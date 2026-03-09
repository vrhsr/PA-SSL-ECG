"""
PA-SSL: Unified ECG Dataset
Supports all three datasets (PTB-XL, MIT-BIH, Chapman-Shaoxing) with:
  - Patient-aware train/val/test splitting (no patient leakage)
  - Temporal adjacency pair mining (for contrastive positives)
  - Pluggable augmentation pipeline
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit
from collections import defaultdict


class ECGBeatDataset(Dataset):
    """
    Core dataset for ECG beats.
    
    Loads pre-processed CSV with columns:
        0..249: signal samples
        label: class label
        patient_id: patient identifier
        record_id: recording identifier
        beat_idx: beat index within recording
        r_peak_pos: R-peak position within the 250-sample window
    """
    
    def __init__(self, csv_file, label_fraction=1.0, seed=42):
        """
        Args:
            csv_file: Path to processed CSV
            label_fraction: Fraction of labels to use (for label-efficiency experiments)
            seed: Random seed for label subsampling
        """
        print(f"Loading ECG dataset from {csv_file}...")
        df = pd.read_csv(csv_file)
        
        # Signal columns
        self.signal_cols = [c for c in df.columns if str(c).isdigit()]
        
        self.X = df[self.signal_cols].values.astype(np.float32)
        self.labels = df['label'].values.astype(np.int64)
        self.patient_ids = df['patient_id'].values
        self.record_ids = df['record_id'].values if 'record_id' in df.columns else df['patient_id'].values
        self.beat_idxs = df['beat_idx'].values.astype(np.int64) if 'beat_idx' in df.columns else np.arange(len(df))
        self.r_peak_positions = df['r_peak_pos'].values.astype(np.int64) if 'r_peak_pos' in df.columns else np.full(len(df), 125)
        
        # Conditional Metadata
        self.age = df['age'].values.astype(np.float32) if 'age' in df.columns else np.full(len(df), 60.0, dtype=np.float32)
        self.sex = df['sex'].values.astype(np.float32) if 'sex' in df.columns else np.full(len(df), 0.5, dtype=np.float32)
        self.weight = df['weight'].values.astype(np.float32) if 'weight' in df.columns else np.full(len(df), 70.0, dtype=np.float32)
        self.height = df['height'].values.astype(np.float32) if 'height' in df.columns else np.full(len(df), 165.0, dtype=np.float32)
        
        # Label subsampling for label-efficiency experiments
        self.label_mask = np.ones(len(self.X), dtype=bool)
        if label_fraction < 1.0:
            rng = np.random.RandomState(seed)
            n_labeled = max(1, int(len(self.X) * label_fraction))
            labeled_indices = rng.choice(len(self.X), n_labeled, replace=False)
            self.label_mask = np.zeros(len(self.X), dtype=bool)
            self.label_mask[labeled_indices] = True
        
        # Build temporal adjacency index: record_id → sorted list of (beat_idx, global_idx)
        self._build_temporal_index()
        
        print(f"  Loaded {len(self.X):,} beats, {len(np.unique(self.patient_ids)):,} patients")
    
    def _build_temporal_index(self):
        """Build index for finding temporally adjacent beats."""
        self.temporal_index = defaultdict(list)
        for global_idx in range(len(self.X)):
            rec_id = self.record_ids[global_idx]
            beat_idx = self.beat_idxs[global_idx]
            self.temporal_index[rec_id].append((beat_idx, global_idx))
        
        # Sort by beat_idx within each record
        for rec_id in self.temporal_index:
            self.temporal_index[rec_id].sort(key=lambda x: x[0])
    
    def get_temporal_neighbor(self, idx):
        """
        Get the index of a temporally adjacent beat (±1 beat from same record).
        Returns None if no neighbor exists.
        """
        rec_id = self.record_ids[idx]
        beats = self.temporal_index[rec_id]
        
        # Find position of this beat in the record's beat list
        for pos, (beat_idx, global_idx) in enumerate(beats):
            if global_idx == idx:
                # Pick adjacent beat (prefer next, fallback to previous)
                candidates = []
                if pos + 1 < len(beats):
                    candidates.append(beats[pos + 1][1])
                if pos - 1 >= 0:
                    candidates.append(beats[pos - 1][1])
                
                if candidates:
                    return candidates[np.random.randint(len(candidates))]
                return None
        return None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        signal = torch.tensor(self.X[idx]).unsqueeze(0)  # (1, 250)
        label = torch.tensor(self.labels[idx])
        r_peak = self.r_peak_positions[idx]
        metadata = torch.tensor([self.age[idx], self.sex[idx], self.weight[idx], self.height[idx]])
        return signal, label, r_peak, idx, metadata


class SSLECGDataset(Dataset):
    """
    Dataset for Self-Supervised Contrastive Learning.
    
    Returns pairs of augmented views:
      - (view1, view2): physiology-aware augmented views of the same beat
      - temporal_view: augmented view of a temporally adjacent beat (optional)
    """
    
    def __init__(self, base_dataset, augmentation_pipeline=None, 
                 use_temporal_positives=True):
        """
        Args:
            base_dataset: ECGBeatDataset instance
            augmentation_pipeline: Callable that takes (signal, r_peak_pos) → augmented signal
            use_temporal_positives: Whether to include temporal adjacency positives
        """
        self.base = base_dataset
        self.augment = augmentation_pipeline
        self.use_temporal = use_temporal_positives
    
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        signal = self.base.X[idx].copy()
        r_peak = self.base.r_peak_positions[idx]
        
        # Generate two augmented views
        if self.augment is not None:
            view1 = self.augment(signal.copy(), r_peak)
            view2 = self.augment(signal.copy(), r_peak)
        else:
            view1 = torch.tensor(signal).float()
            view2 = torch.tensor(signal).float()
        
        # Ensure tensor format: (1, 250)
        if isinstance(view1, np.ndarray):
            view1 = torch.tensor(view1).float()
        if isinstance(view2, np.ndarray):
            view2 = torch.tensor(view2).float()
        
        if view1.dim() == 1:
            view1 = view1.unsqueeze(0)
        if view2.dim() == 1:
            view2 = view2.unsqueeze(0)
        
        result = {'view1': view1, 'view2': view2, 'idx': idx}
        
        # Inject metadata
        if hasattr(self.base, 'age'):
            metadata = torch.tensor([
                self.base.age[idx], 
                self.base.sex[idx], 
                self.base.weight[idx], 
                self.base.height[idx]
            ])
            result['metadata'] = metadata
        
        # Temporal positive
        if self.use_temporal:
            neighbor_idx = self.base.get_temporal_neighbor(idx)
            if neighbor_idx is not None:
                neighbor_signal = self.base.X[neighbor_idx].copy()
                neighbor_rpeak = self.base.r_peak_positions[neighbor_idx]
                if self.augment is not None:
                    temporal_view = self.augment(neighbor_signal, neighbor_rpeak)
                else:
                    temporal_view = torch.tensor(neighbor_signal).float()
                
                if isinstance(temporal_view, np.ndarray):
                    temporal_view = torch.tensor(temporal_view).float()
                if temporal_view.dim() == 1:
                    temporal_view = temporal_view.unsqueeze(0)
                
                result['temporal_view'] = temporal_view
                result['has_temporal'] = True
            else:
                # No neighbor — duplicate view1 as fallback
                result['temporal_view'] = view1.clone()
                result['has_temporal'] = False
        
        return result


# ─── SPLITTING UTILITIES ─────────────────────────────────────────────────────

def patient_aware_split(csv_file, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split dataset ensuring no patient appears in multiple splits.
    
    Returns:
        train_df, val_df, test_df as DataFrames
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    df = pd.read_csv(csv_file)
    
    # First split: train+val vs test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(df, groups=df['patient_id']))
    
    df_trainval = df.iloc[trainval_idx]
    df_test = df.iloc[test_idx]
    
    # Second split: train vs val
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio_adjusted, random_state=seed)
    train_idx, val_idx = next(gss2.split(df_trainval, groups=df_trainval['patient_id']))
    
    df_train = df_trainval.iloc[train_idx]
    df_val = df_trainval.iloc[val_idx]
    
    # Verify no patient leakage
    train_patients = set(df_train['patient_id'].unique())
    val_patients = set(df_val['patient_id'].unique())
    test_patients = set(df_test['patient_id'].unique())
    
    assert len(train_patients & val_patients) == 0, "Patient leakage: train ∩ val"
    assert len(train_patients & test_patients) == 0, "Patient leakage: train ∩ test"
    assert len(val_patients & test_patients) == 0, "Patient leakage: val ∩ test"
    
    print(f"Patient-Aware Split:")
    print(f"  Train: {len(df_train):,} beats, {len(train_patients):,} patients")
    print(f"  Val:   {len(df_val):,} beats, {len(val_patients):,} patients")
    print(f"  Test:  {len(df_test):,} beats, {len(test_patients):,} patients")
    
    return df_train, df_val, df_test


def load_multi_dataset(dataset_names, data_dir='data'):
    """
    Load and concatenate multiple pre-processed ECG datasets.
    
    Args:
        dataset_names: List of dataset names ('ptbxl', 'mitbih', 'chapman')
        data_dir: Base directory containing processed CSVs
    """
    file_map = {
        'ptbxl': 'ptbxl_processed.csv',
        'mitbih': 'mitbih_processed.csv',
        'chapman': 'chapman_processed.csv',
    }
    
    dfs = []
    for name in dataset_names:
        path = f"{data_dir}/{file_map[name]}"
        df = pd.read_csv(path)
        df['source_dataset'] = name
        dfs.append(df)
        print(f"Loaded {name}: {len(df):,} beats")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Combined: {len(combined):,} total beats")
    return combined
