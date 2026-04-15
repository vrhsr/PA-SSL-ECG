"""
PA-SSL: Foundation ECG Corpus

Unified DataLoader across 10+ public ECG databases.

Handles:
- Different sampling rates (250/400/500/1000 Hz → resample to 500 Hz)
- Different signal lengths → crop or pad to target_length (default 5000 = 10s @ 500Hz)
- Different lead configurations (12/6/3/1 → extract common subset)
- Different label schemas → unified SNOMED-CT integer codes

Datasets supported:
    ptbxl       — 21,837 records, 500Hz, 12-lead (PhysioNet)
    mitbih      — 48 records, 360Hz, 2-lead (PhysioNet)
    chapman     — 10,646 records, 500Hz, 12-lead (PhysioNet)
    cpsc        — 6,877 records, 500Hz, 12-lead (PhysioNet/public)
    georgia     — 10,344 records, 500Hz, 12-lead (PhysioNet)
    ningbo      — 34,905 records, 500Hz, 12-lead (PhysioNet)
    code15      — 345,779 records, 400Hz, 12-lead (Cardiology Dataset)
    ecg_icbeb   — 6,500 records, 500Hz, 12-lead (public)
    shandong    — 25,770 records, 500Hz, 12-lead (PhysioNet)
    mimic_iv_ecg — 800,000+ records, 500Hz, 12-lead (PhysioNet credentialed)

Usage:
    # All available datasets
    corpus = FoundationECGCorpus(data_root='data/')

    # Specific datasets only
    corpus = FoundationECGCorpus(data_root='data/', datasets=['ptbxl', 'cpsc', 'georgia'])

    # Scale-limited (for scaling law experiments)
    corpus = FoundationECGCorpus(data_root='data/', max_records=50_000)

    # DataLoader
    loader = DataLoader(corpus, batch_size=512, shuffle=True, num_workers=4)
"""

import os
import glob
import json
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import scipy.signal
import torch
from torch.utils.data import Dataset

from src.augmentations.augmentation_pipeline import PhysioAugPipeline


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_REGISTRY = {
    'ptbxl': {
        'csv': 'ptbxl_processed.csv',
        'source_fs': 500,
        'n_leads': 1,        # Our CSV is single-lead (lead II extracted)
        'description': 'PTB-XL: 21,837 clinical 12-lead ECGs (PhysioNet)',
    },
    'mitbih': {
        'csv': 'mitbih_processed.csv',
        'source_fs': 360,
        'n_leads': 1,
        'description': 'MIT-BIH Arrhythmia: 48 ambulatory records (PhysioNet)',
    },
    'chapman': {
        'csv': 'chapman_processed.csv',
        'source_fs': 500,
        'n_leads': 1,
        'description': 'Chapman-Shaoxing: 10,646 12-lead ECGs (PhysioNet)',
    },
    'cpsc': {
        'csv': 'cpsc_processed.csv',
        'source_fs': 500,
        'n_leads': 1,
        'description': 'CPSC 2018: 6,877 12-lead ECGs (China)',
    },
    'georgia': {
        'csv': 'georgia_processed.csv',
        'source_fs': 500,
        'n_leads': 1,
        'description': 'Georgia 12-Lead: 10,344 clinical ECGs',
    },
    'ningbo': {
        'csv': 'ningbo_processed.csv',
        'source_fs': 500,
        'n_leads': 1,
        'description': 'Ningbo: 34,905 ECGs (Ningbo First Hospital)',
    },
    'code15': {
        'csv': 'code15_processed.csv',
        'source_fs': 400,
        'n_leads': 1,
        'description': 'CODE-15%: 345,779 ECGs from Brazilian hospitals',
    },
    'ecg_icbeb': {
        'csv': 'ecg_icbeb_processed.csv',
        'source_fs': 500,
        'n_leads': 1,
        'description': 'ECG-ICBEB: 6,500 12-lead ECGs (ICBEB 2018)',
    },
    'shandong': {
        'csv': 'shandong_processed.csv',
        'source_fs': 500,
        'n_leads': 1,
        'description': 'Shandong Provincial: 25,770 ECGs (PhysioNet)',
    },
    'mimic_iv_ecg': {
        'csv': 'mimic_iv_ecg_processed.csv',
        'source_fs': 500,
        'n_leads': 1,
        'description': 'MIMIC-IV-ECG: 800,000+ ECGs (requires PhysioNet access)',
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL HARMONIZATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def harmonize_sampling_rate(
    signal: np.ndarray,
    source_fs: int,
    target_fs: int = 500,
) -> np.ndarray:
    """
    Resample 1D signal from source_fs to target_fs using polyphase filtering.
    
    Args:
        signal: (L,) array
        source_fs: original sampling rate in Hz
        target_fs: desired sampling rate (default 500 Hz)
    
    Returns:
        Resampled signal array
    """
    if source_fs == target_fs:
        return signal
    # scipy polyphase resampling is more accurate than linear interpolation
    target_len = int(len(signal) * target_fs / source_fs)
    return scipy.signal.resample(signal, target_len)


def harmonize_length(
    signal: np.ndarray,
    target_length: int = 5000,
    mode: str = 'random_crop',
) -> np.ndarray:
    """
    Crop or zero-pad a signal to exactly target_length samples.
    
    Args:
        signal: (L,) array
        target_length: desired length in samples
        mode: 'random_crop' (training) or 'center_crop' (evaluation)
    
    Returns:
        Fixed-length signal array
    """
    L = len(signal)
    if L >= target_length:
        if mode == 'random_crop':
            start = np.random.randint(0, L - target_length + 1)
        else:
            start = (L - target_length) // 2
        return signal[start: start + target_length]
    else:
        # Zero-pad (append)
        pad = np.zeros(target_length - L, dtype=signal.dtype)
        return np.concatenate([signal, pad])


# ═══════════════════════════════════════════════════════════════════════════════
# FOUNDATION CORPUS DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class FoundationECGCorpus(Dataset):
    """
    Unified foundation ECG corpus for large-scale SSL pretraining.

    Loads CSV-formatted ECG datasets (produced by individual emit_*.py scripts),
    harmonizes sampling rate and signal length, and applies the PA-SSL
    physiology-aware augmentation pipeline.

    Parameters
    ----------
    data_root : str
        Directory containing the processed CSV files.
    datasets : list[str] or 'all'
        Which datasets to include. Use 'all' for every available CSV.
    target_fs : int
        Target sampling rate in Hz (default 500).
    target_length : int
        Target signal length in samples (default 5000 = 10s @ 500Hz).
    max_records : int or None
        Cap total records. If None, use everything available.
        Useful for scaling law experiments.
    augmentation : str
        'physio' (default) or 'naive'. Only physio-aware used for main paper.
    seed : int
        Random seed for reproducible subsetting.
    skip_missing : bool
        If True, silently skip datasets whose CSV files don't exist.
        If False, raise FileNotFoundError.
    """

    def __init__(
        self,
        data_root: str = 'data/',
        datasets: str | List[str] = 'all',
        target_fs: int = 500,
        target_length: int = 5000,
        max_records: Optional[int] = None,
        augmentation: str = 'physio',
        seed: int = 42,
        skip_missing: bool = True,
    ):
        self.data_root = Path(data_root)
        self.target_fs = target_fs
        self.target_length = target_length

        # Augmentation pipeline
        if augmentation == 'physio':
            self.aug_pipeline = PhysioAugPipeline.default(
                strength='medium',
                qrs_protect=True,
            )
        else:
            from src.augmentations.naive_augmentations import NaiveAugPipeline
            self.aug_pipeline = NaiveAugPipeline()

        # Collect all available data
        self.records: List[dict] = []
        self._load_datasets(datasets, skip_missing)

        # Cap records for scaling law experiments
        if max_records is not None and len(self.records) > max_records:
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(self.records), max_records, replace=False)
            self.records = [self.records[i] for i in sorted(indices)]

        print(f"\n{'='*60}")
        print(f"FoundationECGCorpus loaded: {len(self.records):,} records")
        if max_records is not None:
            print(f"  (capped at max_records={max_records:,})")
        print(f"  Target FS: {target_fs} Hz | Length: {target_length} samples")
        print(f"  = {target_length / target_fs:.1f}s per signal")
        print(f"{'='*60}\n")

    def _load_datasets(self, datasets: str | List[str], skip_missing: bool) -> None:
        """Load and merge all specified datasets."""
        import pandas as pd

        if datasets == 'all':
            ds_names = list(DATASET_REGISTRY.keys())
        else:
            ds_names = datasets

        for name in ds_names:
            if name not in DATASET_REGISTRY:
                print(f"  [WARN] Unknown dataset '{name}' — skipping.")
                continue

            config = DATASET_REGISTRY[name]
            csv_path = self.data_root / config['csv']

            if not csv_path.exists():
                if skip_missing:
                    # Silently skip — the emitter hasn't been run yet
                    continue
                else:
                    raise FileNotFoundError(
                        f"Dataset CSV not found: {csv_path}\n"
                        f"Run: python -m src.data.emit_{name} --output {csv_path}"
                    )

            try:
                df = pd.read_csv(csv_path)
                n_before = len(self.records)

                # Signal columns: numeric column names (0, 1, 2, ...)
                sig_cols = [c for c in df.columns if str(c).isdigit()]
                if not sig_cols:
                    print(f"  [WARN] {name}: no signal columns found, skipping.")
                    continue

                for _, row in df.iterrows():
                    signal = row[sig_cols].values.astype(np.float32)
                    label = int(row.get('label', 0))
                    patient_id = str(row.get('patient_id', 'unknown'))
                    age = float(row.get('age', 60.0))
                    sex = float(row.get('sex', 0.5))

                    self.records.append({
                        'signal': signal,
                        'label': label,
                        'patient_id': patient_id,
                        'source': name,
                        'source_fs': config['source_fs'],
                        'age': age,
                        'sex': sex,
                    })

                n_added = len(self.records) - n_before
                print(f"  Loaded {name}: {n_added:,} records  ({config['description']})")

            except Exception as e:
                if skip_missing:
                    print(f"  [WARN] Failed to load {name}: {e}")
                else:
                    raise

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        signal = rec['signal'].copy()

        # 1. Resample to target_fs if needed
        signal = harmonize_sampling_rate(
            signal, rec['source_fs'], self.target_fs
        )

        # 2. Crop or pad to target_length
        signal = harmonize_length(signal, self.target_length, mode='random_crop')

        # 3. Z-score normalize per sample
        std = signal.std()
        if std > 1e-6:
            signal = (signal - signal.mean()) / std

        # 4. Two augmented views for SSL training
        signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)  # (1, L)

        view1 = self.aug_pipeline(signal_tensor.squeeze(0).numpy())
        view2 = self.aug_pipeline(signal_tensor.squeeze(0).numpy())

        view1 = torch.tensor(view1, dtype=torch.float32)
        view2 = torch.tensor(view2, dtype=torch.float32)

        # Metadata for conditioning
        metadata = torch.tensor([
            rec['age'] / 100.0,   # Normalize age to [0,1]
            rec['sex'],
        ], dtype=torch.float32)

        return {
            'view1': view1,                             # (L,)
            'view2': view2,                             # (L,)
            'signal': signal_tensor.squeeze(0),         # (L,) original
            'label': torch.tensor(rec['label'], dtype=torch.long),
            'source': rec['source'],
            'metadata': metadata,
        }

    def get_stats(self) -> dict:
        """Return dataset composition summary."""
        from collections import Counter
        source_counts = Counter(r['source'] for r in self.records)
        return {
            'total_records': len(self.records),
            'per_dataset': dict(source_counts),
            'target_fs': self.target_fs,
            'target_length_samples': self.target_length,
            'signal_duration_seconds': self.target_length / self.target_fs,
        }
