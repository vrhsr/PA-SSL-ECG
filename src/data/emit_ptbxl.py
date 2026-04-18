"""
PA-SSL: PTB-XL ECG Data Processor
Downloads, segments, and normalizes PTB-XL ECG recordings into beat-level CSVs.

Adapted from WavKAN-CL project with enhancements:
  - Preserves R-peak indices within each beat window (needed for physiology-aware augmentations)
  - Stores per-beat metadata (patient_id, beat_index_in_record) for temporal adjacency
  - Supports multi-class labels (not just binary)
"""

import wfdb
import numpy as np
import pandas as pd
from scipy.signal import resample
from src.data.signal_processing import bandpass_filter, z_score_normalize, detect_r_peaks
import ast
from tqdm import tqdm
import os
import requests
import zipfile

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
SAMPLE_RATE = 100       # Target Hz (after resampling)
WINDOW_SEC = 2.5        # Window length in seconds
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)  # 250 samples
FILTER_LOW = 0.5        # Bandpass low cutoff (Hz)
FILTER_HIGH = 40.0      # Bandpass high cutoff (Hz)
SOURCE_RATE = 100       # PTB-XL native sampling rate (using 100Hz records)
LEAD_IDX = 1            # Lead II index in 12-lead order (I, II, III...)
DATA_DIR = r'e:\PhD\PA-SSL-ECG\data\ptb-xl\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
# Fallback paths for Kaggle downloads
DATA_DIR_FALLBACKS = [
    'data/ptb-xl',
    'data/ptb-xl-1.0.3',
    'data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3',
    'data/ptbxl',
]
OUTPUT_FILE = 'data/ptbxl_processed.csv'

# ─── LABEL MAPPING ───────────────────────────────────────────────────────────
# Binary: NORM → 0, MI/STTC/CD/HYP → 1
# Multi-class mapping also available for future extension
DIAGNOSTIC_CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

def map_label_binary(scp_codes, agg_df):
    """Map SCP codes to binary label: 0 (Normal), 1 (Abnormal), -1 (Exclude)."""
    results = []
    for key in scp_codes.keys():
        if key in agg_df.index:
            cls = agg_df.loc[key].diagnostic_class
            if str(cls) != 'nan':
                results.append(cls)
    
    if 'NORM' in results:
        return 0
    if any(c in ['MI', 'STTC', 'CD', 'HYP'] for c in results):
        return 1
    return -1  # Exclude ambiguous

def map_label_multiclass(scp_codes, agg_df):
    """Map SCP codes to multi-class label for fine-grained analysis."""
    results = []
    for key in scp_codes.keys():
        if key in agg_df.index:
            cls = agg_df.loc[key].diagnostic_class
            if str(cls) != 'nan':
                results.append(cls)
    
    if 'NORM' in results:
        return 0
    class_map = {'MI': 1, 'STTC': 2, 'CD': 3, 'HYP': 4}
    for cls in results:
        if cls in class_map:
            return class_map[cls]
    return -1

# ─── SIGNAL PROCESSING ───────────────────────────────────────────────────────



# ─── R-PEAK DETECTION (with fallback) ────────────────────────────────────────



# ─── DOWNLOAD ─────────────────────────────────────────────────────────────────

def download_ptbxl():
    """Download PTB-XL dataset from PhysioNet."""
    global DATA_DIR
    url = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
    zip_path = os.path.join("data", "ptbxl.zip")
    
    os.makedirs("data", exist_ok=True)
    
    # Check all possible locations
    for candidate in DATA_DIR_FALLBACKS:
        if os.path.exists(os.path.join(candidate, 'ptbxl_database.csv')):
            DATA_DIR = candidate
            print(f"PTB-XL already downloaded. Found at: {DATA_DIR}")
            return
    
    # Also check for nested Kaggle structure (data/ptb-xl-1.0.3/ptb-xl-.../ptbxl_database.csv)
    for candidate in DATA_DIR_FALLBACKS:
        if os.path.exists(candidate):
            for sub in os.listdir(candidate):
                nested = os.path.join(candidate, sub, 'ptbxl_database.csv')
                if os.path.exists(nested):
                    DATA_DIR = os.path.join(candidate, sub)
                    print(f"PTB-XL found (nested Kaggle structure): {DATA_DIR}")
                    return
    
    print(f"Downloading PTB-XL from PhysioNet...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(zip_path, 'wb') as f:
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading") as pbar:
            for data in response.iter_content(1024):
                pbar.update(len(data))
                f.write(data)
    
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("data")
    
    # Handle PhysioNet's long folder name
    extracted_dirs = [d for d in os.listdir("data") 
                      if 'ptb-xl' in d and os.path.isdir(os.path.join("data", d))]
    for d in extracted_dirs:
        if d != 'ptb-xl-1.0.3' and 'large-publicly' in d:
            src = os.path.join("data", d)
            dst = os.path.join("data", "ptb-xl-1.0.3")
            if not os.path.exists(dst):
                os.rename(src, dst)
            break
    
    # Re-detect
    for candidate in DATA_DIR_FALLBACKS:
        if os.path.exists(os.path.join(candidate, 'ptbxl_database.csv')):
            DATA_DIR = candidate
            break
    
    os.remove(zip_path)
    print("PTB-XL downloaded and extracted.")

# ─── MAIN PROCESSING ─────────────────────────────────────────────────────────

def process_ptbxl(output_file=None, label_mode='binary', data_dir=None):
    """
    Process PTB-XL dataset into beat-level CSV.
    
    Each row contains:
        - 250 signal columns (0..249): the resampled, filtered, normalized beat
        - 'label': binary (0/1) or multi-class label
        - 'patient_id': patient identifier (for patient-aware splits)
        - 'record_id': ECG record ID (for temporal adjacency)
        - 'beat_idx': beat index within the record (for temporal ordering)
        - 'r_peak_pos': R-peak position within the 250-sample window (for augmentations)
    """
    global DATA_DIR
    if output_file is None:
        output_file = OUTPUT_FILE
    if data_dir is not None:
        DATA_DIR = data_dir
        print(f"Using data_dir: {DATA_DIR}")
    else:
        # Auto-detect or download
        download_ptbxl()
    
    db_csv_path = os.path.join(DATA_DIR, 'ptbxl_database.csv')
    print(f"Loading PTB-XL metadata from {DATA_DIR}...")
    
    Y = pd.read_csv(db_csv_path, index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    agg_df = pd.read_csv(os.path.join(DATA_DIR, 'scp_statements.csv'), index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    
    label_fn = map_label_binary if label_mode == 'binary' else map_label_multiclass
    
    all_beats = []
    all_labels = []
    all_patient_ids = []
    all_record_ids = []
    all_beat_idxs = []
    all_rpeak_positions = []
    
    # Metadata for Phase 9 Conditioning
    all_ages = []
    all_sexes = []
    all_weights = []
    all_heights = []
    
    for ecg_id, row in tqdm(Y.iterrows(), total=len(Y), desc="Processing PTB-XL", mininterval=15.0):
        # Get label
        lbl = label_fn(row.scp_codes, agg_df)
        if lbl == -1:
            continue
            
        # Extract Metadata
        age = float(row.age) if pd.notna(row.age) else 60.0 # Mean imputation
        sex = 1.0 if row.sex == 0 else 0.0 # 0=Male(1), 1=Female(0)
        weight = float(row.weight) if pd.notna(row.weight) else 70.0 # Mean imputation
        height = float(row.height) if pd.notna(row.height) else 165.0 # Mean imputation
        
        # Load signal
        record_path = os.path.join(DATA_DIR, row.filename_lr)
        try:
            signals, fields = wfdb.rdsamp(record_path)
        except Exception as e:
            # Log first few errors to help debug
            if not hasattr(process_ptbxl, '_err_count'):
                process_ptbxl._err_count = 0
            process_ptbxl._err_count += 1
            if process_ptbxl._err_count <= 3:
                print(f"\n  WARNING: Cannot read {record_path}: {e}")
            elif process_ptbxl._err_count == 4:
                print(f"\n  (suppressing further file-read warnings...)")
            continue
        
        # Extract Lead II
        signal = signals[:, LEAD_IDX]
        
        try:
            # Bandpass filter
            signal = bandpass_filter(signal, FILTER_LOW, FILTER_HIGH, SOURCE_RATE)
            
            # R-peak detection
            qrs_inds = detect_r_peaks(signal, fs=SOURCE_RATE)
            
            # Segment beats
            for beat_idx, r_peak in enumerate(qrs_inds):
                window_raw = int(WINDOW_SEC * SOURCE_RATE)
                half_window = window_raw // 2
                
                start = r_peak - half_window
                end = r_peak + half_window
                
                if start < 0 or end > len(signal):
                    continue
                
                beat_raw = signal[start:end]
                
                # Resample to target (250 samples)
                beat_resampled = resample(beat_raw, WINDOW_SAMPLES)
                
                # Compute R-peak position in resampled window
                # R-peak is at half_window in raw → scale to resampled
                r_peak_resampled = int(half_window * WINDOW_SAMPLES / window_raw)
                
                # Normalize
                beat_norm = z_score_normalize(beat_resampled)
                
                all_beats.append(beat_norm)
                all_labels.append(lbl)
                all_patient_ids.append(row.patient_id)
                all_record_ids.append(ecg_id)
                all_beat_idxs.append(beat_idx)
                all_rpeak_positions.append(r_peak_resampled)
                all_ages.append(age)
                all_sexes.append(sex)
                all_weights.append(weight)
                all_heights.append(height)
                
        except Exception:
            continue
    
    # Build DataFrame
    df = pd.DataFrame(np.array(all_beats))
    df['label'] = all_labels
    df['patient_id'] = all_patient_ids
    df['record_id'] = all_record_ids
    df['beat_idx'] = all_beat_idxs
    df['r_peak_pos'] = all_rpeak_positions
    df['age'] = all_ages
    df['sex'] = all_sexes
    df['weight'] = all_weights
    df['height'] = all_heights
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"PTB-XL Processing Complete")
    print(f"  Total beats: {len(df):,}")
    print(f"  Unique patients: {df['patient_id'].nunique():,}")
    print(f"  Label distribution:")
    for lbl, count in df['label'].value_counts().sort_index().items():
        print(f"    Class {lbl}: {count:,} ({100*count/len(df):.1f}%)")
    print(f"  Saved to: {output_file}")
    print(f"{'='*60}")
    
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process PTB-XL dataset")
    parser.add_argument('--output', type=str, default=OUTPUT_FILE,
                        help='Output CSV path')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to PTB-XL root dir (containing ptbxl_database.csv). '
                             'If omitted, auto-detects or downloads.')
    parser.add_argument('--label_mode', type=str, default='binary',
                        choices=['binary', 'multiclass'])
    args = parser.parse_args()
    process_ptbxl(args.output, args.label_mode, args.data_dir)

