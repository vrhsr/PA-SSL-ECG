"""
PA-SSL: MIT-BIH Arrhythmia Database Processor
Downloads, segments, and normalizes MIT-BIH recordings into beat-level CSVs.

Adapted from WavKAN-CL with enhancements:
  - R-peak position preserved per beat
  - Beat index for temporal adjacency
  - Record-level IDs for patient-aware splitting
"""

import wfdb
try:
    import wfdb.processing
    HAS_WFDB_PROCESSING = True
except ImportError:
    HAS_WFDB_PROCESSING = False
import numpy as np
import pandas as pd
from scipy.signal import resample, butter, filtfilt, find_peaks
from tqdm import tqdm
import os

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
SAMPLE_RATE = 100       # Target Hz
WINDOW_SEC = 2.5        # Window length
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)  # 250 samples
FILTER_LOW = 0.5
FILTER_HIGH = 40.0
SOURCE_RATE = 360       # MIT-BIH native
LEAD_IDX = 0            # Lead II is typically index 0 in MIT-BIH
DATA_DIR = 'data/mitbih'
OUTPUT_FILE = 'data/mitbih_processed.csv'

# ─── AAMI MAPPING ────────────────────────────────────────────────────────────
# Following ANSI/AAMI EC57 standard for beat classification
AAMI_MAP = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,   # Normal
    'A': 1, 'a': 1, 'J': 1, 'S': 1,             # SVE (Abnormal)
    'V': 1, 'E': 1,                               # VE (Abnormal)
    'F': 1,                                        # Fusion (Abnormal)
    '/': 1, 'f': 1, 'Q': 1                        # Unknown/Paced (Abnormal)
}

# ─── SIGNAL PROCESSING ───────────────────────────────────────────────────────

def bandpass_filter(data, lowcut, highcut, fs, order=3):
    """Apply Butterworth bandpass filter."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def z_score_normalize(signal):
    """Per-beat z-score normalization."""
    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-6:
        return np.zeros_like(signal)
    return (signal - mean) / std

# ─── R-PEAK DETECTION (with fallback) ────────────────────────────────────────

def detect_r_peaks(signal, fs):
    """Detect R-peaks using wfdb.processing or scipy fallback."""
    if HAS_WFDB_PROCESSING:
        return wfdb.processing.gqrs_detect(signal, fs=fs)
    else:
        height = np.mean(signal) + 0.5 * np.std(signal)
        distance = int(0.4 * fs)
        peaks, _ = find_peaks(signal, height=height, distance=distance)
        return peaks


# ─── RECORD PROCESSING ───────────────────────────────────────────────────────

def process_record(record_name):
    """
    Process a single MIT-BIH record into segmented beats.
    
    Returns:
        beats, labels, patient_ids, beat_idxs, rpeak_positions
    """
    record_path = f"{DATA_DIR}/{record_name}"
    signals, fields = wfdb.rdsamp(record_path)
    annotation = wfdb.rdann(record_path, 'atr')
    
    # Extract Lead II (index 0)
    signal = signals[:, LEAD_IDX]
    
    # Bandpass filter
    signal = bandpass_filter(signal, FILTER_LOW, FILTER_HIGH, SOURCE_RATE)
    
    beats = []
    labels = []
    patient_ids = []
    beat_idxs = []
    rpeak_positions = []
    
    for beat_idx, (sample_idx, symbol) in enumerate(
            zip(annotation.sample, annotation.symbol)):
        if symbol not in AAMI_MAP:
            continue
        
        # Window centered on R-peak
        window_raw = int(WINDOW_SEC * SOURCE_RATE)  # 900 at 360Hz
        half_window = window_raw // 2
        
        start = sample_idx - half_window
        end = sample_idx + half_window
        
        if start < 0 or end > len(signal):
            continue
        
        beat_raw = signal[start:end]
        
        # Resample (900 → 250)
        beat_resampled = resample(beat_raw, WINDOW_SAMPLES)
        
        # R-peak position in resampled window
        r_peak_resampled = int(half_window * WINDOW_SAMPLES / window_raw)
        
        # Normalize
        beat_norm = z_score_normalize(beat_resampled)
        
        beats.append(beat_norm)
        labels.append(AAMI_MAP[symbol])
        patient_ids.append(record_name)
        beat_idxs.append(beat_idx)
        rpeak_positions.append(r_peak_resampled)
    
    return beats, labels, patient_ids, beat_idxs, rpeak_positions

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def process_mitbih(output_file=None):
    """Process all MIT-BIH records into a single CSV."""
    global DATA_DIR
    if output_file is None:
        output_file = OUTPUT_FILE
    
    # Download if needed
    if not os.path.exists(DATA_DIR):
        print(f"Downloading MIT-BIH to {DATA_DIR}...")
        os.makedirs(DATA_DIR, exist_ok=True)
        wfdb.dl_database('mitdb', DATA_DIR)
    
    # Check for .dat files, also look inside subdirectories (Kaggle nesting)
    dat_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.dat')]
    if len(dat_files) == 0:
        # Check nested subdirectories  
        for sub in os.listdir(DATA_DIR):
            subpath = os.path.join(DATA_DIR, sub)
            if os.path.isdir(subpath):
                nested_dats = [f for f in os.listdir(subpath) if f.endswith('.dat')]
                if len(nested_dats) > 0:
                    print(f"  Found MIT-BIH data in nested folder: {subpath}")
                    DATA_DIR = subpath
                    dat_files = nested_dats
                    break
    
    records = sorted(list(set(
        f.split('.')[0] for f in os.listdir(DATA_DIR) if f.endswith('.dat')
    )))
    
    print(f"Found {len(records)} MIT-BIH records. Processing...")
    
    all_beats = []
    all_labels = []
    all_patient_ids = []
    all_record_ids = []
    all_beat_idxs = []
    all_rpeak_positions = []
    
    for rec in tqdm(records, desc="Processing MIT-BIH", mininterval=15.0):
        try:
            beats, lbls, pids, bidxs, rpeaks = process_record(rec)
            all_beats.extend(beats)
            all_labels.extend(lbls)
            all_patient_ids.extend(pids)
            all_record_ids.extend([rec] * len(beats))
            all_beat_idxs.extend(bidxs)
            all_rpeak_positions.extend(rpeaks)
        except Exception as e:
            print(f"Error processing {rec}: {e}")
    
    # Build DataFrame
    df = pd.DataFrame(np.array(all_beats))
    df['label'] = all_labels
    df['patient_id'] = all_patient_ids
    df['record_id'] = all_record_ids
    df['beat_idx'] = all_beat_idxs
    df['r_peak_pos'] = all_rpeak_positions
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"MIT-BIH Processing Complete")
    print(f"  Total beats: {len(df):,}")
    print(f"  Unique records: {df['patient_id'].nunique()}")
    print(f"  Label distribution:")
    for lbl, count in df['label'].value_counts().sort_index().items():
        print(f"    Class {lbl}: {count:,} ({100*count/len(df):.1f}%)")
    print(f"  Saved to: {output_file}")
    print(f"{'='*60}")
    
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process MIT-BIH dataset")
    parser.add_argument('--output', type=str, default=OUTPUT_FILE)
    args = parser.parse_args()
    process_mitbih(args.output)
