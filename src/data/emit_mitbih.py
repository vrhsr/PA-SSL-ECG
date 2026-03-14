"""
PA-SSL: MIT-BIH Arrhythmia Database Processor (CSV/Text version)
Processes MIT-BIH recordings provided as .csv and .annotations.txt files.
"""

import numpy as np
import pandas as pd
from scipy.signal import resample, butter, filtfilt, find_peaks
from tqdm import tqdm
import os
import re

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
SAMPLE_RATE = 100       # Target Hz
WINDOW_SEC = 2.5        # Window length
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)  # 250 samples
FILTER_LOW = 0.5
FILTER_HIGH = 40.0
SOURCE_RATE = 360       # MIT-BIH native
LEAD_IDX = 1            # In CSVs, typically column 0 is index, 1 is MLII
DATA_DIR = 'data/mitbh dataset'
OUTPUT_FILE = 'data/mitbih_processed.csv'

# ─── AAMI MAPPING ────────────────────────────────────────────────────────────
AAMI_MAP = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,   # Normal
    'A': 1, 'a': 1, 'J': 1, 'S': 1,             # SVE (Abnormal)
    'V': 1, 'E': 1,                               # VE (Abnormal)
    'F': 1,                                        # Fusion (Abnormal)
    '/': 1, 'f': 1, 'Q': 1                        # Unknown/Paced (Abnormal)
}

# ─── SIGNAL PROCESSING ───────────────────────────────────────────────────────

def bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def z_score_normalize(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-6:
        return np.zeros_like(signal)
    return (signal - mean) / std

# ─── PARSING ──────────────────────────────────────────────────────────────────

def parse_annotations(ann_path):
    """
    Parses MIT-BIH annotations from .annotations.txt.
    Format is typically: [Time] [Sample] [Type] [Sub] [Chan] [Num] [Aux]
    """
    samples = []
    symbols = []
    with open(ann_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]: # Skip header
            parts = re.split(r'\s+', line.strip())
            if len(parts) >= 3:
                try:
                    sample = int(parts[1])
                    symbol = parts[2]
                    samples.append(sample)
                    symbols.append(symbol)
                except ValueError:
                    continue
    return samples, symbols

def process_record(record_name):
    csv_path = os.path.join(DATA_DIR, f"{record_name}.csv")
    ann_path = os.path.join(DATA_DIR, f"{record_name}annotations.txt")
    
    # Load signal (CSV usually has index, MLII, V1)
    df_sig = pd.read_csv(csv_path)
    # Use second column (MLII)
    signal = df_sig.iloc[:, 1].values
    
    # Load annotations
    samples, symbols = parse_annotations(ann_path)
    
    # Filter
    signal = bandpass_filter(signal, FILTER_LOW, FILTER_HIGH, SOURCE_RATE)
    
    beats, labels, pids, bidxs, rpeaks = [], [], [], [], []
    
    for beat_idx, (sample_idx, symbol) in enumerate(zip(samples, symbols)):
        if symbol not in AAMI_MAP:
            continue
        
        window_raw = int(WINDOW_SEC * SOURCE_RATE)
        half_window = window_raw // 2
        
        start = sample_idx - half_window
        end = sample_idx + half_window
        
        if start < 0 or end > len(signal):
            continue
        
        beat_raw = signal[start:end]
        beat_resampled = resample(beat_raw, WINDOW_SAMPLES)
        r_peak_resampled = int(half_window * WINDOW_SAMPLES / window_raw)
        beat_norm = z_score_normalize(beat_resampled)
        
        beats.append(beat_norm)
        labels.append(AAMI_MAP[symbol])
        pids.append(record_name)
        bidxs.append(beat_idx)
        rpeaks.append(r_peak_resampled)
        
    return beats, labels, pids, bidxs, rpeaks

def process_mitbih(output_file=None):
    if output_file is None:
        output_file = OUTPUT_FILE
        
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: MIT-BIH data not found at {DATA_DIR}")
        return
        
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    records = sorted([f.replace('.csv', '') for f in csv_files])
    
    print(f"Found {len(records)} MIT-BIH records. Processing...")
    
    all_beats, all_labels, all_pids, all_record_ids, all_bidxs, all_rpeaks = [], [], [], [], [], []
    
    for rec in tqdm(records, desc="Processing MIT-BIH", mininterval=15.0):
        try:
            beats, lbls, pids, bidxs, rpeaks = process_record(rec)
            all_beats.extend(beats)
            all_labels.extend(lbls)
            all_pids.extend(pids)
            all_record_ids.extend([rec] * len(beats))
            all_bidxs.extend(bidxs)
            all_rpeaks.extend(rpeaks)
        except Exception as e:
            continue
            
    if len(all_beats) == 0:
        print("WARNING: No beats extracted. Check dataset structure.")
        return
        
    df = pd.DataFrame(np.array(all_beats))
    df['label'] = all_labels
    df['patient_id'] = all_pids
    df['record_id'] = all_record_ids
    df['beat_idx'] = all_bidxs
    df['r_peak_pos'] = all_rpeaks
    
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
    process_mitbih()
