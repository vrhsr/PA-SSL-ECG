"""
PA-SSL: MIT-BIH Arrhythmia Database Processor (WFDB format)

Reads MIT-BIH records in PhysioNet WFDB format (.hea + .dat files),
as downloaded by wfdb.dl_database('mitdb', ...).

Usage:
    python3 -m src.data.emit_mitbih \
        --data_dir data/mitbih/physionet.org/files/mitdb/1.0.0 \
        --output data/mitbih_processed.csv
"""

import numpy as np
import pandas as pd
from scipy.signal import resample
from src.data.signal_processing import bandpass_filter, z_score_normalize, detect_r_peaks, butter, filtfilt
from tqdm import tqdm
import os
import argparse

try:
    import wfdb
    HAS_WFDB = True
except ImportError:
    HAS_WFDB = False

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
SAMPLE_RATE = 100       # Target Hz
WINDOW_SEC = 2.5        # Window length
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)  # 250 samples
FILTER_LOW = 0.5
FILTER_HIGH = 40.0
SOURCE_RATE = 360       # MIT-BIH native rate
OUTPUT_FILE = 'data/mitbih_processed.csv'

# 48 standard MIT-BIH record numbers
MITBIH_RECORDS = [
    '100','101','102','103','104','105','106','107',
    '108','109','111','112','113','114','115','116',
    '117','118','119','121','122','123','124','200',
    '201','202','203','205','207','208','209','210',
    '212','213','214','215','217','219','220','221',
    '222','223','228','230','231','232','233','234',
]

# AAMI beat classification
AAMI_MAP = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,   # Normal
    'A': 1, 'a': 1, 'J': 1, 'S': 1,             # SVE (Abnormal)
    'V': 1, 'E': 1,                               # VE (Abnormal)
    'F': 1,                                        # Fusion (Abnormal)
    '/': 1, 'f': 1, 'Q': 1                        # Unknown/Paced (Abnormal)
}

# ─── SIGNAL PROCESSING ────────────────────────────────────────────────────────

def bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')
    return filtfilt(b, a, data)

def z_score_normalize(signal):
    std = np.std(signal)
    if std < 1e-6:
        return np.zeros_like(signal)
    return (signal - np.mean(signal)) / std

# ─── RECORD PROCESSING ────────────────────────────────────────────────────────

def find_record_path(data_dir, record_name):
    """Search for record files in data_dir and subdirectories."""
    # Direct match
    if os.path.exists(os.path.join(data_dir, record_name + '.hea')):
        return data_dir, record_name

    # Search subdirs (e.g. mitdbdir/)
    for root, dirs, files in os.walk(data_dir):
        if record_name + '.hea' in files:
            return root, record_name

    return None, None


def process_record_wfdb(data_dir, record_name):
    """Process a single WFDB MIT-BIH record."""
    rec_dir, rec_name = find_record_path(data_dir, record_name)
    if rec_dir is None:
        return [], [], [], [], []

    try:
        record = wfdb.rdrecord(os.path.join(rec_dir, rec_name))
        annotation = wfdb.rdann(os.path.join(rec_dir, rec_name), 'atr')
    except Exception as e:
        return [], [], [], [], []

    # Use MLII (Lead II) — typically channel 0
    fs = record.fs or SOURCE_RATE
    signal = record.p_signal[:, 0].astype(np.float32)
    signal = bandpass_filter(signal, FILTER_LOW, FILTER_HIGH, fs)

    samples = annotation.sample
    symbols = annotation.symbol

    beats, labels, pids, bidxs, rpeaks = [], [], [], [], []
    window_raw = int(WINDOW_SEC * fs)
    half_window = window_raw // 2

    for beat_idx, (sample_idx, symbol) in enumerate(zip(samples, symbols)):
        if symbol not in AAMI_MAP:
            continue

        start = sample_idx - half_window
        end   = sample_idx + half_window

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


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def process_mitbih(output_file=None, data_dir=None):
    if output_file is None:
        output_file = OUTPUT_FILE

    if not HAS_WFDB:
        print("ERROR: wfdb not installed. Run: pip install wfdb")
        return

    if data_dir is None:
        # Try common locations
        candidates = [
            'data/mitbih/physionet.org/files/mitdb/1.0.0',
            'data/mitbih',
            'data/mitdb',
        ]
        for c in candidates:
            if os.path.exists(c):
                data_dir = c
                break
        if data_dir is None:
            print("ERROR: MIT-BIH data not found. Use --data_dir to specify path.")
            return

    print(f"Processing MIT-BIH from: {data_dir}")

    all_beats, all_labels, all_pids, all_rids, all_bidxs, all_rpeaks = \
        [], [], [], [], [], []

    found = 0
    for rec in tqdm(MITBIH_RECORDS, desc="Processing MIT-BIH"):
        beats, lbls, pids, bidxs, rpeaks = process_record_wfdb(data_dir, rec)
        if beats:
            found += 1
            all_beats.extend(beats)
            all_labels.extend(lbls)
            all_pids.extend(pids)
            all_rids.extend([rec] * len(beats))
            all_bidxs.extend(bidxs)
            all_rpeaks.extend(rpeaks)

    print(f"  Records found: {found}/{len(MITBIH_RECORDS)}")

    if len(all_beats) == 0:
        print("WARNING: No beats extracted. Check data path.")
        return

    df = pd.DataFrame(np.array(all_beats))
    df['label']      = all_labels
    df['patient_id'] = all_pids
    df['record_id']  = all_rids
    df['beat_idx']   = all_bidxs
    df['r_peak_pos'] = all_rpeaks

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"\n{'='*60}")
    print(f"MIT-BIH Processing Complete")
    print(f"  Total beats: {len(df):,}")
    print(f"  Records processed: {df['patient_id'].nunique()}")
    print(f"  Label distribution:")
    for lbl, count in df['label'].value_counts().sort_index().items():
        print(f"    Class {lbl}: {count:,}  ({100*count/len(df):.1f}%)")
    print(f"  Saved to: {output_file}")
    print(f"{'='*60}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MIT-BIH dataset (WFDB format)")
    parser.add_argument('--output', type=str, default=OUTPUT_FILE,
                        help='Output CSV path')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to MIT-BIH directory (containing .hea/.dat files). '
                             'Auto-detects common locations if omitted.')
    args = parser.parse_args()
    process_mitbih(args.output, args.data_dir)
