"""
PA-SSL: CPSC 2018 ECG Data Emitter

Processes the China Physiological Signal Challenge 2018 (CPSC 2018) dataset
into the standard PA-SSL beat-level CSV format.

Dataset: 6,877 12-lead ECG recordings, 500Hz
Access: https://physionet.org/content/cpsc2018/1.1.1/
Labels: 9 rhythm/morphology classes

Setup:
    1. Download from PhysioNet or https://www.kaggle.com/datasets/bjoernjostein/china-12-lead-ecg-challenge-database
    2. Extract to data/cpsc2018/
    3. python -m src.data.emit_cpsc --data_dir data/cpsc2018 --output data/cpsc_processed.csv

Output CSV columns:
    0..249  — 250-sample beat window at 100Hz
    label   — int (0=Normal, 1=AF, 2=I-AVB, 3=LBBB, 4=RBBB, 5=PAC, 6=PVC, 7=STD, 8=STE)
    patient_id, record_id, beat_idx, r_peak_pos, age, sex
"""

import argparse
import os
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, resample, find_peaks
from tqdm import tqdm


# ─── CONFIGURATION ────────────────────────────────────────────────────────────
SOURCE_FS = 500        # CPSC 2018 recording sampling rate
TARGET_FS = 100        # Resampled rate (matches existing PA-SSL pipeline)
WINDOW_SEC = 2.5       # 250 samples at 100Hz
WINDOW_SAMPLES = int(TARGET_FS * WINDOW_SEC)
LEAD_IDX = 1           # Lead II (index 1 in I, II, III... order)
FILTER_LOW = 0.5
FILTER_HIGH = 40.0

CPSC_LABEL_MAP = {
    'Normal': 0,
    'AF': 1,
    'I-AVB': 2,
    'LBBB': 3,
    'RBBB': 4,
    'PAC': 5,
    'PVC': 6,
    'STD': 7,
    'STE': 8,
}


# ─── UTILITIES ────────────────────────────────────────────────────────────────

def bandpass_filter(signal: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    """Apply Butterworth bandpass filter."""
    nyq = fs / 2.0
    b, a = butter(3, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)


def find_r_peaks(signal: np.ndarray, fs: float = TARGET_FS) -> np.ndarray:
    """Detect R-peaks via scipy find_peaks with physiological constraints."""
    min_distance = int(0.3 * fs)   # Minimum 300ms between peaks (max 200bpm)
    prominence = signal.std() * 0.5
    peaks, _ = find_peaks(signal, distance=min_distance, prominence=prominence)
    return peaks


def extract_beats(signal: np.ndarray, r_peaks: np.ndarray, fs: float = TARGET_FS) -> list:
    """Extract fixed-length windows centered on each R-peak."""
    half = WINDOW_SAMPLES // 2
    beats = []
    for rp in r_peaks:
        start = rp - half
        end = rp + half
        if start < 0 or end > len(signal):
            continue
        beat = signal[start:end].astype(np.float32)
        # Z-score normalize
        std = beat.std()
        if std > 1e-6:
            beat = (beat - beat.mean()) / std
        beats.append((beat, rp))
    return beats


# ─── MAIN PROCESSING ──────────────────────────────────────────────────────────

def process_cpsc(data_dir: str, output_path: str) -> None:
    data_dir = Path(data_dir)

    # Look for reference/label file
    ref_file = None
    for candidate in ['REFERENCE.csv', 'reference.csv', 'REFERENCES.csv', 'labels.csv']:
        p = data_dir / candidate
        if p.exists():
            ref_file = p
            break

    records_metadata = {}
    if ref_file:
        ref_df = pd.read_csv(ref_file, header=None, names=['record', 'label_str'])
        for _, row in ref_df.iterrows():
            rec_name = str(row['record']).strip()
            label_str = str(row['label_str']).strip()
            records_metadata[rec_name] = {
                'label_str': label_str,
                'label': CPSC_LABEL_MAP.get(label_str, -1),
                'age': 60.0,
                'sex': 0.5,
            }
    else:
        print("  [WARN] No reference file found — using label 0 for all records.")

    # Find all ECG record files
    all_hea_files = sorted(data_dir.glob('**/*.hea'))
    if not all_hea_files:
        # Try MATLAB .mat format (PhysioNet CPSC)
        all_mat_files = sorted(data_dir.glob('**/*.mat'))
        print(f"  Found {len(all_mat_files)} .mat files (MATLAB format)")
        _process_mat_files(all_mat_files, records_metadata, output_path)
        return

    print(f"  Found {len(all_hea_files)} .hea files (WFDB format)")

    all_rows = []
    skipped = 0

    for hea_file in tqdm(all_hea_files, desc='Processing CPSC records'):
        rec_name = hea_file.stem
        rec_path = str(hea_file.with_suffix(''))

        try:
            record = wfdb.rdrecord(rec_path)
        except Exception as e:
            skipped += 1
            continue

        # Get lead II signal (index 1 or search by name)
        sig_names = [n.upper() for n in record.sig_name]
        if 'II' in sig_names:
            lead_idx = sig_names.index('II')
        else:
            lead_idx = min(1, len(sig_names) - 1)

        raw_signal = record.p_signal[:, lead_idx].astype(np.float32)
        raw_signal = np.nan_to_num(raw_signal)

        # Resample to TARGET_FS
        if record.fs != TARGET_FS:
            target_len = int(len(raw_signal) * TARGET_FS / record.fs)
            raw_signal = resample(raw_signal, target_len).astype(np.float32)

        # Bandpass filter
        try:
            signal = bandpass_filter(raw_signal, TARGET_FS, FILTER_LOW, FILTER_HIGH)
        except Exception:
            signal = raw_signal

        # Detect R-peaks
        r_peaks = find_r_peaks(signal, TARGET_FS)
        if len(r_peaks) < 2:
            skipped += 1
            continue

        meta = records_metadata.get(rec_name, {'label': 0, 'age': 60.0, 'sex': 0.5})
        label = meta.get('label', 0)
        if label < 0:
            continue  # Skip records with unmapped labels

        # Extract beats
        beats = extract_beats(signal, r_peaks, TARGET_FS)

        for beat_idx, (beat, rp) in enumerate(beats):
            row = {c: beat[c] for c in range(WINDOW_SAMPLES)}
            row['label'] = label
            row['patient_id'] = rec_name
            row['record_id'] = rec_name
            row['beat_idx'] = beat_idx
            row['r_peak_pos'] = WINDOW_SAMPLES // 2
            row['age'] = meta.get('age', 60.0)
            row['sex'] = meta.get('sex', 0.5)
            all_rows.append(row)

    if not all_rows:
        print(f"  [WARN] No beats extracted from {data_dir}. Check data path.")
        return

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n  Saved {len(df):,} beats from {len(all_hea_files) - skipped} records → {output_path}")
    print(f"  Skipped: {skipped} records")
    print(f"  Label distribution: {df['label'].value_counts().to_dict()}")


def _process_mat_files(mat_files, records_metadata, output_path: str) -> None:
    """Process MATLAB .mat format CPSC files."""
    try:
        import scipy.io
    except ImportError:
        print("  scipy.io required for .mat files: pip install scipy")
        return

    all_rows = []
    skipped = 0

    for mat_file in tqdm(mat_files, desc='Processing CPSC .mat files'):
        rec_name = mat_file.stem
        try:
            mat = scipy.io.loadmat(str(mat_file))
            # Common CPSC mat keys
            if 'val' in mat:
                ecg = mat['val'].astype(np.float32)
            elif 'ECG' in mat:
                ecg = mat['ECG']['val'][0][0].astype(np.float32)
            else:
                skipped += 1
                continue

            # (leads, samples) format
            if ecg.ndim == 2:
                raw_signal = ecg[min(LEAD_IDX, ecg.shape[0] - 1), :]
            else:
                raw_signal = ecg.flatten()

            raw_signal = np.nan_to_num(raw_signal).astype(np.float32)

            # Resample
            target_len = int(len(raw_signal) * TARGET_FS / SOURCE_FS)
            signal = resample(raw_signal, target_len).astype(np.float32)

            try:
                signal = bandpass_filter(signal, TARGET_FS, FILTER_LOW, FILTER_HIGH)
            except Exception:
                pass

            r_peaks = find_r_peaks(signal, TARGET_FS)
            if len(r_peaks) < 2:
                skipped += 1
                continue

            meta = records_metadata.get(rec_name, {'label': 0, 'age': 60.0, 'sex': 0.5})
            label = meta.get('label', 0)
            beats = extract_beats(signal, r_peaks, TARGET_FS)

            for beat_idx, (beat, rp) in enumerate(beats):
                row = {c: beat[c] for c in range(WINDOW_SAMPLES)}
                row['label'] = label
                row['patient_id'] = rec_name
                row['record_id'] = rec_name
                row['beat_idx'] = beat_idx
                row['r_peak_pos'] = WINDOW_SAMPLES // 2
                row['age'] = meta.get('age', 60.0)
                row['sex'] = meta.get('sex', 0.5)
                all_rows.append(row)

        except Exception as e:
            skipped += 1
            continue

    if all_rows:
        df = pd.DataFrame(all_rows)
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n  Saved {len(df):,} beats → {output_path}")
    else:
        print(f"  [WARN] No beats extracted.")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Process CPSC 2018 ECG dataset')
    parser.add_argument('--data_dir', type=str, default='data/cpsc2018',
                        help='Directory containing CPSC 2018 data')
    parser.add_argument('--output', type=str, default='data/cpsc_processed.csv',
                        help='Output CSV path')
    args = parser.parse_args()

    print(f"Processing CPSC 2018 from: {args.data_dir}")
    process_cpsc(args.data_dir, args.output)


if __name__ == '__main__':
    main()
