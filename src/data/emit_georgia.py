"""
PA-SSL: Georgia 12-Lead ECG Data Emitter

Processes the Georgia 12-Lead ECG Challenge dataset into the standard
PA-SSL beat-level CSV format.

Dataset: 10,344 12-lead ECG recordings, 500Hz, 27 SNOMED-CT rhythm classes
Access: https://physionet.org/content/challenge-2020/1.0.2/
        (Part of PhysioNet/CinC Challenge 2020 training set)
Format: WFDB format (.hea / .mat)

Setup:
    1. Download: wget -r -N -c -np https://physionet.org/files/challenge-2020/1.0.2/training/georgia/
    2. python -m src.data.emit_georgia --data_dir data/georgia --output data/georgia_processed.csv

Label mapping:
    Maps SNOMED-CT codes (in .hea Dx field) to integer classes.
    We map to the 9 most common rhythm classes for compatibility with PTB-XL.

Output CSV: same schema as emit_ptbxl.py (columns 0..249, label, patient_id, ...)
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample, find_peaks
from tqdm import tqdm

try:
    import wfdb
    HAS_WFDB = True
except ImportError:
    HAS_WFDB = False
    print("[WARN] wfdb not installed. Run: pip install wfdb")

try:
    import scipy.io
    HAS_SCIPY_IO = True
except ImportError:
    HAS_SCIPY_IO = False


# ─── CONFIGURATION ────────────────────────────────────────────────────────────
SOURCE_FS = 500
TARGET_FS = 100
WINDOW_SEC = 2.5
WINDOW_SAMPLES = int(TARGET_FS * WINDOW_SEC)  # 250
LEAD_IDX = 1             # Lead II
FILTER_LOW = 0.5
FILTER_HIGH = 40.0

# SNOMED-CT code → integer label mapping (most common 9 classes)
# Full mapping at: https://github.com/physionetchallenges/evaluation-2020
SNOMED_TO_LABEL = {
    # Normal
    '426783006': 0,   # Sinus rhythm
    '164934002': 0,   # Normal ECG
    # AF/Flutter
    '164889003': 1,   # Atrial fibrillation
    '164890007': 2,   # Atrial flutter
    # Blocks
    '270492004': 3,   # 1st degree AV block
    '713427006': 4,   # Left bundle branch block (complete)
    '713426002': 4,   # Left bundle branch block
    '59118001':  5,   # Right bundle branch block
    # Extrasystoles
    '284470004': 6,   # PAC
    '427172004': 6,   # PAC
    '164884008': 7,   # PVC
    '427345002': 7,   # PVC
    # ST changes
    '164931005': 8,   # ST depression
    '164930006': 8,   # ST elevation
}
DEFAULT_LABEL = 0


# ─── UTILITIES ────────────────────────────────────────────────────────────────

def bandpass_filter(signal, fs, low, high):
    nyq = fs / 2.0
    b, a = butter(3, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)


def find_r_peaks(signal, fs=TARGET_FS):
    min_dist = int(0.3 * fs)
    thresh = signal.std() * 0.5
    peaks, _ = find_peaks(signal, distance=min_dist, prominence=thresh)
    return peaks


def extract_beats(signal, r_peaks):
    half = WINDOW_SAMPLES // 2
    beats = []
    for rp in r_peaks:
        s = rp - half
        e = rp + half
        if s < 0 or e > len(signal):
            continue
        beat = signal[s:e].astype(np.float32)
        std = beat.std()
        if std > 1e-6:
            beat = (beat - beat.mean()) / std
        beats.append(beat)
    return beats


def parse_dx_from_hea(hea_path: str) -> list:
    """Extract SNOMED-CT Dx codes from a .hea header file."""
    codes = []
    try:
        with open(hea_path, 'r') as f:
            for line in f:
                if line.startswith('#Dx:') or line.startswith('# Dx:'):
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        codes = [c.strip() for c in parts[1].split(',')]
                    break
    except Exception:
        pass
    return codes


def snomed_to_label(codes: list) -> int:
    """Map a list of SNOMED codes to a single integer label."""
    for code in codes:
        if code in SNOMED_TO_LABEL:
            return SNOMED_TO_LABEL[code]
    return DEFAULT_LABEL


def parse_age_sex_from_hea(hea_path: str):
    """Extract #Age and #Sex from .hea header."""
    age, sex = 60.0, 0.5
    try:
        with open(hea_path, 'r') as f:
            for line in f:
                if line.startswith('#Age:') or line.startswith('# Age:'):
                    val = line.split(':', 1)[1].strip()
                    try:
                        age = float(val)
                    except ValueError:
                        pass
                elif line.startswith('#Sex:') or line.startswith('# Sex:'):
                    val = line.split(':', 1)[1].strip().upper()
                    sex = 1.0 if val.startswith('M') else 0.0
    except Exception:
        pass
    return age, sex


# ─── MAIN PROCESSING ──────────────────────────────────────────────────────────

def process_georgia(data_dir: str, output_path: str) -> None:
    if not HAS_WFDB:
        print("wfdb is required. Install with: pip install wfdb")
        return

    data_dir = Path(data_dir)

    # Find all .hea files (recursive search)
    hea_files = sorted(data_dir.glob('**/*.hea'))
    if not hea_files:
        print(f"  [ERROR] No .hea files found in {data_dir}")
        print("  Please download Georgia 12-Lead ECG dataset from PhysioNet/CinC 2020")
        return

    print(f"  Found {len(hea_files)} .hea records in {data_dir}")

    all_rows = []
    skipped = 0

    for hea_file in tqdm(hea_files, desc='Processing Georgia ECGs'):
        rec_path = str(hea_file.with_suffix(''))

        # Parse metadata from .hea header directly
        dx_codes = parse_dx_from_hea(str(hea_file))
        label = snomed_to_label(dx_codes)
        age, sex = parse_age_sex_from_hea(str(hea_file))

        try:
            # Load waveform — try WFDB format first, then .mat
            if hea_file.with_suffix('.mat').exists():
                mat = scipy.io.loadmat(rec_path + '.mat') if HAS_SCIPY_IO else None
                if mat is not None:
                    ecg = mat.get('val', mat.get('ECG', {}).get('val', np.array([]))).astype(np.float32)
                    if ecg.ndim == 1:
                        raw = ecg
                    elif ecg.ndim == 2:
                        # (leads, samples) shape
                        lead_i = min(LEAD_IDX, ecg.shape[0] - 1)
                        raw = ecg[lead_i, :]
                    else:
                        skipped += 1
                        continue
                else:
                    skipped += 1
                    continue
            else:
                rec = wfdb.rdrecord(rec_path)
                sig_names = [n.upper() for n in rec.sig_name]
                lead_i = sig_names.index('II') if 'II' in sig_names else min(LEAD_IDX, rec.n_sig - 1)
                raw = rec.p_signal[:, lead_i].astype(np.float32)

        except Exception:
            skipped += 1
            continue

        raw = np.nan_to_num(raw)

        # Resample
        if SOURCE_FS != TARGET_FS:
            target_len = int(len(raw) * TARGET_FS / SOURCE_FS)
            raw = resample(raw, target_len).astype(np.float32)

        # Filter
        try:
            signal = bandpass_filter(raw, TARGET_FS, FILTER_LOW, FILTER_HIGH)
        except Exception:
            signal = raw

        # R-peak detection
        r_peaks = find_r_peaks(signal)
        if len(r_peaks) < 2:
            skipped += 1
            continue

        # Beat extraction
        beats = extract_beats(signal, r_peaks)
        rec_name = hea_file.stem

        for bi, beat in enumerate(beats):
            row = {c: beat[c] for c in range(WINDOW_SAMPLES)}
            row['label'] = label
            row['patient_id'] = rec_name
            row['record_id'] = rec_name
            row['beat_idx'] = bi
            row['r_peak_pos'] = WINDOW_SAMPLES // 2
            row['age'] = age
            row['sex'] = sex
            all_rows.append(row)

    if not all_rows:
        print(f"  [WARN] No beats extracted. Check data directory structure.")
        return

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n  Saved {len(df):,} beats from {len(hea_files) - skipped} records → {output_path}")
    print(f"  Skipped: {skipped} records")
    print(f"  Label distribution:\n{df['label'].value_counts().to_string()}")


def main():
    parser = argparse.ArgumentParser(description='Process Georgia 12-Lead ECG dataset')
    parser.add_argument('--data_dir', type=str, default='data/georgia',
                        help='Directory containing Georgia ECG data')
    parser.add_argument('--output', type=str, default='data/georgia_processed.csv',
                        help='Output CSV path')
    args = parser.parse_args()

    print(f"Processing Georgia 12-Lead ECG from: {args.data_dir}")
    process_georgia(args.data_dir, args.output)


if __name__ == '__main__':
    main()
