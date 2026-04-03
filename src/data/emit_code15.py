"""
PA-SSL: CODE-15% ECG Data Emitter

Processes the CODE-15% dataset (Brazilian longitudinal ECG dataset) into
the standard PA-SSL beat-level CSV format.

Dataset: 345,779 12-lead ECG recordings from ~233K patients, 400Hz
         Largest freely available ECG dataset outside MIMIC-IV-ECG.
Access: https://zenodo.org/record/4916206  (code15_tracings.zip, ~6GB HDF5)
        https://zenodo.org/record/4916206/files/exams.csv (metadata)
Paper: Ribeiro et al., 2021, Nature Communications

Label mapping (6 binary columns in exams.csv):
    1dAVb  — 1st degree AV block
    RBBB   — Right bundle branch block
    LBBB   — Left bundle branch block
    SB     — Sinus bradycardia
    AF     — Atrial fibrillation
    ST     — ST changes

We extract the first positive label as the class (multi-label → single-label
for simplicity; the full multi-hot representation is preserved in metadata).

Setup:
    1. Download HDF5 file: wget https://zenodo.org/record/4916206/files/code15.h5
    2. Download metadata: wget https://zenodo.org/record/4916206/files/exams.csv
    3. python -m src.data.emit_code15 \\
           --hdf5 data/code15/code15.h5 \\
           --metadata data/code15/exams.csv \\
           --output data/code15_processed.csv \\
           --max_records 50000  # Optional cap for testing

Large file notice: 
    - HDF5 file is ~6GB compressed
    - Full processing produces ~10M beats from 345K records
    - Use --max_records to cap for initial experiments
    - Processing time: ~2-4 hours for full dataset on CPU
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample, find_peaks
from tqdm import tqdm


# ─── CONFIGURATION ────────────────────────────────────────────────────────────
SOURCE_FS = 400        # CODE-15% recording sampling rate
TARGET_FS = 100        # PA-SSL target (100Hz)
WINDOW_SEC = 2.5
WINDOW_SAMPLES = int(TARGET_FS * WINDOW_SEC)  # 250
LEAD_IDX = 1           # Lead II (index in standard 12-lead ordering)
FILTER_LOW = 0.5
FILTER_HIGH = 40.0

# Label priority order (first match wins for multi-label → single-label)
LABEL_COLUMNS = ['AF', 'LBBB', 'RBBB', '1dAVb', 'SB', 'ST']
LABEL_MAP = {
    'Normal': 0,   # All zeros → normal
    'AF': 1,
    'LBBB': 2,
    'RBBB': 3,
    '1dAVb': 4,
    'SB': 5,
    'ST': 6,
}


# ─── UTILITIES ────────────────────────────────────────────────────────────────

def bandpass_filter(signal: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    nyq = fs / 2.0
    b, a = butter(3, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)


def find_r_peaks(signal: np.ndarray, fs: float = TARGET_FS) -> np.ndarray:
    min_dist = int(0.3 * fs)
    thresh = signal.std() * 0.5
    peaks, _ = find_peaks(signal, distance=min_dist, prominence=thresh)
    return peaks


def extract_beats(signal: np.ndarray, r_peaks: np.ndarray) -> list:
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


def parse_label(row: pd.Series) -> int:
    """Map multi-label row to single integer label."""
    for col in LABEL_COLUMNS:
        if col in row and row[col] == 1:
            return LABEL_MAP[col]
    return 0  # Normal


# ─── MAIN PROCESSING ──────────────────────────────────────────────────────────

def process_code15(
    hdf5_path: str,
    metadata_path: str,
    output_path: str,
    max_records: Optional[int] = None,
    lead_name: str = 'DII',
) -> None:
    """
    Process CODE-15% HDF5 file into beat-level CSV.

    Args:
        hdf5_path: Path to code15.h5 (HDF5 file with ECG waveforms)
        metadata_path: Path to exams.csv (patient + label metadata)
        output_path: Output CSV path
        max_records: Optional cap on number of recordings processed
        lead_name: ECG lead to extract (default 'DII' = Lead II in CODE-15% naming)
    """
    try:
        import h5py
    except ImportError:
        print("h5py required for CODE-15% processing: pip install h5py")
        return

    hdf5_path = Path(hdf5_path)
    metadata_path = Path(metadata_path)

    if not hdf5_path.exists():
        print(f"  [ERROR] HDF5 file not found: {hdf5_path}")
        print("  Download from: https://zenodo.org/record/4916206/files/code15.h5")
        return

    if not metadata_path.exists():
        print(f"  [ERROR] Metadata CSV not found: {metadata_path}")
        print("  Download from: https://zenodo.org/record/4916206/files/exams.csv")
        return

    print(f"Loading metadata from {metadata_path}...")
    meta_df = pd.read_csv(metadata_path)
    print(f"  Metadata: {len(meta_df):,} rows")

    # Standardize column names (some versions differ)
    meta_df.columns = [c.strip().lower() for c in meta_df.columns]

    # Determine which label columns are available
    available_label_cols = [c for c in LABEL_COLUMNS if c.lower() in meta_df.columns]
    if not available_label_cols:
        # Try alternative naming
        available_label_cols = [c for c in meta_df.columns 
                                   if any(l.lower() in c.lower() for l in LABEL_COLUMNS)]
    
    print(f"  Available label columns: {available_label_cols}")

    # Create exam_id → metadata lookup
    id_col = 'exam_id' if 'exam_id' in meta_df.columns else meta_df.columns[0]
    meta_lookup = {}
    for _, row in meta_df.iterrows():
        eid = str(row[id_col])
        label = 0  # Default normal
        for col in available_label_cols:
            col_norm = col.lower()
            # Find matching column
            matching = [c for c in meta_df.columns if col_norm in c.lower()]
            if matching and row[matching[0]] == 1:
                label = LABEL_MAP.get(col, 0)
                break
        age = float(row.get('age', row.get('patient_age', 60.0)) or 60.0)
        sex_raw = str(row.get('sex', row.get('patient_sex', '')) or '')
        sex = 1.0 if sex_raw.strip().upper().startswith('M') else 0.0
        meta_lookup[eid] = {'label': label, 'age': age, 'sex': sex}

    print(f"Loading ECG waveforms from {hdf5_path}...")
    all_rows = []
    skipped = 0

    with h5py.File(hdf5_path, 'r') as f:
        # Determine available leads
        # CODE-15% structure: f['tracings'][N, T, 12] — N records, T timepoints, 12 leads
        # Or f[exam_id][...] per exam
        
        if 'tracings' in f:
            # Stacked format: (N, 4096, 12)
            tracings = f['tracings']
            exam_ids = f['exam_id'][:] if 'exam_id' in f else np.arange(len(tracings))
            
            n_total = len(tracings)
            if max_records is not None:
                n_process = min(max_records, n_total)
            else:
                n_process = n_total

            print(f"  Processing {n_process:,} / {n_total:,} records...")

            # Standard CODE-15% lead order: DI, DII, DIII, AVR, AVL, AVF, V1-V6
            # Lead II = index 1
            lead_names_code15 = ['DI', 'DII', 'DIII', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            if lead_name in lead_names_code15:
                lead_i = lead_names_code15.index(lead_name)
            else:
                lead_i = 1  # Default Lead II

            for i in tqdm(range(n_process), desc='Processing CODE-15% records'):
                eid = str(exam_ids[i]) if hasattr(exam_ids[i], '__str__') else str(i)
                raw = tracings[i, :, lead_i].astype(np.float32)  # (T,)
                raw = np.nan_to_num(raw)

                # Resample from 400Hz to 100Hz
                target_len = int(len(raw) * TARGET_FS / SOURCE_FS)
                signal = resample(raw, target_len).astype(np.float32)

                # Bandpass filter
                try:
                    signal = bandpass_filter(signal, TARGET_FS, FILTER_LOW, FILTER_HIGH)
                except Exception:
                    pass

                # R-peak detection
                r_peaks = find_r_peaks(signal)
                if len(r_peaks) < 2:
                    skipped += 1
                    continue

                meta = meta_lookup.get(eid, {'label': 0, 'age': 60.0, 'sex': 0.5})
                beats = extract_beats(signal, r_peaks)

                for bi, beat in enumerate(beats):
                    row = {c: beat[c] for c in range(WINDOW_SAMPLES)}
                    row['label'] = meta['label']
                    row['patient_id'] = eid
                    row['record_id'] = eid
                    row['beat_idx'] = bi
                    row['r_peak_pos'] = WINDOW_SAMPLES // 2
                    row['age'] = meta['age']
                    row['sex'] = meta['sex']
                    all_rows.append(row)

        else:
            print("  [WARN] Unexpected HDF5 structure. Expected 'tracings' key.")
            print(f"  Available keys: {list(f.keys())}")
            return

    if not all_rows:
        print("  [WARN] No beats extracted.")
        return

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nSaved {len(df):,} beats from {n_process - skipped:,} records → {output_path}")
    print(f"Skipped: {skipped:,} records")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}")


def process_code15_parts(
    hdf5_dir: str,
    metadata_path: str,
    output_path: str,
    max_records: Optional[int] = None,
    lead_name: str = 'DII',
) -> None:
    """
    Process CODE-15% when downloaded as split HDF5 parts via zenodo_get.
    Expects: exams_part0.hdf5, exams_part1.hdf5, ... in hdf5_dir.
    """
    try:
        import h5py
    except ImportError:
        print("h5py required: pip install h5py")
        return

    import glob
    part_files = sorted(glob.glob(os.path.join(hdf5_dir, 'exams_part*.hdf5')))
    if not part_files:
        print(f"ERROR: No exams_part*.hdf5 files found in {hdf5_dir}")
        return

    print(f"Found {len(part_files)} part files in {hdf5_dir}")
    print(f"Loading metadata from {metadata_path}...")
    meta_df = pd.read_csv(metadata_path)
    meta_df.columns = [c.strip().lower() for c in meta_df.columns]
    available_label_cols = [c for c in LABEL_COLUMNS if c.lower() in meta_df.columns]
    id_col = 'exam_id' if 'exam_id' in meta_df.columns else meta_df.columns[0]
    meta_lookup = {}
    for _, row in meta_df.iterrows():
        eid = str(int(row[id_col]))
        label = 0
        for col in available_label_cols:
            if row.get(col.lower(), 0) == 1:
                label = LABEL_MAP.get(col, 0)
                break
        age = float(row.get('age', row.get('patient_age', 60.0)) or 60.0)
        sex_raw = str(row.get('sex', row.get('patient_sex', '')) or '')
        sex = 1.0 if sex_raw.strip().upper().startswith('M') else 0.0
        meta_lookup[eid] = {'label': label, 'age': age, 'sex': sex}

    lead_names_code15 = ['DI', 'DII', 'DIII', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    lead_i = lead_names_code15.index(lead_name) if lead_name in lead_names_code15 else 1

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    wrote_header = False
    total_beats = 0
    total_records = 0
    total_skipped = 0
    records_processed = 0

    for part_path in part_files:
        part_name = os.path.basename(part_path)
        if max_records is not None and records_processed >= max_records:
            break

        print(f"  Processing {part_name}...")
        all_rows = []
        skipped = 0

        with h5py.File(part_path, 'r') as f:
            tracings = f['tracings']
            exam_ids = f['exam_id'][:]
            n_total = len(tracings)

            if max_records is not None:
                n_process = min(max_records - records_processed, n_total)
            else:
                n_process = n_total

            for i in tqdm(range(n_process), desc=f'  {part_name}', leave=False):
                eid = str(int(exam_ids[i]))
                raw = tracings[i, :, lead_i].astype(np.float32)
                raw = np.nan_to_num(raw)

                target_len = int(len(raw) * TARGET_FS / SOURCE_FS)
                signal = resample(raw, target_len).astype(np.float32)

                try:
                    signal = bandpass_filter(signal, TARGET_FS, FILTER_LOW, FILTER_HIGH)
                except Exception:
                    pass

                r_peaks = find_r_peaks(signal)
                if len(r_peaks) < 2:
                    skipped += 1
                    continue

                meta = meta_lookup.get(eid, {'label': 0, 'age': 60.0, 'sex': 0.5})
                beats = extract_beats(signal, r_peaks)

                for bi, beat in enumerate(beats):
                    row = {c: beat[c] for c in range(WINDOW_SAMPLES)}
                    row['label']      = meta['label']
                    row['patient_id'] = eid
                    row['record_id']  = eid
                    row['beat_idx']   = bi
                    row['r_peak_pos'] = WINDOW_SAMPLES // 2
                    row['age']        = meta['age']
                    row['sex']        = meta['sex']
                    all_rows.append(row)

        records_processed += n_process
        total_skipped    += skipped

        if all_rows:
            part_df = pd.DataFrame(all_rows)
            part_df.to_csv(output_path, mode='a', header=not wrote_header, index=False)
            wrote_header = True
            total_beats   += len(part_df)
            total_records += (n_process - skipped)
            print(f"    → {len(part_df):,} beats | running total: {total_beats:,}")

    print(f"\n{'='*60}")
    print(f"CODE-15% Processing Complete")
    print(f"  Total beats:   {total_beats:,}")
    print(f"  Records used:  {total_records:,}")
    print(f"  Skipped:       {total_skipped:,}")
    print(f"  Saved to:      {output_path}")
    print(f"{'='*60}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Process CODE-15% ECG dataset (Ribeiro et al., Nature Communications 2021)'
    )
    parser.add_argument('--hdf5', type=str, default=None,
                        help='Path to single CODE-15% HDF5 file (legacy single-file format)')
    parser.add_argument('--hdf5_dir', type=str, default=None,
                        help='Path to directory containing exams_part*.hdf5 files (Zenodo multi-part format)')
    parser.add_argument('--metadata', type=str, default='data/code15/exams.csv',
                        help='Path to CODE-15% exams.csv metadata')
    parser.add_argument('--output', type=str, default='data/code15_processed.csv',
                        help='Output CSV path')
    parser.add_argument('--max_records', type=int, default=None,
                        help='Maximum number of records to process (default: all ~345K)')
    parser.add_argument('--lead', type=str, default='DII',
                        help='ECG lead to extract (default: DII = Lead II)')
    args = parser.parse_args()

    if args.hdf5_dir:
        print(f"Processing CODE-15% (multi-part HDF5 format)")
        print(f"  HDF5 dir: {args.hdf5_dir}")
        print(f"  Metadata: {args.metadata}")
        print(f"  Output:   {args.output}")
        if args.max_records:
            print(f"  Max records: {args.max_records:,}")
        process_code15_parts(
            hdf5_dir=args.hdf5_dir,
            metadata_path=args.metadata,
            output_path=args.output,
            max_records=args.max_records,
            lead_name=args.lead,
        )
    elif args.hdf5:
        print(f"Processing CODE-15% ECG dataset")
        print(f"  HDF5: {args.hdf5}")
        print(f"  Metadata: {args.metadata}")
        print(f"  Output: {args.output}")
        if args.max_records:
            print(f"  Max records: {args.max_records:,}")
        process_code15(
            hdf5_path=args.hdf5,
            metadata_path=args.metadata,
            output_path=args.output,
            max_records=args.max_records,
            lead_name=args.lead,
        )
    else:
        # Auto-detect
        if os.path.exists('data/code15/extracted'):
            process_code15_parts(
                hdf5_dir='data/code15/extracted',
                metadata_path=args.metadata,
                output_path=args.output,
                max_records=args.max_records,
                lead_name=args.lead,
            )
        elif os.path.exists('data/code15/code15.h5'):
            process_code15(
                hdf5_path='data/code15/code15.h5',
                metadata_path=args.metadata,
                output_path=args.output,
                max_records=args.max_records,
                lead_name=args.lead,
            )
        else:
            print("ERROR: No CODE-15% data found. Use --hdf5_dir or --hdf5.")


if __name__ == '__main__':
    main()
