#!/usr/bin/env python3
"""
combine_datasets.py
Merges PTB-XL (train+val only) and CODE-15% into a single pretraining corpus.
PTB-XL test patients are excluded using the same patient_aware_split(seed=42)
that evaluate.py uses — guaranteeing zero transductive leakage.

Output: data/combined_pretrain.csv
"""

import pandas as pd
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ecg_dataset import patient_aware_split

COMMON_COLS = [str(i) for i in range(250)] + [
    "label", "patient_id", "record_id", "beat_idx", "r_peak_pos", "age", "sex"
]
CHUNK_SIZE = 50_000


def stream_csv(path: str, source_tag: str, output_handle, write_header: bool) -> int:
    rows_written = 0
    for i, chunk in enumerate(pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False)):
        missing = [c for c in COMMON_COLS if c not in chunk.columns]
        if missing:
            print(f"ERROR: {path} missing columns: {missing}", file=sys.stderr)
            sys.exit(1)
        chunk = chunk[COMMON_COLS].copy()
        chunk["source"] = source_tag
        chunk.to_csv(output_handle, header=(write_header and i == 0), index=False)
        rows_written += len(chunk)
        if (i + 1) % 20 == 0:
            print(f"  [{source_tag}] {rows_written:,} rows written...", flush=True)
    return rows_written


def main():
    ptbxl_path  = PROJECT_ROOT / "data" / "ptbxl_processed.csv"
    code15_path = PROJECT_ROOT / "data" / "code15_processed.csv"
    out_path    = PROJECT_ROOT / "data" / "combined_pretrain.csv"

    for p in [ptbxl_path, code15_path]:
        if not p.exists():
            print(f"ERROR: not found: {p}", file=sys.stderr)
            sys.exit(1)

    # --- PTB-XL: exclude test patients (seed=42 matches evaluate.py exactly) ---
    print("Splitting PTB-XL with patient_aware_split(seed=42)...")
    train_df, val_df, test_df = patient_aware_split(str(ptbxl_path), seed=42)

    test_patients = set(test_df['patient_id'].unique())
    print(f"  Excluded {len(test_patients):,} test patients "
          f"({len(test_df):,} beats) from pretraining corpus.")

    ptbxl_safe = pd.concat([train_df, val_df], ignore_index=True)
    ptbxl_safe = ptbxl_safe[COMMON_COLS].copy()
    ptbxl_safe["source"] = "ptbxl"
    print(f"  PTB-XL train+val: {len(ptbxl_safe):,} beats, "
          f"{ptbxl_safe['patient_id'].nunique():,} patients")

    # Sanity check: no test patient leakage
    pretrain_patients = set(ptbxl_safe['patient_id'].unique())
    overlap = pretrain_patients & test_patients
    assert len(overlap) == 0, f"FATAL: {len(overlap)} test patients in pretraining corpus!"
    print("  ✓ Zero test patient leakage confirmed.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    with open(out_path, "w") as fout:
        # Write PTB-XL safe subset (already in memory, small enough)
        ptbxl_safe.to_csv(fout, header=True, index=False)
        total += len(ptbxl_safe)
        print(f"  PTB-XL written: {total:,} rows")

        # Stream CODE-15% in chunks
        print("Streaming CODE-15%...")
        n = stream_csv(str(code15_path), "code15", fout, write_header=False)
        total += n
        print(f"  CODE-15% done: {n:,} rows")

    print(f"\nCombined corpus: {total:,} rows → {out_path}")
    print("Note: DataLoader shuffle=True handles ordering. --shuffle not required.")


if __name__ == "__main__":
    main()
