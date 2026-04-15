#!/usr/bin/env python3
"""
combine_datasets.py
Merges PTB-XL and CODE-15% preprocessed CSVs into a single pretraining corpus.
Uses chunked I/O to avoid loading 11GB+ into RAM simultaneously.

Output: data/combined_pretrain.csv
Columns: 0..249 (signal), label, patient_id, record_id, beat_idx, r_peak_pos, age, sex, source
"""

import pandas as pd
import argparse
import sys
from pathlib import Path

# Columns present in both datasets
COMMON_COLS = [str(i) for i in range(250)] + [
    "label", "patient_id", "record_id", "beat_idx", "r_peak_pos", "age", "sex"
]

CHUNK_SIZE = 50_000  # rows per chunk


def stream_csv(path: str, source_tag: str, output_handle, write_header: bool) -> int:
    """Stream a CSV in chunks, selecting COMMON_COLS, tagging source, writing to output."""
    rows_written = 0
    for i, chunk in enumerate(pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False)):
        # Keep only common columns
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptbxl",    default="data/ptbxl_processed.csv")
    parser.add_argument("--code15",   default="data/code15_processed.csv")
    parser.add_argument("--output",   default="data/combined_pretrain.csv")
    parser.add_argument("--shuffle",  action="store_true",
                        help="Shuffle final output (loads everything into RAM — skip if low memory)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    ptbxl_path  = project_root / args.ptbxl
    code15_path = project_root / args.code15
    out_path    = project_root / args.output

    for p in [ptbxl_path, code15_path]:
        if not p.exists():
            print(f"ERROR: not found: {p}", file=sys.stderr)
            sys.exit(1)

    print(f"Output: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(out_path, "w") as fout:
        print("Streaming PTB-XL...")
        n = stream_csv(str(ptbxl_path), "ptbxl", fout, write_header=True)
        print(f"  PTB-XL done: {n:,} rows")
        total += n

        print("Streaming CODE-15%...")
        n = stream_csv(str(code15_path), "code15", fout, write_header=False)
        print(f"  CODE-15% done: {n:,} rows")
        total += n

    print(f"\nCombined corpus: {total:,} rows → {out_path}")

    if args.shuffle:
        print("Shuffling (loading into RAM)...")
        df = pd.read_csv(out_path, low_memory=False)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df.to_csv(out_path, index=False)
        print("Shuffle complete.")
    else:
        print("Note: output is PTB-XL rows first, then CODE-15%. Pass --shuffle to randomize.")
        print("For SSL pretraining, DataLoader shuffle=True handles this — --shuffle not required.")


if __name__ == "__main__":
    main()
