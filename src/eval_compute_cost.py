"""
PA-SSL: Computational Cost Benchmark  (Table 9)
================================================
Measures inference latency (ms) and GPU memory (MB) for each encoder at
batch_size=128 over 1000 forward passes, then reports mean ± std.

Usage (run from project root on server):
    python -m src.eval_compute_cost \
        --encoders resnet1d wavkan \
        --n_runs   1000 \
        --batch    128 \
        --output   results/compute_cost.csv \
        2>&1 | tee logs/eval_compute.log
"""

import argparse
import time
import os
import numpy as np
import pandas as pd
import torch

from src.models.encoder import build_encoder


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark(encoder_name, n_runs, batch_size, device):
    encoder = build_encoder(encoder_name, proj_dim=128).to(device)
    encoder.eval()

    n_params = count_params(encoder)
    dummy = torch.randn(batch_size, 1, 250, device=device)

    # ── warm-up (5 runs not counted) ─────────────────────────────────────────
    with torch.no_grad():
        for _ in range(5):
            _ = encoder.encode(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # ── measure latency ───────────────────────────────────────────────────────
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = encoder.encode(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)   # ms

    latency_mean = float(np.mean(latencies))
    latency_std  = float(np.std(latencies))

    # ── measure GPU memory ────────────────────────────────────────────────────
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = encoder.encode(dummy)
        mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    else:
        mem_mb = float("nan")

    return {
        "encoder":       encoder_name,
        "n_params_M":    round(n_params / 1e6, 2),
        "mem_MB":        round(mem_mb, 2),
        "latency_mean":  round(latency_mean, 1),
        "latency_std":   round(latency_std, 1),
    }


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Batch  : {args.batch}  |  Runs: {args.n_runs}\n")

    rows = []
    for enc in args.encoders:
        print(f"Benchmarking {enc} ...")
        row = benchmark(enc, args.n_runs, args.batch, device)
        print(f"  Params: {row['n_params_M']}M  |  "
              f"Mem: {row['mem_MB']} MB  |  "
              f"Latency: {row['latency_mean']} ± {row['latency_std']} ms")
        rows.append(row)

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved → {args.output}")
    print(df.to_string(index=False))

    # ── print ready-to-paste LaTeX row ────────────────────────────────────────
    print("\n--- LaTeX rows (copy into Table 9) ---")
    for _, r in df.iterrows():
        print(f"{r['encoder']} & {r['n_params_M']}M & "
              f"{r['mem_MB']} & "
              f"${r['latency_mean']} \\pm {r['latency_std']}$ \\\\")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoders", nargs="+", default=["resnet1d", "wavkan"])
    parser.add_argument("--n_runs",   type=int,   default=1000)
    parser.add_argument("--batch",    type=int,   default=128)
    parser.add_argument("--output",   default="results/compute_cost.csv")
    main(parser.parse_args())
