"""
PA-SSL: Computational Cost Benchmarking

Measures FLOPs, latency, memory, and throughput for each encoder.
Required for the T-BME reviewer table.

Usage:
    python3 -m src.experiments.computational_cost \
        --output results/compute_cost.csv
"""

import argparse
import time
import os

import numpy as np
import pandas as pd
import torch

from src.models.encoder import build_encoder


def benchmark_encoder(encoder, input_shape=(1, 1, 250), n_warmup=50, n_runs=500, device='cuda'):
    """
    Measure latency, memory, throughput, and parameters for one encoder.

    Args:
        encoder: nn.Module (projection head excluded is fine — just the backbone)
        input_shape: (B, C, T) — default is single-beat (batch=1, 1 channel, 250 samples)
        n_warmup: warmup iterations before timing
        n_runs: timed iterations
        device: 'cuda' or 'cpu'

    Returns dict with metrics.
    """
    encoder = encoder.to(device).eval()
    dummy = torch.randn(*input_shape).to(device)

    # ── Parameters ──────────────────────────────────────────────────────────
    total_params     = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

    # ── Warmup ───────────────────────────────────────────────────────────────
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = encoder(dummy)
    if device == 'cuda':
        torch.cuda.synchronize()

    # ── Latency ──────────────────────────────────────────────────────────────
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = encoder(dummy)
            if device == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)   # ms

    times_arr = np.array(times)
    lat_mean  = float(times_arr.mean())
    lat_p50   = float(np.percentile(times_arr, 50))
    lat_p95   = float(np.percentile(times_arr, 95))
    lat_p99   = float(np.percentile(times_arr, 99))

    # ── Memory ───────────────────────────────────────────────────────────────
    peak_mem_mb = None
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = encoder(dummy)
        torch.cuda.synchronize()
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6

    # ── FLOPs (optional — requires thop) ─────────────────────────────────────
    flops_g = None
    try:
        from thop import profile as thop_profile
        flops, _ = thop_profile(encoder, inputs=(dummy,), verbose=False)
        flops_g = flops / 1e9
    except (ImportError, Exception):
        pass   # thop is optional

    throughput = 1000.0 / lat_mean   # samples / sec (batch=1)

    return {
        'params_M':           round(total_params / 1e6, 3),
        'trainable_M':        round(trainable_params / 1e6, 3),
        'latency_mean_ms':    round(lat_mean, 3),
        'latency_p50_ms':     round(lat_p50, 3),
        'latency_p95_ms':     round(lat_p95, 3),
        'latency_p99_ms':     round(lat_p99, 3),
        'throughput_sps':     round(throughput, 1),
        'peak_memory_MB':     round(peak_mem_mb, 1) if peak_mem_mb is not None else None,
        'flops_G':            round(flops_g, 3) if flops_g is not None else None,
    }


def run_benchmark(output_path='results/compute_cost.csv', device=None, proj_dim=128):
    """Benchmark all encoders and save results to CSV."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    encoders = ['resnet1d', 'wavkan', 'transformer', 'mamba']
    results  = []

    print(f"Benchmarking on: {device.upper()}")
    print(f"{'Encoder':<15} {'Params':>8} {'Latency':>10} {'P95':>8} {'Memory':>10} {'FLOPs':>8}")
    print('─' * 65)

    for enc_name in encoders:
        try:
            encoder = build_encoder(enc_name, proj_dim=proj_dim)
            metrics = benchmark_encoder(encoder, device=device)
            metrics['encoder']   = enc_name
            metrics['device']    = device
            results.append(metrics)

            flops_str  = f"{metrics['flops_G']:.3f}G" if metrics['flops_G'] else 'N/A'
            mem_str    = f"{metrics['peak_memory_MB']:.0f} MB" if metrics['peak_memory_MB'] else 'N/A'
            print(
                f"{enc_name:<15} {metrics['params_M']:>6.3f}M "
                f"{metrics['latency_mean_ms']:>8.2f} ms "
                f"{metrics['latency_p95_ms']:>6.2f} ms "
                f"{mem_str:>10} "
                f"{flops_str:>8}"
            )

            del encoder
            if device == 'cuda':
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"{enc_name:<15}  ERROR: {e}")

    print('─' * 65)

    if results:
        df = pd.DataFrame(results)
        cols = ['encoder', 'params_M', 'trainable_M', 'latency_mean_ms',
                'latency_p50_ms', 'latency_p95_ms', 'latency_p99_ms',
                'throughput_sps', 'peak_memory_MB', 'flops_G', 'device']
        df = df[[c for c in cols if c in df.columns]]
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark PA-SSL encoders (FLOPs, latency, memory)')
    parser.add_argument('--output',   type=str, default='results/compute_cost.csv')
    parser.add_argument('--device',   type=str, default=None,
                        help='cuda or cpu (auto-detect if omitted)')
    parser.add_argument('--proj_dim', type=int, default=128)
    parser.add_argument('--n_runs',   type=int, default=500,
                        help='Number of timed forward passes per encoder')
    args = parser.parse_args()
    run_benchmark(args.output, args.device, args.proj_dim)


if __name__ == '__main__':
    main()
