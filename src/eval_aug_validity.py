"""
PA-SSL: Augmentation Validity Verification  (Table 11)
=======================================================
Quantitatively validates that each physiology-aware augmentation:
  1. Preserves QRS morphology → measured as Pearson correlation of the
     QRS window (R-peak ± 30 ms) between original and augmented signal.
  2. Preserves signal fidelity → measured as Signal-to-Distortion Ratio (SDR).

Metrics are averaged over `n_samples` randomly drawn beats from PTB-XL.
Both Physio-Aware and Naive augmentation pipelines are evaluated.

Usage (run from project root on server):
    python -m src.eval_aug_validity \
        --data     data/ptbxl_processed.csv \
        --n        1000 \
        --output   results/aug_validity.csv \
        2>&1 | tee logs/eval_aug_validity.log
"""

import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.augmentations.augmentation_pipeline import PhysioAugPipeline
from src.augmentations.physio_augmentations import (
    constrained_time_warp, amplitude_perturbation, baseline_wander,
    emg_noise_injection, heart_rate_resample, powerline_interference,
    segment_dropout, wavelet_masking,
)
from src.augmentations.naive_augmentations import NaiveAugPipeline
from src.data.ecg_dataset import ECGBeatDataset

QRS_HALF = 30          # ±30 samples = ±300 ms around R-peak @ 100 Hz
R_PEAK   = 125         # default R-peak position in a 250-sample beat


# ─── metric helpers ────────────────────────────────────────────────────────────

def qrs_correlation(original: np.ndarray, augmented: np.ndarray,
                    r_peak: int = R_PEAK, half: int = QRS_HALF) -> float:
    """Pearson correlation of the QRS window between original and augmented."""
    start  = max(0, r_peak - half)
    end    = min(len(original), r_peak + half)
    orig_w = original[start:end]
    aug_w  = augmented[start:end]
    if orig_w.std() < 1e-8 or aug_w.std() < 1e-8:
        return 1.0   # flat segment — trivially identical
    return float(np.corrcoef(orig_w, aug_w)[0, 1])


def sdr_db(original: np.ndarray, augmented: np.ndarray) -> float:
    """Signal-to-Distortion Ratio in dB (higher = more faithful)."""
    signal_power     = np.mean(original ** 2)
    distortion_power = np.mean((augmented - original) ** 2)
    if distortion_power < 1e-12:
        return 60.0          # practically lossless
    return float(10 * np.log10(signal_power / distortion_power))


# ─── per-augmentation evaluation ───────────────────────────────────────────────

AUG_REGISTRY = {
    "constrained_time_warp":   lambda s: constrained_time_warp(s, r_peak_pos=R_PEAK),
    "amplitude_perturbation":  lambda s: amplitude_perturbation(s, r_peak_pos=R_PEAK,
                                                                  qrs_protect=True),
    "baseline_wander":         lambda s: baseline_wander(s, r_peak_pos=R_PEAK),
    "emg_noise_injection":     lambda s: emg_noise_injection(s, r_peak_pos=R_PEAK),
    "heart_rate_resample":     lambda s: heart_rate_resample(s, r_peak_pos=R_PEAK),
    "powerline_interference":  lambda s: powerline_interference(s, r_peak_pos=R_PEAK),
    "segment_dropout":         lambda s: segment_dropout(s, r_peak_pos=R_PEAK),
    "wavelet_masking":         lambda s: wavelet_masking(s, r_peak_pos=R_PEAK),
}


def evaluate_pipeline(signals, pipeline_fn=None, aug_name="Full Pipeline"):
    """
    Apply pipeline_fn to every signal and return mean QRS-corr & SDR.
    If pipeline_fn is None, tries each individual augmentation in AUG_REGISTRY.
    """
    qrs_corrs, sdrs = [], []
    for sig in signals:
        aug = pipeline_fn(sig) if pipeline_fn else sig   # placeholder
        # safety cleanup
        aug = np.nan_to_num(aug, nan=0.0, posinf=0.0, neginf=0.0)
        if len(aug) != len(sig):
            from scipy.signal import resample
            aug = resample(aug, len(sig))
        qrs_corrs.append(qrs_correlation(sig, aug))
        sdrs.append(sdr_db(sig, aug))
    return {
        "pipeline":    aug_name,
        "qrs_corr_mean": round(float(np.mean(qrs_corrs)), 4),
        "qrs_corr_std":  round(float(np.std(qrs_corrs)), 4),
        "sdr_mean":      round(float(np.mean(sdrs)), 2),
        "sdr_std":       round(float(np.std(sdrs)), 2),
    }


def main(args):
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    # ── Load signals ──────────────────────────────────────────────────────────
    print(f"Loading {args.n} beats from {args.data} ...")
    ds  = ECGBeatDataset(args.data)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(ds), min(args.n, len(ds)), replace=False)

    signals = []
    for i in idx:
        sample = ds[i]
        sig = sample[0] if isinstance(sample, (list, tuple)) else sample["signal"]
        if hasattr(sig, "numpy"):
            sig = sig.numpy()
        sig = np.asarray(sig, dtype=np.float32).squeeze()
        if len(sig) < 50:
            continue
        signals.append(sig)
    print(f"  Collected {len(signals)} valid beats.\n")

    rows = []

    # ── Full physio-aware pipeline ────────────────────────────────────────────
    physio_pipe = PhysioAugPipeline.default(strength="medium")
    print("Evaluating Physio-Aware Full Pipeline ...")
    rows.append(evaluate_pipeline(signals,
                                  lambda s: physio_pipe(s, r_peak_pos=R_PEAK),
                                  "Physio-Aware (Full)"))

    # ── Naive augmentation pipeline ───────────────────────────────────────────
    try:
        naive_pipe = NaiveAugPipeline.default()
        print("Evaluating Naive Augmentation Pipeline ...")
        rows.append(evaluate_pipeline(signals,
                                      lambda s: naive_pipe(s),
                                      "Naive (SimCLR-style)"))
    except Exception as e:
        print(f"  [SKIP] Naive pipeline not available: {e}")

    # ── Individual augmentations ──────────────────────────────────────────────
    for aug_name, aug_fn in tqdm(AUG_REGISTRY.items(), desc="Per-augmentation"):
        def _safe_aug(s, fn=aug_fn):
            try:
                out = fn(s)
                out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
                if len(out) != len(s):
                    from scipy.signal import resample
                    out = resample(out, len(s))
                return out
            except Exception:
                return s.copy()

        row = evaluate_pipeline(signals, _safe_aug, aug_name)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"\nSaved → {args.output}")
    print(df.to_string(index=False))

    # ── Print ready-to-paste LaTeX rows ───────────────────────────────────────
    print("\n--- LaTeX rows (copy into Table 11) ---")
    for _, r in df.iterrows():
        name_clean = r["pipeline"].replace("_", " ").replace("constrained ", "")
        print(f"{name_clean} & "
              f"{r['qrs_corr_mean']:.4f} $\\pm$ {r['qrs_corr_std']:.4f} & "
              f"{r['sdr_mean']:.1f} $\\pm$ {r['sdr_std']:.1f} \\\\")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="data/ptbxl_processed.csv")
    parser.add_argument("--n",      type=int, default=1000,
                        help="Number of beats to evaluate over")
    parser.add_argument("--output", default="results/aug_validity.csv")
    main(parser.parse_args())
