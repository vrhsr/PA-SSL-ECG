"""
PA-SSL: Scaling Laws Experiment
================================
Demonstrates that PA-HybridSSL performance scales log-linearly with
pretraining corpus size — the central empirical result supporting the
foundation-model framing of this work.

Experimental protocol (per scale N):
    1. Sample N records uniformly from FoundationECGCorpus (seeded)
    2. Pretrain PA-HybridSSL for E epochs (contrastive + MAE objectives)
    3. Freeze encoder; fit logistic regression on n_labels labeled examples
       drawn from the PTB-XL evaluation partition
    4. Report macro-AUROC and macro-F1 on the held-out PTB-XL test partition

Scales evaluated (default):
    N ∈ {1 K, 10 K, 50 K, 100 K, 500 K, 1 M, 2 M}

Outputs:
    <output_dir>/scaling_results.csv    — machine-readable tabular results
    <output_dir>/scaling_law_plot.png   — publication-quality log-linear figure
    <output_dir>/encoder_scale<N>.pth   — per-scale encoder checkpoints

Reproducibility:
    All random operations are seeded via --seed (default 42).
    Checkpoints allow resumption without re-training completed scales.

Usage
-----
Full experiment (GPU required; estimate 3–7 days):
    python -m src.experiments.scaling_laws \\
        --data_root  data/ \\
        --eval_csv   data/ptbxl_processed.csv \\
        --encoder    resnet1d \\
        --epochs     100 \\
        --n_labels   100 \\
        --output_dir experiments/scaling_laws

Smoke test (2 scales, 1 epoch, ~5 min):
    python -m src.experiments.scaling_laws \\
        --data_root  data/ \\
        --eval_csv   data/ptbxl_processed.csv \\
        --encoder    resnet1d \\
        --epochs     1 \\
        --scales     1000 10000 \\
        --output_dir experiments/scaling_laws_smoke
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

from src.data.ecg_dataset import ECGBeatDataset
from src.data.foundation_corpus import FoundationECGCorpus
from src.losses import CombinedContrastiveLoss
from src.models.encoder import build_encoder
from src.models.mae import HybridMAE

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SCALES: List[int] = [
    1_000,
    10_000,
    50_000,   # current paper baseline
    100_000,
    500_000,
    1_000_000,
    2_000_000,
]

_DIVIDER_MAJOR = "=" * 68
_DIVIDER_MINOR = "─" * 68


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ScaleResult:
    """Holds all measurements for a single pretraining scale."""

    n_records_requested: int
    n_records_actual: int
    encoder: str
    epochs: int
    n_labels: int
    auroc: float
    macro_f1: float
    n_train_labeled: int
    n_test: int
    n_classes: int
    pretrain_wall_seconds: float = 0.0
    eval_wall_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Pre-training helpers
# ---------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_mae_loss(
    recon: torch.Tensor,
    original: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """
    Patch-level mean-squared reconstruction loss.

    Args:
        recon:    (B, C, L) — decoder output.
        original: (B, C, L) — masked input passed through the model.
        masks:    (B, P)    — boolean mask; True = patch was masked.

    Returns:
        Scalar loss (0 when no patches are masked).
    """
    if not masks.any():
        return torch.tensor(0.0, device=recon.device)

    B, C, L = recon.shape
    n_patches = masks.shape[1]
    patch_len = L // n_patches  # assume equal-length patches

    # Reshape to (B, n_patches, patch_len * C)
    recon_p = recon.permute(0, 2, 1).reshape(B, n_patches, patch_len * C)
    orig_p  = original.permute(0, 2, 1).reshape(B, n_patches, patch_len * C)

    # Select only masked patches
    loss = nn.functional.mse_loss(recon_p[masks], orig_p[masks])
    return loss


def _run_one_epoch(
    encoder:   nn.Module,
    hybrid:    HybridMAE,
    loss_fn:   CombinedContrastiveLoss,
    optimizer: optim.Optimizer,
    loader:    DataLoader,
    device:    torch.device,
    mae_weight: float = 0.1,
) -> float:
    """Train for one epoch; return mean total loss."""
    hybrid.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        view1: torch.Tensor = batch["view1"].to(device, non_blocking=True)
        view2: torch.Tensor = batch["view2"].to(device, non_blocking=True)

        # Ensure shape (B, 1, L)
        if view1.dim() == 2:
            view1 = view1.unsqueeze(1)
            view2 = view2.unsqueeze(1)

        # ── Contrastive objective ──────────────────────────────────────────
        z1 = encoder(view1, return_projection=True)
        z2 = encoder(view2, return_projection=True)
        contrastive_loss, _, _ = loss_fn(z1, z2)

        # ── MAE objective ──────────────────────────────────────────────────
        recon, masks, _ = hybrid.forward_mae(view1)
        mae_loss = _compute_mae_loss(recon, view1, masks)

        loss = contrastive_loss + mae_weight * mae_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(hybrid.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


def pretrain_at_scale(
    corpus:       FoundationECGCorpus,
    encoder_name: str,
    n_records:    int,
    epochs:       int,
    batch_size:   int,
    lr:           float,
    device:       torch.device,
    output_dir:   Path,
    seed:         int = 42,
    mae_weight:   float = 0.1,
    num_workers:  Optional[int] = None,
) -> Tuple[nn.Module, float]:
    """
    Pretrain PA-HybridSSL on *corpus* (capped at *n_records* records).

    Checkpoints are written to *output_dir/encoder_scale<n_records>.pth*
    after every 20 % of training and at completion so that runs can be
    resumed from the most recent checkpoint.

    Returns:
        encoder:           Trained (CPU-resident) encoder module.
        wall_time_seconds: Total wall-clock training time.
    """
    _seed_everything(seed)

    workers = num_workers if num_workers is not None else min(4, os.cpu_count() or 1)

    encoder = build_encoder(encoder_name, proj_dim=128)
    hybrid  = HybridMAE(encoder, repr_dim=256, mask_ratio=0.15)
    hybrid  = hybrid.to(device)

    loss_fn   = CombinedContrastiveLoss(
        temperature=0.1, alpha=1.0, beta=0.5, loss_type="ntxent"
    )
    optimizer = optim.AdamW(hybrid.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loader = DataLoader(
        corpus,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(workers > 0),
    )

    checkpoint_path = output_dir / f"encoder_scale{n_records}.pth"
    log_every = max(1, epochs // 5)   # log ~5 times per run

    log.info(_DIVIDER_MINOR)
    log.info(
        "Pretraining | scale=%s | encoder=%s | epochs=%d | "
        "batches/epoch=%d | device=%s",
        f"{n_records:,}", encoder_name, epochs, len(loader), device,
    )
    log.info(_DIVIDER_MINOR)

    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        avg_loss = _run_one_epoch(
            encoder, hybrid, loss_fn, optimizer, loader, device, mae_weight
        )
        scheduler.step()

        if epoch % log_every == 0 or epoch == epochs:
            elapsed_min = (time.perf_counter() - t0) / 60
            log.info(
                "  epoch %3d/%d | loss %.4f | elapsed %.1f min",
                epoch, epochs, avg_loss, elapsed_min,
            )

        # Intermediate checkpoint every 20 % of training
        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            torch.save(
                {
                    "encoder_state_dict": encoder.state_dict(),
                    "epoch":              epoch,
                    "n_records":          n_records,
                    "encoder_name":       encoder_name,
                    "avg_loss":           avg_loss,
                },
                checkpoint_path,
            )

    wall_seconds = time.perf_counter() - t0
    log.info("Checkpoint saved → %s", checkpoint_path)

    encoder = encoder.cpu()
    return encoder, wall_seconds


# ---------------------------------------------------------------------------
# Linear-probe evaluation
# ---------------------------------------------------------------------------

def evaluate_linear_probe(
    encoder:  nn.Module,
    eval_csv: str,
    device:   torch.device,
    n_labels: int = 100,
    seed:     int = 42,
) -> Tuple[dict, float]:
    """
    Frozen-encoder linear-probe evaluation.

    Protocol:
        • Extract representations for every sample in *eval_csv*.
        • Reserve *n_labels* labeled examples (stratified) for logistic
          regression training; evaluate on the remainder.
        • Report macro-AUROC (OvR) and macro-F1.

    Returns:
        metrics dict, wall_time_seconds
    """
    from src.evaluate import extract_representations  # local import keeps top-level clean

    t0      = time.perf_counter()
    dataset = ECGBeatDataset(eval_csv)

    encoder = encoder.to(device)
    encoder.eval()

    log.info("  Extracting representations from %d samples …", len(dataset))
    reprs, labels = extract_representations(encoder, dataset, device)
    encoder = encoder.cpu()   # free GPU memory early

    rng       = np.random.RandomState(seed)
    classes   = np.unique(labels)
    n_classes = len(classes)
    per_class = max(1, n_labels // n_classes)

    # ── Stratified label split ─────────────────────────────────────────────
    train_idx: List[int] = []
    test_idx:  List[int] = []

    for cls in classes:
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        train_idx.extend(idx[:per_class].tolist())
        test_idx.extend(idx[per_class:].tolist())

    if len(test_idx) == 0:
        log.warning(
            "Stratified split produced an empty test set "
            "(dataset too small for %d labels). Falling back to 80/20 split.",
            n_labels,
        )
        idx     = rng.permutation(len(reprs))
        n_train = max(n_labels, int(0.2 * len(reprs)))
        train_idx = idx[:n_train].tolist()
        test_idx  = idx[n_train:].tolist()

    train_idx = np.array(train_idx)
    test_idx  = np.array(test_idx)

    X_train, y_train = reprs[train_idx], labels[train_idx]
    X_test,  y_test  = reprs[test_idx],  labels[test_idx]

    log.info("  Linear probe | train=%d | test=%d | classes=%d",
             len(X_train), len(X_test), n_classes)

    # ── Logistic regression ────────────────────────────────────────────────
    clf = LogisticRegression(
        max_iter=2_000,
        C=1.0,
        solver="lbfgs",
        multi_class="auto",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    # ── Metrics ───────────────────────────────────────────────────────────
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    try:
        if n_classes == 2:
            auroc = float(roc_auc_score(y_test, y_proba[:, 1]))
        else:
            auroc = float(
                roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
            )
    except ValueError as exc:
        log.warning("AUROC computation failed (%s); recording NaN.", exc)
        auroc = float("nan")

    wall_seconds = time.perf_counter() - t0

    metrics = {
        "auroc":           auroc,
        "macro_f1":        macro_f1,
        "n_train_labeled": int(len(X_train)),
        "n_test":          int(len(X_test)),
        "n_classes":       int(n_classes),
    }
    return metrics, wall_seconds


# ---------------------------------------------------------------------------
# Main experiment driver
# ---------------------------------------------------------------------------

def run_scaling_experiment(
    data_root:    str,
    eval_csv:     str,
    encoder_name: str              = "resnet1d",
    scales:       Optional[List[int]] = None,
    epochs:       int              = 100,
    batch_size:   int              = 256,
    lr:           float            = 3e-4,
    n_labels:     int              = 100,
    output_dir:   str              = "experiments/scaling_laws",
    seed:         int              = 42,
    mae_weight:   float            = 0.1,
    num_workers:  Optional[int]    = None,
) -> List[ScaleResult]:
    """
    Orchestrate the full scaling-law experiment.

    Checkpoints and intermediate CSV results are written after each scale so
    the experiment can be safely interrupted and resumed.

    Returns:
        List of :class:`ScaleResult` instances (one per completed scale).
    """
    if scales is None:
        scales = DEFAULT_SCALES

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_path = out / "scaling_results.csv"

    log.info(_DIVIDER_MAJOR)
    log.info("PA-HybridSSL  ·  Scaling Law Experiment")
    log.info(_DIVIDER_MAJOR)
    log.info("  encoder     : %s",  encoder_name)
    log.info("  scales      : %s",  [f"{n:,}" for n in scales])
    log.info("  epochs/scale: %d",  epochs)
    log.info("  n_labels    : %d",  n_labels)
    log.info("  device      : %s",  device)
    log.info("  output_dir  : %s",  out)
    log.info(_DIVIDER_MAJOR)

    # ── Load any results already on disk (enables resume) ─────────────────
    completed: dict[int, ScaleResult] = {}
    if results_path.exists():
        prev_df = pd.read_csv(results_path)
        for _, row in prev_df.iterrows():
            r = ScaleResult(**{k: row[k] for k in ScaleResult.__dataclass_fields__})
            completed[r.n_records_requested] = r
        log.info("Resuming: %d scale(s) already completed.", len(completed))

    all_results: List[ScaleResult] = list(completed.values())

    for n_records in scales:
        if n_records in completed:
            log.info("Scale %s already done — skipping.", f"{n_records:,}")
            continue

        log.info("")
        log.info(_DIVIDER_MAJOR)
        log.info("Scale: %s pretraining records", f"{n_records:,}")
        log.info(_DIVIDER_MAJOR)

        checkpoint_path = out / f"encoder_scale{n_records}.pth"
        pretrain_wall   = 0.0

        # ── Load or train encoder ──────────────────────────────────────────
        if checkpoint_path.exists():
            log.info("Found checkpoint %s — skipping pretraining.", checkpoint_path)
            encoder = build_encoder(encoder_name, proj_dim=128)
            ckpt    = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            encoder.load_state_dict(ckpt["encoder_state_dict"])
            actual_n = ckpt.get("n_records", n_records)
        else:
            corpus = FoundationECGCorpus(
                data_root=data_root,
                max_records=n_records,
                augmentation="physio",
                seed=seed,
                skip_missing=True,
                target_length=250,
            )

            actual_n = len(corpus)
            if actual_n == 0:
                log.warning("No data for scale %s — skipping.", f"{n_records:,}")
                continue
            if actual_n < n_records:
                log.warning(
                    "Only %s records available (requested %s).",
                    f"{actual_n:,}", f"{n_records:,}",
                )

            encoder, pretrain_wall = pretrain_at_scale(
                corpus=corpus,
                encoder_name=encoder_name,
                n_records=actual_n,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                device=device,
                output_dir=out,
                seed=seed,
                mae_weight=mae_weight,
                num_workers=num_workers,
            )

        # ── Linear probe ───────────────────────────────────────────────────
        log.info("Evaluating linear probe …")
        metrics, eval_wall = evaluate_linear_probe(
            encoder, eval_csv, device, n_labels, seed
        )

        result = ScaleResult(
            n_records_requested=n_records,
            n_records_actual=actual_n,
            encoder=encoder_name,
            epochs=epochs,
            n_labels=n_labels,
            pretrain_wall_seconds=pretrain_wall,
            eval_wall_seconds=eval_wall,
            **metrics,
        )
        all_results.append(result)

        log.info(
            "Scale %s  →  AUROC=%.4f | F1=%.4f",
            f"{n_records:,}", result.auroc, result.macro_f1,
        )

        # ── Persist results after every scale ─────────────────────────────
        pd.DataFrame([asdict(r) for r in all_results]).to_csv(
            results_path, index=False
        )
        log.info("Results written → %s", results_path)

    # ── Final plot ─────────────────────────────────────────────────────────
    if len(all_results) >= 2:
        try:
            _plot_scaling_law(all_results, out)
        except Exception as exc:                          # noqa: BLE001
            log.warning("Plot generation failed: %s", exc)

    return all_results


# ---------------------------------------------------------------------------
# Publication-quality figure
# ---------------------------------------------------------------------------

def _plot_scaling_law(results: List[ScaleResult], output_dir: Path) -> None:
    """
    Produce a two-panel log-linear scaling-law figure.

    Left panel  — macro-AUROC vs. log(N)
    Right panel — macro-F1   vs. log(N)

    A fitted log-linear trend line and its R² are overlaid on each panel.
    A vertical dashed line marks the current paper baseline (N = 50 K).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        from matplotlib.lines import Line2D
    except ImportError:
        log.warning("matplotlib not installed — skipping plot.")
        return

    # ── Style ──────────────────────────────────────────────────────────────
    plt.rcParams.update(
        {
            "font.family":        "DejaVu Sans",
            "font.size":          11,
            "axes.spines.top":    False,
            "axes.spines.right":  False,
            "axes.linewidth":     0.8,
            "xtick.direction":    "out",
            "ytick.direction":    "out",
            "figure.dpi":         150,
        }
    )

    ns     = np.array([r.n_records_actual for r in results], dtype=float)
    aurocs = np.array([r.auroc            for r in results], dtype=float)
    f1s    = np.array([r.macro_f1         for r in results], dtype=float)

    log_ns = np.log10(ns)

    def _fit_and_r2(log_x: np.ndarray, y: np.ndarray):
        """Fit y = a·log10(N) + b; return (coeffs, R²)."""
        valid = np.isfinite(y)
        coeffs = np.polyfit(log_x[valid], y[valid], deg=1)
        y_hat  = np.polyval(coeffs, log_x[valid])
        ss_res = np.sum((y[valid] - y_hat) ** 2)
        ss_tot = np.sum((y[valid] - y[valid].mean()) ** 2)
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return coeffs, r2

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "PA-HybridSSL — Scaling Laws\n"
        "(frozen encoder, 100-label linear probe, PTB-XL)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    panel_cfg = [
        (axes[0], aurocs, "Macro AUROC",  "#2166AC"),
        (axes[1], f1s,    "Macro F1",     "#B2182B"),
    ]

    for ax, values, ylabel, color in panel_cfg:
        coeffs, r2 = _fit_and_r2(log_ns, values)

        # ── Trend line ────────────────────────────────────────────────────
        x_smooth     = np.linspace(log_ns.min(), log_ns.max(), 300)
        trend        = np.polyval(coeffs, x_smooth)
        ns_smooth    = 10 ** x_smooth

        ax.plot(ns_smooth, trend, "--", color=color, alpha=0.55,
                linewidth=1.5, label=f"Log-linear fit (R²={r2:.3f})")

        # ── Data points ───────────────────────────────────────────────────
        ax.semilogx(ns, values, "o", color=color, markersize=8,
                    markeredgecolor="white", markeredgewidth=1.2,
                    zorder=5, label="Observed")

        # ── Baseline reference ────────────────────────────────────────────
        ax.axvline(50_000, color="dimgray", linestyle=":", linewidth=1.2,
                   alpha=0.7, label="Baseline (50 K)")

        # ── Point annotations (alternate above / below to avoid overlap) ──
        for i, (n, v) in enumerate(zip(ns, values)):
            if not np.isfinite(v):
                continue
            offset = 10 if i % 2 == 0 else -14
            ax.annotate(
                f"{v:.3f}",
                xy=(n, v),
                xycoords="data",
                xytext=(0, offset),
                textcoords="offset points",
                fontsize=8.5,
                ha="center",
                color=color,
                fontweight="bold",
            )

        # ── Axes formatting ───────────────────────────────────────────────
        ax.set_xlabel("Pretraining corpus size  N", fontsize=11)
        ax.set_ylabel(f"{ylabel}  (100 labels)", fontsize=11)
        ax.set_xticks(ns)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(
                lambda x, _: (
                    f"{x/1e6:.1f}M" if x >= 1e6 else
                    f"{int(x/1e3)}K" if x >= 1e3 else
                    str(int(x))
                )
            )
        )
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.tick_params(axis="x", which="minor", length=0, rotation=30)
        ax.tick_params(axis="x", which="major", rotation=30)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=9, framealpha=0.6)

    plt.tight_layout()
    plot_path = output_dir / "scaling_law_plot.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Scaling law figure saved → %s", plot_path)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scaling_laws",
        description="PA-SSL Scaling Law Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_root",   type=str,   default="data/",
                   help="Root directory of processed ECG CSV datasets.")
    p.add_argument("--eval_csv",    type=str,   default="data/ptbxl_processed.csv",
                   help="CSV file used for linear-probe evaluation.")
    p.add_argument("--encoder",     type=str,   default="resnet1d",
                   choices=["resnet1d", "wavkan", "transformer", "mamba"],
                   help="Encoder backbone.")
    p.add_argument("--epochs",      type=int,   default=100,
                   help="Pretraining epochs per scale.")
    p.add_argument("--batch_size",  type=int,   default=256)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--mae_weight",  type=float, default=0.1,
                   help="Weight of the MAE loss term.")
    p.add_argument("--n_labels",    type=int,   default=100,
                   help="Labeled examples used for the linear probe.")
    p.add_argument("--scales",      type=int,   nargs="+", default=None,
                   help="Pretraining corpus sizes (overrides defaults).")
    p.add_argument("--output_dir",  type=str,   default="experiments/scaling_laws")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--num_workers", type=int,   default=None,
                   help="DataLoader workers (default: min(4, cpu_count)).")
    return p


def main() -> None:
    args    = _build_parser().parse_args()
    results = run_scaling_experiment(
        data_root=args.data_root,
        eval_csv=args.eval_csv,
        encoder_name=args.encoder,
        scales=args.scales,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mae_weight=args.mae_weight,
        n_labels=args.n_labels,
        output_dir=args.output_dir,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    log.info("")
    log.info(_DIVIDER_MAJOR)
    log.info("SCALING LAW RESULTS SUMMARY")
    log.info(_DIVIDER_MAJOR)
    log.info("  %-14s  %8s  %8s  %12s  %12s",
             "N (actual)", "AUROC", "F1", "Train time", "Eval time")
    log.info("  " + "─" * 62)
    for r in sorted(results, key=lambda x: x.n_records_actual):
        log.info(
            "  %-14s  %8.4f  %8.4f  %9.1f s  %9.1f s",
            f"{r.n_records_actual:,}",
            r.auroc,
            r.macro_f1,
            r.pretrain_wall_seconds,
            r.eval_wall_seconds,
        )
    log.info(_DIVIDER_MAJOR)


if __name__ == "__main__":
    main()