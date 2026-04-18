"""
PA-SSL: Scaling Laws Experiment

Demonstrates that PA-HybridSSL performance improves log-linearly with
pretraining data size — the key experiment that transforms a methods paper
into a foundation model paper.

The "scaling law figure" (log N vs AUROC) is the single most compelling
piece of evidence for top-tier journal submission.

Experiment procedure for each scale N ∈ {1K, 10K, 50K, 100K, 500K, 1M, 2M}:
    1. Sample N records from FoundationECGCorpus
    2. Pretrain PA-HybridSSL for E epochs (default 100)
    3. Freeze encoder, fit linear probe with 100 labels on PTB-XL test set
    4. Record AUROC, macro F1 on PTB-XL evaluation set

Output:
    experiments/scaling_laws/scaling_results.csv   — tabular results
    experiments/scaling_laws/scaling_law_plot.png  — log-linear figure

Usage:
    # Full experiment (GPU required, ~3-7 days)
    python -m src.experiments.scaling_laws \\
        --data_root data/ \\
        --eval_csv data/ptbxl_processed.csv \\
        --encoder resnet1d \\
        --epochs 100 \\
        --n_labels 100 \\
        --output_dir experiments/scaling_laws

    # Quick smoke test (2 scales, 1 epoch)
    python -m src.experiments.scaling_laws \\
        --data_root data/ \\
        --eval_csv data/ptbxl_processed.csv \\
        --encoder resnet1d \\
        --epochs 1 \\
        --scales 1000 10000 \\
        --output_dir experiments/scaling_laws_smoke
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score

from src.data.foundation_corpus import FoundationECGCorpus
from src.data.ecg_dataset import ECGBeatDataset
from src.models.encoder import build_encoder
from src.models.mae import HybridMAE
from src.losses import CombinedContrastiveLoss


# ─── DEFAULT SCALES ───────────────────────────────────────────────────────────
DEFAULT_SCALES = [
    1_000,
    10_000,
    50_000,    # Current paper baseline
    100_000,
    500_000,
    1_000_000,
    2_000_000,
]


# ─── PRETRAINING LOOP ─────────────────────────────────────────────────────────

def pretrain_at_scale(
    corpus: FoundationECGCorpus,
    encoder_name: str,
    n_records: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    output_dir: Path,
    seed: int = 42,
) -> torch.nn.Module:
    """
    Pretrain PA-HybridSSL on a corpus capped at n_records.

    Returns the trained (and frozen) encoder.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build encoder and MAE wrapper
    encoder = build_encoder(encoder_name, proj_dim=128)
    hybrid = HybridMAE(encoder, repr_dim=256, mask_ratio=0.15)
    hybrid = hybrid.to(device)

    loss_fn = CombinedContrastiveLoss(
        temperature=0.1, alpha=1.0, beta=0.5, loss_type='ntxent'
    )

    optimizer = optim.AdamW(hybrid.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loader = DataLoader(
        corpus,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )

    checkpoint_path = output_dir / f"encoder_scale{n_records}.pth"

    print(f"\n{'─'*60}")
    print(f"Pretraining on {n_records:,} records | {encoder_name} | {epochs} epochs")
    print(f"Loader: {len(corpus):,} samples, {len(loader)} batches/epoch")
    print(f"{'─'*60}")

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        hybrid.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            view1 = batch['view1'].to(device)
            view2 = batch['view2'].to(device)

            # Ensure (B, 1, L) shape
            if view1.dim() == 2:
                view1, view2 = view1.unsqueeze(1), view2.unsqueeze(1)

            # Contrastive forward
            z1 = encoder(view1, return_projection=True)
            z2 = encoder(view2, return_projection=True)
            loss, _, _ = loss_fn(z1, z2)

            # MAE forward
            recon, masks, masked = hybrid.forward_mae(view1)
            mae_loss = torch.nn.functional.mse_loss(
                recon[masks.unsqueeze(1).expand_as(recon)],
                view1[masks.unsqueeze(1).expand_as(view1)],
            ) if masks.any() else torch.tensor(0.0, device=device)

            total = loss + 0.1 * mae_loss

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(hybrid.parameters(), 1.0)
            optimizer.step()

            epoch_loss += total.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Elapsed: {elapsed/60:.1f}m")

    # Save checkpoint
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'n_records': n_records,
        'encoder_name': encoder_name,
        'epochs': epochs,
    }, checkpoint_path)
    print(f"  Checkpoint saved: {checkpoint_path}")

    return encoder


# ─── LINEAR PROBE EVALUATION ──────────────────────────────────────────────────

def evaluate_linear_probe(
    encoder: torch.nn.Module,
    eval_csv: str,
    device: torch.device,
    n_labels: int = 100,
    seed: int = 42,
) -> dict:
    """
    Frozen-encoder linear probe evaluation on PTB-XL (or any eval CSV).

    Steps:
        1. Extract representations for the full eval dataset
        2. Use only n_labels labeled examples to train a logistic regression head
        3. Evaluate on the remaining test examples
    """
    from src.evaluate import extract_representations

    dataset = ECGBeatDataset(eval_csv)
    encoder.eval()
    encoder = encoder.to(device)

    print(f"  Extracting representations from {len(dataset):,} samples...")
    reprs, labels = extract_representations(encoder, dataset, device)

    # Stratified label split: n_labels examples per class for training
    rng = np.random.RandomState(seed)
    n_classes = len(np.unique(labels))
    per_class = max(1, n_labels // n_classes)

    train_idx, test_idx = [], []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        train_idx.extend(cls_idx[:per_class].tolist())
        test_idx.extend(cls_idx[per_class:].tolist())

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    if len(test_idx) == 0:
        # Fallback: random 80/20 split
        idx = rng.permutation(len(reprs))
        n_train = max(n_labels, int(0.2 * len(reprs)))
        train_idx = idx[:n_train]
        test_idx = idx[n_train:]

    X_train, y_train = reprs[train_idx], labels[train_idx]
    X_test, y_test = reprs[test_idx], labels[test_idx]

    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # Linear probe
    clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs',
                              multi_class='auto', random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    # Compute metrics
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    try:
        if n_classes == 2:
            auroc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auroc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    except ValueError:
        auroc = float('nan')

    return {
        'macro_f1': float(f1),
        'auroc': float(auroc),
        'n_train_labeled': len(X_train),
        'n_test': len(X_test),
        'n_classes': n_classes,
    }


# ─── MAIN SCALING EXPERIMENT ──────────────────────────────────────────────────

def run_scaling_experiment(
    data_root: str,
    eval_csv: str,
    encoder_name: str = 'resnet1d',
    scales: Optional[List[int]] = None,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 3e-4,
    n_labels: int = 100,
    output_dir: str = 'experiments/scaling_laws',
    seed: int = 42,
) -> list:
    """
    Run the full scaling law experiment.

    For each scale in `scales`:
        1. Build a size-capped corpus
        2. Pretrain the encoder
        3. Evaluate with a linear probe (n_labels labeled examples)
        4. Log results

    Returns list of result dicts.
    """
    if scales is None:
        scales = DEFAULT_SCALES

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Scaling Law Experiment")
    print(f"  Encoder: {encoder_name}")
    print(f"  Scales: {[f'{n:,}' for n in scales]}")
    print(f"  Epochs per scale: {epochs}")
    print(f"  Eval labels: {n_labels}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    all_results = []
    results_path = output_dir / 'scaling_results.csv'

    for n_records in scales:
        print(f"\n{'#'*60}")
        print(f"# Scale: {n_records:,} pretraining records")
        print(f"{'#'*60}")

        # Check if checkpoint already exists (resume-friendly)
        checkpoint_path = output_dir / f"encoder_scale{n_records}.pth"
        if checkpoint_path.exists():
            print(f"  Found existing checkpoint: {checkpoint_path}. Evaluating...")
            encoder = build_encoder(encoder_name, proj_dim=128)
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            encoder.load_state_dict(ckpt['encoder_state_dict'])
        else:
            # Build corpus capped at n_records
            corpus = FoundationECGCorpus(
                data_root=data_root,
                max_records=n_records,
                augmentation='physio',
                seed=seed,
                skip_missing=True,
                target_length=250,
            )

            if len(corpus) == 0:
                print(f"  [SKIP] No data available for scale {n_records:,}.")
                continue

            actual_n = len(corpus)
            if actual_n < n_records:
                print(f"  [NOTE] Only {actual_n:,} records available (requested {n_records:,})")

            # Pretrain
            encoder = pretrain_at_scale(
                corpus=corpus,
                encoder_name=encoder_name,
                n_records=actual_n,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                device=device,
                output_dir=output_dir,
                seed=seed,
            )

        # Linear probe evaluation
        print(f"\n  Evaluating linear probe...")
        metrics = evaluate_linear_probe(encoder, eval_csv, device, n_labels, seed)

        result = {
            'n_records_requested': n_records,
            'n_records_actual': len(corpus) if not checkpoint_path.exists() else n_records,
            'encoder': encoder_name,
            'epochs': epochs,
            'n_labels': n_labels,
            **metrics,
        }
        all_results.append(result)

        print(f"\n  Scale {n_records:,}: AUROC={metrics['auroc']:.4f} | F1={metrics['macro_f1']:.4f}")

        # Save intermediate results after each scale
        import pandas as pd
        pd.DataFrame(all_results).to_csv(results_path, index=False)
        print(f"  Intermediate results saved: {results_path}")

    # Generate scaling law plot
    if len(all_results) >= 2:
        try:
            _plot_scaling_law(all_results, output_dir)
        except Exception as e:
            print(f"  [WARN] Plot failed: {e}")

    return all_results


def _plot_scaling_law(results: list, output_dir: Path) -> None:
    """Generate the log-linear scaling law figure."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("  matplotlib not installed — skip plot")
        return

    ns = [r['n_records_actual'] for r in results]
    aurocs = [r['auroc'] for r in results]
    f1s = [r['macro_f1'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('PA-HybridSSL: Scaling Laws', fontsize=14, fontweight='bold')

    for ax, values, ylabel, color in [
        (axes[0], aurocs, 'Macro AUROC (100 labels)', '#2166AC'),
        (axes[1], f1s, 'Macro F1 (100 labels)', '#B2182B'),
    ]:
        ax.semilogx(ns, values, 'o-', color=color, linewidth=2, markersize=8)
        ax.set_xlabel('Pretraining corpus size (records)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)

        # ── Force ALL 5 tick labels explicitly (prevents matplotlib from hiding 50K/500K) ──
        ax.set_xticks(ns)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f'{x/1e6:.0f}M' if x >= 1e6 else
                          f'{int(x/1e3)}K' if x >= 1e3 else str(int(x))
        ))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.tick_params(axis='x', which='minor', length=0)
        # Rotate labels slightly so they never overlap
        ax.tick_params(axis='x', rotation=30)

        # ── Annotate each point (alternate above/below to prevent overlap) ──
        for i, (n, v) in enumerate(zip(ns, values)):
            # If next point is lower → going down → put label above current
            if i < len(values) - 1 and values[i + 1] < v:
                xytext = (0, 10)
            else:
                xytext = (0, -14)
            ax.annotate(f'{v:.3f}', (n, v), textcoords='offset points',
                        xytext=xytext, fontsize=9, ha='center',
                        fontweight='bold', color=color)

        # Reference line: current paper baseline (50K)
        ax.axvline(x=50_000, color='gray', linestyle='--', alpha=0.6,
                   label='Current (50K)')
        ax.legend(fontsize=9)

    plt.tight_layout()
    plot_path = output_dir / 'scaling_law_plot.png'
    plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Scaling law plot saved: {plot_path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='PA-SSL Scaling Law Experiment'
    )
    parser.add_argument('--data_root', type=str, default='data/',
                        help='Root directory of processed CSV datasets')
    parser.add_argument('--eval_csv', type=str, default='data/ptbxl_processed.csv',
                        help='CSV file for linear probe evaluation')
    parser.add_argument('--encoder', type=str, default='resnet1d',
                        choices=['resnet1d', 'wavkan', 'transformer', 'mamba'])
    parser.add_argument('--epochs', type=int, default=100,
                        help='Pretraining epochs per scale (default 100)')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--n_labels', type=int, default=100,
                        help='Number of labeled examples for linear probe')
    parser.add_argument('--scales', type=int, nargs='+', default=None,
                        help='Pretraining corpus sizes to evaluate')
    parser.add_argument('--output_dir', type=str, default='experiments/scaling_laws')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    results = run_scaling_experiment(
        data_root=args.data_root,
        eval_csv=args.eval_csv,
        encoder_name=args.encoder,
        scales=args.scales,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_labels=args.n_labels,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    print(f"\n{'='*60}")
    print("FINAL SCALING LAW RESULTS")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['n_records_actual']:>10,} records → "
              f"AUROC={r['auroc']:.4f} | F1={r['macro_f1']:.4f}")


if __name__ == '__main__':
    main()
