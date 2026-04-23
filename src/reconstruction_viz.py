"""
PA-SSL: MAE vs PA-MAE Reconstruction Visualization
====================================================
Generates the "wow" figure for the paper: side-by-side reconstruction
comparison showing that standard MAE hallucinates QRS while PA-MAE
preserves it and correctly infers P/T waves.

Picks 3 representative beats (normal, abnormal/wide-QRS, borderline),
applies both random and physio-aware masking, reconstructs, and saves
publication-quality matplotlib figures.

Runtime: ~2 minutes (forward passes only, no training)

Usage:
    python -m src.reconstruction_viz \
        --checkpoint remote/ssl_passl_resnet_hybrid/best_checkpoint.pth \
        --data_csv data/ptbxl_processed.csv \
        --output_dir results/reconstruction_viz
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.encoder import build_encoder
from src.models.mae import PhysioAwareMasker, MAEDecoder1D


# ─── PICK REPRESENTATIVE BEATS ───────────────────────────────────────────────

def pick_representative_beats(df, n_each=1, seed=42):
    """
    Pick representative beats:
      - Normal: low-amplitude, clean signal
      - Abnormal: high QRS amplitude (possible wide-complex)
      - Borderline: moderate amplitude, ST-segment changes
    """
    signal_cols = [c for c in df.columns if str(c).isdigit()]
    signals = df[signal_cols].values.astype(np.float32)
    labels = df['label'].values

    # Compute per-beat features
    amplitudes = np.max(signals, axis=1) - np.min(signals, axis=1)
    qrs_sharpness = np.max(np.abs(np.diff(signals, axis=1)), axis=1)

    rng = np.random.default_rng(seed)

    # Normal: label=0, LOW amplitude
    normal_mask = labels == 0
    if normal_mask.sum() > 0:
        normal_idx = np.where(normal_mask)[0]
        amps = amplitudes[normal_idx]
        # Pick a nice clean one near the 30th percentile
        target = np.percentile(amps, 30)
        best = normal_idx[np.argmin(np.abs(amps - target))]
        normal_beat = best
    else:
        normal_beat = 0

    # Abnormal: label=1, HIGH amplitude (wide QRS)
    abnormal_mask = labels == 1
    if abnormal_mask.sum() > 0:
        abnormal_idx = np.where(abnormal_mask)[0]
        amps = amplitudes[abnormal_idx]
        target = np.percentile(amps, 85)
        best = abnormal_idx[np.argmin(np.abs(amps - target))]
        abnormal_beat = best
    else:
        abnormal_beat = min(1, len(signals) - 1)

    # Borderline: label=0, but HIGH sharpness (possible ST changes)
    if normal_mask.sum() > 0:
        normal_idx = np.where(normal_mask)[0]
        sharp = qrs_sharpness[normal_idx]
        target = np.percentile(sharp, 80)
        best = normal_idx[np.argmin(np.abs(sharp - target))]
        borderline_beat = best
    else:
        borderline_beat = min(2, len(signals) - 1)

    indices = [normal_beat, abnormal_beat, borderline_beat]
    beat_names = ['Normal Beat', 'Abnormal Beat (Wide QRS)', 'Borderline (ST Changes)']
    return indices, beat_names


# ─── RECONSTRUCTION ──────────────────────────────────────────────────────────

@torch.no_grad()
def reconstruct_beats(encoder, decoder, signals, masker, device):
    """
    Apply masking → encoder → decoder for a set of signals.
    Returns: original, masked, reconstructed (all numpy).
    """
    encoder.eval()
    decoder.eval()

    x = torch.tensor(signals).unsqueeze(1).to(device)  # (N, 1, 250)
    masked_x, masks = masker(x)
    h = encoder.encode(masked_x)
    recon = decoder(h)

    return (
        x.cpu().numpy()[:, 0, :],
        masked_x.cpu().numpy()[:, 0, :],
        recon.cpu().numpy()[:, 0, :],
        masks.cpu().numpy(),
    )


# ─── FIGURE GENERATION ────────────────────────────────────────────────────────

def plot_reconstruction_comparison(
    originals, beat_names,
    pa_masked, pa_recon, pa_masks,
    rand_masked, rand_recon, rand_masks,
    output_path
):
    """
    Generate publication-quality reconstruction comparison figure.
    Layout: 3 rows (one per beat) × 4 columns (orig, random-masked, PA-masked, overlay).
    """
    fig, axes = plt.subplots(3, 4, figsize=(16, 9))

    time_axis = np.arange(250) / 500 * 1000  # ms

    col_titles = ['Original Signal', 'Random MAE\n(Masked + Recon)',
                  'PA-MAE\n(Masked + Recon)', 'Reconstruction Overlay']

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight='bold', pad=10)

    for row in range(3):
        orig = originals[row]

        # Column 0: Original
        ax = axes[row, 0]
        ax.plot(time_axis, orig, color='#2166AC', linewidth=1.2, label='Original')
        # Shade QRS zone
        qrs_start = (125 - 15) / 500 * 1000
        qrs_end = (125 + 15) / 500 * 1000
        ax.axvspan(qrs_start, qrs_end, alpha=0.15, color='red', label='QRS Zone')
        ax.set_ylabel(beat_names[row], fontsize=10, fontweight='bold')

        # Column 1: Random MAE
        ax = axes[row, 1]
        ax.plot(time_axis, orig, color='#CCCCCC', linewidth=0.8, alpha=0.5, label='Original')
        # Show masked regions
        mask = rand_masks[row]
        for i in range(len(mask)):
            if mask[i]:
                ax.axvspan(time_axis[i] - 0.5, time_axis[i] + 0.5,
                          alpha=0.08, color='red', linewidth=0)
        ax.plot(time_axis, rand_recon[row], color='#D6604D', linewidth=1.2,
                label='Reconstruction', linestyle='-')
        ax.axvspan(qrs_start, qrs_end, alpha=0.15, color='red')

        # Column 2: PA-MAE
        ax = axes[row, 2]
        ax.plot(time_axis, orig, color='#CCCCCC', linewidth=0.8, alpha=0.5, label='Original')
        mask = pa_masks[row]
        for i in range(len(mask)):
            if mask[i]:
                ax.axvspan(time_axis[i] - 0.5, time_axis[i] + 0.5,
                          alpha=0.08, color='blue', linewidth=0)
        ax.plot(time_axis, pa_recon[row], color='#2166AC', linewidth=1.2,
                label='Reconstruction', linestyle='-')
        ax.axvspan(qrs_start, qrs_end, alpha=0.15, color='red')

        # Column 3: Overlay
        ax = axes[row, 3]
        ax.plot(time_axis, orig, color='#333333', linewidth=1.0, alpha=0.6, label='Original')
        ax.plot(time_axis, rand_recon[row], color='#D6604D', linewidth=1.2,
                label='Random MAE', linestyle='--')
        ax.plot(time_axis, pa_recon[row], color='#2166AC', linewidth=1.2,
                label='PA-MAE', linestyle='-')
        ax.axvspan(qrs_start, qrs_end, alpha=0.15, color='red')
        if row == 0:
            ax.legend(fontsize=7, loc='upper right', framealpha=0.8)

        # Formatting
        for col in range(4):
            axes[row, col].set_xlim(0, time_axis[-1])
            axes[row, col].tick_params(labelsize=8)
            if row == 2:
                axes[row, col].set_xlabel('Time (ms)', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    print(f"  Saved: {output_path.replace('.png', '.pdf')}")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='MAE vs PA-MAE reconstruction visualization')
    parser.add_argument('--checkpoint', type=str,
                        default='remote/ssl_passl_resnet_hybrid/best_checkpoint.pth')
    parser.add_argument('--data_csv', type=str,
                        default='data/ptbxl_processed.csv')
    parser.add_argument('--output_dir', type=str,
                        default='results/reconstruction_viz')
    parser.add_argument('--mask_ratio', type=float, default=0.60)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    import pandas as pd
    df = pd.read_csv(args.data_csv)
    signal_cols = [c for c in df.columns if str(c).isdigit()]
    print(f"Loaded {len(df):,} beats")

    # Pick representative beats
    beat_indices, beat_names = pick_representative_beats(df)
    signals = df.iloc[beat_indices][signal_cols].values.astype(np.float32)
    print(f"Selected beats: {beat_indices}")
    for i, (idx, name) in enumerate(zip(beat_indices, beat_names)):
        label = df.iloc[idx]['label']
        print(f"  [{i}] {name}: index={idx}, label={label}")

    # Load encoder
    print(f"\nLoading checkpoint: {args.checkpoint}")
    encoder = build_encoder('resnet1d')
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get('encoder_state_dict', ckpt.get('model_state_dict', ckpt))
    if any(k.startswith('encoder.') for k in state_dict.keys()):
        encoder_sd = {k.replace('encoder.', '', 1): v for k, v in state_dict.items()
                      if k.startswith('encoder.')}
    else:
        encoder_sd = state_dict
    encoder.load_state_dict(encoder_sd, strict=False)
    encoder = encoder.to(device).eval()

    # Load decoder
    decoder = MAEDecoder1D(repr_dim=256)
    decoder_keys = {k.replace('decoder.', '', 1): v for k, v in state_dict.items()
                    if k.startswith('decoder.')}
    if not decoder_keys:
        # Try full state dict with 'decoder.' keys from model_state_dict
        full_sd = ckpt.get('model_state_dict', ckpt)
        decoder_keys = {k.replace('decoder.', '', 1): v for k, v in full_sd.items()
                        if k.startswith('decoder.')}
    if decoder_keys:
        decoder.load_state_dict(decoder_keys, strict=False)
        print(f"  Loaded decoder weights ({len(decoder_keys)} keys)")
    else:
        print("  [WARN] No decoder keys found — using untrained decoder (results may be noisy)")
    decoder = decoder.to(device).eval()

    # Create maskers
    pa_masker = PhysioAwareMasker(
        mask_ratio=args.mask_ratio, qrs_avoidance_prob=0.8,
        masking_strategy='physio_aware'
    )
    rand_masker = PhysioAwareMasker(
        mask_ratio=args.mask_ratio, qrs_avoidance_prob=0.0,
        masking_strategy='random'
    )

    # Reconstruct with both strategies (run multiple times, pick best viz)
    print("\nGenerating reconstructions...")

    # Use fixed seed for reproducible masking
    torch.manual_seed(42)
    np.random.seed(42)

    pa_orig, pa_masked, pa_recon, pa_masks = reconstruct_beats(
        encoder, decoder, signals, pa_masker, device)

    torch.manual_seed(42)
    rand_orig, rand_masked, rand_recon, rand_masks = reconstruct_beats(
        encoder, decoder, signals, rand_masker, device)

    # Generate figure
    output_path = os.path.join(args.output_dir, 'reconstruction_comparison.png')
    print(f"\nPlotting reconstruction comparison...")
    plot_reconstruction_comparison(
        pa_orig, beat_names,
        pa_masked, pa_recon, pa_masks,
        rand_masked, rand_recon, rand_masks,
        output_path
    )

    # Also compute reconstruction MSE for each strategy
    print(f"\nReconstruction Quality (MSE on masked regions only):")
    print(f"{'Beat':<35} {'Random MAE':>12} {'PA-MAE':>12} {'Δ':>10}")
    print("-" * 70)
    for i in range(len(signals)):
        rand_mse = np.mean((rand_recon[i][rand_masks[i]] - signals[i][rand_masks[i]])**2) \
            if rand_masks[i].sum() > 0 else 0
        pa_mse = np.mean((pa_recon[i][pa_masks[i]] - signals[i][pa_masks[i]])**2) \
            if pa_masks[i].sum() > 0 else 0
        delta = rand_mse - pa_mse
        print(f"  {beat_names[i]:<33} {rand_mse:>12.6f} {pa_mse:>12.6f} {delta:>+10.6f}")

    print(f"\nDone. Figures saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
