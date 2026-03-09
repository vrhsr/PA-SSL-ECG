"""
PA-SSL: Grad-CAM Interpretability Script
Generates Grad-CAM heatmaps showing what the SSL encoder focuses on.
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import argparse
from scipy.ndimage import zoom

from src.data.ecg_dataset import ECGBeatDataset
from src.models.encoder import build_encoder


def generate_gradcam(encoder, dataset, device, n_samples=6, save_path='figures/gradcam_interpretability.png'):
    """Generate Grad-CAM visualizations for random ECG beats."""
    encoder = encoder.to(device).eval()
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle('Grad-CAM Interpretability: What the SSL Encoder Focuses On',
                 fontsize=14, fontweight='bold')
    
    np.random.seed(42)
    
    for ax_idx in range(n_samples):
        idx = np.random.randint(0, len(dataset))
        signal_np = dataset.X[idx]
        label = dataset.labels[idx]
        
        signal = torch.tensor(signal_np).unsqueeze(0).unsqueeze(0).to(device)
        signal.requires_grad_(True)
        
        # Forward through ResNet1D layers
        x = encoder.stem(signal)
        x = encoder.layer1(x)
        x = encoder.layer2(x)
        x = encoder.layer3(x)
        features = encoder.layer4(x)
        features.retain_grad()
        
        pooled = encoder.avgpool(features)
        flat = torch.flatten(pooled, 1)
        proj = encoder.projection_head(flat)
        target = proj.sum()
        target.backward()
        
        # Compute Grad-CAM
        gradients = features.grad[0]  # (C, L)
        weights = gradients.mean(dim=1)  # (C,)
        cam = (weights.unsqueeze(1) * features[0]).sum(dim=0)  # (L,)
        cam = torch.relu(cam).detach().cpu().numpy()
        
        # Interpolate to signal length
        cam_interp = zoom(cam, 250 / len(cam))
        cam_interp = (cam_interp - cam_interp.min()) / (cam_interp.max() - cam_interp.min() + 1e-8)
        
        # Plot
        row, col = ax_idx // 3, ax_idx % 3
        ax = axes[row, col]
        t = np.arange(250) / 100.0
        ax.plot(t, signal_np, 'k-', linewidth=0.8, alpha=0.8)
        ax.fill_between(t, signal_np.min(), signal_np.max(),
                        where=cam_interp > 0.5, alpha=0.3, color='red', label='High Attention')
        ax.fill_between(t, signal_np.min(), signal_np.max(),
                        where=(cam_interp > 0.25) & (cam_interp <= 0.5), alpha=0.15, color='orange')
        ax.set_title(f'Beat {idx} (Class {label})', fontsize=10)
        ax.set_xlabel('Time (s)')
        if col == 0:
            ax.set_ylabel('Amplitude')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Grad-CAM saved: {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PA-SSL Grad-CAM')
    parser.add_argument('--checkpoint', type=str,
                        default='experiments/ssl_resnet1d_physio_temporal/best_checkpoint.pth')
    parser.add_argument('--data_file', type=str, default='data/ptbxl_processed.csv')
    parser.add_argument('--output', type=str, default='figures/gradcam_interpretability.png')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = build_encoder('resnet1d', proj_dim=128)
    ckpt = torch.load(args.checkpoint, map_location=device)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    
    dataset = ECGBeatDataset(args.data_file)
    generate_gradcam(encoder, dataset, device, save_path=args.output)
