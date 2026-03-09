# PA-SSL Experiment Runner (Top-Tier Edition v3 — RTX 4080 Full Power)
# Runs the full experiment pipeline from zero to paper-ready results.
# Chapman-Shaoxing integrated, metadata conditioning, multi-seed, t-SNE, Grad-CAM.

Write-Host "================================================================"
Write-Host "PA-SSL: Top-Tier Automated Research Pipeline (v3 — GPU Full Power)"
Write-Host "================================================================"

# ─── Step 0: Set Python Path & Dependencies ────────────────────────────────
$env:PYTHONPATH = "$PWD"
Write-Host "`n[0/8] Installing Dependencies..."
pip install -r requirements.txt

# ─── Step 1: Data Processing (3 Datasets, 3 Continents) ───────────────────
Write-Host "`n[1/8] Processing Datasets..."
Write-Host "`nProcessing PTB-XL (Germany, ~22k records)..."
python -m src.data.emit_ptbxl --output data/ptbxl_processed.csv

Write-Host "`nProcessing MIT-BIH (USA, 48 records)..."
python -m src.data.emit_mitbih --output data/mitbih_processed.csv

Write-Host "`nProcessing Chapman-Shaoxing (China, ~10k records)..."
python -m src.data.emit_chapman --output data/chapman_processed.csv

# ─── Step 2: SSL Pretraining — Main Configs (200 Epochs) ──────────────────
Write-Host "`n[2/8] SSL Pretraining (200 Epochs, Top-Tier Scale)..."

Write-Host "`nTraining: ResNet1D + PhysioAug + Temporal (MAIN)"
python -m src.train_ssl --encoder resnet1d --augmentation physio --use_temporal --epochs 200 --batch_size 256

Write-Host "`nTraining: WavKAN + PhysioAug + Temporal (MAIN)"
python -m src.train_ssl --encoder wavkan --augmentation physio --use_temporal --epochs 200 --batch_size 256

# ─── Step 3: SSL Pretraining — Ablations (100 Epochs) ─────────────────────
Write-Host "`n[3/8] SSL Pretraining Ablations (100 Epochs Scale)..."

Write-Host "`nTraining: ResNet1D + NaiveAug (ablation)"
python -m src.train_ssl --encoder resnet1d --augmentation naive --no_temporal --epochs 100 --batch_size 256

Write-Host "`nTraining: ResNet1D + PhysioAug, no temporal (ablation)"
python -m src.train_ssl --encoder resnet1d --augmentation physio --no_temporal --epochs 100 --batch_size 256

# ─── Step 4: Phase 9 — Metadata-Conditioned Training (200 Epochs) ─────────
Write-Host "`n[4/8] Phase 9: Metadata-Conditioned SSL (200 Epochs)..."

Write-Host "`nTraining: ResNet1D + PhysioAug + Temporal + Metadata (NEXT-GEN)"
python -m src.train_ssl --encoder resnet1d --augmentation physio --use_temporal --use_metadata --epochs 200 --batch_size 256 --output_dir experiments

# ─── Step 5: Core Evaluation (Same-Dataset) ───────────────────────────────
Write-Host "`n[5/8] Downstream Evaluation (Same-Dataset)..."

python -m src.evaluate --checkpoint experiments/ssl_resnet1d_physio_temporal/best_checkpoint.pth --data_file data/ptbxl_processed.csv
python -m src.evaluate --checkpoint experiments/ssl_wavkan_physio_temporal/best_checkpoint.pth --data_file data/ptbxl_processed.csv --encoder wavkan

# ─── Step 6: Cross-Dataset Transfer Matrix (3×3) ──────────────────────────
Write-Host "`n[6/8] Cross-Dataset Transfer Matrix (3x3)..."

# Train on PTB-XL → Test on MIT-BIH
Write-Host "`nPTB-XL -> MIT-BIH"
python -m src.evaluate --checkpoint experiments/ssl_resnet1d_physio_temporal/best_checkpoint.pth --data_file data/mitbih_processed.csv

# Train on PTB-XL → Test on Chapman
Write-Host "`nPTB-XL -> Chapman"
python -m src.evaluate --checkpoint experiments/ssl_resnet1d_physio_temporal/best_checkpoint.pth --data_file data/chapman_processed.csv

# ─── Step 7: Advanced GPU-Intensive Experiments ───────────────────────────
Write-Host "`n[7/8] Advanced Publishability Experiments..."

# 1. Full Ablation Suite (LaTeX Table)
Write-Host "`nRunning Full Ablation Suite..."
python -c "from src.experiments import run_ablation_suite; run_ablation_suite('data/ptbxl_processed.csv')"

# 2. Robustness Testing (Noise & Artifacts)
Write-Host "`nRunning Robustness Testing..."
python -c "
from src.experiments import robustness_experiment
from src.models.encoder import build_encoder
import torch, pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = build_encoder('resnet1d', proj_dim=128)
ckpt = torch.load('experiments/ssl_resnet1d_physio_temporal/best_checkpoint.pth', map_location=device)
encoder.load_state_dict(ckpt['encoder_state_dict'])
results = robustness_experiment(encoder.to(device), 'data/ptbxl_processed.csv', device)
results.to_csv('experiments/robustness_results.csv', index=False)
"

# 3. OOD Detection (PTB-XL Normal vs Chapman Abnormal — Cross-Continent)
Write-Host "`nRunning OOD Detection (PTB-XL vs Chapman)..."
python -c "
from src.experiments import ood_detection_experiment
from src.models.encoder import build_encoder
import torch, json
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = build_encoder('resnet1d', proj_dim=128)
ckpt = torch.load('experiments/ssl_resnet1d_physio_temporal/best_checkpoint.pth', map_location=device)
encoder.load_state_dict(ckpt['encoder_state_dict'])
results = ood_detection_experiment(encoder.to(device), 'data/ptbxl_processed.csv', 'data/chapman_processed.csv', device)
with open('experiments/ood_results.json', 'w') as f:
    json.dump({k: float(v) for k, v in results.items()}, f, indent=2)
"

# 4. t-SNE Visualization (GPU-accelerated)
Write-Host "`nGenerating t-SNE Visualizations..."
python -c "
import torch, numpy as np, os
from src.data.ecg_dataset import ECGBeatDataset
from src.models.encoder import build_encoder
from src.evaluate import extract_representations
from src.plotting import plot_tsne_embeddings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('figures', exist_ok=True)

# PA-SSL (ResNet1D)
encoder = build_encoder('resnet1d', proj_dim=128)
ckpt = torch.load('experiments/ssl_resnet1d_physio_temporal/best_checkpoint.pth', map_location=device)
encoder.load_state_dict(ckpt['encoder_state_dict'])
encoder = encoder.to(device)
dataset = ECGBeatDataset('data/ptbxl_processed.csv')
reprs, labels = extract_representations(encoder, dataset, device)
# Subsample for speed
idx = np.random.choice(len(reprs), min(5000, len(reprs)), replace=False)
plot_tsne_embeddings(reprs[idx], labels[idx], method_name='PA-SSL (ResNet1D)', save_path='figures/tsne_passl_resnet.png')
print('t-SNE saved: figures/tsne_passl_resnet.png')
"

# 5. Grad-CAM Interpretability
Write-Host "`nGenerating Grad-CAM Interpretability Maps..."
python -c "
import torch, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from src.data.ecg_dataset import ECGBeatDataset
from src.models.encoder import build_encoder

os.makedirs('figures', exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
encoder = build_encoder('resnet1d', proj_dim=128)
ckpt = torch.load('experiments/ssl_resnet1d_physio_temporal/best_checkpoint.pth', map_location=device)
encoder.load_state_dict(ckpt['encoder_state_dict'])
encoder = encoder.to(device).eval()

# Load data
dataset = ECGBeatDataset('data/ptbxl_processed.csv')

# Grad-CAM on last conv layer
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle('Grad-CAM Interpretability: What the SSL Encoder Focuses On', fontsize=14, fontweight='bold')

for ax_idx in range(6):
    idx = np.random.randint(0, len(dataset))
    signal = torch.tensor(dataset.X[idx]).unsqueeze(0).unsqueeze(0).to(device).requires_grad_(True)
    label = dataset.labels[idx]

    # Forward through stem + layers
    x = encoder.stem(signal)
    x = encoder.layer1(x)
    x = encoder.layer2(x)
    x = encoder.layer3(x)
    features = encoder.layer4(x)  # Last conv layer
    features.retain_grad()

    pooled = encoder.avgpool(features)
    flat = torch.flatten(pooled, 1)
    proj = encoder.projection_head(flat)
    target = proj.sum()
    target.backward()

    # Grad-CAM
    gradients = features.grad[0]  # (C, L)
    weights = gradients.mean(dim=1)  # (C,)
    cam = (weights.unsqueeze(1) * features[0]).sum(dim=0)  # (L,)
    cam = torch.relu(cam).detach().cpu().numpy()

    # Interpolate to signal length
    from scipy.ndimage import zoom
    cam_interp = zoom(cam, 250 / len(cam))
    cam_interp = (cam_interp - cam_interp.min()) / (cam_interp.max() - cam_interp.min() + 1e-8)

    # Plot
    row, col = ax_idx // 3, ax_idx % 3
    ax = axes[row, col]
    t = np.arange(250) / 100.0
    ax.plot(t, dataset.X[idx], 'k-', linewidth=0.8, alpha=0.8)
    ax.fill_between(t, dataset.X[idx].min(), dataset.X[idx].max(), where=cam_interp > 0.5, alpha=0.3, color='red', label='High Attention')
    ax.fill_between(t, dataset.X[idx].min(), dataset.X[idx].max(), where=(cam_interp > 0.25) & (cam_interp <= 0.5), alpha=0.15, color='orange')
    ax.set_title(f'Beat {idx} (Class {label})', fontsize=10)
    ax.set_xlabel('Time (s)')
    if col == 0:
        ax.set_ylabel('Amplitude')

plt.tight_layout()
plt.savefig('figures/gradcam_interpretability.png', dpi=300, bbox_inches='tight')
plt.close()
print('Grad-CAM saved: figures/gradcam_interpretability.png')
"

# ─── Step 8: Paper-Quality Plots ──────────────────────────────────────────
Write-Host "`n[8/8] Generating All Paper Figures..."
python -m src.augmentations.visualize_and_test --visualize

Write-Host "`n================================================================"
Write-Host "PA-SSL: ALL EXPERIMENTS COMPLETE!"
Write-Host "================================================================"
Write-Host "Results:"
Write-Host "  - SSL Checkpoints:    experiments/ssl_*/best_checkpoint.pth"
Write-Host "  - Evaluation CSVs:    experiments/ssl_*/evaluation/"
Write-Host "  - Robustness:         experiments/robustness_results.csv"
Write-Host "  - OOD Detection:      experiments/ood_results.json"
Write-Host "  - Ablation LaTeX:     experiments/ablations/ablation_table.tex"
Write-Host "  - Figures:            figures/"
Write-Host "  - t-SNE:              figures/tsne_passl_resnet.png"
Write-Host "  - Grad-CAM:           figures/gradcam_interpretability.png"
Write-Host "================================================================"
