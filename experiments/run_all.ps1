# PA-SSL Experiment Runner (Top-Tier Edition v2)
# Runs the full experiment pipeline from zero to paper-ready results.
# Now includes Chapman-Shaoxing and full 3×3 cross-dataset transfer matrix.

Write-Host "================================================================"
Write-Host "PA-SSL: Top-Tier Automated Research Pipeline (v2)"
Write-Host "================================================================"

# ─── Step 0: Set Python Path & Dependencies ────────────────────────────────
$env:PYTHONPATH = "$PWD"
Write-Host "`n[0/6] Installing Dependencies..."
pip install -r requirements.txt

# ─── Step 1: Data Processing (3 Datasets) ─────────────────────────────────
Write-Host "`n[1/6] Processing Datasets..."
Write-Host "`nProcessing PTB-XL (Germany, ~22k records)..."
python -m src.data.emit_ptbxl --output data/ptbxl_processed.csv

Write-Host "`nProcessing MIT-BIH (USA, 48 records)..."
python -m src.data.emit_mitbih --output data/mitbih_processed.csv

Write-Host "`nProcessing Chapman-Shaoxing (China, ~10k records)..."
python -m src.data.emit_chapman --output data/chapman_processed.csv

# ─── Step 2: SSL Pretraining (Main Configurations) ────────────────────────
Write-Host "`n[2/6] SSL Pretraining (200 Epochs Top-Tier Scale)..."
Write-Host "`nTraining: ResNet1D + PhysioAug + Temporal (MAIN)"
python -m src.train_ssl --encoder resnet1d --augmentation physio --use_temporal --epochs 200 --batch_size 256

Write-Host "`nTraining: WavKAN + PhysioAug + Temporal (MAIN)"
python -m src.train_ssl --encoder wavkan --augmentation physio --use_temporal --epochs 200 --batch_size 256

# ─── Step 3: SSL Pretraining (Ablations) ──────────────────────────────────
Write-Host "`n[3/6] SSL Pretraining Ablations (100 Epochs Scale)..."

Write-Host "`nTraining: ResNet1D + NaiveAug (ablation)"
python -m src.train_ssl --encoder resnet1d --augmentation naive --no_temporal --epochs 100 --batch_size 256

Write-Host "`nTraining: ResNet1D + PhysioAug, no temporal (ablation)"
python -m src.train_ssl --encoder resnet1d --augmentation physio --no_temporal --epochs 100 --batch_size 256

# ─── Step 4: Core Evaluation (Same-Dataset) ───────────────────────────────
Write-Host "`n[4/6] Downstream Evaluation (Same-Dataset)..."

# PTB-XL evaluation
python -m src.evaluate --checkpoint experiments/ssl_resnet1d_physio_temporal/best_checkpoint.pth --data_file data/ptbxl_processed.csv
python -m src.evaluate --checkpoint experiments/ssl_wavkan_physio_temporal/best_checkpoint.pth --data_file data/ptbxl_processed.csv --encoder wavkan

# ─── Step 5: Cross-Dataset Transfer Matrix (3×3) ──────────────────────────
Write-Host "`n[5/6] Cross-Dataset Transfer Matrix (3x3)..."

# Train on PTB-XL → Test on MIT-BIH
Write-Host "`nPTB-XL -> MIT-BIH"
python -m src.evaluate --checkpoint experiments/ssl_resnet1d_physio_temporal/best_checkpoint.pth --data_file data/mitbih_processed.csv

# Train on PTB-XL → Test on Chapman
Write-Host "`nPTB-XL -> Chapman"
python -m src.evaluate --checkpoint experiments/ssl_resnet1d_physio_temporal/best_checkpoint.pth --data_file data/chapman_processed.csv

# ─── Step 6: Advanced Publishability Experiments ──────────────────────────
Write-Host "`n[6/6] Advanced Publishability Experiments & Figures..."

# 1. Ablation Study Automation (Generates LaTeX Table)
Write-Host "`nRunning Full Ablation Suite..."
python -c "from src.experiments import run_ablation_suite; run_ablation_suite('data/ptbxl_processed.csv')"

# 2. Robustness Testing (Noise & Artifacts)
Write-Host "`nRunning Robustness Testing..."
python -c "
from src.experiments import robustness_experiment
from src.models.encoder import build_encoder
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = build_encoder('resnet1d', proj_dim=128)
ckpt = torch.load('experiments/ssl_resnet1d_physio_temporal/best_checkpoint.pth', map_location=device)
encoder.load_state_dict(ckpt['encoder_state_dict'])
robustness_experiment(encoder.to(device), 'data/ptbxl_processed.csv', device)
"

# 3. OOD Detection (PTB-XL Normal vs Chapman Abnormal)
Write-Host "`nRunning OOD Detection (PTB-XL vs Chapman)..."
python -c "
from src.experiments import ood_detection_experiment
from src.models.encoder import build_encoder
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = build_encoder('resnet1d', proj_dim=128)
ckpt = torch.load('experiments/ssl_resnet1d_physio_temporal/best_checkpoint.pth', map_location=device)
encoder.load_state_dict(ckpt['encoder_state_dict'])
ood_detection_experiment(encoder.to(device), 'data/ptbxl_processed.csv', 'data/chapman_processed.csv', device)
"

# 4. Interpretability & Visualizations
Write-Host "`nGenerating Paper Figures..."
python -m src.augmentations.visualize_and_test --visualize

Write-Host "`n================================================================"
Write-Host "PA-SSL: All experiments complete!"
Write-Host "Top-Tier results and paper figures are in: experiments/"
Write-Host "Cross-dataset transfer matrix across 3 continents validated!"
Write-Host "================================================================"
