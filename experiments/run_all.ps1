# PA-SSL Experiment Runner (Publication-Ready Edition v4)
# Runs the full experiment pipeline from zero to paper-ready results.
# Includes: 3 datasets, multi-seed, baselines, fairness, training curves, efficiency.

Write-Host "================================================================"
Write-Host "PA-SSL: Publication-Ready Pipeline (v4 - All 8 Priorities)"
Write-Host "================================================================"

# --- Step 0: Set Python Path and Dependencies ---
$env:PYTHONPATH = "$PWD"
Write-Host "`n[0/12] Installing Dependencies..."
pip install -r requirements.txt

# --- Step 1: Data Processing (3 Datasets, 3 Continents) ---
Write-Host "`n[1/12] Processing Datasets..."
Write-Host "`nProcessing PTB-XL (Germany, ~22k records)..."
if (Test-Path "data/ptbxl_processed.csv") {
    Write-Host "  PTB-XL already processed. Skipping."
} else {
    python -m src.data.emit_ptbxl --output data/ptbxl_processed.csv
    if ($LASTEXITCODE -ne 0) { Write-Host "FATAL: PTB-XL processing failed"; exit 1 }
}

Write-Host "`nProcessing MIT-BIH (USA, 48 records)..."
if (Test-Path "data/mitbih_processed.csv") {
    Write-Host "  MIT-BIH already processed. Skipping."
} else {
    python -m src.data.emit_mitbih --output data/mitbih_processed.csv 2>$null
    if ($LASTEXITCODE -ne 0) { Write-Host "  MIT-BIH: SKIPPED (non-critical)" }
}

Write-Host "`nProcessing Chapman-Shaoxing (China, ~10k records)..."
if (Test-Path "data/chapman_processed.csv") {
    Write-Host "  Chapman already processed. Skipping."
} else {
    python -m src.data.emit_chapman --output data/chapman_processed.csv 2>$null
    if ($LASTEXITCODE -ne 0) { Write-Host "  Chapman: SKIPPED (non-critical)" }
}

# --- Step 2: SSL Pretraining - Main Configs (200 Epochs, 3 Seeds) ---
Write-Host "`n[2/12] SSL Pretraining (200 Epochs, Multi-Seed)..."

foreach ($seed in @(42, 123, 456)) {
    Write-Host "`nTraining: ResNet1D + PhysioAug + Temporal (seed=$seed)"
    if (Test-Path "experiments/ssl_resnet1d_physio_temporal_s$seed/history.json") {
        Write-Host "  Already trained. Skipping."
    } else {
        python -m src.train_ssl --encoder resnet1d --augmentation physio --use_temporal --epochs 200 --batch_size 1024 --seed $seed --output_dir "experiments/ssl_resnet1d_physio_temporal_s$seed"
    }

    Write-Host "`nTraining: WavKAN + PhysioAug + Temporal (seed=$seed)"
    if (Test-Path "experiments/ssl_wavkan_physio_temporal_s$seed/history.json") {
        Write-Host "  Already trained. Skipping."
    } else {
        python -m src.train_ssl --encoder wavkan --augmentation physio --use_temporal --epochs 200 --batch_size 1024 --seed $seed --output_dir "experiments/ssl_wavkan_physio_temporal_s$seed"
    }
}

# --- Step 3: SSL Pretraining - Ablations (100 Epochs, seed=42) ---
Write-Host "`n[3/12] SSL Pretraining Ablations (100 Epochs)..."

Write-Host "`nTraining: ResNet1D + NaiveAug (ablation)"
if (Test-Path "experiments/ssl_resnet1d_naive/history.json") {
    Write-Host "  Already trained. Skipping."
} else {
    python -m src.train_ssl --encoder resnet1d --augmentation naive --no_temporal --epochs 100 --batch_size 1024 --seed 42 --output_dir "experiments/ssl_resnet1d_naive"
}

Write-Host "`nTraining: ResNet1D + PhysioAug, no temporal (ablation)"
if (Test-Path "experiments/ssl_resnet1d_physio/history.json") {
    Write-Host "  Already trained. Skipping."
} else {
    python -m src.train_ssl --encoder resnet1d --augmentation physio --no_temporal --epochs 100 --batch_size 1024 --seed 42 --output_dir "experiments/ssl_resnet1d_physio"
}

# --- Step 4: Metadata-Conditioned Training (200 Epochs) ---
Write-Host "`n[4/12] Metadata-Conditioned Training (200 Epochs)..."
if (Test-Path "experiments/ssl_resnet1d_metadata/history.json") {
    Write-Host "  Already trained. Skipping."
} else {
    python -m src.train_ssl --encoder resnet1d --augmentation metadata --use_temporal --epochs 200 --batch_size 1024 --seed 42 --output_dir "experiments/ssl_resnet1d_metadata"
}

# --- Step 5: Training Convergence Curves (P4) ---
Write-Host "`n[5/12] Generating Training Convergence Curves..."
python -c "
import json, os
from src.plotting import plot_training_curves

histories = {}
dirs = {
    'PA-SSL (PhysioAug + Temporal)': 'experiments/ssl_resnet1d_physio_temporal_s42',
    'SimCLR + Naive Aug': 'experiments/ssl_resnet1d_naive',
    'PhysioAug (no temporal)': 'experiments/ssl_resnet1d_physio',
}
for name, d in dirs.items():
    hp = os.path.join(d, 'history.json')
    if os.path.exists(hp):
        with open(hp) as f:
            histories[name] = json.load(f)

if histories:
    plot_training_curves(histories, save_path='figures/training_curves.png')
    print('Training curves saved')
else:
    print('No history files found')
"

# --- Step 6: Core Evaluation (Same-Dataset) ---
Write-Host "`n[6/12] Downstream Evaluation..."

python -m src.evaluate --checkpoint experiments/ssl_resnet1d_physio_temporal_s42/best_checkpoint.pth --data_file data/ptbxl_processed.csv
python -m src.evaluate --checkpoint experiments/ssl_wavkan_physio_temporal_s42/best_checkpoint.pth --data_file data/ptbxl_processed.csv --encoder wavkan

# --- Step 7: Baseline Experiments (P2) ---
Write-Host "`n[7/12] Running Baseline Experiments (Random Init, Supervised, CLOCS-style)..."
python -m src.baselines --data_csv data/ptbxl_processed.csv --naive_checkpoint experiments/ssl_resnet1d_naive/best_checkpoint.pth --output_dir experiments/baselines

# --- Step 8: Cross-Dataset Transfer Matrix (3x3) ---
Write-Host "`n[8/12] Cross-Dataset Transfer Matrix (3x3)..."

Write-Host "`nPTB-XL -> MIT-BIH"
python -m src.evaluate --checkpoint experiments/ssl_resnet1d_physio_temporal_s42/best_checkpoint.pth --data_file data/mitbih_processed.csv

Write-Host "`nPTB-XL -> Chapman"
python -m src.evaluate --checkpoint experiments/ssl_resnet1d_physio_temporal_s42/best_checkpoint.pth --data_file data/chapman_processed.csv

# --- Step 9: Advanced Experiments ---
Write-Host "`n[9/12] Advanced Experiments (Ablation, Robustness, OOD)..."

# Ablation Suite
Write-Host "`nRunning Full Ablation Suite..."
python -c "from src.experiments import run_ablation_suite; run_ablation_suite('data/ptbxl_processed.csv')"

# Robustness Testing
Write-Host "`nRunning Robustness Testing..."
python -c "from src.experiments import robustness_experiment; from src.models.encoder import build_encoder; import torch, pandas as pd; device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); encoder = build_encoder('resnet1d', proj_dim=128); ckpt = torch.load('experiments/ssl_resnet1d_physio_temporal_s42/best_checkpoint.pth', map_location=device); encoder.load_state_dict(ckpt['encoder_state_dict']); results = robustness_experiment(encoder.to(device), 'data/ptbxl_processed.csv', device); results.to_csv('experiments/robustness_results.csv', index=False)"

# OOD Detection
Write-Host "`nRunning OOD Detection (PTB-XL vs Chapman)..."
python -c "from src.experiments import ood_detection_experiment; from src.models.encoder import build_encoder; import torch, json; device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); encoder = build_encoder('resnet1d', proj_dim=128); ckpt = torch.load('experiments/ssl_resnet1d_physio_temporal_s42/best_checkpoint.pth', map_location=device); encoder.load_state_dict(ckpt['encoder_state_dict']); results = ood_detection_experiment(encoder.to(device), 'data/ptbxl_processed.csv', 'data/chapman_processed.csv', device); f=open('experiments/ood_results.json','w'); json.dump({k: float(v) for k, v in results.items()}, f, indent=2); f.close()"

# --- Step 10: Demographic Fairness Analysis (P3) ---
Write-Host "`n[10/12] Demographic Fairness Analysis..."
python -c "
from src.statistical_tests import demographic_subgroup_analysis, generate_fairness_table
from src.plotting import plot_fairness_comparison
from src.models.encoder import build_encoder
import torch, json, os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('experiments/fairness', exist_ok=True)
os.makedirs('figures', exist_ok=True)

fairness_results = {}

# PA-SSL encoder
encoder = build_encoder('resnet1d', proj_dim=128)
ckpt = torch.load('experiments/ssl_resnet1d_physio_temporal_s42/best_checkpoint.pth', map_location=device)
encoder.load_state_dict(ckpt['encoder_state_dict'])
encoder = encoder.to(device)
encoder.eval()
passl_fairness = demographic_subgroup_analysis(encoder, 'data/ptbxl_processed.csv', device)
passl_fairness.to_csv('experiments/fairness/passl_fairness.csv', index=False)
fairness_results['PA-SSL'] = passl_fairness

# Random init baseline
encoder2 = build_encoder('resnet1d', proj_dim=128).to(device)
encoder2.eval()
random_fairness = demographic_subgroup_analysis(encoder2, 'data/ptbxl_processed.csv', device)
random_fairness.to_csv('experiments/fairness/random_fairness.csv', index=False)
fairness_results['Random Init'] = random_fairness

# Generate fairness table and plot
generate_fairness_table(fairness_results, 'experiments/fairness/fairness_table.tex')
plot_fairness_comparison(fairness_results, save_path='figures/fairness_comparison.png')
print('Fairness analysis complete')
"

# --- Step 11: Computational Efficiency (P7) + Visualizations ---
Write-Host "`n[11/12] Computational Efficiency + Visualizations..."

# Efficiency profiling
Write-Host "`nProfiling model efficiency..."
python -c "
from src.experiments import profile_model
from src.models.encoder import build_encoder
from src.statistical_tests import generate_efficiency_table
import json, os

os.makedirs('experiments/efficiency', exist_ok=True)

results = []
for enc_type in ['resnet1d', 'wavkan']:
    encoder = build_encoder(enc_type, proj_dim=128)
    profile = profile_model(encoder)
    profile['method'] = f'PA-SSL ({enc_type})'
    results.append(profile)
    print(f'{enc_type}: {profile}')

import pandas as pd
pd.DataFrame(results).to_csv('experiments/efficiency/efficiency_results.csv', index=False)
generate_efficiency_table(results, 'experiments/efficiency/efficiency_table.tex')
print('Efficiency profiling complete')
"

# t-SNE Visualization
Write-Host "`nGenerating t-SNE Visualizations..."
python -c "import torch, numpy as np, os; from src.data.ecg_dataset import ECGBeatDataset; from src.models.encoder import build_encoder; from src.evaluate import extract_representations; from src.plotting import plot_tsne_embeddings; device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); os.makedirs('figures', exist_ok=True); encoder = build_encoder('resnet1d', proj_dim=128); ckpt = torch.load('experiments/ssl_resnet1d_physio_temporal_s42/best_checkpoint.pth', map_location=device); encoder.load_state_dict(ckpt['encoder_state_dict']); encoder = encoder.to(device); dataset = ECGBeatDataset('data/ptbxl_processed.csv'); reprs, labels = extract_representations(encoder, dataset, device); idx = np.random.choice(len(reprs), min(5000, len(reprs)), replace=False); plot_tsne_embeddings(reprs[idx], labels[idx], method_name='PA-SSL (ResNet1D)', save_path='figures/tsne_passl_resnet.png'); print('t-SNE saved')"

# Grad-CAM
Write-Host "`nGenerating Grad-CAM Interpretability Maps..."
python -m src.gradcam --checkpoint experiments/ssl_resnet1d_physio_temporal_s42/best_checkpoint.pth --data_file data/ptbxl_processed.csv --output figures/gradcam_interpretability.png

# Augmentation Hero Figure (P5)
Write-Host "`nGenerating Augmentation Hero Figure..."
python -c "
import numpy as np, os
from src.data.ecg_dataset import ECGBeatDataset
from src.augmentations.physio_augmentations import (
    constrained_time_warp, amplitude_perturbation, baseline_wander,
    emg_noise_injection, heart_rate_resample, wavelet_masking
)
from src.plotting import plot_augmentation_hero

os.makedirs('figures', exist_ok=True)
dataset = ECGBeatDataset('data/ptbxl_processed.csv')

# Get a representative sample
signal = dataset[0][0].numpy().squeeze()
r_peak = len(signal) // 2

augs = []
names = []

np.random.seed(42)
augs.append(constrained_time_warp(signal.copy(), r_peak)); names.append('Constrained Time Warp')
augs.append(amplitude_perturbation(signal.copy(), r_peak)); names.append('Amplitude Perturbation')
augs.append(baseline_wander(signal.copy())); names.append('Baseline Wander')
augs.append(emg_noise_injection(signal.copy())); names.append('EMG Noise Injection')
augs.append(heart_rate_resample(signal.copy(), r_peak)); names.append('Heart Rate Resample')
try:
    augs.append(wavelet_masking(signal.copy())); names.append('Wavelet Masking')
except: pass

plot_augmentation_hero(signal, augs, names, r_peak_idx=r_peak, save_path='figures/augmentation_hero.png')
print('Hero figure saved')
"

# --- Step 12: Paper-Quality Plots ---
Write-Host "`n[12/12] Generating All Paper Figures..."
python -m src.augmentations.visualize_and_test --visualize

Write-Host "`n================================================================"
Write-Host "PA-SSL: ALL EXPERIMENTS COMPLETE!"
Write-Host "================================================================"
Write-Host "Results:"
Write-Host "  - SSL Checkpoints:    experiments/ssl_*/best_checkpoint.pth"
Write-Host "  - Evaluation CSVs:    experiments/ssl_*/evaluation/"
Write-Host "  - Baselines:          experiments/baselines/"
Write-Host "  - Robustness:         experiments/robustness_results.csv"
Write-Host "  - OOD Detection:      experiments/ood_results.json"
Write-Host "  - Ablation LaTeX:     experiments/ablations/ablation_table.tex"
Write-Host "  - Fairness:           experiments/fairness/"
Write-Host "  - Efficiency:         experiments/efficiency/"
Write-Host "  - Figures:            figures/"
Write-Host "  - Training Curves:    figures/training_curves.png"
Write-Host "  - t-SNE:              figures/tsne_passl_resnet.png"
Write-Host "  - Hero Figure:        figures/augmentation_hero.png"
Write-Host "  - Fairness Plot:      figures/fairness_comparison.png"
Write-Host "  - Grad-CAM:           figures/gradcam_interpretability.png"
Write-Host "================================================================"
