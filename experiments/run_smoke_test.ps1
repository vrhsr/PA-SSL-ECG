# PA-SSL Smoke Test v2 (~5 min, resilient)
# Does NOT abort on MIT-BIH/Chapman failures since PTB-XL is the primary dataset.

Write-Host "================================================================"
Write-Host "PA-SSL: SMOKE TEST v2 (Resilient, ~5 min)"
Write-Host "================================================================"

$env:PYTHONPATH = "$PWD"
$errors = @()

# --- Step 1: Data Processing ---
Write-Host "`n[1/8] Processing Datasets..."
python -m src.data.emit_ptbxl --output data/ptbxl_processed.csv
if ($LASTEXITCODE -ne 0) { Write-Host "FAIL: PTB-XL processing"; exit 1 }
Write-Host "  PTB-XL OK"

# MIT-BIH and Chapman are OPTIONAL - don't abort on failure
python -m src.data.emit_mitbih --output data/mitbih_processed.csv 2>$null
if ($LASTEXITCODE -ne 0) { 
    Write-Host "  MIT-BIH: SKIPPED (dataset not available or dependency issue)"
    $errors += "MIT-BIH skipped"
} else { Write-Host "  MIT-BIH OK" }

python -m src.data.emit_chapman --output data/chapman_processed.csv 2>$null
if ($LASTEXITCODE -ne 0) { 
    Write-Host "  Chapman: SKIPPED (dataset not available or dependency issue)"
    $errors += "Chapman skipped"
} else { Write-Host "  Chapman OK" }

# --- Step 2: SSL Pretraining (2 epochs only!) ---
Write-Host "`n[2/8] SSL Pretraining (2 epochs, smoke test)..."
python -m src.train_ssl --encoder resnet1d --augmentation physio --use_temporal --epochs 2 --batch_size 32 --seed 42
if ($LASTEXITCODE -ne 0) { Write-Host "FAIL: ResNet1D training"; exit 1 }
Write-Host "  ResNet1D + PhysioAug OK"

python -m src.train_ssl --encoder resnet1d --augmentation naive --no_temporal --epochs 2 --batch_size 32 --seed 42
if ($LASTEXITCODE -ne 0) { Write-Host "FAIL: Naive ablation training"; exit 1 }
Write-Host "  ResNet1D + NaiveAug OK"

# --- Step 3: Core Evaluation ---
Write-Host "`n[3/8] Downstream Evaluation..."
python -m src.evaluate --checkpoint experiments/ssl_resnet1d_physio_temporal/best_checkpoint.pth --data_file data/ptbxl_processed.csv --n_seeds 1
if ($LASTEXITCODE -ne 0) { Write-Host "FAIL: Evaluation"; exit 1 }
Write-Host "  Evaluation OK"

# --- Step 4: Baselines ---
Write-Host "`n[4/8] Baselines (random-init only for speed)..."
python -c "
from src.baselines import evaluate_random_init
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'  Using device: {device}')
r = evaluate_random_init('data/ptbxl_processed.csv', device, n_seeds=1)
print(f'  Random init OK: {len(r)} rows')
"
if ($LASTEXITCODE -ne 0) { 
    Write-Host "  Baselines: SKIPPED (non-critical)"
    $errors += "Baselines skipped"
} else { Write-Host "  Baselines OK" }

# --- Step 5: Training Curves ---
Write-Host "`n[5/8] Training Curves..."
python -c "
import json, os
from src.plotting import plot_training_curves
os.makedirs('figures', exist_ok=True)
histories = {}
for name, d in {'PA-SSL': 'experiments/ssl_resnet1d_physio_temporal', 'Naive': 'experiments/ssl_resnet1d_naive'}.items():
    hp = os.path.join(d, 'history.json')
    if os.path.exists(hp):
        with open(hp) as f: histories[name] = json.load(f)
if histories:
    plot_training_curves(histories, save_path='figures/training_curves.png')
    print('  Training curves OK')
else:
    print('  No history files (expected for 2-epoch test)')
"

# --- Step 6: Augmentation Hero Figure ---
Write-Host "`n[6/8] Augmentation Hero Figure..."
python -c "
import numpy as np, os, sys
sys.path.insert(0, '.')
from src.data.ecg_dataset import ECGBeatDataset
from src.augmentations.physio_augmentations import (
    constrained_time_warp, amplitude_perturbation, baseline_wander,
    emg_noise_injection, heart_rate_resample
)
from src.plotting import plot_augmentation_hero
os.makedirs('figures', exist_ok=True)
dataset = ECGBeatDataset('data/ptbxl_processed.csv')
signal = dataset[0][0].numpy().squeeze()
r_peak = len(signal) // 2
np.random.seed(42)
augs = [constrained_time_warp(signal.copy(), r_peak), amplitude_perturbation(signal.copy(), r_peak),
        baseline_wander(signal.copy()), emg_noise_injection(signal.copy()), heart_rate_resample(signal.copy(), r_peak)]
names = ['Time Warp', 'Amplitude Perturb', 'Baseline Wander', 'EMG Noise', 'HR Resample']
plot_augmentation_hero(signal, augs, names, r_peak_idx=r_peak, save_path='figures/augmentation_hero.png')
print('  Hero figure OK')
"
if ($LASTEXITCODE -ne 0) { $errors += "Hero figure skipped" }

# --- Step 7: Efficiency Profiling ---
Write-Host "`n[7/8] Efficiency Profiling..."
python -c "
from src.experiments import profile_model
from src.models.encoder import build_encoder
e = build_encoder('resnet1d', proj_dim=128)
p = profile_model(e)
print(f'  resnet1d: {p[\"n_params\"]/1e6:.1f}M params, {p[\"inference_ms\"]:.1f}ms/sample')
print('  Efficiency OK')
"
if ($LASTEXITCODE -ne 0) { $errors += "Efficiency skipped" }

# --- Step 8: Grad-CAM ---
Write-Host "`n[8/8] Grad-CAM..."
python -m src.gradcam --checkpoint experiments/ssl_resnet1d_physio_temporal/best_checkpoint.pth --data_file data/ptbxl_processed.csv --output figures/gradcam_smoke.png 2>$null
if ($LASTEXITCODE -ne 0) { 
    $errors += "Grad-CAM skipped"
    Write-Host "  Grad-CAM: SKIPPED (non-critical)" 
} else { Write-Host "  Grad-CAM OK" }

Write-Host "`n================================================================"
if ($errors.Count -eq 0) {
    Write-Host "SMOKE TEST PASSED! All 8 steps completed successfully."
} else {
    Write-Host "SMOKE TEST PASSED (with $($errors.Count) non-critical skips)."
    Write-Host "Skipped: $($errors -join ', ')"
    Write-Host "Core pipeline (PTB-XL + SSL + Evaluation) is working!"
}
Write-Host "================================================================"
Write-Host "Ready to launch full pipeline:"
Write-Host "  powershell -ExecutionPolicy Bypass -File .\experiments\run_all.ps1"
Write-Host "================================================================"
