# PA-SSL: Physiology-Constrained Representation Learning
# Final Paper Automated Experiment Pipeline
# This script executes the minimum required experiment set defined for publication.

Write-Host "================================================================"
Write-Host "PA-SSL: Paper-Ready Pipeline (Official Baselines + Exact Matrix)"
Write-Host "================================================================"

$env:PYTHONPATH = "$PWD"

# --- 1. Data Processing ---
Write-Host "`n[1/7] Ensuring Datasets are Processed..."
if (-not (Test-Path "data/ptbxl_processed.csv")) {
    python -m src.data.emit_ptbxl --output data/ptbxl_processed.csv
}
if (-not (Test-Path "data/mitbih_processed.csv")) {
    python -m src.data.emit_mitbih --output data/mitbih_processed.csv 2>$null
}
if (-not (Test-Path "data/chapman_processed.csv")) {
    python -m src.data.emit_chapman --output data/chapman_processed.csv 2>$null
}

# --- 2. SSL Pretraining Matrix ---
# The matrix: 
# 1. SimCLR Naive (ResNet, NT-Xent)
# 2. PA-SSL (ResNet, NT-Xent)
# 3. PA-SSL (SE-ResNet1D-34, NT-Xent)
# 4. PA-SSL (SE-ResNet1D-34, VICReg)

Write-Host "`n[2/7] Executing SSL Pretraining Matrix (100 Epochs each for speed/fairness)..."

$configs = @(
    @{name="simclr_naive_resnet"; enc="resnet1d"; aug="naive"; temp="False"; loss="ntxent"},
    @{name="passl_resnet_ntxent"; enc="resnet1d"; aug="physio"; temp="True"; loss="ntxent"},
    @{name="passl_seresnet_ntxent"; enc="se_resnet1d34"; aug="physio"; temp="True"; loss="ntxent"},
    @{name="passl_seresnet_vicreg"; enc="se_resnet1d34"; aug="physio"; temp="True"; loss="vicreg"}
)

foreach ($cfg in $configs) {
    Write-Host "`nTraining: $($cfg.name)..."
    $out_dir = "experiments/ssl_$($cfg.name)"
    if (Test-Path "$out_dir/history.json") {
        Write-Host "  Already trained. Skipping."
    } else {
        $temp_flag = if ($cfg.temp -eq "True") { "--use_temporal" } else { "--no_temporal" }
        python -m src.train_ssl --encoder $($cfg.enc) --augmentation $($cfg.aug) $temp_flag --loss_type $($cfg.loss) --epochs 100 --batch_size 512 --seed 42 --output_dir $out_dir
    }
}

# --- 3. Run Baselines (Random, Supervised, TS2Vec, TFC, CLOCS) ---
Write-Host "`n[3/7] Running Official Baselines and Downstream Evaluation..."
# `run_all_baselines` inside baselines.py evaluates Random, Supervised, TS2Vec, TFC, and CLOCS.
# CLOCS uses the simclr_naive_resnet checkpoint.
python -m src.baselines --data_csv data/ptbxl_processed.csv --naive_checkpoint experiments/ssl_simclr_naive_resnet/best_checkpoint.pth --output_dir experiments/baselines

# --- 4. Main PA-SSL Downstream Evaluation ---
Write-Host "`n[4/7] Downstream Evaluation of PA-SSL Variants on PTB-XL..."
foreach ($cfg in $configs) {
    if ($cfg.name -ne "simclr_naive_resnet") {
        python -m src.evaluate --checkpoint experiments/ssl_$($cfg.name)/best_checkpoint.pth --data_file data/ptbxl_processed.csv --encoder $($cfg.enc)
    }
}

# --- 5. Cross-Dataset Transfer Matrix ---
Write-Host "`n[5/7] Cross-Dataset Transfer..."
# Evaluate best model (PA-SSL SE-ResNet NT-Xent or VICReg) on MIT-BIH and Chapman
python -m src.evaluate --checkpoint experiments/ssl_passl_seresnet_vicreg/best_checkpoint.pth --data_file data/mitbih_processed.csv --encoder se_resnet1d34
python -m src.evaluate --checkpoint experiments/ssl_passl_seresnet_vicreg/best_checkpoint.pth --data_file data/chapman_processed.csv --encoder se_resnet1d34

# --- 6. Component & Augmentation Ablations & Validations ---
Write-Host "`n[6/7] Running QRS Validation & Advanced Analytics..."
# QRS metrics and general visual analysis
python -m src.augmentations.visualize_and_test --test --visualize

# Component Ablation (already handled partially via existing models + baselines)
# To fully run the suite and generate the bar chart:
python -c "from src.experiments import run_ablation_suite; import pandas as pd; from src.plotting import plot_ablation_bars; res = run_ablation_suite('data/ptbxl_processed.csv'); plot_ablation_bars(pd.read_csv('experiments/ablations/ablation_results.csv'), save_path='figures/ablation_bars.png')"

# OOD & Robustness test on best model
python -c "from src.experiments import robustness_experiment, ood_detection_experiment; from src.models.encoder import build_encoder; import torch, json; device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); encoder = build_encoder('se_resnet1d34', proj_dim=128); ckpt = torch.load('experiments/ssl_passl_seresnet_vicreg/best_checkpoint.pth', map_location=device); encoder.load_state_dict(ckpt['encoder_state_dict']); results = robustness_experiment(encoder.to(device), 'data/ptbxl_processed.csv', device); results.to_csv('experiments/robustness_results.csv', index=False); ood_res = ood_detection_experiment(encoder.to(device), 'data/ptbxl_processed.csv', 'data/chapman_processed.csv', device); f=open('experiments/ood_results.json','w'); json.dump({k: float(v) for k, v in ood_res.items()}, f, indent=2); f.close()"

# --- 7. Generate Paper Figures ---
Write-Host "`n[7/7] Generating Paper Figures (UMAP, Grad-CAM, etc)..."
# UMAP/t-SNE Representation Visualization
python -c "import torch, numpy as np, os; from src.data.ecg_dataset import ECGBeatDataset; from src.models.encoder import build_encoder; from src.evaluate import extract_representations; from src.plotting import plot_umap_embeddings; device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); os.makedirs('figures', exist_ok=True); encoder = build_encoder('se_resnet1d34', proj_dim=128); ckpt = torch.load('experiments/ssl_passl_seresnet_vicreg/best_checkpoint.pth', map_location=device); encoder.load_state_dict(ckpt['encoder_state_dict']); encoder = encoder.to(device); dataset = ECGBeatDataset('data/ptbxl_processed.csv'); reprs, labels = extract_representations(encoder, dataset, device); plot_umap_embeddings(reprs, labels, method_name='PA-SSL (SE-ResNet34, VICReg)', save_path='figures/umap_passl_vicreg.png'); print('UMAP saved')"

# Grad-CAM 
python -m src.gradcam --checkpoint experiments/ssl_passl_seresnet_vicreg/best_checkpoint.pth --data_file data/ptbxl_processed.csv --output figures/gradcam_passl_vicreg.png

Write-Host "`n================================================================"
Write-Host "PA-SSL: Pipeline Complete. All results and figures are ready."
Write-Host "================================================================"
