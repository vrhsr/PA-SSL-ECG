# PA-SSL: Physiology-Constrained Representation Learning
# SMOKE TEST PIPELINE
# Executes the entire pipeline with minimal epochs and batch counts to ensure code correctness before 60h run.

Write-Host "================================================================"
Write-Host "PA-SSL: SMOKE TEST PIPELINE (Epochs=1, Fast Mode)"
Write-Host "================================================================"

$env:PYTHONPATH = "$PWD"

# Remove any previous smoke test artifacts if you want a clean run
if (Test-Path "experiments/smoke") { Remove-Item -Recurse -Force "experiments/smoke" }
New-Item -ItemType Directory -Force -Path "experiments/smoke"

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
Write-Host "`n[2/7] Executing SSL Pretraining (1 Epoch)..."

$configs = @(
    @{name="simclr_naive_resnet"; enc="resnet1d"; aug="naive"; temp="False"; loss="ntxent"},
    @{name="passl_resnet_ntxent"; enc="resnet1d"; aug="physio"; temp="True"; loss="ntxent"},
    @{name="passl_seresnet_ntxent"; enc="se_resnet1d34"; aug="physio"; temp="True"; loss="ntxent"},
    @{name="passl_seresnet_vicreg"; enc="se_resnet1d34"; aug="physio"; temp="True"; loss="vicreg"}
)

foreach ($cfg in $configs) {
    Write-Host "`nTraining: $($cfg.name)..."
    $out_dir = "experiments/smoke/ssl_$($cfg.name)"
    $temp_flag = if ($cfg.temp -eq "True") { "--use_temporal" } else { "--no_temporal" }
    
    python -m src.train_ssl --encoder $($cfg.enc) --augmentation $($cfg.aug) $temp_flag --loss_type $($cfg.loss) --epochs 1 --batch_size 128 --max_batches 10 --seed 42 --output_dir $out_dir
}

# --- 3. Run Baselines (Random, Supervised, TS2Vec, TFC, CLOCS) ---
Write-Host "`n[3/7] Running Official Baselines and Downstream Evaluation (Epochs=1)..."
python -m src.baselines --data_csv data/ptbxl_processed.csv --naive_checkpoint experiments/smoke/ssl_simclr_naive_resnet/best_checkpoint.pth --output_dir experiments/smoke/baselines --epochs 1 --max_batches 10

# --- 4. Main PA-SSL Downstream Evaluation ---
Write-Host "`n[4/7] Downstream Evaluation of PA-SSL Variants on PTB-XL..."
foreach ($cfg in $configs) {
    if ($cfg.name -ne "simclr_naive_resnet") {
        python -m src.evaluate --checkpoint experiments/smoke/ssl_$($cfg.name)/best_checkpoint.pth --data_file data/ptbxl_processed.csv --encoder $($cfg.enc) --max_batches 10 --n_seeds 1
    }
}

# --- 5. Cross-Dataset Transfer Matrix ---
Write-Host "`n[5/7] Cross-Dataset Transfer..."
# Evaluate best model (PA-SSL SE-ResNet NT-Xent or VICReg) on MIT-BIH and Chapman
python -m src.evaluate --checkpoint experiments/smoke/ssl_passl_seresnet_vicreg/best_checkpoint.pth --data_file data/mitbih_processed.csv --encoder se_resnet1d34 --max_batches 10 --n_seeds 1
python -m src.evaluate --checkpoint experiments/smoke/ssl_passl_seresnet_vicreg/best_checkpoint.pth --data_file data/chapman_processed.csv --encoder se_resnet1d34 --max_batches 10 --n_seeds 1

# --- 6. Component & Augmentation Ablations & Validations ---
Write-Host "`n[6/7] Running QRS Validation & Advanced Analytics..."
# QRS metrics and general visual analysis
python -m src.augmentations.visualize_and_test --test --visualize --output_dir experiments/smoke/figures

# Component Ablation handles data itself, will execute fast enough or we just test bar script
python -c "import pandas as pd; import os; os.makedirs('experiments/smoke/figures', exist_ok=True); from src.plotting import plot_ablation_bars; df = pd.DataFrame({'configuration':['Full','-Temp'], 'accuracy':[0.8,0.7], 'auroc':[0.9, 0.8], 'ece':[0.1,0.2]}); df.to_csv('experiments/smoke/dummy_ablation.csv'); plot_ablation_bars(df, save_path='experiments/smoke/figures/ablation_bars.png'); print('Ablation figure saved!')"

# --- 7. Generate Paper Figures ---
Write-Host "`n[7/7] Generating Paper Figures (UMAP, Grad-CAM, etc)..."
# UMAP/t-SNE Representation Visualization
python -c "import torch, numpy as np, os; from src.data.ecg_dataset import ECGBeatDataset; from src.models.encoder import build_encoder; from src.evaluate import extract_representations; from src.plotting import plot_umap_embeddings; device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); os.makedirs('experiments/smoke/figures', exist_ok=True); encoder = build_encoder('se_resnet1d34', proj_dim=128); ckpt = torch.load('experiments/smoke/ssl_passl_seresnet_vicreg/best_checkpoint.pth', map_location=device); encoder.load_state_dict(ckpt['encoder_state_dict']); encoder = encoder.to(device); dataset = ECGBeatDataset('data/ptbxl_processed.csv'); reprs, labels = extract_representations(encoder, dataset, device); plot_umap_embeddings(reprs, labels, method_name='PA-SSL Smoke', save_path='experiments/smoke/figures/umap.png'); print('UMAP saved')"

# Grad-CAM 
python -m src.gradcam --checkpoint experiments/smoke/ssl_passl_seresnet_vicreg/best_checkpoint.pth --data_file data/ptbxl_processed.csv --output experiments/smoke/figures/gradcam.png

Write-Host "`n================================================================"
Write-Host "PA-SSL: SMOKE TEST Complete!"
Write-Host "================================================================"
