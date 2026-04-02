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

# --- 2. SSL Pretraining Matrix --- 3×2 factorial (encoders × ssl modes)
Write-Host "`n[2/7] Executing SSL Pretraining (1 Epoch)..."

$configs = @(
    @{name="simclr_naive_resnet"; enc="resnet1d"; aug="naive"; temp="False"; loss="ntxent"},
    @{name="passl_resnet_ntxent"; enc="resnet1d"; aug="physio"; temp="True"; loss="ntxent"},
    @{name="passl_resnet_vicreg"; enc="resnet1d"; aug="physio"; temp="True"; loss="vicreg"},
    @{name="passl_resnet_hybrid"; enc="resnet1d"; aug="physio"; temp="True"; loss="hybrid"},
    @{name="passl_wavkan_ntxent"; enc="wavkan";   aug="physio"; temp="True"; loss="ntxent"},
    @{name="passl_wavkan_vicreg"; enc="wavkan";   aug="physio"; temp="True"; loss="vicreg"},
    @{name="passl_wavkan_hybrid"; enc="wavkan";   aug="physio"; temp="True"; loss="hybrid"}
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

# --- 5. Cross-Dataset Transfer Matrix --- use best model (hybrid resnet1d)
Write-Host "`n[5/7] Cross-Dataset Transfer..."
python -m src.evaluate --checkpoint experiments/smoke/ssl_passl_resnet_hybrid/best_checkpoint.pth --data_file data/mitbih_processed.csv --encoder resnet1d --max_batches 10 --n_seeds 1
python -m src.evaluate --checkpoint experiments/smoke/ssl_passl_resnet_hybrid/best_checkpoint.pth --data_file data/chapman_processed.csv --encoder resnet1d --max_batches 10 --n_seeds 1

# --- 6. Component & Augmentation Ablations & Validations ---
Write-Host "`n[6/7] Running QRS Validation & Advanced Analytics..."
# QRS metrics and general visual analysis
python -m src.augmentations.visualize_and_test --test --visualize --output_dir experiments/smoke/figures

# Component Ablation handles data itself, will execute fast enough or we just test bar script
python -c "import pandas as pd; import os; os.makedirs('experiments/smoke/figures', exist_ok=True); from src.plotting import plot_ablation_bars; df = pd.DataFrame({'configuration':['Full','-Temp'], 'accuracy':[0.8,0.7], 'auroc':[0.9, 0.8], 'ece':[0.1,0.2]}); df.to_csv('experiments/smoke/dummy_ablation.csv'); plot_ablation_bars(df, save_path='experiments/smoke/figures/ablation_bars.png'); print('Ablation figure saved!')"

# --- 7. Generate Paper Figures ---
Write-Host "`n[7/7] Generating Paper Figures (UMAP, Grad-CAM, etc)..."
# UMAP/t-SNE Representation Visualization
# UMAP/t-SNE Representation Visualization (dual colored: by dataset + by condition)
python -m src.experiments.plot_umap --encoder resnet1d --checkpoint experiments/smoke/ssl_passl_resnet_hybrid/best_checkpoint.pth --datasets data/ptbxl_processed.csv --max_samples 2000 --output experiments/smoke/figures/umap_dual.png

# Grad-CAM 
python -m src.gradcam --checkpoint experiments/smoke/ssl_passl_resnet_hybrid/best_checkpoint.pth --data_file data/ptbxl_processed.csv --output experiments/smoke/figures/gradcam.png

Write-Host "`n================================================================"
Write-Host "PA-SSL: SMOKE TEST Complete!"
Write-Host "================================================================"
