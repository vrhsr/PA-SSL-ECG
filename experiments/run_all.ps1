# PA-SSL Experiment Runner (Top-Tier Edition v3 - RTX 4080 Full Power)
# Runs the full experiment pipeline from zero to paper-ready results.
# Chapman-Shaoxing integrated, metadata conditioning, multi-seed, t-SNE, Grad-CAM.

Write-Host "================================================================"
Write-Host "PA-SSL: Top-Tier Automated Research Pipeline (v3 - GPU Full Power)"
Write-Host "================================================================"

# --- Step 0: Set Python Path and Dependencies ---
$env:PYTHONPATH = "$PWD"
Write-Host "`n[0/8] Installing Dependencies..."
pip install -r requirements.txt

# --- Step 1: Data Processing (3 Datasets, 3 Continents) ---
Write-Host "`n[1/8] Processing Datasets..."
Write-Host "`nProcessing PTB-XL (Germany, ~22k records)..."
python -m src.data.emit_ptbxl --output data/ptbxl_processed.csv

Write-Host "`nProcessing MIT-BIH (USA, 48 records)..."
python -m src.data.emit_mitbih --output data/mitbih_processed.csv

Write-Host "`nProcessing Chapman-Shaoxing (China, ~10k records)..."
python -m src.data.emit_chapman --output data/chapman_processed.csv

# --- Step 2: SSL Pretraining - Main Configs (200 Epochs) ---
Write-Host "`n[2/8] SSL Pretraining (200 Epochs, Top-Tier Scale)..."

Write-Host "`nTraining: ResNet1D + PhysioAug + Temporal (MAIN)"
python -m src.train_ssl --encoder resnet1d --augmentation physio --use_temporal --epochs 200 --batch_size 256

Write-Host "`nTraining: WavKAN + PhysioAug + Temporal (MAIN)"
python -m src.train_ssl --encoder wavkan --augmentation physio --use_temporal --epochs 200 --batch_size 256

# --- Step 3: SSL Pretraining - Ablations (100 Epochs) ---
Write-Host "`n[3/8] SSL Pretraining Ablations (100 Epochs Scale)..."

Write-Host "`nTraining: ResNet1D + NaiveAug (ablation)"
python -m src.train_ssl --encoder resnet1d --augmentation naive --no_temporal --epochs 100 --batch_size 256

Write-Host "`nTraining: ResNet1D + PhysioAug, no temporal (ablation)"
python -m src.train_ssl --encoder resnet1d --augmentation physio --no_temporal --epochs 100 --batch_size 256

# --- Step 4: Phase 9 - Metadata-Conditioned Training (200 Epochs) ---
Write-Host "`n[4/8] Phase 9: Metadata-Conditioned SSL (200 Epochs)..."

Write-Host "`nTraining: ResNet1D + PhysioAug + Temporal + Metadata (NEXT-GEN)"
python -m src.train_ssl --encoder resnet1d --augmentation physio --use_temporal --use_metadata --epochs 200 --batch_size 256 --output_dir experiments

# --- Step 5: Core Evaluation (Same-Dataset) ---
Write-Host "`n[5/8] Downstream Evaluation (Same-Dataset)..."

python -m src.evaluate --checkpoint experiments/ssl_resnet1d_physio_temporal/best_checkpoint.pth --data_file data/ptbxl_processed.csv
python -m src.evaluate --checkpoint experiments/ssl_wavkan_physio_temporal/best_checkpoint.pth --data_file data/ptbxl_processed.csv --encoder wavkan

# --- Step 6: Cross-Dataset Transfer Matrix (3x3) ---
Write-Host "`n[6/8] Cross-Dataset Transfer Matrix (3x3)..."

Write-Host "`nPTB-XL -> MIT-BIH"
python -m src.evaluate --checkpoint experiments/ssl_resnet1d_physio_temporal/best_checkpoint.pth --data_file data/mitbih_processed.csv

Write-Host "`nPTB-XL -> Chapman"
python -m src.evaluate --checkpoint experiments/ssl_resnet1d_physio_temporal/best_checkpoint.pth --data_file data/chapman_processed.csv

# --- Step 7: Advanced GPU-Intensive Experiments ---
Write-Host "`n[7/8] Advanced Publishability Experiments..."

# 1. Full Ablation Suite (LaTeX Table)
Write-Host "`nRunning Full Ablation Suite..."
python -c "from src.experiments import run_ablation_suite; run_ablation_suite('data/ptbxl_processed.csv')"

# 2. Robustness Testing (Noise and Artifacts)
Write-Host "`nRunning Robustness Testing..."
python -c "from src.experiments import robustness_experiment; from src.models.encoder import build_encoder; import torch, pandas as pd; device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); encoder = build_encoder('resnet1d', proj_dim=128); ckpt = torch.load('experiments/ssl_resnet1d_physio_temporal/best_checkpoint.pth', map_location=device); encoder.load_state_dict(ckpt['encoder_state_dict']); results = robustness_experiment(encoder.to(device), 'data/ptbxl_processed.csv', device); results.to_csv('experiments/robustness_results.csv', index=False)"

# 3. OOD Detection (PTB-XL Normal vs Chapman Abnormal)
Write-Host "`nRunning OOD Detection (PTB-XL vs Chapman)..."
python -c "from src.experiments import ood_detection_experiment; from src.models.encoder import build_encoder; import torch, json; device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); encoder = build_encoder('resnet1d', proj_dim=128); ckpt = torch.load('experiments/ssl_resnet1d_physio_temporal/best_checkpoint.pth', map_location=device); encoder.load_state_dict(ckpt['encoder_state_dict']); results = ood_detection_experiment(encoder.to(device), 'data/ptbxl_processed.csv', 'data/chapman_processed.csv', device); f=open('experiments/ood_results.json','w'); json.dump({k: float(v) for k, v in results.items()}, f, indent=2); f.close()"

# 4. t-SNE Visualization
Write-Host "`nGenerating t-SNE Visualizations..."
python -c "import torch, numpy as np, os; from src.data.ecg_dataset import ECGBeatDataset; from src.models.encoder import build_encoder; from src.evaluate import extract_representations; from src.plotting import plot_tsne_embeddings; device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); os.makedirs('figures', exist_ok=True); encoder = build_encoder('resnet1d', proj_dim=128); ckpt = torch.load('experiments/ssl_resnet1d_physio_temporal/best_checkpoint.pth', map_location=device); encoder.load_state_dict(ckpt['encoder_state_dict']); encoder = encoder.to(device); dataset = ECGBeatDataset('data/ptbxl_processed.csv'); reprs, labels = extract_representations(encoder, dataset, device); idx = np.random.choice(len(reprs), min(5000, len(reprs)), replace=False); plot_tsne_embeddings(reprs[idx], labels[idx], method_name='PA-SSL (ResNet1D)', save_path='figures/tsne_passl_resnet.png'); print('t-SNE saved: figures/tsne_passl_resnet.png')"

# 5. Grad-CAM Interpretability
Write-Host "`nGenerating Grad-CAM Interpretability Maps..."
python -m src.gradcam --checkpoint experiments/ssl_resnet1d_physio_temporal/best_checkpoint.pth --data_file data/ptbxl_processed.csv --output figures/gradcam_interpretability.png

# --- Step 8: Paper-Quality Plots ---
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
Write-Host "================================================================"
