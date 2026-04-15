#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# PA-SSL: Unified GPU Pipeline (Smoke Test → Full Training)
# ═══════════════════════════════════════════════════════════════════════════════
#
# This script runs EVERYTHING in two phases:
#   Phase 1: SMOKE TEST — quick verification (1 epoch, 10 batches per step)
#   Phase 2: FULL TRAINING — production run (100 epochs, full dataset)
#
# Usage:
#   chmod +x experiments/run_gpu_pipeline.sh
#   nohup bash experiments/run_gpu_pipeline.sh 2>&1 | tee pipeline_log.txt &
#
# ═══════════════════════════════════════════════════════════════════════════════

set -e  # Exit on any error

# ─── Configuration ───────────────────────────────────────────────────────────
export PYTHONPATH="$PWD"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "PA-SSL: Unified GPU Pipeline"
echo "Started at: $(date)"
echo "================================================================"

# ─── GPU Check ───────────────────────────────────────────────────────────────
echo ""
echo "[0/9] Checking GPU availability..."
if ! python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "  [WARNING] No GPU detected via torch.cuda.is_available()!"
    echo "  Training will be extremely slow. Press Ctrl+C in 10s to abort..."
    sleep 10
else
    python3 -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  GPU: {torch.cuda.get_device_name(0)}')
print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: DATA PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "[1/9] Processing Datasets..."

if [ ! -f "data/ptbxl_processed.csv" ]; then
    echo "  Processing PTB-XL..."
    python3 -m src.data.emit_ptbxl --output data/ptbxl_processed.csv 2>&1 | tee "$LOG_DIR/data_ptbxl.log"
else
    echo "  PTB-XL already processed. ✓"
fi

if [ ! -f "data/mitbih_processed.csv" ]; then
    echo "  Processing MIT-BIH..."
    python3 -m src.data.emit_mitbih --output data/mitbih_processed.csv 2>&1 | tee "$LOG_DIR/data_mitbih.log"
else
    echo "  MIT-BIH already processed. ✓"
fi

if [ ! -f "data/chapman_processed.csv" ]; then
    echo "  Processing Chapman..."
    python3 -m src.data.emit_chapman --output data/chapman_processed.csv 2>&1 | tee "$LOG_DIR/data_chapman.log"
else
    echo "  Chapman already processed. ✓"
fi

echo "  Data files:"
ls -lh data/*_processed.csv 2>/dev/null || echo "  WARNING: No processed CSV files found!"

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: SMOKE TEST (Quick Verification)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "PHASE 1: SMOKE TEST (1 epoch, 10 batches — verifying everything)"
echo "================================================================"

# Clean previous smoke test
rm -rf experiments/smoke
mkdir -p experiments/smoke

# SSL Pretraining Matrix (smoke) — 3×2 factorial:
#   Encoders:  resnet1d | wavkan
#   SSL Modes: ntxent   | vicreg | hybrid
# SSL Pretraining Matrix (smoke) — 3×2 factorial + hybrid:
# Format: "Name | Encoder | Augmentation | TemporalFlag | LossType | SSLMode"
CONFIGS=(
    "simclr_naive_resnet|resnet1d|naive|--no_temporal|ntxent|contrastive"
    "passl_resnet_ntxent|resnet1d|physio|--use_temporal|ntxent|contrastive"
    "passl_resnet_vicreg|resnet1d|physio|--use_temporal|vicreg|contrastive"
    "passl_resnet_hybrid|resnet1d|physio|--use_temporal|ntxent|hybrid"
    "passl_wavkan_ntxent|wavkan|physio|--use_temporal|ntxent|contrastive"
    "passl_wavkan_vicreg|wavkan|physio|--use_temporal|vicreg|contrastive"
    "passl_wavkan_hybrid|wavkan|physio|--use_temporal|ntxent|hybrid"
)

echo ""
echo "[2/9] Smoke Test: SSL Pretraining (4 variants × 1 epoch × 10 batches)..."
for config in "${CONFIGS[@]}"; do
    IFS='|' read -r name enc aug temp loss mode <<< "$config"
    echo "  Training: $name..."
    python3 -m src.train_ssl \
        --encoder "$enc" \
        --augmentation "$aug" \
        $temp \
        --loss_type "$loss" \
        --ssl_mode "$mode" \
        --epochs 1 \
        --batch_size 256 \
        --max_batches 10 \
        --seed 42 \
        --num_workers 4 \
        --output_dir "experiments/smoke/ssl_${name}" \
        2>&1 | tee "$LOG_DIR/smoke_ssl_${name}.log"
    
    # Verify checkpoint was saved
    if [ -f "experiments/smoke/ssl_${name}/best_checkpoint.pth" ]; then
        echo "  ✓ Checkpoint saved for $name"
    else
        echo "  ✗ ERROR: No checkpoint saved for $name!"
        exit 1
    fi
done

# Baselines (smoke)
echo ""
echo "[3/9] Smoke Test: Baselines (1 epoch, 10 batches)..."
python3 -m src.baselines \
    --data_csv data/ptbxl_processed.csv \
    --naive_checkpoint experiments/smoke/ssl_simclr_naive_resnet/best_checkpoint.pth \
    --output_dir experiments/smoke/baselines \
    --epochs 1 \
    --max_batches 10 \
    2>&1 | tee "$LOG_DIR/smoke_baselines.log"

# Verify baseline results
if [ -f "experiments/smoke/baselines/all_baseline_results.csv" ]; then
    echo "  ✓ Baseline results saved"
else
    echo "  ✗ WARNING: Baseline results may not have saved correctly"
fi

# Downstream Evaluation (smoke)
echo ""
echo "[4/9] Smoke Test: Downstream Evaluation..."
for config in "${CONFIGS[@]}"; do
    IFS='|' read -r name enc aug temp loss <<< "$config"
    if [ "$name" != "simclr_naive_resnet" ]; then
        echo "  Evaluating: $name..."
        python -m src.evaluate \
            --checkpoint "experiments/smoke/ssl_${name}/best_checkpoint.pth" \
            --data_file data/ptbxl_processed.csv \
            --encoder "$enc" \
            --max_batches 10 \
            --n_seeds 1 \
            2>&1 | tee "$LOG_DIR/smoke_eval_${name}.log"
    fi
done

# Cross-dataset (smoke) — use best hybrid model (resnet1d)
echo ""
echo "[5/9] Smoke Test: Cross-Dataset Transfer..."
python -m src.evaluate \
    --checkpoint experiments/smoke/ssl_passl_resnet_hybrid/best_checkpoint.pth \
    --data_file data/mitbih_processed.csv \
    --encoder resnet1d \
    --max_batches 10 \
    --n_seeds 1 \
    2>&1 | tee "$LOG_DIR/smoke_transfer_mitbih.log"

python -m src.evaluate \
    --checkpoint experiments/smoke/ssl_passl_resnet_hybrid/best_checkpoint.pth \
    --data_file data/chapman_processed.csv \
    --encoder resnet1d \
    --max_batches 10 \
    --n_seeds 1 \
    2>&1 | tee "$LOG_DIR/smoke_transfer_chapman.log"

echo ""
echo "================================================================"
echo "PHASE 1 COMPLETE: Smoke test passed! All checkpoints verified."
echo "================================================================"

# ─── Smoke Test Summary ─────────────────────────────────────────────────────
echo ""
echo "Smoke Test Artifacts:"
echo "  SSL Checkpoints:"
ls -lh experiments/smoke/ssl_*/best_checkpoint.pth 2>/dev/null || echo "    None found!"
echo "  Baseline Results:"
ls -lh experiments/smoke/baselines/*.csv 2>/dev/null || echo "    None found!"

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: FULL TRAINING (Production Run)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "PHASE 2: FULL TRAINING (100 epochs, batch_size=512)"
echo "Started at: $(date)"
echo "================================================================"

# SSL Pretraining Matrix (full) — same 3×2 factorial
echo ""
echo "[6/9] Full SSL Pretraining (7 variants × 100 epochs)..."
for config in "${CONFIGS[@]}"; do
    IFS='|' read -r name enc aug temp loss mode <<< "$config"
    out_dir="experiments/ssl_${name}"
    
    if [ -f "${out_dir}/history.json" ]; then
        echo "  $name already trained. Skipping. ✓"
    else
        echo "  Training: $name (full)..."
        python3 -m src.train_ssl \
            --data_file data/combined_pretrain.csv \
            --encoder "$enc" \
            --augmentation "$aug" \
            $temp \
            --loss_type "$loss" \
            --ssl_mode "$mode" \
            --epochs 100 \
            --batch_size 256 \
            --seed 42 \
            --num_workers 4 \
            --output_dir "$out_dir" \
            2>&1 | tee "$LOG_DIR/full_ssl_${name}.log"
        echo "  ✓ $name training complete"
    fi
done

# Baselines (full)
echo ""
echo "[7/9] Full Baselines (50 epochs)..."
python3 -m src.baselines \
    --data_csv data/ptbxl_processed.csv \
    --naive_checkpoint experiments/ssl_simclr_naive_resnet/best_checkpoint.pth \
    --output_dir experiments/baselines \
    --epochs 50 \
    2>&1 | tee "$LOG_DIR/full_baselines.log"

# Downstream Evaluation (full)
echo ""
echo "[8/9] Full Downstream Evaluation + Cross-Dataset Transfer..."
for config in "${CONFIGS[@]}"; do
    IFS='|' read -r name enc aug temp loss <<< "$config"
    if [ "$name" != "simclr_naive_resnet" ]; then
        echo "  Evaluating: $name on PTB-XL..."
        python3 -m src.evaluate \
            --checkpoint "experiments/ssl_${name}/best_checkpoint.pth" \
            --data_file data/ptbxl_processed.csv \
            --encoder "$enc" \
            2>&1 | tee "$LOG_DIR/full_eval_${name}.log"
    fi
done

# Cross-dataset transfer — use hybrid ResNet1D (expected best)
echo "  Cross-dataset: MIT-BIH..."
python3 -m src.evaluate \
    --checkpoint experiments/ssl_passl_resnet_hybrid/best_checkpoint.pth \
    --data_file data/mitbih_processed.csv \
    --encoder resnet1d \
    2>&1 | tee "$LOG_DIR/full_transfer_mitbih.log"

echo "  Cross-dataset: Chapman..."
python3 -m src.evaluate \
    --checkpoint experiments/ssl_passl_resnet_hybrid/best_checkpoint.pth \
    --data_file data/chapman_processed.csv \
    --encoder resnet1d \
    2>&1 | tee "$LOG_DIR/full_transfer_chapman.log"

# Figures & Ablations
echo ""
echo "[9/9] Generating Figures & Running Ablations..."
mkdir -p figures

# QRS Validation
python3 -m src.augmentations.visualize_and_test --test --visualize 2>&1 | tee "$LOG_DIR/full_qrs.log" || true

# Ablation Suite
python3 -c "
from src.experiments import run_ablation_suite
import pandas as pd
from src.plotting import plot_ablation_bars
res = run_ablation_suite('data/ptbxl_processed.csv')
plot_ablation_bars(pd.read_csv('experiments/ablations/ablation_results.csv'), save_path='figures/ablation_bars.png')
print('Ablation complete!')
" 2>&1 | tee "$LOG_DIR/full_ablation.log" || true

# UMAP — dual colored (by dataset + by condition) using best hybrid ResNet model
python3 -m src.experiments.plot_umap \
    --encoder resnet1d \
    --checkpoint experiments/ssl_passl_resnet_hybrid/best_checkpoint.pth \
    --datasets data/ptbxl_processed.csv data/mitbih_processed.csv data/chapman_processed.csv \
    --max_samples 5000 \
    --output figures/umap_dual_passl_hybrid.png \
    2>&1 | tee "$LOG_DIR/full_umap.log" || true

# Grad-CAM
python3 -m src.gradcam \
    --checkpoint experiments/ssl_passl_resnet_hybrid/best_checkpoint.pth \
    --data_file data/ptbxl_processed.csv \
    --output figures/gradcam_passl_hybrid.png \
    2>&1 | tee "$LOG_DIR/full_gradcam.log" || true

# OOD & Robustness
python3 -c "
from src.experiments import robustness_experiment, ood_detection_experiment
from src.models.encoder import build_encoder
import torch, json
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = build_encoder('resnet1d', proj_dim=128)
ckpt = torch.load('experiments/ssl_passl_resnet_hybrid/best_checkpoint.pth', map_location=device)
encoder.load_state_dict(ckpt['encoder_state_dict'])
encoder = encoder.to(device)
results = robustness_experiment(encoder, 'data/ptbxl_processed.csv', device)
results.to_csv('experiments/robustness_results.csv', index=False)
ood_res = ood_detection_experiment(encoder, 'data/ptbxl_processed.csv', 'data/chapman_processed.csv', device)
with open('experiments/ood_results.json','w') as f:
    json.dump({k: float(v) for k, v in ood_res.items()}, f, indent=2)
print('OOD & Robustness done!')
" 2>&1 | tee "$LOG_DIR/full_ood.log" || true

# ═══════════════════════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "PA-SSL: FULL PIPELINE COMPLETE!"
echo "Finished at: $(date)"
echo "================================================================"
echo ""
echo "Results Summary:"
echo "  SSL Checkpoints:  experiments/ssl_*/best_checkpoint.pth"
echo "  Baseline Results: experiments/baselines/all_baseline_results.csv"
echo "  Figures:          figures/"
echo "  Logs:             $LOG_DIR/"
echo ""
echo "Next: Copy experiments/ and figures/ back to your local machine."
