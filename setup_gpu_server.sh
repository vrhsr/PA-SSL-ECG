#!/bin/bash
# =============================================================================
# PA-HybridSSL: GPU Server Setup & Smoke Test Script
# =============================================================================
# Run this once after git clone on your university GPU server.
#
# Usage:
#   chmod +x setup_gpu_server.sh
#   ./setup_gpu_server.sh
#
# What it does:
#   1. Checks GPU availability
#   2. Creates a conda/venv environment
#   3. Installs all dependencies (including GPU-only ones)
#   4. Runs the full unit test suite (28 tests, no data needed)
#   5. Runs a 1-epoch smoke test of the full pipeline (needs data)
# =============================================================================

set -e  # Exit on first error

echo "================================================================"
echo " PA-HybridSSL: GPU Server Setup"
echo " $(date)"
echo "================================================================"

# ── 1. GPU CHECK ─────────────────────────────────────────────────────────────
echo ""
echo "[1/6] Checking GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "  [WARN] nvidia-smi not found. Running on CPU only."
    GPU_AVAILABLE=false
else
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    GPU_AVAILABLE=true
    echo "  GPU detected ✓"
fi

echo ""
echo "[2/6] Checking CUDA + PyTorch..."
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# ── 2. INSTALL DEPENDENCIES ──────────────────────────────────────────────────
echo ""
echo "[3/6] Installing Python dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "  Core requirements installed ✓"

# Install GPU-only extras
if [ "$GPU_AVAILABLE" = true ]; then
    echo "  Installing GPU-only extras (mamba-ssm)..."
    pip install mamba-ssm --quiet 2>/dev/null && echo "  mamba-ssm installed ✓" || echo "  [WARN] mamba-ssm failed — Mamba encoder will use GRU fallback"
fi

# Optional but recommended
pip install wandb --quiet 2>/dev/null && echo "  wandb installed ✓" || true
pip install gdown --quiet 2>/dev/null && echo "  gdown installed ✓" || true

# ── 3. PYTHONPATH ─────────────────────────────────────────────────────────────
echo ""
echo "[4/6] Setting PYTHONPATH..."
export PYTHONPATH="$(pwd):$PYTHONPATH"
echo "  PYTHONPATH=$PYTHONPATH"

# Persist for this session + add to .bashrc
echo "export PYTHONPATH=\"$(pwd):\$PYTHONPATH\"" >> ~/.bashrc
echo "  Added to ~/.bashrc ✓"

# ── 4. UNIT TESTS (no data required) ─────────────────────────────────────────
echo ""
echo "[5/6] Running unit tests (28 tests, no data required)..."
python3 run_tests.py
echo "  ✓ All unit tests passed"

# ── 5. SMOKE TEST (requires at least PTB-XL data) ────────────────────────────
echo ""
echo "[6/6] Checking for processed data files..."
DATA_READY=true

if [ ! -f "data/ptbxl_processed.csv" ]; then
    echo "  [!] data/ptbxl_processed.csv not found."
    echo "  → To prepare: python -m src.data.emit_ptbxl --output data/ptbxl_processed.csv"
    DATA_READY=false
fi

if [ "$DATA_READY" = true ]; then
    echo "  Data found ✓ Running smoke test (1 epoch, 20 batches)..."
    python3 -m src.train_ssl \
        --encoder resnet1d \
        --augmentation physio \
        --use_temporal \
        --loss_type ntxent \
        --epochs 1 \
        --batch_size 128 \
        --max_batches 20 \
        --seed 42 \
        --data_file data/ptbxl_processed.csv \
        --output_dir experiments/smoke/ssl_setup_test
    echo "  ✓ Smoke training test passed"

    python3 -m src.evaluate \
        --checkpoint experiments/smoke/ssl_setup_test/best_checkpoint.pth \
        --data_file data/ptbxl_processed.csv \
        --encoder resnet1d \
        --max_batches 10 \
        --n_seeds 1
    echo "  ✓ Smoke evaluation test passed"
else
    echo "  [SKIP] Skipping smoke test — prepare data first, then:"
    echo "         bash experiments/run_gpu_pipeline.sh"
fi

echo ""
echo "================================================================"
echo " Setup Complete!"
echo "================================================================"
echo ""
echo " Next steps:"
echo "  1. Prepare datasets (if not done):"
echo "       python -m src.data.emit_ptbxl --output data/ptbxl_processed.csv"
echo "       python -m src.data.emit_mitbih --output data/mitbih_processed.csv"
echo "       python -m src.data.emit_chapman --output data/chapman_processed.csv"
echo ""
echo "  2. Run the full smoke test (validates entire pipeline, ~10min):"
echo "       bash experiments/run_gpu_pipeline.sh --smoke"
echo ""
echo "  3. Once smoke test passes, launch full training:"
echo "       nohup bash experiments/run_gpu_pipeline.sh --full > logs/full_pipeline.log 2>&1 &"
echo "       tail -f logs/full_pipeline.log"
echo ""
echo "  4. For hyperparameter search, use:"
echo "       bash experiments/run_hparam_search.sh"
echo ""
