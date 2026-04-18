#!/bin/bash
# PA-SSL: Full Ablation Suite
# Corrected for actual train_ssl.py argument names:
#   --ssl_mode  : contrastive | mae | hybrid
#   --loss_type : ntxent | vicreg | barlow   (the contrastive objective)
#   --mask_ratio: 0.0-1.0 float  (NOT "80_20" format)
#   --no_qrs_protect: flag to disable QRS protection

set -e  # stop on first error
set -o pipefail # ensure exit codes propagate through pipes

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATA="${1:-data/ptbxl_processed.csv}"
EPOCHS="${2:-100}"
BS=256
LR=3e-4
SEEDS=(42 123 456)

if [ ! -f "$DATA" ]; then
    echo "❌ Data file not found: $DATA"
    echo "   Usage: bash experiments/run_ablation.sh [data_csv] [epochs]"
    exit 1
fi

mkdir -p logs experiments/ablation experiments/qrs_ablation \
          experiments/masking_sweep experiments/baselines results

echo "=================================================="
echo " PA-SSL Full Ablation"
echo " Data:   $DATA"
echo " Epochs: $EPOCHS"
echo " Seeds:  ${SEEDS[*]}"
echo " Started: $(date)"
echo "=================================================="

# ═══════════════════════════════════════════════════════════
# BLOCK 1: 3×2 ABLATION  (encoder × ssl_mode, 3 seeds each)
#   Encoders : resnet1d | wavkan
#   SSL modes: contrastive | mae | hybrid
#   = 6 cells × 3 seeds = 18 runs
# ═══════════════════════════════════════════════════════════
echo ""
echo "━━━ BLOCK 1: 3×2 Ablation (18 runs) ━━━"

for SEED in "${SEEDS[@]}"; do
for ENC in resnet1d wavkan; do
for MODE in contrastive mae hybrid; do

    NAME="${ENC}_${MODE}_s${SEED}"
    OUT="experiments/ablation/${NAME}"

    if [ -f "${OUT}/done.flag" ]; then
        echo "  ⏭  $NAME (already done)"
        continue
    fi

    echo ""
    echo "  🚀  $NAME  [encoder=$ENC  ssl_mode=$MODE  seed=$SEED]"
    mkdir -p "$OUT"

    # Only pass loss_type if not pure MAE
    if [ "$MODE" = "mae" ]; then
        LOSS_ARG=""
    else
        LOSS_ARG="--loss_type ntxent"
    fi

    python3 -m src.train_ssl \
        --encoder        "$ENC" \
        --ssl_mode       "$MODE" \
        --augmentation   physio \
        --use_temporal \
        $LOSS_ARG \
        --epochs         $EPOCHS \
        --batch_size     $BS \
        --lr             $LR \
        --seed           $SEED \
        --data_file      "$DATA" \
        --output_dir     "$OUT" \
        --save_every     20 \
        --num_workers    0 \
        2>&1 | tee "logs/abl_${NAME}.log"

    echo "  📊  Evaluating $NAME..."
    python3 -m src.evaluate \
        --checkpoint "$OUT/best_checkpoint.pth" \
        --data_file "$DATA" \
        --n_seeds 3 \
        2>&1 | tee "logs/eval_abl_${NAME}.log"

    touch "${OUT}/done.flag"
    echo "  ✅  $NAME done"

done; done; done

echo ""
echo "BLOCK 1 complete: $(date)"

# ═══════════════════════════════════════════════════════════
# BLOCK 2: QRS PROTECTION ISOLATION
#   Best encoder (resnet1d) + hybrid mode
#   WITH vs WITHOUT QRS protection  (seed=42 only, 2 runs)
# ═══════════════════════════════════════════════════════════
echo ""
echo "━━━ BLOCK 2: QRS Protection Isolation (6 runs) ━━━"

for SEED in "${SEEDS[@]}"; do
for QRS_FLAG in "" "--no_qrs_protect"; do
    if [ -z "$QRS_FLAG" ]; then
        LABEL="qrs_protected"
    else
        LABEL="qrs_unprotected"
    fi

    NAME="resnet1d_hybrid_${LABEL}_s${SEED}"
    OUT="experiments/qrs_ablation/${NAME}"

    if [ -f "${OUT}/done.flag" ]; then
        echo "  ⏭  $NAME (already done)"
        continue
    fi

    echo ""
    echo "  🚀  $NAME  [QRS flag: '${QRS_FLAG:-none}', seed=$SEED]"
    mkdir -p "$OUT"

    python3 -m src.train_ssl \
        --encoder        resnet1d \
        --ssl_mode       hybrid \
        --augmentation   physio \
        --use_temporal \
        --loss_type      ntxent \
        $QRS_FLAG \
        --epochs         $EPOCHS \
        --batch_size     $BS \
        --lr             $LR \
        --seed           $SEED \
        --data_file      "$DATA" \
        --output_dir     "$OUT" \
        --num_workers    4 \
        2>&1 | tee "logs/qrs_${NAME}.log"

    echo "  📊  Evaluating $NAME..."
    python3 -m src.evaluate \
        --checkpoint "$OUT/best_checkpoint.pth" \
        --data_file "$DATA" \
        --n_seeds 3 \
        2>&1 | tee "logs/eval_qrs_${NAME}.log"

    touch "${OUT}/done.flag"
    echo "  ✅  $NAME done"

done; done

echo ""
echo "BLOCK 2 complete: $(date)"

# ═══════════════════════════════════════════════════════════
# BLOCK 3: MASKING RATIO SENSITIVITY
#   --mask_ratio takes a float 0.0–1.0
#   0.0 = no masking (pure contrastive), 1.0 = mask everything
#   Test: 0.0, 0.3, 0.5, 0.6 (default), 0.7, 0.9
# ═══════════════════════════════════════════════════════════
echo ""
echo "━━━ BLOCK 3: Masking Ratio Sensitivity (5 runs) ━━━"

for RATIO in 0.0 0.2 0.5 0.8 1.0; do
    RATIO_LABEL=$(echo "$RATIO" | tr '.' '_')
    NAME="resnet1d_hybrid_mask${RATIO_LABEL}_s42"
    OUT="experiments/masking_sweep/${NAME}"
    
    # Conceptual fix: mask ratio 0.0 is purely contrastive.
    if [ "$RATIO" = "0.0" ]; then
        MODE="contrastive"
    else
        MODE="hybrid"
    fi

    if [ -f "${OUT}/done.flag" ]; then
        echo "  ⏭  $NAME (already done)"
        continue
    fi

    echo ""
    echo "  🚀  $NAME  [mask_ratio=$RATIO, resolved_mode=$MODE]"
    mkdir -p "$OUT"

    python3 -m src.train_ssl \
        --encoder        resnet1d \
        --ssl_mode       "$MODE" \
        --augmentation   physio \
        --use_temporal \
        --loss_type      ntxent \
        --mask_ratio     "$RATIO" \
        --epochs         $EPOCHS \
        --batch_size     $BS \
        --lr             $LR \
        --seed           42 \
        --data_file      "$DATA" \
        --output_dir     "$OUT" \
        --num_workers    4 \
        2>&1 | tee "logs/mask_${NAME}.log"

    echo "  📊  Evaluating $NAME..."
    python3 -m src.evaluate \
        --checkpoint "$OUT/best_checkpoint.pth" \
        --data_file "$DATA" \
        --n_seeds 3 \
        2>&1 | tee "logs/eval_mask_${NAME}.log"

    touch "${OUT}/done.flag"
    echo "  ✅  $NAME done"

done

echo ""
echo "BLOCK 3 complete: $(date)"

# ═══════════════════════════════════════════════════════════
# BLOCK 4: BASELINES
# ═══════════════════════════════════════════════════════════
echo ""
echo "━━━ BLOCK 4: Baselines (2 runs) ━━━"

# SimCLR style: naive augmentations, pure contrastive
NAME="simclr_naive_s42"
OUT="experiments/baselines/${NAME}"
if [ ! -f "${OUT}/done.flag" ]; then
    echo "  🚀  $NAME  [naive augmentation + ntxent]"
    mkdir -p "$OUT"
    python3 -m src.train_ssl \
        --encoder        resnet1d \
        --ssl_mode       contrastive \
        --augmentation   naive \
        --no_temporal \
        --loss_type      ntxent \
        --epochs         $EPOCHS \
        --batch_size     $BS \
        --lr             $LR \
        --seed           42 \
        --data_file      "$DATA" \
        --output_dir     "$OUT" \
        --num_workers    4 \
        2>&1 | tee "logs/bl_${NAME}.log"
        
    echo "  📊  Evaluating $NAME..."
    python3 -m src.evaluate \
        --checkpoint "$OUT/best_checkpoint.pth" \
        --data_file "$DATA" \
        --n_seeds 3 \
        2>&1 | tee "logs/eval_bl_${NAME}.log"
        
    touch "${OUT}/done.flag"
    echo "  ✅  $NAME done"
fi

# Supervised baseline — REAL END-TO-END SUPERVISED (no SSL)
NAME="supervised_true_baseline_s42"
OUT="experiments/baselines/${NAME}"
if [ ! -f "${OUT}/done.flag" ]; then
    echo "  🚀  $NAME  [True end-to-end supervised training Baseline]"
    mkdir -p "$OUT"
    python3 -c "import torch; from src.baselines import train_supervised; \
res = train_supervised('$DATA', torch.device('cuda' if torch.cuda.is_available() else 'cpu'), epochs=$EPOCHS, batch_size=64); \
res.to_csv('$OUT/supervised_results.csv', index=False)" \
        2>&1 | tee "logs/bl_${NAME}.log"
    touch "${OUT}/done.flag"
    echo "  ✅  $NAME done"
fi

echo ""
echo "BLOCK 4 complete: $(date)"

# ═══════════════════════════════════════════════════════════
# BLOCK 5: COMPUTATIONAL COST BENCHMARK
# ═══════════════════════════════════════════════════════════
echo ""
echo "━━━ BLOCK 5: Computational Cost Benchmark ━━━"

python3 -m src.experiments.computational_cost \
    --output results/compute_cost.csv \
    2>&1 | tee logs/compute_cost.log

echo ""
echo "━━━ BLOCK 6: Leave-One-Out (LOO) Augmentation Ablation (8 runs) ━━━"
echo "Isolating the contribution of each physiological augmentation."

AUGS=("constrained_time_warp" "amplitude_perturbation" "baseline_wander" "emg_noise_injection" "heart_rate_resample" "powerline_interference" "segment_dropout" "wavelet_masking")

for AUG in "${AUGS[@]}"; do
    NAME="resnet1d_hybrid_loo_${AUG}_s42"
    OUT="experiments/loo_ablation/${NAME}"

    if [ -f "${OUT}/done.flag" ]; then
        echo "  ⏭  $NAME (already done)"
        continue
    fi

    echo ""
    echo "  🚀  $NAME  [excluding: $AUG]"
    mkdir -p "$OUT"

    python3 -m src.train_ssl \
        --encoder        resnet1d \
        --ssl_mode       hybrid \
        --augmentation   physio \
        --exclude_aug    "$AUG" \
        --use_temporal \
        --loss_type      ntxent \
        --epochs         $EPOCHS \
        --batch_size     $BS \
        --lr             $LR \
        --seed           42 \
        --data_file      "$DATA" \
        --output_dir     "$OUT" \
        --num_workers    4 \
        2>&1 | tee "logs/loo_${NAME}.log"

    echo "  📊  Evaluating $NAME..."
    python3 -m src.evaluate \
        --checkpoint "$OUT/best_checkpoint.pth" \
        --data_file "$DATA" \
        --n_seeds 3 \
        2>&1 | tee "logs/eval_loo_${NAME}.log"

    touch "${OUT}/done.flag"
    echo "  ✅  $NAME done"
done

echo ""
echo "BLOCK 6 complete: $(date)"

echo ""
echo "=================================================="
echo " ALL DONE: $(date)"
echo " Completed runs:"
find experiments -name "done.flag" | wc -l
echo "=================================================="
