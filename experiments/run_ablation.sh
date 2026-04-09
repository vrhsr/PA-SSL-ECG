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

    python3 -m src.train_ssl \
        --encoder        "$ENC" \
        --ssl_mode       "$MODE" \
        --augmentation   physio \
        --use_temporal \
        --loss_type      ntxent \
        --epochs         $EPOCHS \
        --batch_size     $BS \
        --lr             $LR \
        --seed           $SEED \
        --data_file      "$DATA" \
        --output_dir     "$OUT" \
        --save_every     20 \
        --num_workers    0 \
        2>&1 | tee "logs/abl_${NAME}.log"

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
echo "━━━ BLOCK 2: QRS Protection Isolation (2 runs) ━━━"

for QRS_FLAG in "" "--no_qrs_protect"; do
    if [ -z "$QRS_FLAG" ]; then
        LABEL="qrs_protected"
    else
        LABEL="qrs_unprotected"
    fi

    NAME="resnet1d_hybrid_${LABEL}_s42"
    OUT="experiments/qrs_ablation/${NAME}"

    if [ -f "${OUT}/done.flag" ]; then
        echo "  ⏭  $NAME (already done)"
        continue
    fi

    echo ""
    echo "  🚀  $NAME  [QRS flag: '${QRS_FLAG:-none}']"
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
        --seed           42 \
        --data_file      "$DATA" \
        --output_dir     "$OUT" \
        --num_workers    4 \
        2>&1 | tee "logs/qrs_${NAME}.log"

    touch "${OUT}/done.flag"
    echo "  ✅  $NAME done"

done

echo ""
echo "BLOCK 2 complete: $(date)"

# ═══════════════════════════════════════════════════════════
# BLOCK 3: MASKING RATIO SENSITIVITY
#   --mask_ratio takes a float 0.0–1.0
#   0.0 = no masking (pure contrastive), 1.0 = mask everything
#   Test: 0.0, 0.3, 0.5, 0.6 (default), 0.7, 0.9
# ═══════════════════════════════════════════════════════════
echo ""
echo "━━━ BLOCK 3: Masking Ratio Sensitivity (6 runs) ━━━"

for RATIO in 0.0 0.3 0.5 0.6 0.7 0.9; do
    RATIO_LABEL=$(echo "$RATIO" | tr '.' '_')
    NAME="resnet1d_hybrid_mask${RATIO_LABEL}_s42"
    OUT="experiments/masking_sweep/${NAME}"

    if [ -f "${OUT}/done.flag" ]; then
        echo "  ⏭  $NAME (already done)"
        continue
    fi

    echo ""
    echo "  🚀  $NAME  [mask_ratio=$RATIO]"
    mkdir -p "$OUT"

    python3 -m src.train_ssl \
        --encoder        resnet1d \
        --ssl_mode       hybrid \
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
    touch "${OUT}/done.flag"
    echo "  ✅  $NAME done"
fi

# Supervised baseline — no pretraining (random init)
NAME="supervised_random_init_s42"
OUT="experiments/baselines/${NAME}"
if [ ! -f "${OUT}/done.flag" ]; then
    echo "  🚀  $NAME  [no SSL — just to measure random init ceiling]"
    mkdir -p "$OUT"
    python3 -m src.train_ssl \
        --encoder        resnet1d \
        --ssl_mode       contrastive \
        --augmentation   none \
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
echo "=================================================="
echo " ALL DONE: $(date)"
echo " Completed runs:"
find experiments -name "done.flag" | wc -l
echo "=================================================="
