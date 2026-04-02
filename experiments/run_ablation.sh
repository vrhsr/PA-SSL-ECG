#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# PA-SSL: 3×2 Factorial Ablation Runner
# ═══════════════════════════════════════════════════════════════════════════════
#
# Runs the complete 3×2 factorial experiment matrix:
#   Encoders:   CNN (resnet1d) | KAN (wavkan)
#   SSL Modes:  contrastive    | mae          | hybrid
#
# Total: 6 pretraining runs + 6 downstream evaluation runs
#
# Usage:
#   chmod +x experiments/run_ablation.sh
#   ./experiments/run_ablation.sh [--quick]    # quick = smoke test mode
# ═══════════════════════════════════════════════════════════════════════════════

set -e

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_FILE="${DATA_FILE:-data/ptbxl_processed.csv}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="3e-4"
QUICK_TEST=""

if [[ "$1" == "--quick" ]]; then
    echo "═══ QUICK / SMOKE TEST MODE ═══"
    QUICK_TEST="--quick_test"
    EPOCHS=2
fi

ENCODERS=("resnet1d" "wavkan")
SSL_MODES=("contrastive" "mae" "hybrid")

RESULTS_DIR="experiments/ablation_results"
mkdir -p "$RESULTS_DIR"

echo "═══════════════════════════════════════════════════════════════"
echo "  PA-SSL 3×2 Factorial Ablation"
echo "  Data: $DATA_FILE"
echo "  Epochs: $EPOCHS"
echo "  Encoders: ${ENCODERS[*]}"
echo "  SSL Modes: ${SSL_MODES[*]}"
echo "═══════════════════════════════════════════════════════════════"

# ─── Phase 1: Pretraining ─────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════╗"
echo "║  PHASE 1: SSL PRETRAINING (6 runs)  ║"
echo "╚══════════════════════════════════════╝"

for encoder in "${ENCODERS[@]}"; do
    for ssl_mode in "${SSL_MODES[@]}"; do
        EXP_NAME="ssl_${encoder}_${ssl_mode}"
        EXP_DIR="$RESULTS_DIR/$EXP_NAME"
        mkdir -p "$EXP_DIR"

        echo ""
        echo "──────────────────────────────────────"
        echo "  Training: $EXP_NAME"
        echo "──────────────────────────────────────"

        python -m src.train_ssl \
            --encoder "$encoder" \
            --ssl_mode "$ssl_mode" \
            --augmentation physio \
            --data_file "$DATA_FILE" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --lr "$LR" \
            --output_dir "$EXP_DIR" \
            $QUICK_TEST \
            2>&1 | tee "$EXP_DIR/train.log"

        echo "  ✓ $EXP_NAME pretraining complete"
    done
done

# ─── Phase 2: Downstream Evaluation ──────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  PHASE 2: DOWNSTREAM EVALUATION (6 runs)║"
echo "╚══════════════════════════════════════════╝"

for encoder in "${ENCODERS[@]}"; do
    for ssl_mode in "${SSL_MODES[@]}"; do
        EXP_NAME="ssl_${encoder}_${ssl_mode}"
        EXP_DIR="$RESULTS_DIR/$EXP_NAME"
        CKPT="$EXP_DIR/best_checkpoint.pth"

        if [ ! -f "$CKPT" ]; then
            echo "  ⚠ Checkpoint not found for $EXP_NAME, skipping evaluation."
            continue
        fi

        echo ""
        echo "──────────────────────────────────────"
        echo "  Evaluating: $EXP_NAME"
        echo "──────────────────────────────────────"

        python -m src.evaluate \
            --encoder "$encoder" \
            --checkpoint "$CKPT" \
            --data_file "$DATA_FILE" \
            --output_dir "$EXP_DIR" \
            2>&1 | tee "$EXP_DIR/eval.log"

        echo "  ✓ $EXP_NAME evaluation complete"
    done
done

# ─── Phase 3: Dataset Invariance Diagnostic ──────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  PHASE 3: DATASET INVARIANCE DIAGNOSTIC ║"
echo "╚══════════════════════════════════════════╝"

# Only run if all 3 datasets exist
PTBXL="data/ptbxl_processed.csv"
MITBIH="data/mitbih_processed.csv"
CHAPMAN="data/chapman_processed.csv"

if [ -f "$PTBXL" ] && [ -f "$MITBIH" ] && [ -f "$CHAPMAN" ]; then
    for encoder in "${ENCODERS[@]}"; do
        # Use the hybrid model (expected best) for invariance test
        CKPT="$RESULTS_DIR/ssl_${encoder}_hybrid/best_checkpoint.pth"
        if [ -f "$CKPT" ]; then
            echo "  Testing domain invariance for $encoder (hybrid) ..."
            python -m src.experiments.dataset_invariance \
                --encoder "$encoder" \
                --checkpoint "$CKPT" \
                --datasets "$PTBXL" "$MITBIH" "$CHAPMAN" \
                2>&1 | tee "$RESULTS_DIR/domain_invariance_${encoder}.log"
        fi
    done
else
    echo "  ⚠ Not all 3 datasets found. Skipping invariance diagnostic."
    echo "    Expected: $PTBXL, $MITBIH, $CHAPMAN"
fi

# ─── Phase 4: Few-Shot Evaluation ────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════╗"
echo "║  PHASE 4: FEW-SHOT EVALUATION       ║"
echo "╚══════════════════════════════════════╝"

for encoder in "${ENCODERS[@]}"; do
    for ssl_mode in "${SSL_MODES[@]}"; do
        EXP_NAME="ssl_${encoder}_${ssl_mode}"
        CKPT="$RESULTS_DIR/$EXP_NAME/best_checkpoint.pth"

        if [ ! -f "$CKPT" ]; then
            continue
        fi

        echo "  Few-shot evaluation for $EXP_NAME ..."
        python -m src.experiments.few_shot_evaluation \
            --encoder "$encoder" \
            --checkpoint "$CKPT" \
            --data_file "$DATA_FILE" \
            --output "$RESULTS_DIR/${EXP_NAME}_fewshot.csv" \
            2>&1 | tee "$RESULTS_DIR/${EXP_NAME}_fewshot.log"
    done
done

# ─── Phase 5: Morphology Preservation Diagnostic ──────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  PHASE 5: MORPHOLOGY PRESERVATION TEST   ║"
echo "╚══════════════════════════════════════════╝"

echo "  Quantifying augmentation preservation quality..."
python -m src.experiments.morphology_metrics \
    --data_file "$DATA_FILE" \
    --n_samples 500 \
    --output "$RESULTS_DIR/morphology_preservation.csv" \
    2>&1 | tee "$RESULTS_DIR/morphology.log"

# ─── Phase 6: Noise Robustness Diagnostic ───────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  PHASE 6: NOISE ROBUSTNESS TEST          ║"
echo "╚══════════════════════════════════════════╝"

for encoder in "${ENCODERS[@]}"; do
    CKPT="$RESULTS_DIR/ssl_${encoder}_hybrid/best_checkpoint.pth"
    if [ -f "$CKPT" ]; then
        echo "  Testing noise robustness for $encoder (hybrid) ..."
        python -m src.experiments.noise_robustness \
            --encoder "$encoder" \
            --checkpoint "$CKPT" \
            --data_file "$DATA_FILE" \
            --output "$RESULTS_DIR/noise_robustness_${encoder}.csv" \
            2>&1 | tee "$RESULTS_DIR/noise_robustness_${encoder}.log"
    fi
done

# ─── Phase 7: Embedding Collapse Diagnostic ─────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  PHASE 7: EMBEDDING COLLAPSE TEST       ║"
echo "╚══════════════════════════════════════════╝"

for encoder in "${ENCODERS[@]}"; do
    CKPT="$RESULTS_DIR/ssl_${encoder}_hybrid/best_checkpoint.pth"
    if [ -f "$CKPT" ]; then
        echo "  Analyzing embedding richness for $encoder (hybrid) ..."
        python -m src.experiments.embedding_analysis \
            --encoder "$encoder" \
            --checkpoint "$CKPT" \
            --data_file "$DATA_FILE" \
            --output "$RESULTS_DIR/embedding_analysis_${encoder}.json" \
            2>&1 | tee "$RESULTS_DIR/embedding_analysis_${encoder}.log"
    fi
done

# ─── Phase 8: Data Scaling Experiment (Optional) ────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  PHASE 8: PRETRAINING SCALE TEST        ║"
echo "╚══════════════════════════════════════════╝"

FRACTIONS=(0.25 0.50)
for frac in "${FRACTIONS[@]}"; do
    echo "  Running scaling experiment: fraction $frac"
    python -m src.train_ssl \
        --encoder resnet1d \
        --ssl_mode hybrid \
        --data_file "$DATA_FILE" \
        --data_fraction "$frac" \
        --epochs 10 \
        --exp_name "scaling_$frac" \
        --results_dir "$RESULTS_DIR/scaling" \
        2>&1 | tee "$RESULTS_DIR/scaling/log_$frac.txt"
done

# ─── Phase 9: Augmentation Ablation Diagnostic ──────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  PHASE 9: AUGMENTATION ABLATION TEST     ║"
echo "╚══════════════════════════════════════════╝"

# Test specific physiological augmentations vs naive
AUGS=("none" "naive" "physio")
for aug in "${AUGS[@]}"; do
    echo "  Testing augmentation impact: $aug"
    python -m src.train_ssl \
        --encoder resnet1d \
        --ssl_mode contrastive \
        --augmentation "$aug" \
        --epochs 10 \
        --exp_name "aug_ablation_$aug" \
        --output_dir "$RESULTS_DIR/aug_ablation" \
        2>&1 | tee "$RESULTS_DIR/aug_ablation/log_$aug.txt"
done

# ─── Phase 10: Raw Dataset Visualization ─────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  PHASE 10: RAW DATASET DISTRIBUTION     ║"
echo "╚══════════════════════════════════════════╝"

echo "  Generating Raw ECG UMAP (Domain Shift Proof)..."
# We assume a small script or plot command exists in plotting.py linked to a runner
# For now, we'll call a dedicated experiment script if it exists, or just log
python -m src.experiments.plot_umap \
    --datasets "$PTBXL" "$MITBIH" "$CHAPMAN" \
    --encoder resnet1d \
    --checkpoint "$RESULTS_DIR/ssl_resnet1d_hybrid/best_checkpoint.pth" \
    --output "$RESULTS_DIR/umap_domain_invariance.png" \
    2>&1 | tee "$RESULTS_DIR/umap.log"

# ─── Phase 11: Training Stability Analysis ───────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  PHASE 11: TRAINING STABILITY ANALYSIS   ║"
echo "╚══════════════════════════════════════════╝"

echo "  Plotting stability curves for best models..."
# This uses the history.json generated by train_ssl.py
python -m src.experiments.plot_stability \
    --results_dir "$RESULTS_DIR" \
    --output "$RESULTS_DIR/stability_curves.png"

# ─── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ALL ABLATION PHASES COMPLETE"
echo "  Results saved to: $RESULTS_DIR/"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  Pretraining logs:"
for encoder in "${ENCODERS[@]}"; do
    for ssl_mode in "${SSL_MODES[@]}"; do
        echo "    - $RESULTS_DIR/ssl_${encoder}_${ssl_mode}/train.log"
    done
done
echo ""
echo "  Evaluation logs:"
for encoder in "${ENCODERS[@]}"; do
    for ssl_mode in "${SSL_MODES[@]}"; do
        echo "    - $RESULTS_DIR/ssl_${encoder}_${ssl_mode}/eval.log"
    done
done
