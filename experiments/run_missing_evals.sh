#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════════
# MISSING EVALUATIONS FOR PA-HybridSSL PAPER
# Run this in a tmux session: tmux new -s evals
#
# ⏱ TOTAL ESTIMATED TIME: ~3–4 GPU hours
#   Experiment A (Label Efficiency — all fractions): ~1.5h
#   Experiment B (Fine-tuning): ~1.5h
#   Experiment C (Per-class 5-class breakdown): ~0.5h
#   Experiment D (Supervised baseline): ~0.5h
#
# PREREQUISITES: Must be run from ~/projects/PA-SSL-ECG/
# ════════════════════════════════════════════════════════════════════════════════

set -e
cd ~/projects/PA-SSL-ECG

# ─── CHECKPOINT PATHS ──────────────────────────────────────────────────────────
PASSL=experiments/ablation/resnet1d_hybrid_s42/best_checkpoint.pth
SIMCLR=experiments/ablation/resnet1d_contrastive_s42/best_checkpoint.pth

# Verify checkpoints exist
echo "Checking checkpoints..."
[ -f "$PASSL" ]  && echo "  ✓ PA-SSL:  $PASSL" || echo "  ✗ MISSING: $PASSL"
[ -f "$SIMCLR" ] && echo "  ✓ SimCLR:  $SIMCLR" || echo "  ✗ MISSING: $SIMCLR"

OUTPUT_DIR=results/missing_evals
mkdir -p $OUTPUT_DIR


# ════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT A: Full Label Efficiency (1%, 5%, 10%, 25%, 50%, 100%)
# ⏱ ~1.5 GPU-hours
# Fills Table I with 5% and 25%/50%/100% fractions — and gives the label
# efficiency figure its full curve across 6 fractions for both models.
# ════════════════════════════════════════════════════════════════════════════════

echo ""
echo "════ EXPERIMENT A: Full Label Efficiency Curve ════"
echo "Testing PA-HybridSSL — all 6 fractions × 3 seeds = 18 evals..."

python -m src.evaluate \
  --checkpoint $PASSL \
  --data_file data/ptbxl_processed.csv \
  --label_fractions 0.01 0.05 0.1 0.25 0.5 1.0 \
  --n_seeds 3

echo "  → Results saved to: $(dirname $PASSL)/evaluation/label_efficiency.csv"
cp $(dirname $PASSL)/evaluation/label_efficiency.csv \
   $OUTPUT_DIR/label_efficiency_passl.csv

echo ""
echo "Testing SimCLR — all 6 fractions × 3 seeds..."

python -m src.evaluate \
  --checkpoint $SIMCLR \
  --data_file data/ptbxl_processed.csv \
  --label_fractions 0.01 0.05 0.1 0.25 0.5 1.0 \
  --n_seeds 3

cp $(dirname $SIMCLR)/evaluation/label_efficiency.csv \
   $OUTPUT_DIR/label_efficiency_simclr.csv

echo "✓ Experiment A done."


# ════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT B: Supervised Baseline (from-scratch, 1% / 10% / 100% labels)
# ⏱ ~0.5 GPU-hours
# Fills in the "Supervised (100%)" row in Table I which currently says "---"
# ════════════════════════════════════════════════════════════════════════════════

echo ""
echo "════ EXPERIMENT B: Supervised Upper Bound ════"
echo "Training ResNet1D from scratch at 1%, 10%, 100% labels..."

python -c "
import torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader, Subset
import torch.nn as nn, torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from src.data.ecg_dataset import ECGBeatDataset, patient_aware_split
from src.models.encoder import build_encoder
import os, json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_df, val_df, test_df = patient_aware_split('data/ptbxl_processed.csv')

results = []
for frac in [0.01, 0.05, 0.10, 1.0]:
    for seed in [42, 7, 123]:
        print(f'  Fraction={frac*100:.0f}%  seed={seed}')
        torch.manual_seed(seed); np.random.seed(seed)

        train_ds = ECGBeatDataset(train_df, label_fraction=frac, seed=seed)
        test_ds = ECGBeatDataset(test_df)

        # Build fresh encoder (no pretraining)
        enc = build_encoder('resnet1d', proj_dim=128).to(device)
        # Attach classifier head
        feat_dim = enc.encode(torch.zeros(1,1,250).to(device)).shape[-1]
        clf_head = nn.Linear(feat_dim, 2).to(device)
        model_params = list(enc.parameters()) + list(clf_head.parameters())

        optimizer = optim.AdamW(model_params, lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        labeled_mask = train_ds.label_mask
        labeled_idx = np.where(labeled_mask)[0]
        labeled_ds = Subset(train_ds, labeled_idx)
        loader = DataLoader(labeled_ds, batch_size=64, shuffle=True, num_workers=2, drop_last=True)

        # Train 50 epochs from scratch
        for epoch in range(50):
            enc.train(); clf_head.train()
            for batch in loader:
                sig, lab = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                feats = enc.encode(sig)
                logits = clf_head(feats)
                loss = criterion(logits, lab)
                loss.backward()
                optimizer.step()

        # Evaluate
        enc.eval(); clf_head.eval()
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                sig, lab = batch[0].to(device), batch[1]
                logits = clf_head(enc.encode(sig))
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs); all_labels.append(lab.numpy())
        all_probs = np.concatenate(all_probs); all_labels = np.concatenate(all_labels)
        preds = all_probs.argmax(1)

        auroc = roc_auc_score(all_labels, all_probs[:,1])
        auprc = average_precision_score(all_labels, all_probs[:,1])
        acc = accuracy_score(all_labels, preds)
        f1 = f1_score(all_labels, preds, average='macro')
        print(f'    AUROC={auroc:.4f}  AUPRC={auprc:.4f}  Acc={acc:.4f}')
        results.append({'label_fraction': frac, 'seed': seed,
                        'auroc': auroc, 'auprc': auprc, 'accuracy': acc, 'f1_macro': f1})

df = pd.DataFrame(results)
df.to_csv('results/missing_evals/supervised_baseline.csv', index=False)
summary = df.groupby('label_fraction')[['auroc','auprc','accuracy','f1_macro']].agg(['mean','std']).round(4)
print()
print('=== SUPERVISED BASELINE SUMMARY ===')
print(summary.to_string())
"

echo "✓ Experiment B done → results/missing_evals/supervised_baseline.csv"


# ════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT C: Per-Class 5-Class PTB-XL AUROC Breakdown
# ⏱ ~0.5 GPU-hours
# Extracts NORM/MI/STTC/CD/HYP per-class AUROC for Table V
# ════════════════════════════════════════════════════════════════════════════════

echo ""
echo "════ EXPERIMENT C: Per-Class PTB-XL 5-Class AUROC ════"

python -c "
import torch, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from src.data.ecg_dataset import ECGBeatDataset, patient_aware_split
from src.models.encoder import build_encoder
import warnings; warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class mapping
CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

def get_multiclass_data(csv_path, label_col='superclass_label'):
    \"\"\"Load 5-class PTB-XL data.\"\"\"
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        # Try to find the right column
        for col in df.columns:
            if 'super' in col.lower() or 'class' in col.lower():
                print(f'Using column: {col}')
                label_col = col
                break
    return df, label_col

checkpoints = {
    'PA-HybridSSL (ResNet1D)': 'experiments/ablation/resnet1d_hybrid_s42/best_checkpoint.pth',
    'SimCLR + Naive Aug':      'experiments/ablation/resnet1d_contrastive_s42/best_checkpoint.pth',
}

all_results = []
for model_name, ckpt_path in checkpoints.items():
    print(f'Model: {model_name}')
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get('config', {})
    enc_name = config.get('encoder', 'resnet1d')
    enc = build_encoder(enc_name, proj_dim=config.get('proj_dim', 128))
    sd = {k.replace('_orig_mod.',''): v for k,v in ckpt['encoder_state_dict'].items()}
    enc.load_state_dict(sd, strict=False)
    enc = enc.to(device).eval()

    train_df, val_df, test_df = patient_aware_split('data/ptbxl_processed.csv')

    for seed in [42, 7, 123]:
        # Extract representations
        @torch.no_grad()
        def extract(df_in):
            from torch.utils.data import DataLoader
            ds = ECGBeatDataset(df_in)
            loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)
            reprs, labels = [], []
            for batch in loader:
                sig = batch[0].to(device)
                reprs.append(enc.encode(sig).cpu().numpy())
                labels.append(batch[1].numpy())
            return np.concatenate(reprs), np.concatenate(labels)

        tr_r, tr_l = extract(train_df)
        te_r, te_l = extract(test_df)

        # Sanitize NaNs
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(strategy='mean')
        tr_r = imp.fit_transform(tr_r); te_r = imp.transform(te_r)

        # Check if multiclass labels exist (values 0-4)
        unique_labels = np.unique(tr_l)
        if len(unique_labels) <= 2:
            print(f'  Only binary labels found ({unique_labels}). Need superclass labels.')
            # Try loading from multiclass CSV
            mc_csv = 'data/ptbxl_multiclass.csv'
            import os
            if os.path.exists(mc_csv):
                train_df2, val_df2, test_df2 = patient_aware_split(mc_csv)
                tr_r2, tr_l2 = extract(train_df2)
                te_r2, te_l2 = extract(test_df2)
                tr_r, tr_l = tr_r2, tr_l2
                te_r, te_l = te_r2, te_l2
            else:
                print('  Multiclass CSV not found — skipping per-class breakdown.')
                continue

        clf = LogisticRegression(max_iter=1000, C=1.0, solver='saga', multi_class='ovr')
        clf.fit(tr_r, tr_l)
        probs = clf.predict_proba(te_r)
        classes = clf.classes_

        row = {'model': model_name, 'seed': seed}
        for i, cls in enumerate(classes):
            cls_name = CLASS_NAMES[int(cls)] if int(cls) < len(CLASS_NAMES) else f'cls_{cls}'
            # OvR AUROC for this class
            bin_labels = (te_l == cls).astype(int)
            if bin_labels.sum() > 0 and bin_labels.sum() < len(bin_labels):
                auroc = roc_auc_score(bin_labels, probs[:, i])
            else:
                auroc = float('nan')
            row[f'auroc_{cls_name}'] = auroc
            print(f'    {cls_name}: AUROC={auroc:.4f}')
        all_results.append(row)

if all_results:
    df = pd.DataFrame(all_results)
    df.to_csv('results/missing_evals/perclass_5class_auroc.csv', index=False)
    print()
    print('=== PER-CLASS SUMMARY ===')
    summary = df.groupby('model').mean(numeric_only=True).round(4)
    print(summary.to_string())
else:
    print('No multiclass results generated.')
"

echo "✓ Experiment C done → results/missing_evals/perclass_5class_auroc.csv"


# ════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT D: PA-HybridSSL Multi-seed Label Efficiency (seeds 42, 7, 123)
# ⏱ This is already covered by Experiment A (n_seeds=3)
# Just generate a clean summary CSV for easy copy-paste into Table I
# ════════════════════════════════════════════════════════════════════════════════

echo ""
echo "════ GENERATING PAPER TABLE SUMMARIES ════"

python -c "
import pandas as pd, numpy as np

# Combine PA-SSL and SimCLR label efficiency results
pa   = pd.read_csv('results/missing_evals/label_efficiency_passl.csv')
sim  = pd.read_csv('results/missing_evals/label_efficiency_simclr.csv')
pa['model']  = 'PA-HybridSSL (ResNet1D)'
sim['model'] = 'SimCLR + Naive Aug'
combined = pd.concat([pa, sim], ignore_index=True)

# Table I summary: 1%, 5%, 10%, 100%
table_fracs = [0.01, 0.05, 0.10, 1.0]
rows = []
for model in ['PA-HybridSSL (ResNet1D)', 'SimCLR + Naive Aug']:
    mdf = combined[combined['model']==model]
    for frac in table_fracs:
        fdf = mdf[mdf['label_fraction'].round(4) == round(frac, 4)]
        if len(fdf) == 0:
            continue
        rows.append({
            'model': model,
            'label_pct': f'{frac*100:.0f}%',
            'Acc (mean)':   fdf['linear_accuracy'].mean(),
            'Acc (std)':    fdf['linear_accuracy'].std(),
            'F1 (mean)':    fdf['linear_f1_macro'].mean(),
            'AUROC (mean)': fdf['linear_auroc'].mean(),
            'AUROC (std)':  fdf['linear_auroc'].std(),
            'AUPRC (mean)': fdf['linear_auprc'].mean(),
        })
tbl = pd.DataFrame(rows).round(4)
tbl.to_csv('results/missing_evals/TABLE_I_complete.csv', index=False)
print('=== TABLE I (Label Efficiency) ===')
print(tbl.to_string(index=False))

# Supervised baseline summary
try:
    sup = pd.read_csv('results/missing_evals/supervised_baseline.csv')
    sup_summary = sup.groupby('label_fraction')[['auroc','auprc','accuracy','f1_macro']].agg(['mean','std']).round(4)
    print()
    print('=== SUPERVISED BASELINE ===')
    print(sup_summary.to_string())
except:
    print('Supervised baseline not yet available.')
"

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "ALL EXPERIMENTS COMPLETE"
echo "Results in: results/missing_evals/"
echo "  - label_efficiency_passl.csv      (raw PA-SSL results)"
echo "  - label_efficiency_simclr.csv     (raw SimCLR results)"
echo "  - supervised_baseline.csv         (from-scratch training)"
echo "  - perclass_5class_auroc.csv       (NORM/MI/STTC/CD/HYP breakdown)"
echo "  - TABLE_I_complete.csv            (ready to copy into paper)"
echo "════════════════════════════════════════════════════════════════════════════════"
