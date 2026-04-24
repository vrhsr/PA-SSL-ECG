#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════════
# MISSING EVALUATIONS FOR PA-HybridSSL PAPER
# Run: tmux new -s evals  →  bash experiments/run_missing_evals.sh
#
# ⏱ TOTAL: ~3-4 GPU hours
#   Exp A - Full label efficiency (all fractions): ~1.5h
#   Exp B - Supervised baseline (from-scratch):   ~1h
#   Exp C - Per-class 5-class AUROC:              ~0.5h
#
# USAGE: Must run from ~/projects/PA-SSL-ECG/
# ════════════════════════════════════════════════════════════════════════════════

set -e
cd ~/projects/PA-SSL-ECG

PYTHON=python3

# ─── CHECKPOINT PATHS ──────────────────────────────────────────────────────────
PASSL=experiments/ablation/resnet1d_hybrid_s42/best_checkpoint.pth
SIMCLR=experiments/ablation/resnet1d_contrastive_s42/best_checkpoint.pth

echo "Checking checkpoints..."
[ -f "$PASSL" ]  && echo "  OK PA-SSL:  $PASSL"  || { echo "  MISSING: $PASSL";  exit 1; }
[ -f "$SIMCLR" ] && echo "  OK SimCLR:  $SIMCLR" || { echo "  MISSING: $SIMCLR"; exit 1; }

mkdir -p results/missing_evals


# ════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT A: Full Label Efficiency (1%, 5%, 10%, 25%, 50%, 100%)
# ⏱ ~1.5h  | Adds the 5%/25%/50%/100% rows missing from Table I
# ════════════════════════════════════════════════════════════════════════════════

echo ""
echo "==== EXPERIMENT A: Full Label Efficiency Curve ===="

echo "  PA-HybridSSL (all 6 fractions x 3 seeds)..."
$PYTHON -m src.evaluate \
  --checkpoint $PASSL \
  --data_file data/ptbxl_processed.csv \
  --label_fractions 0.01 0.05 0.1 0.25 0.5 1.0 \
  --n_seeds 3 \
  2>&1 | tee logs/eval_label_eff_passl.log

cp $(dirname $PASSL)/evaluation/label_efficiency.csv \
   results/missing_evals/label_efficiency_passl.csv
echo "  Saved: results/missing_evals/label_efficiency_passl.csv"

echo "  SimCLR (all 6 fractions x 3 seeds)..."
$PYTHON -m src.evaluate \
  --checkpoint $SIMCLR \
  --data_file data/ptbxl_processed.csv \
  --label_fractions 0.01 0.05 0.1 0.25 0.5 1.0 \
  --n_seeds 3 \
  2>&1 | tee logs/eval_label_eff_simclr.log

cp $(dirname $SIMCLR)/evaluation/label_efficiency.csv \
   results/missing_evals/label_efficiency_simclr.csv
echo "  Saved: results/missing_evals/label_efficiency_simclr.csv"

echo "DONE: Experiment A"


# ════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT B: Supervised Baseline (from scratch, no pretraining)
# ⏱ ~1h  | Fills "--- upper bound ---" in Table I with real numbers
# ════════════════════════════════════════════════════════════════════════════════

echo ""
echo "==== EXPERIMENT B: Supervised Baseline (from scratch) ===="

$PYTHON - <<'PYEOF'
import torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader, Subset
import torch.nn as nn, torch.optim as optim
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, f1_score)
from src.data.ecg_dataset import ECGBeatDataset, patient_aware_split
from src.models.encoder import build_encoder
import warnings; warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

train_df, val_df, test_df = patient_aware_split('data/ptbxl_processed.csv')

results = []
for frac in [0.01, 0.05, 0.10, 1.0]:
    for seed in [42, 7, 123]:
        print(f'\n  frac={frac*100:.0f}%  seed={seed}')
        torch.manual_seed(seed); np.random.seed(seed)

        train_ds = ECGBeatDataset(train_df, label_fraction=frac, seed=seed)
        test_ds  = ECGBeatDataset(test_df)

        # Fresh encoder — no pretrained weights
        enc      = build_encoder('resnet1d', proj_dim=128).to(device)
        with torch.no_grad():
            feat_dim = enc.encode(torch.zeros(1,1,250).to(device)).shape[-1]
        clf_head = nn.Linear(feat_dim, 2).to(device)

        optimizer = optim.AdamW(
            list(enc.parameters()) + list(clf_head.parameters()),
            lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        labeled_mask = train_ds.label_mask
        labeled_idx  = np.where(labeled_mask)[0]
        labeled_ds   = Subset(train_ds, labeled_idx)

        if len(labeled_ds) == 0:
            print('  No labeled samples, skipping')
            continue

        bs = min(64, len(labeled_ds))
        loader = DataLoader(labeled_ds, batch_size=bs,
                            shuffle=True, num_workers=2, drop_last=False)

        for epoch in range(50):
            enc.train(); clf_head.train()
            for batch in loader:
                sig, lab = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                criterion(clf_head(enc.encode(sig)), lab).backward()
                optimizer.step()

        enc.eval(); clf_head.eval()
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)
        probs_list, labels_list = [], []
        with torch.no_grad():
            for batch in test_loader:
                p = torch.softmax(clf_head(enc.encode(batch[0].to(device))), dim=1)
                probs_list.append(p.cpu().numpy())
                labels_list.append(batch[1].numpy())

        all_probs  = np.concatenate(probs_list)
        all_labels = np.concatenate(labels_list)

        auroc = roc_auc_score(all_labels, all_probs[:,1])
        auprc = average_precision_score(all_labels, all_probs[:,1])
        acc   = accuracy_score(all_labels, all_probs.argmax(1))
        f1    = f1_score(all_labels, all_probs.argmax(1), average='macro')
        print(f'  AUROC={auroc:.4f}  AUPRC={auprc:.4f}  Acc={acc:.4f}  F1={f1:.4f}')

        results.append({'label_fraction': frac, 'seed': seed,
                        'auroc': auroc, 'auprc': auprc,
                        'accuracy': acc, 'f1_macro': f1})

df = pd.DataFrame(results)
df.to_csv('results/missing_evals/supervised_baseline.csv', index=False)

print()
print('=== SUPERVISED BASELINE SUMMARY ===')
summary = df.groupby('label_fraction')[
    ['auroc','auprc','accuracy','f1_macro']
].agg(['mean','std']).round(4)
print(summary.to_string())
PYEOF

echo "DONE: Experiment B -> results/missing_evals/supervised_baseline.csv"


# ════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT C: Per-Class 5-Class AUROC (NORM / MI / STTC / CD / HYP)
# ⏱ ~0.5h  | Adds class breakdown to Table V discussion
# ════════════════════════════════════════════════════════════════════════════════

echo ""
echo "==== EXPERIMENT C: Per-Class 5-Class PTB-XL AUROC ===="

$PYTHON - <<'PYEOF'
import torch, numpy as np, pandas as pd, os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from torch.utils.data import DataLoader
from src.data.ecg_dataset import ECGBeatDataset, patient_aware_split
from src.models.encoder import build_encoder
import warnings; warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

def extract(enc, df_in, device):
    ds     = ECGBeatDataset(df_in)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)
    reprs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            reprs.append(enc.encode(batch[0].to(device)).cpu().numpy())
            labels.append(batch[1].numpy())
    return np.concatenate(reprs), np.concatenate(labels)

checkpoints = {
    'PA-HybridSSL': 'experiments/ablation/resnet1d_hybrid_s42/best_checkpoint.pth',
    'SimCLR':       'experiments/ablation/resnet1d_contrastive_s42/best_checkpoint.pth',
}

# Use multiclass CSV if available, else fall back to binary
use_csv = 'data/ptbxl_multiclass.csv' if os.path.exists('data/ptbxl_multiclass.csv') \
          else 'data/ptbxl_processed.csv'
print(f'Data: {use_csv}')
train_df, val_df, test_df = patient_aware_split(use_csv)

all_results = []
for model_name, ckpt_path in checkpoints.items():
    print(f'\nModel: {model_name}')
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt.get('config', {})
    enc  = build_encoder(cfg.get('encoder','resnet1d'), proj_dim=cfg.get('proj_dim',128))
    sd   = {k.replace('_orig_mod.',''):v for k,v in ckpt['encoder_state_dict'].items()}
    enc.load_state_dict(sd, strict=False)
    enc  = enc.to(device).eval()

    for seed in [42, 7, 123]:
        np.random.seed(seed); torch.manual_seed(seed)
        tr_r, tr_l = extract(enc, train_df, device)
        te_r, te_l = extract(enc, test_df, device)

        imp  = SimpleImputer(strategy='mean')
        tr_r = imp.fit_transform(tr_r); te_r = imp.transform(te_r)

        n_classes = len(np.unique(tr_l))
        print(f'  seed={seed}, n_classes={n_classes}, labels={np.unique(tr_l)}')

        if n_classes <= 2:
            print('  Only binary labels found — skipping per-class (need ptbxl_multiclass.csv)')
            continue

        clf = LogisticRegression(max_iter=1000, C=1.0, solver='saga', multi_class='ovr')
        clf.fit(tr_r, tr_l)
        probs = clf.predict_proba(te_r)

        row = {'model': model_name, 'seed': seed}
        for i, cls in enumerate(clf.classes_):
            name  = CLASS_NAMES[int(cls)] if int(cls) < len(CLASS_NAMES) else f'cls{cls}'
            bl    = (te_l == cls).astype(int)
            auroc = roc_auc_score(bl, probs[:,i]) if 0 < bl.sum() < len(bl) else float('nan')
            row[f'auroc_{name}'] = auroc
            print(f'    {name}: {auroc:.4f}')
        all_results.append(row)

if all_results:
    df = pd.DataFrame(all_results)
    df.to_csv('results/missing_evals/perclass_5class_auroc.csv', index=False)
    print()
    print('=== MEAN PER-CLASS AUROC ===')
    print(df.groupby('model').mean(numeric_only=True).round(4).to_string())
else:
    print('No results — multiclass labels not found in CSV.')
PYEOF

echo "DONE: Experiment C -> results/missing_evals/perclass_5class_auroc.csv"


# ════════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY — print TABLE_I_complete.csv
# ════════════════════════════════════════════════════════════════════════════════

echo ""
echo "==== FINAL TABLE SUMMARY ===="

$PYTHON - <<'PYEOF'
import pandas as pd, numpy as np, os

print('=== TABLE I: Label Efficiency ===')
pa_p  = 'results/missing_evals/label_efficiency_passl.csv'
sim_p = 'results/missing_evals/label_efficiency_simclr.csv'

if os.path.exists(pa_p) and os.path.exists(sim_p):
    pa  = pd.read_csv(pa_p);  pa['model']  = 'PA-HybridSSL'
    sim = pd.read_csv(sim_p); sim['model'] = 'SimCLR'
    combined = pd.concat([pa, sim], ignore_index=True)

    rows = []
    for model in ['PA-HybridSSL', 'SimCLR']:
        mdf = combined[combined['model'] == model]
        for frac in [0.01, 0.05, 0.10, 1.0]:
            fdf = mdf[(mdf['label_fraction'] - frac).abs() < 0.001]
            if len(fdf) == 0:
                continue
            rows.append({
                'Model':  model,
                'Labels': f'{frac*100:.0f}%',
                'Acc':    f"{fdf['linear_accuracy'].mean():.4f}+/-{fdf['linear_accuracy'].std():.4f}",
                'F1':     f"{fdf['linear_f1_macro'].mean():.4f}+/-{fdf['linear_f1_macro'].std():.4f}",
                'AUROC':  f"{fdf['linear_auroc'].mean():.4f}+/-{fdf['linear_auroc'].std():.4f}",
                'AUPRC':  f"{fdf['linear_auprc'].mean():.4f}+/-{fdf['linear_auprc'].std():.4f}",
            })
    tbl = pd.DataFrame(rows)
    tbl.to_csv('results/missing_evals/TABLE_I_complete.csv', index=False)
    print(tbl.to_string(index=False))
else:
    print('Label efficiency CSV not found.')

print()
print('=== SUPERVISED BASELINE ===')
sup_p = 'results/missing_evals/supervised_baseline.csv'
if os.path.exists(sup_p):
    sup = pd.read_csv(sup_p)
    print(sup.groupby('label_fraction')[
        ['auroc','auprc','accuracy','f1_macro']
    ].agg(['mean','std']).round(4).to_string())
else:
    print('Not available.')
PYEOF

echo ""
echo "================================================================"
echo "  ALL DONE — copy results to your local machine:"
echo "    scp -r server:~/projects/PA-SSL-ECG/results/missing_evals/ ."
echo "================================================================"
