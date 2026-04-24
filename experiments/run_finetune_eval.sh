#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════════
# FINE-TUNING EVALUATION FOR PA-HybridSSL PAPER
# Run: tmux new -s finetune  →  bash experiments/run_finetune_eval.sh
#
# ⏱ TOTAL: ~2 GPU hours
#   End-to-end fine-tuning for PA-SSL and SimCLR at 1%, 10%, 100% labels
#   Adds fine-tuning rows to Table I — required for IEEE TBME/JBHI standard
#
# USAGE: Must run from ~/projects/PA-SSL-ECG/
# ════════════════════════════════════════════════════════════════════════════════

set -e
cd ~/projects/PA-SSL-ECG

PYTHON=python3
mkdir -p results/missing_evals logs

# Checkpoints
PASSL=experiments/ablation/resnet1d_hybrid_s42/best_checkpoint.pth
SIMCLR=experiments/ablation/resnet1d_contrastive_s42/best_checkpoint.pth

echo "Checking checkpoints..."
[ -f "$PASSL" ]  && echo "  OK PA-SSL"  || { echo "  MISSING: $PASSL";  exit 1; }
[ -f "$SIMCLR" ] && echo "  OK SimCLR" || { echo "  MISSING: $SIMCLR"; exit 1; }


# ════════════════════════════════════════════════════════════════════════════════
# FINE-TUNING: End-to-end (encoder + head, full backprop)
# ⏱ ~2h  |  Critical missing experiment for IEEE review
# ════════════════════════════════════════════════════════════════════════════════

echo ""
echo "==== FINE-TUNING EVALUATION (end-to-end) ===="

$PYTHON - <<'PYEOF'
import torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader, Subset
import torch.nn as nn, torch.optim as optim
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, f1_score)
from src.data.ecg_dataset import ECGBeatDataset, patient_aware_split
from src.models.encoder import build_encoder
import os, warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

train_df, val_df, test_df = patient_aware_split('data/ptbxl_processed.csv')

CHECKPOINTS = {
    'PA-HybridSSL (ResNet1D)': 'experiments/ablation/resnet1d_hybrid_s42/best_checkpoint.pth',
    'SimCLR + Naive Aug':      'experiments/ablation/resnet1d_contrastive_s42/best_checkpoint.pth',
}

def strip_mod(sd):
    return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}

results = []

for model_name, ckpt_path in CHECKPOINTS.items():
    print(f'\n=== {model_name} ===')
    for frac in [0.01, 0.05, 0.10, 1.0]:
        for seed in [42, 7, 123]:
            print(f'  frac={frac*100:.0f}%  seed={seed}')
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Load pretrained checkpoint
            ckpt   = torch.load(ckpt_path, map_location=device)
            cfg    = ckpt.get('config', {})
            enc    = build_encoder(cfg.get('encoder', 'resnet1d'),
                                   proj_dim=cfg.get('proj_dim', 128))
            enc.load_state_dict(strip_mod(ckpt['encoder_state_dict']), strict=False)
            enc    = enc.to(device)

            # Build classifier head
            with torch.no_grad():
                dummy    = torch.zeros(1, 1, 250).to(device)
                feat_dim = enc.encode(dummy).shape[-1]
            clf_head = nn.Linear(feat_dim, 2).to(device)

            # Use cosine annealing + small LR to avoid catastrophic forgetting
            optimizer = optim.AdamW(
                list(enc.parameters()) + list(clf_head.parameters()),
                lr=1e-4, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=30, eta_min=1e-6)
            criterion = nn.CrossEntropyLoss()

            train_ds    = ECGBeatDataset(train_df, label_fraction=frac, seed=seed)
            test_ds     = ECGBeatDataset(test_df)
            labeled_idx = np.where(train_ds.label_mask)[0]
            labeled_ds  = Subset(train_ds, labeled_idx)

            if len(labeled_ds) == 0:
                print('  No labeled samples — skip')
                continue

            bs     = min(64, len(labeled_ds))
            loader = DataLoader(labeled_ds, batch_size=bs,
                                shuffle=True, num_workers=2, drop_last=False)

            # Fine-tune 30 epochs (enough to adapt head + fine-tune encoder)
            for epoch in range(30):
                enc.train(); clf_head.train()
                for batch in loader:
                    sig, lab = batch[0].to(device), batch[1].to(device)
                    optimizer.zero_grad()
                    criterion(clf_head(enc.encode(sig)), lab).backward()
                    optimizer.step()
                scheduler.step()

            # Evaluate
            enc.eval(); clf_head.eval()
            test_loader = DataLoader(test_ds, batch_size=256,
                                     shuffle=False, num_workers=2)
            probs_list, labels_list = [], []
            with torch.no_grad():
                for batch in test_loader:
                    p = torch.softmax(clf_head(enc.encode(batch[0].to(device))), 1)
                    probs_list.append(p.cpu().numpy())
                    labels_list.append(batch[1].numpy())

            all_probs  = np.concatenate(probs_list)
            all_labels = np.concatenate(labels_list)

            auroc = roc_auc_score(all_labels, all_probs[:, 1])
            auprc = average_precision_score(all_labels, all_probs[:, 1])
            acc   = accuracy_score(all_labels, all_probs.argmax(1))
            f1    = f1_score(all_labels, all_probs.argmax(1), average='macro')
            print(f'    AUROC={auroc:.4f}  AUPRC={auprc:.4f}  Acc={acc:.4f}  F1={f1:.4f}')

            results.append({
                'model': model_name,
                'label_fraction': frac,
                'seed': seed,
                'auroc': auroc,
                'auprc': auprc,
                'accuracy': acc,
                'f1_macro': f1,
            })

df = pd.DataFrame(results)
df.to_csv('results/missing_evals/finetune_results.csv', index=False)

print()
print('='*60)
print('FINE-TUNING SUMMARY (mean +/- std over 3 seeds)')
print('='*60)
for model in df['model'].unique():
    print(f'\n{model}:')
    mdf = df[df['model'] == model]
    for frac in [0.01, 0.05, 0.10, 1.0]:
        fdf = mdf[(mdf['label_fraction'] - frac).abs() < 0.001]
        if len(fdf) == 0:
            continue
        print(f'  {frac*100:5.0f}% labels: '
              f'Acc={fdf["accuracy"].mean():.4f}+/-{fdf["accuracy"].std():.4f}  '
              f'F1={fdf["f1_macro"].mean():.4f}+/-{fdf["f1_macro"].std():.4f}  '
              f'AUROC={fdf["auroc"].mean():.4f}+/-{fdf["auroc"].std():.4f}  '
              f'AUPRC={fdf["auprc"].mean():.4f}+/-{fdf["auprc"].std():.4f}')
PYEOF

echo ""
echo "Done: results/missing_evals/finetune_results.csv"
echo ""
echo "Copy back:"
echo "  scp server:~/projects/PA-SSL-ECG/results/missing_evals/finetune_results.csv ."
