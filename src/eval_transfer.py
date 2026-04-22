"""
PA-SSL: Cross-Dataset Transfer Evaluation  (Table 2)
======================================================
Loads a checkpoint pretrained on PTB-XL, then fits a fresh linear probe
on each target dataset (MIT-BIH, Chapman) using 80/20 patient-aware splits.
Reports AUROC and AUPRC for every source→target pair.

Usage (run from project root on server):
    python -m src.eval_transfer \
        --checkpoint experiments/ablation/resnet1d_hybrid_s42/best_checkpoint.pth \
        --encoder   resnet1d \
        --source    data/ptbxl_processed.csv \
        --targets   data/mitbih_processed.csv data/chapman_processed.csv \
        --output    results/transfer_results.csv \
        2>&1 | tee logs/eval_transfer.log
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from src.models.encoder import build_encoder
from src.data.ecg_dataset import ECGBeatDataset


# ─── helpers ──────────────────────────────────────────────────────────────────

def load_encoder(checkpoint_path, encoder_name, device):
    encoder = build_encoder(encoder_name, proj_dim=128).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("encoder_state_dict", ckpt)
    encoder.load_state_dict(state)
    encoder.eval()
    return encoder


@torch.no_grad()
def extract(encoder, dataset, device, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    reprs, labels = [], []
    for batch in tqdm(loader, desc="  extracting", leave=False):
        x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch["signal"].to(device)
        y = batch[1]            if isinstance(batch, (list, tuple)) else batch["label"]
        r = encoder.encode(x)
        reprs.append(r.cpu().numpy())
        labels.append(y.numpy())
    reprs  = np.concatenate(reprs)
    labels = np.concatenate(labels)
    # drop unknown labels
    valid  = np.isin(labels, [0, 1])
    return reprs[valid], labels[valid]


def patient_split(dataset, train_frac=0.8, seed=42):
    """Simple per-dataset 80/20 split by dataset index (no patient IDs needed)."""
    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(len(dataset))
    cut  = int(len(idx) * train_frac)
    return Subset(dataset, idx[:cut]), Subset(dataset, idx[cut:])


def eval_linear(train_r, train_y, test_r, test_y):
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                             multi_class="auto", n_jobs=-1)
    clf.fit(train_r, train_y)
    proba = clf.predict_proba(test_r)
    if proba.shape[1] == 2:
        proba = proba[:, 1]
    auroc = roc_auc_score(test_y, proba, multi_class="ovr")
    auprc = average_precision_score(test_y, proba)
    return auroc, auprc


# ─── main ─────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    encoder = load_encoder(args.checkpoint, args.encoder, device)
    print(f"Loaded encoder from {args.checkpoint}")

    rows = []

    # ── in-distribution: train on source train split, eval on source test split
    print(f"\nIn-distribution: {args.source}")
    src_ds = ECGBeatDataset(args.source)
    src_train, src_test = patient_split(src_ds, seed=42)
    src_tr_r, src_tr_y  = extract(encoder, src_train, device)
    src_te_r, src_te_y  = extract(encoder, src_test,  device)
    auroc, auprc = eval_linear(src_tr_r, src_tr_y, src_te_r, src_te_y)
    print(f"  PTB-XL (in-distribution): AUROC={auroc:.4f}  AUPRC={auprc:.4f}")
    rows.append({"dataset": "PTB-XL (in-distribution)", "auroc": auroc, "auprc": auprc})

    # ── cross-dataset: train linear probe *on target*, test on target test split
    for target_path in args.targets:
        ds_name = os.path.basename(target_path).replace("_processed.csv", "").upper()
        print(f"\nCross-dataset: {ds_name}  ({target_path})")
        tgt_ds            = ECGBeatDataset(target_path)
        tgt_train, tgt_test = patient_split(tgt_ds, seed=42)
        tgt_tr_r, tgt_tr_y  = extract(encoder, tgt_train, device)
        tgt_te_r, tgt_te_y  = extract(encoder, tgt_test,  device)
        auroc, auprc = eval_linear(tgt_tr_r, tgt_tr_y, tgt_te_r, tgt_te_y)
        print(f"  {ds_name}: AUROC={auroc:.4f}  AUPRC={auprc:.4f}")
        rows.append({"dataset": f"{ds_name} (linear probe transfer)",
                     "auroc": auroc, "auprc": auprc})

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"\nSaved → {args.output}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder",    default="resnet1d")
    parser.add_argument("--source",     default="data/ptbxl_processed.csv")
    parser.add_argument("--targets",    nargs="+",
                        default=["data/mitbih_processed.csv",
                                 "data/chapman_processed.csv"])
    parser.add_argument("--output",     default="results/transfer_results.csv")
    main(parser.parse_args())
