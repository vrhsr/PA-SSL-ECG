"""
PA-SSL: 5-Class PTB-XL Evaluation Script
==========================================
Evaluates all saved PA-SSL checkpoints on the PTB-XL 5-class
diagnostic superclass task (NORM, MI, STTC, CD, HYP) using
linear probing — no retraining required.

This is CRITICAL for top-tier journal publication.
Reviewers expect multi-class evaluation on PTB-XL.

Usage (on remote server):
    python3 -m src.eval_multiclass \
        --ptbxl_dir /path/to/ptb-xl \
        --ptbxl_csv data/ptbxl_processed.csv \
        --checkpoints_dir experiments \
        --output_dir experiments/multiclass_results
"""

import os
import sys
import ast
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, classification_report
)
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.encoder import build_encoder

# ─── 5-CLASS LABEL MAPPING ────────────────────────────────────────────────────

CLASS_NAMES  = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


def build_5class_labels(ptbxl_csv_path, ptbxl_dir):
    """
    Merge 5-class superclass labels from PTB-XL metadata into
    the beat-level CSV. Returns a DataFrame with an added
    'label_5class' column.
    
    Steps:
    1. Load ptbxl_database.csv (has ecg_id, scp_codes, patient_id)
    2. Load scp_statements.csv (maps SCP codes → diagnostic class)
    3. Map each ECG's scp_codes → one of NORM/MI/STTC/CD/HYP
    4. Merge onto beat CSV via record_id = ecg_id
    """
    db_path = os.path.join(ptbxl_dir, 'ptbxl_database.csv')
    scp_path = os.path.join(ptbxl_dir, 'scp_statements.csv')

    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"PTB-XL database CSV not found: {db_path}\n"
            f"Please set --ptbxl_dir to the folder containing ptbxl_database.csv"
        )

    print(f"Loading PTB-XL metadata from: {ptbxl_dir}")
    Y = pd.read_csv(db_path, index_col='ecg_id')
    Y['scp_codes'] = Y['scp_codes'].apply(ast.literal_eval)

    agg_df = pd.read_csv(scp_path, index_col=0)
    agg_df = agg_df[agg_df['diagnostic'] == 1]

    def get_superclass(scp_codes):
        results = []
        for key in scp_codes.keys():
            if key in agg_df.index:
                cls = agg_df.loc[key, 'diagnostic_class']
                if pd.notna(cls):
                    results.append(cls)
        if 'NORM' in results:
            return 'NORM'
        for cls in ['MI', 'STTC', 'CD', 'HYP']:
            if cls in results:
                return cls
        return None  # Ambiguous → exclude

    Y['label_5class'] = Y['scp_codes'].apply(get_superclass)
    ecg_label_map = Y['label_5class'].dropna().to_dict()  # ecg_id → class string

    print(f"  Found {len(ecg_label_map):,} ECGs with unambiguous 5-class labels")
    for cls in CLASS_NAMES:
        n = sum(1 for v in ecg_label_map.values() if v == cls)
        print(f"    {cls}: {n:,}")

    # Load beat CSV and merge
    print(f"\nLoading beat CSV: {ptbxl_csv_path}")
    df = pd.read_csv(ptbxl_csv_path)
    df['label_5class'] = df['record_id'].map(ecg_label_map)
    df = df.dropna(subset=['label_5class'])
    df['label_5class_idx'] = df['label_5class'].map(CLASS_TO_IDX)

    print(f"  Beats with 5-class labels: {len(df):,}")
    for cls in CLASS_NAMES:
        n = (df['label_5class'] == cls).sum()
        print(f"    {cls}: {n:,} beats ({100*n/len(df):.1f}%)")

    return df


def extract_features(encoder, df, device, batch_size=512):
    """Extract representations from all beats using frozen encoder."""
    signal_cols = [c for c in df.columns if str(c).isdigit()]
    X_raw = df[signal_cols].values.astype(np.float32)

    encoder.eval()
    reprs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(X_raw), batch_size), desc="  Extracting features"):
            batch = torch.from_numpy(X_raw[i:i+batch_size]).unsqueeze(1).to(device)
            z = encoder(batch, return_projection=False)
            reprs.append(z.cpu().numpy())

    return np.concatenate(reprs, axis=0)


def run_linear_probe_multiclass(reprs, labels, patient_ids, seed=42):
    """
    Patient-aware train/test split + multinomial logistic regression.
    Returns dict of metrics.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(gss.split(reprs, groups=patient_ids))

    X_train, y_train = reprs[train_idx], labels[train_idx]
    X_test,  y_test  = reprs[test_idx],  labels[test_idx]

    clf = LogisticRegression(
        max_iter=1000, C=1.0, multi_class='multinomial',
        solver='lbfgs', random_state=seed, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    acc   = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # One-vs-Rest AUROC (macro)
    try:
        auroc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
    except Exception:
        auroc = float('nan')

    # Per-class report
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0)

    return {
        'accuracy':     acc,
        'f1_macro':     f1_macro,
        'f1_weighted':  f1_weighted,
        'auroc_macro':  auroc,
        'n_train':      len(train_idx),
        'n_test':       len(test_idx),
        'report':       report,
    }


def evaluate_checkpoint(ckpt_path, encoder_type, df_5class, device, label):
    """Load a checkpoint, extract representations, run 5-class linear probe."""
    print(f"\n{'='*60}")
    print(f"Model: {label}")
    print(f"Checkpoint: {ckpt_path}")

    encoder = build_encoder(encoder_type, proj_dim=128)
    state = torch.load(ckpt_path, map_location=device)

    # Handle different checkpoint formats
    if 'encoder_state_dict' in state:
        encoder.load_state_dict(state['encoder_state_dict'])
    elif 'model_state_dict' in state:
        encoder.load_state_dict(state['model_state_dict'])
    else:
        encoder.load_state_dict(state)

    encoder = encoder.to(device)

    labels_5class = df_5class['label_5class_idx'].values
    patient_ids   = df_5class['patient_id'].values

    reprs = extract_features(encoder, df_5class, device)

    results = []
    for seed in [42, 43, 44]:
        m = run_linear_probe_multiclass(reprs, labels_5class, patient_ids, seed=seed)
        results.append(m)
        print(f"  Seed {seed}: Acc={m['accuracy']:.4f}, F1={m['f1_macro']:.4f}, AUROC={m['auroc_macro']:.4f}")

    mean_acc   = np.mean([r['accuracy']    for r in results])
    mean_f1    = np.mean([r['f1_macro']    for r in results])
    mean_auroc = np.mean([r['auroc_macro'] for r in results])
    std_acc    = np.std([r['accuracy']     for r in results])
    std_f1     = np.std([r['f1_macro']     for r in results])
    std_auroc  = np.std([r['auroc_macro']  for r in results])

    print(f"\n  MEAN RESULTS (3 seeds):")
    print(f"    Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"    F1 Macro: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"    AUROC:    {mean_auroc:.4f} ± {std_auroc:.4f}")
    print(f"\n  Per-class report (seed=42):")
    print(results[0]['report'])

    return {
        'model': label,
        'mean_accuracy': mean_acc, 'std_accuracy': std_acc,
        'mean_f1_macro': mean_f1,  'std_f1_macro': std_f1,
        'mean_auroc':    mean_auroc, 'std_auroc': std_auroc,
    }


def main():
    parser = argparse.ArgumentParser(description="5-Class PTB-XL Evaluation for PA-SSL")
    parser.add_argument('--ptbxl_dir', type=str, required=True,
                        help='Path to PTB-XL root (contains ptbxl_database.csv)')
    parser.add_argument('--ptbxl_csv', type=str, default='data/ptbxl_processed.csv',
                        help='Path to beat-level CSV')
    parser.add_argument('--checkpoints_dir', type=str, default='experiments',
                        help='Root dir containing ssl_*/best_checkpoint.pth')
    parser.add_argument('--output_dir', type=str, default='experiments/multiclass_results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Build 5-class labels
    df_5class = build_5class_labels(args.ptbxl_csv, args.ptbxl_dir)

    # Step 2: Evaluate all checkpoints
    MODELS = [
        # (folder_name, encoder_type, display_label)
        ('ssl_passl_resnet_hybrid',  'resnet1d', 'PA-HybridSSL ResNet1D (Hybrid)'),
        ('ssl_passl_resnet_ntxent',  'resnet1d', 'PA-SSL ResNet1D (NT-Xent)'),
        ('ssl_passl_resnet_vicreg',  'resnet1d', 'PA-SSL ResNet1D (VICReg)'),
        ('ssl_passl_wavkan_hybrid',  'wavkan',   'PA-HybridSSL WavKAN (Hybrid)'),
        ('ssl_passl_wavkan_ntxent',  'wavkan',   'PA-SSL WavKAN (NT-Xent)'),
        ('ssl_passl_wavkan_vicreg',  'wavkan',   'PA-SSL WavKAN (VICReg)'),
        ('ssl_simclr_naive_resnet',  'resnet1d', 'SimCLR + Naive Aug (ResNet1D)'),
    ]

    all_results = []
    for folder, enc_type, display_name in MODELS:
        ckpt = os.path.join(args.checkpoints_dir, folder, 'best_checkpoint.pth')
        if not os.path.exists(ckpt):
            print(f"\nSkipping {display_name}: checkpoint not found at {ckpt}")
            continue
        result = evaluate_checkpoint(ckpt, enc_type, df_5class, device, display_name)
        all_results.append(result)

    # Step 3: Summary table
    df_results = pd.DataFrame(all_results)
    out_csv = os.path.join(args.output_dir, 'multiclass_5class_results.csv')
    df_results.to_csv(out_csv, index=False)

    print(f"\n{'='*60}")
    print("FINAL 5-CLASS SUMMARY TABLE")
    print('='*60)
    for _, row in df_results.iterrows():
        print(f"  {row['model']:<45} "
              f"Acc={row['mean_accuracy']:.4f}±{row['std_accuracy']:.4f}  "
              f"F1={row['mean_f1_macro']:.4f}±{row['std_f1_macro']:.4f}  "
              f"AUROC={row['mean_auroc']:.4f}±{row['std_auroc']:.4f}")

    print(f"\nResults saved to: {out_csv}")

    # Step 4: Generate LaTeX table snippet
    latex_path = os.path.join(args.output_dir, 'multiclass_table.tex')
    with open(latex_path, 'w') as f:
        f.write("% Auto-generated 5-class PTB-XL evaluation table\n")
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{5-class PTB-XL superclass evaluation (linear probe, mean $\\pm$ std over 3 seeds).}\n")
        f.write("\\label{tab:multiclass}\n")
        f.write("\\begin{tabular}{lccc}\n\\toprule\n")
        f.write("\\textbf{Method} & \\textbf{Accuracy} & \\textbf{F1 Macro} & \\textbf{AUROC} \\\\\n")
        f.write("\\midrule\n")
        for _, row in df_results.iterrows():
            name = row['model'].replace('&', '\\&')
            f.write(f"{name} & "
                    f"{row['mean_accuracy']:.4f}$\\pm${row['std_accuracy']:.4f} & "
                    f"{row['mean_f1_macro']:.4f}$\\pm${row['std_f1_macro']:.4f} & "
                    f"{row['mean_auroc']:.4f}$\\pm${row['std_auroc']:.4f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"LaTeX table saved to: {latex_path}")


if __name__ == '__main__':
    main()
