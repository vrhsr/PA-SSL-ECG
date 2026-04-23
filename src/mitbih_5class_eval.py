"""
PA-SSL: MIT-BIH 5-Class AAMI Arrhythmia Evaluation
====================================================
A harder evaluation benchmark that silences the "0.99 AUROC is too easy"
reviewer criticism. The AAMI EC57 standard defines 5 clinically meaningful
beat classes from MIT-BIH, where PA-SSL performance is expected ~0.82-0.85 AUROC
(far from trivial, differences between methods actually matter).

AAMI EC57 Beat Classes:
    N  — Normal beats (LBBB, RBBB, Aberrated atrial beats)
    S  — Supraventricular ectopy (APC, STTC)
    V  — Ventricular ectopy (PVC, VEB)
    F  — Fusion beats (ventricular-normal fusion)
    Q  — Unknown / paced beats

Usage:
    python -m src.mitbih_5class_eval \\
        --mitbih_csv data/mitbih_processed.csv \\
        --checkpoints_dir remote \\
        --output_dir results/mitbih_5class

References:
    ANSI/AAMI EC57:2012 Testing and reporting performance results of
    cardiac rhythm and ST segment measurement algorithms.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, classification_report, confusion_matrix
)
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.encoder import build_encoder

# ─── AAMI EC57 MAPPING ───────────────────────────────────────────────────────
# MIT-BIH annotation symbols → AAMI 5-class
# Reference: https://www.physionet.org/physiobank/database/html/mitdbdir/tables.htm

MITBIH_TO_AAMI = {
    # N — Normal
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    # S — Supraventricular ectopy
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    # V — Ventricular ectopy
    'V': 'V', 'E': 'V',
    # F — Fusion
    'F': 'F',
    # Q — Unknown/paced
    '/': 'Q', 'f': 'Q', 'Q': 'Q',
}

AAMI_CLASSES = ['N', 'S', 'V', 'F', 'Q']
AAMI_TO_IDX = {c: i for i, c in enumerate(AAMI_CLASSES)}


def map_labels_to_aami(df):
    """
    Map MIT-BIH beat annotations → AAMI 5-class labels.
    Expects df to have a 'beat_type' or 'label' column with MIT-BIH symbols.
    Falls back to integer label column → maps 0/1 binary to N/V.
    """
    if 'beat_type' in df.columns:
        df = df.copy()
        df['aami_label'] = df['beat_type'].map(MITBIH_TO_AAMI)
        df = df.dropna(subset=['aami_label'])
        df['label_5class'] = df['aami_label'].map(AAMI_TO_IDX)
    elif 'label' in df.columns:
        # Fallback: label=0 → N (normal), label=1 → V (ventricular)
        # This is the binary dataset we already have; map accordingly
        print("  [WARN] No 'beat_type' column. Using binary label→AAMI mapping:")
        print("         label=0 → N (normal), label=1 → V (ventricular ectopy)")
        df = df.copy()
        df['aami_label'] = df['label'].map({0: 'N', 1: 'V'})
        df['label_5class'] = df['aami_label'].map(AAMI_TO_IDX)
        print(f"         Effective classes: N, V (binary fallback)")
    else:
        raise ValueError("DataFrame must have 'beat_type' or 'label' column")

    print(f"\n  AAMI class distribution:")
    for cls in AAMI_CLASSES:
        n = (df['aami_label'] == cls).sum() if 'aami_label' in df.columns else 0
        if n > 0:
            print(f"    {cls}: {n:,}")
    return df


# ─── CHECKPOINT LOADING ───────────────────────────────────────────────────────

def load_encoder(ckpt_path, encoder_name, device):
    """Load encoder from SSL checkpoint."""
    encoder = build_encoder(encoder_name)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get('encoder_state_dict', ckpt.get('model_state_dict', ckpt))
    if any(k.startswith('encoder.') for k in state_dict.keys()):
        state_dict = {k.replace('encoder.', '', 1): v for k, v in state_dict.items()
                      if k.startswith('encoder.')}
    encoder.load_state_dict(state_dict, strict=False)
    return encoder.to(device).eval()


# ─── REPRESENTATION EXTRACTION ───────────────────────────────────────────────

@torch.no_grad()
def extract_representations(encoder, df, device, batch_size=512):
    """Extract frozen representations from encoder."""
    signal_cols = [c for c in df.columns if str(c).isdigit()]
    signals = df[signal_cols].values.astype(np.float32)
    labels = df['label_5class'].values.astype(np.int64)

    reprs = []
    encoder.eval()
    for start in tqdm(range(0, len(signals), batch_size), desc="  Extracting", leave=False):
        batch = torch.tensor(signals[start:start + batch_size]).to(device)
        r = encoder.encode(batch)
        reprs.append(r.cpu().numpy())

    return np.concatenate(reprs, axis=0), labels


# ─── EVALUATION ───────────────────────────────────────────────────────────────

def sanitize_representations(reprs):
    """Replace NaN/Inf with column means — prevents sklearn crashes from bad encoder outputs."""
    if not np.isfinite(reprs).all():
        n_bad = (~np.isfinite(reprs)).sum()
        print(f"    [WARN] Found {n_bad} NaN/Inf values in representations — imputing with column mean")
        col_means = np.nanmean(reprs, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        inds = np.where(~np.isfinite(reprs))
        reprs[inds] = np.take(col_means, inds[1])
    return reprs


def evaluate_linear_probe(train_reprs, train_labels, test_reprs, test_labels):
    """Fit logistic regression and return metrics."""
    classes = np.unique(np.concatenate([train_labels, test_labels]))
    n_classes = len(classes)

    # Sanitize NaN/Inf before fitting
    train_reprs = sanitize_representations(train_reprs.copy())
    test_reprs = sanitize_representations(test_reprs.copy())

    scaler = StandardScaler()
    tr_s = scaler.fit_transform(train_reprs)
    te_s = scaler.transform(test_reprs)

    # Final safety: replace any residual NaN after scaler
    tr_s = np.nan_to_num(tr_s, nan=0.0, posinf=0.0, neginf=0.0)
    te_s = np.nan_to_num(te_s, nan=0.0, posinf=0.0, neginf=0.0)

    clf = LogisticRegression(
        max_iter=1000, C=1.0, solver='lbfgs',
        multi_class='multinomial' if n_classes > 2 else 'auto',
        n_jobs=-1
    )
    clf.fit(tr_s, train_labels)
    preds = clf.predict(te_s)
    probs = clf.predict_proba(te_s)

    acc = accuracy_score(test_labels, preds)
    f1_macro = f1_score(test_labels, preds, average='macro', zero_division=0)
    f1_weighted = f1_score(test_labels, preds, average='weighted', zero_division=0)

    # AUROC: OvR for multi-class
    try:
        if n_classes == 2:
            auroc = roc_auc_score(test_labels, probs[:, 1])
        else:
            auroc = roc_auc_score(
                test_labels, probs, multi_class='ovr',
                average='macro', labels=list(range(n_classes))
            )
    except Exception:
        auroc = float('nan')

    # AUPRC
    try:
        from sklearn.preprocessing import label_binarize
        if n_classes == 2:
            auprc = average_precision_score(test_labels, probs[:, 1])
        else:
            y_bin = label_binarize(test_labels, classes=list(range(n_classes)))
            auprc = average_precision_score(y_bin, probs, average='macro')
    except Exception:
        auprc = float('nan')

    report = classification_report(
        test_labels, preds,
        target_names=[AAMI_CLASSES[c] for c in sorted(classes)],
        zero_division=0, output_dict=True
    )

    cm = confusion_matrix(test_labels, preds, labels=list(range(n_classes)))

    return {
        'accuracy': float(acc),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'auroc_macro': float(auroc),
        'auprc_macro': float(auprc),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'n_classes': n_classes,
    }


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='MIT-BIH 5-class AAMI arrhythmia evaluation')
    parser.add_argument('--mitbih_csv', type=str,
                        default='data/mitbih_processed.csv')
    parser.add_argument('--passl_checkpoint', type=str, required=True,
                        help='Path to PA-HybridSSL (ResNet1D) best_checkpoint.pth')
    parser.add_argument('--simclr_checkpoint', type=str, required=True,
                        help='Path to SimCLR Naive (ResNet1D) best_checkpoint.pth')
    parser.add_argument('--wavkan_checkpoint', type=str, default='',
                        help='Optional: path to WavKAN hybrid best_checkpoint.pth')
    parser.add_argument('--output_dir', type=str, default='results/mitbih_5class')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 7, 123])
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Cap samples for testing (0 = all)')
    args = parser.parse_args()

    CHECKPOINTS = {
        'PA-HybridSSL (ResNet1D)': (args.passl_checkpoint, 'resnet1d'),
        'SimCLR + Naive Aug':      (args.simclr_checkpoint, 'resnet1d'),
    }
    if args.wavkan_checkpoint:
        CHECKPOINTS['PA-HybridSSL (WavKAN)'] = (args.wavkan_checkpoint, 'wavkan')

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load and map data ──────────────────────────────────────────────────
    print(f"\nLoading MIT-BIH data: {args.mitbih_csv}")
    df = pd.read_csv(args.mitbih_csv)
    if args.max_samples > 0 and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=42).reset_index(drop=True)

    df = map_labels_to_aami(df)
    print(f"Total usable beats: {len(df):,}")

    all_results = []

    for model_name, (ckpt_path, enc_name) in CHECKPOINTS.items():
        if not os.path.exists(ckpt_path):
            print(f"\n  [SKIP] {model_name}: checkpoint not found at {ckpt_path}")
            continue

        print(f"\n{'='*60}")
        print(f"  Model: {model_name}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"{'='*60}")

        encoder = load_encoder(ckpt_path, enc_name, device)

        seed_results = []
        for seed in args.seeds:
            print(f"\n  Seed {seed}:")

            # Patient-aware split
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            split_groups = df['patient_id'].values
            train_idx, test_idx = next(gss.split(df, groups=split_groups))
            df_train = df.iloc[train_idx].reset_index(drop=True)
            df_test = df.iloc[test_idx].reset_index(drop=True)

            print(f"    Train: {len(df_train):,} | Test: {len(df_test):,}")

            tr_reprs, tr_labels = extract_representations(encoder, df_train, device)
            te_reprs, te_labels = extract_representations(encoder, df_test, device)

            metrics = evaluate_linear_probe(tr_reprs, tr_labels, te_reprs, te_labels)
            metrics['model'] = model_name
            metrics['seed'] = seed
            seed_results.append(metrics)
            all_results.append(metrics)

            print(f"    AUROC (macro): {metrics['auroc_macro']:.4f}")
            print(f"    F1 (macro):    {metrics['f1_macro']:.4f}")
            print(f"    Accuracy:      {metrics['accuracy']:.4f}")

        # Per-model summary
        if seed_results:
            aurocs = [r['auroc_macro'] for r in seed_results if not np.isnan(r['auroc_macro'])]
            f1s = [r['f1_macro'] for r in seed_results]
            accs = [r['accuracy'] for r in seed_results]
            print(f"\n  {model_name} — Summary over {len(seed_results)} seeds:")
            print(f"    AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
            print(f"    F1:    {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
            print(f"    Acc:   {np.mean(accs):.4f} ± {np.std(accs):.4f}")

        del encoder
        torch.cuda.empty_cache()

    # ── Aggregate and save ─────────────────────────────────────────────────
    if not all_results:
        print("\n[ERROR] No results collected. Check checkpoint paths.")
        return

    # Strip non-serializable fields for CSV
    csv_results = []
    for r in all_results:
        row = {k: v for k, v in r.items()
               if k not in ('classification_report', 'confusion_matrix')}
        csv_results.append(row)

    results_df = pd.DataFrame(csv_results)
    raw_path = os.path.join(args.output_dir, 'mitbih_5class_raw.csv')
    results_df.to_csv(raw_path, index=False)

    # Summary table
    summary_rows = []
    for model_name in results_df['model'].unique():
        sub = results_df[results_df['model'] == model_name]
        summary_rows.append({
            'model': model_name,
            'auroc_mean': sub['auroc_macro'].mean(),
            'auroc_std': sub['auroc_macro'].std(),
            'f1_mean': sub['f1_macro'].mean(),
            'f1_std': sub['f1_macro'].std(),
            'acc_mean': sub['accuracy'].mean(),
            'acc_std': sub['accuracy'].std(),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir, 'mitbih_5class_summary.csv')
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'='*70}")
    print("  FINAL SUMMARY: MIT-BIH AAMI 5-CLASS RESULTS")
    print(f"{'='*70}")
    print(f"\n{'Model':<35} {'AUROC':>12} {'F1 Macro':>12} {'Accuracy':>12}")
    print("-" * 73)
    for _, row in summary_df.iterrows():
        print(f"  {row['model']:<33} "
              f"{row['auroc_mean']:.4f}±{row['auroc_std']:.4f}  "
              f"{row['f1_mean']:.4f}±{row['f1_std']:.4f}  "
              f"{row['acc_mean']:.4f}±{row['acc_std']:.4f}")

    print(f"\n  Results saved:")
    print(f"    Raw:     {raw_path}")
    print(f"    Summary: {summary_path}")


if __name__ == '__main__':
    main()
