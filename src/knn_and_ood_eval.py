"""
PA-SSL: kNN Evaluation + OOD Detection Comparison
===================================================
Two experiments in one script:

1. kNN Classification (no learned classifier):
   Standard representation quality metric used in DINO, MAE, I-JEPA papers.
   Computes k-nearest-neighbor accuracy on frozen embeddings.

2. OOD Detection Method Comparison:
   Compares Mahalanobis vs Euclidean Distance vs kNN anomaly scoring
   on the PTB-XL (normal) vs Chapman (abnormal) OOD detection task.

Runtime: ~10-15 minutes total (forward passes + distance computation)

Usage:
    python -m src.knn_and_ood_eval \
        --ptbxl_csv data/ptbxl_processed.csv \
        --chapman_csv data/chapman_processed.csv \
        --checkpoints_dir remote \
        --output_dir results/knn_ood_eval
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.covariance import LedoitWolf
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.encoder import build_encoder


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
def extract_representations(encoder, df, device, batch_size=512, max_samples=50000):
    """Extract frozen representations."""
    signal_cols = [c for c in df.columns if str(c).isdigit()]
    if max_samples > 0 and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    signals = df[signal_cols].values.astype(np.float32)
    labels = df['label'].values.astype(np.int64)
    patient_ids = df['patient_id'].values

    reprs = []
    encoder.eval()
    for start in tqdm(range(0, len(signals), batch_size), desc="  Extracting", leave=False):
        batch = torch.tensor(signals[start:start + batch_size]).to(device)
        r = encoder.encode(batch)
        reprs.append(r.cpu().numpy())

    return np.concatenate(reprs), labels, patient_ids


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: kNN EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def knn_evaluate(train_reprs, train_labels, test_reprs, test_labels, k_values=[5, 10, 20]):
    """
    Compute kNN accuracy for multiple k values.
    No model fitting — purely distance-based.
    """
    # Sanitize NaN/Inf
    train_reprs = np.nan_to_num(train_reprs.copy(), nan=0.0, posinf=0.0, neginf=0.0)
    test_reprs = np.nan_to_num(test_reprs.copy(), nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    tr_s = scaler.fit_transform(train_reprs)
    te_s = scaler.transform(test_reprs)

    results = {}
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine', n_jobs=-1)
        knn.fit(tr_s, train_labels)
        preds = knn.predict(te_s)
        acc = accuracy_score(test_labels, preds)
        f1 = f1_score(test_labels, preds, average='macro', zero_division=0)
        results[f'knn_{k}_acc'] = float(acc)
        results[f'knn_{k}_f1'] = float(f1)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: OOD DETECTION COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def mahalanobis_ood(in_reprs, ood_reprs):
    """Mahalanobis distance-based OOD detection."""
    scaler = StandardScaler()
    in_s = scaler.fit_transform(in_reprs)
    ood_s = scaler.transform(ood_reprs)

    # Fit class-conditional Gaussian on in-distribution
    lw = LedoitWolf()
    lw.fit(in_s)
    mean = np.mean(in_s, axis=0)
    precision = lw.precision_

    # Distance for in-distribution
    diff_in = in_s - mean
    in_dists = np.sqrt(np.sum(diff_in @ precision * diff_in, axis=1))

    # Distance for OOD
    diff_ood = ood_s - mean
    ood_dists = np.sqrt(np.sum(diff_ood @ precision * diff_ood, axis=1))

    return in_dists, ood_dists


def euclidean_ood(in_reprs, ood_reprs):
    """Euclidean distance to centroid OOD detection."""
    scaler = StandardScaler()
    in_s = scaler.fit_transform(in_reprs)
    ood_s = scaler.transform(ood_reprs)

    centroid = np.mean(in_s, axis=0)
    in_dists = np.linalg.norm(in_s - centroid, axis=1)
    ood_dists = np.linalg.norm(ood_s - centroid, axis=1)
    return in_dists, ood_dists


def knn_ood(in_reprs, ood_reprs, k=5):
    """kNN distance-based OOD detection (distance to k-th nearest in-distribution neighbor)."""
    from sklearn.neighbors import NearestNeighbors

    scaler = StandardScaler()
    in_s = scaler.fit_transform(in_reprs)
    ood_s = scaler.transform(ood_reprs)

    nn = NearestNeighbors(n_neighbors=k, metric='cosine', n_jobs=-1)
    nn.fit(in_s)

    in_dists, _ = nn.kneighbors(in_s)
    ood_dists_arr, _ = nn.kneighbors(ood_s)

    # Use mean distance to k nearest neighbors as anomaly score
    in_scores = np.mean(in_dists, axis=1)
    ood_scores = np.mean(ood_dists_arr, axis=1)
    return in_scores, ood_scores


def compute_ood_auroc(in_scores, ood_scores):
    """Compute AUROC for OOD detection: in-distribution=0, OOD=1."""
    labels = np.concatenate([np.zeros(len(in_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([in_scores, ood_scores])
    try:
        auroc = roc_auc_score(labels, scores)
    except Exception:
        auroc = float('nan')
    return float(auroc)


def compute_detection_rate(in_scores, ood_scores, specificity=0.95):
    """Compute OOD detection rate at a given specificity level."""
    threshold = np.percentile(in_scores, specificity * 100)
    detection_rate = np.mean(ood_scores > threshold)
    return float(detection_rate), float(threshold)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='kNN evaluation + OOD detection comparison')
    parser.add_argument('--ptbxl_csv', type=str, default='data/ptbxl_processed.csv')
    parser.add_argument('--chapman_csv', type=str, default='data/chapman_processed.csv')
    parser.add_argument('--passl_checkpoint', type=str, required=True,
                        help='Path to PA-HybridSSL best_checkpoint.pth')
    parser.add_argument('--simclr_checkpoint', type=str, required=True,
                        help='Path to SimCLR Naive best_checkpoint.pth')
    parser.add_argument('--output_dir', type=str, default='results/knn_ood_eval')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 7, 123])
    parser.add_argument('--max_samples', type=int, default=30000)
    args = parser.parse_args()

    CHECKPOINTS = {
        'PA-HybridSSL (ResNet1D)': (args.passl_checkpoint, 'resnet1d'),
        'SimCLR + Naive Aug':      (args.simclr_checkpoint, 'resnet1d'),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load PTB-XL Data ─────────────────────────────────────────────────
    print(f"\nLoading PTB-XL: {args.ptbxl_csv}")
    df_ptbxl = pd.read_csv(args.ptbxl_csv)
    print(f"  {len(df_ptbxl):,} beats, {df_ptbxl['patient_id'].nunique()} patients")

    # ── Load Chapman for OOD ─────────────────────────────────────────────
    has_chapman = os.path.exists(args.chapman_csv)
    if has_chapman:
        print(f"Loading Chapman: {args.chapman_csv}")
        df_chapman = pd.read_csv(args.chapman_csv)
        print(f"  {len(df_chapman):,} beats")
    else:
        print(f"  [WARN] Chapman CSV not found at {args.chapman_csv}")
        print(f"         OOD detection comparison will be skipped.")
        df_chapman = None

    all_knn_results = []
    all_ood_results = []

    for model_name, (ckpt_path, enc_name) in CHECKPOINTS.items():
        if not os.path.exists(ckpt_path):
            print(f"\n  [SKIP] {model_name}: {ckpt_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"  Model: {model_name}")
        print(f"{'='*60}")

        encoder = load_encoder(ckpt_path, enc_name, device)

        for seed in args.seeds:
            print(f"\n  Seed {seed}:")

            # ── kNN Evaluation on PTB-XL ─────────────────────────────────
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            train_idx, test_idx = next(gss.split(df_ptbxl, groups=df_ptbxl['patient_id']))

            df_train = df_ptbxl.iloc[train_idx].reset_index(drop=True)
            df_test = df_ptbxl.iloc[test_idx].reset_index(drop=True)

            tr_reprs, tr_labels, _ = extract_representations(
                encoder, df_train, device, max_samples=args.max_samples)
            te_reprs, te_labels, _ = extract_representations(
                encoder, df_test, device, max_samples=args.max_samples // 4)

            knn_res = knn_evaluate(tr_reprs, tr_labels, te_reprs, te_labels)
            knn_res['model'] = model_name
            knn_res['seed'] = seed
            all_knn_results.append(knn_res)

            print(f"    kNN-5 Acc:  {knn_res['knn_5_acc']:.4f}")
            print(f"    kNN-10 Acc: {knn_res['knn_10_acc']:.4f}")
            print(f"    kNN-20 Acc: {knn_res['knn_20_acc']:.4f}")

            # ── OOD Detection Comparison ─────────────────────────────────
            if df_chapman is not None:
                # Normal beats from PTB-XL (in-distribution)
                in_mask = tr_labels == 0
                in_reprs = tr_reprs[in_mask]
                if len(in_reprs) > 5000:
                    rng = np.random.default_rng(seed)
                    idx = rng.choice(len(in_reprs), 5000, replace=False)
                    in_reprs = in_reprs[idx]

                # Chapman beats (out-of-distribution)
                ood_reprs, _, _ = extract_representations(
                    encoder, df_chapman, device, max_samples=5000)

                ood_row = {'model': model_name, 'seed': seed}

                for method_name, method_fn in [
                    ('Mahalanobis', lambda i, o: mahalanobis_ood(i, o)),
                    ('Euclidean', lambda i, o: euclidean_ood(i, o)),
                    ('kNN (k=5)', lambda i, o: knn_ood(i, o, k=5)),
                ]:
                    in_scores, ood_scores = method_fn(in_reprs, ood_reprs)
                    auroc = compute_ood_auroc(in_scores, ood_scores)
                    dr, thresh = compute_detection_rate(in_scores, ood_scores)

                    ood_row[f'{method_name}_auroc'] = auroc
                    ood_row[f'{method_name}_dr@95'] = dr
                    print(f"    OOD [{method_name}]: AUROC={auroc:.4f}, DR@95%={dr:.3f}")

                all_ood_results.append(ood_row)

        del encoder
        torch.cuda.empty_cache()

    # ── Save Results ─────────────────────────────────────────────────────
    if all_knn_results:
        knn_df = pd.DataFrame(all_knn_results)
        knn_df.to_csv(os.path.join(args.output_dir, 'knn_results_raw.csv'), index=False)

        # kNN Summary
        print(f"\n{'='*70}")
        print("  kNN CLASSIFICATION SUMMARY")
        print(f"{'='*70}")
        print(f"\n{'Model':<35} {'kNN-5':>10} {'kNN-10':>10} {'kNN-20':>10}")
        print("-" * 67)
        for model in knn_df['model'].unique():
            sub = knn_df[knn_df['model'] == model]
            print(f"  {model:<33} "
                  f"{sub['knn_5_acc'].mean():.4f}±{sub['knn_5_acc'].std():.3f}  "
                  f"{sub['knn_10_acc'].mean():.4f}±{sub['knn_10_acc'].std():.3f}  "
                  f"{sub['knn_20_acc'].mean():.4f}±{sub['knn_20_acc'].std():.3f}")

        knn_summary = knn_df.groupby('model').agg({
            'knn_5_acc': ['mean', 'std'],
            'knn_10_acc': ['mean', 'std'],
            'knn_20_acc': ['mean', 'std'],
            'knn_5_f1': ['mean', 'std'],
        }).reset_index()
        knn_summary.to_csv(os.path.join(args.output_dir, 'knn_summary.csv'), index=False)

    if all_ood_results:
        ood_df = pd.DataFrame(all_ood_results)
        ood_df.to_csv(os.path.join(args.output_dir, 'ood_comparison_raw.csv'), index=False)

        print(f"\n{'='*70}")
        print("  OOD DETECTION COMPARISON SUMMARY")
        print(f"{'='*70}")
        print(f"\n{'Model':<30} {'Mahalanobis':>14} {'Euclidean':>14} {'kNN (k=5)':>14}")
        print("-" * 74)
        for model in ood_df['model'].unique():
            sub = ood_df[ood_df['model'] == model]
            mah = f"{sub['Mahalanobis_auroc'].mean():.4f}"
            euc = f"{sub['Euclidean_auroc'].mean():.4f}"
            knn = f"{sub['kNN (k=5)_auroc'].mean():.4f}"
            print(f"  {model:<28} {mah:>14} {euc:>14} {knn:>14}")

        ood_summary = ood_df.groupby('model').mean(numeric_only=True).reset_index()
        ood_summary.to_csv(os.path.join(args.output_dir, 'ood_summary.csv'), index=False)

    print(f"\n  All results saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
