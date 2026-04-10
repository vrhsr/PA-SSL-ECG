"""
PA-SSL: Downstream Evaluation Script

Evaluates pretrained SSL encoders via:
  1. Linear probing (frozen encoder + logistic regression)
  2. Fine-tuning (encoder + linear head, end-to-end)
  3. Mahalanobis anomaly detection with calibration
  4. Label-efficiency experiments (1%, 5%, 10%, 25%, 50%, 100%)
  5. Cross-dataset generalization

Usage:
    python -m src.evaluate --checkpoint experiments/ssl_resnet1d_physio_temporal/best_checkpoint.pth
    python -m src.evaluate --checkpoint ... --label_fractions 0.01 0.05 0.1 1.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import argparse
import os
import json
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
    average_precision_score
)

from src.data.ecg_dataset import ECGBeatDataset, patient_aware_split
from src.models.encoder import build_encoder
from src.models.anomaly_scorer import (
    MahalanobisAnomalyScorer, expected_calibration_error, brier_score,
    sensitivity_specificity
)


# ═══════════════════════════════════════════════════════════════════════════════
# REPRESENTATION EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_representations(encoder, dataset, device, batch_size=256, max_batches=None):
    """Extract frozen representations from an encoder for all samples."""
    encoder.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    all_reprs = []
    all_labels = []
    
    # Add progress bar for visibility during extraction
    pbar = tqdm(loader, desc="Extracting Representations", 
                leave=False, disable=(max_batches is not None and max_batches < 50))
    
    for batch_idx, batch in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break
        signals, labels = batch[0], batch[1]
        signals = signals.to(device)
        reprs = encoder.encode(signals)
        all_reprs.append(reprs.cpu().numpy())
        all_labels.append(labels.numpy())
    
    return np.concatenate(all_reprs), np.concatenate(all_labels)


def representation_quality_metrics(reprs, labels=None):
    """
    Computes SSL representation quality metrics.
    - Silhouette Score: Cluster separation (-1 to 1)
    - Davies-Bouldin Index: Cluster compactness (lower is better)
    - Embedding Collapse: Std deviation across dimensions (should be > 0.0)
    - Uniformity: Average pairwise distance between all points
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    from sklearn.metrics.pairwise import pairwise_distances
    
    metrics = {}
    
    # Collapse detection
    feature_stds = np.std(reprs, axis=0)
    metrics['embedding_std_mean'] = float(np.mean(feature_stds))
    metrics['embedding_std_min'] = float(np.min(feature_stds))
    metrics['is_collapsed'] = bool(metrics['embedding_std_mean'] < 1e-4)
    
    if len(reprs) > 10000:
        idx = np.random.choice(len(reprs), 10000, replace=False)
        sub_reprs = reprs[idx]
        sub_labels = labels[idx] if labels is not None else None
    else:
        sub_reprs = reprs
        sub_labels = labels if labels is not None else None
        
    if len(sub_reprs) > 2000:
        u_idx = np.random.choice(len(sub_reprs), 2000, replace=False)
        u_reprs = sub_reprs[u_idx]
    else:
        u_reprs = sub_reprs
        
    dists = pairwise_distances(u_reprs, metric='sqeuclidean')
    np.fill_diagonal(dists, 0)
    metrics['uniformity_l2'] = float(np.sum(dists) / (len(u_reprs) * (len(u_reprs)-1)))
    
    if sub_labels is not None and len(np.unique(sub_labels)) > 1:
        try:
            metrics['silhouette'] = float(silhouette_score(sub_reprs, sub_labels))
            metrics['davies_bouldin'] = float(davies_bouldin_score(sub_reprs, sub_labels))
        except Exception:
            metrics['silhouette'] = np.nan
            metrics['davies_bouldin'] = np.nan
            
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# LINEAR PROBING
# ═══════════════════════════════════════════════════════════════════════════════

def linear_probe(train_reprs, train_labels, test_reprs, test_labels):
    """Fit logistic regression on frozen representations."""
    clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    clf.fit(train_reprs, train_labels)
    
    predictions = clf.predict(test_reprs)
    probabilities = clf.predict_proba(test_reprs)
    
    acc = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, average='macro')
    
    try:
        if probabilities.shape[1] == 2:
            auroc = roc_auc_score(test_labels, probabilities[:, 1])
            auprc = average_precision_score(test_labels, probabilities[:, 1])
            sens, spec = sensitivity_specificity(test_labels, probabilities[:, 1])
        else:
            auroc = roc_auc_score(test_labels, probabilities, multi_class='ovr')
            auprc = average_precision_score(pd.get_dummies(test_labels), probabilities, average='macro')
            sens, spec = 0.0, 0.0
    except Exception:
        auroc, auprc, sens, spec = 0.0, 0.0, 0.0, 0.0
    
    ece, _ = expected_calibration_error(test_labels, probabilities)
    bs = brier_score(test_labels, probabilities)
    
    return {
        'accuracy': acc,
        'f1_macro': f1,
        'auroc': auroc,
        'auprc': auprc,
        'sensitivity': sens,
        'specificity': spec,
        'ece': ece,
        'brier_score': bs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FINE-TUNING
# ═══════════════════════════════════════════════════════════════════════════════

def fine_tune(encoder, train_dataset, test_dataset, device,
              num_classes=2, epochs=30, lr=1e-4, batch_size=64):
    """Fine-tune encoder + linear head end-to-end."""
    model = encoder.set_classifier(num_classes).to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=2)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            signals, labels = batch[0], batch[1]
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(signals)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            signals, labels = batch[0], batch[1]
            signals = signals.to(device)
            logits = model(signals)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    try:
        if all_probs.shape[1] == 2:
            auroc = roc_auc_score(all_labels, all_probs[:, 1])
            auprc = average_precision_score(all_labels, all_probs[:, 1])
            sens, spec = sensitivity_specificity(all_labels, all_probs[:, 1])
        else:
            auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
            auprc = average_precision_score(pd.get_dummies(all_labels), all_probs, average='macro')
            sens, spec = 0.0, 0.0
    except Exception:
        auroc, auprc, sens, spec = 0.0, 0.0, 0.0, 0.0
    
    ece, _ = expected_calibration_error(all_labels, all_probs)
    bs = brier_score(all_labels, all_probs)
    
    return {
        'accuracy': acc,
        'f1_macro': f1,
        'auroc': auroc,
        'auprc': auprc,
        'sensitivity': sens,
        'specificity': spec,
        'ece': ece,
        'brier_score': bs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAHALANOBIS ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def mahalanobis_eval(train_reprs, train_labels, test_reprs, test_labels):
    """Evaluate Mahalanobis-based anomaly detection."""
    scorer = MahalanobisAnomalyScorer(use_shrinkage=True)
    scorer.fit(train_reprs, train_labels)
    
    probs = scorer.predict_proba(test_reprs)
    predictions = probs.argmax(axis=1)
    
    acc = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, average='macro')
    
    try:
        if probs.shape[1] == 2:
            auroc = roc_auc_score(test_labels, probs[:, 1])
            auprc = average_precision_score(test_labels, probs[:, 1])
            sens, spec = sensitivity_specificity(test_labels, probs[:, 1])
        else:
            auroc = roc_auc_score(test_labels, probs, multi_class='ovr')
            auprc = average_precision_score(pd.get_dummies(test_labels), probs, average='macro')
            sens, spec = 0.0, 0.0
    except Exception:
        auroc, auprc, sens, spec = 0.0, 0.0, 0.0, 0.0
    
    ece, _ = expected_calibration_error(test_labels, probs)
    bs = brier_score(test_labels, probs)
    
    return {
        'accuracy': acc,
        'f1_macro': f1,
        'auroc': auroc,
        'auprc': auprc,
        'sensitivity': sens,
        'specificity': spec,
        'ece': ece,
        'brier_score': bs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LABEL EFFICIENCY EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════

def label_efficiency_experiment(encoder, train_csv, test_csv, device,
                                fractions=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
                                n_seeds=3):
    """
    Run label-efficiency experiment across multiple label fractions.
    Each fraction is run with multiple seeds for variance estimation.
    """
    results = []
    
    for frac in fractions:
        for seed in range(n_seeds):
            print(f"\n  Label fraction: {frac*100:.0f}% (seed={seed})")
            
            # Create datasets with subsampled labels
            train_ds = ECGBeatDataset(train_csv, label_fraction=frac, seed=seed)
            test_ds = ECGBeatDataset(test_csv)
            
            # Extract representations
            train_reprs, train_labels = extract_representations(encoder, train_ds, device)
            test_reprs, test_labels = extract_representations(encoder, test_ds, device)
            
            # Only use labeled samples for training
            labeled_mask = train_ds.label_mask
            train_reprs_labeled = train_reprs[labeled_mask]
            train_labels_labeled = train_labels[labeled_mask]
            
            # Linear probe
            lp_metrics = linear_probe(
                train_reprs_labeled, train_labels_labeled,
                test_reprs, test_labels
            )
            
            # Mahalanobis
            mah_metrics = mahalanobis_eval(
                train_reprs_labeled, train_labels_labeled,
                test_reprs, test_labels
            )
            
            result = {
                'label_fraction': frac,
                'seed': seed,
                'n_labeled': int(labeled_mask.sum()),
                **{f'linear_{k}': v for k, v in lp_metrics.items()},
                **{f'mahal_{k}': v for k, v in mah_metrics.items()},
            }
            results.append(result)
            
            print(f"    Linear: acc={lp_metrics['accuracy']:.4f}, "
                  f"auroc={lp_metrics['auroc']:.4f}, ece={lp_metrics['ece']:.4f}")
            print(f"    Mahal:  acc={mah_metrics['accuracy']:.4f}, "
                  f"auroc={mah_metrics['auroc']:.4f}, ece={mah_metrics['ece']:.4f}")
    
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(args):
    """Run full evaluation suite."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"PA-SSL Evaluation")
    print(f"  Device: {device}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"{'='*60}\n")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config', {})
    encoder_name = config.get('encoder', args.encoder)
    
    # Rebuild encoder
    encoder = build_encoder(encoder_name, proj_dim=config.get('proj_dim', 128))
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder = encoder.to(device)
    encoder.eval()
    
    print(f"Loaded encoder: {encoder_name} (epoch {checkpoint.get('epoch', '?')})")
    
    # Output directory
    eval_dir = os.path.join(os.path.dirname(args.checkpoint), 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    
    # ─── Patient-Aware Split ──────────────────────────────────────────────
    train_df, val_df, test_df = patient_aware_split(args.data_file)
    
    # We no longer save massive splits to CSV as it stalls evaluation.
    # Instead, instantiate datasets directly in-memory from the split DataFrames.
    
    # ─── Representation Quality Metrics ─────────────────────────────────────
    print("\n--- Representation Quality Metrics ---")
    
    try:
        test_ds_full = ECGBeatDataset(test_df)
        reprs_full, labels_full = extract_representations(encoder, test_ds_full, device)
        
        rep_metrics = representation_quality_metrics(reprs_full, labels_full)
        for k, v in rep_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
                
        # Save representation metrics
        with open(os.path.join(eval_dir, 'representation_quality.json'), 'w') as f:
            json.dump(rep_metrics, f, indent=2)
    except Exception as e:
        print(f"  Failed to compute representation metrics: {e}")
        
    # ─── Label Efficiency Experiment ──────────────────────────────────────
    print("\n--- Label Efficiency Experiment ---")
    
    fractions = [float(f) for f in args.label_fractions]
    
    le_results = label_efficiency_experiment(
        encoder,
        train_df,
        test_df,
        device,
        fractions=fractions,
        n_seeds=args.n_seeds,
    )
    
    le_results.to_csv(os.path.join(eval_dir, 'label_efficiency.csv'), index=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Label Efficiency Summary (Linear Probe):")
    summary = le_results.groupby('label_fraction').agg({
        'linear_accuracy': ['mean', 'std'],
        'linear_auroc': ['mean', 'std'],
        'linear_ece': ['mean', 'std'],
    }).round(4)
    print(summary.to_string())
    
    print(f"\nResults saved to: {eval_dir}")
    print(f"{'='*60}")
    
    return le_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PA-SSL Evaluation")
    
    # Paths
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_file', type=str, default='data/ptbxl_processed.csv')
    parser.add_argument('--encoder', type=str, default='resnet1d',
                        choices=['resnet1d', 'wavkan', 'transformer', 'mamba'])
    
    # Label efficiency
    parser.add_argument('--label_fractions', nargs='+', type=float,
                        default=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0])
    parser.add_argument('--n_seeds', type=int, default=3)
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Limit number of batches per evaluation (for smoke test)')
    
    args = parser.parse_args()
    evaluate(args)
