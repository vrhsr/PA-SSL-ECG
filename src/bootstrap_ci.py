"""
PA-SSL: Bootstrap Confidence Interval Generator
===============================================
Top-tier medical/engineering journals (like IEEE JBHI) strongly prefer 
95% Confidence Intervals (CIs) over just mean ± std. 
This script implements non-parametric bootstrap resampling (1000 resamples) 
to compute the 95% CI for Accuracy, F1, and AUROC from model predictions.

Usage:
    # Requires a CSV with columns: [true_label, pred_prob_class0, pred_prob_class1, ...]
    python -m src.bootstrap_ci --predictions_csv results.csv --n_bootstraps 1000
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

def compute_metrics(y_true, y_pred, y_prob, is_multiclass=False):
    """Compute standard metrics."""
    acc = accuracy_score(y_true, y_pred)
    
    if is_multiclass:
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        try:
            auroc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        except Exception:
            auroc = float('nan')
    else:
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        try:
            # y_prob should be 1D (probabilities of positive class)
            auroc = roc_auc_score(y_true, y_prob)
        except Exception:
            auroc = float('nan')
            
    return acc, f1, auroc


def bootstrap_ci(y_true, y_pred, y_prob, is_multiclass=False, n_bootstraps=1000, ci_level=0.95):
    """Compute 95% CI using non-parametric bootstrapping."""
    np.random.seed(42)
    n_samples = len(y_true)
    
    boot_acc = []
    boot_f1 = []
    boot_auroc = []
    
    print(f"Running {n_bootstraps} bootstrap resamples...")
    for _ in tqdm(range(n_bootstraps)):
        # Sample with replacement
        indices = np.random.randint(0, n_samples, n_samples)
        
        y_t = y_true[indices]
        y_p = y_pred[indices]
        y_pr = y_prob[indices] if y_prob is not None else None
        
        # Ensure we have at least one of each class in the sample (to compute AUROC)
        if len(np.unique(y_t)) < 2:
            continue
            
        acc, f1, auroc = compute_metrics(y_t, y_p, y_pr, is_multiclass)
        boot_acc.append(acc)
        boot_f1.append(f1)
        boot_auroc.append(auroc)
        
    alpha = (1 - ci_level) / 2.0
    lower_pct = alpha * 100
    upper_pct = (1 - alpha) * 100
    
    results = {}
    for name, vals in [('Accuracy', boot_acc), ('F1 Macro', boot_f1), ('AUROC', boot_auroc)]:
        clean_vals = [v for v in vals if not np.isnan(v)]
        if clean_vals:
            mean_val = np.mean(clean_vals)
            lower = np.percentile(clean_vals, lower_pct)
            upper = np.percentile(clean_vals, upper_pct)
            results[name] = (mean_val, lower, upper)
            
    return results


def format_ci_latex(results):
    """Format the results into a string ready for a LaTeX table."""
    latex_strs = []
    for name, (mean, lower, upper) in results.items():
        # E.g., 0.988 (0.985 - 0.991)
        latex_strs.append(f"{mean:.3f} ({lower:.3f}-{upper:.3f})")
    return " & ".join(latex_strs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Bootstrap Confidence Intervals")
    parser.add_argument("--predictions_csv", type=str, required=True, 
                        help="CSV containing columns 'true_label', 'pred_label', and optionally 'pred_prob' (or pred_prob_0, pred_prob_1...)")
    parser.add_argument("--n_bootstraps", type=int, default=1000)
    parser.add_argument("--is_multiclass", action="store_true")
    args = parser.parse_args()
    
    print(f"Loading predictions from: {args.predictions_csv}")
    df = pd.read_csv(args.predictions_csv)
    
    y_true = df['true_label'].values
    y_pred = df['pred_label'].values
    
    # Try to extract probabilities
    y_prob = None
    if args.is_multiclass:
        prob_cols = [c for c in df.columns if c.startswith('pred_prob_')]
        if prob_cols:
            y_prob = df[prob_cols].values
    else:
        if 'pred_prob' in df.columns:
            y_prob = df['pred_prob'].values
            
    results = bootstrap_ci(y_true, y_pred, y_prob, args.is_multiclass, args.n_bootstraps)
    
    print("\n" + "="*50)
    print("BOOTSTRAP 95% CONFIDENCE INTERVALS")
    print("="*50)
    for name, (mean, lower, upper) in results.items():
        print(f"{name:10}: {mean:.3f} [{lower:.3f}, {upper:.3f}]")
    print("="*50)
    
    print("\nLaTeX ready format:")
    print(format_ci_latex(results))
