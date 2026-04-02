"""
PA-SSL: Multi-Task Clinical Evaluation

Evaluates frozen SSL representations on diverse clinical tasks beyond arrhythmia 
classification. Each task uses a simple linear probe (frozen encoder + trained head),
demonstrating that the learned representations capture general cardiac physiology.

Tasks:
  1. Age regression   — predict patient age from a single ECG beat (MAE, R²)
  2. Sex classification — predict patient sex (AUROC, accuracy)
  3. ECG age gap       — compute predicted_age - actual_age, correlate with pathology
  4. Retrieval @K      — find semantically similar ECGs by cosine distance (P@K, mAP)

All tasks use PTB-XL metadata already present in ECGBeatDataset (self.age, self.sex).
No new data downloads are required.

Usage:
    python -m src.experiments.multi_task_evaluation \\
        --encoder resnet1d \\
        --checkpoint experiments/ssl_passl_resnet_hybrid/best_checkpoint.pth \\
        --data_file data/ptbxl_processed.csv \\
        --output experiments/multi_task_results.json
"""

import argparse
import json
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
)
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from src.data.ecg_dataset import ECGBeatDataset
from src.evaluate import extract_representations
from src.models.encoder import build_encoder


# ═══════════════════════════════════════════════════════════════════════════════
# 1. AGE REGRESSION PROBE
# ═══════════════════════════════════════════════════════════════════════════════

class AgeRegressionProbe:
    """
    Linear probe: frozen SSL representation → predicted patient age.

    Metrics: MAE (years), R², Pearson r
    Clinical meaning: A low MAE (<7 years) indicates the representation
    captures cardiac aging — a known biomarker of cardiovascular risk.
    """

    def __init__(self):
        self.model = LinearRegression()
        self.is_fitted = False

    def fit(self, reprs: np.ndarray, ages: np.ndarray) -> None:
        """Train on (representations, age labels)."""
        self.model.fit(reprs, ages)
        self.is_fitted = True

    def evaluate(
        self, reprs: np.ndarray, ages: np.ndarray
    ) -> dict:
        assert self.is_fitted, "Probe must be fitted before evaluation."
        pred_ages = self.model.predict(reprs)

        mae = mean_absolute_error(ages, pred_ages)
        r2 = r2_score(ages, pred_ages)
        pearson_r, pearson_p = pearsonr(ages, pred_ages)

        return {
            "task": "age_regression",
            "mae_years": float(mae),
            "r2": float(r2),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "n_samples": len(ages),
        }

    def predict(self, reprs: np.ndarray) -> np.ndarray:
        return self.model.predict(reprs)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SEX CLASSIFICATION PROBE
# ═══════════════════════════════════════════════════════════════════════════════

class SexClassificationProbe:
    """
    Linear probe: frozen SSL representation → predicted patient sex.

    Metrics: AUROC, accuracy
    Clinical meaning: ECG morphology (QRS amplitude, T-wave polarity) differs
    significantly between sexes. A high AUROC validates that representations
    capture these physiological differences.
    """

    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        self.is_fitted = False

    def fit(self, reprs: np.ndarray, sexes: np.ndarray) -> None:
        """Train on (representations, sex labels: 0/1)."""
        # Threshold continuous sex values to binary (0 = female, 1 = male)
        labels = (sexes > 0.5).astype(int)
        self.model.fit(reprs, labels)
        self.is_fitted = True
        self._labels_used = labels

    def evaluate(
        self, reprs: np.ndarray, sexes: np.ndarray
    ) -> dict:
        assert self.is_fitted, "Probe must be fitted before evaluation."
        labels = (sexes > 0.5).astype(int)
        preds = self.model.predict(reprs)
        proba = self.model.predict_proba(reprs)[:, 1]

        acc = accuracy_score(labels, preds)
        try:
            auroc = roc_auc_score(labels, proba)
        except ValueError:
            auroc = float("nan")

        return {
            "task": "sex_classification",
            "accuracy": float(acc),
            "auroc": float(auroc),
            "n_samples": len(labels),
            "n_male": int(labels.sum()),
            "n_female": int((1 - labels).sum()),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ECG AGE GAP ANALYSER
# ═══════════════════════════════════════════════════════════════════════════════

class ECGAgeGapAnalyzer:
    """
    Computes predicted_age - actual_age = "ECG age gap" per patient.

    A positive age gap means the ECG looks older than the patient's
    chronological age — an established marker of elevated cardiovascular risk.

    If ECG age gap correlates with known cardiac labels (Spearman r > 0.15),
    this is a publishable clinical biomarker result.
    """

    def __init__(self, age_probe: AgeRegressionProbe):
        self.probe = age_probe

    def compute_age_gap(
        self,
        reprs: np.ndarray,
        actual_ages: np.ndarray,
    ) -> np.ndarray:
        """
        Returns age gap per sample: predicted_age - actual_age.
        Positive gap → ECG looks older → higher risk.
        """
        predicted_ages = self.probe.predict(reprs)
        return predicted_ages - actual_ages

    def correlate_with_labels(
        self,
        reprs: np.ndarray,
        actual_ages: np.ndarray,
        pathology_labels: np.ndarray,
        label_names: Optional[list] = None,
    ) -> dict:
        """
        For each pathology label, compute Spearman correlation between
        ECG age gap and binary disease presence.

        Args:
            reprs: (N, D) representations
            actual_ages: (N,) chronological ages
            pathology_labels: (N, C) or (N,) binary labels
            label_names: List of length C label names

        Returns:
            dict with Spearman r and p per label
        """
        age_gaps = self.compute_age_gap(reprs, actual_ages)

        if pathology_labels.ndim == 1:
            pathology_labels = pathology_labels[:, np.newaxis]
            label_names = label_names or ["pathology"]

        n_labels = pathology_labels.shape[1]
        label_names = label_names or [f"label_{i}" for i in range(n_labels)]

        results = {"age_gap_mean": float(age_gaps.mean()), "age_gap_std": float(age_gaps.std())}
        correlations = {}

        for i, name in enumerate(label_names):
            col = pathology_labels[:, i]
            if col.sum() > 5:  # Skip extremely rare labels
                r, p = spearmanr(age_gaps, col)
                correlations[name] = {"spearman_r": float(r), "spearman_p": float(p)}

        results["label_correlations"] = correlations
        results["task"] = "ecg_age_gap"
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# 4. EMBEDDING RETRIEVAL EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingRetrievalEvaluator:
    """
    Evaluate the representation space as a retrieval engine.

    For each query ECG, retrieve top-K most similar ECGs by cosine distance.
    A retrieval is considered "correct" if it shares the same rhythm label.

    Metrics: Precision@K (for K=1, 5, 10), mean Average Precision (mAP)
    Clinical meaning: "Find me similar patients" — a clinically useful
    capability that requires no additional training beyond the SSL pretraining.
    """

    def __init__(self, k_values: Tuple[int, ...] = (1, 5, 10)):
        self.k_values = k_values

    def evaluate(
        self,
        reprs: np.ndarray,
        labels: np.ndarray,
    ) -> dict:
        """
        Args:
            reprs: (N, D) normalized representations
            labels: (N,) integer class labels

        Returns:
            dict with P@K and mAP scores
        """
        # L2-normalize for cosine similarity
        norms = np.linalg.norm(reprs, axis=1, keepdims=True) + 1e-8
        reprs_norm = reprs / norms

        # Cosine similarity matrix (N, N)
        sim_matrix = reprs_norm @ reprs_norm.T

        # Mask self-similarity
        np.fill_diagonal(sim_matrix, -np.inf)

        results = {"task": "retrieval", "n_samples": len(labels)}
        prec_at_k = {}

        for k in self.k_values:
            if k >= len(labels):
                continue
            top_k_indices = np.argsort(sim_matrix, axis=1)[:, -k:]  # (N, K) highest sim
            hits = np.array([
                (labels[top_k_indices[i]] == labels[i]).sum() / k
                for i in range(len(labels))
            ])
            prec_at_k[f"P@{k}"] = float(hits.mean())

        results["precision_at_k"] = prec_at_k

        # mAP: for each query, compute AP over its retrieved ranking
        # Exclude self (diagonal was set to -inf, so it sorts last — drop it)
        n = len(labels)
        ap_scores = []
        for i in range(n):
            sorted_idx = np.argsort(sim_matrix[i])[::-1]  # descending
            # Remove self index (it has -inf similarity so it's at the end)
            sorted_idx = sorted_idx[sorted_idx != i]       # now length n-1
            relevances = (labels[sorted_idx] == labels[i]).astype(float)
            if relevances.sum() == 0:
                continue
            cumulative = np.cumsum(relevances)             # length n-1
            positions = np.arange(1, n)                    # length n-1
            ap = (cumulative / positions * relevances).sum() / relevances.sum()
            ap_scores.append(ap)

        results["mAP"] = float(np.mean(ap_scores)) if ap_scores else float("nan")
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_multi_task_evaluation(
    encoder,
    dataset: ECGBeatDataset,
    device: torch.device,
    train_frac: float = 0.7,
    seed: int = 42,
) -> dict:
    """
    Run all 4 clinical tasks and return a combined results dict.

    Args:
        encoder: Pretrained (frozen) PA-SSL encoder
        dataset: ECGBeatDataset with metadata (age, sex, labels)
        device: torch device
        train_frac: Fraction of data used to fit linear probes
        seed: Random seed

    Returns:
        dict with keys: age_regression, sex_classification, ecg_age_gap, retrieval
    """
    print("\n" + "=" * 60)
    print("Multi-Task Clinical Evaluation")
    print("=" * 60)

    encoder.eval()

    # Extract representations
    print("Extracting representations...")
    reprs, labels = extract_representations(encoder, dataset, device)

    # Gather metadata arrays
    ages = dataset.age  # Already float32 arrays from ECGBeatDataset
    sexes = dataset.sex

    # Train/test split (index-based, not patient-aware — probes are shallow)
    rng = np.random.RandomState(seed)
    n = len(reprs)
    idx = rng.permutation(n)
    split = int(n * train_frac)
    train_idx, test_idx = idx[:split], idx[split:]

    train_reprs, test_reprs = reprs[train_idx], reprs[test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]
    train_ages, test_ages = ages[train_idx], ages[test_idx]
    train_sexes, test_sexes = sexes[train_idx], sexes[test_idx]

    all_results = {}

    # ── Task 1: Age Regression ─────────────────────────────────────────────
    print("\n[1/4] Age Regression Probe...")
    age_probe = AgeRegressionProbe()
    age_probe.fit(train_reprs, train_ages)
    age_results = age_probe.evaluate(test_reprs, test_ages)
    all_results["age_regression"] = age_results
    print(f"       MAE: {age_results['mae_years']:.2f} years | "
          f"R²: {age_results['r2']:.3f} | "
          f"Pearson r: {age_results['pearson_r']:.3f}")

    # ── Task 2: Sex Classification ─────────────────────────────────────────
    print("\n[2/4] Sex Classification Probe...")
    sex_probe = SexClassificationProbe()
    sex_probe.fit(train_reprs, train_sexes)
    sex_results = sex_probe.evaluate(test_reprs, test_sexes)
    all_results["sex_classification"] = sex_results
    print(f"       AUROC: {sex_results['auroc']:.3f} | "
          f"Accuracy: {sex_results['accuracy']:.3f}")

    # ── Task 3: ECG Age Gap ────────────────────────────────────────────────
    print("\n[3/4] ECG Age Gap Analysis...")
    gap_analyzer = ECGAgeGapAnalyzer(age_probe)
    gap_results = gap_analyzer.correlate_with_labels(
        test_reprs, test_ages, test_labels.astype(float).reshape(-1, 1),
        label_names=["arrhythmia"]
    )
    all_results["ecg_age_gap"] = gap_results
    mean_gap = gap_results["age_gap_mean"]
    std_gap = gap_results["age_gap_std"]
    print(f"       Age gap: {mean_gap:+.2f} ± {std_gap:.2f} years")
    for label, corr in gap_results.get("label_correlations", {}).items():
        print(f"       Corr with {label}: r={corr['spearman_r']:.3f} (p={corr['spearman_p']:.4f})")

    # ── Task 4: Retrieval @K ──────────────────────────────────────────────
    print("\n[4/4] Embedding Retrieval (P@1, P@5, P@10, mAP)...")
    evaluator = EmbeddingRetrievalEvaluator(k_values=(1, 5, 10))
    retrieval_results = evaluator.evaluate(test_reprs, test_labels)
    all_results["retrieval"] = retrieval_results
    for k_label, v in retrieval_results.get("precision_at_k", {}).items():
        print(f"       {k_label}: {v:.3f}")
    print(f"       mAP: {retrieval_results.get('mAP', float('nan')):.3f}")

    print("\n" + "=" * 60)
    print("Multi-Task Evaluation Complete")
    print("=" * 60)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Multi-task clinical evaluation of PA-SSL representations"
    )
    parser.add_argument("--encoder", type=str, default="resnet1d",
                        choices=["resnet1d", "wavkan", "transformer", "mamba"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save JSON results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load encoder
    encoder = build_encoder(args.encoder, proj_dim=args.proj_dim)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_key = "encoder_state_dict" if "encoder_state_dict" in ckpt else None
    encoder.load_state_dict(ckpt[state_key] if state_key else ckpt)
    encoder = encoder.to(device)
    encoder.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Load dataset
    dataset = ECGBeatDataset(args.data_file)

    # Run evaluation
    results = run_multi_task_evaluation(
        encoder, dataset, device,
        train_frac=args.train_frac,
        seed=args.seed,
    )

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")

    return results


if __name__ == "__main__":
    main()
