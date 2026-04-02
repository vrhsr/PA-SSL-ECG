"""
PA-SSL: Monte Carlo Dropout Uncertainty Quantification

Wraps any PA-SSL encoder to produce calibrated uncertainty estimates via
MC-Dropout (Gal & Ghahramani, 2016). At inference, dropout is kept ON and
T forward passes are averaged.

Key results enabled:
  - Coverage-accuracy trade-off curve (selective prediction)
  - Automatic flagging of uncertain ECGs for cardiologist review
  - Improved OOD detection via predictive variance

Usage:
    from src.models.uncertainty import MCDropoutPredictor
    predictor = MCDropoutPredictor(encoder, num_classes=5, T=30, dropout_rate=0.1)
    predictor.fit_classifier(train_reprs, train_labels)
    mean_pred, variance = predictor.predict_with_uncertainty(x_test)
    report = predictor.selective_prediction(x_test, labels_test, threshold_percentile=80)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


# ═══════════════════════════════════════════════════════════════════════════════
# MC-DROPOUT PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════

class MCDropoutPredictor:
    """
    Monte Carlo Dropout uncertainty wrapper for PA-SSL encoders.

    Procedure:
      1. `fit_classifier(train_reprs, train_labels)` trains a logistic regression
         head on frozen representations (no neural training needed).
      2. At inference, `predict_with_uncertainty(x)` runs T stochastic forward
         passes through the encoder with dropout active, averaging softmax outputs
         and reporting predictive variance.

    Parameters
    ----------
    encoder : nn.Module
        Pretrained PA-SSL encoder (ResNet1D, WavKAN, Transformer, Mamba).
    num_classes : int
        Number of downstream classes.
    T : int
        Number of Monte Carlo samples (default 30).
    dropout_rate : float
        Dropout probability injected at each forward pass (default 0.1).
    device : torch.device or None
        If None, auto-detected from encoder parameters.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        T: int = 30,
        dropout_rate: float = 0.1,
        device: torch.device | None = None,
    ):
        self.encoder = encoder
        self.num_classes = num_classes
        self.T = T
        self.dropout_rate = dropout_rate
        self.device = device or next(encoder.parameters()).device

        # Inject dropout into encoder (adds to existing dropout layers or hooks)
        self._dropout = nn.Dropout(p=dropout_rate)

        # Logistic regression head (trained on frozen representations)
        self.clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                       multi_class="auto")
        self._clf_fitted = False

    def fit_classifier(
        self,
        train_reprs: np.ndarray,
        train_labels: np.ndarray,
    ) -> "MCDropoutPredictor":
        """Train the linear classifier on extracted representations."""
        self.clf.fit(train_reprs, train_labels)
        self._clf_fitted = True
        return self

    def _encode_with_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Single stochastic forward pass with dropout applied to features."""
        self.encoder.train()  # Keeps dropout ON
        with torch.no_grad():
            h = self.encoder.encode(x)          # (B, D)
            h = self._dropout(h)                # Apply MC dropout
        return h

    def predict_with_uncertainty(
        self,
        x: torch.Tensor | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run T stochastic forward passes and aggregate.

        Args:
            x: (B, 1, L) tensor or (B, L) numpy array

        Returns:
            mean_proba: (B, C) mean predicted probabilities
            variance:   (B, C) epistemic uncertainty (variance across T passes)
        """
        assert self._clf_fitted, "Call fit_classifier() before predict_with_uncertainty()"

        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.to(self.device)

        all_probas = []
        for _ in range(self.T):
            reprs = self._encode_with_dropout(x).cpu().numpy()  # (B, D)
            proba = self.clf.predict_proba(reprs)               # (B, C)
            all_probas.append(proba)

        all_probas = np.stack(all_probas, axis=0)  # (T, B, C)
        mean_proba = all_probas.mean(axis=0)        # (B, C)
        variance = all_probas.var(axis=0)           # (B, C) epistemic uncertainty

        return mean_proba, variance

    def selective_prediction(
        self,
        x: torch.Tensor | np.ndarray,
        true_labels: np.ndarray,
        threshold_percentile: float = 80,
    ) -> dict:
        """
        Report accuracy at different coverage levels by rejecting
        high-uncertainty samples.

        Args:
            x: Input ECG signals (B, 1, L) or (B, L)
            true_labels: (B,) ground-truth class labels
            threshold_percentile: Samples with uncertainty ABOVE this percentile
                                   are flagged for review (not predicted).

        Returns:
            dict with:
                - coverage_accuracy: list of (coverage, accuracy) tuples
                - flagged_fraction: fraction flagged at given percentile
                - confident_accuracy: accuracy on confident subset
                - full_accuracy: accuracy using all predictions (no rejection)
                - mean_uncertainty: mean epistemic uncertainty across all samples
        """
        assert self._clf_fitted, "Call fit_classifier() before selective_prediction()"

        mean_proba, variance = self.predict_with_uncertainty(x)
        pred_labels = mean_proba.argmax(axis=1)

        # Scalar uncertainty per sample = mean variance across classes
        uncertainty_per_sample = variance.mean(axis=1)  # (B,)

        # Full accuracy (no rejection)
        full_acc = accuracy_score(true_labels, pred_labels)

        # Coverage-accuracy curve at different rejection thresholds
        thresholds = np.percentile(uncertainty_per_sample, np.arange(0, 100, 5))
        coverage_accuracy = []
        for thresh in thresholds:
            confident_mask = uncertainty_per_sample <= thresh
            if confident_mask.sum() == 0:
                continue
            cov = confident_mask.mean()
            acc = accuracy_score(true_labels[confident_mask], pred_labels[confident_mask])
            coverage_accuracy.append((float(cov), float(acc)))

        # Selective accuracy at the given percentile
        sel_thresh = np.percentile(uncertainty_per_sample, threshold_percentile)
        confident_mask = uncertainty_per_sample <= sel_thresh
        flagged_fraction = float((~confident_mask).mean())

        if confident_mask.sum() > 0:
            confident_acc = accuracy_score(
                true_labels[confident_mask], pred_labels[confident_mask]
            )
        else:
            confident_acc = float("nan")

        return {
            "full_accuracy": float(full_acc),
            "confident_accuracy": float(confident_acc),
            "flagged_fraction": flagged_fraction,
            "coverage": float(confident_mask.mean()),
            "threshold_percentile": threshold_percentile,
            "coverage_accuracy_curve": coverage_accuracy,
            "mean_uncertainty": float(uncertainty_per_sample.mean()),
            "n_samples": len(true_labels),
            "n_flagged": int((~confident_mask).sum()),
        }

    def evaluate_ood_detection(
        self,
        in_dist_x: torch.Tensor | np.ndarray,
        ood_x: torch.Tensor | np.ndarray,
    ) -> dict:
        """
        Detect OOD samples using predictive uncertainty.

        In-distribution samples should have LOW uncertainty.
        OOD samples should have HIGH uncertainty.

        Returns AUROC of uncertainty as an OOD detector.
        """
        _, in_var = self.predict_with_uncertainty(in_dist_x)
        _, ood_var = self.predict_with_uncertainty(ood_x)

        in_scores = in_var.mean(axis=1)    # (N_in,) — lower is in-dist
        ood_scores = ood_var.mean(axis=1)  # (N_ood,)

        all_scores = np.concatenate([in_scores, ood_scores])
        all_labels = np.concatenate([
            np.zeros(len(in_scores)),   # in-distribution = 0
            np.ones(len(ood_scores))    # OOD = 1
        ])

        auroc = roc_auc_score(all_labels, all_scores)

        return {
            "auroc_ood": float(auroc),
            "in_dist_uncertainty_mean": float(in_scores.mean()),
            "ood_uncertainty_mean": float(ood_scores.mean()),
            "n_in": len(in_scores),
            "n_ood": len(ood_scores),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DEEP ENSEMBLE (composes multiple MCDropoutPredictors or independent models)
# ═══════════════════════════════════════════════════════════════════════════════

class DeepEnsembleUncertainty:
    """
    Lightweight deep ensemble: combine predictions from M independently
    trained encoders for stronger uncertainty estimates.

    This is complementary to MC-Dropout: ensembles capture disagreement
    between independently trained models (epistemic uncertainty from 
    different weight space modes), while MC-Dropout captures sensitivity
    to data dropout within a single model.

    Usage:
        ensemble = DeepEnsembleUncertainty(list_of_fitted_mcdropout_predictors)
        mean_p, variance = ensemble.predict(x)
    """

    def __init__(self, predictors: list[MCDropoutPredictor]):
        assert len(predictors) >= 2, "Need at least 2 predictors for an ensemble"
        self.predictors = predictors

    def predict(
        self,
        x: torch.Tensor | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Average predictions from all ensemble members.

        Returns:
            mean_proba: (B, C) ensemble mean probability
            variance:   (B, C) ensemble variance (epistemic uncertainty)
        """
        all_preds = []
        for predictor in self.predictors:
            mean_p, _ = predictor.predict_with_uncertainty(x)
            all_preds.append(mean_p)

        all_preds = np.stack(all_preds, axis=0)  # (M, B, C)
        return all_preds.mean(axis=0), all_preds.var(axis=0)
