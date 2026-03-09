"""
PA-SSL: Anomaly Scorer with Calibration Metrics

Implements Mahalanobis-distance anomaly scoring in the learned
representation space, with Expected Calibration Error (ECE) computation.

The Mahalanobis scorer fits class-conditional Gaussians on labeled
support data, then scores test samples by their distance to the
nearest class prototype — providing well-calibrated uncertainty estimates.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from collections import defaultdict


class MahalanobisAnomalyScorer:
    """
    Anomaly scoring via Mahalanobis distance in representation space.
    
    Steps:
        1. Extract representations from a labeled support set
        2. Fit class-conditional Gaussians (mean + shared covariance)
        3. For each test sample, compute Mahalanobis distance to each class
        4. Anomaly score = negative minimum Mahalanobis distance
    """
    
    def __init__(self, use_shrinkage=True):
        """
        Args:
            use_shrinkage: If True, uses Ledoit-Wolf shrinkage for covariance
                          estimation (more stable with small samples)
        """
        self.use_shrinkage = use_shrinkage
        self.class_means = {}
        self.precision_matrix = None  # Shared precision (inverse covariance)
        self.fitted = False
    
    def fit(self, representations, labels):
        """
        Fit class-conditional Gaussians.
        
        Args:
            representations: (N, D) numpy array of encoder outputs
            labels: (N,) array of class labels
        """
        unique_labels = np.unique(labels)
        
        # Compute class means
        for lbl in unique_labels:
            mask = labels == lbl
            self.class_means[lbl] = representations[mask].mean(axis=0)
        
        # Compute shared covariance (centered)
        centered = np.zeros_like(representations)
        for i, lbl in enumerate(labels):
            centered[i] = representations[i] - self.class_means[lbl]
        
        # Fit covariance estimator
        if self.use_shrinkage:
            cov_estimator = LedoitWolf()
        else:
            cov_estimator = EmpiricalCovariance()
        
        cov_estimator.fit(centered)
        self.precision_matrix = cov_estimator.precision_
        self.fitted = True
    
    def score(self, representations):
        """
        Compute anomaly scores for test representations.
        
        Args:
            representations: (N, D) numpy array
        
        Returns:
            scores: (N,) array — higher score = more anomalous
            min_distances: (N,) array — Mahalanobis distance to nearest class
            predicted_classes: (N,) array — predicted class (nearest prototype)
        """
        assert self.fitted, "Must call fit() first"
        
        N = len(representations)
        n_classes = len(self.class_means)
        
        # Distance to each class
        distances = np.zeros((N, n_classes))
        class_labels = sorted(self.class_means.keys())
        
        for j, lbl in enumerate(class_labels):
            diff = representations - self.class_means[lbl]  # (N, D)
            # Mahalanobis: sqrt(diff @ precision @ diff.T)
            left = diff @ self.precision_matrix  # (N, D)
            distances[:, j] = np.sqrt(np.sum(left * diff, axis=1))
        
        min_distances = distances.min(axis=1)
        predicted_classes = np.array([class_labels[j] for j in distances.argmin(axis=1)])
        
        # Anomaly score: higher = more anomalous
        scores = min_distances  
        
        return scores, min_distances, predicted_classes
    
    def predict_proba(self, representations):
        """
        Convert distances to calibrated probabilities via softmax.
        
        Returns:
            probs: (N, n_classes) array of predicted probabilities
        """
        assert self.fitted, "Must call fit() first"
        
        N = len(representations)
        class_labels = sorted(self.class_means.keys())
        n_classes = len(class_labels)
        
        # Negative distances (closer = higher probability)
        neg_distances = np.zeros((N, n_classes))
        for j, lbl in enumerate(class_labels):
            diff = representations - self.class_means[lbl]
            left = diff @ self.precision_matrix
            neg_distances[:, j] = -np.sqrt(np.sum(left * diff, axis=1))
        
        # Softmax
        exp_d = np.exp(neg_distances - neg_distances.max(axis=1, keepdims=True))
        probs = exp_d / exp_d.sum(axis=1, keepdims=True)
        
        return probs


# ═══════════════════════════════════════════════════════════════════════════════
# CALIBRATION METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def expected_calibration_error(y_true, y_proba, n_bins=15):
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures how well predicted probabilities match observed frequencies.
    Lower ECE = better calibrated model.
    
    Args:
        y_true: (N,) true labels
        y_proba: (N, C) predicted probabilities
        n_bins: Number of confidence bins
    
    Returns:
        ece: scalar ECE value
        bin_data: list of (bin_center, accuracy, confidence, count) tuples
    """
    predicted_classes = np.argmax(y_proba, axis=1)
    confidences = np.max(y_proba, axis=1)
    accuracies = (predicted_classes == y_true).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []
    
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        count = mask.sum()
        
        if count > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            bin_center = (bin_boundaries[i] + bin_boundaries[i + 1]) / 2
            ece += (count / len(y_true)) * abs(bin_acc - bin_conf)
            bin_data.append((bin_center, bin_acc, bin_conf, count))
    
    return ece, bin_data


def brier_score(y_true, y_proba):
    """
    Compute Brier score (mean squared error of probability estimates).
    
    Args:
        y_true: (N,) true labels
        y_proba: (N, C) predicted probabilities
    
    Returns:
        Brier score (lower is better)
    """
    n_classes = y_proba.shape[1]
    y_onehot = np.eye(n_classes)[y_true]
    return np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1))


def reliability_diagram_data(y_true, y_proba, n_bins=15):
    """
    Compute data needed for reliability diagram plotting.
    
    Returns:
        dict with 'bin_centers', 'bin_accuracies', 'bin_confidences', 'bin_counts'
    """
    _, bin_data = expected_calibration_error(y_true, y_proba, n_bins)
    
    return {
        'bin_centers': [b[0] for b in bin_data],
        'bin_accuracies': [b[1] for b in bin_data],
        'bin_confidences': [b[2] for b in bin_data],
        'bin_counts': [b[3] for b in bin_data],
    }
