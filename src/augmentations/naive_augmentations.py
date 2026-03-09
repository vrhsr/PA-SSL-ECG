"""
PA-SSL: Naive (Unconstrained) Augmentations Baseline

These are standard augmentations used in prior ECG SSL work (including WavKAN-CL).
They do NOT respect physiology — they can destroy QRS morphology, create
impossible heart rates, etc.

This module exists purely as a BASELINE for ablation studies to show that
physiology-aware augmentations outperform naive ones.
"""

import numpy as np
import torch


def naive_gaussian_noise(signal, noise_std_range=(0.01, 0.1)):
    """Add Gaussian noise (may corrupt QRS)."""
    std = np.random.uniform(*noise_std_range)
    noise = np.random.normal(0, std * np.std(signal), len(signal))
    return (signal + noise).astype(np.float32)


def naive_amplitude_scale(signal, scale_range=(0.5, 1.5)):
    """Uniform amplitude scaling (no QRS protection)."""
    scale = np.random.uniform(*scale_range)
    return (signal * scale).astype(np.float32)


def naive_random_masking(signal, mask_fraction=0.15):
    """Zero-mask a random portion of the signal (may destroy QRS)."""
    result = signal.copy()
    mask_len = int(len(signal) * mask_fraction)
    start = np.random.randint(0, len(signal) - mask_len)
    result[start:start + mask_len] = 0.0
    return result.astype(np.float32)


def naive_time_shift(signal, max_shift=25):
    """Circular shift (breaks P-QRS-T alignment)."""
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(signal, shift).astype(np.float32)


def naive_time_reverse(signal):
    """Full signal reversal (physiologically impossible)."""
    return signal[::-1].copy().astype(np.float32)


def naive_cutout(signal, n_cuts=2, max_cut_len=20):
    """Random cutout (may cut QRS)."""
    result = signal.copy()
    for _ in range(n_cuts):
        cut_len = np.random.randint(5, max_cut_len)
        start = np.random.randint(0, max(1, len(signal) - cut_len))
        result[start:start + cut_len] = 0.0
    return result.astype(np.float32)


class NaiveAugPipeline:
    """
    Baseline augmentation pipeline using naive, unconstrained augmentations.
    Used for ablation comparison against PhysioAugPipeline.
    """
    
    def __init__(self, p=0.5):
        """
        Args:
            p: Probability of applying each augmentation
        """
        self.p = p
        self.augmentations = [
            naive_gaussian_noise,
            naive_amplitude_scale,
            naive_random_masking,
            naive_time_shift,
        ]
    
    def __call__(self, signal, r_peak_pos=None):
        """
        Apply random augmentations (r_peak_pos is ignored — that's the point).
        
        Returns:
            Augmented signal as numpy array (250,)
        """
        result = signal.copy()
        
        for aug_fn in self.augmentations:
            if np.random.rand() < self.p:
                result = aug_fn(result)
        
        return result
