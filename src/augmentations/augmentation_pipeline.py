"""
PA-SSL: Composable Augmentation Pipeline

Provides a configurable pipeline that chains physiology-aware augmentations
with independent per-augmentation probabilities. Guarantees that the
output is always a valid 250-sample ECG beat.
"""

import numpy as np
import torch
from typing import Optional, List, Tuple, Callable

from src.augmentations.physio_augmentations import (
    constrained_time_warp,
    amplitude_perturbation,
    baseline_wander,
    emg_noise_injection,
    heart_rate_resample,
    powerline_interference,
    segment_dropout,
    wavelet_masking,
)


class PhysioAugPipeline:
    """
    Physiology-aware augmentation pipeline for ECG SSL.
    
    Each augmentation is applied independently with its own probability.
    All augmentations receive the R-peak position to enable QRS protection.
    
    Usage:
        pipeline = PhysioAugPipeline.default()
        augmented = pipeline(signal, r_peak_pos=125)
    """
    
    def __init__(self, augmentations: List[Tuple[Callable, dict, float]]):
        """
        Args:
            augmentations: List of (function, kwargs, probability) tuples.
                function: augmentation callable(signal, r_peak_pos, **kwargs)
                kwargs: additional keyword arguments
                probability: probability of applying this augmentation
        """
        self.augmentations = augmentations
    
    @classmethod
    def default(cls, strength='medium', exclude=None, only=None):
        """
        Create default physiology-aware pipeline.
        
        Args:
            strength: 'light', 'medium', or 'strong'
            exclude: list of function names to exclude (leave-one-out)
            only: list of function names to exclusively use (leave-one-in)
        """
        configs = {
            'light': {
                'warp_p': 0.3, 'amp_p': 0.3, 'bw_p': 0.3,
                'emg_p': 0.2, 'hr_p': 0.2, 'pl_p': 0.2, 'drop_p': 0.1,
                'warp_max': 0.08, 'amp_range': (0.9, 1.1),
                'bw_amp': 0.08, 'snr_range': (20, 35),
                'hr_range': (0.92, 1.08),
            },
            'medium': {
                'warp_p': 0.5, 'amp_p': 0.5, 'bw_p': 0.4,
                'emg_p': 0.3, 'hr_p': 0.3, 'pl_p': 0.3, 'drop_p': 0.2,
                'warp_max': 0.15, 'amp_range': (0.8, 1.2),
                'bw_amp': 0.15, 'snr_range': (15, 30),
                'hr_range': (0.85, 1.15),
            },
            'strong': {
                'warp_p': 0.7, 'amp_p': 0.7, 'bw_p': 0.5,
                'emg_p': 0.5, 'hr_p': 0.5, 'pl_p': 0.4, 'drop_p': 0.3,
                'warp_max': 0.20, 'amp_range': (0.7, 1.3),
                'bw_amp': 0.20, 'snr_range': (10, 25),
                'hr_range': (0.80, 1.20),
            },
        }
        
        cfg = configs.get(strength, configs['medium'])
        
        augmentations = [
            (constrained_time_warp, {'max_warp': cfg['warp_max']}, cfg['warp_p']),
            (amplitude_perturbation, {'scale_range': cfg['amp_range'], 'qrs_protect': True}, cfg['amp_p']),
            (baseline_wander, {'max_amplitude': cfg['bw_amp']}, cfg['bw_p']),
            (emg_noise_injection, {'snr_range': cfg['snr_range']}, cfg['emg_p']),
            (heart_rate_resample, {'rate_factor_range': cfg['hr_range']}, cfg['hr_p']),
            (powerline_interference, {}, cfg['pl_p']),
            (segment_dropout, {}, cfg['drop_p']),
            (wavelet_masking, {'max_mask_ratio': 0.3}, cfg['drop_p']), # Tied to drop probability
        ]
        
        if exclude:
            augmentations = [a for a in augmentations if a[0].__name__ not in exclude]
        if only:
            augmentations = [a for a in augmentations if a[0].__name__ in only]
            
        return cls(augmentations)
    
    def __call__(self, signal: np.ndarray, r_peak_pos: int = 125) -> np.ndarray:
        """
        Apply augmentation pipeline.
        
        Args:
            signal: (250,) numpy array — raw ECG beat
            r_peak_pos: R-peak sample index within the beat
        
        Returns:
            Augmented signal as numpy array (250,)
        """
        result = signal.copy()
        
        for aug_fn, kwargs, prob in self.augmentations:
            if np.random.rand() < prob:
                try:
                    result = aug_fn(result, r_peak_pos=r_peak_pos, **kwargs)
                except TypeError:
                    # Some augmentations don't take r_peak_pos
                    try:
                        result = aug_fn(result, **kwargs)
                    except Exception:
                        pass  # Skip if augmentation fails
                except Exception:
                    pass  # Skip and keep previous result
        
        # Safety checks
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure correct length
        if len(result) != len(signal):
            from scipy.signal import resample as sci_resample
            result = sci_resample(result, len(signal))
        
        return result.astype(np.float32)
    
    def __repr__(self):
        lines = ["PhysioAugPipeline:"]
        for aug_fn, kwargs, prob in self.augmentations:
            lines.append(f"  {aug_fn.__name__}: p={prob:.2f}, {kwargs}")
        return "\n".join(lines)
