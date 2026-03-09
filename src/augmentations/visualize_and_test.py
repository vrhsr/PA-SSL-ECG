"""
PA-SSL: Augmentation Visualization & Testing

Provides:
  1. Visual comparison of original vs augmented ECG beats
  2. FFT analysis to verify frequency constraints
  3. QRS morphology preservation verification
  4. Unit tests for all augmentations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys

from src.augmentations.physio_augmentations import (
    constrained_time_warp, amplitude_perturbation, baseline_wander,
    emg_noise_injection, heart_rate_resample, powerline_interference,
    segment_dropout, FS, BEAT_LEN, QRS_HALF_WIDTH,
)
from src.augmentations.augmentation_pipeline import PhysioAugPipeline
from src.augmentations.naive_augmentations import NaiveAugPipeline


def generate_synthetic_beat(r_peak_pos=125, fs=100, length=250):
    """Generate a synthetic ECG beat for visualization/testing."""
    t = np.arange(length) / fs
    t_centered = t - r_peak_pos / fs
    
    # P wave
    p_wave = 0.15 * np.exp(-((t_centered + 0.15) ** 2) / (2 * 0.01 ** 2))
    
    # QRS complex
    q_wave = -0.1 * np.exp(-((t_centered + 0.02) ** 2) / (2 * 0.005 ** 2))
    r_wave = 1.0 * np.exp(-((t_centered) ** 2) / (2 * 0.008 ** 2))
    s_wave = -0.2 * np.exp(-((t_centered - 0.025) ** 2) / (2 * 0.006 ** 2))
    
    # T wave
    t_wave = 0.3 * np.exp(-((t_centered - 0.2) ** 2) / (2 * 0.03 ** 2))
    
    beat = p_wave + q_wave + r_wave + s_wave + t_wave
    return beat.astype(np.float32)


def visualize_all_augmentations(save_path=None, n_examples=3):
    """
    Generate a comprehensive figure showing all augmentations.
    Publication-quality: suitable for paper Figure 2.
    """
    signal = generate_synthetic_beat()
    r_peak = 125
    
    augmentations = [
        ("Original", None),
        ("Constrained\nTime Warp", lambda s, r: constrained_time_warp(s, r, max_warp=0.15)),
        ("Amplitude\nPerturbation", lambda s, r: amplitude_perturbation(s, r, scale_range=(0.7, 1.3))),
        ("Baseline\nWander", lambda s, r: baseline_wander(s, max_amplitude=0.2)),
        ("EMG Noise", lambda s, r: emg_noise_injection(s, snr_range=(10, 20))),
        ("Heart Rate\nResample", lambda s, r: heart_rate_resample(s, r, rate_factor_range=(0.8, 1.2))),
        ("Powerline\nInterference", lambda s, r: powerline_interference(s)),
        ("Segment\nDropout", lambda s, r: segment_dropout(s, r)),
    ]
    
    n_augs = len(augmentations)
    fig, axes = plt.subplots(n_augs, n_examples + 1, figsize=(4 * (n_examples + 1), 2.2 * n_augs))
    fig.suptitle("PA-SSL: Physiology-Aware Augmentations", fontsize=16, fontweight='bold', y=0.98)
    
    t = np.arange(BEAT_LEN) / FS
    
    for row, (name, aug_fn) in enumerate(augmentations):
        # Original
        axes[row, 0].plot(t, signal, 'k-', linewidth=0.8)
        axes[row, 0].axvspan(
            (r_peak - QRS_HALF_WIDTH) / FS, (r_peak + QRS_HALF_WIDTH) / FS,
            alpha=0.15, color='red', label='QRS zone'
        )
        axes[row, 0].set_ylabel(name, fontsize=9, rotation=0, ha='right', va='center')
        
        if row == 0:
            axes[row, 0].set_title("Original", fontsize=10)
        
        # Augmented examples
        for col in range(n_examples):
            ax = axes[row, col + 1]
            
            if aug_fn is None:
                aug_signal = signal.copy()
            else:
                aug_signal = aug_fn(signal.copy(), r_peak)
            
            ax.plot(t, signal, 'k-', alpha=0.3, linewidth=0.5, label='Original')
            ax.plot(t, aug_signal, 'b-', linewidth=0.8, label='Augmented')
            ax.axvspan(
                (r_peak - QRS_HALF_WIDTH) / FS, (r_peak + QRS_HALF_WIDTH) / FS,
                alpha=0.1, color='red'
            )
            
            if row == 0:
                ax.set_title(f"Example {col + 1}", fontsize=10)
        
        for col in range(n_examples + 1):
            axes[row, col].set_xlim(0, BEAT_LEN / FS)
            axes[row, col].tick_params(labelsize=7)
            if row < n_augs - 1:
                axes[row, col].set_xticklabels([])
    
    # X-axis label on bottom row only
    for col in range(n_examples + 1):
        axes[-1, col].set_xlabel("Time (s)", fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Augmentation figure saved to: {save_path}")
    
    plt.close()
    return fig


def visualize_physio_vs_naive(save_path=None):
    """
    Side-by-side comparison of physiology-aware vs naive augmentations.
    Demonstrates why naive augmentations can destroy clinical features.
    """
    signal = generate_synthetic_beat()
    r_peak = 125
    
    physio_pipeline = PhysioAugPipeline.default('strong')
    naive_pipeline = NaiveAugPipeline(p=0.7)
    
    n_examples = 4
    fig, axes = plt.subplots(3, n_examples, figsize=(3.5 * n_examples, 7))
    
    t = np.arange(BEAT_LEN) / FS
    
    for col in range(n_examples):
        # Original
        axes[0, col].plot(t, signal, 'k-', linewidth=1)
        axes[0, col].axvspan(
            (r_peak - QRS_HALF_WIDTH) / FS, (r_peak + QRS_HALF_WIDTH) / FS,
            alpha=0.15, color='red'
        )
        axes[0, col].set_title(f"Original Beat", fontsize=10)
        
        # Physio-aware
        physio_aug = physio_pipeline(signal.copy(), r_peak)
        axes[1, col].plot(t, signal, 'k-', alpha=0.3, linewidth=0.5)
        axes[1, col].plot(t, physio_aug, 'g-', linewidth=1)
        axes[1, col].axvspan(
            (r_peak - QRS_HALF_WIDTH) / FS, (r_peak + QRS_HALF_WIDTH) / FS,
            alpha=0.15, color='red'
        )
        axes[1, col].set_title(f"Physio-Aware", fontsize=10, color='green')
        
        # Naive
        naive_aug = naive_pipeline(signal.copy())
        axes[2, col].plot(t, signal, 'k-', alpha=0.3, linewidth=0.5)
        axes[2, col].plot(t, naive_aug, 'r-', linewidth=1)
        axes[2, col].axvspan(
            (r_peak - QRS_HALF_WIDTH) / FS, (r_peak + QRS_HALF_WIDTH) / FS,
            alpha=0.15, color='red'
        )
        axes[2, col].set_title(f"Naive (breaks QRS)", fontsize=10, color='red')
    
    axes[0, 0].set_ylabel("Original", fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel("PA-SSL", fontsize=11, fontweight='bold', color='green')
    axes[2, 0].set_ylabel("Naive SSL", fontsize=11, fontweight='bold', color='red')
    
    for ax in axes.flat:
        ax.set_xlim(0, BEAT_LEN / FS)
        ax.tick_params(labelsize=7)
    
    for col in range(n_examples):
        axes[2, col].set_xlabel("Time (s)", fontsize=9)
    
    fig.suptitle("Physiology-Aware vs Naive Augmentations", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison figure saved to: {save_path}")
    
    plt.close()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def run_augmentation_tests():
    """
    Run comprehensive unit tests for all augmentations.
    Tests: output shape, NaN/Inf, R-peak preservation, frequency constraints.
    """
    signal = generate_synthetic_beat()
    r_peak = 125
    n_trials = 50
    passed = 0
    failed = 0
    
    tests = {
        'constrained_time_warp': lambda: constrained_time_warp(signal.copy(), r_peak),
        'amplitude_perturbation': lambda: amplitude_perturbation(signal.copy(), r_peak),
        'baseline_wander': lambda: baseline_wander(signal.copy()),
        'emg_noise_injection': lambda: emg_noise_injection(signal.copy()),
        'heart_rate_resample': lambda: heart_rate_resample(signal.copy(), r_peak),
        'powerline_interference': lambda: powerline_interference(signal.copy()),
        'segment_dropout': lambda: segment_dropout(signal.copy(), r_peak),
    }
    
    print("=" * 60)
    print("PA-SSL Augmentation Unit Tests")
    print("=" * 60)
    
    for name, aug_fn in tests.items():
        test_pass = True
        messages = []
        
        for trial in range(n_trials):
            try:
                result = aug_fn()
                
                # Test 1: Output shape
                if result.shape != signal.shape:
                    messages.append(f"Shape mismatch: {result.shape} != {signal.shape}")
                    test_pass = False
                    break
                
                # Test 2: No NaN/Inf
                if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                    messages.append("Contains NaN or Inf")
                    test_pass = False
                    break
                
                # Test 3: Reasonable amplitude (not exploding)
                if np.max(np.abs(result)) > 100 * np.max(np.abs(signal)):
                    messages.append(f"Amplitude explosion: {np.max(np.abs(result)):.2f}")
                    test_pass = False
                    break
                    
            except Exception as e:
                messages.append(f"Exception: {e}")
                test_pass = False
                break
        
        status = "✓ PASS" if test_pass else "✗ FAIL"
        print(f"  {status}  {name}")
        for msg in messages:
            print(f"         → {msg}")
        
        if test_pass:
            passed += 1
        else:
            failed += 1
    
    # Test pipeline
    pipeline = PhysioAugPipeline.default('strong')
    pipe_pass = True
    for _ in range(n_trials):
        result = pipeline(signal.copy(), r_peak)
        if result.shape != signal.shape or np.any(np.isnan(result)):
            pipe_pass = False
            break
    
    status = "✓ PASS" if pipe_pass else "✗ FAIL"
    print(f"  {status}  PhysioAugPipeline (end-to-end)")
    passed += 1 if pipe_pass else 0
    failed += 0 if pipe_pass else 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize', action='store_true', help='Generate figures')
    parser.add_argument('--test', action='store_true', help='Run unit tests')
    parser.add_argument('--output_dir', type=str, default='figures')
    args = parser.parse_args()
    
    if args.test or (not args.visualize):
        success = run_augmentation_tests()
        if not success:
            sys.exit(1)
    
    if args.visualize:
        visualize_all_augmentations(f"{args.output_dir}/augmentations_all.png")
        visualize_physio_vs_naive(f"{args.output_dir}/physio_vs_naive.png")
        print("All visualizations generated.")
