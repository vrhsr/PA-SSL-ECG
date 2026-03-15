"""
PA-SSL: Morphology Preservation Metrics Diagnostic

Quantifies how well augmentations preserve critical ECG morphology:
1. QRS Correlation (Pearson correlation of central QRS zone)
2. PR Interval Stability (estimated via peak-to-peak cross-correlation)
3. RR Interval Preservation (detecting change in peak timing)
4. Overall Mean Squared Error (MSE) within the QRS zone

Compares Physiology-Aware (PA) augmentations vs Naive augmentations.
Goal: Demonstrate that PA-SSL augmentations preserve clinical features significantly better.
"""

import numpy as np
import pandas as pd
import argparse
from scipy.stats import pearsonr
from src.augmentations.augmentation_pipeline import PhysioAugPipeline
from src.augmentations.naive_augmentations import NaiveAugPipeline
from src.data.ecg_dataset import ECGBeatDataset

def calculate_metrics(original, augmented, qrs_radius=30):
    """
    Computes a set of morphology preservation metrics for a single beat.
    """
    L = len(original)
    center = L // 2
    
    # 1. QRS Zone (Central peak)
    orig_qrs = original[center - qrs_radius : center + qrs_radius]
    aug_qrs = augmented[center - qrs_radius : center + qrs_radius]
    
    # QRS Pearson Correlation
    if np.std(orig_qrs) < 1e-8 or np.std(aug_qrs) < 1e-8:
        qrs_corr = 0.0
    else:
        qrs_corr, _ = pearsonr(orig_qrs, aug_qrs)
    
    # QRS MSE
    qrs_mse = np.mean((orig_qrs - aug_qrs)**2)
    
    # 2. Peak Timing Error (R-peak shift)
    orig_r_idx = np.argmax(np.abs(orig_qrs)) + (center - qrs_radius)
    aug_r_idx = np.argmax(np.abs(aug_qrs)) + (center - qrs_radius)
    r_peak_shift = abs(orig_r_idx - aug_r_idx)
    
    # 3. Interval Stability (PR Estimation)
    p_zone_start, p_zone_end = max(0, center - 100), center - 40
    orig_p = original[p_zone_start:p_zone_end]
    aug_p = augmented[p_zone_start:p_zone_end]
    if len(orig_p) > 2 and np.std(orig_p) > 1e-8 and np.std(aug_p) > 1e-8:
        pr_corr, _ = pearsonr(orig_p, aug_p)
    else:
        pr_corr = 0.0
    
    # 4. ST Segment Deviation
    st_start, st_end = center + 40, min(L, center + 100)
    orig_st = original[st_start:st_end]
    aug_st = augmented[st_start:st_end]
    st_mse = np.mean((orig_st - aug_st)**2)
    
    # 5. Full signal correlation
    if np.std(original) < 1e-8 or np.std(augmented) < 1e-8:
        full_corr = 0.0
    else:
        full_corr, _ = pearsonr(original, augmented)
    
    return {
        'qrs_corr': qrs_corr,
        'qrs_mse': qrs_mse,
        'r_peak_shift': r_peak_shift,
        'pr_corr': pr_corr,
        'st_mse': st_mse,
        'full_corr': full_corr
    }

def main(args):
    dataset = ECGBeatDataset(args.data_file)
    n_samples = min(args.n_samples, len(dataset))
    
    # Pipelines operate on numpy arrays, not tensors
    pa_pipeline = PhysioAugPipeline.default(strength='medium')
    naive_pipeline = NaiveAugPipeline(p=0.8)
    
    results = []
    
    print(f"Evaluating {n_samples} samples...")
    for i in range(n_samples):
        # ECGBeatDataset returns (signal, label, r_peak, idx, metadata)
        signal_tensor = dataset[i][0]  # shape: (1, 250)
        r_peak = dataset[i][2]
        
        # Convert to 1D numpy for augmentation pipelines
        orig_signal = signal_tensor.numpy().squeeze()  # shape: (250,)
        
        # Apply PA augmentation (expects numpy array + r_peak_pos)
        pa_aug = pa_pipeline(orig_signal.copy(), r_peak_pos=int(r_peak))
        
        # Apply Naive augmentation (expects numpy array, ignores r_peak)
        naive_aug = naive_pipeline(orig_signal.copy())
        
        # Compute metrics
        pa_metrics = calculate_metrics(orig_signal, pa_aug)
        naive_metrics = calculate_metrics(orig_signal, naive_aug)
        
        results.append({'method': 'Physio-Aware', **pa_metrics})
        results.append({'method': 'Naive', **naive_metrics})
        
    df = pd.DataFrame(results)
    
    # Summary Table
    summary = df.groupby('method').agg(['mean', 'std']).round(4)
    print("\n--- Morphology Preservation Summary ---")
    print(summary)
    
    # Final Verdict
    pa_mean_corr = df[df['method'] == 'Physio-Aware']['qrs_corr'].mean()
    naive_mean_corr = df[df['method'] == 'Naive']['qrs_corr'].mean()
    
    improvement = (pa_mean_corr - naive_mean_corr) * 100
    print(f"\nResult: Physio-Aware augmentations preserve QRS morphology {improvement:+.2f}% better than Naive methods.")
    
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    main(args)
