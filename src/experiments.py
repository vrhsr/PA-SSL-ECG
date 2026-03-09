"""
PA-SSL: Robustness & OOD Detection Experiments

Implements:
  1. Robustness testing — evaluate under controlled noise/artifact corruption
  2. OOD (Out-of-Distribution) detection — detect unseen cardiac pathologies
  3. Ablation study automation — run all configurations and collect results

These experiments directly address key reviewer concerns:
  - "Does the model degrade gracefully under noise?"
  - "Can it detect unknown pathologies?"
  - "What is each component's contribution?"
"""

import torch
import numpy as np
import pandas as pd
import os
import time
import json
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from src.data.ecg_dataset import ECGBeatDataset
from src.models.encoder import build_encoder
from src.models.anomaly_scorer import (
    MahalanobisAnomalyScorer, expected_calibration_error, brier_score
)
from src.evaluate import extract_representations, linear_probe


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ROBUSTNESS TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def add_electrode_noise(signals, snr_db):
    """Add Gaussian electrode noise at specified SNR."""
    power_signal = np.mean(signals ** 2, axis=1, keepdims=True)
    power_noise = power_signal / (10 ** (snr_db / 10))
    noise = np.random.randn(*signals.shape) * np.sqrt(power_noise)
    return signals + noise


def add_motion_artifact(signals, amplitude):
    """
    Add low-frequency motion artifact (simulates patient movement).
    Single sinusoidal component at 0.3-1.0 Hz.
    """
    n_samples = signals.shape[1]
    t = np.arange(n_samples) / 100.0  # 100 Hz
    
    corrupted = signals.copy()
    for i in range(len(signals)):
        freq = np.random.uniform(0.3, 1.0)
        phase = np.random.uniform(0, 2 * np.pi)
        sig_std = np.std(signals[i])
        artifact = amplitude * sig_std * np.sin(2 * np.pi * freq * t + phase)
        corrupted[i] += artifact
    
    return corrupted


def robustness_experiment(encoder, test_csv, device,
                           noise_snrs=(5, 10, 15, 20, 30),
                           motion_amplitudes=(0.05, 0.1, 0.2, 0.5)):
    """
    Evaluate model robustness under controlled corruption.
    
    Reports accuracy degradation curve as noise increases.
    """
    print("\n" + "=" * 60)
    print("Robustness Experiment")
    print("=" * 60)
    
    # Load clean test data
    dataset = ECGBeatDataset(test_csv)
    clean_reprs, clean_labels = extract_representations(encoder, dataset, device)
    
    # Baseline (clean) performance
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000, C=1.0)
    
    # Use half for fitting, half for testing
    n = len(clean_reprs)
    fit_reprs, fit_labels = clean_reprs[:n//2], clean_labels[:n//2]
    
    clf.fit(fit_reprs, fit_labels)
    
    baseline_acc = accuracy_score(clean_labels[n//2:], clf.predict(clean_reprs[n//2:]))
    print(f"Baseline (clean): {baseline_acc:.4f}")
    
    results = [{'corruption': 'clean', 'level': 0, 'accuracy': baseline_acc, 'degradation': 0.0}]
    
    # Electrode noise experiment
    print("\n--- Electrode Noise ---")
    for snr in noise_snrs:
        noisy_signals = add_electrode_noise(dataset.X.copy(), snr)
        
        # Extract representations from noisy signals
        noisy_dataset = ECGBeatDataset.__new__(ECGBeatDataset)
        noisy_dataset.X = noisy_signals.astype(np.float32)
        noisy_dataset.labels = dataset.labels
        noisy_dataset.r_peak_positions = dataset.r_peak_positions
        noisy_dataset.patient_ids = dataset.patient_ids
        noisy_dataset.record_ids = dataset.record_ids
        noisy_dataset.beat_idxs = dataset.beat_idxs
        noisy_dataset.label_mask = dataset.label_mask
        noisy_dataset.signal_cols = dataset.signal_cols
        noisy_dataset.temporal_index = dataset.temporal_index
        
        noisy_reprs, noisy_labels = extract_representations(encoder, noisy_dataset, device)
        noisy_acc = accuracy_score(noisy_labels[n//2:], clf.predict(noisy_reprs[n//2:]))
        degradation = baseline_acc - noisy_acc
        
        print(f"  SNR={snr:2d} dB: acc={noisy_acc:.4f} (Δ={degradation:+.4f})")
        results.append({
            'corruption': 'electrode_noise', 'level': snr,
            'accuracy': noisy_acc, 'degradation': degradation
        })
    
    # Motion artifact experiment
    print("\n--- Motion Artifact ---")
    for amp in motion_amplitudes:
        noisy_signals = add_motion_artifact(dataset.X.copy(), amp)
        
        noisy_dataset = ECGBeatDataset.__new__(ECGBeatDataset)
        noisy_dataset.X = noisy_signals.astype(np.float32)
        noisy_dataset.labels = dataset.labels
        noisy_dataset.r_peak_positions = dataset.r_peak_positions
        noisy_dataset.patient_ids = dataset.patient_ids
        noisy_dataset.record_ids = dataset.record_ids
        noisy_dataset.beat_idxs = dataset.beat_idxs
        noisy_dataset.label_mask = dataset.label_mask
        noisy_dataset.signal_cols = dataset.signal_cols
        noisy_dataset.temporal_index = dataset.temporal_index
        
        noisy_reprs, noisy_labels = extract_representations(encoder, noisy_dataset, device)
        noisy_acc = accuracy_score(noisy_labels[n//2:], clf.predict(noisy_reprs[n//2:]))
        degradation = baseline_acc - noisy_acc
        
        print(f"  amp={amp:.2f}: acc={noisy_acc:.4f} (Δ={degradation:+.4f})")
        results.append({
            'corruption': 'motion_artifact', 'level': amp,
            'accuracy': noisy_acc, 'degradation': degradation
        })
    
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. OOD DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def ood_detection_experiment(encoder, in_dist_csv, ood_csv, device):
    """
    OOD detection experiment using Mahalanobis distance.
    
    Protocol:
      1. Fit Mahalanobis scorer on in-distribution (normal) ECGs
      2. Score both in-distribution and OOD (rare abnormalities)
      3. Evaluate AUROC of distance-based OOD detection
    """
    print("\n" + "=" * 60)
    print("OOD Detection Experiment")
    print("=" * 60)
    
    # Load datasets
    in_dist_dataset = ECGBeatDataset(in_dist_csv)
    
    # Filter to normal only for in-distribution
    normal_mask = in_dist_dataset.labels == 0
    in_dist_dataset_filtered = ECGBeatDataset.__new__(ECGBeatDataset)
    in_dist_dataset_filtered.X = in_dist_dataset.X[normal_mask]
    in_dist_dataset_filtered.labels = in_dist_dataset.labels[normal_mask]
    in_dist_dataset_filtered.r_peak_positions = in_dist_dataset.r_peak_positions[normal_mask]
    in_dist_dataset_filtered.patient_ids = in_dist_dataset.patient_ids[normal_mask]
    in_dist_dataset_filtered.record_ids = in_dist_dataset.record_ids[normal_mask]
    in_dist_dataset_filtered.beat_idxs = in_dist_dataset.beat_idxs[normal_mask]
    in_dist_dataset_filtered.label_mask = in_dist_dataset.label_mask[normal_mask]
    in_dist_dataset_filtered.signal_cols = in_dist_dataset.signal_cols
    in_dist_dataset_filtered.temporal_index = {}
    
    # Extract representations
    in_reprs, in_labels = extract_representations(encoder, in_dist_dataset_filtered, device)
    
    # Fit scorer on normal ECGs
    scorer = MahalanobisAnomalyScorer(use_shrinkage=True)
    scorer.fit(in_reprs, in_labels)
    
    # Load OOD data
    ood_dataset = ECGBeatDataset(ood_csv)
    
    # Score all OOD samples
    ood_reprs, ood_labels = extract_representations(encoder, ood_dataset, device)
    ood_scores, _, _ = scorer.score(ood_reprs)
    
    # Score in-distribution samples (should have lower distances)
    in_scores, _, _ = scorer.score(in_reprs)
    
    # Combine for AUROC
    all_scores = np.concatenate([in_scores, ood_scores])
    all_ood_labels = np.concatenate([
        np.zeros(len(in_scores)),   # in-distribution
        np.ones(len(ood_scores))    # OOD
    ])
    
    auroc = roc_auc_score(all_ood_labels, all_scores)
    
    # Threshold at 95th percentile of in-distribution scores
    threshold = np.percentile(in_scores, 95)
    ood_detected = (ood_scores > threshold).mean()
    
    print(f"  In-distribution: {len(in_scores)} normal ECGs")
    print(f"  OOD: {len(ood_scores)} abnormal ECGs")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  OOD Detection Rate (@ 95% threshold): {ood_detected:.4f}")
    print(f"  Mean in-dist distance: {in_scores.mean():.4f} ± {in_scores.std():.4f}")
    print(f"  Mean OOD distance: {ood_scores.mean():.4f} ± {ood_scores.std():.4f}")
    
    return {
        'auroc_ood': auroc,
        'detection_rate_95': ood_detected,
        'in_dist_mean': in_scores.mean(),
        'ood_mean': ood_scores.mean(),
        'threshold_95': threshold,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ABLATION STUDY AUTOMATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_ablation_suite(data_file, output_dir='experiments/ablations', device=None):
    """
    Run complete ablation study comparing all PA-SSL configurations.
    
    Configurations:
      1. Supervised baseline (random init encoder + labeled data)
      2. SimCLR + naive augmentations
      3. SimCLR + physiology augmentations (no temporal)
      4. SimCLR + temporal positives (no physio aug)
      5. Full PA-SSL (physio aug + temporal)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(output_dir, exist_ok=True)
    
    configurations = [
        {
            'name': 'Supervised (Random Init)',
            'pretrained': False,
            'checkpoint': None,
        },
        {
            'name': 'SimCLR + Naive Aug',
            'pretrained': True,
            'checkpoint': 'experiments/ssl_resnet1d_naive/best_checkpoint.pth',
        },
        {
            'name': 'SimCLR + Physio (no temporal)',
            'pretrained': True,
            'checkpoint': 'experiments/ssl_resnet1d_physio/best_checkpoint.pth',
        },
        {
            'name': 'Full PA-SSL',
            'pretrained': True,
            'checkpoint': 'experiments/ssl_resnet1d_physio_temporal/best_checkpoint.pth',
        },
    ]
    
    results = []
    
    for config in configurations:
        print(f"\n{'='*60}")
        print(f"Ablation: {config['name']}")
        print(f"{'='*60}")
        
        # Load encoder
        encoder = build_encoder('resnet1d', proj_dim=128)
        
        if config['pretrained'] and config['checkpoint'] and os.path.exists(config['checkpoint']):
            checkpoint = torch.load(config['checkpoint'], map_location=device)
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            print(f"  Loaded: {config['checkpoint']}")
        else:
            print(f"  Using random initialization")
        
        encoder = encoder.to(device)
        encoder.eval()
        
        # Evaluate
        dataset = ECGBeatDataset(data_file)
        reprs, labels = extract_representations(encoder, dataset, device)
        
        n = len(reprs)
        train_reprs, train_labels = reprs[:n//2], labels[:n//2]
        test_reprs, test_labels = reprs[n//2:], labels[n//2:]
        
        metrics = linear_probe(train_reprs, train_labels, test_reprs, test_labels)
        
        result = {
            'configuration': config['name'],
            **metrics
        }
        results.append(result)
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  AUROC:    {metrics['auroc']:.4f}")
        print(f"  F1:       {metrics['f1_macro']:.4f}")
        print(f"  ECE:      {metrics['ece']:.4f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'ablation_results.csv'), index=False)
    
    # Print summary table
    print(f"\n{'='*80}")
    print("ABLATION RESULTS SUMMARY")
    print(f"{'='*80}")
    print(results_df.to_string(index=False))
    
    # Generate LaTeX table
    from src.plotting import generate_latex_table
    latex = generate_latex_table(
        results_df.rename(columns={'configuration': 'method'}),
        metrics=('accuracy', 'auroc', 'f1_macro', 'ece'),
        caption="Ablation study results",
        label="tab:ablation"
    )
    
    with open(os.path.join(output_dir, 'ablation_table.tex'), 'w') as f:
        f.write(latex)
    
    print(f"\nResults saved to: {output_dir}")
    return results_df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. COMPUTATIONAL EFFICIENCY PROFILING
# ═══════════════════════════════════════════════════════════════════════════════

def profile_model(encoder, input_shape=(1, 1, 250), n_forward=100, device=None):
    """
    Profile model for training time, inference time, GPU memory, param count.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = encoder.to(device)
    encoder.eval()
    
    n_params = sum(p.numel() for p in encoder.parameters())
    n_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    # Warm up
    dummy = torch.randn(*input_shape, device=device)
    for _ in range(10):
        _ = encoder(dummy)
    
    # Memory
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    # Inference time
    start = time.time()
    for _ in range(n_forward):
        with torch.no_grad():
            _ = encoder(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start
    
    inference_ms = (elapsed / n_forward) * 1000
    
    # Peak memory
    gpu_memory_mb = 0
    if device.type == 'cuda':
        gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    result = {
        'n_parameters': n_params,
        'n_trainable': n_trainable,
        'inference_ms': inference_ms,
        'gpu_memory_mb': gpu_memory_mb,
    }
    
    print(f"  Parameters: {n_params:,} ({n_trainable:,} trainable)")
    print(f"  Inference: {inference_ms:.2f} ms/sample")
    print(f"  GPU Memory: {gpu_memory_mb:.1f} MB")
    
    return result
