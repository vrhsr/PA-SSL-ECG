import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Running Automated Tests for PA-SSL-ECG Upgrades...")

try:
    print("\n[1] Testing New Evaluation Metrics...")
    from src.models.anomaly_scorer import expected_calibration_error, brier_score, sensitivity_specificity
    from src.statistical_tests import bootstrap_confidence_intervals
    
    y_true = np.array([0,0,0,1,1,1,1,0,0,1])
    y_proba = np.random.rand(10, 2)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    
    sens, spec = sensitivity_specificity(y_true, y_proba[:, 1])
    print(f"  Sensitivity: {sens:.4f}, Specificity: {spec:.4f}")
    assert 0 <= sens <= 1 and 0 <= spec <= 1
    
    ci = bootstrap_confidence_intervals(y_true, y_proba[:, 1], metric_fn=lambda y, s: float(np.mean(y == (s > 0.5))), n_bootstrap=100)
    print(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    assert ci[0] <= ci[1]
    print("[1] PASSED")
except Exception as e:
    print(f"[1] FAILED: {e}")

try:
    print("\n[2] Testing Multi-scale Temporal Positives...")
    from src.data.ecg_dataset import ECGBeatDataset
    if os.path.exists('data/ptbxl_processed.csv'):
        ds = ECGBeatDataset('data/ptbxl_processed.csv')
        for scale in [1, 2, 3]:
            neighbor = ds.get_temporal_neighbor(0, scales=[scale])
            print(f"  Scale {scale}: neighbor={neighbor}")
        print("[2] PASSED")
    else:
        print("[2] SKIPPED (ptbxl_processed.csv not found)")
except Exception as e:
    print(f"[2] FAILED: {e}")

try:
    print("\n[3] Testing QRS Preservation Metric and SDR...")
    from src.augmentations.visualize_and_test import generate_synthetic_beat, qrs_preservation_metric, signal_distortion_ratio
    from src.augmentations.physio_augmentations import amplitude_perturbation
    from src.augmentations.naive_augmentations import NaiveAugPipeline
    
    signal = generate_synthetic_beat()
    physio_aug = amplitude_perturbation(signal.copy(), 125, qrs_protect=True)
    naive_pipe = NaiveAugPipeline(p=1.0)
    naive_aug = naive_pipe.naive_amplitude_scale(signal.copy(), scale_range=(0.1, 2.0))
    
    physio_corr = qrs_preservation_metric(signal, physio_aug, r_peak_pos=125)
    naive_corr = qrs_preservation_metric(signal, naive_aug, r_peak_pos=125)
    physio_sdr = signal_distortion_ratio(signal, physio_aug)
    
    print(f"  Physio QRS correlation: {physio_corr:.4f}")
    print(f"  Naive QRS correlation: {naive_corr:.4f}")
    print(f"  Physio SDR: {physio_sdr:.2f}dB")
    assert physio_corr > naive_corr, "Physio should preserve QRS better than naive"
    print("[3] PASSED")
except Exception as e:
    print(f"[3] FAILED: {e}")

try:
    print("\n[4] Running full visualization_and_test module...")
    from src.augmentations.visualize_and_test import run_augmentation_tests
    success = run_augmentation_tests()
    if success:
        print("[4] PASSED")
    else:
        print("[4] FAILED")
except Exception as e:
    print(f"[4] FAILED: {e}")

print("\nAll Tests Executed.")
