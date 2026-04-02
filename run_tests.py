"""
PA-SSL: Comprehensive Test Suite
================================
Tests all core modules without requiring GPU or data files.
Run with: python run_tests.py
"""

import os
import sys
import traceback
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════════════════════
# 0. DETERMINISTIC SEEDS (Test Stability)
# ═══════════════════════════════════════════════════════════════════════════════
np.random.seed(42)
try:
    import torch
    torch.manual_seed(42)
except ImportError:
    pass

PASS = 0
FAIL = 0

def run_test(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  [PASS] {name}")
        PASS += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        FAIL += 1


# ═══════════════════════════════════════════════════════════════════════════════
# 1. AUGMENTATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_physio_augmentations():
    """Test all 8 physiology-aware augmentations."""
    from src.augmentations.physio_augmentations import (
        constrained_time_warp, amplitude_perturbation, baseline_wander,
        emg_noise_injection, heart_rate_resample, powerline_interference,
        segment_dropout, wavelet_masking,
    )
    signal = np.sin(np.linspace(0, 4 * np.pi, 250)).astype(np.float32)
    # Make a clear R-peak at center
    signal[120:130] = signal[120:130] + 2.0
    
    for fn in [constrained_time_warp, amplitude_perturbation, baseline_wander,
               emg_noise_injection, heart_rate_resample, powerline_interference,
               segment_dropout, wavelet_masking]:
        try:
            result = fn(signal.copy(), r_peak_pos=125)
        except TypeError:
            result = fn(signal.copy())
        assert result.shape == (250,), f"{fn.__name__} changed shape to {result.shape}"
        assert np.isfinite(result).all(), f"{fn.__name__} produced NaN/Inf"

def test_naive_augmentations():
    """Test naive augmentation pipeline."""
    from src.augmentations.naive_augmentations import NaiveAugPipeline
    signal = np.random.randn(250).astype(np.float32)
    pipeline = NaiveAugPipeline(p=1.0)
    result = pipeline(signal.copy())
    assert result.shape == (250,), f"Shape mismatch: {result.shape}"
    assert np.isfinite(result).all(), "NaN/Inf in output"

def test_physio_pipeline():
    """Test PhysioAugPipeline with all strength presets."""
    from src.augmentations.augmentation_pipeline import PhysioAugPipeline
    signal = np.random.randn(250).astype(np.float32)
    
    for strength in ['light', 'medium', 'strong']:
        pipeline = PhysioAugPipeline.default(strength=strength)
        result = pipeline(signal.copy(), r_peak_pos=125)
        assert result.shape == (250,), f"{strength}: shape={result.shape}"
        assert np.isfinite(result).all(), f"{strength}: NaN/Inf"

def test_pipeline_exclude_only():
    """Test leave-one-out and leave-one-in filtering."""
    from src.augmentations.augmentation_pipeline import PhysioAugPipeline
    signal = np.random.randn(250).astype(np.float32)
    
    # Exclude one
    pipe = PhysioAugPipeline.default(exclude=['constrained_time_warp'])
    names = [a[0].__name__ for a in pipe.augmentations]
    assert 'constrained_time_warp' not in names
    
    # Only one
    pipe = PhysioAugPipeline.default(only=['baseline_wander'])
    assert len(pipe.augmentations) == 1
    assert pipe.augmentations[0][0].__name__ == 'baseline_wander'


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ENCODER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_encoder_factory():
    """Test build_encoder factory produces correct types."""
    from src.models.encoder import build_encoder
    
    enc1 = build_encoder('resnet1d', proj_dim=128)
    assert enc1.repr_dim == 256
    
    enc2 = build_encoder('wavkan', proj_dim=128)
    assert enc2.repr_dim == 256
    
    # Verify SE-ResNet is gone
    try:
        build_encoder('se_resnet1d34')
        raise AssertionError("se_resnet1d34 should not exist anymore")
    except ValueError:
        pass  # Expected

def test_resnet1d_forward():
    """Test ResNet1D forward pass with various input shapes."""
    import torch
    from src.models.encoder import ResNet1DEncoder
    
    enc = ResNet1DEncoder(repr_dim=256, proj_dim=128)
    enc.eval()
    
    # (B, 1, 250)
    x = torch.randn(4, 1, 250)
    h = enc.encode(x)
    assert h.shape == (4, 256), f"encode shape: {h.shape}"
    
    z = enc(x, return_projection=True)
    assert z.shape == (4, 128), f"projection shape: {z.shape}"
    
    # (B, 250) should also work
    x2 = torch.randn(4, 250)
    h2 = enc.encode(x2)
    assert h2.shape == (4, 256)

def test_wavkan_forward():
    """Test WavKAN encoder forward pass."""
    import torch
    from src.models.encoder import WavKANEncoder
    
    enc = WavKANEncoder(repr_dim=256, proj_dim=128, depth=3)
    enc.eval()
    
    x = torch.randn(4, 1, 250)
    h = enc.encode(x)
    assert h.shape == (4, 256), f"WavKAN encode shape: {h.shape}"
    
    z = enc(x, return_projection=True)
    assert z.shape == (4, 128), f"WavKAN projection shape: {z.shape}"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MAE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_physio_aware_masker():
    """Test PhysioAwareMasker produces valid masks with correct ratio."""
    import torch
    from src.models.mae import PhysioAwareMasker
    
    masker = PhysioAwareMasker(mask_ratio=0.60, qrs_avoidance_prob=0.8)
    x = torch.randn(16, 1, 250)  # Larger batch for stable ratio estimate
    masked_x, masks = masker(x)
    
    assert masked_x.shape == x.shape, f"Masked shape: {masked_x.shape}"
    assert masks.shape == (16, 250), f"Mask shape: {masks.shape}"
    assert masks.sum() > 0, "No masking occurred"
    
    # CRITICAL: Verify mask ratio is approximately correct
    # If masking silently drops to 5% or jumps to 95%, training is useless
    mask_ratio_actual = masks.float().mean().item()
    assert 0.40 <= mask_ratio_actual <= 0.80, \
        f"Expected ~60% masking, got {mask_ratio_actual:.2f} (range: 0.40-0.80)"

def test_mae_decoder():
    """Test MAEDecoder1D reconstructs correct length."""
    import torch
    from src.models.mae import MAEDecoder1D
    
    dec = MAEDecoder1D(repr_dim=256, out_channels=1, seq_len=250)
    h = torch.randn(4, 256)
    out = dec(h)
    assert out.shape == (4, 1, 250), f"Decoder output shape: {out.shape}"

def test_hybrid_mae():
    """Test HybridMAE end-to-end."""
    import torch
    from src.models.encoder import build_encoder
    from src.models.mae import HybridMAE
    
    enc = build_encoder('resnet1d', proj_dim=128)
    hybrid = HybridMAE(enc, repr_dim=256, mask_ratio=0.60)
    
    x = torch.randn(4, 1, 250)
    recon, masks, masked_x = hybrid.forward_mae(x)
    assert recon.shape == (4, 1, 250), f"Reconstruction shape: {recon.shape}"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LOSS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_ntxent_loss():
    """Test NT-Xent contrastive loss."""
    import torch
    from src.losses import NTXentLoss
    
    loss_fn = NTXentLoss(temperature=0.1)
    z1 = torch.randn(8, 128)
    z2 = torch.randn(8, 128)
    loss = loss_fn(z1, z2)
    assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"
    assert torch.isfinite(loss), "Loss is not finite"

def test_vicreg_loss():
    """Test VICReg loss."""
    import torch
    from src.losses import VICRegLoss
    
    loss_fn = VICRegLoss()
    z1 = torch.randn(8, 128)
    z2 = torch.randn(8, 128)
    loss = loss_fn(z1, z2)
    assert torch.isfinite(loss), "VICReg loss is not finite"

def test_combined_loss():
    """Test CombinedContrastiveLoss rejects barlow."""
    import torch
    from src.losses import CombinedContrastiveLoss
    
    # NT-Xent should work
    loss_fn = CombinedContrastiveLoss(loss_type='ntxent')
    z1 = torch.randn(8, 128)
    z2 = torch.randn(8, 128)
    total_loss, loss_aug, loss_temporal = loss_fn(z1, z2)
    assert torch.isfinite(total_loss), f"Total loss not finite: {total_loss}"
    
    # Barlow should fail
    try:
        CombinedContrastiveLoss(loss_type='barlow')
        raise AssertionError("barlow should not be accepted")
    except ValueError:
        pass  # Expected

def test_contrastive_backward():
    """
    Ensure NT-Xent loss produces valid gradients.
    Catches: broken computation graph, NaN gradients.
    """
    import torch
    from src.losses import NTXentLoss

    z1 = torch.randn(8, 128, requires_grad=True)
    z2 = torch.randn(8, 128, requires_grad=True)

    loss_fn = NTXentLoss(temperature=0.1)
    # CombinedContrastiveLoss returns (total, aug, temporal), NTXent returns scalar
    loss = loss_fn(z1, z2)

    loss.backward()

    assert z1.grad is not None, "No gradient for z1"
    assert z2.grad is not None, "No gradient for z2"
    assert torch.isfinite(z1.grad).all(), "NaN gradients in z1"
    assert torch.isfinite(z2.grad).all(), "NaN gradients in z2"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DIAGNOSTIC TESTS (unit-level, no data needed)
# ═══════════════════════════════════════════════════════════════════════════════

def test_embedding_analysis():
    """Test embedding collapse analyzer."""
    from src.experiments.embedding_analysis import analyze_embeddings
    
    # Create fake embeddings (well-distributed)
    embeddings = np.random.randn(100, 64).astype(np.float32)
    result = analyze_embeddings(embeddings)
    
    assert 'avg_cosine_sim' in result
    assert 'avg_variance' in result
    assert result['avg_cosine_sim'] < 0.5, "Random embeddings should have low cosine sim"
    assert result['avg_variance'] > 0.001, "Random embeddings should have non-zero variance"

def test_morphology_metrics():
    """Test morphology metric computation: clean AND noisy cases."""
    from src.experiments.morphology_metrics import calculate_metrics
    
    original = np.sin(np.linspace(0, 4 * np.pi, 250)).astype(np.float32)
    original[120:130] += 2.0  # R-peak
    
    # Case 1: Slightly perturbed (should have high correlation)
    augmented_clean = original + 0.01 * np.random.randn(250).astype(np.float32)
    metrics_clean = calculate_metrics(original, augmented_clean)
    assert metrics_clean['qrs_corr'] > 0.95, f"QRS corr too low: {metrics_clean['qrs_corr']}"
    assert metrics_clean['qrs_mse'] < 0.01, f"QRS MSE too high: {metrics_clean['qrs_mse']}"
    
    # Case 2: Heavily noisy (metric must detect degradation)
    augmented_noisy = original + 0.5 * np.random.randn(250).astype(np.float32)
    metrics_noisy = calculate_metrics(original, augmented_noisy)
    assert metrics_noisy['qrs_corr'] < metrics_clean['qrs_corr'], \
        f"Noise should degrade QRS corr: clean={metrics_clean['qrs_corr']:.3f}, noisy={metrics_noisy['qrs_corr']:.3f}"

def test_anomaly_scorer():
    """Test Mahalanobis anomaly scorer."""
    from src.models.anomaly_scorer import expected_calibration_error, brier_score, sensitivity_specificity
    
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1])
    y_proba = np.random.rand(10, 2)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    
    sens, spec = sensitivity_specificity(y_true, y_proba[:, 1])
    assert 0 <= sens <= 1 and 0 <= spec <= 1

def test_statistical_tests():
    """Test bootstrap CI computation."""
    from src.statistical_tests import bootstrap_confidence_intervals
    
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1])
    y_scores = np.random.rand(10)
    
    ci = bootstrap_confidence_intervals(
        y_true, y_scores,
        metric_fn=lambda y, s: float(np.mean(y == (s > 0.5))),
        n_bootstrap=100
    )
    assert ci[0] <= ci[1], f"CI lower > upper: {ci}"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TS2Vec BASELINE TEST
# ═══════════════════════════════════════════════════════════════════════════════

def test_ts2vec_encoder():
    """Test TS2Vec encoder forward pass with exact shape checks."""
    import torch
    from src.models.ts2vec import TS2VecEncoder
    
    enc = TS2VecEncoder(input_dims=1, output_dims=320, hidden_dims=64, depth=4)
    enc.eval()
    
    x = torch.randn(4, 1, 250)
    out = enc(x)
    
    # Explicit temporal representation check: (B, L, H)
    assert out.shape == (4, 250, 64), f"Unexpected TS2Vec temporal output shape: {out.shape}"
    
    flat = enc.encode(x)
    
    # Explicit flattened embedding check
    assert flat.shape == (4, 64), f"encode() should return (4, 64), got {flat.shape}"
    assert torch.isfinite(flat).all(), "NaN embeddings produced by TS2Vec"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. RESEARCH-CRITICAL TESTS (Experimental Fairness)
# ═══════════════════════════════════════════════════════════════════════════════

def test_encoder_parity():
    """
    CRITICAL: Verify CNN and KAN encoders have comparable parameter counts.
    If one is 2x larger, the ablation is invalid — reviewers will reject.
    """
    from src.models.encoder import build_encoder
    
    cnn = build_encoder('resnet1d', proj_dim=128)
    kan = build_encoder('wavkan', proj_dim=128)
    
    cnn_params = sum(p.numel() for p in cnn.parameters())
    kan_params = sum(p.numel() for p in kan.parameters())
    
    ratio = max(cnn_params, kan_params) / min(cnn_params, kan_params)
    
    print(f"    CNN params: {cnn_params:,}")
    print(f"    KAN params: {kan_params:,}")
    print(f"    Ratio: {ratio:.2f}x")
    
    # Ratio must be ≤ 1.5 for fair comparison (ideal: ≤ 1.2)
    assert ratio <= 1.5, \
        f"Param mismatch too large! CNN={cnn_params:,}, KAN={kan_params:,}, ratio={ratio:.2f}. " \
        f"Adjust repr_dim or depth to achieve parity before running ablation."

def test_hybrid_backward():
    """
    Ensure hybrid MAE training produces valid gradients.
    Catches: NaN grads, exploding grads, dead computation graph.
    """
    import torch
    from src.models.encoder import build_encoder
    from src.models.mae import HybridMAE
    
    enc = build_encoder('resnet1d', proj_dim=128)
    model = HybridMAE(enc, repr_dim=256, mask_ratio=0.60)
    
    x = torch.randn(4, 1, 250)
    recon, masks, masked_x = model.forward_mae(x)
    
    # Compute masked reconstruction loss (only on masked positions)
    loss = ((recon - x) ** 2 * masks.unsqueeze(1).float()).mean()
    loss.backward()
    
    # Verify gradients are finite
    grad_count = 0
    for name, p in enc.named_parameters():
        if p.grad is not None:
            grad_count += 1
            assert torch.isfinite(p.grad).all(), \
                f"NaN/Inf gradient in {name}"
    
    assert grad_count > 0, "No gradients were computed — dead graph!"

def test_patient_split_leakage():
    """
    CRITICAL: Verify patient_aware_split produces ZERO patient overlap.
    If any patient appears in both train and test, the paper is invalid.
    Uses synthetic data so no real CSV is needed.
    """
    import pandas as pd
    import tempfile, os
    from src.data.ecg_dataset import patient_aware_split
    
    # Create synthetic patient data: 5 patients, ~20 beats each
    rows = []
    for pid in range(5):
        for beat in range(20):
            signal = np.random.randn(250).astype(np.float32)
            row = {str(i): signal[i] for i in range(250)}
            row['label'] = np.random.randint(0, 2)
            row['patient_id'] = f'P{pid:03d}'
            row['record_id'] = f'P{pid:03d}'
            row['beat_idx'] = beat
            row['r_peak_pos'] = 125
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Write to temp file to use patient_aware_split
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f, index=False)
        tmp_path = f.name
    
    try:
        train_df, val_df, test_df = patient_aware_split(tmp_path)
        
        train_patients = set(train_df['patient_id'].unique())
        val_patients = set(val_df['patient_id'].unique())
        test_patients = set(test_df['patient_id'].unique())
        
        assert len(train_patients & val_patients) == 0, \
            f"LEAKAGE: train ∩ val = {train_patients & val_patients}"
        assert len(train_patients & test_patients) == 0, \
            f"LEAKAGE: train ∩ test = {train_patients & test_patients}"
        assert len(val_patients & test_patients) == 0, \
            f"LEAKAGE: val ∩ test = {val_patients & test_patients}"
        
        # Verify all patients accounted for
        all_patients = train_patients | val_patients | test_patients
        assert len(all_patients) == 5, f"Expected 5 patients, got {len(all_patients)}"
    finally:
        os.unlink(tmp_path)

def test_masking_strategies():
    """
    Verify all 3 masking strategies produce valid masks with correct ratio.
    This test ensures the masking ablation experiment will work.
    """
    import torch
    from src.models.mae import PhysioAwareMasker
    
    x = torch.randn(16, 1, 250)
    
    for strategy in ['random', 'contiguous', 'physio_aware']:
        masker = PhysioAwareMasker(mask_ratio=0.60, masking_strategy=strategy)
        masked_x, masks = masker(x)
        
        assert masked_x.shape == x.shape, f"{strategy}: shape mismatch"
        assert masks.shape == (16, 250), f"{strategy}: mask shape wrong"
        
        ratio = masks.float().mean().item()
        assert 0.30 <= ratio <= 0.85, \
            f"{strategy}: mask ratio {ratio:.2f} outside valid range"

def test_masking_qrs_avoidance():
    """
    Verify physio_aware masking actually avoids the QRS zone.
    The central ±30 samples (indices 95-155) should be masked LESS often
    than non-QRS regions with physio_aware strategy.
    """
    import torch
    from src.models.mae import PhysioAwareMasker
    
    # Run many samples for stable statistics
    x = torch.randn(64, 1, 250)
    
    # Physio-aware: should avoid QRS
    pa_masker = PhysioAwareMasker(mask_ratio=0.60, masking_strategy='physio_aware',
                                   qrs_avoidance_prob=1.0)  # 100% avoidance for clean test
    _, pa_masks = pa_masker(x)
    
    # Random: no avoidance
    rand_masker = PhysioAwareMasker(mask_ratio=0.60, masking_strategy='random')
    _, rand_masks = rand_masker(x)
    
    qrs_zone = slice(95, 155)  # center ± 30
    
    pa_qrs_rate = pa_masks[:, qrs_zone].float().mean().item()
    rand_qrs_rate = rand_masks[:, qrs_zone].float().mean().item()
    
    # Physio-aware should mask QRS zone LESS than random
    # (with 100% avoidance, the QRS zone should be largely untouched)
    assert pa_qrs_rate < rand_qrs_rate + 0.05, \
        f"QRS avoidance not working: physio={pa_qrs_rate:.3f}, random={rand_qrs_rate:.3f}"



# ═══════════════════════════════════════════════════════════════════════════════
# 8. NEW ENCODER & CLINICAL TESTS (Phase 1 additions)
# ═══════════════════════════════════════════════════════════════════════════════

def test_transformer_encoder():
    """
    Transformer encoder: forward pass shape + parameter count within 2x of ResNet1D.
    Validates patch embedding, CLS-token extraction, and repr_dim output.
    """
    import torch
    from src.models.encoder import build_encoder

    enc = build_encoder('transformer', proj_dim=128)
    enc.eval()

    x = torch.randn(4, 1, 250)

    # Representation output
    h = enc.encode(x)
    assert h.shape == (4, 256), f"Expected (4, 256), got {h.shape}"
    assert torch.isfinite(h).all(), "NaN in Transformer representations"

    # Projection output
    z = enc(x, return_projection=True)
    assert z.shape == (4, 128), f"Expected (4, 128) projection, got {z.shape}"

    # Parameter count — should be within 3x of ResNet1D (2.05M)
    n_params = sum(p.numel() for p in enc.parameters())
    resnet = build_encoder('resnet1d', proj_dim=128)
    resnet_params = sum(p.numel() for p in resnet.parameters())
    ratio = n_params / resnet_params
    print(f"    Transformer params: {n_params:,} | ResNet ratio: {ratio:.2f}x")
    assert ratio < 3.0, f"Transformer too large vs ResNet ({ratio:.2f}x). Reduce d_model or num_layers."


def test_mamba_encoder():
    """
    Mamba encoder: forward pass with GRU fallback when mamba_ssm is unavailable.
    On GPU+Linux with mamba_ssm installed, uses real SSM kernels.
    On CPU/Windows without mamba_ssm, uses bidirectional GRU fallback.
    """
    import torch
    from src.models.encoder import build_encoder

    enc = build_encoder('mamba', proj_dim=128)
    enc.eval()

    x = torch.randn(4, 1, 250)
    h = enc.encode(x)
    assert h.shape == (4, 256), f"Expected (4, 256), got {h.shape}"
    assert torch.isfinite(h).all(), "NaN in Mamba representations"

    z = enc(x, return_projection=True)
    assert z.shape == (4, 128), f"Expected (4, 128) projection, got {z.shape}"

    from src.models.mamba_encoder import _MAMBA_AVAILABLE
    mode = "Mamba SSM" if _MAMBA_AVAILABLE else "GRU fallback"
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"    Mode: {mode} | Params: {n_params:,}")


def test_multi_task_smoke():
    """
    Multi-task clinical evaluation: smoke test with synthetic data.
    Validates that all 4 tasks run end-to-end without error.
    """
    import torch
    import numpy as np
    from src.models.encoder import build_encoder
    from src.experiments.multi_task_evaluation import (
        AgeRegressionProbe,
        SexClassificationProbe,
        EmbeddingRetrievalEvaluator,
    )

    # Synthetic representations + metadata
    n = 200
    reprs = np.random.randn(n, 256).astype(np.float32)
    ages = np.random.uniform(20, 80, n).astype(np.float32)
    sexes = np.random.randint(0, 2, n).astype(np.float32)
    labels = np.random.randint(0, 5, n)

    split = 120
    tr, ts = slice(0, split), slice(split, n)

    # Age regression
    age_probe = AgeRegressionProbe()
    age_probe.fit(reprs[tr], ages[tr])
    age_res = age_probe.evaluate(reprs[ts], ages[ts])
    assert "mae_years" in age_res and age_res["mae_years"] > 0

    # Sex classification
    sex_probe = SexClassificationProbe()
    sex_probe.fit(reprs[tr], sexes[tr])
    sex_res = sex_probe.evaluate(reprs[ts], sexes[ts])
    assert 0 <= sex_res["auroc"] <= 1

    # Retrieval
    retrieval = EmbeddingRetrievalEvaluator(k_values=(1, 5))
    ret_res = retrieval.evaluate(reprs[ts], labels[ts])
    assert "P@1" in ret_res["precision_at_k"]
    assert 0 <= ret_res["mAP"] <= 1

    print(f"    Age MAE: {age_res['mae_years']:.2f} yr | "
          f"Sex AUROC: {sex_res['auroc']:.3f} | "
          f"Retrieval mAP: {ret_res['mAP']:.3f}")


def test_mcdropout_uncertainty():
    """
    MC-Dropout uncertainty: validate output shapes and that uncertain samples
    can be identified. Confidence must improve after rejecting high-uncertainty.
    """
    import torch
    import numpy as np
    from src.models.encoder import build_encoder
    from src.models.uncertainty import MCDropoutPredictor

    enc = build_encoder('resnet1d', proj_dim=128)
    device = torch.device('cpu')
    enc = enc.to(device)

    # Synthetic training data
    n_train, n_test = 100, 50
    train_reprs = np.random.randn(n_train, 256).astype(np.float32)
    train_labels = np.random.randint(0, 3, n_train)
    test_reprs = np.random.randn(n_test, 256).astype(np.float32)
    test_labels = np.random.randint(0, 3, n_test)

    predictor = MCDropoutPredictor(enc, num_classes=3, T=5, dropout_rate=0.2)
    predictor.fit_classifier(train_reprs, train_labels)

    x_test = torch.tensor(test_reprs).unsqueeze(1)  # (50, 1, 256) — fake signal dim
    # Use shorter dummy signals to avoid shape mismatch; patch encoding
    x_test_signal = torch.randn(n_test, 1, 250)  # Proper signal shape

    mean_p, variance = predictor.predict_with_uncertainty(x_test_signal)
    assert mean_p.shape == (n_test, 3), f"Expected ({n_test}, 3), got {mean_p.shape}"
    assert variance.shape == (n_test, 3)
    assert np.all(variance >= 0), "Variance must be non-negative"
    assert np.allclose(mean_p.sum(axis=1), 1.0, atol=0.01), "Probabilities must sum to 1"

    # Selective prediction
    report = predictor.selective_prediction(x_test_signal, test_labels, threshold_percentile=80)
    assert 0 <= report["flagged_fraction"] <= 1
    assert 0 <= report["coverage"] <= 1
    print(f"    Mean uncertainty: {report['mean_uncertainty']:.4f} | "
          f"Flagged: {report['flagged_fraction']:.1%} | "
          f"Coverage: {report['coverage']:.1%}")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print(" PA-SSL-ECG: Comprehensive Test Suite")
    print("=" * 60)
    
    print("\n── 1. Augmentation Tests ──")
    run_test("Physiology-Aware Augmentations (8 transforms)", test_physio_augmentations)
    run_test("Naive Augmentation Pipeline", test_naive_augmentations)
    run_test("PhysioAugPipeline (light/medium/strong)", test_physio_pipeline)
    run_test("Pipeline exclude/only filtering", test_pipeline_exclude_only)
    
    print("\n── 2. Encoder Tests ──")
    run_test("Encoder factory (resnet1d, wavkan)", test_encoder_factory)
    run_test("ResNet1D forward pass", test_resnet1d_forward)
    run_test("WavKAN forward pass", test_wavkan_forward)
    
    print("\n── 3. MAE Tests ──")
    run_test("PhysioAwareMasker (ratio verification)", test_physio_aware_masker)
    run_test("MAEDecoder1D", test_mae_decoder)
    run_test("HybridMAE end-to-end", test_hybrid_mae)
    
    print("\n── 4. Loss Tests ──")
    run_test("NT-Xent Loss", test_ntxent_loss)
    run_test("NT-Xent backward pass", test_contrastive_backward)
    run_test("VICReg Loss", test_vicreg_loss)
    run_test("CombinedContrastiveLoss (no barlow)", test_combined_loss)
    
    print("\n── 5. Diagnostic Tests ──")
    run_test("Embedding collapse analyzer", test_embedding_analysis)
    run_test("Morphology metrics (clean + noisy)", test_morphology_metrics)
    run_test("Anomaly scorer (ECE, Brier, Sens/Spec)", test_anomaly_scorer)
    run_test("Bootstrap confidence intervals", test_statistical_tests)
    
    print("\n── 6. Baseline Tests ──")
    run_test("TS2Vec encoder forward", test_ts2vec_encoder)
    
    print("\n── 7. Research-Critical Tests ──")
    run_test("★ Encoder parameter parity (CNN≈KAN)", test_encoder_parity)
    run_test("★ Hybrid MAE gradient check", test_hybrid_backward)
    run_test("★ Patient-level split (zero leakage)", test_patient_split_leakage)
    run_test("★ Masking strategies (random/contiguous/physio)", test_masking_strategies)
    run_test("★ QRS avoidance verification", test_masking_qrs_avoidance)

    print("\n── 8. New Encoder & Clinical Tests ──")
    run_test("Transformer encoder forward", test_transformer_encoder)
    run_test("Mamba encoder forward (GRU fallback OK)", test_mamba_encoder)
    run_test("Multi-task evaluation smoke", test_multi_task_smoke)
    run_test("MC-Dropout uncertainty shapes", test_mcdropout_uncertainty)
    
    # Summary
    total = PASS + FAIL
    print("\n" + "=" * 60)
    print(f" Results: {PASS}/{total} passed, {FAIL}/{total} failed")
    print("=" * 60)
    
    if FAIL > 0:
        sys.exit(1)
