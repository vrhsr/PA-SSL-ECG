# PA-SSL: Physiology-Aware Self-Supervised Learning for ECG Anomaly Detection

A novel self-supervised learning framework for ECG signals that uses **physiology-preserving augmentations** and **temporal-context contrastive learning** to learn robust representations from unlabeled ECG data.

## Key Contributions

1. **Physiology-Aware Augmentations** — 7 augmentations formally constrained to produce physiologically valid ECGs (QRS protection, heart-rate bounds, realistic noise models)
2. **Temporal Adjacency Contrastive Learning** — Adjacent heartbeats from the same recording serve as natural positive pairs
3. **Calibrated Anomaly Scoring** — Mahalanobis distance-based anomaly detection with ECE evaluation

## Project Structure

```
PA-SSL-ECG/
├── src/
│   ├── data/                    # Dataset processors
│   │   ├── emit_ptbxl.py       # PTB-XL
│   │   ├── emit_mitbih.py      # MIT-BIH Arrhythmia
│   │   ├── emit_chapman.py     # Chapman-Shaoxing
│   │   └── ecg_dataset.py      # Unified dataset + splits
│   ├── augmentations/           # CORE NOVELTY
│   │   ├── physio_augmentations.py   # 7 constrained augmentations
│   │   ├── naive_augmentations.py    # Baseline (unconstrained)
│   │   └── augmentation_pipeline.py  # Composable pipeline
│   ├── models/
│   │   ├── encoder.py           # ResNet1D + WavKAN encoders
│   │   └── anomaly_scorer.py    # Mahalanobis + calibration
│   ├── losses.py                # NT-Xent + temporal contrastive
│   ├── train_ssl.py             # SSL pretraining
│   └── evaluate.py              # Full evaluation suite
├── experiments/
│   └── run_all.ps1              # Master experiment runner
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Process datasets
python -m src.data.emit_ptbxl
python -m src.data.emit_mitbih

# Pretrain (ResNet1D + PhysioAug + Temporal)
python -m src.train_ssl --encoder resnet1d --augmentation physio --use_temporal --epochs 100

# Evaluate
python -m src.evaluate --checkpoint experiments/ssl_resnet1d_physio_temporal/best_checkpoint.pth
```

## Datasets

| Dataset | Source | Beats | Classes |
|---------|--------|-------|---------|
| PTB-XL | PhysioNet | ~100K+ | Normal / Abnormal |
| MIT-BIH | PhysioNet | ~100K+ | Normal / Abnormal (AAMI) |
| Chapman-Shaoxing | PhysioNet | ~100K+ | Normal / Abnormal rhythm |

## Citation

```
@article{passl2026,
  title={Physiology-Aware Self-Supervised Learning for ECG Anomaly Detection},
  author={...},
  year={2026}
}
```
