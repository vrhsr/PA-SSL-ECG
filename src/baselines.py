"""
PA-SSL: Baseline Methods for Comparison

Implements:
  1. CLOCS-style baseline (patient-level contrastive, NO physio augmentations)
  2. Fully supervised baseline (ResNet1D trained end-to-end, no SSL)
  3. Random initialization baseline (frozen random encoder)

These are needed for fair comparison in the paper.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.encoder import build_encoder
from src.data.ecg_dataset import ECGBeatDataset
from src.evaluate import extract_representations, linear_probe


# ==============================================================================
# 1. RANDOM INITIALIZATION BASELINE
# ==============================================================================

def evaluate_random_init(data_csv, device, encoder_type='resnet1d', n_seeds=3, max_batches=None):
    """
    Evaluate a randomly initialized (untrained) encoder.
    This is the lower-bound baseline.
    
    Args:
        data_csv: path to processed dataset CSV
        device: torch device
        encoder_type: 'resnet1d' or 'wavkan'
        n_seeds: number of random seeds
    
    Returns:
        DataFrame with per-seed results
    """
    results = []
    fractions = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    
    for seed in range(n_seeds):
        print(f"  [Random Init] Seed {seed+1}/{n_seeds} — extracting representations...")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Build a fresh random encoder each time
        encoder = build_encoder(encoder_type, proj_dim=128)
        encoder = encoder.to(device)
        encoder.eval()
        
        dataset = ECGBeatDataset(data_csv)
        reprs, labels = extract_representations(encoder, dataset, device, max_batches=max_batches)
        print(f"  [Random Init] Seed {seed+1}/{n_seeds} — running linear probes...")
        
        # Split
        n = len(labels)
        indices = np.random.permutation(n)
        split = int(0.8 * n)
        train_idx, test_idx = indices[:split], indices[split:]
        
        for frac in fractions:
            n_train = max(10, int(frac * len(train_idx)))
            train_sub = train_idx[:n_train]
            
            metrics = linear_probe(
                reprs[train_sub], labels[train_sub],
                reprs[test_idx], labels[test_idx]
            )
            
            result = {
                'method': 'Random Init',
                'label_fraction': frac,
                'seed': seed,
                'n_labeled': n_train,
                **{f'linear_{k}': v for k, v in metrics.items()},
            }
            results.append(result)
    
    return pd.DataFrame(results)


# ==============================================================================
# 2. FULLY SUPERVISED BASELINE
# ==============================================================================

class SupervisedClassifier(nn.Module):
    """ResNet1D + linear head, trained end-to-end with labels."""
    
    def __init__(self, encoder, num_classes=2):
        super().__init__()
        self.encoder = encoder
        # Determine representation dim
        with torch.no_grad():
            dummy = torch.randn(1, 1, 250)
            rep_dim = encoder(dummy).shape[-1]
        self.classifier = nn.Linear(rep_dim, num_classes)
    
    def forward(self, x):
        features = self.encoder(x, return_projection=False)
        return self.classifier(features)


def train_supervised(data_csv, device, encoder_type='resnet1d',
                     epochs=50, batch_size=128, lr=1e-3, n_seeds=3, max_batches=None):
    """
    Train a fully supervised classifier from scratch (no SSL pretraining).
    This is the upper-bound baseline at 100% labels.
    
    Returns:
        DataFrame with per-seed results at multiple label fractions
    """
    results = []
    fractions = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    
    dataset = ECGBeatDataset(data_csv)
    n = len(dataset)
    
    for seed in range(n_seeds):
        print(f"  [Supervised] Seed {seed+1}/{n_seeds}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Split indices
        indices = np.random.permutation(n)
        split = int(0.8 * n)
        train_idx, test_idx = indices[:split], indices[split:]
        
        for frac in fractions:
            print(f"    Label fraction: {int(frac*100)}%...", flush=True)
            torch.manual_seed(seed + int(frac * 1000))
            
            # Build fresh encoder
            encoder = build_encoder(encoder_type, proj_dim=128)
            model = SupervisedClassifier(encoder, num_classes=2).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            # Subsample training data
            n_train = max(10, int(frac * len(train_idx)))
            train_sub_idx = train_idx[:n_train]
            
            # Load data
            signals_train = []
            labels_train = []
            for i in train_sub_idx:
                sample = dataset[i]
                signals_train.append(sample[0])
                labels_train.append(sample[1])
            
            signals_test = []
            labels_test = []
            for i in test_idx:
                sample = dataset[i]
                signals_test.append(sample[0])
                labels_test.append(sample[1])
            
            X_train = torch.stack(signals_train)
            y_train = torch.tensor(labels_train, dtype=torch.long)
            X_test = torch.stack(signals_test)
            y_test = torch.tensor(labels_test, dtype=torch.long)
            
            train_loader = DataLoader(
                TensorDataset(X_train, y_train),
                batch_size=batch_size, shuffle=True
            )
            
            # Train
            model.train()
            pbar = tqdm(range(epochs), desc=f"      Supervised Seed{seed} frac={frac:.0%}", leave=False)
            for ep in pbar:
                ep_loss = 0.0
                for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                    if max_batches is not None and batch_idx >= max_batches:
                        break
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()
                    ep_loss += loss.item()
                pbar.set_postfix({'loss': f'{ep_loss:.4f}'})
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                test_logits = []
                for i in range(0, len(X_test), batch_size):
                    batch = X_test[i:i+batch_size].to(device)
                    test_logits.append(model(batch).cpu())
                test_logits = torch.cat(test_logits)
                test_probs = torch.softmax(test_logits, dim=1).numpy()
                test_preds = test_logits.argmax(dim=1).numpy()
            
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
            from src.models.anomaly_scorer import expected_calibration_error, brier_score, sensitivity_specificity
            
            acc = accuracy_score(y_test.numpy(), test_preds)
            f1 = f1_score(y_test.numpy(), test_preds, average='macro')
            try:
                auroc = roc_auc_score(y_test.numpy(), test_probs[:, 1])
                auprc = average_precision_score(y_test.numpy(), test_probs[:, 1])
                sens, spec = sensitivity_specificity(y_test.numpy(), test_probs[:, 1])
            except Exception:
                auroc, auprc, sens, spec = 0.0, 0.0, 0.0, 0.0
            ece, _ = expected_calibration_error(y_test.numpy(), test_probs)
            bs = brier_score(y_test.numpy(), test_probs)
            
            result = {
                'method': 'Supervised',
                'label_fraction': frac,
                'seed': seed,
                'n_labeled': n_train,
                'linear_accuracy': acc,
                'linear_f1_macro': f1,
                'linear_auroc': auroc,
                'linear_auprc': auprc,
                'linear_sensitivity': sens,
                'linear_specificity': spec,
                'linear_ece': ece,
                'linear_brier_score': bs,
            }
            results.append(result)
            
            print(f"  Supervised (frac={frac:.0%}, seed={seed}): "
                  f"acc={acc:.4f}, auroc={auroc:.4f}")
    
    return pd.DataFrame(results)


# ==============================================================================
# 3. CLOCS-STYLE BASELINE (Patient-level contrastive, NO physio augmentations)
# ==============================================================================

def evaluate_clocs_style(checkpoint_path, data_csv, device, n_seeds=3, max_batches=None):
    """
    Evaluate a CLOCS-style trained encoder (naive augmentations only).
    Uses the ablation checkpoint: ssl_resnet1d_naive/best_checkpoint.pth
    
    This serves as the CLOCS baseline since CLOCS uses:
    - Patient-level contrastive (we have temporal which is similar)
    - Standard augmentations (noise, scaling - same as our naive)
    - No physiology-aware constraints
    
    Args:
        checkpoint_path: path to naive-aug trained checkpoint
        data_csv: path to processed dataset CSV
        device: torch device
        n_seeds: number of random seeds
    
    Returns:
        DataFrame with per-seed results
    """
    results = []
    fractions = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    
    # Load encoder
    encoder = build_encoder('resnet1d', proj_dim=128)
    ckpt = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    encoder = encoder.to(device)
    encoder.eval()
    
    dataset = ECGBeatDataset(data_csv)
    reprs, labels = extract_representations(encoder, dataset, device, max_batches=max_batches)
    
    for seed in range(n_seeds):
        np.random.seed(seed)
        n = len(labels)
        indices = np.random.permutation(n)
        split = int(0.8 * n)
        train_idx, test_idx = indices[:split], indices[split:]
        
        for frac in fractions:
            n_train = max(10, int(frac * len(train_idx)))
            train_sub = train_idx[:n_train]
            
            metrics = linear_probe(
                reprs[train_sub], labels[train_sub],
                reprs[test_idx], labels[test_idx]
            )
            
            result = {
                'method': 'SimCLR + Naive Aug (CLOCS-style)',
                'label_fraction': frac,
                'seed': seed,
                'n_labeled': n_train,
                **{f'linear_{k}': v for k, v in metrics.items()},
            }
            results.append(result)
    
    return pd.DataFrame(results)


# ==============================================================================
# 4. TS2Vec OFFICIAL BASELINE
# ==============================================================================

def train_and_evaluate_ts2vec(data_csv, device, epochs=50, batch_size=128, lr=1e-3, n_seeds=3, max_batches=None):
    from src.models.ts2vec import TS2VecEncoder, HierarchicalContrastiveLoss
    from src.augmentations.naive_augmentations import NaiveAugPipeline
    
    results = []
    fractions = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    dataset = ECGBeatDataset(data_csv)
    n = len(dataset)
    aug_pipe = NaiveAugPipeline(p=1.0)
    
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        indices = np.random.permutation(n)
        split = int(0.8 * n)
        train_idx, test_idx = indices[:split], indices[split:]
        
        # TS2Vec Pretraining
        encoder = TS2VecEncoder(input_dims=1, output_dims=320, hidden_dims=64, depth=10).to(device)
        criterion = HierarchicalContrastiveLoss(alpha=0.5, temp=0.1)
        optimizer = optim.Adam(encoder.parameters(), lr=lr)
        
        X_train_pre = torch.stack([dataset[i][0] for i in train_idx])
        loader = DataLoader(TensorDataset(X_train_pre), batch_size=batch_size, shuffle=True)
        
        encoder.train()
        pbar = tqdm(range(epochs), desc=f"  [TS2Vec] Seed {seed+1}/{n_seeds} pretraining")
        for ep in pbar:
            ep_loss = 0.0
            for batch_x in loader:
                x = batch_x[0]
                # Create two views using standard naive aug
                x1 = torch.stack([torch.tensor(aug_pipe(s.numpy())) for s in x]).to(device)
                x2 = torch.stack([torch.tensor(aug_pipe(s.numpy())) for s in x]).to(device)
                
                optimizer.zero_grad()
                z1 = encoder(x1)
                z2 = encoder(x2)
                loss = criterion(z1, z2)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
            pbar.set_postfix({'loss': f'{ep_loss:.4f}'})
        
        # Evaluation
        encoder.eval()
        with torch.no_grad():
            reprs = []
            loader_all = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            labels = []
            for batch_tuple in loader_all:
                b_x, b_y = batch_tuple[0], batch_tuple[1]
                z = encoder.encode(b_x.to(device)).cpu()
                reprs.append(z)
                labels.append(b_y)
            reprs = torch.cat(reprs).numpy()
            labels = torch.cat(labels).numpy()
            
        for frac in fractions:
            n_train = max(10, int(frac * len(train_idx)))
            train_sub = train_idx[:n_train]
            
            metrics = linear_probe(
                reprs[train_sub], labels[train_sub],
                reprs[test_idx], labels[test_idx]
            )
            
            result = {
                'method': 'TS2Vec (Official)',
                'label_fraction': frac,
                'seed': seed,
                'n_labeled': n_train,
                **{f'linear_{k}': v for k, v in metrics.items()},
            }
            results.append(result)
            
    return pd.DataFrame(results)

# ==============================================================================
# 5. TFC OFFICIAL BASELINE
# ==============================================================================

def train_and_evaluate_tfc(data_csv, device, epochs=50, batch_size=128, lr=1e-3, n_seeds=3, max_batches=None):
    from src.models.tfc import TFCEncoder, TFCLoss
    from src.augmentations.naive_augmentations import NaiveAugPipeline
    
    results = []
    fractions = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    dataset = ECGBeatDataset(data_csv)
    n = len(dataset)
    aug_pipe = NaiveAugPipeline(p=1.0)
    
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        indices = np.random.permutation(n)
        split = int(0.8 * n)
        train_idx, test_idx = indices[:split], indices[split:]
        
        # TFC Pretraining
        encoder = TFCEncoder(in_channels=1, hidden_dim=64, proj_dim=128).to(device)
        criterion = TFCLoss(temperature=0.2)
        optimizer = optim.Adam(encoder.parameters(), lr=lr)
        
        X_train_pre = torch.stack([dataset[i][0] for i in train_idx])
        loader = DataLoader(TensorDataset(X_train_pre), batch_size=batch_size, shuffle=True)
        
        encoder.train()
        pbar = tqdm(range(epochs), desc=f"  [TFC] Seed {seed+1}/{n_seeds} pretraining")
        for ep in pbar:
            ep_loss = 0.0
            for batch_idx, batch_x in enumerate(loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                x = batch_x[0].to(device)
                
                # TFC uses time and frequency views for consistency
                x_aug = torch.stack([torch.tensor(aug_pipe(s.cpu().numpy())) for s in x]).to(device)
                x_f = torch.fft.fft(x_aug).abs()
                
                optimizer.zero_grad()
                _, z_t = encoder.forward_time(x_aug)
                _, z_f = encoder.forward_freq(x_f)
                
                loss = criterion(z_t, z_f)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
            pbar.set_postfix({'loss': f'{ep_loss:.4f}'})
        
        # Evaluation
        encoder.eval()
        with torch.no_grad():
            reprs = []
            loader_all = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            labels = []
            for batch_tuple in loader_all:
                b_x, b_y = batch_tuple[0], batch_tuple[1]
                z = encoder.encode(b_x.to(device)).cpu()
                reprs.append(z)
                labels.append(b_y)
            reprs = torch.cat(reprs).numpy()
            labels = torch.cat(labels).numpy()
            
        for frac in fractions:
            n_train = max(10, int(frac * len(train_idx)))
            train_sub = train_idx[:n_train]
            
            metrics = linear_probe(
                reprs[train_sub], labels[train_sub],
                reprs[test_idx], labels[test_idx]
            )
            
            result = {
                'method': 'TFC (Official)',
                'label_fraction': frac,
                'seed': seed,
                'n_labeled': n_train,
                **{f'linear_{k}': v for k, v in metrics.items()},
            }
            results.append(result)
            
    return pd.DataFrame(results)

# ==============================================================================
# 6. RUN ALL BASELINES
# ==============================================================================

def run_all_baselines(data_csv, device, naive_checkpoint=None, output_dir='experiments/baselines', epochs=50, max_batches=None):
    """
    Run all baseline evaluations and save results.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    
    # 1. Random init
    print("\n=== Evaluating Random Initialization Baseline ===")
    random_results = evaluate_random_init(data_csv, device, max_batches=max_batches)
    random_results.to_csv(os.path.join(output_dir, 'random_init_results.csv'), index=False)
    all_results.append(random_results)
    
    # 2. Supervised
    print("\n=== Training Supervised Baseline ===")
    supervised_results = train_supervised(data_csv, device, epochs=epochs, max_batches=max_batches)
    supervised_results.to_csv(os.path.join(output_dir, 'supervised_results.csv'), index=False)
    all_results.append(supervised_results)
    
    # 3. TS2Vec Official
    print("\n=== Training & Evaluating TS2Vec (Official) Baseline ===")
    ts2vec_results = train_and_evaluate_ts2vec(data_csv, device, epochs=epochs, max_batches=max_batches)
    ts2vec_results.to_csv(os.path.join(output_dir, 'ts2vec_results.csv'), index=False)
    all_results.append(ts2vec_results)
    
    # 4. TFC Official
    print("\n=== Training & Evaluating TFC (Official) Baseline ===")
    tfc_results = train_and_evaluate_tfc(data_csv, device, epochs=epochs, max_batches=max_batches)
    tfc_results.to_csv(os.path.join(output_dir, 'tfc_results.csv'), index=False)
    all_results.append(tfc_results)
    
    # 5. CLOCS-style (if checkpoint exists)
    if naive_checkpoint and os.path.exists(naive_checkpoint):
        print("\n=== Evaluating CLOCS-style Baseline ===")
        clocs_results = evaluate_clocs_style(naive_checkpoint, data_csv, device, max_batches=max_batches)
        clocs_results.to_csv(os.path.join(output_dir, 'clocs_style_results.csv'), index=False)
        all_results.append(clocs_results)
    
    # Combine all
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(os.path.join(output_dir, 'all_baseline_results.csv'), index=False)
    print(f"\nAll baseline results saved to: {output_dir}/all_baseline_results.csv")
    
    return combined


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PA-SSL Baselines")
    parser.add_argument('--data_csv', type=str, default='data/ptbxl_processed.csv')
    parser.add_argument('--naive_checkpoint', type=str, 
                        default='experiments/ssl_resnet1d_naive/best_checkpoint.pth')
    parser.add_argument('--output_dir', type=str, default='experiments/baselines')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--max_batches', type=int, default=None)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    run_all_baselines(args.data_csv, device, args.naive_checkpoint, args.output_dir, args.epochs, args.max_batches)
