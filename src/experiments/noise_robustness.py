"""
Noise Robustness Evaluation Script

Tests how well SSL representations degrade under simulated clinical noise:
  - Baseline Wander (BW): Low-frequency drift from patient breathing/movement
  - Powerline Interference (PLI): 50/60Hz sinusoidal contamination
  - EMG Noise: High-frequency muscle artifact

For each noise type and SNR level, we extract representations from noisy 
signals and evaluate downstream classification performance.

A robust model should show minimal degradation compared to clean signals.
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from src.models.encoder import build_encoder
from src.evaluate import extract_representations
from src.data.ecg_dataset import ECGBeatDataset


class NoisyECGDataset(Dataset):
    """Wraps an ECGBeatDataset and applies controlled noise corruption."""
    
    def __init__(self, base_dataset, noise_type='bw', snr_db=10.0):
        self.base_dataset = base_dataset
        self.noise_type = noise_type
        self.snr_db = snr_db
        
    def __len__(self):
        return len(self.base_dataset)
    
    def _add_noise(self, signal):
        """Add noise at the specified SNR (dB)."""
        sig_power = np.mean(signal ** 2)
        snr_linear = 10 ** (self.snr_db / 10)
        noise_power = sig_power / snr_linear
        
        L = len(signal)
        
        if self.noise_type == 'bw':
            # Baseline Wander: low-frequency sinusoidal drift (0.1-0.5 Hz)
            freq = np.random.uniform(0.1, 0.5)
            t = np.arange(L) / 100.0  # 100Hz sampling
            noise = np.sqrt(noise_power) * np.sin(2 * np.pi * freq * t)
            
        elif self.noise_type == 'pli':
            # Powerline Interference: 50Hz or 60Hz sinusoid
            freq = np.random.choice([50, 60])
            t = np.arange(L) / 100.0
            noise = np.sqrt(noise_power) * np.sin(2 * np.pi * freq * t)
            
        elif self.noise_type == 'emg':
            # EMG Noise: high-frequency white-ish noise
            noise = np.sqrt(noise_power) * np.random.randn(L)
            
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
        
        return signal + noise.astype(np.float32)
    
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        signal = item[0].numpy().squeeze()
        noisy = self._add_noise(signal)
        return torch.tensor(noisy, dtype=torch.float32).unsqueeze(0), item[1]


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Noise Robustness Evaluation using Device: {device}")
    
    # Load encoder
    encoder = build_encoder(args.encoder, proj_dim=128).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if 'encoder_state_dict' in ckpt:
            encoder.load_state_dict(ckpt['encoder_state_dict'])
        else:
            encoder.load_state_dict(ckpt)
    encoder.eval()
    
    # Load clean dataset and split
    dataset = ECGBeatDataset(args.data_file)
    targets = [dataset[i][1].item() for i in range(len(dataset))]
    indices = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=targets, random_state=42)
    
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    test_subset = torch.utils.data.Subset(dataset, test_idx)
    
    # 1. Get clean baseline performance (train LR on clean train, test on clean test)
    print("Extracting clean representations...")
    train_loader = DataLoader(train_subset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=256, shuffle=False)
    
    X_train, y_train = extract_representations(encoder, train_loader, device)
    X_test_clean, y_test = extract_representations(encoder, test_loader, device)
    
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_train, y_train)
    
    clean_f1 = f1_score(y_test, clf.predict(X_test_clean), average='macro')
    clean_acc = accuracy_score(y_test, clf.predict(X_test_clean))
    
    print(f"Clean baseline: F1={clean_f1:.4f}, Acc={clean_acc:.4f}")
    
    # 2. Test under noise conditions
    noise_types = ['bw', 'pli', 'emg']
    snr_levels = [20, 10, 5, 0]  # dB (20=mild, 0=severe)
    
    print(f"\n{'Noise':>6} | {'SNR(dB)':>8} | {'F1':>8} | {'Δ F1':>8} | {'Acc':>8}")
    print("-" * 50)
    
    results = []
    for noise_type in noise_types:
        for snr in snr_levels:
            noisy_test = NoisyECGDataset(test_subset, noise_type=noise_type, snr_db=snr)
            noisy_loader = DataLoader(noisy_test, batch_size=256, shuffle=False)
            
            X_test_noisy, _ = extract_representations(encoder, noisy_loader, device)
            
            noisy_f1 = f1_score(y_test, clf.predict(X_test_noisy), average='macro')
            noisy_acc = accuracy_score(y_test, clf.predict(X_test_noisy))
            delta_f1 = noisy_f1 - clean_f1
            
            print(f"{noise_type:>6} | {snr:>8} | {noisy_f1:>8.4f} | {delta_f1:>+8.4f} | {noisy_acc:>8.4f}")
            results.append({
                'noise_type': noise_type, 'snr_db': snr,
                'f1': noisy_f1, 'delta_f1': delta_f1, 'accuracy': noisy_acc
            })
    
    # Save results
    if args.output:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")
    
    # Summary
    avg_delta = np.mean([r['delta_f1'] for r in results])
    print(f"\nAverage F1 degradation across all noise conditions: {avg_delta:+.4f}")
    if abs(avg_delta) < 0.05:
        print("Verdict: Model is HIGHLY robust to clinical noise.")
    elif abs(avg_delta) < 0.10:
        print("Verdict: Model shows moderate noise robustness.")
    else:
        print("Verdict: Model is sensitive to noise — consider noise-augmented training.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate noise robustness of ECG representations")
    parser.add_argument('--encoder', type=str, default='resnet1d', choices=['resnet1d', 'wavkan'])
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    main(args)
