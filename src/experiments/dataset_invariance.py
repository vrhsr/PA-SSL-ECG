"""
Dataset-Invariance Diagnostic Script

Proves that learned representations are independent of the hospital/dataset origin.
- Loads multiple pre-processed datasets (e.g., PTB-XL, MIT-BIH, Chapman).
- Extracts embeddings via a frozen encoder.
- Trains a logistic regression model to predict which dataset a beat came from.

EXPECTED RESULT:
- Supervised / SimCLR: High accuracy (clusters by dataset / machine).
- PA-SSL (Hybrid): Accuracy closer to random guess (33%), proving true generalization.
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from src.models.encoder import build_encoder
from src.evaluate import extract_representations
from src.data.ecg_dataset import ECGBeatDataset
from torch.utils.data import DataLoader

def load_and_extract(dataset_path, dataset_id, encoder, device, max_samples=10000):
    """
    Loads a dataset and extracts representations, capping at max_samples to balance classes.
    """
    print(f"Loading {dataset_path} (ID: {dataset_id})...")
    dataset = ECGBeatDataset(dataset_path)
    
    # Randomly subset if necessary to balance
    indices = np.random.permutation(len(dataset))
    if len(indices) > max_samples:
        indices = indices[:max_samples]
    
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    
    X_reps, _ = extract_representations(encoder, loader, device, max_batches=None)
    y_domains = np.full((len(X_reps),), dataset_id)
    
    return X_reps, y_domains

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dataset-Invariance Diagnostic using Device: {device}")
    
    # 1. Load Encoder
    encoder = build_encoder(args.encoder, proj_dim=128).to(device)
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        # Handle hybrid vs standard ckpt
        if 'encoder_state_dict' in ckpt:
            encoder.load_state_dict(ckpt['encoder_state_dict'])
        elif 'model_state_dict' in ckpt:
            # Try to rip out encoder from HybridMAE state dict if needed,
            # but usually 'encoder_state_dict' is provided.
            encoder.load_state_dict(ckpt['encoder_state_dict'])
        else:
            encoder.load_state_dict(ckpt)
            
    encoder.eval()
    
    # 2. Extract Representations across datasets
    all_X, all_y = [], []
    for idx, dpath in enumerate(args.datasets):
        if not os.path.exists(dpath):
            print(f"Warning: Dataset path {dpath} does not exist. Skipping.")
            continue
            
        X_domain, y_domain = load_and_extract(dpath, idx, encoder, device, max_samples=args.samples_per_dataset)
        all_X.append(X_domain)
        all_y.append(y_domain)
        print(f"  Extracted {len(X_domain)} samples.")

    if len(all_X) < 2:
        print("Error: Need at least 2 valid datasets to evaluate dataset-invariance.")
        return
        
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    
    # 3. Train Dataset Origin Classifier
    print(f"Training dataset classifier on {len(X)} total samples ({len(args.datasets)} classes)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)
    
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=args.seed)
    clf.fit(X_train, y_train)
    
    # 4. Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print("DATASET-INVARIANCE RESULTS")
    print("="*50)
    print(f"Model: {args.encoder}")
    print(f"Checkpoint: {args.checkpoint if args.checkpoint else 'Random Init'}")
    print(f"Target: Predicting which hospital/dataset the ECG came from.")
    print("-" * 50)
    print(f"Dataset Origin Classification Accuracy: {acc:.4f}")
    
    random_guess = 1.0 / len(all_X)
    print(f"Random Guess Baseline: {random_guess:.4f}")
    
    if acc > 0.8:
        print("Verdict: Model learned spurious domain features (Low Invariance).")
    elif acc < random_guess * 1.5:
        print("Verdict: Model embeddings are highly domain-agnostic (High Invariance). EXCELLENT!")
    else:
        print("Verdict: Moderate domain invariance.")
        
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred, target_names=[os.path.basename(path).split('_')[0] for path in args.datasets if os.path.exists(path)]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate domain-invariance of ECG embeddings")
    parser.add_argument('--encoder', type=str, default='resnet1d', choices=['resnet1d', 'wavkan'])
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to encoder checkpoint. If none, uses random init.')
    parser.add_argument('--datasets', nargs='+', type=str, required=True, help='List of CSV dataset paths (e.g. data/ptbxl_processed.csv data/mitbih_processed.csv)')
    parser.add_argument('--samples_per_dataset', type=int, default=5000, help='Max samples to draw from each dataset')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    main(args)
