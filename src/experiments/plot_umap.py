"""
PA-SSL: Dual UMAP Plotter

Extracts representations from an encoder and generates a side-by-side UMAP plot:
Left: Colored by Dataset Origin (PTB-XL, MIT-BIH, Chapman)
Right: Colored by Condition (Normal, Abnormal)

This acts as a visual and qualitative proof of domain invariance.
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

from src.models.encoder import build_encoder
from src.data.ecg_dataset import ECGBeatDataset
from src.plotting import plot_umap_dual_colored


@torch.no_grad()
def extract_reprs_and_labels(encoder, dataset, device, max_samples=5000):
    encoder.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    all_reprs = []
    all_conds = []
    all_dsets = []
    
    count = 0
    for batch in loader:
        signals = batch['signal'].to(device) if isinstance(batch, dict) else batch[0].to(device)
        conds = batch['label'] if isinstance(batch, dict) else batch[1]
        
        # We need dataset traces. The ECGBeatDataset provides them.
        if isinstance(dataset, ConcatDataset):
            # Complex to trace exact dataset name without custom logic, but if provided as combined
            pass
            
        reprs = encoder.encode(signals)
        all_reprs.append(reprs.cpu().numpy())
        all_conds.append(conds.numpy())
        
        count += len(signals)
        if count >= max_samples:
            break
            
    return np.concatenate(all_reprs)[:max_samples], np.concatenate(all_conds)[:max_samples]


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"UMAP Generation using Device: {device}")
    
    # 1. Load Data
    print("Loading datasets...")
    datasets = []
    for f in args.datasets:
        ds = ECGBeatDataset(f)
        datasets.append(ds)
        
    # Standardize length across datasets to balance the plot
    min_len = min([len(ds) for ds in datasets])
    min_len = min(min_len, args.max_samples // len(datasets))
    
    # Build a combined dataset with explicit string labels for the source
    combined_reprs = []
    combined_conds = []
    combined_dsets = []
    
    # 2. Load Models
    method_name = f"{args.encoder.upper()}"
    encoder = build_encoder(args.encoder, proj_dim=128).to(device)
    if args.checkpoint:
        method_name += " (PA-SSL)"
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if 'encoder_state_dict' in ckpt:
            encoder.load_state_dict(ckpt['encoder_state_dict'])
        else:
            encoder.load_state_dict(ckpt)
    else:
        method_name += " (Random Init)"
        
    encoder.eval()
    
    # 3. Extract and interleave
    for i, ds in enumerate(datasets):
        subset = torch.utils.data.Subset(ds, np.random.choice(len(ds), min_len, replace=False))
        r, c = extract_reprs_and_labels(encoder, subset, device, max_samples=min_len)
        
        ds_name = args.datasets[i].replace('\\', '/').split('/')[-1].replace('_processed.csv', '').upper()
        d_labels = [ds_name] * len(r)
        
        combined_reprs.append(r)
        combined_conds.append(c)
        combined_dsets.extend(d_labels)
        
    combined_reprs = np.concatenate(combined_reprs)
    combined_conds = np.concatenate(combined_conds)
    combined_dsets = np.array(combined_dsets)
    
    # 4. Generate Plot
    fig = plot_umap_dual_colored(
        representations=combined_reprs,
        labels_condition=combined_conds,
        labels_dataset=combined_dsets,
        method_name=method_name,
        save_path=args.output
    )
    
    if fig is None:
        print("Failed to generate UMAP. Make sure umap-learn is installed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default='resnet1d')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--max_samples', type=int, default=5000)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    main(args)
