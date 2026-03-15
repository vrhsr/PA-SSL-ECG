"""
PA-SSL: Embedding Collapse Diagnostic

Measures whether the self-supervised representations are rich or have 
suffered from dimension collapse.

Metrics:
1. Average Pairwise Cosine Similarity: Measures how "bundled" or similar 
   the embeddings are. High similarity (>0.9) suggests collapse.
2. Embedding Variance: Variance of normalized embeddings across the batch.
   Higher variance indicates the model is using more of the representation space.
3. Singular Value Spectrum: (Optional) Decay rate of singular values 
   of the embedding matrix.
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.models.encoder import build_encoder
from src.evaluate import extract_representations
from src.data.ecg_dataset import ECGBeatDataset

def analyze_embeddings(embeddings):
    """
    Args:
        embeddings: numpy array of shape (N, D)
    """
    # Convert to torch for convenience
    z = torch.from_numpy(embeddings).float()
    N, D = z.shape
    
    # 1. Normalize for cosine similarity
    z_norm = F.normalize(z, p=2, dim=1)
    
    # 2. Average Pairwise Cosine Similarity
    # (z_norm @ z_norm.T) is (N, N) matrix of similarities
    # We take a sample if N is very large to avoid OOM
    if N > 2000:
        sample_indices = np.random.choice(N, 2000, replace=False)
        z_sample = z_norm[sample_indices]
    else:
        z_sample = z_norm
        
    sim_matrix = torch.mm(z_sample, z_sample.t())
    # Mask out diagonal
    mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device).bool()
    sim_matrix.masked_fill_(mask, 0)
    avg_cosine_sim = sim_matrix.sum() / (sim_matrix.size(0) * (sim_matrix.size(0) - 1))
    
    # 3. Embedding Variance (Across batch, per dimension)
    # Variance of the normalized embeddings
    var_per_dim = torch.var(z_norm, dim=0)
    avg_variance = torch.mean(var_per_dim).item()
    
    # 4. Collapse Metric (Log-sum of variances)
    # If many variances are near zero, this will be very low
    log_variance = torch.log(var_per_dim + 1e-8).mean().item()
    
    return {
        'avg_cosine_sim': avg_cosine_sim.item(),
        'avg_variance': avg_variance,
        'log_variance': log_variance,
        'num_samples': N,
        'dim': D
    }

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Embedding Analysis using Device: {device}")
    
    # Load encoder
    encoder = build_encoder(args.encoder, proj_dim=128).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if 'encoder_state_dict' in ckpt:
            encoder.load_state_dict(ckpt['encoder_state_dict'])
        else:
            encoder.load_state_dict(ckpt)
    encoder.eval()
    
    # Load dataset
    dataset = ECGBeatDataset(args.data_file)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    print(f"Extracting representations from {len(dataset)} samples...")
    X, _ = extract_representations(encoder, loader, device)
    
    results = analyze_embeddings(X)
    
    print("\n" + "="*40)
    print(" EMBEDDING COLLAPSE DIAGNOSIS")
    print("="*40)
    print(f" Encoder:         {args.encoder}")
    print(f" Dim:             {results['dim']}")
    print(f" Avg Cosine Sim:  {results['avg_cosine_sim']:.4f}  (Lower is better)")
    print(f" Avg Variance:    {results['avg_variance']:.4f}  (Higher is better)")
    print(f" Log-Avg Var:     {results['log_variance']:.4f}")
    print("="*40)
    
    # Verdicts
    if results['avg_cosine_sim'] > 0.9:
        print("Verdict: SEVERE COLLAPSE - All embeddings are nearly identical.")
    elif results['avg_cosine_sim'] > 0.7:
        print("Verdict: MODERATE COLLAPSE - Representations lack diversity.")
    else:
        print("Verdict: HEALTHY - Representations are well-distributed.")
        
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze embedding collapse in SSL models")
    parser.add_argument('--encoder', type=str, default='resnet1d', choices=['resnet1d', 'wavkan'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    main(args)
