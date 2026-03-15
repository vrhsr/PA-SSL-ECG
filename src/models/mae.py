"""
PA-SSL: Masked Autoencoder (MAE) Components

Implements the Physiology-Aware Masking strategy and the 1D Hybrid Decoder.
"""

import torch
import torch.nn as nn
import numpy as np

class PhysioAwareMasker(nn.Module):
    """
    Masking Generator for 1D ECG signals with configurable strategy.
    
    Supports three masking strategies for ablation:
      - 'physio_aware' (default): Multi-span contiguous masking that avoids QRS complex.
      - 'contiguous': Multi-span contiguous masking WITHOUT QRS protection.
      - 'random': Independent per-sample Bernoulli masking (standard MAE baseline).
    
    Since our 2.5s windows are strictly R-peak centered, the QRS is located exactly
    at the center index (L//2 = 125). 
    
    Default: mask_ratio=0.60 (higher ratios force richer representations).
    """
    def __init__(self, mask_ratio=0.60, qrs_avoidance_prob=0.8, seq_len=250, 
                 qrs_radius=30, num_spans=3, masking_strategy='physio_aware'):
        """
        Args:
            mask_ratio: Fraction of signal to mask (default 0.60)
            qrs_avoidance_prob: Probability of avoiding QRS per span (only for 'physio_aware')
            seq_len: Signal length (default 250)
            qrs_radius: QRS protection zone radius in samples (default 30)
            num_spans: Number of contiguous spans (for 'physio_aware' and 'contiguous')
            masking_strategy: One of 'physio_aware', 'contiguous', 'random'
        """
        super().__init__()
        assert masking_strategy in ('physio_aware', 'contiguous', 'random'), \
            f"Unknown masking_strategy: {masking_strategy}"
        
        self.mask_ratio = mask_ratio
        self.qrs_avoidance_prob = qrs_avoidance_prob
        self.seq_len = seq_len
        self.qrs_radius = qrs_radius
        self.center_idx = seq_len // 2
        self.num_spans = num_spans
        self.masking_strategy = masking_strategy
        
        # Each span covers (mask_ratio / num_spans) of the signal
        self.span_len = max(1, int(seq_len * mask_ratio / num_spans))
        
    def forward(self, x):
        """
        Applies masking according to the configured strategy.
        
        Args:
            x: Input tensor of shape (B, C, L) where L = 250
            
        Returns:
            masked_x: The input tensor with masked positions zeroed
            masks: Boolean mask of shape (B, L) — True where masked
        """
        B, C, L = x.shape
        device = x.device
        
        if self.masking_strategy == 'random':
            return self._random_mask(x, B, C, L, device)
        else:
            return self._span_mask(x, B, C, L, device)
    
    def _random_mask(self, x, B, C, L, device):
        """IID Bernoulli masking — standard MAE baseline."""
        masked_x = x.clone()
        masks = torch.rand((B, L), device=device) < self.mask_ratio
        masked_x[:, :, :] = masked_x * (~masks).unsqueeze(1).float()
        return masked_x, masks
    
    def _span_mask(self, x, B, C, L, device):
        """Multi-span contiguous masking (with or without QRS avoidance)."""
        masked_x = x.clone()
        masks = torch.zeros((B, L), dtype=torch.bool, device=device)
        
        qrs_start = self.center_idx - self.qrs_radius
        qrs_end = self.center_idx + self.qrs_radius
        use_qrs_avoidance = (self.masking_strategy == 'physio_aware')
        
        for i in range(B):
            for s in range(self.num_spans):
                # Decide if this span uses physiology-aware masking
                use_physio = use_qrs_avoidance and (torch.rand(1).item() < self.qrs_avoidance_prob)
                
                valid_start = False
                attempts = 0
                
                while not valid_start and attempts < 15:
                    start_idx = torch.randint(0, max(1, L - self.span_len), (1,)).item()
                    end_idx = min(start_idx + self.span_len, L)
                    
                    if use_physio:
                        overlap = not (end_idx <= qrs_start or start_idx >= qrs_end)
                        valid_start = not overlap
                    else:
                        valid_start = True
                        
                    attempts += 1
                    
                # Fallback: random placement
                if not valid_start:
                    start_idx = torch.randint(0, max(1, L - self.span_len), (1,)).item()
                    end_idx = min(start_idx + self.span_len, L)
                    
                masked_x[i, :, start_idx:end_idx] = 0.0
                masks[i, start_idx:end_idx] = True
            
        return masked_x, masks


class MAEDecoder1D(nn.Module):
    """
    Lightweight 1D Decoder to reconstruct the 250-sample sequence from the 256D representation.
    Used for the MAE branch of the Hybrid SSL framework.
    """
    def __init__(self, repr_dim=256, out_channels=1, seq_len=250):
        super().__init__()
        self.seq_len = seq_len
        
        # We start by expanding the 256D representation into a spatial feature map
        self.initial_expansion_len = 10
        self.expand = nn.Linear(repr_dim, 128 * self.initial_expansion_len)
        
        # A sequence of transposed convolutions to upsample back to 250 samples
        # Length progression: 10 -> 25 -> 62 -> 125 -> 250
        self.decoder_blocks = nn.Sequential(
            # 10 -> 25
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=0, output_padding=0), 
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            # 25 -> 62
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=0, output_padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            
            # 62 -> 125
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=1, output_padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            
            # 125 -> 250
            nn.ConvTranspose1d(16, out_channels, kernel_size=4, stride=2, padding=1, output_padding=0),
        )

    def forward(self, h):
        """
        Reconstructs the original signal from the latent representation.
        
        Args:
            h: Encoded representation (B, repr_dim)
            
        Returns:
            reconstructed: Tensor of shape (B, 1, 250)
        """
        B = h.size(0)
        
        # (B, repr_dim) -> (B, 128 * 10)
        x = self.expand(h)
        # (B, 128, 10)
        x = x.view(B, 128, self.initial_expansion_len)
        
        # Upsample blocks
        reconstructed = self.decoder_blocks(x)
        
        # Ensure exact output length (just in case of round/pad issues)
        if reconstructed.size(-1) != self.seq_len:
            reconstructed = torch.nn.functional.interpolate(
                reconstructed, size=self.seq_len, mode='linear', align_corners=False
            )
            
        return reconstructed


class HybridMAE(nn.Module):
    """
    Wrapper model that encapsulates the Encoder, Masker, and Decoder for Hybrid SSL.
    """
    def __init__(self, encoder, repr_dim=256, mask_ratio=0.15, qrs_avoidance_prob=0.8):
        super().__init__()
        self.encoder = encoder
        self.masker = PhysioAwareMasker(mask_ratio=mask_ratio, qrs_avoidance_prob=qrs_avoidance_prob)
        self.decoder = MAEDecoder1D(repr_dim=repr_dim)
        
    def forward_mae(self, x):
        """
        Process the MAE reconstruction path.
        """
        # Ensure input has channel dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # 1. Masking
        masked_x, masks = self.masker(x)
        
        # 2. Encoding (extract representation)
        h = self.encoder.encode(masked_x)
        
        # 3. Decoding
        reconstructed_x = self.decoder(h)
        
        return reconstructed_x, masks, masked_x
        
    def forward_contrastive(self, x1, x2, metadata=None):
        """
        Process the Contrastive path (returns projections).
        """
        z1 = self.encoder(x1, return_projection=True, metadata=metadata)
        z2 = self.encoder(x2, return_projection=True, metadata=metadata)
        return z1, z2

