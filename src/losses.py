"""
PA-SSL: Contrastive Losses

NT-Xent loss for SimCLR-style contrastive learning, with support for
combined augmentation + temporal adjacency contrastive objectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
    
    Standard SimCLR contrastive loss. For a batch of B pairs,
    creates a 2B × 2B similarity matrix and maximizes agreement
    between positive pairs while pushing apart negatives.
    """
    
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_i, z_j):
        """
        Args:
            z_i: (B, D) projections from view 1
            z_j: (B, D) projections from view 2
        
        Returns:
            Scalar loss
        """
        batch_size = z_i.shape[0]
        N = 2 * batch_size
        device = z_i.device
        
        z = torch.cat([z_i, z_j], dim=0)  # (2B, D)
        z = F.normalize(z, dim=1)
        
        sim_matrix = torch.matmul(z, z.T) / self.temperature  # (2B, 2B)
        
        # Mask out self-similarity
        mask = torch.eye(N, dtype=torch.bool, device=device)
        sim_matrix.masked_fill_(mask, -1e4)  # Use -1e4 (not -1e9) for AMP float16 compatibility
        
        # Positive pairs: (i, i+B) and (i+B, i)
        targets = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(0, batch_size, device=device),
        ])
        
        return F.cross_entropy(sim_matrix, targets)


class CombinedContrastiveLoss(nn.Module):
    """
    Combined contrastive loss for PA-SSL.
    
    L_total = α · L_augmentation + β · L_temporal
    
    - L_augmentation: NT-Xent between two augmented views of the same beat
    - L_temporal: NT-Xent between a beat and its temporal neighbor
    """
    
    def __init__(self, temperature=0.5, alpha=1.0, beta=0.5):
        """
        Args:
            temperature: Temperature for NT-Xent
            alpha: Weight for augmentation contrast
            beta: Weight for temporal contrast
        """
        super().__init__()
        self.ntxent = NTXentLoss(temperature)
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, z_view1, z_view2, z_temporal=None, has_temporal=None):
        """
        Args:
            z_view1: (B, D) projections from augmented view 1
            z_view2: (B, D) projections from augmented view 2
            z_temporal: (B, D) projections from temporal neighbor (optional)
            has_temporal: (B,) boolean mask indicating valid temporal pairs
        
        Returns:
            total_loss, loss_aug, loss_temporal
        """
        # Augmentation contrastive loss (always computed)
        loss_aug = self.ntxent(z_view1, z_view2)
        
        # Temporal contrastive loss (optional)
        loss_temporal = torch.tensor(0.0, device=z_view1.device)
        
        if z_temporal is not None and self.beta > 0:
            if has_temporal is not None:
                # Only compute temporal loss for samples that have valid neighbors
                valid_mask = has_temporal.bool()
                if valid_mask.any():
                    z_v1_valid = z_view1[valid_mask]
                    z_temp_valid = z_temporal[valid_mask]
                    if len(z_v1_valid) > 1:  # Need at least 2 for contrastive
                        loss_temporal = self.ntxent(z_v1_valid, z_temp_valid)
            else:
                loss_temporal = self.ntxent(z_view1, z_temporal)
        
        total_loss = self.alpha * loss_aug + self.beta * loss_temporal
        
        return total_loss, loss_aug, loss_temporal
