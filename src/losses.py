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


class VICRegLoss(nn.Module):
    """
    VICReg: Variance-Invariance-Covariance Regularization.
    """
    def __init__(self, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z_i, z_j):
        # invariance loss
        repr_loss = F.mse_loss(z_i, z_j)

        z_i = z_i - z_i.mean(dim=0)
        z_j = z_j - z_j.mean(dim=0)

        # variance loss
        std_i = torch.sqrt(z_i.var(dim=0) + 1e-04)
        std_j = torch.sqrt(z_j.var(dim=0) + 1e-04)
        std_loss = torch.mean(F.relu(1 - std_i)) / 2 + torch.mean(F.relu(1 - std_j)) / 2

        # covariance loss
        N = z_i.size(0)
        D = z_i.size(1)
        cov_i = (z_i.T @ z_i) / (N - 1)
        cov_j = (z_j.T @ z_j) / (N - 1)
        cov_loss = self.off_diagonal(cov_i).pow_(2).sum() / D + self.off_diagonal(cov_j).pow_(2).sum() / D

        loss = self.sim_coeff * repr_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss
        return loss

class CombinedContrastiveLoss(nn.Module):
    """
    Combined contrastive loss for PA-SSL.
    
    L_total = α · L_augmentation + β · L_temporal
    
    Supports switching L_augmentation objective via loss_type.
    """
    
    def __init__(self, temperature=0.5, alpha=1.0, beta=0.5, loss_type='ntxent'):
        """
        Args:
            temperature: Temperature for NT-Xent
            alpha: Weight for augmentation contrast
            beta: Weight for temporal contrast
            loss_type: 'ntxent' or 'vicreg'
        """
        super().__init__()
        self.loss_type = loss_type
        if loss_type == 'ntxent':
            self.aug_loss_fn = NTXentLoss(temperature)
        elif loss_type == 'vicreg':
            self.aug_loss_fn = VICRegLoss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
            
        self.temporal_loss_fn = NTXentLoss(temperature) # Temporal always uses cross-entropy similarity
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, z_view1, z_view2, z_temporal=None, has_temporal=None):
        """
        Args:
            z_view1: (B, D) projections from augmented view 1
            z_view2: (B, D) projections from augmented view 2
            z_temporal: (B, D) projections from temporal neighbor (optional)
            has_temporal: (B,) boolean mask indicating valid temporal pairs
        """
        # Augmentation contrastive loss (always computed)
        loss_aug = self.aug_loss_fn(z_view1, z_view2)
        
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
                        loss_temporal = self.temporal_loss_fn(z_v1_valid, z_temp_valid)
            else:
                loss_temporal = self.temporal_loss_fn(z_view1, z_temporal)
        
        total_loss = self.alpha * loss_aug + self.beta * loss_temporal
        
        return total_loss, loss_aug, loss_temporal
