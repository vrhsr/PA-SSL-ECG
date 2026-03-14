import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size=3, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size=3, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
    
    def forward(self, x):
        res = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + res

class TS2VecEncoder(nn.Module):
    """
    Official TS2Vec architecture (Yue et al., 2022).
    A stack of dilated convolutions with residual connections.
    """
    def __init__(self, input_dims=1, output_dims=320, hidden_dims=64, depth=10, mask_mode='binomial'):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = nn.ModuleList([
            ConvBlock(hidden_dims, hidden_dims, dilation=2**i)
            for i in range(depth)
        ])
        self.repr_dropout = nn.Dropout(p=0.1)
        
    def forward(self, x, mask=None):
        # TS2Vec expects (B, T, C)
        if x.dim() == 2:
            x = x.unsqueeze(2)  # (B, T, 1)
        elif x.size(1) == 1 and x.size(2) > 1:
            x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
            
        x = self.input_fc(x)  # (B, T, hidden_dims)
        x = x.transpose(1, 2)  # (B, hidden_dims, T)
        
        for conv in self.feature_extractor:
            x = conv(x)
            
        x = x.transpose(1, 2)  # (B, T, hidden_dims)
        
        # Max pool over time according to official code to get instance-level representation
        # Actually TS2Vec outputs a representation for each timestep, returning (B, T, hidden)
        return x
        
    def encode(self, x):
        # Extracted flat representation for linear probing
        out = self.forward(x)  # (B, T, hidden)
        # Max pool over time
        return F.max_pool1d(out.transpose(1, 2), kernel_size=out.size(1)).squeeze(2)

class HierarchicalContrastiveLoss(nn.Module):
    """Hierarchical contrastive loss from TS2Vec paper."""
    def __init__(self, alpha=0.5, temp=0.1):
        super().__init__()
        self.alpha = alpha
        self.temp = temp

    def forward(self, z1, z2):
        # z1, z2: (B, T, C)
        B, T = z1.size(0), z1.size(1)
        if B == 1:
            return z1.new_tensor(0.)
            
        loss = torch.tensor(0., device=z1.device)
        d = 0
        while z1.size(1) > 1:
            if self.alpha != 0:
                loss += self.alpha * self.instance_contrastive_loss(z1, z2)
            if d >= 1: # We pool starting from scale 2
                z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
                z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
            d += 1
            
        if z1.size(1) == 1:
            if self.alpha != 0:
                loss += self.alpha * self.instance_contrastive_loss(z1, z2)
                
        return loss / d
        
    def instance_contrastive_loss(self, z1, z2):
        B, T = z1.size(0), z1.size(1)
        if B == 1:
            return z1.new_tensor(0.)
        # z1: (B, T, C)
        z = torch.cat([z1, z2], dim=0)  # (2B, T, C)
        z = z.transpose(0, 1)  # (T, 2B, C)
        sim = torch.matmul(z, z.transpose(1, 2)) / self.temp  # (T, 2B, 2B)
        
        # Labels for contrastive loss
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)
        
        target = torch.arange(B, device=z1.device)
        target = torch.cat([target + B - 1, target])
        
        loss = logits[:, torch.arange(2*B), target].mean()
        return loss
