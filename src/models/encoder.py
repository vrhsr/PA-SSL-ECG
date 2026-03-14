"""
PA-SSL: Encoder Architectures

Provides two backbone encoders (ResNet1D, WavKAN) with detachable
projection heads for contrastive learning and classification heads
for downstream evaluation.

Both encoders follow the same interface:
  - encoder.forward(x) → representation (B, repr_dim)
  - encoder.forward(x, return_projection=True) → projection (B, proj_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ═══════════════════════════════════════════════════════════════════════════════
# PROJECTION HEAD (shared by all encoders)
# ═══════════════════════════════════════════════════════════════════════════════

class ProjectionHead(nn.Module):
    """MLP projection head for SimCLR-style contrastive learning, with optional metadata conditioning."""
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=128, metadata_dim=0):
        super().__init__()
        self.metadata_dim = metadata_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim + metadata_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x, metadata=None):
        if self.metadata_dim > 0 and metadata is not None:
            x = torch.cat([x, metadata], dim=1)
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════════
# RESNET1D ENCODER
# ═══════════════════════════════════════════════════════════════════════════════

class ResBlock1D(nn.Module):
    """Residual block for 1D signals."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, 
                               stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, 
                               stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class ResNet1DEncoder(nn.Module):
    """
    1D ResNet encoder for ECG signals.
    
    Architecture: Conv stem → 4 ResBlocks → Global Average Pool → repr_dim
    Output dimension: 512
    """
    
    def __init__(self, in_channels=1, repr_dim=512, proj_dim=128, **kwargs):
        super().__init__()
        
        self.repr_dim = repr_dim
        
        # Convolutional stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual blocks
        self.layer1 = ResBlock1D(64, 64)
        self.layer2 = ResBlock1D(64, 128, stride=2)
        self.layer3 = ResBlock1D(128, 256, stride=2)
        self.layer4 = ResBlock1D(256, repr_dim, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Projection head (for SSL)
        self.projection_head = ProjectionHead(repr_dim, hidden_dim=256, output_dim=proj_dim, metadata_dim=kwargs.get('metadata_dim', 0))
        
        # Classification head (for downstream)
        self.classifier = None  # Set via set_classifier()
    
    def set_classifier(self, num_classes):
        """Attach a classification head for downstream evaluation."""
        self.classifier = nn.Linear(self.repr_dim, num_classes)
        return self
    
    def encode(self, x):
        """Extract representation without projection/classification."""
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, L) → (B, 1, L)
        
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # (B, repr_dim)
        return x
    
    def forward(self, x, return_projection=False, metadata=None):
        """
        Forward pass.
        
        Args:
            x: (B, 1, 250) or (B, 250)
            return_projection: If True, returns projection head output (for SSL)
                             If False, returns representation or classification
            metadata: Optional (B, metadata_dim) tensor
        """
        h = self.encode(x)
        
        if return_projection:
            return self.projection_head(h, metadata)
        
        if self.classifier is not None:
            return self.classifier(h)
        
        return h


# ═══════════════════════════════════════════════════════════════════════════════
# WAVKAN ENCODER
# ═══════════════════════════════════════════════════════════════════════════════

class WaveletLinear(nn.Module):
    """Wavelet-based linear layer (from WavKAN)."""
    
    def __init__(self, in_features, out_features, wavelet_type='mexican_hat'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type
        
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))
        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        nn.init.uniform_(self.translation, -1, 1)
        nn.init.uniform_(self.scale, 0.1, 1)
    
    def forward(self, x):
        x_expanded = x.unsqueeze(1).expand(-1, self.out_features, -1)
        s = (x_expanded - self.translation) / (self.scale + 1e-8)
        
        if self.wavelet_type == 'mexican_hat':
            wavelet = (1 - s**2) * torch.exp(-0.5 * s**2)
        elif self.wavelet_type == 'morlet':
            wavelet = torch.cos(5 * s) * torch.exp(-0.5 * s**2)
        else:
            raise ValueError(f"Unknown wavelet type: {self.wavelet_type}")
        
        return (self.weights * wavelet).sum(dim=2)


class Conv1DStem(nn.Module):
    """Lightweight 1D CNN feature extractor preserving temporal structure."""
    
    def __init__(self, out_dim=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(out_dim // 4),
        )
        self.out_features = 128 * (out_dim // 4)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        features = self.stem(x)
        return features.view(features.size(0), -1)


class WavKANEncoder(nn.Module):
    """
    WavKAN encoder for ECG signals.
    
    Architecture: Conv1D stem → WaveletLinear layers → repr_dim
    Output dimension: 64 (configurable via hidden_dim)
    """
    
    def __init__(self, input_dim=250, repr_dim=64, depth=3,
                 wavelet_type='mexican_hat', proj_dim=128, **kwargs):
        super().__init__()
        
        self.repr_dim = repr_dim
        
        # Conv1D stem
        self.conv_stem = Conv1DStem(out_dim=repr_dim)
        kan_input_dim = self.conv_stem.out_features
        
        # WaveletLinear layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        self.layers.append(WaveletLinear(kan_input_dim, repr_dim, wavelet_type))
        self.norms.append(nn.LayerNorm(repr_dim))
        
        for _ in range(depth - 2):
            self.layers.append(WaveletLinear(repr_dim, repr_dim, wavelet_type))
            self.norms.append(nn.LayerNorm(repr_dim))
        
        self.layers.append(WaveletLinear(repr_dim, repr_dim, wavelet_type))
        self.norms.append(nn.LayerNorm(repr_dim))
        
        # Projection head (for SSL)
        self.projection_head = ProjectionHead(repr_dim, hidden_dim=128, output_dim=proj_dim, metadata_dim=kwargs.get('metadata_dim', 0))
        
        # Classification head (for downstream)
        self.classifier = None
    
    def set_classifier(self, num_classes):
        self.classifier = nn.Linear(self.repr_dim, num_classes)
        return self
    
    def encode(self, x):
        if x.dim() == 3:
            features = self.conv_stem(x)
        elif x.dim() == 2:
            features = self.conv_stem(x.unsqueeze(1))
        else:
            features = x.view(x.size(0), -1)
        
        for layer, norm in zip(self.layers, self.norms):
            features = norm(F.silu(layer(features)))
        
        return features
    
    def forward(self, x, return_projection=False, metadata=None):
        h = self.encode(x)
        
        if return_projection:
            return self.projection_head(h, metadata)
        
        if self.classifier is not None:
            return self.classifier(h)
        
        return h


# ═══════════════════════════════════════════════════════════════════════════════
# SE-RESNET1D-34 ENCODER
# ═══════════════════════════════════════════════════════════════════════════════

class SqueezeExcitation1D(nn.Module):
    """Squeeze-and-Excitation block for 1D signals."""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class SEResBlock1D(nn.Module):
    """Residual block for 1D signals with Squeeze-and-Excitation."""
    
    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, 
                               stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, 
                               stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SqueezeExcitation1D(out_channels, reduction)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += identity
        return self.relu(out)

class SEResNet1D34Encoder(nn.Module):
    """
    1D SE-ResNet-34 encoder for ECG signals.
    
    Architecture: Conv stem → 4 layers of SE-ResBlocks (3, 4, 6, 3) → Global Average Pool → repr_dim
    Output dimension: 512
    """
    
    def __init__(self, in_channels=1, repr_dim=512, proj_dim=128, **kwargs):
        super().__init__()
        
        self.repr_dim = repr_dim
        
        # Convolutional stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual layers (34-layer equivalent: 3, 4, 6, 3 blocks)
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, repr_dim, 3, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Projection head (for SSL)
        self.projection_head = ProjectionHead(repr_dim, hidden_dim=256, output_dim=proj_dim, metadata_dim=kwargs.get('metadata_dim', 0))
        
        # Classification head (for downstream)
        self.classifier = None
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(SEResBlock1D(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(SEResBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)
        
    def set_classifier(self, num_classes):
        self.classifier = nn.Linear(self.repr_dim, num_classes)
        return self
    
    def encode(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def forward(self, x, return_projection=False, metadata=None):
        h = self.encode(x)
        
        if return_projection:
            return self.projection_head(h, metadata)
        
        if self.classifier is not None:
            return self.classifier(h)
        
        return h


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def build_encoder(name='resnet1d', **kwargs):
    """
    Build encoder by name.
    
    Args:
        name: 'resnet1d' or 'wavkan'
        **kwargs: passed to encoder constructor
    """
    encoders = {
        'resnet1d': ResNet1DEncoder,
        'se_resnet1d34': SEResNet1D34Encoder,
        'wavkan': WavKANEncoder,
    }
    
    if name not in encoders:
        raise ValueError(f"Unknown encoder: {name}. Choose from {list(encoders.keys())}")
    
    return encoders[name](**kwargs)
