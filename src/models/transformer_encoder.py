"""
PA-SSL: ECG Transformer Encoder

1D Vision Transformer (ViT) adapted for single-lead ECG beats.
Architecture: Patch embedding → CLS token + positional encoding → Transformer layers → head

Design choices:
- patch_size=25 → 10 patches from a 250-sample signal (non-overlapping)
- d_model=256: matches the ResNet1D / WavKAN repr_dim for fair comparison
- 6 layers, 8 heads: ~3M parameters (parity with other encoders in the suite)
- GELU activation, pre-LN (more stable than post-LN for smaller datasets)

Interface matches all other PA-SSL encoders:
    enc.encode(x)                       → (B, repr_dim)
    enc.forward(x, return_projection)   → (B, proj_dim) or (B, repr_dim)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ECGTransformerEncoder(nn.Module):
    """
    1D ViT-style encoder for ECG beats.

    Processes a (B, 1, 250) ECG beat as 10 non-overlapping patches of size 25,
    prepends a CLS token, applies positional encoding, then runs N transformer
    layers. The CLS token output is the global representation.

    Parameters
    ----------
    signal_length : int
        Length of the input ECG signal (default 250 samples)
    patch_size : int
        Size of each non-overlapping patch (default 25 → 10 patches)
    d_model : int
        Transformer model dimension (default 256)
    nhead : int
        Number of attention heads (default 8)
    num_layers : int
        Number of transformer encoder layers (default 6)
    repr_dim : int
        Output representation dimension (default 256)
    proj_dim : int
        SSL projection head output dimension (default 128)
    dropout : float
        Dropout rate inside transformer layers (default 0.1)
    """

    def __init__(
        self,
        signal_length: int = 250,
        patch_size: int = 25,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        repr_dim: int = 256,
        proj_dim: int = 128,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        assert signal_length % patch_size == 0, (
            f"signal_length ({signal_length}) must be divisible by patch_size ({patch_size})"
        )
        self.patch_size = patch_size
        self.num_patches = signal_length // patch_size  # 10
        self.d_model = d_model
        self.repr_dim = repr_dim

        # Patch embedding: Conv1d with stride = kernel = patch_size
        # Output shape: (B, d_model, num_patches)
        self.patch_embed = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=patch_size, stride=patch_size, bias=False),
            nn.LayerNorm(d_model),  # Applied after transpose; will be composed below
        )
        # Note: LayerNorm needs (B, L, C) — we transpose after Conv1d
        # Simpler: use linear patch embedding instead
        self.patch_proj = nn.Linear(patch_size, d_model)

        # Learnable CLS token (prepended to patch sequence)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Learnable positional embedding (CLS + num_patches positions)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, d_model)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Pre-LN Transformer encoder (more stable for smaller batch sizes)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN (stable training)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        self.norm = nn.LayerNorm(d_model)

        # Linear head to repr_dim
        self.head = nn.Linear(d_model, repr_dim) if d_model != repr_dim else nn.Identity()

        # Projection head for SSL (same interface as ResNet1D / WavKAN)
        from src.models.encoder import ProjectionHead
        self.projection_head = ProjectionHead(
            repr_dim, hidden_dim=256, output_dim=proj_dim,
            metadata_dim=kwargs.get("metadata_dim", 0),
        )

        # Classification head (set via set_classifier for downstream tasks)
        self.classifier = None

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with truncated normal (ViT convention)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def set_classifier(self, num_classes: int):
        """Attach a classification head for downstream evaluation."""
        self.classifier = nn.Linear(self.repr_dim, num_classes)
        return self

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert (B, 1, L) signal to (B, num_patches, patch_size) patches.
        """
        B, C, L = x.shape
        # Reshape: (B, L) → (B, num_patches, patch_size)
        x = x.squeeze(1)  # (B, L)
        x = x.view(B, self.num_patches, self.patch_size)
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the global CLS-token representation.

        Args:
            x: (B, 1, 250) or (B, 250)

        Returns:
            h: (B, repr_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)   # (B, L) → (B, 1, L)

        B = x.size(0)

        # 1. Patchify + project
        patches = self._patchify(x)           # (B, num_patches, patch_size)
        tokens = self.patch_proj(patches)     # (B, num_patches, d_model)

        # 2. Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, num_patches+1, d_model)

        # 3. Add positional embedding
        tokens = tokens + self.pos_embed

        # 4. Transformer
        encoded = self.transformer(tokens)    # (B, num_patches+1, d_model)
        encoded = self.norm(encoded)

        # 5. CLS token → representation
        cls_output = encoded[:, 0]            # (B, d_model)
        return self.head(cls_output)          # (B, repr_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_projection: bool = False,
        metadata=None,
    ) -> torch.Tensor:
        h = self.encode(x)

        if return_projection:
            return self.projection_head(h, metadata)

        if self.classifier is not None:
            return self.classifier(h)

        return h
