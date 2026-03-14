import torch
from torch import nn
import torch.nn.functional as F

class TFCEncoder(nn.Module):
    """
    Official TFC dual-encoder architecture (Emam et al., 2022).
    Consists of a Time-domain encoder and a Frequency-domain encoder.
    """
    def __init__(self, in_channels=1, hidden_dim=64, proj_dim=128):
        super().__init__()
        
        # Time-domain encoder
        self.time_encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=8, stride=1, padding='same', bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding='same', bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, hidden_dim, kernel_size=8, stride=1, padding='same', bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Frequency-domain encoder
        self.freq_encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=8, stride=1, padding='same', bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding='same', bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, hidden_dim, kernel_size=8, stride=1, padding='same', bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Projectors
        self.time_projector = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, proj_dim)
        )
        
        self.freq_projector = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, proj_dim)
        )
        
    def forward_time(self, x):
        h = self.time_encoder(x).squeeze(2)
        z = self.time_projector(h)
        return h, z
        
    def forward_freq(self, x_f):
        h = self.freq_encoder(x_f).squeeze(2)
        z = self.freq_projector(h)
        return h, z
        
    def encode(self, x):
        """Standard encode method for evaluation"""
        # During inference, TFC concatenates time and freq representations
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # Time forward
        h_t = self.time_encoder(x).squeeze(2)
        
        # Get freq forward
        # Compute FFT
        x_f = torch.fft.fft(x).abs()
        h_f = self.freq_encoder(x_f).squeeze(2)
        
        return torch.cat([h_t, h_f], dim=1)

class TFCLoss(nn.Module):
    """
    Time-Frequency Consistency Loss.
    Contrastive loss between time and frequency pairs of the same sample.
    """
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z_t, z_f):
        # NT-Xent loss between time and freq representations
        B = z_t.size(0)
        
        z = torch.cat([z_t, z_f], dim=0)
        z = F.normalize(z, dim=1)
        
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        
        # Mask out self-similarity
        sim_matrix.fill_diagonal_(-1e9)
        
        # Targets: z_t[i] matches z_f[i] and vice versa
        targets = torch.arange(B, device=z.device)
        targets = torch.cat([targets + B, targets])
        
        return F.cross_entropy(sim_matrix, targets)
