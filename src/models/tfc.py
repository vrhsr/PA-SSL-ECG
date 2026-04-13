import torch
import torch.nn as nn
import torch.nn.functional as F

class TFCEncoder(nn.Module):
    """
    Time-Frequency Consistency (TFC) Encoder.
    Processes time-domain and frequency-domain signals and computes embeddings.
    """
    def __init__(self, in_channels=1, hidden_dim=64, proj_dim=128):
        super().__init__()
        
        # Time-domain encoder (reusing a simple CNN for time series)
        self.encoder_t = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=8, stride=1, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=8, stride=1, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        
        # Frequency-domain encoder (same architecture for amplitude spectrum)
        self.encoder_f = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=8, stride=1, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=8, stride=1, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        
        # Projectors to contrastive space
        self.projector_t = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, proj_dim)
        )
        
        self.projector_f = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, proj_dim)
        )

    def forward_time(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.encoder_t(x).squeeze(-1)
        z = self.projector_t(h)
        return h, z

    def forward_freq(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.encoder_f(x).squeeze(-1)
        z = self.projector_f(h)
        return h, z

    def encode(self, x):
        # TFC normally concatenates both time and freq embeddings for downstream tasks
        h_t, _ = self.forward_time(x)
        # Compute frequency domain
        x_f = torch.fft.fft(x).abs()
        h_f, _ = self.forward_freq(x_f)
        return torch.cat([h_t, h_f], dim=1)


class TFCLoss(nn.Module):
    """
    NT-Xent contrastive loss to enforce consistency between 
    time-domain and frequency-domain embeddings.
    """
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_t, z_f):
        # z_t and z_f are shape (B, proj_dim)
        z_t = F.normalize(z_t, dim=1)
        z_f = F.normalize(z_f, dim=1)
        
        B = z_t.size(0)
        # We treat z_t and z_f of the same item as positive pairs
        labels = torch.arange(B).to(z_t.device)
        
        # Cross-view similarities
        logits_tf = torch.matmul(z_t, z_f.T) / self.temperature
        logits_ft = torch.matmul(z_f, z_t.T) / self.temperature
        
        loss_tf = F.cross_entropy(logits_tf, labels)
        loss_ft = F.cross_entropy(logits_ft, labels)
        
        return (loss_tf + loss_ft) / 2
