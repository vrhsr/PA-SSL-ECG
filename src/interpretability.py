"""
PA-SSL: Interpretability Tools

Implements:
  1. Grad-CAM for 1D ResNet — shows which temporal regions drive predictions
  2. Feature importance via integrated gradients
  3. Visualization of model attention on ECG morphology (P, QRS, T)

These generate publication-quality figures showing the model focuses
on clinically relevant ECG features.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


class GradCAM1D:
    """
    Grad-CAM adapted for 1D convolutional networks (ECG signals).
    
    Computes gradient-weighted class activation maps showing which
    temporal regions of the ECG signal most influence predictions.
    """
    
    def __init__(self, model, target_layer_name='layer4'):
        """
        Args:
            model: Encoder with convolutional layers
            target_layer_name: Name of the conv layer to visualize
        """
        self.model = model
        self.model.eval()
        
        self.gradients = None
        self.activations = None
        
        # Register hooks on target layer
        target_layer = self._find_layer(target_layer_name)
        if target_layer is None:
            raise ValueError(f"Layer '{target_layer_name}' not found")
        
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)
    
    def _find_layer(self, name):
        """Find a named layer in the model."""
        for n, module in self.model.named_modules():
            if n == name or n.endswith(name):
                return module
        
        # Fallback: find the last Conv1d layer
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv1d):
                last_conv = module
        return last_conv
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def __call__(self, input_signal, target_class=None):
        """
        Compute Grad-CAM heatmap.
        
        Args:
            input_signal: (1, 1, L) tensor
            target_class: Class index to explain. If None, uses predicted class.
        
        Returns:
            cam: (L,) numpy array — activation map (0-1 normalized)
            pred_class: predicted class index
        """
        self.model.eval()
        input_signal.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_signal)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()
        
        # Compute Grad-CAM
        weights = self.gradients.mean(dim=2, keepdim=True)  # (1, C, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, T')
        cam = F.relu(cam)
        
        # Upsample to input size
        cam = F.interpolate(cam, size=input_signal.shape[-1], mode='linear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        
        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        
        return cam, target_class


def compute_integrated_gradients(model, input_signal, target_class=None,
                                  n_steps=50, baseline=None):
    """
    Integrated Gradients for ECG feature attribution.
    
    Shows which individual samples in the ECG signal are most important
    for the model's prediction.
    
    Args:
        model: Encoder with classifier
        input_signal: (1, 1, L) tensor
        target_class: Class to explain
        n_steps: Number of interpolation steps
        baseline: Baseline signal (default: zeros)
    
    Returns:
        attributions: (L,) numpy array
    """
    model.eval()
    device = input_signal.device
    
    if baseline is None:
        baseline = torch.zeros_like(input_signal)
    
    if target_class is None:
        with torch.no_grad():
            output = model(input_signal)
            target_class = output.argmax(dim=1).item()
    
    # Interpolate between baseline and input
    alphas = torch.linspace(0, 1, n_steps + 1, device=device)
    gradients_list = []
    
    for alpha in alphas:
        interpolated = baseline + alpha * (input_signal - baseline)
        interpolated.requires_grad_(True)
        
        output = model(interpolated)
        score = output[0, target_class]
        
        model.zero_grad()
        score.backward()
        
        gradients_list.append(interpolated.grad.detach())
    
    # Average gradients × (input - baseline)
    avg_gradients = torch.stack(gradients_list).mean(dim=0)
    attributions = (input_signal - baseline) * avg_gradients
    attributions = attributions.squeeze().detach().cpu().numpy()
    
    return attributions


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_gradcam_ecg(signal, cam, r_peak_pos=125, pred_class=None,
                      true_class=None, save_path=None, title=None):
    """
    Publication-quality Grad-CAM overlay on ECG signal.
    
    Shows which temporal regions the model attends to, with
    annotated ECG morphology regions (P, QRS, T).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), height_ratios=[3, 1],
                                     sharex=True, gridspec_kw={'hspace': 0.05})
    
    t = np.arange(len(signal)) / 100.0  # time in seconds
    
    # ─── ECG with Grad-CAM overlay ──────────────────────────────────────
    # Color-code the ECG line by attention
    for i in range(len(t) - 1):
        color = plt.cm.hot(cam[i])
        ax1.plot(t[i:i+2], signal[i:i+2], color=color, linewidth=1.5)
    
    # Annotate morphological regions
    fs = 100
    regions = [
        ("P", (r_peak_pos - 18) / fs, (r_peak_pos - 8) / fs, 'lightblue'),
        ("QRS", (r_peak_pos - 4) / fs, (r_peak_pos + 4) / fs, 'lightyellow'),
        ("T", (r_peak_pos + 10) / fs, (r_peak_pos + 30) / fs, 'lightgreen'),
    ]
    
    for label, start, end, color in regions:
        if 0 <= start and end <= t[-1]:
            ax1.axvspan(start, end, alpha=0.15, color=color)
            ax1.text((start + end) / 2, ax1.get_ylim()[1] * 0.85, label,
                     ha='center', fontsize=10, fontweight='bold', color='gray')
    
    label_parts = []
    if true_class is not None:
        label_parts.append(f"True: {'Normal' if true_class == 0 else 'Abnormal'}")
    if pred_class is not None:
        label_parts.append(f"Pred: {'Normal' if pred_class == 0 else 'Abnormal'}")
    if label_parts:
        ax1.set_title(" | ".join(label_parts), fontsize=11)
    
    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    
    ax1.set_ylabel("Amplitude (mV)", fontsize=10)
    ax1.grid(True, alpha=0.2)
    
    # ─── Attention heatmap bar ──────────────────────────────────────────
    ax2.imshow(cam.reshape(1, -1), aspect='auto', cmap='hot',
               extent=[t[0], t[-1], 0, 1])
    ax2.set_ylabel("Attention", fontsize=10)
    ax2.set_xlabel("Time (s)", fontsize=10)
    ax2.set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    return fig


def plot_attention_summary(signals, cams, labels, save_path=None):
    """
    Aggregate attention maps across multiple samples to show
    which ECG regions the model consistently focuses on.
    
    Publication Figure: "The model attends primarily to QRS complexes
    for normal ECGs and to ST-segments for abnormal ECGs."
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    
    t = np.arange(signals.shape[1]) / 100.0
    
    for class_idx, class_name in [(0, 'Normal'), (1, 'Abnormal')]:
        mask = labels == class_idx
        if not mask.any():
            continue
        
        class_cams = cams[mask]
        class_signals = signals[mask]
        
        ax = axes[class_idx]
        
        # Mean ± std of attention
        mean_cam = class_cams.mean(axis=0)
        std_cam = class_cams.std(axis=0)
        
        ax.fill_between(t, mean_cam - std_cam, mean_cam + std_cam,
                         alpha=0.2, color='orange')
        ax.plot(t, mean_cam, 'r-', linewidth=2, label='Mean attention')
        
        # Overlay mean ECG
        mean_signal = class_signals.mean(axis=0)
        ax_ecg = ax.twinx()
        ax_ecg.plot(t, mean_signal, 'b-', alpha=0.5, linewidth=0.8, label='Mean ECG')
        ax_ecg.set_ylabel("ECG (mV)", fontsize=9, color='blue')
        
        ax.set_title(f"Class: {class_name} (n={mask.sum()})", fontsize=11)
        ax.set_ylabel("Grad-CAM Attention", fontsize=9, color='red')
        ax.grid(True, alpha=0.2)
    
    axes[-1].set_xlabel("Time (s)", fontsize=10)
    fig.suptitle("Aggregated Model Attention by Class", fontsize=13, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    return fig
