"""
PA-SSL: Paper-Ready Plotting Utilities

Generates all publication-quality figures needed for the PA-SSL paper:
  1. Label efficiency curves
  2. Cross-dataset generalization heatmap
  3. Calibration reliability diagrams
  4. Ablation study comparison tables/bars
  5. t-SNE / UMAP representation analysis
  6. Computational efficiency comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter
from pathlib import Path

# ─── STYLE ────────────────────────────────────────────────────────────────────
# Publication-quality defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color palette for methods
COLORS = {
    'PA-SSL (Ours)': '#2166AC',
    'PA-SSL (ResNet1D)': '#2166AC',
    'PA-SSL (WavKAN)': '#4393C3',
    'SimCLR + Naive Aug': '#F4A582',
    'SimCLR (no temporal)': '#D6604D',
    'Supervised': '#878787',
    'XGBoost': '#B2ABD2',
    'Random Init': '#D9D9D9',
}

MARKERS = {
    'PA-SSL (Ours)': 'o',
    'PA-SSL (ResNet1D)': 'o',
    'PA-SSL (WavKAN)': 's',
    'SimCLR + Naive Aug': '^',
    'SimCLR (no temporal)': 'D',
    'Supervised': 'v',
    'XGBoost': 'X',
    'Random Init': 'P',
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LABEL EFFICIENCY CURVES
# ═══════════════════════════════════════════════════════════════════════════════

def plot_label_efficiency(results_df, metric='accuracy', save_path=None):
    """
    Plot label efficiency curves comparing methods.
    
    Args:
        results_df: DataFrame with columns [method, label_fraction, metric_mean, metric_std]
        metric: Metric name to plot (accuracy, auroc, f1_macro)
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    for method in results_df['method'].unique():
        method_data = results_df[results_df['method'] == method]
        
        fractions = method_data['label_fraction'].values * 100
        means = method_data[f'{metric}_mean'].values
        stds = method_data[f'{metric}_std'].values
        
        color = COLORS.get(method, '#333333')
        marker = MARKERS.get(method, 'o')
        
        ax.plot(fractions, means, color=color, marker=marker, markersize=8,
                linewidth=2, label=method, zorder=3)
        ax.fill_between(fractions, means - stds, means + stds,
                         alpha=0.15, color=color, zorder=2)
    
    ax.set_xlabel("Labeled Data (%)", fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title("Label Efficiency Comparison", fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xticks([1, 5, 10, 25, 50, 100])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='gray')
    
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"Label efficiency plot saved: {save_path}")
    plt.close()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CROSS-DATASET GENERALIZATION HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════

def plot_cross_dataset_heatmap(results_df, metric='accuracy', save_path=None):
    """
    Heatmap of cross-dataset transfer performance.
    
    Args:
        results_df: DataFrame with columns [train_dataset, test_dataset, method, metric_value]
    """
    methods = results_df['method'].unique()
    n_methods = len(methods)
    
    train_datasets = results_df['train_dataset'].unique()
    test_datasets = results_df['test_dataset'].unique()
    
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4))
    if n_methods == 1:
        axes = [axes]
    
    for idx, method in enumerate(methods):
        method_data = results_df[results_df['method'] == method]
        
        matrix = pd.pivot_table(
            method_data, values=metric, index='train_dataset',
            columns='test_dataset', aggfunc='mean'
        )
        
        im = axes[idx].imshow(matrix.values, cmap='YlGnBu', vmin=0.5, vmax=1.0)
        axes[idx].set_xticks(range(len(matrix.columns)))
        axes[idx].set_yticks(range(len(matrix.index)))
        axes[idx].set_xticklabels(matrix.columns, rotation=45, ha='right')
        axes[idx].set_yticklabels(matrix.index)
        axes[idx].set_title(method, fontsize=12, fontweight='bold')
        
        # Annotate cells
        for i in range(len(matrix.index)):
            for j in range(len(matrix.columns)):
                val = matrix.values[i, j]
                color = 'white' if val > 0.75 else 'black'
                axes[idx].text(j, i, f'{val:.3f}', ha='center', va='center',
                             color=color, fontsize=10)
        
        if idx == 0:
            axes[idx].set_ylabel("Train Dataset", fontsize=11)
        axes[idx].set_xlabel("Test Dataset", fontsize=11)
    
    fig.suptitle(f"Cross-Dataset Transfer ({metric.title()})",
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CALIBRATION RELIABILITY DIAGRAMS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_reliability_diagram(calibration_data, save_path=None):
    """
    Reliability diagram with confidence histograms.
    
    Args:
        calibration_data: dict of {method_name: {bin_centers, bin_accuracies, bin_confidences, ece}}
    """
    n_methods = len(calibration_data)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]
    
    for idx, (method, data) in enumerate(calibration_data.items()):
        ax = axes[idx]
        
        bins = np.array(data['bin_centers'])
        acc = np.array(data['bin_accuracies'])
        conf = np.array(data['bin_confidences'])
        ece = data['ece']
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
        
        # Reliability bars
        width = 1.0 / (len(bins) + 1)
        ax.bar(bins, acc, width=width, alpha=0.6,
               color=COLORS.get(method, '#2166AC'), edgecolor='gray',
               label=f'{method}\nECE={ece:.4f}')
        
        # Gap fill
        for b, a, c in zip(bins, acc, conf):
            if a < c:
                ax.fill_between([b - width/2, b + width/2], a, c,
                               alpha=0.2, color='red')
        
        ax.set_xlabel("Confidence", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(method, fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.legend(loc='upper left', fontsize=8)
    
    fig.suptitle("Calibration: Reliability Diagrams",
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ABLATION STUDY BAR CHART
# ═══════════════════════════════════════════════════════════════════════════════

def plot_ablation_bars(ablation_df, metrics=('accuracy', 'auroc', 'ece'),
                        save_path=None):
    """
    Grouped bar chart comparing ablation configurations.
    
    Args:
        ablation_df: DataFrame with columns [configuration, accuracy, auroc, f1, ece]
    """
    configs = ablation_df['configuration'].values
    n_configs = len(configs)
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    x = np.arange(n_configs)
    
    colors_ablation = plt.cm.Set2(np.linspace(0, 1, n_configs))
    
    for ax, metric in zip(axes, metrics):
        values = ablation_df[metric].values
        
        bars = ax.bar(x, values, color=colors_ablation, edgecolor='gray',
                      width=0.6)
        
        # Value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel(metric.upper().replace('_', ' '), fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=30, ha='right', fontsize=9)
        
        # Highlight best
        best_idx = np.argmax(values) if metric != 'ece' else np.argmin(values)
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(2)
    
    fig.suptitle("Ablation Study Results", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 5. t-SNE / UMAP REPRESENTATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_tsne_embeddings(representations, labels, method_name='PA-SSL',
                          perplexity=30, save_path=None):
    """
    t-SNE visualization of learned ECG representations.
    
    Shows clustering quality of the learned representation space.
    """
    from sklearn.manifold import TSNE
    
    # Subsample for speed
    max_samples = 5000
    if len(representations) > max_samples:
        idx = np.random.choice(len(representations), max_samples, replace=False)
        representations = representations[idx]
        labels = labels[idx]
    
    print(f"Computing t-SNE for {len(representations)} samples...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                n_iter=1000, learning_rate='auto', init='pca')
    embeddings_2d = tsne.fit_transform(representations)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    unique_labels = np.unique(labels)
    label_names = {0: 'Normal', 1: 'Abnormal'}
    colors_map = {0: '#2166AC', 1: '#B2182B'}
    
    for lbl in unique_labels:
        mask = labels == lbl
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=colors_map.get(lbl, '#333333'), label=label_names.get(lbl, f'Class {lbl}'),
                   alpha=0.5, s=15, edgecolors='none')
    
    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title(f"t-SNE: {method_name}", fontsize=14, fontweight='bold')
    ax.legend(markerscale=2, framealpha=0.9)
    
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()
    return fig


def plot_umap_embeddings(representations, labels, method_name='PA-SSL',
                          save_path=None):
    """UMAP visualization of learned representations."""
    try:
        import umap
    except ImportError:
        print("UMAP not installed: pip install umap-learn")
        return None
    
    max_samples = 5000
    if len(representations) > max_samples:
        idx = np.random.choice(len(representations), max_samples, replace=False)
        representations = representations[idx]
        labels = labels[idx]
    
    print(f"Computing UMAP for {len(representations)} samples...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15,
                         min_dist=0.1)
    embeddings_2d = reducer.fit_transform(representations)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    unique_labels = np.unique(labels)
    label_names = {0: 'Normal', 1: 'Abnormal'}
    colors_map = {0: '#2166AC', 1: '#B2182B'}
    
    for lbl in unique_labels:
        mask = labels == lbl
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=colors_map.get(lbl, '#333333'), label=label_names.get(lbl, f'Class {lbl}'),
                   alpha=0.5, s=15, edgecolors='none')
    
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(f"UMAP: {method_name}", fontsize=14, fontweight='bold')
    ax.legend(markerscale=2, framealpha=0.9)
    
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()
    return fig


def plot_tsne_comparison(methods_data, save_path=None):
    """
    Side-by-side t-SNE for multiple methods (e.g., PA-SSL vs Naive SSL vs Supervised).
    
    Args:
        methods_data: dict of {method_name: (representations, labels)}
    """
    from sklearn.manifold import TSNE
    
    n_methods = len(methods_data)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]
    
    label_names = {0: 'Normal', 1: 'Abnormal'}
    colors_map = {0: '#2166AC', 1: '#B2182B'}
    
    for idx, (method, (reprs, labels)) in enumerate(methods_data.items()):
        max_samples = 3000
        if len(reprs) > max_samples:
            sample_idx = np.random.choice(len(reprs), max_samples, replace=False)
            reprs = reprs[sample_idx]
            labels = labels[sample_idx]
        
        tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                     n_iter=1000, learning_rate='auto', init='pca')
        emb = tsne.fit_transform(reprs)
        
        for lbl in np.unique(labels):
            mask = labels == lbl
            axes[idx].scatter(emb[mask, 0], emb[mask, 1],
                             c=colors_map.get(lbl, '#333'), s=10, alpha=0.4,
                             label=label_names.get(lbl, f'Class {lbl}'),
                             edgecolors='none')
        
        axes[idx].set_title(method, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel("t-SNE 1", fontsize=10)
        if idx == 0:
            axes[idx].set_ylabel("t-SNE 2", fontsize=10)
        axes[idx].legend(markerscale=2, fontsize=8)
    
    fig.suptitle("Representation Quality Comparison (t-SNE)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 6. COMPUTATIONAL EFFICIENCY
# ═══════════════════════════════════════════════════════════════════════════════

def plot_efficiency_comparison(efficiency_df, save_path=None):
    """
    Scatter plot: Accuracy vs Training Time / GPU Memory.
    
    Args:
        efficiency_df: DataFrame with columns [method, accuracy, train_time_hours, gpu_memory_gb, n_params]
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for _, row in efficiency_df.iterrows():
        method = row['method']
        color = COLORS.get(method, '#333333')
        marker = MARKERS.get(method, 'o')
        
        # Accuracy vs Time
        ax1.scatter(row['train_time_hours'], row['accuracy'],
                    color=color, marker=marker, s=150, edgecolors='gray',
                    zorder=3, label=method)
        ax1.annotate(method, (row['train_time_hours'], row['accuracy']),
                     fontsize=8, ha='left', va='bottom',
                     xytext=(5, 5), textcoords='offset points')
        
        # Accuracy vs Memory
        ax2.scatter(row['gpu_memory_gb'], row['accuracy'],
                    color=color, marker=marker, s=150, edgecolors='gray',
                    zorder=3, label=method)
        ax2.annotate(method, (row['gpu_memory_gb'], row['accuracy']),
                     fontsize=8, ha='left', va='bottom',
                     xytext=(5, 5), textcoords='offset points')
    
    ax1.set_xlabel("Training Time (hours)", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_title("Accuracy vs Training Time", fontsize=12)
    
    ax2.set_xlabel("GPU Memory (GB)", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_title("Accuracy vs GPU Memory", fontsize=12)
    
    fig.suptitle("Computational Efficiency", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# LATEX TABLE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_latex_table(results_df, metrics=('accuracy', 'auroc', 'f1_macro', 'ece'),
                          caption="Comparison of methods", label="tab:results"):
    """
    Generate LaTeX table from results DataFrame.
    
    Returns:
        str: LaTeX table code
    """
    methods = results_df['method'].unique()
    
    # Header
    header = "Method"
    for m in metrics:
        header += f" & {m.upper().replace('_', ' ')}"
    header += " \\\\"
    
    # Rows
    rows = []
    for method in methods:
        method_data = results_df[results_df['method'] == method]
        row = method
        for metric in metrics:
            if f'{metric}_mean' in method_data.columns:
                mean = method_data[f'{metric}_mean'].values[0]
                std = method_data[f'{metric}_std'].values[0]
                row += f" & ${mean:.3f} \\pm {std:.3f}$"
            elif metric in method_data.columns:
                val = method_data[metric].values[0]
                row += f" & {val:.3f}"
        row += " \\\\"
        rows.append(row)
    
    col_format = "l" + "c" * len(metrics)
    
    latex = f"""\\begin{{table}}[ht]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{{col_format}}}
\\toprule
{header}
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    
    return latex


# ═══════════════════════════════════════════════════════════════════════════════
# 7. TRAINING CONVERGENCE CURVES
# ═══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(histories, save_path=None):
    """
    Plot SSL training loss convergence curves for multiple methods.
    
    Args:
        histories: dict of {method_name: list of {'epoch', 'loss', 'loss_aug', 'loss_temporal'}}
        save_path: path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    curve_colors = {
        'PA-SSL (PhysioAug + Temporal)': '#2166AC',
        'PA-SSL + Metadata': '#1A9850',
        'SimCLR + Naive Aug': '#F4A582',
        'PhysioAug (no temporal)': '#D6604D',
    }
    
    for method, history in histories.items():
        epochs = [h['epoch'] for h in history]
        losses = [h['loss'] for h in history]
        color = curve_colors.get(method, '#333333')
        
        ax1.plot(epochs, losses, label=method, color=color, linewidth=2)
        
        # Also plot augmentation component
        if 'loss_aug' in history[0]:
            loss_aug = [h['loss_aug'] for h in history]
            ax2.plot(epochs, loss_aug, label=f"{method} (aug)", 
                     color=color, linewidth=2, linestyle='-')
        
        if 'loss_temporal' in history[0]:
            loss_temp = [h['loss_temporal'] for h in history]
            ax2.plot(epochs, loss_temp, label=f"{method} (temp)", 
                     color=color, linewidth=2, linestyle='--')
    
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Total Loss", fontsize=12)
    ax1.set_title("Training Convergence", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, framealpha=0.9)
    
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss Component", fontsize=12)
    ax2.set_title("Loss Components (Solid=Aug, Dashed=Temporal)", fontsize=13, fontweight='bold')
    ax2.legend(fontsize=8, framealpha=0.9, ncol=2)
    
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"Training curves saved: {save_path}")
    plt.close()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 8. AUGMENTATION HERO FIGURE (Figure 1 of Paper)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_augmentation_hero(original_signal, augmented_signals, aug_names,
                            r_peak_idx=None, save_path=None):
    """
    Create the paper's hero figure showing original ECG + all augmentations.
    
    Args:
        original_signal: 1D numpy array (250 samples)
        augmented_signals: list of 1D numpy arrays
        aug_names: list of augmentation names
        r_peak_idx: index of R-peak in original signal (for annotation)
        save_path: path to save figure
    """
    n_augs = len(augmented_signals)
    n_cols = 4
    n_rows = (n_augs + 1 + n_cols - 1) // n_cols  # +1 for original
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows))
    axes = axes.flatten()
    
    t = np.arange(len(original_signal)) / 100.0  # assuming 100 Hz -> seconds
    
    # Plot original
    axes[0].plot(t, original_signal, color='#2166AC', linewidth=1.5)
    axes[0].set_title('Original ECG', fontsize=11, fontweight='bold', color='#2166AC')
    axes[0].set_ylabel('Amplitude', fontsize=9)
    if r_peak_idx is not None:
        axes[0].axvline(x=r_peak_idx / 100.0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        axes[0].annotate('R-peak', xy=(r_peak_idx / 100.0, original_signal[r_peak_idx]),
                          fontsize=8, color='red', ha='center', va='bottom')
    
    # Plot each augmentation
    aug_colors = ['#D6604D', '#F4A582', '#92C5DE', '#4393C3', '#1A9850', '#762A83', '#E66101']
    
    for i, (aug_signal, aug_name) in enumerate(zip(augmented_signals, aug_names)):
        ax = axes[i + 1]
        # Plot original faintly for reference
        ax.plot(t[:len(original_signal)], original_signal, color='#CCCCCC', 
                linewidth=0.8, alpha=0.6, label='Original')
        # Plot augmented
        color = aug_colors[i % len(aug_colors)]
        ax.plot(t[:len(aug_signal)], aug_signal, color=color, linewidth=1.3, label='Augmented')
        ax.set_title(aug_name, fontsize=10, fontweight='bold', color=color)
        
        if r_peak_idx is not None and r_peak_idx < len(aug_signal):
            ax.axvline(x=r_peak_idx / 100.0, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
    
    # Hide unused axes
    for i in range(n_augs + 1, len(axes)):
        axes[i].set_visible(False)
    
    # Common labels
    for ax in axes:
        if ax.get_visible():
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.tick_params(labelsize=8)
    
    fig.suptitle('PA-SSL: Physiology-Aware Augmentation Library',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Augmentation hero figure saved: {save_path}")
    plt.close()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 9. FAIRNESS SUBGROUP COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fairness_comparison(fairness_results, save_path=None):
    """
    Bar chart comparing accuracy across demographic subgroups for multiple methods.
    
    Args:
        fairness_results: dict of {method_name: DataFrame with 'subgroup' and 'accuracy'}
    """
    subgroups = ['Overall', 'Male', 'Female', '<40', '40-60', '60+']
    methods = list(fairness_results.keys())
    n_methods = len(methods)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    x = np.arange(len(subgroups))
    width = 0.8 / n_methods
    
    method_colors = list(COLORS.values())[:n_methods]
    
    for i, method in enumerate(methods):
        df = fairness_results[method]
        values = []
        for sg in subgroups:
            row = df[df['subgroup'] == sg]
            values.append(row['accuracy'].values[0] if len(row) > 0 else 0)
        
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=method,
                      color=method_colors[i % len(method_colors)],
                      edgecolor='gray', alpha=0.85)
        
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7)
    
    ax.set_xlabel('Demographic Subgroup', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Demographic Fairness Analysis', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(subgroups, fontsize=10)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_ylim(0.5, 1.0)
    
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"Fairness comparison plot saved: {save_path}")
    plt.close()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# RELIABILITY DIAGRAM (ECE VISUALIZATION)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_reliability_diagram(bin_data_dict, ece_dict=None, save_path=None):
    """
    Plot reliability diagrams for one or more methods.
    
    Args:
        bin_data_dict: Dict of {method_name: {'bin_centers': [...], 'bin_accuracies': [...], 
                       'bin_confidences': [...], 'bin_counts': [...]}}
        ece_dict: Optional dict of {method_name: ece_value} for annotation
        save_path: Path to save figure
    """
    n_methods = len(bin_data_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5), squeeze=False)
    
    for idx, (method_name, data) in enumerate(bin_data_dict.items()):
        ax = axes[0, idx]
        centers = np.array(data['bin_centers'])
        accs = np.array(data['bin_accuracies'])
        confs = np.array(data['bin_confidences'])
        counts = np.array(data['bin_counts'])
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        
        # Bar chart of accuracy per confidence bin
        bar_width = 0.06
        color = COLORS.get(method_name, '#2166AC')
        ax.bar(centers, accs, width=bar_width, alpha=0.7, color=color,
               edgecolor='white', label='Model')
        
        # Gap visualization (shaded area showing miscalibration)
        for c, a, conf in zip(centers, accs, confs):
            if a < conf:
                ax.fill_between([c - bar_width/2, c + bar_width/2], a, conf,
                                alpha=0.2, color='red')
            else:
                ax.fill_between([c - bar_width/2, c + bar_width/2], conf, a,
                                alpha=0.2, color='blue')
        
        ax.set_xlabel('Mean Predicted Confidence')
        ax.set_ylabel('Fraction of Positives (Accuracy)')
        ax.set_title(method_name, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        
        if ece_dict and method_name in ece_dict:
            ax.text(0.05, 0.92, f"ECE = {ece_dict[method_name]:.4f}",
                    transform=ax.transAxes, fontsize=11,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))
        
        ax.legend(loc='lower right', fontsize=8)
    
    plt.suptitle('Calibration Reliability Diagrams', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"Reliability diagram saved: {save_path}")
    plt.close()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FEW-SHOT LABEL EFFICIENCY CURVE
# ═══════════════════════════════════════════════════════════════════════════════

def plot_few_shot_curve(results_dict, metric='F1', save_path=None):
    """
    Plot few-shot label efficiency curves for multiple methods.
    
    Args:
        results_dict: Dict of {method_name: pd.DataFrame with columns [Labels, F1, AUROC, Accuracy]}
        metric: Which column to plot ('F1', 'AUROC', 'Accuracy')
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    for method_name, df in results_dict.items():
        color = COLORS.get(method_name, '#333333')
        marker = MARKERS.get(method_name, 'o')
        
        ax.plot(df['Labels'], df[metric], color=color, marker=marker,
                markersize=8, linewidth=2.2, label=method_name, zorder=3)
    
    ax.set_xlabel('Number of Labeled Samples', fontsize=12)
    ax.set_ylabel(f'Macro {metric}', fontsize=12)
    ax.set_title('Few-Shot Label Efficiency', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='gray')
    ax.set_ylim(0.3, 1.0)
    
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"Few-shot curve saved: {save_path}")
    plt.close()
    return fig
# ═══════════════════════════════════════════════════════════════════════════════
# 10. RAW DATASET DISTRIBUTION VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_raw_dataset_distribution(dataset_samples, dataset_names, save_path=None):
    """
    Visualizes the raw distribution of different datasets using UMAP.
    High separability here proves domain shift exists.
    
    Args:
        dataset_samples: list of numpy arrays (each N_i x 250)
        dataset_names: list of strings
    """
    try:
        import umap
    except ImportError:
        print("UMAP not installed")
        return None
        
    all_X = []
    all_y = []
    
    for i, (X, name) in enumerate(zip(dataset_samples, dataset_names)):
        # Normalize each sample to [0, 1] for fair comparison
        X_norm = (X - X.min(axis=1, keepdims=True)) / (X.max(axis=1, keepdims=True) - X.min(axis=1, keepdims=True) + 1e-6)
        all_X.append(X_norm)
        all_y.extend([i] * len(X))
        
    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.array(all_y)
    
    print(f"Computing UMAP for raw data samples: {X_combined.shape}")
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(X_combined)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    spectral_colors = plt.cm.Spectral(np.linspace(0, 1, len(dataset_names)))
    
    for i, name in enumerate(dataset_names):
        mask = y_combined == i
        ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                   label=name, alpha=0.6, s=10, color=spectral_colors[i])
        
    ax.set_title("Raw ECG Distribution (Domain Shift Visualization)", fontsize=14, fontweight='bold')
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# 11. TRAINING STABILITY (BATCH-LEVEL)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_batch_stability(batch_history, save_path=None):
    """
    Plots batch-level loss for training stability diagnostics.
    
    Args:
        batch_history: list of {'loss', 'loss_aug', 'loss_mae'}
    """
    if not batch_history: return
    
    df = pd.DataFrame(batch_history)
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Smooth the curves a bit
    window = 10
    ax.plot(df['loss'].rolling(window).mean(), label='Total Loss', color='#333333', alpha=0.8)
    if 'loss_aug' in df.columns:
        ax.plot(df['loss_aug'].rolling(window).mean(), label='Contrastive Loss', color='#2166AC', alpha=0.6)
    if 'loss_mae' in df.columns:
        ax.plot(df['loss_mae'].rolling(window).mean(), label='MAE Loss', color='#D6604D', alpha=0.6)
        
    ax.set_title("Training Stability (Batch-level Loss)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Batch Step")
    ax.set_ylabel("Loss")
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
    return fig
