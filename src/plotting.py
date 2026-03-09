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
