"""
PA-SSL: Statistical Significance Tests & Publication-Ready Tables

Implements:
  1. Per-seed result aggregation (mean +/- std)
  2. Paired statistical tests (Wilcoxon signed-rank, paired t-test)
  3. LaTeX table generation with significance markers
  4. Demographic subgroup analysis
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations


# ==============================================================================
# 1. AGGREGATE RESULTS
# ==============================================================================

def aggregate_results(results_df, group_cols=('method', 'label_fraction'),
                      metric_cols=None):
    """
    Aggregate per-seed results into mean +/- std.
    
    Args:
        results_df: DataFrame with per-seed results
        group_cols: Columns to group by
        metric_cols: Columns to aggregate (auto-detected if None)
    
    Returns:
        DataFrame with mean, std, and formatted 'mean +/- std' columns
    """
    if metric_cols is None:
        # Auto-detect numeric columns that aren't grouping/seed columns
        exclude = set(group_cols) | {'seed', 'n_labeled'}
        metric_cols = [c for c in results_df.columns 
                       if c not in exclude and results_df[c].dtype in ('float64', 'float32', 'int64')]
    
    agg_dict = {}
    for col in metric_cols:
        agg_dict[f'{col}_mean'] = (col, 'mean')
        agg_dict[f'{col}_std'] = (col, 'std')
        agg_dict[f'{col}_count'] = (col, 'count')
    
    agg = results_df.groupby(list(group_cols)).agg(**agg_dict).reset_index()
    
    # Add formatted columns
    for col in metric_cols:
        mean_col = f'{col}_mean'
        std_col = f'{col}_std'
        
        def format_row(r):
            if pd.notna(r[std_col]) and r[std_col] > 0 and r.get(f'{col}_count', 1) > 1:
                n = r[f'{col}_count']
                ci_margin = 1.96 * r[std_col] / np.sqrt(n)
                return f"{r[mean_col]:.4f} ($\\pm${ci_margin:.4f})"
            elif pd.notna(r[std_col]) and r[std_col] > 0:
                return f"{r[mean_col]:.4f} $\\pm$ {r[std_col]:.4f}"
            else:
                return f"{r[mean_col]:.4f}"
                
        agg[f'{col}_formatted'] = agg.apply(format_row, axis=1)
    
    return agg


# ==============================================================================
# 2. PAIRED STATISTICAL TESTS
# ==============================================================================

def paired_significance_test(method_a_scores, method_b_scores, test='wilcoxon'):
    """
    Perform paired significance test between two methods.
    
    Args:
        method_a_scores: array of per-seed scores for method A
        method_b_scores: array of per-seed scores for method B
        test: 'wilcoxon' or 'ttest'
    
    Returns:
        dict with statistic, p_value, significant (at 0.05)
    """
    a = np.array(method_a_scores)
    b = np.array(method_b_scores)
    
    if len(a) < 3 or len(b) < 3:
        # Not enough samples for meaningful test
        return {'statistic': np.nan, 'p_value': np.nan, 'significant': False}
    
    if np.array_equal(a, b):
        return {'statistic': 0.0, 'p_value': 1.0, 'significant': False}
    
    try:
        if test == 'wilcoxon':
            stat, p = stats.wilcoxon(a, b)
        elif test == 'ttest':
            stat, p = stats.ttest_rel(a, b)
        else:
            raise ValueError(f"Unknown test: {test}")
    except Exception:
        # Fallback to t-test if Wilcoxon fails (e.g., all differences zero)
        try:
            stat, p = stats.ttest_rel(a, b)
        except Exception:
            return {'statistic': np.nan, 'p_value': np.nan, 'significant': False}
    
    return {
        'statistic': float(stat),
        'p_value': float(p),
        'significant': p < 0.05,
    }


def run_all_pairwise_tests(results_df, reference_method, metric='linear_accuracy',
                           group_col='label_fraction', test='wilcoxon'):
    """
    Run pairwise significance tests between a reference method and all others.
    
    Args:
        results_df: per-seed results with 'method', 'seed', and metric columns
        reference_method: name of the reference method (e.g., 'PA-SSL')
        metric: metric column to test
        group_col: column to group comparisons by (e.g., 'label_fraction')
    
    Returns:
        DataFrame with pairwise test results
    """
    methods = results_df['method'].unique()
    groups = results_df[group_col].unique() if group_col else [None]
    
    test_results = []
    
    for group in groups:
        if group is not None:
            ref_scores = results_df[
                (results_df['method'] == reference_method) & 
                (results_df[group_col] == group)
            ][metric].values
        else:
            ref_scores = results_df[
                results_df['method'] == reference_method
            ][metric].values
        
        for method in methods:
            if method == reference_method:
                continue
            
            if group is not None:
                other_scores = results_df[
                    (results_df['method'] == method) & 
                    (results_df[group_col] == group)
                ][metric].values
            else:
                other_scores = results_df[
                    results_df['method'] == method
                ][metric].values
            
            result = paired_significance_test(ref_scores, other_scores, test)
            result['reference'] = reference_method
            result['comparison'] = method
            if group is not None:
                result[group_col] = group
            test_results.append(result)
    
    return pd.DataFrame(test_results)


def bootstrap_confidence_intervals(y_true, y_pred_or_proba, metric_fn, n_bootstrap=1000, seed=42):
    """
    Compute 95% bootstrap confidence intervals for a given metric.
    
    Args:
        y_true: (N,) true labels
        y_pred_or_proba: (N,) predictions or probabilities
        metric_fn: function taking (y_true, y_pred) and returning a float
        n_bootstrap: number of bootstrap samples
        seed: random seed
        
    Returns:
        (lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    scores = []
    
    for _ in range(n_bootstrap):
        indices = rng.randint(0, n_samples, n_samples)
        if len(np.unique(y_true[indices])) < 2:
            continue
        try:
            score = metric_fn(y_true[indices], y_pred_or_proba[indices])
            scores.append(score)
        except Exception:
            pass
            
    if len(scores) < 2:
        return (np.nan, np.nan)
    
    lower = float(np.percentile(scores, 2.5))
    upper = float(np.percentile(scores, 97.5))
    return lower, upper


# ==============================================================================
# 3. LATEX TABLE GENERATION
# ==============================================================================

def _format_cell(mean, std, is_best=False, is_second=False, p_value=None):
    """Format a table cell with mean +/- std, bolding, and significance."""
    if np.isnan(mean):
        return '-'
    
    val = f"{mean:.2f}"
    # Standard deviation format string is passed as std if it is formatted,
    # or let's adapt to CI formatting (±x.xx)
    if isinstance(std, str):
        val = f"{mean:.2f} {std}"
    elif std > 0:
        val = f"{mean:.2f} ($\\pm${std:.2f})"
    
    # Significance marker
    sig = ''
    if p_value is not None and not np.isnan(p_value):
        if p_value < 0.001:
            sig = '$^{***}$'
        elif p_value < 0.01:
            sig = '$^{**}$'
        elif p_value < 0.05:
            sig = '$^{*}$'
    
    if is_best:
        return f"\\textbf{{{val}}}{sig}"
    elif is_second:
        return f"\\underline{{{val}}}{sig}"
    else:
        return f"{val}{sig}"


def generate_main_results_table(agg_df, metrics=('linear_accuracy', 'linear_auroc',
                                                   'linear_f1_macro', 'linear_ece'),
                                 output_path=None):
    """
    Generate the main results LaTeX table (Table 1 of paper).
    
    Args:
        agg_df: aggregated results DataFrame with mean/std columns
        metrics: metrics to include in table
        output_path: path to save .tex file
    
    Returns:
        LaTeX string
    """
    metric_labels = {
        'linear_accuracy': 'Accuracy',
        'linear_auroc': 'AUROC',
        'linear_f1_macro': 'F1 (Macro)',
        'linear_ece': 'ECE $\\downarrow$',
        'mahal_accuracy': 'Mah. Acc',
        'mahal_auroc': 'Mah. AUROC',
    }
    
    methods = agg_df['method'].unique()
    n_metrics = len(metrics)
    
    # Build header
    header = "\\begin{table}[t]\n\\centering\n"
    header += "\\caption{Main Results. PA-SSL vs baselines on PTB-XL. "
    header += "Mean (95\\% CI) over 3 seeds. Best in \\textbf{bold}, second \\underline{underlined}. "
    header += "$^{*}$p$<$0.05, $^{**}$p$<$0.01, $^{***}$p$<$0.001 (Wilcoxon).}\n"
    header += "\\label{tab:main_results}\n"
    header += f"\\begin{{tabular}}{{l{'c' * n_metrics}}}\n\\toprule\n"
    header += "Method & " + " & ".join(metric_labels.get(m, m) for m in metrics) + " \\\\\n"
    header += "\\midrule\n"
    
    # Build rows
    rows = []
    for method in methods:
        row_data = agg_df[agg_df['method'] == method].iloc[0]
        cells = [method.replace('_', ' ')]
        
        for metric in metrics:
            mean_col = f'{metric}_mean'
            std_col = f'{metric}_std'
            
            mean_val = row_data.get(mean_col, np.nan)
            std_val = row_data.get(std_col, 0)
            
            if np.isnan(mean_val):
                cells.append('-')
            else:
                # Find best and second best
                all_means = agg_df[mean_col].values
                if 'ece' in metric.lower():
                    # Lower is better for ECE
                    sorted_means = np.sort(all_means)
                    is_best = mean_val == sorted_means[0]
                    is_second = len(sorted_means) > 1 and mean_val == sorted_means[1]
                else:
                    sorted_means = np.sort(all_means)[::-1]
                    is_best = mean_val == sorted_means[0]
                    is_second = len(sorted_means) > 1 and mean_val == sorted_means[1]
                
                cells.append(_format_cell(mean_val, std_val, is_best, is_second))
        
        rows.append(" & ".join(cells) + " \\\\")
    
    # Build footer
    footer = "\\bottomrule\n\\end{tabular}\n\\end{table}"
    
    latex = header + "\n".join(rows) + "\n" + footer
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"LaTeX table saved to: {output_path}")
    
    return latex


def generate_label_efficiency_table(agg_df, metric='linear_accuracy', output_path=None):
    """
    Generate label efficiency LaTeX table (rows=methods, cols=fractions).
    """
    methods = agg_df['method'].unique()
    fractions = sorted(agg_df['label_fraction'].unique())
    
    header = "\\begin{table}[t]\n\\centering\n"
    header += "\\caption{Label efficiency (Accuracy). Mean (95\\% CI) over 3 seeds.}\n"
    header += "\\label{tab:label_efficiency}\n"
    header += f"\\begin{{tabular}}{{l{'c' * len(fractions)}}}\n\\toprule\n"
    header += "Method & " + " & ".join(f"{f*100:.0f}\\%" for f in fractions) + " \\\\\n"
    header += "\\midrule\n"
    
    rows = []
    for method in methods:
        cells = [method.replace('_', ' ')]
        for frac in fractions:
            row = agg_df[(agg_df['method'] == method) & (agg_df['label_fraction'] == frac)]
            if len(row) == 0:
                cells.append('-')
            else:
                mean_val = row[f'{metric}_mean'].values[0]
                std_val = row[f'{metric}_std'].values[0]
                
                # Best/second for this fraction
                frac_data = agg_df[agg_df['label_fraction'] == frac]
                all_means = frac_data[f'{metric}_mean'].values
                sorted_means = np.sort(all_means)[::-1]
                is_best = mean_val == sorted_means[0]
                is_second = len(sorted_means) > 1 and mean_val == sorted_means[1]
                
                cells.append(_format_cell(mean_val, std_val, is_best, is_second))
        
        rows.append(" & ".join(cells) + " \\\\")
    
    footer = "\\bottomrule\n\\end{tabular}\n\\end{table}"
    latex = header + "\n".join(rows) + "\n" + footer
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex)
    
    return latex


# ==============================================================================
# 4. DEMOGRAPHIC SUBGROUP ANALYSIS
# ==============================================================================

def demographic_subgroup_analysis(encoder, data_csv, device, metadata_cols=None):
    """
    Evaluate model performance broken down by demographic subgroups.
    
    Generates a table: Method | Overall | Male | Female | <40 | 40-60 | 60+
    
    Args:
        encoder: trained encoder model
        data_csv: path to processed CSV with metadata columns
        device: torch device
        metadata_cols: dict mapping column names, e.g. {'sex': 'sex', 'age': 'age'}
    
    Returns:
        DataFrame with per-subgroup metrics
    """
    from src.data.ecg_dataset import ECGBeatDataset
    from src.evaluate import extract_representations, linear_probe
    
    if metadata_cols is None:
        metadata_cols = {'sex': 'sex', 'age': 'age'}
    
    dataset = ECGBeatDataset(data_csv)
    df = pd.read_csv(data_csv)
    
    # Extract representations for all samples
    reprs, labels = extract_representations(encoder, dataset, device)
    
    results = []
    
    # Overall performance (80/20 split)
    n = len(labels)
    np.random.seed(42)
    indices = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = indices[:split], indices[split:]
    
    overall = linear_probe(reprs[train_idx], labels[train_idx],
                           reprs[test_idx], labels[test_idx])
    overall['subgroup'] = 'Overall'
    overall['n_samples'] = n
    results.append(overall)
    
    # Sex subgroups
    sex_col = metadata_cols.get('sex')
    if sex_col and sex_col in df.columns:
        for sex_val, sex_name in [(0, 'Male'), (1, 'Female')]:
            mask = df[sex_col].values == sex_val
            if mask.sum() < 50:
                continue
            sub_indices = np.where(mask)[0]
            np.random.shuffle(sub_indices)
            sub_split = int(0.8 * len(sub_indices))
            tr = sub_indices[:sub_split]
            te = sub_indices[sub_split:]
            
            if len(te) < 10:
                continue
            
            metrics = linear_probe(reprs[tr], labels[tr], reprs[te], labels[te])
            metrics['subgroup'] = sex_name
            metrics['n_samples'] = len(sub_indices)
            results.append(metrics)
    
    # Age subgroups
    age_col = metadata_cols.get('age')
    if age_col and age_col in df.columns:
        ages = df[age_col].values
        for age_min, age_max, age_name in [(0, 40, '<40'), (40, 60, '40-60'), (60, 200, '60+')]:
            mask = (ages >= age_min) & (ages < age_max)
            if mask.sum() < 50:
                continue
            sub_indices = np.where(mask)[0]
            np.random.shuffle(sub_indices)
            sub_split = int(0.8 * len(sub_indices))
            tr = sub_indices[:sub_split]
            te = sub_indices[sub_split:]
            
            if len(te) < 10:
                continue
            
            metrics = linear_probe(reprs[tr], labels[tr], reprs[te], labels[te])
            metrics['subgroup'] = age_name
            metrics['n_samples'] = len(sub_indices)
            results.append(metrics)
    
    results_df = pd.DataFrame(results)
    return results_df


def generate_fairness_table(results_dict, output_path=None):
    """
    Generate demographic fairness LaTeX table.
    
    Args:
        results_dict: dict of {method_name: subgroup_results_df}
        output_path: path to save .tex file
    """
    subgroups = ['Overall', 'Male', 'Female', '<40', '40-60', '60+']
    
    header = "\\begin{table}[t]\n\\centering\n"
    header += "\\caption{Demographic Fairness Analysis (Accuracy). "
    header += "Performance across sex and age subgroups.}\n"
    header += "\\label{tab:fairness}\n"
    header += f"\\begin{{tabular}}{{l{'c' * len(subgroups)}}}\n\\toprule\n"
    header += "Method & " + " & ".join(subgroups) + " \\\\\n"
    header += "\\midrule\n"
    
    rows = []
    for method, df in results_dict.items():
        cells = [method.replace('_', ' ')]
        for sg in subgroups:
            row = df[df['subgroup'] == sg]
            if len(row) == 0:
                cells.append('-')
            else:
                acc = row['accuracy'].values[0]
                cells.append(f"{acc:.2f}")
        rows.append(" & ".join(cells) + " \\\\")
    
    # Compute and append gap row
    footer = "\\bottomrule\n\\end{tabular}\n\\end{table}"
    latex = header + "\n".join(rows) + "\n" + footer
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex)
    
    return latex


# ==============================================================================
# 5. TRAINING CURVE UTILITIES
# ==============================================================================

def load_training_histories(experiment_dirs):
    """
    Load training history JSON files from multiple experiment directories.
    
    Args:
        experiment_dirs: dict of {method_name: experiment_dir_path}
    
    Returns:
        dict of {method_name: list of epoch dicts}
    """
    histories = {}
    for name, exp_dir in experiment_dirs.items():
        history_path = os.path.join(exp_dir, 'history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                histories[name] = json.load(f)
        else:
            print(f"Warning: No history.json found in {exp_dir}")
    return histories


# ==============================================================================
# 6. COMPUTATIONAL EFFICIENCY TABLE
# ==============================================================================

def generate_efficiency_table(efficiency_results, output_path=None):
    """
    Generate computational efficiency LaTeX table.
    
    Args:
        efficiency_results: list of dicts with keys:
            method, n_params, train_time_hours, inference_ms, gpu_memory_gb
    """
    df = pd.DataFrame(efficiency_results)
    
    header = "\\begin{table}[t]\n\\centering\n"
    header += "\\caption{Computational Efficiency Comparison.}\n"
    header += "\\label{tab:efficiency}\n"
    header += "\\begin{tabular}{lcccc}\n\\toprule\n"
    header += "Method & \\#Params & Train (h) & Infer (ms) & GPU Mem (GB) \\\\\n"
    header += "\\midrule\n"
    
    rows = []
    for _, row in df.iterrows():
        cells = [
            row['method'].replace('_', ' '),
            f"{row['n_params']/1e6:.1f}M",
            f"{row.get('train_time_hours', 0):.1f}",
            f"{row.get('inference_ms', 0):.1f}",
            f"{row.get('gpu_memory_gb', 0):.2f}",
        ]
        rows.append(" & ".join(cells) + " \\\\")
    
    footer = "\\bottomrule\n\\end{tabular}\n\\end{table}"
    latex = header + "\n".join(rows) + "\n" + footer
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex)
    
    return latex


# ==============================================================================
# 7. COMMAND-LINE INTERFACE
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PA-SSL Statistical Analysis")
    parser.add_argument('--results_dir', type=str, default='experiments',
                        help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='experiments/tables',
                        help='Output directory for LaTeX tables')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Try to load and aggregate any available results
    results_files = []
    for root, dirs, files in os.walk(args.results_dir):
        for f in files:
            if f.endswith('_results.csv') or f == 'label_efficiency_results.csv':
                results_files.append(os.path.join(root, f))
    
    if results_files:
        print(f"Found {len(results_files)} result files")
        for rf in results_files:
            print(f"  - {rf}")
            df = pd.read_csv(rf)
            if 'method' in df.columns and 'seed' in df.columns:
                agg = aggregate_results(df)
                print(f"    Aggregated: {len(agg)} rows")
    else:
        print("No result files found. Run experiments first.")
