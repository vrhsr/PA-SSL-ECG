#!/usr/bin/env python3
"""
Figure 6 — Label Efficiency: PA-HybridSSL vs Baselines
Publication-ready for IEEE TBME submission.

Reads  : ablation_summary.csv
Outputs: label_efficiency_chart.{pdf,png,svg}
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker   as mticker
import matplotlib.gridspec as gridspec
from matplotlib.lines   import Line2D
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# 0 · Paths
# ══════════════════════════════════════════════════════════════════════════════
CSV_IN  = Path('E:/PhD/PA-SSL-ECG/remote/results/ablation_summary.csv')
FIG_DIR = Path('E:/PhD/PA-SSL-ECG/paper/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)
STEM    = FIG_DIR / 'fig6_label_efficiency'

# ══════════════════════════════════════════════════════════════════════════════
# 1 · Load & tidy
# ══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(CSV_IN)
df.columns = ['idx', 'block', 'experiment', 'label_fraction',
              'acc_mean', 'acc_std',
              'auroc_mean', 'auroc_std',
              'auprc_mean', 'auprc_std']

df = df[df['block'] == 'ablation'].copy()
df['auroc_mean'] = pd.to_numeric(df['auroc_mean'], errors='coerce')
df['auroc_std']  = pd.to_numeric(df['auroc_std'],  errors='coerce')
df = df.dropna(subset=['auroc_mean'])

def classify(name: str) -> str | None:
    n = name.lower()
    if 'wavkan'      in n: return None
    if 'hybrid'      in n: return 'PA-HybridSSL (Ours)'
    if 'contrastive' in n: return 'SimCLR (Contrastive only)'
    if 'mae'         in n: return 'MAE-only'
    return None

df['model'] = df['experiment'].apply(classify)
df = df.dropna(subset=['model'])

# Aggregate seeds → mean ± std
agg = (df.groupby(['model', 'label_fraction'])
         .agg(mu=('auroc_mean', 'mean'),
              sd=('auroc_mean', 'std'))
         .reset_index())

KEEP_FRACS = [0.01, 0.05, 0.10, 1.00]
agg = agg[agg['label_fraction'].isin(KEEP_FRACS)].copy()
agg['pct'] = (agg['label_fraction'] * 100).round(0).astype(int)
agg['sd']  = agg['sd'].fillna(0.0)

SUPERVISED_100 = 0.861   # from Table I — supervised from scratch at 100 %

# ══════════════════════════════════════════════════════════════════════════════
# 2 · Style
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family':          'DejaVu Serif',
    'font.size':            11,
    'axes.labelsize':       12,
    'axes.titlesize':       13,
    'axes.titleweight':     'bold',
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    'axes.linewidth':       0.85,
    'axes.grid':            True,
    'grid.linestyle':       '--',
    'grid.linewidth':       0.55,
    'grid.alpha':           0.35,
    'legend.fontsize':      9.5,
    'legend.framealpha':    0.93,
    'legend.edgecolor':     '#cccccc',
    'legend.handlelength':  2.4,
    'xtick.direction':      'in',
    'ytick.direction':      'in',
    'xtick.minor.visible':  False,
    'ytick.minor.visible':  True,
    'figure.dpi':           150,
    'savefig.dpi':          300,
})

# Per-model visual spec
MODELS = {
    'PA-HybridSSL (Ours)': dict(
        color='#1565C0', marker='o', ms=9,
        lw=2.5, ls='-', zorder=5,
        band_alpha=0.13,
    ),
    'SimCLR (Contrastive only)': dict(
        color='#C62828', marker='s', ms=8,
        lw=2.0, ls='--', zorder=4,
        band_alpha=0.11,
    ),
    'MAE-only': dict(
        color='#2E7D32', marker='^', ms=8,
        lw=1.8, ls='-.', zorder=3,
        band_alpha=0.10,
    ),
}

# ══════════════════════════════════════════════════════════════════════════════
# 3 · Figure layout  (main axes + inset)
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(8.6, 5.6))
gs  = gridspec.GridSpec(
    1, 1, figure=fig,
    left=0.11, right=0.97,
    top=0.88,  bottom=0.13,
)
ax = fig.add_subplot(gs[0])

# Inset: magnify the 1 %–10 % low-label regime
ax_ins = ax.inset_axes([0.54, 0.06, 0.42, 0.44])
ax_ins.set_xscale('log')
ax_ins.set_xlim(0.8, 13)
ax_ins.set_xticks([1, 5, 10])
ax_ins.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax_ins.tick_params(labelsize=8, direction='in')
ax_ins.grid(True, linestyle='--', linewidth=0.45, alpha=0.3)
ax_ins.set_title('Low-label regime (1 – 10 %)', fontsize=8.5,
                 fontweight='bold', pad=4)
for sp in ('top', 'right'):
    ax_ins.spines[sp].set_visible(False)

# ══════════════════════════════════════════════════════════════════════════════
# 4 · Plot curves
# ══════════════════════════════════════════════════════════════════════════════
for model_name, sty in MODELS.items():
    sub = agg[agg['model'] == model_name].sort_values('pct')
    if sub.empty:
        print(f'  [WARN] No data for {model_name}')
        continue

    x   = sub['pct'].values.astype(float)
    y   = sub['mu'].values
    err = sub['sd'].values

    # ── Main axes ─────────────────────────────────────────────────────────────
    ax.plot(x, y,
            color=sty['color'],
            marker=sty['marker'],
            markersize=sty['ms'],
            linewidth=sty['lw'],
            linestyle=sty['ls'],
            zorder=sty['zorder'],
            label=model_name,
            clip_on=False)

    ax.fill_between(x,
                    np.clip(y - err, 0, 1),
                    np.clip(y + err, 0, 1),
                    color=sty['color'],
                    alpha=sty['band_alpha'],
                    zorder=sty['zorder'] - 1)

    ax.errorbar(x, y, yerr=err,
                fmt='none',
                ecolor=sty['color'],
                elinewidth=1.1,
                capsize=4,
                capthick=1.1,
                zorder=sty['zorder'] + 1)

    # ── Inset (1 %, 5 %, 10 % only) ──────────────────────────────────────────
    mask_ins = x <= 10
    if mask_ins.sum() > 1:
        ax_ins.plot(x[mask_ins], y[mask_ins],
                    color=sty['color'],
                    marker=sty['marker'],
                    markersize=sty['ms'] - 1,
                    linewidth=sty['lw'] - 0.4,
                    linestyle=sty['ls'],
                    zorder=sty['zorder'],
                    clip_on=False)
        ax_ins.fill_between(x[mask_ins],
                            np.clip(y[mask_ins] - err[mask_ins], 0, 1),
                            np.clip(y[mask_ins] + err[mask_ins], 0, 1),
                            color=sty['color'],
                            alpha=sty['band_alpha'])
        ax_ins.errorbar(x[mask_ins], y[mask_ins], yerr=err[mask_ins],
                        fmt='none', ecolor=sty['color'],
                        elinewidth=0.9, capsize=3, capthick=0.9)

# ══════════════════════════════════════════════════════════════════════════════
# 5 · Reference line — supervised 100 %
# ══════════════════════════════════════════════════════════════════════════════
ax.axhline(SUPERVISED_100,
           color='#555555', lw=1.3,
           ls=(0, (5, 4)), alpha=0.75,
           zorder=2)
ax.text(102, SUPERVISED_100 + 0.0008,
        f'Supervised 100 % (scratch) = {SUPERVISED_100:.3f}',
        fontsize=8, color='#444444',
        ha='right', va='bottom',
        style='italic')

ax_ins.axhline(SUPERVISED_100,
               color='#555555', lw=1.0,
               ls=(0, (5, 4)), alpha=0.7)

# ══════════════════════════════════════════════════════════════════════════════
# 6 · Headline annotation
#     "PA-HybridSSL with 1 % labels already exceeds full supervision"
# ══════════════════════════════════════════════════════════════════════════════
pa_row = agg[(agg['model'] == 'PA-HybridSSL (Ours)') & (agg['pct'] == 1)]
if not pa_row.empty:
    y_pa   = float(pa_row['mu'].iloc[0])
    gap_pp = (y_pa - SUPERVISED_100) * 100

    ax.annotate(
        f'1 % labels → AUROC {y_pa:.3f}\n'
        f'(+{gap_pp:.1f} pp vs full supervision)',
        xy=(1, y_pa),
        xytext=(4.5, y_pa - 0.0145),
        fontsize=8.8,
        color='#1565C0',
        ha='left',
        va='top',
        arrowprops=dict(
            arrowstyle='->',
            color='#1565C0',
            lw=1.3,
            connectionstyle='arc3,rad=-0.20',
        ),
        bbox=dict(
            boxstyle='round,pad=0.40',
            facecolor='#E8F1FB',
            edgecolor='#1565C0',
            linewidth=0.9,
            alpha=0.92,
        ),
        zorder=10,
    )

# ══════════════════════════════════════════════════════════════════════════════
# 7 · Axes formatting — main
# ══════════════════════════════════════════════════════════════════════════════
ax.set_xscale('log')
ax.set_xticks([1, 5, 10, 100])
ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax.set_xlim(0.75, 135)

# Y limits: 0.5 pp breathing room above & below data range
all_y = agg['mu'].values
y_lo  = max(0.0,  all_y.min() - 0.008)
y_hi  = min(1.0,  all_y.max() + 0.010)
ax.set_ylim(y_lo, y_hi)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

ax.set_xlabel('Labeled Data Fraction (%)', labelpad=7)
ax.set_ylabel('AUROC  (PTB-XL binary task)', labelpad=7)

# ── Inset y-limits derived from visible data ──────────────────────────────────
ins_data = agg[agg['pct'] <= 10]['mu'].values
if len(ins_data):
    ax_ins.set_ylim(ins_data.min() - 0.008, ins_data.max() + 0.010)
    ax_ins.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

ax.indicate_inset_zoom(ax_ins, edgecolor='#999999', linewidth=0.9, alpha=0.6)

# ══════════════════════════════════════════════════════════════════════════════
# 8 · Legend  (manually include the dashed reference)
# ══════════════════════════════════════════════════════════════════════════════
handles, labels = ax.get_legend_handles_labels()
sup_handle = Line2D(
    [0], [0],
    color='#555555', lw=1.3,
    ls=(0, (5, 4)), alpha=0.75,
    label=f'Supervised 100 % (scratch) = {SUPERVISED_100:.3f}',
)
handles.append(sup_handle)
labels.append(sup_handle.get_label())

ax.legend(
    handles, labels,
    loc='lower right',
    frameon=True,
    borderpad=0.75,
    labelspacing=0.55,
    handletextpad=0.6,
)

# ══════════════════════════════════════════════════════════════════════════════
# 9 · Titles & caption
# ══════════════════════════════════════════════════════════════════════════════
fig.suptitle(
    'Figure 6 — Label Efficiency: PA-HybridSSL vs SSL Baselines  (PTB-XL)',
    fontsize=13, fontweight='bold', y=0.975,
)
ax.set_title(
    'ResNet1D encoder · patient-aware split · binary arrhythmia detection',
    fontsize=9.5, color='#444444', pad=6,
)
fig.text(
    0.5, 0.018,
    'Shaded bands and error bars = ±1 std over 3 seeds.  '
    'Dashed line = fully supervised ResNet1D trained on 100 % labels (Table I).  '
    'X-axis is log-scaled.',
    ha='center', fontsize=7.8, color='#666666', style='italic',
)

# ══════════════════════════════════════════════════════════════════════════════
# 10 · Save
# ══════════════════════════════════════════════════════════════════════════════
for ext in ('pdf', 'png', 'svg'):
    out = STEM.with_suffix(f'.{ext}')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f'  Saved -> {out}')

plt.close()
print('Done.')