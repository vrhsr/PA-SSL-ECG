#!/usr/bin/env python3
"""
Figure 6 — Label Efficiency: PA-HybridSSL vs Baselines
Fixed:
  - Inset moved to upper-left (away from legend + annotation)
  - Reference line text moved inside axes, never clipped
  - Inset y-limits computed tightly from visible data
  - Annotation arrow re-routed to avoid inset
  - Figure bottom margin increased so caption never clips
"""

import sys, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker   as mticker
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
CSV_IN  = PROJECT_ROOT / 'remote' / 'results' / 'ablation_summary.csv'
OUT_DIR = PROJECT_ROOT / 'paper' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
STEM = OUT_DIR / 'fig6_label_efficiency'

# ── Load & tidy ───────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_IN)
df.columns = ['idx', 'block', 'experiment', 'label_fraction',
              'acc_mean', 'acc_std',
              'auroc_mean', 'auroc_std',
              'auprc_mean', 'auprc_std']

df = df[df['block'] == 'ablation'].copy()
df['auroc_mean'] = pd.to_numeric(df['auroc_mean'], errors='coerce')
df['auroc_std']  = pd.to_numeric(df['auroc_std'],  errors='coerce')
df = df.dropna(subset=['auroc_mean'])

def classify(name: str):
    n = name.lower()
    if 'wavkan'      in n: return None
    if 'hybrid'      in n: return 'PA-HybridSSL (Ours)'
    if 'contrastive' in n: return 'SimCLR (Contrastive only)'
    if 'mae'         in n: return 'MAE-only'
    return None

df['model'] = df['experiment'].apply(classify)
df = df.dropna(subset=['model'])

agg = (df.groupby(['model', 'label_fraction'])
         .agg(mu=('auroc_mean', 'mean'),
              sd=('auroc_mean', 'std'))
         .reset_index())

KEEP = [0.01, 0.05, 0.10, 1.00]
agg  = agg[agg['label_fraction'].isin(KEEP)].copy()
agg['pct'] = (agg['label_fraction'] * 100).round(0).astype(int)
agg['sd']  = agg['sd'].fillna(0.0)

SUPERVISED_100 = 0.861

# ── Sanity-check: print the aggregated table ──────────────────────────────────
print('\n-- Aggregated data ------------------------------------------')
print(agg.sort_values(['model', 'pct']).to_string(index=False))
print('-------------------------------------------------------------\n')

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':         'DejaVu Serif',
    'font.size':           11,
    'axes.labelsize':      12,
    'axes.titlesize':      12,
    'axes.titleweight':    'bold',
    'axes.spines.top':     False,
    'axes.spines.right':   False,
    'axes.linewidth':      0.85,
    'axes.grid':           True,
    'grid.linestyle':      '--',
    'grid.linewidth':      0.55,
    'grid.alpha':          0.35,
    'legend.fontsize':     9.5,
    'legend.framealpha':   0.95,
    'legend.edgecolor':    '#cccccc',
    'legend.handlelength': 2.4,
    'xtick.direction':     'in',
    'ytick.direction':     'in',
    'ytick.minor.visible': True,
    'figure.dpi':          150,
    'savefig.dpi':         300,
})

MODELS = {
    'PA-HybridSSL (Ours)': dict(
        color='#1565C0', marker='o', ms=9,
        lw=2.5, ls='-', zorder=5, band_alpha=0.13),
    'SimCLR (Contrastive only)': dict(
        color='#C62828', marker='s', ms=8,
        lw=2.0, ls='--', zorder=4, band_alpha=0.11),
    'MAE-only': dict(
        color='#2E7D32', marker='^', ms=8,
        lw=1.8, ls='-.', zorder=3, band_alpha=0.10),
}

# ── Figure ────────────────────────────────────────────────────────────────────
# Extra bottom margin (0.16) so the italic caption line is never clipped
fig = plt.figure(figsize=(9.0, 6.0))
gs  = gridspec.GridSpec(
    1, 1, figure=fig,
    left=0.11, right=0.97,
    top=0.87,  bottom=0.16,   # <── wider bottom margin
)
ax = fig.add_subplot(gs[0])

# ── Inset: upper-LEFT corner — clear of legend (lower-right) and annotation ──
# Position: [x0, y0, width, height] in axes-fraction coords
ax_ins = ax.inset_axes([0.03, 0.52, 0.36, 0.44])
ax_ins.set_xscale('log')
ax_ins.set_xlim(0.80, 12)
ax_ins.set_xticks([1, 5, 10])
ax_ins.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax_ins.tick_params(labelsize=8, direction='in')
ax_ins.grid(True, linestyle='--', linewidth=0.45, alpha=0.30)
ax_ins.set_title('Low-label regime\n(1 – 10 %)',
                 fontsize=8.0, fontweight='bold', pad=3)
for sp in ('top', 'right'):
    ax_ins.spines[sp].set_visible(False)

# ── Plot all models ───────────────────────────────────────────────────────────
ins_y_vals = []   # collect to set inset y-lim precisely

for model_name, sty in MODELS.items():
    sub = agg[agg['model'] == model_name].sort_values('pct')
    if sub.empty:
        print(f'  [WARN] No data for {model_name}')
        continue

    x   = sub['pct'].values.astype(float)
    y   = sub['mu'].values
    err = sub['sd'].values

    # Main axes
    ax.plot(x, y,
            color=sty['color'], marker=sty['marker'],
            markersize=sty['ms'], linewidth=sty['lw'],
            linestyle=sty['ls'], zorder=sty['zorder'],
            label=model_name, clip_on=False)
    ax.fill_between(x,
                    np.clip(y - err, 0, 1),
                    np.clip(y + err, 0, 1),
                    color=sty['color'],
                    alpha=sty['band_alpha'],
                    zorder=sty['zorder'] - 1)
    ax.errorbar(x, y, yerr=err,
                fmt='none', ecolor=sty['color'],
                elinewidth=1.1, capsize=4, capthick=1.1,
                zorder=sty['zorder'] + 1)

    # Inset (1 %, 5 %, 10 % only)
    mask = x <= 10
    if mask.sum() > 1:
        ax_ins.plot(x[mask], y[mask],
                    color=sty['color'], marker=sty['marker'],
                    markersize=sty['ms'] - 1,
                    linewidth=sty['lw'] - 0.4,
                    linestyle=sty['ls'],
                    zorder=sty['zorder'], clip_on=False)
        ax_ins.fill_between(
            x[mask],
            np.clip(y[mask] - err[mask], 0, 1),
            np.clip(y[mask] + err[mask], 0, 1),
            color=sty['color'], alpha=sty['band_alpha'])
        ax_ins.errorbar(x[mask], y[mask], yerr=err[mask],
                        fmt='none', ecolor=sty['color'],
                        elinewidth=0.9, capsize=3, capthick=0.9)
        ins_y_vals.extend((y[mask] - err[mask]).tolist())
        ins_y_vals.extend((y[mask] + err[mask]).tolist())

# ── Inset y-limits ────────────────────────────────────────────────────────────
if ins_y_vals:
    pad_ins = 0.003
    ax_ins.set_ylim(
        max(0.0, min(ins_y_vals) - pad_ins),
        min(1.0, max(ins_y_vals) + pad_ins),
    )
    ax_ins.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    # Keep y-ticks sparse so inset stays readable
    ax_ins.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune='both'))

# Supervised line in inset
ax_ins.axhline(SUPERVISED_100, color='#555555',
               lw=1.0, ls=(0, (5, 4)), alpha=0.70, zorder=2)

# ── Supervised 100 % reference line (main axes) ───────────────────────────────
ax.axhline(SUPERVISED_100, color='#555555',
           lw=1.3, ls=(0, (5, 4)), alpha=0.75, zorder=2)

# Label INSIDE the axes at the far right — never bleeds below the frame
ax.text(
    0.985, SUPERVISED_100,
    f'  Supervised 100 % (scratch) = {SUPERVISED_100:.3f}',
    transform=ax.get_yaxis_transform(),   # x in axes [0,1], y in data coords
    fontsize=8.0, color='#444444',
    ha='left', va='bottom', style='italic',
)

# ── Headline annotation ───────────────────────────────────────────────────────
# Arrow points to the PA-HybridSSL 1 % data point.
# Text box placed lower-centre so it avoids both the inset (upper-left)
# and the legend (lower-right).
pa_row = agg[(agg['model'] == 'PA-HybridSSL (Ours)') & (agg['pct'] == 1)]
if not pa_row.empty:
    y_pa   = float(pa_row['mu'].iloc[0])
    gap_pp = (y_pa - SUPERVISED_100) * 100

    # Determine text position dynamically based on y range
    all_mu = agg['mu'].values
    y_lo   = all_mu.min() - 0.008
    y_hi   = all_mu.max() + 0.010
    y_span = y_hi - y_lo

    # Place text at 25 % height, 40 % across — avoids inset and legend
    text_x = 5.5
    text_y = y_lo + 0.22 * y_span

    ax.annotate(
        f'1 % labels → AUROC {y_pa:.3f}\n'
        f'(+{gap_pp:.1f} pp vs full supervision)',
        xy=(1, y_pa),
        xytext=(text_x, text_y),
        fontsize=9.0, color='#1565C0',
        ha='left', va='center',
        arrowprops=dict(
            arrowstyle='->',
            color='#1565C0',
            lw=1.3,
            connectionstyle='arc3,rad=0.25',   # curves UNDER the inset
        ),
        bbox=dict(
            boxstyle='round,pad=0.45',
            facecolor='#E8F1FB',
            edgecolor='#1565C0',
            linewidth=0.9,
            alpha=0.95,
        ),
        zorder=10,
    )

# ── Main axes formatting ──────────────────────────────────────────────────────
ax.set_xscale('log')
ax.set_xticks([1, 5, 10, 100])
ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax.set_xlim(0.75, 135)

all_y = agg['mu'].values
y_lo  = max(0.0, all_y.min() - 0.010)
y_hi  = min(1.0, all_y.max() + 0.012)
ax.set_ylim(y_lo, y_hi)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

ax.set_xlabel('Labeled Data Fraction (%)', labelpad=7)
ax.set_ylabel('AUROC  (PTB-XL binary task)', labelpad=7)

# Zoom indicator — connect inset to its region on the main axes
ax.indicate_inset_zoom(ax_ins, edgecolor='#999999',
                       linewidth=0.85, alpha=0.55)

# ── Legend — lower-right, unobstructed ────────────────────────────────────────
handles, labels_l = ax.get_legend_handles_labels()
sup_handle = Line2D(
    [0], [0],
    color='#555555', lw=1.3,
    ls=(0, (5, 4)), alpha=0.75,
    label=f'Supervised 100 % (scratch) = {SUPERVISED_100:.3f}',
)
handles.append(sup_handle)
labels_l.append(sup_handle.get_label())

ax.legend(
    handles, labels_l,
    loc='lower right',
    frameon=True,
    borderpad=0.80,
    labelspacing=0.60,
    handletextpad=0.65,
)

# ── Titles ────────────────────────────────────────────────────────────────────
fig.suptitle(
    'Figure 6 — Label Efficiency: PA-HybridSSL vs SSL Baselines  (PTB-XL)',
    fontsize=13, fontweight='bold', y=0.975,
)
ax.set_title(
    'ResNet1D encoder · patient-aware split · binary arrhythmia detection',
    fontsize=9.5, color='#444444', pad=7,
)

# ── Caption — placed well below axes so it never clips ────────────────────────
fig.text(
    0.5, 0.045,                          # higher than before (was 0.018)
    'Shaded bands and error bars = ±1 std over 3 seeds.  '
    'Dashed line = fully supervised ResNet1D trained on 100 % labels (Table I).  '
    'X-axis is log-scaled.',
    ha='center', fontsize=8.0,
    color='#555555', style='italic',
)

# ── Save ──────────────────────────────────────────────────────────────────────
for ext in ('pdf', 'png', 'svg'):
    out = STEM.with_suffix(f'.{ext}')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f'  Saved -> {out}')

plt.close()
print('\nDone.')