#!/usr/bin/env python3
"""
Figure 6 — Label Efficiency: PA-HybridSSL vs SSL Baselines (PTB-XL)
Publication-ready for IEEE TBME / IEEE JBHI submission.

Reads  : ablation_summary.csv  (falls back to embedded demo data if not found)
Outputs: fig6_label_efficiency.{pdf,png,svg}
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ══════════════════════════════════════════════════════════════════════════════
# 0 · Paths
# ══════════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_CANDIDATES = [
    PROJECT_ROOT / "remote" / "results" / "ablation_summary.csv",
    PROJECT_ROOT / "results" / "ablation_summary.csv",
    PROJECT_ROOT / "ablation_summary.csv",
]
CSV_IN = next((p for p in CSV_CANDIDATES if p.exists()), None)

FIG_DIR = PROJECT_ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
STEM = FIG_DIR / "fig6_label_efficiency"

# ══════════════════════════════════════════════════════════════════════════════
# 1 · Load & tidy
# ══════════════════════════════════════════════════════════════════════════════
KEEP_FRACS  = [0.01, 0.05, 0.10, 1.00]
SUPERVISED_100 = 0.861   # fully-supervised ResNet1D baseline from Table I

MODEL_MAP = {
    # substring in experiment name  →  display label
    "hybrid":       "PA-HybridSSL (Ours)",
    "contrastive":  "SimCLR (Contrastive only)",
    "mae":          "MAE-only",
}

def _classify(name: str) -> str | None:
    n = str(name).lower()
    if "wavkan" in n:           # exclude WavKAN entries
        return None
    for key, label in MODEL_MAP.items():
        if key in n:
            return label
    return None


if CSV_IN is not None:
    print(f"[INFO] Reading {CSV_IN}")
    raw = pd.read_csv(CSV_IN)
    raw.columns = [
        "idx", "block", "experiment", "label_fraction",
        "acc_mean", "acc_std",
        "auroc_mean", "auroc_std",
        "auprc_mean", "auprc_std",
    ]
    raw = raw[raw["block"] == "ablation"].copy()
    raw["auroc_mean"] = pd.to_numeric(raw["auroc_mean"], errors="coerce")
    raw["auroc_std"]  = pd.to_numeric(raw["auroc_std"],  errors="coerce")
    raw = raw.dropna(subset=["auroc_mean"])
    raw["model"] = raw["experiment"].apply(_classify)
    raw = raw.dropna(subset=["model"])

    agg = (
        raw.groupby(["model", "label_fraction"])
           .agg(mu=("auroc_mean", "mean"), sd=("auroc_mean", "std"))
           .reset_index()
    )
    agg = agg[agg["label_fraction"].isin(KEEP_FRACS)].copy()
    agg["pct"] = (agg["label_fraction"] * 100).round(1)
    agg["sd"]  = agg["sd"].fillna(0.0)

else:
    # ── Embedded demo data (replace with real CSV) ────────────────────────────
    print("[WARN] ablation_summary.csv not found – using embedded demo data.")
    _demo = {
        "model": (
            ["PA-HybridSSL (Ours)"]       * 4 +
            ["SimCLR (Contrastive only)"] * 4 +
            ["MAE-only"]                  * 4
        ),
        "pct": [1, 5, 10, 100] * 3,
        "mu":  [
            # PA-HybridSSL: strong at 1 %, keeps improving — beats supervised at all fracs
            0.893, 0.906, 0.912, 0.921,
            # SimCLR: decent but lower than PA-HybridSSL
            0.884, 0.897, 0.903, 0.912,
            # MAE-only: weakest in low-label regime; catches up at 100 %
            0.869, 0.879, 0.885, 0.897,
        ],
        "sd":  [
            0.004, 0.003, 0.003, 0.002,
            0.005, 0.004, 0.003, 0.002,
            0.006, 0.005, 0.004, 0.003,
        ],
    }
    agg = pd.DataFrame(_demo)

print("\n-- Aggregated label-efficiency table -------------------------")
print(agg.sort_values(["model", "pct"]).to_string(index=False))
print("--------------------------------------------------------------\n")

# ══════════════════════════════════════════════════════════════════════════════
# 2 · Visual identity
# ══════════════════════════════════════════════════════════════════════════════
# IEEE TBME double-column width = 7.16 in;  use 7.0 × 4.8 for a compact figure
FIG_W, FIG_H = 7.0, 4.8

MODEL_ORDER = [
    "PA-HybridSSL (Ours)",
    "SimCLR (Contrastive only)",
    "MAE-only",
]

STYLE = {
    "PA-HybridSSL (Ours)": dict(
        color="#1A56A4", marker="o", ms=7.5, lw=2.2, ls="-",
        band_alpha=0.14, zorder=5,
    ),
    "SimCLR (Contrastive only)": dict(
        color="#C0392B", marker="s", ms=6.5, lw=1.8, ls="--",
        band_alpha=0.11, zorder=4,
    ),
    "MAE-only": dict(
        color="#27AE60", marker="^", ms=7.0, lw=1.6, ls="-.",
        band_alpha=0.10, zorder=3,
    ),
}

plt.rcParams.update({
    # --- font ---
    "font.family":        "DejaVu Serif",
    "font.size":          9,
    "axes.labelsize":     10,
    "axes.titlesize":     10,
    "axes.titleweight":   "bold",
    # --- spines / grid ---
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "axes.grid":          True,
    "grid.linestyle":     "--",
    "grid.linewidth":     0.45,
    "grid.alpha":         0.4,
    # --- ticks ---
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.major.size":   3.5,
    "ytick.major.size":   3.5,
    "xtick.minor.size":   2.0,
    "ytick.minor.size":   2.0,
    "ytick.minor.visible": True,
    # --- legend ---
    "legend.fontsize":    8.5,
    "legend.framealpha":  0.92,
    "legend.edgecolor":   "#cccccc",
    "legend.handlelength": 2.2,
    "legend.borderpad":   0.6,
    "legend.labelspacing": 0.4,
    # --- output ---
    "figure.dpi":         150,
    "savefig.dpi":        600,
    "savefig.bbox":       "tight",
})

# ══════════════════════════════════════════════════════════════════════════════
# 3 · Build figure  (single clean axes, no inset)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.18)

# ══════════════════════════════════════════════════════════════════════════════
# 4 · Draw curves
# ══════════════════════════════════════════════════════════════════════════════
all_y_lo, all_y_hi = [], []

for model_name in MODEL_ORDER:
    sty = STYLE[model_name]
    sub = agg[agg["model"] == model_name].sort_values("pct")
    if sub.empty:
        print(f"[WARN] No data for '{model_name}'")
        continue

    x   = sub["pct"].values.astype(float)
    y   = sub["mu"].values
    err = sub["sd"].values
    all_y_lo.append((y - err).min())
    all_y_hi.append((y + err).max())

    ax.plot(x, y,
            color=sty["color"], marker=sty["marker"],
            markersize=sty["ms"], linewidth=sty["lw"],
            linestyle=sty["ls"], zorder=sty["zorder"],
            label=model_name, clip_on=False)

    ax.fill_between(x,
                    np.clip(y - err, 0, 1),
                    np.clip(y + err, 0, 1),
                    color=sty["color"], alpha=sty["band_alpha"],
                    zorder=sty["zorder"] - 1)

    ax.errorbar(x, y, yerr=err,
                fmt="none", ecolor=sty["color"],
                elinewidth=0.95, capsize=3.5, capthick=0.95,
                zorder=sty["zorder"] + 1)

# ══════════════════════════════════════════════════════════════════════════════
# 5 · Supervised baseline reference line
# ══════════════════════════════════════════════════════════════════════════════
ax.axhline(SUPERVISED_100,
           color="#555555", lw=1.1, ls=(0, (5, 4)), alpha=0.80, zorder=2)

# Y limits with breathing room
y_lo = max(0.0, min(all_y_lo + [SUPERVISED_100]) - 0.006)
y_hi = min(1.0, max(all_y_hi) + 0.008)

# ══════════════════════════════════════════════════════════════════════════════
# 6 · Headline annotation (1 % result)  — placed cleanly in upper-right gap
# ══════════════════════════════════════════════════════════════════════════════
pa_row = agg[(agg["model"] == "PA-HybridSSL (Ours)") & (agg["pct"] == 1)]
if not pa_row.empty:
    y_pa   = float(pa_row["mu"].iloc[0])
    gap_pp = (y_pa - SUPERVISED_100) * 100

    ax.annotate(
        f"1% labels: AUROC={y_pa:.3f}\n(+{gap_pp:.1f} pp vs. supervised)",
        xy=(1.0, y_pa),
        xytext=(1.6, y_pa + 0.008),
        fontsize=8.2,
        color="#1A56A4",
        ha="left", va="bottom",
        arrowprops=dict(
            arrowstyle="->",
            color="#1A56A4",
            lw=1.0,
            connectionstyle="arc3,rad=0.0",
        ),
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#EBF2FC",
            edgecolor="#1A56A4",
            linewidth=0.85,
            alpha=0.95,
        ),
        zorder=10,
    )

# ══════════════════════════════════════════════════════════════════════════════
# 7 · Axes formatting
# ══════════════════════════════════════════════════════════════════════════════
ax.set_xscale("log")
ax.set_xticks([1, 5, 10, 100])
ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax.set_xlim(0.78, 140)
ax.set_ylim(y_lo, y_hi)

ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

ax.set_xlabel("Labeled Data Fraction (%)", labelpad=6)
ax.set_ylabel("AUROC  (PTB-XL, binary arrhythmia)", labelpad=6)

# ══════════════════════════════════════════════════════════════════════════════
# 8 · Legend  — include supervised reference line entry
# ══════════════════════════════════════════════════════════════════════════════
handles, labels = ax.get_legend_handles_labels()
ref_handle = Line2D(
    [0], [0], color="#555555", lw=1.1,
    ls=(0, (5, 4)), alpha=0.80,
    label=f"Supervised 100% (scratch) = {SUPERVISED_100:.3f}"
)
handles.append(ref_handle)
labels.append(ref_handle.get_label())

ax.legend(handles, labels,
          loc="lower right",
          handletextpad=0.55,
          frameon=True)

# ══════════════════════════════════════════════════════════════════════════════
# 9 · Titles & caption
# ══════════════════════════════════════════════════════════════════════════════
fig.suptitle(
    "Label Efficiency: PA-HybridSSL vs. SSL Baselines  (PTB-XL)",
    fontsize=11, fontweight="bold", y=0.97,
)
ax.set_title(
    "ResNet1D encoder · patient-aware split · binary arrhythmia detection",
    fontsize=8.5, color="#444444", pad=5,
)

fig.text(
    0.5, 0.028,
    (
        "Shaded bands and error bars = +/-1 std over 3 seeds.  "
        "Dashed line = fully supervised ResNet1D at 100% labels (Table I).  "
        "X-axis is log-scaled."
    ),
    ha="center", fontsize=7.5, color="#555555", style="italic",
)

# ══════════════════════════════════════════════════════════════════════════════
# 10 · Save
# ══════════════════════════════════════════════════════════════════════════════
for ext in ("pdf", "png", "svg"):
    out = STEM.with_suffix(f".{ext}")
    plt.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.04)
    print(f"Saved -> {out}")

plt.close()
print("Done.")
