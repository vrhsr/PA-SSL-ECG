# Paper Strengthening Plan for PA-SSL

This document is a research-strength roadmap focused on moving the project from a promising engineering prototype to a paper with strong reviewer confidence.

## Executive assessment

PA-SSL already has a compelling idea (physiology-constrained augmentations + temporal positives), but the current manuscript and repo still look *pre-result* in multiple places. The biggest weakness is not novelty; it is **evidence depth and rigor density**.

If you execute the top-priority items below, your paper strength will increase significantly.

---

## Priority 0 (must-do before submission)

## 1) Replace all placeholder results with complete, multi-seed tables

Current manuscript tables still contain placeholders (`--`). A paper without completed core tables will be desk-rejected or heavily down-scored.

**Required updates**
- Fill label-efficiency table for 1%, 5%, 10%, 25%, 50%, 100% labels.
- Report mean ± std across at least 3 seeds (prefer 5).
- Add AUROC + AUPRC + Macro-F1 as primary triad.
- Add 95% bootstrap CI on test metrics for headline comparisons.

## 2) Run stronger and broader baseline suite

Most reviewers will expect stronger SSL baselines than SimCLR variants only.

**Minimum competitive baseline set**
- Supervised strong baseline (same encoder and training budget).
- SimCLR (existing).
- TS2Vec (already present in repo architecture space).
- CPC/TNC/CLOCS-style temporal SSL (or closest reproducible alternative).
- A simple but strong feature-engineering baseline (XGBoost/LightGBM + wavelet/HRV features).

**Fairness rules**
- Match encoder capacity where possible.
- Match training epochs and data access.
- Run identical patient-aware splits and seeds.

## 3) Add strict statistical testing in the main results section

Reviewers often reject medical-ML work when “best number wins” is used without inferential evidence.

**Required statistical protocol**
- Primary metric: AUROC (binary), secondary: AUPRC, Macro-F1.
- Across seeds: Wilcoxon signed-rank or paired t-test (declare normality assumption).
- Across many methods: Holm-Bonferroni correction.
- Report effect size (Cliff’s delta or Cohen’s d).

---

## Priority 1 (high impact)

## 4) Strengthen leakage and split-validity guarantees

You already use patient-aware splitting, which is excellent. Strengthen confidence further:

- Add a split audit table in appendix:
  - #patients per split
  - #records per split
  - class prevalence per split
  - patient overlap checks (0 expected)
- Keep one **locked final test split** used once for final reporting.

## 5) Expand external validity and transfer claims

Current narrative claims cross-dataset transfer; make this a core contribution with full matrix evidence.

**Required experiments**
- Train on each dataset, test on each other dataset (3x3 transfer matrix).
- Report domain shift drop (% relative AUROC decline).
- Add “few-shot adaptation” experiments (e.g., 1% labels in target domain).

## 6) Clinical utility framing beyond aggregate classification

Aggregate AUROC alone is not enough for clinical papers.

- Report sensitivity at fixed specificity levels (e.g., 90%, 95%).
- Report PPV/NPV at clinically relevant prevalence assumptions.
- Add threshold calibration strategy (validation-selected threshold, then fixed on test).

---

## Priority 2 (differentiation boosters)

## 7) Upgrade ablations from component toggles to mechanism-level analysis

You should demonstrate *why* physiology-aware augmentation works.

- Leave-one-augmentation-out ablations.
- QRS protection stress test: vary protection width and show morphology/accuracy tradeoff.
- Temporal scale ablation: ±1 vs ±1,2 vs ±1,2,3.
- Loss weight sweep for alpha/beta.
- Augmentation validity metrics linked to downstream gains (correlation analysis).

## 8) Robustness and stress testing

- SNR sweep with baseline wander/EMG/powerline mixtures.
- Sampling rate mismatch robustness (100 Hz train, altered test rates).
- Missing-beat and dropout stress tests.
- Adversarially hard negatives (morphologically similar but rhythm-different beats).

## 9) Interpretability with clinician-facing validation

Current Grad-CAM evidence is useful but may be viewed as qualitative only.

- Add cardiologist-guided region overlap scoring (even n=1/2 readers is valuable).
- Compare saliency concentration over P/QRS/T regions across methods.
- Include failure-case interpretability panel.

---

## Writing-level upgrades that materially improve acceptance odds

## 10) Tighten claim calibration

Avoid “state-of-the-art” language unless all major baselines and significance tests are present.
Use precise framing:
- “Consistently outperforms compared methods under patient-aware splits”
- “Improves label efficiency by X% relative AUROC at 1% labels”

## 11) Add explicit threats-to-validity section

Include:
- dataset geographic/device bias
- annotation noise
- binary-label simplification limits
- single-lead scope
- retrospective-only evaluation

## 12) Reproducibility package (camera-ready expectation)

- publish exact run scripts and seeds
- include environment lockfile / Docker image hash
- provide a results manifest (CSV/JSON) for every table and figure
- deterministic mode switch and reproducibility statement

---

## 6-week execution roadmap

## Week 1–2: Evidence completion
- Complete all placeholder tables and figures.
- Run 3–5 seed repeats.
- Freeze split protocol and results schema.

## Week 3–4: Baselines + statistics
- Add missing strong SSL baselines.
- Run statistical tests + effect sizes + multiple-comparison correction.

## Week 5: Robustness + ablations
- Mechanism-level ablations and stress tests.
- Build clinician-meaningful threshold metrics.

## Week 6: Manuscript hardening
- Rewrite results around effect sizes and confidence intervals.
- Add limitations/threats validity/reproducibility package links.

---

## Suggested “Definition of Done” for submission readiness

Use this checklist to decide if the paper is actually submission-ready:

- [ ] No placeholder values remain in manuscript tables.
- [ ] At least 3 strong external baselines included and fairly tuned.
- [ ] All headline claims supported by significance tests + effect sizes.
- [ ] Cross-dataset transfer matrix complete.
- [ ] Clinical operating-point metrics included.
- [ ] Robustness and ablation sections quantify mechanism, not only performance.
- [ ] Reproducibility artifacts published.

