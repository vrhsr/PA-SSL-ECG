import pandas as pd
import numpy as np

SEP = '='*65

print(SEP)
print('DEEP RESULTS ANALYSIS — PA-HybridSSL')
print(SEP)

pa_new = {
    '1%':   {'auroc': 0.8923, 'acc': 0.8151, 'f1': 0.8128, 'std': 0.0058},
    '5%':   {'auroc': 0.9020, 'acc': 0.8232, 'f1': 0.8212, 'std': 0.0009},
    '10%':  {'auroc': 0.9033, 'acc': 0.8237, 'f1': 0.8218, 'std': 0.0003},
    '100%': {'auroc': 0.9071, 'acc': 0.8268, 'f1': 0.8248, 'std': 0.0},
}
sim_new = {
    '1%':   {'auroc': 0.8858, 'acc': 0.8097, 'f1': 0.8071, 'std': 0.0037},
    '5%':   {'auroc': 0.8994, 'acc': 0.8230, 'f1': 0.8211, 'std': 0.0012},
    '10%':  {'auroc': 0.9024, 'acc': 0.8231, 'f1': 0.8212, 'std': 0.0015},
    '100%': {'auroc': 0.9060, 'acc': 0.8269, 'f1': 0.8251, 'std': 0.0},
}
sup = {'1%': 0.8496, '5%': 0.8627, '10%': 0.8609, '100%': 0.8696}

# Old numbers in Table I
sim_old = {'1%': 0.6634, '10%': 0.7260}

print()
print('1) CRITICAL DISCREPANCY: Old SimCLR in paper vs actual eval')
print('-'*60)
for frac in ['1%', '10%']:
    old = sim_old[frac]
    new = sim_new[frac]['auroc']
    print(f'  SimCLR {frac:4s}  Paper says: {old:.4f}  Actual eval: {new:.4f}  Diff: {new-old:+.4f}')

print()
print('2) ACTUAL gap PA-SSL vs SimCLR (linear probe, new eval)')
print('-'*60)
for frac in ['1%', '5%', '10%', '100%']:
    gap = pa_new[frac]['auroc'] - sim_new[frac]['auroc']
    verdict = 'MEANINGFUL' if gap > 0.005 else ('tiny' if gap > 0.001 else 'negligible')
    print(f'  {frac:4s}  PA={pa_new[frac]["auroc"]:.4f}  SIM={sim_new[frac]["auroc"]:.4f}  gap={gap:+.4f}  => {verdict}')

print()
print('3) PA-SSL vs Supervised upper bound (SSL benefit)')
print('-'*60)
for frac in ['1%', '5%', '10%', '100%']:
    gap = pa_new[frac]['auroc'] - sup[frac]
    print(f'  {frac:4s}  PA-SSL={pa_new[frac]["auroc"]:.4f}  Sup={sup[frac]:.4f}  SSL-boost={gap:+.4f}')

print()
print('4) Saturation analysis')
print('-'*60)
pa_gain  = pa_new['100%']['auroc']  - pa_new['1%']['auroc']
sim_gain = sim_new['100%']['auroc'] - sim_new['1%']['auroc']
print(f'  PA-SSL AUROC gain from 1%->100%:  {pa_gain:+.4f}')
print(f'  SimCLR AUROC gain from 1%->100%: {sim_gain:+.4f}')
print('  => Both models plateau fast. SSL is most valuable at low labels.')

print()
print('5) kNN eval (representation quality, no classifier bias)')
print('-'*60)
knn_pa  = 0.8076
knn_sim = 0.7625
print(f'  kNN-20 PA-SSL:  {knn_pa:.4f}')
print(f'  kNN-20 SimCLR:  {knn_sim:.4f}')
print(f'  Gap:            {knn_pa-knn_sim:+.4f}  (5x larger than linear probe gap at 1%!)')

print()
print('6) Cross-dataset transfer MIT-BIH (from existing paper table)')
print('-'*60)
print('  PA-SSL AUROC:  0.9341')
print('  SimCLR AUROC:  0.8785')
print('  Gap:           +0.0556  (strongest advantage in the paper)')

print()
print('7) OOD detection — Euclidean metric (biggest structural gap)')
print('-'*60)
print('  PA-SSL Euclidean AUROC: 0.8053')
print('  SimCLR Euclidean AUROC: 0.6746')
print('  Gap:                    +0.1307')

print()
print(SEP)
print('VERDICT SUMMARY')
print(SEP)
print('''
STRENGTHS (publishable evidence):
  [+] MIT-BIH transfer:    +5.6 AUROC points — clinically significant
  [+] OOD (Euclidean):     +13.1 AUROC points — structural advantage
  [+] kNN-20 accuracy:     +4.5 points — better representation geometry
  [+] SSL > Supervised at 1%: +4.3 AUROC points over from-scratch training
  [+] Physiology probing:  higher R2 on QRS/QT features across all 3 seeds

WEAKNESSES / CONCERNS:
  [!] Linear probe gap at 10% labels: only 0.0009 AUROC — nearly zero
  [!] Linear probe gap at 100% labels: only 0.001 AUROC — negligible
  [!] OLD SimCLR numbers in Table I are wrong (0.6634 -> 0.8858 actual)
  [!] The "+0.1687 absolute improvement" claim in text is based on old/wrong SimCLR numbers
  [!] No fine-tuning results at all (end-to-end) — reviewers WILL ask

REQUIRED FIXES:
  [!] Update Table I SimCLR rows with new correct numbers
  [!] Reframe narrative: advantage is in transfer/OOD, not linear probe
  [!] Remove the incorrect "+0.1687" claim

OPTIONAL BUT RECOMMENDED:
  [ ] Run fine-tuning (end-to-end) for both models (1%, 10%, 100%)
  [ ] Add CLOCS or 3KG as additional SSL baseline
''')
