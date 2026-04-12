import re, json, glob, os
import pandas as pd

LOG_DIR = "remote/logs/20260412_101937"

def parse_metrics_from_log(filepath):
    try:
        with open(filepath, encoding='utf-8', errors='ignore') as f:
            txt = f.read()
        auroc = re.findall(r'auroc[:\s=]+([0-9.]+)', txt, re.IGNORECASE)
        acc   = re.findall(r'accuracy[:\s=]+([0-9.]+)', txt, re.IGNORECASE)
        f1    = re.findall(r'f1_macro[:\s=]+([0-9.]+)', txt, re.IGNORECASE)
        auprc = re.findall(r'auprc[:\s=]+([0-9.]+)', txt, re.IGNORECASE)
        return {
            'auroc': float(auroc[-1]) if auroc else None,
            'acc':   float(acc[-1])   if acc   else None,
            'f1':    float(f1[-1])    if f1    else None,
            'auprc': float(auprc[-1]) if auprc else None,
        }
    except Exception as e:
        return {'error': str(e)}

# 1. Cross-Dataset Transfer
print("\n=== CROSS-DATASET TRANSFER ===")
for ds, fname in [('PTB-XL', 'full_eval_passl_resnet_hybrid.log'),
                  ('MIT-BIH', 'full_transfer_mitbih.log'),
                  ('Chapman', 'full_transfer_chapman.log')]:
    m = parse_metrics_from_log(os.path.join(LOG_DIR, fname))
    print(f"  {ds}: AUROC={m.get('auroc')}, Acc={m.get('acc')}, F1={m.get('f1')}, AUPRC={m.get('auprc')}")

# 2. All SSL model evaluations (label efficiency at 1% and 10%)
print("\n=== PA-SSL MODEL LABEL EFFICIENCY ===")
for model_dir in glob.glob("remote/ssl_*/evaluation/label_efficiency.csv"):
    name = model_dir.split(os.sep)[1]
    try:
        df = pd.read_csv(model_dir)
        for frac in [0.01, 0.1, 1.0]:
            sub = df[df['label_fraction'] == frac]
            if not sub.empty:
                print(f"  {name} @ {int(frac*100)}%: "
                      f"Acc={sub['linear_accuracy'].mean():.4f}, "
                      f"F1={sub['linear_f1_macro'].mean():.4f}, "
                      f"AUROC={sub['linear_auroc'].mean():.4f}, "
                      f"AUPRC={sub['linear_auprc'].mean():.4f}")
    except Exception as e:
        print(f"  {name}: ERROR - {e}")

# 3. Ablation results
print("\n=== ABLATION RESULTS ===")
try:
    df_abl = pd.read_csv("remote/ablation_results.csv")
    for _, row in df_abl.iterrows():
        if pd.notna(row.get('configuration')):
            print(f"  {row['configuration']}: AUROC={row['auroc']:.4f}, F1={row['f1_macro']:.4f}, Acc={row['accuracy']:.4f}")
except Exception as e:
    print(f"  ERROR: {e}")

# 4. OOD Detection
print("\n=== OOD DETECTION ===")
m = parse_metrics_from_log(os.path.join(LOG_DIR, 'full_ood.log'))
print(f"  OOD: {m}")

# 5. Ablation factorial from abl_ logs
print("\n=== FACTORIAL ABLATION (Encoder x Objective) ===")
for enc in ['resnet1d', 'wavkan']:
    for obj in ['contrastive', 'mae', 'hybrid']:
        fname = f"abl_{enc}_{obj}_s42.log"
        m = parse_metrics_from_log(os.path.join("remote/logs", fname))
        print(f"  {enc} x {obj}: AUROC={m.get('auroc')}, Acc={m.get('acc')}")
