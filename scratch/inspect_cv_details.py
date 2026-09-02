import json
import numpy as np
import pandas as pd

with open('scratch/phase2b_cv_results.json') as f:
    data = json.load(f)

patterns = data['patterns']
categories = data['categories']

print("Patterns:", patterns)
print("Categories:", categories)

# Aggregate confusion matrices across 5 folds
base_cm_pat = np.zeros((len(patterns), len(patterns)), dtype=int)
cand_cm_pat = np.zeros((len(patterns), len(patterns)), dtype=int)
base_cm_cat = np.zeros((len(categories), len(categories)), dtype=int)
cand_cm_cat = np.zeros((len(categories), len(categories)), dtype=int)

for fold in range(5):
    base_cm_pat += np.array(data['baseline_raw_folds'][fold]['confusion_pattern'])
    cand_cm_pat += np.array(data['candidate_raw_folds'][fold]['confusion_pattern'])
    base_cm_cat += np.array(data['baseline_raw_folds'][fold]['confusion_category'])
    cand_cm_cat += np.array(data['candidate_raw_folds'][fold]['confusion_category'])

print("\n=== Baseline Pattern CM (All 5 Folds, Total N=112) ===")
print(pd.DataFrame(base_cm_pat, index=patterns, columns=patterns))

print("\n=== Candidate Pattern CM (All 5 Folds, Total N=112) ===")
print(pd.DataFrame(cand_cm_pat, index=patterns, columns=patterns))

print("\n=== Baseline Category CM (All 5 Folds, Total N=112) ===")
print(pd.DataFrame(base_cm_cat, index=categories, columns=categories))

print("\n=== Candidate Category CM (All 5 Folds, Total N=112) ===")
print(pd.DataFrame(cand_cm_cat, index=categories, columns=categories))

# Per-class metrics
def calc_per_class(cm, names):
    prec = np.diag(cm) / np.maximum(cm.sum(axis=0), 1)
    rec = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    support = cm.sum(axis=1)
    return pd.DataFrame({'Support': support, 'Precision': prec.round(4), 'Recall': rec.round(4), 'F1': f1.round(4)}, index=names)

print("\n=== Baseline Pattern Per-Class ===")
print(calc_per_class(base_cm_pat, patterns))

print("\n=== Candidate Pattern Per-Class ===")
print(calc_per_class(cand_cm_pat, patterns))

print("\n=== Baseline Category Per-Class ===")
print(calc_per_class(base_cm_cat, categories))

print("\n=== Candidate Category Per-Class ===")
print(calc_per_class(cand_cm_cat, categories))
