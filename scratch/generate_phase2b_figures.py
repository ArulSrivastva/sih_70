import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('results/figures/p2', exist_ok=True)

with open('scratch/phase2b_cv_results.json') as f:
    cv_data = json.load(f)

with open('scratch/final_test_results.json') as f:
    test_data = json.load(f)

patterns = cv_data['patterns']
categories = cv_data['categories']

# -------------------------------------------------------------
# Figure 1: Confusion Matrices (Baseline vs Candidate)
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# 1.1 Baseline Pattern CM (CV Aggregate)
base_cm_pat = np.sum([f['confusion_pattern'] for f in cv_data['baseline_raw_folds']], axis=0)
cand_cm_pat = np.sum([f['confusion_pattern'] for f in cv_data['candidate_raw_folds']], axis=0)

im00 = axes[0, 0].imshow(base_cm_pat, cmap='Blues')
axes[0, 0].set_title('Baseline Pattern CM (5-Fold CV Aggregate)', fontsize=11, fontweight='bold')
axes[0, 0].set_xticks([0, 1, 2])
axes[0, 0].set_yticks([0, 1, 2])
axes[0, 0].set_xticklabels(['Curved Band', 'Eye Visible', 'Shear'])
axes[0, 0].set_yticklabels(['Curved Band', 'Eye Visible', 'Shear'])
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('True')
for i in range(3):
    for j in range(3):
        color = 'white' if base_cm_pat[i, j] > 30 else 'black'
        axes[0, 0].text(j, i, str(int(base_cm_pat[i, j])), ha='center', va='center', color=color, fontweight='bold')

im01 = axes[0, 1].imshow(cand_cm_pat, cmap='Reds')
axes[0, 1].set_title('Candidate (Dvorak) Pattern CM (5-Fold CV Aggregate)', fontsize=11, fontweight='bold')
axes[0, 1].set_xticks([0, 1, 2])
axes[0, 1].set_yticks([0, 1, 2])
axes[0, 1].set_xticklabels(['Curved Band', 'Eye Visible', 'Shear'])
axes[0, 1].set_yticklabels(['Curved Band', 'Eye Visible', 'Shear'])
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('True')
for i in range(3):
    for j in range(3):
        color = 'white' if cand_cm_pat[i, j] > 30 else 'black'
        axes[0, 1].text(j, i, str(int(cand_cm_pat[i, j])), ha='center', va='center', color=color, fontweight='bold')

# 1.2 Category CM (CV Aggregate)
base_cm_cat = np.sum([f['confusion_category'] for f in cv_data['baseline_raw_folds']], axis=0)
cand_cm_cat = np.sum([f['confusion_category'] for f in cv_data['candidate_raw_folds']], axis=0)

short_cat = ['Cyclonic', 'Deep Dep', 'Dep', 'Ext Severe', 'Severe', 'Very Sev']
im10 = axes[1, 0].imshow(base_cm_cat, cmap='Blues')
axes[1, 0].set_title('Baseline Category CM (5-Fold CV Aggregate)', fontsize=11, fontweight='bold')
axes[1, 0].set_xticks(range(6))
axes[1, 0].set_yticks(range(6))
axes[1, 0].set_xticklabels(short_cat, rotation=25, ha='right', fontsize=9)
axes[1, 0].set_yticklabels(short_cat, fontsize=9)
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('True')
for i in range(6):
    for j in range(6):
        color = 'white' if base_cm_cat[i, j] > 10 else 'black'
        axes[1, 0].text(j, i, str(int(base_cm_cat[i, j])), ha='center', va='center', color=color, fontsize=9)

im11 = axes[1, 1].imshow(cand_cm_cat, cmap='Reds')
axes[1, 1].set_title('Candidate (Dvorak) Category CM (5-Fold CV Aggregate)', fontsize=11, fontweight='bold')
axes[1, 1].set_xticks(range(6))
axes[1, 1].set_yticks(range(6))
axes[1, 1].set_xticklabels(short_cat, rotation=25, ha='right', fontsize=9)
axes[1, 1].set_yticklabels(short_cat, fontsize=9)
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('True')
for i in range(6):
    for j in range(6):
        color = 'white' if cand_cm_cat[i, j] > 10 else 'black'
        axes[1, 1].text(j, i, str(int(cand_cm_cat[i, j])), ha='center', va='center', color=color, fontsize=9)

plt.tight_layout()
plt.savefig('results/figures/p2/p2_phase2b_confusion_matrix.png', dpi=200)
plt.close()
print("Saved p2_phase2b_confusion_matrix.png")

# -------------------------------------------------------------
# Figure 2: Consistency Comparison Across Folds
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
folds = [f'Fold {i+1}' for i in range(5)]
base_phys = [f['physical_consistency_rate'] * 100 for f in cv_data['baseline_raw_folds']]
cand_phys = [f['physical_consistency_rate'] * 100 for f in cv_data['candidate_raw_folds']]

x = np.arange(len(folds))
width = 0.35

ax.bar(x - width/2, base_phys, width, label=f'Baseline (Mean: {np.mean(base_phys):.1f}%)', color='#4A90E2', alpha=0.85)
ax.bar(x + width/2, cand_phys, width, label=f'Candidate Dvorak (Mean: {np.mean(cand_phys):.1f}%)', color='#E94A4A', alpha=0.85)

ax.set_ylabel('Physical Consistency Rate (%)')
ax.set_title('P2 Phase 2B: Physical Consistency Rate Across 5 Folds', fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(folds)
ax.set_ylim(0, 110)
ax.legend(loc='upper right')
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('results/figures/p2/p2_phase2b_consistency_comparison.png', dpi=200)
plt.close()
print("Saved p2_phase2b_consistency_comparison.png")

# -------------------------------------------------------------
# Figure 3: Per-Class F1 Comparison
# -------------------------------------------------------------
def calc_f1s(cm):
    prec = np.diag(cm) / np.maximum(cm.sum(axis=0), 1)
    rec = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    return 2 * prec * rec / np.maximum(prec + rec, 1e-9)

base_cat_f1 = calc_f1s(base_cm_cat)
cand_cat_f1 = calc_f1s(cand_cm_cat)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(short_cat))
width = 0.35

ax.bar(x - width/2, base_cat_f1, width, label='Baseline', color='#4A90E2', alpha=0.85)
ax.bar(x + width/2, cand_cat_f1, width, label='Candidate Dvorak', color='#E94A4A', alpha=0.85)

ax.set_ylabel('F1 Score')
ax.set_title('P2 Phase 2B: Per-Class Category F1 Score (5-Fold CV Aggregate)', fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(short_cat, rotation=15, ha='right')
ax.set_ylim(0, 0.6)
ax.legend(loc='upper right')
ax.grid(axis='y', linestyle='--', alpha=0.5)

for i in range(len(short_cat)):
    ax.text(i - width/2, base_cat_f1[i] + 0.015, f"{base_cat_f1[i]:.2f}", ha='center', fontsize=8, color='#1A4B82')
    ax.text(i + width/2, cand_cat_f1[i] + 0.015, f"{cand_cat_f1[i]:.2f}", ha='center', fontsize=8, color='#9C1A1A')

plt.tight_layout()
plt.savefig('results/figures/p2/p2_phase2b_per_class_f1.png', dpi=200)
plt.close()
print("Saved p2_phase2b_per_class_f1.png")
