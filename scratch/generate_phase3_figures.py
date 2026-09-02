import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('results/figures/p2', exist_ok=True)

with open('scratch/phase3_cv_all.json') as f:
    cv_data = json.load(f)

with open('scratch/phase3_test_all.json') as f:
    test_data = json.load(f)

patterns = cv_data['E0_baseline']['patterns']
categories = cv_data['E0_baseline']['categories']
short_cat = ['Cyclonic', 'Deep Dep', 'Dep', 'Ext Severe', 'Severe', 'Very Sev']

# -------------------------------------------------------------
# 1. Figure 1: Confusion Matrices (E0 vs E1 vs E2 on CV Aggregate)
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for idx, (cfg, title, col) in enumerate([
    ('E0_baseline', 'E0: Baseline (lambda=0)', 'Blues'),
    ('E1_lambda_001', 'E1: Weak Phys (lambda=0.01)', 'Purples'),
    ('E2_lambda_005', 'E2: Mod Phys (lambda=0.05)', 'Oranges')
]):
    cm_pat = np.sum([f['confusion_pattern'] for f in cv_data[cfg]['raw_folds']], axis=0)
    cm_cat = np.sum([f['confusion_category'] for f in cv_data[cfg]['raw_folds']], axis=0)
    
    # Pattern CM
    axes[0, idx].imshow(cm_pat, cmap=col)
    axes[0, idx].set_title(f'{title}\nPattern CM (5-Fold CV Aggregate)', fontsize=10, fontweight='bold')
    axes[0, idx].set_xticks([0, 1, 2])
    axes[0, idx].set_yticks([0, 1, 2])
    axes[0, idx].set_xticklabels(['Curved', 'Eye', 'Shear'])
    axes[0, idx].set_yticklabels(['Curved', 'Eye', 'Shear'])
    axes[0, idx].set_xlabel('Predicted')
    axes[0, idx].set_ylabel('True')
    for i in range(3):
        for j in range(3):
            val = int(cm_pat[i, j])
            axes[0, idx].text(j, i, str(val), ha='center', va='center',
                              color='white' if val > 30 else 'black', fontweight='bold')
                              
    # Category CM
    axes[1, idx].imshow(cm_cat, cmap=col)
    axes[1, idx].set_title(f'{title}\nCategory CM (5-Fold CV Aggregate)', fontsize=10, fontweight='bold')
    axes[1, idx].set_xticks(range(6))
    axes[1, idx].set_yticks(range(6))
    axes[1, idx].set_xticklabels(short_cat, rotation=30, ha='right', fontsize=8)
    axes[1, idx].set_yticklabels(short_cat, fontsize=8)
    axes[1, idx].set_xlabel('Predicted')
    axes[1, idx].set_ylabel('True')
    for i in range(6):
        for j in range(6):
            val = int(cm_cat[i, j])
            axes[1, idx].text(j, i, str(val), ha='center', va='center',
                              color='white' if val > 10 else 'black', fontsize=8)

plt.tight_layout()
plt.savefig('results/figures/p2/p2_phase3_confusion_matrix.png', dpi=200)
plt.close()
print("Saved p2_phase3_confusion_matrix.png")

# -------------------------------------------------------------
# 2. Figure 2: CV Comparison (Metrics across E0, E1, E2)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.5))
metrics = ['Pattern Acc', 'Pattern Macro-F1', 'Category Acc', 'Category Macro-F1', 'Phys Consistency']
keys = ['pattern_accuracy', 'pattern_f1_macro', 'category_accuracy', 'category_f1_macro', 'physical_consistency_rate']

x = np.arange(len(metrics))
width = 0.25

e0_means = [cv_data['E0_baseline']['summary'][k]['mean'] for k in keys]
e0_stds = [cv_data['E0_baseline']['summary'][k]['std'] for k in keys]

e1_means = [cv_data['E1_lambda_001']['summary'][k]['mean'] for k in keys]
e1_stds = [cv_data['E1_lambda_001']['summary'][k]['std'] for k in keys]

e2_means = [cv_data['E2_lambda_005']['summary'][k]['mean'] for k in keys]
e2_stds = [cv_data['E2_lambda_005']['summary'][k]['std'] for k in keys]

ax.bar(x - width, e0_means, width, yerr=e0_stds, capsize=4, label='E0: Baseline (λ=0)', color='#4A90E2', alpha=0.85)
ax.bar(x, e1_means, width, yerr=e1_stds, capsize=4, label='E1: Weak Phys (λ=0.01)', color='#8E44AD', alpha=0.85)
ax.bar(x + width, e2_means, width, yerr=e2_stds, capsize=4, label='E2: Mod Phys (λ=0.05)', color='#E67E22', alpha=0.85)

ax.set_ylabel('Score')
ax.set_title('P2 Phase 3: 5-Fold Cross-Validation Metrics by Regularization Strength', fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.1)
ax.legend(loc='upper right')
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('results/figures/p2/p2_phase3_cv_comparison.png', dpi=200)
plt.close()
print("Saved p2_phase3_cv_comparison.png")

# -------------------------------------------------------------
# 3. Figure 3: Physics Consistency and Incompatible Pairs Rate
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

configs = ['E0 (λ=0)', 'E1 (λ=0.01)', 'E2 (λ=0.05)']
cv_phys = [cv_data['E0_baseline']['summary']['physical_consistency_rate']['mean'] * 100,
           cv_data['E1_lambda_001']['summary']['physical_consistency_rate']['mean'] * 100,
           cv_data['E2_lambda_005']['summary']['physical_consistency_rate']['mean'] * 100]

cv_incomp = [cv_data['E0_baseline']['summary']['incompatible_prediction_rate']['mean'] * 100,
             cv_data['E1_lambda_001']['summary']['incompatible_prediction_rate']['mean'] * 100,
             cv_data['E2_lambda_005']['summary']['incompatible_prediction_rate']['mean'] * 100]

test_phys = [test_data['E0_baseline']['physical_consistency_rate'] * 100,
             test_data['E1_lambda_001']['physical_consistency_rate'] * 100,
             test_data['E2_lambda_005']['physical_consistency_rate'] * 100]

test_incomp = [test_data['E0_baseline']['incompatible_prediction_rate'] * 100,
               test_data['E1_lambda_001']['incompatible_prediction_rate'] * 100,
               test_data['E2_lambda_005']['incompatible_prediction_rate'] * 100]

# Left: CV Consistency & Incompatible Rate
x = np.arange(len(configs))
width = 0.35
ax1.bar(x - width/2, cv_phys, width, label='Consistency Rate (%)', color='#2ECC71', alpha=0.85)
ax1.bar(x + width/2, cv_incomp, width, label='Incompatible Pair Rate (%)', color='#E74C3C', alpha=0.85)
ax1.set_ylabel('Percentage (%)')
ax1.set_title('Cross-Validation Physics Compliance', fontsize=11, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(configs)
ax1.set_ylim(0, 105)
ax1.legend(loc='upper right')
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# Right: Test Set
ax2.bar(x - width/2, test_phys, width, label='Consistency Rate (%)', color='#2ECC71', alpha=0.85)
ax2.bar(x + width/2, test_incomp, width, label='Incompatible Pair Rate (%)', color='#E74C3C', alpha=0.85)
ax2.set_ylabel('Percentage (%)')
ax2.set_title('Held-Out Test Set Physics Compliance (N=21)', fontsize=11, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(configs)
ax2.set_ylim(0, 105)
ax2.legend(loc='upper right')
ax2.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('results/figures/p2/p2_phase3_physics_consistency.png', dpi=200)
plt.close()
print("Saved p2_phase3_physics_consistency.png")

# -------------------------------------------------------------
# 4. Figure 4: Per-Class F1 Comparison on CV
# -------------------------------------------------------------
def get_cv_f1s(raw_folds):
    cm = np.sum([f['confusion_category'] for f in raw_folds], axis=0)
    prec = np.diag(cm) / np.maximum(cm.sum(axis=0), 1)
    rec = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    return 2 * prec * rec / np.maximum(prec + rec, 1e-9)

f1_e0 = get_cv_f1s(cv_data['E0_baseline']['raw_folds'])
f1_e1 = get_cv_f1s(cv_data['E1_lambda_001']['raw_folds'])
f1_e2 = get_cv_f1s(cv_data['E2_lambda_005']['raw_folds'])

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(short_cat))
width = 0.25

ax.bar(x - width, f1_e0, width, label='E0: Baseline (λ=0)', color='#4A90E2', alpha=0.85)
ax.bar(x, f1_e1, width, label='E1: Weak Phys (λ=0.01)', color='#8E44AD', alpha=0.85)
ax.bar(x + width, f1_e2, width, label='E2: Mod Phys (λ=0.05)', color='#E67E22', alpha=0.85)

ax.set_ylabel('F1 Score')
ax.set_title('P2 Phase 3: Category Per-Class F1 Score (5-Fold CV Aggregate)', fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(short_cat, rotation=15, ha='right')
ax.set_ylim(0, 0.55)
ax.legend(loc='upper right')
ax.grid(axis='y', linestyle='--', alpha=0.5)

for i in range(len(short_cat)):
    ax.text(i - width, f1_e0[i] + 0.01, f"{f1_e0[i]:.2f}", ha='center', fontsize=7.5)
    ax.text(i, f1_e1[i] + 0.01, f"{f1_e1[i]:.2f}", ha='center', fontsize=7.5)
    ax.text(i + width, f1_e2[i] + 0.01, f"{f1_e2[i]:.2f}", ha='center', fontsize=7.5)

plt.tight_layout()
plt.savefig('results/figures/p2/p2_phase3_per_class_f1.png', dpi=200)
plt.close()
print("Saved p2_phase3_per_class_f1.png")
