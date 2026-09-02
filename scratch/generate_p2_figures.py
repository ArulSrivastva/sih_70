import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('results/figures/p2', exist_ok=True)

# 1. Confusion Matrix Plot for P2 Baseline
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Pattern Confusion Matrix (Test Set, N=21)
# Classes: curved_band (0), eye_visible (1), shear_pattern (2)
cm_pattern = np.array([
    [1, 5, 0],
    [0, 14, 0],
    [0, 1, 0]
])
im1 = ax1.imshow(cm_pattern, cmap='Blues', interpolation='nearest')
ax1.set_title('P2 Baseline: Pattern Confusion Matrix (Test N=21)', fontsize=11, fontweight='bold')
ax1.set_xticks([0, 1, 2])
ax1.set_yticks([0, 1, 2])
ax1.set_xticklabels(['Curved Band', 'Eye Visible', 'Shear'], rotation=20)
ax1.set_yticklabels(['Curved Band', 'Eye Visible', 'Shear'])
ax1.set_xlabel('Predicted Class')
ax1.set_ylabel('True Class')
for i in range(3):
    for j in range(3):
        color = 'white' if cm_pattern[i, j] > 7 else 'black'
        ax1.text(j, i, str(cm_pattern[i, j]), ha='center', va='center', color=color, fontweight='bold')

# 5-Fold CV Comparison (Baseline vs Candidate)
metrics = ['Pattern Acc', 'Pattern F1 (Macro)', 'Category Acc', 'Category F1 (Macro)']
baseline_means = [0.5897, 0.2767, 0.2573, 0.1422]
baseline_stds = [0.0291, 0.0728, 0.1047, 0.0701]

cand_means = [0.5715, 0.2424, 0.2937, 0.1524]
cand_stds = [0.0194, 0.0053, 0.0528, 0.0393]

x = np.arange(len(metrics))
width = 0.35

rects1 = ax2.bar(x - width/2, baseline_means, width, yerr=baseline_stds, capsize=4, label='Baseline (Unnormalized)', color='#4A90E2', alpha=0.85)
rects2 = ax2.bar(x + width/2, cand_means, width, yerr=cand_stds, capsize=4, label='Candidate (ImageNet Norm)', color='#E94A4A', alpha=0.85)

ax2.set_ylabel('Score')
ax2.set_title('5-Fold Cross-Validation: Baseline vs Candidate', fontsize=11, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(metrics, rotation=15)
ax2.set_ylim(0, 0.8)
ax2.legend(loc='upper right')
ax2.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
fig_path = 'results/figures/p2/p2_kfold_ablation_comparison.png'
plt.savefig(fig_path, dpi=200)
plt.close()
print(f'Saved figure to {fig_path}')
