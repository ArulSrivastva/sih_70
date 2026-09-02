import sys, os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('scratch'))
import phase2b_experiment as exp

print("=== SENSITIVITY ANALYSIS ACROSS BETA VALUES (5-Fold CV) ===")
for beta in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    res = exp.run_phase2b_experiment(alpha=0.1, beta=beta, seed=42)
    p_f1 = res['candidate_cv']['pattern_f1_macro']['mean']
    c_f1 = res['candidate_cv']['category_f1_macro']['mean']
    c_acc = res['candidate_cv']['category_accuracy']['mean']
    p_acc = res['candidate_cv']['pattern_accuracy']['mean']
    phys = res['candidate_cv']['physical_consistency_rate']['mean']
    print(f"Beta = {beta:.1f} | Pat Macro-F1: {p_f1:.4f} | Cat Macro-F1: {c_f1:.4f} | Cat Acc: {c_acc:.4f} | Phys: {phys:.4f}")
