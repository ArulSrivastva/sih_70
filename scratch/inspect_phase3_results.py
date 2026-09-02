import json

with open('scratch/phase3_cv_all.json') as f:
    cv_data = json.load(f)

with open('scratch/phase3_test_all.json') as f:
    test_data = json.load(f)

print("=== 5-FOLD CROSS-VALIDATION SUMMARY (N=112) ===")
configs = ['E0_baseline', 'E1_lambda_001', 'E2_lambda_005']
metrics_to_show = [
    'pattern_accuracy', 'pattern_f1_macro', 'pattern_f1_weighted',
    'category_accuracy', 'category_f1_macro', 'category_f1_weighted',
    'physical_consistency_rate', 'incompatible_prediction_rate', 'mean_physics_loss'
]

for cfg in configs:
    print(f"\n--- Config: {cfg} ---")
    summary = cv_data[cfg]['summary']
    for m in metrics_to_show:
        val = summary[m]
        print(f"  {m:<30}: {val['mean']:.4f} +/- {val['std']:.4f}")

print("\n=== FINAL TEST SET SUMMARY (N=21) ===")
for cfg in configs:
    print(f"\n--- Config: {cfg} ---")
    res = test_data[cfg]
    for m in metrics_to_show:
        if m in res:
            print(f"  {m:<30}: {res[m]:.4f}")

print("\n=== PER-CLASS CATEGORY ON TEST SET ===")
for cfg in configs:
    print(f"\n--- Config: {cfg} ---")
    per_cat = test_data[cfg]['per_class_category']
    for cat_name, cat_m in per_cat.items():
        print(f"  {cat_name:<32}: Prec={cat_m['precision']:.3f}, Rec={cat_m['recall']:.3f}, F1={cat_m['f1']:.3f} (Supp={cat_m['support']})")

print("\n=== PER-CLASS PATTERN ON TEST SET ===")
for cfg in configs:
    print(f"\n--- Config: {cfg} ---")
    per_pat = test_data[cfg]['per_class_pattern']
    for pat_name, pat_m in per_pat.items():
        print(f"  {pat_name:<16}: Prec={pat_m['precision']:.3f}, Rec={pat_m['recall']:.3f}, F1={pat_m['f1']:.3f} (Supp={pat_m['support']})")
