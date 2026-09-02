import json

with open('scratch/phase4_augmentation_results.json') as f:
    aug_cv = json.load(f)

with open('scratch/phase4_finetuning_results.json') as f:
    ft_cv = json.load(f)

with open('scratch/phase4_test_results.json') as f:
    test_res = json.load(f)

print("=== AUGMENTATION 5-FOLD CV ===")
for k in ['A0_baseline', 'A1_spatial_color', 'A2_rotation_spatial']:
    s = aug_cv[k]['summary']
    print(f"\n{k}:")
    print(f"  Pattern Acc: {s['pattern_accuracy']['mean']:.4f} +/- {s['pattern_accuracy']['std']:.4f}")
    print(f"  Pattern Macro-F1: {s['pattern_f1_macro']['mean']:.4f} +/- {s['pattern_f1_macro']['std']:.4f}")
    print(f"  Category Acc: {s['category_accuracy']['mean']:.4f} +/- {s['category_accuracy']['std']:.4f}")
    print(f"  Category Macro-F1: {s['category_f1_macro']['mean']:.4f} +/- {s['category_f1_macro']['std']:.4f}")

print("\n=== FINE-TUNING 5-FOLD CV ===")
for k in ['F0_frozen_backbone', 'F1_partial_finetune', 'F2_full_finetune']:
    s = ft_cv[k]['summary']
    print(f"\n{k}:")
    print(f"  Pattern Acc: {s['pattern_accuracy']['mean']:.4f} +/- {s['pattern_accuracy']['std']:.4f}")
    print(f"  Pattern Macro-F1: {s['pattern_f1_macro']['mean']:.4f} +/- {s['pattern_f1_macro']['std']:.4f}")
    print(f"  Category Acc: {s['category_accuracy']['mean']:.4f} +/- {s['category_accuracy']['std']:.4f}")
    print(f"  Category Macro-F1: {s['category_f1_macro']['mean']:.4f} +/- {s['category_f1_macro']['std']:.4f}")

print("\n=== FINAL TEST SET METRICS (N=21) ===")
for k, v in test_res.items():
    print(f"\n{k}:")
    print(f"  Pattern Acc: {v['pattern_accuracy']:.4f}, Macro-F1: {v['pattern_f1_macro']:.4f}, Weighted-F1: {v['pattern_f1_weighted']:.4f}")
    print(f"  Category Acc: {v['category_accuracy']:.4f}, Macro-F1: {v['category_f1_macro']:.4f}, Weighted-F1: {v['category_f1_weighted']:.4f}")
    print("  Per-class Category F1:")
    for cat_name, cat_m in v['per_class_category'].items():
        print(f"    {cat_name:<30}: F1={cat_m['f1']:.3f} (Supp={cat_m['support']}, Rec={cat_m['recall']:.3f})")
    print("  Per-class Pattern F1:")
    for pat_name, pat_m in v['per_class_pattern'].items():
        print(f"    {pat_name:<16}: F1={pat_m['f1']:.3f} (Supp={pat_m['support']}, Rec={pat_m['recall']:.3f})")
