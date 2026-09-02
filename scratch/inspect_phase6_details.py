import json

with open('scratch/phase6_all_results.json') as f:
    res = json.load(f)

d3 = res['D3_100pct']
print("=== D3 EXPANDED DATASET TEST RESULTS (N=340) ===")
print(f"Pattern Accuracy: {d3['pattern_accuracy']:.4f}")
print(f"Pattern Macro-F1: {d3['pattern_f1_macro']:.4f}")
print(f"Pattern Weighted-F1: {d3['pattern_f1_weighted']:.4f}")
print(f"Category Accuracy: {d3['category_accuracy']:.4f}")
print(f"Category Macro-F1: {d3['category_f1_macro']:.4f}")
print(f"Category Weighted-F1: {d3['category_f1_weighted']:.4f}")

print("\n--- Per-Class Pattern ---")
for k, v in d3['per_class_pattern'].items():
    print(f"  {k:<16}: Prec={v['precision']:.3f}, Rec={v['recall']:.3f}, F1={v['f1']:.3f} (Supp={v['support']})")

print("\n--- Per-Class Category ---")
for k, v in d3['per_class_category'].items():
    print(f"  {k:<32}: Prec={v['precision']:.3f}, Rec={v['recall']:.3f}, F1={v['f1']:.3f} (Supp={v['support']})")

print("\n--- Validation Results (N=280) ---")
print(f"Val Pattern Macro-F1: {d3['val_pattern_f1_macro']:.4f}")
print(f"Val Category Macro-F1: {d3['val_category_f1_macro']:.4f}")
print(f"Val Category Accuracy: {d3['val_category_accuracy']:.4f}")
