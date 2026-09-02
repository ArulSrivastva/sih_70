import zipfile
import io
import pandas as pd
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)

z = zipfile.ZipFile('PS70-main.zip')
train_df = pd.read_csv(io.BytesIO(z.read('PS70-main/data/processed/detection/train_detection.csv')))

ct = pd.crosstab(train_df['structural_pattern'], train_df['category'], margins=True)
print("=== CONTINGENCY TABLE (TRAIN N=93) ===")
print(ct)

print("\n=== P(category | pattern) (TRAIN) ===")
print(pd.crosstab(train_df['structural_pattern'], train_df['category'], normalize='index').round(4))

print("\n=== ALL PAIRS IN TRAIN ===")
for (pat, cat), count in train_df.groupby(['structural_pattern', 'category']).size().items():
    print(f"  Pattern: {pat:<15} | Category: {cat:<32} | Count: {count:>2} ({count/len(train_df)*100:5.2f}%)")

print("\n=== COMBINATIONS NOT IN TRAIN (Count = 0) ===")
patterns = sorted(train_df['structural_pattern'].unique())
categories = sorted(train_df['category'].unique())
for p in patterns:
    for c in categories:
        cnt = len(train_df[(train_df['structural_pattern'] == p) & (train_df['category'] == c)])
        if cnt == 0:
            print(f"  Zero-count pair: Pattern: {p:<15} | Category: {c:<32}")
