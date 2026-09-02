import zipfile
import io
import pandas as pd
import numpy as np
import json

z = zipfile.ZipFile('PS70-main.zip')
train_df = pd.read_csv(io.BytesIO(z.read('PS70-main/data/processed/detection/train_detection.csv')))
val_df = pd.read_csv(io.BytesIO(z.read('PS70-main/data/processed/detection/val_detection.csv')))
test_df = pd.read_csv(io.BytesIO(z.read('PS70-main/data/processed/detection/test_detection.csv')))

print(f"Train samples: {len(train_df)}")
print(f"Val samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")

# Contingency table on TRAINING DATA ONLY
ct_train = pd.crosstab(train_df['structural_pattern'], train_df['category'], margins=True)
ct_train_pct = pd.crosstab(train_df['structural_pattern'], train_df['category'], normalize='all') * 100

print("\n--- Training Data Contingency Table (Counts) ---")
print(ct_train)

print("\n--- Training Data Contingency Table (Percentages %) ---")
print(ct_train_pct.round(2))

# Check conditional distribution P(category | pattern) on train
cond_cat_given_pat = pd.crosstab(train_df['structural_pattern'], train_df['category'], normalize='index')
print("\n--- P(category | structural_pattern) (Train) ---")
print(cond_cat_given_pat.round(3))

# Check conditional distribution P(pattern | category) on train
cond_pat_given_cat = pd.crosstab(train_df['category'], train_df['structural_pattern'], normalize='index')
print("\n--- P(structural_pattern | category) (Train) ---")
print(cond_pat_given_cat.round(3))
