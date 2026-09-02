import os
import json
import hashlib
import glob
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('results', exist_ok=True)
os.makedirs('results/figures/p2/phase6', exist_ok=True)

# -------------------------------------------------------------
# 1. Dataset Count & File System Verification
# -------------------------------------------------------------
manifest_path = 'data/p2_mosdac_expanded/mosdac_expanded_all.csv'
df = pd.read_csv(manifest_path)
print(f"Loaded manifest: {len(df)} rows")

crop_files = glob.glob('data/p2_mosdac_expanded/crops/*.png')
print(f"Found {len(crop_files)} crop files on disk")

# Check unique cyclones
unique_cyclones = df['cyclone_unique_id'].unique()
print(f"Unique cyclones: {len(unique_cyclones)}")
print("Cyclones:", unique_cyclones)

# Verify counts per split
train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'val']
test_df = df[df['split'] == 'test']

print(f"Train rows: {len(train_df)} across {train_df['cyclone_unique_id'].nunique()} cyclones")
print(f"Val rows: {len(val_df)} across {val_df['cyclone_unique_id'].nunique()} cyclones")
print(f"Test rows: {len(test_df)} across {test_df['cyclone_unique_id'].nunique()} cyclones")

# -------------------------------------------------------------
# 2. Cyclone-Disjoint Split Verification
# -------------------------------------------------------------
train_cyclones = set(train_df['cyclone_unique_id'].unique())
val_cyclones = set(val_df['cyclone_unique_id'].unique())
test_cyclones = set(test_df['cyclone_unique_id'].unique())

train_val_overlap = train_cyclones.intersection(val_cyclones)
train_test_overlap = train_cyclones.intersection(test_cyclones)
val_test_overlap = val_cyclones.intersection(test_cyclones)

print("\n--- Split Disjointness ---")
print(f"Train intersect Val overlap: {len(train_val_overlap)} ({train_val_overlap})")
print(f"Train intersect Test overlap: {len(train_test_overlap)} ({train_test_overlap})")
print(f"Val intersect Test overlap: {len(val_test_overlap)} ({val_test_overlap})")

# -------------------------------------------------------------
# 3. Duplicate & Perceptual Hash Audit
# -------------------------------------------------------------
print("\n--- Image Hash & Duplication Audit ---")
hashes = {}
duplicate_pairs = []
per_split_hashes = {'train': set(), 'val': set(), 'test': set()}
cross_split_duplicates = []

for idx, row in df.iterrows():
    c_path = row['crop_path']
    split = row['split']
    with open(c_path, 'rb') as f:
        h = hashlib.sha256(f.read()).hexdigest()
        
    if h in hashes:
        duplicate_pairs.append((row['filename'], hashes[h]['filename']))
        if hashes[h]['split'] != split:
            cross_split_duplicates.append((row['filename'], hashes[h]['filename'], split, hashes[h]['split']))
    else:
        hashes[h] = {'filename': row['filename'], 'split': split}
        
    per_split_hashes[split].add(h)

print(f"Total unique SHA-256 image hashes: {len(hashes)} / {len(df)}")
print(f"Exact intra-dataset duplicates: {len(duplicate_pairs)}")
print(f"Cross-split duplicate image hashes: {len(cross_split_duplicates)}")

# -------------------------------------------------------------
# 4. Geolocation & Crop Quality Check across all 24 unique cyclone systems
# -------------------------------------------------------------
fig, axes = plt.subplots(5, 5, figsize=(16, 16))

for idx, c_id in enumerate(sorted(unique_cyclones)):
    r = idx // 5
    c = idx % 5
    sample_row = df[df['cyclone_unique_id'] == c_id].iloc[len(df[df['cyclone_unique_id'] == c_id]) // 2]
    img = Image.open(sample_row['crop_path'])
    
    axes[r, c].imshow(img)
    axes[r, c].set_title(f"{c_id}\n{sample_row['category']}\n{sample_row['structural_pattern']}\nLat:{sample_row['latitude']} Lon:{sample_row['longitude']}", fontsize=7, fontweight='bold')
    axes[r, c].axis('off')

# Blank out the 25th subplot
axes[4, 4].axis('off')

plt.suptitle("Independent Verification: Representative 224x224 Crops Across Historical Cyclones", fontsize=12, fontweight='bold', y=0.99)
plt.tight_layout()
plt.savefig('results/figures/p2/phase6/p2_phase6_verification_25cyclones.png', dpi=200)
plt.close()
print("Saved results/figures/p2/phase6/p2_phase6_verification_25cyclones.png")

# -------------------------------------------------------------
# 5. Model Reproduction & Per-Cyclone Breakdown
# -------------------------------------------------------------
class CycloneDetector(nn.Module):
    def __init__(self, num_patterns=3, num_categories=7, pretrained=False):
        super().__init__()
        self.backbone = mobilenet_v3_small(weights=None)
        feature_size = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()
        self.presence_head = nn.Sequential(
            nn.Linear(feature_size, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1)
        )
        self.pattern_head = nn.Sequential(
            nn.Linear(feature_size, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_patterns)
        )
        self.category_head = nn.Sequential(
            nn.Linear(feature_size, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_categories)
        )

    def forward(self, x):
        features = self.backbone(x)
        return {
            "presence": self.presence_head(features),
            "pattern": self.pattern_head(features),
            "category": self.category_head(features),
        }

patterns = sorted(df['structural_pattern'].unique())
categories = sorted(df['category'].unique())
pat_to_idx = {p: i for i, p in enumerate(patterns)}
cat_to_idx = {c: i for i, c in enumerate(categories)}

class EvalDataset(Dataset):
    def __init__(self, dframe):
        self.df = dframe.reset_index(drop=True)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['crop_path']).convert('RGB')
        return {
            'image': self.transform(img),
            'structural_pattern': row['structural_pattern'],
            'category': row['category'],
            'cyclone_unique_id': row['cyclone_unique_id'],
            'wind_speed_kmh': row['wind_speed_kmh'],
            'filename': row['filename']
        }

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Candidate Weights
candidate_weights_path = 'models/detection/model_weights_phase6_candidate.pt'
ckpt = torch.load(candidate_weights_path, map_location=device, weights_only=False)
cand_model = CycloneDetector(num_patterns=len(patterns), num_categories=len(categories), pretrained=False).to(device)
cand_model.load_state_dict(ckpt['model_state_dict'])
cand_model.eval()
print(f"\nLoaded candidate checkpoint from {candidate_weights_path}")

test_dataset = EvalDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

p_true, p_pred = [], []
c_true, c_pred = [], []
per_storm_records = {c: {'p_true': [], 'p_pred': [], 'c_true': [], 'c_pred': []} for c in test_cyclones}

with torch.no_grad():
    for batch in test_loader:
        images = batch['image'].to(device)
        out = cand_model(images)
        
        p_p = torch.argmax(out['pattern'], dim=1).cpu().numpy()
        c_p = torch.argmax(out['category'], dim=1).cpu().numpy()
        
        p_t = [pat_to_idx[p] for p in batch['structural_pattern']]
        c_t = [cat_to_idx[c] for c in batch['category']]
        
        p_true.extend(p_t)
        p_pred.extend(p_p)
        c_true.extend(c_t)
        c_pred.extend(c_p)
        
        for i, c_name in enumerate(batch['cyclone_unique_id']):
            per_storm_records[c_name]['p_true'].append(p_t[i])
            per_storm_records[c_name]['p_pred'].append(p_p[i])
            per_storm_records[c_name]['c_true'].append(c_t[i])
            per_storm_records[c_name]['c_pred'].append(c_p[i])

rep_pat_acc = float(accuracy_score(p_true, p_pred))
rep_pat_f1 = float(f1_score(p_true, p_pred, average='macro', zero_division=0))
rep_cat_acc = float(accuracy_score(c_true, c_pred))
rep_cat_f1 = float(f1_score(c_true, c_pred, average='macro', zero_division=0))

print("\n=== CANDIDATE REPRODUCTION ON TEST SET (N=340) ===")
print(f"Pattern Accuracy: {rep_pat_acc*100:.2f}% (Claimed: 98.53%)")
print(f"Pattern Macro-F1: {rep_pat_f1:.4f} (Claimed: 0.9863)")
print(f"Category Accuracy: {rep_cat_acc*100:.2f}% (Claimed: 55.88%)")
print(f"Category Macro-F1: {rep_cat_f1:.4f} (Claimed: 0.4138)")

# Per-Storm Breakdown
per_storm_metrics = {}
print("\n--- Per-Cyclone Test Breakdown ---")
for c_name, data in per_storm_records.items():
    p_acc = accuracy_score(data['p_true'], data['p_pred'])
    p_f1 = f1_score(data['p_true'], data['p_pred'], average='macro', zero_division=0)
    c_acc = accuracy_score(data['c_true'], data['c_pred'])
    c_f1 = f1_score(data['c_true'], data['c_pred'], average='macro', zero_division=0)
    per_storm_metrics[c_name] = {
        'samples': len(data['p_true']),
        'pattern_accuracy': float(p_acc),
        'pattern_macro_f1': float(p_f1),
        'category_accuracy': float(c_acc),
        'category_macro_f1': float(c_f1)
    }
    print(f"  {c_name:<16}: N={len(data['p_true']):<3} | Pat Acc={p_acc*100:5.1f}% Pat F1={p_f1:.3f} | Cat Acc={c_acc*100:5.1f}% Cat F1={c_f1:.3f}")

# -------------------------------------------------------------
# 6. Sanity & Shortcut Investigation for 98.53%
# -------------------------------------------------------------
print("\n--- Sanity / Label Shortcut Audit ---")
img_means, img_stds = [], []
for idx, row in test_df.iterrows():
    im = Image.open(row['crop_path']).convert('L')
    arr = np.array(im, dtype=np.float32)
    img_means.append(arr.mean())
    img_stds.append(arr.std())

test_df_copy = test_df.copy()
test_df_copy['img_mean'] = img_means
test_df_copy['img_std'] = img_stds

mean_by_pattern = test_df_copy.groupby('structural_pattern')['img_mean'].mean().to_dict()
print("Average pixel brightness by structural pattern:")
for p, m in mean_by_pattern.items():
    print(f"  {p:<16}: mean pixel={m:.2f}")

audit_summary = {
    'total_manifest_rows': len(df),
    'total_crop_files': len(crop_files),
    'unique_cyclones_count': len(unique_cyclones),
    'split_counts': {'train': len(train_df), 'val': len(val_df), 'test': len(test_df)},
    'split_cyclone_counts': {'train': len(train_cyclones), 'val': len(val_cyclones), 'test': len(test_cyclones)},
    'train_val_overlap': len(train_val_overlap),
    'train_test_overlap': len(train_test_overlap),
    'val_test_overlap': len(val_test_overlap),
    'unique_image_hashes': len(hashes),
    'exact_duplicates': len(duplicate_pairs),
    'cross_split_duplicates': len(cross_split_duplicates),
    'reproduction_results': {
        'pattern_accuracy': rep_pat_acc,
        'pattern_macro_f1': rep_pat_f1,
        'category_accuracy': rep_cat_acc,
        'category_macro_f1': rep_cat_f1
    },
    'per_storm_metrics': per_storm_metrics
}

with open('scratch/independent_audit_summary.json', 'w') as f:
    json.dump(audit_summary, f, indent=2)
print("Saved scratch/independent_audit_summary.json")
