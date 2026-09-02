import zipfile
import io
import os
import hashlib
import pandas as pd
import numpy as np
from PIL import Image

z = zipfile.ZipFile('PS70-main.zip')
train_df = pd.read_csv(io.BytesIO(z.read('PS70-main/data/processed/detection/train_detection.csv')))
val_df = pd.read_csv(io.BytesIO(z.read('PS70-main/data/processed/detection/val_detection.csv')))
test_df = pd.read_csv(io.BytesIO(z.read('PS70-main/data/processed/detection/test_detection.csv')))
all_df = pd.read_csv(io.BytesIO(z.read('PS70-main/data/processed/detection/detection_all.csv')))

print(f"Total rows in detection_all: {len(all_df)}")
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# 1. Image dimensions, channels, pixel stats, and duplicate check
image_stats = []
md5_hashes = {}
duplicates = []

for idx, row in all_df.iterrows():
    fname = row['filename']
    img_path = f"PS70-main/data/processed/classification/image_only_kaggle/images/{fname}"
    raw_bytes = z.read(img_path)
    
    # MD5 hash for exact duplicate detection
    h = hashlib.md5(raw_bytes).hexdigest()
    if h in md5_hashes:
        duplicates.append((fname, md5_hashes[h]))
    else:
        md5_hashes[h] = fname
        
    img = Image.open(io.BytesIO(raw_bytes))
    arr = np.array(img)
    
    image_stats.append({
        'filename': fname,
        'format': img.format,
        'mode': img.mode,
        'size': img.size, # (width, height)
        'channels': arr.shape[2] if len(arr.shape) == 3 else 1,
        'min_val': float(arr.min()),
        'max_val': float(arr.max()),
        'mean_val': float(arr.mean()),
        'std_val': float(arr.std()),
        'pattern': row['structural_pattern'],
        'category': row['category'],
        'mock_bbox': row['mock_bbox'],
        'cyclone_detected': row['cyclone_detected']
    })

stats_df = pd.DataFrame(image_stats)
print("\n--- Image Format & Mode Distribution ---")
print(stats_df['mode'].value_counts())
print(stats_df['size'].value_counts())

print("\n--- Pixel Dynamic Range Stats ---")
print(f"Min pixel val across all images: {stats_df['min_val'].min()}")
print(f"Max pixel val across all images: {stats_df['max_val'].max()}")
print(f"Mean pixel brightness: {stats_df['mean_val'].mean():.2f} +/- {stats_df['mean_val'].std():.2f}")

print("\n--- Duplicate Images Check ---")
print(f"Exact byte duplicates found: {len(duplicates)}")
for d in duplicates:
    print(f"  Duplicate pair: {d[0]} == {d[1]}")

print("\n--- Bounding Box Check ---")
print(f"Unique bounding box strings: {stats_df['mock_bbox'].unique().tolist()}")
print(f"Constant mock box verified: {len(stats_df['mock_bbox'].unique()) == 1 and stats_df['mock_bbox'].iloc[0] == '[420, 190, 600, 370]'}")

print("\n--- Cyclone Detected Check ---")
print(f"Unique cyclone_detected values: {stats_df['cyclone_detected'].unique().tolist()}")

# Filename pattern analysis (e.g. 33.jpg, 33(1).jpg, 33(2).jpg)
base_names = {}
for fn in stats_df['filename']:
    # extract base number before parens
    base = fn.split('(')[0].split('.')[0]
    base_names[base] = base_names.get(base, 0) + 1

multi_frame_storms = {k: v for k, v in base_names.items() if v > 1}
print(f"\n--- Storm/Sequence Identifiers from Filename ---")
print(f"Total unique base names: {len(base_names)}")
print(f"Base names with multiple frames (potential temporal sequence): {len(multi_frame_storms)}")
print("Examples of multi-frame sets:", list(multi_frame_storms.items())[:10])
