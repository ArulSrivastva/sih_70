import zipfile
import io
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image

os.makedirs('results/figures/p2', exist_ok=True)

z = zipfile.ZipFile('PS70-main.zip')
all_df = pd.read_csv(io.BytesIO(z.read('PS70-main/data/processed/detection/detection_all.csv')))

# Select representative samples from each pattern and category
# Patterns: eye_visible, curved_band, shear_pattern
samples_to_plot = [
    # Pattern: eye_visible
    ('eye_visible', 'Very Severe Cyclonic Storm', '74.jpg'),
    ('eye_visible', 'Severe Cyclonic Storm', '55.jpeg'),
    ('eye_visible', 'Extremely Severe Cyclonic Storm', '101.jpg'),
    # Pattern: curved_band
    ('curved_band', 'Cyclonic Storm', '48(1).jpg'),
    ('curved_band', 'Cyclonic Storm', '43.jpg'),
    ('curved_band', 'Cyclonic Storm', '37.jpg'),
    # Pattern: shear_pattern
    ('shear_pattern', 'Deep Depression', '28.jpg'),
    ('shear_pattern', 'Depression', '25.jpg'),
    ('shear_pattern', 'Deep Depression', '27.jpg')
]

fig, axes = plt.subplots(3, 3, figsize=(11, 11))

for idx, (pat, cat, fname) in enumerate(samples_to_plot):
    r = idx // 3
    c = idx % 3
    
    img_path = f"PS70-main/data/processed/classification/image_only_kaggle/images/{fname}"
    raw_bytes = z.read(img_path)
    img = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
    
    axes[r, c].imshow(img)
    axes[r, c].set_title(f"Pattern: {pat}\nCat: {cat}\nFile: {fname} (Size: {img.size})", fontsize=9, fontweight='bold')
    axes[r, c].axis('off')

plt.suptitle("P2 Phase 4: Image Representation Audit Across Dvorak Structural Classes", fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('results/figures/p2/p2_phase4_image_audit.png', dpi=200)
plt.close()
print("Saved results/figures/p2/p2_phase4_image_audit.png")
