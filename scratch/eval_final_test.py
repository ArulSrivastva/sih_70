import zipfile
import io
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('scratch'))
from phase2b_experiment import (
    ZipImageDataset, CycloneDetector, compute_fold_compatibility_matrix, 
    evaluate_with_consistency, set_seed
)

def run_test_eval():
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    z = zipfile.ZipFile('PS70-main.zip')
    train_df = pd.read_csv(io.BytesIO(z.read('PS70-main/data/processed/detection/train_detection.csv')))
    val_df = pd.read_csv(io.BytesIO(z.read('PS70-main/data/processed/detection/val_detection.csv')))
    test_df = pd.read_csv(io.BytesIO(z.read('PS70-main/data/processed/detection/test_detection.csv')))
    
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    patterns = sorted(combined_df["structural_pattern"].dropna().unique())
    categories = sorted(combined_df["category"].dropna().unique())
    pattern_to_idx = {p: i for i, p in enumerate(patterns)}
    category_to_idx = {c: i for i, c in enumerate(categories)}
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # 1. Compatibility matrix from 93 training samples only! (NO TEST LEAKAGE)
    C_train, D_train, counts_train = compute_fold_compatibility_matrix(train_df, patterns, categories, alpha=0.1)
    
    test_ds = ZipImageDataset(test_df, z, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    # 2. Evaluate the shipped baseline model weights on test
    ckpt_raw = z.read('PS70-main/models/detection/model_weights.pt')
    ckpt = torch.load(io.BytesIO(ckpt_raw), map_location='cpu', weights_only=True)
    
    model = CycloneDetector(num_patterns=len(patterns), num_categories=len(categories), pretrained=False).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    base_test, cand_test = evaluate_with_consistency(model, test_loader, patterns, categories, C_train, D_train, beta=1.0, device=device)
    
    print("=== FINAL TEST SET EVALUATION (N=21) ===")
    print("\n--- BASELINE TEST METRICS ---")
    for k, v in base_test.items():
        if not k.startswith('confusion'):
            print(f"  {k:<30}: {v:.4f}" if isinstance(v, float) else f"  {k:<30}: {v}")
            
    print("\n--- CANDIDATE TEST METRICS (with Dvorak Consistency) ---")
    for k, v in cand_test.items():
        if not k.startswith('confusion'):
            print(f"  {k:<30}: {v:.4f}" if isinstance(v, float) else f"  {k:<30}: {v}")
            
    # Per-class breakdown
    print("\n=== BASELINE TEST PATTERN CONFUSION MATRIX ===")
    print(pd.DataFrame(base_test['confusion_pattern'], index=patterns, columns=patterns))
    print("\n=== CANDIDATE TEST PATTERN CONFUSION MATRIX ===")
    print(pd.DataFrame(cand_test['confusion_pattern'], index=patterns, columns=patterns))
    
    print("\n=== BASELINE TEST CATEGORY CONFUSION MATRIX ===")
    print(pd.DataFrame(base_test['confusion_category'], index=categories, columns=categories))
    print("\n=== CANDIDATE TEST CATEGORY CONFUSION MATRIX ===")
    print(pd.DataFrame(cand_test['confusion_category'], index=categories, columns=categories))
    
    out = {
        'patterns': patterns,
        'categories': categories,
        'baseline_test': base_test,
        'candidate_test': cand_test,
        'compatibility_matrix_train': C_train.tolist(),
        'counts_train': counts_train.tolist()
    }
    with open('scratch/final_test_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved scratch/final_test_results.json")

if __name__ == '__main__':
    run_test_eval()
