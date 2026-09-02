import zipfile
import io
import json
import random
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ZipImageDataset(Dataset):
    def __init__(self, df, z, transform=None):
        self.df = df.reset_index(drop=True)
        self.z = z
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row['filename']
        img_name = f'PS70-main/data/processed/classification/image_only_kaggle/images/{fname}'
        img_bytes = self.z.read(img_name)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        if self.transform:
            img = self.transform(img)
            
        return {
            'image': img,
            'filename': fname,
            'detected': bool(row['cyclone_detected']),
            'structural_pattern': str(row['structural_pattern']),
            'category': str(row['category']),
            'wind_speed_kmh': float(row['wind_speed_kmh'])
        }

class CycloneDetector(nn.Module):
    def __init__(self, num_patterns=3, num_categories=7, pretrained=True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)
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

def is_physically_consistent(pattern_name, category_name):
    severe_cats = {'Severe Cyclonic Storm', 'Very Severe Cyclonic Storm', 'Extremely Severe Cyclonic Storm', 'Super Cyclonic Storm'}
    cyclonic_cats = {'Deep Depression', 'Cyclonic Storm'}
    weak_cats = {'Depression', 'Deep Depression'}
    
    if pattern_name == 'eye_visible':
        return category_name in severe_cats
    elif pattern_name == 'curved_band':
        return category_name in cyclonic_cats or category_name == 'Severe Cyclonic Storm'
    elif pattern_name == 'shear_pattern':
        return category_name in weak_cats or category_name == 'Cyclonic Storm'
    return False

def compute_physics_loss(pattern_logits, category_logits, patterns, categories, incompatible_pairs, device='cpu'):
    """
    Differentiable Physics Loss Term:
    L_phys = mean_b [ sum_{(i, j) in Incompatible} P(pattern=i | x_b) * P(category=j | x_b) ]
    """
    p_probs = torch.softmax(pattern_logits, dim=1) # (B, num_patterns)
    c_probs = torch.softmax(category_logits, dim=1) # (B, num_categories)
    
    pat_to_i = {p: i for i, p in enumerate(patterns)}
    cat_to_j = {c: j for j, c in enumerate(categories)}
    
    loss_phys = torch.tensor(0.0, device=device)
    for (p_name, c_name) in incompatible_pairs:
        if p_name in pat_to_i and c_name in cat_to_j:
            i = pat_to_i[p_name]
            j = cat_to_j[c_name]
            joint_prob = p_probs[:, i] * c_probs[:, j]
            loss_phys = loss_phys + joint_prob.mean()
            
    return loss_phys

def evaluate_model(model, dataloader, patterns, categories, incompatible_pairs, device='cpu'):
    model.eval()
    pat_to_idx = {p: i for i, p in enumerate(patterns)}
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    
    p_true, p_pred = [], []
    c_true, c_pred = [], []
    phys_losses = []
    consistent_cnt = 0
    incompatible_cnt = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            outputs = model(images)
            
            p_logits = outputs['pattern']
            c_logits = outputs['category']
            
            l_phys = compute_physics_loss(p_logits, c_logits, patterns, categories, incompatible_pairs, device=device)
            phys_losses.append(l_phys.item())
            
            pat_p = torch.argmax(p_logits, dim=1).cpu().numpy()
            cat_p = torch.argmax(c_logits, dim=1).cpu().numpy()
            
            p_t = [pat_to_idx[p] for p in batch['structural_pattern']]
            c_t = [cat_to_idx[c] for c in batch['category']]
            
            p_true.extend(p_t)
            p_pred.extend(pat_p)
            c_true.extend(c_t)
            c_pred.extend(cat_p)
            
            for pi, ci in zip(pat_p, cat_p):
                p_name = patterns[pi]
                c_name = categories[ci]
                if is_physically_consistent(p_name, c_name):
                    consistent_cnt += 1
                if (p_name, c_name) in incompatible_pairs:
                    incompatible_cnt += 1
                total += 1
                
    cm_p = confusion_matrix(p_true, p_pred, labels=list(range(len(patterns))))
    cm_c = confusion_matrix(c_true, c_pred, labels=list(range(len(categories))))
    
    # Per class metrics
    def get_per_class(cm, names):
        prec = np.diag(cm) / np.maximum(cm.sum(axis=0), 1)
        rec = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        supp = cm.sum(axis=1)
        return {
            names[i]: {
                'support': int(supp[i]),
                'precision': float(prec[i]),
                'recall': float(rec[i]),
                'f1': float(f1[i])
            } for i in range(len(names))
        }
        
    return {
        'pattern_accuracy': float(accuracy_score(p_true, p_pred)),
        'pattern_precision_weighted': float(precision_score(p_true, p_pred, average='weighted', zero_division=0)),
        'pattern_recall_weighted': float(recall_score(p_true, p_pred, average='weighted', zero_division=0)),
        'pattern_f1_weighted': float(f1_score(p_true, p_pred, average='weighted', zero_division=0)),
        'pattern_f1_macro': float(f1_score(p_true, p_pred, average='macro', zero_division=0)),
        'category_accuracy': float(accuracy_score(c_true, c_pred)),
        'category_precision_weighted': float(precision_score(c_true, c_pred, average='weighted', zero_division=0)),
        'category_recall_weighted': float(recall_score(c_true, c_pred, average='weighted', zero_division=0)),
        'category_f1_weighted': float(f1_score(c_true, c_pred, average='weighted', zero_division=0)),
        'category_f1_macro': float(f1_score(c_true, c_pred, average='macro', zero_division=0)),
        'physical_consistency_rate': float(consistent_cnt / total if total > 0 else 0.0),
        'incompatible_prediction_rate': float(incompatible_cnt / total if total > 0 else 0.0),
        'incompatible_prediction_count': int(incompatible_cnt),
        'mean_physics_loss': float(np.mean(phys_losses)),
        'per_class_pattern': get_per_class(cm_p, patterns),
        'per_class_category': get_per_class(cm_c, categories),
        'confusion_pattern': cm_p.tolist(),
        'confusion_category': cm_c.tolist(),
    }

def run_phase3_cv(lambda_phys=0.0, seed=42, n_splits=5, epochs=10, lr=1e-4):
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    z = zipfile.ZipFile('PS70-main.zip')
    train_df = pd.read_csv(io.BytesIO(z.read('PS70-main/data/processed/detection/train_detection.csv')))
    val_df = pd.read_csv(io.BytesIO(z.read('PS70-main/data/processed/detection/val_detection.csv')))
    
    dev_df = pd.concat([train_df, val_df], ignore_index=True)
    
    patterns = sorted(dev_df["structural_pattern"].dropna().unique())
    categories = sorted(dev_df["category"].dropna().unique())
    
    pattern_to_idx = {p: i for i, p in enumerate(patterns)}
    category_to_idx = {c: i for i, c in enumerate(categories)}
    
    incompatible_pairs = [
        ('eye_visible', 'Cyclonic Storm'),
        ('eye_visible', 'Deep Depression'),
        ('eye_visible', 'Depression'),
        ('shear_pattern', 'Extremely Severe Cyclonic Storm')
    ]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    fold_results = []
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(dev_df, dev_df['structural_pattern'])):
        tr_sub = dev_df.iloc[tr_idx].reset_index(drop=True)
        va_sub = dev_df.iloc[va_idx].reset_index(drop=True)
        
        tr_ds = ZipImageDataset(tr_sub, z, transform=transform)
        va_ds = ZipImageDataset(va_sub, z, transform=transform)
        
        tr_loader = DataLoader(tr_ds, batch_size=16, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=16, shuffle=False)
        
        model = CycloneDetector(num_patterns=len(patterns), num_categories=len(categories), pretrained=True).to(device)
        
        presence_loss = nn.BCEWithLogitsLoss()
        pattern_loss = nn.CrossEntropyLoss()
        category_loss = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=lr)
        
        best_val_loss = float('inf')
        best_state = None
        
        for ep in range(epochs):
            model.train()
            for b in tr_loader:
                imgs = b['image'].to(device)
                det = b['detected'].float().unsqueeze(1).to(device)
                pat = torch.tensor([pattern_to_idx[p] for p in b['structural_pattern']], dtype=torch.long).to(device)
                cat = torch.tensor([category_to_idx[c] for c in b['category']], dtype=torch.long).to(device)
                
                optimizer.zero_grad()
                out = model(imgs)
                
                l_pres = presence_loss(out['presence'], det)
                l_pat = pattern_loss(out['pattern'], pat)
                l_cat = category_loss(out['category'], cat)
                
                l_phys = compute_physics_loss(out['pattern'], out['category'], patterns, categories, incompatible_pairs, device=device)
                
                total_loss = l_pres + l_pat + l_cat + lambda_phys * l_phys
                total_loss.backward()
                optimizer.step()
                
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for b in va_loader:
                    imgs = b['image'].to(device)
                    det = b['detected'].float().unsqueeze(1).to(device)
                    pat = torch.tensor([pattern_to_idx[p] for p in b['structural_pattern']], dtype=torch.long).to(device)
                    cat = torch.tensor([category_to_idx[c] for c in b['category']], dtype=torch.long).to(device)
                    out = model(imgs)
                    
                    l_pres = presence_loss(out['presence'], det)
                    l_pat = pattern_loss(out['pattern'], pat)
                    l_cat = category_loss(out['category'], cat)
                    l_phys = compute_physics_loss(out['pattern'], out['category'], patterns, categories, incompatible_pairs, device=device)
                    
                    val_loss += (l_pres + l_pat + l_cat + lambda_phys * l_phys).item()
                    
            val_loss /= len(va_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
        model.load_state_dict(best_state)
        res = evaluate_model(model, va_loader, patterns, categories, incompatible_pairs, device=device)
        fold_results.append(res)
        
    def summarize(res_list):
        scalar_keys = [k for k in res_list[0].keys() if not (k.startswith('confusion') or k.startswith('per_class'))]
        summary = {}
        for k in scalar_keys:
            vals = [r[k] for r in res_list]
            summary[k] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'values': vals
            }
        return summary
        
    return {
        'lambda_phys': lambda_phys,
        'summary': summarize(fold_results),
        'raw_folds': fold_results,
        'patterns': patterns,
        'categories': categories,
        'incompatible_pairs': incompatible_pairs
    }

def train_and_eval_test(lambda_phys=0.0, seed=42, epochs=10, lr=1e-4):
    set_seed(seed)
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
    
    incompatible_pairs = [
        ('eye_visible', 'Cyclonic Storm'),
        ('eye_visible', 'Deep Depression'),
        ('eye_visible', 'Depression'),
        ('shear_pattern', 'Extremely Severe Cyclonic Storm')
    ]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    train_ds = ZipImageDataset(train_df, z, transform=transform)
    val_ds = ZipImageDataset(val_df, z, transform=transform)
    test_ds = ZipImageDataset(test_df, z, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    if lambda_phys == 0.0:
        # Load locked baseline checkpoint for E0 to ensure exact parity
        ckpt_raw = z.read('PS70-main/models/detection/model_weights.pt')
        ckpt = torch.load(io.BytesIO(ckpt_raw), map_location='cpu', weights_only=True)
        model = CycloneDetector(num_patterns=len(patterns), num_categories=len(categories), pretrained=False).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model = CycloneDetector(num_patterns=len(patterns), num_categories=len(categories), pretrained=True).to(device)
        presence_loss = nn.BCEWithLogitsLoss()
        pattern_loss = nn.CrossEntropyLoss()
        category_loss = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=lr)
        
        best_val_loss = float('inf')
        best_state = None
        
        for ep in range(epochs):
            model.train()
            for b in train_loader:
                imgs = b['image'].to(device)
                det = b['detected'].float().unsqueeze(1).to(device)
                pat = torch.tensor([pattern_to_idx[p] for p in b['structural_pattern']], dtype=torch.long).to(device)
                cat = torch.tensor([category_to_idx[c] for c in b['category']], dtype=torch.long).to(device)
                
                optimizer.zero_grad()
                out = model(imgs)
                
                l_pres = presence_loss(out['presence'], det)
                l_pat = pattern_loss(out['pattern'], pat)
                l_cat = category_loss(out['category'], cat)
                l_phys = compute_physics_loss(out['pattern'], out['category'], patterns, categories, incompatible_pairs, device=device)
                
                total_loss = l_pres + l_pat + l_cat + lambda_phys * l_phys
                total_loss.backward()
                optimizer.step()
                
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for b in val_loader:
                    imgs = b['image'].to(device)
                    det = b['detected'].float().unsqueeze(1).to(device)
                    pat = torch.tensor([pattern_to_idx[p] for p in b['structural_pattern']], dtype=torch.long).to(device)
                    cat = torch.tensor([category_to_idx[c] for c in b['category']], dtype=torch.long).to(device)
                    out = model(imgs)
                    
                    l_pres = presence_loss(out['presence'], det)
                    l_pat = pattern_loss(out['pattern'], pat)
                    l_cat = category_loss(out['category'], cat)
                    l_phys = compute_physics_loss(out['pattern'], out['category'], patterns, categories, incompatible_pairs, device=device)
                    
                    val_loss += (l_pres + l_pat + l_cat + lambda_phys * l_phys).item()
                    
            val_loss /= len(val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
        model.load_state_dict(best_state)
        
    test_res = evaluate_model(model, test_loader, patterns, categories, incompatible_pairs, device=device)
    return test_res

if __name__ == '__main__':
    print("=== RUNNING P2 PHASE-3 5-FOLD CV EXPERIMENTS ===")
    
    print("\n--- E0: Baseline (lambda=0.0) ---")
    e0_cv = run_phase3_cv(lambda_phys=0.0, seed=42)
    for k, v in e0_cv['summary'].items():
        print(f"  {k:<30}: {v['mean']:.4f} +/- {v['std']:.4f}")
        
    print("\n--- E1: Weak Physics Regularization (lambda=0.01) ---")
    e1_cv = run_phase3_cv(lambda_phys=0.01, seed=42)
    for k, v in e1_cv['summary'].items():
        print(f"  {k:<30}: {v['mean']:.4f} +/- {v['std']:.4f}")
        
    print("\n--- E2: Slightly Stronger Regularization (lambda=0.05) ---")
    e2_cv = run_phase3_cv(lambda_phys=0.05, seed=42)
    for k, v in e2_cv['summary'].items():
        print(f"  {k:<30}: {v['mean']:.4f} +/- {v['std']:.4f}")
        
    all_cv = {
        'E0_baseline': e0_cv,
        'E1_lambda_001': e1_cv,
        'E2_lambda_005': e2_cv
    }
    with open('scratch/phase3_cv_all.json', 'w') as f:
        json.dump(all_cv, f, indent=2)
    print("\nSaved scratch/phase3_cv_all.json")
    
    print("\n=== RUNNING FINAL TEST SET EVALUATIONS ===")
    test_e0 = train_and_eval_test(lambda_phys=0.0, seed=42)
    test_e1 = train_and_eval_test(lambda_phys=0.01, seed=42)
    test_e2 = train_and_eval_test(lambda_phys=0.05, seed=42)
    
    all_test = {
        'E0_baseline': test_e0,
        'E1_lambda_001': test_e1,
        'E2_lambda_005': test_e2
    }
    with open('scratch/phase3_test_all.json', 'w') as f:
        json.dump(all_test, f, indent=2)
    print("Saved scratch/phase3_test_all.json")
    
    print("\n--- Test Set Summary ---")
    print(f"E0 (Baseline)    | Pat Macro-F1: {test_e0['pattern_macro_f1']:.4f} | Cat Macro-F1: {test_e0['category_macro_f1']:.4f} | Phys: {test_e0['physical_consistency_rate']:.4f}")
    print(f"E1 (Lambda=0.01) | Pat Macro-F1: {test_e1['pattern_macro_f1']:.4f} | Cat Macro-F1: {test_e1['category_macro_f1']:.4f} | Phys: {test_e1['physical_consistency_rate']:.4f}")
    print(f"E2 (Lambda=0.05) | Pat Macro-F1: {test_e2['pattern_macro_f1']:.4f} | Cat Macro-F1: {test_e2['category_macro_f1']:.4f} | Phys: {test_e2['physical_consistency_rate']:.4f}")
