import zipfile
import io
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
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

def evaluate(model, loader, pattern_to_idx, category_to_idx, device='cpu'):
    model.eval()
    idx_to_pattern = {v: k for k, v in pattern_to_idx.items()}
    idx_to_category = {v: k for k, v in category_to_idx.items()}
    
    p_true, p_pred = [], []
    c_true, c_pred = [], []
    consistent_count = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            outputs = model(images)
            pat = torch.argmax(outputs['pattern'], dim=1).cpu().numpy()
            cat = torch.argmax(outputs['category'], dim=1).cpu().numpy()
            
            p_t = [pattern_to_idx[p] for p in batch['structural_pattern']]
            c_t = [category_to_idx[c] for c in batch['category']]
            
            p_true.extend(p_t)
            p_pred.extend(pat)
            c_true.extend(c_t)
            c_pred.extend(cat)
            
            for p_idx, c_idx in zip(pat, cat):
                if is_physically_consistent(idx_to_pattern[p_idx], idx_to_category[c_idx]):
                    consistent_count += 1
                total += 1
                
    return {
        'pattern_acc': float(accuracy_score(p_true, p_pred)),
        'pattern_f1_weighted': float(f1_score(p_true, p_pred, average='weighted', zero_division=0)),
        'pattern_f1_macro': float(f1_score(p_true, p_pred, average='macro', zero_division=0)),
        'category_acc': float(accuracy_score(c_true, c_pred)),
        'category_f1_weighted': float(f1_score(c_true, c_pred, average='weighted', zero_division=0)),
        'category_f1_macro': float(f1_score(c_true, c_pred, average='macro', zero_division=0)),
        'phys_consistency': float(consistent_count / total if total > 0 else 0.0),
    }

def run_kfold(transform, seed=42, n_splits=5, epochs=10, lr=1e-4):
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    z = zipfile.ZipFile('PS70-main.zip')
    train_df = pd.read_csv(io.BytesIO(z.read('PS70-main/data/processed/detection/train_detection.csv')))
    val_df = pd.read_csv(io.BytesIO(z.read('PS70-main/data/processed/detection/val_detection.csv')))
    test_df = pd.read_csv(io.BytesIO(z.read('PS70-main/data/processed/detection/test_detection.csv')))
    
    # Stratified 5-fold CV on train+val (112 development samples)
    dev_df = pd.concat([train_df, val_df], ignore_index=True)
    
    patterns = sorted(dev_df["structural_pattern"].dropna().unique())
    categories = sorted(dev_df["category"].dropna().unique())
    pattern_to_idx = {p: i for i, p in enumerate(patterns)}
    category_to_idx = {c: i for i, c in enumerate(categories)}
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(dev_df, dev_df['structural_pattern'])):
        tr_sub = dev_df.iloc[train_idx]
        va_sub = dev_df.iloc[val_idx]
        
        tr_ds = ZipImageDataset(tr_sub, z, transform=transform)
        va_ds = ZipImageDataset(va_sub, z, transform=transform)
        
        tr_loader = DataLoader(tr_ds, batch_size=16, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=16, shuffle=False)
        
        model = CycloneDetector(num_patterns=len(pattern_to_idx), num_categories=len(category_to_idx), pretrained=True).to(device)
        
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
                l = presence_loss(out['presence'], det) + pattern_loss(out['pattern'], pat) + category_loss(out['category'], cat)
                l.backward()
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
                    l = presence_loss(out['presence'], det) + pattern_loss(out['pattern'], pat) + category_loss(out['category'], cat)
                    val_loss += l.item()
            val_loss /= len(va_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
        model.load_state_dict(best_state)
        res = evaluate(model, va_loader, pattern_to_idx, category_to_idx, device=device)
        fold_metrics.append(res)
        
    df_res = pd.DataFrame(fold_metrics)
    summary = {
        'mean': df_res.mean().to_dict(),
        'std': df_res.std().to_dict(),
        'min': df_res.min().to_dict(),
        'max': df_res.max().to_dict(),
        'folds': fold_metrics
    }
    return summary

if __name__ == '__main__':
    t_base = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    t_cand = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print('=== 5-Fold CV: Baseline (Unnormalized) ===')
    base_kfold = run_kfold(t_base, seed=42)
    print(json.dumps(base_kfold['mean'], indent=2))
    print('Std:\n', json.dumps(base_kfold['std'], indent=2))
    print()
    
    print('=== 5-Fold CV: Candidate (ImageNet Normalized) ===')
    cand_kfold = run_kfold(t_cand, seed=42)
    print(json.dumps(cand_kfold['mean'], indent=2))
    print('Std:\n', json.dumps(cand_kfold['std'], indent=2))
    print()
    
    # Save results to json
    results = {
        'baseline_kfold': base_kfold,
        'candidate_kfold': cand_kfold
    }
    with open('scratch/kfold_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('Saved scratch/kfold_comparison.json')
