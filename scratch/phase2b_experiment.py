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

def compute_fold_compatibility_matrix(train_df, patterns, categories, alpha=0.1):
    """
    Construct compatibility matrix C[pattern_idx, category_idx] = P_train(category | pattern)
    strictly from training fold data, with Laplace/additive smoothing alpha.
    """
    n_pat = len(patterns)
    n_cat = len(categories)
    counts = np.zeros((n_pat, n_cat), dtype=np.float64)
    
    pat_to_i = {p: i for i, p in enumerate(patterns)}
    cat_to_j = {c: j for j, c in enumerate(categories)}
    
    for _, row in train_df.iterrows():
        p = row['structural_pattern']
        c = row['category']
        if p in pat_to_i and c in cat_to_j:
            counts[pat_to_i[p], cat_to_j[c]] += 1.0
            
    # Additive smoothing
    counts_smoothed = counts + alpha
    # Row normalize to get P(category | pattern)
    C = counts_smoothed / counts_smoothed.sum(axis=1, keepdims=True)
    
    # Also reverse matrix D[category_idx, pattern_idx] = P(pattern | category)
    D = counts_smoothed.T / counts_smoothed.T.sum(axis=1, keepdims=True)
    
    return C, D, counts

def evaluate_with_consistency(model, dataloader, patterns, categories, C, D, beta=1.0, device='cpu'):
    model.eval()
    pat_to_idx = {p: i for i, p in enumerate(patterns)}
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    
    p_true = []
    c_true = []
    
    # Raw CNN predictions
    raw_p_pred = []
    raw_c_pred = []
    
    # Adjusted predictions
    adj_p_pred = []
    adj_c_pred = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            outputs = model(images)
            
            p_logits = outputs['pattern']
            c_logits = outputs['category']
            
            p_probs = torch.softmax(p_logits, dim=1).cpu().numpy() # (B, n_pat)
            c_probs = torch.softmax(c_logits, dim=1).cpu().numpy() # (B, n_cat)
            
            for b in range(len(images)):
                p_vec = p_probs[b] # (n_pat,)
                c_vec = c_probs[b] # (n_cat,)
                
                # Raw predictions
                raw_p = int(np.argmax(p_vec))
                raw_c = int(np.argmax(c_vec))
                raw_p_pred.append(raw_p)
                raw_c_pred.append(raw_c)
                
                # Consistency adjustment:
                # S_cat(j) = sum_i p_i * C[i, j]
                s_cat = p_vec @ C # (n_cat,)
                adj_c_scores = c_vec * (s_cat ** beta)
                adj_c = int(np.argmax(adj_c_scores))
                
                # S_pat(i) = sum_j c_j * D[j, i]
                s_pat = c_vec @ D # (n_pat,)
                adj_p_scores = p_vec * (s_pat ** beta)
                adj_p = int(np.argmax(adj_p_scores))
                
                adj_p_pred.append(adj_p)
                adj_c_pred.append(adj_c)
                
                p_true.append(pat_to_idx[batch['structural_pattern'][b]])
                c_true.append(cat_to_idx[batch['category'][b]])
                
    def calc_metrics(p_p, c_p):
        consistent_cnt = sum(1 for pi, ci in zip(p_p, c_p) if is_physically_consistent(patterns[pi], categories[ci]))
        return {
            'pattern_accuracy': float(accuracy_score(p_true, p_p)),
            'pattern_precision_weighted': float(precision_score(p_true, p_p, average='weighted', zero_division=0)),
            'pattern_recall_weighted': float(recall_score(p_true, p_p, average='weighted', zero_division=0)),
            'pattern_f1_weighted': float(f1_score(p_true, p_p, average='weighted', zero_division=0)),
            'pattern_f1_macro': float(f1_score(p_true, p_p, average='macro', zero_division=0)),
            'category_accuracy': float(accuracy_score(c_true, c_p)),
            'category_precision_weighted': float(precision_score(c_true, c_p, average='weighted', zero_division=0)),
            'category_recall_weighted': float(recall_score(c_true, c_p, average='weighted', zero_division=0)),
            'category_f1_weighted': float(f1_score(c_true, c_p, average='weighted', zero_division=0)),
            'category_f1_macro': float(f1_score(c_true, c_p, average='macro', zero_division=0)),
            'physical_consistency_rate': float(consistent_cnt / len(p_true) if len(p_true) > 0 else 0.0),
            'confusion_pattern': confusion_matrix(p_true, p_p, labels=list(range(len(patterns)))).tolist(),
            'confusion_category': confusion_matrix(c_true, c_p, labels=list(range(len(categories)))).tolist(),
        }
        
    return calc_metrics(raw_p_pred, raw_c_pred), calc_metrics(adj_p_pred, adj_c_pred)

def run_phase2b_experiment(alpha=0.1, beta=1.0, seed=42, n_splits=5, epochs=10, lr=1e-4):
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    z = zipfile.ZipFile('PS70-main.zip')
    train_df = pd.read_csv(io.BytesIO(z.read('PS70-main/data/processed/detection/train_detection.csv')))
    val_df = pd.read_csv(io.BytesIO(z.read('PS70-main/data/processed/detection/val_detection.csv')))
    test_df = pd.read_csv(io.BytesIO(z.read('PS70-main/data/processed/detection/test_detection.csv')))
    
    dev_df = pd.concat([train_df, val_df], ignore_index=True)
    
    # Canonical label list (from dev_df)
    patterns = sorted(dev_df["structural_pattern"].dropna().unique())
    categories = sorted(dev_df["category"].dropna().unique())
    
    pattern_to_idx = {p: i for i, p in enumerate(patterns)}
    category_to_idx = {c: i for i, c in enumerate(categories)}
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    base_fold_results = []
    cand_fold_results = []
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(dev_df, dev_df['structural_pattern'])):
        tr_sub = dev_df.iloc[tr_idx].reset_index(drop=True)
        va_sub = dev_df.iloc[val_idx].reset_index(drop=True)
        
        # Build fold-specific compatibility matrix strictly from tr_sub! (NO LEAKAGE)
        C_fold, D_fold, counts_fold = compute_fold_compatibility_matrix(tr_sub, patterns, categories, alpha=alpha)
        
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
        
        # Evaluate on validation fold
        base_res, cand_res = evaluate_with_consistency(model, va_loader, patterns, categories, C_fold, D_fold, beta=beta, device=device)
        base_fold_results.append(base_res)
        cand_fold_results.append(cand_res)
        
    def summarize(res_list):
        scalar_keys = [k for k in res_list[0].keys() if not k.startswith('confusion')]
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
        'patterns': patterns,
        'categories': categories,
        'baseline_cv': summarize(base_fold_results),
        'candidate_cv': summarize(cand_fold_results),
        'baseline_raw_folds': base_fold_results,
        'candidate_raw_folds': cand_fold_results
    }

if __name__ == '__main__':
    res = run_phase2b_experiment(alpha=0.1, beta=1.0, seed=42)
    
    print("=== BASELINE 5-FOLD CV SUMMARY ===")
    for k, v in res['baseline_cv'].items():
        print(f"  {k:<30}: {v['mean']:.4f} +/- {v['std']:.4f}")
        
    print("\n=== CANDIDATE (DVORAK CONSISTENCY) 5-FOLD CV SUMMARY ===")
    for k, v in res['candidate_cv'].items():
        print(f"  {k:<30}: {v['mean']:.4f} +/- {v['std']:.4f}")
        
    with open('scratch/phase2b_cv_results.json', 'w') as f:
        json.dump(res, f, indent=2)
    print("\nSaved results to scratch/phase2b_cv_results.json")
