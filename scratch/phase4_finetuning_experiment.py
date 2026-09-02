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
    def __init__(self, num_patterns=3, num_categories=7, pretrained=True, freeze_mode='none'):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)
        feature_size = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()
        
        # Apply freezing strategies
        if freeze_mode == 'frozen_backbone': # F0
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif freeze_mode == 'partial_backbone': # F1 (Freeze first 10 feature blocks, unfreeze 10, 11, 12)
            for idx, block in enumerate(self.backbone.features):
                if idx < 10:
                    for param in block.parameters():
                        param.requires_grad = False
                else:
                    for param in block.parameters():
                        param.requires_grad = True
        elif freeze_mode == 'full_finetune': # F2
            for param in self.backbone.parameters():
                param.requires_grad = True
                
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

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def evaluate_model(model, dataloader, patterns, categories, device='cpu'):
    model.eval()
    pat_to_idx = {p: i for i, p in enumerate(patterns)}
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    
    p_true, p_pred = [], []
    c_true, c_pred = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            outputs = model(images)
            
            p_logits = outputs['pattern']
            c_logits = outputs['category']
            
            pat_p = torch.argmax(p_logits, dim=1).cpu().numpy()
            cat_p = torch.argmax(c_logits, dim=1).cpu().numpy()
            
            p_t = [pat_to_idx[p] for p in batch['structural_pattern']]
            c_t = [cat_to_idx[c] for c in batch['category']]
            
            p_true.extend(p_t)
            p_pred.extend(pat_p)
            c_true.extend(c_t)
            c_pred.extend(cat_p)
            
    cm_p = confusion_matrix(p_true, p_pred, labels=list(range(len(patterns))))
    cm_c = confusion_matrix(c_true, c_pred, labels=list(range(len(categories))))
    
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
        'per_class_pattern': get_per_class(cm_p, patterns),
        'per_class_category': get_per_class(cm_c, categories),
        'confusion_pattern': cm_p.tolist(),
        'confusion_category': cm_c.tolist(),
    }

def run_finetuning_cv(freeze_mode='full_finetune', lr=1e-4, seed=42, n_splits=5, epochs=10):
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
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_results = []
    
    dummy_model = CycloneDetector(num_patterns=len(patterns), num_categories=len(categories), pretrained=True, freeze_mode=freeze_mode)
    tot_params, train_params = count_parameters(dummy_model)
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(dev_df, dev_df['structural_pattern'])):
        tr_sub = dev_df.iloc[tr_idx].reset_index(drop=True)
        va_sub = dev_df.iloc[va_idx].reset_index(drop=True)
        
        tr_ds = ZipImageDataset(tr_sub, z, transform=transform)
        va_ds = ZipImageDataset(va_sub, z, transform=transform)
        
        tr_loader = DataLoader(tr_ds, batch_size=16, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=16, shuffle=False)
        
        model = CycloneDetector(num_patterns=len(patterns), num_categories=len(categories), pretrained=True, freeze_mode=freeze_mode).to(device)
        
        presence_loss = nn.BCEWithLogitsLoss()
        pattern_loss = nn.CrossEntropyLoss()
        category_loss = nn.CrossEntropyLoss()
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        
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
                
                total_loss = l_pres + l_pat + l_cat
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
                    val_loss += (l_pres + l_pat + l_cat).item()
                    
            val_loss /= len(va_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
        model.load_state_dict(best_state)
        res = evaluate_model(model, va_loader, patterns, categories, device=device)
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
        'freeze_mode': freeze_mode,
        'learning_rate': lr,
        'total_params': tot_params,
        'trainable_params': train_params,
        'summary': summarize(fold_results),
        'raw_folds': fold_results,
        'patterns': patterns,
        'categories': categories
    }

if __name__ == '__main__':
    print("=== RUNNING PHASE 4 FINE-TUNING EXPERIMENT (5-FOLD CV) ===")
    
    print("\n--- F0: Frozen Backbone (Heads Only) ---")
    f0_res = run_finetuning_cv(freeze_mode='frozen_backbone', lr=1e-4, seed=42)
    print(f"  Params: Trainable {f0_res['trainable_params']:,} / Total {f0_res['total_params']:,}")
    for k, v in f0_res['summary'].items():
        print(f"  {k:<30}: {v['mean']:.4f} +/- {v['std']:.4f}")
        
    print("\n--- F1: Partial Fine-Tuning (Later Blocks + Heads) ---")
    f1_res = run_finetuning_cv(freeze_mode='partial_backbone', lr=1e-4, seed=42)
    print(f"  Params: Trainable {f1_res['trainable_params']:,} / Total {f1_res['total_params']:,}")
    for k, v in f1_res['summary'].items():
        print(f"  {k:<30}: {v['mean']:.4f} +/- {v['std']:.4f}")
        
    print("\n--- F2: Full Fine-Tuning (Full Model Baseline) ---")
    f2_res = run_finetuning_cv(freeze_mode='full_finetune', lr=1e-4, seed=42)
    print(f"  Params: Trainable {f2_res['trainable_params']:,} / Total {f2_res['total_params']:,}")
    for k, v in f2_res['summary'].items():
        print(f"  {k:<30}: {v['mean']:.4f} +/- {v['std']:.4f}")
        
    all_ft = {
        'F0_frozen_backbone': f0_res,
        'F1_partial_finetune': f1_res,
        'F2_full_finetune': f2_res
    }
    with open('scratch/phase4_finetuning_results.json', 'w') as f:
        json.dump(all_ft, f, indent=2)
    print("\nSaved scratch/phase4_finetuning_results.json")
