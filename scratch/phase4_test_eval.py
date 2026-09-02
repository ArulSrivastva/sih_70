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
    def __init__(self, num_patterns=3, num_categories=7, pretrained=True, freeze_mode='full_finetune'):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)
        feature_size = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()
        
        if freeze_mode == 'frozen_backbone':
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif freeze_mode == 'partial_backbone':
            for idx, block in enumerate(self.backbone.features):
                if idx < 10:
                    for param in block.parameters():
                        param.requires_grad = False
                else:
                    for param in block.parameters():
                        param.requires_grad = True
        elif freeze_mode == 'full_finetune':
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

def train_and_eval_test(aug_type='A0', freeze_mode='full_finetune', lr=1e-4, seed=42, epochs=10):
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
    
    if aug_type == 'A0':
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    elif aug_type == 'A1':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor()
        ])
    elif aug_type == 'A2':
        train_transform = transforms.Compose([
            transforms.RandomRotation(degrees=(-10, 10)),
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor()
        ])
        
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    train_ds = ZipImageDataset(train_df, z, transform=train_transform)
    val_ds = ZipImageDataset(val_df, z, transform=val_test_transform)
    test_ds = ZipImageDataset(test_df, z, transform=val_test_transform)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    model = CycloneDetector(num_patterns=len(patterns), num_categories=len(categories), pretrained=True, freeze_mode=freeze_mode).to(device)
    
    presence_loss = nn.BCEWithLogitsLoss()
    pattern_loss = nn.CrossEntropyLoss()
    category_loss = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
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
            
            total_loss = l_pres + l_pat + l_cat
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
                val_loss += (l_pres + l_pat + l_cat).item()
                
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    model.load_state_dict(best_state)
    return evaluate_model(model, test_loader, patterns, categories, device=device)

if __name__ == '__main__':
    # 1. Evaluate locked baseline checkpoint directly
    z = zipfile.ZipFile('PS70-main.zip')
    test_df = pd.read_csv(io.BytesIO(z.read('PS70-main/data/processed/detection/test_detection.csv')))
    combined_df = pd.read_csv(io.BytesIO(z.read('PS70-main/data/processed/detection/detection_all.csv')))
    patterns = sorted(combined_df["structural_pattern"].dropna().unique())
    categories = sorted(combined_df["category"].dropna().unique())
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    test_ds = ZipImageDataset(test_df, z, transform=val_test_transform)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    ckpt_raw = z.read('PS70-main/models/detection/model_weights.pt')
    ckpt = torch.load(io.BytesIO(ckpt_raw), map_location='cpu', weights_only=True)
    baseline_model = CycloneDetector(num_patterns=len(patterns), num_categories=len(categories), pretrained=False)
    baseline_model.load_state_dict(ckpt['model_state_dict'])
    baseline_test = evaluate_model(baseline_model, test_loader, patterns, categories)
    
    print("=== LOCKED BASELINE TEST SET EVALUATION ===")
    print(f"Pattern Acc: {baseline_test['pattern_accuracy']:.4f}, Macro-F1: {baseline_test['pattern_f1_macro']:.4f}")
    print(f"Category Acc: {baseline_test['category_accuracy']:.4f}, Macro-F1: {baseline_test['category_f1_macro']:.4f}")
    
    # 2. Evaluate Best Augmentation / Fine-Tuning candidate (A1 / F1 / F0)
    print("\n=== EVALUATING A1 CANDIDATE ON TEST SET ===")
    a1_test = train_and_eval_test(aug_type='A1', freeze_mode='full_finetune')
    print(f"Pattern Acc: {a1_test['pattern_accuracy']:.4f}, Macro-F1: {a1_test['pattern_f1_macro']:.4f}")
    print(f"Category Acc: {a1_test['category_accuracy']:.4f}, Macro-F1: {a1_test['category_f1_macro']:.4f}")
    
    print("\n=== EVALUATING F0 (FROZEN BACKBONE) ON TEST SET ===")
    f0_test = train_and_eval_test(aug_type='A0', freeze_mode='frozen_backbone')
    print(f"Pattern Acc: {f0_test['pattern_accuracy']:.4f}, Macro-F1: {f0_test['pattern_f1_macro']:.4f}")
    print(f"Category Acc: {f0_test['category_accuracy']:.4f}, Macro-F1: {f0_test['category_f1_macro']:.4f}")
    
    results = {
        'baseline_locked': baseline_test,
        'candidate_A1_aug': a1_test,
        'candidate_F0_frozen': f0_test
    }
    with open('scratch/phase4_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved scratch/phase4_test_results.json")
