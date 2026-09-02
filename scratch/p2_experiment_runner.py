import zipfile
import io
import json
import os
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

# Set seeds
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

def evaluate_model(model, dataloader, pattern_to_idx, category_to_idx, device='cpu'):
    model.eval()
    idx_to_pattern = {v: k for k, v in pattern_to_idx.items()}
    idx_to_category = {v: k for k, v in category_to_idx.items()}
    
    p_true, p_pred = [], []
    c_true, c_pred = [], []
    pres_preds = []
    consistent_count = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            outputs = model(images)
            
            pres = torch.sigmoid(outputs['presence']).cpu().numpy().flatten()
            pres_preds.extend(pres)
            
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
        'n_samples': total,
        'pattern_accuracy': float(accuracy_score(p_true, p_pred)),
        'pattern_f1_weighted': float(f1_score(p_true, p_pred, average='weighted', zero_division=0)),
        'pattern_f1_macro': float(f1_score(p_true, p_pred, average='macro', zero_division=0)),
        'category_accuracy': float(accuracy_score(c_true, c_pred)),
        'category_f1_weighted': float(f1_score(c_true, c_pred, average='weighted', zero_division=0)),
        'category_f1_macro': float(f1_score(c_true, c_pred, average='macro', zero_division=0)),
        'physical_consistency_rate': float(consistent_count / total if total > 0 else 0.0),
    }

def train_and_eval(train_transform, val_transform, use_class_weights=False, epochs=10, lr=1e-4, seed=42):
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    z = zipfile.ZipFile('PS70-main.zip')
    train_df = pd.read_csv(io.BytesIO(z.read('PS70-main/data/processed/detection/train_detection.csv')))
    val_df = pd.read_csv(io.BytesIO(z.read('PS70-main/data/processed/detection/val_detection.csv')))
    test_df = pd.read_csv(io.BytesIO(z.read('PS70-main/data/processed/detection/test_detection.csv')))
    
    # Label maps built on combined as per P2 baseline
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    patterns = sorted(combined_df["structural_pattern"].dropna().unique())
    categories = sorted(combined_df["category"].dropna().unique())
    pattern_to_idx = {pattern: idx for idx, pattern in enumerate(patterns)}
    category_to_idx = {category: idx for idx, category in enumerate(categories)}
    
    train_dataset = ZipImageDataset(train_df, z, transform=train_transform)
    val_dataset = ZipImageDataset(val_df, z, transform=val_transform)
    test_dataset = ZipImageDataset(test_df, z, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    model = CycloneDetector(num_patterns=len(pattern_to_idx), num_categories=len(category_to_idx), pretrained=True).to(device)
    
    presence_loss = nn.BCEWithLogitsLoss()
    
    if use_class_weights:
        # Compute inverse frequency weights from train_df
        pat_counts = train_df['structural_pattern'].value_counts()
        p_weights = torch.tensor([1.0 / pat_counts.get(idx_to_pattern_name, 1.0) for idx_to_pattern_name in patterns], dtype=torch.float).to(device)
        p_weights = p_weights / p_weights.sum() * len(patterns)
        pattern_loss = nn.CrossEntropyLoss(weight=p_weights)
        
        cat_counts = train_df['category'].value_counts()
        c_weights = torch.tensor([1.0 / max(cat_counts.get(idx_to_cat_name, 0.5), 0.5) for idx_to_cat_name in categories], dtype=torch.float).to(device)
        c_weights = c_weights / c_weights.sum() * len(categories)
        category_loss = nn.CrossEntropyLoss(weight=c_weights)
    else:
        pattern_loss = nn.CrossEntropyLoss()
        category_loss = nn.CrossEntropyLoss()
        
    optimizer = Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            images = batch["image"].to(device)
            detected = batch["detected"].float().unsqueeze(1).to(device)
            pattern = torch.tensor([pattern_to_idx[p] for p in batch["structural_pattern"]], dtype=torch.long).to(device)
            category = torch.tensor([category_to_idx[c] for c in batch["category"]], dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            l_pres = presence_loss(outputs["presence"], detected)
            l_pat = pattern_loss(outputs["pattern"], pattern)
            l_cat = category_loss(outputs["category"], category)
            
            loss = l_pres + l_pat + l_cat
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                detected = batch["detected"].float().unsqueeze(1).to(device)
                pattern = torch.tensor([pattern_to_idx[p] for p in batch["structural_pattern"]], dtype=torch.long).to(device)
                category = torch.tensor([category_to_idx[c] for c in batch["category"]], dtype=torch.long).to(device)
                
                outputs = model(images)
                l_pres = presence_loss(outputs["presence"], detected)
                l_pat = pattern_loss(outputs["pattern"], pattern)
                l_cat = category_loss(outputs["category"], category)
                val_loss += (l_pres + l_pat + l_cat).item()
                
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    # Load best model
    model.load_state_dict(best_model_state)
    
    val_metrics = evaluate_model(model, val_loader, pattern_to_idx, category_to_idx, device=device)
    test_metrics = evaluate_model(model, test_loader, pattern_to_idx, category_to_idx, device=device)
    train_metrics = evaluate_model(model, train_loader, pattern_to_idx, category_to_idx, device=device)
    
    return {
        'best_val_loss': best_val_loss,
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics
    }

if __name__ == '__main__':
    t_baseline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    t_normalized = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print('Running Baseline (Unnormalized)...')
    base_res = train_and_eval(t_baseline, t_baseline, use_class_weights=False, seed=42)
    print('Baseline Val:', base_res['val'])
    print('Baseline Test:', base_res['test'])
    print()
    
    print('Running Candidate (ImageNet Normalized Transforms)...')
    cand_res = train_and_eval(t_normalized, t_normalized, use_class_weights=False, seed=42)
    print('Candidate Val:', cand_res['val'])
    print('Candidate Test:', cand_res['test'])
    print()
    
    print('Running Candidate + Class Weights...')
    cand_cw_res = train_and_eval(t_normalized, t_normalized, use_class_weights=True, seed=42)
    print('Candidate CW Val:', cand_cw_res['val'])
    print('Candidate CW Test:', cand_cw_res['test'])
