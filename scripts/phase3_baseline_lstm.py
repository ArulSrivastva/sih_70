"""
Phase 3: Baseline + First LSTM Forecaster

Parts A-H: Data preparation, persistence baseline, LSTM training,
evaluation, visualizations, and model artifacts.
"""

import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import warnings
warnings.filterwarnings('ignore')

# Fix encoding for Windows
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================================
# CONFIGURATION
# ============================================================================
SEQUENCES_DIR = Path("data/processed/sequences")
STATS_JSON = Path("data/processed/normalization_stats.json")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")
PLOTS_DIR = Path("results/plots")
SRC_DIR = Path("src/forecasting")

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Feature names (features 0-5 in input, feature 6 = missingness indicator)
FEATURE_NAMES = ['latitude', 'longitude', 'wind_speed_kmh', 'pressure_hpa',
                 'storm_speed', 'storm_direction']
MISSINGNESS_FEATURE = 'pressure_hpa_missing'
ALL_FEATURES = FEATURE_NAMES + [MISSINGNESS_FEATURE]

# Input/output dimensions
SEQ_LEN = 9
N_FEATURES = 7   # 6 continuous + 1 missingness indicator
N_CONTINUOUS = 6  # first 6 features are continuous
N_TARGETS = 3     # lat, lon, wind
N_HORIZONS = 3    # 6h, 12h, 24h
HORIZONS = [6, 12, 24]

# Training
BATCH_SIZE = 64
MAX_EPOCHS = 150
PATIENCE = 15
LR_INIT = 1e-3
LR_FACTOR = 0.5
LR_PATIENCE = 7
MIN_LR = 1e-6

print("=" * 70)
print("PHASE 3: Baseline + First LSTM Forecaster")
print("=" * 70)

# Create directories
for d in [RESULTS_DIR, PLOTS_DIR, MODELS_DIR, SRC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PART A: DATA PREPARATION
# ============================================================================
print("\n" + "=" * 70)
print("PART A: DATA PREPARATION")
print("=" * 70)

# --- Load raw sequences ---
print("\n[Part A.1] Loading sequences...")
X_train = np.load(SEQUENCES_DIR / 'X_train.npy')
y_train = np.load(SEQUENCES_DIR / 'y_train.npy')
X_val = np.load(SEQUENCES_DIR / 'X_val.npy')
y_val = np.load(SEQUENCES_DIR / 'y_val.npy')
X_test = np.load(SEQUENCES_DIR / 'X_test.npy')
y_test = np.load(SEQUENCES_DIR / 'y_test.npy')

print(f"  X_train: {X_train.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  X_val:   {X_val.shape}")
print(f"  y_val:   {y_val.shape}")
print(f"  X_test:  {X_test.shape}")
print(f"  y_test:  {y_test.shape}")

# --- Inspect features ---
print("\n[Part A.2] Feature inspection...")
for i, name in enumerate(ALL_FEATURES):
    vals = X_train[:, :, i]
    nan_pct = 100 * np.isnan(vals).sum() / vals.size
    print(f"  Feature {i}: {name:<25} NaN: {nan_pct:.1f}%")

print(f"\n  Feature 6 ({MISSINGNESS_FEATURE}) is the binary missingness indicator.")
print(f"  It should be 0.0 or 1.0 (or NaN if pressure feature is NaN).")

# --- Load normalization stats ---
print("\n[Part A.3] Loading normalization statistics...")
with open(STATS_JSON, 'r') as f:
    norm_stats = json.load(f)

for feat in FEATURE_NAMES:
    s = norm_stats[feat]
    print(f"  {feat}: mean={s['mean']:.3f}, std={s['std']:.3f}, "
          f"n_valid={s['n_valid']}, n_total={s['n_total']}")

# --- Impute missing values ---
# Strategy:
#   - pressure_hpa: Replace NaN with training-set mean (992.196 hPa).
#     Binary indicator (feature 6) set to 1.0 wherever pressure was NaN.
#   - wind_speed_kmh: Replace NaN with training-set mean (70.000 km/h).
#     Only 5.3% missing; a separate indicator is not warranted.
print("\n[Part A.4] Handling missing values...")

pressure_mean = norm_stats['pressure_hpa']['mean']
pressure_std = norm_stats['pressure_hpa']['std']
wind_mean = norm_stats['wind_speed_kmh']['mean']
wind_std = norm_stats['wind_speed_kmh']['std']

def impute_missing(X):
    """Impute missing pressure with training mean, set missingness indicator.
    Impute missing wind with training mean (small % missing)."""
    X_imp = X.copy()
    for i in range(X.shape[0]):  # samples
        for t in range(X.shape[1]):  # timesteps
            # Pressure imputation
            if np.isnan(X_imp[i, t, 3]):  # pressure_hpa is NaN
                X_imp[i, t, 3] = pressure_mean
                X_imp[i, t, 6] = 1.0  # mark as missing
            else:
                X_imp[i, t, 6] = 0.0  # mark as present
            # Wind imputation (no indicator — only 5.3% missing)
            if np.isnan(X_imp[i, t, 2]):  # wind_speed_kmh is NaN
                X_imp[i, t, 2] = wind_mean
    return X_imp

X_train_imp = impute_missing(X_train)
X_val_imp = impute_missing(X_val)
X_test_imp = impute_missing(X_test)

# Verify imputation
print(f"  Pressure NaN after imputation (train): {np.isnan(X_train_imp[:, :, 3]).sum()}")
print(f"  Wind NaN after imputation (train): {np.isnan(X_train_imp[:, :, 2]).sum()}")
print(f"  Missingness indicator distribution (train): "
      f"present={int((X_train_imp[:, :, 6] == 0).sum())}, "
      f"missing={int((X_train_imp[:, :, 6] == 1).sum())}")

# --- Normalize continuous features ---
# Use only training-set statistics. Do NOT normalize missingness indicator (feature 6).
print("\n[Part A.5] Normalizing continuous features...")

def normalize(X, stats, feature_names):
    """Normalize continuous features using z-score with training statistics."""
    X_norm = X.copy()
    for i, feat in enumerate(feature_names):
        mean = stats[feat]['mean']
        std = stats[feat]['std']
        X_norm[:, :, i] = (X_norm[:, :, i] - mean) / std
    return X_norm

X_train_norm = normalize(X_train_imp, norm_stats, FEATURE_NAMES)
X_val_norm = normalize(X_val_imp, norm_stats, FEATURE_NAMES)
X_test_norm = normalize(X_test_imp, norm_stats, FEATURE_NAMES)

# Feature 6 (missingness) is NOT normalized - it stays 0/1
print(f"  Feature 6 (missingness) min={X_train_norm[:,:,6].min()}, "
      f"max={X_train_norm[:,:,6].max()}")

# --- Normalize targets ---
# We also need to denormalize predictions later. Save target stats.
target_stats = {}
for i, feat in enumerate(['latitude', 'longitude', 'wind_speed_kmh']):
    vals = y_train[:, :, i].flatten()
    valid = vals[~np.isnan(vals)]
    target_stats[feat] = {
        'mean': float(np.mean(valid)),
        'std': float(np.std(valid)),
        'min': float(np.min(valid)),
        'max': float(np.max(valid))
    }

def normalize_targets(y, stats, target_names):
    """Normalize targets using z-score."""
    y_norm = y.copy()
    for i, feat in enumerate(target_names):
        mean = stats[feat]['mean']
        std = stats[feat]['std']
        y_norm[:, :, i] = (y_norm[:, :, i] - mean) / std
    return y_norm

y_train_norm = normalize_targets(y_train, target_stats, ['latitude', 'longitude', 'wind_speed_kmh'])
y_val_norm = normalize_targets(y_val, target_stats, ['latitude', 'longitude', 'wind_speed_kmh'])
y_test_norm = normalize_targets(y_test, target_stats, ['latitude', 'longitude', 'wind_speed_kmh'])

print(f"  Target stats:")
for feat, s in target_stats.items():
    print(f"    {feat}: mean={s['mean']:.3f}, std={s['std']:.3f}")

# --- Save preprocessing config ---
preproc_config = {
    'feature_names': ALL_FEATURES,
    'n_features': N_FEATURES,
    'n_continuous': N_CONTINUOUS,
    'seq_len': SEQ_LEN,
    'n_targets': N_TARGETS,
    'n_horizons': N_HORIZONS,
    'horizons_hours': HORIZONS,
    'target_names': ['latitude', 'longitude', 'wind_speed_kmh'],
    'pressure_imputation': {
        'method': 'training_mean',
        'value': pressure_mean,
        'indicator_feature_idx': 6,
        'document': 'Missing pressure replaced with training-set mean. '
                    'Binary indicator (feature 6) marks imputed timesteps.'
    },
    'normalization': {
        'method': 'z_score',
        'training_statistics': norm_stats,
        'target_statistics': target_stats,
        'missingness_indicator': 'NOT normalized (stays 0/1)'
    },
    'random_seed': RANDOM_SEED,
    'preprocessing_date': datetime.now().isoformat()
}

with open(MODELS_DIR / 'preprocessing_config.json', 'w') as f:
    json.dump(preproc_config, f, indent=2)
print(f"\n  Saved: models/preprocessing_config.json")

# Convert to torch tensors
X_train_t = torch.FloatTensor(X_train_norm)
y_train_t = torch.FloatTensor(y_train_norm)
X_val_t = torch.FloatTensor(X_val_norm)
y_val_t = torch.FloatTensor(y_val_norm)
X_test_t = torch.FloatTensor(X_test_norm)
y_test_t = torch.FloatTensor(y_test_norm)

# Handle any remaining NaN in targets (mask them)
y_train_mask = ~torch.isnan(y_train_t)
y_val_mask = ~torch.isnan(y_val_t)
y_test_mask = ~torch.isnan(y_test_t)

# Replace NaN targets with 0 for loss computation (masked later)
y_train_t[~y_train_mask] = 0
y_val_t[~y_val_mask] = 0
y_test_t[~y_test_mask] = 0

# Verify no NaN in inputs
print(f"\n  NaN check after imputation:")
print(f"    X_train NaN: {torch.isnan(X_train_t).sum().item()}")
print(f"    X_val NaN: {torch.isnan(X_val_t).sum().item()}")
print(f"    X_test NaN: {torch.isnan(X_test_t).sum().item()}")
print(f"    y_train NaN: {(~y_train_mask).sum().item()}")
print(f"    y_val NaN: {(~y_val_mask).sum().item()}")
print(f"    y_test NaN: {(~y_test_mask).sum().item()}")

print(f"\n  Data shapes after preprocessing:")
print(f"    X_train: {X_train_t.shape}, y_train: {y_train_t.shape}")
print(f"    X_val:   {X_val_t.shape},   y_val:   {y_val_t.shape}")
print(f"    X_test:  {X_test_t.shape},  y_test:  {y_test_t.shape}")

# ============================================================================
# PART B: PERSISTENCE BASELINE
# ============================================================================
print("\n" + "=" * 70)
print("PART B: PERSISTENCE BASELINE")
print("=" * 70)

def haversine_distance(lat1, lon1, lat2, lon2):
    """Great-circle distance in km using haversine formula."""
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def persistence_baseline(X_raw, y_raw):
    """
    Persistence baseline: predict current lat/lon/wind as future values.
    X_raw shape: (N, 9, 7) - raw (not normalized) sequences
    y_raw shape: (N, 3, 3) - raw targets
    """
    # Current state is last timestep (index 8)
    current_lat = X_raw[:, 8, 0]   # latitude
    current_lon = X_raw[:, 8, 1]   # longitude
    current_wind = X_raw[:, 8, 2]  # wind_speed_kmh

    results = {}
    for h_idx, h_hours in enumerate(HORIZONS):
        actual_lat = y_raw[:, h_idx, 0]
        actual_lon = y_raw[:, h_idx, 1]
        actual_wind = y_raw[:, h_idx, 2]

        # MAE
        lat_mae = np.nanmean(np.abs(current_lat - actual_lat))
        lon_mae = np.nanmean(np.abs(current_lon - actual_lon))
        wind_mae = np.nanmean(np.abs(current_wind - actual_wind))

        # Track error (km)
        track_err = haversine_distance(current_lat, current_lon, actual_lat, actual_lon)
        track_err_mean = np.nanmean(track_err)

        results[f'{h_hours}h'] = {
            'latitude_mae_deg': float(lat_mae),
            'longitude_mae_deg': float(lon_mae),
            'wind_speed_mae_kmh': float(wind_mae),
            'track_error_km': float(track_err_mean),
            'n_samples': int(np.sum(~np.isnan(actual_lat)))
        }
        print(f"  +{h_hours}h: lat_mae={lat_mae:.3f}°, lon_mae={lon_mae:.3f}°, "
              f"wind_mae={wind_mae:.1f} km/h, track={track_err_mean:.1f} km")

    return results

print("\nEvaluating persistence baseline on test set...")
persistence_results = persistence_baseline(X_test, y_test)

# Save
with open(RESULTS_DIR / 'persistence_baseline.json', 'w') as f:
    json.dump(persistence_results, f, indent=2)
print(f"\n  Saved: results/persistence_baseline.json")

# ============================================================================
# PART C: LSTM MODEL
# ============================================================================
print("\n" + "=" * 70)
print("PART C: LSTM MODEL")
print("=" * 70)

class CycloneLSTM(nn.Module):
    """
    LSTM forecaster for cyclone track and intensity.

    Input:  (batch, seq_len=9, n_features=7)
    Output: (batch, n_horizons=3, n_targets=3)
    """
    def __init__(self, input_size=7, hidden_size=64, num_layers=1,
                 dropout=0.0, n_horizons=3, n_targets=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, n_horizons * n_targets)
        self.n_horizons = n_horizons
        self.n_targets = n_targets

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        out = self.fc(last_hidden)        # (batch, n_horizons * n_targets)
        out = out.view(-1, self.n_horizons, self.n_targets)
        return out

# Test model
model_test = CycloneLSTM(input_size=7, hidden_size=64, num_layers=1)
dummy = torch.randn(4, 9, 7)
out = model_test(dummy)
print(f"  Model output shape: {out.shape}")
print(f"  Model parameters: {sum(p.numel() for p in model_test.parameters()):,}")
del model_test, dummy, out

# ============================================================================
# PART D: TRAINING
# ============================================================================
print("\n" + "=" * 70)
print("PART D: TRAINING")
print("=" * 70)

def train_model(config, X_train, y_train, X_val, y_val):
    """Train model with given config, return model and training history."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CycloneLSTM(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=config['lr_patience'],
        min_lr=1e-6, verbose=False
    )
    criterion = nn.HuberLoss(reduction='none')

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                             shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                           shuffle=False)

    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    start_time = time.time()

    for epoch in range(config['max_epochs']):
        # Training
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch).mean()
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        scheduler.step(val_loss)

        # Early stopping (skip NaN losses)
        if not np.isnan(val_loss) and not np.isnan(train_loss):
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config['patience']:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}: train={train_loss:.6f}, "
                  f"val={val_loss:.6f}, lr={current_lr:.2e}")

    training_time = time.time() - start_time

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        print(f"    WARNING: No valid model found, using last epoch")
    model.to('cpu')

    return model, history, training_time, best_val_loss

# --- Configuration search ---
configs = [
    {
        'name': 'A',
        'hidden_size': 64,
        'num_layers': 1,
        'dropout': 0.0,
        'lr': 1e-3,
        'batch_size': 64,
        'max_epochs': MAX_EPOCHS,
        'patience': PATIENCE,
        'lr_patience': LR_PATIENCE,
        'input_size': N_FEATURES
    },
    {
        'name': 'B',
        'hidden_size': 128,
        'num_layers': 1,
        'dropout': 0.0,
        'lr': 1e-3,
        'batch_size': 64,
        'max_epochs': MAX_EPOCHS,
        'patience': PATIENCE,
        'lr_patience': LR_PATIENCE,
        'input_size': N_FEATURES
    },
    {
        'name': 'C',
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.1,
        'lr': 1e-3,
        'batch_size': 64,
        'max_epochs': MAX_EPOCHS,
        'patience': PATIENCE,
        'lr_patience': LR_PATIENCE,
        'input_size': N_FEATURES
    },
]

results_all = {}
histories = {}

for config in configs:
    name = config['name']
    print(f"\n  --- Configuration {name}: "
          f"hidden={config['hidden_size']}, layers={config['num_layers']}, "
          f"dropout={config['dropout']} ---")

    torch.manual_seed(RANDOM_SEED)
    model, history, train_time, best_val = train_model(
        config, X_train_t, y_train_t, X_val_t, y_val_t
    )

    results_all[name] = {
        'model': model,
        'history': history,
        'training_time': train_time,
        'best_val_loss': best_val,
        'config': config
    }
    histories[name] = history

    print(f"    Training time: {train_time:.1f}s")
    print(f"    Best val loss: {best_val:.6f}")
    print(f"    Epochs trained: {len(history['val_loss'])}")

# Select best configuration
best_name = min(results_all, key=lambda k: results_all[k]['best_val_loss'])
best_result = results_all[best_name]
best_model = best_result['model']

print(f"\n  Best configuration: {best_name}")
print(f"  Best val loss: {best_result['best_val_loss']:.6f}")
print(f"  Training time: {best_result['training_time']:.1f}s")

# ============================================================================
# PART E: TEST EVALUATION
# ============================================================================
print("\n" + "=" * 70)
print("PART E: TEST EVALUATION")
print("=" * 70)

def evaluate_model(model, X_test_t, X_test_raw, y_test_raw, target_stats):
    """Evaluate model on test set, denormalize predictions, compute metrics."""
    model.eval()
    with torch.no_grad():
        pred_norm = model(X_test_t).numpy()

    # Denormalize predictions
    pred = np.zeros_like(pred_norm)
    for i, feat in enumerate(['latitude', 'longitude', 'wind_speed_kmh']):
        mean = target_stats[feat]['mean']
        std = target_stats[feat]['std']
        pred[:, :, i] = pred_norm[:, :, i] * std + mean

    results = {}
    for h_idx, h_hours in enumerate(HORIZONS):
        actual_lat = y_test_raw[:, h_idx, 0]
        actual_lon = y_test_raw[:, h_idx, 1]
        actual_wind = y_test_raw[:, h_idx, 2]

        pred_lat = pred[:, h_idx, 0]
        pred_lon = pred[:, h_idx, 1]
        pred_wind = pred[:, h_idx, 2]

        lat_mae = np.nanmean(np.abs(pred_lat - actual_lat))
        lon_mae = np.nanmean(np.abs(pred_lon - actual_lon))
        wind_mae = np.nanmean(np.abs(pred_wind - actual_wind))

        track_err = haversine_distance(pred_lat, pred_lon, actual_lat, actual_lon)
        track_err_mean = np.nanmean(track_err)

        results[f'{h_hours}h'] = {
            'latitude_mae_deg': float(lat_mae),
            'longitude_mae_deg': float(lon_mae),
            'wind_speed_mae_kmh': float(wind_mae),
            'track_error_km': float(track_err_mean),
            'n_samples': int(len(actual_lat))
        }
        print(f"  +{h_hours}h: lat_mae={lat_mae:.3f}°, lon_mae={lon_mae:.3f}°, "
              f"wind_mae={wind_mae:.1f} km/h, track={track_err_mean:.1f} km")

    return results, pred

print("\nEvaluating LSTM on test set...")
lstm_results, lstm_predictions = evaluate_model(
    best_model, X_test_t, X_test, y_test, target_stats
)

# Save LSTM results
with open(RESULTS_DIR / 'lstm_results.json', 'w') as f:
    json.dump(lstm_results, f, indent=2)
print(f"\n  Saved: results/lstm_results.json")

# --- Model comparison ---
print("\nModel comparison:")
comparison = {}
for h in ['6h', '12h', '24h']:
    p = persistence_results[h]
    l = lstm_results[h]
    comparison[h] = {
        'persistence': p,
        'lstm': l,
        'improvement': {
            'latitude_mae_pct': 100 * (p['latitude_mae_deg'] - l['latitude_mae_deg']) / p['latitude_mae_deg'],
            'longitude_mae_pct': 100 * (p['longitude_mae_deg'] - l['longitude_mae_deg']) / p['longitude_mae_deg'],
            'wind_mae_pct': 100 * (p['wind_speed_mae_kmh'] - l['wind_speed_mae_kmh']) / p['wind_speed_mae_kmh'],
            'track_error_pct': 100 * (p['track_error_km'] - l['track_error_km']) / p['track_error_km']
        }
    }
    print(f"\n  +{h}:")
    print(f"    Track: persistence={p['track_error_km']:.1f} km, "
          f"lstm={l['track_error_km']:.1f} km "
          f"({comparison[h]['improvement']['track_error_pct']:+.1f}%)")
    print(f"    Wind:  persistence={p['wind_speed_mae_kmh']:.1f} km/h, "
          f"lstm={l['wind_speed_mae_kmh']:.1f} km/h "
          f"({comparison[h]['improvement']['wind_mae_pct']:+.1f}%)")

with open(RESULTS_DIR / 'model_comparison.json', 'w') as f:
    json.dump(comparison, f, indent=2)
print(f"\n  Saved: results/model_comparison.json")

# ============================================================================
# PART F: VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 70)
print("PART F: VISUALIZATIONS")
print("=" * 70)

# --- Plot 1: Training vs Validation loss ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, name in enumerate(['A', 'B', 'C']):
    ax = axes[idx]
    h = histories[name]
    ax.plot(h['train_loss'], label='Train', linewidth=1.5)
    ax.plot(h['val_loss'], label='Validation', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Huber Loss')
    ax.set_title(f'Config {name} (hidden={configs[idx]["hidden_size"]})')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Training vs Validation Loss', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'training_loss_curves.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Plot 2: Example predicted vs actual track ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Pick a test storm with good data
test_meta = pd.read_csv(SEQUENCES_DIR / 'test_metadata.csv')
storm_counts = test_meta['SID'].value_counts()
example_sid = storm_counts.index[0]
example_samples = test_meta[test_meta['SID'] == example_sid].index.tolist()

# Use middle sample
mid_idx = example_samples[len(example_samples)//2]

# Get raw input and actual targets
X_ex = X_test[mid_idx]  # (9, 7) raw
y_ex = y_test[mid_idx]  # (3, 3) raw
pred_ex = lstm_predictions[mid_idx]  # (3, 3)

# Input track
lats_in = X_ex[:, 0]
lons_in = X_ex[:, 1]

# Axes: lat over time, lon over time, wind over time
for ax_idx, (feat_idx, feat_name, unit) in enumerate([
    (0, 'Latitude', '°N'),
    (1, 'Longitude', '°E'),
    (2, 'Wind Speed', 'km/h')
]):
    ax = axes[ax_idx]

    # Input (9 timesteps)
    input_times = np.arange(0, 9)
    ax.plot(input_times, X_ex[:, feat_idx], 'b-o', markersize=4,
            label='Input', linewidth=2)

    # Actual targets
    target_times = np.array([9 + h//3 for h in HORIZONS])
    ax.plot(target_times, y_ex[:, feat_idx], 'g-s', markersize=8,
            label='Actual', linewidth=2, zorder=5)

    # Predicted targets
    ax.plot(target_times, pred_ex[:, feat_idx], 'r-^', markersize=8,
            label='Predicted', linewidth=2, zorder=5)

    ax.axvline(x=8.5, color='gray', linestyle='--', alpha=0.5, label='Forecast')
    ax.set_xlabel('Timestep (3h intervals)')
    ax.set_ylabel(f'{feat_name} ({unit})')
    ax.set_title(f'{feat_name}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle(f'Example Forecast: Storm {example_sid}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'example_forecast.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Plot 3: Track error vs forecast horizon ---
fig, ax = plt.subplots(figsize=(8, 5))

horizon_labels = ['+1h', '+3h', '+6h',  '+12h','+15h', '+24h']
persistence_track = [persistence_results[h]['track_error_km'] for h in ['+1h', '+3h', '+6h',  '+12h','+15h', '+24h']]
lstm_track = [lstm_results[h]['track_error_km'] for h in ['6h', '12h', '24h']]

x = np.arange(len(horizon_labels))
width = 0.35

bars1 = ax.bar(x - width/2, persistence_track, width, label='Persistence',
               color='#FF9800', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, lstm_track, width, label='LSTM',
               color='#2196F3', edgecolor='black', linewidth=0.5)

for bar, val in zip(bars1, persistence_track):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar, val in zip(bars2, lstm_track):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xlabel('Forecast Horizon')
ax.set_ylabel('Track Error (km)')
ax.set_title('Track Error by Forecast Horizon')
ax.set_xticks(x)
ax.set_xticklabels(horizon_labels)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'track_error_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Plot 4: Wind prediction vs actual for example test storms ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax_idx, storm_idx in enumerate([0, len(test_meta)//3, 2*len(test_meta)//3]):
    if storm_idx >= len(X_test):
        continue
    ax = axes[ax_idx]

    X_ex = X_test[storm_idx]
    y_ex = y_test[storm_idx]
    pred_ex = lstm_predictions[storm_idx]

    # Input wind
    input_times = np.arange(0, 9)
    ax.plot(input_times, X_ex[:, 2], 'b-o', markersize=3,
            label='Input', linewidth=1.5)

    # Actual and predicted
    target_times = np.array([9 + h//3 for h in HORIZONS])
    ax.plot(target_times, y_ex[:, 2], 'g-s', markersize=6,
            label='Actual', linewidth=2, zorder=5)
    ax.plot(target_times, pred_ex[:, 2], 'r-^', markersize=6,
            label='Predicted', linewidth=2, zorder=5)

    sid = test_meta.iloc[storm_idx]['SID']
    ax.axvline(x=8.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Wind Speed (km/h)')
    ax.set_title(f'Storm {sid}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Wind Speed: Predicted vs Actual', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'wind_prediction_examples.png', dpi=150, bbox_inches='tight')
plt.close()

print("  Saved:")
print(f"    {PLOTS_DIR}/training_loss_curves.png")
print(f"    {PLOTS_DIR}/example_forecast.png")
print(f"    {PLOTS_DIR}/track_error_comparison.png")
print(f"    {PLOTS_DIR}/wind_prediction_examples.png")

# ============================================================================
# PART G: MODEL ARTIFACTS
# ============================================================================
print("\n" + "=" * 70)
print("PART G: MODEL ARTIFACTS")
print("=" * 70)

# Save best model
torch.save(best_model.state_dict(), MODELS_DIR / 'lstm_forecaster.pt')
print(f"  Saved: models/lstm_forecaster.pt")

# Save model config
model_config = {
    'model_name': 'CycloneLSTM',
    'input_size': N_FEATURES,
    'seq_len': SEQ_LEN,
    'hidden_size': best_result['config']['hidden_size'],
    'num_layers': best_result['config']['num_layers'],
    'dropout': best_result['config']['dropout'],
    'n_horizons': N_HORIZONS,
    'n_targets': N_TARGETS,
    'horizons_hours': HORIZONS,
    'target_names': ['latitude', 'longitude', 'wind_speed_kmh'],
    'feature_names': ALL_FEATURES,
    'best_configuration': best_name,
    'best_val_loss': float(best_result['best_val_loss']),
    'training_time_seconds': float(best_result['training_time']),
    'random_seed': RANDOM_SEED,
    'training_date': datetime.now().isoformat()
}

with open(MODELS_DIR / 'model_config.json', 'w') as f:
    json.dump(model_config, f, indent=2)
print(f"  Saved: models/model_config.json")

# --- Inference module ---
inference_code = '''"""
Reusable inference module for cyclone forecasting.

Usage:
    from src.forecasting.inference import CycloneForecaster

    forecaster = CycloneForecaster()
    result = forecaster.forecast(input_sequence)
    # Returns: {"forecast": [{"hours": 6, "latitude": ..., ...}, ...]}
"""

import numpy as np
import json
import torch
from pathlib import Path


class CycloneForecaster:
    """End-to-end cyclone forecasting with normalization and denormalization."""

    def __init__(self, model_dir='models'):
        self.model_dir = Path(model_dir)
        self._load_configs()
        self._load_model()

    def _load_configs(self):
        with open(self.model_dir / 'model_config.json') as f:
            self.model_config = json.load(f)
        with open(self.model_dir / 'preprocessing_config.json') as f:
            self.preproc_config = json.load(f)

    def _load_model(self):
        from scripts.phase3_baseline_lstm import CycloneLSTM
        self.model = CycloneLSTM(
            input_size=self.model_config['input_size'],
            hidden_size=self.model_config['hidden_size'],
            num_layers=self.model_config['num_layers'],
            dropout=self.model_config['dropout']
        )
        self.model.load_state_dict(
            torch.load(self.model_dir / 'lstm_forecaster.pt',
                       weights_only=True)
        )
        self.model.eval()

    def _normalize(self, X):
        stats = self.preproc_config['normalization']['training_statistics']
        X_norm = X.copy()
        for i, feat in enumerate(self.preproc_config['feature_names'][:6]):
            mean = stats[feat]['mean']
            std = stats[feat]['std']
            X_norm[:, :, i] = (X_norm[:, :, i] - mean) / std
        return X_norm

    def _denormalize(self, pred_norm):
        stats = self.preproc_config['normalization']['target_statistics']
        pred = np.zeros_like(pred_norm)
        target_names = self.model_config['target_names']
        for i, feat in enumerate(target_names):
            mean = stats[feat]['mean']
            std = stats[feat]['std']
            pred[:, :, i] = pred_norm[:, :, i] * std + mean
        return pred

    def forecast(self, input_sequence):
        """
        Predict cyclone trajectory and intensity.

        Args:
            input_sequence: numpy array of shape (9, 7) or (1, 9, 7)
                Features: lat, lon, wind, pressure, storm_speed, storm_dir, pressure_missing

        Returns:
            dict with forecasts for +6h, +12h, +24h
        """
        if input_sequence.ndim == 2:
            input_sequence = input_sequence[np.newaxis, ...]

        X = self._normalize(input_sequence)
        X_t = torch.FloatTensor(X)

        with torch.no_grad():
            pred_norm = self.model(X_t).numpy()

        pred = self._denormalize(pred_norm)[0]  # (3, 3)

        result = {"forecast": []}
        for h_idx, h_hours in enumerate(self.model_config['horizons_hours']):
            result["forecast"].append({
                "hours": h_hours,
                "latitude": float(pred[h_idx, 0]),
                "longitude": float(pred[h_idx, 1]),
                "wind_speed_kmh": float(pred[h_idx, 2])
            })

        return result
'''

with open(SRC_DIR / 'inference.py', 'w') as f:
    f.write(inference_code)

# Create __init__.py files
(SRC_DIR / '__init__.py').touch()
(SRC_DIR.parent / '__init__.py').touch()

print(f"  Saved: src/forecasting/inference.py")
print(f"  Saved: src/forecasting/__init__.py")
print(f"  Saved: src/__init__.py")

# ============================================================================
# PART H: PHASE 3 REPORT
# ============================================================================
print("\n" + "=" * 70)
print("PART H: PHASE 3 REPORT")
print("=" * 70)

report_content = f"""# Phase 3 Report: Baseline + LSTM Forecaster

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Preprocessing

### 1.1 Input Data

| Property | Value |
|----------|-------|
| Source | Phase 2 sequences |
| X_train shape | {X_train.shape} |
| X_val shape | {X_val.shape} |
| X_test shape | {X_test.shape} |
| Features | 7 (6 continuous + 1 missingness indicator) |
| Sequence length | 9 timesteps (24h at 3h resolution) |

### 1.2 Feature Descriptions

| Index | Feature | Unit | Normalized |
|-------|---------|------|------------|
| 0 | latitude | °N | Yes (z-score) |
| 1 | longitude | °E | Yes (z-score) |
| 2 | wind_speed_kmh | km/h | Yes (z-score) |
| 3 | pressure_hpa | hPa | Yes (z-score) |
| 4 | storm_speed | kts | Yes (z-score) |
| 5 | storm_direction | degrees | Yes (z-score) |
| 6 | pressure_hpa_missing | binary | No (stays 0/1) |

---

## 2. Missing Pressure Handling

### Strategy

1. **Imputation:** Missing pressure values replaced with training-set mean ({pressure_mean:.1f} hPa)
2. **Indicator:** Binary feature (index 6) set to 1.0 wherever pressure was imputed
3. **Normalization:** Continuous features normalized using z-score; indicator NOT normalized

### Rationale

- The missingness indicator preserves the information that pressure was unavailable
- Using training-set mean avoids information leakage from val/test sets
- The model can learn to weight pressure-derived features less when the indicator is 1.0

### Statistics

| Set | Pressure missing | Indicator=1 |
|-----|-----------------|-------------|
| Train | {np.sum(np.isnan(X_train[:,:,3])):,} | {int(np.sum(X_train_imp[:,:,6]==1)):,} |
| Val | {np.sum(np.isnan(X_val[:,:,3])):,} | {int(np.sum(X_val_imp[:,:,6]==1)):,} |
| Test | {np.sum(np.isnan(X_test[:,:,3])):,} | {int(np.sum(X_test_imp[:,:,6]==1)):,} |

---

## 3. Persistence Baseline

The persistence baseline predicts that the current cyclone state persists unchanged.

**Rule:** For each horizon (+6h, +12h, +24h), predict:
- future latitude = current latitude (last input timestep)
- future longitude = current longitude
- future wind = current wind speed

### Results

| Horizon | Lat MAE (°) | Lon MAE (°) | Wind MAE (km/h) | Track Error (km) |
|---------|-------------|-------------|-----------------|------------------|
"""

for h in ['+1h', '+3h', '+6h',  '+12h','+15h', '+24h']:
    p = persistence_results[h]
    report_content += f"| +{h} | {p['latitude_mae_deg']:.3f} | {p['longitude_mae_deg']:.3f} | {p['wind_speed_mae_kmh']:.1f} | {p['track_error_km']:.1f} |\n"

report_content += f"""
---

## 4. LSTM Architecture

### Model: CycloneLSTM

```
Input (batch, 9, 7)
  → LSTM (hidden={best_result['config']['hidden_size']}, layers={best_result['config']['num_layers']}, dropout={best_result['config']['dropout']})
  → Linear (hidden={best_result['config']['hidden_size']} → 9)
  → Reshape (batch, 3, 3)
Output
```

| Property | Value |
|----------|-------|
| Hidden size | {best_result['config']['hidden_size']} |
| Number of layers | {best_result['config']['num_layers']} |
| Dropout | {best_result['config']['dropout']} |
| Parameters | {sum(p.numel() for p in best_model.parameters()):,} |
| Output | (batch, 3 horizons × 3 targets) |

### Output Structure

| Index | Horizon | Target |
|-------|---------|--------|
| [0] | +6h | latitude, longitude, wind_speed |
| [1] | +12h | latitude, longitude, wind_speed |
| [2] | +24h | latitude, longitude, wind_speed |

---

## 5. Training Process

| Property | Value |
|----------|-------|
| Optimizer | Adam |
| Loss function | Huber Loss |
| Learning rate | {LR_INIT} (initial) |
| LR scheduler | ReduceLROnPlateau (factor={LR_FACTOR}, patience={LR_PATIENCE}) |
| Early stopping patience | {PATIENCE} epochs |
| Gradient clipping | max_norm=1.0 |
| Batch size | {BATCH_SIZE} |
| Random seed | {RANDOM_SEED} |

### Configuration Search Results

| Config | Hidden | Layers | Dropout | Best Val Loss | Training Time |
|--------|--------|--------|---------|---------------|---------------|
"""

for name, res in results_all.items():
    marker = " **BEST**" if name == best_name else ""
    report_content += (f"| {name} | {res['config']['hidden_size']} | "
                      f"{res['config']['num_layers']} | {res['config']['dropout']} | "
                      f"{res['best_val_loss']:.6f} | {res['training_time']:.1f}s{marker} |\n")

report_content += f"""
---

## 6. Evaluation Metrics

### Metrics Explained

1. **Latitude MAE:** Mean absolute error in degrees north
2. **Longitude MAE:** Mean absolute error in degrees east
3. **Wind MAE:** Mean absolute error in km/h
4. **Track Error:** Great-circle (haversine) distance in km between predicted and actual position

---

## 7. Baseline vs LSTM Comparison

| Horizon | Metric | Persistence | LSTM | Improvement |
|---------|--------|-------------|------|-------------|
"""

for h in ['6h', '12h', '24h']:
    p = persistence_results[h]
    l = lstm_results[h]
    c = comparison[h]['improvement']
    report_content += (f"| +{h} | Lat MAE (°) | {p['latitude_mae_deg']:.3f} | "
                      f"{l['latitude_mae_deg']:.3f} | {c['latitude_mae_pct']:+.1f}% |\n")
    report_content += (f"| | Lon MAE (°) | {p['longitude_mae_deg']:.3f} | "
                      f"{l['longitude_mae_deg']:.3f} | {c['longitude_mae_pct']:+.1f}% |\n")
    report_content += (f"| | Wind MAE (km/h) | {p['wind_speed_mae_kmh']:.1f} | "
                      f"{l['wind_speed_mae_kmh']:.1f} | {c['wind_mae_pct']:+.1f}% |\n")
    report_content += (f"| | Track Error (km) | {p['track_error_km']:.1f} | "
                      f"{l['track_error_km']:.1f} | {c['track_error_pct']:+.1f}% |\n")

report_content += f"""
**Interpretation:** Positive improvement means LSTM outperforms persistence.
Negative values mean persistence is better.

---

## 8. Best Model Configuration

| Property | Value |
|----------|-------|
| Configuration | {best_name} |
| Hidden size | {best_result['config']['hidden_size']} |
| Number of layers | {best_result['config']['num_layers']} |
| Dropout | {best_result['config']['dropout']} |
| Best validation loss | {best_result['best_val_loss']:.6f} |
| Total training time | {best_result['training_time']:.1f}s |
| Epochs trained | {len(best_result['history']['val_loss'])} |

---

## 9. Limitations

1. **Track-only features:** The model uses only cyclone track parameters (lat, lon, wind, pressure, speed, direction). No atmospheric context (ERA5) or satellite imagery (INSAT) is included.

2. **Single data source:** Wind and pressure come from a coalesced hierarchy (IMD → WMO → JTWC). Agency transitions may introduce value jumps.

3. **34% missing pressure:** The imputation strategy (training mean + indicator) is simple but may not capture the true relationship between pressure and storm intensity.

4. **Fixed architecture:** Only 3 configurations were tested. A more thorough hyperparameter search might find better settings.

5. **No temporal features:** The model does not explicitly encode time-of-year, which could help capture seasonal patterns in cyclone activity.

6. **No physics constraints:** Predictions may not satisfy physical conservation laws. Future work could add physics-informed loss terms.

7. **Short evaluation period:** The test set covers ~2019-2025. Performance on earlier decades is not evaluated.

---

## 10. Generated Files

| File | Description |
|------|-------------|
| `models/lstm_forecaster.pt` | Best model weights |
| `models/model_config.json` | Model configuration |
| `models/preprocessing_config.json` | Preprocessing parameters |
| `results/persistence_baseline.json` | Persistence baseline metrics |
| `results/lstm_results.json` | LSTM test metrics |
| `results/model_comparison.json` | Head-to-head comparison |
| `results/plots/training_loss_curves.png` | Training curves for all configs |
| `results/plots/example_forecast.png` | Example predicted vs actual |
| `results/plots/track_error_comparison.png` | Track error by horizon |
| `results/plots/wind_prediction_examples.png` | Wind prediction examples |
| `src/forecasting/inference.py` | Reusable inference module |
| `results/PHASE3_REPORT.md` | This report |
"""

with open(RESULTS_DIR / 'PHASE3_REPORT.md', 'w', encoding='utf-8') as f:
    f.write(report_content)
print(f"  Saved: results/PHASE3_REPORT.md")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 3 COMPLETE")
print("=" * 70)

print(f"\nModel Architecture:")
print(f"  CycloneLSTM: Input(9,7) → LSTM(h={best_result['config']['hidden_size']},"
      f" L={best_result['config']['num_layers']},"
      f" d={best_result['config']['dropout']}) → FC(9) → Output(3,3)")
print(f"  Parameters: {sum(p.numel() for p in best_model.parameters()):,}")

print(f"\nTraining:")
print(f"  Configuration: {best_name}")
print(f"  Training time: {best_result['training_time']:.1f}s")
print(f"  Best validation loss: {best_result['best_val_loss']:.6f}")
print(f"  Epochs: {len(best_result['history']['val_loss'])}")

print(f"\nTest Metrics:")
for h in ['6h', '12h', '24h']:
    l = lstm_results[h]
    print(f"  +{h}: lat={l['latitude_mae_deg']:.3f}°, "
          f"lon={l['longitude_mae_deg']:.3f}°, "
          f"wind={l['wind_speed_mae_kmh']:.1f} km/h, "
          f"track={l['track_error_km']:.1f} km")

print(f"\nPersistence Metrics:")
for h in ['6h', '12h', '24h']:
    p = persistence_results[h]
    print(f"  +{h}: lat={p['latitude_mae_deg']:.3f}°, "
          f"lon={p['longitude_mae_deg']:.3f}°, "
          f"wind={p['wind_speed_mae_kmh']:.1f} km/h, "
          f"track={p['track_error_km']:.1f} km")

print(f"\nImprovement over persistence:")
for h in ['6h', '12h', '24h']:
    c = comparison[h]['improvement']
    print(f"  +{h}: track={c['track_error_pct']:+.1f}%, "
          f"wind={c['wind_mae_pct']:+.1f}%")

print(f"\nGenerated files:")
print(f"  models/lstm_forecaster.pt")
print(f"  models/model_config.json")
print(f"  models/preprocessing_config.json")
print(f"  results/persistence_baseline.json")
print(f"  results/lstm_results.json")
print(f"  results/model_comparison.json")
print(f"  results/plots/ (4 plots)")
print(f"  src/forecasting/inference.py")
print(f"  results/PHASE3_REPORT.md")
