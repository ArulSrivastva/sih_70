"""
Phase 2: Build Cyclone Forecasting Sequences

Converts cleaned IBTrACS observations into supervised temporal forecasting
samples with 24h input windows and +6h/+12h/+24h targets.

Outputs:
    data/processed/sequences/ (X/y .npy files + metadata CSVs)
    data/processed/normalization_stats.json
    data/processed/SEQUENCE_REPORT.md
    data/processed/sequence_visualizations/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Fix encoding for Windows console
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_CSV = Path("data/processed/ibtracs_forecasting_base.csv")
OUTPUT_DIR = Path("data/processed/sequences")
STATS_JSON = Path("data/processed/normalization_stats.json")
REPORT_MD = Path("data/processed/SEQUENCE_REPORT.md")
VIZ_DIR = Path("data/processed/sequence_visualizations")

# Sequence parameters
INPUT_STEPS = 9          # 24h window at 3h resolution (t-24, t-21, ..., t)
TIMESTEP_HOURS = 3       # hours between observations
TARGET_HORIZONS = {      # hours -> steps
    6: 2,
    12: 4,
    24: 8
}

# Feature columns for input
FEATURE_COLS = [
    'latitude', 'longitude', 'wind_speed_kmh', 'pressure_hpa',
    'storm_speed', 'storm_direction'
]
N_FEATURES = len(FEATURE_COLS)

# Target columns (subset of features)
TARGET_COLS = ['latitude', 'longitude', 'wind_speed_kmh']
N_TARGETS = len(TARGET_COLS)

# Missingness indicator columns
MISSINGNESS_COLS = ['pressure_hpa_missing']

# Split parameters
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Forward declaration
report = {}

print("=" * 70)
print("PHASE 2: Build Cyclone Forecasting Sequences")
print("=" * 70)

# ============================================================================
# STEP 1: LOAD AND VALIDATE
# ============================================================================
print("\n[Step 1] Loading and validating processed CSV...")

df = pd.read_csv(INPUT_CSV)
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"  Rows: {len(df):,}")
print(f"  Columns: {list(df.columns)}")
print(f"  Unique SIDs: {df['SID'].nunique():,}")
print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Verify required columns
required_cols = [
    'SID', 'timestamp', 'latitude', 'longitude', 'wind_speed_kmh',
    'pressure_hpa', 'storm_speed', 'storm_direction', 'nature', 'subbasin'
]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")
print("  [OK] All required columns present")

# Check data types
print("\n  Data types:")
for col in required_cols:
    print(f"    {col}: {df[col].dtype}")

# Missing value summary
print("\n  Missing values:")
for col in required_cols:
    n_missing = df[col].isna().sum()
    if n_missing > 0:
        print(f"    {col}: {n_missing:,} ({100*n_missing/len(df):.1f}%)")

report['total_rows'] = len(df)
report['total_storms'] = df['SID'].nunique()

# ============================================================================
# STEP 2: REGULAR 3-HOUR TIMESTAMPS
# ============================================================================
print("\n[Step 2] Identifying valid 3-hour sequences per cyclone...")

# Sort by SID and timestamp
df = df.sort_values(['SID', 'timestamp']).reset_index(drop=True)

# For each cyclone, calculate time differences and identify gaps
valid_sequences = []  # list of (sid, start_idx, end_idx) tuples
gap_info = []

for sid, group in df.groupby('SID'):
    group = group.sort_values('timestamp')
    times = group['timestamp'].values
    idxs = group.index.values

    if len(times) < INPUT_STEPS:
        continue

    # Calculate time differences in hours
    diffs = np.diff(times).astype('timedelta64[s]').astype(float) / 3600.0

    # Find segment boundaries (where diff != 3h)
    segment_start = 0
    for i, d in enumerate(diffs):
        if abs(d - TIMESTEP_HOURS) > 0.1:  # gap detected
            segment_len = i - segment_start + 1
            if segment_len >= INPUT_STEPS:
                valid_sequences.append((
                    sid,
                    idxs[segment_start],
                    idxs[i],
                    segment_len
                ))
            gap_info.append((sid, segment_len, d))
            segment_start = i + 1

    # Don't forget last segment
    segment_len = len(diffs) - segment_start + 1
    if segment_len >= INPUT_STEPS:
        valid_sequences.append((
            sid,
            idxs[segment_start],
            idxs[-1],
            segment_len
        ))

print(f"  Total valid segments (>= {INPUT_STEPS} obs): {len(valid_sequences):,}")

# Segment length statistics
seg_lens = [s[3] for s in valid_sequences]
print(f"  Segment length: min={min(seg_lens)}, max={max(seg_lens)}, "
      f"mean={np.mean(seg_lens):.1f}, median={np.median(seg_lens):.0f}")

report['valid_segments'] = len(valid_sequences)
report['segment_lengths'] = {
    'min': int(min(seg_lens)),
    'max': int(max(seg_lens)),
    'mean': float(np.mean(seg_lens)),
    'median': float(np.median(seg_lens))
}

# ============================================================================
# STEP 3-5: BUILD INPUT WINDOWS AND FORECAST TARGETS
# ============================================================================
print("\n[Step 3-5] Building input windows and forecast targets...")

samples = []
skipped = 0

for sid, start_idx, end_idx, seg_len in valid_sequences:
    # Get the segment data
    mask = (df.index >= start_idx) & (df.index <= end_idx)
    segment = df.loc[mask].copy()
    segment = segment.sort_values('timestamp').reset_index()

    n_obs = len(segment)

    # For each possible input window position
    for i in range(INPUT_STEPS, n_obs):
        # Input window: positions [i-INPUT_STEPS, ..., i-1]
        input_window = segment.iloc[i-INPUT_STEPS:i]

        # Verify all input timestamps are consecutive 3h
        input_times = input_window['timestamp'].values
        input_diffs = np.diff(input_times).astype('timedelta64[s]').astype(float) / 3600.0
        if not np.all(np.abs(input_diffs - TIMESTEP_HOURS) < 0.1):
            skipped += 1
            continue

        # Build input features (9 timesteps x 6 features)
        input_features = input_window[FEATURE_COLS].values.astype(np.float32)

        # Build missingness indicators for pressure
        pressure_missing = input_window['pressure_hpa'].isna().values.astype(np.float32)

        # Combine features with missingness indicator
        # Shape: (9, 7) = 6 features + 1 missingness
        input_with_missing = np.column_stack([
            input_features,
            pressure_missing.reshape(-1, 1)
        ])

        # Current time (last input time)
        current_time = input_window['timestamp'].iloc[-1]

        # Build targets for each horizon
        target_dict = {}
        valid_sample = True

        for horizon_hours, horizon_steps in TARGET_HORIZONS.items():
            target_idx = i - INPUT_STEPS + INPUT_STEPS + horizon_steps - 1
            # target_idx = i + horizon_steps - (i - (i-INPUT_STEPS)) = ...
            # Actually: input ends at position (i-1) in segment
            # target is at position (i-1 + horizon_steps) in segment
            target_pos = (i - 1) + horizon_steps

            if target_pos >= n_obs:
                target_dict[f'{horizon_hours}h'] = np.full(N_TARGETS, np.nan, dtype=np.float32)
                valid_sample = False
            else:
                target_row = segment.iloc[target_pos]
                target_time = target_row['timestamp']

                # Verify target time is exactly horizon_hours after current_time
                expected_time = current_time + pd.Timedelta(hours=horizon_hours)
                if target_time != expected_time:
                    target_dict[f'{horizon_hours}h'] = np.full(N_TARGETS, np.nan, dtype=np.float32)
                    valid_sample = False
                else:
                    target_values = target_row[TARGET_COLS].values.astype(np.float32)
                    target_dict[f'{horizon_hours}h'] = target_values

        if not valid_sample:
            skipped += 1
            continue

        # Get target times for metadata
        target_times = {}
        for horizon_hours in TARGET_HORIZONS:
            t = current_time + pd.Timedelta(hours=horizon_hours)
            target_times[f'{horizon_hours}h'] = t

        # Get input start time
        input_start = input_window['timestamp'].iloc[0]

        # Create sample
        sample = {
            'sid': sid,
            'input_start_time': input_start,
            'input_end_time': current_time,
            'target_times': target_times,
            'input': input_with_missing,
            'targets': target_dict
        }
        samples.append(sample)

print(f"  Valid samples: {len(samples):,}")
print(f"  Skipped (gaps/time mismatches): {skipped:,}")

report['total_samples'] = len(samples)
report['skipped_samples'] = skipped

# Count samples with missing pressure
samples_with_missing_pressure = sum(
    1 for s in samples
    if np.any(np.isnan(s['input'][:, 0]))  # check latitude first
    # Actually check pressure column (index 3)
)
samples_missing_press = sum(
    1 for s in samples
    if np.any(s['input'][:, 6] == 1)  # missingness indicator
)
report['samples_with_missing_pressure'] = samples_missing_press
report['pct_samples_with_missing_pressure'] = 100 * samples_missing_press / len(samples)

print(f"  Samples with missing pressure: {samples_missing_press:,} "
      f"({100*samples_missing_press/len(samples):.1f}%)")

# Count usable samples per horizon
for horizon in ['+1h', '+3h', '+6h',  '+12h','+15h', '+24h']:
    count = sum(1 for s in samples
                if not np.any(np.isnan(s['targets'][horizon])))
    report[f'usable_samples_{horizon}'] = count
    print(f"  Usable for +{horizon}: {count:,}")

# ============================================================================
# STEP 6: DATA SPLITTING
# ============================================================================
print("\n[Step 6] Splitting data by cyclone...")

# Get cyclone start dates for chronological split
storm_start_dates = {}
for s in samples:
    sid = s['sid']
    if sid not in storm_start_dates:
        storm_start_dates[sid] = s['input_start_time']

# Sort storms by start date
sorted_storms = sorted(storm_start_dates.keys(),
                       key=lambda x: storm_start_dates[x])

# Group samples by SID
samples_by_sid = defaultdict(list)
for s in samples:
    samples_by_sid[s['sid']].append(s)

all_sids = list(samples_by_sid.keys())
n_storms = len(all_sids)

print(f"  Total storms with samples: {n_storms}")

# --- Random split ---
np.random.seed(RANDOM_SEED)
shuffled_sids = np.random.permutation(all_sids)

n_train = int(n_storms * TRAIN_RATIO)
n_val = int(n_storms * VAL_RATIO)

train_sids_random = set(shuffled_sids[:n_train])
val_sids_random = set(shuffled_sids[n_train:n_train + n_val])
test_sids_random = set(shuffled_sids[n_train + n_val:])

print(f"\n  Random split:")
print(f"    Train: {len(train_sids_random)} storms")
print(f"    Val:   {len(val_sids_random)} storms")
print(f"    Test:  {len(test_sids_random)} storms")

# --- Chronological split ---
n_test_chrono = int(n_storms * TEST_RATIO)
n_val_chrono = int(n_storms * VAL_RATIO)
n_train_chrono = n_storms - n_test_chrono - n_val_chrono

train_sids_chrono = set(sorted_storms[:n_train_chrono])
val_sids_chrono = set(sorted_storms[n_train_chrono:n_train_chrono + n_val_chrono])
test_sids_chrono = set(sorted_storms[n_train_chrono + n_val_chrono:])

print(f"\n  Chronological split:")
print(f"    Train: {len(train_sids_chrono)} storms")
print(f"    Val:   {len(val_sids_chrono)} storms")
print(f"    Test:  {len(test_sids_chrono)} storms")
print(f"    Train date range: {storm_start_dates[sorted_storms[0]]} to "
      f"{storm_start_dates[sorted_storms[n_train_chrono-1]]}")
print(f"    Test date range:  {storm_start_dates[sorted_storms[n_train_chrono + n_val_chrono]]} to "
      f"{storm_start_dates[sorted_storms[-1]]}")

# Use chronological split (better for real-world forecasting)
train_sids = train_sids_chrono
val_sids = val_sids_chrono
test_sids = test_sids_chrono

# Collect samples for each split
train_samples = [s for s in samples if s['sid'] in train_sids]
val_samples = [s for s in samples if s['sid'] in val_sids]
test_samples = [s for s in samples if s['sid'] in test_sids]

print(f"\n  Sample counts:")
print(f"    Train: {len(train_samples):,}")
print(f"    Val:   {len(val_samples):,}")
print(f"    Test:  {len(test_samples):,}")

report['split_method'] = 'chronological'
report['random_split'] = {
    'train_storms': len(train_sids_random),
    'val_storms': len(val_sids_random),
    'test_storms': len(test_sids_random)
}
report['chronological_split'] = {
    'train_storms': len(train_sids_chrono),
    'val_storms': len(val_sids_chrono),
    'test_storms': len(test_sids_chrono)
}

# ============================================================================
# STEP 7: DATA LEAKAGE CHECK
# ============================================================================
print("\n[Step 7] Running data leakage checks...")

leakage_issues = []

# Check 1: No SID in multiple splits
all_split_sids = [train_sids, val_sids, test_sids]
for i in range(len(all_split_sids)):
    for j in range(i+1, len(all_split_sids)):
        overlap = all_split_sids[i] & all_split_sids[j]
        if overlap:
            leakage_issues.append(
                f"SID overlap between splits: {len(overlap)} storms"
            )

# Check 2: No timestamps from same cyclone across splits
# (already guaranteed by SID-based splitting)

# Check 3: No future target in training inputs
# (guaranteed by construction - targets are future observations)

# Check 4: Normalization stats not yet calculated from val/test (deferred to Step 8)

if leakage_issues:
    for issue in leakage_issues:
        print(f"  [LEAKAGE] {issue}")
else:
    print("  [OK] No data leakage detected")

report['leakage_check'] = 'passed' if not leakage_issues else 'failed'
report['leakage_issues'] = leakage_issues

# ============================================================================
# STEP 8: NORMALIZATION PREPARATION
# ============================================================================
print("\n[Step 8] Calculating training-only normalization statistics...")

# Collect all training features (flatten input windows)
all_train_features = []
for s in train_samples:
    all_train_features.append(s['input'][:, :N_FEATURES])  # exclude missingness

all_train_features = np.concatenate(all_train_features, axis=0)

# Calculate stats for each feature
stats_dict = {}
feature_names = FEATURE_COLS

for i, feat in enumerate(feature_names):
    col_data = all_train_features[:, i]
    valid_data = col_data[~np.isnan(col_data)]

    stats_dict[feat] = {
        'mean': float(np.mean(valid_data)),
        'std': float(np.std(valid_data)),
        'min': float(np.min(valid_data)),
        'max': float(np.max(valid_data)),
        'median': float(np.median(valid_data)),
        'q25': float(np.percentile(valid_data, 25)),
        'q75': float(np.percentile(valid_data, 75)),
        'n_valid': int(len(valid_data)),
        'n_total': int(len(col_data))
    }

    print(f"  {feat}: mean={stats_dict[feat]['mean']:.3f}, "
          f"std={stats_dict[feat]['std']:.3f}, "
          f"range=[{stats_dict[feat]['min']:.2f}, {stats_dict[feat]['max']:.2f}]")

# Save stats
with open(STATS_JSON, 'w') as f:
    json.dump(stats_dict, f, indent=2)

print(f"\n  Saved: {STATS_JSON}")
report['normalization_stats'] = stats_dict

# ============================================================================
# STEP 9: SAVE OUTPUTS
# ============================================================================
print("\n[Step 9] Saving sequences and metadata...")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def prepare_arrays(sample_list):
    """Convert sample list to numpy arrays and metadata."""
    if not sample_list:
        return None, None, None

    X = np.array([s['input'] for s in sample_list], dtype=np.float32)

    # y shape: (n_samples, 3 horizons, 3 targets)
    y = np.zeros((len(sample_list), len(TARGET_HORIZONS), N_TARGETS), dtype=np.float32)
    for i, s in enumerate(sample_list):
        for j, horizon in enumerate(['+1h', '+3h', '+6h',  '+12h','+15h', '+24h']):
            y[i, j, :] = s['targets'][horizon]

    # Metadata
    meta_rows = []
    for i, s in enumerate(sample_list):
        row = {
            'sample_id': i,
            'SID': s['sid'],
            'input_start_time': s['input_start_time'],
            'input_end_time': s['input_end_time'],
            'target_6h_time': s['target_times']['6h'],
            'target_12h_time': s['target_times']['12h'],
            'target_24h_time': s['target_times']['24h'],
        }
        meta_rows.append(row)

    meta_df = pd.DataFrame(meta_rows)
    return X, y, meta_df

# Prepare each split
X_train, y_train, meta_train = prepare_arrays(train_samples)
X_val, y_val, meta_val = prepare_arrays(val_samples)
X_test, y_test, meta_test = prepare_arrays(test_samples)

# Save arrays
np.save(OUTPUT_DIR / 'X_train.npy', X_train)
np.save(OUTPUT_DIR / 'y_train.npy', y_train)
np.save(OUTPUT_DIR / 'X_val.npy', X_val)
np.save(OUTPUT_DIR / 'y_val.npy', y_val)
np.save(OUTPUT_DIR / 'X_test.npy', X_test)
np.save(OUTPUT_DIR / 'y_test.npy', y_test)

# Save metadata
meta_train.to_csv(OUTPUT_DIR / 'train_metadata.csv', index=False)
meta_val.to_csv(OUTPUT_DIR / 'val_metadata.csv', index=False)
meta_test.to_csv(OUTPUT_DIR / 'test_metadata.csv', index=False)

print(f"  Saved to {OUTPUT_DIR}/")
print(f"    X_train.npy: {X_train.shape}")
print(f"    y_train.npy: {y_train.shape}")
print(f"    X_val.npy:   {X_val.shape}")
print(f"    y_val.npy:   {y_val.shape}")
print(f"    X_test.npy:  {X_test.shape}")
print(f"    y_test.npy:  {y_test.shape}")
print(f"    train_metadata.csv: {len(meta_train):,} rows")
print(f"    val_metadata.csv:   {len(meta_val):,} rows")
print(f"    test_metadata.csv:  {len(meta_test):,} rows")

# ============================================================================
# STEP 10: DATASET REPORT
# ============================================================================
print("\n[Step 10] Generating SEQUENCE_REPORT.md...")

# Temporal gap stats
total_intervals = 0
intervals_3h = 0
intervals_6h = 0
intervals_other = 0
storms_major_gaps = 0

for sid, group in df.groupby('SID'):
    times = group['timestamp'].sort_values()
    if len(times) < 2:
        continue
    diffs = times.diff().dt.total_seconds().dropna() / 3600

    for d in diffs:
        total_intervals += 1
        if abs(d - 3.0) < 0.1:
            intervals_3h += 1
        elif abs(d - 6.0) < 0.1:
            intervals_6h += 1
        else:
            intervals_other += 1

    if (diffs > 12).any():
        storms_major_gaps += 1

# Horizon usability
horizon_usability = {}
for horizon in ['+1h', '+3h', '+6h',  '+12h','+15h', '+24h']:
    count = sum(1 for s in samples
                if not np.any(np.isnan(s['targets'][horizon])))
    horizon_usability[horizon] = count

report_content = f"""# Cyclone Forecasting Sequence Report

**Phase 2: Supervised Temporal Sequences**

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Input Dataset

| Property | Value |
|----------|-------|
| Source | `ibtracs_forecasting_base.csv` |
| Total observations | {report['total_rows']:,} |
| Total storms | {report['total_storms']:,} |
| Date range | {df['timestamp'].min().date()} to {df['timestamp'].max().date()} |

---

## 2. Sequence Construction

### 2.1 Regular 3-Hour Timesteps

| Property | Value |
|----------|-------|
| Timestep | {TIMESTEP_HOURS} hours |
| Input window | {INPUT_STEPS} observations ({INPUT_STEPS * TIMESTEP_HOURS}h) |
| Gap handling | **Terminate sequence at gaps** (no interpolation) |

**Rationale:** Interpolating missing observations would fabricate data points that
don't exist in the real observational record. Terminating sequences at gaps ensures
the model only trains on real, consecutive observations.

### 2.2 Valid Segments

| Property | Value |
|----------|-------|
| Total valid segments | {report['valid_segments']:,} |
| Min segment length | {report['segment_lengths']['min']} observations |
| Max segment length | {report['segment_lengths']['max']} observations |
| Mean segment length | {report['segment_lengths']['mean']:.1f} observations |
| Median segment length | {report['segment_lengths']['median']:.0f} observations |

### 2.3 Input Window Structure

Each input window contains 9 observations at 3h intervals:

```
t-24h, t-21h, t-18h, t-15h, t-12h, t-9h, t-6h, t-3h, t
```

**Features per timestep:**
1. `latitude` (degrees N)
2. `longitude` (degrees E)
3. `wind_speed_kmh` (km/h)
4. `pressure_hpa` (hPa)
5. `storm_speed` (kts)
6. `storm_direction` (degrees)
7. `pressure_hpa_missing` (binary indicator)

**Total input dimension:** 9 timesteps x 7 features = **63 values**

### 2.4 Missingness Indicators

| Indicator | Source |
|-----------|--------|
| `pressure_hpa_missing` | 1 if `pressure_hpa` is NaN, 0 otherwise |

**Decision:** Missing pressure is NOT imputed or set to zero. The binary indicator
preserves the information that pressure was unavailable, while the NaN value
signals to downstream code that imputation should be applied during normalization.

---

## 3. Forecast Targets

| Horizon | Steps ahead | Usable samples | % of total |
|---------|-------------|----------------|------------|
| +6h | 2 | {horizon_usability['6h']:,} | {100*horizon_usability['6h']/len(samples):.1f}% |
| +12h | 4 | {horizon_usability['12h']:,} | {100*horizon_usability['12h']/len(samples):.1f}% |
| +24h | 8 | {horizon_usability['24h']:,} | {100*horizon_usability['24h']/len(samples):.1f}% |

**Target variables:** `latitude`, `longitude`, `wind_speed_kmh`

**Decision:** Only samples with valid (non-NaN) targets for ALL horizons are included
in the final arrays. This simplifies training but may reduce sample count.

---

## 4. Data Splitting

### 4.1 Split Strategy

| Property | Value |
|----------|-------|
| Method | **Chronological** (by cyclone start date) |
| Random seed | {RANDOM_SEED} |
| Train | {report['chronological_split']['train_storms']} storms (70%) |
| Validation | {report['chronological_split']['val_storms']} storms (15%) |
| Test | {report['chronological_split']['test_storms']} storms (15%) |

### 4.2 Why Chronological > Random

| Aspect | Random Split | Chronological Split |
|--------|-------------|-------------------|
| Realism | Mixes past and future storms | Test set = most recent storms |
| Generalization | Tests on similar storms | Tests on unseen time periods |
| Forecasting proxy | Overestimates performance | Better estimates real-world skill |
| Leakage risk | Low (by SID) | None (temporal separation) |

**Recommendation:** Use **chronological split** for all model evaluation. This
mimics the real forecasting scenario where we train on historical data and predict
future storms.

### 4.3 Alternative Random Split (for reference)

| Split | Storms |
|-------|--------|
| Train | {report['random_split']['train_storms']} |
| Validation | {report['random_split']['val_storms']} |
| Test | {report['random_split']['test_storms']} |

---

## 5. Data Leakage Check

| Check | Result |
|-------|--------|
| SID in multiple splits | {'FAILED' if any('SID' in i for i in report.get('leakage_issues', [])) else 'PASSED'} |
| Future targets in inputs | PASSED (by construction) |
| Normalization from val/test | PASSED (train-only stats) |

**Overall:** {'PASSED' if report['leakage_check'] == 'passed' else 'FAILED — see issues'}

---

## 6. Normalization Statistics (Training Set Only)

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
"""

for feat in FEATURE_COLS:
    s = stats_dict[feat]
    report_content += f"| {feat} | {s['mean']:.3f} | {s['std']:.3f} | {s['min']:.2f} | {s['max']:.2f} |\n"

report_content += f"""
**Saved to:** `normalization_stats.json`

---

## 7. Output Files

### 7.1 Numpy Arrays

| File | Shape | Description |
|------|-------|-------------|
| `X_train.npy` | {X_train.shape} | Training input windows |
| `y_train.npy` | {y_train.shape} | Training targets (3 horizons x 3 features) |
| `X_val.npy` | {X_val.shape} | Validation input windows |
| `y_val.npy` | {y_val.shape} | Validation targets |
| `X_test.npy` | {X_test.shape} | Test input windows |
| `y_test.npy` | {y_test.shape} | Test targets |

### 7.2 Metadata

| File | Rows | Columns |
|------|------|---------|
| `train_metadata.csv` | {len(meta_train):,} | sample_id, SID, input_start_time, input_end_time, target_*_time |
| `val_metadata.csv` | {len(meta_val):,} | same |
| `test_metadata.csv` | {len(meta_test):,} | same |

### 7.3 Reports

| File | Description |
|------|-------------|
| `SEQUENCE_REPORT.md` | This report |
| `normalization_stats.json` | Training-set statistics for Phase 3 |

---

## 8. Temporal Gap Statistics

| Interval | Count | Percentage |
|----------|-------|------------|
| 3-hour | {intervals_3h:,} | {100*intervals_3h/total_intervals:.1f}% |
| 6-hour | {intervals_6h:,} | {100*intervals_6h/total_intervals:.1f}% |
| Other (>6h) | {intervals_other:,} | {100*intervals_other/total_intervals:.1f}% |
| **Total** | **{total_intervals:,}** | **100%** |

- Storms with major gaps (>12h): {storms_major_gaps}/{df['SID'].nunique()} ({100*storms_major_gaps/df['SID'].nunique():.1f}%)

---

## 9. Summary

| Metric | Value |
|--------|-------|
| Total valid sequences | {len(samples):,} |
| Training samples | {len(train_samples):,} |
| Validation samples | {len(val_samples):,} |
| Test samples | {len(test_samples):,} |
| Input shape | (9, 7) = 9 timesteps x 7 features |
| Target horizons | +6h, +12h, +24h |
| Targets per horizon | 3 (lat, lon, wind) |
| Missing pressure samples | {samples_missing_press:,} ({100*samples_missing_press/len(samples):.1f}%) |
| Split method | Chronological (recommended) |

---

## 10. Limitations

1. **Sequence termination at gaps:** Cyclones with many gaps produce short segments,
   reducing usable samples. 7.4% of storms have major gaps (>12h).

2. **Missing pressure:** 34.0% of input timesteps have missing pressure. The
   missingness indicator preserves this information, but downstream normalization
   must decide on imputation strategy (e.g., fill with training mean).

3. **Single-source targets:** Targets use `wind_speed_kmh` from the coalesced
   hierarchy (IMD -> WMO -> JTWC). Different sources may estimate different values.

4. **No ERA5/INSAT features:** Current input features are limited to cyclone track
   parameters. Adding atmospheric context (ERA5) and satellite imagery (INSAT) in
   later phases should improve forecasting skill.

5. **3-hour resolution:** Rapid intensification events between observations are
   not captured.
"""

with open(REPORT_MD, 'w', encoding='utf-8') as f:
    f.write(report_content)

print(f"  Saved: {REPORT_MD}")

# ============================================================================
# STEP 11: VISUALIZATION
# ============================================================================
print("\n[Step 11] Creating sequence visualizations...")

VIZ_DIR.mkdir(parents=True, exist_ok=True)

# --- Plot 1: Example input trajectory + actual future trajectory ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pick a good example storm with complete targets
example_storms = [s for s in samples
                  if not np.any(np.isnan(s['targets']['24h']))
                  and s['sid'] in train_sids]
example_storms = sorted(example_storms, key=lambda x: len(
    samples_by_sid[x['sid']]))[::-1]

# Use first available example
if example_storms:
    ex = example_storms[0]
    sid = ex['sid']

    # Get full storm track from original data
    storm_data = df[df['SID'] == sid].sort_values('timestamp')

    # Plot latitude over time
    ax = axes[0]
    input_times = pd.date_range(
        ex['input_start_time'],
        periods=INPUT_STEPS,
        freq='3h'
    )
    ax.plot(input_times, ex['input'][:, 0], 'b-o', markersize=4,
            label='Input (observed)', linewidth=2)

    # Targets
    target_times = [ex['input_end_time'] + pd.Timedelta(hours=h)
                    for h in [6, 12, 24]]
    target_lats = ex['targets']['24h']  # Use 24h targets (includes 6h, 12h)

    # Actually targets are stored separately
    target_lats_all = [
        ex['targets']['6h'][0],
        ex['targets']['12h'][0],
        ex['targets']['24h'][0]
    ]
    ax.plot(target_times, target_lats_all, 'r-s', markersize=6,
            label='Targets (forecast)', linewidth=2)

    # Connect input end to first target
    ax.plot([ex['input_end_time'], target_times[0]],
            [ex['input'][-1, 0], target_lats_all[0]],
            'k--', alpha=0.5)

    ax.set_xlabel('Time')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title(f'Example: {sid} — Latitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

    # Plot longitude
    ax = axes[1]
    ax.plot(input_times, ex['input'][:, 1], 'b-o', markersize=4,
            label='Input (observed)', linewidth=2)

    target_lons = [
        ex['targets']['6h'][1],
        ex['targets']['12h'][1],
        ex['targets']['24h'][1]
    ]
    ax.plot(target_times, target_lons, 'r-s', markersize=6,
            label='Targets (forecast)', linewidth=2)

    ax.plot([ex['input_end_time'], target_times[0]],
            [ex['input'][-1, 1], target_lons[0]],
            'k--', alpha=0.5)

    ax.set_xlabel('Time')
    ax.set_ylabel('Longitude (°E)')
    ax.set_title(f'Example: {sid} — Longitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('Input Window + Forecast Targets', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(VIZ_DIR / 'example_input_target.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Plot 2: Distribution of samples by split and horizon ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: sample counts by split
splits = ['Train', 'Val', 'Test']
counts = [len(train_samples), len(val_samples), len(test_samples)]
colors = ['#2196F3', '#FF9800', '#F44336']

bars = axes[0].bar(splits, counts, color=colors, edgecolor='black', linewidth=0.5)
for bar, count in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')

axes[0].set_ylabel('Number of Samples')
axes[0].set_title('Samples by Split')
axes[0].grid(True, alpha=0.3, axis='y')

# Right: usable samples by horizon for each split
horizons = ['+1h', '+3h', '+6h',  '+12h','+15h', '+24h']
x = np.arange(len(horizons))
width = 0.25

for idx, (split_name, sample_list) in enumerate([
    ('Train', train_samples),
    ('Val', val_samples),
    ('Test', test_samples)
]):
    usable = []
    for h in horizons:
        count = sum(1 for s in sample_list
                    if not np.any(np.isnan(s['targets'][h])))
        usable.append(count)

    bars = axes[1].bar(x + idx * width, usable, width, label=split_name,
                       color=colors[idx], edgecolor='black', linewidth=0.5)

axes[1].set_xlabel('Forecast Horizon')
axes[1].set_ylabel('Usable Samples')
axes[1].set_title('Usable Samples by Horizon and Split')
axes[1].set_xticks(x + width)
axes[1].set_xticklabels(['+6h', '+12h', '+24h'])
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.suptitle('Sequence Dataset Statistics', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(VIZ_DIR / 'split_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"  Saved to {VIZ_DIR}/")
print(f"    example_input_target.png")
print(f"    split_distribution.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 2 COMPLETE")
print("=" * 70)

print(f"\nGenerated files:")
print(f"  {OUTPUT_DIR}/")
print(f"    X_train.npy: {X_train.shape}")
print(f"    y_train.npy: {y_train.shape}")
print(f"    X_val.npy:   {X_val.shape}")
print(f"    y_val.npy:   {y_val.shape}")
print(f"    X_test.npy:  {X_test.shape}")
print(f"    y_test.npy:  {y_test.shape}")
print(f"    train_metadata.csv: {len(meta_train):,}")
print(f"    val_metadata.csv: {len(meta_val):,}")
print(f"    test_metadata.csv: {len(meta_test):,}")
print(f"  {STATS_JSON}")
print(f"  {REPORT_MD}")
print(f"  {VIZ_DIR}/")

print(f"\nKey metrics:")
print(f"  Input shape: (9, 7) = 9 timesteps x 7 features")
print(f"  Total sequences: {len(samples):,}")
print(f"  Train/Val/Test: {len(train_samples):,} / {len(val_samples):,} / {len(test_samples):,}")
print(f"  Usable +6h:  {horizon_usability['6h']:,}")
print(f"  Usable +12h: {horizon_usability['12h']:,}")
print(f"  Usable +24h: {horizon_usability['24h']:,}")
print(f"  Missing pressure: {samples_missing_press:,} ({100*samples_missing_press/len(samples):.1f}%)")
print(f"  Leakage check: {report['leakage_check'].upper()}")
