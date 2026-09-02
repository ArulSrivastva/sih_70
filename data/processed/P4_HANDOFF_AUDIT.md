# P4 Data Handoff Audit

**Audit of Person 1 → Person 4 (Cyclone Forecasting) handoff · SIH 2026 PS 26070**
**Date:** 2026-08-29 · **Mode:** READ-ONLY (no file modified, no model trained, no sequences built, nothing downloaded). Inspection used header/sampled reads on NetCDF; full reads on CSVs/npz.
**Delivery under audit:** `PS70-main.zip` (1,182,652,292 B), extracted read-only to `%TEMP%\opencode\PS70_extracted\PS70-main`.

---

## 1. Project data inventory

All forecasting-relevant files found (recursive), with verified attributes:

| File | Path | Type | Size | Rows / dims | Key columns | Time range | Spatial | Missing % | Raw/Proc |
|---|---|---|---|---|---|---|---|---|---|
| `ibtracs_NI_raw.csv` | `data/raw/ibtracs/` | CSV | 27,875,881 B | 62,849 (incl. units row) / 174 cols | SID, ISO_TIME, LAT, LON, WMO/ NEWDELHI wind+pres, NATURE,… | 1842–2025 | global NIO basin | n/a | raw |
| `era5_YYYY_MM.nc` (94 files) | `data/raw/era5/` | NetCDF4 (HDF5) | 1,103,681,364 B total | 141×221 grid × ≤744 t | `sst, msl, u10, v10` + coords | 2013-05-09→2025-12-03 | 30…−5°N × 50…105°E (0.25°) | sst 36.3%, others 0% | raw |
| INSAT images + `insat_3d_ds - Sheet.csv` | `data/raw/insat/` (=`insat_kaggle/`) | image+jpg/png+CSV | ~47 MB | 133–143 img / 136-row csv | img_name, label | none | none | — | raw (duplicated dirs) |
| `ibtracs_clean.csv` | `data/metadata/` | CSV | — | 18,168 / 12 cols | cyclone_id, timestamp, latitude, longitude, wind_speed_kmh, pressure_hpa, category, nature,… | 1980-10-10→2025-12-02 | 0.7–31°N, 41.8–163.7°E | wind 44.7%, pres 47.4% | processed (cleaned) |
| `ibtracs_with_era5.csv` | `data/metadata/` | CSV | — | 5,481 / 14 cols | cyclone_id, timestamp, latitude, longitude, wind_speed_kmh, pressure_hpa, category, u_wind, v_wind, sst_celsius, pressure_msl_hpa | 2013-05-09→2025-12-02 | 1.9–29.2°N, 41.8–141°E | sst 28.1%, wind 23.2%, pres 22.2%, u/v/msl 0% | processed (synchronized) |
| `master_dataset.csv` | `data/metadata/` and `data/processed/` (2 identical, md5 `6b3e93e8…`) | CSV | 738,948 B | 5,481 / 15 cols | cyclone_id, season, name, subbasin, timestamp, lat, lon, wind_speed, pressure, category, sst, wind_u, wind_v, pressure_msl, pre_genesis_favorable | 2013-05-09→2025-12-02 | 1.9–29.2°N, 41.8–141°E | same as with_era5 | processed (aligned master) |
| `train.csv`/`validation.csv`/`test.csv` | `data/metadata/` | CSV | — | cyclone sub-tables + `*_cyclones.csv` (105/22/24 IDs) | master schema | 2013–2025 | same | — | processed (splits) |
| `{train,val,test}_sequences.npz` | `data/processed/forecasting/` | NPZ (float32) | — | X (2275/378/423, 5, 7) · Y (2275/378/423, 3, 3) | `features` [lat,lon,wind_speed,pressure,sst,wind_u,wind_v], `targets` [lat,lon,wind_speed] | 2013–2025 | — | 0% (filled) | processed (sequences — P4 deliverable) |
| `{train,val,test}_sequences_metadata.csv` | `data/processed/forecasting/` | CSV | — | 2,275/378/423 | cyclone_id, t_zero, t_minus_24h, t_plus_24h, origin_lat/lon/wind, target_{6,12,24}h_{lat,lon,wind} | 2013–2025 | — | — | processed |
| `README.md` | `data/processed/forecasting/` | MD | — | schema doc | — | — | — | — | processed |
| `multisource_{train,val,test}.csv` | `data/processed/classification/` | CSV | — | 3,039/518/651 × 15 | master schema | 2013–2025 | same | — | processed (P3) |
| `image_only_kaggle/{images,labels*.csv}` | `data/processed/classification/` | PNG/JPG+CSV | — | 133 images (93/19/21) | filename, wind_speed_kt, wind_speed_kmh, category | none | — | — | processed (P3, label risk) |
| `{detection_all,train,val,test}_detection.csv` | `data/processed/detection/` | CSV | — | 133 (93/19/21) | + `mock_bbox`, `structural_pattern` | none | — | — | processed (P2, mock fields) |
| `QA_REPORT.md` + 4 figures | `data/qa_reports/` | MD/PNG | — | — | — | — | — | — | processed |
| `docs/ERA5_ALIGNMENT.md`, `PS70_Person1_Status.md`, `future_task/tasks4.md` | root/docs/ | MD | — | — | — | — | — | — | processed (docs) |
| `get_ibtracs.py`, `download_era5.py`, `extract_era5_at_points.py`, `build_datasets.py`, `clean_kaggle_intensity.py`, coverage/split helpers | root | PY | — | — | — | — | — | — | processed (scripts) |
| `src/data/{forecasting,classification,detection}_dataset.py`, `preprocess_satellite.py`, `dataloader_example.py` | `src/data/` | PY | — | loaders | — | — | — | — | processed (code) |

*Non-delivery context (pre-existing in repo, untouched):* `data/processed/ibtracs_forecasting_base.csv`, `sequences/{X,y,{train,val,test}}.npy`, `normalization_stats.json`, `SEQUENCE_REPORT.md`, `DATA_QUALITY_REPORT.md`, `scripts/phase*.py`, `models/*`, `results/*` — an independent in-repo pipeline (chronological, 3-h grid, **no ERA5**), superseded by this handoff for P4.

---

## 2. IBTrACS status

**Status: READY (aligned subset partially sparse).**

- Cleaned table (`ibtracs_clean.csv`): 18,168 rows · 471 cyclones · 1980–2025 · 0 duplicates · chronologically ordered per cyclone · NIO basin (IMD-favored fallback hierarchy).
- **Wind selection:** `NEWDELHI_WIND` → `WMO_WIND` (IMD first, WMO fallback; kt × 1.852 → km/h, rounded 1 dp). Missing wind 44.7% (clean) / 23.2% (aligned).
- **Pressure selection:** `NEWDELHI_PRES` → `WMO_PRES` (hPa/mb). Missing pressure 47.4% (clean) / 22.2% (aligned).
- **Category:** derived from wind via RSMC New Delhi IMD scale; missing == wind missingness.
- **Coordinates:** as-is from IBTrACS (lat °N, lon °E, ISO-format `timestamp` UTC); rows without time/position dropped; longitude later mapped to 0–360 for ERA5 lookup.
- **3-hourly?** Yes on {00,03,…,21} UTC for the aligned set (all minute=0, one stray 14:00 row); the full clean table has a few :30 rows and 397 gaps >12 h.

---

## 3. ERA5 status

**Status: READY.**

- 94 monthly NetCDF4 files, 1.10 GB; grid 141×221 at 0.25°; coords `latitude` (30→−5°N) and `longitude` (50→105°E); time = `valid_time`, `seconds since 1970-01-01`, exactly 3-hourly {00,03,…,21}; 5,728 slices; `number`=0, `expver`='01' components normalized by P1 (`extract_era5_at_points.py` merges expver by fillna, drops `number`).
- Variables in-file (exact names): **`sst` (K), `msl` (Pa), `u10` (m s**-1), `v10` (m s**-1)** — all four requested variables present.
- Missing values (sampled ~5 slices/file across all 94 files, 14.65M cells/variable): `sst` **36.3 %**, `msl`/`u10`/`v10` **0 %** (matches a full-file scan of `era5_2013_05.nc`).
- Files cover only cyclone-active months; last month extends to 2025-12-03 21:00.

---

## 4. INSAT status

- **Not usable for P4 forecasting** (no timestamps / geolocation / cyclone identity attached); relevant to P3 (classification/intensity) and P2 (detection).
- `data/raw/insat/` and `data/raw/insat_kaggle/` are byte-identical duplicates.
- **Label caution (P3):** `clean_kaggle_intensity.py` maps the Kaggle `label` column (= image/cyclone number: `25.jpg→25`) to `wind_speed_kt`. Image-only intensity labels are **suspect**; multisource tabular classification is unaffected.

---

## 5. Master dataset structure

`master_dataset.csv` — 5,481 rows × 15 columns (exact names in file order):

```
cyclone_id, season, name, subbasin, timestamp, lat, lon, wind_speed, pressure, category, sst, wind_u, wind_v, pressure_msl, pre_genesis_favorable
```

Exact column mapping to the requested variables:

| Requested | Column | Source | Units |
|---|---|---|---|
| cyclone_id | `cyclone_id` | IBTrACS SID | — |
| timestamp | `timestamp` | IBTrACS ISO_TIME (UTC) | ISO 8601 |
| latitude | `lat` (renamed from `latitude`) | IBTrACS | °N |
| longitude | `lon` (renamed from `longitude`) | IBTrACS | °E |
| wind_speed | `wind_speed` (from `wind_speed_kmh`) | IBTrACS IMD→WMO | km/h |
| pressure | `pressure` (from `pressure_hpa`) | IBTrACS IMD→WMO | hPa |
| SST | `sst` (from `sst_celsius`) | ERA5 `sst` | °C |
| u_wind | `wind_u` (from `u_wind`) | ERA5 `u10` | m/s |
| v_wind | `wind_v` (from `v_wind`) | ERA5 `v10` | m/s |
| mean sea level pressure | `pressure_msl` (from `pressure_msl_hpa`) | ERA5 `msl` | hPa |
| category | `category` | derived IMD scale | categorical |
| — | `pre_genesis_favorable` | derived: `(sst≥26.5) & (category<CS)`; **False whenever sst missing** | bool |

`ibtracs_with_era5.csv` is the pre-rename version (14 cols) using the fuller names (`latitude, longitude, wind_speed_kmh, pressure_hpa, sst_celsius, u_wind, v_wind, pressure_msl_hpa`).

---

## 6. Forecasting feature availability

Coverage per row over all 5,481 aligned observations:

| Feature | Complete rows | % |
|---|---|---|
| lat / lon / timestamp / cyclone_id | 5,481 | 100 |
| `wind_u` / `wind_v` / `pressure_msl` | 5,481 | 100 |
| `sst` | 3,943 | 71.9 |
| `wind_speed` | 4,208 | 76.8 |
| `pressure` | 4,263 | 77.8 |
| **All 4 ERA5 vars (sst,u,v,msl)** | 3,943 | 71.9 |
| **All 6 forecast vars (sst,u,v,msl,wind,pressure)** | 2,986 | 54.5 |

The npz arrays themselves have **0 NaNs** — because the sequence builder interpolated over gaps (see §9). No normalization stats are shipped for the npz (raw float32 arrays); loader `src/data/forecasting_dataset.py::get_feature_stats()` provides train-only mean/std on demand.

---

## 7. Sequence dataset status

**Type: D — already-created sequence data** (class "cyclone + ERA5 synchronized data, windowed into sequences").

`data/processed/forecasting/*_sequences.npz` (float32):

| Dataset | X shape | Y shape | Unique cyclones | t_zero range |
|---|---|---|---|---|
| train | (2,275, 5, 7) | (2,275, 3, 3) | 67 | 2013-05-10 → 2025-11-29 |
| val | (378, 5, 7) | (378, 3, 3) | 14 | 2013-11-07 → 2025-10-28 |
| test | (423, 5, 7) | (423, 3, 3) | 16 | 2013-05-30 → 2025-12-01 |

- **Feature order (axis 2):** `[0] lat, [1] lon, [2] wind_speed, [3] pressure, [4] sst, [5] wind_u, [6] wind_v` (verified via `features` array + npz header/doc).
- **Timestep spacing:** 6 h (t−24, t−18, t−12, t−6, t) — verified `t_minus_24h → t_zero` = exactly 24 h, and per-sequence lon deltas are 6-h track steps.
- **Input window duration:** 24 hours (5 steps).
- **Target horizons:** +6, +12, +24 h — Y row order `[h=+6, h=+12, h=+24] × [lat, lon, wind_speed]`; **verified Y rows match metadata `target_{6,12,24}h_{lat,lon,wind}` to ≤6e-6**.
- **Window end = now:** `X[:, -1, :3]` matches metadata `origin_lat/lon/wind` to ≤6e-6 (verified).
- **Split method:** random 70/15/15 grouped **by cyclone_id** (not chronological); CSVs: 105/22/24 cyclones; sequences usable: 67/14/16 (short storms yield no 24-h sequence).
- **Cyclone-ID overlap between splits: 0 in both the CSV manifests and the npz metadata (verified numerically).**
- Metadata CSV (16 cols) carries per-sample `cyclone_id, t_zero, t_minus_24h, t_plus_24h, origin_*, target_{6,12,24}h_*`.

---

## 8. Train/validation/test split

- Cyclone-grouped (no cyclone appears in >1 split) — **PASS**, verified on CSVs (union 151 = all aligned cyclones) and on npz metadata (97 used).
- **Temporal realism — concern:** the split is random, so train/val/test each span 2013–2025 (interleaved eras). Chronological splitting would better model "train on history → forecast future".
- NPZ usable-sample counts (2275/378/423) are smaller than CSV cyclone counts because many storms are too short to fit a 24-h input window + 24-h target.

---

## 9. Leakage audit

Verified checks (nothing fixed):

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Same cyclone in multiple splits | **PASS** — 0 overlap | CSV + npz metadata sets |
| 2 | Future observations inside inputs | **RISK** — window is t−24…t, but `build_datasets.py` does `grp[feature_cols].interpolate(method='time').ffill().bfill()` (lines 148/164): **`.interpolate` and `.bfill` are non-causal** — a history feature can be filled from observations *after t*, and target values can themselves be interpolated (fabricated) rather than observed. | source code; npz 0-NaN despite 28% sst missing |
| 3 | Target values included as input | **PASS** — features/labels disjoint; inputs carry only ≤t values; Y only encodes t+6/12/24 | npz/X-Y verified vs metadata |
| 4 | ERA5 values from future timestamps | **RISK via (2)** — per-row extraction is at the observation's own timestamp (nearest synoptic), so no direct future-ERA5; only interpolation can mix eras | `extract_era5_at_points.py` |
| 5 | Normalization from val/test | **PASS** — P1 ships raw npz (no normalization applied, no stats shipped); loader offers train-only stats | npz inspection |
| 6 | Incorrect +6/+12/+24 mapping | **PASS** — Y vs metadata diffs ≤6e-6; window/target spans exactly 24 h | verified |
| 7 | `pre_genesis_favorable` bias | WARNING — `sst` missing ⇒ flag False even if environment was favorable (NaNs are not imputed there) | `build_datasets.py` lines 59–61 |

---

## 10. Missing data

| Level | Variable | Missing | Notes |
|---|---|---|---|
| Raw ERA5 | `sst` / `msl` / `u10` / `v10` | 36.3% / 0% / 0% / 0% | sampled scan; sst gaps everywhere, others complete |
| Aligned master | `sst` | 28.1% (1,538/5,481) | direct consequence of raw sst NaN + bbox edges |
| Aligned master | `wind_speed` | 23.2% (1,273) | IBTrACS sparsity |
| Aligned master | `pressure` | 22.2% (1,218) | IBTrACS sparsity |
| Aligned master | `wind_u` / `wind_v` / `pressure_msl` | 0% | complete |
| Aligned master | `category` | 23.2% | == wind missingness |
| npz X/Y | any channel | **0%** | gaps filled by interpolation — see §9 #2 for the integrity caveat |

---

## 11. Unit/coordinate checks

| Item | Verified value |
|---|---|
| wind_speed | km/h (= kt × 1.852, rounded 1 dp; IMD→WMO hierarchy) |
| pressure | hPa (= mb; IMD→WMO hierarchy) |
| sst | °C (= K − 273.15, rounded 2; ERA5 nearest-cell) |
| wind_u / wind_v | m/s (ERA5 `u10`/`v10`) |
| pressure_msl | hPa (= Pa / 100, rounded 1) |
| coordinates | lat °N, lon °E; ERA5 lon normalized to 0–360 for lookup; tracks may exit 0–30N/40–100E (91.7% inside) |
| timestamp | UTC ISO; synoptic 3-hourly; derived from `valid_time` seconds-since-1970 |
| NIO filter | basin=NI at IBTrACS source; a 396-row (7.2%) minority outside the ERA5 grid → edge-clipped env values (not flagged) |
| **ERA5 ↔ CSV cross-validation** | **12/12 sampled cyclone observations matched the raw NetCDF exactly** (ΔSST < 0.005 °C, Δu/v = 0.0, ΔMSLP < 0.05 hPa) by nearest grid cell + synoptic slot |

---

## 12. P1 → P4 readiness assessment

**P4 requirement** (lat, lon, wind_speed → +6/+12/+24 h, inputs lat/lon/wind/pressure/sst/u/v) is **fulfilled by P1's deliverable**:
- Aligned source (master_dataset, 5,481 obs) built by real nearest-neighbor ERA5 extraction, unit/coordinate handling verified, values confirmed against raw NetCDF.
- Ready-made sequence arrays matching the exact contract, correct target horizons, cyclone-disjoint splits, no accidental same-storm leakage, no normalization leakage.

**Blocking for clean model training:**
1. Non-causal gap-filling (`interpolate().ffill().bfill()`) — fabricated inputs/targets, no origin flag.
2. Random (interleaved) split instead of chronological.
3. SST 28% missing with no flag; `pre_genesis_favorable` silently False on missing SST; QA doc's "100% completeness" is wrong.
4. 7.2% of rows edge-clipped at the ERA5 bbox.
5. Only 97 of 151 cyclones yield usable sequences.

None of these prevent building a first model, but each must be mitigated (causal refill + flags, chronological split, SST imputation policy, edge masking) before final training/reporting.

---

## VERDICT

**READY WITH ISSUES**

The P4-required data chain is present, real, correctly aligned, and usable: an ERA5-synchronized master table (values independently verified against the raw NetCDF, 12/12), and pre-built sequence tensors (5×7 → 3×3, +6/+12/+24 h) matching the exact input/target contract with cyclone-disjoint splits and no direct leakage. P4 can start modeling **immediately with caveats**: the sequence builder's non-causal interpolation can fabricate inputs and targets without flagging them, the split is random rather than chronological, SST is ~28% missing (silently False in `pre_genesis_favorable`, misreported as 100% in the QA doc), ~7% of rows carry edge-clipped ERA5 features, and only 97/151 storms yield sequences. Recommend a causal refill with synthetic-origin flags, a chronological cyclone split, and a documented SST/imputation policy as the first P4 step.