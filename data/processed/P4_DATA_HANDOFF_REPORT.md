# P4 Data Handoff Report

**Person 4 — Cyclone Forecasting | SIH 2026 PS 26070**
**Date:** 2026-08-29
**Scope:** READ-ONLY inspection of Person 1's data delivery (`PS70-main.zip`) to establish exactly what forecasting data exists for P4. No files were modified, no download was performed, no model/sequences were built. Verification used actual file metadata and sampled values.

**Source of truth:** read-only extraction of the ZIP at `%TEMP%\opencode\PS70_extracted\PS70-main` (delivery root), plus the pre-existing local project pipeline files for context.

---

## 0. Executive Summary

- P1 delivered a **complete, FYRON-ready forecasting dataset**: `data/processed/forecasting/{train,val,test}_sequences.npz` matching the exact P4 contract (features `lat, lon, wind_speed, pressure, sst, wind_u, wind_v`; targets `lat, lon, wind_speed` at +6/+12/+24 h).
- **IBTrACS: READY** (cleaned 1980–2025 table + cyclone-grouped splits).
- **ERA5: READY** (94 monthly files 2013–2025, the 4 requested variables, 0.25°, 3-hourly).
- **ERA5 is ALREADY aligned with IBTrACS** by P1 (`ibtracs_with_era5.csv` / `master_dataset.csv`, 5,481 matched observations, 151 cyclones).
- Main issues: (1) SST ~28% missing in the matched set (raw ERA5 SST ~36% missing) — no flag column; (2) sequence builder fills gaps with **non-causal interpolation** (future info can enter inputs/targets); (3) split is random-by-cyclone, not chronological; (4) QA report's "100% completeness" figure is wrong.

---

## 1. Files Discovered

### 1.1 Delivery table (ZIP: `PS70-main.zip`, 1,182,652,292 B)

| filename | path | type | size | purpose | usable for P4? |
|---|---|---|---|---|---|
| `ibtracs_NI_raw.csv` | `data/raw/ibtracs/` | CSV | 27,875,881 B | Raw IBTrACS v04r01 NIO export (1842–2025) | reference |
| `era5_YYYY_MM.nc` (94) | `data/raw/era5/` | NetCDF4 (HDF5) | 1,103,681,364 B total | ERA5 SST/MSLP/u10/v10 2013–2025 | YES (needed later) |
| `insat3d_raw_cyclone_ds/…` + `insat3d_ir_cyclone_ds/…` + `insat3d_for_reference_ds/…` | `data/raw/insat/` (= `insat_kaggle/`) | images (jpg/png) | ~5.7+8.1+33.6 MB | INSAT-3D imagery (duplicated) | no (P3/P2 domain) |
| `insat_3d_ds - Sheet.csv` | `data/raw/insat_kaggle/` | CSV | — | label table for imagery | no (P3; label semantics suspect) |
| `ibtracs_clean.csv` | `data/metadata/` | CSV | — | Cleaned IBTrACS 1980–2025, 18,168 rows/471 cyclones | YES (base track data) |
| `ibtracs_with_era5.csv` | `data/metadata/` | CSV | — | Track + per-observation ERA5 (5,481 rows/151 cyclones) | YES (aligned source) |
| `master_dataset.csv` | `data/metadata/` and `data/processed/` | CSV | 738,948 B (2 identical copies, md5 `6b3e93e8…`) | Final aligned master + `pre_genesis_favorable` (5,481×15) | YES (aligned source) |
| `train.csv` / `validation.csv` / `test.csv` | `data/metadata/` | CSV | — | Cyclone-level train/val/test tables | YES (split manifests) |
| `train_cyclones.csv` / `validation_cyclones.csv` / `test_cyclones.csv` | `data/metadata/` | CSV | — | Split cyclone-ID lists (105 / 22 / 24) | YES |
| `{train,val,test}_sequences.npz` | `data/processed/forecasting/` | NPZ | — | **P4 sequence arrays** (5×7 input → 3×3 target, +6/12/24 h) | **YES — primary** |
| `{train,val,test}_sequences_metadata.csv` | `data/processed/forecasting/` | CSV | — | Sequence/cyclone/timestamp metadata | YES |
| `README.md` (forecasting) | `data/processed/forecasting/` | MD | — | Schema spec | YES |
| `multisource_{train,val,test}.csv` | `data/processed/classification/` | CSV | — | Tabular classification stream (3,039/518/651) | context (P3) |
| `image_only_kaggle/{images,labels*(csv)}` | `data/processed/classification/` | images+CSV | — | 133 image-only labels/splits | context (P3, label risk) |
| `{detection_all,train,val,test}_detection.csv` | `data/processed/detection/` | CSV | — | Detection manifests w/ `mock_bbox` | context (P2) |
| `QA_REPORT.md` + `figures/` | `data/qa_reports/` | MD+PNG | — | QA report + 4 figures | reference |
| `docs/ERA5_ALIGNMENT.md` | `docs/` | MD | — | ERA5 variable/grid/alignment spec | reference |
| `PS70_Person1_Status.md`, `README.md`, `future_task/tasks4.md` | root / `future_task/` | MD | — | P1 status & P4 task notes | reference |
| `get_ibtracs.py`, `download_era5.py`, `extract_era5_at_points.py`, `build_datasets.py`, `clean_kaggle_intensity.py`, `check_coverage.py`, `check_missing_files.py`, `find_near_matches.py`, `prioritize_gap.py`, `split_coverage_gap.py` | root | PY | — | Reproduction scripts | YES (read-only audit) |
| `src/data/{forecasting,classification,detection}_dataset.py`, `preprocess_satellite.py`, `dataloader_example.py` | `src/data/` | PY | — | Loaders | YES (`forecasting_dataset.py`) |

### 1.2 Pre-existing local project files (context, NOT part of P1 delivery)

`data/processed/ibtracs_forecasting_base.csv`, `sequences/{X,y,{train,val,test}}.npy`, `normalization_stats.json`, `SEQUENCE_REPORT.md`, `DATA_QUALITY_REPORT.md`; `scripts/phase{1,2,3,3_5}*.py`; `models/*.json`, `lstm_forecaster.pt`; `results/{PHASE3_REPORT,PHASE3_AUDIT}.md` + `*.json` — an earlier in-repo pipeline (chronological split, 3-h grid, **no ERA5 features**). Kept untouched.

---

## 2. IBTrACS Status

Files: `ibtracs_clean.csv` (12 cols), `ibtracs_with_era5.csv` (14 cols), `master_dataset.csv` (15 cols).

| Metric | `ibtracs_clean.csv` | `ibtracs_with_era5.csv` | `master_dataset.csv` |
|---|---|---|---|
| Rows | 18,168 | 5,481 | 5,481 |
| Unique cyclone IDs | 471 | 151 | 151 |
| Timestamp range | 1980-10-10 06:00 → 2025-12-02 18:00 | 2013-05-09 18:00 → 2025-12-02 18:00 | same |
| Timestamp frequency | ~3-hourly (median gap 3 h; 2 rows at :30; hours mostly {0,3,6,9,12,15,18,21}) | all minute=0; hours {0,3,6,…,21} + 1 row at 14:00 | same |
| Latitude range | 0.70 – 31.00 °N | 1.90 – 29.20 °N | same |
| Longitude range | 41.80 – 163.70 °E | 41.80 – 141.00 °E | same |
| Wind column | `wind_speed_kmh` (km/h, kt×1.852) | `wind_speed_kmh` (km/h) | `wind_speed` (km/h) |
| Pressure column | `pressure_hpa` (hPa, from mb) | `pressure_hpa` (hPa) | `pressure` (hPa) |
| Wind missing | 44.73% (10,041 / 18,168) | 23.23% (4,208 / 5,481) | same |
| Pressure missing | 47.42% (9,553 / 18,168) | 22.22% (4,263 / 5,481) | same |
| Category field | `category` (IMD scale), missing 44.73% | present, missing 23.23% | present |
| Category distribution (with_era5) | — | Depression 1,665 · Deep Depression 868 · Cyclonic Storm 769 · Very Severe 353 · Severe 349 · Extremely Severe 178 · Super 25 · Low Pressure Area 1 | same |
| Duplicate rows | 0 (also 0 for (cyclone_id, timestamp)) | 0 | 0 |
| Chronological ordering | True within every cyclone | True | True |

- **NIO-filtered?** Yes at basin level (IBTrACS `NI` list), but 14.2% (clean) / 8.3% (era5 set) of track points lie outside 0–30°N, 40–100°E (basin convention keeps full lifecycles). The aligned set's spatial extent inside the box: 91.7%.
- **Synchronized to 3-hourly?** Effectively yes for the aligned set (all minute=0, on 00/03/06/…/21 UTC grid; one row at 14:00 → nearest 15:00 in ERA5). Gaps >12 h exist for 125 aligned rows; clean has 397 such gaps.
- Category based on IMD/RSMC New Delhi scale (from wind), consistently assigned by `get_ibtracs.py`/`clean_kaggle_intensity.py`.

**Verdict: READY** — chronology, units, categories, uniqueness, and basin filtering all verified.

---

## 3. ERA5 Status

### 3.1 Formats, dimensions, coordinates (verified with h5py on all 94 files)

| Property | Value |
|---|---|
| Format | NetCDF4 (HDF5 container; magic `89 48 44 46`), `<var>` → h5py Datasets with GRIB-derived attrs (`GRIB_centre: ecmf`) |
| Files | 94 (`era5_YYYY_MM.nc`; only cyclone-active months downloaded) |
| Total size | 1,103,681,364 B (≈ 1.10 GB) |
| Dimensions | `valid_time` (≤744), `latitude` (141), `longitude` (221); `number` scalar 0; `expver` field all `01` |
| Coordinates | `latitude`: 30.0 → −5.0 (equal-step 0.25, descending), 141 pts; `longitude`: 50.0 → 105.0, 221 pts; `valid_time`: "seconds since 1970-01-01" |
| Temporal resolution | 3-hourly, exactly hours {00,03,06,09,12,15,18,21} |
| Total time slices | 5,728 |

### 3.2 Variables inside the files (actual names + units)

| Requested (spec) | In-file name | Units (verified) | Present? |
|---|---|---|---|
| sea_surface_temperature | **`sst`** | `K` | YES |
| mean_sea_level_pressure | **`msl`** | `Pa` | YES |
| 10m_u_component_of_wind | **`u10`** | `m s**-1` | YES |
| 10m_v_component_of_wind | **`v10`** | `m s**-1` | YES |

P1 renamed them at match time: `sst`→`sst` (K→°C), `msl`→`pressure_msl`/`pressure_msl_hpa` (Pa→hPa), `u10`→`u_wind`/`wind_u`, `v10`→`v_wind`/`wind_v`.

### 3.3 Per-file table (all 94 files: size, slices, temporal extent)

| file | MB | nt | start | end | file | MB | nt | start | end |
|---|---|---|---|---|---|---|---|---|---|
| era5_2013_05 | 17.1 | 88 | 2013-05-09 | 2013-05-31 | era5_2019_05 | 6.1 | 32 | 2019-05-01 | 2019-05-04 |
| era5_2013_07 | 3.1 | 16 | 2013-07-30 | 2013-07-31 | era5_2019_06 | 17.0 | 88 | 2019-06-08 | 2019-06-18 |
| era5_2013_08 | 7.7 | 40 | 2013-08-01 | 2013-08-23 | era5_2019_08 | 6.1 | 32 | 2019-08-06 | 2019-08-09 |
| era5_2013_10 | 12.3 | 64 | 2013-10-07 | 2013-10-14 | era5_2019_09 | 13.9 | 72 | 2019-09-20 | 2019-09-30 |
| era5_2013_11 | 45.2 | 232 | 2013-11-01 | 2013-11-29 | era5_2019_10 | 17.1 | 88 | 2019-10-01 | 2019-10-31 |
| era5_2013_12 | 12.3 | 64 | 2013-12-05 | 2013-12-12 | era5_2019_11 | 18.5 | 96 | 2019-11-01 | 2019-11-30 |
| era5_2014_01 | 10.6 | 56 | 2014-01-02 | 2014-01-08 | era5_2019_12 | 15.4 | 80 | 2019-12-01 | 2019-12-10 |
| era5_2014_05 | 4.7 | 24 | 2014-05-21 | 2014-05-23 | era5_2020_05 | 15.5 | 80 | 2020-05-15 | 2020-05-31 |
| era5_2014_06 | 11.0 | 56 | 2014-06-08 | 2014-06-14 | era5_2020_06 | 6.1 | 32 | 2020-06-01 | 2020-06-04 |
| era5_2014_07 | 4.7 | 24 | 2014-07-21 | 2014-07-23 | era5_2020_10 | 12.3 | 64 | 2020-10-11 | 2020-10-23 |
| era5_2014_08 | 7.7 | 40 | 2014-08-03 | 2014-08-07 | era5_2020_11 | 15.3 | 80 | 2020-11-20 | 2020-11-30 |
| era5_2014_10 | 28.1 | 144 | 2014-10-06 | 2014-10-31 | era5_2020_12 | 21.2 | 112 | 2020-12-01 | 2020-12-25 |
| era5_2014_11 | 9.3 | 48 | 2014-11-01 | 2014-11-09 | era5_2021_04 | 3.1 | 16 | 2021-04-02 | 2021-04-03 |
| era5_2015_06 | 18.5 | 96 | 2015-06-06 | 2015-06-24 | era5_2021_05 | 18.6 | 96 | 2021-05-13 | 2021-05-27 |
| era5_2015_07 | 13.8 | 72 | 2015-07-10 | 2015-07-31 | era5_2021_09 | 20.0 | 104 | 2021-09-12 | 2021-09-30 |
| era5_2015_08 | 4.6 | 24 | 2015-08-01 | 2015-08-04 | era5_2021_10 | 9.1 | 48 | 2021-10-01 | 2021-10-08 |
| era5_2015_09 | 6.2 | 32 | 2015-09-16 | 2015-09-19 | era5_2021_11 | 10.7 | 56 | 2021-11-08 | 2021-11-30 |
| era5_2015_10 | 17.0 | 88 | 2015-10-07 | 2015-10-31 | era5_2021_12 | 9.2 | 48 | 2021-12-01 | 2021-12-06 |
| era5_2015_11 | 15.4 | 80 | 2015-11-01 | 2015-11-10 | era5_2022_03 | 12.2 | 64 | 2022-03-03 | 2022-03-23 |
| era5_2016_05 | 9.3 | 48 | 2016-05-17 | 2016-05-22 | era5_2022_05 | 15.4 | 80 | 2022-05-05 | 2022-05-20 |
| era5_2016_06 | 6.2 | 32 | 2016-06-26 | 2016-06-29 | era5_2022_07 | 3.2 | 16 | 2022-07-16 | 2022-07-17 |
| era5_2016_08 | 13.9 | 72 | 2016-08-09 | 2016-08-20 | era5_2022_08 | 21.6 | 112 | 2022-08-09 | 2022-08-23 |
| era5_2016_10 | 13.7 | 72 | 2016-10-21 | 2016-10-29 | era5_2022_09 | 3.1 | 16 | 2022-09-11 | 2022-09-12 |
| era5_2016_11 | 10.7 | 56 | 2016-11-02 | 2016-11-30 | era5_2022_10 | 7.7 | 40 | 2022-10-21 | 2022-10-25 |
| era5_2016_12 | 24.3 | 128 | 2016-12-01 | 2016-12-19 | era5_2022_11 | 4.6 | 24 | 2022-11-20 | 2022-11-22 |
| era5_2017_04 | 6.2 | 32 | 2017-04-14 | 2017-04-17 | era5_2022_12 | 29.6 | 152 | 2022-12-04 | 2022-12-25 |
| era5_2017_05 | 6.2 | 32 | 2017-05-27 | 2017-05-30 | era5_2023_01 | 3.1 | 16 | 2023-01-30 | 2023-01-31 |
| era5_2017_06 | 3.1 | 16 | 2017-06-11 | 2017-06-12 | era5_2023_02 | 3.1 | 16 | 2023-02-01 | 2023-02-02 |
| era5_2017_07 | 6.2 | 32 | 2017-07-18 | 2017-07-27 | era5_2023_05 | 12.4 | 64 | 2023-05-08 | 2023-05-15 |
| era5_2017_08 | 10.7 | 56 | 2017-08-19 | 2017-08-25 | era5_2023_06 | 23.0 | 120 | 2023-06-05 | 2023-06-19 |
| era5_2017_10 | 4.6 | 24 | 2017-10-09 | 2017-10-19 | era5_2023_07 | 3.1 | 16 | 2023-07-30 | 2023-07-31 |
| era5_2017_11 | 18.4 | 96 | 2017-11-03 | 2017-11-30 | era5_2023_08 | 4.6 | 24 | 2023-08-01 | 2023-08-03 |
| era5_2017_12 | 15.3 | 80 | 2017-12-01 | 2017-12-10 | era5_2023_09 | 1.6 | 8 | 2023-09-30 | 2023-09-30 |
| era5_2018_03 | 3.1 | 16 | 2018-03-13 | 2018-03-14 | era5_2023_10 | 12.2 | 64 | 2023-10-01 | 2023-10-25 |
| era5_2018_05 | 25.1 | 128 | 2018-05-15 | 2018-05-30 | era5_2023_11 | 10.6 | 56 | 2023-11-13 | 2023-11-30 |
| era5_2018_06 | 1.6 | 8 | 2018-06-10 | 2018-06-10 | era5_2023_12 | 9.3 | 48 | 2023-12-01 | 2023-12-06 |
| era5_2018_07 | 4.7 | 24 | 2018-07-21 | 2018-07-23 | era5_2024_05 | 9.4 | 48 | 2024-05-23 | 2024-05-28 |
| era5_2018_08 | 7.7 | 40 | 2018-08-07 | 2018-08-17 | era5_2024_07 | 3.1 | 16 | 2024-07-19 | 2024-07-20 |
| era5_2018_09 | 9.3 | 48 | 2018-09-06 | 2018-09-22 | era5_2024_08 | 16.9 | 88 | 2024-08-02 | 2024-08-31 |
| era5_2018_10 | 17.2 | 88 | 2018-10-05 | 2018-10-15 | era5_2024_09 | 19.9 | 104 | 2024-09-01 | 2024-09-17 |
| era5_2018_11 | 15.3 | 80 | 2018-11-10 | 2018-11-19 | era5_2024_10 | 15.3 | 80 | 2024-10-13 | 2024-10-26 |
| era5_2018_12 | 10.8 | 56 | 2018-12-13 | 2018-12-31 | era5_2024_11 | 12.2 | 64 | 2024-11-23 | 2024-11-30 |
| era5_2019_01 | 10.6 | 56 | 2019-01-01 | 2019-01-07 | era5_2024_12 | 9.1 | 48 | 2024-12-01 | 2024-12-21 |
| era5_2019_04 | 9.3 | 48 | 2019-04-25 | 2019-04-30 | era5_2025_05 | 4.7 | 24 | 2025-05-24 | 2025-05-30 |
| era5_2019_05 | 6.1 | 32 | 2019-05-01 | 2019-05-04 | era5_2025_07 | 10.8 | 56 | 2025-07-14 | 2025-07-26 |

*(remaining 8: `era5_2025_08` 13.8 MB/72 slices 2025-08-18→27, `2025_09` 21.4/112 09-06→30, `2025_10` 26.5/136 10-01→31, `2025_11` 9.2/48 11-25→30, `2025_12` 3.1/16 12-01→02.)* All 94 use the identical 141×221 0.25° grid; all start/end on 00:00/21:00 UTC synoptic times.

### 3.4 Missing values in ERA5 (sampled-slice scan, ≤5 evenly spaced slices per variable per file; 14,645,670 cells per variable)

| Variable | NaN (sampled) | % |
|---|---|---|
| `sst` | 5,308,650 | **36.25 %** (missing everywhere, all 94 files affected) |
| `msl` | 0 | 0.0 % |
| `u10` | 0 | 0.0 % |
| `v10` | 0 | 0.0 % |

(Sample figures match a full-file scan of `era5_2013_05.nc`: sst 36.25%, others 0%.)

**Verdict: READY** — all four requested variables present with expected units and grid.

---

## 4. Variables Available

| Variable | In master_dataset | In forecasting npz (X) | Units | Real-data coverage |
|---|---|---|---|---|
| cyclone_id / SID | `cyclone_id` | in metadata CSV | — | 151 cyclones |
| timestamp | `timestamp` (UTC) | `t_zero`, `t_minus_24h`, `t_plus_24h` | ISO | 3-hourly |
| latitude | `lat` | feature 0 | °N | 100% |
| longitude | `lon` | feature 1 | °E | 100% |
| wind_speed | `wind_speed` | feature 2 | km/h | 76.8% (rest interpolated) |
| pressure | `pressure` | feature 3 | hPa | 77.8% (rest interpolated) |
| SST | `sst` | feature 4 | °C | 72.0% (rest interpolated) |
| u_wind | `wind_u` | feature 5 | m/s | 100% |
| v_wind | `wind_v` | feature 6 | m/s | 100% |
| (+MSLP) | `pressure_msl` | (not in X) | hPa | 100% |
| category | `category` | — | IMD scale | 76.8% |
| `pre_genesis_favorable` | present | — | bool | True 2,459 / False 3,022 (NaN sst ⇒ False) |

Recommended P4 model inputs are fully covered by the npz features `[lat, lon, wind_speed, pressure, sst, wind_u, wind_v]`.

---

## 5. Spatial Coverage

| Grid | Value |
|---|---|
| ERA5 | lat 30.0 → −5.0 °N (0.25°), lon 50.0 → 105.0 °E (0.25°) |
| Requested NIO region | 0–30 °N, 40–100 °E |
| Overlap | Covers the requested region **and** extends to 5°S / 105°E (catches the Bay/Bengal & SE approaches) |
| Aligned track-point coverage | 91.7% of matched obs inside 0–30N/40–100E; **396 rows (7.2%) outside** (lon up to 141°E, e.g. cyclone `2013305N07141`; lat down to 1.9°N) → those obs' ERA5 features are **edge-clipped to the bbox** |

**No change required to ERA5 files.** Spatial "satellite-buffer" design is appropriate for the NIO forecasting region.

---

## 6. Temporal Coverage

| Source | Coverage |
|---|---|
| IBTrACS (clean) | 1980-10-10 → 2025-12-02 |
| ERA5 | 2013-05-09 00:00 → 2025-12-03 21:00 (5,728 slices; last file carries 16 extra slices past 2025-12-02) |
| Aligned set (master) | 2013-05-09 18:00 → 2025-12-02 18:00 (5,481 obs; monthly files were downloaded only for cyclone-active months) |
| P1 forecasting npz (t_zero) | train 2013-05-10 → 2025-11-29 · val 2013-11-07 → 2025-10-28 · test 2013-05-30 → 2025-12-01 |

---

## 7. Timestamp Compatibility

| Check | Result (verified) |
|---|---|
| Common years | 2013–2025 (both) |
| Common timestamps | Full set — all 5,481 aligned obs are on minute=0 of the 3-hourly synoptic grid {00,03,06,09,12,15,18,21} UTC |
| ERA5 timestep | 003:00 (exactly 3-hourly) |
| IBTrACS timestep | 003:00 nominal (median inter-obs gap 3 h; 125 aligned rows with gaps >12 h) |
| Interpolation/nearest needed? | **Marginally.** Temporal alignment is exact at the synoptic hours. Only 1 aligned obs is off-grid (14:00 → matches nearest ERA5 15:00). No re-interpolation required for the delivered dataset. |

---

## 8. Is ERA5 Already Aligned With IBTrACS?

**YES — P1 already performed the ERA5 ↔ IBTrACS synchronization**, verified across files and scripts:

- **Script:** `extract_era5_at_points.py` runs `extract_era5_at_points.py` → reads each `era5_YYYY_MM.nc`, drops `number`/`expver`, converts `valid_time` (seconds since **1970-01-01**, verified) to `time`, then for each IBTrACS observation performs:
  - **spatial** `.sel(method="nearest")` on lat/lon (0.25° grid),
  - **temporal** nearest `00/03/…` synoptic slot,
  - **unit conversion** `sst` K→°C and `msl` Pa→hPa.
- **Outputs:** `ibtracs_with_era5.csv` (14 cols) → `build_datasets.py` renames (`latitude→lat`, `wind_speed_kmh→wind_speed`, `u_wind→wind_u`, …) → **`master_dataset.csv`** (15 cols incl. `pre_genesis_favorable`; 5,481 rows).
- `docs/ERA5_ALIGNMENT.md` documents the same scheme; `check_coverage.py`/`check_missing_files.py` are verification helpers.
- The forecasting npz reads features from the already-aligned master.

Because the extraction is already done, the statement "ERA5 alignment is not yet complete and must be performed by P4" is **not** applicable — P4 inherits an aligned table.

---

## 9. Missing-Data Statistics

| Column (aligned master) | Missing | Notes |
|---|---|---|
| `wind_speed` | 23.2 % (1,273/5,481) | source IBTrACS sparsity |
| `pressure` | 22.2 % (1,218/5,481) | source IBTrACS sparsity |
| `sst` | 28.1 % (1,538/5,481) | driver: raw ERA5 SST NaN (≈36%) and bbox edges |
| `pressure_msl` | 0 % | complete |
| `wind_u` / `wind_v` | 0 % | complete |
| category | 23.2 % | derived from wind; == wind missingness |
| npz arrays (X/Y) | 0 % (all NaNs filled via storm interpolation) | **see §11 risk** |
| ERA5 raw | sst 36.25 %; msl/u10/v10 0 % | sampled-slice scan |

---

## 10. Recommended Next P4 Step

Proceed to build the forecasting sequences on top of the aligned master / npz, but **mitigate the identified risks first**:

1. **Re-construct sequences causally** from `master_dataset.csv` (avoid `bfill`/non-causal `interpolate`); keep P1's 5-step × 6-h window and +6/+12/+24 targets, but **mask/drop samples whose targets are synthetic** and add a `synthetic`/`missing` flag column for inputs.
2. **Re-split chronologically by `cyclone_id`** (train = older era, test = most recent), since P1's split is random-by-cyclone.
3. Normalize with train-only statistics; treat longitude as circular (0–360) because tracks reach 141°E.
4. Baseline: persistence → LSTM/GRU → seq2seq-attention, evaluated with MAE + Haversine track error per horizon (local in-repo phase-3 already provides persistence/LSTM reference numbers).

---

## 11. Problems Discovered

| # | Severity | Problem | Evidence |
|---|---|---|---|
| 1 | HIGH (target integrity) | Sequence builder fills gaps with **non-causal interpolation** (`interpolate(method='time').ffill().bfill()` in `build_datasets.py`) — history entries can embed post-t info and target wind/pressure can be interpolated, not observed; npz reports 0 NaNs because gaps were filled, not dropped. No filler-src flag. | source code + npz 0-NaN |
| 2 | MEDIUM (validation realism) | Train/val/test split is **random by cyclone, temporally interleaved** (train/val/test all span 2013–2025). Cyclone-disjoint (0 overlap, verified) but not forecast-style. | npz metadata t_zero ranges |
| 3 | MEDIUM (data quality doc) | `QA_REPORT.md` claims "Completeness: **100.0%**" while listing `sst: 1538` missing; actual master SST coverage ≈ 72%. | QA_REPORT.md vs verified counts |
| 4 | MEDIUM (class-label semantics) | `clean_kaggle_intensity.py` treats Kaggle `label` (image number) as **wind speed in knots** (`25.jpg→25 kt`). Image-only classification labels are suspect — impacts P3, not P4 inputs. | labels.csv + raw CSV + script |
| 5 | LOW (feature fidelity) | 396 rows (7.2%) outside ERA5 bbox get **edge-clipped** sst/u/v; not flagged in npz. SST raw missingness ≈ 36%. | master lat/lon ranges vs bbox |
| 6 | LOW (duplication/size) | `data/raw/insat/` & `insat_kaggle/` duplicates add ≈ 50 MB; `master_dataset.csv` duplicated under both metadata/ and processed/. | md5 identical |
| 7 | INFO | ERA5 has `number=0` scalar and `expver='01'` (ERA5 non-Reanalysis-5 ensemble weighting); dropped by P1 at extraction — no action needed. | h5py attrs |
| 8 | INFO | Detection CSVs carry `mock_bbox` / fixed `structural_pattern` (integration placeholders) — affects P2 handoff, not P4. | detection README |

---

*This report is read-only output; it adds no derived/modified data files. The two prior P4 reports (`results/P1_HANDOFF_AUDIT.md`) remain available for cross-reference.*