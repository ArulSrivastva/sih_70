# P1 Handoff Audit — Data Engineering (Forecasting Perspective)

**Auditor:** Person 4 (Cyclone Forecasting)
**Date:** 2026-08-29
**Scope:** Inspection-only audit of Person 1's delivered data (`PS70-main.zip`, 1,182,652,292 bytes) against the P4 forecasting requirements.
**Method:** All facts below were verified by directly reading the delivered files (CSVs, NetCDF via h5py, npz archives, images, scripts, docs). No delivered file was modified. A read-only extraction was made to `%TEMP%\opencode\PS70_extracted\PS70-main` purely for inspection.

---

## 1. Project Structure

**Top level of `PS70-main.zip` (delivered by P1):**

| Path | Contents | Status |
|---|---|---|
| `data/raw/ibtracs/ibtracs_NI_raw.csv` | IBTrACS v04r01, NIO basin raw export (62,849 rows incl. units row) | OK |
| `data/raw/era5/*.nc` | 94 monthly ERA5 NetCDF files, 2013–2025 (1,103,681,364 B ≈ 1.10 GB) | OK |
| `data/raw/insat/`, `data/raw/insat_kaggle/` | INSAT 3D imagery (duplicated folders) + label CSV | OK |
| `data/metadata/` | `ibtracs_clean.csv`, `ibtracs_with_era5.csv`, `master_dataset.csv`, train/val/test + `*_cyclones.csv` | OK |
| `data/processed/forecasting/` | `{train,val,test}_sequences.npz` + `*_sequences_metadata.csv` + `README.md` | OK — P4-relevant |
| `data/processed/classification/` | `multisource_{train,val,test}.csv`, `image_only_kaggle/` (133 images + labels) | OK (label concern, §6/§8) |
| `data/processed/detection/` | `{detection_all,train,val,test}_detection.csv` | OK (mock fields, §8) |
| `data/processed/master_dataset.csv` | byte-identical copy of `data/metadata/master_dataset.csv` (md5 `6b3e93e8…`) | OK |
| `data/qa_reports/` | `QA_REPORT.md` + 4 figures | OK (doc bug, §4) |
| `docs/ERA5_ALIGNMENT.md` | ERA5 variable/grid/alignment spec | OK |
| `future_task/` | `tasks.md`, `tasks2.md`, `tasks3.md`, `tasks4.md` | OK |
| `src/data/` | `forecasting_dataset.py`, `classification_dataset.py`, `detection_dataset.py`, `preprocess_satellite.py`, `dataloader_example.py` | OK |
| Root scripts | `get_ibtracs.py`, `download_era5.py`, `extract_era5_at_points.py`, `build_datasets.py`, `clean_kaggle_intensity.py`, `check_coverage.py`, `check_missing_files.py`, `find_near_matches.py`, `prioritize_gap.py`, `split_coverage_gap.py`, `README.md`, `PS70_Person1_Status.md`, `ps70_team_plan_edited.pdf`, `LICENSE` | OK |

**Local project directory (`cyclone-project\`, where this audit lives):**
- `data/raw/era5/` is **empty**, `data/metadata/` does **not** exist — the ERA5/metadata data lives only inside the ZIP.
- The repo already contains an **independent P4 pipeline** (pre-dates the ZIP, authored outside this handoff):
  - `scripts/phase1_clean_dataset.py`, `phase2_build_sequences.py`, `phase3_baseline_lstm.py`, `phase3_5_audit.py`
  - `data/processed/ibtracs_forecasting_base.csv` (17,778×10), `sequences/{X,y,train,val,test}.npy`, `normalization_stats.json`
  - `models/*.json` + `lstm_forecaster.pt`, `results/{PHASE3_REPORT,PHASE3_AUDIT}.md`, `results/*.json`, plots
  - This local pipeline uses **chronological** 3-hr-resolution sequences **without ERA5 features** (§9/§10 comparison).

---

## 2. Dataset Inventory

| Dataset | Rows | Cyclones | Time range | Verified readiness |
|---|---|---|---|---|
| IBTrACS raw (NI, v04r01) | 62,848 (+units row) | 1,858 | 1842–2025 | Raw, 174 cols |
| `ibtracs_clean.csv` | 18,168 | 471 | 1980–2025 | Cleaned track table, ~3-hourly |
| `ibtracs_with_era5.csv` | 5,481 | 151 | 2013-05-09 → 2025-12-02 | Era5-matched track+env |
| `master_dataset.csv` | 5,481 | 151 | same | + `pre_genesis_favorable` |
| train/validation/test.csv (+ `_cyclones.csv`) | — | 105/22/24 = 151 | 2013–2025 | Cyclone-grouped split manifests |
| `{train,val,test}_sequences.npz` | 2,275 / 378 / 423 | 67/14/16 = 97 | 2013–2025 | **P4-ready sequence arrays** |
| Classification multisource | 3,039 / 518 / 651 | — | 2013–2025 | For P3 |
| Classification image-only | 133 (93/19/21 split) | — | n/a (no timestamps) | For P3 |
| Detection | 133 (93/19/21) | — | n/a | For P2 (mock-ish) |
| ERA5 | 94 files, 5,728 time slices | — | 2013-05-09 → 2025-12-03 | For feature extraction |

---

## 3. IBTrACS Data Assessment

**Raw → clean (`get_ibtracs.py`, verified):**
- Agent/URL: NOAA NCEI IBTrACS v04r01 NI list (`…/ibtracs.NI.list.v04r01.csv`).
- Source hierarchy (IMD-first): wind `NEWDELHI_WIND` → `WMO_WIND` → `USA_WIND`; pressure `NEWDELHI_PRES` → `WMO_PRES`. Coalesced `wind_speed_kmh = kt × 1.852` (rounded 1 dp); IMD/RSMC New Delhi category assigned from wind.
- Filter: years 1980–2025, NIO basin. Result: 18,168 obs, 471 cyclones, 1980-10-10 … 2025-12-02, ~3-hourly.

**Completeness in `ibtracs_clean.csv` (verified):** wind `10,041/18,168` (55%), pressure `9,553/18,168` (53%), category `10,041`. ~45% of pre-split era rows carry no intensity estimate — expected for the pre-2000 record; fine for track, limiting for intensity targets.

**In `ibtracs_with_era5.csv` (the sourced subset, 5,481 rows):**
- `u_wind`, `v_wind`, `pressure_msl_hpa`: 5,481/5,481 (100%).
- `sst_celsius`: 3,943/5,481 (~72%; **1,538 missing**).
- `wind_speed_kmh`: 4,208/5,481 (~77%; 1,273 missing); `pressure_hpa`: 4,263 (~78%).

**Difference vs P1 QA claim:** `QA_REPORT.md` states completeness **100.0%** while simultaneously listing `sst: 1538` missing — the 100% figure is only correct for msl/u/v. **Doc bug to correct; SST is actually ~28% incomplete at source.**

---

## 4. ERA5 Data Assessment

Verified on the actual NetCDF files (h5py; the delivery is NetCDF4 = HDF5, magic `89 48 44 46`):

| Property | Verified value |
|---|---|
| Files | 94 (`era5_YYYY_MM.nc`), 1,103,681,364 B total |
| Grid | lat 30.0 → −5.0 (141 pts, 0.25°), lon 50.0 → 105.0 (221 pts, 0.25°) |
| Variables / units | `sst` (K), `msl` (Pa), `u10`, `v10` (m/s) — renamed `sst`, `pressure_msl_pa`, `u_wind`, `v_wind` at extraction |
| Temporal | 3-hourly, exactly hours {0,3,6,9,12,15,18,21}; 5,728 slices total; 2013-05-09 00:00 → 2025-12-03 21:00 |
| Raw missingness | `sst` ~36% NaN (sample 2013_05); `msl`/`u10`/`v10` 0% (u10 −19.7…21.7, v10 −19.5…23.4) |
| Extraction method | `xarray .sel(method="nearest")` spatial + temporal; `valid_time` decoded as seconds since **1970-01-01** (verified); `expver`/`number` dropped; K→C and Pa→hPa applied |

Notes:
- The last file contains 16 slices (2025-12-03 21:00) beyond the 2025-12-02 end stated in `QA_REPORT.md` — coverage is **≥** the doc's claim.
- P1's `sst` presence in master matches raw-missingness + `pre_genesis_favorable` logic; **rows where SST is missing have `pre_genesis_favorable=False`** even for genuinely favorable environments (verified: True=2,459, False=3,022). P4 should not train on `pre_genesis_favorable` as a feature without fixing this.
- Master rows whose storm position falls outside the ERA5 bbox (lon>105 or <50, lat>30 or <−5): **396 rows = 7.2%** — those `sst/u/v` are **edge-clipped** nearest values (see §8).

---

## 5. INSAT Imagery Assessment

- `data/raw/insat/` and `data/raw/insat_kaggle/` are **byte-identical duplicates** (redundant ~indented in the ZIP; some space could be saved).
- Image subsets: `CYCLONE_DATASET_FINAL` (140 images, ~357×357 RGB), `CYCLONE_DATASET_INFRARED` (136 jpg), `CYCLONE_DATASET` reference (143 files, ~1201×901 / 561×420).
- No timestamps / lat-lon / cyclone identity attached to any image — by design it is an image-only dataset and cannot be fused into the master table.
- **P3-relevant finding (flagged):** P1's `clean_kaggle_intensity.py` maps the Kaggle `label` column of `insat_3d_ds - Sheet.csv` directly to `wind_speed_kt`. In the raw CSV, `label` is the **image/cyclone number** (25.jpg→25, 27.jpg→27, …), i.e. an identifier, not an intensity. The resulting `image_only_kaggle/labels.csv` (133 rows) therefore carries **self-attributed wind speed = filename number**, which is almost certainly **mislabeled** as knots. The multisource (tabular) stream is unaffected. **P3 must re-validate the image-only labels before training.**

---

## 6. P1 Preprocessing Assessment (build → features → sequences)

Pipeline verified end-to-end from scripts:
`get_ibtracs.py` → `ibtracs_clean.csv` → `download_era5.py` → `{era5_YYYY_MM.nc}` → `extract_era5_at_points.py` → `ibtracs_with_era5.csv` → `build_datasets.py` → `master_dataset.csv` + `classification/…` + `detection/…` + `forecasting/{*_sequences.npz,*_metadata.csv}`.

`build_datasets.py` (Dataset C, the P4 part):
- Feature cols: `[lat, lon, wind_speed, pressure, sst, wind_u, wind_v]` — **exact P4 input spec**.
- Target cols: `[lat, lon, wind_speed]`; leads +6/+12/+24 h.
- Input window: 5 steps at 6-h spacing (t−24, t−18, t−12, t−6, t).
- **Critical deficiency observed:** per-storm `interpolate(method="time").ffill().bfill()` before windowing.
  1. Non-causal: a history step can be filled from an **observation after t** (leakage into the input window; also `bfill` backfills leading NaNs from future).
  2. Fabrication: wind-speed and pressure **targets** at +6/+12/+24 h can be interpolated values rather than real observations when the storm has gaps. The npz reports 0 NaNs — because gaps were filled, not dropped. P4 should treat interpolation-filled targets with care (mask or retain only rows where the target comes from a real observation).
- Split: 70/15/15 **random** by `cyclone_id` (not chronological) — see §10.

---

## 7. "P4-Ready" Forecasting Data — Direct Verification

`data/processed/forecasting/{train,val,test}_sequences.npz`:
- Shapes: train X `(2275, 5, 7)` / Y `(2275, 3, 3)`; val X `(378, 5, 7)` / Y `(378, 3, 3)`; test X `(423, 5, 7)` / Y `(423, 3, 3)`; float32; **0 NaNs** (all filled/interpolated).
- `features`: `['lat','lon','wind_speed','pressure','sst','wind_u','wind_v']`; `targets`: `['lat','lon','wind_speed']` — exact match to required P4 schema.
- Value ranges (train): lat 1.9–26.0°, lon 44.2–141.0°, wind 37.0–240.8 km/h, pressure 920–1006 hPa, sst 25.5–31.4 °C, u ±25.4, v ±23.4 m/s.
- Metadata CSV (16 cols): `cyclone_id`, `t_zero`, `t_minus_24h`, `t_plus_24h`, `origin_{lat,lon,wind}`, `target_{6h,12h,24h}_{lat,lon,wind}` → sufficient to reconstruct the anchor points and shadow-test forecasts.
- Loader provided: `src/data/forecasting_dataset.py` (`CycloneForecastingDataset`) exposes raw arrays + `get_feature_stats()` for train-only normalization; **no pre-normalized npz or shipped stats** — normalization is deliberately left to P4.

**Verdict:** The forecasting npz is a valid, directly trainable P4 dataset matching the stated input/target contract.

---

## 8. Feature Check (availability for the P4 model)

| Required input | Source | Provider | Availability in npz | Notes |
|---|---|---|---|---|
| latitude / longitude | IBTrACS track | P1 | 100% | lon up to 141°E (outside ERA5 bbox) |
| wind_speed | IBTrACS IMD/WMO | P1 | 77% real + interpolated fill | interpolated where missing |
| pressure | IBTrACS IMD/WMO | P1 | 78% real + interpolated fill | |
| sst | ERA5 (K→C) | P1 | ~72% real + interpolated fill; else **edge-clipped** for 7.2% rows | raw gaps ≈ 36% |
| u_wind / v_wind (10 m) | ERA5 | P1 | 100%; edge-clipped for 7.2% rows | |
| (targets) lat/lon/wind +6/+12/+24h | IBTrACS | P1 | present | may be interpolated |

- The P4 schema is fully covered. No feature is missing outright.
- **Residual risks:** (a) 7.2% of rows are spatially outside the 50–105°E / 30°N bbox so their ERA5 features are clipped to bbox edge values (SST at 105°E, etc.); (b) filled-vs-real values are not flagged in npz — a flag column would be a cheap P1 fix; (c) `lon` crossing the 100–105° boundary is not wrapped — model must handle longitude as circular.

---

## 9. Timestamp Alignment Check

- `ibtracs_with_era5.csv` timestamps are all on the **exact synoptic grid**: minute=0 in 5,481/5,481; hours {0,3,6,9,12,15,18,21} with one straggler at 14:00 (1 row, matches ERA5's nearest 15:00 slot).
- ERA5 files are 3-hourly on those same synoptic hours → the `method="nearest"` extraction is effectively exact (≤ 3 h) apart from one 1-h offset row.
- Origin timestamps (t_zero) of training sequences: 2013-05-10 18:00 → 2025-11-29 06:00, all on the synoptic grid.
- **Alignment verdict: PASS.** Clean 3-hourly UTC grid shared by all sources.

---

## 10. Train / Val / Test Split Analysis

Verified against the actual CSVs and npz metadata:
- `train_cyclones.csv` 105, `validation_cyclones.csv` 22, `test_cyclones.csv` 24 → 151 cyclones (100% of matched set) — matches QA_REPORT (69.5% / 14.6% / 15.9%).
- But only 97 cyclones actually appear in sequences: train 67, val 14, test 16. The other 54 storms are too short to yield a 5-step input window plus a +24 h target. This is legitimate but under-reported (QA says 3,076 sequences, implying all 151 were used).
- **Temporal interleaving (key finding):** the split is **random by cyclone**, so all three sets span 2013–2025 (train 2013–2025, val 2013–2025, test 2013–2025). Test storms are contemporaneous with training storms. For "train on history, forecast the future" realism, this is weaker than a chronological split (the repo's independent phase-2 pipeline used chronological 243/52/52). Cyclone-grouping itself (no splits share a storm) is correct.
- 54% of matched rows don't produce a sequence; decreasing horizon from +24 h to +6 h only would yield more usable anchors.

---

## 11. Data Leakage Check

| Check | Result (verified) |
|---|---|
| Cyclone (`cyclone_id`/SID) in multiple splits | **PASS** — 0 overlap (npz train∩val∩test = ∅; verified numerically) |
| Future observation inside input window | **PASS at row level** — inputs are t−24…t, targets t+6/12/24 (checked in metadata scheme) |
| **Non-causal interpolation in build_datasets** | **FAIL (potential)** — `interpolate().ffill().bfill()` fills history entries from post-t observations and fills target values from surrounding (future) observations. No filler-source flag exists to distinguish real vs synthetic. |
| Statistics / normalization sourced from test | **PASS** — P1 ships raw arrays; normalization deferred to P4 (loader's `get_feature_stats` = train-only if called on train split). |
| Split temporal overlap | **FAIL (weakness)** — random split; train/val/test eras interleave. Cyclone-disjointness prevents hard leakage but doesn't enforce a forecasting-realistic temporal boundary. |

**Bottom line:** no outright same-storm leakage; the remaining risks are non-causal gap-filling and the temporally interleaved split. Both are P4-controllable (mask unresolved targets; re-split chronologically).

---

## 12. P4 Readiness Score

**Readiness: 8 / 10 — ACTIONABLE with documented caveats.**

- **+** Required 7-feature input schema, 3-target × 3-horizon output, exact synoptic 3-hourly alignment, cyclone-disjoint splits, zero-NaN arrays, loader + metadata, reproduction scripts, QA figures.
- **−** (1) Non-causal gap-filling / fabricated targets not flagged; (2) random-not-chronological split; (3) SST ~28% missing source and no flag; (4) 7.2% edge-clipped ERA5 features; (5) only 97 usable cyclones for the 24-h task (small supervised corpus); (6) the repo's own phase-3 LSTM vs- persistence shows **LSTM better at 12 h/24 h but worse than persistence at 6 h** (track error 104.8 vs 78.4 km) — set realistic baseline expectations.
- **Needs P1 fix (optional but cheap):** add a `target_is_interpolated` / `feature_edge_clipped` flag and re-run builds; correct the QA_REPORT completeness figure.

---

## 13. Recommended Next Steps (P4 Phase)

1. **Adopt the P1 forecasting npz as the primary training corpus** (start). Rebuild sequences with a **causal** fill (rolling backward only, no bfill) and a `synthetic` flag; drop or mask any row whose +6/12/24 h target is interpolated.
2. **Re-split chronologically by `cyclone_id`** (reuse the repo's phase-2 split philosophy) so the test set is the most-recent era; report both splits.
3. Normalize with **train-only statistics** (use the loader's `get_feature_stats()`), encode longitude as circular, and clip/flag ERA5 edge rows.
4. Baseline stack: persistence (numbers exist in `results/`), plain LSTM/GRU (local phase-3 already gives a config: hidden 64, L 1, dropout 0, seq 5×7 → 3×3), then attention/Seq2Seq. Evaluate MAE + Haversine track error per horizon.
5. Add small feature cross-check (e.g., verify X wind values against `ibtracs_with_era5.csv` for 10 random samples before trusting filled rows).
6. Hand P2/P3 owners the §5/§8 flags (image-label semantics; mock bbox) — outside P4 scope but part of a clean handoff.
7. Productionize `src/forecasting/inference.py` path to consume P1 arrays instead of the local 9×7 chrono pipeline when delivering the final forecast service.

---

## P1 HANDOFF STATUS

| # | Item | Status | Action |
|---|---|---|---|
| 1 | Project structure / ZIP completeness | PASS | Keep ZIP as the canonical source; move extras from temp to `data/` |
| 2 | IBTrACS clean + sourced subset | PASS (density caveats) | Document 45% pre-split intensity sparsity |
| 3 | ERA5 files / grid / units | PASS | Correct last-file end date in QA doc |
| 4 | ERA5–IBTrACS alignment | PASS | Confirm the single 14:00 row is intentional |
| 5 | `master_dataset.csv` (151 cyclones) | PASS | Fix QA "100% completeness" wording (SST 1,538 missing) |
| 6 | P4 forecasting npz (5×7 → 3×3) | **PASS (trainable)** | Add synthetic/edge flags; rebuild causally |
| 7 | Feature coverage for P4 | PASS | Watch lon>105, circular lon, edge clipped sst/u/v |
| 8 | Timestamp grid (3-hourly synoptic) | PASS | Reuse grid for real-time serving |
| 9 | Train/val/test cyclone disjointness | PASS | Verified 0 overlap |
| 10 | Split temporal realism | **PARTIAL** | Re-split chronologically for forecasting realism |
| 11 | Leakage (causality / normalization) | **PARTIAL** | Mask interpolated targets; train-only stats |
| 12 | Classification image-only labels | **FAIL** | Image number misread as knots → P3 revalidation |
| 13 | Detection manifests (mock_bbox) | PARTIAL | Confirm P2 accepts contract placeholders |
| **Overall** | **READY (with caveats)** | 8/10 | Proceed to P4 training with steps in §13 |

*P4 forecasting sign-off: proceed with the P1 forecasting npz as primary data after applying causal-fill + chronological re-split (§13). No blocking data defect.*