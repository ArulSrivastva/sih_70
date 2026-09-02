# P4 PHASE 1 FINAL REPORT - FORECASTING DATASET AUDIT & CANONICAL DATASET CREATION

Status at top: **P4_PHASE1_STATUS: PASS_WITH_WARNINGS**

## 1. Scope
Read-only audit of P1's forecasting delivery (PS70-main.zip), leakage detection, canonical dataset creation. Nothing outside p4_forecasting/ was modified. No models were trained.

## 2. File inventory & hashes
See `p4_forecasting/audit/P4_FILE_INVENTORY.md` (23 staged + 94 ERA5 files hash-recorded). ZIP SHA256: `f015bc3b98fe4a99d0a75e65cc1391bc97bca70b1d2b8df4867ad41020a6e5ac`. Baseline stored in p1_source_hashes.json.

## 3. NPZ inspection
X=(2275,5,7)/(378,5,7)/(423,5,7), Y=(3,3) per sample, float32, 0 NaN, 0 inf. See NPZ_AUDIT_REPORT.md. Keys: X, Y, features, targets.

## 4. Feature order
FEATURE_ORDER_VERIFIED = YES. Features: lat, lon, wind_speed, pressure, sst, wind_u, wind_v.

## 5. Target structure
Target structure verified = YES. Y=(N,3,3) horizons +6/+12/+24, columns [lat, lon, wind_speed]. Max abs diff vs metadata across ALL 3076 samples = 6.10e-06 (tol 1e-5).

## 6. Temporal structure
Window = exactly 24h (t-24h..t0); horizons +6/+12/+24. Max window deviation 0.000 h, max horizon deviation 0.000 h. Off-grid t_zero samples: 1. See TEMPORAL_AUDIT.md.

## 7. LEAKAGE - feature/target separation
Confirmed leakage samples: 0 / 3076. All inputs end at t_zero, targets start at t+6h.

## 8. LEAKAGE - splits
Cyclone overlap: train^val=0, train^test=0, val^test=0. Master-level P1 cyclone lists also disjoint. See SPLIT_LEAKAGE_AUDIT.md.

## 9. Spiral/temporal split quality
P1 npz split is a random, cyclone-grouped split (each split contains 2013..2025 cyclones). Not chronological. Secondary chronological split created in canonical_chrono/.

## 10. Duplicates
Zero duplicate (cyclone_id, t_zero) pairs, zero duplicate X/Y rows, zero identical X+Y rows across train/val/test.

## 11. Missing values
npz contains 0 NaN/inf. Source-level (master_dataset.csv aligned) missingness: sst 28.1% of rows, wind 23.2%, pressure 22.2%, u/v 0%. Synthetic (filled) cells are reported per feature and per SST lag in TEMPORAL_AUDIT.md / LEAKAGE_AUDIT.md.

## 12. Physical sanity
All values within nominal physical ranges. 0 violations. SST is in deg C, not Kelvin (see PHYSICAL_SANITY_REPORT.md).

## 13. Normalization
NPZ_NORMALIZATION = NONE. Samples are raw physical units (wind up to 241 km/h, pressure from 920 hPa, sst up to 31.7 deg C).

## 14. Master dataset consistency
For 20 sampled origins per split, all X cells and Y target cells agree with master_dataset.csv values where a real observation exists (max input diff table in code; all sensible). X last step == metadata origin lat/lon/wind to <=1e-5.

## 15. Non-causal interpolation detection
build_datasets.py lines 148 & 164 use interpolate(time).ffill().bfill(). Any input or target cell not backed by a real master observation is flagged NON_CAUSAL_RISK. Affected: 1307/3076 samples (inputs), 659/3076 samples (targets). Full flags in canonical/sample_quality.csv.

## 16. Canonical dataset creation
Created p4_forecasting/canonical/{train,val,test}.npz (+ metadata, + sample_quality.csv). Values preserved byte-for-byte from P1 (no silent alteration). Exclusions due to LEAKAGE/INVALID: 0.

## 17. Canonical policy
CLEAN / MISSING_ENVIRONMENTAL_DATA / NON_CAUSAL_RISK / LEAKAGE / INVALID statuses assigned per sample. NON_CAUSAL_RISK samples are KEPT (flagged), NOT discarded.

## 18. Chronological split
p4_forecasting/canonical_chrono/ created by cyclone start date (70/15/15). Counts: train=68, val=15, test=14 cyclones; train=2259, val=416, test=401 sequences (manifest: split_manifest.csv). Each cyclone appears in exactly one split (0 violations). Temporal ordering verified: all train storm starts precede all val starts, which precede all test starts.

## 19. No normalization in canonical
See section 13.

## 20. DATA CONTRACT
p4_forecasting/canonical/P4_DATA_CONTRACT.md defines shapes, indices, time steps, horizons, units (sst = deg C as delivered), non-normalization, and exclusion policy.

## Warnings
- non-causal interpolation risk present (inputs=1307, synthetic targets=659)

## Files produced
- p4_forecasting/audit/P4_FILE_INVENTORY.md
- p4_forecasting/audit/NPZ_AUDIT_REPORT.md
- p4_forecasting/audit/TEMPORAL_AUDIT.md
- p4_forecasting/audit/LEAKAGE_AUDIT.md
- p4_forecasting/audit/SPLIT_LEAKAGE_AUDIT.md
- p4_forecasting/audit/TEMPORAL_SPLIT_QUALITY.md
- p4_forecasting/audit/PHYSICAL_SANITY_REPORT.md
- p4_forecasting/audit/leakage_results.csv
- p4_forecasting/canonical/{(train,val,test).npz, *_metadata.csv, sample_quality.csv, P4_DATA_CONTRACT.md}
- p4_forecasting/canonical_chrono/{(train,val,test).npz, *_metadata.csv, split_manifest.csv}
- p4_forecasting/reports/p4_phase1_summary.json
