# P1 Data Audit

**Status: PASS_WITH_WARNINGS** (structure is sound; synthetic-label and provenance warnings apply to downstream consumers, and the P1 dataset population differs from what P4 reports used.)

All values in this report were recomputed by reading `PS70-main.zip` contents directly (`data/metadata/*`, `data/processed/*`) and cross-checked against the delivered QA report (`AUDIT_P1_DELIVERY.md`, `DATASET_REPORT.md`).

## 1. Headline counts — match QA report exactly

| Item | Independently computed | QA claim | Match |
|---|---|---|---|
| Master rows | 5,481 | 5,481 | ✓ |
| Master cyclones | 151 | 151 | ✓ |
| Time span | 2013-05-09 18:00 → 2025-12-02 18:00 | 2013-05-09 → 2025-12-02 | ✓ |
| Sub-basins | {AS, BB, MM} | — | — |
| Exact-duplicate rows | 0 | 0 | ✓ |
| Duplicate (cyclone_id, timestamp) | 0 | 0 | ✓ |
| Split rows | train 3,911 / val 752 / test 818 | — | ✓ (sum = 5,481) |
| Split cyclones | 105 / 22 / 24 | 105 / 22 / 24 | ✓ |
| Split disjointness | no cyclone in ≥2 splits; union = 151 | ✓ | ✓ |
| ERA5 join coverage | 5,481 / 5,481 = 100% | “all rows have ERA5” | ✓ |
| Clean IBTrACS | 18,168 rows / 471 cyclones / 1980–2025 | — | — |
| Clean cadence | median 3 h (min 0.5, max 3) | — | — |

## 2. Missingness (master)

| Column | Rows missing | Note |
|---|---|---|
| `sst` | **1,538 (28.1%)** | QA explicitly lists `sst missing: 1538`; **the README-level "Completeness 100.0%" wording is inconsistent with this and must not be quoted as a claim.** |
| `wind_speed` | 1,273 | genesis/lull records |
| `category` | 1,273 | follows wind |
| `pressure` | 1,218 | |
| everything else | 0 | unique ids, positions, ERA5 fields fully matched |

- `multisource_test.csv` repeats the sst gap: 207/651 (31.8%) test rows have NaN sst; pressure 3/651.
- Physical plausibility: lat∈[1.9,29.2], lon∈[41.8,141.0], wind∈[27.8,240.8], pressure∈[920,1008], sst∈[25.43,31.83], wind_u∈[-25.4,24.8], wind_v∈[-19.5,23.4], pressure_msl∈[944.8,1016.5]. **Zero out-of-range values.**

## 3. Class/label sanity

- IMD thresholds derived from wind reproduce the master `category` column on 100% of annotated rows (category-to-wind consistency exact).
- `clean_kaggle_intensity.py` labels: parsed from the Kaggle INSAT-3D sheet (`insat_3d_ds - Sheet.csv`, 136 rows, `img_name → label`); wind km/h = kt × 1.852; IMD category thresholds match (<31 LPA, <50 D, <62 DD, <89 CS, <118 SCS, <166 VSCS, <222 ESCS, ≥222 SuCS). The Kaggle set has **no timestamp / lat / lon / cyclone identity** — it cannot be joined to the master ERA5 track, so the image-only study is fully disconnected from the multisource track data (this is why the P3 image model is evaluated on its own 21-image test).

## 4. Dataset A (detection) — **synthetic labels, flagged**

- `cyclone_detected=True` in 133/133 rows.
- `mock_bbox` identical `[420, 190, 600, 370]` in 133/133 rows.
- `structural_pattern` (3 classes) is **derived from wind_speed_kt by a threshold rule** (checked: 99.25% consistent with a contiguous-threshold rule; the residual is one boundary row). `category` likewise derived from wind.
- Consequence: P2 has **no ground-truth detection/bbox/pattern labels**. “Detection accuracy” cannot be evaluated; only pattern/category classification on derived labels is measurable (which is what `detection_metrics.json` contains).

## 5. Dataset C (forecasting) — **population and fill warnings**

- Sequence sizes (6 h cadence): train 2,275 / val 378 / test 423 (cyclones 67/14/16). Only **97 of 151** master cyclones yield sequences (shorter storms with <5 history or <+24 h target are dropped).
- X/Y have 0 NaN (all NaN sst cells were filled via `interpolate(…).ffill().bfill()`). **`bfill` may back-fill forward-in-time values within a storm** (sst). This is only a train-side feature-window concern where causal ordering was preserved by the P4 clean fix; the risk is documented, not exercised here.
- Test time-range: first test timestamp 2013-05-29; test overlaps train period by design (random cyclone-level split, **not** chronological). Chronological-leakage is *not* present, but `test` is not “predict the future of train” either — see `LEAKAGE_AUDIT.md`.

## 6. Delivered smoothing claims

- `ibtracs_clean.csv` vs `data/metadata/ibtracs_with_era5.csv` vs `master_dataset.csv` are consistent (rename-only join). `data/processed/master_dataset.csv` is byte-identical to `data/metadata/master_dataset.csv` (row-level equality verified).

## Verdicts

| Item | Verdict |
|---|---|
| Structural integrity (counts, splits, ranges, uniqueness, ERA5 coverage) | PASS |
| Reported “Completeness 100.0%” wording | INCONSISTENT (sst missing 1,538 documented in the same audit) |
| Detection labels as ground truth | NOT VALID — synthetic (see P2) |
| Kaggle↔track provenance | LOST (no storm identity/timestamp) |
| Dataset C causal purity | PASS_WITH_WARNINGS (bfill documented; clean pass verified in P4) |
| Deliverable-population reproducibility | WARN (P1 423 ≠ canonical_chrono 401 ≠ feature_dataset 198 test rows) |