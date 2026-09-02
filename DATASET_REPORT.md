# IBTrACS North Indian Ocean Dataset Report

## 1. File Name and Size

| Property | Value |
|----------|-------|
| File | `ibtracs.NI.list.v04r01.csv` |
| Size | ~27.9 MB |

## 2. Number of Rows

- **62,848** observation rows (excluding the 2 header rows: column names + units)

## 3. Number of Cyclone/Storm IDs

- **1,858** unique storm IDs (`SID`)
- Observations per storm range from 1 to 191

## 4. All Column Names (174 columns)

`SID, SEASON, NUMBER, BASIN, SUBBASIN, NAME, ISO_TIME, NATURE, LAT, LON, WMO_WIND, WMO_PRES, WMO_AGENCY, TRACK_TYPE, DIST2LAND, LANDFALL, IFLAG, USA_AGENCY, USA_ATCF_ID, USA_LAT, USA_LON, USA_RECORD, USA_STATUS, USA_WIND, USA_PRES, USA_SSHS, USA_R34_NE, USA_R34_SE, USA_R34_SW, USA_R34_NW, USA_R50_NE, USA_R50_SE, USA_R50_SW, USA_R50_NW, USA_R64_NE, USA_R64_SE, USA_R64_SW, USA_R64_NW, USA_POCI, USA_ROCI, USA_RMW, USA_EYE, TOKYO_LAT, TOKYO_LON, TOKYO_GRADE, TOKYO_WIND, TOKYO_PRES, TOKYO_R50_DIR, TOKYO_R50_LONG, TOKYO_R50_SHORT, TOKYO_R30_DIR, TOKYO_R30_LONG, TOKYO_R30_SHORT, TOKYO_LAND, CMA_LAT, CMA_LON, CMA_CAT, CMA_WIND, CMA_PRES, HKO_LAT, HKO_LON, HKO_CAT, HKO_WIND, HKO_PRES, KMA_LAT, KMA_LON, KMA_CAT, KMA_WIND, KMA_PRES, KMA_R50_DIR, KMA_R50_LONG, KMA_R50_SHORT, KMA_R30_DIR, KMA_R30_LONG, KMA_R30_SHORT, NEWDELHI_LAT, NEWDELHI_LON, NEWDELHI_GRADE, NEWDELHI_WIND, NEWDELHI_PRES, NEWDELHI_CI, NEWDELHI_DP, NEWDELHI_POCI, REUNION_LAT, REUNION_LON, REUNION_TYPE, REUNION_WIND, REUNION_PRES, REUNION_TNUM, REUNION_CI, REUNION_RMW, REUNION_R34_NE, REUNION_R34_SE, REUNION_R34_SW, REUNION_R34_NW, REUNION_R50_NE, REUNION_R50_SE, REUNION_R50_SW, REUNION_R50_NW, REUNION_R64_NE, REUNION_R64_SE, REUNION_R64_SW, REUNION_R64_NW, BOM_LAT, BOM_LON, BOM_TYPE, BOM_WIND, BOM_PRES, BOM_TNUM, BOM_CI, BOM_RMW, BOM_R34_NE, BOM_R34_SE, BOM_R34_SW, BOM_R34_NW, BOM_R50_NE, BOM_R50_SE, BOM_R50_SW, BOM_R50_NW, BOM_R64_NE, BOM_R64_SE, BOM_R64_SW, BOM_R64_NW, BOM_ROCI, BOM_POCI, BOM_EYE, BOM_POS_METHOD, BOM_PRES_METHOD, NADI_LAT, NADI_LON, NADI_CAT, NADI_WIND, NADI_PRES, WELLINGTON_LAT, WELLINGTON_LON, WELLINGTON_WIND, WELLINGTON_PRES, DS824_LAT, DS824_LON, DS824_STAGE, DS824_WIND, DS824_PRES, TD9636_LAT, TD9636_LON, TD9636_STAGE, TD9636_WIND, TD9636_PRES, TD9635_LAT, TD9635_LON, TD9635_WIND, TD9635_PRES, TD9635_ROCI, NEUMANN_LAT, NEUMANN_LON, NEUMANN_CLASS, NEUMANN_WIND, NEUMANN_PRES, MLC_LAT, MLC_LON, MLC_CLASS, MLC_WIND, MLC_PRES, USA_GUST, BOM_GUST, BOM_GUST_PER, REUNION_GUST, REUNION_GUST_PER, USA_SEAHGT, USA_SEARAD_NE, USA_SEARAD_SE, USA_SEARAD_SW, USA_SEARAD_NW, STORM_SPEED, STORM_DIR`

## 5. Key Column Mapping

| Attribute | Column(s) | Units |
|-----------|-----------|-------|
| Storm/Cyclone ID | `SID` | -- |
| Timestamp | `ISO_TIME` | UTC `YYYY-MM-DD HH:MM:SS` |
| Latitude | `LAT` | degrees_north |
| Longitude | `LON` | degrees_east |
| Wind Speed (primary) | `WMO_WIND` | knots |
| Wind Speed (backup) | `USA_WIND`, `NEWDELHI_WIND` | knots |
| Minimum Pressure (primary) | `WMO_PRES` | mb |
| Minimum Pressure (backup) | `USA_PRES`, `NEWDELHI_PRES` | mb |
| Basin | `BASIN` | `NI` (North Indian), `WP`, `NA` |
| Storm Type/Category | `NATURE` | `TS`, `NR`, `DS`, `MX`, `ET` |

**Note:** Multiple agencies provide wind/pressure estimates. `WMO_WIND` is the authoritative best-track value but is missing for ~87% of rows. `USA_WIND` (missing ~86%) and `NEWDELHI_WIND` (missing ~85%) serve as fallbacks. The combined availability across agencies covers most observations.

## 6. Temporal Resolution

- **Dominant resolution: 3-hourly** (96.6% of consecutive observations are 3 hours apart)
- Some 1-hour and 2-hour intervals exist (~0.4% combined)
- Median time difference: **3.0 hours**

## 7. Missing-Value Percentage (Key Columns)

| Column | Missing | % |
|--------|---------|---|
| `SID` | 0 | 0.0% |
| `ISO_TIME` | 0 | 0.0% |
| `LAT` | 0 | 0.0% |
| `LON` | 0 | 0.0% |
| `BASIN` | 0 | 0.0% |
| `NATURE` | 0 | 0.0% |
| `WMO_WIND` | 54,918 | **87.4%** |
| `WMO_PRES` | 54,620 | **86.9%** |
| `USA_WIND` | 53,894 | **85.8%** |
| `USA_PRES` | 46,486 | **74.0%** |
| `STORM_SPEED` | 42 | 0.1% |
| `STORM_DIR` | 42 | 0.1% |

**Critical observation:** Wind and pressure columns are heavily sparse because IBTrACS stores values only when a given agency issued an estimate. No single agency covers all storms across all time periods. For modeling, a strategy to combine values across agencies (e.g., coalesce `WMO_WIND` -> `NEWDELHI_WIND` -> `USA_WIND`) is needed.

## 8. Duplicate Records

- **0 exact duplicate rows** found across the full 62,848 rows.

## 9. Latitude/Longitude Ranges

| Dimension | Min | Max |
|-----------|-----|-----|
| Latitude | 0.70°N | 83.00°N |
| Longitude | -87.70° | 163.70° |

**Note:** Some extreme lat/lon values (e.g., 83°N, -87°W) likely represent post-tropical/extratropical tracks that drifted far from the Indian Ocean basin. The core North Indian Ocean region spans roughly 0-30°N, 40-100°E.

## 10. Wind Speed and Pressure Ranges

| Agency | Wind Speed Range | N | Pressure Range | N |
|--------|-----------------|---|---------------|---|
| WMO | 3.0–140.0 kts | 7,930 | 890.0–1,012.0 mb | 8,228 |
| USA | 10.0–150.0 kts | 16,362 | 898.0–1,014.0 mb | 7,110 |
| New Delhi | 3.0–140.0 kts | 9,467 | 912.0–1,011.0 mb | 8,605 |
| CMA | 17.0–165.0 kts | 2,914 | 890.0–1,010.0 mb | 3,184 |
| HKO | 20.0–130.0 kts | 2,274 | 893.0–1,010.0 mb | 2,274 |
| Tokyo | 35.0–120.0 kts | 1,260 | 890.0–1,012.0 mb | 3,225 |

## 11. North Indian Ocean Cyclone Count

- **1,858** unique cyclone/storm tracks in the dataset
- Time span: **1842–2025** (184 years of data)
- Sub-basin distribution:
  - Bay of Bengal (`BB`): 42,415 obs (67.5%)
  - Arabian Sea (`AS`): 15,426 obs (24.5%)
  - Maritime Continent (`MM`): 4,525 obs (7.2%)
  - North Atlantic (`NA`): 449 obs
  - Other (`GM`, `CS`): 33 obs

- Basin column values: `NI` (57,841), `WP` (4,525), `NA` (482)

## 12. Chronological Ordering

- **All 1,858 storms are chronologically ordered** per cyclone (0 storms with out-of-order timestamps).
- Observations within each storm are sorted by `ISO_TIME` ascending.

## Summary of Key Facts for Modeling

- **1,858 storms**, **62,848 observations**, **3-hourly** resolution spanning 1842–2025.
- Core position (`LAT`, `LON`, `ISO_TIME`, `SID`) and identifier columns (`BASIN`, `NATURE`) are **100% complete**.
- Wind and pressure require **cross-agency coalescing** due to per-agency sparsity (each is 74–87% missing).
- No duplicate rows exist; data is chronologically ordered per storm.
