# IBTrACS North Indian Ocean — Data Quality Report

**Phase 1: Clean Forecasting Dataset**

Generated: 2026-08-26 09:14:06

---

## 1. Filtering Decisions

### 1.1 Temporal Filter

| Parameter | Value |
|-----------|-------|
| Year range | 1980–2025 |
| Rationale | Satellite era; reliable wind/pressure estimates from IMD and JTWC |

**Before:** 62,848 observations, 1,858 storms (1842–2025)
**After:** 18,168 observations, 471 storms

### 1.2 Geographic Filter

| Parameter | Value |
|-----------|-------|
| Latitude | 0–30°N |
| Longitude | 40–100°E |
| Method | Keep all observations for any storm that had ≥1 point within bounds |

**After:** 18,168 observations, 471 storms

### 1.3 Nature (Storm Type) Filter

| Retained | Count | Rationale |
|----------|-------|-----------|
| TS (Tropical Storm) | 13,402 | Organized storms with sustained winds ≥ 34 kt; primary forecasting target |
| NR (Named Storm/Depression) | 3,215 | Tropical depressions and named storms below TS threshold |
| DS (Disturbance) | 1,161 | Organized tropical disturbances that may develop |

| Dropped | Count | Rationale |
|---------|-------|-----------|
| MX (Mixed) | 387 | Ambiguous classification; inconsistent across agencies |
| ET (Extratropical) | 3 | No longer tropical; different dynamics not relevant for cyclone forecasting |

**After:** 17,778 observations, 471 storms

### 1.4 Combined Filter Summary

| Stage | Observations | Storms |
|-------|-------------|--------|
| Raw IBTrACS | 62,848 | 1,858 |
| After temporal (1980–2025) | 18,168 | 471 |
| After geographic (0–30°N, 40–100°E) | 18,168 | 471 |
| After nature filter (TS/NR/DS) | 17,778 | 471 |

---

## 2. Wind and Pressure Source Availability

### 2.1 Per-Source Availability (Post-Filter)

| Source | Available | Missing | % Available |
|--------|-----------|---------|-------------|
| NEWDELHI_WIND (IMD) | 9,253 | 8,525 | 52.0% |
| WMO_WIND (Best-track) | 7,464 | 10,314 | 42.0% |
| USA_WIND (JTWC) | 13,318 | 4,460 | 74.9% |
| NEWDELHI_PRES (IMD) | 8,392 | 9,386 | 47.2% |
| WMO_PRES (Best-track) | 7,283 | 10,495 | 41.0% |
| USA_PRES (JTWC) | 6,766 | 11,012 | 38.1% |

### 2.2 Source Overlap

Wind sources are largely complementary: different agencies cover different eras and storms. Overlap is limited, confirming that a fallback hierarchy is appropriate rather than blind mixing.

### 2.3 Source Hierarchy (IMD-Focused)

| Priority | Wind | Pressure |
|----------|------|----------|
| 1 (Primary) | NEWDELHI_WIND | NEWDELHI_PRES |
| 2 (Fallback) | WMO_WIND | WMO_PRES |
| 3 (Last resort) | USA_WIND | USA_PRES |

**Rationale:** For a North Indian Ocean forecasting system, IMD (New Delhi) is the authoritative Regional Specialized Meteorological Centre (RSMC). WMO best-track provides the consensus value. JTWC (USA) provides additional coverage, especially for older storms.

**After fallback:**
- Wind available: 16,554/17,778 (93.1%)
- Pressure available: 11,728/17,778 (66.0%)

### 2.4 Units

| Variable | Original | Converted | Notes |
|----------|----------|-----------|-------|
| Wind speed | knots | km/h | Multiplied by 1.852 |
| Pressure | mb | hPa | 1 mb = 1 hPa (no conversion needed) |

---

## 3. Temporal Statistics

### 3.1 Observation Frequency

| Interval | Count | Percentage |
|----------|-------|------------|
| 3-hour | 16,993 | 98.2% |
| 6-hour | 20 | 0.1% |
| Other | 294 | 1.7% |
| **Total** | **17,307** | **100%** |

### 3.2 Major Gaps

- Storms with gaps >12 hours: 35 / 471 (7.4%)

---

## 4. Sequence Availability for LSTM Input

### 4.1 Requirements

| Target | Input steps | Future steps | Total min observations |
|--------|-------------|--------------|----------------------|
| +6h | 8 (24h) | 2 | 10 |
| +12h | 8 (24h) | 4 | 12 |
| +24h | 8 (24h) | 8 | 16 |

### 4.2 Usable Storms

| Target | Usable storms | % of total |
|--------|--------------|------------|
| +6h | 415 | 88.1% |
| +12h | 403 | 85.6% |
| +24h | 371 | 78.8% |

---

## 5. Summary

| Metric | Value |
|--------|-------|
| Usable storms | 471 |
| Usable observations | 17,778 |
| Time span | 1980–2025 |
| Avg obs per storm | 37.7 |
| Wind availability (post-fallback) | 93.1% |
| Pressure availability (post-fallback) | 66.0% |
| Chronological order | Verified (0 storms out of order) |

---

## 6. Limitations

1. **Wind/pressure sparsity:** Even after fallback, ~7% of observations lack wind speed. This limits supervised training data.

2. **Agency mixing:** The fallback hierarchy resolves source conflicts by preferring IMD, but introduces inconsistency: different agencies use different estimation methods, leading to potential jumps in wind/pressure values at agency transitions.

3. **No interpolation:** Missing wind/pressure values are preserved as NaN. Downstream models must handle missing data or apply imputation strategies.

4. **Geographic boundaries:** Storms that pass through the region are included with all their observations, including points outside 0–30°N / 40–100°E. This captures full cyclone lifecycles but may include extratropical transitions.

5. **Historical data quality:** Pre-satellite observations (< ~1970) are excluded by the temporal filter, but early satellite-era estimates (1980s) may still have larger uncertainties.

6. **Temporal resolution:** Dominant 3-hour resolution is suitable for tracking but may miss rapid intensification events occurring between observations.

---

## 7. Output Files

| File | Description |
|------|-------------|
| `ibtracs_forecasting_base.csv` | Clean dataset with canonical columns |
| `DATA_QUALITY_REPORT.md` | This report |
| `example_cyclone_tracks.png` | Visualization of 3–5 example cyclone tracks |
