# OFFLINE MODE AUDIT

**Date:** 2026-08-31
**Status:** PASS

## Categories

### 1. Fully Offline Functionality
- Backend inference (P2/P3/P4 models) — no network required
- All model weights loaded from local zip/files
- Forecasting: EXP005 GRU runs on CPU without network
- Detection: MobileNetV3 on reference frame from zip
- Classification: ResNet18 on reference frame from zip
- Health endpoint: reports offline=True

### 2. Local-Backend Functionality
- Frontend runs against localhost backend
- API calls to /api/* via POST
- All data flow is local

### 3. Internet-Dependent Functionality
- Online map tiles (Leaflet/OpenStreetMap)
- Graceful degradation: map shows without tiles if offline
- No map tiles claimed as offline

### 4. Demonstration/Static Functionality
- Reference INSAT-3D frame from zip (NOT a live feed)
- DEMO_HISTORY constant for initial load
- Provenance labels: "reference frame from PS70-main.zip"

## Honest Labeling Verified

| Element | Label | Status |
|---------|-------|--------|
| Reference frame | "reference frame from PS70-main.zip" | Honest |
| DEMO_HISTORY | Used as fallback, not claimed as live | Honest |
| Forecast cone | "illustrative, not calibrated uncertainty" | Honest |
| Landfall | "deterministic server-side heuristic, not ML" | Honest |
| Risk score | "deterministic server-side heuristic, not ML" | Honest |
| Map tiles | Online (OpenStreetMap), graceful degradation | Honest |

## Backend Health Response

```json
{
  "offline": true,
  "model_ready": true,
  ...
}
```

## Verdict

**PASS** — Offline capabilities are honest; online-only features degrade gracefully.
