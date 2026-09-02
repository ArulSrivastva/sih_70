# GEO SCOPE AUDIT — North Indian Ocean Policy

**Date:** 2026-08-31
**Status:** PASS

## Geographic Policy

- Latitude: [0, 30] N
- Longitude: [40, 100] E (degrees East, canonical [0,360) convention)
- Source: IMD/RSMC New Delhi mandate + IBTrACS NI-basin extraction

## Validation Implementation

- Upstream guard on `AnalyzeRequest` (schemas.py, config.py)
- Rejects with `OUT_OF_DOMAIN` error code (never silently clamped)
- Applied to `/api/analyze`, `/api/detect`, `/api/classify`, `/api/forecast`

## Verified Behavior (13 tests + live HTTP)

| Input | Result |
|-------|--------|
| lat=0, lon=40 | 200 (boundary accepted) |
| lat=30, lon=100 | 200 (boundary accepted) |
| lat=-0.001 | 422 OUT_OF_DOMAIN |
| lat=30.001 | 422 OUT_OF_DOMAIN |
| lon=39.999 | 422 OUT_OF_DOMAIN |
| lon=100.001 | 422 OUT_OF_DOMAIN |
| lat=-40 | 422 OUT_OF_DOMAIN |
| lon=120 | 422 OUT_OF_DOMAIN |
| NaN lat | 422 NON_FINITE_VALUE |
| Any obs off-box | 422 OUT_OF_DOMAIN |

## Error Message

"rejected, not clamped" — never repairs input silently.

## Verdict

**PASS** — NIO guard is verified, tested, and honest.
