# API HARDENING AUDIT

**Date:** 2026-08-31
**Status:** PASS

## Endpoints Verified

### GET /api/health
- Returns: 200 with status, service, phase, offline, model_ready, ml, forecasting
- ML module readiness: P2 (ready), P3 image (ready), P3 tabular (NOT_RUN with reason)
- Model ready: reflects EXP005 checkpoint availability

### POST /api/analyze
- Request: AnalyzeRequest (5 observations, 6h spacing, physical bounds, NIO domain)
- Response: full dashboard payload (detection, classification, forecast, landfall, risk, satellite, provenance)
- Validation: 422 for invalid inputs with deterministic codes

### POST /api/image
- Upload: JPEG/PNG only, max 5 MiB, max 4096px
- Response: P2 detection + P3 classification on user image
- Provenance: USER-UPLOADED IMAGE (never claims reference frame)

### POST /api/detect, /api/classify, /api/forecast
- Single-block endpoints sharing AnalyzeRequest
- Same validation as /api/analyze

## Validation Coverage

| Check | Code | Verified |
|-------|------|----------|
| Malformed JSON | INVALID_REQUEST | Yes |
| Missing fields | MISSING_FEATURE | Yes |
| Extra fields | INVALID_REQUEST | Yes |
| NaN/Inf | NON_FINITE_VALUE | Yes |
| Wrong history length | INVALID_HISTORY_LENGTH | Yes |
| Bad timestamp | INVALID_TIMESTAMP | Yes |
| Bad spacing | INVALID_SPACING | Yes |
| Non-monotonic | NON_MONOTONIC_TIMESTAMP | Yes |
| Negative wind | INVALID_REQUEST | Yes |
| Out of NIO | OUT_OF_DOMAIN | Yes |
| Unsupported image type | INVALID_REQUEST | Yes |
| Oversized image | 413 | Yes |
| Path traversal filename | safe (basename only) | Yes |

## Frontend Error Handling

- Loading state: spinner shown during fetch
- Success: data rendered
- Validation error: error message displayed
- Server error: error message displayed
- Offline: mock/offline mode with honest labels

## Verdict

**PASS** — Strict validation, deterministic error codes, no weakening for UI appearance.
