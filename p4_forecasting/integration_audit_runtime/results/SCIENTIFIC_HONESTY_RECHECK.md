# SCIENTIFIC HONESTY AUDIT

**Date:** 2026-08-31
**Status:** PASS

## Classification of Every Displayed Output

### Frontend (cyclone-dashboard)

| Output | Type | Evidence Source |
|--------|------|----------------|
| Detection detected/confidence | REAL MODEL OUTPUT | P2 MobileNetV3 on reference frame |
| Classification category/confidence | REAL MODEL OUTPUT | P3 ResNet18 on reference frame |
| Structural pattern | DERIVED | P2 pattern head on synthetic labels |
| Forecast lat/lon/wind | REAL MODEL OUTPUT | P4 EXP005 via phase6 adapter |
| Forecast pressureHpa | HONEST NULL | No calibrated pressure output |
| Forecast confidence | HONIST NULL | No calibrated uncertainty |
| Landfall estimated/position | HEURISTIC (NOT ML) | Coast proximity heuristic |
| Risk score/level | HEURISTIC (NOT ML) | Wind-based formula |
| Satellite boundingBox | HONEST NULL | No validated localizer |
| Historical track | OBSERVED | From user-provided history |
| Wind/pressure history | OBSERVED | From user-provided history |
| Confidence history | STATIC | Placeholder from P2 detection |
| SST history | OBSERVED | From user-provided history |
| Env wind history | DERIVED | sqrt(wind_u^2 + wind_v^2) |
| Forecast cone | ILLUSTRATIVE | Not calibrated uncertainty |
| "research prototype" label | STATIC | Footer disclaimer |
| "Not an official IMD warning" | STATIC | Footer disclaimer |

### API Responses

| Field | Type | Verified |
|-------|------|----------|
| forecast[].pressureHpa | null | Yes |
| forecast[].confidence | null | Yes |
| detection.location | from history | Yes |
| landfall.* | heuristic | Yes |
| risk.* | heuristic | Yes |
| provenance.notes | full disclosure | Yes |
| satellite.boundingBox | null | Yes |

### Provenance Panel

| Field | Content | Verified |
|-------|---------|----------|
| pipeline | "p2 -> p3 -> p4 EXP005 -> server heuristics" | Yes |
| sources | Model paths specified | Yes |
| notes | Full caveats list | Yes |
| reference_image | Deterministic frame from zip | Yes |

## Fabrication Check

- No fabricated metrics, confidence, forecasts, or operational capabilities
- No "state of the art" or "production-ready" claims
- No "real-time" or "live feed" claims
- All null fields are honest absences
- Movement-vector baseline superiority disclosed

## Verdict

**PASS** — Every output has an identifiable evidence source; no fabrication detected.
