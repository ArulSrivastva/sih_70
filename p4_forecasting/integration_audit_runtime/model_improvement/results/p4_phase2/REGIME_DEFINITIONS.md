# Regime Definitions -- P4 Phase 2

## Thresholds (from TRAINING data only)

- Speed p33: **9.27 km/h** (slow/steady boundary)
- Speed p66: **13.46 km/h** (fast/steady boundary)
- Turning threshold: **15.0 degrees** (heading change)

## Regime Assignments (training set)

| Regime | Count | % |
|--------|-------|---|
| slow | 205 | 14.2% |
| fast | 381 | 26.4% |
| turning | 525 | 36.4% |
| steady | 332 | 23.0% |

## Rules

- **slow**: speed < p33 AND turning < threshold
- **fast**: speed > p66 AND turning < threshold
- **turning**: turning > threshold (regardless of speed)
- **steady**: everything else (moderate speed, low turning)
