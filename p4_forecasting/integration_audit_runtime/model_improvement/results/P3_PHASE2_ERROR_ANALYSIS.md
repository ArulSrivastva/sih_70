# P3 Phase 2 -- Error Analysis

Total errors: 345/651 (53.0%)
Adjacent confusion: 201/345 (58.3%)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|--------|
| Depression | 0.611 | 0.763 | 0.678 | 253 |
| Deep Depression | 0.359 | 0.363 | 0.361 | 102 |
| Cyclonic Storm | 0.356 | 0.282 | 0.315 | 110 |
| Severe Cyclonic Storm | 0.281 | 0.222 | 0.248 | 72 |
| Very Severe Cyclonic Storm | 0.377 | 0.220 | 0.278 | 91 |
| Extremely Severe Cyclonic Storm | 0.257 | 0.391 | 0.310 | 23 |
| Super Cyclonic Storm | 0.000 | 0.000 | 0.000 | 0 |

## Confusion Matrix

| True\Pred | Depressi | Deep Dep | Cyclonic | Severe C | Very Sev | Extremel | Super Cy |
| Depressi | 193 | 34 | 22 | 2 | 2 | 0 | 0 |
| Deep Dep | 57 | 37 | 4 | 1 | 3 | 0 | 0 |
| Cyclonic | 34 | 25 | 31 | 10 | 9 | 1 | 0 |
| Severe C | 22 | 5 | 8 | 16 | 15 | 6 | 0 |
| Very Sev | 7 | 2 | 18 | 25 | 20 | 19 | 0 |
| Extremel | 3 | 0 | 4 | 3 | 4 | 9 | 0 |
| Super Cy | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
