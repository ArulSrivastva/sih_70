# P4 Phase 2 -- Hybrid Analysis

## Overall Results (CV)

| Model | 6h (km) | 12h (km) | 24h (km) |
|-------|---------|----------|----------|
| MV_baseline | 27.77 | 59.19 | 138.92 |
| CatBoost_baseline | 26.82 | 56.57 | 127.27 |
| CatBoost_phase2 | 27.49 | 57.68 | 128.55 |
| Hybrid_rule | 27.44 | 57.75 | 129.42 |
| Hybrid_blend_25 | 26.81 | 56.37 | 130.06 |
| Hybrid_blend_50 | 26.39 | 55.22 | 125.30 |
| Hybrid_blend_75 | 26.62 | 55.70 | 124.82 |

## Regime-Level 24h Performance

| Model | Slow | Fast | Turning | Steady |
|-------|------|------|---------|--------|
| MV_baseline | 135.5 (n=203) | 149.8 (n=384) | 145.2 (n=529) | 118.0 (n=327) |
| CatBoost_baseline | 124.3 (n=203) | 132.7 (n=384) | 130.6 (n=529) | 117.4 (n=327) |
| CatBoost_phase2 | 127.1 (n=203) | 133.1 (n=384) | 131.5 (n=529) | 119.4 (n=327) |
| Hybrid_rule | 135.5 (n=203) | 133.1 (n=384) | 131.5 (n=529) | 118.0 (n=327) |
| Hybrid_blend_25 | 127.9 (n=203) | 136.3 (n=384) | 135.8 (n=529) | 114.8 (n=327) |
| Hybrid_blend_50 | 124.0 (n=203) | 128.9 (n=384) | 130.1 (n=529) | 114.0 (n=327) |
| Hybrid_blend_75 | 123.7 (n=203) | 127.7 (n=384) | 128.9 (n=529) | 115.6 (n=327) |
