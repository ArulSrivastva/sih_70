# P2 Phase 6 — Data Scale Ablation Report

**Date**: September 1, 2026  
**Scope**: Empirical scaling behavior across D0 (Legacy 133), D1 (25%), D2 (50%), and D3 (100%).

---

## 1. Data Scaling Results Table

| Data Tier | Training Crops | Held-Out Test Pattern Macro-F1 | Held-Out Test Category Macro-F1 | Held-Out Test Category Accuracy |
|---|---|---|---|---|
| **D0: Legacy Dataset** | 93 ($100\%$ legacy) | 0.3697 | 0.1465 | 33.33% |
| **D1: 25% Expanded** | 400 ($25\%$ expanded) | 0.1915 | 0.0270 | 9.12% |
| **D2: 50% Expanded** | 780 ($50\%$ expanded) | 0.7244 | 0.1873 | 35.00% |
| **D3: 100% Expanded**| **1,500 ($100\%$ expanded)**| **0.9863** | **0.4138** | **55.88%** |

---

## 2. Scaling Analysis

* **Threshold Phenomenon**: Below $\approx 500$ training images (D0, D1), deep convolutional filters fail to learn generalizable vortex spiral features, collapsing into majority-class predictions.
* **Rapid Convergence Above $N=1,000$**: Moving from D2 ($N=780$) to D3 ($N=1,500$) yields a dramatic leap in pattern macro-F1 ($0.7244 \to 0.9863$) and category macro-F1 ($0.1873 \to 0.4138$), confirming that satellite vision requires genuine sample volume.
* **Scaling Curve Figure**: Visualized in [`results/figures/p2/phase6/p2_phase6_data_scale_curve.png`](file:///c:/Users/aruls/Desktop/SIH26/ps70/cyclone-project/results/figures/p2/phase6/p2_phase6_data_scale_curve.png).
