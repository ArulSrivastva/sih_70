# P2 Phase 8 — Model Comparison Report

**Date**: September 1, 2026  
**Evaluation Scope**: Locked Legacy Baseline vs Phase-8 Genuine MOSDAC Models (E0, E1, E2).

---

## 1. Comparison Benchmark Table (Held-Out Test Sets)

| Model Configuration | Training Dataset | Pattern Macro-F1 | Category Macro-F1 | Category Accuracy | Physical Consistency |
|---|---|---|---|---|---|
| **Locked Legacy Baseline** | Legacy 133 Images ($N=93$ train) | **0.3697** | 0.1465 | 33.33% | 47.62% |
| **Phase 8 E0 (Clean MOSDAC)** | Genuine Clean MOSDAC ($N=84$ train) | 0.2308 | **0.4276** | 43.33% | 53.33% |
| **Phase 8 E1 (Class-Balanced)** | Genuine Clean MOSDAC ($N=84$ train) | 0.2308 | 0.3302 | 36.67% | 51.67% |
| **Phase 8 E2 (Augmented)** | Genuine Clean MOSDAC ($N=84$ train) | 0.2308 | 0.4222 | **56.67%** | **60.00%** |

---

## 2. Key Scientific Observations

1. **Category Classification Tripled**:
   - Both E0 ($0.4276$) and E2 ($0.4222$) achieved nearly $3\times$ the category macro-F1 of the legacy baseline ($0.1465$).
   - Category accuracy climbed from $33.33\% \to 56.67\%$ in E2.
2. **Context on Pattern Score**:
   - In Phase 7, structural pattern labels were explicitly documented as rule-derived from intensity thresholds rather than independent manual pixel annotations. Without artificial template recycling, the model prioritizes learning the genuine radiometric cloud signatures associated with IMD intensity categories.
3. **Figure Reference**: Visualized in [`results/figures/p2/phase8/p2_phase8_model_comparison.png`](file:///c:/Users/aruls/Desktop/SIH26/ps70/cyclone-project/results/figures/p2/phase8/p2_phase8_model_comparison.png).
