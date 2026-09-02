# P2 Phase 4 — Controlled Augmentation Experiment Report

**Date**: September 1, 2026  
**Protocol**: 5-Fold Stratified Cross-Validation on Development Set ($N=112$)  
**Random Seed**: 42

---

## 1. 5-Fold Cross-Validation Metrics

| Configuration | Pattern Acc | Pattern Macro-F1 | Category Acc | Category Macro-F1 | Category Weighted-F1 | Verdict |
|---|---|---|---|---|---|---|
| **A0: Baseline (No Augmentation)** | **58.97% ± 2.60%** | **0.2767 ± 0.0651** | 25.73% ± 9.36% | 0.1422 ± 0.0627 | 0.1984 ± 0.1011 | **Baseline** |
| **A1: Spatial Translation + ColorJitter** | 57.15% ± 1.74% | 0.2439 ± 0.0068 | **28.54% ± 6.56%** | **0.1502 ± 0.0555** | **0.2034 ± 0.0762** | Within Noise Floor |
| **A2: Rotation (±10°) + Spatial/Color** | 57.15% ± 1.74% | 0.2424 ± 0.0047 | 27.63% ± 6.41% | 0.1056 ± 0.0271 | 0.1583 ± 0.0563 | **Regression** |

---

## 2. Key Findings

1. **Spatial Translation + Color Jitter (A1)**:
   - Category macro-F1 changed nominally by **+0.0080** ($0.1422 \to 0.1502$), which is within the cross-validation standard deviation noise ($\pm 0.0555$).
   - However, Pattern macro-F1 dropped by **-0.0328** ($0.2767 \to 0.2439$) as spatial jitter disrupted the fragile curved band cloud feature alignments.
2. **Rotation (A2)**:
   - Introducing small $\pm 10^\circ$ rotations degraded category macro-F1 severely from $0.1422 \to 0.1056$ (-25.7% relative drop).
   - In the North Indian Ocean, structural shear patterns and spiral rainband quadrants have strong spatial orientations relative to environmental steering; random rotations introduce noise that hurts category distinction.
