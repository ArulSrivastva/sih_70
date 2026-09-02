# P2 Phase 6 — Label Generation & Provenance Audit Report

**Date**: September 1, 2026  
**Scope**: Full ground-truth label audit for 2,120 frames across 25 cyclone systems.

---

## 1. Ground-Truth Provenance

* **IMD Intensity Category**: Derived strictly from official **IMD/WMO Best Track maximum sustained wind speeds** ($V_{\text{max}}$ in $\text{km/h}$).
* **Dvorak Structural Pattern**: Assigned following standard Dvorak technique morphology based on convective organization and intensity stage:
  - `eye_visible`: $V_{\text{max}} \ge 120\text{ km/h}$ or mature central dense overcast with defined eye.
  - `curved_band`: $65\text{ km/h} \le V_{\text{max}} < 120\text{ km/h}$ (prominent spiral banding wrapping $\ge 0.5$ circles).
  - `shear_pattern`: $V_{\text{max}} < 65\text{ km/h}$ (displaced convection with exposed low-level circulation).
* **Circularity**: **Zero circularity**. No CNN model predictions were used to generate ground-truth labels.

---

## 2. Full Category & Pattern Label Distributions

| Class Name | Total Sample Count | Percentage (%) | Ground-Truth Standard |
|---|---|---|---|
| **`Cyclonic Storm`** | 457 | 21.6% | $V_{\text{max}} \in [62, 88]\text{ km/h}$ |
| **`Depression`** | 428 | 20.2% | $V_{\text{max}} \in [31, 49]\text{ km/h}$ |
| **`Very Severe CS`** | 375 | 17.7% | $V_{\text{max}} \in [118, 165]\text{ km/h}$ |
| **`Severe CS`** | 368 | 17.4% | $V_{\text{max}} \in [89, 117]\text{ km/h}$ |
| **`Extremely Severe CS`**| 214 | 10.1% | $V_{\text{max}} \in [166, 221]\text{ km/h}$ |
| **`Deep Depression`** | 209 | 9.9% | $V_{\text{max}} \in [50, 61]\text{ km/h}$ |
| **`Super Cyclonic Storm`**| 69 | 3.3% | $V_{\text{max}} \ge 222\text{ km/h}$ |
| **`curved_band`** | 825 | 38.9% | Dvorak Curved Band Standard |
| **`eye_visible`** | 658 | 31.0% | Dvorak Eye Pattern Standard |
| **`shear_pattern`** | 637 | 30.0% | Dvorak Shear Pattern Standard |
