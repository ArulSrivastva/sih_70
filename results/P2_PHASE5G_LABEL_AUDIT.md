# P2 Phase 5G — Label Validation & Provenance Audit Report

**Date**: September 1, 2026  
**Scope**: Verification of ground-truth intensity categories and Dvorak structural-pattern labels.

---

## 1. Ground-Truth Provenance & Standards

* **Intensity Category Supervision**:
  - Derived directly from **official IMD / WMO Best Track maximum sustained wind speeds** recorded in IBTrACS.
  - Standard IMD Scale:
    * *Depression*: $31–49\text{ km/h}$
    * *Deep Depression*: $50–61\text{ km/h}$
    * *Cyclonic Storm*: $62–88\text{ km/h}$
    * *Severe Cyclonic Storm*: $89–117\text{ km/h}$
    * *Very Severe Cyclonic Storm*: $118–165\text{ km/h}$
    * *Extremely Severe Cyclonic Storm*: $166–221\text{ km/h}$
    * *Super Cyclonic Storm*: $\ge 222\text{ km/h}$
* **Structural Pattern Supervision**:
  - Assigned based on IMD operational Dvorak cyclone analysis bulletins.
  - **Zero Circularity**: No CNN predictions or model outputs were used to generate ground-truth labels.

---

## 2. Pilot Label Distribution & Quality Check

| Class Name | Pilot Samples | Ground-Truth Provenance | Ambiguities / Missing |
|---|---|---|---|
| **`Extremely Severe CS`** | 7 | IBTrACS Track $V_{\text{max}} \ge 90\text{ kt}$ | None (0) |
| **`Very Severe CS`** | 5 | IBTrACS Track $V_{\text{max}} \in [64, 89]\text{ kt}$ | None (0) |
| **`Severe CS`** | 3 | IBTrACS Track $V_{\text{max}} \in [48, 63]\text{ kt}$ | None (0) |
| **`Cyclonic Storm`** | 12 | IBTrACS Track $V_{\text{max}} \in [34, 47]\text{ kt}$ | None (0) |
| **`Deep Depression`** | 10 | IBTrACS Track $V_{\text{max}} \in [28, 33]\text{ kt}$ | None (0) |
| **`Depression`** | 8 | IBTrACS Track $V_{\text{max}} \in [17, 27]\text{ kt}$ | None (0) |
| **`eye_visible`** | 12 | IMD Dvorak eye pattern | None (0) |
| **`curved_band`** | 18 | IMD Dvorak curved spiral band | None (0) |
| **`shear_pattern`** | 15 | IMD Dvorak sheared center | None (0) |

---

## 3. Label Audit Status
$$\mathbf{LABEL\_AUDIT\_STATUS = PASS}$$
Ground-truth labels are 100% complete, non-circular, and verified against official meteorological track records.
