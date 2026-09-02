# P2 Phase 8.5 — Label Provenance & Target Leakage Audit Report

**Date**: September 1, 2026  
**Auditor**: Independent Forensic Label Audit

---

## 1. Label Provenance Classification

* **Intensity Category**: **Authoritative (Class A / C)**. Mapped directly from official WMO/IMD Best Track maximum sustained wind speeds in `ibtracs_clean.csv`. Zero AI circularity.
* **Structural Pattern**: **Rule-Derived (Class D)**. Assigned via meteorological Dvorak intensity thresholds ($V_{\text{max}} \ge 120 \implies \text{eye\_visible}$, $65 \le V_{\text{max}} < 120 \implies \text{curved\_band}$, $V_{\text{max}} < 65 \implies \text{shear\_pattern}$).

---

## 2. Target Leakage & Scientific Disclosure
Because structural patterns are rule-derived from the same wind speeds that define intensity categories, the two classification tasks share underlying meteorological thresholds. This is explicitly disclosed: structural pattern performance must not be misrepresented as independent pixel-level human Dvorak annotations.
