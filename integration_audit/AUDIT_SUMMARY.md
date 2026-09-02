# Audit Summary — SIH 2026 PS 26070 Integrated Workspace

Date: 2026-08-30 · Integrator: AUDIT-ONLY (no source/dataset/model/doc modified; no extraction; only `integration_audit/` created)

---

## What was inspected
- **Archive:** `PS70-main.zip` — 1,231,540,094 B · 1,207 entries (42 dirs + 1,165 files) · root `PS70-main/`. Audited via .NET ZipFile streams; contents mirrored to temp for reading only (not extracted into workspace).
- **Datasets (6 groups / 18 items):** IBTrACS (raw 27.8 MB → clean 18,168 rows → era5-joined → master 5,481 rows/151 cyclones), ERA5 (94 NetCDF, 0.25°, 2013–2025, ~1.10 GB), INSAT (419 raw images + label sheet), Dataset A detection (133 rows), Dataset B classification (4,208 tabular + 133 images), Dataset C forecast (3,076 sequences 5×7→3×3).
- **Models (9 trained artifacts):** P2 MobileNetV3 detector; P3 ResNet18 image + LightGBM tabular; P4 6 experiment checkpoints (champion EXP005 GRU) + phase-3 baseline. All distinct (no duplicate weights); metrics verified.
- **APIs:** P4 phase6 FastAPI 4 routes verified live; P2/P3 inference functions; expected P5 endpoints `/api/analyze|detect|classify|forecast` — **not present** (see API_FRONTEND_INVENTORY.md).
- **Frontend:** 2 copies of the Vite/React dashboard (working + pristine), byte-level compared (see DUPLICATE_REPORT.md).

## Duplicate findings
- D1 `data/raw/insat/` ≡ `data/raw/insat_kaggle/` (420+420 files, 47,459,738 B, aggregate SHA-256 identical) — **in-archive only, report → resolve at repack**.
- D2 `data/processed/master_dataset.csv` = `data/metadata/master_dataset.csv` (SHA `003B8B94…`) — in-archive convenience copy.
- D3 PS70-main forecast npz ≡ P4 canonical npz — **value-identical, byte-distinct** (keep both).
- D4 P5 `node_modules` pair — identical deps except one generated `.vite/deps/_metadata.json`; both regenerable.
- D5 working-copy frontend source = **superset** of pristine original (4 extra components, 10 modified files; pristine kept as archival).

## Cleanup executed (SAFE_TO_DELETE only — 115,495,419 B freed)
| # | Item | Size | Basis |
|---|---|---|---|
| C1 | `scripts\__pycache__\*.pyc` | 52,584 B | regenerated cache |
| C2 | `cyclone-dashboard\dist\` (5 files) | 830,806 B | `npm run build` output (inputs pinned) |
| C3 | `SIH26\cyclone-dashboard\…\node_modules\` (8,045 files) | 114,612,029 B | redundant second install, regenerable from identical package-lock.json |
| C4 | `…\src\components\satellite\` | empty | unreferenced empty dir |

Keep-verified active install: working-copy `node_modules` (8,045 files/114,612,029 B) so the dashboard still runs; pristine frontend folder + `cyclone-dashboard.zip`; all models; all data; all source; `_source_p1`; p4 logs; empty placeholder dirs (`notebooks/`, `data/raw/era5/`, `phase4/audit_post_completion/`).

## Immutability verification
Source hash manifest recorded pre- and post-cleanup (`SOURCE_HASHES_BEFORE.json`, `SOURCE_HASHES_AFTER.json` — P1 archive + P4 + both P5 source scopes; 357 file records):
**missing 0 · new 0 · changed 0 → sources byte-for-byte untouched.**

## Post-cleanup state
| Root | Files | Bytes |
|---|---|---|
| `cyclone-project\` (now includes `integration_audit\` 9 files) | 8,423 | 1,390,981,478 |
| `SIH26\cyclone-dashboard\` | 29 (was 8,074) | 163,988 |

## Final status of every component
| Component | Status |
|---|---|
| P1 data/scripts | KEEP — in archive, frozen (unzip pending) |
| P2 detection | KEEP — function-only (no endpoint) |
| P3 classification | KEEP — function-only (no endpoint) |
| P4 forecasting | PASS — FastAPI `:8000` live, contract documented |
| P5 frontend | KEEP — running on mock data (`USE_MOCK=true`) |

## Blockers for full integration (recorded; out of scope here)
1. `PS70-main.zip` not yet extracted (only zipped source exists).
2. No `/api/analyze` `/api/detect` `/api/classify` backend; P2/P3 are functions needing a wrapper.
3. `/api/forecast` contract mismatch (shape/methods) between frontend client and P4 phase6.
4. Frontend still mock-only; P2/P3 accuracy flags (category acc 0.33–0.47) to review before exposing.
5. ERA5 sst coverage inconsistency inside P1 repo (QA vs README) — needs P1 confirmation.

## Next step (for the integration session that follows this audit)
Extract `PS70-main.zip` (purge `__pycache__`, D1/D2 at repack) → wrap P2/P3 as `/api/detect` `/api/classify` → add `/api` prefix + pressure/confidence to P4 forecast → implement unified `POST /api/analyze` per `future_task/tasks.md` + `mockData.js` → flip `USE_MOCK=false`.