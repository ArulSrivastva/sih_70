# Duplicate Report — SIH 2026 PS 26070

All duplicate checks performed at file/content granularity (SHA-256). "Dup" verdict = byte-identical duplicate. Verdict codes: **DUPLICATE** (confirmed identical) · **DUPLICATE-SUBSET** (source subset) · **NOT-DUP-REGENERABLE** (same versions, regenerated) · **NO** (distinct).

## Confirmed duplicate groups (content-identical)

### D1 — INSAT raw image trees (`inside PS70-main.zip`)
| Side A | Side B | Files | Bytes | Hash (aggregate of all file contents) | Verdict |
|---|---|---|---|---|---|
| `data/raw/insat/` | `data/raw/insat_kaggle/` | 420 = 420 | 47,459,738 = 47,459,738 | A=ECB719631205D30F140E35C2218F7689F50679554D82F363F6F2E5DD31C1361E · B=**same** | **DUPLICATE** |

Found inside the archive (= 47.5 MB versioned twice). Do NOT delete inside the zip this task (zip is the download; flag for a later clean extract/repack step). Contents per tree: `insat3d_for_reference_ds/CYCLONE_DATASET` (143), `insat3d_ir_cyclone_ds/CYCLONE_DATASET_INFRARED` (136), `insat3d_raw_cyclone_ds/CYCLONE_DATASET_FINAL` (140), `insat_3d_ds - Sheet.csv` (1).

### D2 — `processed/master_dataset.csv` mirror (`inside PS70-main.zip`)
| Side A | Side B | Size | SHA-256 | Verdict |
|---|---|---|---|---|
| `data/metadata/master_dataset.csv` | `data/processed/master_dataset.csv` | 738,948 | 003B8B944E856FF2763A40CE57288CDA5D9497246EA73D71CCE2AF6D0F62C5B9 | **DUPLICATE** (duplicate placed in processed/ for convenience) |

### D3 — Forecasting arrays: PS70-main `processed/forecasting/*_sequences.npz` vs P4 `canonical/*.npz`
| Criterion | Result |
|---|---|
| `np.array_equal(X / Y)` train+val+test | **True** (value-identical) |
| Byte-compare | differs (different compression/encoding; **NOT** DUPLICATE) |

Same data, different serialization → keep both (source-of-truth vs P4 reference); flagged as value-identical, not byte-duplicate.

### D4 — P5 `node_modules` across the two dashboard installs
| Criterion | Project copy (`cyclone-project/cyclone-dashboard`) | Original (`SIH26/cyclone-dashboard/cyclone-dashboard`) |
|---|---|---|
| Files | 8,045 | 8,045 |
| Bytes | 114,612,029 | 114,612,029 |
| Byte-identical files | 8,044 | 8,044 (same set) |
| Differing | — | `node_modules/.vite/deps/_metadata.json` (Vite generated prebundle cache metadata; regenerated on first `npm run dev`) |
| `package-lock.json` SHA-256 | 384C36812D1B6BBA01DCB9435BDE280456A10D3D943231EE82FDC19F84C39266 | **same** |

Verdict: **NOT-DUP-REGENERABLE** (dependency set identical; one generated Vite cache-metadata file differs). Both fully regenerable from identical `package-lock.json`.

### D5 — P5 source layout (working copy vs pristine original)
| Item | Working copy (`cyclone-project/…/cyclone-dashboard`) | Original (`SIH26/…/cyclone-dashboard`) |
|---|---|---|
| `package.json`, `index.html`, `.oxlintrc.json`, `README.md` | hash-identical | hash-identical |
| `vite.config.js` | 478 B (added `server.fs.allow: [ '../..' ]` to serve the workspace root) | 246 B (pristine) |
| src files shared | same content except the 10 files listed below | — |
| extra components (only in working copy) | `BaselineComparison.jsx`, `LiveCycloneStatus.jsx`, `ModelIntelligence.jsx`, `PipelineStatus.jsx` | absent |
| differing files (same names, changed content) | `App.jsx`, `api/client.js`, `mockData.js`, `components/ChartsPanel.jsx`, `ForecastCard.jsx`, `Header.jsx`, `LandfallPanel.jsx`, `RiskIndicator.jsx`, `SatelliteViewer.jsx`, `components/map/MapView.jsx` | pristine versions |

Verdict: working copy = **DUPLICATE-SUPERSET** (original's source is a strict subset; original kept as archival extract + the download `.zip` is the authoritative original).

## Duplicate NON-candidates (checked, distinct)
- P4 checkpoints EXP003/EXP005/EXP006: exactly 362,018 B each but SHA-256 distinct → not duplicates.
- EXP001 (88,934 B) / EXP002 (480,290 B) / EXP004 (480,866 B): distinct.
- `phase4/configs/EXP*.json` vs `results/experiments/EXP*/config.json`: byte-identical → **intentional experiment-registry copies** (KEEP).
- Kaggle image set: all 133 files hash-checked, zero duplicate pairs.
- P1↔P4 forecast npz: value-identical (D3) but not byte-duplicates.
- `phase3/checkpoints/best_lstm.pt` (79,708 B) vs workspace `models/lstm_forecaster.pt` (79,768 B): **different bytes** → distinct artifacts.

## Near-duplicates: `p4_forecasting/_source_p1/PS70-main/` vs current `PS70-main.zip`
| Item | Verdict |
|---|---|
| `build_datasets.py`, `download_era5.py`, `extract_era5_at_points.py`, `get_ibtracs.py` | **byte-identical** to zip |
| `data/metadata/ibtracs_clean.csv` | byte-identical (hash match) |
| rest of `data/metadata/*` + Dataset C npz | NOT identical — `_source_p1` is the **older P1 build** (pre-final); subset snapshot |

Verdict: `_source_p1` = historical frozen snapshot (P4 relies on it for the P1→P4 contract; keep, do not delete).

## Bottom line
- Confirmed identical byte-duplicates inside the archive (D1, D2) are **flagged for report only**; nothing inside `PS70-main.zip` is modified/deleted by this task.
- No two trained model artifacts are duplicates.
- P5 `node_modules` pair is version-identical regenerable dependency trees (not proto-copies of the codebase) — the **duplicate-install cleanup** is handled in CLEANUP_CANDIDATES.md.
- The P5 working copy is a superset of the original source; original extract + download zip retained.