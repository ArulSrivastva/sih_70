# Cleanup Candidates — SIH 2026 PS 26070

Scope rule applied: delete **only** what is (a) conclusively generated/cache/temp/reproducible or (b) a received duplicate that remains safely recoverable. Everything uncertain is KEEP.

## Candidates found (Phase 8 sweep), with verdicts

| # | Path (this workspace) | Type | Size | Evidence | Verdict |
|---|---|---|---|---|---|
| C1 | `cyclone-project\scripts\__pycache__\phase3_baseline_lstm.cpython-313.pyc` | pyc bytecode cache | 52,584 B | cache artifact of `scripts/phase3_baseline_lstm.py`; regenerated on import; no source value | **DELETE** |
| C2 | `cyclone-project\cyclone-dashboard\dist\` (5 files: index.js 768,359 B, index.css 47,098 B, index.html 796 B, favicon.svg 9,522 B, icons.svg 5,031 B) | Vite build output | 830,806 B | reproducible via `npm run build` (all inputs pinned by identical package.json/vite.config.js); pristine export contained no dist | **DELETE** |
| C3 | `SIH26\cyclone-dashboard\cyclone-dashboard\node_modules\` | dependency install (duplicate install) | 114,612,029 B (8,045 files) | regenerable from identical `package-lock.json` (SHA 384C3681...); dependency byte-set identical to the working copy except one generated `.vite/deps/_metadata.json`; pristine folder + `cyclone-dashboard.zip` untouched | **DELETE** (redundant second install) |
| C4 | `cyclone-project\cyclone-dashboard\src\components\satellite\` | empty directory | 0 B | no files, zero references in repo (grep: no import) | **DELETE** |
| C5 | `p4_forecasting\logs\run_*.log` (6 logs) | generated run logs | ~a few KB | run evidence/trace | KEEP (audit trail) |
| C6 | PS70-main.zip `src\**\__pycache__` (15 .pyc) | caches inside the download | ~60 KB | cannot modify archive in this task | KEEP (report only; purge after unzip) |
| C7 | `cyclone-project\notebooks\`, `cyclone-project\data\raw\era5\`, `cyclone-project\phase4\audit_post_completion\`, `node_modules\.vite-temp\` | empty placeholder dirs | 0 B | Jupyter/scratch/go-live-checkplaceholders | KEEP_UNCERTAIN (inert) |
| C8 | `cyclone-project\cyclone-dashboard\node_modules\` (working copy) | active dependency install | 114,612,029 B | needed to run `npm run dev` right now; deleting would break the live dashboard | KEEP (retain active install) |
| C9 | `SIH26\cyclone-dashboard\cyclone-dashboard\` (pristine source folder) | source tree (subset/superset issue) | (source only) | archival extract of the P5 download; zip is the authoritative original | KEEP (archival) |
| C10 | `cyclone-project\data\processed\sequences\*.npy` (12 files) | legacy 9-step arrays | (dataset) | superseded by canonical npz but is dataset, not garbage | KEEP (don't delete datasets) |
| C11 | inside-zip duplicates D1 (`raw/insat` = `raw/insat_kaggle`) and D2 (`processed/master_dataset.csv` mirror) | duplicated data in archive | 47,459,738 + 738,948 B | cannot modify archive | KEEP (report); resolve at repack |
| C12 | `p4_forecasting\_source_p1\PS70-main\` | old P1 snapshot (mainly older versions) | 4,234,627 B | P4 immutability reference | KEEP |

## Final deletion list (executed) — pre-deletion evidence

| Path | Kind | Size | SHA-256 (file / tree) |
|---|---|---|---|
| `…\scripts\__pycache__\phase3_baseline_lstm.cpython-313.pyc` | cache | 52,584 | B5D8A1816668E0EA66C00BFA10419D109B7F4638D726FA544567F27BE8216EC5 |
| `…\cyclone-dashboard\dist\favicon.svg` | build | 9,522 | 61BC9A161DE58248288E6905425D7180F0624C2865007B97D763FDAC12043A66 |
| `…\cyclone-dashboard\dist\icons.svg` | build | 5,031 | B45FA506195CFCDEF406BA9F0C77B36DDC1A7C224040926EC70ABC2FDEA7B93A |
| `…\cyclone-dashboard\dist\index.html` | build | 796 | 0D6CDBC34BB5A0DE0CCF7F517F39724027CE2C4CC5DD7E5D4CF0DA49FDE4DF5D |
| `…\cyclone-dashboard\dist\assets\index-Ck8t6X48.js` | build | 768,359 | 1697B4682E1255AD4990BAD7A358E3A85B3E6D57E7D16849531CBE5D14012523 |
| `…\cyclone-dashboard\dist\assets\index-pRxyAwG-.css` | build | 47,098 | E0D97673B358EB1BCBF630B72A8B3E47DDA35F3204F15F8748D15E3325F024BD |
| `…\SIH26\cyclone-dashboard\cyclone-dashboard\node_modules\` | deps | 114,612,029 | tree aggregate 9F55808DA9FF84772C2E40A552C4F69ADA080DE0BD955F5239FF2539A0ADF29F |
| `…\cyclone-dashboard\src\components\satellite\` | empty dir | 0 | (empty; removed after confirming no refs) |

Recompute command afterwards: `npm ci` / `npm run build` from the same pinned `package-lock.json` and `package.json`.

## What is NOT a deletion candidate (explicitly retained)
- `PS70-main.zip` and ALL content inside it (nothing in the archive is touched; in-archive duplicates/caches are flagged for the follow-up unzip/repack step).
- working-copy `node_modules` (live dev dependency of the operating dashboard).
- pristine `SIH26\cyclone-dashboard` folder + `cyclone-dashboard.zip` (archival source).
- every model/checkpoint/npz/csv/md/py under p4 phases, models/, results/, canonical/, `_source_p1` (source/data/evidence).
- `AUDIT_P1_DELIVERY.md`, `DATASET_REPORT.md` and all `.md` docs.