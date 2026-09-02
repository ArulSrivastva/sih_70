"""MODEL_REPORT.md generator for P4 Phase 3.

All numbers come from actual computed artifacts (comparison json + training
history); nothing is hand-typed. Baseline/lstm claims are strictly evidence-based.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

HORIZONS = ["6h", "12h", "24h"]
MODELS = ["persistence", "movement_vector", "lstm"]


def _row(model: str, hz: str, data: dict) -> str:
    m = data[hz]
    return (f"| {model:<15} | {hz:>5} | {m['track_error_km_mean']:.2f} | "
            f"{m['wind_mae']:.2f} | {m['wind_rmse']:.2f} |")


def generate_model_report(
    comparison_path: Path,
    history_path: Path,
    config_path: Path,
    run_metadata_path: Path,
    out_path: Path,
    dataset_info: dict,
) -> None:
    with open(comparison_path, "r", encoding="utf-8") as fh:
        cmp = json.load(fh)
    with open(history_path, "r", encoding="utf-8") as fh:
        hist = json.load(fh)
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    with open(run_metadata_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    test = cmp["test"]
    improv_p = cmp["improvement_vs_persistence"]
    improv_mv = cmp["improvement_vs_movement_vector"]

    lines: list = []
    lines.append("# Model Report - LSTM Cyclone Forecast (P4 Phase 3, PS 26070 / SIH 2026)")
    lines.append("")
    lines.append(f"- Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    lines.append("## Dataset")
    lines.append("")
    lines.append("- **Split**: chronological, cyclone-disjoint (train/val/test by cyclone, "
                 "no cyclone appears in more than one split).")
    lines.append("- **Policy**: CLEAN-only (Phase-1 `sample_quality.csv` filter). "
                 "Excludes every sequence with any interpolated/filled (non-causal) cell.")
    lines.append("- **Sequence dimensions**: history (5, 7); targets (3, 3).")
    lines.append(f"- **Train**: {dataset_info['train']['sequences']} sequences / "
                 f"{dataset_info['train']['cyclones']} cyclones")
    lines.append(f"- **Validation**: {dataset_info['val']['sequences']} sequences / "
                 f"{dataset_info['val']['cyclones']} cyclones")
    lines.append(f"- **Test**: {dataset_info['test']['sequences']} sequences / "
                 f"{dataset_info['test']['cyclones']} cyclones")
    lines.append("- **Input features (exact order)**: lat, lon, wind_speed, pressure, "
                 "sst, wind_u, wind_v.")
    lines.append("- **Targets**: lat, lon, wind_speed (per horizon).")
    lines.append("- **SST unit**: degrees Celsius (as delivered; not converted to Kelvin).")
    lines.append("")

    lines.append("## Model")
    lines.append("")
    lines.append("```")
    lines.append("history (batch,5,7)")
    lines.append("  -> LSTM(7 -> 64, num_layers=1, batch_first=True, dropout=0)")
    lines.append("  -> last hidden state")
    lines.append("  -> Linear(64 -> 9)")
    lines.append("  -> reshape (batch, 3 horizons x 3 targets)")
    lines.append("```")
    lines.append("")
    lines.append(f"- Trainable parameters: **{cfg['parameters']}**")
    lines.append(f"- Optimizer: Adam, lr={cfg['learning_rate']}, batch_size={cfg['batch_size']}, "
                 f"seed={cfg['seed']}")
    lines.append("")

    lines.append("## Normalization")
    lines.append("")
    lines.append("Normalization statistics were calculated exclusively from the training "
                 "split and then applied unchanged to validation and test data.")
    lines.append("")
    lines.append("- z = (x - mean) / std for all 7 features; targets normalized the same way.")
    lines.append("- Zero-variance or NaN/Inf statistics are a hard failure (never silently replaced).")
    lines.append("- Source NPZ files are never modified.")
    lines.append("")

    lines.append("## Results (TEST split, primary)")
    lines.append("")
    lines.append("| Model           | Horizon | Track Error km (mean) | Wind MAE km/h | Wind RMSE km/h |")
    lines.append("| --------------- | ------: | --------------------: | ------------: | -------------: |")
    for model in MODELS:
        for hz in HORIZONS:
            lines.append(_row(model, hz, test[model]))
    lines.append("")

    lines.append("## Baseline improvement (LSTM vs baselines, test split)")
    lines.append("")
    lines.append("Improvement % = (baseline - lstm) / |baseline| * 100; positive = LSTM better.")
    lines.append("")
    lines.append("| Horizon | vs Persistence track err % | vs Persistence wind MAE % | vs Movement-vector track err % | vs Movement-vector wind MAE % |")
    lines.append("| ------: | -------------------------: | ------------------------: | -----------------------------: | ----------------------------: |")
    for hz in HORIZONS:
        ip = improv_p[hz]
        imv = improv_mv[hz]
        lines.append(f"| {hz:>6} | {ip['track_error_km_mean_pct']:.2f} | "
                     f"{ip['wind_mae_pct']:.2f} | {imv['track_error_km_mean_pct']:.2f} | "
                     f"{imv['wind_mae_pct']:.2f} |")
    lines.append("")

    for hz in HORIZONS:
        lines.append(f"- **+{hz}**: persistence baseline "
                     f"{'beats' if improv_p[hz]['track_error_km_mean_pct'] < 0 else 'is beaten by'} "
                     f"the LSTM on mean track error "
                     f"({improv_p[hz]['track_error_km_mean_pct']:+.2f}%); movement-vector "
                     f"{'beats' if improv_mv[hz]['track_error_km_mean_pct'] < 0 else 'is beaten by'} "
                     f"the LSTM on mean track error "
                     f"({improv_mv[hz]['track_error_km_mean_pct']:+.2f}%).")
    lines.append("")

    lines.append("## Validation results")
    lines.append("")
    lines.append("| Model           | Horizon | Track Error km (mean) | Wind MAE km/h | Wind RMSE km/h |")
    lines.append("| --------------- | ------: | --------------------: | ------------: | -------------: |")
    for model in MODELS:
        for hz in HORIZONS:
            lines.append(_row(model, hz, cmp["validation"][model]))
    lines.append("")

    lines.append("## Training / overfitting check")
    lines.append("")
    lines.append(f"- Best epoch: {hist['best_epoch']}  (patience {cfg.get('patience', 'n/a')})")
    lines.append(f"- Best validation loss: {hist['best_val_loss']}")
    lines.append(f"- Final train loss: {hist['final_train_loss']}")
    lines.append(f"- Final validation loss: {hist['final_val_loss']}")
    lines.append(f"- Training time: {hist['training_time_s']:.2f} s")
    lines.append(f"- Device: {meta.get('device', 'cpu')}")
    lines.append("")
    if hist.get("final_train_loss") is not None and hist.get("final_val_loss") is not None:
        if hist["final_train_loss"] < hist.get("train_loss_at_best_epoch", 0) and \
                hist["final_val_loss"] > hist["best_val_loss"] * 1.2:
            lines.append("> Note: training loss kept decreasing while validation loss "
                         "worsened substantially (>20%) after the best epoch - evidence of "
                         "overfitting. Reported transparently; no regularization added in Phase 3.")
    lines.append("")

    lines.append("## Config / reproducibility")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps({k: cfg[k] for k in cfg if k != "feature_order" and k != "target_order"},
                            indent=2))
    lines.append("```")
    lines.append("")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")