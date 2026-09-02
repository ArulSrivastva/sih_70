"""Champion selection for P4 Phase-4 (LOCKED RULE).

Primary criterion - lowest equal-weight mean of VALIDATION track errors::

    primary_score = mean(val_track_6h, val_track_12h, val_track_24h)

All three horizons receive equal weight.  Secondary tie-break: wind MAE (then
wind RMSE).  The TEST set is never used for selection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..common import HORIZONS


def horizon_track_mean(metrics: Dict[str, object]) -> List[float]:
    """[mean 6h, mean 12h, mean 24h] track error (km) from a metrics dict."""
    return [float(metrics[h]["track_error_km_mean"]) for h in HORIZONS]


def horizon_wind_mae(metrics: Dict[str, object]) -> List[float]:
    return [float(metrics[h]["wind_mae"]) for h in HORIZONS]


def horizon_wind_rmse(metrics: Dict[str, object]) -> List[float]:
    return [float(metrics[h]["wind_rmse"]) for h in HORIZONS]


def validation_primary_score(metrics: Dict[str, object]) -> float:
    """Equal-weight mean of the three horizon track errors on validation."""
    return float(np.mean(horizon_track_mean(metrics)))


def tiebreak_values(metrics: Dict[str, object]) -> Tuple[float, float]:
    """(mean wind MAE, mean wind RMSE) used to break primary-score ties."""
    return (float(np.mean(horizon_wind_mae(metrics))),
            float(np.mean(horizon_wind_rmse(metrics))))


def rank_experiments(val_results: Dict[str, Dict[str, object]]) -> List[Tuple[str, float]]:
    """Sort experiment ids by (primary_score, mean MAE, mean RMSE)."""
    def key(item: Tuple[str, Dict[str, object]]) -> Tuple[float, float, float]:
        _, m = item
        mae, rmse = tiebreak_values(m)
        return (validation_primary_score(m), mae, rmse)
    return [(eid, validation_primary_score(m)) for eid, m in
            sorted(val_results.items(), key=key)]


def select_champion(val_results: Dict[str, Dict[str, object]]) -> Tuple[str, float, Dict[str, object]]:
    """Return (champion_id, champion_primary_score, rationale-dict)."""
    if not val_results:
        raise ValueError("cannot select champion from empty validation results")
    ranked = rank_experiments(val_results)
    champion_id, primary = ranked[0]
    champion_metrics = val_results[champion_id]
    tie = tiebreak_values(champion_metrics)
    return champion_id, primary, {
        "champion": champion_id,
        "primary_score": primary,
        "component_horizons": {
            h: float(champion_metrics[h]["track_error_km_mean"]) for h in HORIZONS},
        "tiebreak": {"mean_wind_mae": tie[0], "mean_wind_rmse": tie[1]},
        "ranking": [{"experiment_id": eid, "primary_score": sc}
                    for eid, sc in ranked],
        "selection_rule": (
            "lowest equal-weight mean of validation track errors "
            "mean(val_track_6h, val_track_12h, val_track_24h); tie-break wind "
            "MAE then RMSE; TEST IS NEVER USED FOR SELECTION"),
    }


def make_champion_json(
    champion_id: str,
    rationale: Dict[str, object],
    run_metadata: Dict[str, object],
    source_hashes: Dict[str, object],
) -> Dict[str, object]:
    return {
        "experiment_id": champion_id,
        "primary_score": rationale["primary_score"],
        "component_horizon_scores": rationale["component_horizons"],
        "tie_break_values": rationale["tiebreak"],
        "selection_rule": rationale["selection_rule"],
        "ranking": rationale["ranking"],
        "run_metadata": run_metadata,
        "source_hashes": source_hashes,
    }


def save_champion_json(payload: Dict[str, object], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")