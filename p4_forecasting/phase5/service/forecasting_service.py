"""ForecastingService: one object, one call — validate, engineer, predict,
de-normalize, validate output, return the frontend contract.

    service = ForecastingService()
    result  = service.forecast(history)            # model forecast
    cmp     = service.compare_baselines(history)   # + persistence/movement-vector

Pure local inference: the champion lives on disk, no external API is called.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..config import (HISTORY_HOURS, HISTORY_STEPS, N_FEATURES,
                      default_paths)
from ..inference.input_validation import InputError, parse_history
from ..inference.output_contract import (build_error_response,
                                         build_forecast_list,
                                         build_success_response,
                                         validate_response)
from ..inference.predictor import CyclonePredictor
from ..inference.preprocessing import engineer_history
from ..baselines.persistence import persistence_forecast
from ..baselines.movement_vector import movement_vector_forecast


class ForecastingService:
    """Read-only champion inference + reference baselines."""

    def __init__(self, checkpoint_path=None, config_path=None,
                 stats_path=None, device: str = "cpu",
                 predictor: Optional[CyclonePredictor] = None) -> None:
        paths = default_paths()
        self.checkpoint_path = checkpoint_path or paths.champion_checkpoint
        self.config_path = config_path or paths.champion_config
        self.stats_path = stats_path or paths.normalization_stats
        self.champion_id = paths.champion_id
        self.device = device
        self._predictor = predictor

    @property
    def predictor(self) -> CyclonePredictor:
        if self._predictor is None:
            self._predictor = CyclonePredictor(
                self.checkpoint_path, self.config_path, self.stats_path,
                device=self.device)
        return self._predictor

    def _history_bytes(self, history) -> np.ndarray:
        return parse_history(history)

    def forecast(self, history: Any) -> Dict[str, Any]:
        """Validate, engineer, normalise, predict, de-normalise -> contract."""
        try:
            h7 = parse_history(history)
            feas = engineer_history(h7)
            pred = self.predictor.predict_features(feas)
            model_info = self.predictor.model_info(self.champion_id)
            response = build_success_response(
                model_info, build_forecast_list(pred),
                history_hours=HISTORY_HOURS, history_steps=HISTORY_STEPS,
                feature_count=N_FEATURES)
            validate_response(response)
            return response
        except InputError as exc:
            return build_error_response(exc.code, exc.message)
        except Exception as exc:  # never leak stack traces to the frontend
            return build_error_response("internal_error", str(exc))

    def compare_baselines(self, history: Any) -> Dict[str, Any]:
        """Model + persistence + movement-vector forecasts for one history."""
        try:
            h7 = parse_history(history)
            feas = engineer_history(h7)
            pred = self.predictor.predict_features(feas)
            persist = persistence_forecast(h7)
            mvec = movement_vector_forecast(h7)
            model_info = self.predictor.model_info(self.champion_id)
            return {
                "status": "success",
                "model": model_info,
                "input": {
                    "history_hours": HISTORY_HOURS,
                    "history_steps": HISTORY_STEPS,
                    "feature_count": N_FEATURES,
                },
                "model_forecast": build_forecast_list(pred),
                "persistence_forecast": build_forecast_list(persist),
                "movement_vector_forecast": build_forecast_list(mvec),
            }
        except InputError as exc:
            return build_error_response(exc.code, exc.message)
        except Exception as exc:
            return build_error_response("internal_error", str(exc))