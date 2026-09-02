"""
Reusable inference module for cyclone forecasting.

Usage:
    from src.forecasting.inference import CycloneForecaster

    forecaster = CycloneForecaster()
    result = forecaster.forecast(input_sequence)
    # Returns: {"forecast": [{"hours": 6, "latitude": ..., ...}, ...]}
"""

import numpy as np
import json
import torch
from pathlib import Path


class CycloneForecaster:
    """End-to-end cyclone forecasting with normalization and denormalization."""

    def __init__(self, model_dir='models'):
        self.model_dir = Path(model_dir)
        self._load_configs()
        self._load_model()

    def _load_configs(self):
        with open(self.model_dir / 'model_config.json') as f:
            self.model_config = json.load(f)
        with open(self.model_dir / 'preprocessing_config.json') as f:
            self.preproc_config = json.load(f)

    def _load_model(self):
        from scripts.phase3_baseline_lstm import CycloneLSTM
        self.model = CycloneLSTM(
            input_size=self.model_config['input_size'],
            hidden_size=self.model_config['hidden_size'],
            num_layers=self.model_config['num_layers'],
            dropout=self.model_config['dropout']
        )
        self.model.load_state_dict(
            torch.load(self.model_dir / 'lstm_forecaster.pt',
                       weights_only=True)
        )
        self.model.eval()

    def _normalize(self, X):
        stats = self.preproc_config['normalization']['training_statistics']
        X_norm = X.copy()
        for i, feat in enumerate(self.preproc_config['feature_names'][:6]):
            mean = stats[feat]['mean']
            std = stats[feat]['std']
            X_norm[:, :, i] = (X_norm[:, :, i] - mean) / std
        return X_norm

    def _denormalize(self, pred_norm):
        stats = self.preproc_config['normalization']['target_statistics']
        pred = np.zeros_like(pred_norm)
        target_names = self.model_config['target_names']
        for i, feat in enumerate(target_names):
            mean = stats[feat]['mean']
            std = stats[feat]['std']
            pred[:, :, i] = pred_norm[:, :, i] * std + mean
        return pred

    def forecast(self, input_sequence):
        """
        Predict cyclone trajectory and intensity.

        Args:
            input_sequence: numpy array of shape (9, 7) or (1, 9, 7)
                Features: lat, lon, wind, pressure, storm_speed, storm_dir, pressure_missing

        Returns:
            dict with forecasts for +6h, +12h, +24h
        """
        if input_sequence.ndim == 2:
            input_sequence = input_sequence[np.newaxis, ...]

        X = self._normalize(input_sequence)
        X_t = torch.FloatTensor(X)

        with torch.no_grad():
            pred_norm = self.model(X_t).numpy()

        pred = self._denormalize(pred_norm)[0]  # (3, 3)

        result = {"forecast": []}
        for h_idx, h_hours in enumerate(self.model_config['horizons_hours']):
            result["forecast"].append({
                "hours": h_hours,
                "latitude": float(pred[h_idx, 0]),
                "longitude": float(pred[h_idx, 1]),
                "wind_speed_kmh": float(pred[h_idx, 2])
            })

        return result
