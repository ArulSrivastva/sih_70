"""P4 Phase-4 model families package."""

from .improved_lstm import ImprovedLSTM
from .gru import GRUCyclone
from .multitask_lstm import MultiTaskLSTM

MODEL_FAMILIES = {
    "improved_lstm": ImprovedLSTM,
    "gru": GRUCyclone,
    "multitask_lstm": MultiTaskLSTM,
}


def build_model(model: str, config: dict):
    """Instantiate a Phase-4 model from an experiment config dict."""
    key = str(config.get("model", model))
    if key not in MODEL_FAMILIES:
        raise ValueError(f"unknown model family '{key}'; expected one of {sorted(MODEL_FAMILIES)}")
    cls = MODEL_FAMILIES[key]
    return cls(
        input_size=int(config.get("input_size", 16)),
        hidden_size=int(config.get("hidden_size", 64)),
        num_layers=int(config.get("layers", config.get("num_layers", 1))),
        output_size=int(config.get("output_size", 9)),
        dropout=float(config.get("dropout", 0.0)),
    )


__all__ = ["ImprovedLSTM", "GRUCyclone", "MultiTaskLSTM", "MODEL_FAMILIES", "build_model"]