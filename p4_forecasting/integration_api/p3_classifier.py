"""integration_api / p3_classifier.py

Mirror of the audited Phase-3 image-only classifier
(``PS70-main/src/classification/classifier.py`` -> ``ImageOnlyIntensityModel``
and ``inference.py`` -> ``classify_cyclone`` image path).

Same runtime policy as P2: backbone built with ``weights=None``, checkpoint
streamed from ``PS70-main.zip`` in memory.

Honesty note: the image-only model is a weak baseline (audited Phase-3
results: ~38% accuracy, leak-prone 21-frame eval) and its wind regressor is
effectively degenerate, so the analyzer uses ONLY its category + confidence
labels and never its wind estimate.
"""

from __future__ import annotations

import functools
import io
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from .config import (IMD_CLASSES, IMD_SHORT_CODES, P3_IMAGE_WEIGHTS_REL,
                     P3_TABULAR_MODEL_REL)
from .zip_store import ModelNotReady, get_store

try:
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18
    TORCH_AVAILABLE = True
except ImportError:                     # pragma: no cover - torch is required
    TORCH_AVAILABLE = False

try:
    import lightgbm  # noqa: F401
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class ImageOnlyIntensityModel(nn.Module):
    """Structural copy of classifier.py -> ImageOnlyIntensityModel
    (ResNet18 backbone, weights=None)."""

    def __init__(self, num_classes: int = 7):
        super().__init__()
        base = resnet18(weights=None)
        in_feats = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_feats, 128), nn.ReLU(),
            nn.Linear(128, num_classes))
        self.wind_regressor = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_feats, 64), nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, x):
        feats = self.backbone(x)
        return self.classifier(feats), self.wind_regressor(feats).squeeze(-1)


@functools.lru_cache(maxsize=1)
def _load_image_checkpoint() -> Any:
    if not TORCH_AVAILABLE:
        raise ModelNotReady("torch unavailable")
    raw = get_store().read_bytes(P3_IMAGE_WEIGHTS_REL)
    try:
        sd = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ModelNotReady(f"cannot load P3 image checkpoint: {exc}") from exc
    return sd


class P3ImageClassifier:
    """Inference wrapper mirroring inference.py's image-only path."""

    def __init__(self) -> None:
        sd = _load_image_checkpoint()
        self.model = ImageOnlyIntensityModel(num_classes=len(IMD_CLASSES))
        self.model.load_state_dict(sd)
        self.model.eval()

    def classify_image(self, image: Image.Image) -> Dict[str, Any]:
        """Returns the exact P3 image-path result dict (pressure 990.0 is the
        model's own hardcoded value; the analyzer does not display it)."""
        arr = np.array(image.resize((256, 256)), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            logits, pred_wind = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, cat_idx = torch.max(probs, dim=1)
        c_idx = int(cat_idx.item())
        cat = IMD_CLASSES[c_idx] if 0 <= c_idx < len(IMD_CLASSES) \
            else "Cyclonic Storm"
        return {
            "category": cat,
            "imd_code": IMD_SHORT_CODES.get(cat, "CS"),
            "wind_speed": float(round(float(pred_wind.item()), 1)),
            "pressure": 990.0,
            "confidence": float(round(float(conf.item()), 2)),
        }

    def status(self) -> Dict[str, Any]:
        return {
            "model": P3_IMAGE_WEIGHTS_REL,
            "loaded": True,
            "backbone": "resnet18 (weights=None)",
            "classes": len(IMD_CLASSES),
        }


import sys
import types


class MultisourceTabularModel:
    """Stub class matching pickle signature for tabular model."""
    pass


# Register module alias so pickle can deserialize MultisourceTabularModel
try:
    import src
except Exception:
    if "src" not in sys.modules:
        _src = types.ModuleType("src")
        _src.__path__ = [str(PROJECT_ROOT / "src")]
        sys.modules["src"] = _src

if "src" in sys.modules and not hasattr(sys.modules["src"], "__path__"):
    sys.modules["src"].__path__ = [str(PROJECT_ROOT / "src")]

if "src.classification" not in sys.modules:
    _m_cls = types.ModuleType("src.classification")
    sys.modules["src.classification"] = _m_cls
    if "src" in sys.modules:
        setattr(sys.modules["src"], "classification", _m_cls)

if "src.classification.classifier" not in sys.modules:
    _mod = types.ModuleType("src.classification.classifier")
    _mod.MultisourceTabularModel = MultisourceTabularModel
    sys.modules["src.classification.classifier"] = _mod
    if "src.classification" in sys.modules:
        setattr(sys.modules["src.classification"], "classifier", _mod)
else:
    sys.modules["src.classification.classifier"].MultisourceTabularModel = MultisourceTabularModel


class P3TabularClassifier:
    """Inference wrapper for the audited P3 LightGBM multi-source tabular model."""

    def __init__(self) -> None:
        if not LIGHTGBM_AVAILABLE:
            raise ModelNotReady("lightgbm is not available")
        raw = get_store().read_bytes(P3_TABULAR_MODEL_REL)
        import pickle
        try:
            self.model = pickle.loads(raw)
        except Exception as exc:
            raise ModelNotReady(f"failed to unpickle tabular model: {exc}") from exc

    def classify_tabular(self, lat: float, lon: float, sst: float,
                         pressure_msl: float, wind_u: float, wind_v: float) -> Dict[str, Any]:
        """Predicts intensity category, wind speed, and pressure from 6 environmental features."""
        X = np.array([[lat, lon, sst, pressure_msl, wind_u, wind_v]], dtype=np.float32)
        X_scaled = self.model.scaler.transform(X)
        cat_idx = int(self.model.clf.predict(X_scaled)[0])
        probs = self.model.clf.predict_proba(X_scaled)[0]
        conf = float(probs[cat_idx])
        pred_wind = float(self.model.wind_reg.predict(X_scaled)[0])
        try:
            pred_pres = float(self.model.pres_reg.predict(X_scaled)[0])
        except Exception:
            pred_pres = 990.0

        cat_name = IMD_CLASSES[cat_idx] if 0 <= cat_idx < len(IMD_CLASSES) else "Cyclonic Storm"
        return {
            "category": cat_name,
            "imd_code": IMD_SHORT_CODES.get(cat_name, "CS"),
            "confidence": round(conf, 4),
            "wind_speed_kmh": round(pred_wind, 1),
            "pressure_hpa": round(pred_pres, 1),
            "probabilities": {IMD_CLASSES[i]: round(float(probs[i]), 4) for i in range(len(IMD_CLASSES))},
        }


def tabular_status() -> Dict[str, Any]:
    """Tabular MultiSource model status."""
    if not LIGHTGBM_AVAILABLE:
        return {
            "model": P3_TABULAR_MODEL_REL,
            "available": False,
            "reason": "lightgbm package not installed",
        }
    try:
        get_tabular_classifier()
        return {
            "model": P3_TABULAR_MODEL_REL,
            "available": True,
            "reason": "loaded and ready",
        }
    except Exception as exc:
        return {
            "model": P3_TABULAR_MODEL_REL,
            "available": False,
            "reason": f"load failed: {exc}",
        }


_image_classifier: Optional[P3ImageClassifier] = None
_tabular_classifier: Optional[P3TabularClassifier] = None


def get_image_classifier() -> P3ImageClassifier:
    global _image_classifier
    if _image_classifier is None:
        _image_classifier = P3ImageClassifier()
    return _image_classifier


def get_tabular_classifier() -> P3TabularClassifier:
    global _tabular_classifier
    if _tabular_classifier is None:
        _tabular_classifier = P3TabularClassifier()
    return _tabular_classifier