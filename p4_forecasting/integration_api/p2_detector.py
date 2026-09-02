"""integration_api / p2_detector.py

Mirror of the audited Phase-2 detection stack
(``PS70-main/src/detection/detector.py`` + ``inference.py``) that:

  * builds the architecture WITHOUT ImageNet weights (``weights=None``) so
    nothing can download at runtime,
  * loads ``model_state_dict`` straight from ``PS70-main.zip`` (in-memory,
    ``weights_only=True``).

The class/schema parity is deliberate: the checkpoint's
``backbone.* / presence_head.* / pattern_head.* / category_head.*`` keys
bind cleanly, and the inference behaviour (transforms (224,224) + ToTensor,
sigmoid presence >= 0.5) matches the original exactly.
"""

from __future__ import annotations

import functools
import io
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

from .config import (P2_WEIGHTS_REL, P2_CANDIDATE_WEIGHTS_PATH,
                     E9_2_PATTERNS, E9_2_CATEGORIES)
from .zip_store import ModelNotReady, get_store


class CycloneDetector(nn.Module):
    """Structural copy of src/detection/detector.py -> CycloneDetector,
    with ``weights=None`` on the backbone."""

    def __init__(self, num_patterns: int = 3, num_categories: int = 7):
        super().__init__()
        self.backbone = mobilenet_v3_small(weights=None)
        feature_size = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()
        self.presence_head = nn.Sequential(
            nn.Linear(feature_size, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1))
        self.pattern_head = nn.Sequential(
            nn.Linear(feature_size, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_patterns))
        self.category_head = nn.Sequential(
            nn.Linear(feature_size, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_categories))

    def forward(self, x):
        features = self.backbone(x)
        return {
            "presence": self.presence_head(features),
            "pattern": self.pattern_head(features),
            "category": self.category_head(features),
        }


@functools.lru_cache(maxsize=1)
def _load_checkpoint() -> Dict[str, Any]:
    # Priority: Load Phase 9 E9-2 candidate if present on disk
    if P2_CANDIDATE_WEIGHTS_PATH.exists():
        try:
            ck = torch.load(str(P2_CANDIDATE_WEIGHTS_PATH), map_location="cpu", weights_only=False)
            state_dict = ck.get("state_dict") or ck.get("model_state_dict")
            pat_to_idx = {p: i for i, p in enumerate(E9_2_PATTERNS)}
            cat_to_idx = {c: i for i, c in enumerate(E9_2_CATEGORIES)}
            return {
                "source": "models/detection/model_weights_phase9_E9_2.pt (E9-2 candidate)",
                "is_candidate": True,
                "model_state_dict": state_dict,
                "pattern_to_idx": pat_to_idx,
                "category_to_idx": cat_to_idx,
            }
        except Exception as exc:
            pass  # Fall back to zip store baseline

    raw = get_store().read_bytes(P2_WEIGHTS_REL)
    try:
        ck = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ModelNotReady(f"cannot load P2 checkpoint: {exc}") from exc
    if not isinstance(ck, dict) or "model_state_dict" not in ck:
        raise ModelNotReady("P2 checkpoint is missing 'model_state_dict'")
    ck["source"] = P2_WEIGHTS_REL + " (locked baseline fallback)"
    ck["is_candidate"] = False
    return ck


class P2Detector:
    """Inference wrapper mirroring src/detection/inference.py -> CycloneInference."""

    def __init__(self) -> None:
        ck = _load_checkpoint()
        self.source = ck.get("source", P2_WEIGHTS_REL)
        self.is_candidate = ck.get("is_candidate", False)
        self.pattern_to_idx: Dict[str, int] = ck["pattern_to_idx"]
        self.category_to_idx: Dict[str, int] = ck["category_to_idx"]
        self.idx_to_pattern = {i: l for l, i in self.pattern_to_idx.items()}
        self.idx_to_category = {i: l for l, i in self.category_to_idx.items()}
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.model = CycloneDetector(
            num_patterns=len(self.pattern_to_idx),
            num_categories=len(self.category_to_idx),
        )
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.eval()

    def detect(self, image: Image.Image) -> Dict[str, Any]:
        tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(tensor)
        presence = torch.sigmoid(outputs["presence"]).item()
        detected = presence >= 0.5
        pattern_probs = torch.softmax(outputs["pattern"], dim=1)[0]
        pattern_idx = int(torch.argmax(pattern_probs).item())
        pattern_conf = float(pattern_probs[pattern_idx].item())
        category_probs = torch.softmax(outputs["category"], dim=1)[0]
        category_idx = int(torch.argmax(category_probs).item())
        category_conf = float(category_probs[category_idx].item())
        return {
            "detected": detected,
            "confidence": round(presence, 4),
            "structural_pattern": self.idx_to_pattern[pattern_idx],
            "pattern_confidence": round(pattern_conf, 4),
            "category": self.idx_to_category[category_idx],
            "category_confidence": round(category_conf, 4),
            "model_source": self.source,
        }

    def status(self) -> Dict[str, Any]:
        return {
            "model": self.source,
            "is_candidate": self.is_candidate,
            "loaded": True,
            "backbone": "mobilenet_v3_small (weights=None)",
            "patterns": len(self.pattern_to_idx),
            "categories": len(self.category_to_idx),
        }


_detector: Optional[P2Detector] = None


def get_detector() -> P2Detector:
    global _detector
    if _detector is None:
        _detector = P2Detector()
    return _detector