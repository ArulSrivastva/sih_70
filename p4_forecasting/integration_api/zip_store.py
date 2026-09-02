"""integration_api / zip_store.py

In-memory access to the ``PS70-main.zip`` artifact bundle.

The zip is opened read-only and every payload (P2 weights, P3 weights, the
reference INSAT frame) is delivered as ``bytes`` / PIL images pulled straight
from the archive.  Nothing is extracted to disk and the archive handle is
shared process-wide (opened once, lazily).

All failures surface as ``ModelNotReady`` so the API can answer 503 instead
of leaking tracebacks.
"""

from __future__ import annotations

import functools
import io
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

from .config import (INSAT_FINAL_DIR, INSAT_FINAL_DIR_2, P3_IMAGE_DIR,
                     ZIP_PATH, DEFAULT_REFERENCE_IMAGES)


class ModelNotReady(Exception):
    """The artifact bundle cannot provide what the inference path needs."""


class _ZipStore:
    """Lazy, read-only, process-wide zip handle with cached reads."""

    def __init__(self, zip_path: Path) -> None:
        self._zip_path = Path(zip_path)
        self._z: Optional[zipfile.ZipFile] = None
        self._ref: Optional[str] = None

    # -- open/close --------------------------------------------------------
    def _open(self) -> zipfile.ZipFile:
        if self._z is None:
            if not self._zip_path.exists():
                raise ModelNotReady(
                    f"artifact bundle not found: {self._zip_path.name}")
            try:
                self._z = zipfile.ZipFile(str(self._zip_path), "r")
            except zipfile.BadZipFile as exc:
                raise ModelNotReady(
                    f"artifact bundle is corrupt: {exc}") from exc
        return self._z

    @property
    def path(self) -> Path:
        return self._zip_path

    def close(self) -> None:
        if self._z is not None:
            try:
                self._z.close()
            finally:
                self._z = None

    # -- member access -------------------------------------------------------
    def has(self, rel: str) -> bool:
        try:
            self._open().getinfo(rel)
            return True
        except KeyError:
            return False

    @functools.lru_cache(maxsize=16)
    def read_bytes(self, rel: str) -> bytes:
        try:
            return self._open().read(rel)
        except KeyError as exc:
            raise ModelNotReady(f"zip member not found: {rel}") from exc

    def _image_members(self, prefix: str) -> List[str]:
        z = self._open()
        return sorted(
            n for n in z.namelist()
            if n.startswith(prefix)
            and n.lower().endswith((".jpg", ".jpeg", ".png")))

    # -- reference frame ------------------------------------------------------
    def reference_image_name(self) -> str:
        """Deterministic reference INSAT frame used by P2/P3."""
        if self._ref is not None:
            return self._ref
        z = self._open()
        for cand in DEFAULT_REFERENCE_IMAGES:
            try:
                z.getinfo(cand)
                self._ref = cand
                return cand
            except KeyError:
                continue
        finals = self._image_members(INSAT_FINAL_DIR)
        if finals:
            self._ref = finals[len(finals) // 2]
            return self._ref
        finals2 = self._image_members(INSAT_FINAL_DIR_2)
        if finals2:
            self._ref = finals2[len(finals2) // 2]
            return self._ref
        processed = self._image_members(P3_IMAGE_DIR)
        if processed:
            self._ref = processed[0]
            return self._ref
        raise ModelNotReady("no reference INSAT frame available in the bundle")

    def reference_image(self) -> Tuple[str, Image.Image]:
        name = self.reference_image_name()
        try:
            img = Image.open(io.BytesIO(self.read_bytes(name))).convert("RGB")
        except Exception as exc:
            raise ModelNotReady(f"cannot decode reference frame {name}") from exc
        return name, img


_store: Optional[_ZipStore] = None


def get_store() -> _ZipStore:
    global _store
    if _store is None:
        _store = _ZipStore(ZIP_PATH)
    return _store


def reset_store() -> None:
    global _store
    if _store is not None:
        _store.close()
        _store = None