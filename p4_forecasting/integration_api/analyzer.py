"""integration_api / analyzer.py

Orchestrates the dashboard payload from genuinely distinct sources:

  * ``detection``     -> P2 CycloneDetector on a reference INSAT-3D IR frame
                          (model_weights.pt, served from the zip in memory).
  * ``classification``-> P3 image-only classifier category + confidence on the
                          same frame; wind/pressure shown are the latest
                          OBSERVED values from the validated history (the P3
                          wind regressor is degenerate and is never used).
  * ``forecast``      -> the audited Phase-4/5 champion (EXP005) via the
                          phase-6 ForecastingAdapter.
  * ``landfall``/``risk`` -> deterministic server-side heuristics, disclosed
                          as such; never ML output.
  * ``satellite``     -> label + attribution; boundingBox is null because no
                          audited localizer exists in the artifact bundle.

Every number in the response is traceable to either a real model output or
the validated request history; the provenance block says exactly which.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from phase6.integration.forecasting_adapter import AdapterError, ForecastingAdapter

from .config import (DEFAULT_META, FORECAST_HORIZONS, IMD_SHORT_CODES,
                     LANDFALL_THRESHOLD_KM, NIO_COAST_POINTS,
                     P3_IMAGE_WEIGHTS_REL, RISK_WIND_REF_KMH)
from .p2_detector import get_detector
from .p3_classifier import get_image_classifier, get_tabular_classifier, tabular_status
from .schemas import AnalyzeRequest
from .zip_store import get_store

_CARD16 = [
    "North", "North-North-East", "North-East", "East-North-East",
    "East", "East-South-East", "South-East", "South-South-East",
    "South", "South-South-West", "South-West", "West-South-West",
    "West", "West-North-West", "North-West", "North-North-West",
]

_TIME_LABELS = ["-24h", "-18h", "-12h", "-6h", "Now"]


def _parse_ts(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.fromisoformat(value)


def _haversine_km(lat1: float, lon1: float,
                  lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(p2)
    x = (math.cos(p1) * math.sin(p2)
         - math.sin(p1) * math.cos(p2) * math.cos(dlon))
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _compass(bearing: float) -> str:
    idx = int((bearing + 11.25) // 22.5) % 16
    return f"{_CARD16[idx]} ({round(bearing)}°)"


def _nearest_coast(lat: float, lon: float) -> Tuple[float, float, float]:
    best = min(
        ((_haversine_km(lat, lon, clat, clon), clat, clon)
         for clat, clon in NIO_COAST_POINTS))
    return best


class CycloneAnalyzer:
    """Assembles the dashboard contract from real artifact sources."""

    def __init__(self, adapter: Optional[ForecastingAdapter] = None) -> None:
        self._adapter = adapter if adapter is not None else ForecastingAdapter()

    # -- helpers -------------------------------------------------------------
    @property
    def adapter(self) -> ForecastingAdapter:
        return self._adapter

    def _latest_obs(self, req: AnalyzeRequest) -> Dict[str, Any]:
        return req.history[-1]

    def _movement(self, req: AnalyzeRequest) -> Dict[str, Any]:
        a, b = req.history[-2], req.history[-1]
        dist = _haversine_km(a.latitude, a.longitude,
                             b.latitude, b.longitude)
        bearing = _bearing_deg(a.latitude, a.longitude,
                               b.latitude, b.longitude)
        return {
            "bearing_deg": bearing,
            "direction": _compass(bearing),
            "speed_kmh": round(dist / 6.0, 1),
        }

    def _forecast_items(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for f in payload["forecast"]:
            items.append({
                "hour": int(f["hours"]),
                "label": f"+{int(f['hours'])}h",
                "lat": round(float(f["latitude"]), 4),
                "lon": round(float(f["longitude"]), 4),
                "windSpeedKmh": round(float(f["wind_speed_kmh"]), 1),
                "pressureHpa": None,   # no calibrated pressure output
                "confidence": None,    # no calibrated uncertainty
            })
        return items

    def _landfall(self, last_ts: datetime,
                  items: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not items:
            return {"estimated": False, "distanceToLandKm": None}
        best = None
        for f in items:
            dist, clat, clon = _nearest_coast(f["lat"], f["lon"])
            rec = {"hours": f["hour"], "dist": dist, "clat": clat,
                   "clon": clon, "wind": f["windSpeedKmh"]}
            if best is None or dist < best["dist"]:
                best = rec
        if best["dist"] <= LANDFALL_THRESHOLD_KM:
            est_time = last_ts + timedelta(hours=best["hours"])
            return {
                "estimated": True,
                "latitude": round(best["clat"], 4),
                "longitude": round(best["clon"], 4),
                "estimated_time": est_time.astimezone(timezone.utc)
                .isoformat().replace("+00:00", "Z"),
                "predictedWindKmh": best["wind"],
                "distanceToLandKm": int(round(best["dist"])),
            }
        return {
            "estimated": False,
            "latitude": None,
            "longitude": None,
            "estimated_time": None,
            "predictedWindKmh": None,
            "distanceToLandKm": int(round(best["dist"])),
        }

    def _risk(self, wind_now: float, items: List[Dict[str, Any]],
              landfall: Dict[str, Any]) -> Dict[str, Any]:
        fc_wind = max([f["windSpeedKmh"] for f in items] or [wind_now])
        w = max(float(wind_now), float(fc_wind))
        score = 5.0 + (w - RISK_WIND_REF_KMH) * 0.6
        score = min(99.0, max(5.0, score))
        if landfall.get("distanceToLandKm") is not None:
            d = float(landfall["distanceToLandKm"])
            if d < 300:
                score = min(99.0, score + (300.0 - d) / 20.0)
        score = int(round(score))
        level = "HIGH" if score >= 80 else "MODERATE" if score >= 55 else "LOW"
        return {"score": score, "level": level}

    def _history_view(self, req: AnalyzeRequest,
                      confidence_placeholder: float) -> Dict[str, Any]:
        track, wind, pressure, sst, env, conf = [], [], [], [], [], []
        for obs in req.history:
            track.append({
                "lat": round(obs.latitude, 4),
                "lon": round(obs.longitude, 4),
                "timestamp": obs.timestamp,
            })
            wind.append({"value": round(obs.wind_speed_kmh, 1)})
            pressure.append({"value": round(obs.pressure_hpa, 1)})
            sst.append({"value": round(obs.sst, 2)})
            env.append({"value": round(
                math.hypot(obs.wind_u, obs.wind_v), 2)})
            conf.append({"value": round(confidence_placeholder, 1)})
        for lst in (wind, pressure, sst, env, conf):
            for i, label in enumerate(_TIME_LABELS):
                lst[i]["t"] = label
        return {
            "historicalTrack": track,
            "windHistory": wind,
            "pressureHistory": pressure,
            "confidenceHistory": conf,
            "sstHistory": sst,
            "envWindHistory": env,
        }

    # -- public operations ---------------------------------------------------
    def detect(self, req: AnalyzeRequest) -> Dict[str, Any]:
        ref, image = get_store().reference_image()
        det = get_detector().detect(image)
        mov = self._movement(req)
        last = self._latest_obs(req)
        return {
            "detected": det["detected"],
            "confidence": int(round(det["confidence"] * 100)),
            "location": {"lat": round(last.latitude, 4),
                         "lon": round(last.longitude, 4)},
            "movementDirection": mov["direction"],
            "movementSpeedKmh": mov["speed_kmh"],
        }
    def classify(self, req: AnalyzeRequest) -> Dict[str, Any]:
        ref, image = get_store().reference_image()
        det = get_detector().detect(image)
        last = self._latest_obs(req)
        tab_stat = tabular_status()
        if tab_stat["available"]:
            tab_res = get_tabular_classifier().classify_tabular(
                last.latitude, last.longitude, last.sst,
                last.pressure_hpa, last.wind_u, last.wind_v)
            return {
                "category": tab_res["category"],
                "scale": "IMD",
                "windSpeedKmh": round(tab_res["wind_speed_kmh"], 1),
                "pressureHpa": round(tab_res["pressure_hpa"], 1),
                "confidence": int(round(tab_res["confidence"] * 100)),
                "structuralPattern": det["structural_pattern"],
            }
        cls = get_image_classifier().classify_image(image)
        return {
            "category": cls["category"],
            "scale": "IMD",
            "windSpeedKmh": round(last.wind_speed_kmh, 1),
            "pressureHpa": round(last.pressure_hpa, 1),
            "confidence": int(round(cls["confidence"] * 100)),
            "structuralPattern": det["structural_pattern"],
        }

    def forecast(self, req: AnalyzeRequest) -> Dict[str, Any]:
        try:
            res = self._adapter.forecast(req)
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError(
                "INFERENCE_ERROR",
                f"forecasting service failed internally ({exc})") from exc
        return {
            "status": "success",
            "model": dict(res["model"]),
            "forecast": self._forecast_items(res),
        }

    def health(self) -> Dict[str, Any]:
        store = get_store()
        f_health = self._adapter.health()
        det_status = get_detector().status()
        ml = {
            "p2": {"model": det_status["model"], "ready": True, "is_candidate": det_status.get("is_candidate", False)},
            "p3_image": {"model": P3_IMAGE_WEIGHTS_REL, "ready": True},
            "p3_tabular": tabular_status(),
            "reference_frame": store.reference_image_name(),
        }
        try:
            store.reference_image_name()
            get_detector()
            get_image_classifier()
        except Exception as exc:
            for k in ("p2", "p3_image"):
                ml[k]["ready"] = False
                ml[k]["error"] = str(exc)
        return {
            "status": "ok",
            "service": "cyclone-integration",
            "phase": "integration_api",
            "offline": True,
            "model_ready": bool(f_health.get("model_ready")),
            "ml": ml,
            "forecasting": f_health,
        }

    def analyze(self, req: AnalyzeRequest) -> Dict[str, Any]:
        meta_in = req.meta
        meta = {
            "systemId": meta_in.systemId if meta_in else DEFAULT_META["systemId"],
            "systemName": meta_in.systemName if meta_in else DEFAULT_META["systemName"],
            "basin": meta_in.basin if meta_in else DEFAULT_META["basin"],
            "lastPass": (meta_in.lastPass if meta_in and meta_in.lastPass
                         else self._latest_obs(req).timestamp),
            "source": (meta_in.source if meta_in and meta_in.source
                       else DEFAULT_META["source"]),
        }

        ref, image = get_store().reference_image()
        det = get_detector().detect(image)
        mov = self._movement(req)
        last = self._latest_obs(req)

        tab = tabular_status()
        if tab["available"]:
            tab_res = get_tabular_classifier().classify_tabular(
                last.latitude, last.longitude, last.sst,
                last.pressure_hpa, last.wind_u, last.wind_v)
            cat_pred = tab_res["category"]
            cat_conf = int(round(tab_res["confidence"] * 100))
            wind_pred = round(tab_res["wind_speed_kmh"], 1)
            pres_pred = round(tab_res["pressure_hpa"], 1)
            class_source = "P3 LightGBM MultiSource Tabular Model (tabular_multisource_model.pkl)"
        else:
            cls_res = get_image_classifier().classify_image(image)
            cat_pred = cls_res["category"]
            cat_conf = int(round(cls_res["confidence"] * 100))
            wind_pred = round(last.wind_speed_kmh, 1)
            pres_pred = round(last.pressure_hpa, 1)
            class_source = P3_IMAGE_WEIGHTS_REL + " (image baseline fallback)"

        forecast_payload = self._adapter.forecast(req)
        items = self._forecast_items(forecast_payload)
        last_ts = _parse_ts(last.timestamp)
        landfall = self._landfall(last_ts, items)
        risk = self._risk(last.wind_speed_kmh, items, landfall)
        hist = self._history_view(req, round(det["confidence"] * 100, 1))

        notes = [
            f"detection: P2 {det.get('model_source', 'CycloneDetector')} on a reference INSAT-3D IR frame",
            f"classification: {class_source}",
            f"observed wind/pressure at t = {last.timestamp}: {last.wind_speed_kmh} km/h, {last.pressure_hpa} hPa",
            "forecast: audited Phase-4 champion EXP005 (GRU+Huber) via phase6 adapter",
            "landfall & risk: deterministic server-side heuristics",
            "satellite boundingBox: null (no audited localizer in bundle)",
        ]

        preview_b64 = ""
        try:
            import base64
            from PIL import Image
            import io
            p_img = image.resize((224, 224), Image.Resampling.BILINEAR)
            p_buf = io.BytesIO()
            p_img.save(p_buf, format="PNG")
            preview_b64 = "data:image/png;base64," + base64.b64encode(p_buf.getvalue()).decode("ascii")
        except Exception:
            pass

        return {
            "meta": {
                **meta,
                "source": f"{meta['source']} · frame {ref}",
            },
            "detection": {
                "detected": det["detected"],
                "confidence": int(round(det["confidence"] * 100)),
                "location": {"lat": round(last.latitude, 4),
                             "lon": round(last.longitude, 4)},
                "movementDirection": mov["direction"],
                "movementSpeedKmh": mov["speed_kmh"],
            },
            "classification": {
                "category": cat_pred,
                "scale": "IMD",
                "windSpeedKmh": wind_pred,
                "pressureHpa": pres_pred,
                "confidence": cat_conf,
                "structuralPattern": det["structural_pattern"],
            },
            "forecast": items,
            "landfall": landfall,
            "risk": risk,
            **hist,
            "satellite": {
                "label": f"INSAT-3D IR · {ref.rsplit('/', 1)[-1]} (reference frame)",
                "boundingBox": None,
                "source": ref,
            },
            "provenance": {
                "pipeline": "p2 (satellite detect) -> p3 (intensity/classification) -> p4 EXP005 (track/wind forecast) -> server heuristics (landfall/risk)",
                "sources": {
                    "detection": det.get("model_source", "P2 CycloneDetector"),
                    "classification": class_source,
                    "tabular": tab["reason"],
                    "forecast": "phase5 EXP005 (GRU+Huber) via phase6 adapter",
                    "reference_frame": ref,
                },
                "notes": notes,
                "reference_image": ref,
                "tabular": tab,
            },
            "preprocessedPreview": preview_b64,
        }

    def analyze_image(self, image, original_filename: str) -> Dict[str, Any]:
        """P2 + P3 on a user-uploaded image.

        ``image`` is a decoded PIL image (RGB) already validated upstream.  We
        reuse the SAME P2 detector and P3 image classifier as the reference-frame
        path — there is no second inference implementation.

        Honest surface differences from /api/analyze:
          * detection.location / movement are null (no localizer; movement needs
            a history — never invented from a single frame);
          * classification exposes only P3's category + softmax confidence
            (P3 wind regressor is degenerate, image pressure is a hardcoded 990);
          * no forecast / landfall / risk / history (those require the validated
            5-observation history via /api/analyze);
          * boundingBox is null (no validated localizer);
          * tabular stays NOT_RUN unless lightgbm is independently available.
        """
        det = get_detector().detect(image)
        cls_res = get_image_classifier().classify_image(image)
        tab = tabular_status()

        safe_name = str(original_filename)
        # provenance labels image_source as USER-UPLOADED IMAGE vs REFERENCE FRAME
        notes = [
            "image_source: USER-UPLOADED IMAGE (not the reference frame)",
            "detection: P2 model_weights.pt (zip, in-memory) on the uploaded "
            "image; pattern/confidence are real model output on synthetic-label "
            "training and are weak / not a validated localizer",
            "classification: P3 image_only_model.pt category/confidence on the "
            "uploaded image; weak/leak-prone evaluation; the P3 wind regressor "
            "and hardcoded image pressure are NOT exposed",
            "location/movement: null — a single image provides no geographic fix "
            "and no motion without a history; the model has no localizer",
            "boundingBox: null (no audited localizer exists in the artifact "
            "bundle)",
            "forecast/landfall/risk: not computed for an image alone — they "
            "require the validated 5-observation history via /api/analyze",
            "confidence: P3 softmax (not a calibrated uncertainty)",
        ]
        if not tab["available"]:
            notes.append(f"tabular classifier NOT_RUN: {tab['reason']}")
        note = "tabular_multisource_model.pkl NOT_RUN (" + tab["reason"] + ")" \
            if not tab["available"] else "available"

        preview_b64 = ""
        try:
            import base64
            p_img = image.resize((224, 224), Image.Resampling.BILINEAR)
            p_buf = io.BytesIO()
            p_img.save(p_buf, format="PNG")
            preview_b64 = "data:image/png;base64," + base64.b64encode(p_buf.getvalue()).decode("ascii")
        except Exception:
            pass

        return {
            "meta": {
                "systemId": DEFAULT_META["systemId"],
                "systemName": DEFAULT_META["systemName"],
                "basin": DEFAULT_META["basin"],
                "lastPass": DEFAULT_META["lastPass"],
                "source": f"USER-UPLOADED IMAGE · {safe_name}",
            },
            "detection": {
                "detected": det["detected"],
                "confidence": int(round(det["confidence"] * 100)),
                "location": None,
                "movementDirection": None,
                "movementSpeedKmh": None,
            },
            "classification": {
                "category": cls_res["category"],
                "scale": "IMD",
                "confidence": int(round(cls_res["confidence"] * 100)),
                "structuralPattern": det["structural_pattern"],
            },
            "satellite": {
                "label": "User-uploaded satellite image",
                "boundingBox": None,
                "source": safe_name,
            },
            "provenance": {
                "pipeline": "p2 (satellite detect) -> p3 (intensity/"
                            "classification); no history -> no forecast/"
                            "landfall/risk here",
                "sources": {
                    "detection": det.get("model_source", "models/detection/model_weights_phase9_E9_2.pt (E9-2 candidate)"),
                    "classification": P3_IMAGE_WEIGHTS_REL + " (image)",
                    "tabular": note,
                    "image_source": "USER-UPLOADED IMAGE",
                },
                "notes": notes,
                "image_source": "USER-UPLOADED IMAGE",
                "tabular": tab,
            },
            "preprocessedPreview": preview_b64,
        }