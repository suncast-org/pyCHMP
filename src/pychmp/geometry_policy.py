"""Observation-driven geometry policy for pyCHMP workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

EARTH_OBSERVER_ALIASES = {
    "earth",
    "sdo",
    "aia",
    "hmi",
    "eovsa",
    "vla",
    "ovsa",
    "rhessi",
}


@dataclass(frozen=True, slots=True)
class GeometryPolicyDecision:
    """Decision about whether model-saved geometry can be reused."""

    observation_observer: str | None
    model_observer: str | None
    los_aligned: bool | None
    use_model_saved_fov: bool
    geometry_mode: str
    observer_name: str
    observer_lonc_deg: float
    observer_b0sun_deg: float
    observer_dsun_cm: float
    reason: str


def normalize_observer_identity(value: Any | None) -> str | None:
    """Normalize common observer names to stable LOS identities."""

    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    compact = text.replace("_", "-").replace(" ", "-")
    if compact in EARTH_OBSERVER_ALIASES or "earth" in compact:
        return "earth"
    if "sdo" in compact or "aia" in compact or "hmi" in compact:
        return "earth"
    if "eovsa" in compact or "radio" in compact:
        return "earth"
    if compact in {"stereo-a", "ahead"} or "stereo-a" in compact:
        return "stereo-a"
    if compact in {"stereo-b", "behind"} or "stereo-b" in compact:
        return "stereo-b"
    return compact


def infer_observation_observer(obs_map: Any) -> str | None:
    """Infer the observation LOS identity from domain, instrument, or metadata."""

    domain = str(getattr(obs_map, "domain", "") or "").strip().lower()
    instrument = str(getattr(obs_map, "instrument", "") or "").strip()
    observer = str(getattr(obs_map, "observer", "") or "").strip()
    if domain == "mw":
        return "earth"
    normalized_instrument = normalize_observer_identity(instrument)
    if normalized_instrument is not None:
        return normalized_instrument
    return normalize_observer_identity(observer)


def infer_model_observer(model_observer_meta: dict[str, Any]) -> str | None:
    """Infer the model-saved observer identity, if metadata is present."""

    for key in ("observer_name", "observer_label"):
        normalized = normalize_observer_identity(model_observer_meta.get(key))
        if normalized is not None:
            return normalized

    lonc = _optional_float(model_observer_meta.get("observer_lonc_deg"))
    b0 = _optional_float(model_observer_meta.get("observer_b0sun_deg"))
    if lonc is not None and b0 is not None and abs(lonc) <= 1.0e-6 and abs(b0) <= 1.0e-6:
        return "earth"
    return None


def resolve_geometry_policy(
    *,
    obs_map: Any,
    model_observer_meta: dict[str, Any],
    saved_fov: dict[str, Any] | None,
    geometry_overrides_requested: bool,
    explicit_observer_requested: bool,
) -> GeometryPolicyDecision:
    """Resolve whether pyCHMP may reuse the model-saved observer/FOV.

    Policy:
    - explicit geometry/observer CLI overrides are honored by the caller
    - MW observations are Earth LOS
    - AIA/SDO observations and refmaps are Earth LOS
    - model-saved observer/FOV is reused only when the observation LOS is
      compatible with the model-saved observer, or when model observer metadata
      is absent
    - if a saved model LOS disagrees with the observation LOS, model-saved FOV is
      not trusted; pyCHMP uses an observation-inscribed FOV and passes the
      observation observer identity downstream
    """

    obs_observer = infer_observation_observer(obs_map)
    model_observer = infer_model_observer(model_observer_meta)
    if obs_observer is None:
        obs_observer = model_observer or "earth"

    if explicit_observer_requested:
        return GeometryPolicyDecision(
            observation_observer=obs_observer,
            model_observer=model_observer,
            los_aligned=None,
            use_model_saved_fov=not geometry_overrides_requested,
            geometry_mode="explicit" if geometry_overrides_requested else "saved_fov",
            observer_name=str(model_observer_meta.get("observer_name", obs_observer)),
            observer_lonc_deg=float(_optional_float(model_observer_meta.get("observer_lonc_deg")) or 0.0),
            observer_b0sun_deg=float(_optional_float(model_observer_meta.get("observer_b0sun_deg")) or 0.0),
            observer_dsun_cm=float(_optional_float(model_observer_meta.get("observer_dsun_cm")) or 1.495978707e13),
            reason="explicit observer override requested",
        )

    if geometry_overrides_requested:
        return GeometryPolicyDecision(
            observation_observer=obs_observer,
            model_observer=model_observer,
            los_aligned=(None if model_observer is None else obs_observer == model_observer),
            use_model_saved_fov=False,
            geometry_mode="explicit",
            observer_name=obs_observer,
            observer_lonc_deg=0.0 if obs_observer == "earth" else float(_optional_float(model_observer_meta.get("observer_lonc_deg")) or 0.0),
            observer_b0sun_deg=0.0 if obs_observer == "earth" else float(_optional_float(model_observer_meta.get("observer_b0sun_deg")) or 0.0),
            observer_dsun_cm=float(_optional_float(model_observer_meta.get("observer_dsun_cm")) or 1.495978707e13),
            reason="explicit geometry override requested",
        )

    aligned = None if model_observer is None else obs_observer == model_observer
    if saved_fov is not None and (model_observer is None or aligned):
        return GeometryPolicyDecision(
            observation_observer=obs_observer,
            model_observer=model_observer,
            los_aligned=aligned,
            use_model_saved_fov=True,
            geometry_mode="saved_fov",
            observer_name=str(model_observer_meta.get("observer_name", obs_observer)),
            observer_lonc_deg=float(_optional_float(model_observer_meta.get("observer_lonc_deg")) or 0.0),
            observer_b0sun_deg=float(_optional_float(model_observer_meta.get("observer_b0sun_deg")) or 0.0),
            observer_dsun_cm=float(_optional_float(model_observer_meta.get("observer_dsun_cm")) or 1.495978707e13),
            reason="model saved FOV LOS is compatible with observation",
        )

    return GeometryPolicyDecision(
        observation_observer=obs_observer,
        model_observer=model_observer,
        los_aligned=aligned,
        use_model_saved_fov=False,
        geometry_mode="observation_inscribed_fov",
        observer_name=obs_observer,
        observer_lonc_deg=0.0 if obs_observer == "earth" else float(_optional_float(model_observer_meta.get("observer_lonc_deg")) or 0.0),
        observer_b0sun_deg=0.0 if obs_observer == "earth" else float(_optional_float(model_observer_meta.get("observer_b0sun_deg")) or 0.0),
        observer_dsun_cm=float(_optional_float(model_observer_meta.get("observer_dsun_cm")) or 1.495978707e13),
        reason="observation LOS differs from model saved observer; using observation-inscribed FOV",
    )


def _optional_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None
