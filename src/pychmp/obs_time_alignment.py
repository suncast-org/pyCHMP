"""Observation-time alignment against the forward-model reference epoch.

Workflows compare the observational map epoch to the model observation time.
When the shift is small enough, the observation is differentially rotated to the
model epoch before regridding to the render FOV (``obs_preprocessing``). Large or
cross-day mismatches emit warnings only. Metric computation itself does not
rotate or regrid; callers use ``prepare_observation_for_metrics``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
from astropy.io import fits


DEFAULT_MAX_ROTATION_SECONDS = 12.0 * 3600.0
_ROTATION_THRESHOLD_SECONDS = 1.0


def normalize_observation_time_unix(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%d-%b-%Y %H:%M:%S.%f",
        "%d-%b-%Y %H:%M:%S",
    ):
        try:
            parsed = datetime.strptime(text, fmt)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return float(parsed.timestamp())
        except ValueError:
            continue
    try:
        from dateutil.parser import parse as parse_datetime

        parsed = parse_datetime(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return float(parsed.timestamp())
    except Exception:
        return None


def load_model_obs_time_text(model_path: Any) -> str | None:
    import h5py

    path = str(model_path or "").strip()
    if not path:
        return None
    try:
        with h5py.File(path, "r") as handle:
            for key in ("observer/pb0r/obs_date", "metadata/date_obs", "metadata/date-obs"):
                if key not in handle:
                    continue
                raw = np.asarray(handle[key]).reshape(-1)[0]
                if isinstance(raw, bytes):
                    text = raw.decode("utf-8", errors="replace").strip()
                else:
                    text = str(raw).strip()
                if text:
                    return text
    except Exception:
        return None
    return None


@dataclass(frozen=True, slots=True)
class ObsModelTimeAlignment:
    observation_time_text: str | None
    model_time_text: str | None
    observation_time_unix: float | None
    model_time_unix: float | None
    delta_seconds: float | None
    same_utc_day: bool | None
    compatibility: str
    message: str

    def warning_lines(self) -> list[str]:
        if self.compatibility in {"exact", "unknown"}:
            return []
        prefix = "WARNING" if self.compatibility != "incompatible" else "WARNING"
        return [f"{prefix}: {self.message}"]

    def should_rotate(self) -> bool:
        if self.compatibility not in {"rotatable", "warn"}:
            return False
        if self.delta_seconds is None:
            return False
        return abs(float(self.delta_seconds)) > _ROTATION_THRESHOLD_SECONDS


def assess_obs_model_time_alignment(
    observation_time: Any,
    model_time: Any,
    *,
    max_rotation_seconds: float = DEFAULT_MAX_ROTATION_SECONDS,
) -> ObsModelTimeAlignment:
    obs_text = str(observation_time or "").strip() or None
    model_text = str(model_time or "").strip() or None
    obs_unix = normalize_observation_time_unix(obs_text)
    model_unix = normalize_observation_time_unix(model_text)

    if obs_unix is None or model_unix is None:
        return ObsModelTimeAlignment(
            observation_time_text=obs_text,
            model_time_text=model_text,
            observation_time_unix=obs_unix,
            model_time_unix=model_unix,
            delta_seconds=None,
            same_utc_day=None,
            compatibility="unknown",
            message="Could not parse both observation and model times; skipping solar-rotation alignment.",
        )

    delta_seconds = float(model_unix - obs_unix)
    obs_day = datetime.fromtimestamp(obs_unix, tz=timezone.utc).date()
    model_day = datetime.fromtimestamp(model_unix, tz=timezone.utc).date()
    same_utc_day = obs_day == model_day

    if abs(delta_seconds) <= _ROTATION_THRESHOLD_SECONDS:
        return ObsModelTimeAlignment(
            observation_time_text=obs_text,
            model_time_text=model_text,
            observation_time_unix=obs_unix,
            model_time_unix=model_unix,
            delta_seconds=delta_seconds,
            same_utc_day=same_utc_day,
            compatibility="exact",
            message="Observation and model times match within one second.",
        )

    if not same_utc_day:
        return ObsModelTimeAlignment(
            observation_time_text=obs_text,
            model_time_text=model_text,
            observation_time_unix=obs_unix,
            model_time_unix=model_unix,
            delta_seconds=delta_seconds,
            same_utc_day=False,
            compatibility="incompatible",
            message=(
                "Observation and model times fall on different UTC days "
                f"({obs_text!r} vs {model_text!r}, delta={delta_seconds:.1f}s). "
                "Solar-rotation alignment was skipped; provide a reference map closer to the model epoch."
            ),
        )

    if abs(delta_seconds) > float(max_rotation_seconds):
        return ObsModelTimeAlignment(
            observation_time_text=obs_text,
            model_time_text=model_text,
            observation_time_unix=obs_unix,
            model_time_unix=model_unix,
            delta_seconds=delta_seconds,
            same_utc_day=True,
            compatibility="warn",
            message=(
                "Observation and model times are on the same UTC day but differ by "
                f"{abs(delta_seconds)/3600.0:.2f} h (>{max_rotation_seconds/3600.0:.1f} h policy limit). "
                "Solar rotation was applied, but verify that the reference map is close enough to the model epoch."
            ),
        )

    return ObsModelTimeAlignment(
        observation_time_text=obs_text,
        model_time_text=model_text,
        observation_time_unix=obs_unix,
        model_time_unix=model_unix,
        delta_seconds=delta_seconds,
        same_utc_day=True,
        compatibility="rotatable",
        message=(
            "Observation time differs from model time by "
            f"{abs(delta_seconds):.1f} s; applying solar differential rotation to the model epoch."
        ),
    )


def _astropy_time_from_text(value: Any):
    from astropy.time import Time

    unix = normalize_observation_time_unix(value)
    if unix is not None:
        return Time(float(unix), format="unix")
    return Time(str(value))


def align_observation_to_model_time(
    data: np.ndarray,
    header: fits.Header,
    *,
    model_time_text: str,
    alignment: ObsModelTimeAlignment | None = None,
) -> tuple[np.ndarray, fits.Header, dict[str, Any]]:
    alignment = alignment or assess_obs_model_time_alignment(
        header.get("DATE-OBS", header.get("DATE_OBS", "")),
        model_time_text,
    )
    diagnostics: dict[str, Any] = {
        "observation_time_original": alignment.observation_time_text,
        "model_time_reference": alignment.model_time_text,
        "observation_time_delta_s": alignment.delta_seconds,
        "observation_time_alignment": alignment.compatibility,
        "observation_time_rotation_applied": False,
        "observation_time_alignment_message": alignment.message,
    }
    if not alignment.should_rotate():
        return np.asarray(data, dtype=float), header.copy(), diagnostics

    try:
        from sunpy.coordinates import transform_with_sun_center
        from sunpy.map import Map
        from sunpy.physics.differential_rotation import differential_rotate
    except Exception as exc:
        diagnostics["observation_time_rotation_error"] = str(exc)
        diagnostics["observation_time_alignment_message"] = (
            f"{alignment.message} Solar rotation was requested but SunPy alignment dependencies are unavailable."
        )
        return np.asarray(data, dtype=float), header.copy(), diagnostics

    try:
        source_map = Map(np.asarray(data, dtype=float), header.copy())
        target_time = _astropy_time_from_text(model_time_text)
        with transform_with_sun_center():
            rotated_map = differential_rotate(source_map, time=target_time)
    except Exception as exc:
        diagnostics["observation_time_rotation_error"] = str(exc)
        diagnostics["observation_time_alignment_message"] = (
            f"{alignment.message} Solar rotation was requested but failed: {exc}"
        )
        return np.asarray(data, dtype=float), header.copy(), diagnostics

    out_header = rotated_map.fits_header.copy()
    out_header["DATE-OBS"] = str(model_time_text)
    diagnostics["observation_time_rotation_applied"] = True
    diagnostics["observation_time_aligned_to"] = str(model_time_text)
    diagnostics["observation_time_alignment_message"] = alignment.message
    return np.asarray(rotated_map.data, dtype=float), out_header, diagnostics
