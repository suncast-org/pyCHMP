"""Core fit metrics used by pyCHMP search workflows.

These functions mirror the metric definitions used in CHMP-style fitting:
chi2, rho2, eta2, and threshold-based union masks.

Callers must preprocess observations before calling ``compute_metrics``:
solar rotation to the model epoch (when applicable) and regridding to the render
FOV. This module performs pure pixel-wise comparisons on already aligned arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits


@dataclass(frozen=True)
class MetricValues:
    """Container for per-comparison fit metrics."""

    chi2: float
    rho2: float
    eta2: float


def _validate_mask_inputs(
    observed: np.ndarray,
    modeled: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    observed_arr = np.asarray(observed)
    modeled_arr = np.asarray(modeled)
    if observed_arr.shape != modeled_arr.shape:
        raise ValueError("observed and modeled must have identical shapes")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be between 0 and 1")
    return observed_arr, modeled_arr


def threshold_union_mask(
    observed: np.ndarray,
    modeled: np.ndarray,
    threshold: float,
    *,
    obs_max: float | None = None,
) -> np.ndarray:
    """Return mask where observed or modeled is above thresholded maxima."""
    observed, modeled = _validate_mask_inputs(observed, modeled, threshold)

    obs_peak = float(np.max(observed)) if obs_max is None else float(obs_max)
    mod_max = float(np.max(modeled))

    obs_mask = observed > (obs_peak * threshold)
    mod_mask = modeled > (mod_max * threshold)
    return np.logical_or(obs_mask, mod_mask)


def threshold_data_mask(
    observed: np.ndarray,
    _modeled: np.ndarray,
    threshold: float,
    *,
    obs_max: float | None = None,
) -> np.ndarray:
    """Return mask where observed is above thresholded maximum."""
    observed, _modeled = _validate_mask_inputs(observed, _modeled, threshold)
    obs_peak = float(np.max(observed)) if obs_max is None else float(obs_max)
    return observed > (obs_peak * threshold)


def threshold_model_mask(observed: np.ndarray, modeled: np.ndarray, threshold: float) -> np.ndarray:
    """Return mask where modeled is above thresholded maximum."""
    observed, modeled = _validate_mask_inputs(observed, modeled, threshold)
    mod_max = float(np.max(modeled))
    return modeled > (mod_max * threshold)


def threshold_and_mask(
    observed: np.ndarray,
    modeled: np.ndarray,
    threshold: float,
    *,
    obs_max: float | None = None,
) -> np.ndarray:
    """Return mask where both observed and modeled are above thresholded maxima."""
    observed, modeled = _validate_mask_inputs(observed, modeled, threshold)
    obs_peak = float(np.max(observed)) if obs_max is None else float(obs_max)
    mod_max = float(np.max(modeled))
    obs_mask = observed > (obs_peak * threshold)
    mod_mask = modeled > (mod_max * threshold)
    return np.logical_and(obs_mask, mod_mask)


def smoothed_observation_max(
    observed: np.ndarray,
    *,
    pixel_scale_x_arcsec: float,
    pixel_scale_y_arcsec: float | None = None,
) -> float:
    """Estimate obs peak using a local Gaussian fit (CHMP GetSmoothedMax analogue)."""
    arr = np.asarray(observed, dtype=float)
    if arr.size == 0:
        return float("nan")
    peak_flat = int(np.nanargmax(arr))
    peak_y, peak_x = np.unravel_index(peak_flat, arr.shape)
    peak_value = float(arr[peak_y, peak_x])
    sx = max(abs(float(pixel_scale_x_arcsec)), 1e-6)
    sy = max(abs(float(pixel_scale_y_arcsec or pixel_scale_x_arcsec)), 1e-6)
    radius_pix = max(int(np.ceil(max(sx, sy) / min(sx, sy))), 1)
    y0 = max(int(peak_y) - radius_pix, 0)
    y1 = min(int(peak_y) + radius_pix + 1, arr.shape[0])
    x0 = max(int(peak_x) - radius_pix, 0)
    x1 = min(int(peak_x) + radius_pix + 1, arr.shape[1])
    patch = arr[y0:y1, x0:x1]
    if patch.size < 4:
        return peak_value
    yy, xx = np.mgrid[y0:y1, x0:x1]
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    values = patch.ravel()
    floor = float(np.nanmin(values))
    amplitude = max(peak_value - floor, 0.0)

    def gaussian2d(flat_coords: np.ndarray, a: float, b: float, xc: float, yc: float, s: float) -> np.ndarray:
        x = flat_coords[:, 0]
        y = flat_coords[:, 1]
        return b + a * np.exp(-0.5 * (((x - xc) / s) ** 2 + ((y - yc) / s) ** 2))

    try:
        from scipy.optimize import curve_fit

        p0 = (amplitude, floor, float(peak_x), float(peak_y), max(sx, sy))
        fitted, _status = curve_fit(
            gaussian2d,
            coords,
            values,
            p0=p0,
            maxfev=1000,
        )
        return float(fitted[0] + fitted[1])
    except Exception:
        return peak_value


def mask_area_fractions(
    observed: np.ndarray,
    modeled: np.ndarray,
    *,
    threshold: float,
    obs_max: float | None = None,
) -> tuple[float, float]:
    """Return (mask_obs_fraction, mask_mod_fraction) like CHMP maskObs/maskMod."""
    observed_arr, modeled_arr = _validate_mask_inputs(observed, modeled, threshold)
    total = float(observed_arr.size)
    if total <= 0.0:
        return 0.0, 0.0
    obs_peak = float(np.max(observed_arr)) if obs_max is None else float(obs_max)
    mod_max = float(np.max(modeled_arr))
    mask_obs = float(np.count_nonzero(observed_arr > (obs_peak * threshold))) / total
    mask_mod = float(np.count_nonzero(modeled_arr > (mod_max * threshold))) / total
    return mask_obs, mask_mod


def chmp_mask_valid(mask_obs_fraction: float, mask_mod_fraction: float) -> tuple[bool, str]:
    if mask_mod_fraction > 0.99:
        return False, "model mask fraction > 0.99 (GR contribution too low)"
    if mask_obs_fraction > 0.0 and (mask_mod_fraction / mask_obs_fraction) > 4.0:
        return False, "maskMod/maskObs > 4"
    return True, ""


def pixel_scales_from_wcs_header(header: fits.Header) -> tuple[float, float]:
    return abs(float(header["CDELT1"])), abs(float(header["CDELT2"]))


def resolve_metrics_mask_type(diagnostics: dict[str, Any]) -> str:
    """Resolve the threshold mask type used for display for one trial/point."""
    mask_source = str(diagnostics.get("metrics_mask_source", "")).strip().lower()
    if mask_source == "explicit_fits":
        return "explicit"
    selected_index = diagnostics.get("selected_trial_index")
    trial_stages = diagnostics.get("fit_trial_mask_stages") or diagnostics.get("trial_mask_stages")
    if selected_index is not None and trial_stages:
        try:
            stage = trial_stages[int(selected_index)]
            stage_text = str(stage).strip().lower()
            if stage_text in {"union", "data", "model", "and"}:
                return stage_text
        except (IndexError, TypeError, ValueError):
            pass
    mask_type = str(diagnostics.get("mask_type", "union")).strip().lower()
    return mask_type or "union"


def resolve_observation_peak_for_mask(
    observed: np.ndarray,
    diagnostics: dict[str, Any],
    *,
    wcs_header: fits.Header | None = None,
) -> float | None:
    use_smoothed = diagnostics.get("use_smoothed_obs_max")
    if use_smoothed is None:
        use_smoothed = True
    if not bool(use_smoothed):
        return None
    if wcs_header is None:
        return None
    scale_x, scale_y = pixel_scales_from_wcs_header(wcs_header)
    return smoothed_observation_max(
        observed,
        pixel_scale_x_arcsec=scale_x,
        pixel_scale_y_arcsec=scale_y,
    )


def resolve_metrics_threshold_mask(
    observed: np.ndarray,
    modeled: np.ndarray,
    diagnostics: dict[str, Any],
    *,
    wcs_header: fits.Header | None = None,
) -> np.ndarray | None:
    """Build the CHMP-faithful metrics ROI mask for visualization."""
    mask_source = str(diagnostics.get("metrics_mask_source", "")).strip().lower()
    if mask_source == "explicit_fits":
        return None
    mask_type = resolve_metrics_mask_type(diagnostics)
    threshold = float(diagnostics.get("metrics_mask_threshold", diagnostics.get("threshold", 0.1)))
    obs_max = resolve_observation_peak_for_mask(observed, diagnostics, wcs_header=wcs_header)
    mask_fn = resolve_threshold_mask(mask_type)
    if mask_type in {"union", "data", "and"}:
        return mask_fn(observed, modeled, threshold, obs_max=obs_max)
    return mask_fn(observed, modeled, threshold)


def format_metrics_mask_label(diagnostics: dict[str, Any]) -> str:
    mask_source = str(diagnostics.get("metrics_mask_source", "")).strip().lower()
    if mask_source == "explicit_fits":
        mask_path = str(diagnostics.get("metrics_mask_fits", "")).strip()
        return f"ROI mask: explicit ({Path(mask_path).name if mask_path else 'FITS'})"
    mask_type = resolve_metrics_mask_type(diagnostics)
    threshold_value = diagnostics.get("metrics_mask_threshold", diagnostics.get("threshold"))
    try:
        threshold_text = f"{float(threshold_value):.3f}"
    except Exception:
        threshold_text = "n/a"
    peak_note = ""
    if bool(diagnostics.get("use_smoothed_obs_max", True)):
        peak_note = ", smoothed obs peak"
    return f"ROI mask: {mask_type} @ {threshold_text}{peak_note}"


def resolve_threshold_mask(mask_type: str):
    normalized = str(mask_type).strip().lower()
    mask_fn = {
        "union": threshold_union_mask,
        "data": threshold_data_mask,
        "model": threshold_model_mask,
        "and": threshold_and_mask,
    }.get(normalized)
    if mask_fn is None:
        supported = ", ".join(("union", "data", "model", "and"))
        raise ValueError(f"unsupported mask_type {mask_type!r}; expected one of: {supported}")
    return mask_fn


def compute_metrics(
    observed: np.ndarray,
    modeled: np.ndarray,
    sigma: np.ndarray,
    mask: np.ndarray,
) -> MetricValues:
    """Compute chi2, rho2, eta2 on masked elements.

    Formulae:
    - chi2 = mean(((modeled - observed) / sigma)^2)
    - rho2 = mean((modeled / observed - 1)^2)
    - eta2 = mean(((modeled - observed) / mean(observed))^2)
    """
    if not (observed.shape == modeled.shape == sigma.shape == mask.shape):
        raise ValueError("all inputs must have identical shapes")

    idx = np.asarray(mask, dtype=bool)
    if not np.any(idx):
        raise ValueError("mask selects no elements")

    obs = observed[idx].astype(float)
    mod = modeled[idx].astype(float)
    sig = sigma[idx].astype(float)

    if np.any(sig == 0):
        raise ValueError("sigma contains zero values in selected elements")

    nonzero = obs > 0
    if not np.any(nonzero):
        raise ValueError("observed contains no positive values in selected elements")

    chi2 = float(np.mean(((mod - obs) / sig) ** 2))
    rho2 = float(np.mean((mod[nonzero] / obs[nonzero] - 1.0) ** 2))
    eta2 = float(np.mean(((mod - obs) / float(np.mean(obs[nonzero]))) ** 2))

    return MetricValues(chi2=chi2, rho2=rho2, eta2=eta2)
