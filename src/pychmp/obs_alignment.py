"""Observation alignment helpers for CHMP-style per-trial shifting.

Provides FindShift-style correlation search and extraction of a model-FOV
reference window from a padded observation canvas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from scipy.ndimage import map_coordinates

FIND_SHIFT_VERSION = "1"
DEFAULT_MAX_SHIFT_ARSEC = 20.0


@dataclass(frozen=True, slots=True)
class FindShiftResult:
    shift_x_arcsec: float
    shift_y_arcsec: float
    correlation: float
    valid: bool
    message: str = ""


def correlate_maps(observed: np.ndarray, modeled: np.ndarray) -> float:
    """Pearson correlation between two same-shaped maps (CHMP correlateMaps)."""
    obs = np.asarray(observed, dtype=float).ravel()
    mod = np.asarray(modeled, dtype=float).ravel()
    finite = np.isfinite(obs) & np.isfinite(mod)
    if not np.any(finite):
        return float("nan")
    obs = obs[finite]
    mod = mod[finite]
    obs_mean = float(np.mean(obs))
    mod_mean = float(np.mean(mod))
    obs_dev = obs - obs_mean
    mod_dev = mod - mod_mean
    denom = float(np.sqrt(np.sum(obs_dev**2)) * np.sqrt(np.sum(mod_dev**2)))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(obs_dev * mod_dev) / denom)


def build_padded_canvas_header(
    model_header: fits.Header,
    *,
    max_shift_arcsec: float,
) -> tuple[fits.Header, int, int]:
    """Return canvas header enclosing model FOV plus shift padding."""
    if max_shift_arcsec <= 0.0:
        raise ValueError("max_shift_arcsec must be positive")

    dx = abs(float(model_header["CDELT1"]))
    dy = abs(float(model_header["CDELT2"]))
    pad_x = int(np.ceil(float(max_shift_arcsec) / dx))
    pad_y = int(np.ceil(float(max_shift_arcsec) / dy))

    model_nx = int(model_header["NAXIS1"])
    model_ny = int(model_header["NAXIS2"])
    canvas_nx = model_nx + 2 * pad_x
    canvas_ny = model_ny + 2 * pad_y

    canvas_header = model_header.copy()
    canvas_header["NAXIS1"] = canvas_nx
    canvas_header["NAXIS2"] = canvas_ny
    canvas_header["CRPIX1"] = float(model_header["CRPIX1"]) + float(pad_x)
    canvas_header["CRPIX2"] = float(model_header["CRPIX2"]) + float(pad_y)
    return canvas_header, pad_x, pad_y


def model_fov_origin_pixels(pad_x: int, pad_y: int) -> tuple[int, int]:
    """Return (y0, x0) of the model FOV within a padded canvas at zero shift."""
    return int(pad_y), int(pad_x)


def extract_observation_to_model_fov(
    canvas_observed: np.ndarray,
    canvas_header: fits.Header,
    model_header: fits.Header,
    *,
    shift_x_arcsec: float = 0.0,
    shift_y_arcsec: float = 0.0,
    canvas_sigma: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract model-FOV obs/sigma from canvas after applying a CHMP-style shift.

    Positive ``shift_x_arcsec`` / ``shift_y_arcsec`` match CHMP ``ExtractSubmap``:
    the observation is shifted by that amount relative to the fixed model grid.
    """
    from .obs_preprocessing import regrid_observation_to_target_fov

    work_header = canvas_header.copy()
    work_header["CRVAL1"] = float(work_header["CRVAL1"]) - float(shift_x_arcsec)
    work_header["CRVAL2"] = float(work_header["CRVAL2"]) - float(shift_y_arcsec)
    observed = regrid_observation_to_target_fov(
        np.asarray(canvas_observed, dtype=float),
        work_header,
        model_header,
    )
    sigma = None
    if canvas_sigma is not None:
        sigma = regrid_observation_to_target_fov(
            np.asarray(canvas_sigma, dtype=float),
            work_header,
            model_header,
        )
    return observed, sigma


def _extract_at_integer_shift(
    canvas: np.ndarray,
    *,
    y0: float,
    x0: float,
    ny: int,
    nx: int,
) -> np.ndarray:
    yy = np.arange(ny, dtype=float) + float(y0)
    xx = np.arange(nx, dtype=float) + float(x0)
    grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")
    sampled = map_coordinates(
        np.asarray(canvas, dtype=float),
        [grid_y, grid_x],
        order=1,
        mode="constant",
        cval=np.nan,
    )
    return np.asarray(sampled, dtype=float)


def find_shift(
    canvas_observed: np.ndarray,
    canvas_header: fits.Header,
    model_header: fits.Header,
    modeled: np.ndarray,
    *,
    max_shift_arcsec: float = DEFAULT_MAX_SHIFT_ARSEC,
    step_arcsec: float = 1.0,
) -> FindShiftResult:
    """Hill-climb correlation search analogous to CHMP FindShift."""
    if step_arcsec <= 0.0:
        raise ValueError("step_arcsec must be positive")

    _, pad_x, pad_y = build_padded_canvas_header(model_header, max_shift_arcsec=max_shift_arcsec)
    model_ny = int(model_header["NAXIS2"])
    model_nx = int(model_header["NAXIS1"])
    dx = abs(float(model_header["CDELT1"]))
    dy = abs(float(model_header["CDELT2"]))
    max_ix = int(np.floor(float(max_shift_arcsec) / dx))
    max_iy = int(np.floor(float(max_shift_arcsec) / dy))

    best_shift_x = 0.0
    best_shift_y = 0.0
    best_score = float("-inf")
    for ix in range(-max_ix, max_ix + 1):
        for iy in range(-max_iy, max_iy + 1):
            shift_x = float(ix * step_arcsec)
            shift_y = float(iy * step_arcsec)
            obs_patch = extract_observation_to_model_fov(
                canvas_observed,
                canvas_header,
                model_header,
                shift_x_arcsec=shift_x,
                shift_y_arcsec=shift_y,
            )[0]
            score = correlate_maps(obs_patch, modeled)
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_shift_x = shift_x
                best_shift_y = shift_y

    valid = np.isfinite(best_score)
    message = "" if valid else "FindShift did not find a finite correlation"
    return FindShiftResult(
        shift_x_arcsec=best_shift_x,
        shift_y_arcsec=best_shift_y,
        correlation=float(best_score),
        valid=valid,
        message=message,
    )
