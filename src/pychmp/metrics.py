"""Core fit metrics used by pyCHMP search workflows.

These functions mirror the metric definitions used in CHMP-style fitting:
chi2, rho2, eta2, and threshold-based union masks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
) -> np.ndarray:
    """Return mask where observed or modeled is above thresholded maxima."""
    observed, modeled = _validate_mask_inputs(observed, modeled, threshold)

    obs_max = float(np.max(observed))
    mod_max = float(np.max(modeled))

    obs_mask = observed > (obs_max * threshold)
    mod_mask = modeled > (mod_max * threshold)
    return np.logical_or(obs_mask, mod_mask)


def threshold_data_mask(observed: np.ndarray, _modeled: np.ndarray, threshold: float) -> np.ndarray:
    """Return mask where observed is above thresholded maximum."""
    observed, _modeled = _validate_mask_inputs(observed, _modeled, threshold)
    obs_max = float(np.max(observed))
    return observed > (obs_max * threshold)


def threshold_model_mask(observed: np.ndarray, modeled: np.ndarray, threshold: float) -> np.ndarray:
    """Return mask where modeled is above thresholded maximum."""
    observed, modeled = _validate_mask_inputs(observed, modeled, threshold)
    mod_max = float(np.max(modeled))
    return modeled > (mod_max * threshold)


def threshold_and_mask(observed: np.ndarray, modeled: np.ndarray, threshold: float) -> np.ndarray:
    """Return mask where both observed and modeled are above thresholded maxima."""
    observed, modeled = _validate_mask_inputs(observed, modeled, threshold)
    obs_max = float(np.max(observed))
    mod_max = float(np.max(modeled))
    obs_mask = observed > (obs_max * threshold)
    mod_mask = modeled > (mod_max * threshold)
    return np.logical_and(obs_mask, mod_mask)


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
