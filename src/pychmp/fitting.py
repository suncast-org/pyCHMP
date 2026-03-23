"""High-level fitting entry points that connect renderers to optimization."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .metrics import MetricValues, compute_metrics, threshold_union_mask
from .optimize import MetricName, Q0MetricEvaluation, Q0OptimizationResult, find_best_q0


class Q0MapRenderer(Protocol):
    """Protocol for render adapters that synthesize a map for a given Q0."""

    def render(self, q0: float) -> np.ndarray:
        """Return modeled map corresponding to Q0."""


def fit_q0_to_observation(
    renderer: Q0MapRenderer,
    observed: np.ndarray,
    sigma: np.ndarray | None,
    *,
    q0_min: float,
    q0_max: float,
    threshold: float = 0.1,
    target_metric: MetricName = "chi2",
    xatol: float = 1e-3,
    maxiter: int = 200,
    adaptive_bracketing: bool = False,
    q0_start: float | None = None,
    q0_step: float = 1.61803398875,
    max_bracket_steps: int = 12,
) -> Q0OptimizationResult:
    """Optimize Q0 by comparing rendered maps against observed maps.

    If sigma is None or unsuitable for chi2 calculation, falls back to eta2.
    """
    observed_arr = np.asarray(observed, dtype=float)
    
    # Handle None or unsuitable sigma
    actual_target_metric = target_metric
    if sigma is None:
        import warnings
        warnings.warn(
            "sigma is None (map noise estimation failed). "
            f"Falling back from {target_metric} to eta2 metric.",
            UserWarning
        )
        actual_target_metric = "eta2"
        # Use uniform dummy sigma for metric computation
        sigma_arr = np.ones_like(observed_arr)
    else:
        sigma_arr = np.asarray(sigma, dtype=float)

        if observed_arr.shape != sigma_arr.shape:
            raise ValueError("observed and sigma must have identical shapes")

    def metric_function(q0: float) -> Q0MetricEvaluation:
        modeled_arr = np.asarray(renderer.render(float(q0)), dtype=float)
        if modeled_arr.shape != observed_arr.shape:
            raise ValueError("renderer output shape must match observed shape")

        mask = threshold_union_mask(observed_arr, modeled_arr, threshold)
        metrics = compute_metrics(observed_arr, modeled_arr, sigma_arr, mask)
        return Q0MetricEvaluation(
            metrics=metrics,
            total_observed_flux=float(np.sum(observed_arr[mask], dtype=float)),
            total_modeled_flux=float(np.sum(modeled_arr[mask], dtype=float)),
        )

    return find_best_q0(
        metric_function,
        q0_min=q0_min,
        q0_max=q0_max,
        target_metric=actual_target_metric,
        xatol=xatol,
        maxiter=maxiter,
        adaptive_bracketing=adaptive_bracketing,
        q0_start=q0_start,
        q0_step=q0_step,
        max_bracket_steps=max_bracket_steps,
    )
