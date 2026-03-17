"""High-level fitting entry points that connect renderers to optimization."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .metrics import MetricValues, compute_metrics, threshold_union_mask
from .optimize import MetricName, Q0OptimizationResult, find_best_q0


class Q0MapRenderer(Protocol):
    """Protocol for render adapters that synthesize a map for a given Q0."""

    def render(self, q0: float) -> np.ndarray:
        """Return modeled map corresponding to Q0."""


def fit_q0_to_observation(
    renderer: Q0MapRenderer,
    observed: np.ndarray,
    sigma: np.ndarray,
    *,
    q0_min: float,
    q0_max: float,
    threshold: float = 0.1,
    target_metric: MetricName = "chi2",
    xatol: float = 1e-3,
    maxiter: int = 200,
) -> Q0OptimizationResult:
    """Optimize Q0 by comparing rendered maps against observed maps."""
    observed_arr = np.asarray(observed, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)

    if observed_arr.shape != sigma_arr.shape:
        raise ValueError("observed and sigma must have identical shapes")

    def metric_function(q0: float) -> MetricValues:
        modeled_arr = np.asarray(renderer.render(float(q0)), dtype=float)
        if modeled_arr.shape != observed_arr.shape:
            raise ValueError("renderer output shape must match observed shape")

        mask = threshold_union_mask(observed_arr, modeled_arr, threshold)
        return compute_metrics(observed_arr, modeled_arr, sigma_arr, mask)

    return find_best_q0(
        metric_function,
        q0_min=q0_min,
        q0_max=q0_max,
        target_metric=target_metric,
        xatol=xatol,
        maxiter=maxiter,
    )
