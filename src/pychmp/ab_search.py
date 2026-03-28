"""Higher-level single-frequency `(a, b, q0)` search workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import MutableMapping, Protocol

import numpy as np

from .fitting import Q0MapRenderer, fit_q0_to_observation
from .metrics import MetricValues
from .optimize import MetricName, ProgressCallback, Q0OptimizationResult


class ABRendererFactory(Protocol):
    """Factory that returns a Q0 renderer for a specific `(a, b)` point."""

    def __call__(self, a: float, b: float) -> Q0MapRenderer:
        """Create a renderer configured for the specified heating parameters."""


@dataclass(frozen=True)
class ABPointResult:
    """Fit result for one `(a, b)` point."""

    a: float
    b: float
    q0: float
    objective_value: float
    metrics: MetricValues
    target_metric: MetricName
    success: bool
    nfev: int
    nit: int
    message: str
    used_adaptive_bracketing: bool
    bracket_found: bool
    bracket: tuple[float, float, float] | None
    trial_q0: tuple[float, ...]
    trial_objective_values: tuple[float, ...]


@dataclass(frozen=True)
class ABScanResult:
    """Summary of a rectangular `(a, b)` single-frequency scan."""

    a_values: tuple[float, ...]
    b_values: tuple[float, ...]
    target_metric: MetricName
    points: tuple[ABPointResult, ...]
    best_q0: np.ndarray
    objective_values: np.ndarray
    chi2: np.ndarray
    rho2: np.ndarray
    eta2: np.ndarray
    success: np.ndarray

    def point_map(self) -> dict[tuple[float, float], ABPointResult]:
        return {(point.a, point.b): point for point in self.points}


ABPointCache = MutableMapping[tuple[float, float], ABPointResult]


def idl_q0_start_heuristic(a: float, b: float) -> float:
    """Return the empirical IDL `Q0_start(a, b)` heuristic.

    This matches `MultiScanAB.pro`:
        Q0_start = exp(-10.1 - 0.193*a + 2.17*b)
    """

    return float(np.exp(-10.1 - 0.193 * float(a) + 2.17 * float(b)))


def _coerce_q0_start_grid(
    q0_start_grid: float | np.ndarray | None,
    *,
    n_a: int,
    n_b: int,
) -> np.ndarray | None:
    if q0_start_grid is None:
        return None
    if np.isscalar(q0_start_grid):
        arr = np.empty((n_a, n_b), dtype=float)
        arr.fill(float(q0_start_grid))
        return arr
    arr = np.asarray(q0_start_grid, dtype=float)
    if arr.shape != (n_a, n_b):
        raise ValueError("q0_start_grid must be scalar or have shape (len(a_values), len(b_values))")
    return arr


def evaluate_ab_point(
    renderer_factory: ABRendererFactory,
    observed: np.ndarray,
    sigma: np.ndarray | None,
    *,
    a: float,
    b: float,
    q0_min: float,
    q0_max: float,
    hard_q0_min: float | None = None,
    hard_q0_max: float | None = None,
    threshold: float = 0.1,
    target_metric: MetricName = "chi2",
    xatol: float = 1e-3,
    maxiter: int = 200,
    adaptive_bracketing: bool = False,
    q0_start: float | None = None,
    q0_step: float = 1.61803398875,
    max_bracket_steps: int = 12,
    progress_callback: ProgressCallback | None = None,
) -> ABPointResult:
    """Evaluate the best-fit `q0` for one `(a, b)` point."""

    result: Q0OptimizationResult = fit_q0_to_observation(
        renderer_factory(float(a), float(b)),
        observed,
        sigma,
        q0_min=q0_min,
        q0_max=q0_max,
        hard_q0_min=hard_q0_min,
        hard_q0_max=hard_q0_max,
        threshold=threshold,
        target_metric=target_metric,
        xatol=xatol,
        maxiter=maxiter,
        adaptive_bracketing=adaptive_bracketing,
        q0_start=q0_start,
        q0_step=q0_step,
        max_bracket_steps=max_bracket_steps,
        progress_callback=progress_callback,
    )
    return ABPointResult(
        a=float(a),
        b=float(b),
        q0=float(result.q0),
        objective_value=float(result.objective_value),
        metrics=result.metrics,
        target_metric=result.target_metric,
        success=bool(result.success),
        nfev=int(result.nfev),
        nit=int(result.nit),
        message=str(result.message),
        used_adaptive_bracketing=bool(result.used_adaptive_bracketing),
        bracket_found=bool(result.bracket_found),
        bracket=result.bracket,
        trial_q0=tuple(float(v) for v in result.trial_q0),
        trial_objective_values=tuple(float(v) for v in result.trial_objective_values),
    )


def multi_scan_ab(
    renderer_factory: ABRendererFactory,
    observed: np.ndarray,
    sigma: np.ndarray | None,
    *,
    a_values: np.ndarray | list[float] | tuple[float, ...],
    b_values: np.ndarray | list[float] | tuple[float, ...],
    q0_min: float,
    q0_max: float,
    hard_q0_min: float | None = None,
    hard_q0_max: float | None = None,
    threshold: float = 0.1,
    target_metric: MetricName = "chi2",
    xatol: float = 1e-3,
    maxiter: int = 200,
    adaptive_bracketing: bool = False,
    q0_start_grid: float | np.ndarray | None = None,
    use_idl_q0_start_heuristic: bool = False,
    q0_step: float = 1.61803398875,
    max_bracket_steps: int = 12,
    progress_callback: ProgressCallback | None = None,
    cache: ABPointCache | None = None,
) -> ABScanResult:
    """Scan a fixed rectangular `(a, b)` grid using nested Q0 fitting.

    The first implementation mirrors the core `MultiScanAB` behavior without
    yet committing to a persistent on-disk resume format. Callers may pass an
    external mutable `cache` mapping to reuse already-computed points across
    repeated invocations.
    """

    a_arr = np.asarray(a_values, dtype=float)
    b_arr = np.asarray(b_values, dtype=float)
    if a_arr.ndim != 1 or b_arr.ndim != 1:
        raise ValueError("a_values and b_values must be one-dimensional")
    if a_arr.size == 0 or b_arr.size == 0:
        raise ValueError("a_values and b_values must be non-empty")

    q0_start_arr = _coerce_q0_start_grid(q0_start_grid, n_a=int(a_arr.size), n_b=int(b_arr.size))
    if q0_start_arr is None and use_idl_q0_start_heuristic:
        q0_start_arr = np.empty((a_arr.size, b_arr.size), dtype=float)
        for i, a in enumerate(a_arr):
            for j, b in enumerate(b_arr):
                q0_start_arr[i, j] = idl_q0_start_heuristic(float(a), float(b))

    cache_map: ABPointCache = {} if cache is None else cache

    points: list[ABPointResult] = []
    best_q0 = np.full((a_arr.size, b_arr.size), np.nan, dtype=float)
    objective_values = np.full((a_arr.size, b_arr.size), np.nan, dtype=float)
    chi2 = np.full((a_arr.size, b_arr.size), np.nan, dtype=float)
    rho2 = np.full((a_arr.size, b_arr.size), np.nan, dtype=float)
    eta2 = np.full((a_arr.size, b_arr.size), np.nan, dtype=float)
    success = np.zeros((a_arr.size, b_arr.size), dtype=bool)

    for i, a in enumerate(a_arr):
        for j, b in enumerate(b_arr):
            key = (float(a), float(b))
            point = cache_map.get(key)
            if point is None:
                point = evaluate_ab_point(
                    renderer_factory,
                    observed,
                    sigma,
                    a=float(a),
                    b=float(b),
                    q0_min=q0_min,
                    q0_max=q0_max,
                    hard_q0_min=hard_q0_min,
                    hard_q0_max=hard_q0_max,
                    threshold=threshold,
                    target_metric=target_metric,
                    xatol=xatol,
                    maxiter=maxiter,
                    adaptive_bracketing=adaptive_bracketing,
                    q0_start=None if q0_start_arr is None else float(q0_start_arr[i, j]),
                    q0_step=q0_step,
                    max_bracket_steps=max_bracket_steps,
                    progress_callback=progress_callback,
                )
                cache_map[key] = point

            points.append(point)
            best_q0[i, j] = point.q0
            objective_values[i, j] = point.objective_value
            chi2[i, j] = point.metrics.chi2
            rho2[i, j] = point.metrics.rho2
            eta2[i, j] = point.metrics.eta2
            success[i, j] = point.success

    return ABScanResult(
        a_values=tuple(float(v) for v in a_arr),
        b_values=tuple(float(v) for v in b_arr),
        target_metric=target_metric,
        points=tuple(points),
        best_q0=best_q0,
        objective_values=objective_values,
        chi2=chi2,
        rho2=rho2,
        eta2=eta2,
        success=success,
    )
