"""Optimization helpers for CHMP-style one-dimensional Q0 fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from scipy.optimize import minimize_scalar

from .metrics import MetricValues

MetricName = Literal["chi2", "rho2", "eta2"]


@dataclass(frozen=True)
class Q0OptimizationResult:
    """Result container for one-dimensional Q0 optimization."""

    q0: float
    objective_value: float
    metrics: MetricValues
    target_metric: MetricName
    success: bool
    nfev: int
    nit: int
    message: str


def _metric_value(metrics: MetricValues, target_metric: MetricName) -> float:
    if target_metric == "chi2":
        return metrics.chi2
    if target_metric == "rho2":
        return metrics.rho2
    if target_metric == "eta2":
        return metrics.eta2
    raise ValueError(f"unsupported target_metric: {target_metric}")


def find_best_q0(
    metric_function: Callable[[float], MetricValues],
    *,
    q0_min: float,
    q0_max: float,
    target_metric: MetricName = "chi2",
    xatol: float = 1e-3,
    maxiter: int = 200,
) -> Q0OptimizationResult:
    """Find best Q0 in [q0_min, q0_max] with scalar bounded optimization.

    This function intentionally focuses on the refinement stage. Bracketing logic
    analogous to legacy CHMP adaptive expansion can be layered on top.
    """
    if q0_min <= 0 or q0_max <= 0:
        raise ValueError("q0_min and q0_max must be positive")
    if q0_min >= q0_max:
        raise ValueError("q0_min must be less than q0_max")

    def objective(q0: float) -> float:
        metrics = metric_function(float(q0))
        return _metric_value(metrics, target_metric)

    result = minimize_scalar(
        objective,
        bounds=(q0_min, q0_max),
        method="bounded",
        options={"xatol": xatol, "maxiter": maxiter},
    )

    best_q0 = float(result.x)
    best_metrics = metric_function(best_q0)
    best_value = _metric_value(best_metrics, target_metric)

    return Q0OptimizationResult(
        q0=best_q0,
        objective_value=best_value,
        metrics=best_metrics,
        target_metric=target_metric,
        success=bool(result.success),
        nfev=int(result.nfev),
        nit=int(result.nit),
        message=str(result.message),
    )
