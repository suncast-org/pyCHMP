"""Optimization helpers for CHMP-style one-dimensional Q0 fitting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal, TypeAlias

from scipy.optimize import minimize_scalar

from .metrics import MetricValues

MetricName = Literal["chi2", "rho2", "eta2"]


@dataclass(frozen=True)
class Q0MetricEvaluation:
    """Extended per-Q0 evaluation payload used by adaptive search.

    Existing callers may continue returning plain MetricValues; this record is
    only needed when the optimizer should also use flux diagnostics or validity
    flags.
    """

    metrics: MetricValues
    total_observed_flux: float | None = None
    total_modeled_flux: float | None = None
    is_valid: bool = True
    message: str = ""


MetricFunctionResult: TypeAlias = MetricValues | Q0MetricEvaluation


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
    used_adaptive_bracketing: bool = False
    bracket_found: bool = False
    bracket: tuple[float, float, float] | None = None
    trial_q0: tuple[float, ...] = ()
    trial_objective_values: tuple[float, ...] = ()


@dataclass(frozen=True)
class _Q0EvaluationRecord:
    q0: float
    objective_value: float
    metrics: MetricValues
    total_observed_flux: float | None
    total_modeled_flux: float | None
    is_valid: bool
    message: str


@dataclass(frozen=True)
class _BracketSearchResult:
    bracket: tuple[float, float, float] | None
    steps_taken: int
    message: str


def _metric_value(metrics: MetricValues, target_metric: MetricName) -> float:
    if target_metric == "chi2":
        return metrics.chi2
    if target_metric == "rho2":
        return metrics.rho2
    if target_metric == "eta2":
        return metrics.eta2
    raise ValueError(f"unsupported target_metric: {target_metric}")


def _normalize_metric_result(result: MetricFunctionResult) -> Q0MetricEvaluation:
    if isinstance(result, Q0MetricEvaluation):
        return result
    if isinstance(result, MetricValues):
        return Q0MetricEvaluation(metrics=result)
    raise TypeError("metric_function must return MetricValues or Q0MetricEvaluation")


def _evaluate_q0(
    q0: float,
    *,
    metric_function: Callable[[float], MetricFunctionResult],
    target_metric: MetricName,
    cache: dict[float, _Q0EvaluationRecord],
    evaluation_order: list[float],
) -> _Q0EvaluationRecord:
    q0 = float(q0)
    cached = cache.get(q0)
    if cached is not None:
        return cached

    evaluation = _normalize_metric_result(metric_function(q0))
    objective_value = float(_metric_value(evaluation.metrics, target_metric))
    is_valid = bool(evaluation.is_valid) and math.isfinite(objective_value)
    record = _Q0EvaluationRecord(
        q0=q0,
        objective_value=objective_value,
        metrics=evaluation.metrics,
        total_observed_flux=evaluation.total_observed_flux,
        total_modeled_flux=evaluation.total_modeled_flux,
        is_valid=is_valid,
        message=str(evaluation.message or ""),
    )
    cache[q0] = record
    evaluation_order.append(q0)
    return record


def _find_bracket(records: dict[float, _Q0EvaluationRecord]) -> tuple[float, float, float] | None:
    ordered = sorted(records.values(), key=lambda item: item.q0)
    candidates: list[tuple[float, float, float, float]] = []
    for idx in range(1, len(ordered) - 1):
        left = ordered[idx - 1]
        mid = ordered[idx]
        right = ordered[idx + 1]
        if not (left.is_valid and mid.is_valid and right.is_valid):
            continue
        if mid.objective_value <= left.objective_value and mid.objective_value <= right.objective_value:
            candidates.append((mid.objective_value, left.q0, mid.q0, right.q0))
    if not candidates:
        return None
    _obj, qa, qb, qc = min(candidates, key=lambda item: item[0])
    return (qa, qb, qc)


def _step_q0(q0: float, *, direction: int, q0_step: float, q0_min: float, q0_max: float) -> float:
    if direction > 0:
        return min(q0 * q0_step, q0_max)
    return max(q0 / q0_step, q0_min)


def _choose_initial_direction(
    start_record: _Q0EvaluationRecord,
    *,
    q0_start: float,
    q0_min: float,
    q0_max: float,
    q0_step: float,
    metric_function: Callable[[float], MetricFunctionResult],
    target_metric: MetricName,
    cache: dict[float, _Q0EvaluationRecord],
    evaluation_order: list[float],
) -> int:
    if (
        start_record.total_observed_flux is not None
        and start_record.total_modeled_flux is not None
        and math.isfinite(float(start_record.total_observed_flux))
        and math.isfinite(float(start_record.total_modeled_flux))
    ):
        flux_delta = float(start_record.total_observed_flux) - float(start_record.total_modeled_flux)
        if flux_delta > 0.0:
            return 1
        if flux_delta < 0.0:
            return -1

    left_q0 = _step_q0(q0_start, direction=-1, q0_step=q0_step, q0_min=q0_min, q0_max=q0_max)
    right_q0 = _step_q0(q0_start, direction=1, q0_step=q0_step, q0_min=q0_min, q0_max=q0_max)

    left_record = None
    right_record = None
    if left_q0 < q0_start:
        left_record = _evaluate_q0(
            left_q0,
            metric_function=metric_function,
            target_metric=target_metric,
            cache=cache,
            evaluation_order=evaluation_order,
        )
    if right_q0 > q0_start:
        right_record = _evaluate_q0(
            right_q0,
            metric_function=metric_function,
            target_metric=target_metric,
            cache=cache,
            evaluation_order=evaluation_order,
        )

    if left_record and left_record.is_valid and right_record and right_record.is_valid:
        return 1 if right_record.objective_value < left_record.objective_value else -1
    if right_record and right_record.is_valid:
        return 1
    if left_record and left_record.is_valid:
        return -1
    return 1


def _adaptive_multiplicative_bracket(
    metric_function: Callable[[float], MetricFunctionResult],
    *,
    q0_min: float,
    q0_max: float,
    q0_start: float,
    q0_step: float,
    max_bracket_steps: int,
    target_metric: MetricName,
    cache: dict[float, _Q0EvaluationRecord],
    evaluation_order: list[float],
) -> _BracketSearchResult:
    start_record = _evaluate_q0(
        q0_start,
        metric_function=metric_function,
        target_metric=target_metric,
        cache=cache,
        evaluation_order=evaluation_order,
    )
    if not start_record.is_valid:
        return _BracketSearchResult(bracket=None, steps_taken=0, message="adaptive bracketing start point is invalid")

    bracket = _find_bracket(cache)
    if bracket is not None:
        return _BracketSearchResult(bracket=bracket, steps_taken=0, message="adaptive bracketing found initial triplet")

    direction = _choose_initial_direction(
        start_record,
        q0_start=q0_start,
        q0_min=q0_min,
        q0_max=q0_max,
        q0_step=q0_step,
        metric_function=metric_function,
        target_metric=target_metric,
        cache=cache,
        evaluation_order=evaluation_order,
    )

    bracket = _find_bracket(cache)
    if bracket is not None:
        return _BracketSearchResult(bracket=bracket, steps_taken=0, message="adaptive bracketing found triplet around start")

    current_q0 = q0_start
    steps_taken = 0
    while steps_taken < max_bracket_steps:
        next_q0 = _step_q0(current_q0, direction=direction, q0_step=q0_step, q0_min=q0_min, q0_max=q0_max)
        if math.isclose(next_q0, current_q0, rel_tol=0.0, abs_tol=0.0):
            side = "upper" if direction > 0 else "lower"
            return _BracketSearchResult(
                bracket=None,
                steps_taken=steps_taken,
                message=f"adaptive bracketing hit the {side} safety bound without finding an interior minimum",
            )

        current_q0 = next_q0
        steps_taken += 1
        _evaluate_q0(
            current_q0,
            metric_function=metric_function,
            target_metric=target_metric,
            cache=cache,
            evaluation_order=evaluation_order,
        )

        bracket = _find_bracket(cache)
        if bracket is not None:
            return _BracketSearchResult(
                bracket=bracket,
                steps_taken=steps_taken,
                message="adaptive bracketing found a valid interior minimum",
            )

    return _BracketSearchResult(
        bracket=None,
        steps_taken=steps_taken,
        message="adaptive bracketing exhausted the step budget without finding a bracket",
    )


def find_best_q0(
    metric_function: Callable[[float], MetricFunctionResult],
    *,
    q0_min: float,
    q0_max: float,
    target_metric: MetricName = "chi2",
    xatol: float = 1e-3,
    maxiter: int = 200,
    adaptive_bracketing: bool = False,
    q0_start: float | None = None,
    q0_step: float = 1.61803398875,
    max_bracket_steps: int = 12,
) -> Q0OptimizationResult:
    """Find best Q0 with optional adaptive multiplicative bracketing.

    When adaptive bracketing is enabled, q0_min and q0_max act as hard safety
    bounds. If the bracketing phase fails, the optimizer falls back to bounded
    refinement over the full safety interval and reports that fallback.
    """
    if q0_min <= 0 or q0_max <= 0:
        raise ValueError("q0_min and q0_max must be positive")
    if q0_min >= q0_max:
        raise ValueError("q0_min must be less than q0_max")
    if q0_step <= 1.0:
        raise ValueError("q0_step must be greater than 1 for multiplicative bracketing")
    if max_bracket_steps < 1:
        raise ValueError("max_bracket_steps must be at least 1")

    if q0_start is None:
        q0_start = math.sqrt(q0_min * q0_max)
    q0_start = float(q0_start)
    if not (q0_min <= q0_start <= q0_max):
        raise ValueError("q0_start must lie within [q0_min, q0_max]")

    cache: dict[float, _Q0EvaluationRecord] = {}
    evaluation_order: list[float] = []

    def objective(q0: float) -> float:
        record = _evaluate_q0(
            float(q0),
            metric_function=metric_function,
            target_metric=target_metric,
            cache=cache,
            evaluation_order=evaluation_order,
        )
        return record.objective_value

    refinement_bounds = (q0_min, q0_max)
    bracket: tuple[float, float, float] | None = None
    bracket_found = False
    bracket_steps = 0
    message_prefix = ""

    if adaptive_bracketing:
        bracket_result = _adaptive_multiplicative_bracket(
            metric_function,
            q0_min=q0_min,
            q0_max=q0_max,
            q0_start=q0_start,
            q0_step=q0_step,
            max_bracket_steps=max_bracket_steps,
            target_metric=target_metric,
            cache=cache,
            evaluation_order=evaluation_order,
        )
        bracket = bracket_result.bracket
        bracket_found = bracket is not None
        bracket_steps = bracket_result.steps_taken
        if bracket is not None:
            refinement_bounds = (bracket[0], bracket[2])
            message_prefix = bracket_result.message
        else:
            message_prefix = f"{bracket_result.message}; falling back to bounded refinement"

    result = minimize_scalar(
        objective,
        bounds=refinement_bounds,
        method="bounded",
        options={"xatol": xatol, "maxiter": maxiter},
    )

    best_q0 = float(result.x)
    best_record = _evaluate_q0(
        best_q0,
        metric_function=metric_function,
        target_metric=target_metric,
        cache=cache,
        evaluation_order=evaluation_order,
    )

    trial_q0 = tuple(evaluation_order)
    trial_objective_values = tuple(cache[q0].objective_value for q0 in evaluation_order)
    result_message = str(result.message)
    if message_prefix:
        result_message = f"{message_prefix}; {result_message}"

    return Q0OptimizationResult(
        q0=best_q0,
        objective_value=best_record.objective_value,
        metrics=best_record.metrics,
        target_metric=target_metric,
        success=bool(result.success),
        nfev=len(cache),
        nit=int(result.nit) + int(bracket_steps),
        message=result_message,
        used_adaptive_bracketing=bool(adaptive_bracketing),
        bracket_found=bracket_found,
        bracket=bracket,
        trial_q0=trial_q0,
        trial_objective_values=trial_objective_values,
    )
