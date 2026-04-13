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
ProgressStartCallback: TypeAlias = Callable[[int, float], None]
ProgressCallback: TypeAlias = Callable[[float, float, bool, str, float], None]


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
    trial_chi2_values: tuple[float, ...] = ()
    trial_rho2_values: tuple[float, ...] = ()
    trial_eta2_values: tuple[float, ...] = ()


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
    fallback_to_bounded_refinement: bool = True
    boundary_q0: float | None = None


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
    progress_start_callback: ProgressStartCallback | None = None,
    progress_callback: ProgressCallback | None = None,
) -> _Q0EvaluationRecord:
    q0 = float(q0)
    cached = cache.get(q0)
    if cached is not None:
        return cached

    t0 = math.nan
    try:
        import time as _time

        if progress_start_callback is not None:
            progress_start_callback(len(evaluation_order) + 1, q0)
        t0 = _time.perf_counter()
        evaluation = _normalize_metric_result(metric_function(q0))
        elapsed_s = _time.perf_counter() - t0
    except Exception:
        # Preserve original exception behavior while still allowing the normal
        # control flow to surface the failure to callers.
        raise
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
    if progress_callback is not None:
        progress_callback(record.q0, record.objective_value, record.is_valid, record.message, float(elapsed_s))
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


def _trial_metric_histories(
    cache: dict[float, _Q0EvaluationRecord],
    evaluation_order: list[float],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    return (
        tuple(float(cache[q0].metrics.chi2) for q0 in evaluation_order),
        tuple(float(cache[q0].metrics.rho2) for q0 in evaluation_order),
        tuple(float(cache[q0].metrics.eta2) for q0 in evaluation_order),
    )


def _valid_records_in_range(
    records: dict[float, _Q0EvaluationRecord],
    *,
    q0_lo: float,
    q0_hi: float,
) -> list[_Q0EvaluationRecord]:
    lo = min(float(q0_lo), float(q0_hi))
    hi = max(float(q0_lo), float(q0_hi))
    return sorted(
        (
            record
            for record in records.values()
            if record.is_valid and lo <= float(record.q0) <= hi
        ),
        key=lambda item: item.q0,
    )


def _nearest_valid_neighbors(
    records: dict[float, _Q0EvaluationRecord],
    *,
    center_q0: float,
    q0_lo: float,
    q0_hi: float,
) -> tuple[_Q0EvaluationRecord | None, _Q0EvaluationRecord | None]:
    valid = _valid_records_in_range(records, q0_lo=q0_lo, q0_hi=q0_hi)
    center = float(center_q0)
    left = None
    right = None
    for record in valid:
        if float(record.q0) < center:
            left = record
        elif float(record.q0) > center and right is None:
            right = record
            break
    return left, right


def _geometric_midpoint(q0_a: float, q0_b: float) -> float:
    return float(math.sqrt(float(q0_a) * float(q0_b)))


def _refine_sampled_neighborhood(
    *,
    bracket: tuple[float, float, float],
    metric_function: Callable[[float], MetricFunctionResult],
    target_metric: MetricName,
    cache: dict[float, _Q0EvaluationRecord],
    evaluation_order: list[float],
    progress_start_callback: ProgressStartCallback | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[float, float]:
    q0_lo = float(bracket[0])
    q0_hi = float(bracket[2])
    candidates = _valid_records_in_range(cache, q0_lo=q0_lo, q0_hi=q0_hi)
    if len(candidates) < 3:
        return (q0_lo, q0_hi)

    best_record = min(candidates, key=lambda item: item.objective_value)
    left_neighbor, right_neighbor = _nearest_valid_neighbors(
        cache,
        center_q0=float(best_record.q0),
        q0_lo=q0_lo,
        q0_hi=q0_hi,
    )

    midpoint_candidates: list[float] = []
    if left_neighbor is not None and float(left_neighbor.q0) > 0.0 and float(left_neighbor.q0) < float(best_record.q0):
        midpoint_candidates.append(_geometric_midpoint(float(left_neighbor.q0), float(best_record.q0)))
    if right_neighbor is not None and float(best_record.q0) > 0.0 and float(best_record.q0) < float(right_neighbor.q0):
        midpoint_candidates.append(_geometric_midpoint(float(best_record.q0), float(right_neighbor.q0)))

    for q0_value in midpoint_candidates:
        _evaluate_q0(
            q0_value,
            metric_function=metric_function,
            target_metric=target_metric,
            cache=cache,
            evaluation_order=evaluation_order,
            progress_start_callback=progress_start_callback,
            progress_callback=progress_callback,
        )

    candidates = _valid_records_in_range(cache, q0_lo=q0_lo, q0_hi=q0_hi)
    if not candidates:
        return (q0_lo, q0_hi)
    best_record = min(candidates, key=lambda item: item.objective_value)
    left_neighbor, right_neighbor = _nearest_valid_neighbors(
        cache,
        center_q0=float(best_record.q0),
        q0_lo=q0_lo,
        q0_hi=q0_hi,
    )
    refined_lo = float(left_neighbor.q0) if left_neighbor is not None else q0_lo
    refined_hi = float(right_neighbor.q0) if right_neighbor is not None else q0_hi
    return (refined_lo, refined_hi)


def _resolve_effective_xatol(xatol: float, *, bounds: tuple[float, float]) -> float:
    span = abs(float(bounds[1]) - float(bounds[0]))
    if span <= 0.0:
        return float(xatol)
    return min(float(xatol), max(0.05 * span, 1e-12))


def _boundary_best_q0(
    records: dict[float, _Q0EvaluationRecord],
    *,
    side: Literal["lower", "upper"],
) -> float | None:
    ordered = sorted(records.values(), key=lambda item: item.q0)
    if len(ordered) < 2:
        return None

    if side == "lower":
        boundary = ordered[0]
        neighbor = ordered[1]
    else:
        boundary = ordered[-1]
        neighbor = ordered[-2]

    if not (boundary.is_valid and neighbor.is_valid):
        return None
    if boundary.objective_value <= neighbor.objective_value:
        return boundary.q0
    return None


def _best_valid_q0(records: dict[float, _Q0EvaluationRecord]) -> float | None:
    candidates = [record for record in records.values() if record.is_valid]
    if not candidates:
        return None
    return float(min(candidates, key=lambda item: item.objective_value).q0)


def _step_q0(
    q0: float,
    *,
    direction: int,
    q0_step: float,
    hard_q0_min: float | None,
    hard_q0_max: float | None,
) -> float:
    if direction > 0:
        next_q0 = q0 * q0_step
        if hard_q0_max is not None:
            next_q0 = min(next_q0, hard_q0_max)
        return next_q0
    next_q0 = q0 / q0_step
    if hard_q0_min is not None:
        next_q0 = max(next_q0, hard_q0_min)
    return next_q0


def _choose_direction_from_triplet(
    left_record: _Q0EvaluationRecord | None,
    middle_record: _Q0EvaluationRecord,
    right_record: _Q0EvaluationRecord | None,
) -> int:
    left_obj = left_record.objective_value if left_record is not None and left_record.is_valid else math.inf
    middle_obj = middle_record.objective_value if middle_record.is_valid else math.inf
    right_obj = right_record.objective_value if right_record is not None and right_record.is_valid else math.inf

    if left_obj < middle_obj and left_obj <= right_obj:
        return -1
    if right_obj < middle_obj and right_obj < left_obj:
        return 1
    if math.isfinite(right_obj) and not math.isfinite(left_obj):
        return 1
    if math.isfinite(left_obj) and not math.isfinite(right_obj):
        return -1
    return 1


def _choose_initial_direction(
    start_record: _Q0EvaluationRecord,
    *,
    q0_start: float,
    hard_q0_min: float | None,
    hard_q0_max: float | None,
    q0_step: float,
    metric_function: Callable[[float], MetricFunctionResult],
    target_metric: MetricName,
    cache: dict[float, _Q0EvaluationRecord],
    evaluation_order: list[float],
    progress_start_callback: ProgressStartCallback | None = None,
    progress_callback: ProgressCallback | None = None,
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

    left_q0 = _step_q0(q0_start, direction=-1, q0_step=q0_step, hard_q0_min=hard_q0_min, hard_q0_max=hard_q0_max)
    right_q0 = _step_q0(q0_start, direction=1, q0_step=q0_step, hard_q0_min=hard_q0_min, hard_q0_max=hard_q0_max)

    left_record = None
    right_record = None
    if left_q0 < q0_start:
        left_record = _evaluate_q0(
            left_q0,
            metric_function=metric_function,
            target_metric=target_metric,
            cache=cache,
            evaluation_order=evaluation_order,
            progress_start_callback=progress_start_callback,
            progress_callback=progress_callback,
        )
    if right_q0 > q0_start:
        right_record = _evaluate_q0(
            right_q0,
            metric_function=metric_function,
            target_metric=target_metric,
            cache=cache,
            evaluation_order=evaluation_order,
            progress_start_callback=progress_start_callback,
            progress_callback=progress_callback,
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
    hard_q0_min: float | None,
    hard_q0_max: float | None,
    q0_step: float,
    max_bracket_steps: int,
    target_metric: MetricName,
    cache: dict[float, _Q0EvaluationRecord],
    evaluation_order: list[float],
    progress_start_callback: ProgressStartCallback | None = None,
    progress_callback: ProgressCallback | None = None,
) -> _BracketSearchResult:
    start_record = None
    left_record = None
    right_record = None
    for q0_value in (float(q0_min), float(q0_start), float(q0_max)):
        record = _evaluate_q0(
            q0_value,
            metric_function=metric_function,
            target_metric=target_metric,
            cache=cache,
            evaluation_order=evaluation_order,
            progress_start_callback=progress_start_callback,
            progress_callback=progress_callback,
        )
        if math.isclose(q0_value, float(q0_start), rel_tol=0.0, abs_tol=0.0):
            start_record = record
        if math.isclose(q0_value, float(q0_min), rel_tol=0.0, abs_tol=0.0):
            left_record = record
        if math.isclose(q0_value, float(q0_max), rel_tol=0.0, abs_tol=0.0):
            right_record = record

    assert start_record is not None
    if not start_record.is_valid:
        return _BracketSearchResult(bracket=None, steps_taken=0, message="adaptive bracketing start point is invalid")

    bracket = _find_bracket(cache)
    if bracket is not None:
        return _BracketSearchResult(bracket=bracket, steps_taken=0, message="adaptive bracketing found initial triplet")

    direction = _choose_direction_from_triplet(left_record, start_record, right_record)
    current_q0 = float(q0_min) if direction < 0 else float(q0_max)
    steps_taken = 0
    first_q0 = _step_q0(
        current_q0,
        direction=direction,
        q0_step=q0_step,
        hard_q0_min=hard_q0_min,
        hard_q0_max=hard_q0_max,
    )
    if math.isclose(first_q0, current_q0, rel_tol=0.0, abs_tol=0.0):
        side = "upper" if direction > 0 else "lower"
        boundary_q0 = _boundary_best_q0(cache, side=side)
        if boundary_q0 is not None:
            return _BracketSearchResult(
                bracket=None,
                steps_taken=steps_taken,
                message=f"adaptive bracketing hit the {side} safety bound while the objective was still improving; stopping at the boundary best instead of falling back",
                fallback_to_bounded_refinement=False,
                boundary_q0=boundary_q0,
            )
        return _BracketSearchResult(
            bracket=None,
            steps_taken=steps_taken,
            message=f"adaptive bracketing hit the {side} safety bound without finding an interior minimum",
        )

    while steps_taken < max_bracket_steps:
        current_q0 = first_q0 if steps_taken == 0 else _step_q0(
            current_q0,
            direction=direction,
            q0_step=q0_step,
            hard_q0_min=hard_q0_min,
            hard_q0_max=hard_q0_max,
        )
        if steps_taken > 0 and math.isclose(current_q0, previous_q0, rel_tol=0.0, abs_tol=0.0):
            side = "upper" if direction > 0 else "lower"
            boundary_q0 = _boundary_best_q0(cache, side=side)
            if boundary_q0 is not None:
                return _BracketSearchResult(
                    bracket=None,
                    steps_taken=steps_taken,
                    message=f"adaptive bracketing hit the {side} safety bound while the objective was still improving; stopping at the boundary best instead of falling back",
                    fallback_to_bounded_refinement=False,
                    boundary_q0=boundary_q0,
                )
            if hard_q0_min is not None and hard_q0_max is not None:
                return _BracketSearchResult(
                    bracket=None,
                    steps_taken=steps_taken,
                    message=f"adaptive bracketing hit the {side} safety bound without finding an interior minimum",
                    fallback_to_bounded_refinement=True,
                )
            best_q0 = _best_valid_q0(cache)
            return _BracketSearchResult(
                bracket=None,
                steps_taken=steps_taken,
                message="adaptive bracketing exhausted the reachable search region without finding an interior minimum",
                fallback_to_bounded_refinement=False,
                boundary_q0=best_q0,
            )

        previous_q0 = current_q0
        eval_count_before = len(evaluation_order)
        _evaluate_q0(
            current_q0,
            metric_function=metric_function,
            target_metric=target_metric,
            cache=cache,
            evaluation_order=evaluation_order,
            progress_start_callback=progress_start_callback,
            progress_callback=progress_callback,
        )
        if len(evaluation_order) > eval_count_before:
            steps_taken += 1

        bracket = _find_bracket(cache)
        if bracket is not None:
            return _BracketSearchResult(
                bracket=bracket,
                steps_taken=steps_taken,
                message="adaptive bracketing found a valid interior minimum",
            )
        first_q0 = _step_q0(
            current_q0,
            direction=direction,
            q0_step=q0_step,
            hard_q0_min=hard_q0_min,
            hard_q0_max=hard_q0_max,
        )

    if hard_q0_min is not None and hard_q0_max is not None:
        return _BracketSearchResult(
            bracket=None,
            steps_taken=steps_taken,
            message="adaptive bracketing exhausted the step budget without finding a bracket",
            fallback_to_bounded_refinement=True,
        )

    best_q0 = _best_valid_q0(cache)
    return _BracketSearchResult(
        bracket=None,
        steps_taken=steps_taken,
        message="adaptive bracketing exhausted the step budget without finding a bracket",
        fallback_to_bounded_refinement=False,
        boundary_q0=best_q0,
    )


def find_best_q0(
    metric_function: Callable[[float], MetricFunctionResult],
    *,
    q0_min: float,
    q0_max: float,
    hard_q0_min: float | None = None,
    hard_q0_max: float | None = None,
    target_metric: MetricName = "chi2",
    xatol: float = 1e-3,
    maxiter: int = 200,
    adaptive_bracketing: bool = False,
    q0_start: float | None = None,
    q0_step: float = 1.61803398875,
    max_bracket_steps: int = 12,
    progress_start_callback: ProgressStartCallback | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Q0OptimizationResult:
    """Find best Q0 with optional adaptive multiplicative bracketing.

    q0_min and q0_max define the user-provided initialization interval. In
    adaptive mode the search may expand beyond that interval unless explicit
    hard_q0_min / hard_q0_max safety bounds are supplied.
    """
    if q0_min <= 0 or q0_max <= 0:
        raise ValueError("q0_min and q0_max must be positive")
    if q0_min >= q0_max:
        raise ValueError("q0_min must be less than q0_max")
    if hard_q0_min is not None and hard_q0_min <= 0:
        raise ValueError("hard_q0_min must be positive")
    if hard_q0_max is not None and hard_q0_max <= 0:
        raise ValueError("hard_q0_max must be positive")
    if hard_q0_min is not None and hard_q0_max is not None and hard_q0_min >= hard_q0_max:
        raise ValueError("hard_q0_min must be less than hard_q0_max")
    if hard_q0_min is not None and q0_min < hard_q0_min:
        raise ValueError("q0_min must not lie below hard_q0_min")
    if hard_q0_max is not None and q0_max > hard_q0_max:
        raise ValueError("q0_max must not lie above hard_q0_max")
    if q0_step <= 1.0:
        raise ValueError("q0_step must be greater than 1 for multiplicative bracketing")
    if max_bracket_steps < 1:
        raise ValueError("max_bracket_steps must be at least 1")

    if q0_start is None:
        q0_start = math.sqrt(q0_min * q0_max)
    q0_start = float(q0_start)
    if not (q0_min <= q0_start <= q0_max):
        raise ValueError("q0_start must lie within [q0_min, q0_max]")
    if hard_q0_min is not None and q0_start < hard_q0_min:
        raise ValueError("q0_start must not lie below hard_q0_min")
    if hard_q0_max is not None and q0_start > hard_q0_max:
        raise ValueError("q0_start must not lie above hard_q0_max")

    cache: dict[float, _Q0EvaluationRecord] = {}
    evaluation_order: list[float] = []

    def objective(q0: float) -> float:
        record = _evaluate_q0(
            float(q0),
            metric_function=metric_function,
            target_metric=target_metric,
            cache=cache,
            evaluation_order=evaluation_order,
            progress_start_callback=progress_start_callback,
            progress_callback=progress_callback,
        )
        return record.objective_value

    lower_refinement_bound = float(hard_q0_min) if hard_q0_min is not None else float(q0_min)
    upper_refinement_bound = float(hard_q0_max) if hard_q0_max is not None else float(q0_max)
    refinement_bounds = (lower_refinement_bound, upper_refinement_bound)
    bracket: tuple[float, float, float] | None = None
    bracket_found = False
    bracket_steps = 0
    message_prefix = ""
    boundary_q0: float | None = None

    if adaptive_bracketing:
        bracket_result = _adaptive_multiplicative_bracket(
            metric_function,
            q0_min=q0_min,
            q0_max=q0_max,
            q0_start=q0_start,
            hard_q0_min=hard_q0_min,
            hard_q0_max=hard_q0_max,
            q0_step=q0_step,
            max_bracket_steps=max_bracket_steps,
            target_metric=target_metric,
            cache=cache,
            evaluation_order=evaluation_order,
            progress_start_callback=progress_start_callback,
            progress_callback=progress_callback,
        )
        bracket = bracket_result.bracket
        bracket_found = bracket is not None
        bracket_steps = bracket_result.steps_taken
        boundary_q0 = bracket_result.boundary_q0
        if bracket is not None:
            refinement_bounds = _refine_sampled_neighborhood(
                bracket=bracket,
                metric_function=metric_function,
                target_metric=target_metric,
                cache=cache,
                evaluation_order=evaluation_order,
                progress_start_callback=progress_start_callback,
                progress_callback=progress_callback,
            )
            message_prefix = bracket_result.message
        elif not bracket_result.fallback_to_bounded_refinement and boundary_q0 is not None:
            boundary_record = cache[float(boundary_q0)]
            trial_q0 = tuple(evaluation_order)
            trial_objective_values = tuple(cache[q0].objective_value for q0 in evaluation_order)
            trial_chi2_values, trial_rho2_values, trial_eta2_values = _trial_metric_histories(cache, evaluation_order)
            return Q0OptimizationResult(
                q0=float(boundary_record.q0),
                objective_value=boundary_record.objective_value,
                metrics=boundary_record.metrics,
                target_metric=target_metric,
                success=False,
                nfev=len(cache),
                nit=int(bracket_steps),
                message=bracket_result.message,
                used_adaptive_bracketing=True,
                bracket_found=False,
                bracket=None,
                trial_q0=trial_q0,
                trial_objective_values=trial_objective_values,
                trial_chi2_values=trial_chi2_values,
                trial_rho2_values=trial_rho2_values,
                trial_eta2_values=trial_eta2_values,
            )
        else:
            message_prefix = f"{bracket_result.message}; falling back to bounded refinement"

    effective_xatol = _resolve_effective_xatol(xatol, bounds=refinement_bounds)
    result = minimize_scalar(
        objective,
        bounds=refinement_bounds,
        method="bounded",
        options={"xatol": effective_xatol, "maxiter": maxiter},
    )

    best_q0 = float(result.x)
    best_record = _evaluate_q0(
        best_q0,
        metric_function=metric_function,
        target_metric=target_metric,
        cache=cache,
        evaluation_order=evaluation_order,
        progress_start_callback=progress_start_callback,
        progress_callback=progress_callback,
    )

    trial_q0 = tuple(evaluation_order)
    trial_objective_values = tuple(cache[q0].objective_value for q0 in evaluation_order)
    trial_chi2_values, trial_rho2_values, trial_eta2_values = _trial_metric_histories(cache, evaluation_order)
    result_message = str(result.message)
    if message_prefix:
        result_message = f"{message_prefix}; {result_message}"

    # Post-processing: flag as not successful if best_q0 is at the boundary
    tol = 1e-8
    boundary_failure = False
    if abs(best_q0 - refinement_bounds[0]) < tol or abs(best_q0 - refinement_bounds[1]) < tol:
        boundary_failure = True
        result_message += " WARNING: Minimum is at the boundary of the search region; true minimum may lie outside."

    return Q0OptimizationResult(
        q0=best_q0,
        objective_value=best_record.objective_value,
        metrics=best_record.metrics,
        target_metric=target_metric,
        success=bool(result.success) and not boundary_failure,
        nfev=len(cache),
        nit=int(result.nit) + int(bracket_steps),
        message=result_message,
        used_adaptive_bracketing=bool(adaptive_bracketing),
        bracket_found=bracket_found,
        bracket=bracket,
        trial_q0=trial_q0,
        trial_objective_values=trial_objective_values,
        trial_chi2_values=trial_chi2_values,
        trial_rho2_values=trial_rho2_values,
        trial_eta2_values=trial_eta2_values,
    )
