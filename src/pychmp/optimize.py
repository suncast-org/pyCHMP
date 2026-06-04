"""Optimization helpers for CHMP-style one-dimensional Q0 fitting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal, Mapping, TypeAlias

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
    shift_x_arcsec: float = 0.0
    shift_y_arcsec: float = 0.0
    find_shift_valid: bool = True
    mask_obs_fraction: float | None = None
    mask_mod_fraction: float | None = None
    find_shift_version: str = ""
    chmp_eval_policy_version: str = ""
    mask_stage: str = ""


MetricFunctionResult: TypeAlias = MetricValues | Q0MetricEvaluation
InitialQ0Evaluations: TypeAlias = Mapping[float, MetricFunctionResult]
ProgressStartCallback: TypeAlias = Callable[[int, float], None]
ProgressCallback: TypeAlias = (
    Callable[[float, float, bool, str, float, MetricValues, Q0MetricEvaluation], None]
    | Callable[[float, float, bool, str, float, MetricValues], None]
)


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
    boundary_constrained: bool = False
    trial_q0: tuple[float, ...] = ()
    trial_objective_values: tuple[float, ...] = ()
    trial_chi2_values: tuple[float, ...] = ()
    trial_rho2_values: tuple[float, ...] = ()
    trial_eta2_values: tuple[float, ...] = ()
    trial_shift_x_arcsec: tuple[float, ...] = ()
    trial_shift_y_arcsec: tuple[float, ...] = ()
    trial_find_shift_valid: tuple[bool, ...] = ()
    q0_search_stages: tuple[str, ...] = ()
    trial_mask_stages: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Q0EvaluationRecord:
    q0: float
    objective_value: float
    metrics: MetricValues
    total_observed_flux: float | None
    total_modeled_flux: float | None
    is_valid: bool
    message: str
    shift_x_arcsec: float = 0.0
    shift_y_arcsec: float = 0.0
    find_shift_valid: bool = True
    mask_stage: str = ""


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
        shift_x_arcsec=float(evaluation.shift_x_arcsec),
        shift_y_arcsec=float(evaluation.shift_y_arcsec),
        find_shift_valid=bool(evaluation.find_shift_valid),
        mask_stage=str(evaluation.mask_stage or ""),
    )
    cache[q0] = record
    evaluation_order.append(q0)
    if progress_callback is not None:
        evaluation_payload = Q0MetricEvaluation(
            metrics=record.metrics,
            total_observed_flux=record.total_observed_flux,
            total_modeled_flux=record.total_modeled_flux,
            is_valid=record.is_valid,
            message=record.message,
            shift_x_arcsec=float(record.shift_x_arcsec),
            shift_y_arcsec=float(record.shift_y_arcsec),
            find_shift_valid=bool(record.find_shift_valid),
            mask_stage=str(record.mask_stage or ""),
        )
        try:
            progress_callback(
                record.q0,
                record.objective_value,
                record.is_valid,
                record.message,
                float(elapsed_s),
                record.metrics,
                evaluation_payload,
            )
        except TypeError:
            progress_callback(
                record.q0,
                record.objective_value,
                record.is_valid,
                record.message,
                float(elapsed_s),
                record.metrics,
            )
    return record


def _seed_initial_evaluations(
    initial_evaluations: InitialQ0Evaluations | None,
    *,
    target_metric: MetricName,
    cache: dict[float, _Q0EvaluationRecord],
    evaluation_order: list[float],
    hard_q0_min: float | None = None,
    hard_q0_max: float | None = None,
) -> int:
    """Load already-scored Q0 samples into the optimizer cache."""
    if not initial_evaluations:
        return 0
    seeded = 0
    seed_items: list[tuple[float, MetricFunctionResult]] = []
    for q0_raw, evaluation_raw in initial_evaluations.items():
        try:
            q0 = float(q0_raw)
        except Exception:
            continue
        seed_items.append((q0, evaluation_raw))
    for q0, evaluation_raw in sorted(seed_items, key=lambda item: item[0]):
        if not math.isfinite(q0) or q0 <= 0.0:
            continue
        if hard_q0_min is not None and q0 < float(hard_q0_min):
            continue
        if hard_q0_max is not None and q0 > float(hard_q0_max):
            continue
        if q0 in cache:
            continue
        try:
            evaluation = _normalize_metric_result(evaluation_raw)
            objective_value = float(_metric_value(evaluation.metrics, target_metric))
        except Exception:
            continue
        is_valid = bool(evaluation.is_valid) and math.isfinite(objective_value)
        record = _Q0EvaluationRecord(
            q0=q0,
            objective_value=objective_value,
            metrics=evaluation.metrics,
            total_observed_flux=evaluation.total_observed_flux,
            total_modeled_flux=evaluation.total_modeled_flux,
            is_valid=is_valid,
            message=str(evaluation.message or "seeded from saved trial map"),
            shift_x_arcsec=float(evaluation.shift_x_arcsec),
            shift_y_arcsec=float(evaluation.shift_y_arcsec),
            find_shift_valid=bool(evaluation.find_shift_valid),
            mask_stage=str(evaluation.mask_stage or ""),
        )
        cache[q0] = record
        evaluation_order.append(q0)
        seeded += 1
    return seeded


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


def _trial_shift_histories(
    cache: dict[float, _Q0EvaluationRecord],
    evaluation_order: list[float],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[bool, ...]]:
    return (
        tuple(float(cache[q0].shift_x_arcsec) for q0 in evaluation_order),
        tuple(float(cache[q0].shift_y_arcsec) for q0 in evaluation_order),
        tuple(bool(cache[q0].find_shift_valid) for q0 in evaluation_order),
    )


def _trial_mask_stage_histories(
    cache: dict[float, _Q0EvaluationRecord],
    evaluation_order: list[float],
) -> tuple[str, ...]:
    return tuple(str(cache[q0].mask_stage or "") for q0 in evaluation_order)


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
        return 1 if right_obj < middle_obj else -1
    if math.isfinite(left_obj) and not math.isfinite(right_obj):
        return -1 if left_obj < middle_obj else 1
    if right_obj > middle_obj and left_obj > middle_obj:
        return -1 if middle_obj > right_obj else 1
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


_IDL_RELATIVE_ACC_DEFAULT = 1e-2
_IDL_GOLDEN_RATIO = (1.0 + math.sqrt(5.0)) / 2.0


def _i_ratio(a: float, b: float) -> float:
    """Mirror CHMP ``IRatio.pro`` used in golden/Brent q0 refinement."""
    if a < b:
        return 1.0e100 if a <= 0.0 else b / a
    if a > b:
        return 1.0e100 if b <= 0.0 else a / b
    return 1.0e100


def _ebtel_table_violation(record: _Q0EvaluationRecord, *, emthreshold: float) -> bool:
    message = str(record.message or "")
    if "EBTEL miss ratio" not in message or "exceeds" not in message:
        return False
    return True


def _idl_expansion_direction(
    q_grid: list[float],
    cache: dict[float, _Q0EvaluationRecord],
    *,
    emthreshold: float,
) -> int:
    """Return expansion sign: negative=left, positive=right, zero=bracketed (FindBestFitQ)."""
    records = [cache[float(q)] for q in q_grid]
    nq = len(records)
    if nq == 1:
        aw = 0.0
        record = records[0]
        if (
            record.total_observed_flux is not None
            and record.total_modeled_flux is not None
            and math.isfinite(float(record.total_observed_flux))
            and math.isfinite(float(record.total_modeled_flux))
        ):
            flux_delta = float(record.total_observed_flux) - float(record.total_modeled_flux)
            if flux_delta > 0.0:
                aw = 1.0
            elif flux_delta < 0.0:
                aw = -1.0
        return 1 if aw == 0.0 else int(math.copysign(1, aw))

    lmins = 0
    rmins = 0
    if any(not math.isfinite(record.objective_value) for record in records):
        return 0
    min_index = min(range(nq), key=lambda index: records[index].objective_value)
    if min_index == 0:
        lmins = 1
    if min_index == nq - 1:
        rmins = 1
    if _ebtel_table_violation(records[0], emthreshold=emthreshold):
        lmins = 0
    if _ebtel_table_violation(records[-1], emthreshold=emthreshold):
        rmins = 0
    if lmins == 0 and rmins == 0:
        return 0
    aw = rmins - lmins
    return 1 if aw == 0 else int(math.copysign(1, aw))


def _idl_propose_expansion_q0(
    q_grid: list[float],
    *,
    direction: int,
    q0_step: float,
    hard_q0_min: float | None,
    hard_q0_max: float | None,
) -> float | None:
    if direction < 0:
        anchor = min(q_grid)
        proposed = anchor / q0_step
        if hard_q0_min is not None:
            proposed = max(proposed, float(hard_q0_min))
        if proposed >= anchor * (1.0 - 1e-15):
            return None
        return proposed
    anchor = max(q_grid)
    proposed = anchor * q0_step
    if hard_q0_max is not None:
        proposed = min(proposed, float(hard_q0_max))
    if proposed <= anchor * (1.0 + 1e-15):
        return None
    return proposed


def _idl_interior_minimum_count(
    q_grid: list[float],
    cache: dict[float, _Q0EvaluationRecord],
) -> int:
    count = 0
    for index in range(1, len(q_grid) - 1):
        left = cache[float(q_grid[index - 1])]
        middle = cache[float(q_grid[index])]
        right = cache[float(q_grid[index + 1])]
        if not (left.is_valid and middle.is_valid and right.is_valid):
            continue
        if middle.objective_value < left.objective_value and middle.objective_value < right.objective_value:
            count += 1
    return count


def _idl_golden_brent_refine(
    metric_function: Callable[[float], MetricFunctionResult],
    *,
    q_grid: list[float],
    cache: dict[float, _Q0EvaluationRecord],
    evaluation_order: list[float],
    target_metric: MetricName,
    relative_acc: float,
    maxiter: int,
    progress_start_callback: ProgressStartCallback | None,
    progress_callback: ProgressCallback | None,
    q_bound_min: float | None = None,
    q_bound_max: float | None = None,
) -> tuple[list[float], int]:
    """Refine q0 on a sorted grid using CHMP golden/Brent steps (``FindBestFitQ.pro``)."""
    grid = list(q_grid)
    refine_steps = 0
    for _ in range(max(1, int(maxiter))):
        valid_indices = [index for index, q0 in enumerate(grid) if cache[float(q0)].is_valid]
        if len(valid_indices) < 3:
            break
        ib = min(
            valid_indices,
            key=lambda index: cache[float(grid[index])].objective_value,
        )
        if ib <= 0 or ib >= len(grid) - 1:
            break

        qa = float(grid[ib - 1])
        qb = float(grid[ib])
        qc = float(grid[ib + 1])
        mtra = float(cache[qa].objective_value)
        mtrb = float(cache[qb].objective_value)
        mtrc = float(cache[qc].objective_value)
        if (qc + qa) <= 0.0 or ((qc - qa) / (qc + qa)) < relative_acc:
            break

        if (qc - qb) > (qb - qa):
            qxg = qb + (qc - qb) * (1.0 - 1.0 / _IDL_GOLDEN_RATIO)
        else:
            qxg = qb - (qb - qa) * (1.0 - 1.0 / _IDL_GOLDEN_RATIO)

        denom = (qb - qa) * (mtrb - mtrc) - (qb - qc) * (mtrb - mtra)
        if math.isclose(denom, 0.0, rel_tol=0.0, abs_tol=1e-30):
            qxb = qxg
        else:
            qxb = qb - 0.5 * (
                (qb - qa) ** 2 * (mtrb - mtrc) - (qb - qc) ** 2 * (mtrb - mtra)
            ) / denom

        if qxb > qb:
            rb_hit = _i_ratio(qc - qxb, qxb - qb)
            rb_miss = _i_ratio(qxb - qb, qb - qa)
        else:
            rb_hit = _i_ratio(qb - qxb, qxb - qa)
            rb_miss = _i_ratio(qc - qb, qb - qxb)

        if qxb > qb:
            dnewg = (qc - qb + qxg - qa) / 2.0
            dnewb = (qc - qb + qxb - qa) / 2.0
        else:
            dnewg = (qb - qa + qc - qxg) / 2.0
            dnewb = (qb - qa + qc - qxb) / 2.0

        use_brent = (
            qa < qxb < qc
            and dnewb <= dnewg
            and rb_hit <= 10.0
            and rb_miss <= 10.0
        )
        qx = qxb if use_brent else qxg
        if q_bound_min is not None:
            qx = max(float(qx), float(q_bound_min))
        if q_bound_max is not None:
            qx = min(float(qx), float(q_bound_max))
        if math.isclose(qx, qb, rel_tol=0.0, abs_tol=1e-15):
            break

        _evaluate_q0(
            qx,
            metric_function=metric_function,
            target_metric=target_metric,
            cache=cache,
            evaluation_order=evaluation_order,
            progress_start_callback=progress_start_callback,
            progress_callback=progress_callback,
        )
        insert_at = ib if qx > qb else ib - 1
        grid = grid[: insert_at + 1] + [float(qx)] + grid[insert_at + 1 :]
        refine_steps += 1

    return grid, refine_steps


def _idl_sorted_q_grid(cache: dict[float, _Q0EvaluationRecord]) -> list[float]:
    return [float(q0) for q0 in sorted(cache.keys())]


def _idl_warm_best_edge_q0(
    q_grid: list[float],
    cache: dict[float, _Q0EvaluationRecord],
) -> float:
    valid_records = [cache[float(q0)] for q0 in q_grid if cache[float(q0)].is_valid]
    if not valid_records:
        return float(q_grid[0])
    left_q0 = float(q_grid[0])
    right_q0 = float(q_grid[-1])
    left = cache[left_q0]
    right = cache[right_q0]
    if left.is_valid and right.is_valid:
        return left_q0 if float(left.objective_value) <= float(right.objective_value) else right_q0
    if left.is_valid:
        return left_q0
    if right.is_valid:
        return right_q0
    return float(min(valid_records, key=lambda item: item.objective_value).q0)


def _idl_build_q0_result(
    *,
    q_grid: list[float],
    cache: dict[float, _Q0EvaluationRecord],
    evaluation_order: list[float],
    target_metric: MetricName,
    bracket_steps: int,
    message_parts: list[str],
    bracket_found: bool,
    bracket: tuple[float, float, float] | None,
) -> Q0OptimizationResult:
    valid_records = [record for record in cache.values() if record.is_valid]
    if not valid_records:
        best_record = cache[float(q_grid[0])]
        success = False
    else:
        best_record = min(valid_records, key=lambda item: item.objective_value)
        success = bool(bracket_found)
    trial_q0 = tuple(evaluation_order)
    trial_chi2, trial_rho2, trial_eta2 = _trial_metric_histories(cache, evaluation_order)
    shift_x, shift_y, shift_valid = _trial_shift_histories(cache, evaluation_order)
    return Q0OptimizationResult(
        q0=float(best_record.q0),
        objective_value=float(best_record.objective_value),
        metrics=best_record.metrics,
        target_metric=target_metric,
        success=success,
        nfev=len(cache),
        nit=bracket_steps,
        message="; ".join(message_parts),
        used_adaptive_bracketing=True,
        bracket_found=bracket_found,
        bracket=bracket,
        boundary_constrained=not bracket_found,
        trial_q0=trial_q0,
        trial_objective_values=tuple(cache[q0].objective_value for q0 in trial_q0),
        trial_chi2_values=trial_chi2,
        trial_rho2_values=trial_rho2,
        trial_eta2_values=trial_eta2,
        trial_shift_x_arcsec=shift_x,
        trial_shift_y_arcsec=shift_y,
        trial_find_shift_valid=shift_valid,
        trial_mask_stages=_trial_mask_stage_histories(cache, evaluation_order),
    )


def _idl_chmp_expand_q0_grid(
    metric_function: Callable[[float], MetricFunctionResult],
    *,
    q_grid: list[float],
    q0_step: float,
    max_bracket_steps: int,
    hard_q0_min: float | None,
    hard_q0_max: float | None,
    target_metric: MetricName,
    emthreshold: float,
    cache: dict[float, _Q0EvaluationRecord],
    evaluation_order: list[float],
    progress_start_callback: ProgressStartCallback | None,
    progress_callback: ProgressCallback | None,
) -> tuple[list[float], int, list[str], bool]:
    bracket_steps = 0
    message_parts: list[str] = []
    boundary_constrained = False
    done = False
    while not done and bracket_steps < int(max_bracket_steps):
        direction = _idl_expansion_direction(q_grid, cache, emthreshold=emthreshold)
        if direction == 0:
            done = True
            break
        proposed = _idl_propose_expansion_q0(
            q_grid,
            direction=direction,
            q0_step=q0_step,
            hard_q0_min=hard_q0_min,
            hard_q0_max=hard_q0_max,
        )
        if proposed is None:
            side = "upper" if direction > 0 else "lower"
            message_parts.append(f"CHMP q0 search hit the {side} expansion limit")
            boundary_constrained = True
            break
        if direction < 0:
            q_grid = [float(proposed)] + q_grid
        else:
            q_grid = q_grid + [float(proposed)]
        _evaluate_q0(
            q_grid[0] if direction < 0 else q_grid[-1],
            metric_function=metric_function,
            target_metric=target_metric,
            cache=cache,
            evaluation_order=evaluation_order,
            progress_start_callback=progress_start_callback,
            progress_callback=progress_callback,
        )
        bracket_steps += 1
    return q_grid, bracket_steps, message_parts, boundary_constrained


def _idl_chmp_finalize_bracket_and_refine(
    metric_function: Callable[[float], MetricFunctionResult],
    *,
    q_grid: list[float],
    cache: dict[float, _Q0EvaluationRecord],
    evaluation_order: list[float],
    target_metric: MetricName,
    xatol: float,
    maxiter: int,
    bracket_steps: int,
    message_parts: list[str],
    progress_start_callback: ProgressStartCallback | None,
    progress_callback: ProgressCallback | None,
    q_bound_min: float | None = None,
    q_bound_max: float | None = None,
    allow_refinement: bool = True,
) -> Q0OptimizationResult:
    interior_minima = _idl_interior_minimum_count(q_grid, cache)
    bracket: tuple[float, float, float] | None = None
    bracket_found = interior_minima == 1
    if bracket_found:
        bracket = _find_bracket(cache)
        if bracket is not None:
            message_parts.append("CHMP q0 bracketing found a valid interior minimum")
        else:
            bracket_found = False
            message_parts.append("CHMP q0 bracketing found no interior minimum")
    elif interior_minima == 0:
        message_parts.append("CHMP q0 bracketing found no interior minimum")
    else:
        message_parts.append("CHMP q0 bracketing found more than one interior minimum")

    relative_acc = max(float(xatol), _IDL_RELATIVE_ACC_DEFAULT)
    if bracket_found and bracket is not None and allow_refinement:
        q_grid, refine_steps = _idl_golden_brent_refine(
            metric_function,
            q_grid=q_grid,
            cache=cache,
            evaluation_order=evaluation_order,
            target_metric=target_metric,
            relative_acc=relative_acc,
            maxiter=maxiter,
            progress_start_callback=progress_start_callback,
            progress_callback=progress_callback,
            q_bound_min=q_bound_min,
            q_bound_max=q_bound_max,
        )
        message_parts.append("CHMP golden/Brent q0 refinement")
        bracket_steps += refine_steps

    return _idl_build_q0_result(
        q_grid=q_grid,
        cache=cache,
        evaluation_order=evaluation_order,
        target_metric=target_metric,
        bracket_steps=bracket_steps,
        message_parts=message_parts,
        bracket_found=bracket_found,
        bracket=bracket,
    )


def _idl_chmp_find_best_q0_warm_start(
    metric_function: Callable[[float], MetricFunctionResult],
    *,
    q0_step: float,
    max_bracket_steps: int,
    hard_q0_min: float | None,
    hard_q0_max: float | None,
    target_metric: MetricName,
    xatol: float,
    maxiter: int,
    emthreshold: float,
    cache: dict[float, _Q0EvaluationRecord],
    evaluation_order: list[float],
    progress_start_callback: ProgressStartCallback | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Q0OptimizationResult:
    """Continue CHMP q0 search from a rescored warm trial curve (IDL policy)."""
    q_grid = _idl_sorted_q_grid(cache)
    if not q_grid:
        raise RuntimeError("warm-start q0 search requires seeded evaluations")

    warm_lo = float(q_grid[0])
    warm_hi = float(q_grid[-1])
    message_parts = ["CHMP warm-start q0 search"]
    interior_minima = _idl_interior_minimum_count(q_grid, cache)

    if interior_minima == 1:
        message_parts.append(
            "rescored curve has one interior minimum; refine within stored q0 range only"
        )
        return _idl_chmp_finalize_bracket_and_refine(
            metric_function,
            q_grid=q_grid,
            cache=cache,
            evaluation_order=evaluation_order,
            target_metric=target_metric,
            xatol=xatol,
            maxiter=maxiter,
            bracket_steps=0,
            message_parts=message_parts,
            progress_start_callback=progress_start_callback,
            progress_callback=progress_callback,
            q_bound_min=warm_lo,
            q_bound_max=warm_hi,
            allow_refinement=True,
        )

    if interior_minima > 1:
        message_parts.append(
            "rescored curve has more than one interior minimum; IDL skips refinement"
        )
        return _idl_chmp_finalize_bracket_and_refine(
            metric_function,
            q_grid=q_grid,
            cache=cache,
            evaluation_order=evaluation_order,
            target_metric=target_metric,
            xatol=xatol,
            maxiter=maxiter,
            bracket_steps=0,
            message_parts=message_parts,
            progress_start_callback=progress_start_callback,
            progress_callback=progress_callback,
            allow_refinement=False,
        )

    message_parts.append(
        "rescored curve has no interior minimum; expand from best metric edge"
    )
    q_grid, bracket_steps, expand_messages, _boundary = _idl_chmp_expand_q0_grid(
        metric_function,
        q_grid=q_grid,
        q0_step=q0_step,
        max_bracket_steps=max_bracket_steps,
        hard_q0_min=hard_q0_min,
        hard_q0_max=hard_q0_max,
        target_metric=target_metric,
        emthreshold=emthreshold,
        cache=cache,
        evaluation_order=evaluation_order,
        progress_start_callback=progress_start_callback,
        progress_callback=progress_callback,
    )
    message_parts.extend(expand_messages)
    return _idl_chmp_finalize_bracket_and_refine(
        metric_function,
        q_grid=q_grid,
        cache=cache,
        evaluation_order=evaluation_order,
        target_metric=target_metric,
        xatol=xatol,
        maxiter=maxiter,
        bracket_steps=bracket_steps,
        message_parts=message_parts,
        progress_start_callback=progress_start_callback,
        progress_callback=progress_callback,
    )


def _idl_chmp_find_best_q0(
    metric_function: Callable[[float], MetricFunctionResult],
    *,
    q0_start: float,
    q0_step: float,
    max_bracket_steps: int,
    hard_q0_min: float | None,
    hard_q0_max: float | None,
    target_metric: MetricName,
    xatol: float,
    maxiter: int,
    emthreshold: float,
    cache: dict[float, _Q0EvaluationRecord],
    evaluation_order: list[float],
    progress_start_callback: ProgressStartCallback | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Q0OptimizationResult:
    """CHMP IDL ``FindBestFitQ``-style multiplicative bracketing plus golden/Brent refinement."""
    q_grid = [float(q0_start)]
    _evaluate_q0(
        q_grid[0],
        metric_function=metric_function,
        target_metric=target_metric,
        cache=cache,
        evaluation_order=evaluation_order,
        progress_start_callback=progress_start_callback,
        progress_callback=progress_callback,
    )

    bracket_steps = 0
    message_parts: list[str] = []
    done = False
    while not done and bracket_steps < int(max_bracket_steps):
        direction = _idl_expansion_direction(q_grid, cache, emthreshold=emthreshold)
        if direction == 0:
            done = True
            break
        proposed = _idl_propose_expansion_q0(
            q_grid,
            direction=direction,
            q0_step=q0_step,
            hard_q0_min=hard_q0_min,
            hard_q0_max=hard_q0_max,
        )
        if proposed is None:
            side = "upper" if direction > 0 else "lower"
            boundary_q0 = _boundary_best_q0(cache, side=side)
            message_parts.append(f"CHMP q0 search hit the {side} expansion limit")
            if boundary_q0 is not None:
                record = cache[float(boundary_q0)]
                trial_q0 = tuple(evaluation_order)
                chi2, rho2, eta2 = _trial_metric_histories(cache, evaluation_order)
                shift_x, shift_y, shift_valid = _trial_shift_histories(cache, evaluation_order)
                return Q0OptimizationResult(
                    q0=float(record.q0),
                    objective_value=float(record.objective_value),
                    metrics=record.metrics,
                    target_metric=target_metric,
                    success=False,
                    nfev=len(cache),
                    nit=bracket_steps,
                    message="; ".join(message_parts),
                    used_adaptive_bracketing=True,
                    bracket_found=False,
                    bracket=None,
                    boundary_constrained=True,
                    trial_q0=trial_q0,
                    trial_objective_values=tuple(cache[q0].objective_value for q0 in trial_q0),
                    trial_chi2_values=chi2,
                    trial_rho2_values=rho2,
                    trial_eta2_values=eta2,
                    trial_shift_x_arcsec=shift_x,
                    trial_shift_y_arcsec=shift_y,
                    trial_find_shift_valid=shift_valid,
                    trial_mask_stages=_trial_mask_stage_histories(cache, evaluation_order),
                )
            break
        if direction < 0:
            q_grid = [float(proposed)] + q_grid
        else:
            q_grid = q_grid + [float(proposed)]
        _evaluate_q0(
            q_grid[0] if direction < 0 else q_grid[-1],
            metric_function=metric_function,
            target_metric=target_metric,
            cache=cache,
            evaluation_order=evaluation_order,
            progress_start_callback=progress_start_callback,
            progress_callback=progress_callback,
        )
        bracket_steps += 1

    interior_minima = _idl_interior_minimum_count(q_grid, cache)
    bracket: tuple[float, float, float] | None = None
    bracket_found = interior_minima == 1
    if bracket_found:
        for index in range(1, len(q_grid) - 1):
            left = cache[float(q_grid[index - 1])]
            middle = cache[float(q_grid[index])]
            right = cache[float(q_grid[index + 1])]
            if not (left.is_valid and middle.is_valid and right.is_valid):
                continue
            if middle.objective_value <= left.objective_value and middle.objective_value <= right.objective_value:
                bracket = (float(left.q0), float(middle.q0), float(right.q0))
                break
        message_parts.append("CHMP q0 bracketing found a valid interior minimum")
    elif interior_minima == 0:
        message_parts.append("CHMP q0 bracketing found no interior minimum")
    else:
        message_parts.append("CHMP q0 bracketing found more than one interior minimum")

    relative_acc = max(float(xatol), _IDL_RELATIVE_ACC_DEFAULT)
    if bracket_found and bracket is not None:
        q_grid, refine_steps = _idl_golden_brent_refine(
            metric_function,
            q_grid=q_grid,
            cache=cache,
            evaluation_order=evaluation_order,
            target_metric=target_metric,
            relative_acc=relative_acc,
            maxiter=maxiter,
            progress_start_callback=progress_start_callback,
            progress_callback=progress_callback,
        )
        message_parts.append("CHMP golden/Brent q0 refinement")
        bracket_steps += refine_steps

    valid_records = [record for record in cache.values() if record.is_valid]
    if not valid_records:
        best_record = cache[float(q_grid[0])]
        success = False
    else:
        best_record = min(valid_records, key=lambda item: item.objective_value)
        success = bracket_found

    trial_q0 = tuple(evaluation_order)
    trial_chi2, trial_rho2, trial_eta2 = _trial_metric_histories(cache, evaluation_order)
    shift_x, shift_y, shift_valid = _trial_shift_histories(cache, evaluation_order)
    return Q0OptimizationResult(
        q0=float(best_record.q0),
        objective_value=float(best_record.objective_value),
        metrics=best_record.metrics,
        target_metric=target_metric,
        success=success,
        nfev=len(cache),
        nit=bracket_steps,
        message="; ".join(message_parts),
        used_adaptive_bracketing=True,
        bracket_found=bracket_found,
        bracket=bracket,
        boundary_constrained=not bracket_found,
        trial_q0=trial_q0,
        trial_objective_values=tuple(cache[q0].objective_value for q0 in trial_q0),
        trial_chi2_values=trial_chi2,
        trial_rho2_values=trial_rho2,
        trial_eta2_values=trial_eta2,
        trial_shift_x_arcsec=shift_x,
        trial_shift_y_arcsec=shift_y,
        trial_find_shift_valid=shift_valid,
        trial_mask_stages=_trial_mask_stage_histories(cache, evaluation_order),
    )


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

    if (
        start_record.is_valid
        and right_record is not None
        and right_record.is_valid
        and right_record.objective_value > start_record.objective_value
        and (left_record is None or not left_record.is_valid)
    ):
        return _BracketSearchResult(
            bracket=None,
            steps_taken=0,
            message=(
                "adaptive bracketing: q0_start is the best valid seed and the upper "
                "initialization bound is worse; skipping upward expansion"
            ),
            fallback_to_bounded_refinement=True,
            boundary_q0=float(start_record.q0),
        )

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
    initial_evaluations: InitialQ0Evaluations | None = None,
    emthreshold: float = 0.1,
) -> Q0OptimizationResult:
    """Find best Q0 with optional CHMP IDL-style adaptive search.

    When ``adaptive_bracketing`` is enabled, bracket expansion and golden/Brent
    refinement follow ``FindBestFitQ.pro`` (start at ``q0_start``, grow by
    ``q0_step``, not the legacy triplet-at-bounds algorithm).

    ``q0_min`` and ``q0_max`` bound the non-adaptive SciPy path and warm-start
    validation; adaptive expansion may continue beyond them unless
    ``hard_q0_min`` / ``hard_q0_max`` are set.
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
    _seed_initial_evaluations(
        initial_evaluations,
        target_metric=target_metric,
        cache=cache,
        evaluation_order=evaluation_order,
        hard_q0_min=hard_q0_min,
        hard_q0_max=hard_q0_max,
    )

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
        if cache:
            return _idl_chmp_find_best_q0_warm_start(
                metric_function,
                q0_step=q0_step,
                max_bracket_steps=max_bracket_steps,
                hard_q0_min=hard_q0_min,
                hard_q0_max=hard_q0_max,
                target_metric=target_metric,
                xatol=xatol,
                maxiter=maxiter,
                emthreshold=float(emthreshold),
                cache=cache,
                evaluation_order=evaluation_order,
                progress_start_callback=progress_start_callback,
                progress_callback=progress_callback,
            )
        return _idl_chmp_find_best_q0(
            metric_function,
            q0_start=q0_start,
            q0_step=q0_step,
            max_bracket_steps=max_bracket_steps,
            hard_q0_min=hard_q0_min,
            hard_q0_max=hard_q0_max,
            target_metric=target_metric,
            xatol=xatol,
            maxiter=maxiter,
            emthreshold=float(emthreshold),
            cache=cache,
            evaluation_order=evaluation_order,
            progress_start_callback=progress_start_callback,
            progress_callback=progress_callback,
        )

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

    result_message = str(result.message)
    if message_prefix:
        result_message = f"{message_prefix}; {result_message}"

    sampled_override = False
    valid_sampled_records = [record for record in cache.values() if record.is_valid]
    if valid_sampled_records:
        sampled_best = min(valid_sampled_records, key=lambda item: item.objective_value)
        objective_tol = max(1e-12, 1e-9 * max(1.0, abs(float(best_record.objective_value))))
        if float(sampled_best.objective_value) < float(best_record.objective_value) - objective_tol:
            best_q0 = float(sampled_best.q0)
            best_record = sampled_best
            sampled_override = True
            sampled_q0_values = [float(record.q0) for record in valid_sampled_records]
            lower_sampled = min(sampled_q0_values)
            upper_sampled = max(sampled_q0_values)
            if math.isclose(best_q0, lower_sampled, rel_tol=0.0, abs_tol=1e-15):
                sampled_location = "lower sampled edge"
            elif math.isclose(best_q0, upper_sampled, rel_tol=0.0, abs_tol=1e-15):
                sampled_location = "upper sampled edge"
            else:
                sampled_location = "sampled point outside the refined local bracket"
            result_message += (
                f" WARNING: A {sampled_location} had a lower {target_metric} "
                "than the refined local minimum; reporting the sampled best instead."
            )

    trial_q0 = tuple(evaluation_order)
    trial_objective_values = tuple(cache[q0].objective_value for q0 in evaluation_order)
    trial_chi2_values, trial_rho2_values, trial_eta2_values = _trial_metric_histories(cache, evaluation_order)
    trial_shift_x, trial_shift_y, trial_shift_valid = _trial_shift_histories(cache, evaluation_order)
    trial_mask_stages = _trial_mask_stage_histories(cache, evaluation_order)

    boundary_tol = max(float(effective_xatol), 1e-12)
    boundary_failure = False
    if (
        math.isclose(best_q0, refinement_bounds[0], rel_tol=0.0, abs_tol=boundary_tol)
        or math.isclose(best_q0, refinement_bounds[1], rel_tol=0.0, abs_tol=boundary_tol)
        or sampled_override
    ):
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
        boundary_constrained=boundary_failure,
        trial_q0=trial_q0,
        trial_objective_values=trial_objective_values,
        trial_chi2_values=trial_chi2_values,
        trial_rho2_values=trial_rho2_values,
        trial_eta2_values=trial_eta2_values,
        trial_shift_x_arcsec=trial_shift_x,
        trial_shift_y_arcsec=trial_shift_y,
        trial_find_shift_valid=trial_shift_valid,
        trial_mask_stages=trial_mask_stages,
    )
