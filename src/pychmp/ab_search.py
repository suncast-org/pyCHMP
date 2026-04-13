"""Higher-level single-frequency `(a, b, q0)` search workflows."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterator, MutableMapping, Protocol

import numpy as np

from .ab_scan_execution import ABExecutionSettings, ABRequestedExecutionPolicy, iter_execute_tasks, resolve_execution_plan
from .ab_scan_tasks import ABPointTask, ABSliceTaskDescriptor, compile_rectangular_point_tasks
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
    trial_chi2_values: tuple[float, ...] = ()
    trial_rho2_values: tuple[float, ...] = ()
    trial_eta2_values: tuple[float, ...] = ()
    elapsed_seconds: float = float("nan")


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


@dataclass(frozen=True)
class ABLocalSearchResult:
    """Summary of adaptive local `(a, b)` search on one slice."""

    a_values: tuple[float, ...]
    b_values: tuple[float, ...]
    best_a: float
    best_b: float
    target_metric: MetricName
    threshold_metric: float
    points: tuple[ABPointResult, ...]
    best_q0: np.ndarray
    objective_values: np.ndarray
    chi2: np.ndarray
    rho2: np.ndarray
    eta2: np.ndarray
    success: np.ndarray
    n_phase1_iters: int
    n_phase2_iters: int
    best_is_interior: bool
    best_boundary_axes: tuple[str, ...]
    minimum_certified: bool
    termination_reason: str
    frontier_open_axes: tuple[str, ...]
    evaluated_point_count: int

    def point_map(self) -> dict[tuple[float, float], ABPointResult]:
        return {(point.a, point.b): point for point in self.points}


ABPointCache = MutableMapping[tuple[float, float], ABPointResult]


@dataclass(frozen=True)
class ABPointEvaluationRequest:
    """Structured evaluation request for one `(a, b)` point."""

    task: ABPointTask
    hard_q0_min: float | None
    hard_q0_max: float | None
    threshold: float
    mask_type: str
    target_metric: MetricName
    xatol: float
    maxiter: int
    adaptive_bracketing: bool
    q0_start: float | None
    q0_step: float
    max_bracket_steps: int


@dataclass(frozen=True)
class ABSearchWorkerPayload:
    """Bootstrap payload for persistent AB-search workers."""

    renderer_factory: ABRendererFactory
    observed: np.ndarray
    sigma: np.ndarray | None


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
    mask_type: str = "union",
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

    started = time.perf_counter()
    result: Q0OptimizationResult = fit_q0_to_observation(
        renderer_factory(float(a), float(b)),
        observed,
        sigma,
        q0_min=q0_min,
        q0_max=q0_max,
        hard_q0_min=hard_q0_min,
        hard_q0_max=hard_q0_max,
        threshold=threshold,
        mask_type=mask_type,
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
        trial_chi2_values=tuple(float(v) for v in result.trial_chi2_values),
        trial_rho2_values=tuple(float(v) for v in result.trial_rho2_values),
        trial_eta2_values=tuple(float(v) for v in result.trial_eta2_values),
        elapsed_seconds=float(time.perf_counter() - started),
    )


def _bootstrap_ab_search_worker(payload: ABSearchWorkerPayload) -> ABSearchWorkerPayload:
    return payload


def _evaluate_ab_search_request(
    request: ABPointEvaluationRequest,
    worker_payload: ABSearchWorkerPayload,
) -> ABPointResult:
    a = float(request.task.a)
    b = float(request.task.b)
    q0_start = request.q0_start
    point_started = time.perf_counter()
    q0_start_text = "auto" if q0_start is None else f"{float(q0_start):.6g}"
    print(
        "    Starting point: "
        f"a={a:.3f} b={b:.3f} "
        f"q0_range=({float(request.task.q0_min):.6g}, {float(request.task.q0_max):.6g}) "
        f"q0_start={q0_start_text}",
        flush=True,
    )
    try:
        result = evaluate_ab_point(
            worker_payload.renderer_factory,
            worker_payload.observed,
            worker_payload.sigma,
            a=a,
            b=b,
            q0_min=float(request.task.q0_min),
            q0_max=float(request.task.q0_max),
            hard_q0_min=request.hard_q0_min,
            hard_q0_max=request.hard_q0_max,
            threshold=float(request.threshold),
            mask_type=str(request.mask_type),
            target_metric=request.target_metric,
            xatol=float(request.xatol),
            maxiter=int(request.maxiter),
            adaptive_bracketing=bool(request.adaptive_bracketing),
            q0_start=q0_start,
            q0_step=float(request.q0_step),
            max_bracket_steps=int(request.max_bracket_steps),
            progress_callback=None,
        )
    except BaseException:
        print(
            "    Point interrupted: "
            f"a={a:.3f} b={b:.3f} "
            f"elapsed={time.perf_counter() - point_started:.3f}s",
            flush=True,
        )
        raise
    print(
        "    Finished point: "
        f"a={a:.3f} b={b:.3f} "
        f"q0={float(result.q0):.6g} "
        f"{str(result.target_metric)}={float(result.objective_value):.6e} "
        f"elapsed={float(result.elapsed_seconds):.3f}s",
        flush=True,
    )
    return result


def _validate_q0_interval(*, q0_min: float, q0_max: float) -> None:
    if float(q0_min) >= float(q0_max):
        raise ValueError("q0_min must be smaller than q0_max")


def _resolve_q0_start_array(
    *,
    q0_start_grid: float | np.ndarray | None,
    use_idl_q0_start_heuristic: bool,
    a_arr: np.ndarray,
    b_arr: np.ndarray,
) -> np.ndarray | None:
    q0_start_arr = _coerce_q0_start_grid(q0_start_grid, n_a=int(a_arr.size), n_b=int(b_arr.size))
    if q0_start_arr is None and use_idl_q0_start_heuristic:
        q0_start_arr = np.empty((a_arr.size, b_arr.size), dtype=float)
        for i, a in enumerate(a_arr):
            for j, b in enumerate(b_arr):
                q0_start_arr[i, j] = idl_q0_start_heuristic(float(a), float(b))
    return q0_start_arr


def _resolve_execution_plan_for_requests(
    *,
    request_count: int,
    execution_policy: ABRequestedExecutionPolicy,
    max_workers: int | None,
    progress_callback: ProgressCallback | None,
):
    execution_plan = resolve_execution_plan(
        task_count=int(request_count),
        requested_policy=execution_policy,
        max_workers=max_workers,
    )
    if execution_plan.policy != "serial" and progress_callback is not None:
        raise ValueError("progress_callback is only supported in serial execution mode")
    return execution_plan


def _execute_ab_requests(
    renderer_factory: ABRendererFactory,
    observed: np.ndarray,
    sigma: np.ndarray | None,
    *,
    pending_requests: list[ABPointEvaluationRequest],
    execution_policy: ABRequestedExecutionPolicy,
    max_workers: int | None,
    worker_chunksize: int,
    progress_callback: ProgressCallback | None,
) -> Iterator[ABPointResult]:
    if not pending_requests:
        return iter(())

    execution_plan = _resolve_execution_plan_for_requests(
        request_count=len(pending_requests),
        execution_policy=execution_policy,
        max_workers=max_workers,
        progress_callback=progress_callback,
    )
    execution_settings = ABExecutionSettings(
        policy=execution_plan.policy,
        max_workers=execution_plan.max_workers,
        chunksize=int(worker_chunksize),
        yield_completion_order=True,
    )
    worker_payload = ABSearchWorkerPayload(
        renderer_factory=renderer_factory,
        observed=np.asarray(observed, dtype=float),
        sigma=None if sigma is None else np.asarray(sigma, dtype=float),
    )
    return iter_execute_tasks(
        pending_requests,
        bootstrap_worker=_bootstrap_ab_search_worker,
        bootstrap_payload=worker_payload,
        evaluate_task=_evaluate_ab_search_request,
        settings=execution_settings,
    )


def _cache_set_pending_points(
    cache_map: ABPointCache | None,
    pending_requests: list[ABPointEvaluationRequest],
) -> None:
    if cache_map is None:
        return
    setter = getattr(cache_map, "set_pending_points", None)
    if setter is None:
        return
    try:
        setter(
            [
                (float(request.task.a), float(request.task.b))
                for request in pending_requests
            ]
        )
    except Exception:
        pass


def _cache_clear_pending_points(cache_map: ABPointCache | None) -> None:
    if cache_map is None:
        return
    clearer = getattr(cache_map, "clear_pending_points", None)
    if clearer is None:
        return
    try:
        clearer()
    except Exception:
        pass


def _persist_adaptive_point(
    *,
    cache_map: ABPointCache,
    point_results: dict[tuple[float, float], ABPointResult],
    evaluated_points: list[ABPointResult],
    point: ABPointResult,
) -> None:
    key = (float(point.a), float(point.b))
    cache_map[key] = point
    point_results[key] = point
    evaluated_points.append(point)


def _materialize_scan_arrays(
    *,
    a_arr: np.ndarray,
    b_arr: np.ndarray,
    point_tasks: list[ABPointTask],
    point_results: dict[tuple[float, float], ABPointResult],
    allow_missing: bool = False,
) -> tuple[tuple[ABPointResult, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points: list[ABPointResult] = []
    best_q0 = np.full((a_arr.size, b_arr.size), np.nan, dtype=float)
    objective_values = np.full((a_arr.size, b_arr.size), np.nan, dtype=float)
    chi2 = np.full((a_arr.size, b_arr.size), np.nan, dtype=float)
    rho2 = np.full((a_arr.size, b_arr.size), np.nan, dtype=float)
    eta2 = np.full((a_arr.size, b_arr.size), np.nan, dtype=float)
    success = np.zeros((a_arr.size, b_arr.size), dtype=bool)

    for point_task in point_tasks:
        key = (float(point_task.a), float(point_task.b))
        point = point_results.get(key)
        if point is None:
            if bool(allow_missing):
                continue
            raise KeyError(key)
        points.append(point)
        best_q0[point_task.a_index, point_task.b_index] = point.q0
        objective_values[point_task.a_index, point_task.b_index] = point.objective_value
        chi2[point_task.a_index, point_task.b_index] = point.metrics.chi2
        rho2[point_task.a_index, point_task.b_index] = point.metrics.rho2
        eta2[point_task.a_index, point_task.b_index] = point.metrics.eta2
        success[point_task.a_index, point_task.b_index] = point.success

    return (tuple(points), best_q0, objective_values, chi2, rho2, eta2, success)


def _expand_axis_around_index(
    axis_values: np.ndarray,
    *,
    current_index: int,
    step: float,
    bounds: tuple[float, float],
) -> tuple[np.ndarray, int, bool]:
    values = [float(v) for v in np.asarray(axis_values, dtype=float)]
    index = int(current_index)
    expanded = False
    lower_bound, upper_bound = (float(bounds[0]), float(bounds[1]))
    tolerance = max(1e-12, abs(float(step)) * 1e-9)

    if index == 0:
        candidate = float(values[0] - float(step))
        if candidate >= lower_bound - tolerance:
            values.insert(0, candidate)
            index += 1
            expanded = True

    if index == len(values) - 1:
        candidate = float(values[-1] + float(step))
        if candidate <= upper_bound + tolerance:
            values.append(candidate)
            expanded = True

    return np.asarray(values, dtype=float), int(index), bool(expanded)


def _make_adaptive_task(
    *,
    a_values: np.ndarray,
    b_values: np.ndarray,
    a_index: int,
    b_index: int,
    q0_min: float,
    q0_max: float,
    target_metric: MetricName,
) -> ABPointTask:
    return ABPointTask(
        slice_key="single_slice",
        slice_domain="generic",
        slice_label="single-slice",
        slice_display_label="single-slice",
        a=float(a_values[a_index]),
        b=float(b_values[b_index]),
        a_index=int(a_index),
        b_index=int(b_index),
        q0_min=float(q0_min),
        q0_max=float(q0_max),
        target_metric=str(target_metric),
        source_kind="adaptive",
    )


def _normalize_q0_seed(
    q0_seed: float | None,
    *,
    q0_min: float,
    q0_max: float,
) -> float | None:
    if q0_seed is None:
        return None
    try:
        seed = float(q0_seed)
    except Exception:
        return None
    if not np.isfinite(seed):
        return None
    lower = float(q0_min)
    upper = float(q0_max)
    tolerance = max(1e-12, max(abs(lower), abs(upper)) * 1e-12)
    if seed < lower - tolerance or seed > upper + tolerance:
        return None
    return min(max(seed, lower), upper)


def _append_adaptive_request_if_needed(
    pending_requests: list[ABPointEvaluationRequest],
    pending_keys: set[tuple[float, float]],
    *,
    a_values: np.ndarray,
    b_values: np.ndarray,
    a_index: int,
    b_index: int,
    q0_min: float,
    q0_max: float,
    hard_q0_min: float | None,
    hard_q0_max: float | None,
    threshold: float,
    mask_type: str,
    target_metric: MetricName,
    xatol: float,
    maxiter: int,
    adaptive_bracketing: bool,
    q0_seed: float | None,
    q0_step: float,
    max_bracket_steps: int,
    point_results: dict[tuple[float, float], ABPointResult],
    cache_map: ABPointCache,
) -> None:
    key = (float(a_values[a_index]), float(b_values[b_index]))
    cached_point = cache_map.get(key)
    if cached_point is not None:
        point_results[key] = cached_point
        return
    if key in point_results or key in pending_keys:
        return

    normalized_q0_seed = _normalize_q0_seed(
        q0_seed,
        q0_min=float(q0_min),
        q0_max=float(q0_max),
    )
    pending_keys.add(key)
    pending_requests.append(
        ABPointEvaluationRequest(
            task=_make_adaptive_task(
                a_values=a_values,
                b_values=b_values,
                a_index=int(a_index),
                b_index=int(b_index),
                q0_min=q0_min,
                q0_max=q0_max,
                target_metric=target_metric,
            ),
            hard_q0_min=hard_q0_min,
            hard_q0_max=hard_q0_max,
            threshold=float(threshold),
            mask_type=str(mask_type),
            target_metric=target_metric,
            xatol=float(xatol),
            maxiter=int(maxiter),
            adaptive_bracketing=bool(adaptive_bracketing),
            q0_start=normalized_q0_seed,
            q0_step=float(q0_step),
            max_bracket_steps=int(max_bracket_steps),
        )
    )


def _evaluate_adaptive_index_batch(
    renderer_factory: ABRendererFactory,
    observed: np.ndarray,
    sigma: np.ndarray | None,
    *,
    a_values: np.ndarray,
    b_values: np.ndarray,
    candidate_points: list[tuple[int, int, float | None]],
    q0_min: float,
    q0_max: float,
    hard_q0_min: float | None,
    hard_q0_max: float | None,
    threshold: float,
    mask_type: str,
    target_metric: MetricName,
    xatol: float,
    maxiter: int,
    adaptive_bracketing: bool,
    q0_step: float,
    max_bracket_steps: int,
    point_results: dict[tuple[float, float], ABPointResult],
    cache_map: ABPointCache,
    execution_policy: ABRequestedExecutionPolicy,
    max_workers: int | None,
    worker_chunksize: int,
    progress_callback: ProgressCallback | None,
) -> list[ABPointResult]:
    pending_requests: list[ABPointEvaluationRequest] = []
    pending_keys: set[tuple[float, float]] = set()

    for a_index, b_index, q0_seed in candidate_points:
        _append_adaptive_request_if_needed(
            pending_requests,
            pending_keys,
            a_values=a_values,
            b_values=b_values,
            a_index=int(a_index),
            b_index=int(b_index),
            q0_min=q0_min,
            q0_max=q0_max,
            hard_q0_min=hard_q0_min,
            hard_q0_max=hard_q0_max,
            threshold=float(threshold),
            mask_type=str(mask_type),
            target_metric=target_metric,
            xatol=float(xatol),
            maxiter=int(maxiter),
            adaptive_bracketing=bool(adaptive_bracketing),
            q0_seed=q0_seed,
            q0_step=float(q0_step),
            max_bracket_steps=int(max_bracket_steps),
            point_results=point_results,
            cache_map=cache_map,
        )

    execution_plan = _resolve_execution_plan_for_requests(
        request_count=len(pending_requests),
        execution_policy=execution_policy,
        max_workers=max_workers,
        progress_callback=progress_callback,
    )

    if pending_requests and execution_plan.policy != "serial":
        _cache_set_pending_points(cache_map, pending_requests)
        point_tokens = [
            f"(a={float(request.task.a):.3f}, b={float(request.task.b):.3f})"
            for request in pending_requests
        ]
        preview = ", ".join(point_tokens[:6])
        if len(point_tokens) > 6:
            preview += f", ... (+{len(point_tokens) - 6} more)"
        print(
            "  Adaptive batch dispatch: "
            f"{len(pending_requests)} point(s) via {execution_policy} "
            f"{preview}"
        )

    evaluated_points: list[ABPointResult] = []
    try:
        if execution_plan.policy == "serial":
            for request in pending_requests:
                _cache_set_pending_points(cache_map, [request])
                for point in _execute_ab_requests(
                    renderer_factory,
                    observed,
                    sigma,
                    pending_requests=[request],
                    execution_policy="serial",
                    max_workers=1,
                    worker_chunksize=worker_chunksize,
                    progress_callback=progress_callback,
                ):
                    _persist_adaptive_point(
                        cache_map=cache_map,
                        point_results=point_results,
                        evaluated_points=evaluated_points,
                        point=point,
                    )
        else:
            for point in _execute_ab_requests(
                renderer_factory,
                observed,
                sigma,
                pending_requests=pending_requests,
                execution_policy=execution_policy,
                max_workers=max_workers,
                worker_chunksize=worker_chunksize,
                progress_callback=progress_callback,
            ):
                _persist_adaptive_point(
                    cache_map=cache_map,
                    point_results=point_results,
                    evaluated_points=evaluated_points,
                    point=point,
                )
    finally:
        if pending_requests:
            _cache_clear_pending_points(cache_map)
    if pending_requests:
        print(f"  Adaptive batch complete: {len(evaluated_points)} point(s) finished")
    return evaluated_points


def _evaluate_adaptive_neighbor_batch(
    renderer_factory: ABRendererFactory,
    observed: np.ndarray,
    sigma: np.ndarray | None,
    *,
    a_values: np.ndarray,
    b_values: np.ndarray,
    center_a_index: int,
    center_b_index: int,
    q0_min: float,
    q0_max: float,
    hard_q0_min: float | None,
    hard_q0_max: float | None,
    threshold: float,
    mask_type: str,
    target_metric: MetricName,
    xatol: float,
    maxiter: int,
    adaptive_bracketing: bool,
    explicit_q0_start: float | None,
    warm_start_q0: float | None,
    q0_step: float,
    max_bracket_steps: int,
    point_results: dict[tuple[float, float], ABPointResult],
    cache_map: ABPointCache,
    execution_policy: ABRequestedExecutionPolicy,
    max_workers: int | None,
    worker_chunksize: int,
    progress_callback: ProgressCallback | None,
) -> list[ABPointResult]:
    candidate_points: list[tuple[int, int, float | None]] = []

    for i in range(max(0, int(center_a_index) - 1), min(int(a_values.size), int(center_a_index) + 2)):
        for j in range(max(0, int(center_b_index) - 1), min(int(b_values.size), int(center_b_index) + 2)):
            q0_seed = explicit_q0_start if explicit_q0_start is not None else warm_start_q0
            candidate_points.append((int(i), int(j), None if q0_seed is None else float(q0_seed)))

    return _evaluate_adaptive_index_batch(
        renderer_factory,
        observed,
        sigma,
        a_values=a_values,
        b_values=b_values,
        candidate_points=candidate_points,
        q0_min=q0_min,
        q0_max=q0_max,
        hard_q0_min=hard_q0_min,
        hard_q0_max=hard_q0_max,
        threshold=float(threshold),
        mask_type=str(mask_type),
        target_metric=target_metric,
        xatol=float(xatol),
        maxiter=int(maxiter),
        adaptive_bracketing=bool(adaptive_bracketing),
        q0_step=float(q0_step),
        max_bracket_steps=int(max_bracket_steps),
        point_results=point_results,
        cache_map=cache_map,
        execution_policy=execution_policy,
        max_workers=max_workers,
        worker_chunksize=worker_chunksize,
        progress_callback=progress_callback,
    )


def _current_best_point(
    *,
    a_values: np.ndarray,
    b_values: np.ndarray,
    point_results: dict[tuple[float, float], ABPointResult],
) -> tuple[int, int, ABPointResult]:
    best_point: ABPointResult | None = None
    best_index: tuple[int, int] | None = None
    for i, a_value in enumerate(np.asarray(a_values, dtype=float)):
        for j, b_value in enumerate(np.asarray(b_values, dtype=float)):
            point = point_results.get((float(a_value), float(b_value)))
            if point is None:
                continue
            if best_point is None or float(point.objective_value) < float(best_point.objective_value):
                best_point = point
                best_index = (int(i), int(j))
    if best_point is None or best_index is None:
        raise RuntimeError("adaptive search could not identify a best point")
    return int(best_index[0]), int(best_index[1]), best_point


def _expand_axis_for_indices(
    axis_values: np.ndarray,
    *,
    active_indices: list[int],
    step: float,
    bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, bool]:
    values = [float(v) for v in np.asarray(axis_values, dtype=float)]
    lower_bound, upper_bound = (float(bounds[0]), float(bounds[1]))
    tolerance = max(1e-12, abs(float(step)) * 1e-9)
    original_len = len(values)
    prepended = False
    expanded = False

    if active_indices and any(int(index) == 0 for index in active_indices):
        candidate = float(values[0] - float(step))
        if candidate >= lower_bound - tolerance:
            values.insert(0, candidate)
            prepended = True
            expanded = True

    if active_indices and any(int(index) == original_len - 1 for index in active_indices):
        candidate = float(values[-1] + float(step))
        if candidate <= upper_bound + tolerance:
            values.append(candidate)
            expanded = True

    adjusted_indices = np.asarray(active_indices, dtype=int)
    if prepended:
        adjusted_indices = adjusted_indices + 1

    return np.asarray(values, dtype=float), adjusted_indices, bool(expanded)


def _resolve_threshold_limit(*, best_objective_value: float, threshold_metric: float) -> float:
    best_value = float(best_objective_value)
    factor = max(float(threshold_metric), 1.0)
    if best_value > 0.0:
        return float(best_value * factor)
    return float(best_value + (factor - 1.0) * max(abs(best_value), 1e-12))


def _collect_threshold_region_points(
    *,
    a_values: np.ndarray,
    b_values: np.ndarray,
    point_results: dict[tuple[float, float], ABPointResult],
    best_a_index: int,
    best_b_index: int,
    best_point: ABPointResult,
    threshold_metric: float,
) -> list[tuple[int, int, ABPointResult]]:
    threshold_limit = _resolve_threshold_limit(
        best_objective_value=float(best_point.objective_value),
        threshold_metric=float(threshold_metric),
    )
    tolerance = max(1e-12, abs(float(threshold_limit)) * 1e-12)
    threshold_points: list[tuple[int, int, ABPointResult]] = []

    for i, a_value in enumerate(np.asarray(a_values, dtype=float)):
        for j, b_value in enumerate(np.asarray(b_values, dtype=float)):
            point = point_results.get((float(a_value), float(b_value)))
            if point is None:
                continue
            if float(point.objective_value) <= threshold_limit + tolerance:
                threshold_points.append((int(i), int(j), point))

    if threshold_points:
        return threshold_points
    return [(int(best_a_index), int(best_b_index), best_point)]


def _boundary_axes_for_best_point(
    *,
    a_values: np.ndarray,
    b_values: np.ndarray,
    best_a_index: int,
    best_b_index: int,
) -> tuple[str, ...]:
    axes: list[str] = []
    if int(best_a_index) == 0:
        axes.append("a_min")
    elif int(best_a_index) == int(len(a_values)) - 1:
        axes.append("a_max")
    if int(best_b_index) == 0:
        axes.append("b_min")
    elif int(best_b_index) == int(len(b_values)) - 1:
        axes.append("b_max")
    return tuple(axes)


def _seed_point_results_from_cache(
    cache_map: ABPointCache,
    *,
    a_range: tuple[float, float],
    b_range: tuple[float, float],
) -> dict[tuple[float, float], ABPointResult]:
    out: dict[tuple[float, float], ABPointResult] = {}
    a_tol = max(1e-12, abs(float(a_range[1]) - float(a_range[0])) * 1e-12)
    b_tol = max(1e-12, abs(float(b_range[1]) - float(b_range[0])) * 1e-12)
    for key, point in cache_map.items():
        a_value = float(key[0])
        b_value = float(key[1])
        if not (float(a_range[0]) - a_tol <= a_value <= float(a_range[1]) + a_tol):
            continue
        if not (float(b_range[0]) - b_tol <= b_value <= float(b_range[1]) + b_tol):
            continue
        out[(a_value, b_value)] = point
    return out


def _initialize_adaptive_axes(
    point_results: dict[tuple[float, float], ABPointResult],
    *,
    a_start: float,
    b_start: float,
) -> tuple[np.ndarray, np.ndarray]:
    a_values = {float(a_start)}
    b_values = {float(b_start)}
    for a_value, b_value in point_results:
        a_values.add(float(a_value))
        b_values.add(float(b_value))
    return np.asarray(sorted(a_values), dtype=float), np.asarray(sorted(b_values), dtype=float)


def _iter_neighbor_indices(a_index: int, b_index: int) -> list[tuple[int, int]]:
    neighbors: list[tuple[int, int]] = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            neighbors.append((int(a_index) + int(di), int(b_index) + int(dj)))
    return neighbors


def _connected_threshold_region(
    *,
    a_values: np.ndarray,
    b_values: np.ndarray,
    point_results: dict[tuple[float, float], ABPointResult],
    best_a_index: int,
    best_b_index: int,
    threshold_limit: float,
) -> list[tuple[int, int, ABPointResult]]:
    tolerance = max(1e-12, abs(float(threshold_limit)) * 1e-12)
    best_key = (float(a_values[best_a_index]), float(b_values[best_b_index]))
    if best_key not in point_results:
        raise RuntimeError("best adaptive point is missing from point_results")

    pending: list[tuple[int, int]] = [(int(best_a_index), int(best_b_index))]
    visited: set[tuple[int, int]] = set()
    component: list[tuple[int, int, ABPointResult]] = []

    while pending:
        current_a_index, current_b_index = pending.pop()
        if (current_a_index, current_b_index) in visited:
            continue
        visited.add((current_a_index, current_b_index))
        if not (0 <= current_a_index < int(a_values.size) and 0 <= current_b_index < int(b_values.size)):
            continue
        point = point_results.get((float(a_values[current_a_index]), float(b_values[current_b_index])))
        if point is None:
            continue
        if float(point.objective_value) > float(threshold_limit) + tolerance:
            continue
        component.append((int(current_a_index), int(current_b_index), point))
        for neighbor_a_index, neighbor_b_index in _iter_neighbor_indices(current_a_index, current_b_index):
            if (neighbor_a_index, neighbor_b_index) in visited:
                continue
            pending.append((int(neighbor_a_index), int(neighbor_b_index)))

    if component:
        return component
    return [(int(best_a_index), int(best_b_index), point_results[best_key])]


def _collect_basin_frontier_candidates(
    *,
    a_values: np.ndarray,
    b_values: np.ndarray,
    basin_points: list[tuple[int, int, ABPointResult]],
    point_results: dict[tuple[float, float], ABPointResult],
    explicit_q0_start: float | None,
) -> list[tuple[int, int, float | None]]:
    candidate_map: dict[tuple[int, int], float | None] = {}
    basin_index_set = {(int(a_index), int(b_index)) for a_index, b_index, _point in basin_points}

    for a_index, b_index, point in basin_points:
        q0_seed = explicit_q0_start if explicit_q0_start is not None else float(point.q0)
        for neighbor_a_index, neighbor_b_index in _iter_neighbor_indices(int(a_index), int(b_index)):
            if not (0 <= neighbor_a_index < int(a_values.size) and 0 <= neighbor_b_index < int(b_values.size)):
                continue
            if (neighbor_a_index, neighbor_b_index) in basin_index_set:
                continue
            neighbor_key = (float(a_values[neighbor_a_index]), float(b_values[neighbor_b_index]))
            if neighbor_key in point_results:
                continue
            candidate_map[(int(neighbor_a_index), int(neighbor_b_index))] = None if q0_seed is None else float(q0_seed)

    return [
        (int(a_index), int(b_index), q0_seed)
        for (a_index, b_index), q0_seed in sorted(candidate_map.items())
    ]


def _certify_local_minimum_basin(
    *,
    a_values: np.ndarray,
    b_values: np.ndarray,
    point_results: dict[tuple[float, float], ABPointResult],
    basin_points: list[tuple[int, int, ABPointResult]],
    best_a_index: int,
    best_b_index: int,
) -> tuple[bool, tuple[str, ...]]:
    open_axes = set(_boundary_axes_for_best_point(
        a_values=a_values,
        b_values=b_values,
        best_a_index=best_a_index,
        best_b_index=best_b_index,
    ))
    basin_index_set = {(int(a_index), int(b_index)) for a_index, b_index, _point in basin_points}

    for a_index, b_index, _point in basin_points:
        if int(a_index) == 0:
            open_axes.add("a_min")
        if int(a_index) == int(a_values.size) - 1:
            open_axes.add("a_max")
        if int(b_index) == 0:
            open_axes.add("b_min")
        if int(b_index) == int(b_values.size) - 1:
            open_axes.add("b_max")
        for neighbor_a_index, neighbor_b_index in _iter_neighbor_indices(int(a_index), int(b_index)):
            if neighbor_a_index < 0:
                open_axes.add("a_min")
                continue
            if neighbor_a_index >= int(a_values.size):
                open_axes.add("a_max")
                continue
            if neighbor_b_index < 0:
                open_axes.add("b_min")
                continue
            if neighbor_b_index >= int(b_values.size):
                open_axes.add("b_max")
                continue
            if (neighbor_a_index, neighbor_b_index) in basin_index_set:
                continue
            neighbor_key = (float(a_values[neighbor_a_index]), float(b_values[neighbor_b_index]))
            if neighbor_key not in point_results:
                return False, tuple(sorted(open_axes))

    return not bool(open_axes), tuple(sorted(open_axes))


def search_local_minimum_ab(
    renderer_factory: ABRendererFactory,
    observed: np.ndarray,
    sigma: np.ndarray | None,
    *,
    a_start: float,
    b_start: float,
    da: float,
    db: float,
    a_range: tuple[float, float] = (-9.999, 9.999),
    b_range: tuple[float, float] = (-9.999, 9.999),
    q0_min: float,
    q0_max: float,
    hard_q0_min: float | None = None,
    hard_q0_max: float | None = None,
    threshold: float = 0.1,
    mask_type: str = "union",
    target_metric: MetricName = "chi2",
    xatol: float = 1e-3,
    maxiter: int = 200,
    adaptive_bracketing: bool = False,
    q0_start: float | None = None,
    q0_step: float = 1.61803398875,
    max_bracket_steps: int = 12,
    threshold_metric: float = 2.0,
    no_area: bool = False,
    progress_callback: ProgressCallback | None = None,
    cache: ABPointCache | None = None,
    execution_policy: ABRequestedExecutionPolicy = "serial",
    max_workers: int | None = None,
    worker_chunksize: int = 1,
) -> ABLocalSearchResult:
    """Search for a local minimum in `(a, b)` using adaptive Phase 1 and Phase 2 expansion."""
    if float(da) <= 0 or float(db) <= 0:
        raise ValueError("da and db must be positive")
    if float(a_range[0]) > float(a_range[1]) or float(b_range[0]) > float(b_range[1]):
        raise ValueError("search bounds must be ordered")
    if not (float(a_range[0]) <= float(a_start) <= float(a_range[1])):
        raise ValueError("a_start must lie within a_range")
    if not (float(b_range[0]) <= float(b_start) <= float(b_range[1])):
        raise ValueError("b_start must lie within b_range")
    _validate_q0_interval(q0_min=q0_min, q0_max=q0_max)

    cache_map: ABPointCache = {} if cache is None else cache
    point_results = _seed_point_results_from_cache(
        cache_map,
        a_range=(float(a_range[0]), float(a_range[1])),
        b_range=(float(b_range[0]), float(b_range[1])),
    )
    a_arr, b_arr = _initialize_adaptive_axes(
        point_results,
        a_start=float(a_start),
        b_start=float(b_start),
    )
    n_phase1_iters = 0
    n_phase2_iters = 0
    termination_reason = "not_started"
    minimum_certified = False
    frontier_open_axes: tuple[str, ...] = ()

    while True:
        n_phase1_iters += 1
        best_a_index, best_b_index, best_point_before = _current_best_point(
            a_values=a_arr,
            b_values=b_arr,
            point_results=point_results,
        ) if point_results else (0, 0, None)

        if best_point_before is None:
            _evaluate_adaptive_neighbor_batch(
                renderer_factory,
                observed,
                sigma,
                a_values=a_arr,
                b_values=b_arr,
                center_a_index=0,
                center_b_index=0,
                q0_min=q0_min,
                q0_max=q0_max,
                hard_q0_min=hard_q0_min,
                hard_q0_max=hard_q0_max,
                threshold=float(threshold),
                mask_type=mask_type,
                target_metric=target_metric,
                xatol=float(xatol),
                maxiter=int(maxiter),
                adaptive_bracketing=bool(adaptive_bracketing),
                explicit_q0_start=q0_start,
                warm_start_q0=None,
                q0_step=float(q0_step),
                max_bracket_steps=int(max_bracket_steps),
                point_results=point_results,
                cache_map=cache_map,
                execution_policy=execution_policy,
                max_workers=max_workers,
                worker_chunksize=int(worker_chunksize),
                progress_callback=progress_callback,
            )
            if not point_results:
                raise RuntimeError("adaptive search failed to evaluate the starting point")
            best_a_index, best_b_index, best_point_before = _current_best_point(
                a_values=a_arr,
                b_values=b_arr,
                point_results=point_results,
            )

        a_arr, best_a_index, _expanded_a = _expand_axis_around_index(
            a_arr,
            current_index=best_a_index,
            step=float(da),
            bounds=(float(a_range[0]), float(a_range[1])),
        )
        b_arr, best_b_index, _expanded_b = _expand_axis_around_index(
            b_arr,
            current_index=best_b_index,
            step=float(db),
            bounds=(float(b_range[0]), float(b_range[1])),
        )

        _evaluate_adaptive_neighbor_batch(
            renderer_factory,
            observed,
            sigma,
            a_values=a_arr,
            b_values=b_arr,
            center_a_index=best_a_index,
            center_b_index=best_b_index,
            q0_min=q0_min,
            q0_max=q0_max,
            hard_q0_min=hard_q0_min,
            hard_q0_max=hard_q0_max,
            threshold=float(threshold),
            mask_type=mask_type,
            target_metric=target_metric,
            xatol=float(xatol),
            maxiter=int(maxiter),
            adaptive_bracketing=bool(adaptive_bracketing),
            explicit_q0_start=q0_start,
            warm_start_q0=None if q0_start is not None else float(best_point_before.q0),
            q0_step=float(q0_step),
            max_bracket_steps=int(max_bracket_steps),
            point_results=point_results,
            cache_map=cache_map,
            execution_policy=execution_policy,
            max_workers=max_workers,
            worker_chunksize=int(worker_chunksize),
            progress_callback=progress_callback,
        )

        best_a_index, best_b_index, best_point_after = _current_best_point(
            a_values=a_arr,
            b_values=b_arr,
            point_results=point_results,
        )
        if float(best_point_after.objective_value) >= float(best_point_before.objective_value):
            break

    if not bool(no_area):
        while True:
            best_a_index, best_b_index, best_point = _current_best_point(
                a_values=a_arr,
                b_values=b_arr,
                point_results=point_results,
            )
            threshold_points = _connected_threshold_region(
                a_values=a_arr,
                b_values=b_arr,
                point_results=point_results,
                best_a_index=best_a_index,
                best_b_index=best_b_index,
                threshold_limit=_resolve_threshold_limit(
                    best_objective_value=float(best_point.objective_value),
                    threshold_metric=float(threshold_metric),
                ),
            )
            minimum_certified, frontier_open_axes = _certify_local_minimum_basin(
                a_values=a_arr,
                b_values=b_arr,
                point_results=point_results,
                basin_points=threshold_points,
                best_a_index=best_a_index,
                best_b_index=best_b_index,
            )
            if minimum_certified:
                termination_reason = "certified_local_minimum"
                break

            a_arr, shifted_a_indices, expanded_a = _expand_axis_for_indices(
                a_arr,
                active_indices=[int(index) for index, _, _point in threshold_points],
                step=float(da),
                bounds=(float(a_range[0]), float(a_range[1])),
            )
            b_arr, shifted_b_indices, expanded_b = _expand_axis_for_indices(
                b_arr,
                active_indices=[int(index) for _index, index, _point in threshold_points],
                step=float(db),
                bounds=(float(b_range[0]), float(b_range[1])),
            )

            shifted_threshold_points = [
                (int(a_index), int(b_index), point)
                for a_index, b_index, point in zip(
                    shifted_a_indices.tolist(),
                    shifted_b_indices.tolist(),
                    [point for _i, _j, point in threshold_points],
                    strict=False,
                )
            ]
            candidate_points = _collect_basin_frontier_candidates(
                a_values=a_arr,
                b_values=b_arr,
                basin_points=shifted_threshold_points,
                point_results=point_results,
                explicit_q0_start=q0_start,
            )

            evaluated_phase2_points = _evaluate_adaptive_index_batch(
                renderer_factory,
                observed,
                sigma,
                a_values=a_arr,
                b_values=b_arr,
                candidate_points=candidate_points,
                q0_min=q0_min,
                q0_max=q0_max,
                hard_q0_min=hard_q0_min,
                hard_q0_max=hard_q0_max,
                threshold=float(threshold),
                mask_type=mask_type,
                target_metric=target_metric,
                xatol=float(xatol),
                maxiter=int(maxiter),
                adaptive_bracketing=bool(adaptive_bracketing),
                q0_step=float(q0_step),
                max_bracket_steps=int(max_bracket_steps),
                point_results=point_results,
                cache_map=cache_map,
                execution_policy=execution_policy,
                max_workers=max_workers,
                worker_chunksize=int(worker_chunksize),
                progress_callback=progress_callback,
            )
            if not expanded_a and not expanded_b and not evaluated_phase2_points:
                termination_reason = "frontier_exhausted_without_certification"
                break
            n_phase2_iters += 1
    else:
        termination_reason = "phase1_only"

    if termination_reason == "not_started":
        termination_reason = "frontier_exhausted_without_certification"

    slice_descriptor = ABSliceTaskDescriptor(
        key="single_slice",
        domain="generic",
        label="single-slice",
        display_label="single-slice",
    )
    point_tasks = compile_rectangular_point_tasks(
        a_values=a_arr,
        b_values=b_arr,
        slice_descriptor=slice_descriptor,
        q0_min=q0_min,
        q0_max=q0_max,
        target_metric=target_metric,
    )
    points, best_q0, objective_values, chi2, rho2, eta2, success = _materialize_scan_arrays(
        a_arr=a_arr,
        b_arr=b_arr,
        point_tasks=point_tasks,
        point_results=point_results,
        allow_missing=True,
    )
    final_best_a_index, final_best_b_index, _final_best_point = _current_best_point(
        a_values=a_arr,
        b_values=b_arr,
        point_results=point_results,
    )
    best_boundary_axes = _boundary_axes_for_best_point(
        a_values=a_arr,
        b_values=b_arr,
        best_a_index=final_best_a_index,
        best_b_index=final_best_b_index,
    )

    return ABLocalSearchResult(
        a_values=tuple(float(v) for v in a_arr),
        b_values=tuple(float(v) for v in b_arr),
        best_a=float(a_arr[final_best_a_index]),
        best_b=float(b_arr[final_best_b_index]),
        target_metric=target_metric,
        threshold_metric=float(threshold_metric),
        points=points,
        best_q0=best_q0,
        objective_values=objective_values,
        chi2=chi2,
        rho2=rho2,
        eta2=eta2,
        success=success,
        n_phase1_iters=int(n_phase1_iters),
        n_phase2_iters=int(n_phase2_iters),
        best_is_interior=not bool(best_boundary_axes),
        best_boundary_axes=best_boundary_axes,
        minimum_certified=bool(minimum_certified),
        termination_reason=str(termination_reason),
        frontier_open_axes=tuple(str(axis) for axis in frontier_open_axes),
        evaluated_point_count=int(len(point_results)),
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
    mask_type: str = "union",
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
    execution_policy: ABRequestedExecutionPolicy = "serial",
    max_workers: int | None = None,
    worker_chunksize: int = 1,
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

    _validate_q0_interval(q0_min=q0_min, q0_max=q0_max)
    q0_start_arr = _resolve_q0_start_array(
        q0_start_grid=q0_start_grid,
        use_idl_q0_start_heuristic=bool(use_idl_q0_start_heuristic),
        a_arr=a_arr,
        b_arr=b_arr,
    )

    cache_map: ABPointCache = {} if cache is None else cache

    slice_descriptor = ABSliceTaskDescriptor(
        key="single_slice",
        domain="generic",
        label="single-slice",
        display_label="single-slice",
    )
    point_tasks = compile_rectangular_point_tasks(
        a_values=a_arr,
        b_values=b_arr,
        slice_descriptor=slice_descriptor,
        q0_min=q0_min,
        q0_max=q0_max,
        target_metric=target_metric,
    )

    point_results: dict[tuple[float, float], ABPointResult] = {}
    pending_requests: list[ABPointEvaluationRequest] = []
    for point_task in point_tasks:
        key = (float(point_task.a), float(point_task.b))
        cached_point = cache_map.get(key)
        if cached_point is not None:
            point_results[key] = cached_point
            continue
        pending_requests.append(
            ABPointEvaluationRequest(
                task=point_task,
                hard_q0_min=hard_q0_min,
                hard_q0_max=hard_q0_max,
                threshold=float(threshold),
                mask_type=str(mask_type),
                target_metric=target_metric,
                xatol=float(xatol),
                maxiter=int(maxiter),
                adaptive_bracketing=bool(adaptive_bracketing),
                q0_start=None if q0_start_arr is None else float(q0_start_arr[point_task.a_index, point_task.b_index]),
                q0_step=float(q0_step),
                max_bracket_steps=int(max_bracket_steps),
            )
        )

    if pending_requests:
        for point in _execute_ab_requests(
            renderer_factory,
            observed,
            sigma,
            pending_requests=pending_requests,
            execution_policy=execution_policy,
            max_workers=max_workers,
            worker_chunksize=int(worker_chunksize),
            progress_callback=progress_callback,
        ):
            key = (float(point.a), float(point.b))
            cache_map[key] = point
            point_results[key] = point

    points, best_q0, objective_values, chi2, rho2, eta2, success = _materialize_scan_arrays(
        a_arr=a_arr,
        b_arr=b_arr,
        point_tasks=point_tasks,
        point_results=point_results,
    )

    return ABScanResult(
        a_values=tuple(float(v) for v in a_arr),
        b_values=tuple(float(v) for v in b_arr),
        target_metric=target_metric,
        points=points,
        best_q0=best_q0,
        objective_values=objective_values,
        chi2=chi2,
        rho2=rho2,
        eta2=eta2,
        success=success,
    )
