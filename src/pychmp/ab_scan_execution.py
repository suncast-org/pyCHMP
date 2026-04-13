"""Execution helpers for single-slice `(a, b)` task evaluation."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Literal, TypeVar

from .ab_scan_tasks import ABExecutionPolicy


ABRequestedExecutionPolicy = Literal["auto", "serial", "process-pool"]


TaskT = TypeVar("TaskT")
BootstrapPayloadT = TypeVar("BootstrapPayloadT")
WorkerStateT = TypeVar("WorkerStateT")
ResultT = TypeVar("ResultT")

BootstrapFn = Callable[[BootstrapPayloadT], WorkerStateT]
EvaluateFn = Callable[[TaskT, WorkerStateT], ResultT]


@dataclass(frozen=True)
class ABExecutionSettings:
    """Execution settings shared by single-slice scan engines."""

    policy: ABExecutionPolicy = "serial"
    max_workers: int | None = None
    chunksize: int = 1
    yield_completion_order: bool = False


@dataclass(frozen=True)
class ABResolvedExecutionPlan:
    """Resolved execution decision for one scan."""

    requested_policy: ABRequestedExecutionPolicy
    policy: ABExecutionPolicy
    max_workers: int
    available_cpus: int
    task_count: int


def _resolve_auto_worker_count(*, task_count: int, capped_workers: int) -> int:
    """Return a conservative process-pool width for auto mode.

    The heuristic intentionally avoids parallel startup overhead for tiny scans
    and caps worker growth below the full task count so users get a moderate,
    machine-aware default rather than immediate saturation.
    """

    if int(task_count) < 4 or int(capped_workers) <= 1:
        return 1

    preferred_workers = max(2, (int(task_count) + 1) // 2)
    return min(int(capped_workers), preferred_workers)


def resolve_execution_plan(
    *,
    task_count: int,
    requested_policy: ABRequestedExecutionPolicy,
    max_workers: int | None,
) -> ABResolvedExecutionPlan:
    """Resolve requested execution settings into an executable plan."""

    if int(task_count) < 0:
        raise ValueError("task_count must be non-negative")
    if max_workers is not None and int(max_workers) <= 0:
        raise ValueError("max_workers must be positive when provided")

    available_cpus = max(1, int(os.cpu_count() or 1))
    capped_workers = min(int(task_count) if int(task_count) > 0 else 1, available_cpus)
    if max_workers is not None:
        capped_workers = min(capped_workers, int(max_workers))
    capped_workers = max(1, capped_workers)

    if requested_policy == "serial":
        resolved_policy: ABExecutionPolicy = "serial"
        resolved_workers = 1
    elif requested_policy == "process-pool":
        resolved_policy = "process-pool"
        resolved_workers = capped_workers
    elif requested_policy == "auto":
        auto_workers = _resolve_auto_worker_count(task_count=int(task_count), capped_workers=capped_workers)
        resolved_policy = "process-pool" if auto_workers > 1 else "serial"
        resolved_workers = auto_workers if resolved_policy == "process-pool" else 1
    else:
        raise ValueError(f"unsupported execution policy request: {requested_policy}")

    return ABResolvedExecutionPlan(
        requested_policy=requested_policy,
        policy=resolved_policy,
        max_workers=int(resolved_workers),
        available_cpus=int(available_cpus),
        task_count=int(task_count),
    )


_WORKER_STATE: Any = None
_WORKER_EVALUATE: Callable[[Any, Any], Any] | None = None


def _initialize_task_worker(
    bootstrap_worker: Callable[[Any], Any],
    bootstrap_payload: Any,
    evaluate_task: Callable[[Any, Any], Any],
) -> None:
    global _WORKER_STATE, _WORKER_EVALUATE
    _WORKER_STATE = bootstrap_worker(bootstrap_payload)
    _WORKER_EVALUATE = evaluate_task


def _evaluate_task_in_worker(task: Any) -> Any:
    if _WORKER_EVALUATE is None:
        raise RuntimeError("task worker is not initialized")
    return _WORKER_EVALUATE(task, _WORKER_STATE)


def iter_execute_tasks(
    tasks: Iterable[TaskT],
    *,
    bootstrap_worker: BootstrapFn[BootstrapPayloadT, WorkerStateT],
    bootstrap_payload: BootstrapPayloadT,
    evaluate_task: EvaluateFn[TaskT, WorkerStateT, ResultT],
    settings: ABExecutionSettings,
) -> Iterator[ResultT]:
    """Yield task results under the selected execution policy."""

    task_list = list(tasks)
    if not task_list:
        return iter(())

    if settings.policy == "serial":
        worker_state = bootstrap_worker(bootstrap_payload)
        return (evaluate_task(task, worker_state) for task in task_list)

    if settings.policy != "process-pool":
        raise ValueError(f"unsupported execution policy: {settings.policy}")

    max_workers = settings.max_workers
    chunksize = max(1, int(settings.chunksize))
    yield_completion_order = bool(settings.yield_completion_order)

    def _result_iterator() -> Iterator[ResultT]:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_initialize_task_worker,
            initargs=(bootstrap_worker, bootstrap_payload, evaluate_task),
        ) as executor:
            if yield_completion_order:
                futures = [executor.submit(_evaluate_task_in_worker, task) for task in task_list]
                for future in as_completed(futures):
                    yield future.result()
            else:
                yield from executor.map(_evaluate_task_in_worker, task_list, chunksize=chunksize)

    return _result_iterator()
