from __future__ import annotations

import os
import time

import pytest

from pychmp.ab_scan_execution import ABExecutionSettings, iter_execute_tasks, resolve_execution_plan
from pychmp.ab_scan_tasks import ABPointTask


def _make_task(index: int) -> ABPointTask:
    return ABPointTask(
        slice_key="single_slice",
        slice_domain="generic",
        slice_label="single-slice",
        slice_display_label="single-slice",
        a=float(index),
        b=float(index + 1),
        a_index=index,
        b_index=0,
        q0_min=0.1,
        q0_max=10.0,
        target_metric="chi2",
        source_kind="rectangular",
    )


def _bootstrap_counter_worker(offset: float) -> dict[str, float | int]:
    return {"offset": float(offset), "count": 0}


def _evaluate_counter_task(task: ABPointTask, worker_state: dict[str, float | int]) -> tuple[int, int, int]:
    worker_state["count"] = int(worker_state["count"]) + 1
    return (os.getpid(), int(worker_state["count"]), int(task.a_index))


def test_iter_execute_tasks_reuses_serial_worker_state() -> None:
    """Keep serial worker state alive across task iteration."""
    results = list(
        iter_execute_tasks(
            [_make_task(0), _make_task(1), _make_task(2)],
            bootstrap_worker=_bootstrap_counter_worker,
            bootstrap_payload=1.0,
            evaluate_task=_evaluate_counter_task,
            settings=ABExecutionSettings(policy="serial"),
        )
    )

    assert [count for _pid, count, _index in results] == [1, 2, 3]
    assert [index for _pid, _count, index in results] == [0, 1, 2]
    assert len({pid for pid, _count, _index in results}) == 1


def test_iter_execute_tasks_reuses_process_worker_state() -> None:
    """Keep process-pool worker state alive across submitted tasks."""
    results = list(
        iter_execute_tasks(
            [_make_task(0), _make_task(1), _make_task(2)],
            bootstrap_worker=_bootstrap_counter_worker,
            bootstrap_payload=1.0,
            evaluate_task=_evaluate_counter_task,
            settings=ABExecutionSettings(policy="process-pool", max_workers=1, chunksize=1),
        )
    )

    assert [count for _pid, count, _index in results] == [1, 2, 3]
    assert [index for _pid, _count, index in results] == [0, 1, 2]
    assert len({pid for pid, _count, _index in results}) == 1


def _evaluate_delayed_counter_task(task: ABPointTask, worker_state: dict[str, float | int]) -> tuple[int, int]:
    if int(task.a_index) == 0:
        time.sleep(0.15)
    else:
        time.sleep(0.01)
    worker_state["count"] = int(worker_state["count"]) + 1
    return (int(task.a_index), int(worker_state["count"]))


def test_iter_execute_tasks_can_yield_process_results_in_completion_order() -> None:
    """Allow process-pool consumers to stream whichever task completes first."""
    results = list(
        iter_execute_tasks(
            [_make_task(0), _make_task(1), _make_task(2)],
            bootstrap_worker=_bootstrap_counter_worker,
            bootstrap_payload=1.0,
            evaluate_task=_evaluate_delayed_counter_task,
            settings=ABExecutionSettings(
                policy="process-pool",
                max_workers=2,
                chunksize=1,
                yield_completion_order=True,
            ),
        )
    )

    assert sorted(index for index, _count in results) == [0, 1, 2]
    assert results[0][0] != 0


def test_resolve_execution_plan_auto_chooses_process_pool_for_larger_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Promote larger scans to process-pool execution in auto mode."""
    monkeypatch.setattr(os, "cpu_count", lambda: 20)

    plan = resolve_execution_plan(task_count=9, requested_policy="auto", max_workers=9)

    assert plan.policy == "process-pool"
    assert plan.max_workers == 5
    assert plan.task_count == 9


def test_resolve_execution_plan_auto_stays_serial_for_tiny_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tiny scans on serial execution in auto mode."""
    monkeypatch.setattr(os, "cpu_count", lambda: 20)

    plan = resolve_execution_plan(task_count=3, requested_policy="auto", max_workers=9)

    assert plan.policy == "serial"
    assert plan.max_workers == 1


def test_resolve_execution_plan_auto_respects_user_worker_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Honor the caller worker cap when auto mode selects processes."""
    monkeypatch.setattr(os, "cpu_count", lambda: 20)

    plan = resolve_execution_plan(task_count=9, requested_policy="auto", max_workers=3)

    assert plan.policy == "process-pool"
    assert plan.max_workers == 3


def test_resolve_execution_plan_honors_single_task_process_request() -> None:
    """Cap explicit process execution to one worker for one task."""
    plan = resolve_execution_plan(task_count=1, requested_policy="process-pool", max_workers=4)

    assert plan.policy == "process-pool"
    assert plan.max_workers == 1


def test_resolve_execution_plan_process_pool_caps_workers_by_task_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Limit process-pool workers to the number of available tasks."""
    monkeypatch.setattr(os, "cpu_count", lambda: 20)

    plan = resolve_execution_plan(task_count=3, requested_policy="process-pool", max_workers=9)

    assert plan.policy == "process-pool"
    assert plan.max_workers == 3
