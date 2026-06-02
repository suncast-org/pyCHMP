from __future__ import annotations

import numpy as np

from pychmp.viewer_navigation import (
    best_point_selection,
    execution_is_serial,
    search_metric_best_trial_index,
    should_follow_active_refresh_event,
)


def test_execution_is_serial_defaults_to_serial() -> None:
    assert execution_is_serial({}) is True
    assert execution_is_serial({"execution_policy": "process-pool"}) is False


def test_should_follow_active_refresh_event_serial_vs_parallel() -> None:
    assert should_follow_active_refresh_event("point_assigned", serial=True) is True
    assert should_follow_active_refresh_event("trial_committed", serial=True) is True
    assert should_follow_active_refresh_event("point_assigned", serial=False) is False
    assert should_follow_active_refresh_event("point_completed", serial=False) is True


def test_best_point_selection_prefers_latest_completed_on_tie() -> None:
    payload = {
        "a_values": np.asarray([0.0, 0.3], dtype=float),
        "b_values": np.asarray([2.4, 2.7], dtype=float),
        "point_records": [
            {
                "a": 0.0,
                "b": 2.4,
                "a_index": 0,
                "b_index": 0,
                "status": "computed",
                "success": True,
                "metrics": {"eta2": 0.5},
                "fit_q0_trials": (1e-4,),
                "fit_eta2_trials": (0.5,),
                "fit_metric_trials": (0.5,),
                "diagnostics": {"grid_point_id": "p000000", "completed_utc": "2026-05-31T18:00:00Z"},
            },
            {
                "a": 0.3,
                "b": 2.7,
                "a_index": 1,
                "b_index": 1,
                "status": "computed",
                "success": True,
                "metrics": {"eta2": 0.5},
                "fit_q0_trials": (2e-4,),
                "fit_eta2_trials": (0.5,),
                "fit_metric_trials": (0.5,),
                "diagnostics": {"grid_point_id": "p000001", "completed_utc": "2026-05-31T18:05:00Z"},
            },
        ],
    }
    selection = best_point_selection(payload, "eta2")
    assert selection is not None
    assert selection.a_index == 1
    assert selection.b_index == 1
    assert len(selection.tied_records) == 2


def test_search_metric_best_trial_index() -> None:
    record = {
        "fit_q0_trials": (1e-5, 1e-4, 1e-3),
        "fit_eta2_trials": (0.9, 0.4, 0.6),
    }
    assert search_metric_best_trial_index(record, "eta2") == 1
