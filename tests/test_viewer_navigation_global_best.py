from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pychmp.ab_scan_artifacts import search_lifecycle_is_in_progress
from pychmp.viewer_navigation import find_global_best_domain, metric_value_from_record


def test_search_lifecycle_is_in_progress_ignores_completed_active_flag() -> None:
    lifecycle = {
        "active": True,
        "status": "complete",
        "completed_at": "2026-05-28T12:00:00Z",
    }
    assert search_lifecycle_is_in_progress(lifecycle) is False


def test_search_lifecycle_is_in_progress_honors_running_flag() -> None:
    assert search_lifecycle_is_in_progress({"in_progress": True, "status": "running"}) is True


def test_find_global_best_domain_prefers_lower_eta2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h5_path = tmp_path / "scan.h5"

    def _payload(slice_key: str, search_id: str) -> dict:
        if slice_key == "slice_a":
            return {
                "a_values": [0.0],
                "b_values": [2.0],
                "point_records": [
                    {"status": "computed", "a": 0.0, "b": 2.0, "metrics": {"eta2": 0.8}, "diagnostics": {}},
                ],
            }
        return {
            "a_values": [0.0],
            "b_values": [3.0],
            "point_records": [
                {"status": "computed", "a": 0.0, "b": 3.0, "metrics": {"eta2": 0.3}, "diagnostics": {}},
            ],
        }

    def _fake_load_scan_file(path: Path, *, slice_key: str | None = None, search_id: str | None = None, **kwargs):
        return _payload(str(slice_key), str(search_id))

    monkeypatch.setattr("pychmp.viewer_navigation.load_scan_file", _fake_load_scan_file)
    catalog = [
        {"slice_key": "slice_a", "search_id": "search_a"},
        {"slice_key": "slice_b", "search_id": "search_b"},
    ]
    best_slice, best_search = find_global_best_domain(h5_path, catalog=catalog, target_metric="eta2")
    assert best_slice == "slice_b"
    assert best_search == "search_b"
    payload = _payload("slice_b", "search_b")
    value = metric_value_from_record(payload["point_records"][0], "eta2")
    assert value == pytest.approx(0.3)
