from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import pychmp.viewer as viewer_mod
from pychmp.psf import PSFMetadata
from pychmp.viewer import PychmpViewApp, _SelectedSolutionWindow, _center_kernel_to_shape


class _Var:
    def __init__(self, value=None) -> None:
        self._value = value

    def get(self):
        return self._value

    def set(self, value) -> None:
        self._value = value


class _AxisStub:
    transAxes = object()

    def clear(self) -> None:
        pass

    def set_axis_on(self) -> None:
        pass

    def set_axis_off(self) -> None:
        pass

    def add_collection(self, _collection) -> None:
        pass

    def set_title(self, *_args, **_kwargs) -> None:
        pass

    def set_xlabel(self, *_args, **_kwargs) -> None:
        pass

    def set_ylabel(self, *_args, **_kwargs) -> None:
        pass

    def set_xlim(self, *_args, **_kwargs) -> None:
        pass

    def set_ylim(self, *_args, **_kwargs) -> None:
        pass

    def set_box_aspect(self, *_args, **_kwargs) -> None:
        pass

    def set_aspect(self, *_args, **_kwargs) -> None:
        pass

    def tick_params(self, *_args, **_kwargs) -> None:
        pass

    def text(self, *_args, **_kwargs) -> None:
        pass

    def plot(self, *_args, **_kwargs) -> None:
        pass

    def axvline(self, *_args, **_kwargs) -> None:
        pass

    def grid(self, *_args, **_kwargs) -> None:
        pass

    def scatter(self, *_args, **_kwargs) -> None:
        pass


class _AxisLimitsStub(_AxisStub):
    def __init__(self) -> None:
        self.x_limits = None
        self.y_limits = None

    def set_xlim(self, *args, **_kwargs) -> None:
        self.x_limits = args

    def set_ylim(self, *args, **_kwargs) -> None:
        self.y_limits = args

    def annotate(self, *_args, **_kwargs) -> None:
        pass


class _ColorbarAxisStub:
    def clear(self) -> None:
        pass

    def set_axis_on(self) -> None:
        pass

    def set_axis_off(self) -> None:
        pass


class _ColorbarStub:
    def set_label(self, *_args, **_kwargs) -> None:
        pass


class _FigureStub:
    def colorbar(self, *_args, **_kwargs):
        return _ColorbarStub()

    def subplots_adjust(self, *_args, **_kwargs) -> None:
        pass


class _EventStub:
    def __init__(self, *, inaxes, xdata, ydata) -> None:
        self.inaxes = inaxes
        self.xdata = xdata
        self.ydata = ydata


def _make_app(tmp_path: Path, *, phase: str, refresh_active: bool) -> PychmpViewApp:
    app = object.__new__(PychmpViewApp)
    app.artifact_h5 = tmp_path / "adaptive.h5"
    app.payload = {
        "points": {"0": {"status": "computed", "a": 0.3, "b": 2.7}},
        "selected_slice": {"label": "MW: 2.874 GHz"},
        "diagnostics": {
            "artifact_kind": "pychmp_ab_scan_sparse_points",
            "search_mode": "adaptive_local_single_observation",
        },
    }
    app.available_slices = []
    app.refresh_signal_path = tmp_path / "adaptive.h5.refresh"
    app._refresh_signal_phase = phase
    app._ACTIVE_REFRESH_GRACE_S = 10.0
    app.metric_var = _Var("chi2")
    app.run_target_metric = "chi2"
    app.a_index_var = _Var(0)
    app.b_index_var = _Var(0)
    app.navigation_mode_var = _Var("free")
    if refresh_active:
        app.refresh_signal_path.write_text(f"0.0 {phase}\n", encoding="utf-8")
    return app


def test_adaptive_sparse_active_refresh_reports_running(tmp_path: Path) -> None:
    """Adaptive sparse artifacts stay RUNNING while a fresh refresh heartbeat is present."""

    app = _make_app(tmp_path, phase="point 1 saved", refresh_active=True)

    badge, toolbar_detail, info_detail, _color, _foreground = app._scan_state_snapshot()

    assert badge == "RUNNING"
    assert toolbar_detail == "MW: 2.874 GHz | 1/1 computed"
    assert "Last phase: point 1 saved" in info_detail


def test_adaptive_sparse_complete_phase_reports_finished(tmp_path: Path) -> None:
    """Adaptive sparse artifacts report FINISHED once the completion phase is written."""

    app = _make_app(tmp_path, phase="scan complete", refresh_active=False)

    badge, _toolbar_detail, info_detail, _color, _foreground = app._scan_state_snapshot()

    assert badge == "FINISHED"
    assert "Last phase: scan complete" in info_detail


def test_fresh_scan_complete_refresh_does_not_report_running(tmp_path: Path) -> None:
    app = _make_app(tmp_path, phase="scan complete", refresh_active=True)
    app.payload = {
        **dict(app.payload),
        "selected_search": {
            "status": "complete",
            "lifecycle": {"completed_at": "2026-05-31T23:52:49Z"},
        },
        "selected_search_id": "search_live",
        "a_values": [0.0, 0.3],
        "b_values": [2.4, 3.0],
        "points": {
            (0, 0): {"status": "computed", "a": 0.0, "b": 2.4},
            (0, 1): {"status": "computed", "a": 0.0, "b": 3.0},
        },
    }
    app.a_values = np.asarray([0.0, 0.3], dtype=float)
    app.b_values = np.asarray([2.4, 3.0], dtype=float)
    app._refresh_signal_active_point = (0.0, 3.0)
    app._refresh_signal_event = "search_completed"
    app._runner_pid_from_log = lambda: None

    assert app._live_runner_detected() is False
    assert app._heartbeat_activity_present() is False

    app._apply_refresh_signal_payload({"event": "search_completed", "phase": "scan complete"})
    assert app._refresh_signal_active_point is None

    badge, _toolbar_detail, info_detail, _color, _foreground = app._scan_state_snapshot()
    assert badge == "FINISHED"
    assert "Active point:" not in info_detail


def test_scan_state_reports_loading_during_initial_background_load(tmp_path: Path) -> None:
    app = object.__new__(PychmpViewApp)
    app.artifact_h5 = tmp_path / "adaptive.h5"
    app.payload = {}
    app._initial_reload_in_progress = True

    badge, toolbar_detail, info_detail, _color, _foreground = app._scan_state_snapshot()

    assert badge == "LOADING"
    assert toolbar_detail == "Opening artifact"
    assert "load in progress" in info_detail


def test_selected_point_summary_shows_elapsed_seconds_when_available(tmp_path: Path) -> None:
    """Show per-point elapsed time in the Selected Point summary when diagnostics provide it."""

    app = object.__new__(PychmpViewApp)
    app.payload = {
        "points": {
            (0, 0): {
                "status": "computed",
                "q0": 0.067975,
                "target_metric": "chi2",
                "diagnostics": {
                    "chi2": 1.232974e2,
                    "rho2": 2.332496,
                    "eta2": 3.491346e-1,
                    "elapsed_seconds": 176.076,
                },
                "fit_q0_trials": (),
                "fit_metric_trials": (),
            }
        },
        "selected_slice": {"display_label": "MW: 2.874 GHz"},
        "diagnostics": {},
    }
    app.a_values = [ -3.0 ]
    app.b_values = [ 3.9 ]
    app.a_index_var = _Var(0)
    app.b_index_var = _Var(0)
    app.metric_var = _Var("chi2")
    app.run_target_metric = "chi2"
    app.summary_var = _Var("")

    app._refresh_summary()

    assert "elapsed = 176.076 s" in app.summary_var.get()


def test_read_refresh_signal_reads_phase_and_routing_hints_only(tmp_path: Path) -> None:
    app = object.__new__(PychmpViewApp)
    app.refresh_signal_path = tmp_path / "adaptive.h5.refresh"
    app.refresh_signal_path.write_text(
        json.dumps(
            {
                "phase": "2 point(s) pending",
                "slice_key": "euv_193",
                "search_id": "search_b",
                "pending_points": [{"a": -1.2, "b": 2.1}],
                "active_point": {"a": -1.2, "b": 2.1},
                "live_trials": {"q0_trials": [1.0e-5]},
            }
        ),
        encoding="utf-8",
    )

    parsed = app._read_refresh_signal_payload()

    assert parsed["phase"] == "2 point(s) pending"
    assert parsed["slice_key"] == "euv_193"
    assert parsed["search_id"] == "search_b"
    assert "active_point" not in parsed
    assert "live_trials" not in parsed


def test_live_navigation_available_after_artifact_sync(tmp_path: Path) -> None:
    app = object.__new__(PychmpViewApp)
    app.artifact_h5 = tmp_path / "adaptive.h5"
    app.refresh_signal_path = tmp_path / "adaptive.h5.refresh"
    app.run_target_metric = "eta2"
    app._ACTIVE_REFRESH_GRACE_S = 300
    app.refresh_signal_path.write_text(
        json.dumps(
            {
                "version": "2",
                "event": "point_assigned",
                "point_id": "p000000",
                "slice_key": "euv_193",
                "search_id": "search_b",
            }
        ),
        encoding="utf-8",
    )
    app._live_runner_detected = lambda: True
    app.payload = {"selected_slice_key": "euv_193", "selected_search_id": "search_b"}
    app.available_searches = []

    import pychmp.viewer as viewer_module

    viewer_module.load_grid_point_live_state = lambda *_args, **_kwargs: {
        "slice_key": "euv_193",
        "search_id": "search_b",
        "a": -1.2,
        "b": 2.1,
        "metric_name": "eta2",
        "fit_q0_trials": np.asarray([], dtype=float),
        "fit_metric_trials": np.asarray([], dtype=float),
    }

    app._sync_refresh_signal_from_disk()

    assert app._refresh_signal_active_point == (-1.2, 2.1)
    assert app._live_navigation_available() is True


def test_initialize_navigation_mode_does_not_force_active_after_user_chose_free() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("free")
    app._navigation_mode_initialized = True
    app._navigation_mode_user_chosen = True
    app._default_navigation_mode = lambda: "active"
    app._refresh_navigation_control_states = lambda: None

    app._initialize_navigation_mode()

    assert app.navigation_mode_var.get() == "free"


def test_restore_slice_view_state_restores_free_even_with_live_runner() -> None:
    app = object.__new__(PychmpViewApp)
    app.artifact_h5 = Path("/tmp/adaptive.h5")
    app.search_id_var = _Var("search_a")
    app.metric_var = _Var("chi2")
    app.run_target_metric = "chi2"
    app._preferred_initial_metric = None
    app.trials_xmin_var = _Var("")
    app.trials_xmax_var = _Var("")
    app.trials_ymin_var = _Var("")
    app.trials_ymax_var = _Var("")
    app.trials_xscale_var = _Var("linear scale")
    app.trials_yscale_var = _Var("linear scale")
    app.a_index_var = _Var(0)
    app.b_index_var = _Var(0)
    app.navigation_mode_var = _Var("active")
    app._live_runner_detected = lambda: True
    app._default_navigation_mode = lambda: "active"
    app._coerce_navigation_mode_to_availability = PychmpViewApp._coerce_navigation_mode_to_availability.__get__(app)
    app._navigation_mode_radios = {}
    app._live_navigation_available = lambda: True
    app._active_navigation_mode_available = lambda: True
    app._best_navigation_available = lambda: True
    app.a_values = np.asarray([0.0, 0.3], dtype=float)
    app.b_values = np.asarray([2.4, 2.7], dtype=float)
    app.payload = {
        "selected_slice_key": "euv_171",
        "selected_search_id": "search_a",
        "a_values": [0.0, 0.3],
        "b_values": [2.4, 2.7],
        "points": {(0, 0): {"status": "computed", "a": 0.0, "b": 2.4, "metrics": {"chi2": 1.0}}},
    }
    token = app._slice_state_token("euv_171")
    app.slice_display_state = {
        token: {
            "metric": "chi2",
            "a_index": 0,
            "b_index": 0,
            "navigation_mode": "free",
            "trials_by_metric": {},
        }
    }
    app._reset_trials_controls_only = lambda: None
    app._restore_trials_controls_for_metric = lambda _metric: None

    app._restore_slice_view_state()

    assert app.navigation_mode_var.get() == "free"


def test_refresh_signal_payload_parses_phase_and_routing_hints(tmp_path: Path) -> None:
    app = object.__new__(PychmpViewApp)
    app.refresh_signal_path = tmp_path / "adaptive.h5.refresh"
    payload = {
        "phase": "trial 03 complete",
        "slice_key": "euv_193",
        "search_id": "search_live",
        "active_point": {"a": 0.0, "b": 2.4},
        "live_trials": {"q0_trials": [1.0e-5]},
    }
    app.refresh_signal_path.write_text(json.dumps(payload), encoding="utf-8")

    parsed = app._read_refresh_signal_payload()

    assert parsed["phase"] == "trial 03 complete"
    assert parsed["slice_key"] == "euv_193"
    assert parsed["search_id"] == "search_live"


def test_live_trial_series_from_state_reads_heartbeat_q0_trials() -> None:
    app = object.__new__(PychmpViewApp)
    app.run_target_metric = "eta2"
    app.metric_var = _Var("eta2")
    app._load_live_active_point_snapshot = lambda _live_state: None

    q0_trials, metric_trials, point_metric, snapshot = app._live_trial_series_from_state(
        {
            "metric_name": "eta2",
            "q0_trials": [1.0e-5, 1.0e-4, 1.0e-3],
            "metric_trials": [3.0, 2.0, 1.0],
        }
    )

    assert point_metric == "eta2"
    assert snapshot is None
    assert q0_trials.tolist() == [1.0e-5, 1.0e-4, 1.0e-3]
    assert metric_trials.tolist() == [3.0, 2.0, 1.0]


def test_live_trial_series_from_state_uses_display_metric_not_search_target() -> None:
    app = object.__new__(PychmpViewApp)
    app.run_target_metric = "eta2"
    app.metric_var = _Var("chi2")
    app.navigation_mode_var = _Var("free")
    app._load_live_active_point_snapshot = lambda _live_state: None

    q0_trials, metric_trials, point_metric, snapshot = app._live_trial_series_from_state(
        {
            "metric_name": "eta2",
            "fit_q0_trials": [1.0e-5, 1.0e-4],
            "fit_metric_trials": [0.8, 0.5],
            "fit_chi2_trials": [1.2, 0.9],
            "fit_rho2_trials": [0.55, 0.44],
            "fit_eta2_trials": [0.8, 0.5],
        }
    )

    assert point_metric == "chi2"
    assert snapshot is None
    assert q0_trials.tolist() == [1.0e-5, 1.0e-4]
    assert metric_trials.tolist() == [1.2, 0.9]


def test_live_trial_series_from_state_does_not_fallback_to_target_metric() -> None:
    app = object.__new__(PychmpViewApp)
    app.run_target_metric = "eta2"
    app.metric_var = _Var("chi2")
    app.navigation_mode_var = _Var("free")
    app._load_live_active_point_snapshot = lambda _live_state: None

    q0_trials, metric_trials, point_metric, snapshot = app._live_trial_series_from_state(
        {
            "metric_name": "eta2",
            "fit_q0_trials": [1.0e-5, 1.0e-4],
            "fit_metric_trials": [0.8, 0.5],
        }
    )

    assert point_metric == "chi2"
    assert q0_trials.tolist() == [1.0e-5, 1.0e-4]
    assert metric_trials.size == 0
    assert snapshot is None


def test_sync_live_trial_state_populates_live_state_from_artifact() -> None:
    app = object.__new__(PychmpViewApp)
    app.artifact_h5 = Path("/tmp/adaptive.h5")
    app.payload = {"selected_slice_key": "euv_193", "selected_search_id": "search_b"}
    app.available_searches = []
    app.run_target_metric = "eta2"
    app._refresh_signal_point_id = "p000000"

    import pychmp.viewer as viewer_module

    viewer_module.load_grid_point_live_state = lambda *_args, **_kwargs: {
        "slice_key": "euv_193",
        "search_id": "search_b",
        "a": 0.3,
        "b": 2.7,
        "metric_name": "eta2",
        "fit_q0_trials": np.asarray([1.0e-5, 1.0e-4, 1.0e-3], dtype=float),
        "fit_metric_trials": np.asarray([3.0, 2.0, 1.0], dtype=float),
        "fit_shift_x_trials": np.asarray([1.25, -0.5, 0.0], dtype=float),
        "fit_shift_y_trials": np.asarray([-0.75, 2.0, 0.0], dtype=float),
        "fit_find_shift_valid_trials": np.asarray([True, True, False], dtype=bool),
        "q0": 1.0e-2,
        "trial_index": 3,
    }

    app._sync_live_trial_state_from_artifact()

    assert app._refresh_signal_active_point == (0.3, 2.7)
    merged = dict(app._refresh_signal_live_trials or {})
    assert len(np.asarray(merged["fit_q0_trials"], dtype=float)) == 3
    assert merged["active_trial_q0"] == pytest.approx(1.0e-2)
    assert merged["active_trial_index"] == 3
    np.testing.assert_allclose(np.asarray(merged["fit_shift_x_trials"], dtype=float), [1.25, -0.5, 0.0])
    np.testing.assert_allclose(np.asarray(merged["fit_shift_y_trials"], dtype=float), [-0.75, 2.0, 0.0])


def test_live_search_matches_when_heartbeat_search_id_is_null() -> None:
    app = object.__new__(PychmpViewApp)
    app._selected_search_id = lambda: "search_abc"
    app._selected_slice_key = lambda: "mw_2p873584ghz"
    app._live_runner_detected = lambda: True
    app.payload = {
        "selected_search_id": "search_abc",
        "selected_search": {
            "status": "running",
            "in_progress": True,
        },
    }
    app.available_searches = [{"search_id": "search_abc", "in_progress": True, "slice_key": "mw_2p873584ghz"}]

    assert app._live_search_matches_selected({"search_id": None}) is True
    assert app._live_search_matches_selected({"search_id": "search_abc"}) is True
    assert app._live_search_matches_selected({"search_id": "search_other"}) is True


def test_live_search_rejects_unknown_heartbeat_id_for_completed_catalog_search() -> None:
    app = object.__new__(PychmpViewApp)
    app._selected_search_id = lambda: "search_abc"
    app._selected_slice_key = lambda: "mw_2p873584ghz"
    app._live_runner_detected = lambda: True
    app.payload = {
        "selected_search_id": "search_abc",
        "selected_search": {
            "status": "running",
            "in_progress": True,
        },
    }
    app.available_searches = [
        {"search_id": "search_abc", "in_progress": True, "slice_key": "mw_2p873584ghz"},
        {"search_id": "search_other", "status": "complete", "in_progress": False, "slice_key": "mw_2p873584ghz"},
    ]

    assert app._live_search_matches_selected({"search_id": "search_other"}) is False


def test_live_search_does_not_match_completed_search_when_heartbeat_search_id_is_null() -> None:
    app = object.__new__(PychmpViewApp)
    app._selected_search_id = lambda: "search_legacy"
    app._selected_slice_key = lambda: "mw_2p873584ghz"
    app._live_runner_detected = lambda: True
    app._refresh_signal_is_fresh = lambda: True
    app.payload = {
        "selected_slice_key": "mw_2p873584ghz",
        "selected_search": {
            "status": "complete",
            "active": True,
            "lifecycle": {"active": True, "completed_at": "2026-05-28T12:00:00Z"},
        },
    }
    app.available_searches = [
        {
            "search_id": "search_legacy",
            "status": "complete",
            "in_progress": False,
            "slice_key": "mw_2p873584ghz",
        },
        {
            "search_id": "search_live",
            "status": "running",
            "in_progress": True,
            "slice_key": "mw_2p873584ghz",
        },
    ]

    assert app._live_search_matches_selected({"search_id": None, "slice_key": "mw_2p873584ghz"}) is False


def test_should_force_live_trials_when_selection_matches_active_point() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("active")
    app._refresh_signal_active_point = (0.3, 2.1)
    app._refresh_signal_slice_key = "mw_2p873584ghz"
    app._selected_search_id = lambda: "search_abc"
    app._live_runner_detected = lambda: True
    app.a_index_var = _Var(0)
    app.b_index_var = _Var(0)
    app.payload = {
        "selected_slice_key": "mw_2p873584ghz",
        "selected_search_id": "search_abc",
        "selected_search": {"status": "running", "in_progress": True},
        "points": {},
    }
    app.a_values = np.asarray([0.3], dtype=float)
    app.b_values = np.asarray([2.1], dtype=float)
    app._live_trial_series_from_state = lambda _live_state: (
        np.asarray([1.0e-5], dtype=float),
        np.asarray([1.0], dtype=float),
        "eta2",
        None,
    )

    assert app._should_force_live_trials(
        {
            "slice_key": "mw_2p873584ghz",
            "search_id": "search_abc",
            "a_index": 0,
            "b_index": 0,
        }
    ) is True


def test_should_force_live_trials_false_when_selection_differs_from_active_point() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("free")
    app._refresh_signal_active_point = (0.0, 2.4)
    app._refresh_signal_slice_key = "mw_2p873584ghz"
    app.a_index_var = _Var(1)
    app.b_index_var = _Var(1)
    app.metric_var = _Var("eta2")
    app._live_runner_detected = lambda: True
    app._refresh_signal_is_fresh = lambda: True
    app.payload = {
        "selected_slice_key": "mw_2p873584ghz",
        "selected_search_id": "search_abc",
        "points": {
            (1, 1): {"a": 0.3, "b": 2.7, "status": "computed"},
        },
    }
    app.a_values = np.asarray([0.0, 0.3], dtype=float)
    app.b_values = np.asarray([2.4, 2.7], dtype=float)

    assert app._should_force_live_trials(
        {
            "slice_key": "mw_2p873584ghz",
            "search_id": "search_abc",
            "a_index": 1,
            "b_index": 1,
        }
    ) is False


def test_heartbeat_requires_payload_reload_for_point_assigned_event() -> None:
    app = object.__new__(PychmpViewApp)
    app._last_payload_reload_at_s = 0.0
    app._MIN_HEARTBEAT_RELOAD_INTERVAL_S = 2.0

    assert app._heartbeat_requires_payload_reload({"version": 2, "event": "point_assigned"}) is False


def test_heartbeat_requires_payload_reload_bypasses_rate_limit_for_point_completed() -> None:
    app = object.__new__(PychmpViewApp)
    app._last_payload_reload_at_s = float(__import__("time").time())
    app._MIN_HEARTBEAT_RELOAD_INTERVAL_S = 60.0

    assert app._heartbeat_requires_payload_reload({"version": 2, "event": "point_completed"}) is False


def test_heartbeat_requires_payload_reload_for_saved_phase() -> None:
    app = object.__new__(PychmpViewApp)
    app._last_payload_reload_at_s = 0.0
    app._MIN_HEARTBEAT_RELOAD_INTERVAL_S = 0.0

    assert app._heartbeat_requires_payload_reload({"phase": "point 12 saved"}) is True


def test_heartbeat_requires_payload_reload_ignores_trial_only_phase() -> None:
    app = object.__new__(PychmpViewApp)
    app._last_payload_reload_at_s = 0.0
    app._MIN_HEARTBEAT_RELOAD_INTERVAL_S = 0.0

    assert app._heartbeat_requires_payload_reload({"phase": "trial 03 complete"}) is False


def test_poll_external_refresh_signal_uses_lightweight_refresh_for_trial_phase(tmp_path: Path) -> None:
    app = object.__new__(PychmpViewApp)
    app._is_closing = False
    app.refresh_signal_path = tmp_path / "adaptive.h5.refresh"
    app.refresh_signal_path.write_text("{}\n", encoding="utf-8")
    app._refresh_signal_mtime_ns = -1
    app._refresh_signal_phase = ""
    app._refresh_signal_slice_key = None
    app._refresh_signal_pending_points = []
    app._refresh_signal_active_point = None
    app._refresh_signal_live_trials = None
    app._external_refresh_after_id = None
    app._EXTERNAL_REFRESH_POLL_MS = 1200
    app._last_payload_reload_at_s = 0.0
    app._MIN_HEARTBEAT_RELOAD_INTERVAL_S = 0.0

    calls = {"reload": 0, "refresh": 0, "sync": 0}
    app._read_refresh_signal_payload = lambda: {
        "phase": "trial 03 complete",
        "slice_key": "euv_171",
        "search_id": "search_live",
    }
    app._reload_payload = lambda: calls.__setitem__("reload", calls["reload"] + 1)
    app._sync_live_trial_state_from_artifact = lambda: calls.__setitem__("sync", calls["sync"] + 1)
    app._refresh_scan_state_display = lambda: None
    app._refresh_action_states = lambda: None
    app._refresh_all = lambda **_kwargs: calls.__setitem__("refresh", calls["refresh"] + 1)
    app.payload = {"points": {}}
    app.navigation_mode_var = _Var("free")

    class _RootStub:
        def after(self, *_args, **_kwargs):
            return "after-id"

    app.root = _RootStub()

    app._poll_external_refresh_signal()

    assert calls["reload"] == 0
    assert calls["sync"] == 1
    assert calls["refresh"] == 0


def test_scan_state_reports_interrupted_for_empty_search_with_stale_live_state(tmp_path: Path) -> None:
    app = _make_app(tmp_path, phase="trial 06 active", refresh_active=False)
    app.payload = {
        "points": {},
        "selected_slice_key": "mw_2p873584ghz",
        "selected_slice": {"label": "MW: 2.874 GHz", "key": "mw_2p873584ghz"},
        "selected_search_id": "search_a4c3736655362921",
        "selected_search": {
            "status": "empty",
            "active": True,
            "lifecycle": {"active": True, "in_progress": True, "status": "empty"},
            "target_metric": "eta2",
        },
        "diagnostics": {
            "artifact_kind": "pychmp_ab_scan_sparse_points",
            "search_mode": "adaptive_local_single_observation",
        },
    }
    app._refresh_signal_active_point = (0.3, 2.7)
    app._refresh_signal_live_trials = {
        "slice_key": "mw_2p873584ghz",
        "search_id": "search_a4c3736655362921",
        "metric_name": "eta2",
    }
    app._refresh_signal_slice_key = "mw_2p873584ghz"
    app._live_runner_detected = lambda: False
    app._active_point_scoped_to_selection = lambda: None

    badge, toolbar_detail, info_detail, _color, _foreground = app._scan_state_snapshot()

    assert badge == "INTERRUPTED"
    assert toolbar_detail == "MW: 2.874 GHz | search_a4c3736655362921 | 0/0 computed"
    assert "Computed: 0" in info_detail


def test_scan_state_reports_finished_when_saved_search_complete_even_if_refresh_fresh(tmp_path: Path) -> None:
    app = _make_app(tmp_path, phase="trial 03 complete", refresh_active=True)
    app.payload["selected_search"] = {
        "status": "complete",
        "lifecycle": {"completed_at": "2026-05-28T12:00:00Z", "active": True},
        "target_metric": "eta2",
    }
    app._refresh_signal_active_point = (0.3, 2.7)
    app._refresh_signal_live_trials = {
        "slice_key": "mw_6p929688ghz",
        "search_id": "search_a",
        "metric_name": "eta2",
        "fit_q0_trials": np.asarray([1.0e-5], dtype=float),
        "fit_metric_trials": np.asarray([0.5], dtype=float),
    }
    app._refresh_signal_slice_key = "mw_6p929688ghz"
    app._runner_pid_from_log = lambda: 4242
    app._process_is_running = lambda _pid: True
    app.navigation_mode_var = _Var("active")

    badge, _toolbar_detail, info_detail, _color, _foreground = app._scan_state_snapshot()

    assert badge == "FINISHED"
    assert "Navigation mode: active" in info_detail


def test_scan_state_reports_running_for_live_runner_on_other_slice(tmp_path: Path) -> None:
    app = _make_app(tmp_path, phase="8 point(s) pending", refresh_active=True)
    app.payload["selected_slice_key"] = "euv_171"
    app.payload["selected_slice"] = {"key": "euv_171", "label": "EUV: 171 A", "domain": "euv"}
    app.payload["selected_search"] = {
        "status": "in_progress",
        "lifecycle": {"active": True, "in_progress": True},
        "target_metric": "eta2",
    }
    app.available_slices = [
        {"key": "euv_171", "label": "EUV: 171 A", "domain": "euv"},
        {"key": "euv_193", "label": "EUV: 193 A", "domain": "euv"},
    ]
    app._refresh_signal_slice_key = "euv_193"
    app._refresh_signal_search_id = "search_a"
    app._runner_pid_from_log = lambda: 4242
    app._process_is_running = lambda _pid: True

    badge, _toolbar_detail, info_detail, _color, _foreground = app._scan_state_snapshot()

    assert badge == "RUNNING"
    assert "Live scan slice: EUV: 193 A" in info_detail


def test_default_navigation_mode_is_free_for_stale_active_search(tmp_path: Path) -> None:
    app = _make_app(tmp_path, phase="", refresh_active=False)
    app.available_searches = [{"search_id": "search_a", "active": True, "lifecycle": {"active": True}}]
    app.payload["selected_search"] = {"active": True, "lifecycle": {"active": True}, "target_metric": "eta2"}
    app._live_runner_detected = lambda: False

    mode = app._default_navigation_mode()

    assert mode == "free"
    assert "interrupted" in app._stale_active_notice.lower()


def test_grid_summary_prefers_existing_png_over_plot_script(tmp_path: Path) -> None:
    app = object.__new__(PychmpViewApp)
    app.artifact_h5 = tmp_path / "adaptive.h5"
    app.status_var = _Var("")
    expected_png = tmp_path / "adaptive_grid.png"
    expected_png.write_bytes(b"png")
    calls = {"open": 0, "plot": 0}

    app._open_external_file = lambda path: calls.__setitem__("open", calls["open"] + 1) or (path == expected_png)
    app._open_plot_script = lambda *_args: calls.__setitem__("plot", calls["plot"] + 1)

    app._open_grid_summary()

    assert calls["open"] == 1
    assert calls["plot"] == 0
    assert "Opened saved grid summary PNG" in str(app.status_var.get())


def test_grid_summary_falls_back_to_plot_script_when_png_missing(tmp_path: Path) -> None:
    app = object.__new__(PychmpViewApp)
    app.artifact_h5 = tmp_path / "adaptive.h5"
    app.status_var = _Var("")
    calls = {"plot": 0}

    app._open_plot_script = lambda *_args: calls.__setitem__("plot", calls["plot"] + 1)

    app._open_grid_summary()

    assert calls["plot"] == 1
    assert "Generating grid summary plot" in str(app.status_var.get())


def test_selected_solution_plot_context_uses_live_trials_without_saved_point() -> None:
    app = object.__new__(PychmpViewApp)
    app._has_selected_point = lambda: False
    app._live_trial_state = lambda: {
        "active_a": 0.3,
        "active_b": 2.7,
        "metric_name": "eta2",
        "q0_trials": [1.0e-5, 1.0e-4, 1.0e-3],
        "metric_trials": [3.0, 2.0, 1.0],
    }
    app._should_use_live_trials = lambda _live_state: True
    app.trial_index_var = _Var(1)
    app.run_target_metric = "eta2"
    app.metric_var = _Var("eta2")
    app.payload = {
        "diagnostics": {
            "model_path": "/tmp/model.h5",
            "ebtel_path": "/tmp/ebtel.sav",
            "spectral_domain": "mw",
            "active_frequency_ghz": 2.874,
        },
        "observed": np.zeros((2, 2), dtype=float),
        "wcs_header": fits.Header(),
        "selected_slice": {"display_label": "MW: 2.874 GHz"},
        "blos_reference": None,
        "psf_kernel": None,
    }
    app._slice_label = lambda _descriptor: "MW: 2.874 GHz"
    app.trials_xmin_var = _Var("")
    app.trials_xmax_var = _Var("")
    app.trials_ymin_var = _Var("")
    app.trials_ymax_var = _Var("")
    app.trials_xscale_var = _Var("linear scale")
    app.trials_yscale_var = _Var("linear scale")
    app._parse_axis_limit = lambda _text: None
    app._live_trial_series_from_state = lambda _live_state: (
        np.asarray([1.0e-5, 1.0e-4, 1.0e-3], dtype=float),
        np.asarray([3.0, 2.0, 1.0], dtype=float),
        "eta2",
        {
            "trial_raw_modeled_maps": np.stack(
                [
                    np.ones((2, 2), dtype=float),
                    np.ones((2, 2), dtype=float) * 2.0,
                    np.ones((2, 2), dtype=float) * 3.0,
                ]
            ),
        },
    )

    context = app._selected_solution_plot_context()

    assert context is not None
    diagnostics = dict(context["diagnostics"])
    assert diagnostics["selected_trial_index"] == 1
    assert diagnostics["selected_trial_maps_available"] is True
    assert float(diagnostics["q0_recovered"]) == 1.0e-4


def test_selected_solution_plot_context_prefers_artifact_snapshot_for_live_trials(monkeypatch: pytest.MonkeyPatch) -> None:
    app = object.__new__(PychmpViewApp)
    app.artifact_h5 = Path("/tmp/live-artifact.h5")
    app._has_selected_point = lambda: False
    app._live_trial_state = lambda: {
        "active_a": 0.3,
        "active_b": 2.7,
        "metric_name": "eta2",
        "q0_trials": [1.0e-5, 1.0e-4],
        "metric_trials": [3.0, 2.0],
    }
    app._should_use_live_trials = lambda _live_state: True
    app.trial_index_var = _Var(1)
    app.run_target_metric = "eta2"
    app.metric_var = _Var("eta2")
    app.payload = {
        "diagnostics": {
            "model_path": "/tmp/model.h5",
            "ebtel_path": "/tmp/ebtel.sav",
            "spectral_domain": "mw",
            "active_frequency_ghz": 2.874,
        },
        "observed": np.zeros((2, 2), dtype=float),
        "wcs_header": fits.Header(),
        "selected_slice": {"display_label": "MW: 2.874 GHz"},
        "selected_slice_key": "mw_2p874000ghz",
        "selected_search_id": "search-live",
        "blos_reference": None,
        "psf_kernel": None,
    }
    app._slice_label = lambda _descriptor: "MW: 2.874 GHz"
    app.trials_xmin_var = _Var("")
    app.trials_xmax_var = _Var("")
    app.trials_ymin_var = _Var("")
    app.trials_ymax_var = _Var("")
    app.trials_xscale_var = _Var("linear scale")
    app.trials_yscale_var = _Var("linear scale")
    app._parse_axis_limit = lambda _text: None
    app._render_live_selected_trial_maps = lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not rerender"))
    app._live_trial_series_from_state = lambda _live_state: (
        np.asarray([1.0e-5, 1.0e-4], dtype=float),
        np.asarray([3.0, 2.0], dtype=float),
        "eta2",
        {
            "trial_raw_modeled_maps": np.stack(
                [
                    np.full((2, 2), 1.0, dtype=float),
                    np.full((2, 2), 2.0, dtype=float),
                ]
            ),
        },
    )

    context = app._selected_solution_plot_context()

    assert context is not None
    np.testing.assert_allclose(context["modeled_best"], np.full((2, 2), 2.0, dtype=float))
    diagnostics = dict(context["diagnostics"])
    assert diagnostics["selected_trial_maps_available"] is True


def test_sync_live_trial_state_prefers_artifact_coords_over_stale_heartbeat() -> None:
    app = object.__new__(PychmpViewApp)
    app.artifact_h5 = Path("/tmp/adaptive.h5")
    app.payload = {"selected_slice_key": "euv_171", "selected_search_id": "search-live"}
    app.available_searches = []
    app.run_target_metric = "eta2"
    app._refresh_signal_slice_key = "euv_171"
    app._refresh_signal_search_id = "search-live"
    app._refresh_signal_point_id = "p000001"

    artifact_live = {
        "slice_key": "euv_171",
        "search_id": "search-live",
        "a": 0.3,
        "b": 2.7,
        "metric_name": "eta2",
        "fit_q0_trials": np.asarray([1.0e-5, 1.0e-4]),
        "fit_metric_trials": np.asarray([0.8, 0.7]),
    }

    viewer_mod.load_grid_point_live_state = lambda *_args, **_kwargs: artifact_live

    app._sync_live_trial_state_from_artifact()

    assert app._refresh_signal_active_point == (0.3, 2.7)
    merged = dict(app._refresh_signal_live_trials or {})
    assert float(merged["a"]) == pytest.approx(0.3)
    assert float(merged["b"]) == pytest.approx(2.7)
    assert len(np.asarray(merged["fit_q0_trials"], dtype=float)) == 2


def test_load_selected_trial_maps_reads_grid_point_trial_maps(tmp_path: Path) -> None:
    from pychmp.ab_scan_artifacts import UNIFIED_ARTIFACT_KIND, write_point_scan_artifact
    from pychmp.grid_points import (
        GRID_POINTS_CONTRACT_VERSION,
        GridPointAssignedEvent,
        GridTrialCommittedEvent,
        apply_grid_point_event_with_retry,
    )

    out_h5 = tmp_path / "artifact_grid_maps.h5"
    diagnostics = {
        "artifact_kind": UNIFIED_ARTIFACT_KIND,
        "slice_key": "euv_171",
        "target_slice_key": "euv_171",
        "search_id": "search-live",
        "selected_search_id": "search-live",
        "spectral_domain": "euv",
        "spectral_label": "171 A",
        "euv_channel": "171",
        "euv_instrument": "AIA",
        "target_metric": "eta2",
        "contract_version": GRID_POINTS_CONTRACT_VERSION,
        "model_path": str(tmp_path / "model.h5"),
        "ebtel_path": str(tmp_path / "ebtel.sav"),
        "tbase": 1.0,
        "nbase": 1.0,
        "map_nx": 2,
        "map_ny": 2,
        "map_dx_arcsec": 2.0,
        "map_dy_arcsec": 2.0,
        "model_id": "model-123",
        "model_sha256": "a" * 64,
        "fits_file": "/tmp/obs.fits",
        "fits_sha256": "b" * 64,
        "ebtel_sha256": "c" * 64,
        "frequency_ghz": 5.7,
        "map_xc_arcsec": 0.0,
        "map_yc_arcsec": 0.0,
        "observer_name": "earth",
        "observer_lonc_deg": 0.0,
        "observer_b0sun_deg": 0.0,
        "observer_dsun_cm": 1.495978707e13,
        "observer_obs_time": "2020-11-26T20:00:00",
    }
    observed = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma = np.ones_like(observed)
    wcs_header = fits.Header()
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma,
        wcs_header=wcs_header,
        diagnostics=diagnostics,
        point_records=[],
    )
    point_id = apply_grid_point_event_with_retry(
        out_h5,
        GridPointAssignedEvent(
            a=0.3,
            b=2.7,
            q0_start=1.0e-5,
            next_q0=1.0e-5,
            metric_name="eta2",
        ),
        observed=observed,
        sigma_map=sigma,
        wcs_header=wcs_header,
        diagnostics=diagnostics,
    )
    apply_grid_point_event_with_retry(
        out_h5,
        GridTrialCommittedEvent(
            point_id=str(point_id),
            trial_index=0,
            q0=1.0e-5,
            metric=0.8,
            next_q0=1.0e-5,
            best_trial_index=0,
            best_metric=0.8,
            raw_modeled_map=np.array([[10.0, 11.0], [12.0, 13.0]], dtype=np.float32),
        ),
        observed=observed,
        sigma_map=sigma,
        wcs_header=wcs_header,
        diagnostics=diagnostics,
    )

    app = object.__new__(PychmpViewApp)
    app.artifact_h5 = out_h5
    app.payload = {
        "selected_slice_key": "euv_171",
        "selected_search_id": "search-live",
    }
    app._selected_trial_map_cache_key = None
    app._selected_trial_map_cache_value = None

    payload = app._load_selected_trial_maps(a_value=0.3, b_value=2.7, selected_trial_index=0)

    assert payload is not None
    np.testing.assert_allclose(
        np.asarray(payload["raw_modeled_best"], dtype=float),
        np.array([[10.0, 11.0], [12.0, 13.0]], dtype=float),
    )


def test_refresh_action_states_enables_display_for_live_trials() -> None:
    class _ButtonStub:
        def __init__(self) -> None:
            self.states: list[tuple[str, ...]] = []

        def state(self, args) -> None:
            self.states.append(tuple(args))

    app = object.__new__(PychmpViewApp)
    app.artifact_h5 = Path("/tmp/adaptive.h5")
    app.open_artifact_button = _ButtonStub()
    app.display_selected_button = _ButtonStub()
    app.summary_button = _ButtonStub()
    app.refresh_button = _ButtonStub()
    app._has_selected_point = lambda: False
    app._live_trial_state = lambda: {"active_a": 0.3, "active_b": 2.7}
    app._live_trials_context_available = lambda _state: True
    app._live_trials_streaming = lambda _state: True
    app._refresh_navigation_control_states = lambda: None

    app._refresh_action_states()

    assert app.display_selected_button.states[-1] == ("!disabled",)


def test_open_selected_maps_uses_live_context_without_saved_point() -> None:
    app = object.__new__(PychmpViewApp)
    app._selected_solution_plot_context = lambda: {"diagnostics": {}}
    app.status_var = _Var("")

    class _WindowStub:
        def __init__(self) -> None:
            self.present_calls = 0
            self.update_calls = 0

        def present(self) -> None:
            self.present_calls += 1

        def update_selection(self) -> None:
            self.update_calls += 1

    window = _WindowStub()
    app.selected_solution_window = window

    app._open_selected_maps()

    assert window.present_calls == 1
    assert window.update_calls == 1


def test_schedule_selected_solution_update_defers_and_coalesces_refresh() -> None:
    app = object.__new__(PychmpViewApp)
    app._is_closing = False
    scheduled: dict[str, Any] = {"callback": None, "cancelled": []}

    class _RootStub:
        def after_idle(self, callback):
            scheduled["callback"] = callback
            return "after-id-1"

        def after_cancel(self, after_id):
            scheduled["cancelled"].append(after_id)

    class _WindowStub:
        def __init__(self) -> None:
            self.update_calls = 0

        def update_selection(self) -> None:
            self.update_calls += 1

    app.root = _RootStub()
    app.selected_solution_window = _WindowStub()
    app._selected_solution_update_after_id = None

    app._schedule_selected_solution_update()
    assert app._selected_solution_update_after_id == "after-id-1"
    assert scheduled["cancelled"] == []

    app._schedule_selected_solution_update()
    assert scheduled["cancelled"] == ["after-id-1"]

    callback = scheduled["callback"]
    assert callback is not None
    callback()

    assert app._selected_solution_update_after_id is None
    assert app.selected_solution_window.update_calls == 1


def test_apply_active_point_selection_keeps_existing_valid_selection() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("free")
    app._refresh_signal_active_point = (0.3, 2.7)
    app.a_values = np.asarray([0.0, 0.3], dtype=float)
    app.b_values = np.asarray([2.4, 2.7], dtype=float)
    app.a_index_var = _Var(0)
    app.b_index_var = _Var(0)
    app.metric_var = _Var("chi2")
    app.payload = {
        "points": {
            (0, 0): {
                "status": "computed",
                "fit_q0_trials": (),
                "fit_metric_trials": (),
            }
        }
    }

    app._apply_active_point_selection()

    assert app.a_index_var.get() == 0
    assert app.b_index_var.get() == 0


def test_apply_active_point_selection_force_overrides_existing_selection() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("free")
    app.run_target_metric = "chi2"
    app._refresh_navigation_control_states = lambda: None
    app._refresh_selector_values = lambda: None
    app._live_navigation_available = lambda: True
    app._active_navigation_mode_available = lambda: True
    app._refresh_signal_active_point = (0.3, 2.7)
    app.a_values = np.asarray([0.0, 0.3], dtype=float)
    app.b_values = np.asarray([2.4, 2.7], dtype=float)
    app.a_index_var = _Var(0)
    app.b_index_var = _Var(0)
    app.metric_var = _Var("chi2")
    app.payload = {
        "points": {
            (0, 0): {
                "status": "computed",
                "fit_q0_trials": (),
                "fit_metric_trials": (),
            }
        }
    }

    app._apply_active_point_selection(force=True)

    assert app.a_index_var.get() == 1
    assert app.b_index_var.get() == 1


def test_resolve_restored_point_indices_preserves_unsaved_cell() -> None:
    app = object.__new__(PychmpViewApp)
    app.a_values = np.asarray([0.0, 0.3], dtype=float)
    app.b_values = np.asarray([2.4, 2.7], dtype=float)
    app.payload = {
        "a_values": [0.0, 0.3],
        "b_values": [2.4, 2.7],
        "points": {
            (0, 0): {"status": "computed", "a": 0.0, "b": 2.4, "metrics": {"chi2": 1.0}},
            (1, 1): {"status": "missing", "a": 0.3, "b": 2.7},
        },
    }

    restored = app._resolve_restored_point_indices(
        metric_name="chi2",
        a_index=1,
        b_index=1,
        default_a=0,
        default_b=0,
    )

    assert restored == (1, 1)


def test_resolve_restored_point_indices_preserves_grid_cell_without_record() -> None:
    app = object.__new__(PychmpViewApp)
    app.a_values = np.asarray([0.0, 0.3], dtype=float)
    app.b_values = np.asarray([2.4, 2.7], dtype=float)
    app.payload = {
        "a_values": [0.0, 0.3],
        "b_values": [2.4, 2.7],
        "points": {
            (0, 0): {"status": "computed", "a": 0.0, "b": 2.4, "metrics": {"chi2": 1.0}},
        },
    }

    restored = app._resolve_restored_point_indices(
        metric_name="chi2",
        a_index=1,
        b_index=1,
        default_a=0,
        default_b=0,
    )

    assert restored == (1, 1)


def test_ensure_selected_point_exists_keeps_follow_active_selection() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("active")
    app._refresh_signal_active_point = (0.3, 2.7)
    app._live_navigation_available = lambda: True
    app._active_navigation_mode_available = lambda: True
    app.a_values = np.asarray([0.0, 0.3], dtype=float)
    app.b_values = np.asarray([2.4, 2.7], dtype=float)
    app.a_index_var = _Var(1)
    app.b_index_var = _Var(1)
    app.metric_var = _Var("chi2")
    app.payload = {
        "a_values": [0.0, 0.3],
        "b_values": [2.4, 2.7],
        "points": {
            (0, 0): {"status": "computed", "a": 0.0, "b": 2.4, "metrics": {"chi2": 1.0}},
        },
    }
    app._refresh_selector_values = lambda: None

    assert app._ensure_selected_point_exists(metric_name="chi2") is True
    assert app.a_index_var.get() == 1
    assert app.b_index_var.get() == 1


def test_ensure_selected_point_exists_does_not_remap_valid_unsaved_grid_cell() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("free")
    app.a_values = np.asarray([0.0, 0.3], dtype=float)
    app.b_values = np.asarray([2.4, 2.7], dtype=float)
    app.a_index_var = _Var(1)
    app.b_index_var = _Var(1)
    app.metric_var = _Var("chi2")
    app.payload = {
        "a_values": [0.0, 0.3],
        "b_values": [2.4, 2.7],
        "points": {
            (0, 0): {"status": "computed", "a": 0.0, "b": 2.4, "metrics": {"chi2": 1.0}},
        },
    }
    app._refresh_selector_values = lambda: None

    assert app._ensure_selected_point_exists(metric_name="chi2") is True
    assert app.a_index_var.get() == 1
    assert app.b_index_var.get() == 1


def test_restore_slice_view_state_preserves_free_coordinates_from_pending_ab() -> None:
    app = object.__new__(PychmpViewApp)
    app.artifact_h5 = Path("/tmp/adaptive.h5")
    app.search_id_var = _Var("search_b")
    app.metric_var = _Var("eta2")
    app.run_target_metric = "eta2"
    app._preferred_initial_metric = None
    app.trials_xmin_var = _Var("")
    app.trials_xmax_var = _Var("")
    app.trials_ymin_var = _Var("")
    app.trials_ymax_var = _Var("")
    app.trials_xscale_var = _Var("linear scale")
    app.trials_yscale_var = _Var("linear scale")
    app.a_index_var = _Var(99)
    app.b_index_var = _Var(99)
    app.navigation_mode_var = _Var("free")
    app._refresh_navigation_control_states = lambda: None
    app._free_selection_ab = (0.25, 4.0)
    app.a_values = np.asarray([-0.75, -0.25, 0.25], dtype=float)
    app.b_values = np.asarray([2.25, 4.0], dtype=float)
    app.payload = {
        "selected_slice_key": "mw_2874",
        "selected_search_id": "search_b",
        "target_metric": "eta2",
        "a_values": [-0.75, -0.25, 0.25],
        "b_values": [2.25, 4.0],
        "points": {
            (0, 0): {"status": "computed", "a": -0.75, "b": 2.25, "metrics": {"eta2": 0.9}},
            (2, 1): {"status": "computed", "a": 0.25, "b": 4.0, "metrics": {"eta2": 0.4}},
        },
    }
    app.slice_display_state = {}
    app._reset_trials_controls_only = lambda: None
    app._restore_trials_controls_for_metric = lambda _metric: None
    app._default_point_selection = lambda _metric: (0, 0)

    app._restore_slice_view_state()

    assert app.a_index_var.get() == 2
    assert app.b_index_var.get() == 1
    assert app._free_selection_ab == (0.25, 4.0)


def test_restore_free_selection_keeps_coordinates_outside_current_slice_grid() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("free")
    app._free_selection_ab = (0.9, 3.0)
    app.a_index_var = _Var(0)
    app.b_index_var = _Var(0)
    app.a_values = np.asarray([-0.75, -0.25, 0.25], dtype=float)
    app.b_values = np.asarray([2.25, 4.0], dtype=float)
    app.payload = {
        "a_values": [-0.75, -0.25, 0.25],
        "b_values": [2.25, 4.0],
        "points": {
            (0, 0): {"status": "computed", "a": -0.75, "b": 2.25},
        },
    }

    assert app._restore_free_grid_selection_from_coords() is True
    assert app._free_selection_ab == (0.9, 3.0)
    assert app._free_grid_selection_indices() is None
    assert app._has_saved_selected_point() is False
    assert app._ensure_selected_point_exists(metric_name="eta2") is True


def test_set_free_grid_selection_accepts_off_grid_coordinates() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("free")
    app._free_selection_ab = None
    app.a_index_var = _Var(1)
    app.b_index_var = _Var(1)
    app.a_values = np.asarray([-0.75, -0.25, 0.25], dtype=float)
    app.b_values = np.asarray([2.25, 4.0], dtype=float)
    app.payload = {
        "a_values": [-0.75, -0.25, 0.25],
        "b_values": [2.25, 4.0],
        "points": {},
    }

    app._set_free_grid_selection(0.9, 3.0)

    assert app._free_selection_ab == (0.9, 3.0)
    assert app.a_index_var.get() == 1
    assert app.b_index_var.get() == 1


def test_restore_slice_view_state_preserves_unsaved_active_selection() -> None:
    app = object.__new__(PychmpViewApp)
    app.artifact_h5 = Path("/tmp/adaptive.h5")
    app.search_id_var = _Var("search_a")
    app.metric_var = _Var("chi2")
    app.run_target_metric = "chi2"
    app._preferred_initial_metric = None
    app.trials_xmin_var = _Var("")
    app.trials_xmax_var = _Var("")
    app.trials_ymin_var = _Var("")
    app.trials_ymax_var = _Var("")
    app.trials_xscale_var = _Var("linear scale")
    app.trials_yscale_var = _Var("linear scale")
    app.a_index_var = _Var(0)
    app.b_index_var = _Var(0)
    app.navigation_mode_var = _Var("free")
    app._refresh_navigation_control_states = lambda: None
    app.a_values = np.asarray([0.0, 0.3], dtype=float)
    app.b_values = np.asarray([2.4, 2.7], dtype=float)
    app.payload = {
        "selected_slice_key": "euv_171",
        "selected_search_id": "search_a",
        "a_values": [0.0, 0.3],
        "b_values": [2.4, 2.7],
        "points": {
            (0, 0): {"status": "computed", "a": 0.0, "b": 2.4, "metrics": {"chi2": 1.0}},
            (1, 1): {"status": "missing", "a": 0.3, "b": 2.7},
        },
    }
    token = app._slice_state_token("euv_171")
    assert token is not None
    app.slice_display_state = {
        token: {
            "metric": "chi2",
            "a_index": 1,
            "b_index": 1,
            "navigation_mode": "free",
            "trials_by_metric": {},
        }
    }
    app._reset_trials_controls_only = lambda: None
    app._restore_trials_controls_for_metric = lambda _metric: None

    app._restore_slice_view_state()

    assert app.a_index_var.get() == 1
    assert app.b_index_var.get() == 1


def test_active_navigation_mode_selects_live_coordinates() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("active")
    app.run_target_metric = "chi2"
    app._refresh_navigation_control_states = lambda: None
    app._live_navigation_available = lambda: True
    app._active_navigation_mode_available = lambda: True
    app._refresh_signal_active_point = (0.3, 2.7)
    app.a_values = np.asarray([0.0, 0.3], dtype=float)
    app.b_values = np.asarray([2.4, 2.7], dtype=float)
    app.a_index_var = _Var(0)
    app.b_index_var = _Var(0)
    app.metric_var = _Var("chi2")
    app._selected_trial_token = object()
    app.payload = {"points": {}}
    app._selected_slice_key = lambda: "euv_171"
    app._selected_search_id = lambda: "search_a"
    calls: list[str] = []
    app._refresh_selector_values = lambda: calls.append("selectors")

    app._apply_locked_navigation_selection(schedule_slice_reload=False)

    assert app.a_index_var.get() == 1
    assert app.b_index_var.get() == 1
    assert app._selected_trial_token is None
    assert calls == ["selectors"]


def test_active_navigation_mode_is_noop_without_live_coordinates() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("active")
    app.run_target_metric = "chi2"
    app._refresh_navigation_control_states = lambda: None
    app._live_navigation_available = lambda: False
    app._active_navigation_mode_available = lambda: False
    app._refresh_signal_active_point = None
    app.a_values = np.asarray([0.0, 0.3], dtype=float)
    app.b_values = np.asarray([2.4, 2.7], dtype=float)
    app.a_index_var = _Var(0)
    app.b_index_var = _Var(0)
    app.metric_var = _Var("chi2")
    app._selected_trial_token = object()
    app.payload = {"points": {}}
    calls: list[str] = []
    app._refresh_selector_values = lambda: calls.append("selectors")

    app._apply_locked_navigation_selection(schedule_slice_reload=False)

    assert app.a_index_var.get() == 0
    assert app.b_index_var.get() == 0
    assert calls == []


def test_on_navigation_mode_changed_rejects_active_without_live_runner() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("active")
    app.status_var = _Var("")
    app._live_navigation_available = lambda: False
    app._active_navigation_mode_available = lambda: False
    app._active_navigation_mode_available = lambda: False
    app._best_navigation_available = lambda: True
    app._coerce_navigation_mode_to_availability = PychmpViewApp._coerce_navigation_mode_to_availability.__get__(app)
    app._navigation_mode_radios = {}
    app._refresh_navigation_control_states = lambda: None
    app._apply_locked_navigation_selection = lambda **_kwargs: False
    refresh_calls = {"count": 0}
    app._refresh_all = lambda **_kwargs: refresh_calls.__setitem__("count", refresh_calls["count"] + 1)

    app._on_navigation_mode_changed()

    assert app.navigation_mode_var.get() == "free"
    assert "in-progress" in str(app.status_var.get()).lower()
    assert refresh_calls["count"] == 1


def test_live_navigation_available_uses_single_active_point() -> None:
    app = object.__new__(PychmpViewApp)
    app._live_runner_detected = lambda: True
    app._refresh_signal_active_point = (0.3, 2.7)
    app._refresh_signal_pending_points = []
    app.artifact_h5 = None
    app.a_values = np.asarray([0.0, 0.3], dtype=float)
    app.b_values = np.asarray([2.4, 2.7], dtype=float)
    app._sync_live_trial_state_from_artifact = lambda: None

    assert app._live_navigation_available() is True
    assert app._refresh_signal_active_point == (0.3, 2.7)


def test_live_navigation_available_uses_first_of_multiple_pending_points() -> None:
    app = object.__new__(PychmpViewApp)
    app._live_runner_detected = lambda: True
    app._refresh_signal_active_point = (0.0, 2.4)
    app._refresh_signal_pending_points = []
    app._refresh_signal_slice_key = "euv_193"
    app.artifact_h5 = None
    app._sync_live_trial_state_from_artifact = lambda: None

    assert app._live_navigation_available() is True
    assert app._refresh_signal_active_point == (0.0, 2.4)


def test_default_navigation_mode_is_active_for_live_runner_with_active_point() -> None:
    app = object.__new__(PychmpViewApp)
    app._stale_active_search = lambda: False
    app._live_runner_detected = lambda: True
    app._heartbeat_activity_present = lambda: True
    app._live_scan_slice_key = lambda: "euv_193"
    app._resolve_live_active_point = lambda: (0.0, 2.4)
    app._refresh_signal_pending_points = []

    assert app._default_navigation_mode() == "active"


def test_apply_locked_navigation_selection_switches_to_live_slice_in_active_mode() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("active")
    app.slice_key_var = _Var("euv_171")
    app.search_id_var = _Var("search_old")
    app.metric_var = _Var("chi2")
    app.a_index_var = _Var(0)
    app.b_index_var = _Var(0)
    app.run_target_metric = "eta2"
    app.payload = {"selected_slice_key": "euv_171", "points": {}}
    app._refresh_signal_slice_key = "euv_193"
    app._refresh_signal_search_id = "search_live"
    app._live_navigation_available = lambda: True
    app._active_navigation_mode_available = lambda: True
    app._run_target_metric_for_selection = lambda: "eta2"
    app._resolve_active_session_slice_search = lambda: ("euv_193", "search_live")
    app._search_id_in_artifact = lambda *_args, **_kwargs: True
    scheduled: list[str] = []
    app._schedule_payload_reload = lambda *, status_text=None: scheduled.append(str(status_text or ""))

    assert app._apply_locked_navigation_selection() is True
    assert app.slice_key_var.get() == "euv_193"
    assert scheduled == ["Loading active slice..."]


def test_sync_live_trial_state_from_artifact_clears_state_when_artifact_has_no_live_point() -> None:
    app = object.__new__(PychmpViewApp)
    app.artifact_h5 = Path("/tmp/adaptive.h5")
    app.payload = {"selected_slice_key": "euv_193", "selected_search_id": "search_b"}
    app.available_searches = []
    app.run_target_metric = "eta2"
    app._refresh_signal_active_point = (-0.6, 2.1)
    app._refresh_signal_live_trials = {"a": -0.6, "b": 2.1, "metric_name": "eta2"}
    app._refresh_signal_point_id = "p000002"

    import pychmp.viewer as viewer_module

    viewer_module.load_grid_point_live_state = lambda *_args, **_kwargs: None

    app._sync_live_trial_state_from_artifact()

    assert app._refresh_signal_active_point is None
    assert app._refresh_signal_live_trials is None


def test_on_navigation_mode_changed_free_preserves_best_selection() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("free")
    app.status_var = _Var("")
    app.payload = {
        "a_values": [-0.75, -0.25, 0.25],
        "b_values": [2.5, 3.0, 3.5, 4.0],
        "points": {
            (1, 2): {
                "status": "computed",
                "a": -0.25,
                "b": 3.5,
                "metrics": {"eta2": 0.9},
                "fit_q0_trials": (1e-4, 1e-3),
                "fit_eta2_trials": (0.9, 0.5),
            },
            (0, 0): {
                "status": "computed",
                "a": -0.75,
                "b": 2.5,
                "metrics": {"eta2": 0.4},
                "fit_q0_trials": (1e-4,),
                "fit_eta2_trials": (0.4,),
            },
        },
    }
    app.a_values = np.asarray([-0.75, -0.25, 0.25], dtype=float)
    app.b_values = np.asarray([2.5, 3.0, 3.5, 4.0], dtype=float)
    app.run_target_metric = "eta2"
    app.metric_var = _Var("eta2")
    app.a_index_var = _Var(1)
    app.b_index_var = _Var(2)
    app._free_selection_ab = (-0.75, 2.5)
    app._live_navigation_available = lambda: False
    app._active_navigation_mode_available = lambda: False
    app._best_navigation_available = lambda: True
    app._coerce_navigation_mode_to_availability = lambda: None
    app._navigation_mode_radios = {}
    app._refresh_navigation_control_states = lambda: None
    app._apply_locked_navigation_selection = lambda **_kwargs: False
    app._selected_slice_key = lambda: "mw_2874"
    app._selected_search_id = lambda: "search_a"
    app._refresh_selector_values = lambda: None
    app._applying_navigation_selection = False
    refresh_calls = {"count": 0}
    app._refresh_all = lambda **_kwargs: refresh_calls.__setitem__("count", refresh_calls["count"] + 1)

    app._on_navigation_mode_changed()

    assert app.navigation_mode_var.get() == "free"
    assert app.a_index_var.get() == 1
    assert app.b_index_var.get() == 2
    assert app._free_selection_ab == (-0.25, 3.5)
    assert refresh_calls["count"] == 1


def test_on_navigation_mode_changed_best_refreshes_immediately() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("best")
    app.status_var = _Var("")
    app.payload = {
        "a_values": [0.0, 0.3],
        "b_values": [2.4, 2.7],
        "points": {
            (0, 0): {
                "status": "computed",
                "a": 0.0,
                "b": 2.4,
                "metrics": {"eta2": 1.0, "chi2": 9.0},
                "diagnostics": {"eta2": 1.0, "chi2": 9.0},
            },
            (1, 1): {
                "status": "computed",
                "a": 0.3,
                "b": 2.7,
                "metrics": {"eta2": 2.0, "chi2": 2.0},
                "diagnostics": {"eta2": 2.0, "chi2": 2.0},
            },
        },
    }
    app.a_values = np.asarray([0.0, 0.3], dtype=float)
    app.b_values = np.asarray([2.4, 2.7], dtype=float)
    app.run_target_metric = "eta2"
    app.metric_var = _Var("chi2")
    app.a_index_var = _Var(1)
    app.b_index_var = _Var(1)
    app._live_navigation_available = lambda: False
    app._active_navigation_mode_available = lambda: False
    app._best_navigation_available = lambda: True
    app._coerce_navigation_mode_to_availability = lambda: None
    app._navigation_mode_radios = {}
    app._refresh_navigation_control_states = lambda: None
    app._selected_slice_key = lambda: "euv_171"
    app._selected_search_id = lambda: "search_a"
    app._refresh_selector_values = lambda: None
    app._applying_navigation_selection = False
    refresh_calls = {"count": 0}
    app._refresh_all = lambda **_kwargs: refresh_calls.__setitem__("count", refresh_calls["count"] + 1)

    app._on_navigation_mode_changed()

    assert app.navigation_mode_var.get() == "best"
    assert app.a_index_var.get() == 1
    assert app.b_index_var.get() == 1
    assert refresh_calls["count"] == 1


def test_slice_selection_locked_in_active_and_best_modes() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("free")
    assert app._slice_and_search_selection_locked() is False
    app.navigation_mode_var = _Var("best")
    assert app._slice_and_search_selection_locked() is True
    app.navigation_mode_var = _Var("active")
    assert app._slice_and_search_selection_locked() is True


def test_best_point_index_uses_visible_slice_payload_only() -> None:
    app = object.__new__(PychmpViewApp)
    app.run_target_metric = "eta2"
    app.payload = {
        "a_values": [0.0, 0.3],
        "b_values": [2.4, 2.7],
        "points": {
            (0, 0): {"status": "computed", "a": 0.0, "b": 2.4, "metrics": {"eta2": 5.0}, "diagnostics": {"eta2": 5.0}},
            (1, 1): {"status": "computed", "a": 0.3, "b": 2.7, "metrics": {"eta2": 1.0}, "diagnostics": {"eta2": 1.0}},
        },
    }
    assert app._best_point_index("eta2") == (1, 1)


def test_on_slice_changed_blocked_in_best_mode() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("best")
    app.available_slices = [{"key": "euv_131", "label": "EUV: 131 A"}, {"key": "euv_193", "label": "EUV: 193 A"}]
    app.payload = {"selected_slice_key": "euv_131"}
    app._selected_slice_key = lambda: "euv_131"
    app.slice_key_var = _Var("euv_131")
    app.search_id_var = _Var("search_a")
    app._last_rendered_metric = "eta2"
    app._capture_current_slice_view_state = lambda *_args, **_kwargs: None

    class _SliceMenu:
        def current(self) -> int:
            return 1

    app.slice_menu = _SliceMenu()
    scheduled: list[str] = []

    def _schedule(**kwargs) -> None:
        scheduled.append(str(kwargs.get("status_text", "")))

    app._schedule_payload_reload = _schedule
    app._on_slice_changed()

    assert scheduled == []
    assert app.slice_key_var.get() == "euv_131"


def test_apply_locked_navigation_best_follows_slice_local_best() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("best")
    app.metric_var = _Var("chi2")
    app.run_target_metric = "eta2"
    app.a_index_var = _Var(0)
    app.b_index_var = _Var(0)
    app._applying_navigation_selection = False
    app._selected_trial_token = "stale"
    app._refresh_selector_values = lambda: None
    app.payload = {
        "a_values": [0.0, 0.3],
        "b_values": [2.4, 2.7],
        "points": {
            (0, 0): {
                "status": "computed",
                "a": 0.0,
                "b": 2.4,
                "metrics": {"eta2": 9.0, "chi2": 9.0},
                "diagnostics": {"eta2": 9.0, "chi2": 9.0},
            },
            (1, 1): {
                "status": "computed",
                "a": 0.3,
                "b": 2.7,
                "metrics": {"eta2": 2.0, "chi2": 2.0},
                "diagnostics": {"eta2": 2.0, "chi2": 2.0},
            },
        },
    }
    app._best_navigation_available = lambda: True
    app._live_navigation_available = lambda: False
    app._active_navigation_mode_available = lambda: False

    assert app._apply_locked_navigation_selection(schedule_slice_reload=False) is False
    assert app.a_index_var.get() == 1
    assert app.b_index_var.get() == 1
    assert app.metric_var.get() == "chi2"
    assert app._selected_trial_token is None


def test_on_navigation_mode_changed_selects_live_coordinates() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("active")
    app.run_target_metric = "chi2"
    app._refresh_signal_active_point = (0.3, 2.7)
    app.a_values = np.asarray([0.0, 0.3], dtype=float)
    app.b_values = np.asarray([2.4, 2.7], dtype=float)
    app.a_index_var = _Var(0)
    app.b_index_var = _Var(0)
    app.metric_var = _Var("chi2")
    app.payload = {"points": {}}
    app._selected_slice_key = lambda: "euv_171"
    app._selected_search_id = lambda: "search_a"
    app._live_navigation_available = lambda: True
    app._active_navigation_mode_available = lambda: True
    app._best_navigation_available = lambda: True
    app._coerce_navigation_mode_to_availability = lambda: None
    app._navigation_mode_radios = {}
    app._refresh_navigation_control_states = lambda: None
    app._refresh_selector_values = lambda: None
    app._refresh_all = lambda **_kwargs: None

    app._on_navigation_mode_changed()

    assert app.navigation_mode_var.get() == "active"
    assert app.a_index_var.get() == 1
    assert app.b_index_var.get() == 1


def test_free_mode_allows_active_star_heatmap_click() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("free")
    app.ax_heatmap = object()
    app.a_values = np.asarray([0.0, 0.3, 0.6, 0.9], dtype=float)
    app.b_values = np.asarray([2.4, 2.7, 3.0, 3.3], dtype=float)
    app.display_model = {
        "records": [],
        "a_min": -0.15,
        "a_max": 0.75,
        "b_min": 2.25,
        "b_max": 3.45,
    }
    app.payload = {
        "selected_slice_key": "mw_6p929688ghz",
        "points": {
            (0, 0): {"status": "computed", "a": 0.0, "b": 2.4},
        },
    }
    app._refresh_signal_active_point = (0.9, 3.0)
    app._refresh_signal_slice_key = "mw_6p929688ghz"
    app.run_target_metric = "eta2"
    app.a_index_var = _Var(0)
    app.b_index_var = _Var(0)
    calls: list[str] = []
    app._refresh_selector_values = lambda: calls.append("selectors")
    app._refresh_action_states = lambda: calls.append("actions")
    app._refresh_all = lambda: calls.append("refresh")

    app._on_canvas_click(_EventStub(inaxes=app.ax_heatmap, xdata=0.9, ydata=3.0))

    assert app.a_index_var.get() == 3
    assert app.b_index_var.get() == 2
    assert app._selected_trial_token is None
    assert calls == ["selectors", "actions", "refresh"]


def test_free_mode_selects_clicked_coordinates_without_patch_record() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("free")
    app.ax_heatmap = object()
    app.a_values = np.asarray([0.0, 0.3, 0.6, 0.9], dtype=float)
    app.b_values = np.asarray([2.4, 2.7, 3.0, 3.3], dtype=float)
    app.display_model = {
        "records": [],
        "a_min": -0.15,
        "a_max": 0.75,
        "b_min": 2.25,
        "b_max": 3.45,
    }
    app.payload = {
        "selected_slice_key": "mw_6p929688ghz",
        "a_values": [0.0, 0.3, 0.6, 0.9],
        "b_values": [2.4, 2.7, 3.0, 3.3],
        "points": {
            (0, 0): {"status": "computed", "a": 0.0, "b": 2.4},
        },
    }
    app._refresh_signal_active_point = (0.9, 3.0)
    app._refresh_signal_slice_key = "mw_6p929688ghz"
    app.run_target_metric = "eta2"
    app.a_index_var = _Var(0)
    app.b_index_var = _Var(0)
    calls: list[str] = []
    app._refresh_selector_values = lambda: calls.append("selectors")
    app._refresh_action_states = lambda: calls.append("actions")
    app._refresh_all = lambda: calls.append("refresh")

    app._on_canvas_click(_EventStub(inaxes=app.ax_heatmap, xdata=0.6, ydata=2.7))

    assert app._free_selection_ab == (0.6, 2.7)
    assert app.a_index_var.get() == 2
    assert app.b_index_var.get() == 1
    assert app._selected_trial_token is None
    assert calls == ["selectors", "actions", "refresh"]


def test_free_mode_selects_off_grid_click_coordinates() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("free")
    app.ax_heatmap = object()
    app.a_values = np.asarray([0.0, 0.3, 0.6], dtype=float)
    app.b_values = np.asarray([2.4, 2.7, 3.0], dtype=float)
    app.display_model = {
        "records": [],
        "a_min": -0.15,
        "a_max": 0.75,
        "b_min": 2.25,
        "b_max": 3.15,
    }
    app.payload = {
        "selected_slice_key": "mw_6p929688ghz",
        "a_values": [0.0, 0.3, 0.6],
        "b_values": [2.4, 2.7, 3.0],
    }
    app._refresh_signal_active_point = (0.6, 3.0)
    app._refresh_signal_live_trials = {
        "slice_key": "mw_6p929688ghz",
        "metric_name": "eta2",
        "q0_trials": [1.0e-6],
        "metric_trials": [0.5],
    }
    app.a_index_var = _Var(0)
    app.b_index_var = _Var(0)
    calls: list[str] = []
    app._refresh_selector_values = lambda: calls.append("selectors")
    app._refresh_action_states = lambda: calls.append("actions")
    app._refresh_all = lambda: calls.append("refresh")

    app._on_canvas_click(_EventStub(inaxes=app.ax_heatmap, xdata=0.9, ydata=3.2))

    assert app._free_selection_ab == (0.9, 3.2)
    assert app._free_grid_selection_indices() is None
    assert app._selected_trial_token is None
    assert calls == ["selectors", "actions", "refresh"]


def test_point_indices_for_coordinates_matches_grid_values(tmp_path: Path) -> None:
    app = object.__new__(PychmpViewApp)
    app.a_values = np.asarray([0.0, 0.3, 0.6], dtype=float)
    app.b_values = np.asarray([2.1, 2.4, 2.7], dtype=float)

    assert app._point_indices_for_coordinates(0.3, 2.4) == (1, 1)
    assert app._point_indices_for_coordinates(1.0, 2.4) is None


def test_live_trial_state_is_available_when_selection_differs_from_active_point() -> None:
    app = object.__new__(PychmpViewApp)
    app._refresh_signal_active_point = (0.3, 2.7)
    app._refresh_signal_live_trials = {
        "metric_name": "eta2",
        "q0_trials": [1.0e-5, 1.0e-4],
        "metric_trials": [0.8, 0.7],
        "active_trial_index": 1,
        "active_trial_q0": 1.0e-4,
    }
    app.a_values = np.asarray([0.0, 0.3], dtype=float)
    app.b_values = np.asarray([2.4, 2.7], dtype=float)
    app.a_index_var = _Var(0)
    app.b_index_var = _Var(0)

    live = app._live_trial_state()

    assert live is not None
    assert live["metric_name"] == "eta2"
    assert live["a_index"] == 1
    assert live["b_index"] == 1


def test_live_trial_state_is_available_for_unsaved_active_point() -> None:
    app = object.__new__(PychmpViewApp)
    app._refresh_signal_active_point = (0.0, 2.4)
    app._refresh_signal_live_trials = {
        "metric_name": "eta2",
        "q0_trials": [1.0e-5],
        "metric_trials": [0.8],
        "active_trial_index": 1,
        "active_trial_q0": 1.0e-5,
    }
    # Active point is not yet represented in the saved grid coordinates.
    app.a_values = np.asarray([0.3], dtype=float)
    app.b_values = np.asarray([2.7], dtype=float)
    app.a_index_var = _Var(0)
    app.b_index_var = _Var(0)

    live = app._live_trial_state()

    assert live is not None
    assert live["metric_name"] == "eta2"
    assert live["active_a"] == pytest.approx(0.0)
    assert live["active_b"] == pytest.approx(2.4)
    assert "a_index" not in live
    assert "b_index" not in live


def test_should_force_live_trials_for_unsaved_active_point_on_selected_slice() -> None:
    app = object.__new__(PychmpViewApp)
    app.payload = {"selected_slice_key": "mw_6p929688ghz", "points": {}}
    app.a_values = np.asarray([0.6], dtype=float)
    app.b_values = np.asarray([2.7], dtype=float)
    app.navigation_mode_var = _Var("active")
    app._refresh_signal_active_point = (0.6, 2.7)

    assert app._should_force_live_trials(
        {
            "slice_key": "mw_6p929688ghz",
            "active_a": 0.6,
            "active_b": 2.7,
        }
    ) is True


def test_should_not_force_live_trials_in_free_mode_during_live_scan() -> None:
    app = object.__new__(PychmpViewApp)
    app.payload = {
        "selected_slice_key": "euv_131",
        "a_values": [-0.3, 0.0, 0.3],
        "b_values": [2.1, 2.4, 2.7, 3.0],
        "points": {
            (1, 1): {
                "a": 0.0,
                "b": 2.4,
                "status": "computed",
                "success": True,
                "fit_q0_trials": [1.0e-5, 1.0e-4, 1.0e-3, 6.18e-6],
                "fit_metric_trials": [1.0, 2.0, 3.0, 4.0],
            }
        },
    }
    app.navigation_mode_var = _Var("free")
    app.metric_var = _Var("eta2")
    app.a_values = np.asarray([-0.3, 0.0, 0.3], dtype=float)
    app.b_values = np.asarray([2.1, 2.4, 2.7, 3.0], dtype=float)
    app.a_index_var = _Var(1)
    app.b_index_var = _Var(1)
    app._refresh_signal_active_point = (-0.9, 2.1)
    app._selected_search_id = lambda: "search_abc"
    app._live_trial_series_from_state = lambda _live_state: (
        np.asarray([1.0e-5, 2.0e-5, 3.0e-5], dtype=float),
        np.asarray([1.0, 2.0, 3.0], dtype=float),
        "eta2",
        None,
    )

    live_state = {
        "slice_key": "euv_131",
        "search_id": None,
        "active_a": -0.9,
        "active_b": 2.1,
        "metric_name": "eta2",
        "q0_trials": [1.0e-5, 2.0e-5, 3.0e-5],
        "metric_trials": [1.0, 2.0, 3.0],
        "active_trial_q0": 3.0e-5,
    }

    assert app._should_force_live_trials(live_state) is False
    assert app._should_use_live_trials(live_state) is False


def test_should_not_use_live_trials_in_best_mode_while_scan_runs_elsewhere() -> None:
    app = object.__new__(PychmpViewApp)
    app.payload = {
        "selected_slice_key": "mw_2p873584ghz",
        "selected_search_id": "search_abc",
        "a_values": [0.0, 0.3],
        "b_values": [2.4, 3.0],
        "points": {
            (0, 0): {
                "a": 0.0,
                "b": 2.4,
                "status": "computed",
                "success": True,
                "fit_q0_trials": [1.0e-5, 2.0e-5],
                "fit_metric_trials": [0.5, 0.4],
                "fit_eta2_trials": [0.5, 0.4],
                "metrics": {"eta2": 0.4},
                "diagnostics": {"eta2": 0.4},
            }
        },
    }
    app.navigation_mode_var = _Var("best")
    app.metric_var = _Var("eta2")
    app.run_target_metric = "eta2"
    app.a_values = np.asarray([0.0, 0.3], dtype=float)
    app.b_values = np.asarray([2.4, 3.0], dtype=float)
    app.a_index_var = _Var(0)
    app.b_index_var = _Var(0)
    app._refresh_signal_active_point = (0.0, 3.0)
    app._refresh_signal_slice_key = "mw_2p873584ghz"
    app._live_runner_detected = lambda: True
    app._refresh_signal_is_fresh = lambda: True
    app._selected_search_id = lambda: "search_abc"
    app._live_trial_series_from_state = lambda _live_state: (
        np.asarray([1.0e-5, 2.0e-5, 3.0e-5], dtype=float),
        np.asarray([0.9, 0.8, 0.7], dtype=float),
        "eta2",
        None,
    )

    live_state = {
        "slice_key": "mw_2p873584ghz",
        "search_id": "search_abc",
        "active_a": 0.0,
        "active_b": 3.0,
        "a_index": 0,
        "b_index": 1,
        "metric_name": "eta2",
        "fit_q0_trials": [1.0e-5, 2.0e-5, 3.0e-5],
        "fit_metric_trials": [0.9, 0.8, 0.7],
        "active_trial_q0": 3.0e-5,
    }

    assert app._should_force_live_trials(live_state) is False
    assert app._should_use_live_trials(live_state) is False
    assert app._live_trials_context_available(live_state) is False


def test_should_not_force_live_trials_when_active_point_saved_without_streaming() -> None:
    app = object.__new__(PychmpViewApp)
    app.payload = {
        "selected_slice_key": "euv_131",
        "a_values": [0.3],
        "b_values": [2.4],
        "points": {
            (0, 0): {
                "a": 0.3,
                "b": 2.4,
                "status": "computed",
                "fit_q0_trials": [1e-5, 2e-5],
                "fit_eta2_trials": [0.1, 0.2],
            }
        },
    }
    app.a_values = np.asarray([0.3], dtype=float)
    app.b_values = np.asarray([2.4], dtype=float)
    app.navigation_mode_var = _Var("active")
    app._refresh_signal_active_point = (0.3, 2.4)
    app.run_target_metric = "eta2"
    app.metric_var = _Var("eta2")
    app._live_trial_series_from_state = lambda _live_state: (
        np.asarray([], dtype=float),
        np.asarray([], dtype=float),
        "eta2",
        None,
    )

    assert app._active_point_is_saved_in_artifact() is True
    assert app._should_force_live_trials(
        {
            "slice_key": "euv_131",
            "search_id": "search_abc",
            "active_a": 0.3,
            "active_b": 2.4,
            "a_index": 0,
            "b_index": 0,
        }
    ) is False


def test_should_force_live_trials_when_active_point_streams_trials() -> None:
    app = object.__new__(PychmpViewApp)
    app.payload = {
        "selected_slice_key": "euv_131",
        "a_values": [0.3],
        "b_values": [2.4],
        "points": {
            (0, 0): {
                "a": 0.3,
                "b": 2.4,
                "status": "computed",
            }
        },
    }
    app.a_values = np.asarray([0.3], dtype=float)
    app.b_values = np.asarray([2.4], dtype=float)
    app.navigation_mode_var = _Var("active")
    app._refresh_signal_active_point = (0.3, 2.4)
    app.run_target_metric = "eta2"
    app.metric_var = _Var("eta2")
    app._selected_search_id = lambda: "search_abc"
    app._live_trial_series_from_state = lambda _live_state: (
        np.asarray([1e-5], dtype=float),
        np.asarray([0.1], dtype=float),
        "eta2",
        None,
    )

    assert app._should_force_live_trials(
        {
            "slice_key": "euv_131",
            "search_id": "search_abc",
            "active_a": 0.3,
            "active_b": 2.4,
            "a_index": 0,
            "b_index": 0,
        }
    ) is True


def test_heatmap_plot_limits_ignore_live_active_point_outside_grid() -> None:
    """Axis limits follow grid/search extents, not transient live-runner coordinates."""
    from pychmp.ab_scan_artifacts import grid_extents_from_ab_values

    app = object.__new__(PychmpViewApp)
    app.shared_heatmap_axes_var = _Var(False)
    app._shared_heatmap_extents = None
    app.payload = {"selected_slice_key": "mw_6p929688ghz"}
    app.display_model = {
        "records": [],
        "a_min": -0.15,
        "a_max": 0.45,
        "b_min": 2.25,
        "b_max": 3.45,
    }
    a_values = np.asarray([-0.15, 0.45], dtype=float)
    b_values = np.asarray([2.25, 3.45], dtype=float)
    app.a_values = a_values
    app.b_values = b_values
    app._refresh_signal_slice_key = "mw_6p929688ghz"
    app._refresh_signal_pending_points = []
    app._refresh_signal_active_point = (0.6, 2.7)
    app.run_target_metric = "eta2"
    expected = grid_extents_from_ab_values(a_values, b_values)

    a_min, a_max, b_min, b_max = app._heatmap_plot_limits()

    assert a_min == pytest.approx(expected["a_min"])
    assert a_max == pytest.approx(expected["a_max"])
    assert b_min == pytest.approx(expected["b_min"])
    assert b_max == pytest.approx(expected["b_max"])


def test_active_star_scoped_to_selected_slice_only() -> None:
    app = object.__new__(PychmpViewApp)
    app.payload = {"selected_slice_key": "euv_171", "selected_search_id": "search_a"}
    app._refresh_signal_slice_key = "mw_6p929688ghz"
    app._refresh_signal_active_point = (0.6, 2.7)
    app._refresh_signal_live_trials = {"search_id": "search_a"}
    app.run_target_metric = "eta2"

    assert app._active_point_scoped_to_selection() is None


def test_active_star_hidden_for_completed_search_on_live_slice() -> None:
    app = object.__new__(PychmpViewApp)
    app.payload = {
        "selected_slice_key": "mw_2p873584ghz",
        "selected_search_id": "search_legacy",
        "selected_search": {
            "status": "complete",
            "active": True,
            "lifecycle": {"active": True, "completed_at": "2026-05-28T12:00:00Z"},
        },
    }
    app.available_searches = [
        {
            "search_id": "search_legacy",
            "status": "complete",
            "in_progress": False,
            "slice_key": "mw_2p873584ghz",
        },
        {
            "search_id": "search_live",
            "status": "running",
            "in_progress": True,
            "slice_key": "mw_2p873584ghz",
        },
    ]
    app._refresh_signal_slice_key = "mw_2p873584ghz"
    app._refresh_signal_active_point = (-0.9, 2.4)
    app._refresh_signal_live_trials = {
        "slice_key": "mw_2p873584ghz",
        "search_id": None,
        "metric_name": "eta2",
    }
    app._live_runner_detected = lambda: True
    app.run_target_metric = "eta2"

    assert app._active_point_scoped_to_selection() is None


def test_active_star_shown_for_live_search_in_active_mode_with_stale_complete_status() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("active")
    app.payload = {
        "selected_slice_key": "mw_2p873584ghz",
        "selected_search_id": "search_live",
        "selected_search": {
            "status": "complete",
            "active": True,
            "lifecycle": {"active": True, "completed_at": "2026-05-28T12:00:00Z"},
        },
    }
    app.available_searches = [
        {
            "search_id": "search_legacy",
            "status": "complete",
            "in_progress": False,
            "active": True,
            "slice_key": "mw_2p873584ghz",
        },
        {
            "search_id": "search_live",
            "status": "complete",
            "in_progress": True,
            "active": True,
            "slice_key": "mw_2p873584ghz",
        },
    ]
    app._refresh_signal_slice_key = "mw_2p873584ghz"
    app._refresh_signal_active_point = (-0.9, 2.7)
    app._refresh_signal_live_trials = {
        "slice_key": "mw_2p873584ghz",
        "search_id": None,
        "metric_name": "eta2",
    }
    app._live_runner_detected = lambda: True
    app._refresh_signal_is_fresh = lambda: True
    app.run_target_metric = "eta2"

    scoped = app._active_point_scoped_to_selection()

    assert scoped == (-0.9, 2.7)


def test_active_star_shown_for_live_search_in_free_mode_with_stale_complete_status() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("free")
    app.payload = {
        "selected_slice_key": "mw_2p873584ghz",
        "selected_search_id": "search_live",
        "selected_search": {
            "status": "complete",
            "active": True,
            "lifecycle": {"active": True, "completed_at": "2026-05-28T12:00:00Z"},
        },
    }
    app.available_searches = [
        {
            "search_id": "search_legacy",
            "status": "complete",
            "in_progress": False,
            "active": True,
            "slice_key": "mw_2p873584ghz",
        },
        {
            "search_id": "search_live",
            "status": "complete",
            "in_progress": True,
            "active": True,
            "slice_key": "mw_2p873584ghz",
        },
    ]
    app._refresh_signal_slice_key = "mw_2p873584ghz"
    app._refresh_signal_active_point = (-0.9, 2.7)
    app._refresh_signal_live_trials = {
        "slice_key": "mw_2p873584ghz",
        "search_id": None,
        "metric_name": "eta2",
    }
    app._live_runner_detected = lambda: True
    app._refresh_signal_is_fresh = lambda: True
    app.run_target_metric = "eta2"

    assert app._active_point_scoped_to_selection() == (-0.9, 2.7)


def test_catalog_live_search_id_does_not_prefer_selected_when_multiple_active() -> None:
    app = object.__new__(PychmpViewApp)
    app._selected_search_id = lambda: "search_legacy"
    app._live_runner_detected = lambda: True
    app._refresh_signal_is_fresh = lambda: True
    app.available_searches = [
        {"search_id": "search_legacy", "active": True, "slice_key": "mw_2p873584ghz"},
        {"search_id": "search_live", "active": True, "in_progress": True, "slice_key": "mw_2p873584ghz"},
    ]

    assert app._catalog_live_search_id_for_slice("mw_2p873584ghz") == "search_live"


def test_heatmap_plot_limits_keep_shared_extents_without_live_padding() -> None:
    app = object.__new__(PychmpViewApp)
    app.shared_heatmap_axes_var = _Var(True)
    app._shared_heatmap_extents = {
        "a_min": -0.5,
        "a_max": 0.5,
        "b_min": 2.0,
        "b_max": 3.2,
    }
    app.display_model = {
        "a_min": -0.1,
        "a_max": 0.2,
        "b_min": 2.4,
        "b_max": 2.7,
    }
    app.payload = {"selected_slice_key": "mw_2p873584ghz", "selected_search_id": "search_live"}
    app._refresh_signal_slice_key = "mw_2p873584ghz"
    app._refresh_signal_active_point = (-0.9, 3.0)
    app.run_target_metric = "eta2"
    app.available_searches = [
        {"search_id": "search_live", "in_progress": True, "slice_key": "mw_2p873584ghz"},
    ]
    app._live_runner_detected = lambda: True
    app._refresh_signal_is_fresh = lambda: True

    a_min, a_max, b_min, b_max = app._heatmap_plot_limits()

    assert a_min == pytest.approx(-0.5)
    assert a_max == pytest.approx(0.5)
    assert b_min == pytest.approx(2.0)
    assert b_max == pytest.approx(3.2)


def test_should_force_live_trials_in_free_mode_when_selection_matches_active_point() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("free")
    app.payload = {
        "selected_slice_key": "mw_2p873584ghz",
        "selected_search_id": "search_live",
        "selected_search": {"status": "running", "in_progress": True},
        "points": {},
    }
    app.a_values = np.asarray([-0.9, -0.6], dtype=float)
    app.b_values = np.asarray([2.4, 2.7], dtype=float)
    app.a_index_var = _Var(1)
    app.b_index_var = _Var(1)
    app.metric_var = _Var("eta2")
    app._refresh_signal_active_point = (-0.6, 2.7)
    app._refresh_signal_slice_key = "mw_2p873584ghz"
    app._live_runner_detected = lambda: True
    app._refresh_signal_is_fresh = lambda: True
    app._live_trial_series_from_state = lambda _live_state: (
        np.asarray([1.0e-5, 1.0e-4], dtype=float),
        np.asarray([0.5, 0.9], dtype=float),
        "eta2",
        None,
    )

    live_state = app._live_trial_state()

    assert live_state is not None
    assert app._should_force_live_trials(live_state) is True
    assert app._should_use_live_trials(live_state) is True


def test_free_mode_resolves_unsaved_live_active_point() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("free")
    app.payload = {
        "selected_slice_key": "mw_6p929688ghz",
        "points": {(0, 0): {"status": "computed", "a": 0.0, "b": 2.4}},
    }
    app.a_values = np.asarray([0.0, 0.3, 0.6, 0.9], dtype=float)
    app.b_values = np.asarray([2.4, 2.7, 3.0, 3.3], dtype=float)
    app.a_index_var = _Var(3)
    app.b_index_var = _Var(2)
    app.metric_var = _Var("eta2")
    app._refresh_signal_active_point = (0.9, 3.0)
    app._refresh_signal_slice_key = "mw_6p929688ghz"
    app.run_target_metric = "eta2"

    assert app._resolved_point_index(metric_name="eta2") == (3, 2)


def test_live_trials_context_available_for_force_live_unsaved_active_point() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("active")
    app.payload = {
        "selected_slice_key": "mw_6p929688ghz",
        "selected_search_id": "search_live",
        "points": {},
    }
    app.a_index_var = _Var(1)
    app.b_index_var = _Var(1)
    app.a_values = np.asarray([0.3, 0.6], dtype=float)
    app.b_values = np.asarray([2.4, 2.7, 3.0], dtype=float)
    app._refresh_signal_active_point = (0.6, 2.7)
    app._refresh_signal_slice_key = "mw_6p929688ghz"
    app._refresh_signal_live_trials = {
        "slice_key": "mw_6p929688ghz",
        "search_id": "search_live",
        "metric_name": "eta2",
        "q0_trials": [1.0e-5],
        "metric_trials": [0.5],
    }
    app.run_target_metric = "eta2"

    live_state = app._live_trial_state()

    assert app._should_force_live_trials(live_state) is True
    assert app._live_trials_context_available(live_state) is True


def test_selected_solution_plot_context_uses_live_context_when_force_live() -> None:
    app = object.__new__(PychmpViewApp)
    live_state = {"a_index": 1, "b_index": 1, "metric_name": "eta2"}
    app._live_trial_state = lambda: live_state
    app._has_selected_point = lambda: True
    app._should_force_live_trials = lambda _live_state: True
    app._should_use_live_trials = lambda _live_state: False
    app._live_trials_context_available = lambda _live_state: True
    live_context = {"diagnostics": {"source": "live"}}
    app._live_selected_solution_plot_context = lambda: live_context

    context = app._selected_solution_plot_context()

    assert context is live_context


def test_draw_heatmap_handles_empty_grid_without_crashing() -> None:
    app = object.__new__(PychmpViewApp)
    app.metric_var = _Var("eta2")
    app.payload = {"a_values": [], "b_values": [], "best_q0": np.zeros((0, 0), dtype=float)}
    app.display_model = {
        "records": [],
        "a_min": 0.0,
        "a_max": 1.0,
        "b_min": 0.0,
        "b_max": 1.0,
    }
    app.a_values = np.asarray([], dtype=float)
    app.b_values = np.asarray([], dtype=float)
    app.a_index_var = _Var(0)
    app.b_index_var = _Var(0)
    app.ax_heatmap = _AxisStub()
    app.heatmap_figure = _FigureStub()
    app._reset_heatmap_colorbar = lambda: None
    app._apply_figure_autolayout = lambda _figure: None
    app._refresh_signal_pending_points = []
    app._refresh_signal_active_point = None
    app._navigation_mode = lambda: "free"
    app._best_tied_records = []
    app._heatmap_display_metric = lambda: "eta2"
    app._heatmap_display_model = lambda: app.display_model
    app.a_values = np.asarray([-0.5, 0.5], dtype=float)
    app.b_values = np.asarray([2.0, 3.0], dtype=float)
    app.shared_heatmap_axes_var = _Var(False)
    app._shared_heatmap_extents = None
    app._use_shared_grid_axes = lambda: False
    app._heatmap_plot_limits = lambda _model: (0.0, 1.0, 0.0, 1.0)

    # Should not raise IndexError when no records/grid values exist yet.
    app._draw_heatmap()


def test_draw_trials_handles_empty_grid_without_crashing() -> None:
    app = object.__new__(PychmpViewApp)
    app._live_trial_state = lambda: None
    app._should_use_live_trials = lambda _live_state: False
    app._should_force_live_trials = lambda _live_state: False
    app._has_selected_point = lambda: True
    app._has_saved_selected_point = lambda: True
    app._selected_point = lambda: {
        "a": 0.3,
        "b": 2.7,
        "status": "computed",
        "success": True,
        "q0": np.nan,
        "fit_q0_trials": (),
        "fit_metric_trials": (),
    }
    app._trial_series_for_point = lambda _point: (np.asarray([], dtype=float), np.asarray([], dtype=float), "eta2")
    app._refresh_trial_selector_controls = lambda *_args, **_kwargs: None
    app._selected_trial_index_for_point = lambda *_args, **_kwargs: None
    app._capture_current_slice_view_state = lambda *_args, **_kwargs: None
    app._slice_label = lambda _descriptor: "MW: 2.874 GHz"
    app.metric_var = _Var("eta2")
    app.run_target_metric = "eta2"
    app.trials_xscale_var = _Var("linear scale")
    app.trials_yscale_var = _Var("linear scale")
    app.ax_trials = _AxisStub()
    app.trials_figure = _FigureStub()
    app.trials_canvas = type("_Canvas", (), {"draw_idle": lambda self: None})()
    app._apply_trials_layout_margins = lambda: None
    app._apply_figure_autolayout = lambda _figure: None
    app.navigation_mode_var = _Var("active")
    app.status_var = _Var("")
    app.payload = {"selected_slice": {"display_label": "MW: 2.874 GHz"}}
    app.a_values = np.asarray([], dtype=float)
    app.b_values = np.asarray([], dtype=float)
    app.a_index_var = _Var(0)
    app.b_index_var = _Var(0)

    # Should not raise IndexError when selected point exists but grid arrays are still empty.
    app._draw_trials()
    assert "Selected point:" in app.status_var.get()


def test_draw_trials_uses_unsaved_live_active_point_status_when_same_slice_selected() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("active")
    app._refresh_signal_active_point = (0.6, 2.7)
    app._live_trial_state = lambda: {
        "slice_key": "mw_6p929688ghz",
        "active_a": 0.6,
        "active_b": 2.7,
        "metric_name": "eta2",
        "q0_trials": [1.0e-5, 1.0e-4, 1.0e-3],
        "metric_trials": [0.92, 0.89, 2.92],
        "active_trial_index": 2,
        "active_trial_q0": 1.0e-3,
    }
    app._should_use_live_trials = lambda _live_state: False
    app._should_force_live_trials = lambda _live_state: True
    app._live_slice_matches_selected = lambda _live_state: True
    app._live_search_matches_selected = lambda _live_state: True
    app._live_trial_series_from_state = lambda _live_state: (
        np.asarray([1.0e-5, 1.0e-4, 1.0e-3], dtype=float),
        np.asarray([0.92, 0.89, 2.92], dtype=float),
        "eta2",
        None,
    )
    app._selection_matches_live_active_point = lambda _a, _b: False
    app._best_trial_index_from_metric = lambda _metric_trials: 0
    app._has_selected_point = lambda: True
    app._selected_point = lambda: {
        "a": 0.3,
        "b": 3.3,
        "status": "computed",
        "success": True,
        "q0": 9.0e-7,
        "fit_q0_trials": np.asarray([9.0e-7, 1.0e-6], dtype=float),
        "fit_metric_trials": np.asarray([0.41, 0.42], dtype=float),
    }
    app.metric_var = _Var("eta2")
    app.run_target_metric = "eta2"
    app.trial_index_var = _Var(0)
    app._selected_trial_token = None
    app._refresh_trial_selector_controls = lambda *_args, **_kwargs: None
    app._apply_trials_axis_controls = lambda **_kwargs: None
    app._sync_trials_axis_controls_from_axes = lambda: None
    app._capture_current_slice_view_state = lambda *_args, **_kwargs: None
    app.trials_xscale_var = _Var("linear scale")
    app.trials_yscale_var = _Var("linear scale")
    app.ax_trials = _AxisStub()
    app.trials_figure = _FigureStub()
    app.trials_canvas = type("_Canvas", (), {"draw_idle": lambda self: None})()
    app._apply_trials_layout_margins = lambda: None
    app._apply_figure_autolayout = lambda _figure: None
    app.status_var = _Var("")
    app.payload = {
        "selected_slice": {"display_label": "MW: 6.930 GHz"},
        "selected_slice_key": "mw_6p929688ghz",
        "points": {},
    }
    app.a_values = np.asarray([0.3, 0.6], dtype=float)
    app.b_values = np.asarray([2.7, 3.3], dtype=float)
    app.a_index_var = _Var(0)
    app.b_index_var = _Var(0)

    app._draw_trials()

    assert "Active point: a=0.600, b=2.700" in app.status_var.get()


def test_sync_trials_axis_controls_displays_current_limits_for_autoscale() -> None:
    app = object.__new__(PychmpViewApp)

    class _AxisWithLimits:
        def get_xlim(self):
            return (0.125, 9.5)

        def get_ylim(self):
            return (1.0e-3, 2.0)

        def get_xscale(self):
            return "linear"

        def get_yscale(self):
            return "log"

    app.ax_trials = _AxisWithLimits()
    app.trials_xmin_var = _Var("")
    app.trials_xmax_var = _Var("")
    app.trials_ymin_var = _Var("")
    app.trials_ymax_var = _Var("")
    app.trials_xscale_var = _Var("linear scale")
    app.trials_yscale_var = _Var("linear scale")

    app._sync_trials_axis_controls_from_axes()

    assert app.trials_xmin_var.get() == "0.125"
    assert app.trials_xmax_var.get() == "9.5"
    assert app.trials_ymin_var.get() == "0.001"
    assert app.trials_ymax_var.get() == "2"
    assert app.trials_xscale_var.get() == "linear scale"
    assert app.trials_yscale_var.get() == "log scale"


def test_apply_trials_axis_controls_ignores_displayed_limits_without_manual_flags() -> None:
    app = object.__new__(PychmpViewApp)

    class _AxisRecorder:
        def __init__(self) -> None:
            self.xscale = "linear"
            self.yscale = "linear"
            self.x_limits = []
            self.y_limits = []

        def set_xscale(self, value):
            self.xscale = value

        def set_yscale(self, value):
            self.yscale = value

        def get_xscale(self):
            return self.xscale

        def get_yscale(self):
            return self.yscale

        def set_xlim(self, *args, **kwargs):
            self.x_limits.append((args, kwargs))

        def set_ylim(self, *args, **kwargs):
            self.y_limits.append((args, kwargs))

    app.ax_trials = _AxisRecorder()
    app.trials_xmin_var = _Var("0.125")
    app.trials_xmax_var = _Var("9.5")
    app.trials_ymin_var = _Var("0.001")
    app.trials_ymax_var = _Var("2")
    app._trials_xmin_manual = False
    app._trials_xmax_manual = False
    app._trials_ymin_manual = False
    app._trials_ymax_manual = False
    app.trials_xscale_var = _Var("linear scale")
    app.trials_yscale_var = _Var("linear scale")

    app._apply_trials_axis_controls(q0_trials=np.asarray([1.0, 2.0]), metric_trials=np.asarray([3.0, 4.0]))

    assert app.ax_trials.x_limits == []
    assert app.ax_trials.y_limits == []


def test_sync_trials_axis_controls_updates_explicit_manual_limits() -> None:
    app = object.__new__(PychmpViewApp)

    class _AxisWithLimits:
        def get_xlim(self):
            return (0.125, 9.5)

        def get_ylim(self):
            return (1.0e-3, 2.0)

        def get_xscale(self):
            return "linear"

        def get_yscale(self):
            return "log"

    app.ax_trials = _AxisWithLimits()
    app.trials_xmin_var = _Var("0")
    app.trials_xmax_var = _Var("10")
    app.trials_ymin_var = _Var("0.1")
    app.trials_ymax_var = _Var("2")
    app.trials_xscale_var = _Var("log scale")
    app.trials_yscale_var = _Var("linear scale")

    app._sync_trials_axis_controls_from_axes()

    assert app.trials_xmin_var.get() == "0.125"
    assert app.trials_xmax_var.get() == "9.5"
    assert app.trials_ymin_var.get() == "0.001"
    assert app.trials_ymax_var.get() == "2"
    assert app.trials_xscale_var.get() == "linear scale"
    assert app.trials_yscale_var.get() == "log scale"


def test_trial_slider_changes_selection_in_live_mode_without_saved_point() -> None:
    app = object.__new__(PychmpViewApp)
    app._updating_trial_slider = False
    app._has_selected_point = lambda: False
    app._should_use_live_trials = lambda _live_state: True
    app._live_trial_state = lambda: {
        "a_index": 0,
        "b_index": 1,
        "metric_name": "eta2",
        "q0_trials": [1.0e-5, 1.0e-4, 1.0e-3],
        "metric_trials": [3.0, 2.0, 1.0],
    }
    app._live_trial_series_from_state = lambda _live_state: (
        np.asarray([1.0e-5, 1.0e-4, 1.0e-3], dtype=float),
        np.asarray([3.0, 2.0, 1.0], dtype=float),
        "eta2",
        None,
    )
    app.run_target_metric = "eta2"
    app.metric_var = _Var("eta2")
    app.trial_index_var = _Var(0)
    app._selected_trial_token = None
    calls = {"refresh": 0}

    def _refresh() -> None:
        calls["refresh"] += 1

    app._refresh_all = _refresh

    app._on_trial_slider_changed("1")

    assert app.trial_index_var.get() == 1
    assert app._selected_trial_token == (0, 1, "eta2", -1)
    assert calls["refresh"] == 1


def test_jump_to_best_trial_works_in_live_mode_without_saved_point() -> None:
    app = object.__new__(PychmpViewApp)
    app._has_selected_point = lambda: False
    app._should_use_live_trials = lambda _live_state: True
    app._live_trial_state = lambda: {
        "a_index": 0,
        "b_index": 1,
        "metric_name": "eta2",
        "q0_trials": [1.0e-5, 1.0e-4, 1.0e-3],
        "metric_trials": [2.0, 0.5, 1.0],
    }
    app._live_trial_series_from_state = lambda _live_state: (
        np.asarray([1.0e-5, 1.0e-4, 1.0e-3], dtype=float),
        np.asarray([2.0, 0.5, 1.0], dtype=float),
        "eta2",
        None,
    )
    app.run_target_metric = "eta2"
    app.metric_var = _Var("eta2")
    app.trial_index_var = _Var(0)
    app._selected_trial_token = None
    calls = {"refresh": 0}

    def _refresh() -> None:
        calls["refresh"] += 1

    app._refresh_all = _refresh

    app._jump_to_best_trial()

    assert app.trial_index_var.get() == 1
    assert app._selected_trial_token == (0, 1, "eta2", -1)
    assert calls["refresh"] == 1


def test_should_use_live_trials_when_no_selected_point() -> None:
    app = object.__new__(PychmpViewApp)
    app._has_selected_point = lambda: False
    app.a_index_var = _Var(0)
    app.b_index_var = _Var(0)
    live_state = {"a_index": 0, "b_index": 0, "active_a": 0.0, "active_b": 2.4}

    assert app._should_use_live_trials(live_state) is True


def test_should_not_use_live_trials_for_unsaved_active_point_when_selection_exists() -> None:
    app = object.__new__(PychmpViewApp)
    app._has_selected_point = lambda: True
    app.a_index_var = _Var(0)
    app.b_index_var = _Var(0)
    live_state = {"active_a": 0.0, "active_b": 2.4}

    assert app._should_use_live_trials(live_state) is False


def test_should_use_live_trials_only_when_selection_matches_active_index() -> None:
    app = object.__new__(PychmpViewApp)
    app._has_selected_point = lambda: True
    app.a_index_var = _Var(1)
    app.b_index_var = _Var(2)

    assert app._should_use_live_trials({"a_index": 1, "b_index": 2}) is True
    assert app._should_use_live_trials({"a_index": 0, "b_index": 2}) is False


def test_trial_slider_uses_force_live_mode_for_unsaved_active_point() -> None:
    app = object.__new__(PychmpViewApp)
    app._updating_trial_slider = False
    app.navigation_mode_var = _Var("active")
    app._refresh_signal_active_point = (0.9, 3.0)
    app._has_selected_point = lambda: True
    app._selected_point = lambda: {
        "fit_q0_trials": np.asarray([1.0e-5, 1.0e-4]),
        "fit_metric_trials": np.asarray([1.0, 2.0]),
        "target_metric": "eta2",
    }
    app._trial_series_for_point = lambda _point: (np.asarray([1.0e-5, 1.0e-4]), np.asarray([1.0, 2.0]), "eta2")
    app._live_trial_state = lambda: {
        "slice_key": "mw_6p929688ghz",
        "active_a": 0.9,
        "active_b": 3.0,
        "metric_name": "eta2",
        "q0_trials": [1.0e-6, 1.0e-5, 1.0e-4],
        "metric_trials": [0.5, 0.6, 0.9],
    }
    app._live_trial_series_from_state = lambda _live_state: (
        np.asarray([1.0e-6, 1.0e-5, 1.0e-4], dtype=float),
        np.asarray([0.5, 0.6, 0.9], dtype=float),
        "eta2",
        None,
    )
    app.payload = {"selected_slice_key": "mw_6p929688ghz", "points": {}}
    app.a_values = np.asarray([0.9], dtype=float)
    app.b_values = np.asarray([3.0], dtype=float)
    app.trial_index_var = _Var(2)
    app.run_target_metric = "eta2"
    app.metric_var = _Var("eta2")
    app._selected_trial_token = None
    calls = {"refresh": 0}

    def _refresh() -> None:
        calls["refresh"] += 1

    app._refresh_all = _refresh

    app._on_trial_slider_changed("0")

    assert app.trial_index_var.get() == 0
    assert app._selected_trial_token == (0.9, 3.0, "eta2", -1)
    assert calls["refresh"] == 1


def test_trial_slider_uses_live_mode_for_saved_active_point() -> None:
    app = object.__new__(PychmpViewApp)
    app._updating_trial_slider = False
    app._has_selected_point = lambda: True
    app._should_force_live_trials = lambda _live_state: False
    app._should_use_live_trials = lambda _live_state: True
    app._selected_point = lambda: {
        "fit_q0_trials": np.asarray([1.0e-5, 1.0e-4], dtype=float),
        "fit_metric_trials": np.asarray([1.0, 2.0], dtype=float),
        "target_metric": "eta2",
    }
    app._trial_series_for_point = lambda _point: (np.asarray([1.0e-5, 1.0e-4]), np.asarray([1.0, 2.0]), "eta2")
    app._live_trial_state = lambda: {
        "a_index": 1,
        "b_index": 2,
        "metric_name": "eta2",
        "q0_trials": [1.0e-6, 1.0e-5, 1.0e-4],
        "metric_trials": [0.5, 0.6, 0.9],
    }
    app._live_trial_series_from_state = lambda _live_state: (
        np.asarray([1.0e-6, 1.0e-5, 1.0e-4], dtype=float),
        np.asarray([0.5, 0.6, 0.9], dtype=float),
        "eta2",
        None,
    )
    app.trial_index_var = _Var(2)
    app.run_target_metric = "eta2"
    app.metric_var = _Var("eta2")
    app._selected_trial_token = None
    calls = {"refresh": 0}

    def _refresh() -> None:
        calls["refresh"] += 1

    app._refresh_all = _refresh

    app._on_trial_slider_changed("0")

    assert app.trial_index_var.get() == 0
    assert app._selected_trial_token == (1, 2, "eta2", -1)
    assert calls["refresh"] == 1


def test_selected_solution_plot_context_prefers_live_context_for_saved_active_point() -> None:
    app = object.__new__(PychmpViewApp)
    app._live_trial_state = lambda: {"a_index": 1, "b_index": 2, "metric_name": "eta2"}
    app._has_selected_point = lambda: True
    app._should_force_live_trials = lambda _live_state: False
    app._should_use_live_trials = lambda _live_state: True
    live_context = {"diagnostics": {"source": "live"}}
    app._live_selected_solution_plot_context = lambda: live_context

    context = app._selected_solution_plot_context()

    assert context is live_context


def test_selected_solution_plot_context_includes_point_ab_for_saved_point() -> None:
    app = object.__new__(PychmpViewApp)
    app._live_trial_state = lambda: None
    app._live_trials_context_available = lambda _live_state: False
    app._has_selected_point = lambda: True
    app._selected_point = lambda: {
        "a": 0.0,
        "b": 2.5,
        "fit_q0_trials": np.asarray([1.0e-3, 3.9e-3], dtype=float),
        "fit_metric_trials": np.asarray([0.5, 0.4], dtype=float),
        "fit_eta2_trials": np.asarray([0.5, 0.4], dtype=float),
        "target_metric": "eta2",
        "diagnostics": {"target_metric": "eta2"},
        "raw_modeled_best": np.ones((2, 2), dtype=float),
        "modeled_best": np.ones((2, 2), dtype=float),
        "residual": np.zeros((2, 2), dtype=float),
    }
    app.payload = {
        "diagnostics": {"model_path": "/tmp/model.h5"},
        "observed": np.zeros((2, 2), dtype=float),
        "wcs_header": fits.Header(),
        "selected_slice": {"display_label": "MW: 2.874 GHz"},
        "selected_slice_key": "mw_2p874000ghz",
        "selected_search_id": "search-live",
        "points": {(0, 1): app._selected_point()},
    }
    app._selected_diagnostics = lambda: dict(app.payload["diagnostics"])
    app._diagnostics_with_search_shift = lambda diagnostics: dict(diagnostics)
    app._trial_series_for_point = lambda _point: (
        np.asarray([1.0e-3, 3.9e-3], dtype=float),
        np.asarray([0.5, 0.4], dtype=float),
        "eta2",
    )
    app._selected_trial_index_for_point = lambda *_args, **_kwargs: 1
    app._best_trial_index_from_metric = lambda _metric_trials: 1
    app._metric_history_for_point = lambda _point, _metric_name: np.asarray([], dtype=float)
    app._load_selected_trial_maps = lambda **_kwargs: {
        "raw_modeled_best": np.ones((2, 2), dtype=float),
        "modeled_best": np.ones((2, 2), dtype=float),
        "residual": np.zeros((2, 2), dtype=float),
        "trial_index": 1,
    }
    app._slice_label = lambda _descriptor: "MW: 2.874 GHz"
    app.trials_xmin_var = _Var("")
    app.trials_xmax_var = _Var("")
    app.trials_ymin_var = _Var("")
    app.trials_ymax_var = _Var("")
    app.trials_xscale_var = _Var("linear scale")
    app.trials_yscale_var = _Var("linear scale")
    app._parse_axis_limit = lambda _text: None
    app._display_observation_for_trial = lambda **_kwargs: np.zeros((2, 2), dtype=float)
    app.trial_index_var = _Var(1)
    app.metric_var = _Var("eta2")

    context = app._selected_solution_plot_context()

    assert context is not None
    diagnostics = dict(context["diagnostics"])
    assert diagnostics["a"] == 0.0
    assert diagnostics["b"] == 2.5


def test_trials_canvas_click_uses_live_mode_for_saved_active_point() -> None:
    app = object.__new__(PychmpViewApp)
    app.ax_trials = object()
    app._has_selected_point = lambda: True
    app._should_force_live_trials = lambda _live_state: False
    app._should_use_live_trials = lambda _live_state: True
    app._selected_point = lambda: {
        "fit_q0_trials": np.asarray([1.0e-5, 1.0e-4], dtype=float),
        "fit_metric_trials": np.asarray([1.0, 2.0], dtype=float),
        "target_metric": "eta2",
    }
    app._trial_series_for_point = lambda _point: (np.asarray([1.0e-5, 1.0e-4]), np.asarray([1.0, 2.0]), "eta2")
    app._live_trial_state = lambda: {
        "a_index": 1,
        "b_index": 2,
        "metric_name": "eta2",
        "q0_trials": [1.0e-6, 1.0e-5, 1.0e-4],
        "metric_trials": [0.7, 0.2, 0.9],
    }
    app._live_trial_series_from_state = lambda _live_state: (
        np.asarray([1.0e-6, 1.0e-5, 1.0e-4], dtype=float),
        np.asarray([0.7, 0.2, 0.9], dtype=float),
        "eta2",
        None,
    )
    app.run_target_metric = "eta2"
    app.metric_var = _Var("eta2")
    app.trial_index_var = _Var(2)
    app._selected_trial_token = None
    calls = {"refresh": 0}

    def _refresh() -> None:
        calls["refresh"] += 1

    app._refresh_all = _refresh

    class _Event:
        inaxes = app.ax_trials
        xdata = 1.0e-5
        ydata = 0.2

    app._on_trials_canvas_click(_Event())

    assert app.trial_index_var.get() == 1
    assert app._selected_trial_token == (1, 2, "eta2", -1)
    assert calls["refresh"] == 1


def test_trial_token_matches_context_ignores_metric_and_size_suffix() -> None:
    app = object.__new__(PychmpViewApp)

    assert app._trial_token_matches_context((0, 1, "eta2", 11), a_token=0, b_token=1, point_metric="eta2") is True
    assert app._trial_token_matches_context((0, 1, "eta2", -1), a_token=0, b_token=1, point_metric="eta2") is True
    assert app._trial_token_matches_context((0, 1, "chi2", 11), a_token=0, b_token=1, point_metric="eta2") is True
    assert app._trial_token_matches_context((0, 2, "eta2", 11), a_token=0, b_token=1, point_metric="eta2") is False


def test_trial_token_matches_context_tolerates_small_live_float_drift() -> None:
    app = object.__new__(PychmpViewApp)

    assert (
        app._trial_token_matches_context(
            (0.9, 3.0, "eta2", -1),
            a_token=0.9000004,
            b_token=2.9999997,
            point_metric="eta2",
        )
        is True
    )


def test_selected_trial_index_preserves_manual_selection_when_trial_count_changes() -> None:
    app = object.__new__(PychmpViewApp)
    app.a_index_var = _Var(1)
    app.b_index_var = _Var(2)
    app.trial_index_var = _Var(1)
    app._selected_trial_token = (1, 2, "eta2", 2)

    selected = app._selected_trial_index_for_point(
        {},
        np.asarray([1.0e-6, 1.0e-5, 1.0e-4], dtype=float),
        np.asarray([0.5, 0.3, 0.2], dtype=float),
        "eta2",
    )

    assert selected == 1
    assert app.trial_index_var.get() == 1
    assert app._selected_trial_token == (1, 2, "eta2", 3)


def test_selected_trial_index_resets_to_display_metric_best_when_context_changes() -> None:
    app = object.__new__(PychmpViewApp)
    app.a_index_var = _Var(1)
    app.b_index_var = _Var(2)
    app.trial_index_var = _Var(2)
    app._selected_trial_token = (0, 2, "eta2", 3)
    app.run_target_metric = "eta2"

    selected = app._selected_trial_index_for_point(
        {"target_metric": "eta2", "q0": 1.0e-5},
        np.asarray([1.0e-6, 1.0e-5, 1.0e-4], dtype=float),
        np.asarray([2.0, 0.1, 0.5], dtype=float),
        "eta2",
    )

    assert selected == 1
    assert app.trial_index_var.get() == 1
    assert app._selected_trial_token == (1, 2, "eta2", 3)


def test_selected_trial_index_preserves_q0_when_display_metric_changes() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("free")
    app.a_index_var = _Var(1)
    app.b_index_var = _Var(1)
    app.trial_index_var = _Var(3)
    app._selected_trial_token = (1, 1, "eta2", 4)
    app.run_target_metric = "eta2"
    q0_trials = np.asarray([1.0e-5, 1.0e-4, 1.0e-3, 2.0e-3], dtype=float)
    chi2_trials = np.asarray([240.0, 183.0, 190.0, 210.0], dtype=float)

    selected = app._selected_trial_index_for_point(
        {"target_metric": "eta2", "q0": 2.0e-3},
        q0_trials,
        chi2_trials,
        "chi2",
    )

    assert selected == 3
    assert app.trial_index_var.get() == 3


def test_free_mode_metric_change_keeps_trial_selection() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("free")
    app.metric_var = _Var("chi2")
    app.a_index_var = _Var(1)
    app.b_index_var = _Var(1)
    app.trial_index_var = _Var(3)
    app._selected_trial_token = (1, 1, "eta2", 4)
    app._last_rendered_metric = "eta2"
    app._metric_selection_locked = lambda: False
    app._capture_current_slice_view_state = lambda _metric: None
    app._restore_trials_controls_for_metric = lambda _metric: None
    app._refresh_all = lambda **_kwargs: None

    app._on_metric_changed()

    assert app.trial_index_var.get() == 3
    assert app._selected_trial_token == (1, 1, "eta2", 4)


def test_free_mode_grid_change_defaults_trial_to_display_metric_best() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("free")
    app.a_index_var = _Var(0)
    app.b_index_var = _Var(0)
    app.trial_index_var = _Var(3)
    app._selected_trial_token = (0, 0, "chi2", 4)
    app.run_target_metric = "eta2"
    q0_trials = np.asarray([1.0e-5, 1.0e-4, 2.0e-3, 1.0e-3], dtype=float)
    chi2_trials = np.asarray([240.0, 183.0, 190.0, 210.0], dtype=float)
    point = {
        "target_metric": "eta2",
        "q0": 2.0e-3,
        "fit_q0_trials": q0_trials.tolist(),
        "fit_chi2_trials": chi2_trials.tolist(),
    }

    app.a_index_var.set(1)
    app.b_index_var.set(1)
    app._selected_trial_token = None

    selected = app._selected_trial_index_for_point(point, q0_trials, chi2_trials, "chi2")

    assert selected == 1
    assert app.trial_index_var.get() == 1


def test_selected_trial_index_defaults_to_display_metric_best() -> None:
    app = object.__new__(PychmpViewApp)
    app.a_index_var = _Var(1)
    app.b_index_var = _Var(1)
    app.trial_index_var = _Var(0)
    app._selected_trial_token = None
    app.run_target_metric = "eta2"
    q0_trials = np.asarray([1.0e-5, 1.0e-4, 2.0e-3, 1.0e-3], dtype=float)
    chi2_trials = np.asarray([240.0, 183.0, 190.0, 210.0], dtype=float)
    point = {
        "target_metric": "eta2",
        "q0": 2.0e-3,
        "fit_q0_trials": q0_trials.tolist(),
        "fit_eta2_trials": np.asarray([1.0, 0.8, 0.55, 0.7], dtype=float).tolist(),
        "fit_chi2_trials": chi2_trials.tolist(),
    }

    selected = app._selected_trial_index_for_point(point, q0_trials, chi2_trials, "chi2")

    assert selected == 1
    assert app.trial_index_var.get() == 1


def test_slice_change_schedules_deferred_reload() -> None:
    app = object.__new__(PychmpViewApp)
    app.slice_menu = type("_Menu", (), {"current": lambda self: 0})()
    app.available_slices = [{"key": "euv_193"}]
    app.slice_key_var = _Var("euv_171")
    app.search_id_var = _Var("search-1")
    app._last_rendered_metric = "eta2"
    app._capture_current_slice_view_state = lambda _metric: None
    app.trials_xscale_var = _Var("linear scale")
    app.trials_yscale_var = _Var("linear scale")
    app.trials_xmin_var = _Var("")
    app.trials_xmax_var = _Var("")
    app.trials_ymin_var = _Var("")
    app.trials_ymax_var = _Var("")
    calls: list[str] = []
    app._schedule_payload_reload = lambda *, status_text=None: calls.append(str(status_text))

    app._on_slice_changed()

    assert app.slice_key_var.get() == "euv_193"
    assert app.search_id_var.get() == ""
    assert calls == ["Loading selected slice..."]


def test_search_change_schedules_deferred_reload() -> None:
    app = object.__new__(PychmpViewApp)
    app.search_menu = type("_Menu", (), {"current": lambda self: 0})()
    app.available_searches = [{"search_id": "search-2"}]
    app.search_id_var = _Var("search-1")
    app._last_rendered_metric = "eta2"
    app._capture_current_slice_view_state = lambda _metric: None
    app._selected_trial_token = object()
    calls: list[str] = []
    app._schedule_payload_reload = lambda *, status_text=None: calls.append(str(status_text))

    app._on_search_changed()

    assert app.search_id_var.get() == "search-2"
    assert app._selected_trial_token is None
    assert calls == ["Loading selected search..."]


def test_refresh_search_controls_prefers_ui_search_over_stale_payload() -> None:
    app = object().__new__(PychmpViewApp)
    app.payload = {"selected_search_id": "search_old", "selected_slice_key": "mw_2p873584ghz"}
    app.available_searches = [
        {"search_id": "search_old"},
        {"search_id": "search_new"},
    ]
    app.search_id_var = _Var("search_new")
    app.search_display_var = _Var("")
    app.search_menu = type("_Menu", (), {"configure": lambda *a, **k: None, "current": lambda self, index=0: None, "grid": lambda *a, **k: None, "grid_remove": lambda *a, **k: None})()
    app.search_display_label = None
    app._search_label = PychmpViewApp._search_label.__get__(app, PychmpViewApp)
    app._unique_menu_labels = lambda labels, keys: labels

    app._refresh_search_controls()

    assert app.search_id_var.get() == "search_new"


def test_refresh_all_schedules_reload_when_payload_search_is_stale() -> None:
    app = object().__new__(PychmpViewApp)
    app.payload = {
        "selected_search_id": "search_old",
        "selected_slice_key": "mw_2p873584ghz",
        "point_records": [{"a": 0.25, "b": 2.75, "a_index": 0, "b_index": 0, "status": "computed", "metrics": {"eta2": 0.2}}],
        "a_values": [0.25],
        "b_values": [2.75],
        "target_metric": "eta2",
    }
    app.display_model = {"records": [{"a": 0.25, "b": 2.75, "a_index": 0, "b_index": 0, "status": "computed", "metrics": {"eta2": 0.2}, "a0": 0.0, "a1": 1.0, "b0": 0.0, "b1": 1.0}]}
    app.search_id_var = _Var("search_new")
    app.artifact_h5 = Path("/tmp/adaptive.h5")
    app.metric_var = _Var("eta2")
    app.navigation_mode_var = _Var("free")
    app.a_index_var = _Var(0)
    app.b_index_var = _Var(0)
    app.a_values = np.asarray([0.25], dtype=float)
    app.b_values = np.asarray([2.75], dtype=float)
    app._selected_search_id = lambda: str(app.search_id_var.get())
    app._navigation_mode = lambda: "free"
    app._ensure_selected_point_exists = lambda **kwargs: True
    app._payload_has_computed_grid = lambda: True
    app._live_trial_state = lambda: None
    app._has_selected_point = lambda: True
    app._refresh_action_states = lambda: None
    app._refresh_scan_state_display = lambda: None
    app._draw_heatmap = lambda: (_ for _ in ()).throw(AssertionError("heatmap should not draw"))
    app._draw_trials = lambda: None
    app._refresh_summary = lambda: None
    app._refresh_info_text = lambda: None
    app.heatmap_canvas = type("_Canvas", (), {"draw_idle": lambda self: None})()
    app.trials_canvas = type("_Canvas", (), {"draw_idle": lambda self: None})()
    scheduled: list[str] = []
    app._schedule_payload_reload = lambda *, status_text=None: scheduled.append(str(status_text or ""))

    app._refresh_all(update_selected_solution=False)

    assert scheduled == ["Loading selected search..."]


def test_refresh_selector_controls_tolerates_missing_available_lists() -> None:
    app = object.__new__(PychmpViewApp)
    app.payload = {}
    app.slice_key_var = _Var("")
    app.slice_display_var = _Var("")
    app.search_id_var = _Var("")
    app.search_display_var = _Var("")
    app.slice_menu = None
    app.slice_display_label = None
    app.search_menu = None
    app.search_display_label = None
    app._selected_slice_key = lambda: ""
    app._selected_search_id = lambda: ""

    app._refresh_slice_controls()
    app._refresh_search_controls()

    assert app.slice_key_var.get() == ""
    assert app.search_id_var.get() == ""


def test_center_kernel_to_shape_pads_smaller_kernel() -> None:
    kernel = np.zeros((3, 3), dtype=float)
    kernel[1, 1] = 1.0

    mapped, fraction = _center_kernel_to_shape(kernel, (7, 7))

    assert mapped.shape == (7, 7)
    assert mapped[3, 3] == 1.0
    assert fraction == 1.0


def test_center_kernel_to_shape_crops_larger_kernel() -> None:
    kernel = np.ones((7, 7), dtype=float)

    mapped, fraction = _center_kernel_to_shape(kernel, (3, 3))

    assert mapped.shape == (3, 3)
    np.testing.assert_allclose(mapped, np.ones((3, 3), dtype=float))
    assert fraction == pytest.approx(9.0 / 49.0)


def test_psf_metadata_from_diagnostics_prefers_resolved_gaussian_for_mw() -> None:
    window = object.__new__(_SelectedSolutionWindow)
    diagnostics = {
        "spectral_domain": "mw",
        "resolved_psf": {
            "kind": "gaussian",
            "source": "mw_gaussian",
            "active_bmaj_arcsec": 5.5,
            "active_bmin_arcsec": 3.2,
            "active_bpa_deg": 14.0,
        },
    }

    metadata = window._psf_metadata_from_diagnostics(diagnostics)

    assert metadata is not None
    assert metadata.kind == "gaussian"
    assert metadata.source == "mw_gaussian"
    assert metadata.bmaj_arcsec == pytest.approx(5.5)
    assert metadata.bmin_arcsec == pytest.approx(3.2)
    assert metadata.bpa_deg == pytest.approx(14.0)


def test_psf_metadata_from_diagnostics_uses_finite_euv_gaussian_fields() -> None:
    window = object.__new__(_SelectedSolutionWindow)
    diagnostics = {
        "spectral_domain": "euv",
        "psf_source": "gaussian:171",
        "psf_bmaj_arcsec": np.nan,
        "psf_bmin_arcsec": np.nan,
        "psf_bpa_deg": np.nan,
        "active_bmaj_arcsec": 1.8,
        "active_bmin_arcsec": 1.5,
        "active_bpa_deg": 22.0,
    }

    metadata = window._psf_metadata_from_diagnostics(diagnostics)

    assert metadata is not None
    assert metadata.kind == "gaussian"
    assert metadata.source == "gaussian:171"
    assert metadata.bmaj_arcsec == pytest.approx(1.8)
    assert metadata.bmin_arcsec == pytest.approx(1.5)
    assert metadata.bpa_deg == pytest.approx(22.0)


def test_psf_metadata_from_diagnostics_caches_default_psf(monkeypatch: pytest.MonkeyPatch) -> None:
    window = object.__new__(_SelectedSolutionWindow)
    window._psf_metadata_cache = {}

    calls = {"count": 0}

    def _fake_default_psf_metadata(**_kwargs):
        calls["count"] += 1
        return PSFMetadata(
            source="aiapy_psf:171",
            kind="kernel",
            kernel=np.ones((3, 3), dtype=float),
            allows_frequency_scaling=False,
        )

    monkeypatch.setattr(viewer_mod, "default_psf_metadata", _fake_default_psf_metadata)

    diagnostics = {
        "psf_source": "aiapy_psf:171",
        "spectral_domain": "euv",
        "observation_instrument": "AIA",
        "observer_obs_time": "2026-05-24T00:00:00",
    }

    first = window._psf_metadata_from_diagnostics(diagnostics)
    second = window._psf_metadata_from_diagnostics(diagnostics)

    assert first is not None
    assert second is not None
    assert first is second
    assert calls["count"] == 1


def test_psf_kernel_cache_persists_across_selected_solution_windows() -> None:
    app = object.__new__(PychmpViewApp)
    app._psf_metadata_cache = {}
    app._psf_kernel_cache = {}

    window_one = object.__new__(_SelectedSolutionWindow)
    window_one.app = app
    window_one._psf_metadata_cache = {}
    window_one._psf_kernel_cache = {}

    window_two = object.__new__(_SelectedSolutionWindow)
    window_two.app = app
    window_two._psf_metadata_cache = {}
    window_two._psf_kernel_cache = {}

    diagnostics = {
        "spectral_domain": "mw",
        "resolved_psf": {
            "kind": "gaussian",
            "source": "mw_gaussian",
            "active_bmaj_arcsec": 5.5,
            "active_bmin_arcsec": 3.2,
            "active_bpa_deg": 14.0,
        },
    }

    metadata_one = window_one._psf_metadata_from_diagnostics(diagnostics)
    assert metadata_one is not None

    kernel_key = (
        str(metadata_one.source),
        str(metadata_one.kind),
        float(metadata_one.bmaj_arcsec or 0.0),
        float(metadata_one.bmin_arcsec or 0.0),
        float(metadata_one.bpa_deg or 0.0),
        2.0,
        2.0,
        5.7,
    )
    app._psf_kernel_cache[kernel_key] = np.ones((5, 5), dtype=float)

    assert window_two._get_psf_kernel_cache().get(kernel_key) is not None


def test_heatmap_plot_limits_use_shared_extents_when_enabled() -> None:
    app = object.__new__(PychmpViewApp)
    app.shared_heatmap_axes_var = _Var(True)
    app._shared_heatmap_extents = {
        "a_min": -0.5,
        "a_max": 0.5,
        "b_min": 2.0,
        "b_max": 3.2,
    }
    app.display_model = {
        "a_min": -0.1,
        "a_max": 0.2,
        "b_min": 2.4,
        "b_max": 2.7,
    }
    app._refresh_signal_active_point = None
    app.payload = {"selected_slice_key": "euv_171"}

    a_min, a_max, b_min, b_max = app._heatmap_plot_limits()

    assert a_min == pytest.approx(-0.5)
    assert a_max == pytest.approx(0.5)
    assert b_min == pytest.approx(2.0)
    assert b_max == pytest.approx(3.2)


def test_open_session_defaults_select_heartbeat_slice_and_active_mode(tmp_path: Path) -> None:
    artifact_h5 = tmp_path / "adaptive.h5"
    refresh_path = Path(f"{artifact_h5}.refresh")
    refresh_path.write_text(
        json.dumps(
            {
                "phase": "adaptive search running",
                "slice_key": "mw_2p873584ghz",
                "search_id": "search_live",
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    app = object.__new__(PychmpViewApp)
    app.artifact_h5 = artifact_h5
    app.artifact_search_catalog = []
    app.run_target_metric = "eta2"
    app.slice_key_var = _Var("")
    app.search_id_var = _Var("")
    app.navigation_mode_var = _Var("free")
    app._navigation_mode_initialized = False
    app._refresh_signal_mtime_ns = -1
    app._refresh_signal_phase = ""
    app._refresh_signal_slice_key = None
    app._refresh_signal_search_id = None
    app._refresh_signal_pending_points = []
    app._refresh_signal_active_point = None
    app._refresh_signal_live_trials = None
    app._live_runner_detected = lambda: True
    app._artifact_has_in_progress_search = lambda: False
    for method_name in (
        "_bind_refresh_signal_path",
        "_read_refresh_signal_payload",
        "_apply_refresh_signal_payload",
        "_sync_live_trial_state_from_artifact",
    ):
        setattr(app, method_name, getattr(PychmpViewApp, method_name).__get__(app, PychmpViewApp))

    app._apply_open_session_defaults()

    assert app.slice_key_var.get() == "mw_2p873584ghz"
    assert app.search_id_var.get() == ""
    assert app.navigation_mode_var.get() == "active"
    assert app._navigation_mode_initialized is True


def test_open_session_defaults_select_known_search_id(tmp_path: Path) -> None:
    artifact_h5 = tmp_path / "adaptive.h5"
    refresh_path = Path(f"{artifact_h5}.refresh")
    refresh_path.write_text(
        json.dumps(
            {
                "phase": "adaptive search running",
                "slice_key": "mw_2p873584ghz",
                "search_id": "search_live",
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    app = object.__new__(PychmpViewApp)
    app.artifact_h5 = artifact_h5
    app.artifact_search_catalog = [{"search_id": "search_live", "slice_key": "mw_2p873584ghz"}]
    app.payload = {"search_records": [{"search_id": "search_live"}]}
    app.available_searches = [{"search_id": "search_live"}]
    app.run_target_metric = "eta2"
    app.slice_key_var = _Var("")
    app.search_id_var = _Var("")
    app.navigation_mode_var = _Var("free")
    app._navigation_mode_initialized = False
    app._refresh_signal_mtime_ns = -1
    app._refresh_signal_phase = ""
    app._refresh_signal_slice_key = None
    app._refresh_signal_search_id = None
    app._refresh_signal_pending_points = []
    app._refresh_signal_active_point = None
    app._refresh_signal_live_trials = None
    app._live_runner_detected = lambda: True
    app._artifact_has_in_progress_search = lambda: False
    for method_name in (
        "_bind_refresh_signal_path",
        "_read_refresh_signal_payload",
        "_apply_refresh_signal_payload",
        "_sync_live_trial_state_from_artifact",
    ):
        setattr(app, method_name, getattr(PychmpViewApp, method_name).__get__(app, PychmpViewApp))

    app._apply_open_session_defaults()

    assert app.search_id_var.get() == "search_live"


def test_active_session_coerce_downgrades_stale_active_mode_to_best() -> None:
    app = object.__new__(PychmpViewApp)
    app.artifact_h5 = None
    app.artifact_search_catalog = []
    app.navigation_mode_var = _Var("active")
    app._open_session_prefers_active = True
    app._live_navigation_available = lambda: False
    app._active_navigation_mode_available = lambda: False
    app._best_navigation_available = lambda: True
    app._should_follow_active_session = lambda: True

    app._coerce_navigation_mode_to_availability()

    assert app.navigation_mode_var.get() == "best"


def test_apply_active_session_selection_schedules_runner_slice_reload() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("active")
    app._live_runner_detected = lambda: True
    app._artifact_has_in_progress_search = lambda: False
    app.artifact_h5 = Path("/tmp/adaptive.h5")
    app.slice_key_var = _Var("euv_171")
    app.search_id_var = _Var("search_old")
    app._selected_slice_key = lambda: str(app.slice_key_var.get())
    app._selected_search_id = lambda: str(app.search_id_var.get())
    app._refresh_signal_mtime_ns = -1
    app._refresh_signal_phase = ""
    app._refresh_signal_slice_key = "mw_2p873584ghz"
    app._refresh_signal_pending_points = []
    app._refresh_signal_active_point = (0.3, 2.7)
    app._refresh_signal_live_trials = {"search_id": "search_live"}
    app._bind_refresh_signal_path = PychmpViewApp._bind_refresh_signal_path.__get__(app, PychmpViewApp)
    app._sync_refresh_signal_from_disk = lambda: None
    scheduled: list[str] = []
    app._schedule_payload_reload = lambda *, status_text=None: scheduled.append(str(status_text or ""))

    changed = app._apply_active_session_selection_if_needed()

    assert changed is True
    assert app.slice_key_var.get() == "mw_2p873584ghz"
    assert scheduled == ["Loading active slice..."]


def test_apply_active_session_selection_ignores_missing_search_id() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("active")
    app._live_runner_detected = lambda: True
    app._artifact_has_in_progress_search = lambda: False
    app.artifact_h5 = Path("/tmp/adaptive.h5")
    app.slice_key_var = _Var("mw_2p873584ghz")
    app.search_id_var = _Var("search_saved")
    app.payload = {"selected_search_id": "search_saved", "search_records": [{"search_id": "search_saved"}]}
    app.available_searches = [{"search_id": "search_saved"}]
    app._selected_slice_key = lambda: str(app.slice_key_var.get())
    app._selected_search_id = lambda: str(app.search_id_var.get())
    app._refresh_signal_slice_key = "mw_2p873584ghz"
    app._refresh_signal_live_trials = {"search_id": "search_missing"}
    app._sync_refresh_signal_from_disk = lambda: None
    scheduled: list[str] = []
    app._schedule_payload_reload = lambda *, status_text=None: scheduled.append(str(status_text or ""))

    changed = app._apply_active_session_selection_if_needed()

    assert changed is False
    assert scheduled == []


def test_apply_active_session_selection_ignores_in_progress_without_live_runner() -> None:
    app = object.__new__(PychmpViewApp)
    app.navigation_mode_var = _Var("active")
    app._live_runner_detected = lambda: False
    app._should_follow_active_session = lambda: False
    app._artifact_has_in_progress_search = lambda: True
    app.artifact_h5 = Path("/tmp/adaptive.h5")
    app.slice_key_var = _Var("euv_171")
    app.search_id_var = _Var("search_old")
    app._selected_slice_key = lambda: str(app.slice_key_var.get())
    app._selected_search_id = lambda: str(app.search_id_var.get())
    app._resolve_active_session_slice_search = lambda: ("mw_2p873584ghz", "search_live")
    app._search_id_in_artifact = lambda *_args, **_kwargs: True
    app._sync_refresh_signal_from_disk = lambda: None
    scheduled: list[str] = []
    app._schedule_payload_reload = lambda *, status_text=None: scheduled.append(str(status_text or ""))

    changed = app._apply_active_session_selection_if_needed()

    assert changed is False
    assert scheduled == []


def test_payload_has_computed_grid_detects_saved_points() -> None:
    app = object.__new__(PychmpViewApp)
    app.display_model = {"records": []}
    app.payload = {
        "points": {
            (0, 0): {"status": "computed", "a": 0.0, "b": 2.4},
        },
    }
    assert app._payload_has_computed_grid() is True


def test_refresh_all_draws_heatmap_when_search_completed_without_live_point() -> None:
    app = object.__new__(PychmpViewApp)
    app.payload = {
        "selected_slice_key": "mw_2p873584ghz",
        "selected_search_id": "search_live",
        "target_metric": "eta2",
        "a_values": [0.0, 0.3],
        "b_values": [2.4, 2.7],
        "points": {
            (0, 0): {
                "status": "computed",
                "a": 0.0,
                "b": 2.4,
                "success": True,
                "metrics": {"eta2": 0.5},
                "fit_q0_trials": [1e-5],
                "fit_metric_trials": [0.5],
            },
        },
        "diagnostics": {},
    }
    app.display_model = {
        "records": [
            {
                "a_index": 0,
                "b_index": 0,
                "a": 0.0,
                "b": 2.4,
                "a0": -0.15,
                "a1": 0.15,
                "b0": 2.25,
                "b1": 2.55,
                "a_center": 0.0,
                "b_center": 2.4,
                "status": "computed",
                "metrics": {"eta2": 0.5},
            }
        ],
        "a_min": -0.15,
        "a_max": 0.45,
        "b_min": 2.25,
        "b_max": 2.85,
    }
    app.a_values = np.asarray([0.0, 0.3], dtype=float)
    app.b_values = np.asarray([2.4, 2.7], dtype=float)
    app.navigation_mode_var = _Var("active")
    app.metric_var = _Var("eta2")
    app.run_target_metric = "eta2"
    app.a_index_var = _Var(0)
    app.b_index_var = _Var(0)
    app._active_follow_point_id = "p000024"
    app._refresh_signal_active_point = None
    app._live_navigation_available = lambda: False
    app._active_navigation_mode_available = lambda: False
    app._live_trial_state = lambda: None
    app._grid_point_coords_for_id = lambda point_id: (-0.9, 3.0) if point_id == "p000024" else None
    app._point_indices_for_coordinates = lambda a, b: (1, 1) if np.isclose(a, -0.9) and np.isclose(b, 3.0) else (0, 0)
    app._refresh_selector_values = lambda: None
    app._clear_trial_selector_controls = lambda: None
    app._refresh_action_states = lambda: None
    app._refresh_scan_state_display = lambda: None
    app._refresh_info_text = lambda: None
    app._refresh_summary = lambda: None
    app._reset_heatmap_colorbar = lambda: None
    app.ax_heatmap = _AxisStub()
    app.ax_trials = _AxisStub()
    app.heatmap_figure = _FigureStub()
    app.trials_figure = _FigureStub()
    app.heatmap_canvas = type("Canvas", (), {"draw_idle": lambda self: None})()
    app.trials_canvas = type("Canvas", (), {"draw_idle": lambda self: None})()
    app.status_var = _Var("")
    app.summary_var = _Var("")
    app.selected_solution_window = None
    app._refresh_signal_pending_points = []
    app._best_tied_records = []
    drawn = {"heatmap": 0, "trials": 0}
    app._draw_heatmap = lambda: drawn.__setitem__("heatmap", drawn["heatmap"] + 1)
    app._draw_trials = lambda: drawn.__setitem__("trials", drawn["trials"] + 1)

    app._refresh_all()

    assert drawn["heatmap"] == 1
    assert drawn["trials"] == 1
    assert "Waiting for first completed" not in app.status_var.get()
