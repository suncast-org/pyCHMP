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
    def set_axis_on(self) -> None:
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


def test_refresh_signal_payload_parses_live_trials_and_active_point(tmp_path: Path) -> None:
    app = object.__new__(PychmpViewApp)
    app.refresh_signal_path = tmp_path / "adaptive.h5.refresh"
    app.run_target_metric = "eta2"
    app.a_values = np.asarray([0.0, 0.3], dtype=float)
    app.b_values = np.asarray([2.4, 2.7], dtype=float)
    payload = {
        "phase": "trial 03 complete",
        "pending_points": [{"a": 0.0, "b": 2.4}],
        "active_point": {"a": 0.0, "b": 2.4},
        "live_trials": {
            "a": 0.0,
            "b": 2.4,
            "metric_name": "eta2",
            "q0_trials": [1.0e-5, 1.0e-4, 1.0e-3],
            "metric_trials": [3.0, 2.0, 1.0],
            "active_trial_index": 3,
            "active_trial_q0": 1.0e-2,
        },
    }
    app.refresh_signal_path.write_text(json.dumps(payload), encoding="utf-8")

    parsed = app._read_refresh_signal_payload()

    assert parsed["phase"] == "trial 03 complete"
    assert parsed["pending_points"] == [(0.0, 2.4)]
    assert parsed["active_point"] == (0.0, 2.4)
    assert parsed["live_trials"]["metric_name"] == "eta2"
    assert parsed["live_trials"]["active_trial_index"] == 3
    assert parsed["live_trials"]["active_trial_q0"] == 1.0e-2


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

    calls = {"reload": 0, "refresh": 0}
    app._read_refresh_signal_payload = lambda: {
        "phase": "trial 03 complete",
        "slice_key": "euv_171",
        "pending_points": [(0.0, 2.4)],
        "active_point": (0.0, 2.4),
        "live_trials": {"metric_name": "eta2", "q0_trials": [1.0e-5], "metric_trials": [1.0]},
    }
    app._reload_payload = lambda: calls.__setitem__("reload", calls["reload"] + 1)
    app._refresh_all = lambda **_kwargs: calls.__setitem__("refresh", calls["refresh"] + 1)

    class _RootStub:
        def after(self, *_args, **_kwargs):
            return "after-id"

    app.root = _RootStub()

    app._poll_external_refresh_signal()

    assert calls["reload"] == 1
    assert calls["refresh"] == 0


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

    monkeypatch.setattr(
        viewer_mod,
        "load_active_point_snapshot",
        lambda *_args, **_kwargs: {
            "a": 0.3,
            "b": 2.7,
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


def test_jump_to_active_point_selects_live_coordinates() -> None:
    app = object.__new__(PychmpViewApp)
    app._refresh_signal_active_point = (0.3, 2.7)
    app.a_values = np.asarray([0.0, 0.3], dtype=float)
    app.b_values = np.asarray([2.4, 2.7], dtype=float)
    app.a_index_var = _Var(0)
    app.b_index_var = _Var(0)
    app._selected_trial_token = object()
    calls: list[str] = []
    app._refresh_selector_values = lambda: calls.append("selectors")
    app._refresh_all = lambda: calls.append("refresh")

    app._jump_to_active_point()

    assert app.a_index_var.get() == 1
    assert app.b_index_var.get() == 1
    assert app._selected_trial_token is None
    assert calls == ["selectors", "refresh"]


def test_jump_to_active_point_is_noop_without_live_coordinates() -> None:
    app = object.__new__(PychmpViewApp)
    app._refresh_signal_active_point = None
    app.a_values = np.asarray([0.0, 0.3], dtype=float)
    app.b_values = np.asarray([2.4, 2.7], dtype=float)
    app.a_index_var = _Var(0)
    app.b_index_var = _Var(0)
    app._selected_trial_token = object()
    calls: list[str] = []
    app._refresh_selector_values = lambda: calls.append("selectors")
    app._refresh_all = lambda: calls.append("refresh")

    app._jump_to_active_point()

    assert app.a_index_var.get() == 0
    assert app.b_index_var.get() == 0
    assert calls == []


def test_heatmap_click_selects_unsaved_live_active_point_overlay() -> None:
    app = object.__new__(PychmpViewApp)
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
    app.payload = {"selected_slice_key": "mw_6p929688ghz"}
    app._refresh_signal_active_point = (0.9, 3.0)
    app._refresh_signal_live_trials = {
        "slice_key": "mw_6p929688ghz",
        "metric_name": "eta2",
        "q0_trials": [1.0e-6],
        "metric_trials": [0.5],
    }
    app.a_index_var = _Var(0)
    app.b_index_var = _Var(0)
    app._selected_trial_token = object()
    calls: list[str] = []
    app._refresh_selector_values = lambda: calls.append("selectors")
    app._refresh_action_states = lambda: calls.append("actions")
    app._refresh_all = lambda: calls.append("refresh")

    app._on_canvas_click(_EventStub(inaxes=app.ax_heatmap, xdata=3.0, ydata=0.9))

    assert app.a_index_var.get() == 3
    assert app.b_index_var.get() == 2
    assert app._selected_trial_token is None
    assert calls == ["selectors", "actions", "refresh"]


def test_heatmap_click_miss_does_not_select_when_no_patch_or_active_star_match() -> None:
    app = object.__new__(PychmpViewApp)
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
    app.payload = {"selected_slice_key": "mw_6p929688ghz"}
    app._refresh_signal_active_point = (0.6, 3.0)
    app._refresh_signal_live_trials = {
        "slice_key": "mw_6p929688ghz",
        "metric_name": "eta2",
        "q0_trials": [1.0e-6],
        "metric_trials": [0.5],
    }
    app.a_index_var = _Var(0)
    app.b_index_var = _Var(0)
    token = object()
    app._selected_trial_token = token
    calls: list[str] = []
    app._refresh_selector_values = lambda: calls.append("selectors")
    app._refresh_action_states = lambda: calls.append("actions")
    app._refresh_all = lambda: calls.append("refresh")

    app._on_canvas_click(_EventStub(inaxes=app.ax_heatmap, xdata=2.4, ydata=0.0))

    assert app.a_index_var.get() == 0
    assert app.b_index_var.get() == 0
    assert app._selected_trial_token is token
    assert calls == []


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
    app.payload = {"selected_slice_key": "mw_6p929688ghz"}

    assert app._should_force_live_trials(
        {
            "slice_key": "mw_6p929688ghz",
            "active_a": 0.6,
            "active_b": 2.7,
        }
    ) is True


def test_heatmap_plot_limits_include_same_slice_live_active_point() -> None:
    app = object.__new__(PychmpViewApp)
    app.payload = {"selected_slice_key": "mw_6p929688ghz"}
    app.display_model = {
        "records": [],
        "a_min": -0.15,
        "a_max": 0.45,
        "b_min": 2.25,
        "b_max": 3.45,
    }
    app._refresh_signal_slice_key = "mw_6p929688ghz"
    app._refresh_signal_pending_points = []
    app._refresh_signal_active_point = (0.6, 2.7)

    b_min, b_max, a_min, a_max = app._heatmap_plot_limits()

    assert b_min <= 2.25
    assert b_max >= 3.45
    assert a_min <= -0.15
    assert a_max > 0.6


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
    app.ax_heatmap_cbar = _ColorbarAxisStub()
    app.heatmap_figure = _FigureStub()
    app._reset_heatmap_colorbar_axis = lambda: None
    app._refresh_signal_pending_points = []
    app._refresh_signal_active_point = None

    # Should not raise IndexError when no records/grid values exist yet.
    app._draw_heatmap()


def test_draw_trials_handles_empty_grid_without_crashing() -> None:
    app = object.__new__(PychmpViewApp)
    app._live_trial_state = lambda: None
    app._should_use_live_trials = lambda _live_state: False
    app._has_selected_point = lambda: True
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
    app.trials_xscale_var = _Var("linear")
    app.trials_yscale_var = _Var("linear")
    app.ax_trials = _AxisStub()
    app.trials_figure = _FigureStub()
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
    app.status_var = _Var("")
    app.payload = {
        "selected_slice": {"display_label": "MW: 6.930 GHz"},
        "selected_slice_key": "mw_6p929688ghz",
    }
    app.a_values = np.asarray([0.3], dtype=float)
    app.b_values = np.asarray([3.3], dtype=float)
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
    live_state = {"active_a": 0.0, "active_b": 2.4}

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
    app.payload = {"selected_slice_key": "mw_6p929688ghz"}
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


def test_trial_token_matches_context_ignores_size_suffix() -> None:
    app = object.__new__(PychmpViewApp)

    assert app._trial_token_matches_context((0, 1, "eta2", 11), a_token=0, b_token=1, point_metric="eta2") is True
    assert app._trial_token_matches_context((0, 1, "eta2", -1), a_token=0, b_token=1, point_metric="eta2") is True
    assert app._trial_token_matches_context((0, 1, "chi2", 11), a_token=0, b_token=1, point_metric="eta2") is False
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


def test_selected_trial_index_resets_to_best_when_context_changes() -> None:
    app = object.__new__(PychmpViewApp)
    app.a_index_var = _Var(1)
    app.b_index_var = _Var(2)
    app.trial_index_var = _Var(2)
    app._selected_trial_token = (0, 2, "eta2", 3)

    selected = app._selected_trial_index_for_point(
        {},
        np.asarray([1.0e-6, 1.0e-5, 1.0e-4], dtype=float),
        np.asarray([2.0, 0.1, 0.5], dtype=float),
        "eta2",
    )

    assert selected == 1
    assert app.trial_index_var.get() == 1
    assert app._selected_trial_token == (1, 2, "eta2", 3)


def test_slice_change_schedules_deferred_reload() -> None:
    app = object.__new__(PychmpViewApp)
    app.slice_menu = type("_Menu", (), {"current": lambda self: 0})()
    app.available_slices = [{"key": "euv_193"}]
    app.slice_key_var = _Var("euv_171")
    app.search_id_var = _Var("search-1")
    app._last_rendered_metric = "eta2"
    app._capture_current_slice_view_state = lambda _metric: None
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
