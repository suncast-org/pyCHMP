from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

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


def test_trial_token_matches_context_ignores_size_suffix() -> None:
    app = object.__new__(PychmpViewApp)

    assert app._trial_token_matches_context((0, 1, "eta2", 11), a_token=0, b_token=1, point_metric="eta2") is True
    assert app._trial_token_matches_context((0, 1, "eta2", -1), a_token=0, b_token=1, point_metric="eta2") is True
    assert app._trial_token_matches_context((0, 1, "chi2", 11), a_token=0, b_token=1, point_metric="eta2") is False
    assert app._trial_token_matches_context((0, 2, "eta2", 11), a_token=0, b_token=1, point_metric="eta2") is False


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
