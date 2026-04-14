from __future__ import annotations

from pathlib import Path

from pychmp.viewer import PychmpViewApp


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
            "search_mode": "adaptive_local_single_frequency",
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
