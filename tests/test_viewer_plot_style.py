from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from pychmp.viewer_plot_style import (
    apply_figure_autolayout,
    apply_heatmap_data_limits,
    heatmap_facecolors_from_values,
    apply_q0_panel_trials_axis_style,
    limits_frame_finite_data,
    normalize_axis_scale_choice,
    reserve_q0_trials_subplot,
    resolve_grid_axis_limits,
    resolve_heatmap_color_norm,
    resolve_trial_metric_arrays,
)
from matplotlib.colors import LogNorm, Normalize


def test_q0_artifact_panel_calls_autolayout_after_update(monkeypatch: pytest.MonkeyPatch) -> None:
    from pychmp.q0_artifact_panel import Q0ArtifactPanelFigure

    calls: list[str] = []

    def _record() -> None:
        calls.append("autolayout")

    panel = Q0ArtifactPanelFigure()
    monkeypatch.setattr(panel, "apply_autolayout", _record)
    monkeypatch.setattr(panel, "_ensure_layout", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(panel, "_update_blos_panel", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(panel, "_update_common_panel", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(panel, "_update_trials", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(panel, "_draw_mask_contours", lambda *_args, **_kwargs: None)

    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 4
    header["NAXIS2"] = 4
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CRPIX1"] = 2.0
    header["CRPIX2"] = 2.0
    header["CDELT1"] = 1.0
    header["CDELT2"] = 1.0
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    shape = (4, 4)
    data = np.ones(shape, dtype=float)

    panel.update(
        model_path=Path("model.h5"),
        observed_noisy=data,
        raw_modeled_best=data,
        modeled_best=data,
        residual=np.zeros(shape, dtype=float),
        wcs_header=header,
        frequency_ghz=1.418,
        diagnostics={"a": 0.0, "b": 2.0, "target_metric": "chi2"},
    )
    assert calls == ["autolayout"]


def test_apply_figure_autolayout_uses_safe_fallback_when_tight_layout_fails() -> None:
    class _Figure:
        def __init__(self) -> None:
            self.tight_called = False
            self.adjust_args: tuple[float, float, float, float] | None = None

        def tight_layout(self, **_kwargs) -> None:
            self.tight_called = True
            raise RuntimeError("tight layout failed")

        def subplots_adjust(self, *, left: float, right: float, bottom: float, top: float) -> None:
            self.adjust_args = (left, bottom, right, top)

    figure = _Figure()
    apply_figure_autolayout(figure)
    assert figure.adjust_args is not None
    left, bottom, right, top = figure.adjust_args
    assert left < right
    assert bottom < top


def test_apply_heatmap_data_limits_uses_auto_aspect() -> None:
    class _Axis:
        def __init__(self) -> None:
            self.xlim: tuple[float, float] | None = None
            self.ylim: tuple[float, float] | None = None
            self.aspect: str | None = None

        def set_xlim(self, left: float, right: float) -> None:
            self.xlim = (left, right)

        def set_ylim(self, bottom: float, top: float) -> None:
            self.ylim = (bottom, top)

        def set_aspect(self, value: str) -> None:
            self.aspect = value

    ax = _Axis()
    apply_heatmap_data_limits(ax, a_min=-1.0, a_max=1.0, b_min=2.0, b_max=4.0)
    assert ax.xlim == (-1.0, 1.0)
    assert ax.ylim == (2.0, 4.0)
    assert ax.aspect == "auto"


def test_resolve_heatmap_color_norm_uses_log_for_positive_values() -> None:
    values = np.asarray([1.0, 10.0, 100.0], dtype=float)
    norm, vmin, vmax = resolve_heatmap_color_norm(values, log_scale=True)
    assert isinstance(norm, LogNorm)
    assert vmin == pytest.approx(1.0)
    assert vmax == pytest.approx(100.0)


def test_heatmap_colorbar_reset_after_remove_recreates_cax() -> None:
    from matplotlib.figure import Figure
    import matplotlib.cm as mpl_cm
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig = Figure(layout=None)
    gs = fig.add_gridspec(1, 2, width_ratios=[22, 1], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    cmap = plt.colormaps["viridis"]
    mappable = mpl_cm.ScalarMappable(norm=LogNorm(0.75, 1.05), cmap=cmap)
    mappable.set_array([0.8, 1.0])
    cb = fig.colorbar(mappable, cax=cax)
    cb.remove()
    parent_fig = cax.get_figure(root=False)
    if parent_fig is None:
        cax = fig.add_subplot(gs[0, 1])
    else:
        cax.cla()
    fig.colorbar(mappable, cax=cax)


def test_heatmap_colorbar_cax_does_not_shrink_main_axes_on_repeat() -> None:
    from matplotlib.figure import Figure
    import matplotlib.cm as mpl_cm
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig = Figure(layout=None)
    gs = fig.add_gridspec(1, 2, width_ratios=[22, 1], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    width0 = float(ax.get_position().width)
    cmap = plt.colormaps["viridis"]
    mappable = mpl_cm.ScalarMappable(norm=LogNorm(0.75, 1.05), cmap=cmap)
    mappable.set_array([0.8, 1.0, 1.04])
    for _ in range(6):
        cax.cla()
        fig.colorbar(mappable, cax=cax)
        ax.clear()
    width_after = float(ax.get_position().width)
    assert width_after == pytest.approx(width0, rel=1e-4)


def test_heatmap_facecolors_differ_between_linear_and_log_norm() -> None:
    import matplotlib.pyplot as plt

    values = np.asarray([0.6, 0.8, 1.0], dtype=float)
    cmap = plt.colormaps["viridis"]
    linear_norm, _, _ = resolve_heatmap_color_norm(values, log_scale=False)
    log_norm, _, _ = resolve_heatmap_color_norm(values, log_scale=True)
    linear_colors = heatmap_facecolors_from_values(values, norm=linear_norm, cmap=cmap)
    log_colors = heatmap_facecolors_from_values(values, norm=log_norm, cmap=cmap)
    assert not np.allclose(linear_colors, log_colors)


def test_resolve_heatmap_color_norm_falls_back_to_linear_without_positive_values() -> None:
    values = np.asarray([0.0, -1.0, 0.5], dtype=float)
    norm, vmin, vmax = resolve_heatmap_color_norm(values, log_scale=True)
    assert isinstance(norm, Normalize)
    assert not isinstance(norm, LogNorm)
    assert vmin == pytest.approx(-1.0)
    assert vmax == pytest.approx(0.5)


def test_resolve_grid_axis_limits_uses_shared_extents_when_enabled() -> None:
    display_model = {"a_min": -0.1, "a_max": 0.1, "b_min": 2.0, "b_max": 2.5}
    a_min, a_max, b_min, b_max = resolve_grid_axis_limits(
        display_model=display_model,
        a_values=np.asarray([-0.5, 0.5], dtype=float),
        b_values=np.asarray([2.0, 3.0], dtype=float),
        shared_extents={"a_min": -1.0, "a_max": 1.0, "b_min": 1.0, "b_max": 4.0},
        use_shared_axes=True,
    )
    assert (a_min, a_max) == (-1.0, 1.0)
    assert (b_min, b_max) == (1.0, 4.0)


def test_q0_panel_trials_axis_style_sets_labelpad() -> None:
    class _Axis:
        def __init__(self) -> None:
            self.title: dict[str, object] = {}
            self.xlabel: dict[str, object] = {}
            self.ylabel: dict[str, object] = {}
            self.tick_kw: dict[str, object] = {}
            self.aspect: str | None = None

        def set_title(self, text: str, **kwargs: object) -> None:
            self.title = {"text": text, **kwargs}

        def set_xlabel(self, text: str, **kwargs: object) -> None:
            self.xlabel = {"text": text, **kwargs}

        def set_ylabel(self, text: str, **kwargs: object) -> None:
            self.ylabel = {"text": text, **kwargs}

        def tick_params(self, **kwargs: object) -> None:
            self.tick_kw = kwargs

        def set_aspect(self, value: str, **kwargs: object) -> None:
            self.aspect = value

        def get_position(self) -> object:
            return type("Pos", (), {"x0": 0.5, "y0": 0.1, "width": 0.3, "height": 0.35})()

        def set_position(self, _box: list[float]) -> None:
            return None

    ax = _Axis()
    apply_q0_panel_trials_axis_style(ax, metric_name="chi2")
    assert ax.xlabel["text"] == "q0"
    assert ax.xlabel["labelpad"] == 14
    reserve_q0_trials_subplot(ax)


def test_resolve_trial_metric_arrays_uses_display_metric_history() -> None:
    diag = {
        "trials_display_metric": "chi2",
        "point_target_metric": "eta2",
        "fit_q0_trials": [0.001, 0.002, 0.003],
        "fit_chi2_trials": [20.0, 18.0, 22.0],
        "fit_metric_trials": [0.42, 0.55, 0.61],
    }
    q0, metric = resolve_trial_metric_arrays(diag, "chi2")
    assert q0.size == 3
    assert metric.tolist() == [20.0, 18.0, 22.0]


def test_resolve_trial_metric_arrays_does_not_use_search_metric_for_other_display() -> None:
    diag = {
        "trials_display_metric": "chi2",
        "point_target_metric": "eta2",
        "fit_q0_trials": [0.001, 0.002, 0.003],
        "fit_chi2_trials": [],
        "fit_metric_trials": [0.42, 0.55, 0.61],
    }
    q0, metric = resolve_trial_metric_arrays(diag, "chi2")
    assert q0.size == 3
    assert metric.size == 0


def test_limits_frame_finite_data_rejects_stale_parent_limits() -> None:
    q0 = np.asarray([0.001, 0.002, 0.003], dtype=float)
    metric = np.asarray([20.0, 18.0, 22.0], dtype=float)
    assert limits_frame_finite_data((0.0, 1.0), (0.0, 1.0), q0, metric) is False
    assert limits_frame_finite_data((0.0, 0.004), (15.0, 35.0), q0, metric) is True


def test_normalize_axis_scale_choice_accepts_display_labels() -> None:
    assert normalize_axis_scale_choice("linear scale") == "linear"
    assert normalize_axis_scale_choice("log scale") == "log"
    assert normalize_axis_scale_choice("LOG") == "log"


def test_resolve_grid_axis_limits_uses_full_grid_when_shared_disabled() -> None:
    display_model = {"a_min": -0.1, "a_max": 0.1, "b_min": 2.2, "b_max": 2.4}
    a_min, a_max, b_min, b_max = resolve_grid_axis_limits(
        display_model=display_model,
        a_values=np.asarray([-0.25, 0.25], dtype=float),
        b_values=np.asarray([2.2, 2.8], dtype=float),
        shared_extents={"a_min": -1.0, "a_max": 1.0, "b_min": 1.0, "b_max": 4.0},
        use_shared_axes=False,
    )
    assert a_min < -0.2
    assert a_max > 0.2
    assert b_min < 2.2
    assert b_max > 2.8
    assert (a_min, a_max) != (-1.0, 1.0)
    assert (b_min, b_max) != (1.0, 4.0)
