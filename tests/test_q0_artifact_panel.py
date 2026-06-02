from __future__ import annotations

import numpy as np
import pytest
from matplotlib.colors import LogNorm, Normalize, SymLogNorm

from pychmp.q0_artifact_panel import (
    _euv_channel_from_diagnostics,
    _euv_channel_token,
    _intensity_colormap_for_panel,
    _resolve_image_render_state,
    _sunpy_euv_colormap_name,
)


def test_resolve_image_render_state_linear_common_map() -> None:
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    display_data, norm, limits, applied_scale = _resolve_image_render_state(data, scale="linear")
    assert isinstance(norm, Normalize)
    assert not isinstance(norm, (LogNorm, SymLogNorm))
    assert applied_scale == "linear"
    assert limits == (1.0, 4.0)
    np.testing.assert_allclose(np.asarray(display_data), data)


def test_resolve_image_render_state_log_masks_nonpositive_values() -> None:
    data = np.array([[0.0, 1.0], [10.0, -5.0]], dtype=float)
    display_data, norm, limits, applied_scale = _resolve_image_render_state(data, scale="log")
    assert isinstance(norm, LogNorm)
    assert applied_scale == "log"
    assert limits == (1.0, 10.0)
    assert bool(np.ma.getmaskarray(display_data)[0, 0])
    assert bool(np.ma.getmaskarray(display_data)[1, 1])


def test_resolve_image_render_state_symlog_residual() -> None:
    data = np.array([[-10.0, -1.0], [1.0, 8.0]], dtype=float)
    _display_data, norm, limits, applied_scale = _resolve_image_render_state(
        data,
        scale="symlog",
        symmetric=True,
    )
    assert isinstance(norm, SymLogNorm)
    assert applied_scale == "symlog"
    assert limits == (-10.0, 10.0)


def test_euv_channel_token_normalizes_aia_labels() -> None:
    assert _euv_channel_token("171") == "171"
    assert _euv_channel_token("A171") == "171"
    assert _euv_channel_token(" 193 ") == "193"


def test_euv_channel_from_diagnostics_prefers_explicit_channel() -> None:
    assert _euv_channel_from_diagnostics({"euv_channel": "304"}) == "304"
    assert _euv_channel_from_diagnostics({"wavelength_angstrom": 193.0}) == "193"


def test_intensity_colormap_for_panel_uses_sunpy_for_euv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "pychmp.q0_artifact_panel._sunpy_colormap_registry",
        lambda: {"sdoaia171": object(), "sdoaia193": object()},
    )
    diagnostics = {
        "spectral_domain": "euv",
        "euv_instrument": "AIA",
        "euv_channel": "171",
    }
    assert _sunpy_euv_colormap_name(diagnostics) == "sdoaia171"
    assert _intensity_colormap_for_panel("observed", diagnostics) == "sdoaia171"
    assert _intensity_colormap_for_panel("modeled", diagnostics) == "sdoaia171"
    assert _intensity_colormap_for_panel("residual", diagnostics) == "coolwarm"


def test_intensity_colormap_for_panel_falls_back_to_inferno_for_mw() -> None:
    diagnostics = {
        "spectral_domain": "mw",
        "frequency_ghz": 17.0,
    }
    assert _sunpy_euv_colormap_name(diagnostics) is None
    assert _intensity_colormap_for_panel("observed", diagnostics) == "inferno"
    assert _intensity_colormap_for_panel("residual", diagnostics) == "coolwarm"


def test_draw_mask_contours_applies_to_all_map_axes() -> None:
    from pychmp.q0_artifact_panel import Q0ArtifactPanelFigure

    class _Axis:
        def __init__(self) -> None:
            self._mask_contours: object = ()
            self.contour_calls = 0

        def contour(self, *_args, **_kwargs):
            self.contour_calls += 1
            return object()

    panel = Q0ArtifactPanelFigure()
    panel._shape = (4, 4)
    axes = {name: _Axis() for name in ("observed", "raw_modeled", "modeled", "residual")}
    panel._common_axes = axes
    blos_axis = _Axis()
    panel._blos_ax = blos_axis
    panel._blos_image = object()

    observed = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    modeled = observed.copy()
    diagnostics = {"mask_type": "union", "metrics_mask_threshold": 0.1}

    panel._draw_mask_contours(True, observed, modeled, diagnostics)

    assert axes["observed"].contour_calls == 1
    assert axes["raw_modeled"].contour_calls == 1
    assert axes["modeled"].contour_calls == 1
    assert axes["residual"].contour_calls == 1
    assert blos_axis.contour_calls == 1
    assert _intensity_colormap_for_panel("observed", diagnostics) == "inferno"
