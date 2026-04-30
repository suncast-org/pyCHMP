from __future__ import annotations

import numpy as np
from matplotlib.colors import LogNorm, Normalize, SymLogNorm

from pychmp.q0_artifact_panel import _resolve_image_render_state


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
