from __future__ import annotations

import numpy as np
import pytest

from pychmp.ab_scan_tasks import ABSliceTaskDescriptor, compile_rectangular_point_tasks, compile_sparse_point_tasks


def _slice_descriptor() -> ABSliceTaskDescriptor:
    return ABSliceTaskDescriptor(
        key="mw_2p874ghz",
        domain="mw",
        label="2.874 GHz",
        display_label="MW: 2.874 GHz",
    )


def test_compile_rectangular_point_tasks_builds_full_grid() -> None:
    """Expand rectangular scan inputs into the full task grid."""
    tasks = compile_rectangular_point_tasks(
        a_values=[0.0, 0.3],
        b_values=[2.1, 2.4, 2.7],
        slice_descriptor=_slice_descriptor(),
        q0_min=1e-5,
        q0_max=1e-3,
        target_metric="chi2",
    )
    assert len(tasks) == 6
    assert tasks[0].source_kind == "rectangular"
    assert tasks[0].slice_key == "mw_2p874ghz"
    assert tasks[0].a_index == 0
    assert tasks[0].b_index == 0
    assert tasks[-1].a_index == 1
    assert tasks[-1].b_index == 2


def test_compile_sparse_point_tasks_preserves_requested_points() -> None:
    """Preserve sparse point coordinates and per-point q0 bounds."""
    tasks = compile_sparse_point_tasks(
        point_specs=[(0.3, 2.1, None, None), (0.6, 2.7, 2e-5, 5e-4)],
        a_values=np.asarray([0.0, 0.3, 0.6], dtype=float),
        b_values=np.asarray([2.1, 2.4, 2.7], dtype=float),
        slice_descriptor=_slice_descriptor(),
        default_q0_min=1e-5,
        default_q0_max=1e-3,
        target_metric="chi2",
    )
    assert len(tasks) == 2
    assert tasks[0].source_kind == "sparse"
    assert tasks[0].a_index == 1
    assert tasks[0].b_index == 0
    assert tasks[0].q0_min == pytest.approx(1e-5)
    assert tasks[1].q0_min == pytest.approx(2e-5)
    assert tasks[1].q0_max == pytest.approx(5e-4)


def test_compile_sparse_point_tasks_rejects_unknown_grid_point() -> None:
    """Reject sparse points that do not land on the declared grid."""
    with pytest.raises(ValueError, match="not present"):
        compile_sparse_point_tasks(
            point_specs=[(0.9, 2.1, None, None)],
            a_values=[0.0, 0.3, 0.6],
            b_values=[2.1, 2.4, 2.7],
            slice_descriptor=_slice_descriptor(),
            default_q0_min=1e-5,
            default_q0_max=1e-3,
            target_metric="chi2",
        )
