"""Shared plot styling helpers for the pyCHMP scan viewer."""

from __future__ import annotations

from typing import Any

import numpy as np

from .ab_scan_artifacts import grid_extents_from_ab_values


def resolve_grid_axis_limits(
    *,
    display_model: dict[str, Any],
    a_values: np.ndarray,
    b_values: np.ndarray,
    shared_extents: dict[str, float] | None,
    use_shared_axes: bool,
) -> tuple[float, float, float, float]:
    """Return (a_min, a_max, b_min, b_max) using a deterministic policy."""
    if use_shared_axes and isinstance(shared_extents, dict):
        return (
            float(shared_extents["a_min"]),
            float(shared_extents["a_max"]),
            float(shared_extents["b_min"]),
            float(shared_extents["b_max"]),
        )
    if np.asarray(a_values, dtype=float).size and np.asarray(b_values, dtype=float).size:
        extents = grid_extents_from_ab_values(a_values, b_values)
        return (
            float(extents["a_min"]),
            float(extents["a_max"]),
            float(extents["b_min"]),
            float(extents["b_max"]),
        )
    return (
        float(display_model.get("a_min", 0.0)),
        float(display_model.get("a_max", 1.0)),
        float(display_model.get("b_min", 0.0)),
        float(display_model.get("b_max", 1.0)),
    )


def format_heatmap_title(metric_name: str) -> str:
    return f"{metric_name} over (a, b)"


def format_trials_title(metric_name: str) -> str:
    return f"{metric_name} vs q0"


def apply_heatmap_axis_style(ax: Any, *, metric_name: str) -> None:
    ax.set_title(format_heatmap_title(metric_name), fontsize=12, pad=10)
    ax.set_xlabel("a", labelpad=8)
    ax.set_ylabel("b", labelpad=10)
    ax.tick_params(axis="both", which="major", pad=4)


def apply_figure_autolayout(figure: Any, *, pad: float = 1.0) -> None:
    """Fit subplots to titles, tick labels, and colorbar using the renderer (like IDL char margins)."""
    try:
        figure.tight_layout(pad=pad, h_pad=0.8, w_pad=0.8)
        return
    except Exception:
        pass
    left, bottom, right, top = 0.12, 0.14, 0.90, 0.92
    if left < right and bottom < top:
        try:
            figure.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
        except Exception:
            pass


def normalize_axis_scale_choice(value: str | None) -> str:
    text = str(value or "").strip().lower()
    return "log" if "log" in text else "linear"


def apply_q0_solution_panel_layout(figure: Any) -> None:
    """Fixed margins for the 2x3 selected-solution WCS grid (no constrained layout)."""
    try:
        figure.set_layout_engine(None)
    except Exception:
        pass
    try:
        figure.subplots_adjust(
            left=0.06,
            right=0.99,
            bottom=0.12,
            top=0.90,
            wspace=0.38,
            hspace=0.58,
        )
    except Exception:
        pass


def apply_heatmap_data_limits(
    ax: Any,
    *,
    a_min: float,
    a_max: float,
    b_min: float,
    b_max: float,
) -> None:
    """Set (a, b) limits and stretch the heatmap to the axes box (non-isometric)."""
    ax.set_xlim(a_min, a_max)
    ax.set_ylim(b_min, b_max)
    ax.set_aspect("auto")


def apply_trials_axis_style(ax: Any, *, metric_name: str) -> None:
    ax.set_title(format_trials_title(metric_name), fontsize=12, pad=12)
    ax.set_xlabel("q0", labelpad=8)
    ax.set_ylabel(metric_name, labelpad=10)
    ax.tick_params(axis="both", which="major", pad=4)


def resolve_trial_metric_arrays(
    diagnostics: dict[str, Any],
    target_metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve q0/metric trial series for the viewer's selected display metric."""
    q0 = np.asarray(diagnostics.get("fit_q0_trials", ()), dtype=float)
    metric_name = str(
        target_metric
        or diagnostics.get("trials_display_metric")
        or diagnostics.get("target_metric")
        or "chi2"
    ).strip().lower()
    metric_key = f"fit_{metric_name}_trials"
    metric = np.asarray(diagnostics.get(metric_key, ()), dtype=float)
    if metric.size == q0.size and metric.size > 0 and np.any(np.isfinite(metric)):
        return q0, metric
    point_target = str(
        diagnostics.get("point_target_metric") or diagnostics.get("search_target_metric") or ""
    ).strip().lower()
    if metric_name == point_target:
        metric = np.asarray(diagnostics.get("fit_metric_trials", ()), dtype=float)
        if metric.size == q0.size and metric.size > 0 and np.any(np.isfinite(metric)):
            return q0, metric
    return q0, np.asarray([], dtype=float)


def limits_frame_finite_data(
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    q0: np.ndarray,
    metric: np.ndarray,
) -> bool:
    x0, x1 = (float(min(xlim)), float(max(xlim)))
    y0, y1 = (float(min(ylim)), float(max(ylim)))
    finite = np.isfinite(q0) & np.isfinite(metric)
    if not np.any(finite) or x1 <= x0 or y1 <= y0:
        return False
    xf = np.asarray(q0[finite], dtype=float)
    yf = np.asarray(metric[finite], dtype=float)
    x_margin = max((x1 - x0) * 0.05, 1e-12)
    y_margin = max((y1 - y0) * 0.05, 1e-12)
    return bool(
        np.all((xf >= x0 - x_margin) & (xf <= x1 + x_margin))
        and np.all((yf >= y0 - y_margin) & (yf <= y1 + y_margin))
    )


def autoscale_trials_from_data(ax: Any, q0: np.ndarray, metric: np.ndarray) -> None:
    finite = np.isfinite(q0) & np.isfinite(metric)
    if not np.any(finite):
        return
    xf = np.asarray(q0[finite], dtype=float)
    yf = np.asarray(metric[finite], dtype=float)
    x_min = float(np.min(xf))
    x_max = float(np.max(xf))
    y_min = float(np.min(yf))
    y_max = float(np.max(yf))
    x_pad = (x_max - x_min) * 0.08 if x_max > x_min else max(abs(x_min) * 0.05, 1e-9)
    y_pad = (y_max - y_min) * 0.12 if y_max > y_min else max(abs(y_min) * 0.1, 1e-12)
    if ax.get_xscale() == "log":
        if x_min > 0 and x_max > 0:
            ax.set_xlim(x_min / 1.15, x_max * 1.15)
    else:
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
    if ax.get_yscale() == "log":
        if y_min > 0 and y_max > 0:
            ax.set_ylim(y_min / 1.2, y_max * 1.2)
    else:
        ax.set_ylim(y_min - y_pad, y_max + y_pad)


def apply_q0_panel_trials_axis_style(ax: Any, *, metric_name: str) -> None:
    """Trials axis styling for the selected-solution 2x3 panel (extra bottom room for labels)."""
    ax.set_title(format_trials_title(metric_name), fontsize=9, pad=10)
    ax.set_xlabel("q0", fontsize=9, labelpad=14)
    ax.set_ylabel(metric_name, fontsize=9, labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=8, pad=5)


def reserve_q0_trials_subplot(ax: Any) -> None:
    """Lift the trials cell slightly so the q0 label is not clipped by the window edge."""
    pos = ax.get_position()
    lift = min(0.035, max(0.012, pos.height * 0.08))
    shrink = lift * 0.4
    ax.set_position([pos.x0, pos.y0 + lift, pos.width, max(pos.height - shrink, 0.15)])


def show_trials_message(ax: Any, message: str) -> None:
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        wrap=True,
    )
