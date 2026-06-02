"""Interactive viewer for consolidated `(a, b)` scan artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import textwrap
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from types import SimpleNamespace
from typing import Any
import numpy as np
from astropy.io import fits
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from scipy.signal import fftconvolve

from .obs_preprocessing import (
    format_observation_shift_label,
    format_search_shift_policy_label,
    resolve_trial_observation_for_display,
)
from .search_contract import apply_search_shift_diagnostics
from .ab_scan_artifacts import (
    find_in_progress_slice_search,
    list_artifact_search_catalog,
    list_scan_slices,
    METRICS,
    best_grid_index,
    build_patch_grid_model,
    default_point_index,
    extend_patch_grid_model_with_pending_point,
    find_record_for_point,
    grid_indices_for_coordinates,
    grid_patch_rectangle,
    nearest_index,
    load_selected_trial_plot_payload,
    load_scan_file,
    load_shared_grid_extents,
    resolve_point_index,
    search_lifecycle_is_in_progress,
    with_observer_metadata,
)
from .grid_points import load_grid_point_live_state
from .metrics import format_metrics_mask_label, resolve_metrics_threshold_mask, resolve_threshold_mask
from .viewer_plot_style import (
    apply_heatmap_axis_style,
    apply_figure_autolayout,
    apply_heatmap_data_limits,
    apply_trials_axis_style,
    resolve_grid_axis_limits,
    show_trials_message,
)
from .viewer_navigation import (
    REFRESH_EVENTS_REQUIRING_SLICE_RELOAD,
    assigned_only_records,
    best_point_selection,
    execution_is_serial,
    find_global_best_domain,
    search_metric_best_trial_index,
    should_follow_active_refresh_event,
)
from .psf import KernelConvolvedRenderer, PSFMetadata, build_psf_kernel, default_psf_metadata
from .q0_artifact_panel import Q0ArtifactPanelFigure
from .gxrender_adapter import GXRenderEUVAdapter, GXRenderMWAdapter

NAVIGATION_MODES = ("free", "active", "best")
_NO_GRID_SOLUTION_MESSAGE = "No solution for the selected grid point."


def _heatmap_grid_rectangle(record: dict[str, Any]) -> Rectangle:
    x, y, width, height = grid_patch_rectangle(record)
    return Rectangle((x, y), width, height)


def _viewer_state_path() -> Path:
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "pychmp" / "viewer_state.json"
    return home / ".config" / "pychmp" / "viewer_state.json"


def _read_viewer_state() -> dict[str, Any]:
    state_path = _viewer_state_path()
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        return dict(payload) if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_viewer_state(payload: dict[str, Any]) -> None:
    try:
        state_path = _viewer_state_path()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def _load_last_directory() -> Path | None:
    last_dir = _read_viewer_state().get("last_artifact_dir")
    if not last_dir:
        return None
    try:
        path = Path(str(last_dir)).expanduser()
        return path if path.is_dir() else None
    except Exception:
        return None


def _save_last_directory(path: Path | None) -> None:
    if path is None:
        return
    payload = _read_viewer_state()
    payload["last_artifact_dir"] = str(path)
    _write_viewer_state(payload)


def _load_shared_grid_axes_pref() -> bool:
    value = _read_viewer_state().get("shared_grid_axes")
    return bool(value) if isinstance(value, bool) else False


def _save_shared_grid_axes_pref(enabled: bool) -> None:
    payload = _read_viewer_state()
    payload["shared_grid_axes"] = bool(enabled)
    _write_viewer_state(payload)


def _format_scalar(value: Any, pattern: str) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "n/a"
    if not np.isfinite(numeric):
        return "nan"
    if pattern.endswith("f") and numeric != 0.0 and abs(numeric) < 1e-4:
        precision_text = pattern[1:-1] if pattern.startswith(".") else ""
        try:
            precision = max(1, int(precision_text))
        except Exception:
            precision = 6
        return format(numeric, f".{precision}e")
    return format(numeric, pattern)


def _optional_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    return numeric if np.isfinite(numeric) else None


def _first_finite_float(*values: Any) -> float | None:
    for value in values:
        numeric = _optional_float(value)
        if numeric is not None:
            return numeric
    return None


def _sequence_as_list(value: Any) -> list[Any]:
    """Convert trial-series payloads to plain lists without numpy truthiness traps."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _center_kernel_to_shape(kernel: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, float]:
    target_ny, target_nx = int(shape[0]), int(shape[1])
    source = np.asarray(kernel, dtype=float)
    if source.ndim != 2 or target_ny <= 0 or target_nx <= 0:
        return np.zeros((max(target_ny, 1), max(target_nx, 1)), dtype=float), 0.0

    src_ny, src_nx = source.shape
    out = np.zeros((target_ny, target_nx), dtype=float)

    src_y0 = max(0, (src_ny - target_ny) // 2)
    src_x0 = max(0, (src_nx - target_nx) // 2)
    dst_y0 = max(0, (target_ny - src_ny) // 2)
    dst_x0 = max(0, (target_nx - src_nx) // 2)

    copy_ny = min(src_ny, target_ny)
    copy_nx = min(src_nx, target_nx)
    if copy_ny > 0 and copy_nx > 0:
        out[dst_y0 : dst_y0 + copy_ny, dst_x0 : dst_x0 + copy_nx] = source[
            src_y0 : src_y0 + copy_ny,
            src_x0 : src_x0 + copy_nx,
        ]

    source_sum = float(np.nansum(source))
    displayed_sum = float(np.nansum(out))
    fraction = displayed_sum / source_sum if np.isfinite(source_sum) and source_sum > 0.0 else 0.0
    return out, float(np.clip(fraction, 0.0, 1.0))


def _convolve_raw_map(raw_map: np.ndarray, psf_kernel: np.ndarray | None) -> np.ndarray:
    raw = np.asarray(raw_map, dtype=float)
    kernel = None if psf_kernel is None else np.asarray(psf_kernel, dtype=float)
    if kernel is None or kernel.ndim != 2 or kernel.size == 0:
        return raw.copy()
    return np.asarray(fftconvolve(raw, kernel, mode="same"), dtype=float)


def _make_reset_icon_image(master: tk.Misc, *, size: int = 14, color: str = "#4a4a4a") -> tk.PhotoImage:
    image = tk.PhotoImage(master=master, width=size, height=size)
    image.put("", to=(0, 0, size, size))
    pixels = {
        (8, 1), (9, 1), (10, 1),
        (6, 2), (7, 2), (8, 2),
        (5, 3), (6, 3),
        (4, 4), (5, 4),
        (3, 5), (4, 5),
        (2, 6), (3, 6),
        (2, 7), (3, 7),
        (2, 8), (3, 8),
        (3, 9), (4, 9),
        (4, 10), (5, 10),
        (5, 11), (6, 11), (7, 11),
        (7, 12), (8, 12), (9, 12),
        (9, 11), (10, 11),
        (10, 10), (11, 10),
        (10, 2), (10, 3), (10, 4),
        (9, 4), (8, 4),
        (7, 4),
        (10, 9), (9, 9),
    }
    for x, y in pixels:
        if 0 <= x < size and 0 <= y < size:
            image.put(color, (x, y))
    return image


def _make_best_trial_icon_image(master: tk.Misc, *, size: int = 14) -> tk.PhotoImage:
    image = tk.PhotoImage(master=master, width=size, height=size)
    image.put("", to=(0, 0, size, size))
    orange = "#f08c00"
    white = "#ffffff"
    dark = "#8a4f00"
    star_pixels = {
        (7, 1),
        (6, 3), (7, 3), (8, 3),
        (3, 4), (4, 4), (5, 4), (6, 4), (7, 4), (8, 4), (9, 4), (10, 4), (11, 4),
        (5, 5), (6, 5), (7, 5), (8, 5), (9, 5),
        (5, 6), (6, 6), (7, 6), (8, 6), (9, 6),
        (4, 7), (5, 7), (6, 7), (7, 7), (8, 7), (9, 7), (10, 7),
        (3, 9), (4, 9), (5, 9), (9, 9), (10, 9), (11, 9),
        (5, 10), (9, 10),
        (6, 11), (8, 11),
    }
    for x, y in star_pixels:
        if 0 <= x < size and 0 <= y < size:
            image.put(orange, (x, y))
    outline_pixels = {(7, 1), (3, 4), (11, 4), (3, 9), (11, 9), (6, 11), (8, 11)}
    for x, y in outline_pixels:
        if 0 <= x < size and 0 <= y < size:
            image.put(dark, (x, y))
    center_pixels = {(7, 4), (7, 5), (7, 6), (7, 7)}
    for x, y in center_pixels:
        if 0 <= x < size and 0 <= y < size:
            image.put(white, (x, y))
    return image


class _ToolTip:
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tipwindow: tk.Toplevel | None = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def _show(self, _event: Any) -> None:
        if self.tipwindow is not None or not self.text:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#fff8dc",
            relief=tk.SOLID,
            borderwidth=1,
            font=("TkDefaultFont", 9),
            padx=6,
            pady=3,
        )
        label.pack()

    def _hide(self, _event: Any) -> None:
        if self.tipwindow is not None:
            self.tipwindow.destroy()
            self.tipwindow = None


class _SelectedSolutionWindow:
    def __init__(self, app: PychmpViewApp) -> None:
        self.app = app
        self.window = tk.Toplevel(app.root)
        self.window.title("pyCHMP Selected Solution")
        screen_w = max(1, int(self.window.winfo_screenwidth()))
        screen_h = max(1, int(self.window.winfo_screenheight()))
        width = min(max(1180, int(round(screen_w * 0.82))), max(1180, screen_w - 60))
        height = min(max(820, int(round(screen_h * 0.84))), max(820, screen_h - 80))
        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 2)
        self.window.geometry(f"{width}x{height}+{x}+{y}")
        self.window.minsize(1040, 720)
        self.window.resizable(True, True)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.load_blos = False
        self.current_context: dict[str, Any] | None = None
        self.show_mask_contours_var = tk.BooleanVar(value=True)
        self.mask_type_var = tk.StringVar(value="ROI mask: union")
        self.common_map_scale_var = tk.StringVar(value="linear")
        self.common_map_vmin_var = tk.StringVar(value="")
        self.common_map_vmax_var = tk.StringVar(value="")
        self.residual_map_scale_var = tk.StringVar(value="linear")
        self.residual_map_vmin_var = tk.StringVar(value="")
        self.residual_map_vmax_var = tk.StringVar(value="")

        outer = ttk.Frame(self.window, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)
        outer.rowconfigure(2, weight=0)

        toolbar = ttk.Frame(outer)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        toolbar.columnconfigure(13, weight=1)
        self.blos_button = ttk.Button(toolbar, text="Load B_los", command=self._load_blos_now)
        self.blos_button.grid(row=0, column=0, sticky="w")
        _ToolTip(self.blos_button, "Load B_los: fetch and cache the external reference panel on demand for the current model.")
        self.export_png_button = ttk.Button(toolbar, text="Export PNG", command=self._export_current_png)
        self.export_png_button.grid(row=0, column=1, sticky="w", padx=(8, 0))
        _ToolTip(self.export_png_button, "Export the currently displayed selected-solution figure to PNG, preserving the current zoom/pan axis state.")
        self.plot_beam_button = ttk.Button(toolbar, text="Plot Beam", command=self._plot_convolving_beam)
        self.plot_beam_button.grid(row=0, column=2, sticky="w", padx=(8, 0))
        _ToolTip(self.plot_beam_button, "Plot the convolving beam at the same map FOV and pixel scale as the synthetic image.")
        self.mask_contour_check = ttk.Checkbutton(
            toolbar,
            text="Show ROI Mask Contour",
            variable=self.show_mask_contours_var,
            command=self._toggle_mask_contours,
        )
        self.mask_contour_check.grid(row=0, column=3, sticky="w", padx=(10, 0))
        _ToolTip(
            self.mask_contour_check,
            "Toggle the metrics ROI threshold contour on all map panels (B_los, observed, raw modeled, modeled, residual).",
        )
        ttk.Label(toolbar, textvariable=self.mask_type_var, anchor=tk.W).grid(row=0, column=4, sticky="w", padx=(10, 0))
        self.status_var = tk.StringVar(value="Selected solution window is ready.")
        ttk.Label(toolbar, textvariable=self.status_var, anchor=tk.W).grid(row=0, column=5, columnspan=9, sticky="ew", padx=(10, 0))

        ttk.Label(toolbar, text="Maps").grid(row=1, column=0, sticky="w", pady=(8, 0))
        maps_scale_menu = ttk.Combobox(
            toolbar,
            width=7,
            state="readonly",
            values=("linear", "log"),
            textvariable=self.common_map_scale_var,
        )
        maps_scale_menu.grid(row=1, column=1, sticky="w", padx=(4, 6), pady=(8, 0))
        maps_vmin_entry = ttk.Entry(toolbar, width=9, textvariable=self.common_map_vmin_var)
        maps_vmin_entry.grid(row=1, column=2, sticky="w", padx=(0, 4), pady=(8, 0))
        ttk.Label(toolbar, text="to").grid(row=1, column=3, sticky="w", pady=(8, 0))
        maps_vmax_entry = ttk.Entry(toolbar, width=9, textvariable=self.common_map_vmax_var)
        maps_vmax_entry.grid(row=1, column=4, sticky="w", padx=(4, 6), pady=(8, 0))
        maps_auto_button = ttk.Button(toolbar, text="Auto", width=6, command=self._reset_common_map_display)
        maps_auto_button.grid(row=1, column=5, sticky="w", pady=(8, 0))

        ttk.Label(toolbar, text="Residual").grid(row=1, column=6, sticky="w", padx=(14, 0), pady=(8, 0))
        residual_scale_menu = ttk.Combobox(
            toolbar,
            width=7,
            state="readonly",
            values=("linear", "symlog"),
            textvariable=self.residual_map_scale_var,
        )
        residual_scale_menu.grid(row=1, column=7, sticky="w", padx=(4, 6), pady=(8, 0))
        residual_vmin_entry = ttk.Entry(toolbar, width=9, textvariable=self.residual_map_vmin_var)
        residual_vmin_entry.grid(row=1, column=8, sticky="w", padx=(0, 4), pady=(8, 0))
        ttk.Label(toolbar, text="to").grid(row=1, column=9, sticky="w", pady=(8, 0))
        residual_vmax_entry = ttk.Entry(toolbar, width=9, textvariable=self.residual_map_vmax_var)
        residual_vmax_entry.grid(row=1, column=10, sticky="w", padx=(4, 6), pady=(8, 0))
        residual_auto_button = ttk.Button(toolbar, text="Auto", width=6, command=self._reset_residual_map_display)
        residual_auto_button.grid(row=1, column=11, sticky="w", pady=(8, 0))

        _ToolTip(maps_scale_menu, "Shared intensity scale for the observed, raw modeled, and PSF-convolved modeled panels.")
        _ToolTip(maps_auto_button, "Reset the shared observed/raw/modeled intensity display to automatic limits.")
        _ToolTip(residual_scale_menu, "Residual intensity scale. Use symlog for signed residuals with wide dynamic range.")
        _ToolTip(residual_auto_button, "Reset the residual intensity display to automatic limits.")
        for widget in (maps_scale_menu, residual_scale_menu):
            widget.bind("<<ComboboxSelected>>", lambda _event: self.update_selection())
        for entry in (maps_vmin_entry, maps_vmax_entry, residual_vmin_entry, residual_vmax_entry):
            entry.bind("<Return>", lambda _event: self.update_selection())

        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(15.0, 10.5), dpi=100)
        self.panel = Q0ArtifactPanelFigure(self.figure)
        self.canvas = FigureCanvasTkAgg(self.figure, master=outer)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.grid(row=1, column=0, sticky="nsew")
        self._autolayout_after_id: str | None = None
        canvas_widget.bind("<Configure>", lambda _event: self._schedule_autolayout())
        toolbar_frame = ttk.Frame(outer)
        toolbar_frame.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.grid(row=0, column=0, sticky="w")

    def _schedule_autolayout(self) -> None:
        if getattr(self.app, "_is_closing", False):
            return
        after_id = self._autolayout_after_id
        if after_id is not None:
            try:
                self.window.after_cancel(after_id)
            except Exception:
                pass
        self._autolayout_after_id = self.window.after(60, self._run_autolayout)

    def _sync_figure_to_canvas(self) -> None:
        widget = self.canvas.get_tk_widget()
        try:
            width_px = int(widget.winfo_width())
            height_px = int(widget.winfo_height())
        except Exception:
            return
        if width_px < 120 or height_px < 120:
            return
        dpi = float(self.figure.get_dpi())
        self.figure.set_size_inches(width_px / dpi, height_px / dpi, forward=True)

    def _run_autolayout(self) -> None:
        self._autolayout_after_id = None
        if getattr(self.app, "_is_closing", False):
            return
        self._sync_figure_to_canvas()
        self.panel.apply_autolayout()
        self.canvas.draw_idle()

    def present(self) -> None:
        self.window.deiconify()
        self.window.lift()
        self.window.focus_force()
        self.window.after_idle(self._run_autolayout)

    def _parse_display_limit(self, value: str) -> float | None:
        text = str(value).strip()
        if not text:
            return None
        try:
            numeric = float(text)
        except Exception:
            return None
        return numeric if np.isfinite(numeric) else None

    def _get_psf_metadata_cache(self) -> dict[str, PSFMetadata | None]:
        app = getattr(self, "app", None)
        if app is not None:
            cache = getattr(app, "_psf_metadata_cache", None)
            if cache is None:
                cache = {}
                app._psf_metadata_cache = cache
            return cache
        cache = getattr(self, "_psf_metadata_cache", None)
        if cache is None:
            cache = {}
            self._psf_metadata_cache = cache
        return cache

    def _get_psf_kernel_cache(self) -> dict[tuple[Any, ...], np.ndarray]:
        app = getattr(self, "app", None)
        if app is not None:
            cache = getattr(app, "_psf_kernel_cache", None)
            if cache is None:
                cache = {}
                app._psf_kernel_cache = cache
            return cache
        cache = getattr(self, "_psf_kernel_cache", None)
        if cache is None:
            cache = {}
            self._psf_kernel_cache = cache
        return cache

    def close(self) -> None:
        try:
            self.window.destroy()
        finally:
            self.app.selected_solution_window = None

    def update_selection(self) -> None:
        context = self.app._selected_solution_plot_context()
        if context is None:
            self.status_var.set("No completed point available yet.")
            return
        self.current_context = context
        diagnostics = dict(context["diagnostics"])
        self._sync_figure_to_canvas()
        self.panel.update(
            model_path=context["model_path"],
            observed_noisy=context["observed_noisy"],
            raw_modeled_best=context["raw_modeled_best"],
            modeled_best=context["modeled_best"],
            residual=context["residual"],
            wcs_header=context["wcs_header"],
            frequency_ghz=context["frequency_ghz"],
            diagnostics=diagnostics,
            trials_xmin=context.get("trials_xmin"),
            trials_xmax=context.get("trials_xmax"),
            trials_ymin=context.get("trials_ymin"),
            trials_ymax=context.get("trials_ymax"),
            trials_xscale=context.get("trials_xscale"),
            trials_yscale=context.get("trials_yscale"),
            trials_xlim=context.get("trials_xlim"),
            trials_ylim=context.get("trials_ylim"),
            trials_match_parent_view=bool(context.get("trials_match_parent_view")),
            common_map_scale=str(self.common_map_scale_var.get() or "linear"),
            common_map_vmin=self._parse_display_limit(self.common_map_vmin_var.get()),
            common_map_vmax=self._parse_display_limit(self.common_map_vmax_var.get()),
            residual_map_scale=str(self.residual_map_scale_var.get() or "linear"),
            residual_map_vmin=self._parse_display_limit(self.residual_map_vmin_var.get()),
            residual_map_vmax=self._parse_display_limit(self.residual_map_vmax_var.get()),
            wcs_header_transform=context["wcs_header_transform"],
            load_blos=self.load_blos,
            blos_reference=context.get("blos_reference"),
        )
        self.panel.set_mask_contours_visible(bool(self.show_mask_contours_var.get()))
        self.panel._draw_mask_contours(
            bool(self.show_mask_contours_var.get()),
            np.asarray(context["observed_noisy"], dtype=float),
            np.asarray(context["modeled_best"], dtype=float),
            diagnostics,
        )
        self.panel.apply_autolayout()
        slice_label = str(context["slice_label"])
        mask_type = str(diagnostics.get("mask_type", "union")).strip() or "union"
        selected_trial_index = diagnostics.get("selected_trial_index")
        selected_trial_count = diagnostics.get("selected_trial_count")
        trial_label = ""
        if selected_trial_index is not None and selected_trial_count is not None:
            trial_label = f" - trial {int(selected_trial_index) + 1}/{int(selected_trial_count)}"
        self.mask_type_var.set(format_metrics_mask_label(diagnostics))
        self.window.title(
            "pyCHMP Selected Solution"
            f" - {slice_label}{trial_label} - a={float(diagnostics.get('a', np.nan)):.3f}"
            f" b={float(diagnostics.get('b', np.nan)):.3f}"
        )
        elapsed_seconds = diagnostics.get("elapsed_seconds")
        try:
            elapsed_value = float(elapsed_seconds)
        except Exception:
            elapsed_value = float("nan")
        elapsed_text = f" | elapsed={elapsed_value:.3f} s" if np.isfinite(elapsed_value) else ""
        has_saved_blos = context.get("blos_reference") is not None
        blos_state = "saved" if has_saved_blos else ("loaded" if self.load_blos else "lazy")
        self.status_var.set(
            f"Slice: {slice_label}{trial_label} | q0={_format_scalar(diagnostics.get('q0_recovered', np.nan), '.6f')}"
            f" | B_los: {blos_state}{elapsed_text}"
        )
        self.canvas.draw_idle()

    def _load_blos_now(self) -> None:
        self.load_blos = True
        self.update_selection()

    def _reset_common_map_display(self) -> None:
        self.common_map_scale_var.set("linear")
        self.common_map_vmin_var.set("")
        self.common_map_vmax_var.set("")
        self.update_selection()

    def _reset_residual_map_display(self) -> None:
        self.residual_map_scale_var.set("linear")
        self.residual_map_vmin_var.set("")
        self.residual_map_vmax_var.set("")
        self.update_selection()

    def _toggle_mask_contours(self) -> None:
        self.panel.set_mask_contours_visible(bool(self.show_mask_contours_var.get()))
        if self.current_context is None:
            return
        diagnostics = dict(self.current_context["diagnostics"])
        self.panel._draw_mask_contours(
            bool(self.show_mask_contours_var.get()),
            np.asarray(self.current_context["observed_noisy"], dtype=float),
            np.asarray(self.current_context["modeled_best"], dtype=float),
            diagnostics,
        )
        self.panel.apply_autolayout()
        self.canvas.draw_idle()

    def _export_current_png(self) -> None:
        if self.current_context is None:
            self.status_var.set("No selected solution is available to export.")
            return

        diagnostics = dict(self.current_context.get("diagnostics", {}))
        artifact_dir = self.app.last_artifact_dir or (
            self.app.artifact_h5.parent if self.app.artifact_h5 is not None else Path.cwd()
        )
        slice_label = str(self.current_context.get("slice_label", "slice")).strip() or "slice"
        safe_slice = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in slice_label).strip("_") or "slice"
        try:
            a_text = f"{float(diagnostics.get('a', np.nan)):.3f}"
            b_text = f"{float(diagnostics.get('b', np.nan)):.3f}"
        except Exception:
            a_text = "na"
            b_text = "na"
        default_name = f"selected_solution_{safe_slice}_a{a_text}_b{b_text}.png"
        out_path = filedialog.asksaveasfilename(
            parent=self.window,
            title="Export Selected Solution PNG",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
            initialdir=str(artifact_dir),
            initialfile=default_name,
        )
        if not out_path:
            return
        out_png = Path(out_path).expanduser()
        try:
            out_png.parent.mkdir(parents=True, exist_ok=True)
            self.panel.apply_autolayout()
            self.figure.savefig(out_png, dpi=self.figure.dpi)
            self.status_var.set(f"Exported PNG: {out_png}")
            _save_last_directory(out_png.parent)
            self.app.last_artifact_dir = out_png.parent
        except Exception as exc:
            self.status_var.set(f"PNG export failed: {exc}")

    def _psf_metadata_from_diagnostics(self, diagnostics: dict[str, Any]) -> PSFMetadata | None:
        diag = dict(diagnostics or {})
        cache = self._get_psf_metadata_cache()
        cache_payload = {
            "psf_source": diag.get("psf_source"),
            "resolved_psf": diag.get("resolved_psf"),
            "spectral_domain": diag.get("spectral_domain"),
            "observation_instrument": diag.get("observation_instrument"),
            "euv_instrument": diag.get("euv_instrument"),
            "observer_obs_time": diag.get("observer_obs_time"),
        }
        try:
            cache_key = json.dumps(cache_payload, sort_keys=True, default=str)
        except Exception:
            cache_key = repr(cache_payload)
        if cache_key in cache:
            return cache[cache_key]

        resolved = dict(diag.get("resolved_psf") or {})
        source = str(diag.get("psf_source") or resolved.get("source") or "").strip()
        kind = str(resolved.get("kind") or "").strip().lower()

        if kind == "gaussian":
            bmaj = _first_finite_float(
                resolved.get("active_bmaj_arcsec"),
                resolved.get("psf_bmaj_arcsec"),
                resolved.get("reference_bmaj_arcsec"),
            )
            bmin = _first_finite_float(
                resolved.get("active_bmin_arcsec"),
                resolved.get("psf_bmin_arcsec"),
                resolved.get("reference_bmin_arcsec"),
            )
            bpa = _first_finite_float(
                resolved.get("active_bpa_deg"),
                resolved.get("psf_bpa_deg"),
                resolved.get("reference_bpa_deg"),
                0.0,
            )
            if bmaj is not None and bmin is not None and bpa is not None:
                metadata = PSFMetadata(
                    source=source or "resolved_psf",
                    kind="gaussian",
                    bmaj_arcsec=float(bmaj),
                    bmin_arcsec=float(bmin),
                    bpa_deg=float(bpa),
                    allows_frequency_scaling=False,
                )
                cache[cache_key] = metadata
                return metadata

        if source.startswith("aiapy_psf:"):
            wavelength = _optional_float(source.split(":", 1)[1])
            instrument = str(diag.get("observation_instrument") or diag.get("euv_instrument") or "AIA")
            domain = str(diag.get("spectral_domain") or "euv")
            metadata = default_psf_metadata(
                domain=domain,
                instrument_name=instrument,
                wavelength_angstrom=wavelength,
                date_obs=str(diag.get("observer_obs_time") or ""),
            )
            if metadata is not None:
                cache[cache_key] = metadata
                return metadata

        bmaj = _first_finite_float(
            diag.get("active_bmaj_arcsec"),
            diag.get("psf_bmaj_arcsec"),
        )
        bmin = _first_finite_float(
            diag.get("active_bmin_arcsec"),
            diag.get("psf_bmin_arcsec"),
        )
        bpa = _first_finite_float(
            diag.get("active_bpa_deg"),
            diag.get("psf_bpa_deg"),
            0.0,
        )
        if bmaj is not None and bmin is not None and bpa is not None:
            metadata = PSFMetadata(
                source=source or "diagnostics",
                kind="gaussian",
                bmaj_arcsec=float(bmaj),
                bmin_arcsec=float(bmin),
                bpa_deg=float(bpa),
                allows_frequency_scaling=False,
            )
            cache[cache_key] = metadata
            return metadata
        cache[cache_key] = None
        return None

    def _plot_convolving_beam(self) -> None:
        if self.current_context is None:
            self.status_var.set("No selected solution available for beam plotting.")
            return

        diagnostics = dict(self.current_context.get("diagnostics", {}))
        map_shape = tuple(np.asarray(self.current_context.get("modeled_best", np.zeros((1, 1))), dtype=float).shape)
        if len(map_shape) != 2:
            self.status_var.set("Beam plot unavailable: modeled map is not 2D.")
            return

        dx = _optional_float(diagnostics.get("map_dx_arcsec"))
        dy = _optional_float(diagnostics.get("map_dy_arcsec"))
        if dx is None or dy is None:
            header = self.current_context.get("wcs_header")
            if isinstance(header, fits.Header):
                dx = dx or abs(_optional_float(header.get("CDELT1")) or 0.0)
                dy = dy or abs(_optional_float(header.get("CDELT2")) or 0.0)
        if dx is None or dy is None or dx <= 0.0 or dy <= 0.0:
            self.status_var.set("Beam plot unavailable: invalid map pixel scale.")
            return

        kernel: np.ndarray | None = None
        source = ""
        stored_kernel = self.current_context.get("psf_kernel")
        if stored_kernel is not None:
            candidate = np.asarray(stored_kernel, dtype=float)
            if candidate.ndim == 2 and candidate.size > 0:
                kernel_sum = float(np.nansum(candidate))
                if np.isfinite(kernel_sum) and kernel_sum != 0.0:
                    candidate = candidate / kernel_sum
                kernel = candidate
                source = f"{str(diagnostics.get('psf_source') or 'artifact_common_psf_kernel')} (stored)"

        if kernel is None:
            metadata = self._psf_metadata_from_diagnostics(diagnostics)
            if metadata is None:
                self.status_var.set("Beam plot unavailable: missing PSF metadata.")
                return

            active_frequency_ghz = _optional_float(
                diagnostics.get("active_frequency_ghz", diagnostics.get("frequency_ghz", diagnostics.get("mw_frequency_ghz")))
            )
            kernel_cache = self._get_psf_kernel_cache()
            kernel_cache_key = (
                str(metadata.source),
                str(metadata.kind),
                None if metadata.bmaj_arcsec is None else float(metadata.bmaj_arcsec),
                None if metadata.bmin_arcsec is None else float(metadata.bmin_arcsec),
                None if metadata.bpa_deg is None else float(metadata.bpa_deg),
                float(dx),
                float(dy),
                None if active_frequency_ghz is None else float(active_frequency_ghz),
            )
            kernel = kernel_cache.get(kernel_cache_key)
            if kernel is None:
                kernel, _resolved = build_psf_kernel(
                    metadata=metadata,
                    dx_arcsec=float(dx),
                    dy_arcsec=float(dy),
                    active_frequency_ghz=active_frequency_ghz,
                    ref_frequency_ghz=None,
                    scale_inverse_frequency=False,
                )
                if kernel is not None:
                    kernel_cache[kernel_cache_key] = np.asarray(kernel, dtype=float)
            source = str(metadata.source or diagnostics.get("psf_source") or "unknown")
        if kernel is None:
            self.status_var.set("Beam plot unavailable: failed to build PSF kernel.")
            return

        beam_map, captured_fraction = _center_kernel_to_shape(np.asarray(kernel, dtype=float), map_shape)
        center_y = int(map_shape[0] // 2)
        center_x = int(map_shape[1] // 2)
        x_axis_arcsec = (np.arange(map_shape[1], dtype=float) - float(center_x)) * float(dx)
        y_axis_arcsec = (np.arange(map_shape[0], dtype=float) - float(center_y)) * float(dy)
        x_extent = (x_axis_arcsec[0] - 0.5 * float(dx), x_axis_arcsec[-1] + 0.5 * float(dx))
        y_extent = (y_axis_arcsec[0] - 0.5 * float(dy), y_axis_arcsec[-1] + 0.5 * float(dy))
        profile = np.asarray(beam_map[center_y, :], dtype=float)

        fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), dpi=120)
        linear_im = axes[0].imshow(
            beam_map,
            origin="lower",
            cmap="magma",
            extent=(x_extent[0], x_extent[1], y_extent[0], y_extent[1]),
        )
        axes[0].set_title("Beam (linear)")
        axes[0].set_xlabel("x offset [arcsec]")
        axes[0].set_ylabel("y offset [arcsec]")
        linear_cb = fig.colorbar(linear_im, ax=axes[0], fraction=0.046, pad=0.04)
        linear_cb.set_label("kernel value")

        floor = max(float(np.nanmax(beam_map)) * 1.0e-8, 1.0e-20)
        log_im = axes[1].imshow(
            np.log10(np.clip(beam_map, floor, None)),
            origin="lower",
            cmap="magma",
            extent=(x_extent[0], x_extent[1], y_extent[0], y_extent[1]),
        )
        axes[1].set_title("Beam (log10)")
        axes[1].set_xlabel("x offset [arcsec]")
        axes[1].set_ylabel("y offset [arcsec]")
        log_cb = fig.colorbar(log_im, ax=axes[1], fraction=0.046, pad=0.04)
        log_cb.set_label("log10(kernel value)")

        axes[2].plot(x_axis_arcsec, np.clip(profile, 1.0e-20, None), lw=1.8)
        axes[2].set_yscale("log")
        axes[2].set_title("Center-line profile")
        axes[2].set_xlabel("x offset [arcsec]")
        axes[2].set_ylabel("kernel value")
        axes[2].grid(alpha=0.25)

        if not source:
            source = str(diagnostics.get("psf_source") or "unknown")
        fig.suptitle(
            f"Convolving beam at map FOV {map_shape[1]}x{map_shape[0]} | dx={float(dx):.3f}\" dy={float(dy):.3f}\" | source={source} | displayed energy={captured_fraction:.2%}",
            fontsize=10,
        )
        fig.tight_layout()
        plt.show(block=False)
        self.status_var.set("Opened convolving-beam plot window.")


class PychmpViewApp:
    _LOAD_RETRY_ATTEMPTS = 8
    _LOAD_RETRY_DELAY_S = 0.35
    _EXTERNAL_REFRESH_POLL_MS = 1200
    _MIN_HEARTBEAT_RELOAD_INTERVAL_S = 2.0
    _MAX_LOG_LINES = 3000
    _INITIAL_LOG_READ_BYTES = 262_144
    _HEATMAP_FOOTER_HEIGHT = 108
    _HEATMAP_COLORBAR_FRACTION = 0.046
    _HEATMAP_COLORBAR_PAD = 0.04
    _TRIALS_FOOTER_HEIGHT = 104
    _TOOLBAR_HEIGHT = 78
    _NOTEBOOK_TAB_HEIGHT = 28
    _WINDOW_VERTICAL_CHROME = 28
    _DISPLAY_PAD = 2
    _PLOT_PANEL_GAP = 2
    _FIGURE_DPI = 100
    _MIN_PLOT_ROW_HEIGHT = 300
    _PLOT_COLUMNS_UNIFORM = "pychmp_plot_columns"
    _ACTIVE_REFRESH_GRACE_S = 10.0
    _MIN_LEFT_PANEL_WIDTH = 420
    _MIN_CENTER_PANEL_WIDTH = 520
    _MIN_RIGHT_PANEL_WIDTH = 300
    _MIN_TOOLBAR_WIDTH = 1080

    def __init__(
        self,
        root: tk.Tk,
        artifact_h5: Path | None,
        *,
        initial_metric: str | None = None,
    ) -> None:
        self.root = root
        if artifact_h5 is None:
            self.artifact_h5 = None
        else:
            try:
                self.artifact_h5 = artifact_h5.expanduser().resolve()
            except Exception:
                self.artifact_h5 = Path(artifact_h5).expanduser()
        self.payload = {}
        self.a_values = np.asarray([], dtype=float)
        self.b_values = np.asarray([], dtype=float)
        self.display_model: dict[str, Any] = {"records": []}
        self.run_target_metric = "chi2"
        self._preferred_initial_metric = initial_metric if initial_metric in METRICS else None
        self.metric_var = tk.StringVar(value=self._preferred_initial_metric or "chi2")
        self._last_rendered_metric = self.metric_var.get() if self.metric_var.get() in METRICS else "chi2"
        self.slice_key_var = tk.StringVar(value="")
        self.slice_display_var = tk.StringVar(value="")
        self.search_id_var = tk.StringVar(value="")
        self.search_display_var = tk.StringVar(value="")
        self.trials_xmin_var = tk.StringVar(value="")
        self.trials_xmax_var = tk.StringVar(value="")
        self.trials_ymin_var = tk.StringVar(value="")
        self.trials_ymax_var = tk.StringVar(value="")
        self._trials_xmin_manual = False
        self._trials_xmax_manual = False
        self._trials_ymin_manual = False
        self._trials_ymax_manual = False
        self.trials_xscale_var = tk.StringVar(value="linear scale")
        self.trials_yscale_var = tk.StringVar(value="linear scale")
        self.trial_index_var = tk.IntVar(value=0)
        self.trial_label_var = tk.StringVar(value="trial: n/a")
        self.shared_heatmap_axes_var = tk.BooleanVar(value=_load_shared_grid_axes_pref())
        self._shared_heatmap_extents: dict[str, float] | None = None
        self._open_global_best_applied = False
        self.a_index_var = tk.IntVar(value=0)
        self.b_index_var = tk.IntVar(value=0)
        self.scan_state_var = tk.StringVar(value="NO ARTIFACT")
        self.scan_state_detail_var = tk.StringVar(value="No artifact loaded")
        self.scan_state_info_var = tk.StringVar(value="No artifact loaded")
        self.status_var = tk.StringVar(value="")
        self.summary_var = tk.StringVar(value="")
        self.slice_display_state: dict[str, dict[str, Any]] = {}
        self.available_slices: list[dict[str, Any]] = []
        self.available_searches: list[dict[str, Any]] = []
        self.artifact_search_catalog: list[dict[str, Any]] = []
        self._open_session_prefers_active = False
        self.last_artifact_dir = _load_last_directory()
        self.refresh_signal_path: Path | None = None
        self._refresh_signal_mtime_ns = -1
        self._refresh_signal_phase = ""
        self._refresh_signal_slice_key: str | None = None
        self._refresh_signal_search_id: str | None = None
        self._refresh_signal_pending_points: list[tuple[float, float]] = []
        self._free_selection_ab: tuple[float, float] | None = None
        self._refresh_signal_active_point: tuple[float, float] | None = None
        self._refresh_signal_live_trials: dict[str, Any] | None = None
        self._refresh_signal_event: str = ""
        self._refresh_signal_point_id: str | None = None
        self._active_follow_point_id: str | None = None
        self._best_tied_records: list[dict[str, Any]] = []
        self._external_refresh_after_id: str | None = None
        self._last_payload_reload_at_s = 0.0
        self._payload_cache_artifact_path = ""
        self._payload_cache_artifact_mtime_ns = -1
        self._payload_cache_by_selection: dict[tuple[str, str], dict[str, Any]] = {}
        self._scheduled_reload_after_id: str | None = None
        self._initial_reload_after_id: str | None = None
        self._initial_reload_in_progress = False
        self._initial_reload_token = 0
        self._present_window_after_id: str | None = None
        self._selected_solution_update_after_id: str | None = None
        self._psf_metadata_cache: dict[str, PSFMetadata | None] = {}
        self._psf_kernel_cache: dict[tuple[Any, ...], np.ndarray] = {}
        self._live_trial_render_cache_key: tuple[Any, ...] | None = None
        self._live_trial_render_cache_value: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._live_snapshot_cache_key: tuple[Any, ...] | None = None
        self._live_snapshot_cache_value: dict[str, Any] | None = None
        self._selected_trial_map_cache_key: tuple[Any, ...] | None = None
        self._selected_trial_map_cache_value: dict[str, Any] | None = None
        self.refresh_button: ttk.Button | None = None
        self.metric_menu: ttk.Combobox | None = None
        self.navigation_mode_var = tk.StringVar(value="free")
        self._navigation_mode_initialized = False
        self._navigation_mode_user_chosen = False
        self._navigation_mode_radios: dict[str, ttk.Radiobutton] = {}
        self._stale_active_notice = ""
        self.scan_state_badge: tk.Label | None = None
        self.scan_state_label: ttk.Label | None = None
        self.scan_state_info_label: ttk.Label | None = None
        self.left_notebook: ttk.Notebook | None = None
        self.center_notebook: ttk.Notebook | None = None
        self.info_notebook: ttk.Notebook | None = None
        self._plot_canvas_resize_after_id: str | None = None
        self.selected_solution_window: _SelectedSolutionWindow | None = None
        self._selected_trial_token: tuple[int, int, str, int] | None = None
        self._updating_trial_slider = False
        self._applying_navigation_selection = False
        self._is_closing = False

        self._bind_refresh_signal_path()
        self._refresh_artifact_catalog()
        self._apply_open_session_defaults()
        self.root.title(f"pychmp-view: {artifact_h5.name}" if artifact_h5 is not None else "pychmp-view")
        self.root.protocol("WM_DELETE_WINDOW", self._close_root_window)
        self._configure_initial_window_geometry()
        self._reset_icon_image = _make_reset_icon_image(self.root)
        self._best_trial_icon_image = _make_best_trial_icon_image(self.root)

        self._build_ui()
        self._refresh_all()
        self._external_refresh_after_id = self.root.after(self._EXTERNAL_REFRESH_POLL_MS, self._poll_external_refresh_signal)
        self._present_window_after_id = self.root.after(50, self._present_window)

    def _configure_initial_window_geometry(self) -> None:
        screen_w = max(1, int(self.root.winfo_screenwidth()))
        screen_h = max(1, int(self.root.winfo_screenheight()))

        min_body_width = self._MIN_LEFT_PANEL_WIDTH + self._MIN_CENTER_PANEL_WIDTH + self._MIN_RIGHT_PANEL_WIDTH + 32
        max_width = max(min_body_width, screen_w - 40)
        width = min(max(max(self._MIN_TOOLBAR_WIDTH, min_body_width), int(round(screen_w * 0.88))), max_width)
        self._update_panel_widths(width - 16)

        # Height is derived from required vertical stack: shared toolbar + tab header +
        # plot canvas region + fixed tab footer + outer chrome/padding.
        required_height = (
            16  # outer frame vertical padding (8 top + 8 bottom)
            + self._TOOLBAR_HEIGHT
            + 6  # gap below toolbar
            + self._NOTEBOOK_TAB_HEIGHT
            + self._MIN_PLOT_ROW_HEIGHT
            + 2 * self._DISPLAY_PAD
            + max(self._HEATMAP_FOOTER_HEIGHT, self._TRIALS_FOOTER_HEIGHT)
            + self._WINDOW_VERTICAL_CHROME
        )
        height = min(max(560, required_height), max(560, screen_h - 40))

        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.minsize(max(self._MIN_TOOLBAR_WIDTH, min_body_width), height)
        self.root.resizable(True, True)

    def _update_panel_widths(self, total_width: int) -> None:
        available_width = max(
            self._MIN_LEFT_PANEL_WIDTH + self._MIN_CENTER_PANEL_WIDTH + self._MIN_RIGHT_PANEL_WIDTH + 8,
            int(total_width),
        )
        right_width = max(self._MIN_RIGHT_PANEL_WIDTH, min(360, int(round(available_width * 0.24))))
        left_center_width = max(
            self._MIN_LEFT_PANEL_WIDTH + self._MIN_CENTER_PANEL_WIDTH,
            available_width - right_width - 8,
        )
        plot_column_width = max(
            self._MIN_LEFT_PANEL_WIDTH,
            self._MIN_CENTER_PANEL_WIDTH,
            int(left_center_width // 2),
        )
        self.left_panel_width = int(plot_column_width)
        self.center_panel_width = int(plot_column_width)
        self.right_panel_width = int(right_width)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(outer)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        toolbar.columnconfigure(0, weight=1)

        primary_toolbar = ttk.Frame(toolbar)
        primary_toolbar.grid(row=0, column=0, sticky="ew")
        primary_toolbar.columnconfigure(9, weight=1)

        secondary_toolbar = ttk.Frame(toolbar)
        secondary_toolbar.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        secondary_toolbar.columnconfigure(7, weight=1)

        ttk.Label(primary_toolbar, text="Metric").grid(row=0, column=0, sticky="w")
        metric_menu = ttk.Combobox(primary_toolbar, width=8, state="readonly", values=list(METRICS), textvariable=self.metric_var)
        metric_menu.grid(row=0, column=1, sticky="w", padx=(6, 8))
        metric_menu.bind("<<ComboboxSelected>>", lambda _event: self._on_metric_changed())
        self.metric_menu = metric_menu

        navigation_frame = ttk.Frame(primary_toolbar)
        navigation_frame.grid(row=0, column=2, sticky="w", padx=(0, 10))
        for column, (mode, label) in enumerate(
            (
                ("free", "Free"),
                ("active", "Active"),
                ("best", "Best"),
            )
        ):
            radio = ttk.Radiobutton(
                navigation_frame,
                text=label,
                value=mode,
                variable=self.navigation_mode_var,
                command=self._on_navigation_mode_changed,
            )
            radio.grid(row=0, column=column, sticky="w", padx=(0 if column == 0 else 6, 0))
            self._navigation_mode_radios[mode] = radio

        ttk.Label(primary_toolbar, text="Slice").grid(row=0, column=4, sticky="w")
        self.slice_menu = ttk.Combobox(primary_toolbar, width=18, state="readonly")
        self.slice_menu.grid(row=0, column=5, sticky="w", padx=(6, 6))
        self.slice_menu.bind("<<ComboboxSelected>>", lambda _event: self._on_slice_changed())
        self.slice_display_label = ttk.Label(primary_toolbar, textvariable=self.slice_display_var, anchor=tk.W)
        self.slice_display_label.grid(row=0, column=5, sticky="w", padx=(6, 6))

        ttk.Label(primary_toolbar, text="Search").grid(row=0, column=6, sticky="w")
        self.search_menu = ttk.Combobox(primary_toolbar, width=28, state="readonly")
        self.search_menu.grid(row=0, column=7, sticky="w", padx=(6, 10))
        self.search_menu.bind("<<ComboboxSelected>>", lambda _event: self._on_search_changed())
        self.search_display_label = ttk.Label(primary_toolbar, textvariable=self.search_display_var, anchor=tk.W)
        self.search_display_label.grid(row=0, column=7, sticky="w", padx=(6, 10))

        ttk.Label(primary_toolbar, text="a").grid(row=0, column=8, sticky="w")
        self.a_menu = ttk.Combobox(primary_toolbar, width=10, state="readonly")
        self.a_menu.configure(width=10)
        self.a_menu.grid(row=0, column=9, sticky="w", padx=(6, 10))
        self.a_menu.bind("<<ComboboxSelected>>", lambda _event: self._on_a_changed())

        ttk.Label(primary_toolbar, text="b").grid(row=0, column=10, sticky="w")
        self.b_menu = ttk.Combobox(primary_toolbar, width=10, state="readonly")
        self.b_menu.configure(width=10)
        self.b_menu.grid(row=0, column=11, sticky="w", padx=(6, 12))
        self.b_menu.bind("<<ComboboxSelected>>", lambda _event: self._on_b_changed())

        ttk.Separator(secondary_toolbar, orient=tk.VERTICAL).grid(row=0, column=0, sticky="ns", padx=(0, 10))

        open_artifact_button = ttk.Button(secondary_toolbar, text="📂", width=3, command=self._open_artifact)
        open_artifact_button.grid(row=0, column=1, sticky="w", padx=(0, 6))
        self.open_artifact_button = open_artifact_button
        display_selected_button = ttk.Button(secondary_toolbar, text="🖼", width=3, command=self._open_selected_maps)
        display_selected_button.grid(row=0, column=2, sticky="w", padx=(0, 6))
        self.display_selected_button = display_selected_button
        summary_button = ttk.Button(secondary_toolbar, text="▦", width=3, command=self._open_grid_summary)
        summary_button.grid(row=0, column=3, sticky="w", padx=(0, 6))
        self.summary_button = summary_button
        refresh_button = ttk.Button(secondary_toolbar, text="↻", width=3, command=self._reload_payload)
        refresh_button.grid(row=0, column=4, sticky="w")
        self.refresh_button = refresh_button
        ttk.Separator(secondary_toolbar, orient=tk.VERTICAL).grid(row=0, column=5, sticky="ns", padx=(10, 10))
        scan_state_badge = tk.Label(
            secondary_toolbar,
            textvariable=self.scan_state_var,
            bg="#6c757d",
            fg="white",
            font=("TkDefaultFont", 9, "bold"),
            padx=8,
            pady=3,
        )
        scan_state_badge.grid(row=0, column=6, sticky="w")
        self.scan_state_badge = scan_state_badge
        scan_state_label = ttk.Label(secondary_toolbar, textvariable=self.scan_state_detail_var, anchor=tk.W)
        scan_state_label.grid(row=0, column=7, sticky="ew", padx=(8, 0))
        self.scan_state_label = scan_state_label
        _ToolTip(open_artifact_button, "Open Artifact: choose a consolidated pyCHMP scan H5 artifact file.")
        _ToolTip(display_selected_button, "Display Selected Solution: open or refresh the persistent detailed solution window for the current (a, b) point.")
        _ToolTip(summary_button, "Display Grid Summary: open the external grid-summary plot for the current artifact.")
        _ToolTip(refresh_button, "Refresh Artifact: reload the current artifact and update the viewer if the scan has advanced.")
        _ToolTip(scan_state_badge, "Scan status: indicates whether the current slice appears to be running, finished, incomplete, or empty. Full details are shown in the Info pane.")

        content = ttk.Frame(outer)
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(
            0,
            minsize=self._MIN_LEFT_PANEL_WIDTH + self._MIN_CENTER_PANEL_WIDTH + self._PLOT_PANEL_GAP,
            weight=1,
        )
        content.columnconfigure(1, minsize=self.right_panel_width, weight=0)
        content.columnconfigure(2, weight=0)
        content.rowconfigure(0, weight=1)

        plot_area = ttk.Frame(content)
        plot_area.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        plot_area.columnconfigure(0, weight=1, uniform=self._PLOT_COLUMNS_UNIFORM)
        plot_area.columnconfigure(1, weight=1, uniform=self._PLOT_COLUMNS_UNIFORM)
        plot_area.rowconfigure(0, weight=1)

        info_panel = ttk.Frame(content)
        info_panel.grid(row=0, column=1, sticky="nsew")
        info_panel.columnconfigure(0, minsize=self.right_panel_width, weight=0)
        info_panel.rowconfigure(0, weight=1)

        info_notebook = ttk.Notebook(info_panel)
        info_notebook.grid(row=0, column=0, sticky="nsew")
        self.info_notebook = info_notebook
        self.info_notebook.configure(width=self.right_panel_width)

        info_tab = ttk.Frame(info_notebook)
        info_tab.columnconfigure(0, weight=1)
        info_tab.columnconfigure(1, weight=0)
        info_tab.rowconfigure(0, weight=0)
        info_tab.rowconfigure(1, weight=1)
        self.info_tab = info_tab
        info_notebook.add(info_tab, text="Info")

        info_actions = ttk.Frame(info_tab)
        info_actions.grid(row=0, column=0, columnspan=2, sticky="ew", padx=(4, 4), pady=(4, 0))
        info_actions.columnconfigure(0, weight=1)
        self.info_actions_menu_button = ttk.Menubutton(info_actions, text="...", width=3)
        self.info_actions_menu_button.grid(row=0, column=1, sticky="e")
        info_actions_menu = tk.Menu(self.info_actions_menu_button, tearoff=False)
        info_actions_menu.add_command(label="Select All", command=self._info_select_all)
        info_actions_menu.add_command(label="Copy Selection", command=self._info_copy_selection)
        info_actions_menu.add_command(label="Copy All", command=self._info_copy_all)
        self.info_actions_menu_button.configure(menu=info_actions_menu)
        self.info_actions_menu = info_actions_menu

        info_text = tk.Text(
            info_tab,
            wrap=tk.WORD,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
            font=("TkDefaultFont", 10),
            width=1,
            height=1,
            padx=8,
            pady=8,
            cursor="arrow",
            undo=False,
        )
        info_text.grid(row=1, column=0, sticky="nsew")
        info_scrollbar = ttk.Scrollbar(info_tab, orient=tk.VERTICAL, command=info_text.yview)
        info_scrollbar.grid(row=1, column=1, sticky="ns")
        info_text.configure(yscrollcommand=info_scrollbar.set)
        self.info_text = info_text
        self.info_scrollbar = info_scrollbar
        info_text.bind("<KeyPress>", self._on_info_text_keypress)
        info_text.bind("<Command-c>", lambda _event: (self._info_copy_selection(), "break")[1])
        info_text.bind("<Control-c>", lambda _event: (self._info_copy_selection(), "break")[1])
        info_text.tag_configure("heading", font=("TkDefaultFont", 10, "bold"), spacing1=2, spacing3=4)
        info_text.tag_configure("body", font=("TkDefaultFont", 10))

        left_notebook = ttk.Notebook(plot_area)
        half_gap = max(1, self._PLOT_PANEL_GAP // 2)
        left_notebook.grid(row=0, column=0, sticky="nsew", padx=(0, half_gap))
        self.left_notebook = left_notebook
        right_notebook = ttk.Notebook(plot_area)
        right_notebook.grid(row=0, column=1, sticky="nsew", padx=(half_gap, 0))
        self.center_notebook = right_notebook

        heatmap_tab = ttk.Frame(left_notebook)
        heatmap_tab.columnconfigure(0, weight=1)
        heatmap_tab.rowconfigure(0, weight=1)
        heatmap_tab.rowconfigure(1, minsize=self._HEATMAP_FOOTER_HEIGHT, weight=0)
        self.heatmap_tab = heatmap_tab
        left_notebook.add(heatmap_tab, text="Grid Metric")

        trials_tab = ttk.Frame(right_notebook)
        trials_tab.columnconfigure(0, weight=1)
        trials_tab.rowconfigure(0, weight=1)
        trials_tab.rowconfigure(1, minsize=self._TRIALS_FOOTER_HEIGHT, weight=0)
        self.trials_tab = trials_tab
        right_notebook.add(trials_tab, text="Trials")
        self.q0_trials_tab = trials_tab

        self.heatmap_figure = Figure(figsize=(6.0, 4.0), dpi=self._FIGURE_DPI)
        self.ax_heatmap = self.heatmap_figure.add_subplot(111)

        self.trials_figure = Figure(figsize=(6.0, 4.0), dpi=self._FIGURE_DPI)
        self.ax_trials = self.trials_figure.add_subplot(111)
        self._heatmap_colorbar = None
        self.heatmap_canvas = FigureCanvasTkAgg(self.heatmap_figure, master=heatmap_tab)
        self.heatmap_canvas_widget = self.heatmap_canvas.get_tk_widget()
        self.heatmap_canvas_widget.grid(row=0, column=0, sticky="nsew", padx=self._DISPLAY_PAD, pady=self._DISPLAY_PAD)
        self.heatmap_canvas_widget.bind("<Configure>", lambda _event: self._schedule_plot_canvas_resize())
        self.heatmap_canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self.heatmap_footer = ttk.Frame(heatmap_tab)
        self.heatmap_footer.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        self.heatmap_footer.columnconfigure(0, weight=1)

        self.heatmap_legend_row = ttk.Frame(self.heatmap_footer)
        self.heatmap_legend_row.grid(row=0, column=0, sticky="ew")
        self._build_heatmap_legend_row()

        shared_axes_row = ttk.Frame(self.heatmap_footer)
        shared_axes_row.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        self._build_shared_grid_axes_control(shared_axes_row)

        trial_status = ttk.Label(self.heatmap_footer, textvariable=self.trial_label_var, anchor="w", justify=tk.LEFT)
        trial_status.grid(row=2, column=0, sticky="ew", padx=(2, 0), pady=(2, 0))
        self.trial_status_label = trial_status

        self.trials_canvas = FigureCanvasTkAgg(self.trials_figure, master=trials_tab)
        self.trials_canvas_widget = self.trials_canvas.get_tk_widget()
        self.trials_canvas_widget.grid(row=0, column=0, sticky="nsew", padx=self._DISPLAY_PAD, pady=self._DISPLAY_PAD)
        self.trials_canvas_widget.bind("<Configure>", lambda _event: self._schedule_plot_canvas_resize())
        self.trials_canvas.mpl_connect("button_press_event", self._on_trials_canvas_click)
        trials_controls = ttk.Frame(trials_tab)
        trials_controls.grid(row=1, column=0, sticky="ew", padx=self._DISPLAY_PAD, pady=(0, 4))
        self.trials_controls = trials_controls
        self._build_trials_controls()

        self.root.bind("<Configure>", self._on_resize)

        if self.artifact_h5 is not None:
            self.status_var.set("Loading artifact; controls will populate shortly...")
            self.summary_var.set("Loading artifact...")
            self._refresh_all()
            # Let the window realize first so startup remains responsive.
            self._initial_reload_after_id = self.root.after(150, self._run_initial_payload_reload)
        else:
            self.status_var.set("No artifact loaded. Use Open Artifact to choose a consolidated scan H5 file.")
            self.summary_var.set("No artifact loaded.")
            self._refresh_all()
        self._refresh_selector_values()
        self.root.after_idle(self._schedule_plot_canvas_resize)

    def _run_initial_payload_reload(self) -> None:
        self._initial_reload_after_id = None
        if self._is_closing or self.artifact_h5 is None:
            return
        if self._initial_reload_in_progress:
            return
        self._initial_reload_in_progress = True
        self._initial_reload_token += 1
        token = int(self._initial_reload_token)
        requested_slice_key = self._selected_slice_key()
        requested_search_id = self._selected_search_id()
        try:
            artifact_path = Path(self.artifact_h5).expanduser().resolve()
        except Exception:
            artifact_path = Path(self.artifact_h5)
        self.status_var.set("Loading artifact data in background...")
        self.summary_var.set("Loading artifact...")
        self._refresh_all()

        def _worker() -> None:
            payload: dict[str, Any] | None = None
            error: Exception | None = None
            try:
                last_exc: Exception | None = None
                for attempt in range(1, self._LOAD_RETRY_ATTEMPTS + 1):
                    try:
                        payload = load_scan_file(
                            artifact_path,
                            slice_key=requested_slice_key,
                            search_id=requested_search_id,
                            include_maps=False,
                        )
                        break
                    except (BlockingIOError, PermissionError, OSError) as exc:
                        last_exc = exc
                        if attempt >= self._LOAD_RETRY_ATTEMPTS:
                            break
                        time.sleep(self._LOAD_RETRY_DELAY_S)
                if payload is None and last_exc is not None:
                    raise last_exc
                if payload is None:
                    raise RuntimeError("unknown artifact loading failure")
            except Exception as exc:
                error = exc

            def _apply_result() -> None:
                if self._is_closing or int(self._initial_reload_token) != token:
                    return
                self._initial_reload_in_progress = False
                if error is not None:
                    if isinstance(error, FileNotFoundError):
                        self.status_var.set(
                            "Artifact file could not be found at the requested path.\n"
                            f"Path: {artifact_path}\n"
                            f"Details: {error}"
                        )
                        self.summary_var.set("Artifact path is missing or inaccessible. Use Open Artifact to reselect it.")
                        self._refresh_action_states()
                        self._refresh_all()
                        return
                    self.status_var.set(
                        "Artifact is currently being written or is temporarily locked. "
                        "Please retry in a moment.\n"
                        f"Details: {error}"
                    )
                    self.summary_var.set("Could not refresh artifact right now. Existing view remains unchanged.")
                    self._refresh_action_states()
                    self._refresh_all()
                    return
                assert payload is not None
                selected_cache_key = (
                    str(payload.get("selected_slice_key", "") or ""),
                    str(payload.get("selected_search_id", "") or ""),
                )
                requested_cache_key = (
                    str(requested_slice_key or ""),
                    str(requested_search_id or ""),
                )
                self._payload_cache_by_selection[requested_cache_key] = payload
                self._payload_cache_by_selection[selected_cache_key] = payload
                self._apply_payload(payload)

            try:
                self.root.after(0, _apply_result)
            except Exception:
                pass

        threading.Thread(target=_worker, name="pychmp-view-initial-load", daemon=True).start()

    def _present_window(self) -> None:
        self._present_window_after_id = None
        if self._is_closing:
            return
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
            # Briefly mark the window topmost so macOS brings it forward.
            self.root.attributes("-topmost", True)
            self.root.after(200, lambda: self.root.attributes("-topmost", False))
            self.root.after(120, self._schedule_plot_canvas_resize)
        except Exception:
            pass

    def _schedule_selected_solution_update(self) -> None:
        if self.selected_solution_window is None or self._is_closing:
            return
        if self._selected_solution_update_after_id is not None:
            try:
                self.root.after_cancel(self._selected_solution_update_after_id)
            except Exception:
                pass
            self._selected_solution_update_after_id = None

        def _run() -> None:
            self._selected_solution_update_after_id = None
            if self.selected_solution_window is None or self._is_closing:
                return
            self.selected_solution_window.update_selection()

        self._selected_solution_update_after_id = self.root.after_idle(_run)

    def _close_root_window(self) -> None:
        self._is_closing = True
        for after_id in (
            self._external_refresh_after_id,
            self._scheduled_reload_after_id,
            self._initial_reload_after_id,
            self._present_window_after_id,
            self._selected_solution_update_after_id,
            self._plot_canvas_resize_after_id,
        ):
            if after_id is None:
                continue
            try:
                self.root.after_cancel(after_id)
            except Exception:
                pass
        self._external_refresh_after_id = None
        self._scheduled_reload_after_id = None
        self._initial_reload_after_id = None
        self._initial_reload_in_progress = False
        self._present_window_after_id = None
        self._selected_solution_update_after_id = None
        if self.selected_solution_window is not None:
            try:
                self.selected_solution_window.window.destroy()
            except Exception:
                pass
            self.selected_solution_window = None
        try:
            self.root.destroy()
        except Exception:
            pass

    def _on_resize(self, _event: Any) -> None:
        self._update_panel_widths(int(self.root.winfo_width()) - 16)
        self._apply_layout_geometry()

    def _apply_layout_geometry(self) -> None:
        if self.info_notebook is not None:
            self.info_notebook.configure(width=self.right_panel_width)
        self._schedule_plot_canvas_resize()

    def _schedule_plot_canvas_resize(self) -> None:
        if getattr(self, "_is_closing", False):
            return
        root = getattr(self, "root", None)
        if root is None:
            return
        after_id = getattr(self, "_plot_canvas_resize_after_id", None)
        if after_id is not None:
            try:
                root.after_cancel(after_id)
            except Exception:
                pass
        self._plot_canvas_resize_after_id = root.after(40, self._sync_plot_canvas_sizes)

    def _sync_plot_canvas_sizes(self) -> None:
        self._plot_canvas_resize_after_id = None
        if getattr(self, "_is_closing", False):
            return
        if getattr(self, "heatmap_canvas", None) is not None:
            self._resize_figure_to_canvas_widget(self.heatmap_canvas, self.heatmap_figure)
        if getattr(self, "trials_canvas", None) is not None:
            self._resize_figure_to_canvas_widget(self.trials_canvas, self.trials_figure)

    def _resize_figure_to_canvas_widget(self, canvas: FigureCanvasTkAgg, figure: Figure) -> None:
        widget = canvas.get_tk_widget()
        width_px = int(widget.winfo_width())
        height_px = int(widget.winfo_height())
        if width_px < 80 or height_px < 80:
            return
        dpi = float(figure.get_dpi())
        figure.set_size_inches(width_px / dpi, height_px / dpi, forward=False)
        canvas.draw_idle()

    def _reset_heatmap_colorbar(self) -> None:
        if self._heatmap_colorbar is None:
            return
        try:
            self._heatmap_colorbar.remove()
        except Exception:
            pass
        self._heatmap_colorbar = None

    def _refresh_info_text(self) -> None:
        if self.info_text is None:
            return
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, "Scan State\n", "heading")
        self.info_text.insert(tk.END, f"{self._wrap_info_block(self.scan_state_info_var.get())}\n\n", "body")
        self.info_text.insert(tk.END, "Artifact\n", "heading")
        self.info_text.insert(tk.END, f"{self._wrap_info_block(self.status_var.get())}\n\n", "body")
        self.info_text.insert(tk.END, "Selected Point\n", "heading")
        self.info_text.insert(tk.END, f"{self._wrap_info_block(self.summary_var.get())}\n", "body")

    def _on_info_text_keypress(self, event: tk.Event) -> str | None:
        # Keep the pane read-only while preserving navigation and selection.
        if event.keysym in {
            "Up",
            "Down",
            "Left",
            "Right",
            "Prior",
            "Next",
            "Home",
            "End",
            "Shift_L",
            "Shift_R",
            "Control_L",
            "Control_R",
            "Command",
            "Meta_L",
            "Meta_R",
        }:
            return None
        if (event.state & 0x4) or (event.state & 0x8):
            return None
        if event.keysym in {"BackSpace", "Delete", "Return", "KP_Enter", "Tab"}:
            return "break"
        if event.char:
            return "break"
        return None

    def _info_select_all(self) -> None:
        if self.info_text is None:
            return
        self.info_text.focus_set()
        self.info_text.tag_add(tk.SEL, "1.0", "end-1c")
        self.info_text.mark_set(tk.INSERT, "1.0")
        self.info_text.see(tk.INSERT)

    def _copy_text_to_clipboard(self, text: str) -> None:
        if not text:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update_idletasks()

    def _info_copy_selection(self) -> None:
        if self.info_text is None:
            return
        try:
            text = self.info_text.get(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            return
        self._copy_text_to_clipboard(str(text))

    def _info_copy_all(self) -> None:
        if self.info_text is None:
            return
        text = self.info_text.get("1.0", "end-1c")
        self._copy_text_to_clipboard(str(text))

    def _wrap_info_block(self, text: str) -> str:
        wrapped_lines: list[str] = []
        width = self._info_wrap_width_chars()
        for raw_line in str(text).splitlines():
            if not raw_line.strip():
                wrapped_lines.append("")
                continue
            wrapped_lines.extend(
                textwrap.wrap(
                    raw_line,
                    width=width,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                or [raw_line]
            )
        return "\n".join(wrapped_lines)

    def _info_wrap_width_chars(self) -> int:
        if self.info_text is not None:
            try:
                widget_width = int(self.info_text.winfo_width())
                if widget_width > 40:
                    return max(28, int((widget_width - 24) / 7.2))
            except Exception:
                pass
        return max(28, int((self.right_panel_width - 24) / 7.2))

    def _navigation_mode(self) -> str:
        mode = str(getattr(self, "navigation_mode_var", None).get() if getattr(self, "navigation_mode_var", None) is not None else "free").strip().lower()
        return mode if mode in NAVIGATION_MODES else "free"

    def _slice_and_search_selection_locked(self) -> bool:
        return self._navigation_mode() in {"active", "best"}

    def _grid_point_selection_locked(self) -> bool:
        return self._navigation_mode() in {"active", "best"}

    def _metric_selection_locked(self) -> bool:
        return self._navigation_mode() == "active"

    def _navigation_is_locked(self) -> bool:
        return self._grid_point_selection_locked()

    def _payload_has_computed_grid(self) -> bool:
        model = dict(getattr(self, "display_model", {}) or {})
        for record in list(model.get("records", [])):
            status = str(record.get("status", "computed")).strip().lower()
            if status in {"missing", "pending"} or bool(record.get("live_pending")):
                continue
            return True
        points = dict((getattr(self, "payload", {}) or {}).get("points", {}))
        for point in points.values():
            status = str(point.get("status", "computed")).strip().lower()
            if status not in {"missing", "pending"}:
                return True
        return False

    def _followed_point_coords(self) -> tuple[float, float] | None:
        followed_point_id = str(getattr(self, "_active_follow_point_id", None) or "").strip()
        if not followed_point_id:
            return None
        return self._grid_point_coords_for_id(followed_point_id)

    def _best_point_index(self, metric: str | None = None) -> tuple[int, int] | None:
        if not getattr(self, "payload", None):
            return None
        search_metric = str(metric or self._run_target_metric_for_selection()).strip().lower()
        selection = best_point_selection(self.payload, search_metric)
        if selection is None:
            self._best_tied_records = []
            return None
        self._best_tied_records = list(selection.tied_records)
        return int(selection.a_index), int(selection.b_index)

    def _run_target_metric_for_selection(self) -> str:
        metric = str(getattr(self, "run_target_metric", "") or "").strip().lower()
        if metric in METRICS:
            return metric
        selected = dict((getattr(self, "payload", {}) or {}).get("selected_search") or {})
        metric = str(selected.get("target_metric") or "").strip().lower()
        if metric in METRICS:
            return metric
        return "chi2"

    def _execution_is_serial(self) -> bool:
        diagnostics = dict((getattr(self, "payload", {}) or {}).get("diagnostics") or {})
        return execution_is_serial(diagnostics)

    def _heatmap_display_metric(self) -> str:
        if self._navigation_mode() == "active":
            return self._run_target_metric_for_selection()
        metric_var = getattr(self, "metric_var", None)
        metric = str(metric_var.get() if metric_var is not None else "").strip().lower()
        return metric if metric in METRICS else self._run_target_metric_for_selection()

    def _trials_display_metric(self) -> str:
        if self._navigation_mode() == "active":
            return self._run_target_metric_for_selection()
        metric_var = getattr(self, "metric_var", None)
        metric = str(metric_var.get() if metric_var is not None else "").strip().lower()
        return metric if metric in METRICS else self._run_target_metric_for_selection()

    def _refresh_phase_is_terminal(self, phase: str | None = None) -> bool:
        text = str(phase if phase is not None else getattr(self, "_refresh_signal_phase", "") or "").strip().lower()
        if not text:
            return False
        if text in {"scan complete", "complete"}:
            return True
        return text in {
            "scan interrupted",
            "interrupted",
            "scan aborted",
            "aborted",
            "scan failed",
            "failed",
        }

    def _refresh_event_is_terminal(self, event: str | None = None) -> bool:
        name = str(event if event is not None else getattr(self, "_refresh_signal_event", "") or "").strip().lower()
        return name in {"search_completed", "search_failed"}

    def _search_run_terminal(self) -> bool:
        if self._refresh_event_is_terminal() or self._refresh_phase_is_terminal():
            return True
        payload = getattr(self, "payload", None) or {}
        if not payload:
            return False
        selected = dict(payload.get("selected_search") or {})
        lifecycle = dict(selected.get("lifecycle") or {})
        status = str(selected.get("status", "") or "").strip().lower()
        if status in {"complete", "completed", "failed", "aborted", "interrupted"}:
            return True
        if str(lifecycle.get("completed_at") or "").strip():
            return True
        return False

    def _clear_live_runner_state(self) -> None:
        self._refresh_signal_active_point = None
        self._refresh_signal_live_trials = None
        self._active_follow_point_id = None
        self._refresh_signal_point_id = None

    def _refresh_signal_is_fresh(self) -> bool:
        refresh_signal_path = getattr(self, "refresh_signal_path", None)
        if refresh_signal_path is None or not refresh_signal_path.exists():
            return False
        try:
            refresh_age_s = max(0.0, time.time() - float(refresh_signal_path.stat().st_mtime))
            return refresh_age_s <= float(self._ACTIVE_REFRESH_GRACE_S)
        except Exception:
            return False

    def _runner_pid_from_log(self) -> int | None:
        artifact_h5 = getattr(self, "artifact_h5", None)
        if artifact_h5 is None:
            return None
        log_path = Path(f"{artifact_h5}.log")
        if not log_path.exists():
            return None
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None
        matches = re.findall(r"pid=(\d+)", text)
        if not matches:
            return None
        try:
            return int(matches[-1])
        except Exception:
            return None

    def _process_is_running(self, pid: int) -> bool:
        try:
            os.kill(int(pid), 0)
            return True
        except OSError:
            return False

    def _artifact_marks_active_search(self) -> bool:
        for record in list(getattr(self, "available_searches", []) or []):
            lifecycle = dict(record.get("lifecycle") or {})
            if bool(record.get("active", lifecycle.get("active", False))):
                return True
        selected = dict((getattr(self, "payload", {}) or {}).get("selected_search") or {})
        lifecycle = dict(selected.get("lifecycle") or {})
        return bool(selected.get("active", lifecycle.get("active", False)))

    def _live_runner_detected(self) -> bool:
        if self._search_run_terminal():
            return False
        if self._refresh_signal_is_fresh():
            return True
        pid = self._runner_pid_from_log()
        return pid is not None and self._process_is_running(pid)

    def _bind_refresh_signal_path(self) -> None:
        artifact_h5 = getattr(self, "artifact_h5", None)
        if artifact_h5 is None:
            self.refresh_signal_path = None
            return
        self.refresh_signal_path = Path(f"{artifact_h5}.refresh")

    def _sync_refresh_signal_from_disk(self) -> None:
        self._bind_refresh_signal_path()
        if self.refresh_signal_path is None or not self.refresh_signal_path.exists():
            return
        try:
            self._refresh_signal_mtime_ns = int(self.refresh_signal_path.stat().st_mtime_ns)
            refresh_payload = self._read_refresh_signal_payload()
            self._refresh_signal_phase = str(refresh_payload.get("phase", ""))
            self._apply_refresh_signal_payload(refresh_payload)
            if getattr(self, "payload", None):
                self._sync_live_trial_state_from_artifact()
        except Exception:
            pass

    def _heartbeat_slice_and_search(self) -> tuple[str | None, str | None]:
        live_slice_key = str(getattr(self, "_refresh_signal_slice_key", None) or "").strip() or None
        live_search_id = str(getattr(self, "_refresh_signal_search_id", None) or "").strip() or None
        return live_slice_key, live_search_id

    def _search_id_in_artifact(self, search_id: str | None, *, slice_key: str | None = None) -> bool:
        key = str(search_id or "").strip()
        if not key:
            return False
        slice_filter = str(slice_key or "").strip()
        for record in list(getattr(self, "artifact_search_catalog", []) or []):
            if str(record.get("search_id", "")).strip() != key:
                continue
            record_slice = str(record.get("slice_key", "")).strip()
            if slice_filter and record_slice and record_slice != slice_filter:
                continue
            return True
        for record in list(getattr(self, "available_searches", []) or []):
            if str(record.get("search_id", "")).strip() == key:
                return True
        payload = getattr(self, "payload", None) or {}
        if str(payload.get("selected_search_id", "") or "").strip() == key:
            return True
        for record in list(payload.get("search_records", []) or []):
            if str(record.get("search_id", "")).strip() == key:
                return True
        return False

    def _refresh_artifact_catalog(self) -> None:
        if self.artifact_h5 is None:
            self.available_slices = []
            self.artifact_search_catalog = []
            return
        artifact_path = Path(self.artifact_h5)
        try:
            self.available_slices = list(list_scan_slices(artifact_path))
            self.artifact_search_catalog = list(list_artifact_search_catalog(artifact_path))
        except Exception:
            self.available_slices = list(getattr(self, "available_slices", []) or [])
            self.artifact_search_catalog = list(getattr(self, "artifact_search_catalog", []) or [])

    def _artifact_has_in_progress_search(self) -> bool:
        for record in list(getattr(self, "artifact_search_catalog", []) or []):
            lifecycle = dict(record.get("lifecycle") or {})
            if search_lifecycle_is_in_progress(lifecycle):
                return True
        artifact_h5 = getattr(self, "artifact_h5", None)
        if artifact_h5 is None:
            return False
        return find_in_progress_slice_search(Path(artifact_h5)) != (None, None)

    def _resolve_active_session_slice_search(self) -> tuple[str | None, str | None]:
        self._sync_refresh_signal_from_disk()
        live_slice_key, live_search_id = self._heartbeat_slice_and_search()
        if live_slice_key:
            return live_slice_key, live_search_id
        artifact_h5 = getattr(self, "artifact_h5", None)
        if artifact_h5 is None:
            return None, None
        return find_in_progress_slice_search(Path(artifact_h5))

    def _should_follow_active_session(self) -> bool:
        return self._live_runner_detected()

    def _apply_open_session_defaults(self) -> None:
        if self.artifact_h5 is None:
            return
        if not self._live_runner_detected():
            return
        target_slice_key, target_search_id = self._resolve_active_session_slice_search()
        if not target_slice_key and not target_search_id:
            return
        if target_slice_key:
            self.slice_key_var.set(str(target_slice_key))
        if target_search_id and self._search_id_in_artifact(
            target_search_id,
            slice_key=target_slice_key,
        ):
            self.search_id_var.set(str(target_search_id))
        self.navigation_mode_var.set("active")
        self._navigation_mode_initialized = True
        self._open_session_prefers_active = True

    def _apply_open_global_best_domain(self) -> bool:
        if getattr(self, "_open_global_best_applied", False):
            return False
        if getattr(self, "_navigation_mode_user_chosen", False):
            self._open_global_best_applied = True
            return False
        if self._navigation_mode() != "best" or self.artifact_h5 is None:
            self._open_global_best_applied = True
            return False
        catalog = list(getattr(self, "artifact_search_catalog", []) or [])
        if not catalog:
            self._open_global_best_applied = True
            return False
        target_metric = str(self.run_target_metric or "chi2")
        best_slice, best_search = find_global_best_domain(
            Path(self.artifact_h5),
            catalog=catalog,
            target_metric=target_metric,
        )
        self._open_global_best_applied = True
        if not best_slice:
            return False
        current_slice = str(self._selected_slice_key() or "").strip()
        current_search = str(self._selected_search_id() or "").strip()
        if best_slice == current_slice and (not best_search or best_search == current_search):
            self._apply_best_navigation(force_reanchor=True)
            return False
        self.slice_key_var.set(str(best_slice))
        if best_search:
            self.search_id_var.set(str(best_search))
        self._schedule_payload_reload(status_text="Loading global best domain...")
        return True

    def _apply_active_session_selection_if_needed(self) -> bool:
        if self._navigation_mode() != "active":
            return False
        if not self._should_follow_active_session():
            return False
        self._bind_refresh_signal_path()
        self._sync_refresh_signal_from_disk()
        target_slice_key, target_search_id = self._resolve_active_session_slice_search()
        selected_slice_key = str(self._selected_slice_key() or "").strip()
        selected_search_id = str(self._selected_search_id() or "").strip()
        if target_slice_key and target_slice_key != selected_slice_key:
            self.slice_key_var.set(target_slice_key)
            if target_search_id:
                self.search_id_var.set(target_search_id)
            self._schedule_payload_reload(status_text="Loading active slice...")
            return True
        if target_search_id and target_search_id != selected_search_id:
            if not self._search_id_in_artifact(target_search_id, slice_key=target_slice_key or selected_slice_key):
                return False
            self.search_id_var.set(target_search_id)
            self._schedule_payload_reload(status_text="Loading active search...")
            return True
        return False

    def _heartbeat_activity_present(self) -> bool:
        if self._search_run_terminal():
            return False
        if getattr(self, "_refresh_signal_active_point", None) is not None:
            return True
        live_trials = dict(getattr(self, "_refresh_signal_live_trials", None) or {})
        for q0_key in ("fit_q0_trials", "q0_trials"):
            q0_values = np.asarray(live_trials.get(q0_key, ()), dtype=float)
            if q0_values.ndim == 1 and q0_values.size > 0:
                return True
        if live_trials.get("active_trial_q0") is not None:
            return True
        phase = str(getattr(self, "_refresh_signal_phase", "") or "").strip().lower()
        if phase and any(token in phase for token in ("active", "trial", "point")):
            return True
        return False

    def _live_scan_slice_key(self) -> str:
        return str(getattr(self, "_refresh_signal_slice_key", None) or "").strip()

    def _slice_label_for_key(self, slice_key: str) -> str:
        key = str(slice_key or "").strip()
        if not key:
            return ""
        for descriptor in list(getattr(self, "available_slices", []) or []):
            if str(descriptor.get("key", "")).strip() == key:
                return self._slice_label(descriptor)
        return key

    def _stale_active_search(self) -> bool:
        return self._artifact_marks_active_search() and not self._live_runner_detected()

    def _resolve_live_active_point(self) -> tuple[float, float] | None:
        active_point = getattr(self, "_refresh_signal_active_point", None)
        if active_point is not None:
            try:
                active_a = float(active_point[0])
                active_b = float(active_point[1])
            except Exception:
                active_a = np.nan
                active_b = np.nan
            if np.isfinite(active_a) and np.isfinite(active_b):
                return (float(active_a), float(active_b))

        live_trials = dict(getattr(self, "_refresh_signal_live_trials", None) or {})
        try:
            active_a = float(live_trials.get("a", np.nan))
            active_b = float(live_trials.get("b", np.nan))
        except Exception:
            active_a = np.nan
            active_b = np.nan
        if np.isfinite(active_a) and np.isfinite(active_b):
            return (float(active_a), float(active_b))
        return None

    def _live_navigation_available(self) -> bool:
        if not self._live_runner_detected():
            return False
        if not self._heartbeat_activity_present() and not self._live_scan_slice_key():
            return False
        resolved = self._resolve_live_active_point()
        if resolved is None and self.artifact_h5 is not None:
            self._sync_live_trial_state_from_artifact()
            resolved = self._resolve_live_active_point()
        if resolved is None:
            return False
        return True

    def _active_navigation_mode_available(self) -> bool:
        if self._live_navigation_available():
            return True
        return self._artifact_has_in_progress_search()

    def _best_navigation_available(self) -> bool:
        if not getattr(self, "payload", None):
            return False
        try:
            default_point_index(self.payload, self._run_target_metric_for_selection())
        except ValueError:
            return False
        return True

    def _coerce_navigation_mode_to_availability(self) -> None:
        mode = self._navigation_mode()
        if mode == "active" and not self._active_navigation_mode_available():
            fallback = "best" if self._best_navigation_available() else "free"
            self.navigation_mode_var.set(fallback)
        elif mode == "best" and not self._best_navigation_available():
            self.navigation_mode_var.set("free")

    def _default_navigation_mode(self) -> str:
        if self._stale_active_search():
            self._stale_active_notice = (
                "Search is marked active in the artifact, but no live runner was detected (interrupted?)."
            )
        else:
            self._stale_active_notice = ""
        if self._live_runner_detected() and self._active_navigation_mode_available():
            return "active"
        if self._best_navigation_available():
            return "best"
        return "free"

    def _parse_refresh_active_point(self, active_point: Any) -> tuple[float, float] | None:
        if isinstance(active_point, tuple):
            try:
                active_a = float(active_point[0])
                active_b = float(active_point[1])
            except Exception:
                return None
            if np.isfinite(active_a) and np.isfinite(active_b):
                return (float(active_a), float(active_b))
            return None
        if isinstance(active_point, dict):
            try:
                active_a = float(active_point.get("a"))
                active_b = float(active_point.get("b"))
            except Exception:
                return None
            if np.isfinite(active_a) and np.isfinite(active_b):
                return (float(active_a), float(active_b))
        return None

    def _apply_refresh_signal_payload(self, refresh_payload: dict[str, Any]) -> None:
        self._refresh_signal_slice_key = str(refresh_payload.get("slice_key", "") or "").strip() or None
        self._refresh_signal_search_id = str(refresh_payload.get("search_id", "") or "").strip() or None
        self._refresh_signal_event = str(refresh_payload.get("event", "") or "").strip().lower()
        self._refresh_signal_point_id = str(refresh_payload.get("point_id", "") or "").strip() or None
        self._refresh_signal_pending_points = []
        phase = str(refresh_payload.get("phase", "") or refresh_payload.get("event", "") or "").strip()
        if self._refresh_event_is_terminal(self._refresh_signal_event) or self._refresh_phase_is_terminal(phase):
            self._clear_live_runner_state()

    def _grid_point_coords_for_id(self, point_id: str) -> tuple[float, float] | None:
        target_id = str(point_id or "").strip()
        if not target_id:
            return None
        payload = getattr(self, "payload", None) or {}
        for record in list(payload.get("point_records") or []):
            grid_point_id = str(dict(record.get("diagnostics") or {}).get("grid_point_id") or "").strip()
            if grid_point_id == target_id:
                try:
                    return float(record["a"]), float(record["b"])
                except Exception:
                    return None
        artifact_h5 = getattr(self, "artifact_h5", None)
        slice_key = str(
            getattr(self, "_refresh_signal_slice_key", None)
            or payload.get("selected_slice_key")
            or ""
        ).strip() or None
        search_id = str(
            getattr(self, "_refresh_signal_search_id", None)
            or payload.get("selected_search_id")
            or ""
        ).strip() or None
        if artifact_h5 is None or slice_key is None:
            return None
        try:
            live_state = load_grid_point_live_state(
                Path(artifact_h5),
                slice_key=str(slice_key),
                search_id=search_id,
                point_id=target_id,
            )
        except Exception:
            live_state = None
        if not isinstance(live_state, dict):
            return None
        try:
            active_a = float(live_state.get("a", np.nan))
            active_b = float(live_state.get("b", np.nan))
        except Exception:
            return None
        if np.isfinite(active_a) and np.isfinite(active_b):
            return float(active_a), float(active_b)
        return None

    def _refresh_slice_grid_metadata(self) -> bool:
        if self.artifact_h5 is None:
            return False
        requested_slice_key = self._selected_slice_key()
        requested_search_id = self._selected_search_id()
        cache_key = (str(requested_slice_key or ""), str(requested_search_id or ""))
        self._payload_cache_by_selection.pop(cache_key, None)
        try:
            payload = self._load_scan_file_with_retries(
                self.artifact_h5,
                slice_key=requested_slice_key,
                search_id=requested_search_id,
            )
        except Exception:
            return False
        selected_cache_key = (
            str(payload.get("selected_slice_key", "") or ""),
            str(payload.get("selected_search_id", "") or ""),
        )
        self.payload = payload
        self._payload_cache_by_selection[cache_key] = payload
        self._payload_cache_by_selection[selected_cache_key] = payload
        self.available_slices = list(self.payload.get("available_slices", []))
        self.available_searches = list(self.payload.get("search_records", []))
        self.a_values = np.asarray(self.payload["a_values"], dtype=float)
        self.b_values = np.asarray(self.payload["b_values"], dtype=float)
        self.display_model = build_patch_grid_model(self.payload)
        self.run_target_metric = str(self.payload.get("target_metric", "chi2"))
        self._refresh_shared_heatmap_extents()
        self._last_payload_reload_at_s = float(time.time())
        return True

    def _apply_active_navigation_from_refresh(self, refresh_payload: dict[str, Any]) -> None:
        slice_key = str(refresh_payload.get("slice_key", "") or "").strip()
        search_id = str(refresh_payload.get("search_id", "") or "").strip()
        if slice_key:
            self.slice_key_var.set(slice_key)
        if search_id and self._search_id_in_artifact(search_id):
            self.search_id_var.set(search_id)
        search_metric = self._run_target_metric_for_selection()
        if str(self.metric_var.get()).strip().lower() != search_metric:
            self.metric_var.set(search_metric)
        event = str(refresh_payload.get("event", "") or "").strip().lower()
        point_id = str(refresh_payload.get("point_id", "") or "").strip()
        if not point_id or not should_follow_active_refresh_event(
            event,
            serial=self._execution_is_serial(),
        ):
            return
        coords = self._grid_point_coords_for_id(point_id)
        if coords is None:
            return
        self._active_follow_point_id = point_id
        self._refresh_signal_active_point = coords
        indices = self._point_indices_for_coordinates(float(coords[0]), float(coords[1]))
        if indices is None:
            return
        a_index, b_index = indices
        if int(self.a_index_var.get()) != int(a_index) or int(self.b_index_var.get()) != int(b_index):
            self.a_index_var.set(int(a_index))
            self.b_index_var.set(int(b_index))
            self._selected_trial_token = None

    def _apply_best_navigation(self, *, force_reanchor: bool = False) -> None:
        search_metric = self._run_target_metric_for_selection()
        selection = best_point_selection(self.payload, search_metric)
        if selection is None:
            self._best_tied_records = []
            return
        self._best_tied_records = list(selection.tied_records)
        prev = (int(self.a_index_var.get()), int(self.b_index_var.get()))
        target = (int(selection.a_index), int(selection.b_index))
        if force_reanchor or prev != target:
            self.a_index_var.set(target[0])
            self.b_index_var.set(target[1])
            self._selected_trial_token = None
            if selection.search_metric_best_trial_index is not None:
                self.trial_index_var.set(int(selection.search_metric_best_trial_index))

    def _apply_navigation_from_refresh(self, refresh_payload: dict[str, Any]) -> None:
        mode = self._navigation_mode()
        if mode == "active":
            self._apply_active_navigation_from_refresh(refresh_payload)
        elif mode == "best":
            self._apply_best_navigation(force_reanchor=False)

    def _process_refresh_event(self, refresh_payload: dict[str, Any]) -> None:
        self._apply_refresh_signal_payload(refresh_payload)
        if self.artifact_h5 is None:
            return
        if not getattr(self, "payload", None):
            self._reload_payload()
            return
        mode = self._navigation_mode()
        if mode == "active":
            slice_key = str(refresh_payload.get("slice_key", "") or "").strip()
            search_id = str(refresh_payload.get("search_id", "") or "").strip()
            if slice_key:
                self.slice_key_var.set(slice_key)
            if search_id and self._search_id_in_artifact(search_id):
                self.search_id_var.set(search_id)
        if not self._refresh_slice_grid_metadata():
            return
        self._apply_navigation_from_refresh(refresh_payload)
        self._sync_live_trial_state_from_artifact()
        if mode in {"active", "best"}:
            self._apply_locked_navigation_selection(schedule_slice_reload=False)
        self._refresh_selector_values()
        self._refresh_scan_state_display()
        self._refresh_action_states()
        if self.payload:
            self._refresh_all(update_selected_solution=True)

    def _point_is_saved(self, a_index: int, b_index: int) -> bool:
        points = dict((getattr(self, "payload", {}) or {}).get("points", {}))
        key = (int(a_index), int(b_index))
        if key not in points:
            return False
        status = str(points[key].get("status", "computed")).strip().lower()
        return status not in {"missing", "pending"}

    def _refresh_navigation_control_states(self) -> None:
        self._coerce_navigation_mode_to_availability()
        slice_locked = self._slice_and_search_selection_locked()
        grid_locked = self._grid_point_selection_locked()
        metric_locked = self._metric_selection_locked()
        active_available = self._active_navigation_mode_available()
        best_available = self._best_navigation_available()
        for mode, radio in getattr(self, "_navigation_mode_radios", {}).items():
            if mode == "free":
                radio.state(["!disabled"])
            elif mode == "active":
                radio.state(["!disabled"] if active_available else ["disabled"])
            elif mode == "best":
                radio.state(["!disabled"] if best_available else ["disabled"])
        for widget in (
            getattr(self, "slice_menu", None),
            getattr(self, "search_menu", None),
        ):
            if widget is None:
                continue
            if slice_locked:
                widget.state(["disabled"])
            else:
                widget.state(["!disabled", "readonly"])
        for widget in (
            getattr(self, "a_menu", None),
            getattr(self, "b_menu", None),
        ):
            if widget is None:
                continue
            if grid_locked:
                widget.state(["disabled"])
            else:
                widget.state(["!disabled", "readonly"])
        metric_menu = getattr(self, "metric_menu", None)
        if metric_menu is not None:
            if metric_locked:
                metric_menu.state(["disabled"])
            else:
                metric_menu.state(["!disabled", "readonly"])

    def _apply_locked_navigation_selection(self, *, schedule_slice_reload: bool = True) -> bool:
        mode = self._navigation_mode()
        if mode == "free" or not getattr(self, "payload", None):
            return False
        if mode == "active" and not self._active_navigation_mode_available():
            return False
        if mode == "best" and not self._best_navigation_available():
            return False
        if bool(getattr(self, "_applying_navigation_selection", False)):
            return False
        self._applying_navigation_selection = True
        try:
            if mode == "active":
                target_metric = self._run_target_metric_for_selection()
                if str(self.metric_var.get()) != target_metric:
                    self.metric_var.set(target_metric)
                target_slice_key, target_search_id = self._resolve_active_session_slice_search()
                live_slice_key = str(target_slice_key or getattr(self, "_refresh_signal_slice_key", None) or "").strip()
                live_search_id = str(target_search_id or getattr(self, "_refresh_signal_search_id", None) or "").strip()
                selected_slice_key = str(self._selected_slice_key() or "").strip()
                selected_search_id = str(self._selected_search_id() or "").strip()
                if schedule_slice_reload and live_slice_key and live_slice_key != selected_slice_key:
                    self.slice_key_var.set(live_slice_key)
                    self._schedule_payload_reload(status_text="Loading active slice...")
                    return True
                if live_search_id and live_search_id != selected_search_id:
                    if not self._search_id_in_artifact(live_search_id, slice_key=live_slice_key or selected_slice_key):
                        return False
                    self.search_id_var.set(live_search_id)
                    self._schedule_payload_reload(status_text="Loading active search...")
                    return True
                resolved = self._active_point_indices()
                if resolved is None:
                    return False
                a_index, b_index = resolved
                if int(self.a_index_var.get()) != int(a_index) or int(self.b_index_var.get()) != int(b_index):
                    self.a_index_var.set(int(a_index))
                    self.b_index_var.set(int(b_index))
                    self._selected_trial_token = None
                    self._refresh_selector_values()
            elif mode == "best":
                self._apply_best_navigation(force_reanchor=True)
        finally:
            self._applying_navigation_selection = False
        return False

    def _initialize_navigation_mode(self) -> None:
        if self._navigation_mode_initialized or getattr(self, "_navigation_mode_user_chosen", False):
            self._refresh_navigation_control_states()
            return
        if getattr(self, "_open_session_prefers_active", False) and self._active_navigation_mode_available():
            self.navigation_mode_var.set("active")
        else:
            self.navigation_mode_var.set(self._default_navigation_mode())
        self._navigation_mode_initialized = True
        self._coerce_navigation_mode_to_availability()
        self._refresh_navigation_control_states()

    def _on_navigation_mode_changed(self) -> None:
        requested = self._navigation_mode()
        if requested == "active" and not self._active_navigation_mode_available():
            self.navigation_mode_var.set("free")
            self.status_var.set("Active navigation requires a live or in-progress search.")
            requested = "free"
        elif requested == "best" and not self._best_navigation_available():
            self.navigation_mode_var.set("free")
            self.status_var.set("Best navigation requires a saved grid point for the run target metric.")
            requested = "free"
        elif requested == "best":
            self._apply_best_navigation(force_reanchor=True)
        self._navigation_mode_user_chosen = True
        if requested == "free":
            self._stale_active_notice = ""
        self._refresh_navigation_control_states()
        if self._apply_locked_navigation_selection():
            return
        self._refresh_all()

    def _refresh_action_states(self) -> None:
        has_artifact = self.artifact_h5 is not None
        has_selected_point = has_artifact and self._has_selected_point()
        live_state = self._live_trial_state() if has_artifact else None
        live_solution_available = (
            live_state is not None
            and self._live_trials_context_available(live_state)
            and self._live_trials_streaming(live_state)
        )
        if self.open_artifact_button is not None:
            self.open_artifact_button.state(["!disabled"])
        if self.display_selected_button is not None:
            self.display_selected_button.state(
                ["!disabled"] if (has_selected_point or live_solution_available) else ["disabled"]
            )
        if self.summary_button is not None:
            self.summary_button.state(["!disabled"] if has_artifact else ["disabled"])
        if self.refresh_button is not None:
            self.refresh_button.state(["!disabled"] if has_artifact else ["disabled"])
        self._refresh_navigation_control_states()

    def _scan_state_snapshot(self) -> tuple[str, str, str, str, str]:
        if self.artifact_h5 is not None and bool(getattr(self, "_initial_reload_in_progress", False)) and not self.payload:
            return "LOADING", "Opening artifact", "Artifact load in progress", "#1d6fd6", "white"
        if self.artifact_h5 is None or not self.payload:
            return "NO ARTIFACT", "No artifact", "No artifact loaded", "#6c757d", "white"

        points = self.payload.get("points", {})
        diagnostics = dict(self.payload.get("diagnostics") or {})
        total = int(len(points))
        live_pending = 0
        scoped_active = self._active_point_scoped_to_selection()
        if scoped_active is not None and not self._active_point_is_saved_in_artifact():
            live_pending = 1
        total += live_pending
        slice_descriptor = dict(self.payload.get("selected_slice") or {})
        slice_label = self._slice_label(slice_descriptor) if slice_descriptor else "current slice"
        slice_count = max(1, int(len(self.available_slices) or 1))
        selected_search = dict(self.payload.get("selected_search") or {})
        search_display_var = getattr(self, "search_display_var", None)
        search_display_text = search_display_var.get() if search_display_var is not None else ""
        selected_slice_key = str(self.payload.get("selected_slice_key", "")).strip()
        selected_search_id = str(self.payload.get("selected_search_id", "")).strip()
        search_label = str(selected_search_id or search_display_text or selected_search.get("label") or "").strip()
        search_request = dict(selected_search.get("request") or {})
        search_lifecycle = dict(selected_search.get("lifecycle") or {})
        search_diagnostics = dict(selected_search.get("diagnostics") or {})
        search_metric = str(
            selected_search.get("target_metric")
            or search_request.get("target_metric")
            or search_diagnostics.get("target_metric")
            or ""
        ).strip()
        threshold_value = search_request.get("metrics_mask_threshold", search_diagnostics.get("metrics_mask_threshold"))
        try:
            search_threshold = f"{float(threshold_value):.3f}"
        except Exception:
            search_threshold = ""
        search_signature = str(
            search_request.get("compatibility_signature")
            or search_diagnostics.get("compatibility_signature")
            or ""
        ).strip()
        started_at = str(search_lifecycle.get("started_at") or "").strip()
        completed_at = str(search_lifecycle.get("completed_at") or "").strip()
        search_status = str(selected_search.get("status") or "").strip()
        search_active_flag = bool(selected_search.get("active", search_lifecycle.get("active", False)))
        toolbar_prefix = f"{slice_label} | {search_label}" if search_label else slice_label
        search_mode = str(diagnostics.get("search_mode", "")).strip().lower()
        adaptive_point_run = search_mode in {"adaptive_local_single_observation", "adaptive_local_single_frequency"}
        if total == 0:
            refresh_active = False
            if self.refresh_signal_path is not None and self.refresh_signal_path.exists():
                try:
                    refresh_age_s = max(0.0, time.time() - float(self.refresh_signal_path.stat().st_mtime))
                    refresh_active = refresh_age_s <= float(self._ACTIVE_REFRESH_GRACE_S)
                except Exception:
                    refresh_active = False
            heartbeat_slice_key = str(getattr(self, "_refresh_signal_slice_key", None) or "").strip()
            heartbeat_matches_selected = not (heartbeat_slice_key and selected_slice_key and heartbeat_slice_key != selected_slice_key)
            artifact_live = self._heartbeat_activity_present() and heartbeat_matches_selected
            runner_active = self._live_runner_detected()
            heartbeat_running = artifact_live and runner_active

            if runner_active or (adaptive_point_run and refresh_active and heartbeat_matches_selected):
                badge = "RUNNING"
                color = "#0b7285"
            elif artifact_live or search_active_flag:
                badge = "INTERRUPTED"
                color = "#c92a2a"
            else:
                badge = "EMPTY"
                color = "#6c757d"
            toolbar_detail = f"{toolbar_prefix} | 0/{total} computed"
            info_lines_empty = [f"Current slice: {slice_label}"]
            if selected_slice_key:
                info_lines_empty.append(f"Slice key: {selected_slice_key}")
            if search_label:
                info_lines_empty.append(f"Selected search: {search_label}")
            if selected_search_id:
                info_lines_empty.append(f"Selected search ID: {selected_search_id}")
            info_lines_empty.extend([f"Grid points: {total}", "Computed: 0", f"Artifact slices: {slice_count}"])
            phase_display = str(self._refresh_signal_phase or "").strip()
            if phase_display:
                info_lines_empty.append(f"Last phase: {phase_display}")
            info_detail = "\n".join(info_lines_empty)
            return badge, toolbar_detail, info_detail, color, "white"

        statuses = [str(point.get("status", "computed")).strip().lower() for point in points.values()]
        pending = int(sum(status == "pending" for status in statuses)) + live_pending
        failed = int(sum(status == "failed" for status in statuses))
        computed = int(sum(status == "computed" for status in statuses))
        other = max(0, total - pending - failed - computed)
        noncomputed = pending + failed + other

        refresh_active = False
        if self.refresh_signal_path is not None and self.refresh_signal_path.exists():
            try:
                refresh_age_s = max(0.0, time.time() - float(self.refresh_signal_path.stat().st_mtime))
                refresh_active = refresh_age_s <= float(self._ACTIVE_REFRESH_GRACE_S)
            except Exception:
                refresh_active = False

        phase = str(self._refresh_signal_phase or "").strip().lower()
        phase_complete = phase in {"scan complete", "complete"}
        phase_aborted = phase in {
            "scan interrupted",
            "interrupted",
            "scan aborted",
            "aborted",
            "scan failed",
            "failed",
        }
        heartbeat_slice_key = str(getattr(self, "_refresh_signal_slice_key", None) or "").strip()
        heartbeat_matches_selected = not (heartbeat_slice_key and selected_slice_key and heartbeat_slice_key != selected_slice_key)
        heartbeat_search_id = str(getattr(self, "_refresh_signal_search_id", None) or "").strip()
        heartbeat_matches_selected_search = not (
            heartbeat_search_id and selected_search_id and heartbeat_search_id != selected_search_id
        )
        scoped_refresh_active = refresh_active and heartbeat_matches_selected
        scoped_phase_complete = phase_complete and heartbeat_matches_selected
        artifact_live_active = self._heartbeat_activity_present() and heartbeat_matches_selected and heartbeat_matches_selected_search
        search_completed = (search_status == "complete") or bool(completed_at)
        runner_active = self._live_runner_detected()
        live_runner = runner_active and (artifact_live_active or self._heartbeat_activity_present())

        if runner_active or artifact_live_active or (adaptive_point_run and scoped_refresh_active and not scoped_phase_complete):
            if self._search_run_terminal():
                badge = "FINISHED"
                color = "#2b8a3e"
            else:
                badge = "RUNNING"
                color = "#0b7285"
        elif scoped_phase_complete or (search_completed and not runner_active):
            badge = "FINISHED"
            color = "#2b8a3e"
        elif phase_aborted:
            badge = "INTERRUPTED"
            color = "#c92a2a"
        elif not search_completed:
            badge = "INCOMPLETE"
            color = "#b26a00"
        else:
            badge = "FINISHED"
            color = "#2b8a3e"

        toolbar_detail = f"{toolbar_prefix} | {computed}/{total} computed"
        info_lines = [
            f"Current slice: {slice_label}",
            f"Grid points: {total}",
            f"Computed: {computed}",
        ]
        if selected_slice_key:
            info_lines.insert(1, f"Slice key: {selected_slice_key}")
        if search_label:
            info_lines.insert(1, f"Selected search: {search_label}")
        if selected_search_id:
            info_lines.append(f"Selected search ID: {selected_search_id}")
        info_lines.append(f"Search active marker: {'yes' if search_active_flag else 'no'}")
        if search_metric:
            info_lines.append(f"Search metric: {search_metric}")
        if search_threshold:
            info_lines.append(f"Search threshold: {search_threshold}")
        if search_signature:
            info_lines.append(f"Search signature: {search_signature}")
        if started_at:
            info_lines.append(f"Search started: {started_at}")
        if completed_at:
            info_lines.append(f"Search completed: {completed_at}")
        if search_status:
            if (
                search_active_flag
                and search_status == "complete"
                and runner_active
                and not self._search_run_terminal()
            ):
                info_lines.append("Search status: in_progress (runner active)")
            else:
                info_lines.append(f"Search status: {search_status}")
        shift_diag = self._diagnostics_with_search_shift(dict(diagnostics))
        info_lines.append(format_search_shift_policy_label(shift_diag))
        info_lines.append(f"Navigation mode: {self._navigation_mode()}")
        stale_notice = str(getattr(self, "_stale_active_notice", "") or "").strip()
        if stale_notice:
            info_lines.append(stale_notice)
        available_searches = list(getattr(self, "available_searches", []) or [])
        if available_searches:
            info_lines.append(f"Stored searches: {len(available_searches)}")
            if not any(bool(record.get("active", dict(record.get("lifecycle") or {}).get("active", False))) for record in available_searches):
                info_lines.append("Active search marker in artifact: none")
                info_lines.append("Selection fallback: latest/selected search record")
        run_history = list(self.payload.get("run_history", [])) if self.payload else []
        info_lines.append(f"Run history entries: {len(run_history)}")
        if run_history:
            latest_history = dict(run_history[-1])
            latest_timestamp = str(latest_history.get("timestamp_utc", "")).strip()
            latest_action = str(latest_history.get("action", "")).strip()
            latest_command = str(
                latest_history.get("wrapper_command")
                or latest_history.get("effective_python_command")
                or ""
            ).strip()
            if latest_timestamp or latest_action:
                label = " ".join(part for part in (latest_timestamp, f"({latest_action})" if latest_action else "") if part)
                info_lines.append(f"Latest run: {label}".strip())
            if latest_command:
                info_lines.append(f"Latest command: {latest_command}")
        if pending > 0:
            info_lines.append(f"Pending: {pending}")
        if failed > 0:
            info_lines.append(f"Failed: {failed}")
        if other > 0:
            info_lines.append(f"Other: {other}")
        info_lines.append(f"Artifact slices: {slice_count}")
        phase_display = str(self._refresh_signal_phase or "").strip()
        if phase_display:
            info_lines.append(f"Last phase: {phase_display}")
        if runner_active:
            info_lines.append(f"Live runner: yes (pid from {self.artifact_h5}.log)" if self.artifact_h5 is not None else "Live runner: yes")
        if heartbeat_slice_key:
            info_lines.append(f"Heartbeat slice key: {heartbeat_slice_key}")
        if runner_active and heartbeat_slice_key and not heartbeat_matches_selected:
            live_slice_label = self._slice_label_for_key(heartbeat_slice_key) or heartbeat_slice_key
            info_lines.append(f"Live scan slice: {live_slice_label} (switch Slice menu to follow)")
        active_point = getattr(self, "_refresh_signal_active_point", None)
        if (
            active_point is not None
            and not self._search_run_terminal()
            and heartbeat_matches_selected
            and heartbeat_matches_selected_search
        ):
            try:
                active_a = float(active_point[0])
                active_b = float(active_point[1])
            except Exception:
                active_a = np.nan
                active_b = np.nan
            if np.isfinite(active_a) and np.isfinite(active_b):
                info_lines.append(f"Active point: (a={active_a:.3f}, b={active_b:.3f})")
        elif active_point is not None:
            info_lines.append("Active point: running on a different slice/search")
        live_trials = dict(getattr(self, "_refresh_signal_live_trials", None) or {})
        active_trial_q0 = live_trials.get("active_trial_q0")
        if active_trial_q0 is not None and heartbeat_matches_selected and heartbeat_matches_selected_search:
            try:
                info_lines.append(f"Live trial q0: {float(active_trial_q0):.6e}")
            except Exception:
                pass
        if self._has_selected_point():
            try:
                point = self._selected_point()
                q0_trials, metric_trials, point_metric = self._trial_series_for_point(point)
                selected_trial_index = self._selected_trial_index_for_point(point, q0_trials, metric_trials, point_metric)
                point_diag = self._selected_diagnostics()
                trial_shift_label = format_observation_shift_label(
                    diagnostics=point_diag,
                    trial_index=selected_trial_index,
                    fit_shift_x_trials=point.get("fit_shift_x_trials"),
                    fit_shift_y_trials=point.get("fit_shift_y_trials"),
                    fit_find_shift_valid_trials=point.get("fit_find_shift_valid_trials"),
                )
                if trial_shift_label:
                    info_lines.append(f"Selected trial: {trial_shift_label}")
            except Exception:
                pass
        return badge, toolbar_detail, "\n".join(info_lines), color, "white"

    def _read_refresh_signal_payload(self) -> dict[str, Any]:
        empty: dict[str, Any] = {
            "phase": "",
            "slice_key": None,
            "search_id": None,
            "version": 1,
            "event": "",
            "point_id": None,
            "trial_index": None,
            "sequence": None,
        }
        if self.refresh_signal_path is None or not self.refresh_signal_path.exists():
            return dict(empty)
        try:
            text = self.refresh_signal_path.read_text(encoding="utf-8").strip()
        except Exception:
            return dict(empty)
        if not text:
            return dict(empty)
        if text.startswith("{"):
            try:
                payload = json.loads(text)
            except Exception:
                return dict(empty)
            event = str(payload.get("event", "") or "").strip()
            phase = str(payload.get("phase", "") or event).strip()
            trial_index = payload.get("trial_index")
            return {
                "version": int(payload.get("version", 1) or 1),
                "event": event,
                "phase": phase,
                "slice_key": str(payload.get("slice_key", "") or "").strip() or None,
                "search_id": str(payload.get("search_id", "") or "").strip() or None,
                "point_id": str(payload.get("point_id", "") or "").strip() or None,
                "trial_index": None if trial_index is None else int(trial_index),
                "sequence": int(payload.get("sequence", 0) or 0),
            }
        parts = text.split(maxsplit=1)
        phase = parts[1].strip() if len(parts) == 2 else parts[0].strip()
        return {"phase": phase, "slice_key": None, "search_id": None, "version": 1, "event": "", "point_id": None, "trial_index": None, "sequence": None}

    def _point_indices_for_coordinates(self, a_value: float, b_value: float) -> tuple[int, int] | None:
        a_values = np.asarray(getattr(self, "a_values", []), dtype=float)
        b_values = np.asarray(getattr(self, "b_values", []), dtype=float)
        if a_values.size == 0 or b_values.size == 0:
            return None
        a_matches = np.where(np.isclose(a_values, float(a_value), rtol=0.0, atol=1e-12))[0]
        b_matches = np.where(np.isclose(b_values, float(b_value), rtol=0.0, atol=1e-12))[0]
        if a_matches.size == 0 or b_matches.size == 0:
            return None
        return int(a_matches[0]), int(b_matches[0])

    def _apply_active_point_selection(self, *, force: bool = False) -> None:
        navigation_mode_var = getattr(self, "navigation_mode_var", None)
        if force and navigation_mode_var is not None:
            navigation_mode_var.set("active")
            self._refresh_navigation_control_states()
        self._apply_locked_navigation_selection(schedule_slice_reload=False)

    def _active_point_indices(self) -> tuple[int, int] | None:
        active_point = getattr(self, "_refresh_signal_active_point", None)
        if active_point is None:
            return None
        if not getattr(self, "payload", None):
            payload = {}
        else:
            payload = self.payload
        active_a, active_b = active_point
        exact = self._point_indices_for_coordinates(active_a, active_b)
        if exact is not None:
            return exact
        a_values = getattr(self, "a_values", np.asarray([], dtype=float))
        b_values = getattr(self, "b_values", np.asarray([], dtype=float))
        if a_values.size == 0 or b_values.size == 0:
            return None
        nearest_a_idx = int(np.argmin(np.abs(a_values - float(active_a))))
        nearest_b_idx = int(np.argmin(np.abs(b_values - float(active_b))))
        return nearest_a_idx, nearest_b_idx

    def _live_active_ab(self, live_state: dict[str, Any] | None = None) -> tuple[float, float] | None:
        if live_state is not None:
            for a_key, b_key in (("active_a", "active_b"), ("a", "b")):
                if a_key not in live_state or b_key not in live_state:
                    continue
                try:
                    active_a = float(live_state[a_key])
                    active_b = float(live_state[b_key])
                except Exception:
                    continue
                if np.isfinite(active_a) and np.isfinite(active_b):
                    return float(active_a), float(active_b)
        active_point = getattr(self, "_refresh_signal_active_point", None)
        if active_point is None:
            return None
        try:
            active_a = float(active_point[0])
            active_b = float(active_point[1])
        except Exception:
            return None
        if np.isfinite(active_a) and np.isfinite(active_b):
            return float(active_a), float(active_b)
        return None

    def _live_trial_state(self) -> dict[str, Any] | None:
        live_trials = dict(getattr(self, "_refresh_signal_live_trials", None) or {})
        active_coords = self._live_active_ab(live_trials)
        if active_coords is None:
            return None
        active_a, active_b = active_coords
        if not live_trials:
            live_trials = {
                "metric_name": str(getattr(self, "run_target_metric", None) or "chi2"),
                "q0_trials": [],
                "metric_trials": [],
            }
        slice_key = str(
            live_trials.get("slice_key") or getattr(self, "_refresh_signal_slice_key", None) or ""
        ).strip()
        if slice_key:
            live_trials["slice_key"] = slice_key
        resolved_search_id = self._resolved_heartbeat_search_id(live_trials)
        if resolved_search_id and not str(live_trials.get("search_id") or "").strip():
            live_trials["search_id"] = resolved_search_id
        live_trials["active_a"] = active_a
        live_trials["active_b"] = active_b
        resolved = self._point_indices_for_coordinates(active_a, active_b)
        if resolved is not None:
            live_trials["a_index"] = int(resolved[0])
            live_trials["b_index"] = int(resolved[1])
        return live_trials

    def _sync_live_trial_state_from_artifact(self) -> None:
        self._refresh_signal_pending_points = []
        if self._search_run_terminal():
            self._clear_live_runner_state()
            return
        artifact_h5 = getattr(self, "artifact_h5", None)
        payload = getattr(self, "payload", None) or {}
        if artifact_h5 is None or not payload:
            self._refresh_signal_live_trials = None
            return
        slice_key = str(
            getattr(self, "_refresh_signal_slice_key", None)
            or payload.get("selected_slice_key", "")
            or ""
        ).strip() or None
        search_id = str(
            getattr(self, "_refresh_signal_search_id", None)
            or payload.get("selected_search_id", "")
            or ""
        ).strip() or None
        if search_id and not self._search_id_in_artifact(search_id):
            catalog_search_id = self._catalog_live_search_id_for_slice(str(slice_key))
            if catalog_search_id and self._search_id_in_artifact(catalog_search_id):
                search_id = catalog_search_id
        if slice_key is None:
            self._refresh_signal_live_trials = None
            return
        live_state: dict[str, Any] | None = None
        followed_point_id = str(
            getattr(self, "_active_follow_point_id", None)
            or getattr(self, "_refresh_signal_point_id", None)
            or ""
        ).strip() or None
        if followed_point_id:
            try:
                live_state = load_grid_point_live_state(
                    Path(artifact_h5),
                    slice_key=str(slice_key),
                    search_id=search_id,
                    point_id=followed_point_id,
                )
            except Exception:
                live_state = None
        if live_state is None:
            self._refresh_signal_live_trials = None
            followed_coords = self._followed_point_coords()
            if followed_coords is not None:
                self._refresh_signal_active_point = followed_coords
            else:
                self._refresh_signal_active_point = None
            return
        resolved_slice_key = str(live_state.get("slice_key", "") or slice_key).strip() or slice_key
        if not str(getattr(self, "_refresh_signal_slice_key", None) or "").strip():
            self._refresh_signal_slice_key = resolved_slice_key
        if getattr(self, "_refresh_signal_active_point", None) is None:
            try:
                active_a = float(live_state.get("a", np.nan))
                active_b = float(live_state.get("b", np.nan))
            except Exception:
                active_a = np.nan
                active_b = np.nan
            if np.isfinite(active_a) and np.isfinite(active_b):
                self._refresh_signal_active_point = (float(active_a), float(active_b))
        fit_q0_trials = np.asarray(live_state.get("fit_q0_trials", ()), dtype=float)
        fit_metric_trials = np.asarray(live_state.get("fit_metric_trials", ()), dtype=float)
        active_trial_q0 = live_state.get("q0")
        active_trial_index = live_state.get("trial_index")
        resolved_search_id = str(search_id or live_state.get("search_id", "") or "").strip() or None
        if resolved_search_id and not str(getattr(self, "_refresh_signal_search_id", None) or "").strip():
            self._refresh_signal_search_id = resolved_search_id
        self._refresh_signal_live_trials = {
            "a": None if self._refresh_signal_active_point is None else float(self._refresh_signal_active_point[0]),
            "b": None if self._refresh_signal_active_point is None else float(self._refresh_signal_active_point[1]),
            "metric_name": str(live_state.get("metric_name", self.run_target_metric or "chi2")),
            "slice_key": resolved_slice_key,
            "search_id": resolved_search_id,
            "active_trial_index": None if active_trial_index is None else int(active_trial_index),
            "active_trial_q0": None if active_trial_q0 is None else float(active_trial_q0),
            "fit_q0_trials": fit_q0_trials,
            "fit_metric_trials": fit_metric_trials,
            "fit_chi2_trials": np.asarray(live_state.get("fit_chi2_trials", ()), dtype=float),
            "fit_rho2_trials": np.asarray(live_state.get("fit_rho2_trials", ()), dtype=float),
            "fit_eta2_trials": np.asarray(live_state.get("fit_eta2_trials", ()), dtype=float),
            "fit_shift_x_trials": np.asarray(live_state.get("fit_shift_x_trials", ()), dtype=float),
            "fit_shift_y_trials": np.asarray(live_state.get("fit_shift_y_trials", ()), dtype=float),
            "fit_find_shift_valid_trials": np.asarray(
                live_state.get("fit_find_shift_valid_trials", ()), dtype=bool
            ),
        }

    def _should_use_live_trials(self, live_state: dict[str, Any] | None) -> bool:
        if live_state is None:
            return False
        if self._navigation_mode() == "best":
            return False
        if not self._live_slice_matches_selected(live_state):
            return False
        if not self._live_search_matches_selected(live_state):
            return False
        if not self._has_selected_point():
            if "a_index" not in live_state or "b_index" not in live_state:
                if self._navigation_mode() == "free":
                    active_indices = self._active_point_indices()
                    if active_indices is None:
                        return False
                    return (
                        int(self.a_index_var.get()),
                        int(self.b_index_var.get()),
                    ) == active_indices
                return False
            return (
                int(self.a_index_var.get()),
                int(self.b_index_var.get()),
            ) == (
                int(live_state["a_index"]),
                int(live_state["b_index"]),
            )
        active_indices = self._active_point_indices()
        if active_indices is not None and (
            int(self.a_index_var.get()),
            int(self.b_index_var.get()),
        ) == active_indices:
            return True
        if "a_index" not in live_state or "b_index" not in live_state:
            return False
        return (
            int(self.a_index_var.get()),
            int(self.b_index_var.get()),
        ) == (
            int(live_state["a_index"]),
            int(live_state["b_index"]),
        )

    def _live_slice_matches_selected(self, live_state: dict[str, Any] | None) -> bool:
        if live_state is None:
            return False
        selected_slice_key = str(self._selected_slice_key() or "").strip()
        heartbeat_slice_key = str(getattr(self, "_refresh_signal_slice_key", None) or "").strip()
        if heartbeat_slice_key and selected_slice_key and heartbeat_slice_key != selected_slice_key:
            return False
        live_slice_key = str(live_state.get("slice_key") or heartbeat_slice_key or "").strip()
        return not (live_slice_key and selected_slice_key and live_slice_key != selected_slice_key)

    def _in_progress_search_id_for_slice(self, slice_key: str) -> str:
        key = str(slice_key or "").strip()
        for record in list(getattr(self, "available_searches", []) or []):
            lifecycle = dict(record.get("lifecycle") or {})
            if not bool(record.get("in_progress", lifecycle.get("in_progress", False))):
                continue
            record_slice = str(record.get("slice_key", "") or "").strip()
            if key and record_slice and record_slice != key:
                continue
            search_id = str(record.get("search_id", "") or "").strip()
            if search_id:
                return search_id
        return ""

    def _catalog_live_search_id_for_slice(self, slice_key: str) -> str:
        in_progress = self._in_progress_search_id_for_slice(slice_key)
        if in_progress:
            return in_progress
        if not (self._refresh_signal_is_fresh() or self._live_runner_detected()):
            return ""
        key = str(slice_key or "").strip()
        active_ids: list[str] = []
        for record in list(getattr(self, "available_searches", []) or []):
            lifecycle = dict(record.get("lifecycle") or {})
            if not bool(record.get("active", lifecycle.get("active", False))):
                continue
            record_slice = str(record.get("slice_key", "") or "").strip()
            if key and record_slice and record_slice != key:
                continue
            search_id = str(record.get("search_id", "") or "").strip()
            if search_id:
                active_ids.append(search_id)
        if len(active_ids) == 1:
            return active_ids[0]
        return ""

    def _resolved_heartbeat_search_id(self, live_state: dict[str, Any] | None) -> str:
        if live_state is None:
            return ""
        live_search_id = str(live_state.get("search_id") or "").strip()
        if live_search_id and not self._search_id_in_artifact(live_search_id):
            live_search_id = ""
        slice_key = str(
            live_state.get("slice_key")
            or getattr(self, "_refresh_signal_slice_key", None)
            or self._selected_slice_key()
            or ""
        ).strip()
        in_progress_id = self._in_progress_search_id_for_slice(slice_key)
        if in_progress_id:
            return in_progress_id
        if live_search_id:
            return live_search_id
        return self._catalog_live_search_id_for_slice(slice_key)

    def _selected_search_is_in_progress(self) -> bool:
        selected = dict((getattr(self, "payload", {}) or {}).get("selected_search") or {})
        if not selected:
            return True
        lifecycle = dict(selected.get("lifecycle") or {})
        status = str(selected.get("status", "") or "").strip().lower()
        if status in {"complete", "completed", "failed", "aborted", "interrupted"}:
            return False
        if bool(selected.get("in_progress", lifecycle.get("in_progress", False))):
            return True
        if str(lifecycle.get("completed_at") or "").strip():
            return False
        return bool(selected.get("active", lifecycle.get("active", False)))

    def _live_search_matches_selected(self, live_state: dict[str, Any] | None) -> bool:
        if live_state is None:
            return False
        if not self._live_slice_matches_selected(live_state):
            return False

        selected_search_id = str(self._selected_search_id() or "").strip()
        slice_key = str(
            live_state.get("slice_key")
            or getattr(self, "_refresh_signal_slice_key", None)
            or self._selected_slice_key()
            or ""
        ).strip()
        in_progress_id = self._in_progress_search_id_for_slice(slice_key)
        explicit_live_id = str(live_state.get("search_id") or "").strip()
        if explicit_live_id and selected_search_id and explicit_live_id != selected_search_id:
            if self._search_id_in_artifact(explicit_live_id):
                return False
            explicit_live_id = ""
        catalog_live_id = self._catalog_live_search_id_for_slice(slice_key)
        live_owner_id = in_progress_id or explicit_live_id or catalog_live_id

        if in_progress_id and selected_search_id and in_progress_id != selected_search_id:
            return False

        if live_owner_id and selected_search_id:
            return live_owner_id == selected_search_id

        if live_owner_id and not selected_search_id:
            return True

        if getattr(self, "_refresh_signal_active_point", None) is not None and (
            self._refresh_signal_is_fresh() or self._live_runner_detected()
        ):
            return self._selected_search_is_in_progress()

        return self._selected_search_is_in_progress()

    def _selection_matches_live_active_point(self, a_index: int, b_index: int) -> bool:
        live_state = self._live_trial_state()
        if live_state is None:
            return False
        if not self._live_slice_matches_selected(live_state):
            return False
        if not self._live_search_matches_selected(live_state):
            return False
        active_indices = self._active_point_indices()
        if active_indices is None:
            return False
        return int(a_index) == int(active_indices[0]) and int(b_index) == int(active_indices[1])

    def _active_point_scoped_to_selection(self) -> tuple[float, float] | None:
        active_point = getattr(self, "_refresh_signal_active_point", None)
        if active_point is None:
            return None
        live_state = self._live_trial_state()
        if live_state is None:
            return None
        if not self._live_slice_matches_selected(live_state):
            return None
        if not self._live_search_matches_selected(live_state):
            return None
        try:
            active_a = float(active_point[0])
            active_b = float(active_point[1])
        except Exception:
            return None
        if np.isfinite(active_a) and np.isfinite(active_b):
            return (float(active_a), float(active_b))
        return None

    def _live_trials_context_available(self, live_state: dict[str, Any] | None) -> bool:
        if live_state is None:
            return False
        if not self._live_slice_matches_selected(live_state):
            return False
        if not self._live_search_matches_selected(live_state):
            return False
        return self._should_use_live_trials(live_state) or self._should_force_live_trials(live_state)

    def _active_point_is_saved_in_artifact(self) -> bool:
        active_point = getattr(self, "_refresh_signal_active_point", None)
        if active_point is None or not getattr(self, "payload", None):
            return False
        try:
            active_a = float(active_point[0])
            active_b = float(active_point[1])
        except Exception:
            return False
        indices = self._point_indices_for_coordinates(active_a, active_b)
        if indices is None:
            return False
        points = dict(self.payload.get("points", {}))
        key = (int(indices[0]), int(indices[1]))
        if key not in points:
            return False
        status = str(points[key].get("status", "computed")).strip().lower()
        return status not in {"missing", "pending"}

    def _live_trials_streaming(self, live_state: dict[str, Any] | None) -> bool:
        if live_state is None:
            return False
        q0_trials, _, _, _ = self._live_trial_series_from_state(live_state)
        return int(q0_trials.size) > 0

    def _active_point_waiting_summary(self, live_state: dict[str, Any] | None) -> str:
        if self._live_trials_streaming(live_state):
            return (
                "Live trials are streaming for the active point.\n"
                "The detailed point summary will populate after the point is saved."
            )
        return (
            "Computing the active grid point.\n"
            "Trial curves appear here after the point is saved to the artifact."
        )

    def _should_force_live_trials(self, live_state: dict[str, Any] | None) -> bool:
        if live_state is None:
            return False
        if self._navigation_mode() == "best":
            return False
        if not self._live_slice_matches_selected(live_state):
            return False
        if not self._live_search_matches_selected(live_state):
            return False
        if getattr(self, "_refresh_signal_active_point", None) is None:
            return False
        if self._navigation_mode() == "free":
            a_index_var = getattr(self, "a_index_var", None)
            b_index_var = getattr(self, "b_index_var", None)
            if a_index_var is None or b_index_var is None:
                return False
            if not self._selection_matches_live_active_point(
                int(a_index_var.get()),
                int(b_index_var.get()),
            ):
                return False
        if self._active_point_is_saved_in_artifact():
            return self._live_trials_streaming(live_state)
        return True

    def _use_shared_grid_axes(self) -> bool:
        return bool(getattr(self, "shared_heatmap_axes_var", None) and self.shared_heatmap_axes_var.get())

    def _ensure_shared_grid_extents(self) -> None:
        if not self._use_shared_grid_axes():
            return
        if isinstance(getattr(self, "_shared_heatmap_extents", None), dict):
            return
        self._refresh_shared_heatmap_extents()

    def _heatmap_plot_limits(self, display_model: dict[str, Any] | None = None) -> tuple[float, float, float, float]:
        model = dict(display_model or getattr(self, "display_model", {}) or {})
        self._ensure_shared_grid_extents()
        return resolve_grid_axis_limits(
            display_model=model,
            a_values=np.asarray(getattr(self, "a_values", ()), dtype=float),
            b_values=np.asarray(getattr(self, "b_values", ()), dtype=float),
            shared_extents=getattr(self, "_shared_heatmap_extents", None),
            use_shared_axes=self._use_shared_grid_axes(),
        )

    def _refresh_scan_state_display(self) -> None:
        badge, toolbar_detail, info_detail, bg, fg = self._scan_state_snapshot()
        self.scan_state_var.set(badge)
        self.scan_state_detail_var.set(toolbar_detail)
        self.scan_state_info_var.set(info_detail)
        if self.scan_state_badge is not None:
            self.scan_state_badge.configure(bg=bg, fg=fg)

    def _build_heatmap_legend_row(self) -> None:
        markers = (
            ("○", "#d62728", "best chi2"),
            ("□", "#1f77b4", "best rho2"),
            ("△", "#2ca02c", "best eta2"),
            ("×", "#444444", "selected"),
            ("*", "#f08c00", "active"),
        )
        for index, (symbol, color, label) in enumerate(markers):
            item = ttk.Frame(self.heatmap_legend_row)
            item.grid(row=index // 3, column=index % 3, sticky="w", padx=(0, 10), pady=(0, 2))
            tk.Label(item, text=symbol, fg=color, font=("TkDefaultFont", 12, "bold")).pack(side=tk.LEFT)
            ttk.Label(item, text=label).pack(side=tk.LEFT, padx=(4, 0))

    def _apply_figure_autolayout(self, figure: Figure) -> None:
        apply_figure_autolayout(figure)

    def _build_shared_grid_axes_control(self, parent: ttk.Frame) -> None:
        shared_axes_check = ttk.Checkbutton(
            parent,
            text="Shared grid axes",
            variable=self.shared_heatmap_axes_var,
            command=self._on_shared_heatmap_axes_changed,
        )
        shared_axes_check.pack(side=tk.LEFT, padx=(2, 4))
        self.shared_heatmap_axes_check = shared_axes_check
        _ToolTip(
            shared_axes_check,
            "Use the same (a, b) axis limits for every slice (plot fills the panel; cells are not forced square).",
        )

    def _refresh_shared_heatmap_extents(self) -> None:
        self._shared_heatmap_extents = None
        if self.artifact_h5 is None:
            return
        slice_keys = [str(descriptor.get("key", "")).strip() for descriptor in list(self.available_slices or [])]
        slice_keys = [key for key in slice_keys if key]
        if not slice_keys:
            selected_key = str(self.payload.get("selected_slice_key", "") or "").strip()
            if selected_key:
                slice_keys = [selected_key]
        if not slice_keys:
            return
        try:
            self._shared_heatmap_extents = load_shared_grid_extents(
                Path(self.artifact_h5),
                slice_keys=slice_keys,
            )
        except Exception:
            self._shared_heatmap_extents = None

    def _on_shared_heatmap_axes_changed(self) -> None:
        enabled = bool(self.shared_heatmap_axes_var.get())
        _save_shared_grid_axes_pref(enabled)
        self._shared_heatmap_extents = None
        if enabled:
            self._refresh_shared_heatmap_extents()
        self._refresh_all(update_selected_solution=False)

    def _build_trials_controls(self) -> None:
        if self.trials_controls is None:
            return

        # Layout lock-in: this Trials/footer arrangement has been explicitly
        # user-reviewed on macOS. Do not change the footer control geometry,
        # button style, or left-panel trial summary placement unless explicitly
        # requested.
        controls = self.trials_controls
        for column in range(8):
            controls.columnconfigure(column, weight=0)
        controls.columnconfigure(7, weight=1)

        ttk.Label(controls, text="x").grid(row=0, column=0, sticky="w")
        xmin_entry = ttk.Entry(controls, width=9, textvariable=self.trials_xmin_var)
        xmin_entry.grid(row=0, column=1, sticky="w", padx=(4, 0))
        ttk.Label(controls, text="-").grid(row=0, column=2, sticky="w")
        xmax_entry = ttk.Entry(controls, width=9, textvariable=self.trials_xmax_var)
        xmax_entry.grid(row=0, column=3, sticky="w", padx=(0, 6))

        xscale_menu = ttk.Combobox(
            controls,
            width=12,
            state="readonly",
            values=("linear scale", "log scale"),
            textvariable=self.trials_xscale_var,
        )
        xscale_menu.grid(row=0, column=4, sticky="w", padx=(0, 6))
        xscale_menu.bind("<<ComboboxSelected>>", lambda _event: self._refresh_all())
        reset_x_button = tk.Button(
            controls,
            image=self._reset_icon_image,
            width=22,
            height=22,
            relief=tk.GROOVE,
            borderwidth=1,
            padx=0,
            pady=0,
            command=self._reset_trials_x_axis,
        )
        reset_x_button.grid(row=0, column=5, sticky="w")

        ttk.Label(controls, text="y").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ymin_entry = ttk.Entry(controls, width=9, textvariable=self.trials_ymin_var)
        ymin_entry.grid(row=1, column=1, sticky="w", padx=(4, 0), pady=(8, 0))
        ymax_entry = ttk.Entry(controls, width=9, textvariable=self.trials_ymax_var)
        ttk.Label(controls, text="-").grid(row=1, column=2, sticky="w", pady=(8, 0))
        ymax_entry.grid(row=1, column=3, sticky="w", padx=(0, 6), pady=(8, 0))

        yscale_menu = ttk.Combobox(
            controls,
            width=12,
            state="readonly",
            values=("linear scale", "log scale"),
            textvariable=self.trials_yscale_var,
        )
        yscale_menu.grid(row=1, column=4, sticky="w", padx=(0, 6), pady=(8, 0))
        yscale_menu.bind("<<ComboboxSelected>>", lambda _event: self._refresh_all())
        reset_y_button = tk.Button(
            controls,
            image=self._reset_icon_image,
            width=22,
            height=22,
            relief=tk.GROOVE,
            borderwidth=1,
            padx=0,
            pady=0,
            command=self._reset_trials_y_axis,
        )
        reset_y_button.grid(row=1, column=5, sticky="w", pady=(8, 0))

        ttk.Label(controls, text="trial").grid(row=2, column=0, sticky="w", pady=(8, 0))
        trial_slider = ttk.Scale(controls, from_=0.0, to=0.0, orient=tk.HORIZONTAL, command=self._on_trial_slider_changed)
        trial_slider.grid(row=2, column=1, columnspan=4, sticky="ew", padx=(4, 6), pady=(8, 0))
        self.trial_slider = trial_slider
        best_button = tk.Button(
            controls,
            image=self._best_trial_icon_image,
            width=22,
            height=22,
            relief=tk.GROOVE,
            borderwidth=1,
            padx=0,
            pady=0,
            command=self._jump_to_best_trial,
        )
        best_button.grid(row=2, column=5, sticky="w", pady=(8, 0))
        self.trial_best_button = best_button

        _ToolTip(xmin_entry, "X range minimum: enter a lower q0 limit and press Enter to redraw the Trials plot.")
        _ToolTip(xmax_entry, "X range maximum: enter an upper q0 limit and press Enter to redraw the Trials plot.")
        _ToolTip(xscale_menu, "X scale: switch the Trials plot q0 axis between linear and logarithmic scaling.")
        _ToolTip(reset_x_button, "Reset X axis: restore the Trials plot x-axis to its default linear autoscaled view for this slice.")
        _ToolTip(ymin_entry, "Y range minimum: enter a lower metric limit and press Enter to redraw the Trials plot.")
        _ToolTip(ymax_entry, "Y range maximum: enter an upper metric limit and press Enter to redraw the Trials plot.")
        _ToolTip(yscale_menu, "Y scale: switch the Trials plot metric axis between linear and logarithmic scaling.")
        _ToolTip(reset_y_button, "Reset Y axis: restore the Trials plot y-axis to its default linear autoscaled view for this slice.")
        _ToolTip(trial_slider, "Trial slider: move the selected trial pointer across the stored q0 trials for the current point.")
        _ToolTip(best_button, "Best trial: jump the selected trial pointer to the optimal trial for the displayed q0-curve metric.")

        for entry in (xmin_entry, xmax_entry, ymin_entry, ymax_entry):
            entry.bind("<Return>", lambda _event: self._commit_trials_axis_entries())

    def _parse_axis_limit(self, text: str) -> float | None:
        value = str(text).strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def _normalize_scale_choice(self, value: str) -> str:
        text = str(value).strip().lower()
        if text.startswith("log"):
            return "log"
        return "linear"

    def _display_scale_choice(self, value: str) -> str:
        return "log scale" if self._normalize_scale_choice(value) == "log" else "linear scale"

    def _commit_trials_axis_entries(self) -> None:
        self._trials_xmin_manual = bool(str(self.trials_xmin_var.get()).strip())
        self._trials_xmax_manual = bool(str(self.trials_xmax_var.get()).strip())
        self._trials_ymin_manual = bool(str(self.trials_ymin_var.get()).strip())
        self._trials_ymax_manual = bool(str(self.trials_ymax_var.get()).strip())
        self._refresh_all()

    def _apply_trials_axis_controls(self, *, q0_trials: np.ndarray, metric_trials: np.ndarray) -> None:
        xscale = self._normalize_scale_choice(str(self.trials_xscale_var.get() or "linear scale"))
        yscale = self._normalize_scale_choice(str(self.trials_yscale_var.get() or "linear scale"))

        if xscale == "log":
            positive_q0 = q0_trials[np.isfinite(q0_trials) & (q0_trials > 0)]
            if positive_q0.size:
                self.ax_trials.set_xscale("log")
        else:
            self.ax_trials.set_xscale("linear")

        if yscale == "log":
            positive_metric = metric_trials[np.isfinite(metric_trials) & (metric_trials > 0)]
            if positive_metric.size:
                self.ax_trials.set_yscale("log")
        else:
            self.ax_trials.set_yscale("linear")

        xmin = self._parse_axis_limit(self.trials_xmin_var.get()) if self._trials_xmin_manual else None
        xmax = self._parse_axis_limit(self.trials_xmax_var.get()) if self._trials_xmax_manual else None
        ymin = self._parse_axis_limit(self.trials_ymin_var.get()) if self._trials_ymin_manual else None
        ymax = self._parse_axis_limit(self.trials_ymax_var.get()) if self._trials_ymax_manual else None

        if self.ax_trials.get_xscale() == "log":
            if xmin is not None and xmin <= 0:
                xmin = None
            if xmax is not None and xmax <= 0:
                xmax = None
        if self.ax_trials.get_yscale() == "log":
            if ymin is not None and ymin <= 0:
                ymin = None
            if ymax is not None and ymax <= 0:
                ymax = None

        if xmin is not None and xmax is not None and xmin < xmax:
            self.ax_trials.set_xlim(xmin, xmax)
        elif xmin is not None:
            self.ax_trials.set_xlim(left=xmin)
        elif xmax is not None:
            self.ax_trials.set_xlim(right=xmax)

        if ymin is not None and ymax is not None and ymin < ymax:
            self.ax_trials.set_ylim(ymin, ymax)
        elif ymin is not None:
            self.ax_trials.set_ylim(bottom=ymin)
        elif ymax is not None:
            self.ax_trials.set_ylim(top=ymax)

    def _sync_trials_axis_controls_from_axes(self) -> None:
        x0, x1 = self.ax_trials.get_xlim()
        y0, y1 = self.ax_trials.get_ylim()

        self.trials_xmin_var.set(f"{float(x0):.6g}")
        self.trials_xmax_var.set(f"{float(x1):.6g}")
        self.trials_ymin_var.set(f"{float(y0):.6g}")
        self.trials_ymax_var.set(f"{float(y1):.6g}")

        self.trials_xscale_var.set(self._display_scale_choice(str(self.ax_trials.get_xscale())))
        self.trials_yscale_var.set(self._display_scale_choice(str(self.ax_trials.get_yscale())))

    def _trials_axis_view_from_parent(self) -> dict[str, Any]:
        """Limits and scales currently shown on the main viewer trials axis."""
        ax = getattr(self, "ax_trials", None)
        if ax is not None:
            has_curve = any(
                len(np.asarray(line.get_xdata(), dtype=float).reshape(-1)) > 0 for line in ax.lines
            )
            if has_curve:
                x0, x1 = ax.get_xlim()
                y0, y1 = ax.get_ylim()
                return {
                    "trials_xlim": (float(x0), float(x1)),
                    "trials_ylim": (float(y0), float(y1)),
                    "trials_xscale": self._normalize_scale_choice(str(ax.get_xscale())),
                    "trials_yscale": self._normalize_scale_choice(str(ax.get_yscale())),
                    "trials_match_parent_view": True,
                }
        return {
            "trials_xmin": self._parse_axis_limit(self.trials_xmin_var.get()),
            "trials_xmax": self._parse_axis_limit(self.trials_xmax_var.get()),
            "trials_ymin": self._parse_axis_limit(self.trials_ymin_var.get()),
            "trials_ymax": self._parse_axis_limit(self.trials_ymax_var.get()),
            "trials_xscale": str(self.trials_xscale_var.get() or "linear"),
            "trials_yscale": str(self.trials_yscale_var.get() or "linear"),
            "trials_match_parent_view": False,
        }

    def _reset_trials_view(self) -> None:
        self.trials_xscale_var.set("linear scale")
        self.trials_yscale_var.set("linear scale")
        self._trials_xmin_manual = False
        self._trials_xmax_manual = False
        self._trials_ymin_manual = False
        self._trials_ymax_manual = False
        self.trials_xmin_var.set("")
        self.trials_xmax_var.set("")
        self.trials_ymin_var.set("")
        self.trials_ymax_var.set("")
        self._refresh_all()

    def _reset_trials_x_axis(self) -> None:
        self.trials_xscale_var.set("linear scale")
        self._trials_xmin_manual = False
        self._trials_xmax_manual = False
        self.trials_xmin_var.set("")
        self.trials_xmax_var.set("")
        self._refresh_all()

    def _reset_trials_y_axis(self) -> None:
        self.trials_yscale_var.set("linear scale")
        self._trials_ymin_manual = False
        self._trials_ymax_manual = False
        self.trials_ymin_var.set("")
        self.trials_ymax_var.set("")
        self._refresh_all()

    def _slice_label(self, descriptor: dict[str, Any]) -> str:
        display_label = str(descriptor.get("display_label", "")).strip()
        if display_label:
            return display_label
        label = str(descriptor.get("label", descriptor.get("key", ""))).strip()
        return label or "default"

    def _unique_menu_labels(self, labels: list[str], ids: list[str]) -> list[str]:
        counts: dict[str, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        out: list[str] = []
        for label, item_id in zip(labels, ids, strict=False):
            if counts.get(label, 0) > 1 and str(item_id).strip():
                out.append(f"{label} [{item_id}]")
            else:
                out.append(label)
        return out

    def _selected_slice_key(self) -> str | None:
        slice_key_var = getattr(self, "slice_key_var", None)
        if slice_key_var is not None:
            try:
                value = str(slice_key_var.get()).strip()
            except Exception:
                value = ""
            if value:
                return value
        payload = getattr(self, "payload", {}) or {}
        value = str(payload.get("selected_slice_key", "")).strip()
        return value or None

    def _selected_search_id(self) -> str | None:
        search_id_var = getattr(self, "search_id_var", None)
        if search_id_var is not None:
            value = str(search_id_var.get()).strip()
            return value or None
        payload = getattr(self, "payload", {}) or {}
        value = str(payload.get("selected_search_id", "")).strip()
        return value or None

    def _slice_state_token(self, slice_key: str | None = None) -> str | None:
        if self.artifact_h5 is None:
            return None
        key = str(slice_key or self.payload.get("selected_slice_key") or self._selected_slice_key() or "").strip()
        if not key:
            return None
        try:
            artifact_text = str(self.artifact_h5.expanduser().resolve())
        except Exception:
            artifact_text = str(self.artifact_h5)
        search = str(self.search_id_var.get()).strip()
        return f"{artifact_text}::{key}::{search}"

    def _default_point_selection(self, metric_name: str) -> tuple[int, int]:
        if not self.payload:
            return 0, 0
        try:
            return default_point_index(self.payload, metric_name)
        except Exception:
            return 0, 0

    def _is_valid_grid_index(self, a_index: int, b_index: int) -> bool:
        a_size = int(np.asarray(getattr(self, "a_values", ()), dtype=float).size)
        b_size = int(np.asarray(getattr(self, "b_values", ()), dtype=float).size)
        if a_size <= 0 or b_size <= 0:
            return False
        return 0 <= int(a_index) < a_size and 0 <= int(b_index) < b_size

    def _resolve_restored_point_indices(
        self,
        *,
        metric_name: str,
        a_index: int,
        b_index: int,
        default_a: int,
        default_b: int,
    ) -> tuple[int, int]:
        key = (int(a_index), int(b_index))
        if self._is_valid_grid_index(key[0], key[1]):
            if self._navigation_mode() == "free":
                return key
            points = dict(self.payload.get("points", {}))
            if key not in points:
                return key
            status = str(points[key].get("status", "computed")).strip().lower()
            if status in {"missing", "pending"}:
                return key
        try:
            return resolve_point_index(
                self.payload,
                metric=metric_name,
                a_index=int(a_index),
                b_index=int(b_index),
            )
        except Exception:
            return default_a, default_b

    def _free_grid_selection_ab(self) -> tuple[float, float] | None:
        if self._navigation_mode() != "free":
            return None
        pending = getattr(self, "_free_selection_ab", None)
        if pending is not None:
            try:
                a_value = float(pending[0])
                b_value = float(pending[1])
            except Exception:
                return None
            if np.isfinite(a_value) and np.isfinite(b_value):
                return a_value, b_value
        a_index_var = getattr(self, "a_index_var", None)
        b_index_var = getattr(self, "b_index_var", None)
        if a_index_var is None or b_index_var is None:
            return None
        a_index = int(a_index_var.get())
        b_index = int(b_index_var.get())
        if not self._is_valid_grid_index(a_index, b_index):
            return None
        a_values = np.asarray(getattr(self, "a_values", ()), dtype=float)
        b_values = np.asarray(getattr(self, "b_values", ()), dtype=float)
        return float(a_values[a_index]), float(b_values[b_index])

    def _set_free_grid_selection(self, a_value: float, b_value: float) -> None:
        if self._navigation_mode() != "free":
            return
        self._free_selection_ab = (float(a_value), float(b_value))
        payload = dict(getattr(self, "payload", None) or {})
        if "a_values" not in payload:
            payload["a_values"] = np.asarray(getattr(self, "a_values", ()), dtype=float)
        if "b_values" not in payload:
            payload["b_values"] = np.asarray(getattr(self, "b_values", ()), dtype=float)
        resolved = grid_indices_for_coordinates(payload, float(a_value), float(b_value))
        if resolved is None:
            resolved = self._point_indices_for_coordinates(float(a_value), float(b_value))
        if resolved is not None:
            self.a_index_var.set(int(resolved[0]))
            self.b_index_var.set(int(resolved[1]))

    def _sync_free_selection_from_indices(self) -> None:
        if self._navigation_mode() != "free":
            return
        a_index_var = getattr(self, "a_index_var", None)
        b_index_var = getattr(self, "b_index_var", None)
        if a_index_var is None or b_index_var is None:
            return
        a_index = int(a_index_var.get())
        b_index = int(b_index_var.get())
        if not self._is_valid_grid_index(a_index, b_index):
            return
        a_values = np.asarray(getattr(self, "a_values", ()), dtype=float)
        b_values = np.asarray(getattr(self, "b_values", ()), dtype=float)
        self._free_selection_ab = (
            float(a_values[a_index]),
            float(b_values[b_index]),
        )

    def _capture_free_grid_selection_coords(self) -> None:
        try:
            if self._navigation_mode() != "free":
                return
            selection = self._free_grid_selection_ab()
            if selection is None:
                return
            self._free_selection_ab = selection
        except Exception:
            return

    def _restore_free_grid_selection_from_coords(self) -> bool:
        pending = getattr(self, "_free_selection_ab", None)
        if pending is None:
            return False
        try:
            a_value = float(pending[0])
            b_value = float(pending[1])
        except Exception:
            return False
        if not (np.isfinite(a_value) and np.isfinite(b_value)):
            return False
        self._set_free_grid_selection(a_value, b_value)
        return True

    def _free_grid_selection_indices(self) -> tuple[int, int] | None:
        if self._navigation_mode() != "free":
            return None
        selection = self._free_grid_selection_ab()
        if selection is None:
            return None
        payload = dict(getattr(self, "payload", None) or {})
        if "a_values" not in payload:
            payload["a_values"] = np.asarray(getattr(self, "a_values", ()), dtype=float)
        if "b_values" not in payload:
            payload["b_values"] = np.asarray(getattr(self, "b_values", ()), dtype=float)
        resolved = grid_indices_for_coordinates(payload, float(selection[0]), float(selection[1]))
        if resolved is not None:
            return resolved
        return self._point_indices_for_coordinates(float(selection[0]), float(selection[1]))

    def _point_payload_for_selection(self) -> dict[str, Any] | None:
        resolved = self._resolved_point_index(metric_name=self._trials_display_metric())
        if resolved is None:
            return None
        points = dict(self.payload.get("points", {}))
        if resolved not in points:
            return None
        return dict(points[resolved])

    def _selected_grid_point_has_solution(self) -> bool:
        if not self._has_saved_selected_point():
            return False
        point = self._point_payload_for_selection()
        if not point:
            return False
        q0_trials, metric_trials, _point_metric = self._trial_series_for_point(point)
        return (
            q0_trials.size > 0
            and metric_trials.size == q0_trials.size
            and np.any(np.isfinite(metric_trials))
        )

    def _resolved_point_index(self, *, metric_name: str | None = None) -> tuple[int, int] | None:
        payload = getattr(self, "payload", None) or {}
        if not payload:
            return None
        a_index_var = getattr(self, "a_index_var", None)
        b_index_var = getattr(self, "b_index_var", None)
        if a_index_var is None or b_index_var is None:
            return None
        if self._navigation_mode() == "free":
            free_indices = self._free_grid_selection_indices()
            if free_indices is not None:
                return free_indices
            return None
        a_index = int(a_index_var.get())
        b_index = int(b_index_var.get())
        points = dict(self.payload.get("points", {}))
        key = (a_index, b_index)
        if key in points:
            status = str(points[key].get("status", "computed")).strip().lower()
            if status not in {"missing", "pending"}:
                return key
            return None
        if self._is_valid_grid_index(a_index, b_index):
            if self._selection_matches_live_active_point(a_index, b_index):
                return key
            return None
        try:
            return resolve_point_index(
                self.payload,
                metric=metric_name or str(self.metric_var.get()),
                a_index=a_index,
                b_index=b_index,
            )
        except Exception:
            return None

    def _ensure_selected_point_exists(self, *, metric_name: str | None = None) -> bool:
        if self._navigation_mode() == "free":
            return self._free_grid_selection_ab() is not None
        if self._navigation_mode() == "active":
            if self._live_navigation_available():
                resolved_active = self._active_point_indices()
                if resolved_active is not None:
                    a_index, b_index = resolved_active
                    if int(self.a_index_var.get()) != int(a_index) or int(self.b_index_var.get()) != int(b_index):
                        self.a_index_var.set(int(a_index))
                        self.b_index_var.set(int(b_index))
                        self._refresh_selector_values()
                    return True
            followed_coords = self._followed_point_coords()
            if followed_coords is not None:
                resolved = self._point_indices_for_coordinates(float(followed_coords[0]), float(followed_coords[1]))
                if resolved is not None:
                    a_index, b_index = resolved
                    self._refresh_signal_active_point = followed_coords
                    if int(self.a_index_var.get()) != int(a_index) or int(self.b_index_var.get()) != int(b_index):
                        self.a_index_var.set(int(a_index))
                        self.b_index_var.set(int(b_index))
                        self._refresh_selector_values()
                    return True
            if self._resolved_point_index(metric_name=metric_name or self._trials_display_metric()) is not None:
                return True
            return False
        if self._navigation_mode() == "best":
            resolved_best = self._best_point_index(metric_name)
            if resolved_best is None:
                return False
            a_index, b_index = resolved_best
            if int(self.a_index_var.get()) != int(a_index) or int(self.b_index_var.get()) != int(b_index):
                self.a_index_var.set(int(a_index))
                self.b_index_var.set(int(b_index))
                self._refresh_selector_values()
            return True
        resolved = self._resolved_point_index(metric_name=metric_name)
        if resolved is None:
            return False
        a_index, b_index = resolved
        changed = int(self.a_index_var.get()) != int(a_index) or int(self.b_index_var.get()) != int(b_index)
        if changed:
            self.a_index_var.set(int(a_index))
            self.b_index_var.set(int(b_index))
            self._refresh_selector_values()
        return True

    def _capture_current_slice_view_state(self, metric_name: str | None = None) -> None:
        token = self._slice_state_token()
        if token is None:
            return
        existing = dict(self.slice_display_state.get(token, {}))
        per_metric = dict(existing.get("trials_by_metric", {}))
        current_metric = str(metric_name or self.metric_var.get())
        if current_metric in METRICS:
            per_metric[current_metric] = {
                "trials_xmin": str(self.trials_xmin_var.get()),
                "trials_xmax": str(self.trials_xmax_var.get()),
                "trials_ymin": str(self.trials_ymin_var.get()),
                "trials_ymax": str(self.trials_ymax_var.get()),
                "trials_xmin_manual": bool(self._trials_xmin_manual),
                "trials_xmax_manual": bool(self._trials_xmax_manual),
                "trials_ymin_manual": bool(self._trials_ymin_manual),
                "trials_ymax_manual": bool(self._trials_ymax_manual),
                "trials_xscale": self._normalize_scale_choice(str(self.trials_xscale_var.get() or "linear scale")),
                "trials_yscale": self._normalize_scale_choice(str(self.trials_yscale_var.get() or "linear scale")),
            }
        existing.update(
            {
                "metric": current_metric,
                "a_index": int(self.a_index_var.get()),
                "b_index": int(self.b_index_var.get()),
                "navigation_mode": self._navigation_mode(),
                "trials_by_metric": per_metric,
            }
        )
        self.slice_display_state[token] = existing

    def _reset_trials_controls_only(self) -> None:
        self.trials_xscale_var.set("linear scale")
        self.trials_yscale_var.set("linear scale")
        self._trials_xmin_manual = False
        self._trials_xmax_manual = False
        self._trials_ymin_manual = False
        self._trials_ymax_manual = False
        self.trials_xmin_var.set("")
        self.trials_xmax_var.set("")
        self.trials_ymin_var.set("")
        self.trials_ymax_var.set("")

    def _restore_trials_controls_for_metric(self, metric_name: str) -> None:
        token = self._slice_state_token()
        state = self.slice_display_state.get(token or "", {})
        per_metric = dict(state.get("trials_by_metric", {}))
        metric_state = dict(per_metric.get(str(metric_name), {}))
        if not metric_state:
            self._reset_trials_controls_only()
            return
        self.trials_xscale_var.set(self._display_scale_choice(str(metric_state.get("trials_xscale", "linear") or "linear")))
        self.trials_yscale_var.set(self._display_scale_choice(str(metric_state.get("trials_yscale", "linear") or "linear")))
        self.trials_xmin_var.set(str(metric_state.get("trials_xmin", "")))
        self.trials_xmax_var.set(str(metric_state.get("trials_xmax", "")))
        self.trials_ymin_var.set(str(metric_state.get("trials_ymin", "")))
        self.trials_ymax_var.set(str(metric_state.get("trials_ymax", "")))
        self._trials_xmin_manual = bool(metric_state.get("trials_xmin_manual", bool(str(metric_state.get("trials_xmin", "")).strip())))
        self._trials_xmax_manual = bool(metric_state.get("trials_xmax_manual", bool(str(metric_state.get("trials_xmax", "")).strip())))
        self._trials_ymin_manual = bool(metric_state.get("trials_ymin_manual", bool(str(metric_state.get("trials_ymin", "")).strip())))
        self._trials_ymax_manual = bool(metric_state.get("trials_ymax_manual", bool(str(metric_state.get("trials_ymax", "")).strip())))

    def _restore_slice_view_state(self) -> None:
        selected_key = str(self.payload.get("selected_slice_key", "")).strip()
        token = self._slice_state_token(selected_key)
        metric_name = (
            self._preferred_initial_metric
            or (str(self.metric_var.get()) if self.metric_var.get() in METRICS else self.run_target_metric)
            or self.run_target_metric
        )
        default_a, default_b = self._default_point_selection(metric_name)
        state = self.slice_display_state.get(token or "", None)

        if state is None:
            self.metric_var.set(metric_name if metric_name in METRICS else "chi2")
            if self._navigation_mode() == "free" and self._restore_free_grid_selection_from_coords():
                self._preferred_initial_metric = None
                self._reset_trials_controls_only()
                return
            if self._navigation_mode() == "best":
                resolved_best = self._best_point_index(metric_name)
                if resolved_best is not None:
                    default_a, default_b = resolved_best
            elif self._navigation_mode() == "free" and self._is_valid_grid_index(
                int(self.a_index_var.get()),
                int(self.b_index_var.get()),
            ):
                default_a = int(self.a_index_var.get())
                default_b = int(self.b_index_var.get())
            self.a_index_var.set(default_a)
            self.b_index_var.set(default_b)
            if self._navigation_mode() == "free":
                self._sync_free_selection_from_indices()
            self._reset_trials_controls_only()
            self._preferred_initial_metric = None
            return

        if not getattr(self, "_open_session_prefers_active", False):
            saved_mode = str(state.get("navigation_mode", "") or "").strip().lower()
            if saved_mode == "active" and not self._active_navigation_mode_available():
                saved_mode = "best" if self._best_navigation_available() else "free"
            if saved_mode in NAVIGATION_MODES:
                self.navigation_mode_var.set(saved_mode)
                if saved_mode != "free" or not self._live_runner_detected():
                    self._navigation_mode_user_chosen = True
        self._coerce_navigation_mode_to_availability()

        saved_metric = str(state.get("metric", metric_name))
        if saved_metric not in METRICS:
            saved_metric = metric_name if metric_name in METRICS else "chi2"
        self.metric_var.set(saved_metric)

        if self._navigation_mode() == "free" and self._restore_free_grid_selection_from_coords():
            self._preferred_initial_metric = None
            self._reset_trials_controls_only()
            return
        if self._navigation_mode() == "best":
            resolved_best = self._best_point_index(saved_metric)
            if resolved_best is not None:
                restored_a, restored_b = resolved_best
            else:
                restored_a, restored_b = default_a, default_b
        else:
            restored_a, restored_b = self._resolve_restored_point_indices(
                metric_name=saved_metric,
                a_index=int(state.get("a_index", default_a)),
                b_index=int(state.get("b_index", default_b)),
                default_a=default_a,
                default_b=default_b,
            )
        self.a_index_var.set(restored_a)
        self.b_index_var.set(restored_b)
        if self._navigation_mode() == "free":
            self._sync_free_selection_from_indices()
        self._preferred_initial_metric = None
        self._reset_trials_controls_only()

    def _refresh_slice_controls(self) -> None:
        artifact_h5 = getattr(self, "artifact_h5", None)
        if artifact_h5 is not None:
            try:
                catalog_slices = list(list_scan_slices(Path(artifact_h5)))
                if catalog_slices:
                    self.available_slices = catalog_slices
            except Exception:
                pass
        descriptors = list(getattr(self, "available_slices", []) or [])
        labels = [self._slice_label(item) for item in descriptors]
        keys = [str(item.get("key", "")) for item in descriptors]
        labels = self._unique_menu_labels(labels, keys)
        selected_key = str(self.payload.get("selected_slice_key", self._selected_slice_key() or "")).strip()
        if selected_key and selected_key in keys:
            selected_index = keys.index(selected_key)
        else:
            selected_index = 0 if keys else -1
            selected_key = keys[selected_index] if selected_index >= 0 else ""
        self.slice_key_var.set(selected_key)
        self.slice_display_var.set(labels[selected_index] if selected_index >= 0 else "")
        if self.slice_menu is not None:
            self.slice_menu.configure(values=labels)
            if len(labels) > 1:
                self.slice_menu.grid()
                if self.slice_display_label is not None:
                    self.slice_display_label.grid_remove()
                self.slice_menu.current(selected_index)
            else:
                self.slice_menu.grid_remove()
                if self.slice_display_label is not None:
                    self.slice_display_label.grid()
        elif self.slice_display_label is not None:
            self.slice_display_label.grid()

    def _search_label(self, record: dict[str, Any]) -> str:
        search_id = str(record.get("search_id", "")).strip() or "legacy_current"
        lifecycle = dict(record.get("lifecycle") or {})
        in_progress = bool(record.get("in_progress", lifecycle.get("in_progress", False)))
        suffix = " *" if in_progress else ""
        return f"{search_id}{suffix}"

    def _refresh_search_controls(self) -> None:
        records = list(getattr(self, "available_searches", []) or [])
        labels = [self._search_label(record) for record in records]
        ids = [str(record.get("search_id", "")) for record in records]
        labels = self._unique_menu_labels(labels, ids)
        selected_id = str(self.payload.get("selected_search_id", self._selected_search_id() or "") or "").strip()
        if selected_id and selected_id in ids:
            selected_index = ids.index(selected_id)
        else:
            selected_index = 0 if ids else -1
            selected_id = ids[selected_index] if selected_index >= 0 else ""
        self.search_id_var.set(selected_id)
        self.search_display_var.set(labels[selected_index] if selected_index >= 0 else "legacy current")
        if self.search_menu is not None:
            self.search_menu.configure(values=labels)
            if labels:
                self.search_menu.grid()
                if self.search_display_label is not None:
                    self.search_display_label.grid_remove()
                self.search_menu.current(selected_index)
            else:
                self.search_menu.grid_remove()
                if self.search_display_label is not None:
                    self.search_display_label.grid()
        elif self.search_display_label is not None:
            self.search_display_label.grid()

    def _refresh_selector_values(self) -> None:
        self._refresh_slice_controls()
        self._refresh_search_controls()
        a_labels = [f"{i}: {value:.3f}" for i, value in enumerate(self.a_values)]
        b_labels = [f"{i}: {value:.3f}" for i, value in enumerate(self.b_values)]
        self.a_menu.configure(values=a_labels)
        self.b_menu.configure(values=b_labels)
        if a_labels:
            self.a_menu.current(int(np.clip(self.a_index_var.get(), 0, max(0, len(a_labels) - 1))))
        if b_labels:
            self.b_menu.current(int(np.clip(self.b_index_var.get(), 0, max(0, len(b_labels) - 1))))

    def _reload_payload(self, artifact_path: Path | None = None) -> None:
        self._capture_current_slice_view_state(self._last_rendered_metric)
        if artifact_path is not None:
            try:
                resolved_path = artifact_path.expanduser().resolve()
            except Exception:
                resolved_path = Path(artifact_path).expanduser()
            if self.artifact_h5 is None or str(resolved_path) != str(self.artifact_h5):
                self._navigation_mode_initialized = False
                self._navigation_mode_user_chosen = False
                self._open_global_best_applied = False
                self._stale_active_notice = ""
            try:
                self.artifact_h5 = resolved_path
            except Exception:
                self.artifact_h5 = Path(artifact_path).expanduser()
        if self.artifact_h5 is None:
            self.status_var.set("No artifact loaded. Use Open Artifact to choose a consolidated scan H5 file.")
            self.summary_var.set("No artifact loaded.")
            self.payload = {}
            self.available_slices = []
            self.available_searches = []
            self.slice_key_var.set("")
            self.slice_display_var.set("")
            self.search_id_var.set("")
            self.search_display_var.set("")
            self.a_values = np.asarray([], dtype=float)
            self.b_values = np.asarray([], dtype=float)
            self.refresh_signal_path = None
            self._refresh_signal_mtime_ns = -1
            self._refresh_signal_phase = ""
            self._refresh_signal_slice_key = None
            self._refresh_signal_pending_points = []
            self._refresh_signal_active_point = None
            self._refresh_signal_live_trials = None
            self._payload_cache_artifact_path = ""
            self._payload_cache_artifact_mtime_ns = -1
            self._payload_cache_by_selection = {}
            self._live_snapshot_cache_key = None
            self._live_snapshot_cache_value = None
            self._selected_trial_map_cache_key = None
            self._selected_trial_map_cache_value = None
            self._navigation_mode_initialized = False
            self._navigation_mode_user_chosen = False
            self._open_global_best_applied = False
            self._stale_active_notice = ""
            self._refresh_action_states()
            self._refresh_all()
            return
        self.last_artifact_dir = self.artifact_h5.expanduser().resolve().parent
        self._bind_refresh_signal_path()
        self._refresh_artifact_catalog()
        self._sync_refresh_signal_from_disk()
        _save_last_directory(self.last_artifact_dir)
        prev_a = int(self.a_index_var.get())
        prev_b = int(self.b_index_var.get())
        requested_slice_key = self._selected_slice_key()
        requested_search_id = self._selected_search_id()
        artifact_signature_path = str(self.artifact_h5.expanduser().resolve())
        try:
            artifact_mtime_ns = int(self.artifact_h5.stat().st_mtime_ns)
        except Exception:
            artifact_mtime_ns = -1
        if (
            artifact_signature_path != self._payload_cache_artifact_path
            or int(artifact_mtime_ns) != int(self._payload_cache_artifact_mtime_ns)
        ):
            self._payload_cache_artifact_path = artifact_signature_path
            self._payload_cache_artifact_mtime_ns = int(artifact_mtime_ns)
            self._payload_cache_by_selection = {}
            self._live_snapshot_cache_key = None
            self._live_snapshot_cache_value = None
            self._selected_trial_map_cache_key = None
            self._selected_trial_map_cache_value = None
        cache_key = (str(requested_slice_key or ""), str(requested_search_id or ""))
        cached_payload = self._payload_cache_by_selection.get(cache_key)
        try:
            if cached_payload is not None:
                self.payload = cached_payload
            else:
                self.payload = self._load_scan_file_with_retries(
                    self.artifact_h5,
                    slice_key=requested_slice_key,
                    search_id=requested_search_id,
                )
        except Exception as exc:
            self.status_var.set(
                "Artifact is currently being written or is temporarily locked. "
                "Please retry in a moment.\n"
                f"Details: {exc}"
            )
            self.summary_var.set("Could not refresh artifact right now. Existing view remains unchanged.")
            return
        selected_cache_key = (
            str(self.payload.get("selected_slice_key", "") or ""),
            str(self.payload.get("selected_search_id", "") or ""),
        )
        self._payload_cache_by_selection[cache_key] = self.payload
        self._payload_cache_by_selection[selected_cache_key] = self.payload
        self._apply_payload(self.payload)

    def _apply_payload(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        if self.artifact_h5 is not None:
            self.root.title(f"pychmp-view: {self.artifact_h5.name}")
        self._refresh_artifact_catalog()
        self._sync_refresh_signal_from_disk()
        self.available_searches = list(self.payload.get("search_records", []))
        open_prefers_active = bool(getattr(self, "_open_session_prefers_active", False))
        if open_prefers_active:
            target_slice_key, target_search_id = self._resolve_active_session_slice_search()
            loaded_slice_key = str(self.payload.get("selected_slice_key", "")).strip()
            if target_slice_key and target_slice_key != loaded_slice_key:
                self.slice_key_var.set(str(target_slice_key))
                if target_search_id:
                    self.search_id_var.set(str(target_search_id))
                self._schedule_payload_reload(status_text="Loading active slice...")
                return
            if target_search_id:
                self.search_id_var.set(str(target_search_id))
        else:
            self.slice_key_var.set(str(self.payload.get("selected_slice_key", "")))
            self.search_id_var.set(str(self.payload.get("selected_search_id") or ""))
        self.a_values = np.asarray(self.payload["a_values"], dtype=float)
        self.b_values = np.asarray(self.payload["b_values"], dtype=float)
        self.display_model = build_patch_grid_model(self.payload)
        self.run_target_metric = str(self.payload.get("target_metric", "chi2"))
        self._refresh_shared_heatmap_extents()
        self._sync_live_trial_state_from_artifact()
        self._restore_slice_view_state()
        self._initialize_navigation_mode()
        self._coerce_navigation_mode_to_availability()
        if open_prefers_active:
            self._open_session_prefers_active = False
        if self._navigation_mode() != "active" and self._apply_open_global_best_domain():
            return
        if self._navigation_mode() == "best":
            self._apply_best_navigation(force_reanchor=True)
        if self._apply_active_session_selection_if_needed():
            return
        if self._apply_locked_navigation_selection():
            return
        self._refresh_selector_values()
        self._refresh_action_states()
        self._refresh_all()
        self._last_payload_reload_at_s = float(time.time())
        self._live_snapshot_cache_key = None
        self._live_snapshot_cache_value = None
        self._selected_trial_map_cache_key = None
        self._selected_trial_map_cache_value = None

    def _schedule_payload_reload(self, *, status_text: str | None = None) -> None:
        if self._scheduled_reload_after_id is not None:
            try:
                self.root.after_cancel(self._scheduled_reload_after_id)
            except Exception:
                pass
            self._scheduled_reload_after_id = None
        if status_text:
            self.status_var.set(str(status_text))

        def _run() -> None:
            self._scheduled_reload_after_id = None
            if self._is_closing:
                return
            self._reload_payload()

        self._scheduled_reload_after_id = self.root.after_idle(_run)

    def _load_scan_file_with_retries(
        self,
        artifact_h5: Path,
        *,
        slice_key: str | None = None,
        search_id: str | None = None,
    ) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(1, self._LOAD_RETRY_ATTEMPTS + 1):
            try:
                return load_scan_file(artifact_h5, slice_key=slice_key, search_id=search_id, include_maps=False)
            except (BlockingIOError, PermissionError, OSError) as exc:
                last_exc = exc
                if attempt >= self._LOAD_RETRY_ATTEMPTS:
                    break
                self.status_var.set(
                    "Artifact is busy (possibly being written); "
                    f"retrying {attempt}/{self._LOAD_RETRY_ATTEMPTS - 1}..."
                )
                time.sleep(self._LOAD_RETRY_DELAY_S)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("unknown artifact loading failure")

    def _poll_external_refresh_signal(self) -> None:
        self._external_refresh_after_id = None
        if self._is_closing:
            return
        try:
            if bool(getattr(self, "_initial_reload_in_progress", False)):
                return
            if self.refresh_signal_path is None:
                return
            if self.refresh_signal_path.exists():
                mtime_ns = int(self.refresh_signal_path.stat().st_mtime_ns)
                if mtime_ns > int(self._refresh_signal_mtime_ns):
                    self._refresh_signal_mtime_ns = mtime_ns
                    refresh_payload = self._read_refresh_signal_payload()
                    self._refresh_signal_phase = str(refresh_payload.get("phase", ""))
                    event = str(refresh_payload.get("event", "") or "").strip().lower()
                    version = int(refresh_payload.get("version", 1) or 1)
                    if version >= 2 and event in REFRESH_EVENTS_REQUIRING_SLICE_RELOAD:
                        self._process_refresh_event(refresh_payload)
                    elif self._heartbeat_requires_payload_reload(refresh_payload):
                        self._reload_payload()
                    else:
                        self._apply_refresh_signal_payload(refresh_payload)
                        self._sync_live_trial_state_from_artifact()
                        if self._navigation_mode() in {"active", "best"}:
                            if self._apply_locked_navigation_selection():
                                return
                        self._refresh_scan_state_display()
                        self._refresh_action_states()
                        if self._navigation_mode() in {"active", "best"} and self.payload:
                            self._refresh_all(update_selected_solution=True)
        except Exception:
            pass
        finally:
            if not self._is_closing:
                try:
                    self._external_refresh_after_id = self.root.after(self._EXTERNAL_REFRESH_POLL_MS, self._poll_external_refresh_signal)
                except Exception:
                    self._external_refresh_after_id = None

    def _heartbeat_requires_payload_reload(self, refresh_payload: dict[str, Any]) -> bool:
        event = str(refresh_payload.get("event", "") or "").strip().lower()
        if int(refresh_payload.get("version", 1) or 1) >= 2 and event in REFRESH_EVENTS_REQUIRING_SLICE_RELOAD:
            return False
        if int(refresh_payload.get("version", 1) or 1) >= 2 and event:
            return False
        phase = str(refresh_payload.get("phase", "")).strip().lower()
        if not phase:
            return False
        write_markers = (
            "saved",
            "promoted",
            "initialized",
            "resume",
            "loaded",
            "scan complete",
            "scan failed",
            "scan interrupted",
        )
        if not any(marker in phase for marker in write_markers):
            return False
        now = float(time.time())
        if now - float(self._last_payload_reload_at_s) < float(self._MIN_HEARTBEAT_RELOAD_INTERVAL_S):
            return False
        return True

    def _current_metric_name(self) -> str:
        metric_var = getattr(self, "metric_var", None)
        if metric_var is not None:
            return str(metric_var.get())
        payload = getattr(self, "payload", None) or {}
        selected_search = dict(payload.get("selected_search") or {})
        search_request = dict(selected_search.get("request") or {})
        search_diagnostics = dict(selected_search.get("diagnostics") or {})
        metric_name = str(
            selected_search.get("target_metric")
            or search_request.get("target_metric")
            or search_diagnostics.get("target_metric")
            or getattr(self, "run_target_metric", None)
            or "chi2"
        ).strip()
        return metric_name if metric_name in METRICS else "chi2"

    def _selected_point(self) -> dict[str, Any]:
        resolved = self._resolved_point_index(metric_name=self._current_metric_name())
        if resolved is None:
            raise KeyError("no stored point is available")
        return self.payload["points"][resolved]

    def _has_selected_point(self) -> bool:
        if self._navigation_mode() == "free" and self._free_grid_selection_ab() is not None:
            return True
        return self._resolved_point_index(metric_name=self._trials_display_metric()) is not None

    def _has_saved_selected_point(self) -> bool:
        resolved = self._resolved_point_index(metric_name=self._trials_display_metric())
        if resolved is None or not getattr(self, "payload", None):
            return False
        points = dict(self.payload.get("points", {}))
        key = (int(resolved[0]), int(resolved[1]))
        if key not in points:
            return False
        status = str(points[key].get("status", "computed")).strip().lower()
        return status not in {"missing", "pending"}

    def _search_shift_request(self) -> dict[str, Any]:
        return dict((self.payload.get("selected_search") or {}).get("request") or {})

    def _diagnostics_with_search_shift(self, diagnostics: dict[str, Any]) -> dict[str, Any]:
        return apply_search_shift_diagnostics(diagnostics, search_request=self._search_shift_request())

    def _diagnostics_with_point_ab(
        self,
        diagnostics: dict[str, Any],
        point: dict[str, Any] | None,
    ) -> dict[str, Any]:
        merged = dict(diagnostics)
        if point is None:
            return merged
        for key in ("a", "b"):
            try:
                value = float(point.get(key, np.nan))
            except Exception:
                continue
            if np.isfinite(value):
                merged[key] = float(value)
        return merged

    def _display_observation_for_trial(
        self,
        *,
        point: dict[str, Any] | None = None,
        trial_index: int | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> np.ndarray:
        diag = self._diagnostics_with_search_shift(dict(diagnostics or self.payload.get("diagnostics") or {}))
        if point is not None:
            diag.update(dict(point.get("diagnostics") or {}))
            diag = self._diagnostics_with_search_shift(diag)
        header = self.payload.get("wcs_header")
        model_header = header if isinstance(header, fits.Header) else None
        shift_x = None if point is None else point.get("fit_shift_x_trials")
        shift_y = None if point is None else point.get("fit_shift_y_trials")
        if shift_x is None:
            shift_x = diag.get("fit_shift_x_trials")
        if shift_y is None:
            shift_y = diag.get("fit_shift_y_trials")
        observed, _sigma = resolve_trial_observation_for_display(
            observed=np.asarray(self.payload.get("observed"), dtype=float),
            sigma=self.payload.get("sigma_map"),
            model_header=model_header,
            diagnostics=diag,
            observation_canvas=self.payload.get("observation_canvas"),
            sigma_canvas=self.payload.get("sigma_canvas"),
            canvas_header=self.payload.get("canvas_wcs_header"),
            trial_index=trial_index,
            fit_shift_x_trials=shift_x,
            fit_shift_y_trials=shift_y,
        )
        return observed

    def _selected_diagnostics(self) -> dict[str, Any]:
        diagnostics = dict(self.payload["diagnostics"])
        diagnostics.update(self._selected_point()["diagnostics"])
        return self._diagnostics_with_search_shift(diagnostics)

    def _metrics_mask_summary(self, diagnostics: dict[str, Any]) -> str | None:
        source = str(diagnostics.get("metrics_mask_source", "")).strip().lower()
        if source == "explicit_fits":
            mask_path = str(diagnostics.get("metrics_mask_fits", "")).strip()
            return f"explicit FITS: {mask_path}" if mask_path else "explicit FITS"
        threshold = diagnostics.get("metrics_mask_threshold")
        try:
            threshold_value = float(threshold)
        except Exception:
            threshold_value = float("nan")
        if np.isfinite(threshold_value):
            return f"union threshold={threshold_value:.3f} + observed>0"
        mask_type = str(diagnostics.get("mask_type", "")).strip()
        if mask_type:
            return mask_type
        return None

    def _tr_mask_summary(self, diagnostics: dict[str, Any]) -> str | None:
        source = str(diagnostics.get("tr_mask_source", "")).strip().lower()
        if not source or source == "unavailable":
            return None if not source else "unavailable"
        threshold = diagnostics.get("tr_mask_bmin_gauss")
        try:
            threshold_value = float(threshold)
        except Exception:
            threshold_value = float("nan")
        if source in {"abs_blos_ge_bmin", "blos_ge_bmin"} and np.isfinite(threshold_value):
            prefix = "|B_los|" if source == "abs_blos_ge_bmin" else "B_los"
            return f"{prefix} >= {threshold_value:.1f} G"
        if np.isfinite(threshold_value):
            return f"{source} ({threshold_value:.1f} G)"
        return source

    def _selected_modeled_map_for_mask(self, point: dict[str, Any]) -> np.ndarray:
        modeled = np.asarray(point["modeled_best"], dtype=float)
        q0_trials, metric_trials, point_metric = self._trial_series_for_point(point)
        selected_trial_index = self._selected_trial_index_for_point(point, q0_trials, metric_trials, point_metric)
        if selected_trial_index is None:
            return modeled
        modeled_trial_maps = point.get("trial_modeled_maps")
        if modeled_trial_maps is None:
            return modeled
        modeled_trial_maps = np.asarray(modeled_trial_maps, dtype=float)
        if modeled_trial_maps.ndim < 3 or int(selected_trial_index) >= int(modeled_trial_maps.shape[0]):
            return modeled
        return np.asarray(modeled_trial_maps[int(selected_trial_index)], dtype=float)

    def _metrics_mask_pixel_summary(self, point: dict[str, Any], diagnostics: dict[str, Any]) -> str | None:
        observed = np.asarray(self.payload.get("observed"), dtype=float)
        modeled = self._selected_modeled_map_for_mask(point)
        if observed.shape != modeled.shape:
            return None
        source = str(diagnostics.get("metrics_mask_source", "")).strip().lower()
        mask: np.ndarray | None = None
        if source == "explicit_fits":
            mask_path = str(diagnostics.get("metrics_mask_fits", "")).strip()
            if not mask_path:
                return "unavailable"
            try:
                data = np.asarray(fits.getdata(Path(mask_path).expanduser()))
                data = np.squeeze(data)
                if data.shape != observed.shape:
                    return f"shape mismatch: {data.shape} vs {observed.shape}"
                mask = np.asarray(np.isfinite(data) & (data != 0), dtype=bool)
            except Exception:
                return "unavailable"
        else:
            mask_type = str(diagnostics.get("mask_type", "union")).strip() or "union"
            threshold_value = diagnostics.get("metrics_mask_threshold", diagnostics.get("threshold", 0.1))
            try:
                threshold = float(threshold_value)
                mask_fn = resolve_threshold_mask(mask_type)
                mask = np.asarray(mask_fn(observed, modeled, threshold), dtype=bool)
            except Exception:
                return "unavailable"
        mask = np.asarray(mask, dtype=bool) & (observed > 0)
        selected = int(np.count_nonzero(mask))
        total = int(mask.size)
        return f"{selected}/{total} ({selected / max(total, 1):.1%})"

    def _metric_history_for_point(self, point: dict[str, Any], metric_name: str) -> np.ndarray:
        q0_trials = np.asarray(point.get("fit_q0_trials", ()), dtype=float)
        metric_trials = np.asarray(point.get(f"fit_{metric_name}_trials", ()), dtype=float)
        if metric_trials.size == q0_trials.size and metric_trials.size > 0:
            return metric_trials
        if str(point.get("target_metric", "")) == str(metric_name):
            metric_trials = np.asarray(point.get("fit_metric_trials", ()), dtype=float)
            if metric_trials.size == q0_trials.size and metric_trials.size > 0:
                return metric_trials
        return np.asarray([], dtype=float)

    def _populate_metric_trial_histories_in_diagnostics(
        self,
        diagnostics: dict[str, Any],
        *,
        point: dict[str, Any] | None = None,
        live_state: dict[str, Any] | None = None,
    ) -> str:
        """Attach per-metric trial histories and the UI-selected display metric."""
        display_metric = self._trials_display_metric()
        diagnostics["trials_display_metric"] = display_metric
        diagnostics["target_metric"] = display_metric
        if point is not None:
            diagnostics["point_target_metric"] = str(point.get("target_metric", "")).strip().lower()
            q0 = np.asarray(point.get("fit_q0_trials", ()), dtype=float)
            if q0.ndim == 1 and q0.size > 0:
                diagnostics["fit_q0_trials"] = [float(v) for v in q0]
                for metric_name in METRICS:
                    history = self._metric_history_for_point(point, metric_name)
                    if history.size == q0.size and history.size > 0:
                        diagnostics[f"fit_{metric_name}_trials"] = [float(v) for v in history]
        if live_state is not None:
            if "point_target_metric" not in diagnostics:
                diagnostics["point_target_metric"] = str(
                    live_state.get("metric_name", self._run_target_metric_for_selection()) or ""
                ).strip().lower()
            q0 = np.asarray(live_state.get("fit_q0_trials", ()), dtype=float)
            if q0.ndim == 1 and q0.size > 0:
                diagnostics["fit_q0_trials"] = [float(v) for v in q0]
            for metric_name in METRICS:
                key = f"fit_{metric_name}_trials"
                arr = np.asarray(live_state.get(key, ()), dtype=float)
                if arr.ndim == 1 and arr.size > 0:
                    diagnostics[key] = [float(v) for v in arr]
        return display_metric

    def _trial_series_for_point(self, point: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
        q0_trials = np.asarray(point.get("fit_q0_trials", ()), dtype=float)
        selected_metric = self._trials_display_metric()
        metric_trials = self._metric_history_for_point(point, selected_metric)
        if metric_trials.size != q0_trials.size or not np.any(np.isfinite(metric_trials)):
            point_target = str(point.get("target_metric", "")).strip().lower()
            if selected_metric == point_target:
                fallback = np.asarray(point.get("fit_metric_trials", ()), dtype=float)
                if fallback.size == q0_trials.size:
                    metric_trials = fallback
        return q0_trials, metric_trials, selected_metric

    def _best_trial_index_from_metric(self, metric_trials: np.ndarray) -> int | None:
        finite = np.isfinite(metric_trials)
        if metric_trials.ndim != 1 or metric_trials.size == 0 or not np.any(finite):
            return None
        return int(np.nanargmin(metric_trials))

    def _current_trial_token(self, point_metric: str, q0_trials: np.ndarray) -> tuple[int, int, str, int]:
        a_index_var = getattr(self, "a_index_var", None)
        b_index_var = getattr(self, "b_index_var", None)
        return (
            int(a_index_var.get()) if a_index_var is not None else 0,
            int(b_index_var.get()) if b_index_var is not None else 0,
            str(point_metric),
            int(q0_trials.size),
        )

    def _trial_token_matches_context(self, token: Any, *, a_token: Any, b_token: Any, point_metric: str) -> bool:
        _ = point_metric
        if not isinstance(token, tuple) or len(token) < 2:
            return False
        try:
            token_a = float(token[0])
            token_b = float(token[1])
            context_a = float(a_token)
            context_b = float(b_token)
        except Exception:
            return False
        return bool(
            np.isclose(token_a, context_a, rtol=0.0, atol=1e-6)
            and np.isclose(token_b, context_b, rtol=0.0, atol=1e-6)
        )

    def _default_trial_index_for_point(
        self,
        point: dict[str, Any],
        q0_trials: np.ndarray,
        metric_trials: np.ndarray,
    ) -> int | None:
        q0_value = point.get("q0")
        if q0_value is not None:
            try:
                q0_numeric = float(q0_value)
            except Exception:
                q0_numeric = float("nan")
            if np.isfinite(q0_numeric) and q0_trials.size > 0:
                return int(np.argmin(np.abs(q0_trials - q0_numeric)))
        target_metric = str(point.get("target_metric", self.run_target_metric or "chi2"))
        search_metric_trials = self._metric_history_for_point(point, target_metric)
        if search_metric_trials.size == q0_trials.size and search_metric_trials.size > 0:
            return self._best_trial_index_from_metric(search_metric_trials)
        return self._best_trial_index_from_metric(metric_trials)

    def _selected_trial_index_for_point(
        self,
        point: dict[str, Any],
        q0_trials: np.ndarray,
        metric_trials: np.ndarray,
        point_metric: str,
        *,
        force_best: bool = False,
    ) -> int | None:
        trial_index_var = getattr(self, "trial_index_var", None)
        if q0_trials.ndim != 1 or q0_trials.size == 0 or metric_trials.size != q0_trials.size:
            self._selected_trial_token = None
            if trial_index_var is not None:
                trial_index_var.set(0)
            return None
        token = self._current_trial_token(point_metric, q0_trials)
        display_best_index = self._best_trial_index_from_metric(metric_trials)
        current_index = int(trial_index_var.get()) if trial_index_var is not None else -1
        token_matches_context = self._trial_token_matches_context(
            self._selected_trial_token,
            a_token=token[0],
            b_token=token[1],
            point_metric=point_metric,
        )
        if (
            force_best
            or not token_matches_context
            or current_index < 0
            or current_index >= q0_trials.size
        ):
            if force_best:
                current_index = 0 if display_best_index is None else int(display_best_index)
            else:
                default_index = self._default_trial_index_for_point(point, q0_trials, metric_trials)
                current_index = 0 if default_index is None else int(default_index)
            if trial_index_var is not None:
                trial_index_var.set(current_index)
        self._selected_trial_token = token
        return current_index

    def _refresh_trial_selector_controls(
        self,
        point: dict[str, Any] | None,
        q0_trials: np.ndarray,
        metric_trials: np.ndarray,
        point_metric: str,
        *,
        selected_index_override: int | None = None,
    ) -> None:
        if selected_index_override is None and point is not None:
            selected_index = self._selected_trial_index_for_point(point, q0_trials, metric_trials, point_metric)
        elif q0_trials.ndim == 1 and q0_trials.size > 0 and metric_trials.size == q0_trials.size:
            selected_index = int(np.clip(0 if selected_index_override is None else selected_index_override, 0, max(0, len(q0_trials) - 1)))
            if point is not None:
                self._selected_trial_token = self._current_trial_token(point_metric, q0_trials)
            self.trial_index_var.set(selected_index)
        else:
            selected_index = None
        enabled = selected_index is not None
        trial_label_var = getattr(self, "trial_label_var", None)
        if enabled:
            best_index = self._best_trial_index_from_metric(metric_trials)
            suffix = " [best]" if best_index is not None and int(selected_index) == int(best_index) else ""
            shift_suffix = ""
            if point is not None:
                trial_shift_label = format_observation_shift_label(
                    diagnostics=self._selected_diagnostics(),
                    trial_index=int(selected_index),
                    fit_shift_x_trials=point.get("fit_shift_x_trials"),
                    fit_shift_y_trials=point.get("fit_shift_y_trials"),
                    fit_find_shift_valid_trials=point.get("fit_find_shift_valid_trials"),
                )
                if trial_shift_label:
                    shift_suffix = f"  {trial_shift_label}"
            if trial_label_var is not None:
                trial_label_var.set(
                    f"trial #{int(selected_index) + 1}/{len(q0_trials)}  q0={float(q0_trials[int(selected_index)]):.6g}  "
                    f"{point_metric}={float(metric_trials[int(selected_index)]):.6g}{suffix}{shift_suffix}"
                )
        else:
            if trial_label_var is not None:
                trial_label_var.set("trial: n/a")
        if self.trial_slider is not None:
            self._updating_trial_slider = True
            try:
                if enabled:
                    self.trial_slider.configure(
                        from_=0.0,
                        to=float(max(0, len(q0_trials) - 1)),
                        state=tk.NORMAL,
                    )
                    self.trial_slider.set(float(selected_index))
                else:
                    self.trial_slider.configure(from_=0.0, to=0.0, state=tk.DISABLED)
                    self.trial_slider.set(0.0)
            finally:
                self._updating_trial_slider = False
        if self.trial_best_button is not None:
            self.trial_best_button.configure(state=(tk.NORMAL if enabled else tk.DISABLED))

    def _clear_trial_selector_controls(self) -> None:
        self._selected_trial_token = None
        self.trial_index_var.set(0)
        self.trial_label_var.set("trial: n/a")
        if self.trial_slider is not None:
            self._updating_trial_slider = True
            try:
                self.trial_slider.configure(from_=0.0, to=0.0, state=tk.DISABLED)
                self.trial_slider.set(0.0)
            finally:
                self._updating_trial_slider = False
        if self.trial_best_button is not None:
            self.trial_best_button.configure(state=tk.DISABLED)

    def _jump_to_best_trial(self) -> None:
        live_state = self._live_trial_state()
        use_live_trials = self._should_force_live_trials(live_state) or self._should_use_live_trials(live_state)
        if self._has_selected_point() and not use_live_trials:
            point = self._selected_point()
            q0_trials, metric_trials, point_metric = self._trial_series_for_point(point)
            best_index = self._best_trial_index_from_metric(metric_trials)
            if best_index is None:
                return
            self.trial_index_var.set(int(best_index))
            self._selected_trial_token = self._current_trial_token(point_metric, q0_trials)
            self._refresh_all()
            return

        if not use_live_trials:
            return
        q0_trials, metric_trials, point_metric, _snapshot = self._live_trial_series_from_state(live_state)
        if q0_trials.size == 0 or metric_trials.size != q0_trials.size:
            return
        best_index = self._best_trial_index_from_metric(metric_trials)
        if best_index is None:
            return
        a_token = int(live_state["a_index"]) if "a_index" in live_state else float(live_state.get("active_a", np.nan))
        b_token = int(live_state["b_index"]) if "b_index" in live_state else float(live_state.get("active_b", np.nan))
        self.trial_index_var.set(int(best_index))
        self._selected_trial_token = (a_token, b_token, point_metric, -1)
        self._refresh_all()

    def _on_trial_slider_changed(self, value: str) -> None:
        if self._updating_trial_slider:
            return
        live_state = self._live_trial_state()
        use_live_trials = self._should_force_live_trials(live_state) or self._should_use_live_trials(live_state)
        if self._has_selected_point() and not use_live_trials:
            point = self._selected_point()
            q0_trials, _metric_trials, point_metric = self._trial_series_for_point(point)
            if q0_trials.size == 0:
                return
            current = int(np.clip(round(float(value)), 0, max(0, len(q0_trials) - 1)))
            if current == int(self.trial_index_var.get()):
                return
            self.trial_index_var.set(current)
            self._selected_trial_token = self._current_trial_token(point_metric, q0_trials)
            self._refresh_all()
            return

        if not use_live_trials:
            return
        q0_trials, metric_trials, point_metric, _snapshot = self._live_trial_series_from_state(live_state)
        if q0_trials.size == 0 or metric_trials.size != q0_trials.size:
            return
        current = int(np.clip(round(float(value)), 0, max(0, len(q0_trials) - 1)))
        if current == int(self.trial_index_var.get()):
            return
        a_token = int(live_state["a_index"]) if "a_index" in live_state else float(live_state.get("active_a", np.nan))
        b_token = int(live_state["b_index"]) if "b_index" in live_state else float(live_state.get("active_b", np.nan))
        self.trial_index_var.set(current)
        self._selected_trial_token = (a_token, b_token, point_metric, -1)
        self._refresh_all()

    def _selected_solution_plot_context(self) -> dict[str, Any] | None:
        live_state = self._live_trial_state()
        if self._live_trials_context_available(live_state):
            live_context = self._live_selected_solution_plot_context()
            if live_context is not None:
                if self._has_saved_selected_point():
                    try:
                        point = self._selected_point()
                        live_diag = dict(live_context.get("diagnostics") or {})
                        if not _sequence_as_list(live_diag.get("fit_shift_x_trials")):
                            live_diag["fit_shift_x_trials"] = _sequence_as_list(point.get("fit_shift_x_trials"))
                            live_diag["fit_shift_y_trials"] = _sequence_as_list(point.get("fit_shift_y_trials"))
                            live_diag["fit_find_shift_valid_trials"] = _sequence_as_list(
                                point.get("fit_find_shift_valid_trials")
                            )
                        live_diag = self._diagnostics_with_point_ab(live_diag, point)
                        live_context = {**live_context, "diagnostics": live_diag}
                    except KeyError:
                        pass
                return live_context
        if not self._has_selected_point():
            return None
        point = self._selected_point()
        diagnostics = self._diagnostics_with_point_ab(self._selected_diagnostics(), point)
        search_diagnostics = dict((self.payload.get("selected_search") or {}).get("diagnostics") or {})
        if "use_smoothed_obs_max" in search_diagnostics:
            diagnostics["use_smoothed_obs_max"] = search_diagnostics.get("use_smoothed_obs_max")
        diagnostics["fit_trial_mask_stages"] = list(point.get("fit_trial_mask_stages") or ())
        diagnostics["fit_shift_x_trials"] = _sequence_as_list(point.get("fit_shift_x_trials"))
        diagnostics["fit_shift_y_trials"] = _sequence_as_list(point.get("fit_shift_y_trials"))
        diagnostics["fit_find_shift_valid_trials"] = _sequence_as_list(point.get("fit_find_shift_valid_trials"))
        self._populate_metric_trial_histories_in_diagnostics(diagnostics, point=point)
        q0_trials, metric_trials, point_metric = self._trial_series_for_point(point)
        selected_trial_index = self._selected_trial_index_for_point(point, q0_trials, metric_trials, point_metric)
        best_trial_index = self._best_trial_index_from_metric(metric_trials)
        diagnostics["best_q0_recovered"] = float(point.get("q0", np.nan))
        diagnostics["best_target_metric_value"] = float(diagnostics.get("target_metric_value", np.nan))
        raw_modeled = None
        modeled = None
        residual = None
        if selected_trial_index is not None:
            diagnostics["selected_trial_index"] = int(selected_trial_index)
            diagnostics["selected_trial_count"] = int(q0_trials.size)
            diagnostics["selected_trial_q0"] = float(q0_trials[int(selected_trial_index)])
            diagnostics["selected_trial_metric_name"] = str(point_metric)
            diagnostics["selected_trial_metric_value"] = float(metric_trials[int(selected_trial_index)])
            diagnostics["selected_trial_is_best"] = bool(
                best_trial_index is not None and int(selected_trial_index) == int(best_trial_index)
            )
            diagnostics["q0_recovered"] = float(q0_trials[int(selected_trial_index)])
            diagnostics["target_metric_value"] = float(metric_trials[int(selected_trial_index)])
            for metric_name in METRICS:
                metric_history = self._metric_history_for_point(point, metric_name)
                if metric_history.size == q0_trials.size:
                    diagnostics[metric_name] = float(metric_history[int(selected_trial_index)])
            diagnostics["selected_trial_maps_available"] = False
            loaded_maps = self._load_selected_trial_maps(
                a_value=float(point.get("a", np.nan)),
                b_value=float(point.get("b", np.nan)),
                selected_trial_index=int(selected_trial_index),
            )
            if loaded_maps is not None:
                raw_modeled = np.asarray(loaded_maps.get("raw_modeled_best"), dtype=float)
                modeled = np.asarray(loaded_maps.get("modeled_best"), dtype=float)
                residual = np.asarray(loaded_maps.get("residual"), dtype=float)
                diagnostics["selected_trial_maps_available"] = True
                diagnostics["selected_trial_index"] = int(loaded_maps.get("trial_index", selected_trial_index))
            else:
                raw_trial_maps = point.get("trial_raw_modeled_maps")
                modeled_trial_maps = point.get("trial_modeled_maps")
                residual_trial_maps = point.get("trial_residual_maps")
                if raw_trial_maps is not None and modeled_trial_maps is not None and residual_trial_maps is not None:
                    if int(selected_trial_index) < int(np.asarray(raw_trial_maps).shape[0]):
                        raw_modeled = np.asarray(raw_trial_maps[int(selected_trial_index)], dtype=float)
                        modeled = np.asarray(modeled_trial_maps[int(selected_trial_index)], dtype=float)
                        residual = np.asarray(residual_trial_maps[int(selected_trial_index)], dtype=float)
                        diagnostics["selected_trial_maps_available"] = True
        else:
            diagnostics["q0_recovered"] = float(point.get("q0", np.nan))
            loaded_maps = self._load_selected_trial_maps(
                a_value=float(point.get("a", np.nan)),
                b_value=float(point.get("b", np.nan)),
                selected_trial_index=None,
            )
            if loaded_maps is not None:
                raw_modeled = np.asarray(loaded_maps.get("raw_modeled_best"), dtype=float)
                modeled = np.asarray(loaded_maps.get("modeled_best"), dtype=float)
                residual = np.asarray(loaded_maps.get("residual"), dtype=float)
                diagnostics["selected_trial_maps_available"] = True
                if loaded_maps.get("trial_index") is not None:
                    diagnostics["selected_trial_index"] = int(loaded_maps["trial_index"])
            else:
                try:
                    raw_modeled = np.asarray(point["raw_modeled_best"], dtype=float)
                    modeled = np.asarray(point["modeled_best"], dtype=float)
                    residual = np.asarray(point["residual"], dtype=float)
                except Exception:
                    raw_modeled = None
                    modeled = None
                    residual = None
        if raw_modeled is None or modeled is None or residual is None:
            return None
        slice_descriptor = dict(self.payload.get("selected_slice") or {})
        slice_label = self._slice_label(slice_descriptor) if slice_descriptor else "single slice"
        frequency_ghz = None
        for key in ("active_frequency_ghz", "frequency_ghz", "mw_frequency_ghz"):
            value = diagnostics.get(key)
            if value is None:
                continue
            try:
                numeric = float(value)
            except Exception:
                continue
            if np.isfinite(numeric):
                frequency_ghz = numeric
                break
        display_observed = self._display_observation_for_trial(
            point=point,
            trial_index=int(selected_trial_index) if selected_trial_index is not None else None,
            diagnostics=diagnostics,
        )
        if modeled is not None:
            residual = np.asarray(modeled, dtype=float) - np.asarray(display_observed, dtype=float)
        return {
            "model_path": Path(str(diagnostics.get("model_path", ""))),
            "observed_noisy": np.asarray(display_observed, dtype=float),
            "raw_modeled_best": np.asarray(raw_modeled, dtype=float),
            "modeled_best": np.asarray(modeled, dtype=float),
            "residual": np.asarray(residual, dtype=float),
            "wcs_header": self.payload.get("wcs_header"),
            "frequency_ghz": frequency_ghz,
            "diagnostics": diagnostics,
            "psf_kernel": self.payload.get("psf_kernel"),
            "slice_label": slice_label,
            "blos_reference": self.payload.get("blos_reference"),
            **self._trials_axis_view_from_parent(),
            "wcs_header_transform": lambda hdr: with_observer_metadata(hdr, self.payload["wcs_header"], diagnostics),
        }

    def _build_live_renderer_for_point(self, *, diagnostics: dict[str, Any], a_value: float, b_value: float) -> Any | None:
        model_path = Path(str(diagnostics.get("model_path", "")).strip()).expanduser()
        if not model_path.exists():
            return None
        ebtel_text = str(diagnostics.get("ebtel_path", "")).strip()
        ebtel_path = ebtel_text or None
        tbase = _optional_float(diagnostics.get("tbase"))
        nbase = _optional_float(diagnostics.get("nbase"))
        if tbase is None or nbase is None:
            return None

        geometry = SimpleNamespace(
            xc=float(_optional_float(diagnostics.get("map_xc_arcsec")) or 0.0),
            yc=float(_optional_float(diagnostics.get("map_yc_arcsec")) or 0.0),
            dx=float(_optional_float(diagnostics.get("map_dx_arcsec")) or 2.0),
            dy=float(_optional_float(diagnostics.get("map_dy_arcsec")) or 2.0),
            nx=int(_optional_float(diagnostics.get("map_nx")) or 0),
            ny=int(_optional_float(diagnostics.get("map_ny")) or 0),
        )
        if geometry.nx <= 0 or geometry.ny <= 0:
            return None

        observer = SimpleNamespace(
            dsun_cm=_optional_float(diagnostics.get("observer_dsun_cm")),
            lonc_deg=_optional_float(diagnostics.get("observer_lonc_deg")),
            b0sun_deg=_optional_float(diagnostics.get("observer_b0sun_deg")),
        )
        observer_name = str(diagnostics.get("observer_name", "")).strip() or None
        domain = str(diagnostics.get("spectral_domain", "")).strip().lower()

        if domain == "mw":
            frequency_ghz = _first_finite_float(
                diagnostics.get("active_frequency_ghz"),
                diagnostics.get("frequency_ghz"),
                diagnostics.get("mw_frequency_ghz"),
            )
            if frequency_ghz is None:
                return None
            render_freqs: list[float] = [float(frequency_ghz)]
            for item in list(diagnostics.get("render_frequencies_ghz", []) or []):
                numeric = _optional_float(item)
                if numeric is None:
                    continue
                if not any(np.isclose(float(numeric), float(existing), rtol=0.0, atol=1e-12) for existing in render_freqs):
                    render_freqs.append(float(numeric))
            renderer = GXRenderMWAdapter(
                model_path=str(model_path),
                ebtel_path=ebtel_path,
                frequency_ghz=float(frequency_ghz),
                render_frequencies_ghz=tuple(render_freqs),
                tbase=float(tbase),
                nbase=float(nbase),
                a=float(a_value),
                b=float(b_value),
                geometry=geometry,
                observer=observer,
                observer_name=observer_name,
                pixel_scale_arcsec=float(abs(geometry.dx)),
            )
        elif domain in {"euv", "uv"}:
            channel = str(diagnostics.get("euv_channel", "")).strip()
            if not channel:
                return None
            render_channels: list[str] = [channel]
            for item in list(diagnostics.get("render_channels", []) or []):
                text = str(item).strip()
                if text and text not in render_channels:
                    render_channels.append(text)
            renderer = GXRenderEUVAdapter(
                model_path=str(model_path),
                channel=str(channel),
                render_channels=tuple(render_channels),
                instrument=str(diagnostics.get("euv_instrument") or diagnostics.get("observation_instrument") or "AIA"),
                response_sav=(str(diagnostics.get("euv_response_sav", "")).strip() or None),
                ebtel_path=ebtel_path,
                tbase=float(tbase),
                nbase=float(nbase),
                a=float(a_value),
                b=float(b_value),
                geometry=geometry,
                observer=observer,
                observer_name=observer_name,
                pixel_scale_arcsec=float(abs(geometry.dx)),
            )
        else:
            return None

        psf_kernel = self.payload.get("psf_kernel")
        if psf_kernel is not None:
            kernel = np.asarray(psf_kernel, dtype=float)
            if kernel.ndim == 2 and kernel.size:
                return KernelConvolvedRenderer(renderer, kernel)
        return renderer

    def _render_live_selected_trial_maps(
        self,
        *,
        diagnostics: dict[str, Any],
        a_value: float,
        b_value: float,
        q0_value: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        observed = np.asarray(self.payload.get("observed"), dtype=float)
        if observed.ndim != 2 or not np.isfinite(q0_value):
            return None
        cache_key = (
            float(a_value),
            float(b_value),
            float(q0_value),
            str(diagnostics.get("spectral_domain", "")),
            str(diagnostics.get("spectral_label", "")),
            str(diagnostics.get("model_path", "")),
            str(diagnostics.get("ebtel_path", "")),
        )
        if self._live_trial_render_cache_key == cache_key and self._live_trial_render_cache_value is not None:
            return self._live_trial_render_cache_value

        renderer = self._build_live_renderer_for_point(
            diagnostics=diagnostics,
            a_value=float(a_value),
            b_value=float(b_value),
        )
        if renderer is None:
            return None

        if hasattr(renderer, "render_pair"):
            raw_modeled, modeled = renderer.render_pair(float(q0_value))
        else:
            modeled = renderer.render(float(q0_value))
            raw_modeled = modeled
        raw_arr = np.asarray(raw_modeled, dtype=float)
        modeled_arr = np.asarray(modeled, dtype=float)
        if raw_arr.shape != observed.shape or modeled_arr.shape != observed.shape:
            return None
        residual_arr = np.asarray(modeled_arr - observed, dtype=float)
        payload = (raw_arr, modeled_arr, residual_arr)
        self._live_trial_render_cache_key = cache_key
        self._live_trial_render_cache_value = payload
        return payload

    def _load_live_active_point_snapshot(self, live_state: dict[str, Any]) -> dict[str, Any] | None:
        return None

    def _load_selected_trial_maps(
        self,
        *,
        a_value: float,
        b_value: float,
        selected_trial_index: int | None,
    ) -> dict[str, Any] | None:
        artifact_h5 = getattr(self, "artifact_h5", None)
        if artifact_h5 is None:
            return None
        slice_key = str(self.payload.get("selected_slice_key", "")).strip() or None
        search_id = str(self.payload.get("selected_search_id", "")).strip() or None
        if slice_key is None:
            return None
        try:
            artifact_mtime_ns = int(Path(artifact_h5).stat().st_mtime_ns)
        except Exception:
            artifact_mtime_ns = -1
        cache_key = (
            str(Path(artifact_h5)),
            str(slice_key),
            str(search_id or ""),
            float(a_value),
            float(b_value),
            -1 if selected_trial_index is None else int(selected_trial_index),
            int(artifact_mtime_ns),
        )
        if cache_key == getattr(self, "_selected_trial_map_cache_key", None):
            return getattr(self, "_selected_trial_map_cache_value", None)
        try:
            payload = load_selected_trial_plot_payload(
                Path(artifact_h5),
                a=float(a_value),
                b=float(b_value),
                trial_index=selected_trial_index,
                slice_key=slice_key,
                search_id=search_id,
            )
        except Exception:
            payload = None
        self._selected_trial_map_cache_key = cache_key
        self._selected_trial_map_cache_value = payload
        return payload

    def _live_trial_arrays_from_state(
        self,
        live_state: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        selected_metric = self._trials_display_metric()
        q0_trials = np.asarray([], dtype=float)
        for q0_key in ("fit_q0_trials", "q0_trials"):
            candidate = np.asarray(live_state.get(q0_key, ()), dtype=float)
            if candidate.ndim == 1 and candidate.size > 0:
                q0_trials = candidate
                break
        if q0_trials.size == 0:
            return q0_trials, np.asarray([], dtype=float)
        metric_trials = np.asarray(live_state.get(f"fit_{selected_metric}_trials", ()), dtype=float)
        if metric_trials.size != q0_trials.size:
            target_metric = str(
                live_state.get("metric_name", self._run_target_metric_for_selection()) or ""
            ).strip().lower()
            if selected_metric == target_metric:
                for metric_key in ("fit_metric_trials", "metric_trials"):
                    candidate = np.asarray(live_state.get(metric_key, ()), dtype=float)
                    if candidate.ndim == 1 and candidate.size == q0_trials.size:
                        metric_trials = candidate
                        break
            else:
                metric_trials = np.asarray([], dtype=float)
        if metric_trials.ndim == 1 and metric_trials.size == q0_trials.size:
            return q0_trials, metric_trials
        return q0_trials, np.asarray([], dtype=float)

    def _live_trial_series_from_state(
        self,
        live_state: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, str, dict[str, Any] | None]:
        point_metric = self._trials_display_metric()
        q0_trials, metric_trials = self._live_trial_arrays_from_state(live_state)
        if q0_trials.size > 0 and metric_trials.size == q0_trials.size:
            return q0_trials, metric_trials, point_metric, None
        snapshot = self._load_live_active_point_snapshot(live_state)
        if snapshot is not None:
            try:
                snapshot_a = float(snapshot.get("a", np.nan))
                snapshot_b = float(snapshot.get("b", np.nan))
            except Exception:
                snapshot_a = float("nan")
                snapshot_b = float("nan")
            try:
                active_a = float(live_state.get("active_a", live_state.get("a", np.nan)))
                active_b = float(live_state.get("active_b", live_state.get("b", np.nan)))
            except Exception:
                active_a = float("nan")
                active_b = float("nan")
            coords_match = (
                np.isfinite(snapshot_a)
                and np.isfinite(snapshot_b)
                and np.isfinite(active_a)
                and np.isfinite(active_b)
                and np.isclose(snapshot_a, active_a, rtol=0.0, atol=1e-6)
                and np.isclose(snapshot_b, active_b, rtol=0.0, atol=1e-6)
            )
            if coords_match:
                q0_trials = np.asarray(snapshot.get("fit_q0_trials", ()), dtype=float)
                metric_trials = np.asarray(snapshot.get(f"fit_{point_metric}_trials", ()), dtype=float)
                if metric_trials.size != q0_trials.size or metric_trials.size == 0:
                    target_metric = str(
                        snapshot.get("target_metric", live_state.get("metric_name", self._run_target_metric_for_selection()))
                        or ""
                    ).strip().lower()
                    if point_metric == target_metric:
                        metric_trials = np.asarray(snapshot.get("fit_metric_trials", ()), dtype=float)
                    else:
                        metric_trials = np.asarray([], dtype=float)
                    if metric_trials.size != q0_trials.size:
                        metric_trials = np.asarray([], dtype=float)
                if q0_trials.ndim == 1 and q0_trials.size > 0 and metric_trials.size == q0_trials.size:
                    return q0_trials, metric_trials, point_metric, snapshot
        if q0_trials.size == 0:
            for q0_key in ("fit_q0_trials", "q0_trials"):
                candidate = np.asarray(live_state.get(q0_key, ()), dtype=float)
                if candidate.ndim == 1 and candidate.size > 0:
                    q0_trials = candidate
                    break
        return q0_trials, np.asarray([], dtype=float), point_metric, snapshot

    def _live_selected_solution_plot_context(self) -> dict[str, Any] | None:
        live_state = self._live_trial_state()
        if not self._live_trials_context_available(live_state):
            return None
        q0_trials, metric_trials, point_metric, snapshot = self._live_trial_series_from_state(live_state)
        if q0_trials.ndim != 1 or q0_trials.size == 0 or metric_trials.size != q0_trials.size:
            return None

        current_index = int(np.clip(int(self.trial_index_var.get()), 0, max(0, int(q0_trials.size) - 1)))
        selected_q0 = float(q0_trials[current_index])
        selected_metric = float(metric_trials[current_index])
        best_index = self._best_trial_index_from_metric(metric_trials)
        best_q0 = float(q0_trials[int(best_index)]) if best_index is not None else selected_q0

        active_coords = self._live_active_ab(live_state)
        if active_coords is None:
            return None
        active_a, active_b = active_coords
        diagnostics = self._diagnostics_with_search_shift(dict(self.payload.get("diagnostics") or {}))
        diagnostics.update(
            {
                "a": float(active_a),
                "b": float(active_b),
                "target_metric_value": float(selected_metric),
                "fit_shift_x_trials": _sequence_as_list(live_state.get("fit_shift_x_trials")),
                "fit_shift_y_trials": _sequence_as_list(live_state.get("fit_shift_y_trials")),
                "fit_find_shift_valid_trials": _sequence_as_list(live_state.get("fit_find_shift_valid_trials")),
                "selected_trial_index": int(current_index),
                "selected_trial_count": int(q0_trials.size),
                "selected_trial_q0": float(selected_q0),
                "selected_trial_metric_name": point_metric,
                "selected_trial_metric_value": float(selected_metric),
                "selected_trial_is_best": bool(best_index is not None and int(current_index) == int(best_index)),
                "q0_recovered": float(selected_q0),
                "best_q0_recovered": float(best_q0),
            }
        )
        saved_point = self._selected_point() if self._has_saved_selected_point() else None
        display_metric = self._populate_metric_trial_histories_in_diagnostics(
            diagnostics,
            point=saved_point,
            live_state=live_state,
        )
        point_metric = display_metric
        diagnostics["selected_trial_metric_name"] = str(point_metric)
        diagnostics["selected_trial_maps_available"] = False
        loaded_maps = self._load_selected_trial_maps(
            a_value=float(active_a),
            b_value=float(active_b),
            selected_trial_index=int(current_index),
        )
        raw_modeled = None
        modeled = None
        residual = None
        if loaded_maps is not None:
            raw_modeled = np.asarray(loaded_maps.get("raw_modeled_best"), dtype=float)
            modeled = np.asarray(loaded_maps.get("modeled_best"), dtype=float)
            residual = np.asarray(loaded_maps.get("residual"), dtype=float)
            diagnostics["selected_trial_maps_available"] = True
            diagnostics["selected_trial_index"] = int(loaded_maps.get("trial_index", current_index))
        elif snapshot is not None:
            raw_trials = snapshot.get("trial_raw_modeled_maps")
            if raw_trials is not None:
                raw_trials_arr = np.asarray(raw_trials, dtype=float)
                if raw_trials_arr.ndim == 3 and raw_trials_arr.shape[0] == q0_trials.size and current_index < raw_trials_arr.shape[0]:
                    raw_modeled = np.asarray(raw_trials_arr[current_index], dtype=float)
                    modeled = _convolve_raw_map(raw_modeled, self.payload.get("psf_kernel"))
                    display_observed = self._display_observation_for_trial(
                        trial_index=int(current_index),
                        diagnostics=diagnostics,
                    )
                    residual = np.asarray(modeled - np.asarray(display_observed, dtype=float), dtype=float)
                    diagnostics["selected_trial_maps_available"] = True
        if raw_modeled is None or modeled is None or residual is None:
            rendered = self._render_live_selected_trial_maps(
                diagnostics=diagnostics,
                a_value=float(active_a),
                b_value=float(active_b),
                q0_value=float(selected_q0),
            )
            if rendered is not None:
                raw_modeled, modeled, residual = rendered
                diagnostics["selected_trial_maps_available"] = True
        if raw_modeled is None or modeled is None or residual is None:
            return None
        for metric_name in METRICS:
            diagnostics[metric_name] = float(selected_metric) if metric_name == point_metric else float("nan")

        slice_descriptor = dict(self.payload.get("selected_slice") or {})
        slice_label = self._slice_label(slice_descriptor) if slice_descriptor else "single slice"
        frequency_ghz = None
        for key in ("active_frequency_ghz", "frequency_ghz", "mw_frequency_ghz"):
            value = diagnostics.get(key)
            if value is None:
                continue
            try:
                numeric = float(value)
            except Exception:
                continue
            if np.isfinite(numeric):
                frequency_ghz = numeric
                break
        display_observed = self._display_observation_for_trial(
            trial_index=int(current_index),
            diagnostics=diagnostics,
        )
        if modeled is not None:
            residual = np.asarray(modeled, dtype=float) - np.asarray(display_observed, dtype=float)
        return {
            "model_path": Path(str(diagnostics.get("model_path", ""))),
            "observed_noisy": np.asarray(display_observed, dtype=float),
            "raw_modeled_best": np.asarray(raw_modeled, dtype=float),
            "modeled_best": np.asarray(modeled, dtype=float),
            "residual": np.asarray(residual, dtype=float),
            "wcs_header": self.payload["wcs_header"],
            "frequency_ghz": frequency_ghz,
            "diagnostics": diagnostics,
            "psf_kernel": self.payload.get("psf_kernel"),
            "slice_label": slice_label,
            "blos_reference": self.payload.get("blos_reference"),
            **self._trials_axis_view_from_parent(),
            "wcs_header_transform": lambda hdr: with_observer_metadata(hdr, self.payload["wcs_header"], diagnostics),
        }

    def _on_metric_changed(self) -> None:
        if self._metric_selection_locked():
            self.metric_var.set(self._run_target_metric_for_selection())
            return
        self._capture_current_slice_view_state(self._last_rendered_metric)
        self._restore_trials_controls_for_metric(str(self.metric_var.get()))
        self._refresh_all()

    def _on_a_changed(self) -> None:
        if self._grid_point_selection_locked():
            return
        current = int(self.a_menu.current())
        if current >= 0:
            self.a_index_var.set(current)
        self._sync_free_selection_from_indices()
        self._selected_trial_token = None
        self._refresh_all()

    def _on_slice_changed(self) -> None:
        if self._slice_and_search_selection_locked():
            return
        if self.slice_menu is None:
            return
        current = int(self.slice_menu.current())
        if current < 0 or current >= len(self.available_slices):
            return
        selected_key = str(self.available_slices[current].get("key", "")).strip()
        if not selected_key or selected_key == self._selected_slice_key():
            return
        self._capture_current_slice_view_state(self._last_rendered_metric)
        self._capture_free_grid_selection_coords()
        self._reset_trials_controls_only()
        self._navigation_mode_user_chosen = True
        if self._use_shared_grid_axes():
            self._shared_heatmap_extents = None
            self._refresh_shared_heatmap_extents()
        self.slice_key_var.set(selected_key)
        self.search_id_var.set("")
        self._schedule_payload_reload(status_text="Loading selected slice...")

    def _on_search_changed(self) -> None:
        if self._slice_and_search_selection_locked():
            return
        if self.search_menu is None:
            return
        current = int(self.search_menu.current())
        if current < 0 or current >= len(self.available_searches):
            return
        selected_id = str(self.available_searches[current].get("search_id", "")).strip()
        if not selected_id or selected_id == self._selected_search_id():
            return
        self._capture_current_slice_view_state(self._last_rendered_metric)
        self._capture_free_grid_selection_coords()
        self.search_id_var.set(selected_id)
        self._selected_trial_token = None
        self._schedule_payload_reload(status_text="Loading selected search...")

    def _on_b_changed(self) -> None:
        if self._grid_point_selection_locked():
            return
        current = int(self.b_menu.current())
        if current >= 0:
            self.b_index_var.set(current)
        self._sync_free_selection_from_indices()
        self._selected_trial_token = None
        self._refresh_all()

    def _on_trials_canvas_click(self, event: Any) -> None:
        if event.inaxes is not self.ax_trials or event.xdata is None:
            return
        live_state = self._live_trial_state()
        use_live_trials = self._should_force_live_trials(live_state) or self._should_use_live_trials(live_state)
        if self._has_selected_point() and not use_live_trials:
            point = self._selected_point()
            q0_trials, metric_trials, point_metric = self._trial_series_for_point(point)
            if q0_trials.size == 0 or metric_trials.size != q0_trials.size:
                return
            token = self._current_trial_token(point_metric, q0_trials)
        else:
            if not use_live_trials:
                return
            q0_trials, metric_trials, point_metric, _snapshot = self._live_trial_series_from_state(live_state)
            if q0_trials.size == 0 or metric_trials.size != q0_trials.size:
                return
            a_token = int(live_state["a_index"]) if "a_index" in live_state else float(live_state.get("active_a", np.nan))
            b_token = int(live_state["b_index"]) if "b_index" in live_state else float(live_state.get("active_b", np.nan))
            token = (a_token, b_token, point_metric, -1)

        if event.ydata is None:
            selected_index = int(np.argmin(np.abs(q0_trials - float(event.xdata))))
        else:
            x_span = max(float(np.nanmax(q0_trials) - np.nanmin(q0_trials)), 1e-12)
            y_span = max(float(np.nanmax(metric_trials) - np.nanmin(metric_trials)), 1e-12)
            distances = ((q0_trials - float(event.xdata)) / x_span) ** 2 + ((metric_trials - float(event.ydata)) / y_span) ** 2
            selected_index = int(np.nanargmin(distances))
        self.trial_index_var.set(selected_index)
        self._selected_trial_token = token
        self._refresh_all()

    def _on_canvas_click(self, event: Any) -> None:
        if self._navigation_mode() != "free":
            return
        if self.a_values.size == 0 or self.b_values.size == 0:
            return
        if event.inaxes is not self.ax_heatmap or event.xdata is None or event.ydata is None:
            return
        click_a = float(event.xdata)
        click_b = float(event.ydata)
        active_indices = self._active_point_indices_from_heatmap_click(click_a, click_b)
        if active_indices is not None:
            self._set_free_grid_selection(click_a, click_b)
            self._selected_trial_token = None
            self._refresh_selector_values()
            self._refresh_action_states()
            self._refresh_all()
            return
        record = find_record_for_point(self.display_model, click_a, click_b)
        if record is not None:
            self._set_free_grid_selection(float(record["a"]), float(record["b"]))
        else:
            resolved_indices = grid_indices_for_coordinates(self.payload, click_a, click_b)
            if resolved_indices is not None:
                a_values = np.asarray(self.a_values, dtype=float)
                b_values = np.asarray(self.b_values, dtype=float)
                self._set_free_grid_selection(
                    float(a_values[int(resolved_indices[0])]),
                    float(b_values[int(resolved_indices[1])]),
                )
            else:
                self._set_free_grid_selection(click_a, click_b)
        self._selected_trial_token = None
        self._refresh_selector_values()
        self._refresh_action_states()
        self._refresh_all()

    def _active_point_indices_from_heatmap_click(self, xdata: float, ydata: float) -> tuple[int, int] | None:
        live_state = self._live_trial_state()
        if live_state is None or not self._live_slice_matches_selected(live_state):
            return None
        try:
            active_a = float(live_state.get("active_a", live_state.get("a", np.nan)))
            active_b = float(live_state.get("active_b", live_state.get("b", np.nan)))
        except Exception:
            return None
        if not (np.isfinite(active_a) and np.isfinite(active_b)):
            return None
        resolved = self._point_indices_for_coordinates(active_a, active_b)
        if resolved is None:
            return None

        def _click_tolerance(values: np.ndarray, lower: float, upper: float) -> float:
            finite = np.asarray(values, dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size >= 2:
                diffs = np.diff(np.unique(np.sort(finite)))
                diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
                if diffs.size:
                    return float(np.min(diffs)) * 0.5
            span = max(float(upper - lower), 1e-6)
            return 0.05 * span

        x_tol = _click_tolerance(np.asarray(self.a_values, dtype=float), float(self.display_model.get("a_min", active_a)), float(self.display_model.get("a_max", active_a)))
        y_tol = _click_tolerance(np.asarray(self.b_values, dtype=float), float(self.display_model.get("b_min", active_b)), float(self.display_model.get("b_max", active_b)))
        if abs(float(xdata) - active_a) > x_tol or abs(float(ydata) - active_b) > y_tol:
            return None
        return resolved

    def _refresh_all(self, *, update_selected_solution: bool = True) -> None:
        if not self.payload:
            self._clear_trial_selector_controls()
            self._refresh_action_states()
            self._refresh_scan_state_display()
            self.ax_heatmap.clear()
            self._reset_heatmap_colorbar()
            self.ax_trials.clear()
            self.ax_heatmap.set_axis_off()
            self.ax_trials.set_axis_off()
            self._refresh_info_text()
            self.heatmap_canvas.draw_idle()
            self.trials_canvas.draw_idle()
            if self.selected_solution_window is not None:
                self.selected_solution_window.status_var.set("No artifact loaded.")
            return
        live_state = self._live_trial_state()
        if (
            not self._ensure_selected_point_exists(metric_name=str(self.metric_var.get()))
            and live_state is None
            and not self._payload_has_computed_grid()
        ):
            self._clear_trial_selector_controls()
            self._refresh_scan_state_display()
            self.ax_heatmap.clear()
            self._reset_heatmap_colorbar()
            self.ax_trials.clear()
            self.ax_heatmap.text(
                0.5,
                0.5,
                "Waiting for first completed (a,b) point.\nUse Refresh Artifact as the scan advances.",
                transform=self.ax_heatmap.transAxes,
                ha="center",
                va="center",
            )
            self.ax_heatmap.set_axis_off()
            self.ax_trials.text(
                0.5,
                0.5,
                "No completed search yet for the selected (a,b) pair.",
                transform=self.ax_trials.transAxes,
                ha="center",
                va="center",
            )
            self.ax_trials.set_axis_off()
            self.status_var.set(
                "Displayed grid metric: waiting\n"
                "Run target metric: "
                f"{self.run_target_metric}\n"
                "No completed search yet for the (a,b) pair."
            )
            self.summary_var.set(
                "No completed point is available yet in this artifact.\n"
                "Keep the viewer open and press Refresh Artifact while the scan is running."
            )
            self._refresh_action_states()
            self._refresh_info_text()
            self.heatmap_canvas.draw_idle()
            self.trials_canvas.draw_idle()
            if self.selected_solution_window is not None:
                self.selected_solution_window.status_var.set("No completed point available yet.")
            return
        if live_state is not None and not self._has_selected_point():
            self._refresh_action_states()
            self._refresh_scan_state_display()
            self._draw_heatmap()
            self._draw_trials()
            displayed_metric = (
                str(self.metric_var.get())
                if list(self.display_model.get("records", []))
                else "waiting"
            )
            self.status_var.set(
                f"Displayed grid metric: {displayed_metric}\n"
                f"Run target metric: {self.run_target_metric}\n"
                "Active point has not been saved to the artifact yet."
            )
            self.summary_var.set(self._active_point_waiting_summary(live_state))
            self._refresh_info_text()
            self.heatmap_canvas.draw_idle()
            self.trials_canvas.draw_idle()
            if self.selected_solution_window is not None:
                if self._live_trials_streaming(live_state):
                    self.selected_solution_window.status_var.set(
                        "Live trial maps are available; open Display Selected Solution to inspect them."
                    )
                else:
                    self.selected_solution_window.status_var.set("Waiting for active point to be saved.")
            if update_selected_solution and self.selected_solution_window is not None:
                self._schedule_selected_solution_update()
            return
        self._refresh_action_states()
        self._refresh_scan_state_display()
        self._draw_heatmap()
        self._draw_trials()
        self._refresh_summary()
        self._refresh_info_text()
        self.heatmap_canvas.draw_idle()
        self.trials_canvas.draw_idle()
        self._schedule_plot_canvas_resize()
        if update_selected_solution and self.selected_solution_window is not None:
            self._schedule_selected_solution_update()

    def _heatmap_display_model(self) -> dict[str, Any]:
        model = dict(getattr(self, "display_model", {}) or {})
        active = self._active_point_scoped_to_selection()
        if active is not None and not self._active_point_is_saved_in_artifact():
            model = extend_patch_grid_model_with_pending_point(
                model,
                a_value=float(active[0]),
                b_value=float(active[1]),
                diagnostics=dict(self.payload.get("diagnostics") or {}),
            )
        return model

    def _draw_heatmap(self) -> None:
        metric_name = self._heatmap_display_metric()
        display_model = self._heatmap_display_model()
        records = list(display_model.get("records", []))
        record_lookup = {(int(record["a_index"]), int(record["b_index"])): record for record in records}
        self._reset_heatmap_colorbar()
        self.ax_heatmap.clear()
        computed_records = [
            record
            for record in records
            if str(record.get("status", "computed")).strip().lower() not in {"pending", "missing"}
            and not bool(record.get("live_pending"))
        ]
        pending_records = [record for record in records if record not in computed_records]
        patches = [_heatmap_grid_rectangle(record) for record in computed_records]
        values = np.asarray(
            [float(record["metrics"].get(metric_name, np.nan)) for record in computed_records],
            dtype=float,
        )
        metric_collection = None
        if patches:
            metric_collection = PatchCollection(patches, cmap="viridis", edgecolor="none", linewidth=0.0)
            metric_collection.set_array(values)
            finite = np.isfinite(values)
            if np.any(finite):
                metric_collection.set_clim(float(np.nanmin(values[finite])), float(np.nanmax(values[finite])))
            self.ax_heatmap.add_collection(metric_collection)
        if pending_records:
            pending_patches = [_heatmap_grid_rectangle(record) for record in pending_records]
            pending_collection = PatchCollection(
                pending_patches,
                facecolors="none",
                edgecolors="#666666",
                linewidths=1.8,
                linestyles="dashed",
            )
            self.ax_heatmap.add_collection(pending_collection)
        apply_heatmap_axis_style(self.ax_heatmap, metric_name=metric_name)
        heatmap_a_min, heatmap_a_max, heatmap_b_min, heatmap_b_max = self._heatmap_plot_limits(display_model)
        apply_heatmap_data_limits(
            self.ax_heatmap,
            a_min=heatmap_a_min,
            a_max=heatmap_a_max,
            b_min=heatmap_b_min,
            b_max=heatmap_b_max,
        )

        locked_active_view = (
            self._navigation_mode() == "active"
            and getattr(self, "_refresh_signal_active_point", None) is not None
        )
        best_markers = {"chi2": ("#d62728", "o"), "rho2": ("#1f77b4", "s"), "eta2": ("#2ca02c", "^")}
        for name, (color, marker) in best_markers.items():
            if locked_active_view:
                continue
            try:
                a_index, b_index = best_grid_index(self.payload, name)
            except ValueError:
                # No finite values for this metric yet; skip marker until refresh after new points land.
                continue
            record = record_lookup.get((a_index, b_index))
            if record is not None:
                b_value = float(record["b_center"])
                a_value = float(record["a_center"])
            else:
                if not (0 <= int(a_index) < int(self.a_values.size) and 0 <= int(b_index) < int(self.b_values.size)):
                    continue
                b_value = float(self.b_values[b_index])
                a_value = float(self.a_values[a_index])
            self.ax_heatmap.scatter(
                [a_value],
                [b_value],
                s=120,
                marker=marker,
                facecolor="none",
                edgecolor="white",
                linewidth=3.2,
                zorder=5,
            )
            self.ax_heatmap.scatter(
                [a_value],
                [b_value],
                s=80,
                marker=marker,
                facecolor="none",
                edgecolor=color,
                linewidth=1.8,
                zorder=6,
            )

        current_b_value: float | None = None
        current_a_value: float | None = None
        if self._navigation_mode() == "free":
            free_ab = self._free_grid_selection_ab()
            if free_ab is not None:
                current_a_value, current_b_value = free_ab
        if current_a_value is None or current_b_value is None:
            current_a = int(self.a_index_var.get())
            current_b = int(self.b_index_var.get())
            current_record = record_lookup.get((current_a, current_b))
            if current_record is not None:
                current_b_value = float(current_record["b_center"])
                current_a_value = float(current_record["a_center"])
            elif 0 <= int(current_a) < int(self.a_values.size) and 0 <= int(current_b) < int(self.b_values.size):
                current_b_value = float(self.b_values[current_b])
                current_a_value = float(self.a_values[current_a])
        if current_b_value is not None and current_a_value is not None and not locked_active_view:
            self.ax_heatmap.scatter(
                [current_a_value],
                [current_b_value],
                s=170,
                marker="x",
                color="black",
                linewidth=3.6,
                zorder=7,
            )
            self.ax_heatmap.scatter(
                [current_a_value],
                [current_b_value],
                s=120,
                marker="x",
                color="white",
                linewidth=2.2,
                zorder=8,
            )
        for record in assigned_only_records(self.payload):
            try:
                assigned_a = float(record["a"])
                assigned_b = float(record["b"])
            except Exception:
                continue
            assigned_record = None
            for candidate in records:
                if np.isclose(float(candidate["a"]), assigned_a, rtol=0.0, atol=1e-9) and np.isclose(
                    float(candidate["b"]), assigned_b, rtol=0.0, atol=1e-9
                ):
                    assigned_record = candidate
                    break
            if assigned_record is not None:
                marker_b = float(assigned_record["b_center"])
                marker_a = float(assigned_record["a_center"])
            else:
                marker_b = assigned_b
                marker_a = assigned_a
            self.ax_heatmap.scatter(
                [marker_a],
                [marker_b],
                s=180,
                marker="*",
                facecolor="#9775fa",
                edgecolor="black",
                linewidth=0.9,
                zorder=8,
            )
        if self._navigation_mode() == "best" and len(self._best_tied_records) > 1:
            for tied_record in self._best_tied_records:
                try:
                    tied_a = float(tied_record["a"])
                    tied_b = float(tied_record["b"])
                except Exception:
                    continue
                tied_display = None
                for candidate in records:
                    if np.isclose(float(candidate["a"]), tied_a, rtol=0.0, atol=1e-9) and np.isclose(
                        float(candidate["b"]), tied_b, rtol=0.0, atol=1e-9
                    ):
                        tied_display = candidate
                        break
                if tied_display is not None:
                    tied_b_value = float(tied_display["b_center"])
                    tied_a_value = float(tied_display["a_center"])
                else:
                    tied_b_value = tied_b
                    tied_a_value = tied_a
                self.ax_heatmap.scatter(
                    [tied_a_value],
                    [tied_b_value],
                    s=140,
                    marker="o",
                    facecolor="none",
                    edgecolor="#495057",
                    linewidth=2.0,
                    zorder=8,
                )
        active_point = (
            self._refresh_signal_active_point
            if self._navigation_mode() in {"active", "best"} and not self._search_run_terminal()
            else None
        )
        if active_point is not None:
            active_a, active_b = active_point
            if np.isfinite(active_a) and np.isfinite(active_b):
                live_state = self._live_trial_state()
                streaming = live_state is not None and self._live_trials_streaming(live_state)
                resolved = self._point_indices_for_coordinates(float(active_a), float(active_b))
                has_committed_trials = False
                if resolved is not None and getattr(self, "payload", None):
                    point_payload = dict(self.payload.get("points", {})).get(resolved)
                    if point_payload is not None:
                        trials = np.asarray(point_payload.get("fit_q0_trials", ()), dtype=float)
                        has_committed_trials = trials.size > 0
                if streaming or has_committed_trials:
                    self.ax_heatmap.scatter(
                        [active_a],
                        [active_b],
                        s=260,
                        marker="*",
                        facecolor="#ffd43b",
                        edgecolor="black",
                        linewidth=1.1,
                        zorder=9,
                    )
                    self.ax_heatmap.annotate(
                        "active",
                        xy=(active_a, active_b),
                        xytext=(8, 8),
                        textcoords="offset points",
                        fontsize=8,
                        color="#8a5b00",
                        zorder=10,
                    )

        has_colorbar = metric_collection is not None
        if has_colorbar:
            self._heatmap_colorbar = self.heatmap_figure.colorbar(
                metric_collection,
                ax=self.ax_heatmap,
                fraction=self._HEATMAP_COLORBAR_FRACTION,
                pad=self._HEATMAP_COLORBAR_PAD,
            )
            self._heatmap_colorbar.set_label(metric_name)
            self._heatmap_colorbar.ax.tick_params(length=3, pad=2)
        else:
            self._heatmap_colorbar = None
        self._apply_figure_autolayout(self.heatmap_figure)

    def _draw_trials(self) -> None:
        live_state = self._live_trial_state()
        force_live_trials = self._should_force_live_trials(live_state)
        if live_state is not None and not force_live_trials and not self._should_use_live_trials(live_state):
            keep_live = (
                self._navigation_mode() != "best"
                and getattr(self, "_refresh_signal_active_point", None) is not None
                and self._live_search_matches_selected(live_state)
                and self._live_trials_streaming(live_state)
                and self._selection_matches_live_active_point(
                    int(self.a_index_var.get()),
                    int(self.b_index_var.get()),
                )
            )
            if not keep_live:
                live_state = None
        point: dict[str, Any] | None = None
        if live_state is None or not force_live_trials:
            if self._has_selected_point():
                if self._navigation_mode() == "free":
                    point = self._point_payload_for_selection()
                else:
                    try:
                        point = self._selected_point()
                    except KeyError:
                        point = None
        selected_metric = self._trials_display_metric()
        active_trial_index = None
        active_trial_q0 = None
        if live_state is not None:
            q0_trials, metric_trials, point_metric, _snapshot = self._live_trial_series_from_state(live_state)
            live_a_token = (
                int(live_state["a_index"])
                if "a_index" in live_state
                else float(live_state.get("active_a", np.nan))
            )
            live_b_token = (
                int(live_state["b_index"])
                if "b_index" in live_state
                else float(live_state.get("active_b", np.nan))
            )
            current_live_token = (
                live_a_token,
                live_b_token,
                point_metric,
                -1,
            )
            if q0_trials.size == 0:
                selected_trial_index = None
            else:
                current_index = int(self.trial_index_var.get())
                token_matches_context = self._trial_token_matches_context(
                    self._selected_trial_token,
                    a_token=live_a_token,
                    b_token=live_b_token,
                    point_metric=point_metric,
                )
                if not token_matches_context or current_index < 0 or current_index >= q0_trials.size:
                    selected_trial_index = int(np.clip(q0_trials.size - 1, 0, max(0, q0_trials.size - 1)))
                    self.trial_index_var.set(selected_trial_index)
                    self._selected_trial_token = current_live_token
                else:
                    selected_trial_index = current_index
            raw_active_trial_index = live_state.get("active_trial_index")
            if raw_active_trial_index is not None:
                try:
                    active_trial_index = int(raw_active_trial_index)
                except Exception:
                    active_trial_index = None
            raw_active_trial_q0 = live_state.get("active_trial_q0")
            if raw_active_trial_q0 is not None:
                try:
                    active_trial_q0 = float(raw_active_trial_q0)
                except Exception:
                    active_trial_q0 = None
                else:
                    if not np.isfinite(active_trial_q0):
                        active_trial_q0 = None
            point_status = "running"
        else:
            if point is None:
                self.ax_trials.clear()
                self.ax_trials.set_axis_on()
                trials_metric = self._trials_display_metric()
                apply_trials_axis_style(self.ax_trials, metric_name=trials_metric)
                show_trials_message(self.ax_trials, "Select a grid point to view trial history.")
                self._apply_figure_autolayout(self.trials_figure)
                self.trials_canvas.draw_idle()
                return
            if point is None:
                point_metric = selected_metric
                q0_trials = np.asarray([], dtype=float)
                metric_trials = np.asarray([], dtype=float)
                selected_trial_index = None
                point_status = "missing"
            else:
                q0_trials, metric_trials, point_metric = self._trial_series_for_point(point)
                selected_trial_index = None
                point_status = str(point.get("status", "computed"))
        self.ax_trials.clear()
        self.ax_trials.set_axis_on()
        apply_trials_axis_style(self.ax_trials, metric_name=point_metric)
        if point_status == "missing" or (point is None and live_state is None):
            self._refresh_trial_selector_controls(point, np.asarray([], dtype=float), np.asarray([], dtype=float), point_metric)
            show_trials_message(self.ax_trials, _NO_GRID_SOLUTION_MESSAGE)
            self._apply_figure_autolayout(self.trials_figure)
            self.trials_canvas.draw_idle()
            return
        if point_status == "pending":
            self._refresh_trial_selector_controls(point, np.asarray([], dtype=float), np.asarray([], dtype=float), point_metric)
            show_trials_message(
                self.ax_trials,
                "No data yet for this grid point.\nUse Refresh Artifact to reload partial scan progress.",
            )
        elif live_state is not None and q0_trials.size == 0:
            self._refresh_trial_selector_controls(None, np.asarray([], dtype=float), np.asarray([], dtype=float), point_metric)
            if active_trial_q0 is not None:
                waiting_text = f"Rendering active trial at q0={active_trial_q0:.6g}."
            elif self._live_trials_streaming(live_state):
                waiting_text = "Waiting for the first completed trial for the active point."
            else:
                waiting_text = (
                    "Computing the active grid point.\n"
                    "Trial curves appear after the point is saved."
                )
            show_trials_message(self.ax_trials, waiting_text)
        elif q0_trials.size > 0 and metric_trials.size == q0_trials.size:
            if not np.any(np.isfinite(metric_trials)):
                self._refresh_trial_selector_controls(point, q0_trials, metric_trials, point_metric)
                show_trials_message(
                    self.ax_trials,
                    f"No finite {point_metric} trial history for this grid point.",
                )
            else:
                if selected_trial_index is None:
                    selected_trial_index = self._selected_trial_index_for_point(point, q0_trials, metric_trials, point_metric)
                self._refresh_trial_selector_controls(
                    point,
                    q0_trials,
                    metric_trials,
                    point_metric,
                    selected_index_override=selected_trial_index,
                )
                order = np.argsort(q0_trials)
                self.ax_trials.plot(q0_trials[order], metric_trials[order], "-o", ms=4, lw=1.2, color="#2b6cb0")
                self.ax_trials.scatter(q0_trials, metric_trials, s=24, color="#2b6cb0", alpha=0.55)
                best_trial_index = self._best_trial_index_from_metric(metric_trials)
                if selected_trial_index is not None:
                    self.ax_trials.axvline(float(q0_trials[int(selected_trial_index)]), color="#d62728", ls="--", lw=1.6)
                    self.ax_trials.scatter(
                        [float(q0_trials[int(selected_trial_index)])],
                        [float(metric_trials[int(selected_trial_index)])],
                        color="#d62728",
                        s=42,
                        zorder=5,
                    )
                if active_trial_q0 is not None:
                    active_label = "active trial"
                    if active_trial_index is not None:
                        active_label = f"active trial #{int(active_trial_index) + 1}"
                    self.ax_trials.axvline(float(active_trial_q0), color="#f08c00", ls=":", lw=1.8, zorder=3)
                    self.ax_trials.text(
                        0.5,
                        0.98,
                        f"{active_label} at q0={float(active_trial_q0):.6g}",
                        transform=self.ax_trials.transAxes,
                        va="top",
                        ha="center",
                        color="#b26a00",
                    )
                if best_trial_index is not None:
                    self.ax_trials.scatter(
                        [float(q0_trials[best_trial_index])],
                        [float(metric_trials[best_trial_index])],
                        facecolor="none",
                        edgecolor="#f08c00",
                        linewidth=1.8,
                        s=74,
                        zorder=4,
                    )
                self.ax_trials.grid(alpha=0.25)
                self._apply_trials_axis_controls(q0_trials=q0_trials, metric_trials=metric_trials)
                self._sync_trials_axis_controls_from_axes()
                self._last_rendered_metric = selected_metric if selected_metric in METRICS else point_metric
                self._capture_current_slice_view_state(self._last_rendered_metric)
        else:
            self._refresh_trial_selector_controls(point, np.asarray([], dtype=float), np.asarray([], dtype=float), point_metric)
            show_trials_message(self.ax_trials, "Trial history unavailable")
            self.trials_xscale_var.set("linear scale")
            self.trials_yscale_var.set("linear scale")
            self._last_rendered_metric = self.metric_var.get() if self.metric_var.get() in METRICS else point_metric
            self._capture_current_slice_view_state(self._last_rendered_metric)
        self._apply_figure_autolayout(self.trials_figure)

        point_success = bool(point.get("success")) if isinstance(point, dict) else False
        success_text = point_status if point_status != "computed" else ("success" if point_success else "boundary/failed")
        best_metric_value = float(np.nanmin(metric_trials)) if metric_trials.size else float("nan")
        if selected_trial_index is None and point is not None:
            selected_trial_index = self._selected_trial_index_for_point(point, q0_trials, metric_trials, point_metric)
        selected_trial_text = "n/a"
        selected_metric_text = "n/a"
        if selected_trial_index is not None:
            selected_trial_text = (
                f"{int(selected_trial_index) + 1}/{len(q0_trials)} "
                f"(q0={_format_scalar(q0_trials[int(selected_trial_index)], '.6f')}, "
                f"{point_metric}={_format_scalar(metric_trials[int(selected_trial_index)], '.6e')})"
            )
            selected_metric_text = _format_scalar(metric_trials[int(selected_trial_index)], ".6e")
        selected_a_value = float("nan")
        selected_b_value = float("nan")
        a_values_arr = np.asarray(self.a_values, dtype=float)
        b_values_arr = np.asarray(self.b_values, dtype=float)
        selected_subject_label = "Selected point"
        if live_state is not None and point is None:
            selected_subject_label = "Active point"
            try:
                selected_a_value = float(live_state.get("active_a", live_state.get("a", np.nan)))
            except Exception:
                selected_a_value = float("nan")
            try:
                selected_b_value = float(live_state.get("active_b", live_state.get("b", np.nan)))
            except Exception:
                selected_b_value = float("nan")
        elif point is not None:
            try:
                selected_a_value = float(point.get("a", np.nan))
            except Exception:
                selected_a_value = float("nan")
            try:
                selected_b_value = float(point.get("b", np.nan))
            except Exception:
                selected_b_value = float("nan")
        free_ab = self._free_grid_selection_ab() if self._navigation_mode() == "free" else None
        if not np.isfinite(selected_a_value):
            if free_ab is not None:
                selected_a_value = float(free_ab[0])
            else:
                try:
                    a_idx = int(self.a_index_var.get())
                except Exception:
                    a_idx = -1
                if 0 <= a_idx < int(a_values_arr.size):
                    selected_a_value = float(a_values_arr[a_idx])
        if not np.isfinite(selected_b_value):
            if free_ab is not None:
                selected_b_value = float(free_ab[1])
            else:
                try:
                    b_idx = int(self.b_index_var.get())
                except Exception:
                    b_idx = -1
                if 0 <= b_idx < int(b_values_arr.size):
                    selected_b_value = float(b_values_arr[b_idx])
        slice_descriptor = dict(self.payload.get("selected_slice") or {})
        slice_text = self._slice_label(slice_descriptor) if slice_descriptor else "single slice"
        best_trial_index = self._best_trial_index_from_metric(metric_trials)
        best_q0_value = point.get("q0", np.nan) if point is not None else np.nan
        if point is None and best_trial_index is not None and q0_trials.size:
            best_q0_value = q0_trials[int(best_trial_index)]
        self.status_var.set(
            f"Slice: {slice_text}\n"
            f"Displayed grid metric: {self.metric_var.get()}\n"
            f"Displayed q0-curve metric: {point_metric}\n"
            f"Run target metric: {self.run_target_metric}\n"
            f"{selected_subject_label}: a={_format_scalar(selected_a_value, '.3f')}, "
            f"b={_format_scalar(selected_b_value, '.3f')}\n"
            f"Selected trial: {selected_trial_text}\n"
            f"Best q0: {_format_scalar(best_q0_value, '.6f')}\n"
            f"Selected {point_metric}: {selected_metric_text}\n"
            f"Best {point_metric}: {best_metric_value:.6e}\n"
            f"Status: {success_text}"
        )

    def _refresh_summary(self) -> None:
        live_state = self._live_trial_state()
        if self._should_force_live_trials(live_state):
            self.summary_var.set(self._active_point_waiting_summary(live_state))
            return
        if not self._has_saved_selected_point():
            free_ab = self._free_grid_selection_ab()
            selected_a_value = float("nan")
            selected_b_value = float("nan")
            if free_ab is not None:
                selected_a_value, selected_b_value = free_ab
            slice_descriptor = dict(self.payload.get("selected_slice") or {})
            self.summary_var.set(
                "\n".join(
                    [
                        f"slice = {self._slice_label(slice_descriptor) if slice_descriptor else 'single slice'}",
                        f"a = {_format_scalar(selected_a_value, '.3f')}",
                        f"b = {_format_scalar(selected_b_value, '.3f')}",
                        _NO_GRID_SOLUTION_MESSAGE,
                    ]
                )
            )
            return
        point = self._selected_point()
        diagnostics = self._selected_diagnostics()
        q0_trials, metric_trials, point_metric = self._trial_series_for_point(point)
        selected_trial_index = self._selected_trial_index_for_point(point, q0_trials, metric_trials, point_metric)
        metric_snapshot: dict[str, float] = {}
        if selected_trial_index is not None:
            for metric_name in METRICS:
                metric_history = self._metric_history_for_point(point, metric_name)
                if metric_history.size == q0_trials.size:
                    metric_snapshot[metric_name] = float(metric_history[int(selected_trial_index)])
        selected_a_value = float("nan")
        selected_b_value = float("nan")
        a_values_arr = np.asarray(self.a_values, dtype=float)
        b_values_arr = np.asarray(self.b_values, dtype=float)
        if point is not None:
            try:
                selected_a_value = float(point.get("a", np.nan))
            except Exception:
                selected_a_value = float("nan")
            try:
                selected_b_value = float(point.get("b", np.nan))
            except Exception:
                selected_b_value = float("nan")
        if not np.isfinite(selected_a_value):
            try:
                a_idx = int(self.a_index_var.get())
            except Exception:
                a_idx = -1
            if 0 <= a_idx < int(a_values_arr.size):
                selected_a_value = float(a_values_arr[a_idx])
        if not np.isfinite(selected_b_value):
            try:
                b_idx = int(self.b_index_var.get())
            except Exception:
                b_idx = -1
            if 0 <= b_idx < int(b_values_arr.size):
                selected_b_value = float(b_values_arr[b_idx])
        slice_descriptor = dict(self.payload.get("selected_slice") or {})
        lines = [
            f"slice = {self._slice_label(slice_descriptor) if slice_descriptor else 'single slice'}",
            f"a = {_format_scalar(selected_a_value, '.3f')}",
            f"b = {_format_scalar(selected_b_value, '.3f')}",
            f"status = {point.get('status', 'computed')}",
            f"best_q0 = {_format_scalar(point.get('q0', np.nan), '.6f')}",
            f"chi2 = {_format_scalar(metric_snapshot.get('chi2', diagnostics.get('chi2', np.nan)), '.6e')}",
            f"rho2 = {_format_scalar(metric_snapshot.get('rho2', diagnostics.get('rho2', np.nan)), '.6e')}",
            f"eta2 = {_format_scalar(metric_snapshot.get('eta2', diagnostics.get('eta2', np.nan)), '.6e')}",
        ]
        if selected_trial_index is not None:
            lines[4:4] = [
                f"selected_trial = {int(selected_trial_index) + 1}/{len(q0_trials)}",
                f"selected_q0 = {_format_scalar(q0_trials[int(selected_trial_index)], '.6f')}",
                f"selected_{point_metric} = {_format_scalar(metric_trials[int(selected_trial_index)], '.6e')}",
            ]
        elapsed_seconds = diagnostics.get("elapsed_seconds")
        try:
            elapsed_value = float(elapsed_seconds)
        except Exception:
            elapsed_value = float("nan")
        if np.isfinite(elapsed_value):
            lines[3] = f"{lines[3]} in {elapsed_value:.3f} s"
            lines.append(f"elapsed = {elapsed_value:.3f} s")
        metrics_mask_text = self._metrics_mask_summary(diagnostics)
        if metrics_mask_text:
            lines.append(f"metrics_mask = {metrics_mask_text}")
            mask_pixels_text = self._metrics_mask_pixel_summary(point, diagnostics)
            if mask_pixels_text:
                lines.append(f"metrics_mask_pixels = {mask_pixels_text}")
        tr_mask_text = self._tr_mask_summary(diagnostics)
        if tr_mask_text:
            lines.append(f"tr_mask = {tr_mask_text}")
        message = diagnostics.get("optimizer_message")
        if message:
            lines.extend(["", "message:", str(message)])
        bracket = diagnostics.get("adaptive_bracket")
        if bracket is not None:
            lines.append(f"adaptive_bracket = {bracket}")
        selected_trials = np.asarray(point.get(f"fit_{self.metric_var.get()}_trials", ()), dtype=float)
        if self.metric_var.get() != str(self._selected_point().get("target_metric", self.run_target_metric)) and selected_trials.size == 0:
            lines.extend(
                [
                    "",
                    "note:",
                    "current artifact stores q0-trial history only for the run target metric,",
                    f"so the right-hand plot remains {self.run_target_metric}(q0)",
                ]
            )
        self.summary_var.set("\n".join(lines))

    def _open_plot_script(self, *extra_args: str) -> None:
        if self.artifact_h5 is None:
            return
        script_path = Path(__file__).resolve().parents[2] / "examples" / "plot_ab_scan_artifacts.py"
        cmd = [sys.executable, str(script_path), str(self.artifact_h5)]
        selected_slice_key = self._selected_slice_key()
        if selected_slice_key:
            cmd.extend(["--slice-key", selected_slice_key])
        cmd.extend(extra_args)
        subprocess.Popen(
            cmd,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )

    def _open_artifact(self) -> None:
        initial_dir = None
        if self.last_artifact_dir is not None and self.last_artifact_dir.is_dir():
            initial_dir = str(self.last_artifact_dir)
        elif self.artifact_h5 is not None:
            initial_dir = str(self.artifact_h5.parent)
        selected = filedialog.askopenfilename(
            title="Open pyCHMP scan artifact",
            initialdir=initial_dir,
            filetypes=[("pyCHMP HDF5 artifacts", "*.h5 *.hdf5 *.h5r")],
        )
        if not selected:
            return
        self._reload_payload(Path(selected))

    def _open_selected_maps(self) -> None:
        if self._selected_solution_plot_context() is None:
            live_state = self._live_trial_state()
            if live_state is not None and self._live_trials_streaming(live_state):
                self.status_var.set(
                    "Live trial maps are not available yet for the selected trial; wait for the next artifact refresh."
                )
            else:
                self.status_var.set("No completed point available yet; refresh after the scan advances.")
            return
        if self.selected_solution_window is None:
            self.selected_solution_window = _SelectedSolutionWindow(self)
        self.selected_solution_window.present()
        self.selected_solution_window.update_selection()

    def _grid_summary_png_candidate(self) -> Path | None:
        if self.artifact_h5 is None:
            return None
        candidate = self.artifact_h5.with_name(f"{self.artifact_h5.stem}_grid.png")
        return candidate if candidate.exists() else None

    def _open_external_file(self, path: Path) -> bool:
        try:
            resolved = Path(path).expanduser().resolve()
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(resolved)], start_new_session=True)
                return True
            if os.name == "nt":
                os.startfile(str(resolved))
                return True
            subprocess.Popen(["xdg-open", str(resolved)], start_new_session=True)
            return True
        except Exception:
            return False

    def _open_grid_summary(self) -> None:
        png_candidate = self._grid_summary_png_candidate()
        if png_candidate is not None and self._open_external_file(png_candidate):
            self.status_var.set(f"Opened saved grid summary PNG: {png_candidate}")
            return
        self.status_var.set("Generating grid summary plot; this can take a while for large sparse artifacts...")
        self._open_plot_script("--grid")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive viewer for consolidated `(a,b)` scan artifacts.")
    parser.add_argument("artifact_h5", type=Path, nargs="?", help="Optional consolidated H5 produced by scan_ab_obs_map.py")
    parser.add_argument("--metric", choices=METRICS, default=None, help="Optional initial metric override shown in the heatmap.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = tk.Tk()
    PychmpViewApp(
        root,
        args.artifact_h5,
        initial_metric=str(args.metric),
    )
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
