"""Interactive viewer for consolidated `(a, b)` scan artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
import time
import tkinter as tk
from pathlib import Path
from tkinter import scrolledtext
from tkinter import filedialog, ttk
from typing import Any

import numpy as np
from astropy.io import fits
from matplotlib.collections import PatchCollection
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .ab_scan_artifacts import (
    METRICS,
    best_grid_index,
    build_patch_grid_model,
    find_record_for_point,
    load_scan_file,
    with_observer_metadata,
)
from .metrics import resolve_threshold_mask
from .q0_artifact_panel import Q0ArtifactPanelFigure


def _viewer_state_path() -> Path:
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "pychmp" / "viewer_state.json"
    return home / ".config" / "pychmp" / "viewer_state.json"


def _load_last_directory() -> Path | None:
    state_path = _viewer_state_path()
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        last_dir = payload.get("last_artifact_dir")
        if not last_dir:
            return None
        path = Path(last_dir).expanduser()
        return path if path.is_dir() else None
    except Exception:
        return None


def _save_last_directory(path: Path | None) -> None:
    if path is None:
        return
    try:
        state_path = _viewer_state_path()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps({"last_artifact_dir": str(path)}, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


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
        toolbar.columnconfigure(12, weight=1)
        self.blos_button = ttk.Button(toolbar, text="Load B_los", command=self._load_blos_now)
        self.blos_button.grid(row=0, column=0, sticky="w")
        _ToolTip(self.blos_button, "Load B_los: fetch and cache the external reference panel on demand for the current model.")
        self.export_png_button = ttk.Button(toolbar, text="Export PNG", command=self._export_current_png)
        self.export_png_button.grid(row=0, column=1, sticky="w", padx=(8, 0))
        _ToolTip(self.export_png_button, "Export the currently displayed selected-solution figure to PNG, preserving the current zoom/pan axis state.")
        self.mask_contour_check = ttk.Checkbutton(
            toolbar,
            text="Show ROI Mask Contour",
            variable=self.show_mask_contours_var,
            command=self._toggle_mask_contours,
        )
        self.mask_contour_check.grid(row=0, column=2, sticky="w", padx=(10, 0))
        _ToolTip(self.mask_contour_check, "Toggle the displayed ROI mask contour overlay on the observed, modeled, and residual panels.")
        ttk.Label(toolbar, textvariable=self.mask_type_var, anchor=tk.W).grid(row=0, column=3, sticky="w", padx=(10, 0))
        self.status_var = tk.StringVar(value="Selected solution window is ready.")
        ttk.Label(toolbar, textvariable=self.status_var, anchor=tk.W).grid(row=0, column=4, columnspan=9, sticky="ew", padx=(10, 0))

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

        self.figure = Figure(figsize=(14.8, 9.2), dpi=100, constrained_layout=True)
        self.panel = Q0ArtifactPanelFigure(self.figure)
        self.canvas = FigureCanvasTkAgg(self.figure, master=outer)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        toolbar_frame = ttk.Frame(outer)
        toolbar_frame.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.grid(row=0, column=0, sticky="w")

    def present(self) -> None:
        self.window.deiconify()
        self.window.lift()
        self.window.focus_force()

    def _parse_display_limit(self, value: str) -> float | None:
        text = str(value).strip()
        if not text:
            return None
        try:
            numeric = float(text)
        except Exception:
            return None
        return numeric if np.isfinite(numeric) else None

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
        slice_label = str(context["slice_label"])
        mask_type = str(diagnostics.get("mask_type", "union")).strip() or "union"
        selected_trial_index = diagnostics.get("selected_trial_index")
        selected_trial_count = diagnostics.get("selected_trial_count")
        trial_label = ""
        if selected_trial_index is not None and selected_trial_count is not None:
            trial_label = f" - trial {int(selected_trial_index) + 1}/{int(selected_trial_count)}"
        metrics_mask_source = str(diagnostics.get("metrics_mask_source", "")).strip().lower()
        if metrics_mask_source == "explicit_fits":
            mask_path = str(diagnostics.get("metrics_mask_fits", "")).strip()
            threshold_text = Path(mask_path).name if mask_path else "explicit FITS"
        else:
            threshold_value = diagnostics.get("metrics_mask_threshold", diagnostics.get("threshold"))
            try:
                threshold_text = f"{float(threshold_value):.3f}"
            except Exception:
                threshold_text = "n/a"
        self.mask_type_var.set(f"ROI mask: {mask_type} @ {threshold_text}")
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
            self.figure.savefig(out_png, dpi=self.figure.dpi)
            self.status_var.set(f"Exported PNG: {out_png}")
            _save_last_directory(out_png.parent)
            self.app.last_artifact_dir = out_png.parent
        except Exception as exc:
            self.status_var.set(f"PNG export failed: {exc}")


class PychmpViewApp:
    _LOAD_RETRY_ATTEMPTS = 8
    _LOAD_RETRY_DELAY_S = 0.35
    _EXTERNAL_REFRESH_POLL_MS = 1200
    _MAX_LOG_LINES = 3000
    _INITIAL_LOG_READ_BYTES = 262_144
    _HEATMAP_FOOTER_HEIGHT = 82
    _TRIALS_FOOTER_HEIGHT = 104
    _TOOLBAR_HEIGHT = 78
    _NOTEBOOK_TAB_HEIGHT = 28
    _WINDOW_VERTICAL_CHROME = 28
    _DISPLAY_PAD = 8
    _ACTIVE_REFRESH_GRACE_S = 10.0
    _MIN_LEFT_PANEL_WIDTH = 420
    _MIN_CENTER_PANEL_WIDTH = 520
    _MIN_RIGHT_PANEL_WIDTH = 300
    _MIN_TOOLBAR_WIDTH = 1080

    def __init__(self, root: tk.Tk, artifact_h5: Path | None, *, initial_metric: str | None = None) -> None:
        self.root = root
        self.artifact_h5 = artifact_h5
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
        self.trials_xmin_var = tk.StringVar(value="")
        self.trials_xmax_var = tk.StringVar(value="")
        self.trials_ymin_var = tk.StringVar(value="")
        self.trials_ymax_var = tk.StringVar(value="")
        self.trials_xscale_var = tk.StringVar(value="linear scale")
        self.trials_yscale_var = tk.StringVar(value="linear scale")
        self.trial_index_var = tk.IntVar(value=0)
        self.trial_label_var = tk.StringVar(value="trial: n/a")
        self.a_index_var = tk.IntVar(value=0)
        self.b_index_var = tk.IntVar(value=0)
        self.scan_state_var = tk.StringVar(value="NO ARTIFACT")
        self.scan_state_detail_var = tk.StringVar(value="No artifact loaded")
        self.scan_state_info_var = tk.StringVar(value="No artifact loaded")
        self.status_var = tk.StringVar(value="")
        self.summary_var = tk.StringVar(value="")
        self.last_artifact_dir = _load_last_directory()
        self.refresh_signal_path: Path | None = None
        self._refresh_signal_mtime_ns = -1
        self._refresh_signal_phase = ""
        self._refresh_signal_pending_points: list[tuple[float, float]] = []
        self._external_refresh_after_id: str | None = None
        self._present_window_after_id: str | None = None
        self._is_closing = False
        self.info_notebook: ttk.Notebook | None = None
        self.info_tab: ttk.Frame | None = None
        self.center_notebook: ttk.Notebook | None = None
        self.q0_trials_tab: ttk.Frame | None = None
        self.left_panel_width = 0
        self.center_panel_width = 0
        self.right_panel_width = 0
        self.left_notebook: ttk.Notebook | None = None
        self.right_notebook: ttk.Notebook | None = None
        self.heatmap_tab: ttk.Frame | None = None
        self.trials_tab: ttk.Frame | None = None
        self.trials_controls: ttk.Frame | None = None
        self.trial_slider: ttk.Scale | None = None
        self.trial_status_label: ttk.Label | None = None
        self.trial_best_button: ttk.Button | None = None
        self._reset_icon_image: tk.PhotoImage | None = None
        self._best_trial_icon_image: tk.PhotoImage | None = None
        self.info_frame: ttk.Frame | None = None
        self.info_canvas: tk.Canvas | None = None
        self.info_canvas_window: int | None = None
        self.info_scrollbar: ttk.Scrollbar | None = None
        self.info_text: tk.Text | None = None
        self.slice_menu: ttk.Combobox | None = None
        self.slice_display_label: ttk.Label | None = None
        self.available_slices: list[dict[str, Any]] = []
        self.slice_display_state: dict[str, dict[str, Any]] = {}
        self.open_artifact_button: ttk.Button | None = None
        self.display_selected_button: ttk.Button | None = None
        self.summary_button: ttk.Button | None = None
        self.refresh_button: ttk.Button | None = None
        self.scan_state_badge: tk.Label | None = None
        self.scan_state_label: ttk.Label | None = None
        self.scan_state_info_label: ttk.Label | None = None
        self.square_display_side = 0
        self.selected_solution_window: _SelectedSolutionWindow | None = None
        self._selected_trial_token: tuple[int, int, str, int] | None = None
        self._updating_trial_slider = False

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
        # square display region + fixed tab footer + outer chrome/padding.
        required_height = (
            16  # outer frame vertical padding (8 top + 8 bottom)
            + self._TOOLBAR_HEIGHT
            + 6  # gap below toolbar
            + self._NOTEBOOK_TAB_HEIGHT
            + self.square_display_side
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
        center_width = max(self._MIN_CENTER_PANEL_WIDTH, int(round(left_center_width * 0.54)))
        left_width = max(self._MIN_LEFT_PANEL_WIDTH, left_center_width - center_width)
        center_width = max(self._MIN_CENTER_PANEL_WIDTH, left_center_width - left_width)

        self.left_panel_width = int(left_width)
        self.center_panel_width = int(center_width)
        self.right_panel_width = int(right_width)
        self.square_display_side = max(220, min(self.left_panel_width, self.center_panel_width) - 24)

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

        ttk.Button(primary_toolbar, text="Best", command=self._jump_to_best).grid(row=0, column=2, sticky="w", padx=(0, 10))

        ttk.Label(primary_toolbar, text="Slice").grid(row=0, column=3, sticky="w")
        self.slice_menu = ttk.Combobox(primary_toolbar, width=18, state="readonly")
        self.slice_menu.grid(row=0, column=4, sticky="w", padx=(6, 6))
        self.slice_menu.bind("<<ComboboxSelected>>", lambda _event: self._on_slice_changed())
        self.slice_display_label = ttk.Label(primary_toolbar, textvariable=self.slice_display_var, anchor=tk.W)
        self.slice_display_label.grid(row=0, column=4, sticky="w", padx=(6, 6))

        ttk.Label(primary_toolbar, text="a").grid(row=0, column=5, sticky="w")
        self.a_menu = ttk.Combobox(primary_toolbar, width=10, state="readonly")
        self.a_menu.configure(width=10)
        self.a_menu.grid(row=0, column=6, sticky="w", padx=(6, 10))
        self.a_menu.bind("<<ComboboxSelected>>", lambda _event: self._on_a_changed())

        ttk.Label(primary_toolbar, text="b").grid(row=0, column=7, sticky="w")
        self.b_menu = ttk.Combobox(primary_toolbar, width=10, state="readonly")
        self.b_menu.configure(width=10)
        self.b_menu.grid(row=0, column=8, sticky="w", padx=(6, 12))
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
        content.columnconfigure(0, minsize=self.left_panel_width + self.center_panel_width + 8, weight=1)
        content.columnconfigure(1, minsize=self.right_panel_width, weight=0)
        content.columnconfigure(2, weight=0)
        content.rowconfigure(0, weight=1)

        plot_area = ttk.Frame(content)
        plot_area.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        plot_area.columnconfigure(0, minsize=self.left_panel_width, weight=1)
        plot_area.columnconfigure(1, minsize=self.center_panel_width, weight=1)
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
        info_tab.rowconfigure(0, weight=1)
        self.info_tab = info_tab
        info_notebook.add(info_tab, text="Info")

        info_text = scrolledtext.ScrolledText(
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
        )
        info_text.grid(row=0, column=0, sticky="nsew")
        self.info_text = info_text
        self.info_scrollbar = None
        try:
            info_text.vbar.configure(width=14)
        except Exception:
            pass
        info_text.configure(state=tk.DISABLED)
        info_text.tag_configure("heading", font=("TkDefaultFont", 10, "bold"), spacing1=2, spacing3=4)
        info_text.tag_configure("body", font=("TkDefaultFont", 10))

        left_notebook = ttk.Notebook(plot_area)
        left_notebook.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left_notebook.configure(width=self.left_panel_width)
        self.left_notebook = left_notebook
        right_notebook = ttk.Notebook(plot_area)
        right_notebook.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        right_notebook.configure(width=self.center_panel_width)
        self.center_notebook = right_notebook

        heatmap_tab = ttk.Frame(left_notebook)
        heatmap_tab.columnconfigure(0, weight=1)
        heatmap_tab.rowconfigure(0, weight=1)
        heatmap_tab.rowconfigure(1, minsize=self._HEATMAP_FOOTER_HEIGHT, weight=0)
        heatmap_tab.rowconfigure(2, weight=0)
        self.heatmap_tab = heatmap_tab
        left_notebook.add(heatmap_tab, text="Grid Metric")

        trials_tab = ttk.Frame(right_notebook)
        trials_tab.columnconfigure(0, weight=1)
        trials_tab.rowconfigure(0, weight=1)
        trials_tab.rowconfigure(1, minsize=self._TRIALS_FOOTER_HEIGHT, weight=0)
        self.trials_tab = trials_tab
        right_notebook.add(trials_tab, text="Trials")
        self.q0_trials_tab = trials_tab

        self.heatmap_figure = Figure(figsize=(4.8, 4.2), dpi=100)
        self.ax_heatmap = self.heatmap_figure.add_subplot(111)
        self.heatmap_divider = make_axes_locatable(self.ax_heatmap)
        self.ax_heatmap_cbar = self.heatmap_divider.append_axes("right", size="5%", pad=0.08)
        self.ax_heatmap.set_box_aspect(1.0)
        self.heatmap_figure.subplots_adjust(left=0.12, right=0.88, bottom=0.14, top=0.92)

        self.trials_figure = Figure(figsize=(4.8, 4.2), dpi=100)
        self.ax_trials = self.trials_figure.add_subplot(111)
        self.ax_trials.set_box_aspect(1.0)
        self.trials_figure.subplots_adjust(left=0.14, right=0.98, bottom=0.14, top=0.92)
        self._heatmap_colorbar = None
        self.heatmap_canvas = FigureCanvasTkAgg(self.heatmap_figure, master=heatmap_tab)
        self.heatmap_canvas_widget = self.heatmap_canvas.get_tk_widget()
        self.heatmap_canvas_widget.grid(row=0, column=0, sticky="nsew", padx=self._DISPLAY_PAD, pady=self._DISPLAY_PAD)
        self.heatmap_canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self.heatmap_legend_row = ttk.Frame(heatmap_tab)
        self.heatmap_legend_row.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        self._build_heatmap_legend_row()
        trial_status = ttk.Label(heatmap_tab, textvariable=self.trial_label_var, anchor="w", justify=tk.LEFT)
        trial_status.grid(row=2, column=0, sticky="ew", padx=(10, 0), pady=(2, 0))
        self.trial_status_label = trial_status

        self.trials_canvas = FigureCanvasTkAgg(self.trials_figure, master=trials_tab)
        self.trials_canvas_widget = self.trials_canvas.get_tk_widget()
        self.trials_canvas_widget.grid(row=0, column=0, sticky="nsew", padx=self._DISPLAY_PAD, pady=self._DISPLAY_PAD)
        self.trials_canvas.mpl_connect("button_press_event", self._on_trials_canvas_click)
        trials_controls = ttk.Frame(trials_tab)
        trials_controls.grid(row=1, column=0, sticky="ew", padx=self._DISPLAY_PAD, pady=(0, 4))
        self.trials_controls = trials_controls
        self._build_trials_controls()

        self.root.bind("<Configure>", self._on_resize)

        if self.artifact_h5 is not None:
            self._reload_payload()
        else:
            self.status_var.set("No artifact loaded. Use Open Artifact to choose a consolidated scan H5 file.")
            self.summary_var.set("No artifact loaded.")
            self._refresh_all()
        self._refresh_selector_values()
        self._apply_fixed_body_geometry()

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
        except Exception:
            pass

    def _close_root_window(self) -> None:
        self._is_closing = True
        for after_id in (self._external_refresh_after_id, self._present_window_after_id):
            if after_id is None:
                continue
            try:
                self.root.after_cancel(after_id)
            except Exception:
                pass
        self._external_refresh_after_id = None
        self._present_window_after_id = None
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
        self._apply_fixed_body_geometry()

    def _apply_fixed_body_geometry(self) -> None:
        square_side = int(self.square_display_side)
        canvas_side = max(180, square_side - 2 * self._DISPLAY_PAD)

        if self.left_notebook is not None:
            self.left_notebook.configure(width=self.left_panel_width)
        if self.right_notebook is not None:
            self.right_notebook.configure(width=self.center_panel_width)
        if self.info_notebook is not None:
            self.info_notebook.configure(width=self.right_panel_width)

        if self.heatmap_tab is not None:
            self.heatmap_tab.rowconfigure(0, minsize=square_side, weight=0)
        if self.trials_tab is not None:
            self.trials_tab.rowconfigure(0, minsize=square_side, weight=0)
        self.heatmap_canvas_widget.configure(width=canvas_side, height=canvas_side)
        self.trials_canvas_widget.configure(width=canvas_side, height=canvas_side)
        self.heatmap_figure.set_size_inches(canvas_side / 100.0, canvas_side / 100.0, forward=True)
        self.trials_figure.set_size_inches(canvas_side / 100.0, canvas_side / 100.0, forward=True)

    def _reset_heatmap_colorbar_axis(self) -> None:
        if self._heatmap_colorbar is not None:
            try:
                self._heatmap_colorbar.remove()
            except Exception:
                pass
            self._heatmap_colorbar = None
        try:
            self.ax_heatmap_cbar.remove()
        except Exception:
            pass
        self.heatmap_divider = make_axes_locatable(self.ax_heatmap)
        self.ax_heatmap_cbar = self.heatmap_divider.append_axes("right", size="5%", pad=0.08)

    def _refresh_info_text(self) -> None:
        if self.info_text is None:
            return
        self.info_text.configure(state=tk.NORMAL)
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, "Scan State\n", "heading")
        self.info_text.insert(tk.END, f"{self._wrap_info_block(self.scan_state_info_var.get())}\n\n", "body")
        self.info_text.insert(tk.END, "Artifact\n", "heading")
        self.info_text.insert(tk.END, f"{self._wrap_info_block(self.status_var.get())}\n\n", "body")
        self.info_text.insert(tk.END, "Selected Point\n", "heading")
        self.info_text.insert(tk.END, f"{self._wrap_info_block(self.summary_var.get())}\n", "body")
        self.info_text.configure(state=tk.DISABLED)

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

    def _refresh_action_states(self) -> None:
        has_artifact = self.artifact_h5 is not None
        has_selected_point = has_artifact and self._has_selected_point()
        if self.open_artifact_button is not None:
            self.open_artifact_button.state(["!disabled"])
        if self.display_selected_button is not None:
            self.display_selected_button.state(["!disabled"] if has_selected_point else ["disabled"])
        if self.summary_button is not None:
            self.summary_button.state(["!disabled"] if has_artifact else ["disabled"])
        if self.refresh_button is not None:
            self.refresh_button.state(["!disabled"] if has_artifact else ["disabled"])

    def _scan_state_snapshot(self) -> tuple[str, str, str, str, str]:
        if self.artifact_h5 is None or not self.payload:
            return "NO ARTIFACT", "No artifact", "No artifact loaded", "#6c757d", "white"

        points = self.payload.get("points", {})
        diagnostics = dict(self.payload.get("diagnostics") or {})
        total = int(len(points))
        slice_descriptor = dict(self.payload.get("selected_slice") or {})
        slice_label = self._slice_label(slice_descriptor) if slice_descriptor else "current slice"
        slice_count = max(1, int(len(self.available_slices) or 1))
        adaptive_sparse = (
            str(diagnostics.get("artifact_kind", "")).strip().lower() == "pychmp_ab_scan_sparse_points"
            and str(diagnostics.get("search_mode", "")).strip().lower() == "adaptive_local_single_frequency"
        )
        if total == 0:
            toolbar_detail = f"{slice_label} | 0/{total} computed"
            info_detail = f"Current slice: {slice_label}\nGrid points: {total}\nComputed: 0\nArtifact slices: {slice_count}"
            return "EMPTY", toolbar_detail, info_detail, "#6c757d", "white"

        statuses = [str(point.get("status", "computed")).strip().lower() for point in points.values()]
        pending = int(sum(status == "pending" for status in statuses))
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
        phase_running = bool(phase) and phase not in {"scan complete", "complete"}
        phase_complete = phase in {"scan complete", "complete"}

        if phase_complete:
            badge = "FINISHED"
            color = "#2b8a3e"
        elif adaptive_sparse and refresh_active:
            badge = "RUNNING"
            color = "#0b7285"
        elif (refresh_active or phase_running) and noncomputed > 0:
            badge = "RUNNING"
            color = "#0b7285"
        elif adaptive_sparse and phase_running:
            badge = "INCOMPLETE"
            color = "#b26a00"
        elif noncomputed > 0:
            badge = "INCOMPLETE"
            color = "#b26a00"
        else:
            badge = "FINISHED"
            color = "#2b8a3e"

        toolbar_detail = f"{slice_label} | {computed}/{total} computed"
        info_lines = [
            f"Current slice: {slice_label}",
            f"Grid points: {total}",
            f"Computed: {computed}",
        ]
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
        pending_points = list(getattr(self, "_refresh_signal_pending_points", []) or [])
        if pending_points:
            active_points = ", ".join(
                f"(a={a_value:.3f}, b={b_value:.3f})"
                for a_value, b_value in pending_points[:4]
            )
            if len(pending_points) > 4:
                active_points += f", ... (+{len(pending_points) - 4} more)"
            info_lines.append(f"Active point(s): {active_points}")
        return badge, toolbar_detail, "\n".join(info_lines), color, "white"

    def _read_refresh_signal_payload(self) -> dict[str, Any]:
        if self.refresh_signal_path is None or not self.refresh_signal_path.exists():
            return {"phase": "", "pending_points": []}
        try:
            text = self.refresh_signal_path.read_text(encoding="utf-8").strip()
        except Exception:
            return {"phase": "", "pending_points": []}
        if not text:
            return {"phase": "", "pending_points": []}
        if text.startswith("{"):
            try:
                payload = json.loads(text)
                phase = str(payload.get("phase", "")).strip()
                pending_points: list[tuple[float, float]] = []
                for item in payload.get("pending_points", []) or []:
                    try:
                        a_value = float(item.get("a"))
                        b_value = float(item.get("b"))
                    except Exception:
                        continue
                    if np.isfinite(a_value) and np.isfinite(b_value):
                        pending_points.append((a_value, b_value))
                return {"phase": phase, "pending_points": pending_points}
            except Exception:
                return {"phase": "", "pending_points": []}
        parts = text.split(maxsplit=1)
        phase = parts[1].strip() if len(parts) == 2 else parts[0].strip()
        return {"phase": phase, "pending_points": []}

    def _refresh_scan_state_display(self) -> None:
        badge, toolbar_detail, info_detail, bg, fg = self._scan_state_snapshot()
        self.scan_state_var.set(badge)
        self.scan_state_detail_var.set(toolbar_detail)
        self.scan_state_info_var.set(info_detail)
        if self.scan_state_badge is not None:
            self.scan_state_badge.configure(bg=bg, fg=fg)

    def _build_heatmap_legend_row(self) -> None:
        def add_item(symbol: str, color: str, label: str) -> None:
            item = ttk.Frame(self.heatmap_legend_row)
            item.pack(side=tk.LEFT, padx=(0, 12))
            tk.Label(item, text=symbol, fg=color, font=("TkDefaultFont", 12, "bold")).pack(side=tk.LEFT)
            ttk.Label(item, text=label).pack(side=tk.LEFT, padx=(4, 0))

        add_item("○", "#d62728", "best chi2")
        add_item("□", "#1f77b4", "best rho2")
        add_item("△", "#2ca02c", "best eta2")
        add_item("×", "#444444", "selected point")
        add_item("*", "#f08c00", "active point")

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
            entry.bind("<Return>", lambda _event: self._refresh_all())

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

        xmin = self._parse_axis_limit(self.trials_xmin_var.get())
        xmax = self._parse_axis_limit(self.trials_xmax_var.get())
        ymin = self._parse_axis_limit(self.trials_ymin_var.get())
        ymax = self._parse_axis_limit(self.trials_ymax_var.get())

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

    def _reset_trials_view(self) -> None:
        self.trials_xscale_var.set("linear scale")
        self.trials_yscale_var.set("linear scale")
        self.trials_xmin_var.set("")
        self.trials_xmax_var.set("")
        self.trials_ymin_var.set("")
        self.trials_ymax_var.set("")
        self._refresh_all()

    def _reset_trials_x_axis(self) -> None:
        self.trials_xscale_var.set("linear scale")
        self.trials_xmin_var.set("")
        self.trials_xmax_var.set("")
        self._refresh_all()

    def _reset_trials_y_axis(self) -> None:
        self.trials_yscale_var.set("linear scale")
        self.trials_ymin_var.set("")
        self.trials_ymax_var.set("")
        self._refresh_all()

    def _slice_label(self, descriptor: dict[str, Any]) -> str:
        display_label = str(descriptor.get("display_label", "")).strip()
        if display_label:
            return display_label
        label = str(descriptor.get("label", descriptor.get("key", ""))).strip()
        return label or "default"

    def _selected_slice_key(self) -> str | None:
        value = str(self.slice_key_var.get()).strip()
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
        return f"{artifact_text}::{key}"

    def _default_point_selection(self, metric_name: str) -> tuple[int, int]:
        if not self.payload or self.a_values.size == 0 or self.b_values.size == 0:
            return 0, 0
        try:
            if np.any(np.isfinite(np.asarray(self.payload[metric_name], dtype=float))):
                return best_grid_index(self.payload, metric_name)
        except Exception:
            pass
        return 0, 0

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
                "trials_xscale": self._normalize_scale_choice(str(self.trials_xscale_var.get() or "linear scale")),
                "trials_yscale": self._normalize_scale_choice(str(self.trials_yscale_var.get() or "linear scale")),
            }
        existing.update(
            {
                "metric": current_metric,
                "a_index": int(self.a_index_var.get()),
                "b_index": int(self.b_index_var.get()),
                "trials_by_metric": per_metric,
            }
        )
        self.slice_display_state[token] = existing

    def _reset_trials_controls_only(self) -> None:
        self.trials_xscale_var.set("linear scale")
        self.trials_yscale_var.set("linear scale")
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
            self.a_index_var.set(default_a)
            self.b_index_var.set(default_b)
            self._reset_trials_controls_only()
            self._preferred_initial_metric = None
            return

        saved_metric = str(state.get("metric", metric_name))
        if saved_metric not in METRICS:
            saved_metric = metric_name if metric_name in METRICS else "chi2"
        self.metric_var.set(saved_metric)

        restored_a, restored_b = self._default_point_selection(saved_metric)
        if self.a_values.size:
            restored_a = int(np.clip(int(state.get("a_index", restored_a)), 0, max(0, self.a_values.size - 1)))
        if self.b_values.size:
            restored_b = int(np.clip(int(state.get("b_index", restored_b)), 0, max(0, self.b_values.size - 1)))
        self.a_index_var.set(restored_a)
        self.b_index_var.set(restored_b)
        self._preferred_initial_metric = None
        self._restore_trials_controls_for_metric(saved_metric)

    def _refresh_slice_controls(self) -> None:
        descriptors = list(self.available_slices)
        labels = [self._slice_label(item) for item in descriptors]
        keys = [str(item.get("key", "")) for item in descriptors]
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

    def _refresh_selector_values(self) -> None:
        self._refresh_slice_controls()
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
            self.artifact_h5 = artifact_path
        if self.artifact_h5 is None:
            self.status_var.set("No artifact loaded. Use Open Artifact to choose a consolidated scan H5 file.")
            self.summary_var.set("No artifact loaded.")
            self.payload = {}
            self.available_slices = []
            self.slice_key_var.set("")
            self.slice_display_var.set("")
            self.a_values = np.asarray([], dtype=float)
            self.b_values = np.asarray([], dtype=float)
            self.refresh_signal_path = None
            self._refresh_signal_mtime_ns = -1
            self._refresh_signal_phase = ""
            self._refresh_signal_pending_points = []
            self._refresh_action_states()
            self._refresh_all()
            return
        self.last_artifact_dir = self.artifact_h5.expanduser().resolve().parent
        self.refresh_signal_path = Path(f"{self.artifact_h5}.refresh")
        if self.refresh_signal_path.exists():
            try:
                self._refresh_signal_mtime_ns = int(self.refresh_signal_path.stat().st_mtime_ns)
                refresh_payload = self._read_refresh_signal_payload()
                self._refresh_signal_phase = str(refresh_payload.get("phase", ""))
                self._refresh_signal_pending_points = list(refresh_payload.get("pending_points", []))
            except Exception:
                self._refresh_signal_mtime_ns = -1
                self._refresh_signal_phase = ""
                self._refresh_signal_pending_points = []
        _save_last_directory(self.last_artifact_dir)
        prev_a = int(self.a_index_var.get())
        prev_b = int(self.b_index_var.get())
        requested_slice_key = self._selected_slice_key()
        try:
            self.payload = self._load_scan_file_with_retries(self.artifact_h5, slice_key=requested_slice_key)
        except Exception as exc:
            self.status_var.set(
                "Artifact is currently being written or is temporarily locked. "
                "Please retry in a moment.\n"
                f"Details: {exc}"
            )
            self.summary_var.set("Could not refresh artifact right now. Existing view remains unchanged.")
            return
        self.root.title(f"pychmp-view: {self.artifact_h5.name}")
        self.available_slices = list(self.payload.get("available_slices", []))
        self.slice_key_var.set(str(self.payload.get("selected_slice_key", "")))
        self.a_values = np.asarray(self.payload["a_values"], dtype=float)
        self.b_values = np.asarray(self.payload["b_values"], dtype=float)
        self.display_model = build_patch_grid_model(self.payload)
        self.run_target_metric = str(self.payload.get("target_metric", "chi2"))
        self._restore_slice_view_state()
        self._refresh_selector_values()
        self._refresh_action_states()
        self._refresh_all()

    def _load_scan_file_with_retries(self, artifact_h5: Path, *, slice_key: str | None = None) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(1, self._LOAD_RETRY_ATTEMPTS + 1):
            try:
                return load_scan_file(artifact_h5, slice_key=slice_key)
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
            if self.refresh_signal_path is None:
                return
            if self.refresh_signal_path.exists():
                mtime_ns = int(self.refresh_signal_path.stat().st_mtime_ns)
                if mtime_ns > int(self._refresh_signal_mtime_ns):
                    self._refresh_signal_mtime_ns = mtime_ns
                    refresh_payload = self._read_refresh_signal_payload()
                    self._refresh_signal_phase = str(refresh_payload.get("phase", ""))
                    self._refresh_signal_pending_points = list(refresh_payload.get("pending_points", []))
                    self._reload_payload()
        except Exception:
            pass
        finally:
            if not self._is_closing:
                try:
                    self._external_refresh_after_id = self.root.after(self._EXTERNAL_REFRESH_POLL_MS, self._poll_external_refresh_signal)
                except Exception:
                    self._external_refresh_after_id = None

    def _selected_point(self) -> dict[str, Any]:
        return self.payload["points"][(int(self.a_index_var.get()), int(self.b_index_var.get()))]

    def _has_selected_point(self) -> bool:
        if not self.payload:
            return False
        if self.a_values.size == 0 or self.b_values.size == 0:
            return False
        points = self.payload.get("points", {})
        if not points:
            return False
        key = (int(self.a_index_var.get()), int(self.b_index_var.get()))
        return key in points

    def _selected_diagnostics(self) -> dict[str, Any]:
        diagnostics = dict(self.payload["diagnostics"])
        diagnostics.update(self._selected_point()["diagnostics"])
        return diagnostics

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

    def _trial_series_for_point(self, point: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
        q0_trials = np.asarray(point.get("fit_q0_trials", ()), dtype=float)
        selected_metric = str(self.metric_var.get())
        metric_trials = self._metric_history_for_point(point, selected_metric)
        point_metric = selected_metric
        if metric_trials.size != q0_trials.size or metric_trials.size == 0:
            metric_trials = np.asarray(point.get("fit_metric_trials", ()), dtype=float)
            point_metric = str(point.get("target_metric", selected_metric))
        return q0_trials, metric_trials, point_metric

    def _best_trial_index_from_metric(self, metric_trials: np.ndarray) -> int | None:
        finite = np.isfinite(metric_trials)
        if metric_trials.ndim != 1 or metric_trials.size == 0 or not np.any(finite):
            return None
        return int(np.nanargmin(metric_trials))

    def _current_trial_token(self, point_metric: str, q0_trials: np.ndarray) -> tuple[int, int, str, int]:
        return (
            int(self.a_index_var.get()),
            int(self.b_index_var.get()),
            str(point_metric),
            int(q0_trials.size),
        )

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
        best_index = self._best_trial_index_from_metric(metric_trials)
        current_index = int(trial_index_var.get()) if trial_index_var is not None else -1
        if (
            force_best
            or self._selected_trial_token != token
            or current_index < 0
            or current_index >= q0_trials.size
        ):
            current_index = 0 if best_index is None else int(best_index)
            if trial_index_var is not None:
                trial_index_var.set(current_index)
            self._selected_trial_token = token
        return current_index

    def _refresh_trial_selector_controls(
        self,
        point: dict[str, Any],
        q0_trials: np.ndarray,
        metric_trials: np.ndarray,
        point_metric: str,
    ) -> None:
        selected_index = self._selected_trial_index_for_point(point, q0_trials, metric_trials, point_metric)
        enabled = selected_index is not None
        trial_label_var = getattr(self, "trial_label_var", None)
        if enabled:
            best_index = self._best_trial_index_from_metric(metric_trials)
            suffix = " [best]" if best_index is not None and int(selected_index) == int(best_index) else ""
            if trial_label_var is not None:
                trial_label_var.set(
                    f"trial #{int(selected_index) + 1}/{len(q0_trials)}  q0={float(q0_trials[int(selected_index)]):.6g}  "
                    f"{point_metric}={float(metric_trials[int(selected_index)]):.6g}{suffix}"
                )
        else:
            if trial_label_var is not None:
                trial_label_var.set("trial: n/a")
        if self.trial_slider is not None:
            self._updating_trial_slider = True
            try:
                if enabled:
                    self.trial_slider.configure(from_=0.0, to=float(max(0, len(q0_trials) - 1)), state=tk.NORMAL)
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
        if not self._has_selected_point():
            return
        point = self._selected_point()
        q0_trials, metric_trials, point_metric = self._trial_series_for_point(point)
        best_index = self._best_trial_index_from_metric(metric_trials)
        if best_index is None:
            return
        self.trial_index_var.set(int(best_index))
        self._selected_trial_token = self._current_trial_token(point_metric, q0_trials)
        self._refresh_all()

    def _on_trial_slider_changed(self, value: str) -> None:
        if self._updating_trial_slider:
            return
        if not self._has_selected_point():
            return
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

    def _selected_solution_plot_context(self) -> dict[str, Any] | None:
        if not self._has_selected_point():
            return None
        point = self._selected_point()
        diagnostics = self._selected_diagnostics()
        q0_trials, metric_trials, point_metric = self._trial_series_for_point(point)
        selected_trial_index = self._selected_trial_index_for_point(point, q0_trials, metric_trials, point_metric)
        best_trial_index = self._best_trial_index_from_metric(metric_trials)
        diagnostics["fit_q0_trials"] = np.asarray(point["fit_q0_trials"], dtype=float).tolist()
        diagnostics["fit_metric_trials"] = np.asarray(point["fit_metric_trials"], dtype=float).tolist()
        diagnostics["target_metric"] = str(point_metric)
        diagnostics["best_q0_recovered"] = float(point.get("q0", np.nan))
        diagnostics["best_target_metric_value"] = float(diagnostics.get("target_metric_value", np.nan))
        raw_modeled = np.asarray(point["raw_modeled_best"], dtype=float)
        modeled = np.asarray(point["modeled_best"], dtype=float)
        residual = np.asarray(point["residual"], dtype=float)
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
            raw_trial_maps = point.get("trial_raw_modeled_maps")
            modeled_trial_maps = point.get("trial_modeled_maps")
            residual_trial_maps = point.get("trial_residual_maps")
            diagnostics["selected_trial_maps_available"] = False
            if raw_trial_maps is not None and modeled_trial_maps is not None and residual_trial_maps is not None:
                if int(selected_trial_index) < int(np.asarray(raw_trial_maps).shape[0]):
                    raw_modeled = np.asarray(raw_trial_maps[int(selected_trial_index)], dtype=float)
                    modeled = np.asarray(modeled_trial_maps[int(selected_trial_index)], dtype=float)
                    residual = np.asarray(residual_trial_maps[int(selected_trial_index)], dtype=float)
                    diagnostics["selected_trial_maps_available"] = True
        else:
            diagnostics["q0_recovered"] = float(point.get("q0", np.nan))
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
        return {
            "model_path": Path(str(diagnostics.get("model_path", ""))),
            "observed_noisy": np.asarray(self.payload["observed"], dtype=float),
            "raw_modeled_best": raw_modeled,
            "modeled_best": modeled,
            "residual": residual,
            "wcs_header": self.payload["wcs_header"],
            "frequency_ghz": frequency_ghz,
            "diagnostics": diagnostics,
            "slice_label": slice_label,
            "blos_reference": self.payload.get("blos_reference"),
            "trials_xmin": self._parse_axis_limit(self.trials_xmin_var.get()),
            "trials_xmax": self._parse_axis_limit(self.trials_xmax_var.get()),
            "trials_ymin": self._parse_axis_limit(self.trials_ymin_var.get()),
            "trials_ymax": self._parse_axis_limit(self.trials_ymax_var.get()),
            "trials_xscale": str(self.trials_xscale_var.get() or "linear"),
            "trials_yscale": str(self.trials_yscale_var.get() or "linear"),
            "wcs_header_transform": lambda hdr: with_observer_metadata(hdr, self.payload["wcs_header"], diagnostics),
        }

    def _jump_to_best(self) -> None:
        try:
            a_index, b_index = best_grid_index(self.payload, self.metric_var.get())
            self.a_index_var.set(a_index)
            self.b_index_var.set(b_index)
            self._selected_trial_token = None
            self._refresh_selector_values()
        except ValueError:
            # No finite metric values yet (common at scan start).
            pass
        self._refresh_all()

    def _on_metric_changed(self) -> None:
        self._capture_current_slice_view_state(self._last_rendered_metric)
        self._restore_trials_controls_for_metric(str(self.metric_var.get()))
        self._jump_to_best()

    def _on_a_changed(self) -> None:
        current = int(self.a_menu.current())
        if current >= 0:
            self.a_index_var.set(current)
        self._selected_trial_token = None
        self._refresh_all()

    def _on_slice_changed(self) -> None:
        if self.slice_menu is None:
            return
        current = int(self.slice_menu.current())
        if current < 0 or current >= len(self.available_slices):
            return
        selected_key = str(self.available_slices[current].get("key", "")).strip()
        if not selected_key or selected_key == self._selected_slice_key():
            return
        self._capture_current_slice_view_state(self._last_rendered_metric)
        self.slice_key_var.set(selected_key)
        self._reload_payload()

    def _on_b_changed(self) -> None:
        current = int(self.b_menu.current())
        if current >= 0:
            self.b_index_var.set(current)
        self._selected_trial_token = None
        self._refresh_all()

    def _on_trials_canvas_click(self, event: Any) -> None:
        if not self._has_selected_point():
            return
        if event.inaxes is not self.ax_trials or event.xdata is None:
            return
        point = self._selected_point()
        q0_trials, metric_trials, point_metric = self._trial_series_for_point(point)
        if q0_trials.size == 0 or metric_trials.size != q0_trials.size:
            return
        if event.ydata is None:
            selected_index = int(np.argmin(np.abs(q0_trials - float(event.xdata))))
        else:
            x_span = max(float(np.nanmax(q0_trials) - np.nanmin(q0_trials)), 1e-12)
            y_span = max(float(np.nanmax(metric_trials) - np.nanmin(metric_trials)), 1e-12)
            distances = ((q0_trials - float(event.xdata)) / x_span) ** 2 + ((metric_trials - float(event.ydata)) / y_span) ** 2
            selected_index = int(np.nanargmin(distances))
        self.trial_index_var.set(selected_index)
        self._selected_trial_token = self._current_trial_token(point_metric, q0_trials)
        self._refresh_all()

    def _on_canvas_click(self, event: Any) -> None:
        if self.a_values.size == 0 or self.b_values.size == 0:
            return
        if event.inaxes is not self.ax_heatmap or event.xdata is None or event.ydata is None:
            return
        record = find_record_for_point(self.display_model, float(event.xdata), float(event.ydata))
        if record is None:
            return
        a_index = int(record["a_index"])
        b_index = int(record["b_index"])
        self.a_index_var.set(a_index)
        self.b_index_var.set(b_index)
        self._selected_trial_token = None
        self._refresh_selector_values()
        self._refresh_action_states()
        self._refresh_all()

    def _refresh_all(self) -> None:
        if not self.payload:
            self._clear_trial_selector_controls()
            self._refresh_action_states()
            self._refresh_scan_state_display()
            self.ax_heatmap.clear()
            self._reset_heatmap_colorbar_axis()
            self.ax_trials.clear()
            self.ax_heatmap.set_axis_off()
            self.ax_heatmap_cbar.set_axis_off()
            self.ax_trials.set_axis_off()
            self._refresh_info_text()
            self.heatmap_canvas.draw_idle()
            self.trials_canvas.draw_idle()
            if self.selected_solution_window is not None:
                self.selected_solution_window.status_var.set("No artifact loaded.")
            return
        if not self._has_selected_point():
            self._clear_trial_selector_controls()
            self._refresh_scan_state_display()
            self.ax_heatmap.clear()
            self._reset_heatmap_colorbar_axis()
            self.ax_trials.clear()
            self.ax_heatmap_cbar.set_axis_off()
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
        self._refresh_action_states()
        self._refresh_scan_state_display()
        self._draw_heatmap()
        self._draw_trials()
        self._refresh_summary()
        self._refresh_info_text()
        self.heatmap_canvas.draw_idle()
        self.trials_canvas.draw_idle()
        if self.selected_solution_window is not None:
            self.selected_solution_window.update_selection()

    def _draw_heatmap(self) -> None:
        metric_name = self.metric_var.get()
        records = list(self.display_model.get("records", []))
        record_lookup = {(int(record["a_index"]), int(record["b_index"])): record for record in records}
        self.ax_heatmap.clear()
        self._reset_heatmap_colorbar_axis()
        self.ax_heatmap_cbar.set_axis_on()
        patches = [
            Rectangle(
                (float(record["b0"]), float(record["a0"])),
                float(record["b1"]) - float(record["b0"]),
                float(record["a1"]) - float(record["a0"]),
            )
            for record in records
        ]
        values = np.asarray([float(record["metrics"].get(metric_name, np.nan)) for record in records], dtype=float)
        collection = PatchCollection(patches, cmap="viridis", edgecolor="none", linewidth=0.0)
        collection.set_array(values)
        finite = np.isfinite(values)
        if np.any(finite):
            collection.set_clim(float(np.nanmin(values[finite])), float(np.nanmax(values[finite])))
        self.ax_heatmap.add_collection(collection)
        self.ax_heatmap.set_title(f"{metric_name} over (a, b)", fontsize=12)
        self.ax_heatmap.set_xlabel("b")
        self.ax_heatmap.set_ylabel("a")
        self.ax_heatmap.set_xlim(float(self.display_model["b_min"]), float(self.display_model["b_max"]))
        self.ax_heatmap.set_ylim(float(self.display_model["a_min"]), float(self.display_model["a_max"]))
        self.ax_heatmap.set_box_aspect(1.0)
        self._heatmap_colorbar = self.heatmap_figure.colorbar(collection, cax=self.ax_heatmap_cbar)
        self._heatmap_colorbar.set_label(metric_name)

        best_markers = {"chi2": ("#d62728", "o"), "rho2": ("#1f77b4", "s"), "eta2": ("#2ca02c", "^")}
        for name, (color, marker) in best_markers.items():
            try:
                a_index, b_index = best_grid_index(self.payload, name)
            except ValueError:
                # No finite values for this metric yet; skip marker until refresh after new points land.
                continue
            record = record_lookup.get((a_index, b_index))
            b_value = float(record["b_center"]) if record is not None else float(self.b_values[b_index])
            a_value = float(record["a_center"]) if record is not None else float(self.a_values[a_index])
            self.ax_heatmap.scatter(
                [b_value],
                [a_value],
                s=120,
                marker=marker,
                facecolor="none",
                edgecolor="white",
                linewidth=3.2,
                zorder=5,
            )
            self.ax_heatmap.scatter(
                [b_value],
                [a_value],
                s=80,
                marker=marker,
                facecolor="none",
                edgecolor=color,
                linewidth=1.8,
                zorder=6,
            )

        current_a = int(self.a_index_var.get())
        current_b = int(self.b_index_var.get())
        current_record = record_lookup.get((current_a, current_b))
        current_b_value = float(current_record["b_center"]) if current_record is not None else float(self.b_values[current_b])
        current_a_value = float(current_record["a_center"]) if current_record is not None else float(self.a_values[current_a])
        self.ax_heatmap.scatter(
            [current_b_value],
            [current_a_value],
            s=170,
            marker="x",
            color="black",
            linewidth=3.6,
            zorder=7,
        )
        self.ax_heatmap.scatter(
            [current_b_value],
            [current_a_value],
            s=120,
            marker="x",
            color="white",
            linewidth=2.2,
            zorder=8,
        )
        if self._refresh_signal_pending_points:
            pending_b = [float(b_value) for _a_value, b_value in self._refresh_signal_pending_points]
            pending_a = [float(a_value) for a_value, _b_value in self._refresh_signal_pending_points]
            self.ax_heatmap.scatter(
                pending_b,
                pending_a,
                s=180,
                marker="*",
                facecolor="#f08c00",
                edgecolor="black",
                linewidth=0.8,
                zorder=7,
            )

    def _draw_trials(self) -> None:
        point = self._selected_point()
        selected_metric = str(self.metric_var.get())
        q0_trials, metric_trials, point_metric = self._trial_series_for_point(point)
        self.ax_trials.clear()
        self.ax_trials.set_axis_on()
        point_status = str(point.get("status", "computed"))
        if point_status == "pending":
            self._refresh_trial_selector_controls(point, np.asarray([], dtype=float), np.asarray([], dtype=float), point_metric)
            self.ax_trials.text(
                0.02,
                0.98,
                "No data yet for this grid point.\nUse Refresh Artifact to reload partial scan progress.",
                transform=self.ax_trials.transAxes,
                va="top",
                ha="left",
            )
            self.ax_trials.set_axis_off()
        elif q0_trials.size > 0 and metric_trials.size == q0_trials.size:
            selected_trial_index = self._selected_trial_index_for_point(point, q0_trials, metric_trials, point_metric)
            self._refresh_trial_selector_controls(point, q0_trials, metric_trials, point_metric)
            order = np.argsort(q0_trials)
            self.ax_trials.plot(q0_trials[order], metric_trials[order], "-o", ms=4, lw=1.2, color="#2b6cb0")
            self.ax_trials.scatter(q0_trials, metric_trials, s=24, color="#2b6cb0", alpha=0.55)
            best_trial_index = int(np.nanargmin(metric_trials))
            if selected_trial_index is not None:
                self.ax_trials.axvline(float(q0_trials[int(selected_trial_index)]), color="#d62728", ls="--", lw=1.6)
                self.ax_trials.scatter(
                    [float(q0_trials[int(selected_trial_index)])],
                    [float(metric_trials[int(selected_trial_index)])],
                    color="#d62728",
                    s=42,
                    zorder=5,
                )
            self.ax_trials.scatter(
                [float(q0_trials[best_trial_index])],
                [float(metric_trials[best_trial_index])],
                facecolor="none",
                edgecolor="#f08c00",
                linewidth=1.8,
                s=74,
                zorder=4,
            )
            self.ax_trials.set_title(f"{point_metric} vs q0", fontsize=12)
            self.ax_trials.set_xlabel("q0")
            self.ax_trials.set_ylabel(point_metric)
            self.ax_trials.grid(alpha=0.25)
            self._apply_trials_axis_controls(q0_trials=q0_trials, metric_trials=metric_trials)
            self._sync_trials_axis_controls_from_axes()
            self._last_rendered_metric = selected_metric if selected_metric in METRICS else point_metric
            self._capture_current_slice_view_state(self._last_rendered_metric)
        else:
            self._refresh_trial_selector_controls(point, np.asarray([], dtype=float), np.asarray([], dtype=float), point_metric)
            self.ax_trials.text(
                0.02,
                0.98,
                "Trial history unavailable",
                transform=self.ax_trials.transAxes,
                va="top",
                ha="left",
            )
            self.ax_trials.set_axis_off()
            self.trials_xscale_var.set("linear")
            self.trials_yscale_var.set("linear")
            self._last_rendered_metric = self.metric_var.get() if self.metric_var.get() in METRICS else point_metric
            self._capture_current_slice_view_state(self._last_rendered_metric)
        self.ax_trials.set_box_aspect(1.0)
        self.trials_figure.subplots_adjust(left=0.14, right=0.98, bottom=0.14, top=0.92)

        success_text = point_status if point_status != "computed" else ("success" if bool(point["success"]) else "boundary/failed")
        best_metric_value = float(np.nanmin(metric_trials)) if metric_trials.size else float("nan")
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
        slice_descriptor = dict(self.payload.get("selected_slice") or {})
        slice_text = self._slice_label(slice_descriptor) if slice_descriptor else "single slice"
        self.status_var.set(
            f"Slice: {slice_text}\n"
            f"Displayed grid metric: {self.metric_var.get()}\n"
            f"Displayed q0-curve metric: {point_metric}\n"
            f"Run target metric: {self.run_target_metric}\n"
            f"Selected point: a={self.a_values[int(self.a_index_var.get())]:.3f}, "
            f"b={self.b_values[int(self.b_index_var.get())]:.3f}\n"
            f"Selected trial: {selected_trial_text}\n"
            f"Best q0: {_format_scalar(point.get('q0', np.nan), '.6f')}\n"
            f"Selected {point_metric}: {selected_metric_text}\n"
            f"Best {point_metric}: {best_metric_value:.6e}\n"
            f"Status: {success_text}"
        )

    def _refresh_summary(self) -> None:
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
        slice_descriptor = dict(self.payload.get("selected_slice") or {})
        lines = [
            f"slice = {self._slice_label(slice_descriptor) if slice_descriptor else 'single slice'}",
            f"a = {self.a_values[int(self.a_index_var.get())]:.3f}",
            f"b = {self.b_values[int(self.b_index_var.get())]:.3f}",
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
        if not self._has_selected_point():
            self.status_var.set("No completed point available yet; refresh after the scan advances.")
            return
        if self.selected_solution_window is None:
            self.selected_solution_window = _SelectedSolutionWindow(self)
        self.selected_solution_window.present()
        self.selected_solution_window.update_selection()

    def _open_grid_summary(self) -> None:
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
    PychmpViewApp(root, args.artifact_h5, initial_metric=str(args.metric))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
