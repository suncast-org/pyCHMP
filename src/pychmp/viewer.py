"""Interactive viewer for consolidated `(a, b)` scan artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import textwrap
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from types import SimpleNamespace
import numpy as np
from astropy.io import fits
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import fftconvolve

from .ab_scan_artifacts import (
    METRICS,
    best_grid_index,
    build_patch_grid_model,
    default_point_index,
    find_record_for_point,
    load_active_point_snapshot,
    load_live_trial_point,
    load_selected_trial_plot_payload,
    load_scan_file,
    resolve_point_index,
    with_observer_metadata,
)
from .metrics import resolve_threshold_mask
from .psf import KernelConvolvedRenderer, PSFMetadata, build_psf_kernel, default_psf_metadata
from .q0_artifact_panel import Q0ArtifactPanelFigure
from .gxrender_adapter import GXRenderEUVAdapter, GXRenderMWAdapter


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
        _ToolTip(self.mask_contour_check, "Toggle the displayed ROI mask contour overlay on the observed, modeled, and residual panels.")
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
        self.last_artifact_dir = _load_last_directory()
        self.refresh_signal_path: Path | None = None
        self._refresh_signal_mtime_ns = -1
        self._refresh_signal_phase = ""
        self._refresh_signal_slice_key: str | None = None
        self._refresh_signal_pending_points: list[tuple[float, float]] = []
        self._refresh_signal_active_point: tuple[float, float] | None = None
        self._refresh_signal_live_trials: dict[str, Any] | None = None
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
        self.active_point_button: ttk.Button | None = None
        self.scan_state_badge: tk.Label | None = None
        self.scan_state_label: ttk.Label | None = None
        self.scan_state_info_label: ttk.Label | None = None
        self.left_notebook: ttk.Notebook | None = None
        self.center_notebook: ttk.Notebook | None = None
        self.info_notebook: ttk.Notebook | None = None
        self.square_display_side = 0
        self.selected_solution_window: _SelectedSolutionWindow | None = None
        self._selected_trial_token: tuple[int, int, str, int] | None = None
        self._updating_trial_slider = False
        self._is_closing = False

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
        active_point_button = ttk.Button(primary_toolbar, text="Active", command=self._jump_to_active_point)
        active_point_button.grid(row=0, column=3, sticky="w", padx=(0, 10))
        self.active_point_button = active_point_button

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
        _ToolTip(active_point_button, "Active Point: jump to the currently running (a, b) point for the selected slice/search when a live refresh signal is present.")

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
        self._apply_fixed_body_geometry()

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
        self._apply_fixed_body_geometry()

    def _apply_fixed_body_geometry(self) -> None:
        square_side = int(self.square_display_side)
        canvas_side = max(180, square_side - 2 * self._DISPLAY_PAD)

        if self.left_notebook is not None:
            self.left_notebook.configure(width=self.left_panel_width)
        if self.center_notebook is not None:
            self.center_notebook.configure(width=self.center_panel_width)
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

    def _refresh_action_states(self) -> None:
        has_artifact = self.artifact_h5 is not None
        has_selected_point = has_artifact and self._has_selected_point()
        has_live_point = has_artifact and self._refresh_signal_active_point is not None
        live_search_id = str((self._refresh_signal_live_trials or {}).get("search_id", "") or "").strip()
        selected_search_id = str(self.payload.get("selected_search_id", "") or "").strip()
        has_active_point = has_live_point and (
            self._active_point_indices() is not None
            or (bool(live_search_id) and live_search_id != selected_search_id)
        )
        if self.open_artifact_button is not None:
            self.open_artifact_button.state(["!disabled"])
        if self.display_selected_button is not None:
            self.display_selected_button.state(["!disabled"] if has_selected_point else ["disabled"])
        if self.summary_button is not None:
            self.summary_button.state(["!disabled"] if has_artifact else ["disabled"])
        if self.refresh_button is not None:
            self.refresh_button.state(["!disabled"] if has_artifact else ["disabled"])
        if self.active_point_button is not None:
            self.active_point_button.state(["!disabled"] if has_active_point else ["disabled"])

    def _scan_state_snapshot(self) -> tuple[str, str, str, str, str]:
        if self.artifact_h5 is not None and bool(getattr(self, "_initial_reload_in_progress", False)) and not self.payload:
            return "LOADING", "Opening artifact", "Artifact load in progress", "#1d6fd6", "white"
        if self.artifact_h5 is None or not self.payload:
            return "NO ARTIFACT", "No artifact", "No artifact loaded", "#6c757d", "white"

        points = self.payload.get("points", {})
        diagnostics = dict(self.payload.get("diagnostics") or {})
        total = int(len(points))
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
            heartbeat_live = bool(getattr(self, "_refresh_signal_live_trials", None)) and heartbeat_matches_selected
            heartbeat_pending = bool(getattr(self, "_refresh_signal_pending_points", None)) and heartbeat_matches_selected
            heartbeat_active_point = bool(getattr(self, "_refresh_signal_active_point", None)) and heartbeat_matches_selected
            heartbeat_running = heartbeat_live or heartbeat_pending or heartbeat_active_point

            badge = "RUNNING" if (heartbeat_running or (adaptive_point_run and refresh_active and heartbeat_matches_selected)) else "EMPTY"
            color = "#0b7285" if badge == "RUNNING" else "#6c757d"
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
        live_state = dict(getattr(self, "_refresh_signal_live_trials", None) or {})
        heartbeat_search_id = str(live_state.get("search_id", "") or "").strip()
        heartbeat_matches_selected_search = not (
            heartbeat_search_id and selected_search_id and heartbeat_search_id != selected_search_id
        )
        scoped_refresh_active = refresh_active and heartbeat_matches_selected
        scoped_phase_complete = phase_complete and heartbeat_matches_selected
        scoped_live_state_active = bool(live_state) and heartbeat_matches_selected and heartbeat_matches_selected_search
        scoped_pending_points = bool(getattr(self, "_refresh_signal_pending_points", None)) and heartbeat_matches_selected
        scoped_active_point = bool(getattr(self, "_refresh_signal_active_point", None)) and heartbeat_matches_selected
        scoped_heartbeat_running = scoped_live_state_active or scoped_pending_points or scoped_active_point
        search_completed = (search_status == "complete") or bool(completed_at)

        if scoped_phase_complete:
            badge = "FINISHED"
            color = "#2b8a3e"
        elif search_completed:
            badge = "FINISHED"
            color = "#2b8a3e"
        elif scoped_heartbeat_running:
            badge = "RUNNING"
            color = "#0b7285"
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
            info_lines.append(f"Search status: {search_status}")
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
        if heartbeat_slice_key:
            info_lines.append(f"Heartbeat slice key: {heartbeat_slice_key}")
        pending_points = list(getattr(self, "_refresh_signal_pending_points", []) or [])
        if pending_points and heartbeat_matches_selected and heartbeat_matches_selected_search:
            active_points = ", ".join(
                f"(a={a_value:.3f}, b={b_value:.3f})"
                for a_value, b_value in pending_points[:4]
            )
            if len(pending_points) > 4:
                active_points += f", ... (+{len(pending_points) - 4} more)"
            info_lines.append(f"Active point(s): {active_points}")
        elif pending_points:
            info_lines.append("Active point(s): running on a different slice/search")
        active_point = getattr(self, "_refresh_signal_active_point", None)
        if active_point is not None:
            try:
                active_a = float(active_point[0])
                active_b = float(active_point[1])
            except Exception:
                pass
            else:
                if (
                    np.isfinite(active_a)
                    and np.isfinite(active_b)
                    and heartbeat_matches_selected
                    and heartbeat_matches_selected_search
                ):
                    info_lines.append(f"Heartbeat active point: (a={active_a:.3f}, b={active_b:.3f})")
                elif np.isfinite(active_a) and np.isfinite(active_b):
                    info_lines.append("Heartbeat active point: running on a different slice/search")
        live_trials = live_state
        active_trial_q0 = live_trials.get("active_trial_q0")
        if active_trial_q0 is not None and heartbeat_matches_selected and heartbeat_matches_selected_search:
            try:
                info_lines.append(f"Live trial q0: {float(active_trial_q0):.6e}")
            except Exception:
                pass
        return badge, toolbar_detail, "\n".join(info_lines), color, "white"

    def _read_refresh_signal_payload(self) -> dict[str, Any]:
        if self.refresh_signal_path is None or not self.refresh_signal_path.exists():
            return {"phase": "", "slice_key": None, "pending_points": [], "active_point": None, "live_trials": None}
        try:
            text = self.refresh_signal_path.read_text(encoding="utf-8").strip()
        except Exception:
            return {"phase": "", "slice_key": None, "pending_points": [], "active_point": None, "live_trials": None}
        if not text:
            return {"phase": "", "slice_key": None, "pending_points": [], "active_point": None, "live_trials": None}
        if text.startswith("{"):
            try:
                payload = json.loads(text)
                phase = str(payload.get("phase", "")).strip()
                payload_slice_key = str(payload.get("slice_key", "")).strip() or None
                pending_points: list[tuple[float, float]] = []
                for item in payload.get("pending_points", []) or []:
                    try:
                        a_value = float(item.get("a"))
                        b_value = float(item.get("b"))
                    except Exception:
                        continue
                    if np.isfinite(a_value) and np.isfinite(b_value):
                        pending_points.append((a_value, b_value))
                active_point = payload.get("active_point")
                active_coords = None
                if isinstance(active_point, dict):
                    try:
                        active_a = float(active_point.get("a"))
                        active_b = float(active_point.get("b"))
                    except Exception:
                        active_coords = None
                    else:
                        if np.isfinite(active_a) and np.isfinite(active_b):
                            active_coords = (active_a, active_b)
                live_trials = None
                live_payload = payload.get("live_trials")
                if isinstance(live_payload, dict):
                    default_a = active_coords[0] if active_coords is not None else np.nan
                    default_b = active_coords[1] if active_coords is not None else np.nan
                    active_trial_index = live_payload.get("active_trial_index")
                    try:
                        parsed_active_trial_index = None if active_trial_index is None else int(active_trial_index)
                    except Exception:
                        parsed_active_trial_index = None
                    active_trial_q0 = live_payload.get("active_trial_q0")
                    try:
                        parsed_active_trial_q0 = None if active_trial_q0 is None else float(active_trial_q0)
                    except Exception:
                        parsed_active_trial_q0 = None
                    if parsed_active_trial_q0 is not None and not np.isfinite(parsed_active_trial_q0):
                        parsed_active_trial_q0 = None
                    trial_count = live_payload.get("trial_count")
                    try:
                        parsed_trial_count = None if trial_count is None else int(trial_count)
                    except Exception:
                        parsed_trial_count = None
                    live_trials = {
                        "a": float(live_payload.get("a", default_a)),
                        "b": float(live_payload.get("b", default_b)),
                        "metric_name": str(live_payload.get("metric_name", self.run_target_metric or "chi2")),
                        "slice_key": payload_slice_key,
                        "trial_count": parsed_trial_count,
                        "active_trial_index": parsed_active_trial_index,
                        "active_trial_q0": parsed_active_trial_q0,
                    }
                return {
                    "phase": phase,
                    "slice_key": payload_slice_key,
                    "pending_points": pending_points,
                    "active_point": active_coords,
                    "live_trials": live_trials,
                }
            except Exception:
                return {"phase": "", "slice_key": None, "pending_points": [], "active_point": None, "live_trials": None}
        parts = text.split(maxsplit=1)
        phase = parts[1].strip() if len(parts) == 2 else parts[0].strip()
        return {"phase": phase, "slice_key": None, "pending_points": [], "active_point": None, "live_trials": None}

    def _point_indices_for_coordinates(self, a_value: float, b_value: float) -> tuple[int, int] | None:
        if self.a_values.size == 0 or self.b_values.size == 0:
            return None
        a_matches = np.where(np.isclose(self.a_values, float(a_value), rtol=0.0, atol=1e-12))[0]
        b_matches = np.where(np.isclose(self.b_values, float(b_value), rtol=0.0, atol=1e-12))[0]
        if a_matches.size == 0 or b_matches.size == 0:
            return None
        return int(a_matches[0]), int(b_matches[0])

    def _apply_active_point_selection(self) -> None:
        if self._refresh_signal_active_point is None:
            return
        if self._has_selected_point():
            return
        resolved = self._active_point_indices()
        if resolved is None:
            return
        a_index, b_index = resolved
        if int(self.a_index_var.get()) != a_index or int(self.b_index_var.get()) != b_index:
            self.a_index_var.set(a_index)
            self.b_index_var.set(b_index)
            self._selected_trial_token = None

    def _active_point_indices(self) -> tuple[int, int] | None:
        if self._refresh_signal_active_point is None:
            return None
        if not self.payload:
            return None
        active_a, active_b = self._refresh_signal_active_point
        exact = self._point_indices_for_coordinates(active_a, active_b)
        if exact is not None:
            # For Active navigation, prefer the exact live grid cell when it is
            # representable on the current grid, even if that cell has not been
            # persisted yet.
            return exact
        # Translate live (a, b) to the nearest grid index, then resolve to
        # the nearest actual *computed* point (handles both the case where the
        # live point is outside the current grid and the case where it sits on
        # an uncomputed placeholder position).
        a_values = self.a_values
        b_values = self.b_values
        if a_values.size == 0 or b_values.size == 0:
            return None
        nearest_a_idx = int(np.argmin(np.abs(a_values - float(active_a))))
        nearest_b_idx = int(np.argmin(np.abs(b_values - float(active_b))))
        try:
            return resolve_point_index(
                self.payload,
                metric=str(self.metric_var.get()),
                a_index=nearest_a_idx,
                b_index=nearest_b_idx,
            )
        except Exception:
            return None

    def _live_trial_state(self) -> dict[str, Any] | None:
        live_trials = dict(getattr(self, "_refresh_signal_live_trials", None) or {})
        active_point = getattr(self, "_refresh_signal_active_point", None)
        if not live_trials or active_point is None:
            return None
        try:
            active_a = float(active_point[0])
            active_b = float(active_point[1])
        except Exception:
            return None
        if not (np.isfinite(active_a) and np.isfinite(active_b)):
            return None
        live_trials["active_a"] = active_a
        live_trials["active_b"] = active_b
        resolved = self._point_indices_for_coordinates(active_a, active_b)
        if resolved is not None:
            live_trials["a_index"] = int(resolved[0])
            live_trials["b_index"] = int(resolved[1])
        return live_trials

    def _sync_live_trial_state_from_artifact(self) -> None:
        self._refresh_signal_slice_key = None
        self._refresh_signal_pending_points = []
        self._refresh_signal_active_point = None
        self._refresh_signal_live_trials = None
        artifact_h5 = getattr(self, "artifact_h5", None)
        if artifact_h5 is None:
            return
        slice_key = str(self.payload.get("selected_slice_key", "") or "").strip() or None
        search_id = str(self.payload.get("selected_search_id", "") or "").strip() or None
        if slice_key is None:
            return
        try:
            live_state = load_live_trial_point(Path(artifact_h5), slice_key=slice_key, search_id=search_id)
        except Exception:
            live_state = None
        if not isinstance(live_state, dict):
            newest_state: dict[str, Any] | None = None
            newest_updated_utc = ""
            for record in list(self.available_searches or []):
                candidate_search_id = str(record.get("search_id", "") or "").strip()
                if not candidate_search_id:
                    continue
                try:
                    candidate_state = load_live_trial_point(
                        Path(artifact_h5),
                        slice_key=slice_key,
                        search_id=candidate_search_id,
                    )
                except Exception:
                    candidate_state = None
                if not isinstance(candidate_state, dict):
                    continue
                updated_utc = str(candidate_state.get("updated_utc", "") or "")
                if newest_state is None or updated_utc > newest_updated_utc:
                    newest_state = candidate_state
                    newest_updated_utc = updated_utc
            live_state = newest_state
        if not isinstance(live_state, dict):
            return
        self._refresh_signal_slice_key = str(live_state.get("slice_key", "") or slice_key).strip() or slice_key
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
        if self._refresh_signal_active_point is not None and active_trial_q0 is not None:
            self._refresh_signal_pending_points = [self._refresh_signal_active_point]
        self._refresh_signal_live_trials = {
            "a": None if self._refresh_signal_active_point is None else float(self._refresh_signal_active_point[0]),
            "b": None if self._refresh_signal_active_point is None else float(self._refresh_signal_active_point[1]),
            "metric_name": str(live_state.get("metric_name", self.run_target_metric or "chi2")),
            "slice_key": self._refresh_signal_slice_key,
            "search_id": str(live_state.get("search_id", "") or "").strip() or None,
            "active_trial_index": None if active_trial_index is None else int(active_trial_index),
            "active_trial_q0": None if active_trial_q0 is None else float(active_trial_q0),
            "fit_q0_trials": fit_q0_trials,
            "fit_metric_trials": fit_metric_trials,
        }

    def _should_use_live_trials(self, live_state: dict[str, Any] | None) -> bool:
        if live_state is None:
            return False
        if not self._live_slice_matches_selected(live_state):
            return False
        if not self._live_search_matches_selected(live_state):
            return False
        if not self._has_selected_point():
            if "a_index" not in live_state or "b_index" not in live_state:
                return False
            return (
                int(self.a_index_var.get()),
                int(self.b_index_var.get()),
            ) == (
                int(live_state["a_index"]),
                int(live_state["b_index"]),
            )
        if "a_index" not in live_state or "b_index" not in live_state:
            # Active point is not yet stored in the artifact grid; do not hijack
            # an explicit user selection.
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
        live_slice_key = str(live_state.get("slice_key", "")).strip()
        selected_slice_key = str(self._selected_slice_key() or "").strip()
        return not (live_slice_key and selected_slice_key and live_slice_key != selected_slice_key)

    def _live_search_matches_selected(self, live_state: dict[str, Any] | None) -> bool:
        if live_state is None:
            return False
        live_search_id = str(live_state.get("search_id", "")).strip()
        selected_search_id = str(self._selected_search_id() or "").strip()
        return not (live_search_id and selected_search_id and live_search_id != selected_search_id)

    def _should_force_live_trials(self, live_state: dict[str, Any] | None) -> bool:
        if live_state is None:
            return False
        if not self._live_slice_matches_selected(live_state):
            return False
        if not self._live_search_matches_selected(live_state):
            return False
        # When the active point is not yet saved into the sparse artifact grid,
        # showing the fallback stored point is misleading. Follow the live point.
        return "a_index" not in live_state or "b_index" not in live_state

    def _heatmap_plot_limits(self) -> tuple[float, float, float, float]:
        b_min = float(self.display_model["b_min"])
        b_max = float(self.display_model["b_max"])
        a_min = float(self.display_model["a_min"])
        a_max = float(self.display_model["a_max"])

        selected_slice_key = str(self._selected_slice_key() or "").strip()
        heartbeat_slice_key = str(getattr(self, "_refresh_signal_slice_key", None) or "").strip()
        if heartbeat_slice_key and selected_slice_key and heartbeat_slice_key != selected_slice_key:
            return b_min, b_max, a_min, a_max

        live_a: list[float] = []
        live_b: list[float] = []
        for a_value, b_value in list(getattr(self, "_refresh_signal_pending_points", []) or []):
            try:
                parsed_a = float(a_value)
                parsed_b = float(b_value)
            except Exception:
                continue
            if np.isfinite(parsed_a) and np.isfinite(parsed_b):
                live_a.append(parsed_a)
                live_b.append(parsed_b)
        active_point = getattr(self, "_refresh_signal_active_point", None)
        if active_point is not None:
            try:
                active_a = float(active_point[0])
                active_b = float(active_point[1])
            except Exception:
                active_a = np.nan
                active_b = np.nan
            if np.isfinite(active_a) and np.isfinite(active_b):
                live_a.append(active_a)
                live_b.append(active_b)
        if not live_a or not live_b:
            return b_min, b_max, a_min, a_max

        a_span = max(a_max - a_min, max(live_a) - min(live_a), 1e-6)
        b_span = max(b_max - b_min, max(live_b) - min(live_b), 1e-6)
        a_pad = max(0.05 * a_span, 1e-6)
        b_pad = max(0.05 * b_span, 1e-6)
        return (
            min(b_min, min(live_b) - b_pad),
            max(b_max, max(live_b) + b_pad),
            min(a_min, min(live_a) - a_pad),
            max(a_max, max(live_a) + a_pad),
        )

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
        value = str(self.search_id_var.get()).strip()
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

    def _resolved_point_index(self, *, metric_name: str | None = None) -> tuple[int, int] | None:
        if not self.payload:
            return None
        a_index = int(self.a_index_var.get())
        b_index = int(self.b_index_var.get())
        points = dict(self.payload.get("points", {}))
        key = (a_index, b_index)
        if key in points:
            status = str(points[key].get("status", "computed")).strip().lower()
            if status not in {"missing", "pending"}:
                return key
            # Preserve explicit selection on unsaved cells instead of silently
            # remapping to a different computed point.
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
            self.a_index_var.set(default_a)
            self.b_index_var.set(default_b)
            self._reset_trials_controls_only()
            self._preferred_initial_metric = None
            return

        saved_metric = str(state.get("metric", metric_name))
        if saved_metric not in METRICS:
            saved_metric = metric_name if metric_name in METRICS else "chi2"
        self.metric_var.set(saved_metric)

        try:
            restored_a, restored_b = resolve_point_index(
                self.payload,
                metric=saved_metric,
                a_index=int(state.get("a_index", default_a)),
                b_index=int(state.get("b_index", default_b)),
            )
        except Exception:
            restored_a, restored_b = default_a, default_b
        self.a_index_var.set(restored_a)
        self.b_index_var.set(restored_b)
        self._preferred_initial_metric = None
        self._restore_trials_controls_for_metric(saved_metric)

    def _refresh_slice_controls(self) -> None:
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
                self.artifact_h5 = artifact_path.expanduser().resolve()
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
            except Exception:
                self._refresh_signal_mtime_ns = -1
                self._refresh_signal_phase = ""
                self._refresh_signal_slice_key = None
                self._refresh_signal_pending_points = []
                self._refresh_signal_active_point = None
                self._refresh_signal_live_trials = None
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
        self.available_slices = list(self.payload.get("available_slices", []))
        self.available_searches = list(self.payload.get("search_records", []))
        self.slice_key_var.set(str(self.payload.get("selected_slice_key", "")))
        self.search_id_var.set(str(self.payload.get("selected_search_id") or ""))
        self.a_values = np.asarray(self.payload["a_values"], dtype=float)
        self.b_values = np.asarray(self.payload["b_values"], dtype=float)
        self.display_model = build_patch_grid_model(self.payload)
        self.run_target_metric = str(self.payload.get("target_metric", "chi2"))
        self._restore_slice_view_state()
        self._sync_live_trial_state_from_artifact()
        self._apply_active_point_selection()
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
                    self._reload_payload()
        except Exception:
            pass
        finally:
            if not self._is_closing:
                try:
                    self._external_refresh_after_id = self.root.after(self._EXTERNAL_REFRESH_POLL_MS, self._poll_external_refresh_signal)
                except Exception:
                    self._external_refresh_after_id = None

    def _heartbeat_requires_payload_reload(self, refresh_payload: dict[str, Any]) -> bool:
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

    def _selected_point(self) -> dict[str, Any]:
        resolved = self._resolved_point_index(metric_name=str(self.metric_var.get()))
        if resolved is None:
            raise KeyError("no stored point is available")
        return self.payload["points"][resolved]

    def _has_selected_point(self) -> bool:
        return self._resolved_point_index(metric_name=str(self.metric_var.get())) is not None

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
        a_index_var = getattr(self, "a_index_var", None)
        b_index_var = getattr(self, "b_index_var", None)
        return (
            int(a_index_var.get()) if a_index_var is not None else 0,
            int(b_index_var.get()) if b_index_var is not None else 0,
            str(point_metric),
            int(q0_trials.size),
        )

    def _trial_token_matches_context(self, token: Any, *, a_token: Any, b_token: Any, point_metric: str) -> bool:
        if not isinstance(token, tuple) or len(token) < 3:
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
            and str(token[2]) == str(point_metric)
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
            current_index = 0 if best_index is None else int(best_index)
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
        if not self._has_selected_point() or self._should_force_live_trials(live_state) or self._should_use_live_trials(live_state):
            live_context = self._live_selected_solution_plot_context()
            if live_context is not None:
                return live_context
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
        return {
            "model_path": Path(str(diagnostics.get("model_path", ""))),
            "observed_noisy": np.asarray(self.payload.get("observed"), dtype=float),
            "raw_modeled_best": np.asarray(raw_modeled, dtype=float),
            "modeled_best": np.asarray(modeled, dtype=float),
            "residual": np.asarray(residual, dtype=float),
            "wcs_header": self.payload.get("wcs_header"),
            "frequency_ghz": frequency_ghz,
            "diagnostics": diagnostics,
            "psf_kernel": self.payload.get("psf_kernel"),
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
        artifact_h5 = getattr(self, "artifact_h5", None)
        if artifact_h5 is None:
            return None
        slice_key = str(self.payload.get("selected_slice_key", "")).strip() or None
        search_id = str(self.payload.get("selected_search_id", "")).strip() or None
        if slice_key is None or search_id is None:
            return None
        try:
            active_a = float(live_state.get("active_a", live_state.get("a", np.nan)))
        except Exception:
            active_a = float("nan")
        try:
            active_b = float(live_state.get("active_b", live_state.get("b", np.nan)))
        except Exception:
            active_b = float("nan")
        if not (np.isfinite(active_a) and np.isfinite(active_b)):
            return None
        try:
            artifact_mtime_ns = int(Path(artifact_h5).stat().st_mtime_ns)
        except Exception:
            artifact_mtime_ns = -1
        cache_key = (
            str(Path(artifact_h5)),
            str(slice_key),
            str(search_id),
            float(active_a),
            float(active_b),
            int(artifact_mtime_ns),
        )
        if cache_key == self._live_snapshot_cache_key:
            return self._live_snapshot_cache_value
        try:
            snapshot = load_active_point_snapshot(
                Path(artifact_h5),
                slice_key=slice_key,
                search_id=search_id,
                include_maps=False,
            )
        except Exception:
            self._live_snapshot_cache_key = cache_key
            self._live_snapshot_cache_value = None
            return None
        self._live_snapshot_cache_key = cache_key
        self._live_snapshot_cache_value = snapshot
        return snapshot

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

    def _live_trial_series_from_state(
        self,
        live_state: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, str, dict[str, Any] | None]:
        point_metric = str(live_state.get("metric_name", self.run_target_metric or self.metric_var.get()))
        q0_trials = np.asarray(live_state.get("fit_q0_trials", ()), dtype=float)
        metric_trials = np.asarray(live_state.get("fit_metric_trials", ()), dtype=float)
        if q0_trials.ndim == 1 and q0_trials.size > 0 and metric_trials.size == q0_trials.size:
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
                    metric_trials = np.asarray(snapshot.get("fit_metric_trials", ()), dtype=float)
                    if metric_trials.size != q0_trials.size:
                        metric_trials = np.asarray([], dtype=float)
                if q0_trials.ndim == 1 and q0_trials.size > 0 and metric_trials.size == q0_trials.size:
                    return q0_trials, metric_trials, point_metric, snapshot
        return np.asarray([], dtype=float), np.asarray([], dtype=float), point_metric, snapshot

    def _live_selected_solution_plot_context(self) -> dict[str, Any] | None:
        live_state = self._live_trial_state()
        if not self._should_use_live_trials(live_state):
            return None
        q0_trials, metric_trials, point_metric, snapshot = self._live_trial_series_from_state(live_state)
        if q0_trials.ndim != 1 or q0_trials.size == 0 or metric_trials.size != q0_trials.size:
            return None

        current_index = int(np.clip(int(self.trial_index_var.get()), 0, max(0, int(q0_trials.size) - 1)))
        selected_q0 = float(q0_trials[current_index])
        selected_metric = float(metric_trials[current_index])
        best_index = self._best_trial_index_from_metric(metric_trials)
        best_q0 = float(q0_trials[int(best_index)]) if best_index is not None else selected_q0

        diagnostics = dict(self.payload.get("diagnostics") or {})
        diagnostics.update(
            {
                "a": float(live_state.get("active_a", np.nan)),
                "b": float(live_state.get("active_b", np.nan)),
                "target_metric": point_metric,
                "target_metric_value": float(selected_metric),
                "fit_q0_trials": [float(v) for v in q0_trials],
                "fit_metric_trials": [float(v) for v in metric_trials],
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
        diagnostics["selected_trial_maps_available"] = False
        loaded_maps = self._load_selected_trial_maps(
            a_value=float(live_state.get("active_a", np.nan)),
            b_value=float(live_state.get("active_b", np.nan)),
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
                    residual = np.asarray(modeled - np.asarray(self.payload.get("observed"), dtype=float), dtype=float)
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
        return {
            "model_path": Path(str(diagnostics.get("model_path", ""))),
            "observed_noisy": np.asarray(self.payload.get("observed"), dtype=float),
            "raw_modeled_best": np.asarray(raw_modeled, dtype=float),
            "modeled_best": np.asarray(modeled, dtype=float),
            "residual": np.asarray(residual, dtype=float),
            "wcs_header": self.payload["wcs_header"],
            "frequency_ghz": frequency_ghz,
            "diagnostics": diagnostics,
            "psf_kernel": self.payload.get("psf_kernel"),
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
            a_index, b_index = default_point_index(self.payload, self.metric_var.get())
            self.a_index_var.set(a_index)
            self.b_index_var.set(b_index)
            self._selected_trial_token = None
            self._refresh_selector_values()
        except ValueError:
            # No finite metric values yet (common at scan start).
            pass
        self._refresh_all()

    def _jump_to_active_point(self) -> None:
        live_search_id = str((self._refresh_signal_live_trials or {}).get("search_id", "") or "").strip()
        selected_search_id = str(self.payload.get("selected_search_id", "") or "").strip()
        if live_search_id and selected_search_id and live_search_id != selected_search_id:
            self.search_id_var.set(live_search_id)
            self._reload_payload()
        resolved = self._active_point_indices()
        if resolved is None:
            return
        a_index, b_index = resolved
        if int(self.a_index_var.get()) != a_index or int(self.b_index_var.get()) != b_index:
            self.a_index_var.set(a_index)
            self.b_index_var.set(b_index)
            self._selected_trial_token = None
            self._refresh_selector_values()
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
        self.search_id_var.set("")
        self._schedule_payload_reload(status_text="Loading selected slice...")

    def _on_search_changed(self) -> None:
        if self.search_menu is None:
            return
        current = int(self.search_menu.current())
        if current < 0 or current >= len(self.available_searches):
            return
        selected_id = str(self.available_searches[current].get("search_id", "")).strip()
        if not selected_id or selected_id == self._selected_search_id():
            return
        self._capture_current_slice_view_state(self._last_rendered_metric)
        self.search_id_var.set(selected_id)
        self._selected_trial_token = None
        self._schedule_payload_reload(status_text="Loading selected search...")

    def _on_b_changed(self) -> None:
        current = int(self.b_menu.current())
        if current >= 0:
            self.b_index_var.set(current)
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
        if self.a_values.size == 0 or self.b_values.size == 0:
            return
        if event.inaxes is not self.ax_heatmap or event.xdata is None or event.ydata is None:
            return
        record = find_record_for_point(self.display_model, float(event.xdata), float(event.ydata))
        if record is not None:
            a_index = int(record["a_index"])
            b_index = int(record["b_index"])
        else:
            resolved = self._active_point_indices_from_heatmap_click(float(event.xdata), float(event.ydata))
            if resolved is None:
                return
            a_index, b_index = resolved
        self.a_index_var.set(a_index)
        self.b_index_var.set(b_index)
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

        x_tol = _click_tolerance(np.asarray(self.b_values, dtype=float), float(self.display_model.get("b_min", active_b)), float(self.display_model.get("b_max", active_b)))
        y_tol = _click_tolerance(np.asarray(self.a_values, dtype=float), float(self.display_model.get("a_min", active_a)), float(self.display_model.get("a_max", active_a)))
        if abs(float(xdata) - active_b) > x_tol or abs(float(ydata) - active_a) > y_tol:
            return None
        return resolved

    def _refresh_all(self, *, update_selected_solution: bool = True) -> None:
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
        live_state = self._live_trial_state()
        if not self._ensure_selected_point_exists(metric_name=str(self.metric_var.get())) and live_state is None:
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
        if live_state is not None and not self._has_selected_point():
            self._refresh_action_states()
            self._refresh_scan_state_display()
            self._draw_heatmap()
            self._draw_trials()
            self.status_var.set(
                "Displayed grid metric: waiting\n"
                f"Run target metric: {self.run_target_metric}\n"
                "Active point has not been saved to the artifact yet."
            )
            self.summary_var.set(
                "Live trials are streaming for the active point.\n"
                "The detailed point summary will populate after the point is saved."
            )
            self._refresh_info_text()
            self.heatmap_canvas.draw_idle()
            self.trials_canvas.draw_idle()
            if self.selected_solution_window is not None:
                self.selected_solution_window.status_var.set("Waiting for active point to be saved.")
            return
        self._refresh_action_states()
        self._refresh_scan_state_display()
        self._draw_heatmap()
        self._draw_trials()
        self._refresh_summary()
        self._refresh_info_text()
        self.heatmap_canvas.draw_idle()
        self.trials_canvas.draw_idle()
        if update_selected_solution and self.selected_solution_window is not None:
            self._schedule_selected_solution_update()

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
        heatmap_b_min, heatmap_b_max, heatmap_a_min, heatmap_a_max = self._heatmap_plot_limits()
        self.ax_heatmap.set_xlim(heatmap_b_min, heatmap_b_max)
        self.ax_heatmap.set_ylim(heatmap_a_min, heatmap_a_max)
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
            if record is not None:
                b_value = float(record["b_center"])
                a_value = float(record["a_center"])
            else:
                if not (0 <= int(a_index) < int(self.a_values.size) and 0 <= int(b_index) < int(self.b_values.size)):
                    continue
                b_value = float(self.b_values[b_index])
                a_value = float(self.a_values[a_index])
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
        current_b_value: float | None = None
        current_a_value: float | None = None
        if current_record is not None:
            current_b_value = float(current_record["b_center"])
            current_a_value = float(current_record["a_center"])
        elif 0 <= int(current_a) < int(self.a_values.size) and 0 <= int(current_b) < int(self.b_values.size):
            current_b_value = float(self.b_values[current_b])
            current_a_value = float(self.a_values[current_a])
        if current_b_value is not None and current_a_value is not None:
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
        active_point = getattr(self, "_refresh_signal_active_point", None)
        if active_point is not None:
            try:
                active_a = float(active_point[0])
                active_b = float(active_point[1])
            except Exception:
                active_a = np.nan
                active_b = np.nan
            if np.isfinite(active_a) and np.isfinite(active_b):
                self.ax_heatmap.scatter(
                    [active_b],
                    [active_a],
                    s=260,
                    marker="*",
                    facecolor="#ffd43b",
                    edgecolor="black",
                    linewidth=1.1,
                    zorder=9,
                )
                self.ax_heatmap.annotate(
                    "active",
                    xy=(active_b, active_a),
                    xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=8,
                    color="#8a5b00",
                    zorder=10,
                )

    def _draw_trials(self) -> None:
        live_state = self._live_trial_state()
        force_live_trials = self._should_force_live_trials(live_state)
        if not force_live_trials and not self._should_use_live_trials(live_state):
            live_state = None
        point = None if (live_state is not None and force_live_trials) else (self._selected_point() if self._has_selected_point() else None)
        selected_metric = str(self.metric_var.get())
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
                self.ax_trials.set_axis_off()
                return
            q0_trials, metric_trials, point_metric = self._trial_series_for_point(point)
            selected_trial_index = None
            point_status = str(point.get("status", "computed"))
        self.ax_trials.clear()
        self.ax_trials.set_axis_on()
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
        elif live_state is not None and q0_trials.size == 0:
            self._refresh_trial_selector_controls(None, np.asarray([], dtype=float), np.asarray([], dtype=float), point_metric)
            self.ax_trials.text(
                0.02,
                0.98,
                (
                    "Waiting for the first completed trial for the active point."
                    if active_trial_q0 is None
                    else f"Rendering active trial at q0={active_trial_q0:.6g}."
                ),
                transform=self.ax_trials.transAxes,
                va="top",
                ha="left",
            )
            self.ax_trials.set_axis_off()
        elif q0_trials.size > 0 and metric_trials.size == q0_trials.size:
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
            if active_trial_q0 is not None:
                active_label = "active trial"
                if active_trial_index is not None:
                    active_label = f"active trial #{int(active_trial_index) + 1}"
                self.ax_trials.axvline(float(active_trial_q0), color="#f08c00", ls=":", lw=1.8, zorder=3)
                self.ax_trials.text(
                    0.02,
                    0.98,
                    f"{active_label} at q0={float(active_trial_q0):.6g}",
                    transform=self.ax_trials.transAxes,
                    va="top",
                    ha="left",
                    color="#b26a00",
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
        slice_text = self._slice_label(slice_descriptor) if slice_descriptor else "single slice"
        self.status_var.set(
            f"Slice: {slice_text}\n"
            f"Displayed grid metric: {self.metric_var.get()}\n"
            f"Displayed q0-curve metric: {point_metric}\n"
            f"Run target metric: {self.run_target_metric}\n"
            f"{selected_subject_label}: a={_format_scalar(selected_a_value, '.3f')}, "
            f"b={_format_scalar(selected_b_value, '.3f')}\n"
            f"Selected trial: {selected_trial_text}\n"
            f"Best q0: {_format_scalar((point.get('q0', np.nan) if point is not None else (q0_trials[int(np.nanargmin(metric_trials))] if metric_trials.size else np.nan)), '.6f')}\n"
            f"Selected {point_metric}: {selected_metric_text}\n"
            f"Best {point_metric}: {best_metric_value:.6e}\n"
            f"Status: {success_text}"
        )

    def _refresh_summary(self) -> None:
        live_state = self._live_trial_state()
        if self._should_force_live_trials(live_state):
            self.summary_var.set(
                "Live trials are streaming for the active point.\n"
                "The detailed point summary will populate after the point is saved."
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
    PychmpViewApp(root, args.artifact_h5, initial_metric=str(args.metric))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
