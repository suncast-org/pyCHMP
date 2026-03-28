"""Interactive viewer for consolidated `(a, b)` scan artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .ab_scan_artifacts import (
    METRICS,
    best_grid_index,
    build_patch_grid_model,
    find_record_for_point,
    load_scan_file,
)


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


class PychmpViewApp:
    def __init__(self, root: tk.Tk, artifact_h5: Path | None, *, initial_metric: str = "chi2") -> None:
        self.root = root
        self.artifact_h5 = artifact_h5
        self.payload = {}
        self.a_values = np.asarray([], dtype=float)
        self.b_values = np.asarray([], dtype=float)
        self.display_model: dict[str, Any] = {"records": []}
        self.run_target_metric = "chi2"
        self.metric_var = tk.StringVar(value=initial_metric if initial_metric in METRICS else "chi2")
        self.a_index_var = tk.IntVar(value=0)
        self.b_index_var = tk.IntVar(value=0)
        self.status_var = tk.StringVar(value="")
        self.summary_var = tk.StringVar(value="")
        self.last_artifact_dir = _load_last_directory()

        self.root.title(f"pychmp-view: {artifact_h5.name}" if artifact_h5 is not None else "pychmp-view")
        self._configure_initial_window_geometry()

        self._build_ui()
        self._refresh_all()
        self.root.after(50, self._present_window)

    def _configure_initial_window_geometry(self) -> None:
        screen_w = max(1, int(self.root.winfo_screenwidth()))
        screen_h = max(1, int(self.root.winfo_screenheight()))

        # Preserve the proportions that work on the user's Mac by expressing the
        # previous fixed geometry as screen-relative ratios instead of pixels.
        width_ratio = 1380.0 / 1440.0
        height_ratio = 760.0 / 900.0
        min_width_ratio = 1240.0 / 1440.0
        min_height_ratio = 700.0 / 900.0

        width = min(screen_w - 24, max(980, int(round(screen_w * width_ratio))))
        height = min(screen_h - 60, max(620, int(round(screen_h * height_ratio))))
        min_width = min(width, max(980, int(round(screen_w * min_width_ratio))))
        min_height = min(height, max(620, int(round(screen_h * min_height_ratio))))

        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.minsize(min_width, min_height)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)

        controls = ttk.Frame(outer)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        ttk.Label(controls, text="Metric").pack(side=tk.LEFT)
        metric_menu = ttk.Combobox(controls, width=10, state="readonly", values=list(METRICS), textvariable=self.metric_var)
        metric_menu.pack(side=tk.LEFT, padx=(6, 12))
        metric_menu.bind("<<ComboboxSelected>>", lambda _event: self._on_metric_changed())

        ttk.Button(controls, text="Jump To Best", command=self._jump_to_best).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(controls, text="a").pack(side=tk.LEFT)
        self.a_menu = ttk.Combobox(controls, width=14, state="readonly")
        self.a_menu.pack(side=tk.LEFT, padx=(6, 12))
        self.a_menu.bind("<<ComboboxSelected>>", lambda _event: self._on_a_changed())

        ttk.Label(controls, text="b").pack(side=tk.LEFT)
        self.b_menu = ttk.Combobox(controls, width=14, state="readonly")
        self.b_menu.pack(side=tk.LEFT, padx=(6, 12))
        self.b_menu.bind("<<ComboboxSelected>>", lambda _event: self._on_b_changed())

        actions = ttk.Frame(outer)
        actions.grid(row=1, column=0, sticky="w", pady=(0, 6))
        ttk.Button(actions, text="Open Artifact", command=self._open_artifact).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(actions, text="Open Selected Maps", command=self._open_selected_maps).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(actions, text="Open Grid Summary", command=self._open_grid_summary).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(actions, text="Refresh Artifact", command=self._reload_payload).pack(side=tk.LEFT)

        content = ttk.Frame(outer)
        content.grid(row=2, column=0, sticky="nsew")
        content.columnconfigure(0, weight=9)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(0, weight=1)

        plot_area = ttk.Frame(content)
        plot_area.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        plot_area.columnconfigure(0, weight=1)
        plot_area.columnconfigure(1, weight=1)
        plot_area.rowconfigure(0, weight=1)

        info_panel = ttk.Frame(content)
        info_panel.grid(row=0, column=1, sticky="nsew")
        info_panel.columnconfigure(0, weight=1)
        info_panel.rowconfigure(0, weight=1)

        info_canvas = tk.Canvas(info_panel, highlightthickness=0)
        info_canvas.grid(row=0, column=0, sticky="nsew")
        info_scroll_y = ttk.Scrollbar(info_panel, orient=tk.VERTICAL, command=info_canvas.yview)
        info_scroll_y.grid(row=0, column=1, sticky="ns")
        info_scroll_x = ttk.Scrollbar(info_panel, orient=tk.HORIZONTAL, command=info_canvas.xview)
        info_scroll_x.grid(row=1, column=0, sticky="ew")
        info_canvas.configure(yscrollcommand=info_scroll_y.set, xscrollcommand=info_scroll_x.set)

        self.info_inner = ttk.Frame(info_canvas)
        self.info_canvas_window = info_canvas.create_window((0, 0), window=self.info_inner, anchor="nw")
        self.info_inner.columnconfigure(0, weight=1)
        self.info_canvas = info_canvas

        run_info_frame = ttk.LabelFrame(self.info_inner, text="Run Info", padding=10)
        run_info_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.status_label = ttk.Label(
            run_info_frame,
            textvariable=self.status_var,
            justify=tk.LEFT,
            anchor=tk.NW,
            font=("TkDefaultFont", 12),
        )
        self.status_label.pack(fill=tk.X)

        left_notebook = ttk.Notebook(plot_area)
        left_notebook.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        right_notebook = ttk.Notebook(plot_area)
        right_notebook.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        heatmap_tab = ttk.Frame(left_notebook)
        heatmap_tab.columnconfigure(0, weight=1)
        heatmap_tab.rowconfigure(0, weight=1)
        heatmap_tab.rowconfigure(1, weight=0)
        left_notebook.add(heatmap_tab, text="Grid Metric")

        trials_tab = ttk.Frame(right_notebook)
        trials_tab.columnconfigure(0, weight=1)
        trials_tab.rowconfigure(0, weight=1)
        right_notebook.add(trials_tab, text="Q0 Trials")

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
        self.heatmap_canvas_widget.grid(row=0, column=0, sticky="nsew")
        self.heatmap_canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self.heatmap_legend_row = ttk.Frame(heatmap_tab)
        self.heatmap_legend_row.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        self._build_heatmap_legend_row()

        self.trials_canvas = FigureCanvasTkAgg(self.trials_figure, master=trials_tab)
        self.trials_canvas_widget = self.trials_canvas.get_tk_widget()
        self.trials_canvas_widget.grid(row=0, column=0, sticky="nsew")

        summary_frame = ttk.LabelFrame(self.info_inner, text="Selected Point", padding=10)
        summary_frame.grid(row=1, column=0, sticky="nsew")
        self.info_inner.rowconfigure(1, weight=1)
        self.summary_label = ttk.Label(
            summary_frame,
            textvariable=self.summary_var,
            justify=tk.LEFT,
            anchor=tk.NW,
            font=("TkFixedFont", 11),
        )
        self.summary_label.pack(fill=tk.BOTH, expand=True, anchor=tk.NW)

        self.root.bind("<Configure>", self._on_resize)
        self.info_inner.bind("<Configure>", self._on_info_panel_configure)
        self.info_canvas.bind("<Configure>", self._on_info_canvas_configure)

        if self.artifact_h5 is not None:
            self._reload_payload()
        else:
            self.status_var.set("No artifact loaded. Use Open Artifact to choose a consolidated scan H5 file.")
            self.summary_var.set("No artifact loaded.")
            self._refresh_all()
        self._refresh_selector_values()

    def _present_window(self) -> None:
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
            # Briefly mark the window topmost so macOS brings it forward.
            self.root.attributes("-topmost", True)
            self.root.after(200, lambda: self.root.attributes("-topmost", False))
        except Exception:
            pass

    def _on_resize(self, _event: Any) -> None:
        total_width = max(400, int(self.root.winfo_width()) - 60)
        side_width = max(240, int(total_width * 0.22))
        self.status_label.configure(wraplength=side_width)
        self.summary_label.configure(wraplength=side_width)

    def _on_info_panel_configure(self, _event: Any) -> None:
        self.info_canvas.configure(scrollregion=self.info_canvas.bbox("all"))

    def _on_info_canvas_configure(self, event: Any) -> None:
        self.info_canvas.itemconfigure(self.info_canvas_window, width=event.width)

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

    def _refresh_selector_values(self) -> None:
        a_labels = [f"{i}: {value:.3f}" for i, value in enumerate(self.a_values)]
        b_labels = [f"{i}: {value:.3f}" for i, value in enumerate(self.b_values)]
        self.a_menu.configure(values=a_labels)
        self.b_menu.configure(values=b_labels)
        if a_labels:
            self.a_menu.current(int(np.clip(self.a_index_var.get(), 0, max(0, len(a_labels) - 1))))
        if b_labels:
            self.b_menu.current(int(np.clip(self.b_index_var.get(), 0, max(0, len(b_labels) - 1))))

    def _reload_payload(self, artifact_path: Path | None = None) -> None:
        if artifact_path is not None:
            self.artifact_h5 = artifact_path
        if self.artifact_h5 is None:
            self.status_var.set("No artifact loaded. Use Open Artifact to choose a consolidated scan H5 file.")
            self.summary_var.set("No artifact loaded.")
            self.payload = {}
            self.a_values = np.asarray([], dtype=float)
            self.b_values = np.asarray([], dtype=float)
            self._refresh_all()
            return
        self.last_artifact_dir = self.artifact_h5.expanduser().resolve().parent
        _save_last_directory(self.last_artifact_dir)
        prev_a = int(self.a_index_var.get())
        prev_b = int(self.b_index_var.get())
        self.payload = load_scan_file(self.artifact_h5)
        self.root.title(f"pychmp-view: {self.artifact_h5.name}")
        self.a_values = np.asarray(self.payload["a_values"], dtype=float)
        self.b_values = np.asarray(self.payload["b_values"], dtype=float)
        self.display_model = build_patch_grid_model(self.payload)
        self.run_target_metric = str(self.payload.get("target_metric", "chi2"))
        if self.metric_var.get() not in METRICS:
            self.metric_var.set(self.run_target_metric)
        if np.any(np.isfinite(np.asarray(self.payload[self.metric_var.get()], dtype=float))):
            default_a, default_b = best_grid_index(self.payload, self.metric_var.get())
        else:
            default_a, default_b = 0, 0
        self.a_index_var.set(int(np.clip(prev_a, 0, max(0, self.a_values.size - 1))) if self.a_values.size else default_a)
        self.b_index_var.set(int(np.clip(prev_b, 0, max(0, self.b_values.size - 1))) if self.b_values.size else default_b)
        self._refresh_selector_values()
        self._refresh_all()

    def _selected_point(self) -> dict[str, Any]:
        return self.payload["points"][(int(self.a_index_var.get()), int(self.b_index_var.get()))]

    def _selected_diagnostics(self) -> dict[str, Any]:
        diagnostics = dict(self.payload["diagnostics"])
        diagnostics.update(self._selected_point()["diagnostics"])
        return diagnostics

    def _jump_to_best(self) -> None:
        a_index, b_index = best_grid_index(self.payload, self.metric_var.get())
        self.a_index_var.set(a_index)
        self.b_index_var.set(b_index)
        self._refresh_selector_values()
        self._refresh_all()

    def _on_metric_changed(self) -> None:
        self._jump_to_best()

    def _on_a_changed(self) -> None:
        self.a_index_var.set(int(self.a_menu.current()))
        self._refresh_all()

    def _on_b_changed(self) -> None:
        self.b_index_var.set(int(self.b_menu.current()))
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
        self._refresh_selector_values()
        self._refresh_all()

    def _refresh_all(self) -> None:
        if not self.payload:
            self.ax_heatmap.clear()
            if self._heatmap_colorbar is not None:
                self._heatmap_colorbar = None
            self.ax_heatmap_cbar.clear()
            self.ax_trials.clear()
            self.ax_heatmap.set_axis_off()
            self.ax_heatmap_cbar.set_axis_off()
            self.ax_trials.set_axis_off()
            self.heatmap_canvas.draw_idle()
            self.trials_canvas.draw_idle()
            return
        self._draw_heatmap()
        self._draw_trials()
        self._refresh_summary()
        self.heatmap_canvas.draw_idle()
        self.trials_canvas.draw_idle()

    def _draw_heatmap(self) -> None:
        metric_name = self.metric_var.get()
        records = list(self.display_model.get("records", []))
        record_lookup = {(int(record["a_index"]), int(record["b_index"])): record for record in records}
        self.ax_heatmap.clear()
        if self._heatmap_colorbar is not None:
            self._heatmap_colorbar = None
        self.ax_heatmap_cbar.clear()
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
            a_index, b_index = best_grid_index(self.payload, name)
            record = record_lookup.get((a_index, b_index))
            b_value = float(record["b_center"]) if record is not None else float(self.b_values[b_index])
            a_value = float(record["a_center"]) if record is not None else float(self.a_values[a_index])
            self.ax_heatmap.scatter(
                [b_value],
                [a_value],
                s=80,
                marker=marker,
                facecolor="none",
                edgecolor=color,
                linewidth=2.0,
                zorder=5,
            )

        current_a = int(self.a_index_var.get())
        current_b = int(self.b_index_var.get())
        current_record = record_lookup.get((current_a, current_b))
        current_b_value = float(current_record["b_center"]) if current_record is not None else float(self.b_values[current_b])
        current_a_value = float(current_record["a_center"]) if current_record is not None else float(self.a_values[current_a])
        self.ax_heatmap.scatter(
            [current_b_value],
            [current_a_value],
            s=120,
            marker="x",
            color="white",
            linewidth=2.2,
            zorder=6,
        )

    def _draw_trials(self) -> None:
        point = self._selected_point()
        q0_trials = np.asarray(point["fit_q0_trials"], dtype=float)
        selected_metric = str(self.metric_var.get())
        selected_metric_key = f"fit_{selected_metric}_trials"
        metric_trials = np.asarray(point.get(selected_metric_key, ()), dtype=float)
        point_metric = selected_metric
        if metric_trials.size != q0_trials.size or metric_trials.size == 0:
            metric_trials = np.asarray(point["fit_metric_trials"], dtype=float)
            point_metric = str(point["target_metric"])
        self.ax_trials.clear()
        self.ax_trials.set_axis_on()
        point_status = str(point.get("status", "computed"))
        if point_status == "pending":
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
            order = np.argsort(q0_trials)
            self.ax_trials.plot(q0_trials[order], metric_trials[order], "-o", ms=4, lw=1.2, color="#2b6cb0")
            self.ax_trials.scatter(q0_trials, metric_trials, s=24, color="#2b6cb0", alpha=0.55)
            self.ax_trials.axvline(float(point["q0"]), color="#d62728", ls="--", lw=1.6)
            best_trial_index = int(np.nanargmin(metric_trials))
            self.ax_trials.scatter([float(point["q0"])], [float(metric_trials[best_trial_index])], color="#d62728", s=42, zorder=4)
            self.ax_trials.set_title(f"{point_metric} vs q0", fontsize=12)
            self.ax_trials.set_xlabel("q0")
            self.ax_trials.set_ylabel(point_metric)
            self.ax_trials.grid(alpha=0.25)
        else:
            self.ax_trials.text(
                0.02,
                0.98,
                "Trial history unavailable",
                transform=self.ax_trials.transAxes,
                va="top",
                ha="left",
            )
            self.ax_trials.set_axis_off()
        self.ax_trials.set_box_aspect(1.0)
        self.trials_figure.subplots_adjust(left=0.14, right=0.98, bottom=0.14, top=0.92)

        success_text = point_status if point_status != "computed" else ("success" if bool(point["success"]) else "boundary/failed")
        best_metric_value = float(np.nanmin(metric_trials)) if metric_trials.size else float("nan")
        self.status_var.set(
            f"Displayed grid metric: {self.metric_var.get()}\n"
            f"Displayed q0-curve metric: {point_metric}\n"
            f"Run target metric: {self.run_target_metric}\n"
            f"Selected point: a={self.a_values[int(self.a_index_var.get())]:.3f}, "
            f"b={self.b_values[int(self.b_index_var.get())]:.3f}\n"
            f"Best q0: {float(point['q0']):.6f}\n"
            f"Point {point_metric}: {best_metric_value:.6e}\n"
            f"Status: {success_text}"
        )

    def _refresh_summary(self) -> None:
        point = self._selected_point()
        diagnostics = self._selected_diagnostics()
        lines = [
            f"a = {self.a_values[int(self.a_index_var.get())]:.3f}",
            f"b = {self.b_values[int(self.b_index_var.get())]:.3f}",
            f"status = {point.get('status', 'computed')}",
            f"q0 = {float(point['q0']):.6f}",
            f"chi2 = {float(diagnostics.get('chi2', np.nan)):.6e}",
            f"rho2 = {float(diagnostics.get('rho2', np.nan)):.6e}",
            f"eta2 = {float(diagnostics.get('eta2', np.nan)):.6e}",
        ]
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
        cmd = [sys.executable, str(script_path), str(self.artifact_h5), *extra_args]
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
        self._open_plot_script("--a-index", str(self.a_index_var.get()), "--b-index", str(self.b_index_var.get()))

    def _open_grid_summary(self) -> None:
        self._open_plot_script("--grid")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive viewer for consolidated `(a,b)` scan artifacts.")
    parser.add_argument("artifact_h5", type=Path, nargs="?", help="Optional consolidated H5 produced by scan_ab_obs_map.py")
    parser.add_argument("--metric", choices=METRICS, default="chi2", help="Initial metric shown in the heatmap.")
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
