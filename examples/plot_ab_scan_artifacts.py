#!/usr/bin/env python
"""Plot consolidated `(a,b)` scan artifacts saved by `scan_ab_obs_map.py`."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    from q0_artifact_plot import plot_q0_artifact_panel
except ModuleNotFoundError:
    from examples.q0_artifact_plot import plot_q0_artifact_panel

try:
    from pychmp.ab_scan_artifacts import (
        METRICS,
        best_grid_index,
        build_patch_grid_model,
        load_scan_file,
        nearest_index,
        with_observer_metadata,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from pychmp.ab_scan_artifacts import (
        METRICS,
        best_grid_index,
        build_patch_grid_model,
        load_scan_file,
        nearest_index,
        with_observer_metadata,
    )



def _plot_grid_summary(
    payload: dict[str, Any],
    *,
    a_index: int,
    b_index: int,
    out_png: Path | None,
    show_plot: bool,
    defer_show: bool = False,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Rectangle
    except ModuleNotFoundError:
        return

    a_values = np.asarray(payload["a_values"], dtype=float)
    b_values = np.asarray(payload["b_values"], dtype=float)
    grid_model = build_patch_grid_model(payload)
    records = list(grid_model["records"])
    record_lookup = {(int(record["a_index"]), int(record["b_index"])): record for record in records}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0), constrained_layout=True)
    panels = [
        ("best_q0", np.asarray(payload["best_q0"], dtype=float), "Best Q0"),
        ("chi2", np.asarray(payload["chi2"], dtype=float), "Best chi2"),
        ("rho2", np.asarray(payload["rho2"], dtype=float), "Best rho2"),
        ("eta2", np.asarray(payload["eta2"], dtype=float), "Best eta2"),
    ]

    best_specs = [
        ("chi2", "#d62728", "o"),
        ("rho2", "#1f77b4", "s"),
        ("eta2", "#2ca02c", "^"),
    ]
    best_points: list[tuple[str, str, str, int, int, float, float, float, float]] = []
    for metric_name, color, marker in best_specs:
        ai, bi = best_grid_index(payload, metric_name)
        q0_best = float(np.asarray(payload["best_q0"], dtype=float)[ai, bi])
        metric_value = float(np.asarray(payload[metric_name], dtype=float)[ai, bi])
        record = record_lookup.get((ai, bi))
        a_center = float(record["a_center"]) if record is not None else float(a_values[ai])
        b_center = float(record["b_center"]) if record is not None else float(b_values[bi])
        a_val = float(record["a"]) if record is not None else float(a_values[ai])
        b_val = float(record["b"]) if record is not None else float(b_values[bi])
        best_points.append((metric_name, color, marker, ai, bi, a_center, b_center, a_val, b_val, q0_best, metric_value))

    for ax, (name, arr, title) in zip(axes.flat, panels, strict=True):
        if name == "best_q0":
            values = np.asarray(
                [float(payload["best_q0"][record["a_index"], record["b_index"]]) for record in records],
                dtype=float,
            )
        else:
            values = np.asarray([float(record["metrics"].get(name, np.nan)) for record in records], dtype=float)
        patches = [
            Rectangle(
                (float(record["b0"]), float(record["a0"])),
                float(record["b1"]) - float(record["b0"]),
                float(record["a1"]) - float(record["a0"]),
            )
            for record in records
        ]
        collection = PatchCollection(patches, cmap="viridis", edgecolor="none", linewidth=0.0)
        collection.set_array(values)
        finite = np.isfinite(values)
        if np.any(finite):
            collection.set_clim(float(np.nanmin(values[finite])), float(np.nanmax(values[finite])))
        ax.add_collection(collection)
        ax.set_xlim(float(grid_model["b_min"]), float(grid_model["b_max"]))
        ax.set_ylim(float(grid_model["a_min"]), float(grid_model["a_max"]))
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("b")
        ax.set_ylabel("a")
        fig.colorbar(collection, ax=ax, fraction=0.046, pad=0.04)
        for metric_name, color, marker, _ai, _bi, a_center, b_center, _a_val, _b_val, _q0_best, _metric_value in best_points:
            ax.scatter(
                [b_center],
                [a_center],
                s=80,
                marker=marker,
                facecolor="none",
                edgecolor=color,
                linewidth=2.0,
                zorder=5,
            )
        selected_record = record_lookup.get((a_index, b_index))
        selected_b = float(selected_record["b_center"]) if selected_record is not None else float(b_values[b_index])
        selected_a = float(selected_record["a_center"]) if selected_record is not None else float(a_values[a_index])
        ax.scatter(
            [selected_b],
            [selected_a],
            s=120,
            marker="x",
            color="white",
            linewidth=2.2,
            zorder=6,
        )
    fig.suptitle("(a, b) Scan Summary", fontsize=14)
    legend_handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=marker,
            markersize=9,
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=2.0,
            label=(
                f"best {metric_name}: "
                f"a={a_val:.3f} b={b_val:.3f} q0={q0_best:.6f} {metric_name}={metric_value:.6e}"
            ),
        )
        for metric_name, color, marker, _ai, _bi, _a_center, _b_center, a_val, b_val, q0_best, metric_value in best_points
    ]
    legend_handles.append(
        Line2D(
            [],
            [],
            linestyle="none",
            marker="x",
            markersize=10,
            color="white",
            markeredgewidth=2.2,
            label=f"selected point: a={float(a_values[a_index]):.3f} b={float(b_values[b_index]):.3f}",
        )
    )
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, framealpha=0.9, fontsize=9)
    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=160)
    if show_plot and not defer_show:
        plt.show(block=False)
        plt.pause(0.1)
    if not defer_show:
        plt.close(fig)


def _plot_selected_point(
    payload: dict[str, Any],
    *,
    a_index: int,
    b_index: int,
    out_png: Path | None,
    show_plot: bool,
    defer_show: bool = False,
) -> None:
    a_values = np.asarray(payload["a_values"], dtype=float)
    b_values = np.asarray(payload["b_values"], dtype=float)
    a_index = int(np.clip(int(a_index), 0, max(0, a_values.size - 1)))
    b_index = int(np.clip(int(b_index), 0, max(0, b_values.size - 1)))
    point = payload["points"][(a_index, b_index)]
    diagnostics = dict(payload["diagnostics"])
    diagnostics.update(point["diagnostics"])
    diagnostics["fit_q0_trials"] = np.asarray(point["fit_q0_trials"], dtype=float).tolist()
    diagnostics["fit_metric_trials"] = np.asarray(point["fit_metric_trials"], dtype=float).tolist()
    diagnostics["q0_recovered"] = float(point["q0"])
    diagnostics["target_metric"] = str(point["target_metric"])
    out_path = out_png or Path("/tmp") / "pychmp_ab_scan_point.png"
    plot_q0_artifact_panel(
        out_path,
        model_path=Path(str(diagnostics.get("model_path", ""))),
        observed_noisy=np.asarray(payload["observed"], dtype=float),
        raw_modeled_best=np.asarray(point["raw_modeled_best"], dtype=float),
        modeled_best=np.asarray(point["modeled_best"], dtype=float),
        residual=np.asarray(point["residual"], dtype=float),
        wcs_header=payload["wcs_header"],
        frequency_ghz=float(diagnostics.get("active_frequency_ghz", diagnostics.get("frequency_ghz", 17.0))),
        diagnostics=diagnostics,
        show_plot=show_plot,
        defer_show=defer_show,
        wcs_header_transform=lambda hdr: with_observer_metadata(hdr, payload["wcs_header"], diagnostics),
    )


def plot_ab_scan_file(
    h5_path: Path,
    *,
    a_index: int = 0,
    b_index: int = 0,
    show_grid: bool = False,
    show_point: bool = True,
    out_grid_png: Path | None = None,
    out_point_png: Path | None = None,
    show_plot: bool = False,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        plt = None  # type: ignore[assignment]

    payload = load_scan_file(h5_path)
    wants_grid = bool(show_grid or out_grid_png is not None)
    wants_point = bool(show_point or out_point_png is not None)
    if wants_grid:
        try:
            _plot_grid_summary(
                payload,
                a_index=a_index,
                b_index=b_index,
                out_png=out_grid_png,
                show_plot=show_plot,
                defer_show=show_plot,
            )
        except ValueError as exc:
            if "No finite values available for metric" not in str(exc):
                raise
            print(f"WARNING: skipping grid-summary plot: {exc}")
    if wants_point:
        _plot_selected_point(
            payload,
            a_index=a_index,
            b_index=b_index,
            out_png=out_point_png,
            show_plot=show_plot,
            defer_show=show_plot,
        )
    if show_plot and plt is not None:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot consolidated `(a,b)` scan artifacts.")
    parser.add_argument("artifact_h5", type=Path, help="Consolidated H5 produced by scan_ab_obs_map.py")
    parser.add_argument("--a-index", type=int, default=None, help="Selected a-grid index for the point panel. If omitted, defaults to the best point for the file target metric.")
    parser.add_argument("--b-index", type=int, default=None, help="Selected b-grid index for the point panel. If omitted, defaults to the best point for the file target metric.")
    parser.add_argument("--a-value", type=float, default=None, help="Alternative a selection by nearest value.")
    parser.add_argument("--b-value", type=float, default=None, help="Alternative b selection by nearest value.")
    parser.add_argument(
        "--best-of-grid",
        nargs="?",
        const="chi2",
        default=None,
        choices=METRICS,
        help="Select the grid point minimizing the chosen summary metric; defaults to chi2 if no metric is given.",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Also show or save the grid-summary figure. If this is the only plot request, no selected-point plot is shown.",
    )
    parser.add_argument("--grid-png", type=Path, default=None, help="Optional output path for the grid-summary PNG.")
    parser.add_argument("--point-png", type=Path, default=None, help="Optional output path for the selected-point PNG.")
    parser.add_argument("--no-plot", action="store_true", help="Do not display the generated plot(s); useful when only saving PNGs.")
    parser.add_argument("--display-worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _spawn_display_worker() -> int:
    cmd = [sys.executable, str(Path(__file__).resolve())]
    cmd.extend(arg for arg in sys.argv[1:] if arg != "--display-worker")
    cmd.append("--display-worker")
    subprocess.Popen(
        cmd,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )
    return 0


def main() -> int:
    args = parse_args()
    if not args.no_plot and not args.display_worker:
        return _spawn_display_worker()
    payload = load_scan_file(args.artifact_h5)
    a_index = int(args.a_index) if args.a_index is not None else 0
    b_index = int(args.b_index) if args.b_index is not None else 0
    explicit_point_selection = any(
        [
            args.best_of_grid is not None,
            args.a_value is not None,
            args.b_value is not None,
            args.a_index is not None,
            args.b_index is not None,
            args.point_png is not None,
        ]
    )
    if args.best_of_grid is not None:
        a_index, b_index = best_grid_index(payload, str(args.best_of_grid))
    elif not explicit_point_selection:
        a_index, b_index = best_grid_index(payload, str(payload.get("target_metric", "chi2")))
    if args.a_value is not None:
        a_index = nearest_index(np.asarray(payload["a_values"], dtype=float), float(args.a_value))
    if args.b_value is not None:
        b_index = nearest_index(np.asarray(payload["b_values"], dtype=float), float(args.b_value))
    show_point = (not bool(args.grid)) or explicit_point_selection
    show_plot = not bool(args.no_plot)
    plot_ab_scan_file(
        args.artifact_h5,
        a_index=a_index,
        b_index=b_index,
        show_grid=bool(args.grid),
        show_point=bool(show_point),
        out_grid_png=args.grid_png,
        out_point_png=args.point_png,
        show_plot=bool(show_plot),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
