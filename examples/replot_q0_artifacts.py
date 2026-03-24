#!/usr/bin/env python3
"""Regenerate PNG panels from a saved pyCHMP artifacts H5 file.

This avoids rerunning the Q0 search/render workflow when you only want to
rebuild the visualization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from astropy.io import fits

from q0_artifact_plot import plot_q0_artifact_panel


def _decode_h5_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        item = value.item()
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace")
        return str(item)
    return str(value)


def _parse_artifact_h5(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        cube = np.asarray(f["maps/data"], dtype=np.float32)
        # Stored as (nx, ny, n_maps, 2[TI/TV]); convert TI planes back to (ny, nx)
        ti_stack = np.transpose(cube[:, :, :3, 0], (1, 0, 2))
        observed = ti_stack[:, :, 0]
        modeled = ti_stack[:, :, 1]
        residual = ti_stack[:, :, 2]

        raw_modeled = None
        if "analysis/raw_modeled_best_ti" in f:
            raw_modeled = np.asarray(f["analysis/raw_modeled_best_ti"], dtype=np.float32)

        fit_q0_trials = None
        fit_metric_trials = None
        if "analysis/fit_q0_trials" in f and "analysis/fit_metric_trials" in f:
            fit_q0_trials = np.asarray(f["analysis/fit_q0_trials"], dtype=np.float64)
            fit_metric_trials = np.asarray(f["analysis/fit_metric_trials"], dtype=np.float64)
        elif "analysis/fit_q0_trials" in f and "analysis/fit_chi2_trials" in f:
            fit_q0_trials = np.asarray(f["analysis/fit_q0_trials"], dtype=np.float64)
            fit_metric_trials = np.asarray(f["analysis/fit_chi2_trials"], dtype=np.float64)

        header_text = _decode_h5_scalar(f["metadata/wcs_header"][()])
        wcs_header = fits.Header.fromstring(header_text, sep="\n")

        diagnostics = {}
        if "metadata/diagnostics_json" in f:
            try:
                diagnostics = json.loads(_decode_h5_scalar(f["metadata/diagnostics_json"][()]))
            except Exception:
                diagnostics = {}

    return {
        "observed": observed,
        "modeled": modeled,
        "residual": residual,
        "raw_modeled": raw_modeled,
        "fit_q0_trials": fit_q0_trials,
        "fit_metric_trials": fit_metric_trials,
        "wcs_header": wcs_header,
        "diagnostics": diagnostics,
    }


def _plot_from_artifact(data: dict[str, Any], out_png: Path, *, log_metrics_override: bool | None = None, log_q0_override: bool | None = None, zoom2best_override: int | None = None, show_plot: bool = False) -> None:
    observed = data["observed"]
    modeled = data["modeled"]
    residual = data["residual"]
    raw_modeled = data["raw_modeled"]
    fit_q0_trials = data.get("fit_q0_trials")
    fit_metric_trials = data.get("fit_metric_trials")
    wcs_header = data["wcs_header"]
    diagnostics = data["diagnostics"]
    freq = float(diagnostics.get("mw_frequency_ghz", 17.0))
    log_metrics = bool(diagnostics.get("log_metrics", False))
    if log_metrics_override is not None:
        log_metrics = bool(log_metrics_override)
    log_q0 = bool(diagnostics.get("log_q0", False))
    if log_q0_override is not None:
        log_q0 = bool(log_q0_override)
    zoom2best = int(diagnostics.get("zoom2best", 0)) or None
    if zoom2best_override is not None:
        zoom2best = zoom2best_override

    q0_trials_for_plot = fit_q0_trials
    metric_trials_for_plot = fit_metric_trials
    if q0_trials_for_plot is None:
        q0_trials_for_plot = diagnostics.get("fit_q0_trials")
    if metric_trials_for_plot is None:
        metric_trials_for_plot = diagnostics.get("fit_metric_trials", diagnostics.get("fit_chi2_trials"))

    diagnostics_for_plot = dict(diagnostics)
    diagnostics_for_plot["fit_q0_trials"] = q0_trials_for_plot
    diagnostics_for_plot["fit_metric_trials"] = metric_trials_for_plot

    plot_q0_artifact_panel(
        out_png,
        model_path=Path(str(diagnostics.get("model_path", ""))),
        observed_noisy=observed,
        raw_modeled_best=(raw_modeled if raw_modeled is not None else modeled),
        modeled_best=modeled,
        residual=residual,
        wcs_header=wcs_header,
        frequency_ghz=freq,
        diagnostics=diagnostics_for_plot,
        log_metrics=log_metrics,
        log_q0=log_q0,
        zoom2best=zoom2best,
        show_plot=show_plot,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild pyCHMP artifact PNG from artifact H5")
    p.add_argument("artifacts_h5_pos", nargs="?", metavar="ARTIFACTS_H5", default=None, help="Path to q0_recovery_artifacts.h5 (positional shorthand)")
    p.add_argument("--artifacts-h5", default=None, help="Path to q0_recovery_artifacts.h5")
    p.add_argument("--output-png", default=None, help="Output PNG path (default: alongside H5)")
    p.add_argument("--log-metrics", action="store_true", help="Force logarithmic y-axis (metric axis) for trials panel.")
    p.add_argument("--log-q0", action="store_true", help="Force logarithmic x-axis (q0 axis) for trials panel.")
    p.add_argument("--zoom2best", type=int, default=None, metavar="N", help="Restrict trials panel axes to ±N trials around the best-metric trial.")
    p.add_argument("--show-plot", action="store_true", help="Display the PNG panel interactively after saving (calls plt.show()).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    raw = args.artifacts_h5 or args.artifacts_h5_pos
    if not raw:
        raise SystemExit("error: provide the artifacts H5 path as a positional argument or via --artifacts-h5")
    h5_path = Path(raw).expanduser()
    if not h5_path.exists():
        raise SystemExit(f"artifact H5 not found: {h5_path}")

    if args.output_png:
        out_png = Path(args.output_png).expanduser()
    else:
        out_png = h5_path.with_name(h5_path.stem + "_replot.png")

    data = _parse_artifact_h5(h5_path)
    _plot_from_artifact(
        data, out_png,
        log_metrics_override=(True if args.log_metrics else None),
        log_q0_override=(True if args.log_q0 else None),
        zoom2best_override=args.zoom2best,
        show_plot=args.show_plot,
    )
    print(f"wrote PNG: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
