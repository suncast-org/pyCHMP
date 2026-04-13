#!/usr/bin/env python
"""Adaptive `(a, b)` search against a single real observational map.

This example targets the current reliable EOVSA 2.874 GHz workflow by default,
but it also accepts explicit FITS/model/EBTEL paths for other single-frequency
 runs. It persists every evaluated `(a, b)` point into a sparse HDF5 artifact so
the viewer can inspect progress while the search is still running.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

def _format_console_scalar(value: float, *, fixed_precision: int = 6) -> str:
    numeric = float(value)
    if not np.isfinite(numeric):
        return "nan"
    if numeric != 0.0 and abs(numeric) < 10 ** (-fixed_precision):
        return f"{numeric:.{max(1, fixed_precision)}e}"
    return f"{numeric:.{fixed_precision}f}"


def _build_run_history_entry(
    *,
    artifact_h5: Path,
    log_path: Path,
    viewer_cmd_text: str,
    action: str,
    target_metric: str,
    recompute_existing: bool,
) -> dict[str, Any]:
    effective_python_argv = [str(sys.executable), *[str(item) for item in sys.argv]]
    wrapper_command = os.environ.get("PYCHMP_WRAPPER_COMMAND", "").strip()
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "action": str(action),
        "artifact_path": str(artifact_h5),
        "cwd": os.getcwd(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "wrapper_command": wrapper_command or None,
        "effective_python_argv": effective_python_argv,
        "effective_python_command": shlex.join(effective_python_argv),
        "viewer_command": str(viewer_cmd_text),
        "log_path": str(log_path),
        "target_metric": str(target_metric),
        "recompute_existing": bool(recompute_existing),
    }


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples"
for candidate in (REPO_ROOT, EXAMPLES_ROOT):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

from pychmp import ABPointResult, GXRenderMWAdapter, estimate_map_noise, search_local_minimum_ab
from pychmp.ab_scan_artifacts import (
    ScanArtifactCompatibilityError,
    append_sparse_point_record,
    append_run_history_entry,
    build_computed_point_payload,
    detect_scan_artifact_format,
    load_scan_file,
    load_run_history,
    validate_scan_artifact_compatibility,
    write_sparse_scan_file,
)

try:
    from fit_q0_obs_map import (
        DEFAULT_A,
        DEFAULT_B,
        DEFAULT_NBASE,
        DEFAULT_TBASE,
        PSFConvolvedRenderer,
        _build_target_header,
        _effective_psf_parameters,
        _elliptical_gaussian_kernel,
        _extract_psf_from_header,
        _format_psf_report,
        _resolve_selected_psf,
        _load_model_observer_metadata,
        _load_saved_fov_from_model,
        _regrid_full_disk_to_target,
        _resolve_observer_overrides,
        _with_observer_wcs_keywords,
        load_eovsa_map,
    )
except ModuleNotFoundError:
    from examples.fit_q0_obs_map import (
        DEFAULT_A,
        DEFAULT_B,
        DEFAULT_NBASE,
        DEFAULT_TBASE,
        PSFConvolvedRenderer,
        _build_target_header,
        _effective_psf_parameters,
        _elliptical_gaussian_kernel,
        _extract_psf_from_header,
        _format_psf_report,
        _resolve_selected_psf,
        _load_model_observer_metadata,
        _load_saved_fov_from_model,
        _regrid_full_disk_to_target,
        _resolve_observer_overrides,
        _with_observer_wcs_keywords,
        load_eovsa_map,
    )

try:
    from plot_ab_scan_artifacts import plot_ab_scan_file
except ModuleNotFoundError:
    from examples.plot_ab_scan_artifacts import plot_ab_scan_file


DEFAULT_FREQ_FITS = "eovsa.synoptic_daily.20201126T200000Z.f2.874GHz.tb.disk.fits"
DEFAULT_MODEL_H5 = "hmi.M_720s.20201126_195831.E18S19CR.CEA.NAS.GEN.CHR.h5"
DEFAULT_EBTEL = "ebtel.sav"


def _validate_gxrender_runtime() -> None:
    """Fail fast on clearly incompatible gxrender native-extension layouts."""

    if not sys.platform.startswith("win"):
        return
    try:
        gxrender_pkg = import_module("gxrender")
    except Exception:
        return

    pkg_root = Path(getattr(gxrender_pkg, "__file__", "")).resolve().parent
    if not pkg_root.exists():
        return

    bad_candidates = sorted(pkg_root.glob("RenderGRFF*.so"))
    darwin_candidates = [path for path in bad_candidates if "darwin" in path.name.lower()]
    if darwin_candidates:
        candidate_list = ", ".join(str(path) for path in darwin_candidates)
        raise SystemExit(
            "Incompatible gxrender native extension detected on Windows. "
            "The current import path resolves to macOS RenderGRFF binaries: "
            f"{candidate_list}. "
            "Fix the gxrender installation or import path before running pyCHMP."
        )


class _TeeStream:
    def __init__(self, *streams: Any) -> None:
        self._streams = [stream for stream in streams if stream is not None]

    def write(self, data: str) -> int:
        for stream in self._streams:
            try:
                stream.write(data)
            except UnicodeEncodeError:
                encoding = getattr(stream, "encoding", None) or "utf-8"
                safe_text = str(data).encode(encoding, errors="replace").decode(encoding, errors="replace")
                stream.write(safe_text)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            try:
                stream.flush()
            except Exception:
                pass


class _ViewerRefreshHeartbeat:
    def __init__(self, signal_path: Path, *, interval_s: float = 2.0) -> None:
        self._signal_path = Path(signal_path)
        self._interval_s = max(0.5, float(interval_s))
        self._phase = ""
        self._pending_points: tuple[tuple[float, float], ...] = ()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _write_signal(self) -> None:
        try:
            with self._lock:
                payload = {
                    "timestamp": float(time.time()),
                    "phase": str(self._phase),
                    "pending_points": [
                        {"a": float(a_value), "b": float(b_value)}
                        for a_value, b_value in self._pending_points
                    ],
                }
            self._signal_path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
        except Exception:
            pass

    @property
    def phase(self) -> str:
        with self._lock:
            return str(self._phase)

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = str(phase).strip()
        self._write_signal()

    def set_pending_points(self, points: list[tuple[float, float]] | tuple[tuple[float, float], ...]) -> None:
        normalized: list[tuple[float, float]] = []
        for a_value, b_value in points:
            normalized.append((float(a_value), float(b_value)))
        with self._lock:
            self._pending_points = tuple(normalized)
        self._write_signal()

    def clear_pending_points(self) -> None:
        with self._lock:
            self._pending_points = ()
        self._write_signal()

    def start(self, phase: str) -> None:
        self.set_phase(phase)
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="pychmp-viewer-heartbeat", daemon=True)
        self._thread.start()

    def stop(self, phase: str) -> None:
        with self._lock:
            self._phase = str(phase).strip()
            self._pending_points = ()
        self._write_signal()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_s + 0.5)
            self._thread = None

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            self._write_signal()


@dataclass(frozen=True)
class _FactoryGeometry:
    xc: float
    yc: float
    dx: float
    dy: float
    nx: int
    ny: int


@dataclass(frozen=True)
class _ObserverOverrideData:
    dsun_cm: float | None
    lonc_deg: float | None
    b0sun_deg: float | None


@dataclass(frozen=True)
class _AdaptiveRendererFactory:
    model_path: str
    ebtel_path: str
    frequency_ghz: float
    tbase: float
    nbase: float
    geometry: _FactoryGeometry
    observer_overrides: _ObserverOverrideData | None
    pixel_scale_arcsec: float
    psf_kernel: np.ndarray | None

    def __call__(self, a: float, b: float) -> Any:
        sdk = import_module("gxrender.sdk")
        geometry = sdk.MapGeometry(
            xc=float(self.geometry.xc),
            yc=float(self.geometry.yc),
            dx=float(self.geometry.dx),
            dy=float(self.geometry.dy),
            nx=int(self.geometry.nx),
            ny=int(self.geometry.ny),
        )
        observer = None
        if self.observer_overrides is not None:
            observer = sdk.ObserverOverrides(
                dsun_cm=self.observer_overrides.dsun_cm,
                lonc_deg=self.observer_overrides.lonc_deg,
                b0sun_deg=self.observer_overrides.b0sun_deg,
            )
        base = GXRenderMWAdapter(
            model_path=self.model_path,
            ebtel_path=self.ebtel_path,
            frequency_ghz=float(self.frequency_ghz),
            tbase=float(self.tbase),
            nbase=float(self.nbase),
            a=float(a),
            b=float(b),
            geometry=geometry,
            observer=observer,
            pixel_scale_arcsec=float(self.pixel_scale_arcsec),
        )
        if self.psf_kernel is None:
            return base
        return PSFConvolvedRenderer(base, np.asarray(self.psf_kernel, dtype=float))


def _latest_dated_dir(parent: Path, prefix: str) -> Path | None:
    candidates = sorted(path for path in parent.glob(f"{prefix}_*") if path.is_dir())
    return candidates[-1] if candidates else None


def _default_testdata_repo(repo_root: Path) -> Path:
    return repo_root.parent / "pyGXrender-test-data"


def _coerce_path(value: Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def _resolve_runtime_paths(args: argparse.Namespace, *, repo_root: Path) -> tuple[Path, Path, Path]:
    fits_path = _coerce_path(args.fits_file)
    model_h5 = _coerce_path(args.model_h5)
    ebtel_path = _coerce_path(args.ebtel_path)

    if fits_path is not None and model_h5 is not None and ebtel_path is not None:
        return fits_path, model_h5, ebtel_path

    testdata_repo = _coerce_path(args.testdata_repo) or _default_testdata_repo(repo_root)
    eovsa_root = testdata_repo / "raw" / "eovsa_maps"
    model_root = testdata_repo / "raw" / "models"
    ebtel_root = testdata_repo / "raw" / "ebtel" / "ebtel_gxsimulator_euv"
    latest_eovsa = _latest_dated_dir(eovsa_root, "eovsa_maps")
    latest_model = _latest_dated_dir(model_root, "models")

    if fits_path is None:
        if latest_eovsa is None:
            raise SystemExit(f"No dated EOVSA map folder found under {eovsa_root}")
        fits_path = latest_eovsa / DEFAULT_FREQ_FITS
    if model_h5 is None:
        if latest_model is None:
            raise SystemExit(f"No dated model folder found under {model_root}")
        model_h5 = latest_model / DEFAULT_MODEL_H5
    if ebtel_path is None:
        ebtel_path = ebtel_root / DEFAULT_EBTEL

    return fits_path.resolve(), model_h5.resolve(), ebtel_path.resolve()


def _point_payload_from_result(
    point: ABPointResult,
    *,
    renderer_factory: _AdaptiveRendererFactory,
    observed_template: np.ndarray,
    target_metric: str,
    psf_source: str,
) -> dict[str, Any]:
    renderer = renderer_factory(float(point.a), float(point.b))
    if hasattr(renderer, "render_pair"):
        raw_modeled_best, modeled_best = renderer.render_pair(float(point.q0))
    else:
        modeled_best = renderer.render(float(point.q0))
        raw_modeled_best = modeled_best
    modeled_best = np.asarray(modeled_best, dtype=float)
    raw_modeled_best = np.asarray(raw_modeled_best, dtype=float)
    residual = np.asarray(modeled_best - np.asarray(observed_template, dtype=float), dtype=float)
    fit_metric_trials = tuple(float(v) for v in point.trial_objective_values)
    fit_chi2_trials = tuple(float(v) for v in point.trial_chi2_values)
    fit_rho2_trials = tuple(float(v) for v in point.trial_rho2_values)
    fit_eta2_trials = tuple(float(v) for v in point.trial_eta2_values)
    elapsed_seconds = float(point.elapsed_seconds)
    diagnostics = {
        "a": float(point.a),
        "b": float(point.b),
        "target_metric": str(target_metric),
        "target_metric_value": float(point.objective_value),
        "chi2": float(point.metrics.chi2),
        "rho2": float(point.metrics.rho2),
        "eta2": float(point.metrics.eta2),
        "fit_success": bool(point.success),
        "optimizer_message": str(point.message),
        "nfev": int(point.nfev),
        "nit": int(point.nit),
        "used_adaptive_bracketing": bool(point.used_adaptive_bracketing),
        "bracket_found": bool(point.bracket_found),
        "bracket": None if point.bracket is None else [float(v) for v in point.bracket],
        "fit_q0_trials": [float(v) for v in point.trial_q0],
        "fit_metric_trials": [float(v) for v in point.trial_objective_values],
        "fit_chi2_trials": [float(v) for v in fit_chi2_trials],
        "fit_rho2_trials": [float(v) for v in fit_rho2_trials],
        "fit_eta2_trials": [float(v) for v in fit_eta2_trials],
        "psf_source": str(psf_source),
        "point_status": "computed",
    }
    if np.isfinite(elapsed_seconds):
        diagnostics["elapsed_seconds"] = elapsed_seconds
    return build_computed_point_payload(
        a_value=float(point.a),
        b_value=float(point.b),
        q0=float(point.q0),
        success=bool(point.success),
        status="computed",
        modeled_best=modeled_best,
        raw_modeled_best=raw_modeled_best,
        residual=residual,
        fit_q0_trials=tuple(float(v) for v in point.trial_q0),
        fit_metric_trials=fit_metric_trials,
        fit_chi2_trials=fit_chi2_trials,
        fit_rho2_trials=fit_rho2_trials,
        fit_eta2_trials=fit_eta2_trials,
        nfev=int(point.nfev),
        nit=int(point.nit),
        message=str(point.message),
        used_adaptive_bracketing=bool(point.used_adaptive_bracketing),
        bracket_found=bool(point.bracket_found),
        bracket=None if point.bracket is None else tuple(float(v) for v in point.bracket),
        target_metric=str(target_metric),
        diagnostics=diagnostics,
    )


class _MetricValues:
    def __init__(self, *, chi2: float, rho2: float, eta2: float) -> None:
        self.chi2 = float(chi2)
        self.rho2 = float(rho2)
        self.eta2 = float(eta2)


def _point_from_record(record: dict[str, Any], *, target_metric: str) -> ABPointResult:
    diagnostics = dict(record.get("diagnostics", {}))
    return ABPointResult(
        a=float(record["a"]),
        b=float(record["b"]),
        q0=float(record.get("q0", np.nan)),
        objective_value=float(diagnostics.get("target_metric_value", np.nan)),
        metrics=_MetricValues(
            chi2=float(diagnostics.get("chi2", np.nan)),
            rho2=float(diagnostics.get("rho2", np.nan)),
            eta2=float(diagnostics.get("eta2", np.nan)),
        ),
        target_metric=str(record.get("target_metric", target_metric)),
        success=bool(record.get("success", False)),
        nfev=int(record.get("nfev", diagnostics.get("nfev", -1))),
        nit=int(record.get("nit", diagnostics.get("nit", -1))),
        message=str(record.get("message", diagnostics.get("optimizer_message", ""))),
        used_adaptive_bracketing=bool(record.get("used_adaptive_bracketing", diagnostics.get("used_adaptive_bracketing", False))),
        bracket_found=bool(record.get("bracket_found", diagnostics.get("bracket_found", False))),
        bracket=None if record.get("bracket") is None else tuple(float(v) for v in record.get("bracket")),
        trial_q0=tuple(float(v) for v in record.get("fit_q0_trials", ())),
        trial_objective_values=tuple(float(v) for v in record.get("fit_metric_trials", ())),
        trial_chi2_values=tuple(float(v) for v in record.get("fit_chi2_trials", ())),
        trial_rho2_values=tuple(float(v) for v in record.get("fit_rho2_trials", ())),
        trial_eta2_values=tuple(float(v) for v in record.get("fit_eta2_trials", ())),
        elapsed_seconds=float(diagnostics.get("elapsed_seconds", np.nan)),
    )


class _PersistentPointCache(MutableMapping[tuple[float, float], ABPointResult]):
    def __init__(
        self,
        *,
        artifact_h5: Path,
        observed: np.ndarray,
        sigma_map: np.ndarray,
        target_header: Any,
        diagnostics: dict[str, Any],
        renderer_factory: _AdaptiveRendererFactory,
        target_metric: str,
        psf_source: str,
        viewer_heartbeat: _ViewerRefreshHeartbeat | None = None,
    ) -> None:
        self._artifact_h5 = Path(artifact_h5)
        self._observed = np.asarray(observed, dtype=float)
        self._sigma_map = np.asarray(sigma_map, dtype=float)
        self._target_header = target_header
        self._diagnostics = dict(diagnostics)
        self._renderer_factory = renderer_factory
        self._target_metric = str(target_metric)
        self._psf_source = str(psf_source)
        self._viewer_heartbeat = viewer_heartbeat
        self._point_map: dict[tuple[float, float], ABPointResult] = {}

    def set_pending_points(self, points: list[tuple[float, float]] | tuple[tuple[float, float], ...]) -> None:
        if self._viewer_heartbeat is not None:
            self._viewer_heartbeat.set_pending_points(points)

    def clear_pending_points(self) -> None:
        if self._viewer_heartbeat is not None:
            self._viewer_heartbeat.clear_pending_points()

    def hydrate_from_existing(self) -> int:
        if not self._artifact_h5.exists():
            return 0
        payload = load_scan_file(self._artifact_h5)
        validate_scan_artifact_compatibility(
            payload,
            observed=self._observed,
            sigma_map=self._sigma_map,
            wcs_header=self._target_header,
            diagnostics=self._diagnostics,
            artifact_path=self._artifact_h5,
        )
        count = 0
        for record in payload.get("point_records", []):
            point = _point_from_record(record, target_metric=self._target_metric)
            self._point_map[(float(point.a), float(point.b))] = point
            count += 1
        return count

    def __getitem__(self, key: tuple[float, float]) -> ABPointResult:
        return self._point_map[key]

    def __setitem__(self, key: tuple[float, float], value: ABPointResult) -> None:
        normalized_key = (float(key[0]), float(key[1]))
        existing = self._point_map.get(normalized_key)
        if existing is not None and np.isclose(existing.q0, value.q0, rtol=0.0, atol=1e-12):
            self._point_map[normalized_key] = value
            return
        payload = _point_payload_from_result(
            value,
            renderer_factory=self._renderer_factory,
            observed_template=self._observed,
            target_metric=self._target_metric,
            psf_source=self._psf_source,
        )
        append_sparse_point_record(
            self._artifact_h5,
            observed=self._observed,
            sigma_map=self._sigma_map,
            wcs_header=self._target_header,
            diagnostics=self._diagnostics,
            point_payload=payload,
        )
        self._point_map[normalized_key] = value
        if self._viewer_heartbeat is not None:
            self._viewer_heartbeat.set_phase(
                f"point {len(self._point_map)} saved"
            )
        print(
            f"  Saved point to sparse artifact: a={float(value.a):.3f} b={float(value.b):.3f} "
            f"q0={_format_console_scalar(float(value.q0), fixed_precision=6)} "
            f"{self._target_metric}={float(value.objective_value):.6e}"
        )

    def __delitem__(self, key: tuple[float, float]) -> None:
        del self._point_map[key]

    def __iter__(self) -> Iterator[tuple[float, float]]:
        return iter(self._point_map)

    def __len__(self) -> int:
        return len(self._point_map)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an adaptive local `(a, b)` search against a single real observational map.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("fits_file", nargs="?", type=Path, help="Path to the observational FITS map")
    parser.add_argument("model_h5", nargs="?", type=Path, help="Path to the model H5 file")
    parser.add_argument("--ebtel-path", type=Path, default=None, help="Path to the matching EBTEL .sav file")
    parser.add_argument("--testdata-repo", type=Path, default=None, help="Optional sibling pyGXrender-test-data checkout used for default input resolution")
    parser.add_argument("--a-start", type=float, default=DEFAULT_A, help="Adaptive search starting a value")
    parser.add_argument("--b-start", type=float, default=DEFAULT_B, help="Adaptive search starting b value")
    parser.add_argument("--da", type=float, default=0.3, help="Adaptive a step size")
    parser.add_argument("--db", type=float, default=0.3, help="Adaptive b step size")
    parser.add_argument("--a-min", type=float, default=0.0, help="Adaptive search lower a bound")
    parser.add_argument("--a-max", type=float, default=1.2, help="Adaptive search upper a bound")
    parser.add_argument("--b-min", type=float, default=2.1, help="Adaptive search lower b bound")
    parser.add_argument("--b-max", type=float, default=3.6, help="Adaptive search upper b bound")
    parser.add_argument("--q0-min", type=float, default=1e-5, help="Lower edge of the initial q0 interval")
    parser.add_argument("--q0-max", type=float, default=1e-3, help="Upper edge of the initial q0 interval")
    parser.add_argument("--hard-q0-min", type=float, default=None, help="Optional hard lower q0 bound")
    parser.add_argument("--hard-q0-max", type=float, default=None, help="Optional hard upper q0 bound")
    parser.add_argument("--target-metric", choices=("chi2", "rho2", "eta2"), default="chi2", help="Metric minimized during the search")
    parser.add_argument("--threshold", type=float, default=0.1, help="Mask threshold passed into each q0 fit")
    parser.add_argument("--mask-type", choices=("union", "data", "model", "and"), default="union", help="Mask type for metric computation: union (OR), data, model, or and (AND)")
    parser.add_argument("--threshold-metric", type=float, default=1.1, help="Phase-2 threshold region multiplier around the best point")
    parser.add_argument("--no-area", action="store_true", help="Stop after phase 1 without threshold-region expansion")
    parser.add_argument("--adaptive-bracketing", action=argparse.BooleanOptionalAction, default=True, help="Enable adaptive q0 bracketing")
    parser.add_argument("--q0-start", type=float, default=None, help="Optional explicit q0 start for every point")
    parser.add_argument("--q0-step", type=float, default=1.61803398875, help="Multiplicative q0 step for adaptive bracketing")
    parser.add_argument("--max-bracket-steps", type=int, default=12, help="Maximum adaptive q0 bracket expansions")
    parser.add_argument("--tbase", type=float, default=DEFAULT_TBASE, help="Base temperature in K")
    parser.add_argument("--nbase", type=float, default=DEFAULT_NBASE, help="Base density in cm^-3")
    parser.add_argument("--observer", default=None, help="Observer name override, e.g. earth")
    parser.add_argument("--dsun-cm", type=float, default=None, help="Observer-Sun distance override in cm")
    parser.add_argument("--lonc-deg", type=float, default=None, help="Observer Carrington longitude override in degrees")
    parser.add_argument("--b0sun-deg", type=float, default=None, help="Observer latitude override in degrees")
    parser.add_argument("--pixel-scale-arcsec", type=float, default=2.0, help="Pixel scale used with the model saved FOV")
    parser.add_argument("--psf-bmaj-arcsec", type=float, default=None, help="PSF major-axis FWHM in arcsec")
    parser.add_argument("--psf-bmin-arcsec", type=float, default=None, help="PSF minor-axis FWHM in arcsec")
    parser.add_argument("--psf-bpa-deg", type=float, default=None, help="PSF position angle in degrees")
    parser.add_argument(
        "--override-header-psf",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use user-supplied PSF or fallback reference-beam parameters even when the FITS header already contains a PSF beam",
    )
    parser.add_argument("--fallback-psf-bmaj-arcsec", type=float, default=None, help="Fallback PSF major-axis FWHM used only when the FITS header has no beam and no explicit PSF override is supplied")
    parser.add_argument("--fallback-psf-bmin-arcsec", type=float, default=None, help="Fallback PSF minor-axis FWHM used only when the FITS header has no beam and no explicit PSF override is supplied")
    parser.add_argument("--fallback-psf-bpa-deg", type=float, default=None, help="Fallback PSF position angle used only when the FITS header has no beam and no explicit PSF override is supplied")
    parser.add_argument("--psf-ref-frequency-ghz", type=float, default=None, help="Reference frequency for the supplied PSF")
    parser.add_argument(
        "--psf-scale-inverse-frequency",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Scale PSF axes by reference_frequency / active_frequency",
    )
    parser.add_argument("--artifact-h5", type=Path, default=None, help="Explicit sparse artifact path to create/update")
    parser.add_argument("--artifacts-dir", type=Path, default=None, help="Directory used when --artifact-h5 is not supplied")
    parser.add_argument("--artifacts-stem", default=None, help="Base filename stem used when --artifact-h5 is not supplied")
    parser.add_argument(
        "--recompute-existing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reset a compatible sparse artifact at the target path and recompute all points from scratch.",
    )
    parser.add_argument("--no-grid-png", action="store_true", help="Skip the summary grid PNG at the end")
    parser.add_argument("--no-point-png", action="store_true", help="Skip the selected-point PNG at the end")
    parser.add_argument("--show-plot", action="store_true", help="Display the final plots interactively")
    parser.add_argument("--no-viewer", action="store_true", help="Do not launch pychmp-view automatically when the scan starts")
    parser.add_argument(
        "--require-interior-best",
        action="store_true",
        help="Exit nonzero unless the adaptive search certifies an interior best point",
    )
    parser.add_argument("--execution-policy", choices=("serial", "process-pool", "auto"), default="serial", help="Execution policy used by the adaptive search")
    parser.add_argument("--max-workers", type=int, default=None, help="Optional worker cap for process execution")
    parser.add_argument("--worker-chunksize", type=int, default=1, help="Task chunksize for process execution")
    parser.add_argument("--dry-run", action="store_true", help="Resolve inputs and print run settings without starting the search")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _validate_gxrender_runtime()
    repo_root = Path(__file__).resolve().parents[2]
    fits_path, model_h5, ebtel_path = _resolve_runtime_paths(args, repo_root=repo_root)
    if not fits_path.exists():
        raise SystemExit(f"Observational FITS file not found: {fits_path}")
    if not model_h5.exists():
        raise SystemExit(f"Model H5 file not found: {model_h5}")
    if not ebtel_path.exists():
        raise SystemExit(f"EBTEL file not found: {ebtel_path}")

    artifacts_dir = _coerce_path(args.artifacts_dir) or (repo_root / "ab_scan_artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stem = args.artifacts_stem or f"{fits_path.stem}_adaptive_ab_{args.target_metric}"
    artifact_h5 = _coerce_path(args.artifact_h5) or (artifacts_dir / f"{stem}.h5")
    grid_png = None if args.no_grid_png else artifact_h5.with_name(f"{artifact_h5.stem}_grid.png")
    point_png = None if args.no_point_png else artifact_h5.with_name(f"{artifact_h5.stem}_point.png")
    log_path = Path(f"{artifact_h5}.log")
    viewer_script = repo_root / "examples" / "pychmp_view.py"
    viewer_cmd = [sys.executable, str(viewer_script), str(artifact_h5)]
    viewer_cmd_text = shlex.join(viewer_cmd)
    viewer_process = None
    suppress_auto_viewer = os.environ.get("PYCHMP_NO_AUTO_VIEWER", "").strip().lower() in {"1", "true", "yes", "on"}
    auto_viewer_enabled = not bool(args.no_viewer) and not suppress_auto_viewer

    def _maybe_launch_viewer(phase: str) -> None:
        nonlocal viewer_process
        if not auto_viewer_enabled:
            return
        if viewer_process is not None and viewer_process.poll() is None:
            print(f"pychmp-view already running ({phase}) pid={viewer_process.pid}")
            return
        try:
            proc = subprocess.Popen(viewer_cmd, start_new_session=True)
            if proc.poll() is not None:
                print(f"WARNING: pychmp-view exited immediately after auto-launch ({phase}). Try running manually: {viewer_cmd_text}")
                return
            viewer_process = proc
            print(f"Launched pychmp-view ({phase}) pid={proc.pid}")
        except Exception as exc:
            print(f"WARNING: failed to launch pychmp-view automatically ({phase}): {exc}")

    print(f"Using FITS map: {fits_path}")
    print(f"Using model H5: {model_h5}")
    print(f"Using EBTEL: {ebtel_path}")
    print(f"Artifact H5: {artifact_h5}")
    print(f"Live viewer command: {viewer_cmd_text}")
    if auto_viewer_enabled:
        print("Viewer auto-launch: enabled")
    elif bool(args.no_viewer):
        print("Viewer auto-launch: disabled by --no-viewer")
    else:
        print("Viewer auto-launch: disabled by PYCHMP_NO_AUTO_VIEWER")
    print(
        "Adaptive search: "
        f"a_start={float(args.a_start):.3f} b_start={float(args.b_start):.3f} "
        f"da={float(args.da):.3f} db={float(args.db):.3f} "
        f"a_range=({float(args.a_min):.3f}, {float(args.a_max):.3f}) "
        f"b_range=({float(args.b_min):.3f}, {float(args.b_max):.3f})"
    )

    if args.dry_run:
        print("Dry run only; no artifact was created and no search was started.")
        return 0

    live_log_handle = None
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        live_log_handle = log_path.open("a", encoding="utf-8", buffering=1)
        sys.stdout = _TeeStream(sys.stdout, live_log_handle)
        sys.stderr = _TeeStream(sys.stderr, live_log_handle)
        print(f"Live log file: {log_path}")
    except Exception as exc:
        print(f"WARNING: failed to initialize live log sidecar: {exc}")

    print(f"Loading FITS file: {fits_path.name}")
    observed, header, freq_ghz = load_eovsa_map(fits_path)
    print(f"  Shape: {observed.shape}")
    print(f"  Frequency: {float(freq_ghz):.3f} GHz")

    print("Estimating noise from map...")
    noise_result = estimate_map_noise(observed, method="histogram_clip")
    if noise_result is None:
        sigma_map = np.full_like(observed, observed.std())
        noise_diag = None
        print(f"  Noise estimate unavailable; using sigma={float(observed.std()):.2f} K")
    else:
        sigma_map = np.asarray(noise_result.sigma_map, dtype=float)
        noise_diag = noise_result.diagnostics
        print(f"  Estimated sigma: {float(noise_result.sigma):.2f} K")

    sdk = import_module("gxrender.sdk")
    header_psf, header_psf_source = _extract_psf_from_header(header)
    psf_bmaj_arcsec = float(args.psf_bmaj_arcsec) if args.psf_bmaj_arcsec is not None else None
    psf_bmin_arcsec = float(args.psf_bmin_arcsec) if args.psf_bmin_arcsec is not None else None
    psf_bpa_deg = float(args.psf_bpa_deg) if args.psf_bpa_deg is not None else None
    fallback_psf_bmaj_arcsec = float(args.fallback_psf_bmaj_arcsec) if args.fallback_psf_bmaj_arcsec is not None else None
    fallback_psf_bmin_arcsec = float(args.fallback_psf_bmin_arcsec) if args.fallback_psf_bmin_arcsec is not None else None
    fallback_psf_bpa_deg = float(args.fallback_psf_bpa_deg) if args.fallback_psf_bpa_deg is not None else None
    (
        psf_bmaj_arcsec,
        psf_bmin_arcsec,
        psf_bpa_deg,
        psf_source,
        psf_allows_frequency_scaling,
    ) = _resolve_selected_psf(
        header_psf=header_psf,
        header_psf_source=header_psf_source,
        cli_psf_bmaj_arcsec=psf_bmaj_arcsec,
        cli_psf_bmin_arcsec=psf_bmin_arcsec,
        cli_psf_bpa_deg=psf_bpa_deg,
        fallback_psf_bmaj_arcsec=fallback_psf_bmaj_arcsec,
        fallback_psf_bmin_arcsec=fallback_psf_bmin_arcsec,
        fallback_psf_bpa_deg=fallback_psf_bpa_deg,
        override_header_psf=bool(args.override_header_psf),
    )

    observer_overrides, observer_source = _resolve_observer_overrides(
        sdk,
        model_path=model_h5,
        observer_name=args.observer,
        dsun_cm=args.dsun_cm,
        lonc_deg=args.lonc_deg,
        b0sun_deg=args.b0sun_deg,
    )
    model_observer_meta = _load_model_observer_metadata(model_h5)
    saved_fov = _load_saved_fov_from_model(model_h5)
    if saved_fov is None:
        raise SystemExit("model does not expose a saved FOV; this example requires one")
    geometry = sdk.MapGeometry(
        xc=float(saved_fov["xc_arcsec"]),
        yc=float(saved_fov["yc_arcsec"]),
        dx=float(args.pixel_scale_arcsec),
        dy=float(args.pixel_scale_arcsec),
        nx=max(16, int(round(float(saved_fov["xsize_arcsec"]) / abs(float(args.pixel_scale_arcsec))))),
        ny=max(16, int(round(float(saved_fov["ysize_arcsec"]) / abs(float(args.pixel_scale_arcsec))))),
    )

    target_header = _build_target_header(
        nx=int(geometry.nx),
        ny=int(geometry.ny),
        xc_arcsec=float(geometry.xc),
        yc_arcsec=float(geometry.yc),
        dx_arcsec=float(geometry.dx),
        dy_arcsec=float(geometry.dy),
        template_header=header,
    )
    target_header = _with_observer_wcs_keywords(
        target_header,
        observer_name=str(model_observer_meta.get("observer_name", args.observer or "earth")),
        hgln_obs_deg=float(model_observer_meta.get("observer_lonc_deg", 0.0)),
        hglt_obs_deg=float(model_observer_meta.get("observer_b0sun_deg", 0.0)),
        dsun_obs_m=float(model_observer_meta.get("observer_dsun_cm", 1.495978707e13)) / 100.0,
    )
    observed_cropped = _regrid_full_disk_to_target(observed, header, target_header)
    sigma_cropped = _regrid_full_disk_to_target(sigma_map, header, target_header)
    if np.isnan(observed_cropped).any():
        observed_cropped = np.nan_to_num(observed_cropped, nan=float(np.nanmedian(observed_cropped)))
    if np.isnan(sigma_cropped).any():
        fill_sigma = float(np.nanmedian(sigma_cropped))
        if not np.isfinite(fill_sigma) or fill_sigma <= 0:
            fill_sigma = float(np.nanmedian(sigma_map))
        sigma_cropped = np.nan_to_num(sigma_cropped, nan=fill_sigma)

    print(f"  Observer mode: {'saved metadata' if observer_overrides is None else 'overrides'} ({observer_source})")
    print(
        "  Geometry: "
        f"xc={float(geometry.xc):.3f} yc={float(geometry.yc):.3f} "
        f"dx={float(geometry.dx):.3f} dy={float(geometry.dy):.3f} "
        f"nx={int(geometry.nx)} ny={int(geometry.ny)}"
    )
    print(
        "  "
        + _format_psf_report(
            source=str(psf_source),
            bmaj_arcsec=psf_bmaj_arcsec,
            bmin_arcsec=psf_bmin_arcsec,
            bpa_deg=psf_bpa_deg,
            active_frequency_ghz=float(freq_ghz),
            ref_frequency_ghz=float(args.psf_ref_frequency_ghz) if args.psf_ref_frequency_ghz is not None else None,
            scale_inverse_frequency=bool(args.psf_scale_inverse_frequency and psf_allows_frequency_scaling),
        )
    )

    psf_kernel = None
    if psf_bmaj_arcsec is not None and psf_bmin_arcsec is not None and psf_bpa_deg is not None:
        psf_meta = _effective_psf_parameters(
            bmaj_arcsec=psf_bmaj_arcsec,
            bmin_arcsec=psf_bmin_arcsec,
            bpa_deg=psf_bpa_deg,
            active_frequency_ghz=float(freq_ghz),
            ref_frequency_ghz=float(args.psf_ref_frequency_ghz) if args.psf_ref_frequency_ghz is not None else None,
                scale_inverse_frequency=bool(args.psf_scale_inverse_frequency and psf_allows_frequency_scaling),
        )
        assert psf_meta is not None
        psf_kernel = _elliptical_gaussian_kernel(
            bmaj_arcsec=float(psf_meta["active_bmaj_arcsec"]),
            bmin_arcsec=float(psf_meta["active_bmin_arcsec"]),
            bpa_deg=float(psf_bpa_deg),
            dx_arcsec=float(geometry.dx),
            dy_arcsec=float(geometry.dy),
        )

    root_diag = {
        "artifact_kind": "pychmp_ab_scan_sparse_points",
        "spectral_domain": "mw",
        "spectral_label": f"{float(freq_ghz):.3f} GHz",
        "model_path": str(model_h5),
        "fits_file": str(fits_path),
        "ebtel_path": str(ebtel_path),
        "target_metric": str(args.target_metric),
        "frequency_ghz": float(freq_ghz),
        "map_xc_arcsec": float(geometry.xc),
        "map_yc_arcsec": float(geometry.yc),
        "map_dx_arcsec": float(geometry.dx),
        "map_dy_arcsec": float(geometry.dy),
        "noise_diagnostics": noise_diag,
        "observer_name": str(model_observer_meta.get("observer_name", args.observer or "earth")),
        "observer_lonc_deg": float(model_observer_meta.get("observer_lonc_deg", 0.0)),
        "observer_b0sun_deg": float(model_observer_meta.get("observer_b0sun_deg", 0.0)),
        "observer_dsun_cm": float(model_observer_meta.get("observer_dsun_cm", 1.495978707e13)),
        "observer_obs_time": target_header.get("DATE-OBS", ""),
        "search_mode": "adaptive_local_single_frequency",
        "a_start": float(args.a_start),
        "b_start": float(args.b_start),
        "da": float(args.da),
        "db": float(args.db),
        "a_range": [float(args.a_min), float(args.a_max)],
        "b_range": [float(args.b_min), float(args.b_max)],
        "threshold_metric": float(args.threshold_metric),
        "no_area": bool(args.no_area),
        "execution_policy": str(args.execution_policy),
        "execution_max_workers": None if args.max_workers is None else int(args.max_workers),
    }

    artifact_preexisting = artifact_h5.exists()
    existing_format = detect_scan_artifact_format(artifact_h5) if artifact_preexisting else None
    viewer_refresh_signal = Path(f"{artifact_h5}.refresh")
    viewer_heartbeat = _ViewerRefreshHeartbeat(viewer_refresh_signal)
    if artifact_h5.exists() and existing_format not in {None, "sparse"}:
        raise SystemExit(
            f"Existing artifact {artifact_h5} is rectangular; this adaptive example requires a sparse artifact path."
        )
    existing_run_history = load_run_history(artifact_h5) if artifact_preexisting else []
    if bool(args.recompute_existing) and artifact_h5.exists():
        print(f"Recompute existing: resetting sparse artifact at {artifact_h5}")
    if not artifact_h5.exists() or bool(args.recompute_existing):
        write_sparse_scan_file(
            artifact_h5,
            observed=observed_cropped,
            sigma_map=sigma_cropped,
            wcs_header=target_header,
            diagnostics=root_diag,
            point_records=[],
            run_history=existing_run_history,
        )
        viewer_heartbeat.set_phase("adaptive sparse artifact initialized")
        print(f"Initialized sparse artifact: {artifact_h5}")

    factory = _AdaptiveRendererFactory(
        model_path=str(model_h5),
        ebtel_path=str(ebtel_path),
        frequency_ghz=float(freq_ghz),
        tbase=float(args.tbase),
        nbase=float(args.nbase),
        geometry=_FactoryGeometry(
            xc=float(geometry.xc),
            yc=float(geometry.yc),
            dx=float(geometry.dx),
            dy=float(geometry.dy),
            nx=int(geometry.nx),
            ny=int(geometry.ny),
        ),
        observer_overrides=None
        if observer_overrides is None
        else _ObserverOverrideData(
            dsun_cm=None if getattr(observer_overrides, "dsun_cm", None) is None else float(observer_overrides.dsun_cm),
            lonc_deg=None if getattr(observer_overrides, "lonc_deg", None) is None else float(observer_overrides.lonc_deg),
            b0sun_deg=None if getattr(observer_overrides, "b0sun_deg", None) is None else float(observer_overrides.b0sun_deg),
        ),
        pixel_scale_arcsec=float(args.pixel_scale_arcsec),
        psf_kernel=None if psf_kernel is None else np.asarray(psf_kernel, dtype=float),
    )
    cache = _PersistentPointCache(
        artifact_h5=artifact_h5,
        observed=observed_cropped,
        sigma_map=sigma_cropped,
        target_header=target_header,
        diagnostics=root_diag,
        renderer_factory=factory,
        target_metric=str(args.target_metric),
        psf_source=str(psf_source),
        viewer_heartbeat=viewer_heartbeat,
    )
    try:
        reused_points = 0 if bool(args.recompute_existing) else cache.hydrate_from_existing()
    except ScanArtifactCompatibilityError as exc:
        raise SystemExit(str(exc)) from exc
    append_run_history_entry(
        artifact_h5,
        _build_run_history_entry(
            artifact_h5=artifact_h5,
            log_path=log_path,
            viewer_cmd_text=viewer_cmd_text,
            action=(
                "recompute"
                if bool(args.recompute_existing) and artifact_preexisting
                else ("resume" if artifact_preexisting and not bool(args.recompute_existing) else "create")
            ),
            target_metric=str(args.target_metric),
            recompute_existing=bool(args.recompute_existing),
        ),
    )
    print(f"Resume state: {reused_points} compatible point(s) loaded from the existing artifact")
    if auto_viewer_enabled:
        print("Viewer command above is shown for reference; pychmp-view will also be auto-launched unless it is already running.")
    else:
        print("Open the viewer command above now if you want to inspect the artifact while the run is in progress.")
    _maybe_launch_viewer("scan start")

    started = time.perf_counter()
    viewer_heartbeat.start("adaptive search running")
    try:
        result = search_local_minimum_ab(
            factory,
            observed_cropped,
            sigma_cropped,
            a_start=float(args.a_start),
            b_start=float(args.b_start),
            da=float(args.da),
            db=float(args.db),
            a_range=(float(args.a_min), float(args.a_max)),
            b_range=(float(args.b_min), float(args.b_max)),
            q0_min=float(args.q0_min),
            q0_max=float(args.q0_max),
            hard_q0_min=args.hard_q0_min,
            hard_q0_max=args.hard_q0_max,
            threshold=float(args.threshold),
            mask_type=str(args.mask_type),
            target_metric=str(args.target_metric),
            adaptive_bracketing=bool(args.adaptive_bracketing),
            q0_start=args.q0_start,
            q0_step=float(args.q0_step),
            max_bracket_steps=int(args.max_bracket_steps),
            threshold_metric=float(args.threshold_metric),
            no_area=bool(args.no_area),
            cache=cache,
            execution_policy=str(args.execution_policy),
            max_workers=args.max_workers,
            worker_chunksize=int(args.worker_chunksize),
        )
    except Exception:
        viewer_heartbeat.stop("adaptive search failed")
        raise
    viewer_heartbeat.stop("scan complete")
    elapsed = time.perf_counter() - started

    payload = load_scan_file(artifact_h5)
    if grid_png is not None or point_png is not None or bool(args.show_plot):
        plot_ab_scan_file(
            artifact_h5,
            payload=payload,
            out_grid_png=grid_png,
            out_point_png=point_png,
            show_plot=bool(args.show_plot),
        )

    boundary_axes = tuple(str(axis) for axis in result.best_boundary_axes)
    boundary_constrained = not bool(result.best_is_interior)

    print("\nAdaptive search complete")
    print(f"  Best point: a={float(result.best_a):.3f} b={float(result.best_b):.3f}")
    best_ai = int(np.nanargmin(result.objective_values) // result.objective_values.shape[1]) if np.any(np.isfinite(result.objective_values)) else 0
    best_bi = int(np.nanargmin(result.objective_values) % result.objective_values.shape[1]) if np.any(np.isfinite(result.objective_values)) else 0
    best_q0 = float(result.best_q0[best_ai, best_bi]) if result.best_q0.size else float("nan")
    best_metric = float(np.nanmin(result.objective_values)) if np.any(np.isfinite(result.objective_values)) else float("nan")
    print(f"  Best q0: {best_q0:.6f}")
    print(f"  Best {args.target_metric}: {best_metric:.6e}")
    print(f"  Phase 1 iterations: {int(result.n_phase1_iters)}")
    print(f"  Phase 2 iterations: {int(result.n_phase2_iters)}")
    print(f"  Confirmed interior minimum: {'yes' if result.best_is_interior else 'no'}")
    print(f"  Minimum certified: {'yes' if result.minimum_certified else 'no'}")
    print(f"  Termination reason: {result.termination_reason}")
    print(f"  Evaluated points: {int(result.evaluated_point_count)}")
    if boundary_axes:
        print(f"  Boundary-constrained best point: {', '.join(boundary_axes)}")
    if result.frontier_open_axes:
        print(f"  Open frontier axes: {', '.join(str(axis) for axis in result.frontier_open_axes)}")
    print(f"  Stored points: {len(payload.get('point_records', []))}")
    print(f"  Artifact H5: {artifact_h5}")
    if grid_png is not None:
        print(f"  Grid PNG: {grid_png}")
    if point_png is not None:
        print(f"  Point PNG: {point_png}")
    print(f"  Total elapsed: {elapsed:.3f}s")
    if not bool(result.minimum_certified):
        print(
            "WARNING: this run did not certify a closed local-minimum basin around the best point. "
            "Resume with wider a/b bounds if you want the search to continue expanding around the currently evaluated basin."
        )
        if bool(args.require_interior_best):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
