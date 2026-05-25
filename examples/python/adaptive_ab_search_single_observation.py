#!/usr/bin/env python
"""Adaptive `(a, b)` search against a single real observational map.

This is the generic observation-oriented entrypoint for the adaptive search.
It supports both MW external FITS observations and normalized model-refmap
selections for EUV/UV slices.
"""

from __future__ import annotations

import argparse
import queue
import hashlib
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
from astropy.io import fits

def _format_console_scalar(value: float, *, fixed_precision: int = 6) -> str:
    numeric = float(value)
    if not np.isfinite(numeric):
        return "nan"
    if numeric != 0.0 and abs(numeric) < 10 ** (-fixed_precision):
        return f"{numeric:.{max(1, fixed_precision)}e}"
    return f"{numeric:.{fixed_precision}f}"


def _build_command_compatibility_signature(argv: list[str]) -> str:
    normalized = json.dumps([str(item) for item in argv], separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _build_physical_compatibility_signature(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, separators=(",", ":"), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _canonical_json_sha256(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, separators=(",", ":"), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


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
    compatibility_signature = _build_command_compatibility_signature(effective_python_argv)
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
        "compatibility_signature": compatibility_signature,
        "viewer_command": str(viewer_cmd_text),
        "log_path": str(log_path),
        "target_metric": str(target_metric),
        "recompute_existing": bool(recompute_existing),
    }


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples"
DEFAULT_Q0_XATOL = 1e-3
DEFAULT_Q0_MAXITER = 200
for candidate in (REPO_ROOT, EXAMPLES_ROOT):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

from pychmp import (
    ABPointResult,
    GXRenderEUVAdapter,
    GXRenderMWAdapter,
    build_psf_kernel,
    build_tr_region_mask_from_blos,
    estimate_obs_map_noise,
    format_psf_report,
    load_obs_map,
    obs_map_noise_unit_label,
    resolve_render_geometry_via_gxrender,
    resolve_default_testdata_fixture_paths,
    search_local_minimum_ab,
    validate_obs_map_identity,
)
from pychmp.spectral import (
    default_euv_channels_for_instrument,
    euv_slice_request,
    mw_slice_request,
    parse_csv_floats,
    parse_csv_tokens,
    unique_preserve_order,
)
from pychmp.ab_scan_artifacts import (
    COMPATIBILITY_SIGNATURE_KEY,
    ScanArtifactCompatibilityError,
    append_point_record,
    append_run_history_entry,
    build_computed_point_payload,
    detect_scan_artifact_format,
    load_auxiliary_map_store_point_records,
    load_scan_file,
    load_run_history,
    point_record_matches_compatibility_signature,
    validate_scan_artifact_compatibility,
    validate_scan_artifact_reuse_preflight,
    write_point_scan_artifact,
)
from pychmp.geometry_policy import resolve_geometry_policy
from pychmp.metrics import compute_metrics, resolve_threshold_mask

try:
    from fit_q0_obs_map import (
        DEFAULT_A,
        DEFAULT_B,
        DEFAULT_NBASE,
        DEFAULT_TBASE,
        PSFConvolvedRenderer,
        _build_target_header,
        _compute_file_sha256,
        _effective_psf_parameters,
        _elliptical_gaussian_kernel,
        _extract_psf_from_header,
        _format_psf_report,
        _format_static_psf_report,
        _load_explicit_metric_mask,
        _load_model_identity,
        _make_trial_progress_reporter,
        _resolve_render_selection,
        _resolve_selected_psf_metadata,
        load_blos_reference_for_fov,
        _load_model_observer_metadata,
        _load_saved_fov_from_model,
        _regrid_full_disk_to_target,
        _resolve_observer_overrides,
        _with_observer_wcs_keywords,
    )
except ModuleNotFoundError:
    from examples.fit_q0_obs_map import (
        DEFAULT_A,
        DEFAULT_B,
        DEFAULT_NBASE,
        DEFAULT_TBASE,
        PSFConvolvedRenderer,
        _build_target_header,
        _compute_file_sha256,
        _effective_psf_parameters,
        _elliptical_gaussian_kernel,
        _extract_psf_from_header,
        _format_psf_report,
        _format_static_psf_report,
        _load_explicit_metric_mask,
        _load_model_identity,
        _make_trial_progress_reporter,
        _resolve_render_selection,
        _resolve_selected_psf_metadata,
        load_blos_reference_for_fov,
        _load_model_observer_metadata,
        _load_saved_fov_from_model,
        _regrid_full_disk_to_target,
        _resolve_observer_overrides,
        _with_observer_wcs_keywords,
    )

try:
    from plot_ab_scan_artifacts import plot_ab_scan_file
except ModuleNotFoundError:
    from examples.plot_ab_scan_artifacts import plot_ab_scan_file


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
    def __init__(self, signal_path: Path, *, interval_s: float = 2.0, slice_key: str | None = None) -> None:
        self._signal_path = Path(signal_path)
        self._interval_s = max(0.5, float(interval_s))
        self._slice_key = None if slice_key is None else str(slice_key).strip() or None
        self._phase = ""
        self._pending_points: tuple[tuple[float, float], ...] = ()
        self._active_point: tuple[float, float] | None = None
        self._live_trial_payload: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _write_signal(self) -> None:
        try:
            with self._lock:
                payload = {
                    "timestamp": float(time.time()),
                    "phase": str(self._phase),
                    "slice_key": self._slice_key,
                    "pending_points": [
                        {"a": float(a_value), "b": float(b_value)}
                        for a_value, b_value in self._pending_points
                    ],
                    "active_point": (
                        None
                        if self._active_point is None
                        else {"a": float(self._active_point[0]), "b": float(self._active_point[1])}
                    ),
                    "live_trials": None if self._live_trial_payload is None else dict(self._live_trial_payload),
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

    def set_active_point(self, a_value: float, b_value: float) -> None:
        with self._lock:
            self._active_point = (float(a_value), float(b_value))
            self._live_trial_payload = None
        self._write_signal()

    def update_live_trials(
        self,
        *,
        a_value: float,
        b_value: float,
        metric_name: str,
        q0_trials: list[float],
        metric_trials: list[float],
        active_trial_index: int | None = None,
        active_trial_q0: float | None = None,
    ) -> None:
        with self._lock:
            self._active_point = (float(a_value), float(b_value))
            self._live_trial_payload = {
                "a": float(a_value),
                "b": float(b_value),
                "metric_name": str(metric_name),
                "q0_trials": [float(value) for value in q0_trials],
                "metric_trials": [float(value) for value in metric_trials],
                "active_trial_index": (
                    None if active_trial_index is None else int(active_trial_index)
                ),
                "active_trial_q0": (
                    None if active_trial_q0 is None else float(active_trial_q0)
                ),
            }
        self._write_signal()

    def clear_active_trial(self) -> None:
        with self._lock:
            self._active_point = None
            self._live_trial_payload = None
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
            self._active_point = None
            self._live_trial_payload = None
        self._write_signal()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_s + 0.5)
            self._thread = None

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            self._write_signal()


class _PointRenderRecord:
    def __init__(self) -> None:
        self.raw_modeled_by_q0: dict[str, np.ndarray] = {}
        self.modeled_by_q0: dict[str, np.ndarray] = {}
        self.components_by_q0: dict[str, dict[str, Any]] = {}
        self.cube_by_q0: dict[str, dict[str, Any]] = {}


class _PointRenderStream:
    def __init__(self) -> None:
        self._records: dict[tuple[float, float], _PointRenderRecord] = {}
        self._lock = threading.Lock()

    def __getstate__(self) -> dict[str, Any]:
        return {"_records": self._records}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._records = dict(state.get("_records", {}))
        self._lock = threading.Lock()

    @staticmethod
    def _q0_key(q0_value: float) -> str:
        return f"{float(q0_value):.17g}"

    def _get_or_create(self, a_value: float, b_value: float) -> _PointRenderRecord:
        point_key = (float(a_value), float(b_value))
        with self._lock:
            record = self._records.get(point_key)
            if record is None:
                record = _PointRenderRecord()
                self._records[point_key] = record
            return record

    def record_render_pair(
        self,
        *,
        a_value: float,
        b_value: float,
        q0_value: float,
        raw_modeled: np.ndarray,
        modeled: np.ndarray,
    ) -> None:
        record = self._get_or_create(a_value, b_value)
        key = self._q0_key(q0_value)
        record.raw_modeled_by_q0[key] = np.asarray(raw_modeled, dtype=np.float32)
        record.modeled_by_q0[key] = np.asarray(modeled, dtype=np.float32)

    def record_components(
        self,
        *,
        a_value: float,
        b_value: float,
        q0_value: float,
        components: dict[str, Any],
    ) -> None:
        record = self._get_or_create(a_value, b_value)
        key = self._q0_key(q0_value)
        record.components_by_q0[key] = dict(components)

    def record_cube(
        self,
        *,
        a_value: float,
        b_value: float,
        q0_value: float,
        cube_payload: dict[str, Any],
    ) -> None:
        record = self._get_or_create(a_value, b_value)
        key = self._q0_key(q0_value)
        record.cube_by_q0[key] = dict(cube_payload)

    def pop_record(self, *, a_value: float, b_value: float) -> _PointRenderRecord | None:
        with self._lock:
            return self._records.pop((float(a_value), float(b_value)), None)


class _TrackedBaseRendererProxy:
    def __init__(self, base_renderer: Any, *, stream: _PointRenderStream, a_value: float, b_value: float) -> None:
        self._base = base_renderer
        self._stream = stream
        self._a_value = float(a_value)
        self._b_value = float(b_value)

    def render_components(self, q0: float) -> dict[str, Any]:
        payload = self._base.render_components(float(q0))
        self._stream.record_components(
            a_value=self._a_value,
            b_value=self._b_value,
            q0_value=float(q0),
            components=payload,
        )
        return payload

    def render_cube(self, q0: float) -> dict[str, Any]:
        payload = self._base.render_cube(float(q0))
        self._stream.record_cube(
            a_value=self._a_value,
            b_value=self._b_value,
            q0_value=float(q0),
            cube_payload=payload,
        )
        return payload

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


class _TrackedRendererProxy:
    def __init__(
        self,
        renderer: Any,
        *,
        stream: _PointRenderStream,
        a_value: float,
        b_value: float,
        renderer_factory: _AdaptiveRendererFactory,
        observed_template: np.ndarray,
        target_metric: str,
        psf_source: str,
        compatibility_signature: str,
        store_trial_map_cubes: bool,
    ) -> None:
        self._renderer = renderer
        self._stream = stream
        self._a_value = float(a_value)
        self._b_value = float(b_value)
        self._renderer_factory = renderer_factory
        self._observed_template = np.asarray(observed_template, dtype=float)
        self._target_metric = str(target_metric)
        self._psf_source = str(psf_source)
        self._compatibility_signature = str(compatibility_signature)
        self._store_trial_map_cubes = bool(store_trial_map_cubes)
        base = getattr(renderer, "_base", None)
        self._base = None
        if base is not None:
            self._base = _TrackedBaseRendererProxy(
                base,
                stream=stream,
                a_value=float(a_value),
                b_value=float(b_value),
            )

    def render_pair(self, q0: float) -> tuple[np.ndarray, np.ndarray]:
        q0_value = float(q0)
        if hasattr(self._renderer, "render_pair"):
            raw_modeled, modeled = self._renderer.render_pair(q0_value)
        else:
            modeled = self._renderer.render(q0_value)
            raw_modeled = modeled
        raw_arr = np.asarray(raw_modeled, dtype=np.float32)
        modeled_arr = np.asarray(modeled, dtype=np.float32)
        self._stream.record_render_pair(
            a_value=self._a_value,
            b_value=self._b_value,
            q0_value=q0_value,
            raw_modeled=raw_arr,
            modeled=modeled_arr,
        )
        return raw_arr, modeled_arr

    def render(self, q0: float) -> np.ndarray:
        _raw_modeled, modeled = self.render_pair(float(q0))
        return modeled

    def build_artifact_payload(self, point: ABPointResult) -> dict[str, Any]:
        stream_record = self._stream.pop_record(a_value=self._a_value, b_value=self._b_value)
        return _point_payload_from_result(
            point,
            renderer_factory=self._renderer_factory,
            observed_template=self._observed_template,
            target_metric=self._target_metric,
            psf_source=self._psf_source,
            compatibility_signature=self._compatibility_signature,
            store_trial_map_cubes=self._store_trial_map_cubes,
            stream_record=stream_record,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._renderer, name)


class _StreamingRendererFactory:
    def __init__(
        self,
        base_factory: _AdaptiveRendererFactory,
        *,
        stream: _PointRenderStream,
        observed_template: np.ndarray,
        target_metric: str,
        psf_source: str,
        compatibility_signature: str,
        store_trial_map_cubes: bool,
    ) -> None:
        self._base_factory = base_factory
        self._stream = stream
        self._observed_template = np.asarray(observed_template, dtype=float)
        self._target_metric = str(target_metric)
        self._psf_source = str(psf_source)
        self._compatibility_signature = str(compatibility_signature)
        self._store_trial_map_cubes = bool(store_trial_map_cubes)

    def __call__(self, a: float, b: float) -> Any:
        renderer = self._base_factory(float(a), float(b))
        return _TrackedRendererProxy(
            renderer,
            stream=self._stream,
            a_value=float(a),
            b_value=float(b),
            renderer_factory=self._base_factory,
            observed_template=self._observed_template,
            target_metric=self._target_metric,
            psf_source=self._psf_source,
            compatibility_signature=self._compatibility_signature,
            store_trial_map_cubes=self._store_trial_map_cubes,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_factory, name)


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
class _ObservationRequest:
    source_mode: str
    obs_path: Path | None
    obs_map_id: str | None
    model_h5: Path
    ebtel_path: Path | None


@dataclass(frozen=True)
class _AdaptiveRendererFactory:
    model_path: str
    ebtel_path: str
    spectral_domain: str
    spectral_label: str
    frequency_ghz: float | None
    wavelength_angstrom: float | None
    euv_channel: str | None
    euv_instrument: str | None
    euv_response_sav: str | None
    render_frequencies_ghz: tuple[float, ...]
    render_channels: tuple[str, ...]
    tbase: float
    nbase: float
    geometry: _FactoryGeometry
    observer_overrides: _ObserverOverrideData | None
    observer_name: str | None
    pixel_scale_arcsec: float
    psf_kernel: np.ndarray | None
    tr_region_mask: np.ndarray | None = None

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
        if str(self.spectral_domain).lower() == "mw":
            if self.frequency_ghz is None:
                raise ValueError("MW adaptive renderer factory requires frequency_ghz")
            base = GXRenderMWAdapter(
                model_path=self.model_path,
                ebtel_path=self.ebtel_path,
                frequency_ghz=float(self.frequency_ghz),
                render_frequencies_ghz=self.render_frequencies_ghz,
                tbase=float(self.tbase),
                nbase=float(self.nbase),
                a=float(a),
                b=float(b),
                geometry=geometry,
                observer=observer,
                observer_name=self.observer_name,
                pixel_scale_arcsec=float(self.pixel_scale_arcsec),
            )
            if self.psf_kernel is None:
                return base
            return PSFConvolvedRenderer(base, np.asarray(self.psf_kernel, dtype=float))

        if self.euv_channel is None:
            raise ValueError(
                f"non-MW adaptive renderer factory for {self.spectral_label!r} is missing euv_channel metadata"
            )
        base = GXRenderEUVAdapter(
            model_path=self.model_path,
            channel=str(self.euv_channel),
            render_channels=self.render_channels,
            instrument=str(self.euv_instrument or "AIA"),
            response_sav=self.euv_response_sav,
            ebtel_path=self.ebtel_path,
            tbase=float(self.tbase),
            nbase=float(self.nbase),
            a=float(a),
            b=float(b),
            geometry=geometry,
            observer=observer,
            observer_name=self.observer_name,
            tr_region_mask=None if self.tr_region_mask is None else np.asarray(self.tr_region_mask, dtype=bool),
            pixel_scale_arcsec=float(self.pixel_scale_arcsec),
        )
        if self.psf_kernel is None:
            return base
        return PSFConvolvedRenderer(base, np.asarray(self.psf_kernel, dtype=float))


def _default_testdata_repo(repo_root: Path) -> Path:
    return repo_root.parent / "pyGXrender-test-data"


def _coerce_path(value: Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def _default_testdata_roots(repo_root: Path, *, testdata_repo: Path | None) -> tuple[Path, Path, Path]:
    resolved_testdata_repo = testdata_repo or _default_testdata_repo(repo_root)
    eovsa_root = resolved_testdata_repo / "raw" / "eovsa_maps"
    model_root = resolved_testdata_repo / "raw" / "models"
    ebtel_root = resolved_testdata_repo / "raw" / "ebtel" / "ebtel_gxsimulator_euv"
    return eovsa_root, model_root, ebtel_root


def _resolve_observation_request(args: argparse.Namespace, *, repo_root: Path) -> _ObservationRequest:
    positional_fits = _coerce_path(args.fits_file)
    explicit_obs_path = _coerce_path(args.obs_path)
    model_h5 = _coerce_path(args.model_h5)
    ebtel_path = _coerce_path(args.ebtel_path)
    testdata_repo = _coerce_path(args.testdata_repo)
    obs_map_id = None if args.obs_map_id is None else str(args.obs_map_id).strip() or None
    explicit_source = None if args.obs_source is None else str(args.obs_source).strip().lower() or None

    obs_path = explicit_obs_path or positional_fits

    if explicit_source is None:
        explicit_source = "model_refmap" if obs_map_id is not None else "external_fits"

    _eovsa_root, model_root, ebtel_root = _default_testdata_roots(repo_root, testdata_repo=testdata_repo)
    _default_eovsa_fits, default_model_h5, default_ebtel_path = resolve_default_testdata_fixture_paths(
        repo_root=repo_root,
        testdata_repo=testdata_repo,
    )

    if model_h5 is None:
        if default_model_h5 is None:
            raise SystemExit(
                f"Default CHR test-data model not found under {model_root}; "
                "install the 2020-11-26 CHR fixture set or pass --model-h5"
            )
        model_h5 = default_model_h5
    if ebtel_path is None:
        ebtel_path = default_ebtel_path

    return _ObservationRequest(
        source_mode=str(explicit_source),
        obs_path=None if obs_path is None else obs_path.resolve(),
        obs_map_id=obs_map_id,
        model_h5=model_h5.resolve(),
        ebtel_path=None if ebtel_path is None else ebtel_path.resolve(),
    )


def _default_artifact_stem(obs_request: _ObservationRequest, *, target_metric: str) -> str:
    source_token = (
        obs_request.obs_path.stem
        if obs_request.obs_path is not None
        else (obs_request.obs_map_id or "observation")
    )
    return f"{source_token}_adaptive_ab_{target_metric}"


def _resolve_render_slice_requests(
    *,
    domain: str,
    frequency_ghz: float | None,
    euv_channel: str | None,
    euv_instrument: str | None,
    all_channels: bool,
    render_channels_csv: str | None,
    render_frequencies_csv: str | None,
) -> tuple[list[dict[str, Any]], tuple[float, ...], tuple[str, ...]]:
    resolved_domain = str(domain).strip().lower()
    if resolved_domain == "mw":
        if all_channels:
            raise SystemExit("--all-channels is only defined for fixed-channel EUV/UV instruments")
        if frequency_ghz is None:
            raise SystemExit("MW rendering requires a target observation frequency")
        extra_freqs = parse_csv_floats(render_frequencies_csv, option_name="--render-frequencies-ghz")
        requests = [mw_slice_request(float(frequency_ghz), is_target=True)]
        for freq in extra_freqs:
            if not np.isclose(float(freq), float(frequency_ghz), rtol=0.0, atol=1e-12):
                requests.append(mw_slice_request(float(freq), is_target=False))
        return [item.as_descriptor() for item in requests], tuple(float(item.frequency_ghz) for item in requests), tuple()

    requested_channels = list(parse_csv_tokens(render_channels_csv, option_name="--render-channels"))
    if all_channels:
        requested_channels.extend(default_euv_channels_for_instrument(euv_instrument))
        if not requested_channels:
            raise SystemExit(
                f"--all-channels is not known for EUV/UV instrument {euv_instrument!r}; use --render-channels instead"
            )
    if euv_channel:
        requested_channels.insert(0, str(euv_channel))
    channels = tuple(str(value) for value in unique_preserve_order(requested_channels))
    if not channels:
        raise SystemExit("EUV/UV rendering requires a target channel")
    requests = [
        euv_slice_request(channel, domain=resolved_domain, is_target=(str(channel) == str(euv_channel)))
        for channel in channels
    ]
    return [item.as_descriptor() for item in requests], tuple(), channels


def _resolve_existing_file(path_text: str | None) -> Path | None:
    if path_text is None:
        return None
    candidate = Path(path_text).expanduser()
    try:
        resolved = candidate.resolve()
    except Exception:
        resolved = candidate
    return resolved if resolved.exists() else None


def _build_synthetic_map_identity(
    *,
    renderer_factory: _AdaptiveRendererFactory,
    a_value: float,
    b_value: float,
    q0_value: float,
    domain_label: str,
    channel_or_frequency_label: str,
    map_role: str,
) -> dict[str, Any]:
    geometry = getattr(renderer_factory, "geometry", None)
    spectral_domain = getattr(renderer_factory, "spectral_domain", "unknown")
    spectral_label = getattr(renderer_factory, "spectral_label", str(channel_or_frequency_label))
    observer_name = getattr(renderer_factory, "observer_name", None)
    render_frequencies_ghz = tuple(getattr(renderer_factory, "render_frequencies_ghz", ()) or ())
    render_channels = tuple(getattr(renderer_factory, "render_channels", ()) or ())
    return {
        "schema": "pychmp.synthetic_map_db.v1",
        "spectral_domain": str(spectral_domain),
        "spectral_label": str(spectral_label),
        "domain_label": str(domain_label),
        "channel_or_frequency": str(channel_or_frequency_label),
        "map_role": str(map_role),
        "a": float(a_value),
        "b": float(b_value),
        "q0": float(q0_value),
        "geometry": {
            "xc": float(getattr(geometry, "xc", np.nan)),
            "yc": float(getattr(geometry, "yc", np.nan)),
            "dx": float(getattr(geometry, "dx", np.nan)),
            "dy": float(getattr(geometry, "dy", np.nan)),
            "nx": int(getattr(geometry, "nx", 0)),
            "ny": int(getattr(geometry, "ny", 0)),
        },
        "observer_name": None if observer_name is None else str(observer_name),
        "render_frequencies_ghz": [float(v) for v in render_frequencies_ghz],
        "render_channels": [str(v) for v in render_channels],
    }


def _lookup_stream_value_by_q0(
    values_by_q0: dict[str, Any],
    q0_value: float,
) -> Any | None:
    exact_key = _PointRenderStream._q0_key(float(q0_value))
    if exact_key in values_by_q0:
        return values_by_q0[exact_key]
    target = float(q0_value)
    for key_text, value in values_by_q0.items():
        try:
            candidate = float(key_text)
        except Exception:
            continue
        if np.isclose(candidate, target, rtol=0.0, atol=1e-12):
            return value
    return None


def _point_payload_from_result(
    point: ABPointResult,
    *,
    renderer_factory: _AdaptiveRendererFactory,
    observed_template: np.ndarray,
    target_metric: str,
    psf_source: str,
    compatibility_signature: str,
    store_trial_map_cubes: bool = False,
    stream_record: _PointRenderRecord,
) -> dict[str, Any]:
    point_a = float(point.a)
    point_b = float(point.b)
    point_q0 = float(point.q0)
    trial_q0_values = [float(v) for v in point.trial_q0]
    raw_modeled_best = _lookup_stream_value_by_q0(stream_record.raw_modeled_by_q0, point_q0)
    modeled_best = _lookup_stream_value_by_q0(stream_record.modeled_by_q0, point_q0)
    if raw_modeled_best is None or modeled_best is None:
        raise RuntimeError("point payload stream record is missing the best rendered maps")
    modeled_best = np.asarray(modeled_best, dtype=float)
    raw_modeled_best = np.asarray(raw_modeled_best, dtype=float)
    residual = np.asarray(modeled_best - np.asarray(observed_template, dtype=float), dtype=float)
    fit_metric_trials = tuple(float(v) for v in point.trial_objective_values)
    fit_chi2_trials = tuple(float(v) for v in point.trial_chi2_values)
    fit_rho2_trials = tuple(float(v) for v in point.trial_rho2_values)
    fit_eta2_trials = tuple(float(v) for v in point.trial_eta2_values)
    trial_raw_modeled_maps = None
    trial_modeled_maps = None
    trial_residual_maps = None
    trial_euv_coronal_maps = None
    trial_euv_tr_maps = None
    map_store_arrays: dict[str, np.ndarray] = {}
    synthetic_map_keys: list[dict[str, Any]] = []

    def _register_synthetic_map(*, identity: dict[str, Any], array: np.ndarray, label: str) -> None:
        machine_key = _canonical_json_sha256(identity)
        map_store_name = f"synthetic/{machine_key}"
        map_store_arrays[map_store_name] = np.asarray(array, dtype=np.float32)
        synthetic_map_keys.append(
            {
                "label": str(label),
                "map_store_array": str(map_store_name),
                "machine_key": str(machine_key),
                "identity": dict(identity),
            }
        )

    cube_payload = None
    cube_payload = _lookup_stream_value_by_q0(stream_record.cube_by_q0, point_q0)
    if isinstance(cube_payload, dict):
        for freq, rendered in dict(cube_payload.get("raw_modeled_by_frequency", {})).items():
            freq_label = f"{float(freq):.6f}ghz"
            _register_synthetic_map(
                identity=_build_synthetic_map_identity(
                    renderer_factory=renderer_factory,
                    a_value=point_a,
                    b_value=point_b,
                    q0_value=point_q0,
                    domain_label="mw",
                    channel_or_frequency_label=freq_label,
                    map_role="raw_modeled_best",
                ),
                array=np.asarray(rendered, dtype=np.float32),
                label=f"MW {freq_label} raw modeled",
            )
    components = None
    components = _lookup_stream_value_by_q0(stream_record.components_by_q0, point_q0)
    if isinstance(components, dict):
        for channel, rendered in dict(components.get("rendered_by_channel", {})).items():
            _register_synthetic_map(
                identity=_build_synthetic_map_identity(
                    renderer_factory=renderer_factory,
                    a_value=point_a,
                    b_value=point_b,
                    q0_value=point_q0,
                    domain_label="euv",
                    channel_or_frequency_label=str(channel),
                    map_role="rendered_best",
                ),
                array=np.asarray(rendered, dtype=np.float32),
                label=f"EUV {channel} rendered",
            )
        for channel, rendered in dict(components.get("flux_corona_by_channel", {})).items():
            _register_synthetic_map(
                identity=_build_synthetic_map_identity(
                    renderer_factory=renderer_factory,
                    a_value=point_a,
                    b_value=point_b,
                    q0_value=point_q0,
                    domain_label="euv",
                    channel_or_frequency_label=str(channel),
                    map_role="flux_corona_best",
                ),
                array=np.asarray(rendered, dtype=np.float32),
                label=f"EUV {channel} coronal flux",
            )
        for channel, rendered in dict(components.get("flux_tr_by_channel", {})).items():
            _register_synthetic_map(
                identity=_build_synthetic_map_identity(
                    renderer_factory=renderer_factory,
                    a_value=point_a,
                    b_value=point_b,
                    q0_value=point_q0,
                    domain_label="euv",
                    channel_or_frequency_label=str(channel),
                    map_role="flux_tr_best",
                ),
                array=np.asarray(rendered, dtype=np.float32),
                label=f"EUV {channel} TR flux",
            )

    trial_raw_by_q0: dict[str, np.ndarray] = {}
    trial_modeled_by_q0: dict[str, np.ndarray] = {}
    if stream_record is not None:
        trial_raw_by_q0 = dict(stream_record.raw_modeled_by_q0)
        trial_modeled_by_q0 = dict(stream_record.modeled_by_q0)

    if trial_q0_values and bool(store_trial_map_cubes):
        raw_trials: list[np.ndarray] = []
        modeled_trials: list[np.ndarray] = []
        residual_trials: list[np.ndarray] = []
        euv_coronal_trials: list[np.ndarray] = []
        euv_tr_trials: list[np.ndarray] = []
        for trial_index, q0_value in enumerate(trial_q0_values):
            raw_trial = _lookup_stream_value_by_q0(trial_raw_by_q0, q0_value)
            modeled_trial = _lookup_stream_value_by_q0(trial_modeled_by_q0, q0_value)
            if raw_trial is None or modeled_trial is None:
                raise RuntimeError("point payload stream record is missing trial rendered maps")
            raw_trial_arr = np.asarray(raw_trial, dtype=np.float32)
            modeled_trial_arr = np.asarray(modeled_trial, dtype=np.float32)
            raw_trials.append(raw_trial_arr)
            modeled_trials.append(modeled_trial_arr)
            residual_trials.append(modeled_trial_arr - np.asarray(observed_template, dtype=np.float32))

            trial_components = _lookup_stream_value_by_q0(stream_record.components_by_q0, q0_value)
            if isinstance(trial_components, dict):
                components = trial_components
                coronal = components.get("flux_corona")
                tr_flux = components.get("flux_tr")
                if coronal is not None and tr_flux is not None:
                    euv_coronal_trials.append(np.asarray(coronal, dtype=np.float32))
                    euv_tr_trials.append(np.asarray(tr_flux, dtype=np.float32))
                for channel, rendered in dict(components.get("rendered_by_channel", {})).items():
                    _register_synthetic_map(
                        identity=_build_synthetic_map_identity(
                            renderer_factory=renderer_factory,
                            a_value=point_a,
                            b_value=point_b,
                            q0_value=float(q0_value),
                            domain_label="euv",
                            channel_or_frequency_label=str(channel),
                            map_role=f"trial_{trial_index:03d}_rendered",
                        ),
                        array=np.asarray(rendered, dtype=np.float32),
                        label=f"EUV {channel} trial {trial_index:03d} rendered",
                    )
            trial_cube = _lookup_stream_value_by_q0(stream_record.cube_by_q0, q0_value)
            if isinstance(trial_cube, dict):
                cube_payload = trial_cube
                for freq, rendered in dict(cube_payload.get("raw_modeled_by_frequency", {})).items():
                    freq_label = f"{float(freq):.6f}ghz"
                    _register_synthetic_map(
                        identity=_build_synthetic_map_identity(
                            renderer_factory=renderer_factory,
                            a_value=point_a,
                            b_value=point_b,
                            q0_value=float(q0_value),
                            domain_label="mw",
                            channel_or_frequency_label=freq_label,
                            map_role=f"trial_{trial_index:03d}_raw_modeled",
                        ),
                        array=np.asarray(rendered, dtype=np.float32),
                        label=f"MW {freq_label} trial {trial_index:03d} raw modeled",
                    )
        if raw_trials and len(raw_trials) == len(trial_q0_values):
            trial_raw_modeled_maps = np.stack(raw_trials, axis=0)
            trial_modeled_maps = np.stack(modeled_trials, axis=0)
            trial_residual_maps = np.stack(residual_trials, axis=0)
        if euv_coronal_trials and len(euv_coronal_trials) == len(trial_q0_values):
            trial_euv_coronal_maps = np.stack(euv_coronal_trials, axis=0)
            trial_euv_tr_maps = np.stack(euv_tr_trials, axis=0)
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
        "synthetic_map_db_version": 1,
        "synthetic_map_keys": synthetic_map_keys,
        "store_trial_map_cubes": bool(store_trial_map_cubes),
        COMPATIBILITY_SIGNATURE_KEY: str(compatibility_signature),
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
        trial_raw_modeled_maps=trial_raw_modeled_maps,
        trial_modeled_maps=trial_modeled_maps,
        trial_residual_maps=trial_residual_maps,
        trial_euv_coronal_maps=trial_euv_coronal_maps,
        trial_euv_tr_maps=trial_euv_tr_maps,
        map_store_arrays=map_store_arrays,
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


class _ArtifactWriteDispatcher:
    def __init__(
        self,
        *,
        artifact_h5: Path,
        observed: np.ndarray,
        sigma_map: np.ndarray,
        target_header: Any,
        diagnostics: dict[str, Any],
        blos_reference: tuple[np.ndarray, fits.Header] | None,
        psf_kernel: np.ndarray | None,
    ) -> None:
        self._artifact_h5 = Path(artifact_h5)
        self._observed = np.asarray(observed, dtype=float)
        self._sigma_map = np.asarray(sigma_map, dtype=float)
        self._target_header = target_header
        self._diagnostics = dict(diagnostics)
        self._blos_reference = blos_reference
        self._psf_kernel = None if psf_kernel is None else np.asarray(psf_kernel, dtype=float)
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=16)
        self._closed = False
        self._failed: BaseException | None = None
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, name="pychmp-artifact-dispatcher", daemon=True)
        self._worker.start()

    def _raise_if_failed(self) -> None:
        with self._lock:
            if self._failed is not None:
                raise RuntimeError("artifact write dispatcher failed") from self._failed

    def write_point(self, point_payload: dict[str, Any]) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("artifact write dispatcher is closed")
        self._raise_if_failed()
        # Bounded queue applies backpressure while still allowing the caller to
        # continue without waiting for each point write to complete.
        self._queue.put(dict(point_payload))

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(None)
        self._queue.join()
        self._worker.join(timeout=5.0)
        self._raise_if_failed()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            try:
                append_point_record(
                    self._artifact_h5,
                    observed=self._observed,
                    sigma_map=self._sigma_map,
                    wcs_header=self._target_header,
                    diagnostics=self._diagnostics,
                    blos_reference=self._blos_reference,
                    psf_kernel=self._psf_kernel,
                    point_payload=item,
                )
            except BaseException as exc:
                with self._lock:
                    if self._failed is None:
                        self._failed = exc
            finally:
                self._queue.task_done()


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


def _target_metric_value(metrics: Any, target_metric: str) -> float:
    metric_name = str(target_metric)
    if metric_name == "chi2":
        return float(metrics.chi2)
    if metric_name == "rho2":
        return float(metrics.rho2)
    if metric_name == "eta2":
        return float(metrics.eta2)
    raise ValueError(f"unsupported target metric: {target_metric!r}")


def _rescore_auxiliary_map_record(
    record: dict[str, Any],
    *,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    threshold: float,
    explicit_mask: np.ndarray | None,
    target_metric: str,
) -> tuple[ABPointResult, dict[str, Any]] | None:
    trial_q0 = tuple(float(value) for value in record.get("fit_q0_trials", ()))
    trial_metric_trials = tuple(float(value) for value in record.get("fit_metric_trials", ()))
    trial_maps_raw = record.get("trial_modeled_maps")
    observed_arr = np.asarray(observed, dtype=float)
    sigma_arr = np.asarray(sigma_map, dtype=float)
    explicit_mask_arr = None if explicit_mask is None else np.asarray(explicit_mask, dtype=bool)
    mask_fn = resolve_threshold_mask("union")

    if trial_maps_raw is None:
        diagnostics = dict(record.get("diagnostics") or {})
        source_metric = str(record.get("target_metric", diagnostics.get("target_metric", ""))).strip().lower()
        if not trial_q0 or source_metric != str(target_metric).strip().lower():
            return None
        if len(trial_metric_trials) != len(trial_q0):
            return None
        modeled_best_raw = record.get("modeled_best")
        if modeled_best_raw is None:
            return None
        finite_indices = [idx for idx, value in enumerate(trial_metric_trials) if np.isfinite(float(value))]
        if not finite_indices:
            return None
        best_index = min(finite_indices, key=lambda idx: float(trial_metric_trials[idx]))
        best_q0 = float(trial_q0[best_index])
        best_objective = float(trial_metric_trials[best_index])
        best_map = np.asarray(modeled_best_raw, dtype=float)
        trial_chi2 = tuple(float(value) for value in record.get("fit_chi2_trials", trial_metric_trials))
        trial_rho2 = tuple(float(value) for value in record.get("fit_rho2_trials", (np.nan,) * len(trial_q0)))
        trial_eta2 = tuple(float(value) for value in record.get("fit_eta2_trials", (np.nan,) * len(trial_q0)))
        diagnostics.update(
            {
                "target_metric": str(target_metric),
                "target_metric_value": float(best_objective),
                "chi2": float(trial_chi2[best_index]) if best_index < len(trial_chi2) else float("nan"),
                "rho2": float(trial_rho2[best_index]) if best_index < len(trial_rho2) else float("nan"),
                "eta2": float(trial_eta2[best_index]) if best_index < len(trial_eta2) else float("nan"),
                "map_store_reused": True,
                "map_store_reused_without_trial_maps": True,
                "map_store_source_slice_key": record.get("source_slice_key"),
                "map_store_source_search_id": record.get("source_search_id"),
            }
        )
        residual = np.asarray(best_map, dtype=float) - observed_arr
        point = ABPointResult(
            a=float(record["a"]),
            b=float(record["b"]),
            q0=float(best_q0),
            objective_value=float(best_objective),
            metrics=_MetricValues(
                chi2=float(diagnostics.get("chi2", np.nan)),
                rho2=float(diagnostics.get("rho2", np.nan)),
                eta2=float(diagnostics.get("eta2", np.nan)),
            ),
            target_metric=str(target_metric),
            success=True,
            nfev=int(len(trial_q0)),
            nit=0,
            message="reused from saved trial metrics",
            used_adaptive_bracketing=bool(record.get("used_adaptive_bracketing", False)),
            bracket_found=bool(record.get("bracket_found", False)),
            bracket=None if record.get("bracket") is None else tuple(float(v) for v in record.get("bracket")),
            trial_q0=tuple(float(v) for v in trial_q0),
            trial_objective_values=tuple(float(v) for v in trial_metric_trials),
            trial_chi2_values=trial_chi2,
            trial_rho2_values=trial_rho2,
            trial_eta2_values=trial_eta2,
            elapsed_seconds=0.0,
        )
        payload = build_computed_point_payload(
            a_value=float(point.a),
            b_value=float(point.b),
            a_index=int(record.get("a_index", 0)),
            b_index=int(record.get("b_index", 0)),
            q0=float(point.q0),
            success=True,
            status="computed",
            modeled_best=np.asarray(best_map, dtype=float),
            raw_modeled_best=np.asarray(record.get("raw_modeled_best", best_map), dtype=float),
            residual=residual,
            fit_q0_trials=tuple(float(v) for v in trial_q0),
            fit_metric_trials=tuple(float(v) for v in trial_metric_trials),
            fit_chi2_trials=trial_chi2,
            fit_rho2_trials=trial_rho2,
            fit_eta2_trials=trial_eta2,
            nfev=int(len(trial_q0)),
            nit=0,
            message="reused from saved trial metrics",
            used_adaptive_bracketing=bool(record.get("used_adaptive_bracketing", False)),
            bracket_found=bool(record.get("bracket_found", False)),
            bracket=None if record.get("bracket") is None else tuple(float(v) for v in record.get("bracket")),
            target_metric=str(target_metric),
            diagnostics=diagnostics,
        )
        return point, payload

    if not trial_q0:
        return None
    trial_maps = np.asarray(trial_maps_raw, dtype=float)
    if trial_maps.ndim != 3 or int(trial_maps.shape[0]) != len(trial_q0):
        return None

    rescored: list[tuple[float, Any, float, np.ndarray]] = []
    for q0_value, modeled in zip(trial_q0, trial_maps, strict=False):
        modeled_arr = np.asarray(modeled, dtype=float)
        try:
            mask = explicit_mask_arr if explicit_mask_arr is not None else mask_fn(observed_arr, modeled_arr, float(threshold))
            metrics = compute_metrics(observed_arr, modeled_arr, sigma_arr, mask)
            objective = _target_metric_value(metrics, target_metric)
        except Exception:
            continue
        if np.isfinite(objective):
            rescored.append((float(q0_value), metrics, float(objective), modeled_arr))
    if not rescored:
        return None

    best_q0, best_metrics, best_objective, best_map = min(rescored, key=lambda item: item[2])
    diagnostics = dict(record.get("diagnostics") or {})
    diagnostics.update(
        {
            "chi2": float(best_metrics.chi2),
            "rho2": float(best_metrics.rho2),
            "eta2": float(best_metrics.eta2),
            "target_metric": str(target_metric),
            "target_metric_value": float(best_objective),
            "map_store_reused": True,
            "map_store_source_slice_key": record.get("source_slice_key"),
            "map_store_source_search_id": record.get("source_search_id"),
        }
    )
    trial_metric_values = tuple(float(objective) for _q0, _metrics, objective, _modeled in rescored)
    trial_chi2 = tuple(float(metrics.chi2) for _q0, metrics, _objective, _modeled in rescored)
    trial_rho2 = tuple(float(metrics.rho2) for _q0, metrics, _objective, _modeled in rescored)
    trial_eta2 = tuple(float(metrics.eta2) for _q0, metrics, _objective, _modeled in rescored)
    ordered_trial_q0 = tuple(float(q0) for q0, _metrics, _objective, _modeled in rescored)
    ordered_trial_maps = np.stack([np.asarray(modeled, dtype=float) for _q0, _metrics, _objective, modeled in rescored], axis=0)
    residual = np.asarray(best_map, dtype=float) - observed_arr

    point = ABPointResult(
        a=float(record["a"]),
        b=float(record["b"]),
        q0=float(best_q0),
        objective_value=float(best_objective),
        metrics=_MetricValues(chi2=float(best_metrics.chi2), rho2=float(best_metrics.rho2), eta2=float(best_metrics.eta2)),
        target_metric=str(target_metric),
        success=True,
        nfev=int(len(rescored)),
        nit=0,
        message="reused from map_store",
        used_adaptive_bracketing=bool(record.get("used_adaptive_bracketing", False)),
        bracket_found=bool(record.get("bracket_found", False)),
        bracket=None if record.get("bracket") is None else tuple(float(v) for v in record.get("bracket")),
        trial_q0=ordered_trial_q0,
        trial_objective_values=trial_metric_values,
        trial_chi2_values=trial_chi2,
        trial_rho2_values=trial_rho2,
        trial_eta2_values=trial_eta2,
        elapsed_seconds=0.0,
    )
    payload = build_computed_point_payload(
        a_value=float(point.a),
        b_value=float(point.b),
        a_index=int(record.get("a_index", 0)),
        b_index=int(record.get("b_index", 0)),
        q0=float(point.q0),
        success=True,
        status="computed",
        modeled_best=np.asarray(best_map, dtype=float),
        raw_modeled_best=np.asarray(record.get("raw_modeled_best", best_map), dtype=float),
        residual=residual,
        fit_q0_trials=ordered_trial_q0,
        fit_metric_trials=trial_metric_values,
        fit_chi2_trials=trial_chi2,
        fit_rho2_trials=trial_rho2,
        fit_eta2_trials=trial_eta2,
        trial_raw_modeled_maps=np.asarray(record.get("trial_raw_modeled_maps", ordered_trial_maps), dtype=float),
        trial_modeled_maps=ordered_trial_maps,
        trial_residual_maps=ordered_trial_maps - observed_arr[None, :, :],
        nfev=int(len(rescored)),
        nit=0,
        message="reused from map_store",
        used_adaptive_bracketing=bool(record.get("used_adaptive_bracketing", False)),
        bracket_found=bool(record.get("bracket_found", False)),
        bracket=None if record.get("bracket") is None else tuple(float(v) for v in record.get("bracket")),
        target_metric=str(target_metric),
        diagnostics=diagnostics,
    )
    return point, payload


def _collect_start_over_seed_point_payloads(
    *,
    artifact_h5: Path,
    slice_key: str | None,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
    threshold: float,
    explicit_mask: np.ndarray | None,
    target_metric: str,
    compatibility_signature: str,
) -> list[dict[str, Any]]:
    if not artifact_h5.exists() or not slice_key:
        return []
    try:
        current_payload = load_scan_file(artifact_h5, slice_key=slice_key)
    except KeyError:
        return []
    validate_scan_artifact_compatibility(
        current_payload,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=wcs_header,
        diagnostics=diagnostics,
        artifact_path=artifact_h5,
    )
    if bool(dict(current_payload.get("diagnostics") or {}).get("render_only_slice", False)):
        return []

    payloads_by_key: dict[tuple[float, float], dict[str, Any]] = {}
    for search in current_payload.get("search_records", []):
        search_id = str(search.get("search_id", "")).strip()
        if not search_id:
            continue
        try:
            search_payload = load_scan_file(artifact_h5, slice_key=slice_key, search_id=search_id)
        except KeyError:
            continue
        for record in search_payload.get("point_records", []):
            key = (float(record["a"]), float(record["b"]))
            if key in payloads_by_key:
                continue
            rescored = _rescore_auxiliary_map_record(
                {
                    **dict(record),
                    "source_slice_key": slice_key,
                    "source_search_id": search_id,
                },
                observed=observed,
                sigma_map=sigma_map,
                threshold=float(threshold),
                explicit_mask=explicit_mask,
                target_metric=target_metric,
            )
            if rescored is None:
                continue
            _point, point_payload = rescored
            point_payload["diagnostics"] = {
                **dict(point_payload.get("diagnostics") or {}),
                COMPATIBILITY_SIGNATURE_KEY: compatibility_signature,
            }
            payloads_by_key[key] = point_payload
    return [payloads_by_key[key] for key in sorted(payloads_by_key.keys())]


class _PersistentPointCache(MutableMapping[tuple[float, float], ABPointResult]):
    def __init__(
        self,
        *,
        artifact_h5: Path,
        observed: np.ndarray,
        sigma_map: np.ndarray,
        target_header: Any,
        diagnostics: dict[str, Any],
        blos_reference: tuple[np.ndarray, fits.Header] | None,
        renderer_factory: _AdaptiveRendererFactory,
        target_metric: str,
        psf_source: str,
        psf_kernel: np.ndarray | None,
        compatibility_signature: str,
        store_trial_map_cubes: bool = False,
        viewer_heartbeat: _ViewerRefreshHeartbeat | None = None,
    ) -> None:
        self._artifact_h5 = Path(artifact_h5)
        self._observed = np.asarray(observed, dtype=float)
        self._sigma_map = np.asarray(sigma_map, dtype=float)
        self._target_header = target_header
        self._diagnostics = dict(diagnostics)
        self._blos_reference = blos_reference
        self._renderer_factory = renderer_factory
        self._target_metric = str(target_metric)
        self._psf_source = str(psf_source)
        self._psf_kernel = None if psf_kernel is None else np.asarray(psf_kernel, dtype=float)
        self._compatibility_signature = str(compatibility_signature)
        self._store_trial_map_cubes = bool(store_trial_map_cubes)
        self._viewer_heartbeat = viewer_heartbeat
        self._render_stream = _PointRenderStream()
        self._writer = _ArtifactWriteDispatcher(
            artifact_h5=self._artifact_h5,
            observed=self._observed,
            sigma_map=self._sigma_map,
            target_header=self._target_header,
            diagnostics=self._diagnostics,
            blos_reference=self._blos_reference,
            psf_kernel=self._psf_kernel,
        )
        self._point_map: dict[tuple[float, float], ABPointResult] = {}

    @property
    def streaming_renderer_factory(self) -> Any:
        return _StreamingRendererFactory(
            self._renderer_factory,
            stream=self._render_stream,
            observed_template=self._observed,
            target_metric=self._target_metric,
            psf_source=self._psf_source,
            compatibility_signature=self._compatibility_signature,
            store_trial_map_cubes=self._store_trial_map_cubes,
        )

    def close(self) -> None:
        self._writer.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def set_pending_points(self, points: list[tuple[float, float]] | tuple[tuple[float, float], ...]) -> None:
        if self._viewer_heartbeat is not None:
            self._viewer_heartbeat.set_pending_points(points)

    def clear_pending_points(self) -> None:
        if self._viewer_heartbeat is not None:
            self._viewer_heartbeat.clear_pending_points()

    def _target_slice_key(self) -> str | None:
        value = str(self._diagnostics.get("target_slice_key") or self._diagnostics.get("slice_key") or "").strip()
        return value or None

    def promote_auxiliary_maps_from_store(
        self,
        *,
        threshold: float,
        explicit_mask: np.ndarray | None,
    ) -> int:
        if not self._artifact_h5.exists():
            return 0
        slice_key = self._target_slice_key()
        if not slice_key:
            return 0
        try:
            records = load_auxiliary_map_store_point_records(self._artifact_h5, slice_key=slice_key)
        except KeyError:
            return 0
        promoted_count = 0
        for record in records:
            rescored = _rescore_auxiliary_map_record(
                record,
                observed=self._observed,
                sigma_map=self._sigma_map,
                threshold=float(threshold),
                explicit_mask=explicit_mask,
                target_metric=self._target_metric,
            )
            if rescored is None:
                continue
            point, point_payload = rescored
            key = (float(point.a), float(point.b))
            if key in self._point_map:
                continue
            point_payload["diagnostics"] = {
                **dict(point_payload.get("diagnostics") or {}),
                COMPATIBILITY_SIGNATURE_KEY: self._compatibility_signature,
            }
            self._writer.write_point(point_payload)
            self._point_map[key] = point
            promoted_count += 1
        if promoted_count and self._viewer_heartbeat is not None:
            self._viewer_heartbeat.set_phase(f"{promoted_count} map_store point(s) promoted")
        return promoted_count

    def hydrate_from_existing(self) -> int:
        if not self._artifact_h5.exists():
            return 0
        try:
            payload = load_scan_file(self._artifact_h5, slice_key=self._target_slice_key())
        except KeyError:
            return 0
        if bool(dict(payload.get("diagnostics") or {}).get("render_only_slice", False)) and not payload.get("point_records"):
            return 0
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
            if not point_record_matches_compatibility_signature(
                record,
                compatibility_signature=self._compatibility_signature,
            ):
                continue
            point = _point_from_record(record, target_metric=self._target_metric)
            self._point_map[(float(point.a), float(point.b))] = point
            count += 1
        return count

    def promote_current_slice_trial_maps(
        self,
        *,
        threshold: float,
        explicit_mask: np.ndarray | None,
        include_matching_signature: bool = False,
    ) -> int:
        if not self._artifact_h5.exists():
            return 0
        try:
            current_payload = load_scan_file(self._artifact_h5, slice_key=self._target_slice_key())
        except KeyError:
            return 0
        if bool(dict(current_payload.get("diagnostics") or {}).get("render_only_slice", False)):
            return 0
        validate_scan_artifact_compatibility(
            current_payload,
            observed=self._observed,
            sigma_map=self._sigma_map,
            wcs_header=self._target_header,
            diagnostics=self._diagnostics,
            artifact_path=self._artifact_h5,
        )
        promoted_count = 0
        for search in current_payload.get("search_records", []):
            search_id = str(search.get("search_id", "")).strip()
            if not search_id:
                continue
            try:
                search_payload = load_scan_file(self._artifact_h5, slice_key=self._target_slice_key(), search_id=search_id)
            except KeyError:
                continue
            for record in search_payload.get("point_records", []):
                key = (float(record["a"]), float(record["b"]))
                if key in self._point_map:
                    continue
                if point_record_matches_compatibility_signature(
                    record,
                    compatibility_signature=self._compatibility_signature,
                ) and not bool(include_matching_signature):
                    continue
                rescored = _rescore_auxiliary_map_record(
                    {
                        **dict(record),
                        "source_slice_key": self._target_slice_key(),
                        "source_search_id": search_id,
                    },
                    observed=self._observed,
                    sigma_map=self._sigma_map,
                    threshold=float(threshold),
                    explicit_mask=explicit_mask,
                    target_metric=self._target_metric,
                )
                if rescored is None:
                    continue
                point, point_payload = rescored
                point_payload["diagnostics"] = {
                    **dict(point_payload.get("diagnostics") or {}),
                    COMPATIBILITY_SIGNATURE_KEY: self._compatibility_signature,
                }
                self._writer.write_point(point_payload)
                self._point_map[key] = point
                promoted_count += 1
        if promoted_count and self._viewer_heartbeat is not None:
            self._viewer_heartbeat.set_phase(f"{promoted_count} current-slice map_store point(s) promoted")
        return promoted_count

    def __getitem__(self, key: tuple[float, float]) -> ABPointResult:
        return self._point_map[key]

    def __setitem__(self, key: tuple[float, float], value: ABPointResult) -> None:
        normalized_key = (float(key[0]), float(key[1]))
        existing = self._point_map.get(normalized_key)
        if existing is not None and np.isclose(existing.q0, value.q0, rtol=0.0, atol=1e-12):
            self._point_map[normalized_key] = value
            return
        payload = value.artifact_payload
        if payload is None:
            raise RuntimeError("point result is missing the artifact payload produced by the worker")
        print(
            f"  Serializing point payload: a={float(value.a):.3f} b={float(value.b):.3f} "
            f"(trials={len(tuple(value.trial_q0))})"
        )
        if self._viewer_heartbeat is not None:
            self._viewer_heartbeat.set_phase(
                f"serializing point a={float(value.a):.3f} b={float(value.b):.3f}"
            )
        save_started = time.perf_counter()
        self._writer.write_point(payload)
        save_elapsed = time.perf_counter() - save_started
        self._point_map[normalized_key] = value
        if self._viewer_heartbeat is not None:
            self._viewer_heartbeat.clear_active_trial()
            self._viewer_heartbeat.set_phase(
                f"point {len(self._point_map)} saved"
            )
        print(
            f"  Saved point to sparse artifact: a={float(value.a):.3f} b={float(value.b):.3f} "
            f"q0={_format_console_scalar(float(value.q0), fixed_precision=6)} "
            f"{self._target_metric}={float(value.objective_value):.6e} "
            f"enqueue_elapsed={save_elapsed:.3f}s"
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
    parser.add_argument("--obs-source", choices=("external_fits", "model_refmap"), default=None, help="Select whether the observation comes from an external FITS product or an internal model refmap")
    parser.add_argument("--obs-path", type=Path, default=None, help="Explicit path to an external observational FITS map")
    parser.add_argument("--obs-map-id", default=None, help="Internal model refmap identifier, for example AIA_171")
    parser.add_argument("--obs-domain", choices=("mw", "euv", "uv", "generic"), default=None, help="Optional observation-domain hint used only for validation or to fill missing metadata")
    parser.add_argument("--obs-frequency-ghz", type=float, default=None, help="Optional MW frequency hint used only when the selected observation is missing frequency metadata")
    parser.add_argument("--obs-wavelength-angstrom", type=float, default=None, help="Optional EUV/UV wavelength hint used only when the selected observation is missing wavelength metadata")
    parser.add_argument("--model-h5", dest="model_h5_override", type=Path, default=None, help="Explicit model H5 path used when positional model_h5 is omitted")
    parser.add_argument("--ebtel-path", type=Path, default=None, help="Path to the matching EBTEL .sav file")
    parser.add_argument("--testdata-repo", type=Path, default=None, help="Optional sibling pyGXrender-test-data checkout used for default input resolution")
    parser.add_argument("--euv-instrument", type=str, default=None, help="Optional EUV/UV instrument override. Must agree with the selected observation if that observation already declares an instrument.")
    parser.add_argument("--euv-response-sav", type=Path, default=None, help="Optional gxresponse SAV override used for EUV/UV rendering. If omitted, gximagecomputing will use its default pyEUVTools-backed response path for supported instruments.")
    parser.add_argument("--all-channels", action="store_true", help="For fixed-channel EUV/UV instruments, render all known channels for the same geometry while fitting only the selected observation slice.")
    parser.add_argument("--render-channels", default=None, help="Comma-separated EUV/UV channels to render in addition to the fitted observation channel, for example 94,131,193.")
    parser.add_argument("--render-frequencies-ghz", default=None, help="Comma-separated MW frequencies to render in addition to the fitted observation frequency. MW extra frequencies must be explicit.")
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
    parser.add_argument(
        "--store-trial-map-cubes",
        action="store_true",
        help="Persist full per-trial rendered map cubes (expensive; disabled by default)",
    )
    parser.add_argument("--metrics-mask-threshold", type=float, default=0.1, help="Relative threshold used by the default union metrics mask.")
    parser.add_argument("--metrics-mask-fits", type=Path, default=None, help="Optional FITS bit mask used for metrics evaluation. Non-zero finite pixels are treated as in-mask and override --metrics-mask-threshold.")
    parser.add_argument("--tr-mask-bmin-gauss", type=float, default=1000.0, help="For EUV/UV, build the default TR-region mask from abs(B_los) >= Bmin [G]. Negative inputs are treated as abs(Bmin).")
    parser.add_argument("--threshold", dest="metrics_mask_threshold", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--mask-type", choices=("union", "data", "model", "and"), default="union", help=argparse.SUPPRESS)
    parser.add_argument("--threshold-metric", type=float, default=1.1, help="Phase-2 threshold region multiplier around the best point")
    parser.add_argument("--no-area", action="store_true", help="Stop after phase 1 without threshold-region expansion")
    parser.add_argument("--adaptive-bracketing", action=argparse.BooleanOptionalAction, default=True, help="Enable adaptive q0 bracketing")
    parser.add_argument("--q0-start", type=float, default=None, help="Optional explicit q0 start for every point")
    parser.add_argument("--q0-step", type=float, default=1.61803398875, help="Multiplicative q0 step for adaptive bracketing")
    parser.add_argument("--xatol", type=float, default=DEFAULT_Q0_XATOL, help="Absolute q0 tolerance for bounded minimization")
    parser.add_argument("--maxiter", type=int, default=DEFAULT_Q0_MAXITER, help="Maximum bounded-minimizer iterations")
    parser.add_argument(
        "--max-bracket-steps",
        type=int,
        default=12,
        help="Maximum additional adaptive q0 bracket expansions after the initial q0_min/q0_start/q0_max triplet; not a cap on total trial evaluations",
    )
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
        help="Use user-supplied PSF parameters even when the FITS header already contains a PSF beam",
    )
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
    reset_group = parser.add_mutually_exclusive_group()
    reset_group.add_argument(
        "--recompute-existing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reset a compatible sparse artifact at the target path and recompute all points from scratch.",
    )
    reset_group.add_argument(
        "--start-over",
        action="store_true",
        help="Start a fresh adaptive search instance in the existing artifact while seeding from the stored maps instead of resuming the previous compatible search state.",
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


def _resolve_geometry_request_flags(args: argparse.Namespace) -> tuple[bool, bool]:
    explicit_observer_requested = any(
        getattr(args, field_name, None) is not None
        for field_name in ("observer", "dsun_cm", "lonc_deg", "b0sun_deg")
    )
    geometry_overrides_requested = False
    return geometry_overrides_requested, explicit_observer_requested


def main() -> int:
    args = _parse_args()
    _validate_gxrender_runtime()
    repo_root = Path(__file__).resolve().parents[2]
    obs_request = _resolve_observation_request(args, repo_root=repo_root)
    model_h5 = obs_request.model_h5
    if obs_request.obs_path is not None and not obs_request.obs_path.exists():
        raise SystemExit(f"Observational FITS file not found: {obs_request.obs_path}")
    if not model_h5.exists():
        raise SystemExit(f"Model H5 file not found: {model_h5}")

    try:
        obs_map = load_obs_map(
            obs_path=obs_request.obs_path,
            model_h5=model_h5,
            map_id=obs_request.obs_map_id,
            source_mode=obs_request.source_mode,
        )
        obs_map = validate_obs_map_identity(
            obs_map,
            domain_hint=args.obs_domain,
            frequency_ghz_hint=args.obs_frequency_ghz,
            wavelength_angstrom_hint=args.obs_wavelength_angstrom,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    obs_source_detail = (
        str(obs_request.obs_path)
        if obs_request.obs_path is not None
        else f"{model_h5}::{obs_request.obs_map_id}"
    )
    try:
        render_selection = _resolve_render_selection(args, obs_map)
        slice_descriptors, render_frequencies_ghz, render_channels = _resolve_render_slice_requests(
            domain=render_selection.domain,
            frequency_ghz=render_selection.active_frequency_ghz,
            euv_channel=render_selection.euv_channel,
            euv_instrument=render_selection.euv_instrument,
            all_channels=bool(args.all_channels),
            render_channels_csv=args.render_channels,
            render_frequencies_csv=args.render_frequencies_ghz,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    artifacts_dir = _coerce_path(args.artifacts_dir) or (repo_root / "ab_scan_artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stem = args.artifacts_stem or _default_artifact_stem(obs_request, target_metric=str(args.target_metric))
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

    print(f"Using observation source: {obs_request.source_mode}")
    print(f"Observation selection: {obs_source_detail}")
    print(
        "Resolved observation: "
        f"domain={obs_map.domain} "
        f"label={obs_map.spectral_label or 'n/a'} "
        f"instrument={obs_map.instrument or 'n/a'}"
    )
    print(f"Using model H5: {model_h5}")
    if obs_request.ebtel_path is not None:
        print(f"Using EBTEL: {obs_request.ebtel_path}")
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

    ebtel_path = obs_request.ebtel_path
    if ebtel_path is None or not ebtel_path.exists():
        raise SystemExit(f"EBTEL file not found: {ebtel_path}")

    live_log_handle = None
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        live_log_handle = log_path.open("a", encoding="utf-8", buffering=1)
        sys.stdout = _TeeStream(sys.stdout, live_log_handle)
        sys.stderr = _TeeStream(sys.stderr, live_log_handle)
        print("=" * 88)
        print(
            "Run start: "
            f"utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
            f"pid={os.getpid()} argv={shlex.join([str(sys.executable), *[str(item) for item in sys.argv]])}"
        )
        print(f"Live log file: {log_path}")
    except Exception as exc:
        print(f"WARNING: failed to initialize live log sidecar: {exc}")

    print("Using resolved observational map payload")
    observed = np.asarray(obs_map.data, dtype=float)
    header = obs_map.header.copy()
    freq_ghz = None if obs_map.frequency_ghz is None else float(obs_map.frequency_ghz)
    print(f"  Shape: {observed.shape}")
    if freq_ghz is not None:
        print(f"  Frequency: {float(freq_ghz):.3f} GHz")
    elif obs_map.wavelength_angstrom is not None:
        print(f"  Wavelength: {float(obs_map.wavelength_angstrom):.3f} A")

    sdk = import_module("gxrender.sdk")
    header_psf, header_psf_source = _extract_psf_from_header(header)
    psf_bmaj_arcsec = float(args.psf_bmaj_arcsec) if args.psf_bmaj_arcsec is not None else None
    psf_bmin_arcsec = float(args.psf_bmin_arcsec) if args.psf_bmin_arcsec is not None else None
    psf_bpa_deg = float(args.psf_bpa_deg) if args.psf_bpa_deg is not None else None
    selected_psf_metadata = _resolve_selected_psf_metadata(
        header_psf=header_psf,
        header_psf_source=header_psf_source,
        domain=render_selection.domain,
        instrument_name=render_selection.euv_instrument if render_selection.domain != "mw" else None,
        wavelength_angstrom=None if obs_map.wavelength_angstrom is None else float(obs_map.wavelength_angstrom),
        date_obs=obs_map.date_obs,
        cli_psf_bmaj_arcsec=psf_bmaj_arcsec,
        cli_psf_bmin_arcsec=psf_bmin_arcsec,
        cli_psf_bpa_deg=psf_bpa_deg,
        override_header_psf=bool(args.override_header_psf),
    )
    psf_source = "none" if selected_psf_metadata is None else str(selected_psf_metadata.source)
    psf_allows_frequency_scaling = bool(
        False if selected_psf_metadata is None else selected_psf_metadata.allows_frequency_scaling
    )
    if selected_psf_metadata is not None and selected_psf_metadata.kind == "gaussian":
        psf_bmaj_arcsec = selected_psf_metadata.bmaj_arcsec
        psf_bmin_arcsec = selected_psf_metadata.bmin_arcsec
        psf_bpa_deg = selected_psf_metadata.bpa_deg
    else:
        psf_bmaj_arcsec = None
        psf_bmin_arcsec = None
        psf_bpa_deg = None

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
    geometry_overrides_requested, explicit_observer_requested = _resolve_geometry_request_flags(args)
    geometry_policy = resolve_geometry_policy(
        obs_map=obs_map,
        model_observer_meta=model_observer_meta,
        saved_fov=saved_fov,
        geometry_overrides_requested=geometry_overrides_requested,
        explicit_observer_requested=explicit_observer_requested,
    )
    geometry_observer_name = str(args.observer or geometry_policy.observer_name)
    geometry_observer = observer_overrides if explicit_observer_requested else None
    resolved_geometry = resolve_render_geometry_via_gxrender(
        model_path=model_h5,
        model_format="auto",
        ebtel_path=str(ebtel_path) if ebtel_path is not None else None,
        pixel_scale_arcsec=float(args.pixel_scale_arcsec),
        observer_name=geometry_observer_name,
        observer=geometry_observer,
        omp_threads=int(getattr(args, "omp_threads", 8)),
    )
    geometry = resolved_geometry.geometry
    if not explicit_observer_requested:
        observer_overrides = sdk.ObserverOverrides(
            dsun_cm=float(geometry_policy.observer_dsun_cm),
            lonc_deg=float(geometry_policy.observer_lonc_deg),
            b0sun_deg=float(geometry_policy.observer_b0sun_deg),
        )
        observer_source = f"geometry_policy:{geometry_policy.observation_observer}"
    effective_observer_name = str(args.observer or geometry_policy.observer_name)
    effective_observer_lonc_deg = float(
        getattr(observer_overrides, "lonc_deg", None)
        if observer_overrides is not None and getattr(observer_overrides, "lonc_deg", None) is not None
        else geometry_policy.observer_lonc_deg
    )
    effective_observer_b0sun_deg = float(
        getattr(observer_overrides, "b0sun_deg", None)
        if observer_overrides is not None and getattr(observer_overrides, "b0sun_deg", None) is not None
        else geometry_policy.observer_b0sun_deg
    )
    effective_observer_dsun_cm = float(
        getattr(observer_overrides, "dsun_cm", None)
        if observer_overrides is not None and getattr(observer_overrides, "dsun_cm", None) is not None
        else geometry_policy.observer_dsun_cm
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
    # Do not set observer distance in header unless explicitly overridden; delegate to gxrender
    target_header = _with_observer_wcs_keywords(
        target_header,
        observer_name=effective_observer_name,
        hgln_obs_deg=effective_observer_lonc_deg,
        hglt_obs_deg=effective_observer_b0sun_deg,
        dsun_obs_m=effective_observer_dsun_cm / 100.0,
    )

    artifact_preexisting = artifact_h5.exists()
    existing_format = detect_scan_artifact_format(artifact_h5) if artifact_preexisting else None
    if artifact_preexisting and existing_format not in {None, "sparse", "unified"}:
        raise SystemExit(
            f"Existing artifact {artifact_h5} is rectangular; this adaptive example requires a sparse artifact path."
        )

    model_sha256 = _compute_file_sha256(model_h5)
    observation_source_path = obs_map.source_path
    observation_source_file = _resolve_existing_file(observation_source_path)
    observation_source_sha256 = (
        _compute_file_sha256(observation_source_file)
        if observation_source_file is not None and observation_source_file.is_file()
        else None
    )
    ebtel_sha256 = _compute_file_sha256(ebtel_path)
    target_slice_key = next(
        (str(item["key"]) for item in slice_descriptors if bool(item.get("is_target"))),
        str(slice_descriptors[0]["key"]) if slice_descriptors else "default",
    )
    preflight_diag = {
        "artifact_kind": "pychmp_ab_scan_sparse_points",
        "spectral_domain": str(render_selection.domain),
        "spectral_label": str(render_selection.spectral_label),
        "target_slice_key": target_slice_key,
        "model_sha256": str(model_sha256),
        "fits_sha256": str(observation_source_sha256 or ""),
        "ebtel_sha256": str(ebtel_sha256),
        "frequency_ghz": None if freq_ghz is None else float(freq_ghz),
        "wavelength_angstrom": None if obs_map.wavelength_angstrom is None else float(obs_map.wavelength_angstrom),
        "euv_channel": render_selection.euv_channel,
        "euv_instrument": render_selection.euv_instrument,
        "map_xc_arcsec": float(geometry.xc),
        "map_yc_arcsec": float(geometry.yc),
        "map_dx_arcsec": float(geometry.dx),
        "map_dy_arcsec": float(geometry.dy),
        "map_nx": int(geometry.nx),
        "map_ny": int(geometry.ny),
        "observer_name": effective_observer_name,
        "observer_lonc_deg": effective_observer_lonc_deg,
        "observer_b0sun_deg": effective_observer_b0sun_deg,
        "observer_dsun_cm": effective_observer_dsun_cm,
        "observer_obs_time": target_header.get("DATE-OBS", ""),
    }
    if artifact_preexisting and not bool(args.recompute_existing):
        try:
            current_payload = load_scan_file(artifact_h5, slice_key=target_slice_key)
        except Exception:
            current_payload = None
        if current_payload is not None:
            try:
                validate_scan_artifact_reuse_preflight(
                    current_payload,
                    wcs_header=target_header,
                    diagnostics=preflight_diag,
                    artifact_path=artifact_h5,
                )
            except ScanArtifactCompatibilityError as exc:
                raise SystemExit(str(exc)) from exc

    print("Estimating noise from map...")
    noise_result = estimate_obs_map_noise(obs_map, method="histogram_clip")
    sigma_map = np.asarray(noise_result.sigma_map, dtype=float)
    noise_diag = noise_result.diagnostics
    noise_unit = obs_map_noise_unit_label(obs_map)
    print(
        f"  Estimated sigma: {float(noise_result.sigma):.2f} {noise_unit} "
        f"(method={str(noise_result.method_used)})"
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

    blos_reference_for_fov = load_blos_reference_for_fov(
        model_h5,
        header=target_header,
        shape=np.asarray(observed_cropped, dtype=float).shape,
        wcs_header_transform=None,
    )
    euv_tr_mask = None
    if render_selection.domain != "mw":
        if blos_reference_for_fov is not None:
            tr_mask_bmin_gauss = abs(float(args.tr_mask_bmin_gauss))
            euv_tr_mask = build_tr_region_mask_from_blos(
                np.asarray(blos_reference_for_fov[0], dtype=float),
                threshold_gauss=tr_mask_bmin_gauss,
            )
            selected = int(np.count_nonzero(euv_tr_mask))
            total = int(euv_tr_mask.size)
            print(
                f"  EUV TR mask: |B_los| >= {tr_mask_bmin_gauss:.1f} G "
                f"({selected}/{total} pixels, {selected / max(total, 1):.1%})"
            )
        else:
            print("  EUV TR mask: unavailable (B_los reference could not be loaded); summing full TR component")

    explicit_metric_mask = None
    if args.metrics_mask_fits is not None:
        explicit_metric_mask = _load_explicit_metric_mask(
            args.metrics_mask_fits,
            expected_shape=tuple(np.asarray(observed_cropped, dtype=float).shape),
        )
        selected = int(np.count_nonzero(explicit_metric_mask))
        total = int(explicit_metric_mask.size)
        print(
            f"  Metrics mask: explicit FITS {Path(args.metrics_mask_fits).expanduser()} "
            f"({selected}/{total} pixels, {selected / max(total, 1):.1%})"
        )
    else:
        print(f"  Metrics mask: union threshold={float(args.metrics_mask_threshold):.3f}")

    print(f"  Observer mode: {'saved metadata' if observer_overrides is None else 'overrides'} ({observer_source})")
    print(
        "  Geometry policy: "
        f"obs_los={geometry_policy.observation_observer or '<unknown>'} "
        f"model_los={geometry_policy.model_observer or '<unknown>'} "
        f"aligned={geometry_policy.los_aligned}; render_geometry_resolver=gxrender"
    )
    print(
        "  Geometry: "
        f"xc={float(geometry.xc):.3f} yc={float(geometry.yc):.3f} "
        f"dx={float(geometry.dx):.3f} dy={float(geometry.dy):.3f} "
        f"nx={int(geometry.nx)} ny={int(geometry.ny)}"
    )
    if render_selection.domain == "mw":
        print(
            "  "
            + format_psf_report(
                metadata=selected_psf_metadata,
                active_frequency_ghz=float(freq_ghz),
                ref_frequency_ghz=float(args.psf_ref_frequency_ghz) if args.psf_ref_frequency_ghz is not None else None,
                scale_inverse_frequency=bool(args.psf_scale_inverse_frequency),
            )
        )
    else:
        print(
            "  "
            + format_psf_report(
                metadata=selected_psf_metadata,
                active_frequency_ghz=None,
                ref_frequency_ghz=None,
                scale_inverse_frequency=False,
            )
        )

    psf_kernel = None
    resolved_psf_meta = None
    if selected_psf_metadata is not None:
        psf_kernel, resolved_psf_meta = build_psf_kernel(
            metadata=selected_psf_metadata,
            dx_arcsec=float(geometry.dx),
            dy_arcsec=float(geometry.dy),
            active_frequency_ghz=float(freq_ghz) if render_selection.domain == "mw" else None,
            ref_frequency_ghz=float(args.psf_ref_frequency_ghz) if args.psf_ref_frequency_ghz is not None else None,
            scale_inverse_frequency=bool(args.psf_scale_inverse_frequency),
        )

    root_diag = {
        "artifact_kind": "pychmp_ab_scan_sparse_points",
        "spectral_domain": str(render_selection.domain),
        "spectral_label": str(render_selection.spectral_label),
        "slice_descriptors": slice_descriptors,
        "target_slice_key": target_slice_key,
        "render_frequencies_ghz": [float(v) for v in render_frequencies_ghz],
        "render_channels": [str(v) for v in render_channels],
        "render_extra_slices": int(max(0, len(slice_descriptors) - 1)),
        "model_path": str(model_h5),
        "model_id": str(_load_model_identity(model_h5)),
        "model_sha256": str(model_sha256),
        "fits_file": str(observation_source_path or ""),
        "fits_sha256": str(observation_source_sha256 or ""),
        "observation_source_mode": str(obs_map.source_mode),
        "observation_source_path": observation_source_path,
        "observation_source_map_id": obs_map.source_map_id,
        "observation_source_sha256": observation_source_sha256,
        "observation_instrument": obs_map.instrument,
        "observation_observer": obs_map.observer,
        "ebtel_path": str(ebtel_path),
        "ebtel_sha256": str(ebtel_sha256),
        "target_metric": str(args.target_metric),
        "frequency_ghz": None if freq_ghz is None else float(freq_ghz),
        "active_frequency_ghz": None if freq_ghz is None else float(freq_ghz),
        "wavelength_angstrom": None if obs_map.wavelength_angstrom is None else float(obs_map.wavelength_angstrom),
        "euv_channel": render_selection.euv_channel,
        "euv_instrument": render_selection.euv_instrument,
        "euv_response_sav": None if render_selection.euv_response_sav is None else str(render_selection.euv_response_sav),
        "map_xc_arcsec": float(geometry.xc),
        "map_yc_arcsec": float(geometry.yc),
        "map_dx_arcsec": float(geometry.dx),
        "map_dy_arcsec": float(geometry.dy),
        "map_nx": int(geometry.nx),
        "map_ny": int(geometry.ny),
        "noise_diagnostics": noise_diag,
        "observer_name": effective_observer_name,
        "observer_lonc_deg": effective_observer_lonc_deg,
        "observer_b0sun_deg": effective_observer_b0sun_deg,
        "observer_dsun_cm": effective_observer_dsun_cm,
        "observer_obs_time": target_header.get("DATE-OBS", ""),
        "geometry_policy_mode": geometry_policy.geometry_mode,
        "geometry_policy_reason": "resolved_by_gxrender_observer_fov_policy",
        "geometry_policy_observation_los": geometry_policy.observation_observer,
        "geometry_policy_model_los": geometry_policy.model_observer,
        "geometry_policy_los_aligned": geometry_policy.los_aligned,
        "search_mode": "adaptive_local_single_observation",
        "a_start": float(args.a_start),
        "b_start": float(args.b_start),
        "da": float(args.da),
        "db": float(args.db),
        "a_range": [float(args.a_min), float(args.a_max)],
        "b_range": [float(args.b_min), float(args.b_max)],
        "threshold": float(args.metrics_mask_threshold),
        "metrics_mask_threshold": float(args.metrics_mask_threshold),
        "metrics_mask_fits": None if args.metrics_mask_fits is None else str(Path(args.metrics_mask_fits).expanduser()),
        "metrics_mask_source": "explicit_fits" if explicit_metric_mask is not None else "union_threshold",
        "mask_type": "explicit_fits" if explicit_metric_mask is not None else "union",
        "threshold_metric": float(args.threshold_metric),
        "tr_mask_bmin_gauss": abs(float(args.tr_mask_bmin_gauss)) if render_selection.domain != "mw" else None,
        "tr_mask_source": (
            "abs_blos_ge_bmin" if euv_tr_mask is not None else ("unavailable" if render_selection.domain != "mw" else None)
        ),
        "no_area": bool(args.no_area),
        "execution_policy": str(args.execution_policy),
        "execution_max_workers": None if args.max_workers is None else int(args.max_workers),
        "psf_source": str(psf_source),
        "resolved_psf": resolved_psf_meta,
    }
    compatibility_signature = _build_physical_compatibility_signature(
        {
            "artifact_kind": root_diag["artifact_kind"],
            "target_metric": root_diag["target_metric"],
            "model_sha256": root_diag["model_sha256"],
            "fits_sha256": root_diag["fits_sha256"],
            "observation_source_mode": root_diag["observation_source_mode"],
            "observation_source_map_id": root_diag["observation_source_map_id"],
            "spectral_domain": root_diag["spectral_domain"],
            "spectral_label": root_diag["spectral_label"],
            "ebtel_sha256": root_diag["ebtel_sha256"],
            "frequency_ghz": root_diag["frequency_ghz"],
            "wavelength_angstrom": root_diag["wavelength_angstrom"],
            "euv_channel": root_diag["euv_channel"],
            "euv_instrument": root_diag["euv_instrument"],
            "euv_response_sav": root_diag["euv_response_sav"],
            "map_xc_arcsec": root_diag["map_xc_arcsec"],
            "map_yc_arcsec": root_diag["map_yc_arcsec"],
            "map_dx_arcsec": root_diag["map_dx_arcsec"],
            "map_dy_arcsec": root_diag["map_dy_arcsec"],
            "map_nx": root_diag["map_nx"],
            "map_ny": root_diag["map_ny"],
            "observer_name": root_diag["observer_name"],
            "observer_lonc_deg": root_diag["observer_lonc_deg"],
            "observer_b0sun_deg": root_diag["observer_b0sun_deg"],
            "observer_dsun_cm": root_diag["observer_dsun_cm"],
            "observer_obs_time": root_diag["observer_obs_time"],
            "threshold": root_diag["threshold"],
            "metrics_mask_threshold": root_diag["metrics_mask_threshold"],
            "metrics_mask_fits": root_diag["metrics_mask_fits"],
            "threshold_metric": root_diag["threshold_metric"],
            "mask_type": root_diag["mask_type"],
            "tr_mask_bmin_gauss": root_diag["tr_mask_bmin_gauss"],
            "tr_mask_source": root_diag["tr_mask_source"],
            "no_area": root_diag["no_area"],
            "psf_source": root_diag["psf_source"],
            "resolved_psf": root_diag["resolved_psf"],
        }
    )
    root_diag[COMPATIBILITY_SIGNATURE_KEY] = compatibility_signature
    common_blos_reference = blos_reference_for_fov

    viewer_refresh_signal = Path(f"{artifact_h5}.refresh")
    viewer_heartbeat = _ViewerRefreshHeartbeat(
        viewer_refresh_signal,
        slice_key=str(root_diag.get("target_slice_key") or "").strip() or None,
    )
    existing_run_history = load_run_history(artifact_h5) if artifact_preexisting else []
    if bool(args.recompute_existing) and artifact_h5.exists():
        print(f"Recompute existing: resetting sparse artifact at {artifact_h5}")
    start_over_seed_payloads: list[dict[str, Any]] = []
    if bool(args.start_over) and artifact_h5.exists():
        start_over_seed_payloads = _collect_start_over_seed_point_payloads(
            artifact_h5=artifact_h5,
            slice_key=str(root_diag.get("target_slice_key") or "").strip() or None,
            observed=observed_cropped,
            sigma_map=sigma_cropped,
            wcs_header=target_header,
            diagnostics=root_diag,
            threshold=float(args.metrics_mask_threshold),
            explicit_mask=explicit_metric_mask,
            target_metric=str(args.target_metric),
            compatibility_signature=compatibility_signature,
        )
        print(f"Start over: resetting search history in {artifact_h5} while seeding from stored compatible maps")
    if not artifact_h5.exists() or bool(args.recompute_existing) or bool(args.start_over):
        write_point_scan_artifact(
            artifact_h5,
            observed=observed_cropped,
            sigma_map=sigma_cropped,
            wcs_header=target_header,
            diagnostics=root_diag,
            blos_reference=common_blos_reference,
            psf_kernel=None if psf_kernel is None else np.asarray(psf_kernel, dtype=float),
            point_records=start_over_seed_payloads if bool(args.start_over) else [],
            run_history=[] if bool(args.start_over) else existing_run_history,
            preserve_existing_searches=not bool(args.start_over),
        )
        viewer_heartbeat.set_phase("adaptive sparse artifact initialized")
        print(f"Initialized sparse artifact: {artifact_h5}")

    factory = _AdaptiveRendererFactory(
        model_path=str(model_h5),
        ebtel_path=str(ebtel_path),
        spectral_domain=str(render_selection.domain),
        spectral_label=str(render_selection.spectral_label),
        frequency_ghz=None if freq_ghz is None else float(freq_ghz),
        wavelength_angstrom=None if obs_map.wavelength_angstrom is None else float(obs_map.wavelength_angstrom),
        euv_channel=render_selection.euv_channel,
        euv_instrument=render_selection.euv_instrument,
        euv_response_sav=None if render_selection.euv_response_sav is None else str(render_selection.euv_response_sav),
        render_frequencies_ghz=render_frequencies_ghz,
        render_channels=render_channels,
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
        observer_name=effective_observer_name,
        pixel_scale_arcsec=float(args.pixel_scale_arcsec),
        psf_kernel=None if psf_kernel is None else np.asarray(psf_kernel, dtype=float),
        tr_region_mask=None if euv_tr_mask is None else np.asarray(euv_tr_mask, dtype=bool),
    )
    cache = _PersistentPointCache(
        artifact_h5=artifact_h5,
        observed=observed_cropped,
        sigma_map=sigma_cropped,
        target_header=target_header,
        diagnostics=root_diag,
        blos_reference=common_blos_reference,
        renderer_factory=factory,
        target_metric=str(args.target_metric),
        psf_source=str(psf_source),
        psf_kernel=None if psf_kernel is None else np.asarray(psf_kernel, dtype=float),
        compatibility_signature=compatibility_signature,
        store_trial_map_cubes=bool(args.store_trial_map_cubes),
        viewer_heartbeat=viewer_heartbeat,
    )
    search_renderer_factory = cache.streaming_renderer_factory
    try:
        reused_points = 0 if bool(args.recompute_existing) else cache.hydrate_from_existing()
        if not bool(args.recompute_existing):
            current_promoted_points = cache.promote_current_slice_trial_maps(
                threshold=float(args.metrics_mask_threshold),
                explicit_mask=explicit_metric_mask,
                include_matching_signature=False,
            )
            if current_promoted_points:
                reused_points += current_promoted_points
                print(
                    f"Map store reuse: promoted {current_promoted_points} current-slice point(s) "
                    f"into search {root_diag.get(COMPATIBILITY_SIGNATURE_KEY)}"
                )
            promoted_points = cache.promote_auxiliary_maps_from_store(
                threshold=float(args.metrics_mask_threshold),
                explicit_mask=explicit_metric_mask,
            )
            if promoted_points:
                reused_points += promoted_points
                print(f"Map store reuse: promoted {promoted_points} point(s) into slice {root_diag.get('target_slice_key')}")
    except ScanArtifactCompatibilityError as exc:
        cache.close()
        raise SystemExit(str(exc)) from exc
    append_run_history_entry(
        artifact_h5,
        {
            **_build_run_history_entry(
            artifact_h5=artifact_h5,
            log_path=log_path,
            viewer_cmd_text=viewer_cmd_text,
            action=(
                "recompute"
                if bool(args.recompute_existing) and artifact_preexisting
                else (
                    "start_over"
                    if bool(args.start_over) and artifact_preexisting
                    else ("resume" if artifact_preexisting and not bool(args.recompute_existing) else "create")
                )
            ),
            target_metric=str(args.target_metric),
            recompute_existing=bool(args.recompute_existing),
            ),
            "compatibility_signature": compatibility_signature,
        },
    )
    if bool(args.start_over):
        print(f"Start-over seed state: {reused_points} point(s) restored into the reset search")
    else:
        print(f"Resume state: {reused_points} compatible point(s) loaded from the existing artifact")
    if auto_viewer_enabled:
        print("Viewer command above is shown for reference; pychmp-view will also be auto-launched unless it is already running.")
    else:
        print("Open the viewer command above now if you want to inspect the artifact while the run is in progress.")
    _maybe_launch_viewer("scan start")

    started = time.perf_counter()
    viewer_heartbeat.start("adaptive search running")
    progress_start_callback = None
    progress_callback = None
    if str(args.execution_policy) == "serial":
        progress_start_callback, progress_callback = _make_trial_progress_reporter(target_metric=str(args.target_metric))
        if viewer_heartbeat is not None:
            live_q0_trials: list[float] = []
            live_metric_trials: list[float] = []
            active_point: dict[str, tuple[float, float] | None] = {"value": None}
            console_start_callback = progress_start_callback
            console_progress_callback = progress_callback

            def _viewer_point_start(a_value: float, b_value: float) -> None:
                active_point["value"] = (float(a_value), float(b_value))
                live_q0_trials.clear()
                live_metric_trials.clear()
                viewer_heartbeat.set_active_point(float(a_value), float(b_value))

            def _viewer_point_complete(a_value: float, b_value: float) -> None:
                point = active_point.get("value")
                if point is not None:
                    viewer_heartbeat.set_phase("serializing point payload")

            def _viewer_progress_start(trial_index: int, q0: float) -> None:
                if console_start_callback is not None:
                    console_start_callback(trial_index, q0)
                point = active_point.get("value")
                if point is None:
                    return
                viewer_heartbeat.set_phase(f"trial {int(trial_index):02d} rendering")
                viewer_heartbeat.update_live_trials(
                    a_value=float(point[0]),
                    b_value=float(point[1]),
                    metric_name=str(args.target_metric),
                    q0_trials=list(live_q0_trials),
                    metric_trials=list(live_metric_trials),
                    active_trial_index=int(trial_index),
                    active_trial_q0=float(q0),
                )

            def _viewer_progress_report(q0: float, objective_value: float, is_valid: bool, message: str, elapsed_s: float) -> None:
                if console_progress_callback is not None:
                    console_progress_callback(q0, objective_value, is_valid, message, elapsed_s)
                point = active_point.get("value")
                if point is None:
                    return
                live_q0_trials.append(float(q0))
                live_metric_trials.append(float(objective_value))
                viewer_heartbeat.set_phase(f"trial {len(live_q0_trials):02d} complete")
                viewer_heartbeat.update_live_trials(
                    a_value=float(point[0]),
                    b_value=float(point[1]),
                    metric_name=str(args.target_metric),
                    q0_trials=list(live_q0_trials),
                    metric_trials=list(live_metric_trials),
                    active_trial_index=None,
                    active_trial_q0=None,
                )

            progress_start_callback = _viewer_progress_start
            progress_callback = _viewer_progress_report
        else:
            _viewer_point_start = None
            _viewer_point_complete = None
    else:
        _viewer_point_start = None
        _viewer_point_complete = None
    try:
        result = search_local_minimum_ab(
            search_renderer_factory,
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
            threshold=float(args.metrics_mask_threshold),
            mask_type="union" if explicit_metric_mask is None else "explicit_fits",
            explicit_mask=explicit_metric_mask,
            target_metric=str(args.target_metric),
            xatol=float(args.xatol),
            maxiter=int(args.maxiter),
            adaptive_bracketing=bool(args.adaptive_bracketing),
            q0_start=args.q0_start,
            q0_step=float(args.q0_step),
            max_bracket_steps=int(args.max_bracket_steps),
            threshold_metric=float(args.threshold_metric),
            no_area=bool(args.no_area),
            progress_start_callback=progress_start_callback,
            progress_callback=progress_callback,
            cache=cache,
            execution_policy=str(args.execution_policy),
            max_workers=args.max_workers,
            worker_chunksize=int(args.worker_chunksize),
            point_start_callback=_viewer_point_start,
            point_complete_callback=_viewer_point_complete,
        )
    except Exception:
        viewer_heartbeat.stop("adaptive search failed")
        cache.close()
        raise
    viewer_heartbeat.stop("scan complete")
    elapsed = time.perf_counter() - started

    cache.close()

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
