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
from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from scipy.signal import fftconvolve

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


def _find_existing_viewer_pid(*, viewer_script: Path, artifact_h5: Path) -> int | None:
    """Return an existing pychmp-view PID for this artifact, if visible."""
    try:
        artifact_text = str(Path(artifact_h5).expanduser().resolve())
        script_name = Path(viewer_script).name
        proc = subprocess.run(
            ["ps", "-axo", "pid=,command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    current_pid = os.getpid()
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == current_pid:
            continue
        if script_name in command and artifact_text in command:
            return pid
    return None


def _focus_existing_viewer_pid(pid: int) -> bool:
    """Try to bring an existing pychmp-view process to the foreground."""
    if not sys.platform.startswith("darwin"):
        return False
    script = (
        'tell application "System Events"\n'
        f"set frontmost of first process whose unix id is {int(pid)} to true\n"
        "end tell"
    )
    try:
        proc = subprocess.run(
            ["osascript", "-e", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return False
    return proc.returncode == 0


def _is_hdf5_lock_contention_error(exc: BaseException) -> bool:
    if isinstance(exc, BlockingIOError):
        return True
    text = str(exc).strip().lower()
    return (
        "unable to lock file" in text
        or "resource temporarily unavailable" in text
        or "errno = 35" in text
        or "already open for read-only" in text
        or "already open for read" in text
        or "unable to synchronously open file" in text
    )


def _is_snapshot_target_missing_error(exc: BaseException) -> bool:
    if not isinstance(exc, KeyError):
        return False
    text = str(exc).strip().lower()
    return (
        "search not found for slice" in text
        or "search not found" in text
        or "slice not found" in text
    )


def _artifact_has_target_slice(artifact_h5: Path, slice_key: str | None) -> bool:
    key = str(slice_key or "").strip()
    if not key or not artifact_h5.exists():
        return False
    try:
        import h5py

        with h5py.File(str(artifact_h5), "r", locking=False) as handle:
            slices_group = handle.get(SLICE_CONTAINER_GROUP)
            return slices_group is not None and key in slices_group
    except Exception:
        return False


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
    PSFMetadata,
    build_psf_kernel,
    build_tr_region_mask_from_blos,
    estimate_obs_map_noise,
    format_psf_report,
    load_obs_map,
    load_model_obs_time_text,
    obs_map_noise_unit_label,
    prepare_observation_for_metrics,
    resolve_geometry_policy,
    resolve_slice_observation_reference,
    SliceObservationReferenceError,
    resolve_euv_response_identity,
    resolve_render_geometry_via_gxrender,
    compute_forward_model_identity_placeholder,
    resolve_default_testdata_fixture_paths,
    search_local_minimum_ab,
    validate_obs_map_identity,
)
from pychmp.search_options import add_chmp_search_cli_arguments, resolve_chmp_search_settings, resolve_shift_policy_from_args
from pychmp.spectral import (
    default_euv_channels_for_instrument,
    euv_slice_request,
    mw_slice_request,
    parse_csv_floats,
    parse_csv_tokens,
    unique_preserve_order,
)
from pychmp.search_contract import compatibility_signature_from_diagnostics, search_id_from_evaluation_config
from pychmp.ab_scan_artifacts import (
    COMPATIBILITY_SIGNATURE_KEY,
    FORWARD_MODEL_IDENTITY_VERSION,
    SLICE_CONTAINER_GROUP,
    SEARCHES_GROUP,
    ScanArtifactCompatibilityError,
    append_point_record,
    append_run_history_entry,
    artifact_geometry_sha256,
    build_artifact_geometry_block,
    build_computed_point_payload,
    build_map_identity,
    finalize_search_runner_state,
    _derive_euv_raw_best_from_components,
    _derive_euv_trial_raw_from_components,
    detect_scan_artifact_format,
    load_auxiliary_map_store_point_records,
    load_scan_file,
    load_run_history,
    load_slice_observation_reference_payload,
    load_search_observation_reference_payload,
    point_record_matches_compatibility_signature,
    read_slice_active_search_id,
    validate_scan_artifact_compatibility,
    validate_scan_artifact_reuse_preflight,
    write_point_scan_artifact,
)
from pychmp.grid_points import (
    GRID_POINTS_CONTRACT_VERSION,
    GridPointActiveQ0Event,
    GridPointAssignedEvent,
    GridPointCompletedEvent,
    GridPointFailedEvent,
    GridTrialCommittedEvent,
    apply_grid_point_event_with_retry,
    classify_grid_point_state,
    find_grid_point_group,
    list_grid_point_headers,
    hydrate_render_maps_from_grid_point,
    resolve_trial_shift_commit_fields,
)
from pychmp.refresh_signal import RefreshSignalWriter
from pychmp.metrics import MetricValues, compute_metrics, resolve_threshold_mask

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


def _load_slice_psf_metadata_from_artifact(*, artifact_h5: Path, slice_key: str) -> PSFMetadata | None:
    try:
        import h5py  # local import so runtime still works when h5py is unavailable
    except Exception:
        return None

    try:
        with h5py.File(str(artifact_h5), "r", locking=False) as f:
            common = f["slices"][str(slice_key)]["common"]
            if "psf_kernel" not in common:
                return None
            kernel = np.asarray(common["psf_kernel"], dtype=float)
            if kernel.ndim != 2 or kernel.size == 0:
                return None
            source = "artifact_slice_psf"
            if "psf_kernel_meta_json" in common:
                try:
                    raw_meta = common["psf_kernel_meta_json"][()]
                    if isinstance(raw_meta, bytes):
                        raw_meta = raw_meta.decode("utf-8", errors="replace")
                    parsed = json.loads(str(raw_meta))
                    origin_source = str(parsed.get("source") or "").strip()
                    if origin_source:
                        source = f"artifact_slice_psf:{origin_source}"
                except Exception:
                    pass
            return PSFMetadata(
                source=source,
                kind="kernel",
                kernel=kernel,
                allows_frequency_scaling=False,
            )
    except TypeError:
        # Older h5py versions may not support the locking kwarg.
        try:
            with h5py.File(str(artifact_h5), "r") as f:
                common = f["slices"][str(slice_key)]["common"]
                if "psf_kernel" not in common:
                    return None
                kernel = np.asarray(common["psf_kernel"], dtype=float)
                if kernel.ndim != 2 or kernel.size == 0:
                    return None
                return PSFMetadata(
                    source="artifact_slice_psf",
                    kind="kernel",
                    kernel=kernel,
                    allows_frequency_scaling=False,
                )
        except Exception:
            return None
    except Exception:
        return None


def _load_slice_preflight_payload(
    *,
    artifact_h5: Path,
    slice_key: str,
    include_maps: bool = False,
    search_id: str | None = None,
) -> dict[str, Any] | None:
    try:
        payload = load_slice_observation_reference_payload(
            artifact_h5,
            slice_key=slice_key,
            search_id=search_id,
        )
    except Exception:
        payload = None
    if payload is None:
        return None
    try:
        import h5py
    except Exception:
        return None
    try:
        with h5py.File(str(artifact_h5), "r", locking=False) as f:
            common = f["slices"][str(slice_key)]["common"]
            header_text = common["wcs_header"][()]
            if isinstance(header_text, bytes):
                header_text = header_text.decode("utf-8", errors="replace")
            diagnostics_raw = common["diagnostics_json"][()]
            if isinstance(diagnostics_raw, bytes):
                diagnostics_raw = diagnostics_raw.decode("utf-8", errors="replace")
            common_diagnostics = json.loads(str(diagnostics_raw))
            if not isinstance(common_diagnostics, dict):
                common_diagnostics = {}
    except TypeError:
        try:
            with h5py.File(str(artifact_h5), "r") as f:
                common = f["slices"][str(slice_key)]["common"]
                header_text = common["wcs_header"][()]
                if isinstance(header_text, bytes):
                    header_text = header_text.decode("utf-8", errors="replace")
                diagnostics_raw = common["diagnostics_json"][()]
                if isinstance(diagnostics_raw, bytes):
                    diagnostics_raw = diagnostics_raw.decode("utf-8", errors="replace")
                common_diagnostics = json.loads(str(diagnostics_raw))
                if not isinstance(common_diagnostics, dict):
                    common_diagnostics = {}
        except Exception:
            return None
    except Exception:
        return None
    obs_diagnostics = dict(payload.get("diagnostics") or {})
    merged_diagnostics = {**common_diagnostics, **obs_diagnostics}
    wcs_header = payload.get("wcs_header")
    if wcs_header is None:
        wcs_header = fits.Header.fromstring(str(header_text), sep="\n")
    observed = np.asarray(payload["observed"], dtype=float) if include_maps and payload.get("observed") is not None else None
    sigma_map = np.asarray(payload["sigma_map"], dtype=float) if include_maps and payload.get("sigma_map") is not None else None
    return {
        "wcs_header": wcs_header,
        "diagnostics": merged_diagnostics,
        "observed": observed,
        "sigma_map": sigma_map,
        "observation_canvas": payload.get("observation_canvas"),
        "sigma_canvas": payload.get("sigma_canvas"),
        "canvas_wcs_header": payload.get("canvas_wcs_header"),
    }

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


def _compact_kernel_for_target_shape(
    kernel: np.ndarray | None,
    *,
    target_ny: int,
    target_nx: int,
) -> np.ndarray | None:
    if kernel is None:
        return None
    arr = np.asarray(kernel, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return arr

    # Keep only the center support needed for target-resolution same-mode convolution.
    max_ny = max(1, int(target_ny) * 2 + 1)
    max_nx = max(1, int(target_nx) * 2 + 1)
    if arr.shape[0] <= max_ny and arr.shape[1] <= max_nx:
        kernel_sum = float(np.nansum(arr))
        if np.isfinite(kernel_sum) and kernel_sum != 0.0:
            return np.asarray(arr / kernel_sum, dtype=float)
        return arr

    cy = int(arr.shape[0] // 2)
    cx = int(arr.shape[1] // 2)
    half_ny = int(max_ny // 2)
    half_nx = int(max_nx // 2)

    y0 = max(0, cy - half_ny)
    y1 = min(arr.shape[0], y0 + max_ny)
    x0 = max(0, cx - half_nx)
    x1 = min(arr.shape[1], x0 + max_nx)

    compact = np.asarray(arr[y0:y1, x0:x1], dtype=float)
    compact_sum = float(np.nansum(compact))
    if np.isfinite(compact_sum) and compact_sum != 0.0:
        compact = np.asarray(compact / compact_sum, dtype=float)
    return compact


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
    """Refresh v2 routing hints for pychmp-view (HDF5 is authoritative)."""

    def __init__(
        self,
        signal_path: Path,
        *,
        interval_s: float = 2.0,
        slice_key: str | None = None,
        search_id: str | None = None,
    ) -> None:
        self._signal_path = Path(signal_path)
        self._interval_s = max(0.5, float(interval_s))
        self._writer = RefreshSignalWriter(self._signal_path)
        self._writer.set_routing(slice_key=slice_key, search_id=search_id)
        self._phase = ""
        self._pending_points: list[tuple[float, float]] = []
        self._active_point: tuple[float, float] | None = None
        self._last_point_id: str | None = None
        self._lock = threading.Lock()

    @property
    def phase(self) -> str:
        with self._lock:
            return str(self._phase)

    def set_search_id(self, search_id: str | None) -> None:
        with self._lock:
            self._writer.set_routing(search_id=search_id)

    def emit_event(
        self,
        event: str,
        *,
        point_id: str | None = None,
        trial_index: int | None = None,
        legacy_phase: str | None = None,
    ) -> None:
        with self._lock:
            if legacy_phase is not None:
                self._phase = str(legacy_phase)
            if point_id is not None and str(point_id).strip():
                self._last_point_id = str(point_id).strip()
            self._writer.write_event(
                str(event),
                point_id=point_id,
                trial_index=trial_index,
                legacy_phase=self._phase or legacy_phase,
            )

    def notify_refresh(self) -> None:
        with self._lock:
            point_id = str(self._last_point_id or "").strip() or None
            if point_id:
                self._writer.write_event(
                    "point_assigned",
                    point_id=point_id,
                    legacy_phase=self._phase or None,
                )
            else:
                self._writer.write_event(
                    "search_initialized",
                    legacy_phase=self._phase or None,
                )

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = str(phase)

    def set_pending_points(self, points: list[tuple[float, float]] | tuple[tuple[float, float], ...]) -> None:
        with self._lock:
            self._pending_points = [(float(a), float(b)) for a, b in points]
            if self._pending_points:
                self._active_point = self._pending_points[0]
            else:
                self._active_point = None

    def pending_points(self) -> list[tuple[float, float]]:
        with self._lock:
            return list(self._pending_points)

    def remove_pending_point(self, a_value: float, b_value: float) -> None:
        key = (float(a_value), float(b_value))
        with self._lock:
            self._pending_points = [
                point for point in self._pending_points if not np.allclose(point, key, rtol=0.0, atol=1e-12)
            ]
            if self._active_point is not None and np.allclose(self._active_point, key, rtol=0.0, atol=1e-12):
                self._active_point = self._pending_points[0] if self._pending_points else None

    def clear_pending_points(self) -> None:
        with self._lock:
            self._pending_points = []
            self._active_point = None

    def set_active_point(self, a_value: float, b_value: float) -> None:
        with self._lock:
            self._active_point = (float(a_value), float(b_value))

    def update_from_live_snapshot(self, live_state: dict[str, Any]) -> None:
        _ = live_state

    def clear_active_trial(self) -> None:
        return

    def start(self, phase: str) -> None:
        with self._lock:
            self._phase = str(phase)
        self.emit_event("search_initialized", legacy_phase=str(phase))

    def stop(self, phase: str) -> None:
        with self._lock:
            self._phase = str(phase)
            self._pending_points = []
            self._active_point = None
        lowered = str(phase).strip().lower()
        if "complete" in lowered:
            self.emit_event("search_completed", legacy_phase=str(phase))
        elif "fail" in lowered or "interrupt" in lowered:
            self.emit_event("search_completed", legacy_phase=str(phase))


class _PointRenderRecord:
    def __init__(self) -> None:
        self.raw_modeled_by_q0: dict[str, np.ndarray] = {}
        self.stokes_v_by_q0: dict[str, np.ndarray] = {}
        self.modeled_by_q0: dict[str, np.ndarray] = {}
        self.components_by_q0: dict[str, dict[str, Any]] = {}
        self.cube_by_q0: dict[str, dict[str, Any]] = {}

    def clone(self) -> _PointRenderRecord:
        cloned = _PointRenderRecord()
        cloned.raw_modeled_by_q0 = {
            str(key): np.asarray(value, dtype=np.float32).copy()
            for key, value in self.raw_modeled_by_q0.items()
        }
        cloned.stokes_v_by_q0 = {
            str(key): np.asarray(value, dtype=np.float32).copy()
            for key, value in self.stokes_v_by_q0.items()
        }
        cloned.modeled_by_q0 = {
            str(key): np.asarray(value, dtype=np.float32).copy()
            for key, value in self.modeled_by_q0.items()
        }
        cloned.components_by_q0 = {
            str(key): dict(value)
            for key, value in self.components_by_q0.items()
        }
        cloned.cube_by_q0 = {
            str(key): dict(value)
            for key, value in self.cube_by_q0.items()
        }
        return cloned


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
        stokes_v: np.ndarray | None = None,
    ) -> None:
        record = self._get_or_create(a_value, b_value)
        key = self._q0_key(q0_value)
        record.raw_modeled_by_q0[key] = np.asarray(raw_modeled, dtype=np.float32)
        record.modeled_by_q0[key] = np.asarray(modeled, dtype=np.float32)
        if stokes_v is not None:
            record.stokes_v_by_q0[key] = np.asarray(stokes_v, dtype=np.float32)

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

    def snapshot_record(self, *, a_value: float, b_value: float) -> _PointRenderRecord | None:
        with self._lock:
            record = self._records.get((float(a_value), float(b_value)))
            return None if record is None else record.clone()

    def drop_trial_render(self, *, a_value: float, b_value: float, q0_value: float) -> None:
        point_key = (float(a_value), float(b_value))
        key = self._q0_key(q0_value)
        with self._lock:
            record = self._records.get(point_key)
            if record is None:
                return
            record.raw_modeled_by_q0.pop(key, None)
            record.modeled_by_q0.pop(key, None)
            record.stokes_v_by_q0.pop(key, None)
            record.components_by_q0.pop(key, None)
            record.cube_by_q0.pop(key, None)


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
        base = object.__getattribute__(self, "_base")
        return getattr(base, name)


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
        self._artifact_h5: Path | None = None
        self._slice_key: str | None = None
        self._search_id: str | None = None
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
        stokes_v = getattr(self._renderer, "_last_stokes_v", None)
        if stokes_v is None and hasattr(self._renderer, "render_stokes_raw"):
            _stokes_i, stokes_v_raw = self._renderer.render_stokes_raw(q0_value)
            if stokes_v_raw is not None:
                stokes_v = stokes_v_raw
        raw_arr = np.asarray(raw_modeled, dtype=np.float32)
        modeled_arr = np.asarray(modeled, dtype=np.float32)
        self._stream.record_render_pair(
            a_value=self._a_value,
            b_value=self._b_value,
            q0_value=q0_value,
            raw_modeled=raw_arr,
            modeled=modeled_arr,
            stokes_v=None if stokes_v is None else np.asarray(stokes_v, dtype=np.float32),
        )
        return raw_arr, modeled_arr

    def render(self, q0: float) -> np.ndarray:
        _raw_modeled, modeled = self.render_pair(float(q0))
        return modeled

    def build_artifact_payload(self, point: ABPointResult) -> dict[str, Any]:
        stream_record = self._stream.pop_record(a_value=self._a_value, b_value=self._b_value)
        if stream_record is None:
            stream_record = _PointRenderRecord()
        artifact_h5 = getattr(self, "_artifact_h5", None)
        if artifact_h5 is not None:
            _hydrate_stream_record_from_grid_artifact(
                stream_record,
                artifact_h5=Path(artifact_h5),
                slice_key=getattr(self, "_slice_key", None),
                search_id=getattr(self, "_search_id", None),
                a_value=float(self._a_value),
                b_value=float(self._b_value),
            )
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
        renderer = object.__getattribute__(self, "_renderer")
        return getattr(renderer, name)


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
        artifact_h5: Path | None = None,
        slice_key: str | None = None,
        search_id: str | None = None,
    ) -> None:
        self._base_factory = base_factory
        self._stream = stream
        self._observed_template = np.asarray(observed_template, dtype=float)
        self._target_metric = str(target_metric)
        self._psf_source = str(psf_source)
        self._compatibility_signature = str(compatibility_signature)
        self._store_trial_map_cubes = bool(store_trial_map_cubes)
        self._artifact_h5 = None if artifact_h5 is None else Path(artifact_h5)
        self._slice_key = None if slice_key is None else str(slice_key).strip() or None
        self._search_id = None if search_id is None else str(search_id).strip() or None

    def __call__(self, a: float, b: float) -> Any:
        renderer = self._base_factory(float(a), float(b))
        proxy = _TrackedRendererProxy(
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
        proxy._artifact_h5 = self._artifact_h5
        proxy._slice_key = self._slice_key
        proxy._search_id = self._search_id
        return proxy

    def __getattr__(self, name: str) -> Any:
        base_factory = object.__getattribute__(self, "_base_factory")
        return getattr(base_factory, name)

    def __getstate__(self) -> dict[str, Any]:
        return {
            "_base_factory": self._base_factory,
            "_stream": self._stream,
            "_observed_template": self._observed_template,
            "_target_metric": self._target_metric,
            "_psf_source": self._psf_source,
            "_compatibility_signature": self._compatibility_signature,
            "_store_trial_map_cubes": self._store_trial_map_cubes,
            "_artifact_h5": self._artifact_h5,
            "_slice_key": self._slice_key,
            "_search_id": self._search_id,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)


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
    forward_model_sha256: str = ""
    forward_model_identity_version: str = FORWARD_MODEL_IDENTITY_VERSION
    artifact_geometry_sha256: str = ""
    ebtel_sha256: str = ""
    euv_response_sha256: str | None = None
    euv_response_identity_version: str | None = None

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
    component: str,
) -> dict[str, Any]:
    return build_map_identity(
        a=float(a_value),
        b=float(b_value),
        q0=float(q0_value),
        domain=str(domain_label),
        channel_or_frequency=str(channel_or_frequency_label),
        component=str(component),
        forward_model_sha256=str(renderer_factory.forward_model_sha256),
        forward_model_identity_version=str(renderer_factory.forward_model_identity_version),
        ebtel_sha256=str(renderer_factory.ebtel_sha256),
        artifact_geometry_sha256=str(renderer_factory.artifact_geometry_sha256),
        euv_response_sha256=renderer_factory.euv_response_sha256,
        euv_response_identity_version=renderer_factory.euv_response_identity_version,
    )


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


def _convolve_with_psf_kernel(raw_map: np.ndarray, psf_kernel: np.ndarray | None) -> np.ndarray:
    raw = np.asarray(raw_map, dtype=float)
    kernel = None if psf_kernel is None else np.asarray(psf_kernel, dtype=float)
    if kernel is None or kernel.ndim != 2 or kernel.size == 0:
        return raw.copy()
    return np.asarray(fftconvolve(raw, kernel, mode="same"), dtype=float)


def _hydrate_stream_record_from_grid_artifact(
    record: _PointRenderRecord,
    *,
    artifact_h5: Path,
    slice_key: str | None,
    search_id: str | None,
    a_value: float,
    b_value: float,
) -> None:
    hydrate_render_maps_from_grid_point(
        Path(artifact_h5),
        slice_key=slice_key,
        search_id=search_id,
        a=float(a_value),
        b=float(b_value),
        raw_modeled_by_q0=record.raw_modeled_by_q0,
        modeled_by_q0=record.modeled_by_q0,
    )


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
        metric_trials = [float(v) for v in point.trial_objective_values]
        if len(metric_trials) == len(trial_q0_values) and metric_trials:
            finite = [(idx, value) for idx, value in enumerate(metric_trials) if np.isfinite(value)]
            if finite:
                best_index = min(finite, key=lambda item: item[1])[0]
                fallback_q0 = float(trial_q0_values[best_index])
                raw_modeled_best = _lookup_stream_value_by_q0(stream_record.raw_modeled_by_q0, fallback_q0)
                modeled_best = _lookup_stream_value_by_q0(stream_record.modeled_by_q0, fallback_q0)
                if raw_modeled_best is not None and modeled_best is not None:
                    point_q0 = float(fallback_q0)
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
    euv_coronal_best = None
    euv_tr_best = None
    stokes_v_best = None
    trial_stokes_v_maps = None
    euv_tr_mask = None
    tr_mask_attr = getattr(renderer_factory, "tr_region_mask", None)
    if tr_mask_attr is not None:
        euv_tr_mask = np.asarray(tr_mask_attr, dtype=bool)
    map_store_arrays: dict[str, np.ndarray] = {}
    synthetic_map_keys: list[dict[str, Any]] = []

    def _register_synthetic_map(*, identity: dict[str, Any], array: np.ndarray, label: str, map_role: str) -> None:
        machine_key = _canonical_json_sha256(identity)
        map_store_name = f"synthetic/{machine_key}"
        map_store_arrays[map_store_name] = np.asarray(array, dtype=np.float32)
        synthetic_map_keys.append(
            {
                "label": str(label),
                "map_store_array": str(map_store_name),
                "machine_key": str(machine_key),
                "identity": {**dict(identity), "map_role": str(map_role)},
            }
        )

    cube_payload = None
    cube_payload = _lookup_stream_value_by_q0(stream_record.cube_by_q0, point_q0)
    if isinstance(cube_payload, dict):
        stokes_v_by_frequency = dict(cube_payload.get("stokes_v_by_frequency", {}))
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
                    component="stokes_i",
                ),
                array=np.asarray(rendered, dtype=np.float32),
                label=f"MW {freq_label} Stokes I",
                map_role="stokes_i",
            )
            rendered_v = stokes_v_by_frequency.get(freq)
            if rendered_v is None:
                rendered_v = stokes_v_by_frequency.get(float(freq))
            if rendered_v is not None:
                _register_synthetic_map(
                    identity=_build_synthetic_map_identity(
                        renderer_factory=renderer_factory,
                        a_value=point_a,
                        b_value=point_b,
                        q0_value=point_q0,
                        domain_label="mw",
                        channel_or_frequency_label=freq_label,
                        component="stokes_v",
                    ),
                    array=np.asarray(rendered_v, dtype=np.float32),
                    label=f"MW {freq_label} Stokes V",
                    map_role="stokes_v",
                )
    components = None
    components = _lookup_stream_value_by_q0(stream_record.components_by_q0, point_q0)
    if isinstance(components, dict):
        corona_target = components.get("flux_corona")
        tr_target = components.get("flux_tr")
        if corona_target is not None:
            euv_coronal_best = np.asarray(corona_target, dtype=np.float32)
        if tr_target is not None:
            euv_tr_best = np.asarray(tr_target, dtype=np.float32)
        for channel, rendered in dict(components.get("flux_corona_by_channel", {})).items():
            _register_synthetic_map(
                identity=_build_synthetic_map_identity(
                    renderer_factory=renderer_factory,
                    a_value=point_a,
                    b_value=point_b,
                    q0_value=point_q0,
                    domain_label="euv",
                    channel_or_frequency_label=str(channel),
                    component="corona",
                ),
                array=np.asarray(rendered, dtype=np.float32),
                label=f"EUV {channel} corona",
                map_role="corona",
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
                    component="tr",
                ),
                array=np.asarray(rendered, dtype=np.float32),
                label=f"EUV {channel} TR",
                map_role="tr",
            )

    trial_raw_by_q0: dict[str, np.ndarray] = {}
    trial_modeled_by_q0: dict[str, np.ndarray] = {}
    trial_stokes_v_by_q0: dict[str, np.ndarray] = {}
    if stream_record is not None:
        trial_raw_by_q0 = dict(stream_record.raw_modeled_by_q0)
        trial_modeled_by_q0 = dict(stream_record.modeled_by_q0)
        trial_stokes_v_by_q0 = dict(stream_record.stokes_v_by_q0)
    stokes_v_best_lookup = _lookup_stream_value_by_q0(trial_stokes_v_by_q0, point_q0)
    if stokes_v_best_lookup is not None:
        stokes_v_best = np.asarray(stokes_v_best_lookup, dtype=np.float32)

    if trial_q0_values:
        raw_trials: list[np.ndarray] = []
        modeled_trials: list[np.ndarray] = []
        residual_trials: list[np.ndarray] = []
        euv_coronal_trials: list[np.ndarray] = []
        euv_tr_trials: list[np.ndarray] = []
        stokes_v_trials: list[np.ndarray] = []
        for trial_index, q0_value in enumerate(trial_q0_values):
            raw_trial = _lookup_stream_value_by_q0(trial_raw_by_q0, q0_value)
            if raw_trial is None:
                raw_trials = []
                modeled_trials = []
                residual_trials = []
                euv_coronal_trials = []
                euv_tr_trials = []
                stokes_v_trials = []
                break
            raw_trial_arr = np.asarray(raw_trial, dtype=np.float32)
            raw_trials.append(raw_trial_arr)
            modeled_trial = _lookup_stream_value_by_q0(trial_modeled_by_q0, q0_value)
            modeled_trial_arr = raw_trial_arr if modeled_trial is None else np.asarray(modeled_trial, dtype=np.float32)
            modeled_trials.append(modeled_trial_arr)
            residual_trials.append(modeled_trial_arr - np.asarray(observed_template, dtype=np.float32))
            stokes_v_trial = _lookup_stream_value_by_q0(trial_stokes_v_by_q0, q0_value)
            if stokes_v_trial is not None:
                stokes_v_trials.append(np.asarray(stokes_v_trial, dtype=np.float32))

            if bool(store_trial_map_cubes):
                trial_components = _lookup_stream_value_by_q0(stream_record.components_by_q0, q0_value)
                if isinstance(trial_components, dict):
                    components = trial_components
                    coronal = components.get("flux_corona")
                    tr_flux = components.get("flux_tr")
                    if coronal is not None and tr_flux is not None:
                        euv_coronal_trials.append(np.asarray(coronal, dtype=np.float32))
                        euv_tr_trials.append(np.asarray(tr_flux, dtype=np.float32))
                    for channel, rendered in dict(components.get("flux_corona_by_channel", {})).items():
                        _register_synthetic_map(
                            identity=_build_synthetic_map_identity(
                                renderer_factory=renderer_factory,
                                a_value=point_a,
                                b_value=point_b,
                                q0_value=float(q0_value),
                                domain_label="euv",
                                channel_or_frequency_label=str(channel),
                                component="corona",
                            ),
                            array=np.asarray(rendered, dtype=np.float32),
                            label=f"EUV {channel} trial {trial_index:03d} corona",
                            map_role=f"trial_{trial_index:03d}_corona",
                        )
                    for channel, rendered in dict(components.get("flux_tr_by_channel", {})).items():
                        _register_synthetic_map(
                            identity=_build_synthetic_map_identity(
                                renderer_factory=renderer_factory,
                                a_value=point_a,
                                b_value=point_b,
                                q0_value=float(q0_value),
                                domain_label="euv",
                                channel_or_frequency_label=str(channel),
                                component="tr",
                            ),
                            array=np.asarray(rendered, dtype=np.float32),
                            label=f"EUV {channel} trial {trial_index:03d} TR",
                            map_role=f"trial_{trial_index:03d}_tr",
                        )
                trial_cube = _lookup_stream_value_by_q0(stream_record.cube_by_q0, q0_value)
                if isinstance(trial_cube, dict):
                    cube_payload = trial_cube
                    stokes_v_by_frequency = dict(cube_payload.get("stokes_v_by_frequency", {}))
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
                                component="stokes_i",
                            ),
                            array=np.asarray(rendered, dtype=np.float32),
                            label=f"MW {freq_label} trial {trial_index:03d} Stokes I",
                            map_role=f"trial_{trial_index:03d}_stokes_i",
                        )
                        rendered_v = stokes_v_by_frequency.get(freq)
                        if rendered_v is None:
                            rendered_v = stokes_v_by_frequency.get(float(freq))
                        if rendered_v is not None:
                            _register_synthetic_map(
                                identity=_build_synthetic_map_identity(
                                    renderer_factory=renderer_factory,
                                    a_value=point_a,
                                    b_value=point_b,
                                    q0_value=float(q0_value),
                                    domain_label="mw",
                                    channel_or_frequency_label=freq_label,
                                    component="stokes_v",
                                ),
                                array=np.asarray(rendered_v, dtype=np.float32),
                                label=f"MW {freq_label} trial {trial_index:03d} Stokes V",
                                map_role=f"trial_{trial_index:03d}_stokes_v",
                            )
        if raw_trials and len(raw_trials) == len(trial_q0_values):
            trial_raw_modeled_maps = np.stack(raw_trials, axis=0)
            trial_modeled_maps = np.stack(modeled_trials, axis=0)
            trial_residual_maps = np.stack(residual_trials, axis=0)
        if euv_coronal_trials and len(euv_coronal_trials) == len(trial_q0_values):
            trial_euv_coronal_maps = np.stack(euv_coronal_trials, axis=0)
            trial_euv_tr_maps = np.stack(euv_tr_trials, axis=0)
        if stokes_v_trials and len(stokes_v_trials) == len(trial_q0_values):
            trial_stokes_v_maps = np.stack(stokes_v_trials, axis=0)
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
        fit_shift_x_trials=tuple(float(v) for v in point.trial_shift_x_arcsec),
        fit_shift_y_trials=tuple(float(v) for v in point.trial_shift_y_arcsec),
        fit_find_shift_valid_trials=tuple(bool(v) for v in point.trial_find_shift_valid),
        fit_trial_mask_stages=tuple(str(v) for v in point.trial_mask_stages),
        trial_raw_modeled_maps=trial_raw_modeled_maps,
        trial_modeled_maps=trial_modeled_maps,
        trial_residual_maps=trial_residual_maps,
        trial_euv_coronal_maps=trial_euv_coronal_maps,
        trial_euv_tr_maps=trial_euv_tr_maps,
        euv_coronal_best=euv_coronal_best,
        euv_tr_best=euv_tr_best,
        euv_tr_mask=euv_tr_mask,
        stokes_v_best=stokes_v_best,
        trial_stokes_v_maps=trial_stokes_v_maps,
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


def _build_live_point_snapshot_payload(
    *,
    a_value: float,
    b_value: float,
    q0_trials: list[float],
    metric_trials: list[float],
    observed_template: np.ndarray,
    target_metric: str,
    compatibility_signature: str,
    stream_record: _PointRenderRecord | None,
) -> dict[str, Any] | None:
    if stream_record is None:
        return None
    if not q0_trials or len(q0_trials) != len(metric_trials):
        return None

    raw_trials: list[np.ndarray] = []
    modeled_trials: list[np.ndarray] = []
    residual_trials: list[np.ndarray] = []
    for q0_value in q0_trials:
        raw_trial = _lookup_stream_value_by_q0(stream_record.raw_modeled_by_q0, float(q0_value))
        if raw_trial is None:
            return None
        raw_trial_arr = np.asarray(raw_trial, dtype=np.float32)
        raw_trials.append(raw_trial_arr)
        modeled_trial = _lookup_stream_value_by_q0(stream_record.modeled_by_q0, float(q0_value))
        modeled_trial_arr = raw_trial_arr if modeled_trial is None else np.asarray(modeled_trial, dtype=np.float32)
        modeled_trials.append(modeled_trial_arr)
        residual_trials.append(modeled_trial_arr - np.asarray(observed_template, dtype=np.float32))

    metric_name = str(target_metric)
    fit_metric_trials = tuple(float(v) for v in metric_trials)
    nan_trials = tuple(float("nan") for _ in metric_trials)
    fit_chi2_trials = fit_metric_trials if metric_name == "chi2" else nan_trials
    fit_rho2_trials = fit_metric_trials if metric_name == "rho2" else nan_trials
    fit_eta2_trials = fit_metric_trials if metric_name == "eta2" else nan_trials
    best_index = int(np.nanargmin(np.asarray(metric_trials, dtype=float)))
    diagnostics = {
        "a": float(a_value),
        "b": float(b_value),
        "target_metric": metric_name,
        "target_metric_value": float(metric_trials[best_index]),
        "chi2": float(metric_trials[best_index]) if metric_name == "chi2" else float("nan"),
        "rho2": float(metric_trials[best_index]) if metric_name == "rho2" else float("nan"),
        "eta2": float(metric_trials[best_index]) if metric_name == "eta2" else float("nan"),
        "point_status": "running",
        COMPATIBILITY_SIGNATURE_KEY: str(compatibility_signature),
    }
    return build_computed_point_payload(
        a_value=float(a_value),
        b_value=float(b_value),
        q0=float(q0_trials[best_index]),
        success=False,
        status="running",
        modeled_best=np.asarray(raw_trials[best_index], dtype=float),
        raw_modeled_best=np.asarray(raw_trials[best_index], dtype=float),
        residual=np.asarray(raw_trials[best_index] - np.asarray(observed_template, dtype=np.float32), dtype=float),
        fit_q0_trials=tuple(float(v) for v in q0_trials),
        fit_metric_trials=fit_metric_trials,
        fit_chi2_trials=fit_chi2_trials,
        fit_rho2_trials=fit_rho2_trials,
        fit_eta2_trials=fit_eta2_trials,
        trial_raw_modeled_maps=np.stack(raw_trials, axis=0),
        trial_modeled_maps=np.stack(modeled_trials, axis=0),
        trial_residual_maps=np.stack(residual_trials, axis=0),
        nfev=int(len(q0_trials)),
        nit=max(0, int(len(q0_trials) - 1)),
        message="running",
        used_adaptive_bracketing=False,
        bracket_found=False,
        bracket=None,
        target_metric=metric_name,
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
        viewer_heartbeat: _ViewerRefreshHeartbeat | None = None,
    ) -> None:
        self._artifact_h5 = Path(artifact_h5)
        self._observed = np.asarray(observed, dtype=float)
        self._sigma_map = np.asarray(sigma_map, dtype=float)
        self._target_header = target_header
        self._diagnostics = dict(diagnostics)
        self._blos_reference = blos_reference
        self._psf_kernel = None if psf_kernel is None else np.asarray(psf_kernel, dtype=float)
        self._viewer_heartbeat = viewer_heartbeat
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=16)
        self._closed = False
        self._failed: BaseException | None = None
        self._snapshot_lock_warning_emitted = False
        self._snapshot_missing_target_warning_emitted = False
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, name="pychmp-artifact-dispatcher", daemon=True)
        self._worker.start()

    def _run_h5_write_with_retry(self, fn: Any, *, attempts: int) -> None:
        last_exc: BaseException | None = None
        for attempt in range(max(1, int(attempts))):
            try:
                fn()
                return
            except BaseException as exc:
                last_exc = exc
                if not _is_hdf5_lock_contention_error(exc):
                    raise
                if attempt >= int(attempts) - 1:
                    raise
                time.sleep(min(0.5, 0.05 * (2 ** attempt)))
        if last_exc is not None:
            raise last_exc

    def _raise_if_failed(self) -> None:
        with self._lock:
            if self._failed is not None:
                raise RuntimeError("artifact write dispatcher failed") from self._failed

    def write_grid_event(self, event: Any) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("artifact write dispatcher is closed")
        self._raise_if_failed()
        self._queue.put(("grid_event", event))

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(None)
        self._queue.join()
        self._worker.join(timeout=5.0)
        self._raise_if_failed()

    def drain(self) -> None:
        self._queue.join()
        self._raise_if_failed()

    def _target_slice_key(self) -> str | None:
        value = str(self._diagnostics.get("target_slice_key") or self._diagnostics.get("slice_key") or "").strip()
        return value or None

    def _target_search_id(self) -> str | None:
        value = str(self._diagnostics.get("selected_search_id") or self._diagnostics.get("search_id") or "").strip()
        return value or None

    def _emit_refresh_for_event(self, event: Any, *, point_id: str | None = None) -> None:
        if self._viewer_heartbeat is None:
            return
        if isinstance(event, GridPointAssignedEvent):
            self._viewer_heartbeat.emit_event(
                "point_assigned",
                point_id=str(point_id or event.point_id or ""),
                legacy_phase=f"point assigned {event.a:.3f},{event.b:.3f}",
            )
        elif isinstance(event, GridTrialCommittedEvent):
            self._viewer_heartbeat.emit_event(
                "trial_committed",
                point_id=str(event.point_id),
                trial_index=int(event.trial_index),
                legacy_phase=f"trial {int(event.trial_index):02d} complete",
            )
        elif isinstance(event, GridPointCompletedEvent):
            self._viewer_heartbeat.emit_event(
                "point_completed",
                point_id=str(event.point_id),
                legacy_phase="point saved",
            )
        elif isinstance(event, GridPointFailedEvent):
            self._viewer_heartbeat.emit_event(
                "point_failed",
                point_id=str(event.point_id),
                legacy_phase="point failed",
            )
        elif isinstance(event, GridPointActiveQ0Event):
            if point_id is not None:
                self._viewer_heartbeat.emit_event(
                    "point_assigned",
                    point_id=str(point_id),
                    legacy_phase="trial active",
                )

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            try:
                operation, payload = item
                if operation == "grid_event":
                    assigned_point_id: str | None = None

                    def _apply() -> None:
                        nonlocal assigned_point_id
                        assigned_point_id = apply_grid_point_event_with_retry(
                            self._artifact_h5,
                            payload,
                            observed=self._observed,
                            sigma_map=self._sigma_map,
                            wcs_header=self._target_header,
                            diagnostics=self._diagnostics,
                            blos_reference=self._blos_reference,
                            psf_kernel=self._psf_kernel,
                        )

                    self._run_h5_write_with_retry(_apply, attempts=12)
                    self._emit_refresh_for_event(payload, point_id=assigned_point_id)
                else:
                    raise RuntimeError(f"unsupported dispatcher operation: {operation}")
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
    psf_kernel: np.ndarray | None = None,
    tr_region_mask: np.ndarray | None = None,
) -> tuple[ABPointResult, dict[str, Any]] | None:
    if record.get("euv_tr_mask") is None and tr_region_mask is not None:
        record = {**dict(record), "euv_tr_mask": np.asarray(tr_region_mask, dtype=bool)}
    trial_q0 = tuple(float(value) for value in record.get("fit_q0_trials", ()))
    trial_metric_trials = tuple(float(value) for value in record.get("fit_metric_trials", ()))
    trial_maps_raw = record.get("trial_raw_modeled_maps")
    if trial_maps_raw is None:
        trial_maps_raw = _derive_euv_trial_raw_from_components(record)
    if trial_maps_raw is None:
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
        modeled_best_raw = record.get("raw_modeled_best")
        if modeled_best_raw is None:
            modeled_best_raw = _derive_euv_raw_best_from_components(record)
        if modeled_best_raw is None:
            modeled_best_raw = record.get("modeled_best")
        if modeled_best_raw is None:
            return None
        finite_indices = [idx for idx, value in enumerate(trial_metric_trials) if np.isfinite(float(value))]
        if not finite_indices:
            return None
        best_index = min(finite_indices, key=lambda idx: float(trial_metric_trials[idx]))
        best_q0 = float(trial_q0[best_index])
        best_objective = float(trial_metric_trials[best_index])
        best_raw_map = np.asarray(modeled_best_raw, dtype=float)
        best_map = _convolve_with_psf_kernel(best_raw_map, psf_kernel)
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
            raw_modeled_best=np.asarray(best_raw_map, dtype=float),
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
        raw_arr = np.asarray(modeled, dtype=float)
        modeled_arr = _convolve_with_psf_kernel(raw_arr, psf_kernel)
        try:
            mask = explicit_mask_arr if explicit_mask_arr is not None else mask_fn(observed_arr, modeled_arr, float(threshold))
            metrics = compute_metrics(observed_arr, modeled_arr, sigma_arr, mask)
            objective = _target_metric_value(metrics, target_metric)
        except Exception:
            continue
        if np.isfinite(objective):
            rescored.append((float(q0_value), metrics, float(objective), raw_arr, modeled_arr))
    if not rescored:
        return None

    best_q0, best_metrics, best_objective, best_raw_map, best_map = min(rescored, key=lambda item: item[2])
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
    trial_metric_values = tuple(float(objective) for _q0, _metrics, objective, _raw, _modeled in rescored)
    trial_chi2 = tuple(float(metrics.chi2) for _q0, metrics, _objective, _raw, _modeled in rescored)
    trial_rho2 = tuple(float(metrics.rho2) for _q0, metrics, _objective, _raw, _modeled in rescored)
    trial_eta2 = tuple(float(metrics.eta2) for _q0, metrics, _objective, _raw, _modeled in rescored)
    ordered_trial_q0 = tuple(float(q0) for q0, _metrics, _objective, _raw, _modeled in rescored)
    ordered_trial_raw_maps = np.stack([np.asarray(raw, dtype=float) for _q0, _metrics, _objective, raw, _modeled in rescored], axis=0)
    ordered_trial_maps = np.stack([np.asarray(modeled, dtype=float) for _q0, _metrics, _objective, _raw, modeled in rescored], axis=0)
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
        raw_modeled_best=np.asarray(best_raw_map, dtype=float),
        residual=residual,
        fit_q0_trials=ordered_trial_q0,
        fit_metric_trials=trial_metric_values,
        fit_chi2_trials=trial_chi2,
        fit_rho2_trials=trial_rho2,
        fit_eta2_trials=trial_eta2,
        trial_raw_modeled_maps=np.asarray(ordered_trial_raw_maps, dtype=float),
        trial_modeled_maps=np.asarray(ordered_trial_maps, dtype=float),
        trial_residual_maps=np.asarray(ordered_trial_maps - observed_arr[None, :, :], dtype=float),
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
    psf_kernel: np.ndarray | None,
) -> list[dict[str, Any]]:
    if not artifact_h5.exists() or not slice_key:
        return []
    try:
        current_payload = load_scan_file(artifact_h5, slice_key=slice_key, include_maps=False)
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
                psf_kernel=psf_kernel,
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
    
def _maybe_validate_artifact_preflight(
    *,
    artifact_h5: Path,
    artifact_preexisting: bool,
    recompute_existing: bool,
    target_slice_key: str,
    target_header: fits.Header,
    diagnostics: dict[str, Any],
) -> None:
    if not artifact_preexisting or bool(recompute_existing):
        return
    try:
        current_payload = _load_slice_preflight_payload(
            artifact_h5=artifact_h5,
            slice_key=target_slice_key,
            include_maps=False,
        )
    except Exception:
        return
    if current_payload is None:
        return
    if bool(dict(current_payload.get("diagnostics") or {}).get("render_only_slice", False)):
        return
    validate_scan_artifact_reuse_preflight(
        current_payload,
        wcs_header=target_header,
        diagnostics=diagnostics,
        artifact_path=artifact_h5,
    )


def _grid_trial_shift_commit_kwargs(
    *,
    trial_index: int,
    payload: dict[str, Any] | None = None,
    result: ABPointResult | None = None,
    shift_x_trials: Any = None,
    shift_y_trials: Any = None,
    shift_valid_trials: Any = None,
) -> dict[str, Any]:
    stages: tuple[str, ...] = ()
    if payload is not None:
        stages = tuple(str(v) for v in payload.get("fit_trial_mask_stages", ()) or ())
    if not stages and result is not None:
        stages = tuple(str(v) for v in result.trial_mask_stages or ())
    stage = str(stages[trial_index]) if trial_index < len(stages) else ""
    if shift_x_trials is None:
        shift_x_trials = payload.get("fit_shift_x_trials") if payload is not None else None
        if shift_x_trials is None and result is not None:
            shift_x_trials = result.trial_shift_x_arcsec
    if shift_y_trials is None:
        shift_y_trials = payload.get("fit_shift_y_trials") if payload is not None else None
        if shift_y_trials is None and result is not None:
            shift_y_trials = result.trial_shift_y_arcsec
    if shift_valid_trials is None:
        shift_valid_trials = payload.get("fit_find_shift_valid_trials") if payload is not None else None
        if shift_valid_trials is None and result is not None:
            shift_valid_trials = result.trial_find_shift_valid
    trial_metadata, shift_x, shift_y, shift_valid = resolve_trial_shift_commit_fields(
        trial_index=int(trial_index),
        shift_x_trials=shift_x_trials,
        shift_y_trials=shift_y_trials,
        shift_valid_trials=shift_valid_trials,
        stage=stage,
    )
    return {
        "trial_metadata": trial_metadata,
        "shift_x": shift_x,
        "shift_y": shift_y,
        "shift_valid": shift_valid,
    }


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
            viewer_heartbeat=self._viewer_heartbeat,
        )
        self._point_map: dict[tuple[float, float], ABPointResult] = {}
        self._point_ids: dict[tuple[float, float], str] = {}
        self._trials_committed: dict[tuple[float, float], int] = {}
        self._resume_q0: dict[tuple[float, float], float] = {}
        self._retry_failed = False
        self._resume_incomplete_from = "next_q0"
        self._recompute_existing = False

    def set_resume_policy(
        self,
        *,
        retry_failed: bool = False,
        resume_incomplete_from: str = "next_q0",
    ) -> None:
        self._retry_failed = bool(retry_failed)
        resume_from = str(resume_incomplete_from or "next_q0").strip().lower()
        self._resume_incomplete_from = resume_from if resume_from in {"next_q0", "q0_start"} else "next_q0"

    def pending_resume_q0_start(self, a_value: float, b_value: float) -> float | None:
        return self._resume_q0.get((float(a_value), float(b_value)))

    def _resolve_search_id_from_artifact(self) -> str | None:
        configured = str(
            self._diagnostics.get("selected_search_id") or self._diagnostics.get("search_id") or ""
        ).strip()
        slice_key = self._target_slice_key()
        if not slice_key or not self._artifact_h5.exists():
            return configured or None
        import h5py

        from pychmp.ab_scan_artifacts import ACTIVE_SEARCH_ID_DATASET

        try:
            with h5py.File(str(self._artifact_h5), "r") as f:
                if SLICE_CONTAINER_GROUP not in f or slice_key not in f[SLICE_CONTAINER_GROUP]:
                    return configured or None
                slice_group = f[SLICE_CONTAINER_GROUP][slice_key]
                if ACTIVE_SEARCH_ID_DATASET in slice_group:
                    active = str(slice_group[ACTIVE_SEARCH_ID_DATASET][()].decode("utf-8")).strip()
                    if active:
                        return active
        except Exception:
            pass
        return configured or None

    def _sync_point_id_from_artifact(self, a_value: float, b_value: float) -> str | None:
        import h5py

        slice_key = self._target_slice_key()
        search_id = self._resolve_search_id_from_artifact()
        if not slice_key or not search_id or not self._artifact_h5.exists():
            return None
        with h5py.File(str(self._artifact_h5), "r") as f:
            if SLICE_CONTAINER_GROUP not in f or slice_key not in f[SLICE_CONTAINER_GROUP]:
                return None
            slice_group = f[SLICE_CONTAINER_GROUP][slice_key]
            if SEARCHES_GROUP not in slice_group or search_id not in slice_group[SEARCHES_GROUP]:
                return None
            found = find_grid_point_group(
                slice_group[SEARCHES_GROUP][search_id],
                a=float(a_value),
                b=float(b_value),
            )
            if found is None:
                return None
            point_id = str(found[0])
        self._point_ids[(float(a_value), float(b_value))] = point_id
        return point_id

    def set_recompute_existing(self, enabled: bool) -> None:
        self._recompute_existing = bool(enabled)

    def point_id_for(self, a_value: float, b_value: float) -> str | None:
        return self._point_ids.get((float(a_value), float(b_value)))

    def assign_grid_points(
        self,
        assignments: list[tuple[float, float, float | None]],
        *,
        force_new: bool | None = None,
    ) -> None:
        force = self._recompute_existing if force_new is None else bool(force_new)
        for a_value, b_value, q0_start in assignments:
            q0_seed = float(q0_start) if q0_start is not None else float(np.sqrt(float(self._diagnostics.get("q0_min", 1e-5)) * float(self._diagnostics.get("q0_max", 1e-3))))
            event = GridPointAssignedEvent(
                a=float(a_value),
                b=float(b_value),
                q0_start=float(q0_seed),
                next_q0=float(q0_seed),
                metric_name=str(self._target_metric),
                force_new_point_id=bool(force),
            )
            self._writer.write_grid_event(event)
            self._writer.drain()
            point_key = (float(a_value), float(b_value))
            point_id = self._sync_point_id_from_artifact(float(a_value), float(b_value))
            if point_id is not None and (force or point_key not in self._trials_committed):
                self._trials_committed[point_key] = 0

    def set_pending_points(
        self,
        points: list[tuple[float, float]] | tuple[tuple[float, float], ...],
        *,
        q0_starts: list[float | None] | None = None,
    ) -> None:
        normalized = [(float(a), float(b)) for a, b in points]
        if self._viewer_heartbeat is not None:
            self._viewer_heartbeat.set_pending_points(normalized)
        assignments: list[tuple[float, float, float | None]] = []
        for index, (a_value, b_value) in enumerate(normalized):
            q0_start = None
            if q0_starts is not None and index < len(q0_starts):
                q0_start = q0_starts[index]
            assignments.append((float(a_value), float(b_value), q0_start))
        if assignments:
            self.assign_grid_points(assignments)

    def clear_pending_points(self) -> None:
        if self._viewer_heartbeat is not None:
            self._viewer_heartbeat.clear_pending_points()

    def clear_live_trial_snapshot(self) -> None:
        return

    def update_active_q0(self, *, a_value: float, b_value: float, q0_value: float) -> None:
        point_id = self.point_id_for(a_value, b_value)
        if point_id is None:
            return
        self._writer.write_grid_event(
            GridPointActiveQ0Event(point_id=str(point_id), next_q0=float(q0_value))
        )

    def commit_trial(
        self,
        *,
        a_value: float,
        b_value: float,
        trial_index: int,
        q0_value: float,
        metric_value: float,
        raw_modeled_map: np.ndarray | None,
        next_q0: float | None = None,
        trial_metadata: dict[str, Any] | None = None,
        chi2: float | None = None,
        rho2: float | None = None,
        eta2: float | None = None,
        shift_x: float | None = None,
        shift_y: float | None = None,
        shift_valid: bool | None = None,
    ) -> None:
        point_key = (float(a_value), float(b_value))
        point_id = self.point_id_for(a_value, b_value)
        if point_id is None:
            return
        q0_trials = [float(q0_value)]
        metric_trials = [float(metric_value)]
        best_index = 0
        if trial_index > 0:
            best_index = int(trial_index)
        shift_kwargs = _grid_trial_shift_commit_kwargs(trial_index=int(trial_index))
        if shift_x is not None:
            shift_kwargs["shift_x"] = float(shift_x)
        if shift_y is not None:
            shift_kwargs["shift_y"] = float(shift_y)
        if shift_valid is not None:
            shift_kwargs["shift_valid"] = bool(shift_valid)
        merged_metadata = {**dict(shift_kwargs.get("trial_metadata") or {}), **dict(trial_metadata or {})}
        event = GridTrialCommittedEvent(
            point_id=str(point_id),
            trial_index=int(trial_index),
            q0=float(q0_value),
            metric=float(metric_value),
            next_q0=None if next_q0 is None else float(next_q0),
            best_trial_index=int(best_index),
            best_metric=float(metric_value),
            raw_modeled_map=raw_modeled_map,
            trial_metadata=merged_metadata,
            shift_x=shift_kwargs.get("shift_x"),
            shift_y=shift_kwargs.get("shift_y"),
            shift_valid=shift_kwargs.get("shift_valid"),
            chi2=chi2,
            rho2=rho2,
            eta2=eta2,
        )
        self._writer.write_grid_event(event)
        self._trials_committed[point_key] = int(trial_index) + 1

    def write_live_trial_snapshot(
        self,
        *,
        a_value: float,
        b_value: float,
        q0_trials: list[float],
        metric_trials: list[float],
        active_trial_index: int | None = None,
        active_trial_q0: float | None = None,
        completed_trial_index: int | None = None,
        completed_trial_raw_map: np.ndarray | None = None,
        trial_metadata: dict[str, Any] | None = None,
        chi2_trials: list[float] | None = None,
        rho2_trials: list[float] | None = None,
        eta2_trials: list[float] | None = None,
        shift_x_trials: list[float] | None = None,
        shift_y_trials: list[float] | None = None,
        shift_valid_trials: list[bool] | None = None,
    ) -> None:
        if active_trial_q0 is not None and active_trial_index is not None:
            self.update_active_q0(a_value=float(a_value), b_value=float(b_value), q0_value=float(active_trial_q0))
        if completed_trial_index is None or completed_trial_raw_map is None:
            return
        metric_value = float(metric_trials[completed_trial_index]) if completed_trial_index < len(metric_trials) else float("nan")
        q0_value = float(q0_trials[completed_trial_index]) if completed_trial_index < len(q0_trials) else float("nan")
        trial_chi2 = None
        trial_rho2 = None
        trial_eta2 = None
        if chi2_trials is not None and completed_trial_index < len(chi2_trials):
            trial_chi2 = float(chi2_trials[completed_trial_index])
        if rho2_trials is not None and completed_trial_index < len(rho2_trials):
            trial_rho2 = float(rho2_trials[completed_trial_index])
        if eta2_trials is not None and completed_trial_index < len(eta2_trials):
            trial_eta2 = float(eta2_trials[completed_trial_index])
        committed = int(self._trials_committed.get((float(a_value), float(b_value)), 0))
        best_index = int(completed_trial_index)
        best_metric = float(metric_value)
        if metric_trials:
            finite = [(idx, float(v)) for idx, v in enumerate(metric_trials) if np.isfinite(float(v))]
            if finite:
                best_index, best_metric = min(finite, key=lambda item: item[1])
        shift_kwargs = _grid_trial_shift_commit_kwargs(
            trial_index=int(completed_trial_index),
            shift_x_trials=shift_x_trials,
            shift_y_trials=shift_y_trials,
            shift_valid_trials=shift_valid_trials,
        )
        merged_metadata = {**dict(shift_kwargs.get("trial_metadata") or {}), **dict(trial_metadata or {})}
        self._writer.write_grid_event(
            GridTrialCommittedEvent(
                point_id=str(self.point_id_for(a_value, b_value) or ""),
                trial_index=int(completed_trial_index),
                q0=float(q0_value),
                metric=float(metric_value),
                next_q0=float(q0_value),
                best_trial_index=int(best_index),
                best_metric=float(best_metric),
                raw_modeled_map=np.asarray(completed_trial_raw_map, dtype=np.float32),
                trial_metadata=merged_metadata,
                shift_x=shift_kwargs.get("shift_x"),
                shift_y=shift_kwargs.get("shift_y"),
                shift_valid=shift_kwargs.get("shift_valid"),
                chi2=trial_chi2,
                rho2=trial_rho2,
                eta2=trial_eta2,
            )
        )
        self._trials_committed[(float(a_value), float(b_value))] = max(
            committed,
            int(completed_trial_index) + 1,
        )

    def _commit_completed_point_from_payload(
        self,
        *,
        a_value: float,
        b_value: float,
        payload: dict[str, Any],
        result: ABPointResult,
    ) -> None:
        point_key = (float(a_value), float(b_value))
        point_id = self.point_id_for(a_value, b_value)
        if point_id is None:
            q0_start = float(payload.get("q0", result.q0))
            self.assign_grid_points([(float(a_value), float(b_value), q0_start)])
            point_id = self.point_id_for(a_value, b_value)
        if point_id is None:
            raise RuntimeError("grid point assignment did not produce a point_id")
        q0_trials = [float(v) for v in payload.get("fit_q0_trials", result.trial_q0)]
        metric_trials = [float(v) for v in payload.get("fit_metric_trials", result.trial_objective_values)]
        chi2_trials = [float(v) for v in payload.get("fit_chi2_trials", result.trial_chi2_values)]
        rho2_trials = [float(v) for v in payload.get("fit_rho2_trials", result.trial_rho2_values)]
        eta2_trials = [float(v) for v in payload.get("fit_eta2_trials", result.trial_eta2_values)]
        shift_x_trials = payload.get("fit_shift_x_trials", result.trial_shift_x_arcsec)
        shift_y_trials = payload.get("fit_shift_y_trials", result.trial_shift_y_arcsec)
        shift_valid_trials = payload.get("fit_find_shift_valid_trials", result.trial_find_shift_valid)
        for trial_index, q0_value in enumerate(q0_trials):
            raw_map = None
            trial_maps = payload.get("trial_raw_modeled_maps")
            if trial_maps is not None:
                arr = np.asarray(trial_maps, dtype=float)
                if arr.ndim == 3 and trial_index < arr.shape[0]:
                    raw_map = np.asarray(arr[trial_index], dtype=np.float32)
            if raw_map is None:
                raw_map = self.trial_raw_map_for(float(a_value), float(b_value), float(q0_value))
            metric_value = float(metric_trials[trial_index]) if trial_index < len(metric_trials) else float("nan")
            finite_metrics = [
                (idx, float(metric_trials[idx]))
                for idx in range(len(metric_trials))
                if idx < len(metric_trials) and np.isfinite(float(metric_trials[idx]))
            ]
            best_index = trial_index
            best_metric = float(metric_value)
            if finite_metrics:
                best_index, best_metric = min(finite_metrics, key=lambda item: item[1])
            shift_kwargs = _grid_trial_shift_commit_kwargs(
                trial_index=int(trial_index),
                payload=payload,
                result=result,
                shift_x_trials=shift_x_trials,
                shift_y_trials=shift_y_trials,
                shift_valid_trials=shift_valid_trials,
            )
            self._writer.write_grid_event(
                GridTrialCommittedEvent(
                    point_id=str(point_id),
                    trial_index=int(trial_index),
                    q0=float(q0_value),
                    metric=float(metric_value),
                    next_q0=float(q0_value),
                    best_trial_index=int(best_index),
                    best_metric=float(best_metric),
                    raw_modeled_map=None if raw_map is None else np.asarray(raw_map, dtype=np.float32),
                    trial_metadata=dict(shift_kwargs.get("trial_metadata") or {}),
                    shift_x=shift_kwargs.get("shift_x"),
                    shift_y=shift_kwargs.get("shift_y"),
                    shift_valid=shift_kwargs.get("shift_valid"),
                    chi2=float(chi2_trials[trial_index]) if trial_index < len(chi2_trials) else None,
                    rho2=float(rho2_trials[trial_index]) if trial_index < len(rho2_trials) else None,
                    eta2=float(eta2_trials[trial_index]) if trial_index < len(eta2_trials) else None,
                )
            )
        finite_metrics = [
            (idx, float(metric_trials[idx]))
            for idx in range(len(metric_trials))
            if np.isfinite(float(metric_trials[idx]))
        ]
        best_index = int(np.nanargmin(np.asarray(metric_trials, dtype=float))) if metric_trials else 0
        best_metric = float(metric_trials[best_index]) if metric_trials else float(result.objective_value)
        if finite_metrics:
            best_index, best_metric = min(finite_metrics, key=lambda item: item[1])
        self._writer.write_grid_event(
            GridPointCompletedEvent(
                point_id=str(point_id),
                best_trial_index=int(best_index),
                best_metric=float(best_metric),
                best_q0=float(q0_trials[best_index]) if q0_trials else float(result.q0),
            )
        )
        self._trials_committed[point_key] = len(q0_trials)

    @property
    def streaming_renderer_factory(self) -> Any:
        slice_key = str(self._diagnostics.get("target_slice_key") or self._diagnostics.get("slice_key") or "").strip() or None
        search_id = str(
            self._diagnostics.get("selected_search_id") or self._diagnostics.get("search_id") or ""
        ).strip() or None
        return _StreamingRendererFactory(
            self._renderer_factory,
            stream=self._render_stream,
            observed_template=self._observed,
            target_metric=self._target_metric,
            psf_source=self._psf_source,
            compatibility_signature=self._compatibility_signature,
            store_trial_map_cubes=self._store_trial_map_cubes,
            artifact_h5=self._artifact_h5,
            slice_key=slice_key,
            search_id=search_id,
        )

    def close(self) -> None:
        self._writer.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def trial_raw_map_for(self, a_value: float, b_value: float, q0_value: float) -> np.ndarray | None:
        record = self._render_stream.snapshot_record(a_value=float(a_value), b_value=float(b_value))
        if record is None:
            return None
        raw_map = _lookup_stream_value_by_q0(record.raw_modeled_by_q0, float(q0_value))
        if raw_map is None:
            return None
        return np.asarray(raw_map, dtype=np.float32)

    def drop_persisted_trial_render(self, a_value: float, b_value: float, q0_value: float) -> None:
        self._render_stream.drop_trial_render(
            a_value=float(a_value),
            b_value=float(b_value),
            q0_value=float(q0_value),
        )

    def flush_pending_writes(self) -> None:
        self._writer.drain()

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
                psf_kernel=self._psf_kernel,
                tr_region_mask=getattr(self._renderer_factory, "tr_region_mask", None),
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
            self._commit_completed_point_from_payload(
                a_value=float(point.a),
                b_value=float(point.b),
                payload=point_payload,
                result=point,
            )
            self._writer.drain()
            self._point_map[key] = point
            promoted_count += 1
        if promoted_count and self._viewer_heartbeat is not None:
            self._viewer_heartbeat.set_phase(f"{promoted_count} map_store point(s) promoted")
        return promoted_count

    def hydrate_from_existing(self) -> int:
        if not self._artifact_h5.exists():
            return 0
        try:
            payload = load_scan_file(self._artifact_h5, slice_key=self._target_slice_key(), include_maps=False)
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
            key = (float(point.a), float(point.b))
            self._point_map[key] = point
            grid_point_id = str(dict(record.get("diagnostics") or {}).get("grid_point_id") or "").strip()
            if grid_point_id:
                self._point_ids[key] = grid_point_id
            count += 1
        import h5py

        slice_key = self._target_slice_key()
        search_id = str(self._diagnostics.get("selected_search_id") or self._diagnostics.get("search_id") or "").strip()
        if slice_key and search_id:
            try:
                with h5py.File(str(self._artifact_h5), "r") as f:
                    if SLICE_CONTAINER_GROUP in f and slice_key in f[SLICE_CONTAINER_GROUP]:
                        slice_group = f[SLICE_CONTAINER_GROUP][slice_key]
                        if SEARCHES_GROUP in slice_group and search_id in slice_group[SEARCHES_GROUP]:
                            for header in list_grid_point_headers(slice_group[SEARCHES_GROUP][search_id]):
                                key = (float(header["a"]), float(header["b"]))
                                self._point_ids[key] = str(header["point_id"])
                                self._trials_committed[key] = int(header.get("n_trials", 0))
                                state = classify_grid_point_state(header)
                                if state == "complete":
                                    continue
                                if state == "failed":
                                    if not self._retry_failed:
                                        continue
                                    resume_q0 = float(header.get("q0_start", np.nan))
                                elif state == "assigned_no_trials":
                                    resume_q0 = float(header.get("q0_start", header.get("next_q0", np.nan)))
                                else:
                                    if self._resume_incomplete_from == "q0_start":
                                        resume_q0 = float(header.get("q0_start", np.nan))
                                    else:
                                        resume_q0 = float(header.get("next_q0", header.get("q0_start", np.nan)))
                                if np.isfinite(resume_q0):
                                    self._resume_q0[key] = float(resume_q0)
            except Exception:
                pass
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
            current_payload = load_scan_file(self._artifact_h5, slice_key=self._target_slice_key(), include_maps=False)
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
                    psf_kernel=self._psf_kernel,
                    tr_region_mask=getattr(self._renderer_factory, "tr_region_mask", None),
                )
                if rescored is None:
                    continue
                point, point_payload = rescored
                point_payload["diagnostics"] = {
                    **dict(point_payload.get("diagnostics") or {}),
                    COMPATIBILITY_SIGNATURE_KEY: self._compatibility_signature,
                }
                self._commit_completed_point_from_payload(
                    a_value=float(point.a),
                    b_value=float(point.b),
                    payload=point_payload,
                    result=point,
                )
                self._writer.drain()
                self._point_map[key] = point
                promoted_count += 1
        if promoted_count:
            self._writer.drain()
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
        self._commit_completed_point_from_payload(
            a_value=float(value.a),
            b_value=float(value.b),
            payload=payload,
            result=value,
        )
        self._writer.drain()
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
    parser.add_argument("--a-min", type=float, default=-1.2, help="Adaptive search lower a bound")
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
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="On resume, requeue grid points whose status is FAILED.",
    )
    parser.add_argument(
        "--resume-incomplete-from",
        choices=("next_q0", "q0_start"),
        default="next_q0",
        help="When resuming incomplete grid points, seed Q0 from stored next_q0 or original q0_start.",
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
    add_chmp_search_cli_arguments(parser)
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
    target_slice_key = next(
        (str(item["key"]) for item in slice_descriptors if bool(item.get("is_target"))),
        str(slice_descriptors[0]["key"]) if slice_descriptors else "default",
    )

    artifacts_dir = _coerce_path(args.artifacts_dir) or (repo_root / "ab_scan_artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stem = args.artifacts_stem or _default_artifact_stem(obs_request, target_metric=str(args.target_metric))
    artifact_h5 = _coerce_path(args.artifact_h5) or (artifacts_dir / f"{stem}.h5")
    try:
        artifact_h5 = artifact_h5.expanduser().resolve()
    except Exception:
        artifact_h5 = artifact_h5.expanduser()
    artifact_preexisting = artifact_h5.exists()
    resume_slice_payload = (
        _load_slice_preflight_payload(
            artifact_h5=artifact_h5,
            slice_key=target_slice_key,
            include_maps=True,
        )
        if artifact_preexisting and not bool(args.recompute_existing)
        else None
    )
    resume_slice_diagnostics = dict(resume_slice_payload.get("diagnostics") or {}) if resume_slice_payload is not None else {}
    if (
        render_selection.domain != "mw"
        and render_selection.euv_response_sav is None
        and not bool(args.recompute_existing)
    ):
        cached_response_override = str(resume_slice_diagnostics.get("euv_response_override_path") or "").strip()
        if cached_response_override:
            override_path = Path(cached_response_override).expanduser()
            if override_path.exists():
                render_selection = replace(
                    render_selection,
                    euv_response_sav=override_path.resolve(),
                )
                print(f"  EUV response preload: restored override from artifact diagnostics ({override_path})")
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
        existing_viewer_pid = _find_existing_viewer_pid(viewer_script=viewer_script, artifact_h5=artifact_h5)
        if existing_viewer_pid is not None:
            if _focus_existing_viewer_pid(int(existing_viewer_pid)):
                print(f"Reusing existing pychmp-view ({phase}) pid={existing_viewer_pid} (brought to front)")
                return
            print(
                f"Existing pychmp-view detected ({phase}) pid={existing_viewer_pid} but could not be focused; launching a fresh viewer"
            )
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

    # Launch as early as possible for existing artifacts so long preflight work
    # (e.g. PSF/model setup) does not look like a stalled run.
    if artifact_h5.exists():
        _maybe_launch_viewer("preflight")

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
    selected_psf_metadata = None
    if (
        artifact_preexisting
        and not bool(args.override_header_psf)
        and psf_bmaj_arcsec is None
        and psf_bmin_arcsec is None
        and psf_bpa_deg is None
    ):
        cached_slice_psf = _load_slice_psf_metadata_from_artifact(
            artifact_h5=artifact_h5,
            slice_key=target_slice_key,
        )
        if cached_slice_psf is not None:
            selected_psf_metadata = cached_slice_psf
            print(f"  PSF preload: restored kernel from artifact slice metadata ({target_slice_key})")
    if selected_psf_metadata is None:
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
    geometry_observer_name = (
        None if bool(geometry_policy.use_model_saved_fov) and not explicit_observer_requested else str(args.observer or geometry_policy.observer_name)
    )
    geometry_observer = None if bool(geometry_policy.use_model_saved_fov) else (observer_overrides if explicit_observer_requested else None)
    resolved_geometry = resolve_render_geometry_via_gxrender(
        model_path=model_h5,
        model_format="auto",
        ebtel_path=str(ebtel_path) if ebtel_path is not None else None,
        pixel_scale_arcsec=float(args.pixel_scale_arcsec),
        observer_name=geometry_observer_name,
        observer=geometry_observer,
        omp_threads=int(getattr(args, "omp_threads", 8)),
        use_saved_fov=bool(geometry_policy.use_model_saved_fov),
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

    model_obs_time = str(model_observer_meta.get("observer_obs_time") or load_model_obs_time_text(model_h5) or "").strip()
    obs_time_text = str(obs_map.date_obs or header.get("DATE-OBS", header.get("DATE_OBS", "")) or "").strip()

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

    existing_format = detect_scan_artifact_format(artifact_h5) if artifact_preexisting else None
    if artifact_preexisting and existing_format not in {None, "sparse", "unified"}:
        raise SystemExit(
            f"Existing artifact {artifact_h5} is rectangular; this adaptive example requires a sparse artifact path."
        )

    model_sha256 = _compute_file_sha256(model_h5)
    forward_model_identity = compute_forward_model_identity_placeholder(model_path=model_h5)
    forward_model_sha256 = str(forward_model_identity.sha256)
    forward_model_identity_version = str(forward_model_identity.version)
    artifact_geometry_block = build_artifact_geometry_block(
        {
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
    )
    artifact_geometry_sha256_value = artifact_geometry_sha256(artifact_geometry_block)
    observation_source_path = obs_map.source_path
    observation_source_file = _resolve_existing_file(observation_source_path)
    observation_source_sha256 = (
        _compute_file_sha256(observation_source_file)
        if observation_source_file is not None and observation_source_file.is_file()
        else None
    )
    ebtel_sha256 = _compute_file_sha256(ebtel_path)
    stored_slice_payload = resume_slice_payload if artifact_preexisting and not bool(args.recompute_existing) else None
    sigma_map_full: np.ndarray | None = None
    noise_diag: dict[str, Any] = {}
    if stored_slice_payload is None:
        print("Estimating noise from map...")
        noise_result = estimate_obs_map_noise(obs_map, method="histogram_clip")
        sigma_map_full = np.asarray(noise_result.sigma_map, dtype=float)
        noise_diag = dict(noise_result.diagnostics)
        noise_unit = obs_map_noise_unit_label(obs_map)
        print(
            f"  Estimated sigma: {float(noise_result.sigma):.2f} {noise_unit} "
            f"(method={str(noise_result.method_used)})"
        )
    try:
        shift_policy, max_shift_arcsec, xy_shift_arcsec = resolve_shift_policy_from_args(args)
        slice_obs_ref = resolve_slice_observation_reference(
            observed,
            header,
            target_header,
            sigma=sigma_map_full,
            observation_source_sha256=observation_source_sha256,
            artifact_geometry_sha256=artifact_geometry_sha256_value,
            model_time_text=model_obs_time or None,
            observation_time_text=obs_time_text or None,
            stored_slice_payload=stored_slice_payload,
            force_recompute=bool(args.recompute_existing),
            shift_policy=shift_policy,
            max_shift_arcsec=max_shift_arcsec,
            xy_shift_arcsec=xy_shift_arcsec,
            slice_canvas_max_shift_arcsec=max_shift_arcsec,
        )
    except SliceObservationReferenceError as exc:
        raise SystemExit(str(exc)) from exc
    observed_cropped = np.asarray(slice_obs_ref.observed, dtype=float)
    sigma_cropped = np.asarray(slice_obs_ref.sigma, dtype=float)
    header = slice_obs_ref.source_header
    obs_time_alignment_diag = dict(slice_obs_ref.diagnostics)
    if slice_obs_ref.restored_from_artifact:
        noise_diag = dict(resume_slice_diagnostics.get("noise_diagnostics") or noise_diag)
        finite_sigma = sigma_cropped[np.isfinite(sigma_cropped)]
        restored_sigma = float(np.nanmedian(finite_sigma)) if finite_sigma.size else float("nan")
        noise_unit = obs_map_noise_unit_label(obs_map)
        print(
            "  Slice observation reference: restored rotated+regridded observed/sigma maps "
            f"from artifact slice metadata (median sigma={restored_sigma:.2f} {noise_unit})"
        )
    else:
        for warning_line in obs_time_alignment_diag.get("observation_time_warning_lines", ()):
            print(warning_line)
        if obs_time_alignment_diag.get("observation_time_rotation_applied"):
            print(f"  {obs_time_alignment_diag.get('observation_time_alignment_message', '')}")
        elif str(obs_time_alignment_diag.get("observation_time_alignment", "")) not in {"exact", "unknown"}:
            print(f"  {obs_time_alignment_diag.get('observation_time_alignment_message', '')}")

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

    metrics_mask_type = "explicit_fits" if explicit_metric_mask is not None else "union"
    chmp_settings = resolve_chmp_search_settings(
        args,
        mask_type=metrics_mask_type,
        explicit_mask=explicit_metric_mask,
    )
    if len(chmp_settings.q0_search_stages) > 1:
        print(f"  Q0 search stages: {', '.join(chmp_settings.q0_search_stages)}")
    print(
        "  CHMP evaluation: "
        f"shift_policy={chmp_settings.shift_policy} "
        f"smoothed_obs_max={chmp_settings.use_smoothed_obs_max} "
        f"emthreshold_gate={chmp_settings.use_emthreshold}"
    )

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
        if psf_kernel is not None:
            original_kernel_shape = tuple(int(v) for v in np.asarray(psf_kernel, dtype=float).shape)
            psf_kernel = _compact_kernel_for_target_shape(
                np.asarray(psf_kernel, dtype=float),
                target_ny=int(geometry.ny),
                target_nx=int(geometry.nx),
            )
            compact_kernel_shape = tuple(int(v) for v in np.asarray(psf_kernel, dtype=float).shape)
            if compact_kernel_shape != original_kernel_shape:
                print(
                    "  PSF compact: "
                    f"cropped kernel {original_kernel_shape} -> {compact_kernel_shape} "
                    "for target-grid convolution"
                )
                if isinstance(resolved_psf_meta, dict):
                    resolved_psf_meta = {
                        **resolved_psf_meta,
                        "psf_kernel_shape_original": original_kernel_shape,
                        "psf_kernel_shape": compact_kernel_shape,
                        "psf_kernel_compacted": True,
                    }

    euv_response_identity = None
    euv_response_identity_version = None
    euv_response_sha256 = None
    euv_response_identity_summary = None
    if render_selection.domain != "mw":
        reused_identity = False
        if artifact_preexisting and not bool(args.recompute_existing):
            cached_version = str(resume_slice_diagnostics.get("euv_response_identity_version") or "").strip()
            cached_sha = str(resume_slice_diagnostics.get("euv_response_sha256") or "").strip()
            cached_summary = resume_slice_diagnostics.get("euv_response_identity_summary")
            if cached_version and cached_sha and isinstance(cached_summary, dict):
                euv_response_identity_version = cached_version
                euv_response_sha256 = cached_sha
                euv_response_identity_summary = dict(cached_summary)
                reused_identity = True
                print("  EUV response identity: reused cached artifact diagnostics")
        if not reused_identity:
            identity_started = time.perf_counter()
            euv_response_identity = resolve_euv_response_identity(
                model_path=str(model_h5),
                channel=str(render_selection.euv_channel),
                render_channels=render_channels,
                instrument=str(render_selection.euv_instrument),
                response_sav=render_selection.euv_response_sav,
                ebtel_path=str(ebtel_path),
                tbase=float(args.tbase),
                nbase=float(args.nbase),
                a=float(args.a_start),
                b=float(args.b_start),
                geometry=geometry,
                observer=observer_overrides,
                observer_name=effective_observer_name,
                tr_region_mask=euv_tr_mask,
                pixel_scale_arcsec=float(args.pixel_scale_arcsec),
            )
            if euv_response_identity is not None:
                euv_response_identity_version = str(euv_response_identity.version)
                euv_response_sha256 = str(euv_response_identity.sha256)
                euv_response_identity_summary = dict(euv_response_identity.summary)
            identity_elapsed = time.perf_counter() - identity_started
            print(f"  EUV response identity: computed in {identity_elapsed:.2f}s")
    euv_response_origin = "pyEUVTools" if render_selection.euv_response_sav is None else "response_sav"
    euv_response_override_path = None if render_selection.euv_response_sav is None else str(render_selection.euv_response_sav)
    euv_response_resolver = "pychmp.gxrender_adapter.resolve_euv_response_identity"

    preflight_diag = {
        "artifact_kind": "pychmp_ab_scan_sparse_points",
        "spectral_domain": str(render_selection.domain),
        "spectral_label": str(render_selection.spectral_label),
        "target_slice_key": target_slice_key,
        "model_sha256": str(model_sha256),
        "forward_model_sha256": str(forward_model_sha256),
        "forward_model_identity_version": str(forward_model_identity_version),
        "artifact_geometry_sha256": str(artifact_geometry_sha256_value),
        "fits_sha256": str(observation_source_sha256 or ""),
        "ebtel_sha256": str(ebtel_sha256),
        "euv_response_identity_version": euv_response_identity_version,
        "euv_response_sha256": euv_response_sha256,
        "euv_response_origin": euv_response_origin,
        "euv_response_override_path": euv_response_override_path,
        "euv_response_resolver": euv_response_resolver,
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
    try:
        _maybe_validate_artifact_preflight(
            artifact_h5=artifact_h5,
            artifact_preexisting=artifact_preexisting,
            recompute_existing=bool(args.recompute_existing),
            target_slice_key=target_slice_key,
            target_header=target_header,
            diagnostics=preflight_diag,
        )
    except ScanArtifactCompatibilityError as exc:
        raise SystemExit(str(exc)) from exc

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
        "forward_model_sha256": str(forward_model_sha256),
        "forward_model_identity_version": str(forward_model_identity_version),
        "artifact_geometry_sha256": str(artifact_geometry_sha256_value),
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
        "euv_response_identity_version": euv_response_identity_version,
        "euv_response_sha256": euv_response_sha256,
        "euv_response_origin": euv_response_origin,
        "euv_response_override_path": euv_response_override_path,
        "euv_response_resolver": euv_response_resolver,
        "euv_response_source": (
            None if euv_response_identity_summary is None else euv_response_identity_summary.get("source")
        ),
        "euv_response_mode": None if euv_response_identity_summary is None else euv_response_identity_summary.get("mode"),
        "euv_response_identity_summary": (
            None if euv_response_identity_summary is None else dict(euv_response_identity_summary)
        ),
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
        **obs_time_alignment_diag,
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
        "shift_policy": chmp_settings.shift_policy,
        "max_shift_arcsec": chmp_settings.max_shift_arcsec,
        "xy_shift_arcsec": [float(chmp_settings.xy_shift_arcsec[0]), float(chmp_settings.xy_shift_arcsec[1])],
        "use_smoothed_obs_max": bool(chmp_settings.use_smoothed_obs_max),
        "use_emthreshold": bool(chmp_settings.use_emthreshold),
        "emthreshold": float(chmp_settings.emthreshold),
        "q0_search_stages": [str(stage) for stage in chmp_settings.q0_search_stages],
    }
    compatibility_signature = compatibility_signature_from_diagnostics(
        root_diag,
        layout={"kind": "point_list"},
    )
    root_diag[COMPATIBILITY_SIGNATURE_KEY] = compatibility_signature
    target_search_id = search_id_from_evaluation_config(root_diag, layout={"kind": "point_list"})
    root_diag["selected_search_id"] = target_search_id
    root_diag["search_id"] = target_search_id
    root_diag["search_active"] = True
    root_diag["contract_version"] = GRID_POINTS_CONTRACT_VERSION
    common_blos_reference = blos_reference_for_fov

    viewer_refresh_signal = Path(f"{artifact_h5}.refresh")
    viewer_heartbeat = _ViewerRefreshHeartbeat(
        viewer_refresh_signal,
        slice_key=str(root_diag.get("target_slice_key") or "").strip() or None,
        search_id=target_search_id,
    )
    print(f"Viewer heartbeat signal: {viewer_refresh_signal}")
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
            psf_kernel=None if psf_kernel is None else np.asarray(psf_kernel, dtype=float),
        )
        print(f"Start over: resetting search history in {artifact_h5} while seeding from stored compatible maps")
    artifact_initialized_this_run = False
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
            observation_canvas=slice_obs_ref.observation_canvas,
            sigma_canvas=slice_obs_ref.sigma_canvas,
            canvas_wcs_header=slice_obs_ref.canvas_header,
        )
        from pychmp.ab_scan_artifacts import _sync_preprocessed_content_identity_diagnostics

        root_diag.update(
            _sync_preprocessed_content_identity_diagnostics(
                root_diag,
                observed=observed_cropped,
                sigma_map=sigma_cropped,
                observation_canvas=slice_obs_ref.observation_canvas,
                sigma_canvas=slice_obs_ref.sigma_canvas,
            )
        )
        artifact_initialized_this_run = True
        resolved_search_id = read_slice_active_search_id(artifact_h5, slice_key=target_slice_key)
        if resolved_search_id:
            target_search_id = resolved_search_id
            root_diag["selected_search_id"] = resolved_search_id
            root_diag["search_id"] = resolved_search_id
        viewer_heartbeat.set_search_id(target_search_id)
        viewer_heartbeat.set_phase("initialized")
        viewer_heartbeat.notify_refresh()
        print(f"Initialized sparse artifact: {artifact_h5}")
    elif artifact_preexisting and not bool(args.recompute_existing) and not _artifact_has_target_slice(
        artifact_h5, target_slice_key
    ):
        print(f"Initializing target slice shell in existing artifact: {target_slice_key}")
        write_point_scan_artifact(
            artifact_h5,
            observed=observed_cropped,
            sigma_map=sigma_cropped,
            wcs_header=target_header,
            diagnostics=root_diag,
            blos_reference=common_blos_reference,
            psf_kernel=None if psf_kernel is None else np.asarray(psf_kernel, dtype=float),
            point_records=[],
            run_history=existing_run_history,
            preserve_existing_searches=True,
            observation_canvas=slice_obs_ref.observation_canvas,
            sigma_canvas=slice_obs_ref.sigma_canvas,
            canvas_wcs_header=slice_obs_ref.canvas_header,
        )
        from pychmp.ab_scan_artifacts import _sync_preprocessed_content_identity_diagnostics

        root_diag.update(
            _sync_preprocessed_content_identity_diagnostics(
                root_diag,
                observed=observed_cropped,
                sigma_map=sigma_cropped,
                observation_canvas=slice_obs_ref.observation_canvas,
                sigma_canvas=slice_obs_ref.sigma_canvas,
            )
        )
        artifact_initialized_this_run = True
        resolved_search_id = read_slice_active_search_id(artifact_h5, slice_key=target_slice_key)
        if resolved_search_id:
            target_search_id = resolved_search_id
            root_diag["selected_search_id"] = resolved_search_id
            root_diag["search_id"] = resolved_search_id
            viewer_heartbeat.set_search_id(resolved_search_id)
        viewer_heartbeat.set_phase("initialized")
        viewer_heartbeat.notify_refresh()

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
        forward_model_sha256=str(forward_model_sha256),
        forward_model_identity_version=str(forward_model_identity_version),
        artifact_geometry_sha256=str(artifact_geometry_sha256_value),
        ebtel_sha256=str(ebtel_sha256),
        euv_response_sha256=euv_response_sha256,
        euv_response_identity_version=euv_response_identity_version,
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
    cache.set_recompute_existing(bool(args.recompute_existing))
    cache.set_resume_policy(
        retry_failed=bool(getattr(args, "retry_failed", False)),
        resume_incomplete_from=str(getattr(args, "resume_incomplete_from", "next_q0")),
    )
    search_renderer_factory = cache.streaming_renderer_factory
    try:
        if artifact_initialized_this_run and not bool(args.recompute_existing):
            print("Resume preload: skipped for artifact initialized in this run")
            reused_points = 0
        elif not bool(args.recompute_existing):
            print("Resume preload: loading compatible points from existing artifact...")
            reused_points = cache.hydrate_from_existing()
        else:
            reused_points = 0
        promoted_same_slice = 0
        promoted_auxiliary = 0
        if not bool(args.recompute_existing):
            promoted_same_slice = cache.promote_current_slice_trial_maps(
                threshold=float(args.metrics_mask_threshold),
                explicit_mask=explicit_metric_mask,
                include_matching_signature=False,
            )
            promoted_auxiliary = cache.promote_auxiliary_maps_from_store(
                threshold=float(args.metrics_mask_threshold),
                explicit_mask=explicit_metric_mask,
            )
            if promoted_same_slice or promoted_auxiliary:
                cache.flush_pending_writes()
            reused_points += promoted_same_slice + promoted_auxiliary
            print(
                f"Resume preload: loaded {reused_points} compatible point(s) "
                f"(hydrated={reused_points - promoted_same_slice - promoted_auxiliary}, "
                f"same_slice={promoted_same_slice}, auxiliary={promoted_auxiliary})"
            )
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
    viewer_heartbeat.set_phase("search initialized")
    viewer_heartbeat.notify_refresh()
    if str(args.execution_policy) != "serial":
        print(
            "  Note: process-pool mode reports progress per completed point; "
            "per-trial console/viewer updates remain serial-only"
        )
    progress_start_callback = None
    progress_callback = None
    if str(args.execution_policy) == "serial":
        progress_start_callback, progress_callback = _make_trial_progress_reporter(target_metric=str(args.target_metric))
        if viewer_heartbeat is not None:
            live_q0_trials: list[float] = []
            live_metric_trials: list[float] = []
            live_chi2_trials: list[float] = []
            live_rho2_trials: list[float] = []
            live_eta2_trials: list[float] = []
            live_shift_x_trials: list[float] = []
            live_shift_y_trials: list[float] = []
            live_shift_valid_trials: list[bool] = []
            active_point: dict[str, tuple[float, float] | None] = {"value": None}
            console_start_callback = progress_start_callback
            console_progress_callback = progress_callback

            def _viewer_point_start(a_value: float, b_value: float) -> None:
                active_point["value"] = (float(a_value), float(b_value))
                live_q0_trials.clear()
                live_metric_trials.clear()
                live_chi2_trials.clear()
                live_rho2_trials.clear()
                live_eta2_trials.clear()
                live_shift_x_trials.clear()
                live_shift_y_trials.clear()
                live_shift_valid_trials.clear()
                cache.clear_live_trial_snapshot()

            def _viewer_point_complete(a_value: float, b_value: float) -> None:
                _ = (a_value, b_value)
                cache.clear_live_trial_snapshot()

            def _viewer_progress_start(trial_index: int, q0: float) -> None:
                if console_start_callback is not None:
                    console_start_callback(trial_index, q0)
                point = active_point.get("value")
                if point is None:
                    return
                cache.write_live_trial_snapshot(
                    a_value=float(point[0]),
                    b_value=float(point[1]),
                    q0_trials=list(live_q0_trials),
                    metric_trials=list(live_metric_trials),
                    chi2_trials=list(live_chi2_trials),
                    rho2_trials=list(live_rho2_trials),
                    eta2_trials=list(live_eta2_trials),
                    shift_x_trials=list(live_shift_x_trials),
                    shift_y_trials=list(live_shift_y_trials),
                    shift_valid_trials=list(live_shift_valid_trials),
                    active_trial_index=int(trial_index),
                    active_trial_q0=float(q0),
                )
                cache.flush_pending_writes()

            def _viewer_progress_report(
                q0: float,
                objective_value: float,
                is_valid: bool,
                message: str,
                elapsed_s: float,
                metrics: MetricValues,
                evaluation: Any = None,
            ) -> None:
                if console_progress_callback is not None:
                    console_progress_callback(q0, objective_value, is_valid, message, elapsed_s, metrics)
                point = active_point.get("value")
                if point is None:
                    return
                live_q0_trials.append(float(q0))
                live_metric_trials.append(float(objective_value))
                live_chi2_trials.append(float(metrics.chi2))
                live_rho2_trials.append(float(metrics.rho2))
                live_eta2_trials.append(float(metrics.eta2))
                if evaluation is not None:
                    live_shift_x_trials.append(float(getattr(evaluation, "shift_x_arcsec", np.nan)))
                    live_shift_y_trials.append(float(getattr(evaluation, "shift_y_arcsec", np.nan)))
                    live_shift_valid_trials.append(bool(getattr(evaluation, "find_shift_valid", True)))
                else:
                    live_shift_x_trials.append(float("nan"))
                    live_shift_y_trials.append(float("nan"))
                    live_shift_valid_trials.append(True)
                completed_index = int(len(live_q0_trials) - 1)
                completed_raw_map = cache.trial_raw_map_for(float(point[0]), float(point[1]), float(q0))
                cache.write_live_trial_snapshot(
                    a_value=float(point[0]),
                    b_value=float(point[1]),
                    q0_trials=list(live_q0_trials),
                    metric_trials=list(live_metric_trials),
                    chi2_trials=list(live_chi2_trials),
                    rho2_trials=list(live_rho2_trials),
                    eta2_trials=list(live_eta2_trials),
                    shift_x_trials=list(live_shift_x_trials),
                    shift_y_trials=list(live_shift_y_trials),
                    shift_valid_trials=list(live_shift_valid_trials),
                    completed_trial_index=completed_index,
                    completed_trial_raw_map=completed_raw_map,
                )
                cache.flush_pending_writes()
                if completed_raw_map is not None:
                    cache.drop_persisted_trial_render(float(point[0]), float(point[1]), float(q0))

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
            observation_reference=slice_obs_ref,
            q0_search_stages=chmp_settings.q0_search_stages,
            use_smoothed_obs_max=chmp_settings.use_smoothed_obs_max,
            use_emthreshold=chmp_settings.use_emthreshold,
            emthreshold=chmp_settings.emthreshold,
        )
    except Exception:
        root_diag["search_active"] = False
        finalize_search_runner_state(
            artifact_h5,
            slice_key=str(root_diag.get("target_slice_key") or "").strip() or None,
            search_id=target_search_id,
            completed=False,
        )
        viewer_heartbeat.stop("adaptive search failed")
        cache.close()
        raise
    root_diag["search_active"] = False
    cache._diagnostics["search_active"] = False
    finalize_search_runner_state(
        artifact_h5,
        slice_key=str(root_diag.get("target_slice_key") or "").strip() or None,
        search_id=target_search_id,
        completed=True,
    )
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
