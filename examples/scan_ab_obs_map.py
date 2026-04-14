#!/usr/bin/env python
"""Scan a rectangular `(a, b)` grid against a real observational map.

This is the first usable single-frequency `MultiScanAB`-style workflow for
pyCHMP. It reuses the validated observational preprocessing path from
`fit_q0_obs_map.py`, then fits one best `q0` per `(a, b)` point and stores all
results in one consolidated HDF5 file.
"""

from __future__ import annotations

import argparse
import csv
from importlib import import_module
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from astropy.io import fits

from pychmp import GXRenderMWContext, estimate_map_noise, fit_q0_to_observation
from pychmp.ab_scan_artifacts import (
    COMPATIBILITY_SIGNATURE_KEY,
    ScanArtifactCompatibilityError,
    append_point_record,
    build_computed_point_payload,
    detect_scan_artifact_format,
    load_scan_file,
    load_run_history,
    point_record_matches_compatibility_signature,
    save_rectangular_scan_file,
    slice_descriptor_from_diagnostics,
    validate_scan_artifact_compatibility,
    write_sparse_scan_file,
)
from pychmp.ab_scan_execution import ABExecutionSettings, iter_execute_tasks, resolve_execution_plan
from pychmp.ab_scan_tasks import ABSliceTaskDescriptor, compile_rectangular_point_tasks, compile_sparse_point_tasks
from pychmp.ab_search import idl_q0_start_heuristic


def _build_command_compatibility_signature(argv: list[str]) -> str:
    normalized = json.dumps([str(item) for item in argv], separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _build_physical_compatibility_signature(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, separators=(",", ":"), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _build_run_history_entry(
    *,
    artifact_h5: Path,
    viewer_cmd_text: str,
    action: str,
    target_metric: str,
    recompute_existing: bool,
) -> dict[str, Any]:
    effective_python_argv = [str(sys.executable), *[str(item) for item in sys.argv]]
    wrapper_command = os.environ.get("PYCHMP_WRAPPER_COMMAND", "").strip()
    compatibility_signature = _build_command_compatibility_signature(effective_python_argv)
    log_path = Path(f"{artifact_h5}.log")
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "action": str(action),
        "artifact_path": str(artifact_h5),
        "cwd": os.getcwd(),
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

try:
    from fit_q0_obs_map import (
        DEFAULT_A,
        DEFAULT_B,
        DEFAULT_NBASE,
        DEFAULT_TBASE,
        PSFConvolvedRenderer,
        _elliptical_gaussian_kernel,
        _build_target_header,
        _colorize,
        _effective_psf_parameters,
        _extract_psf_from_header,
        _format_psf_report,
        _format_q0_value,
        _compute_file_sha256,
        _lookup_cached_render_pair,
        _load_model_identity,
        _load_model_observer_metadata,
        _load_saved_fov_from_model,
        _make_trial_progress_reporter,
        _regrid_full_disk_to_target,
        _resolve_observer_overrides,
        _run_stage,
        save_prepared_observation_bundle,
        save_q0_artifact,
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
        _elliptical_gaussian_kernel,
        _build_target_header,
        _colorize,
        _effective_psf_parameters,
        _extract_psf_from_header,
        _format_psf_report,
        _format_q0_value,
        _compute_file_sha256,
        _lookup_cached_render_pair,
        _load_model_identity,
        _load_model_observer_metadata,
        _load_saved_fov_from_model,
        _make_trial_progress_reporter,
        _regrid_full_disk_to_target,
        _resolve_observer_overrides,
        _run_stage,
        save_prepared_observation_bundle,
        save_q0_artifact,
        _with_observer_wcs_keywords,
        load_eovsa_map,
    )

try:
    from plot_ab_scan_artifacts import plot_ab_scan_file
except ModuleNotFoundError:
    from examples.plot_ab_scan_artifacts import plot_ab_scan_file


METRIC_CHOICES = ("chi2", "rho2", "eta2")


@dataclass(frozen=True)
class GridPointSpec:
    a: float
    b: float
    q0_min: float | None = None
    q0_max: float | None = None


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

    def isatty(self) -> bool:
        for stream in self._streams:
            try:
                if bool(stream.isatty()):
                    return True
            except Exception:
                continue
        return False

    @property
    def encoding(self) -> str:
        for stream in self._streams:
            enc = getattr(stream, "encoding", None)
            if isinstance(enc, str) and enc:
                return enc
        return "utf-8"


class _SharedContextPointRenderer:
    def __init__(
        self,
        context: GXRenderMWContext,
        *,
        frequency_ghz: float,
        tbase: float,
        nbase: float,
        a: float,
        b: float,
        mode: int = 0,
        selective_heating: bool = False,
        shtable: Any | None = None,
    ) -> None:
        self._context = context
        self._frequency_ghz = float(frequency_ghz)
        self._tbase = float(tbase)
        self._nbase = float(nbase)
        self._a = float(a)
        self._b = float(b)
        self._mode = int(mode)
        self._selective_heating = bool(selective_heating)
        self._shtable = shtable
        self.render_call_count = 0

    def render(self, q0: float) -> np.ndarray:
        self.render_call_count += 1
        return self._context.render(
            frequency_ghz=self._frequency_ghz,
            tbase=self._tbase,
            nbase=self._nbase,
            q0=float(q0),
            a=self._a,
            b=self._b,
            mode=self._mode,
            selective_heating=self._selective_heating,
            shtable=self._shtable,
        )


def _parse_float_list(text: str) -> list[float]:
    values = [item.strip() for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("expected a comma-separated list of floats")
    return [float(v) for v in values]


def _parse_grid_point_token(text: str) -> GridPointSpec:
    parts = [item.strip() for item in str(text).split(":")]
    if len(parts) < 2 or len(parts) > 4:
        raise ValueError(f"invalid grid point '{text}'; expected a:b[:q0_min[:q0_max]]")
    a_text, b_text = parts[0], parts[1]
    if not a_text or not b_text:
        raise ValueError(f"invalid grid point '{text}'; a and b are required")
    q0_min = None
    q0_max = None
    if len(parts) >= 3 and parts[2] != "":
        q0_min = float(parts[2])
    if len(parts) >= 4 and parts[3] != "":
        q0_max = float(parts[3])
    return GridPointSpec(a=float(a_text), b=float(b_text), q0_min=q0_min, q0_max=q0_max)


def _parse_ab_pairs(text: str) -> list[GridPointSpec]:
    parts = [item.strip() for item in str(text).split(",") if item.strip()]
    if not parts:
        raise ValueError("expected a comma-separated list of a:b pairs")
    return [_parse_grid_point_token(item) for item in parts]


def _load_grid_file(path: Path) -> list[GridPointSpec]:
    suffix = str(path.suffix).lower()
    if suffix == ".csv":
        rows: list[GridPointSpec] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader, start=2):
                try:
                    a_text = str(row.get("a", "")).strip()
                    b_text = str(row.get("b", "")).strip()
                    if not a_text or not b_text:
                        raise ValueError("columns 'a' and 'b' are required")
                    q0_min_text = str(row.get("q0_min", "")).strip()
                    q0_max_text = str(row.get("q0_max", "")).strip()
                    rows.append(
                        GridPointSpec(
                            a=float(a_text),
                            b=float(b_text),
                            q0_min=float(q0_min_text) if q0_min_text else None,
                            q0_max=float(q0_max_text) if q0_max_text else None,
                        )
                    )
                except Exception as exc:
                    raise ValueError(f"invalid CSV grid-file row {index}: {exc}") from exc
        return rows
    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"failed to parse JSON grid-file {path}: {exc}") from exc
        if not isinstance(payload, list):
            raise ValueError("JSON grid-file must contain a list of point objects")
        rows = []
        for index, item in enumerate(payload, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"invalid JSON grid-file entry {index}: expected object")
            try:
                rows.append(
                    GridPointSpec(
                        a=float(item["a"]),
                        b=float(item["b"]),
                        q0_min=None if item.get("q0_min") in {None, ""} else float(item["q0_min"]),
                        q0_max=None if item.get("q0_max") in {None, ""} else float(item["q0_max"]),
                    )
                )
            except Exception as exc:
                raise ValueError(f"invalid JSON grid-file entry {index}: {exc}") from exc
        return rows
    raise ValueError(f"unsupported --grid-file format for {path}; expected .csv or .json")


def _merge_sparse_point_specs(*groups: list[GridPointSpec]) -> list[GridPointSpec]:
    merged: dict[tuple[float, float], GridPointSpec] = {}
    for group in groups:
        for spec in group:
            key = (float(spec.a), float(spec.b))
            merged.pop(key, None)
            merged[key] = GridPointSpec(
                a=float(spec.a),
                b=float(spec.b),
                q0_min=None if spec.q0_min is None else float(spec.q0_min),
                q0_max=None if spec.q0_max is None else float(spec.q0_max),
            )
    return list(merged.values())


def _parse_grid_values(
    *,
    values_text: str | None,
    start: float | None,
    stop: float | None,
    step: float | None,
    name: str,
) -> np.ndarray:
    if values_text:
        arr = np.asarray(_parse_float_list(values_text), dtype=float)
    else:
        if start is None or stop is None or step is None:
            raise ValueError(f"provide either --{name}-values or --{name}-start/--{name}-stop/--{name}-step")
        if step == 0:
            raise ValueError(f"--{name}-step must be non-zero")
        count = int(np.floor((float(stop) - float(start)) / float(step))) + 1
        if count <= 0:
            raise ValueError(f"invalid {name} grid specification")
        arr = np.asarray([float(start) + i * float(step) for i in range(count)], dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} grid must be one-dimensional and non-empty")
    return arr


def _decode_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        item = value.item()
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace")
        return str(item)
    return str(value)


def _blank_map(template: np.ndarray) -> np.ndarray:
    return np.full_like(np.asarray(template, dtype=float), np.nan, dtype=float)


def _pending_point_payload(
    *,
    a_value: float,
    b_value: float,
    a_index: int,
    b_index: int,
    observed_template: np.ndarray,
    target_metric: str,
    status: str,
    message: str,
) -> dict[str, Any]:
    blank = _blank_map(observed_template)
    diagnostics = {
        "a": float(a_value),
        "b": float(b_value),
        "target_metric": str(target_metric),
        "optimizer_message": str(message),
        "fit_success": False,
        "point_status": str(status),
    }
    return {
        "a": float(a_value),
        "b": float(b_value),
        "a_index": int(a_index),
        "b_index": int(b_index),
        "q0": np.nan,
        "success": False,
        "status": str(status),
        "modeled_best": blank,
        "raw_modeled_best": blank.copy(),
        "residual": blank.copy(),
        "fit_q0_trials": tuple(),
        "fit_metric_trials": tuple(),
        "fit_chi2_trials": tuple(),
        "fit_rho2_trials": tuple(),
        "fit_eta2_trials": tuple(),
        "target_metric": str(target_metric),
        "diagnostics": diagnostics,
    }


def _match_existing_index(values: np.ndarray, value: float) -> int | None:
    matches = np.where(np.isclose(np.asarray(values, dtype=float), float(value), rtol=0.0, atol=1e-12))[0]
    if matches.size == 0:
        return None
    return int(matches[0])


def _merge_existing_rectangular_payload(
    *,
    existing_payload: dict[str, Any],
    a_values: np.ndarray,
    b_values: np.ndarray,
    point_payloads: dict[tuple[int, int], dict[str, Any]],
    best_q0: np.ndarray,
    objective_values: np.ndarray,
    chi2: np.ndarray,
    rho2: np.ndarray,
    eta2: np.ndarray,
    success: np.ndarray,
    target_metric: str,
) -> tuple[dict[tuple[int, int], dict[str, Any]], int, int]:
    existing_points = int(len(existing_payload.get("points", {})))
    reused_points = 0

    for payload in existing_payload.get("points", {}).values():
        match_i = _match_existing_index(a_values, float(payload["a"]))
        match_j = _match_existing_index(b_values, float(payload["b"]))
        if match_i is None or match_j is None:
            continue
        key = (int(match_i), int(match_j))
        existing_diag = dict(payload["diagnostics"])
        point_payloads[key] = {
            "a": float(payload["a"]),
            "b": float(payload["b"]),
            "a_index": int(match_i),
            "b_index": int(match_j),
            "q0": float(payload["q0"]),
            "success": bool(payload["success"]),
            "status": str(payload.get("status", "computed")),
            "modeled_best": np.asarray(payload["modeled_best"], dtype=float),
            "raw_modeled_best": np.asarray(payload["raw_modeled_best"], dtype=float),
            "residual": np.asarray(payload["residual"], dtype=float),
            "fit_q0_trials": tuple(float(v) for v in payload.get("fit_q0_trials", ())),
            "fit_metric_trials": tuple(float(v) for v in payload.get("fit_metric_trials", ())),
            "fit_chi2_trials": tuple(float(v) for v in payload.get("fit_chi2_trials", ())),
            "fit_rho2_trials": tuple(float(v) for v in payload.get("fit_rho2_trials", ())),
            "fit_eta2_trials": tuple(float(v) for v in payload.get("fit_eta2_trials", ())),
            "target_metric": str(payload.get("target_metric", target_metric)),
            "diagnostics": existing_diag,
        }
        if np.isfinite(float(payload["q0"])):
            best_q0[key] = float(payload["q0"])
        objective_values[key] = float(existing_diag.get("target_metric_value", np.nan))
        chi2[key] = float(existing_diag.get("chi2", np.nan))
        rho2[key] = float(existing_diag.get("rho2", np.nan))
        eta2[key] = float(existing_diag.get("eta2", np.nan))
        success[key] = bool(payload["success"])
        reused_points += 1

    return point_payloads, int(existing_points), int(reused_points)


def _build_rectangular_pending_requests(
    *,
    target_tasks: list[Any],
    point_payloads: dict[tuple[int, int], dict[str, Any]],
    recompute_existing: bool,
    q0_start_scalar: float | None,
    use_idl_q0_start_heuristic: bool,
    hard_q0_min: float | None,
    hard_q0_max: float | None,
    target_metric: str,
    adaptive_bracketing: bool,
    q0_step: float,
    max_bracket_steps: int,
) -> tuple[list[_RectangularPointEvaluationRequest], list[tuple[float, float]]]:
    pending_requests: list[_RectangularPointEvaluationRequest] = []
    skipped_points: list[tuple[float, float]] = []

    for point_task in target_tasks:
        i = int(point_task.a_index)
        j = int(point_task.b_index)
        a_value = float(point_task.a)
        b_value = float(point_task.b)
        existing_payload = point_payloads[(int(i), int(j))]
        existing_status = str(existing_payload.get("status", "pending"))

        if existing_status != "pending" and existing_status != "failed" and not bool(recompute_existing):
            skipped_points.append((float(a_value), float(b_value)))
            continue

        q0_start = None
        if q0_start_scalar is not None:
            q0_start = float(q0_start_scalar)
        elif use_idl_q0_start_heuristic:
            q0_start = float(idl_q0_start_heuristic(a_value, b_value))

        pending_requests.append(
            _RectangularPointEvaluationRequest(
                task=point_task,
                q0_start=q0_start,
                hard_q0_min=hard_q0_min,
                hard_q0_max=hard_q0_max,
                target_metric=str(target_metric),
                adaptive_bracketing=bool(adaptive_bracketing),
                q0_step=float(q0_step),
                max_bracket_steps=int(max_bracket_steps),
            )
        )

    return pending_requests, skipped_points


def _build_sparse_pending_tasks(
    *,
    target_tasks: list[Any],
    existing_points: dict[tuple[float, float], dict[str, Any]],
    recompute_existing: bool,
) -> tuple[list[Any], list[tuple[float, float]], list[tuple[float, float]]]:
    pending_tasks: list[Any] = []
    skipped_points: list[tuple[float, float]] = []
    recompute_points: list[tuple[float, float]] = []

    for point_task in target_tasks:
        a_value = float(point_task.a)
        b_value = float(point_task.b)
        existing_point = existing_points.get((float(a_value), float(b_value)))
        if existing_point is None:
            pending_tasks.append(point_task)
            continue

        existing_status = str(existing_point.get("status", "computed"))
        if existing_status == "failed" or bool(recompute_existing):
            pending_tasks.append(point_task)
            recompute_points.append((float(a_value), float(b_value)))
            continue

        if existing_status not in {"pending", "missing"}:
            skipped_points.append((float(a_value), float(b_value)))
            continue

        pending_tasks.append(point_task)

    return pending_tasks, skipped_points, recompute_points


def _decode_h5_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        item = value.item()
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace")
        return str(item)
    return str(value)


def _load_fit_q0_artifact_payload(
    h5_path: Path,
    *,
    a_value: float,
    b_value: float,
    fallback_target_metric: str,
) -> dict[str, Any]:
    with h5py.File(h5_path, "r") as f:
        diagnostics = json.loads(_decode_h5_scalar(f["diagnostics_json"][()])) if "diagnostics_json" in f else {}
        metrics_grp = f["metrics"]
        q0_value = float(f.attrs.get("q0_fitted", np.nan))
        target_metric = str(diagnostics.get("target_metric", fallback_target_metric))
        target_metric_value = float(diagnostics.get("target_metric_value", np.nan))
        if not np.isfinite(target_metric_value):
            metric_lookup = {
                "chi2": float(metrics_grp.attrs.get("chi2", np.nan)),
                "rho2": float(metrics_grp.attrs.get("rho2", np.nan)),
                "eta2": float(metrics_grp.attrs.get("eta2", np.nan)),
            }
            target_metric_value = metric_lookup.get(target_metric, np.nan)
        return {
            "a": float(a_value),
            "b": float(b_value),
            "q0": q0_value,
            "success": bool(diagnostics.get("fit_success", False)),
            "status": "computed",
            "modeled_best": np.asarray(f["modeled_best"], dtype=float),
            "raw_modeled_best": np.asarray(f["raw_modeled_best"], dtype=float),
            "residual": np.asarray(f["residual"], dtype=float),
            "fit_q0_trials": tuple(float(v) for v in diagnostics.get("fit_q0_trials", [])),
            "fit_metric_trials": tuple(float(v) for v in diagnostics.get("fit_metric_trials", [])),
            "fit_chi2_trials": tuple(float(v) for v in diagnostics.get("fit_chi2_trials", [])),
            "fit_rho2_trials": tuple(float(v) for v in diagnostics.get("fit_rho2_trials", [])),
            "fit_eta2_trials": tuple(float(v) for v in diagnostics.get("fit_eta2_trials", [])),
            "nfev": int(diagnostics.get("nfev", -1)),
            "nit": int(diagnostics.get("nit", -1)),
            "message": str(diagnostics.get("optimizer_message", "")),
            "used_adaptive_bracketing": bool(diagnostics.get("used_adaptive_bracketing", False)),
            "bracket_found": bool(diagnostics.get("bracket_found", False)),
            "bracket": tuple(float(v) for v in diagnostics.get("bracket", [])) if diagnostics.get("bracket") is not None else None,
            "target_metric": target_metric,
            "diagnostics": {
                **diagnostics,
                "a": float(a_value),
                "b": float(b_value),
                "target_metric": target_metric,
                "target_metric_value": target_metric_value,
                "chi2": float(metrics_grp.attrs.get("chi2", np.nan)),
                "rho2": float(metrics_grp.attrs.get("rho2", np.nan)),
                "eta2": float(metrics_grp.attrs.get("eta2", np.nan)),
                "point_status": "computed",
            },
        }


def _failed_sparse_point_payload(
    *,
    observed_template: np.ndarray,
    a_value: float,
    b_value: float,
    target_metric: str,
    message: str,
    status: str = "failed",
) -> dict[str, Any]:
    return {
        "a": float(a_value),
        "b": float(b_value),
        "q0": np.nan,
        "success": False,
        "status": str(status),
        "modeled_best": _blank_map(observed_template),
        "raw_modeled_best": _blank_map(observed_template),
        "residual": _blank_map(observed_template),
        "fit_q0_trials": tuple(),
        "fit_metric_trials": tuple(),
        "fit_chi2_trials": tuple(),
        "fit_rho2_trials": tuple(),
        "fit_eta2_trials": tuple(),
        "nfev": -1,
        "nit": -1,
        "message": str(message),
        "used_adaptive_bracketing": False,
        "bracket_found": False,
        "bracket": None,
        "target_metric": str(target_metric),
        "diagnostics": {
            "a": float(a_value),
            "b": float(b_value),
            "target_metric": str(target_metric),
            "optimizer_message": str(message),
            "fit_success": False,
            "nfev": -1,
            "nit": -1,
            "used_adaptive_bracketing": False,
            "bracket_found": False,
            "bracket": None,
            "point_status": str(status),
        },
    }


def _run_point_worker(
    cmd: list[str],
    *,
    timeout_s: float | None,
    progress: bool,
    label: str,
) -> None:
    proc = subprocess.Popen(cmd, start_new_session=True)
    if progress:
        print(
            f"    {label} PID/PGID: {proc.pid} "
            f"(stop with: kill -9 -{proc.pid} or pkill -9 -f fit_q0_obs_map.py)",
            flush=True,
        )

    if timeout_s is None or timeout_s <= 0:
        while proc.poll() is None:
            time.sleep(0.25)
        if proc.returncode != 0:
            raise RuntimeError(f"worker exited with status {proc.returncode}")
        return

    deadline = time.monotonic() + float(timeout_s)
    while True:
        rc = proc.poll()
        if rc is not None:
            if rc != 0:
                raise RuntimeError(f"worker exited with status {rc}")
            return
        if time.monotonic() >= deadline:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            kill_deadline = time.monotonic() + 2.0
            while time.monotonic() < kill_deadline:
                rc = proc.poll()
                if rc is not None:
                    raise TimeoutError(
                        f"{label} exceeded timeout of {timeout_s:.0f}s "
                        f"and worker PGID {proc.pid} was terminated"
                    )
                time.sleep(0.1)
            raise RuntimeError(
                f"{label} exceeded timeout of {timeout_s:.0f}s, SIGKILL was sent to worker PGID {proc.pid}, "
                "but the worker did not exit; native code is likely wedged and a reboot may be required"
            )
        time.sleep(0.25)


@dataclass(frozen=True)
class _SparsePointEvaluationRequest:
    task: Any
    fits_file: str
    model_h5: str
    ebtel_path: str
    prepared_observation_h5: str
    point_artifacts_dir: str
    fit_script: str
    point_stem: str
    q0_start: float | None
    hard_q0_min: float | None
    hard_q0_max: float | None
    target_metric: str
    adaptive_bracketing: bool
    q0_step: float
    max_bracket_steps: int
    psf_bmaj_arcsec: float | None
    psf_bmin_arcsec: float | None
    psf_bpa_deg: float | None
    psf_ref_frequency_ghz: float | None
    psf_scale_inverse_frequency: bool
    preflight_render: bool
    preflight_q0: float | None
    preflight_timeout_s: float
    point_timeout_s: float | None
    progress: bool
    spinner: bool
    keep_point_artifacts: bool


@dataclass(frozen=True)
class _SparsePointEvaluationResult:
    task: Any
    point_payload: dict[str, Any] | None
    elapsed_seconds: float
    error_message: str | None = None


def _bootstrap_sparse_worker(_payload: None) -> None:
    return None


def _evaluate_sparse_point_request(
    request: _SparsePointEvaluationRequest,
    _worker_state: None,
) -> _SparsePointEvaluationResult:
    started = time.perf_counter()
    task = request.task
    point_h5 = Path(request.point_artifacts_dir) / f"{request.point_stem}.h5"

    base_point_cmd = [
        sys.executable,
        str(request.fit_script),
        str(request.fits_file),
        str(request.model_h5),
        "--ebtel-path",
        str(request.ebtel_path),
        "--prepared-observation-h5",
        str(request.prepared_observation_h5),
        "--a",
        str(float(task.a)),
        "--b",
        str(float(task.b)),
        "--q0-min",
        str(float(task.q0_min)),
        "--q0-max",
        str(float(task.q0_max)),
        "--target-metric",
        str(request.target_metric),
        "--q0-step",
        str(float(request.q0_step)),
        "--max-bracket-steps",
        str(int(request.max_bracket_steps)),
        "--artifacts-dir",
        str(request.point_artifacts_dir),
        "--artifacts-stem",
        str(request.point_stem),
        "--no-artifacts-png",
    ]
    if bool(request.adaptive_bracketing):
        base_point_cmd.append("--adaptive-bracketing")
    else:
        base_point_cmd.append("--no-adaptive-bracketing")
    if request.hard_q0_min is not None:
        base_point_cmd.extend(["--hard-q0-min", str(float(request.hard_q0_min))])
    if request.hard_q0_max is not None:
        base_point_cmd.extend(["--hard-q0-max", str(float(request.hard_q0_max))])
    if request.q0_start is not None:
        base_point_cmd.extend(["--q0-start", str(float(request.q0_start))])
    if request.psf_bmaj_arcsec is not None:
        base_point_cmd.extend(["--psf-bmaj-arcsec", str(float(request.psf_bmaj_arcsec))])
    if request.psf_bmin_arcsec is not None:
        base_point_cmd.extend(["--psf-bmin-arcsec", str(float(request.psf_bmin_arcsec))])
    if request.psf_bpa_deg is not None:
        base_point_cmd.extend(["--psf-bpa-deg", str(float(request.psf_bpa_deg))])
    if request.psf_ref_frequency_ghz is not None:
        base_point_cmd.extend(["--psf-ref-frequency-ghz", str(float(request.psf_ref_frequency_ghz))])
    if bool(request.psf_scale_inverse_frequency):
        base_point_cmd.append("--psf-scale-inverse-frequency")
    if not bool(request.progress):
        base_point_cmd.append("--no-progress")
    if not bool(request.spinner):
        base_point_cmd.append("--no-spinner")

    try:
        if bool(request.preflight_render):
            preflight_cmd = list(base_point_cmd)
            preflight_cmd.extend(["--preflight-render-only", "--no-artifacts"])
            if request.preflight_q0 is not None:
                preflight_cmd.extend(["--preflight-q0", str(float(request.preflight_q0))])
            _run_point_worker(
                preflight_cmd,
                timeout_s=float(request.preflight_timeout_s),
                progress=bool(request.progress),
                label="preflight worker",
            )

        _run_point_worker(
            list(base_point_cmd),
            timeout_s=None if request.point_timeout_s is None else float(request.point_timeout_s),
            progress=bool(request.progress),
            label="worker",
        )
        if not point_h5.exists():
            raise FileNotFoundError(f"expected point artifact not found: {point_h5}")
        point_payload = _load_fit_q0_artifact_payload(
            point_h5,
            a_value=float(task.a),
            b_value=float(task.b),
            fallback_target_metric=str(request.target_metric),
        )
        return _SparsePointEvaluationResult(
            task=task,
            point_payload=point_payload,
            elapsed_seconds=time.perf_counter() - started,
        )
    except Exception as exc:
        return _SparsePointEvaluationResult(
            task=task,
            point_payload=None,
            elapsed_seconds=time.perf_counter() - started,
            error_message=str(exc),
        )
    finally:
        if point_h5.exists() and not bool(request.keep_point_artifacts):
            point_h5.unlink(missing_ok=True)


def _load_single_point_artifact(h5_path: Path) -> dict[str, Any]:
    with h5py.File(h5_path, "r") as f:
        diagnostics = json.loads(_decode_scalar(f["diagnostics_json"][()]))
        metrics_grp = f["metrics"]
        return {
            "observed": np.asarray(f["observed"], dtype=float),
            "sigma_map": np.asarray(f["sigma_map"], dtype=float),
            "modeled_best": np.asarray(f["modeled_best"], dtype=float),
            "raw_modeled_best": np.asarray(f["raw_modeled_best"], dtype=float),
            "residual": np.asarray(f["residual"], dtype=float),
            "q0": float(f.attrs["q0_fitted"]),
            "chi2": float(metrics_grp.attrs["chi2"]),
            "rho2": float(metrics_grp.attrs["rho2"]),
            "eta2": float(metrics_grp.attrs["eta2"]),
            "target_metric": str(diagnostics.get("target_metric", "chi2")),
            "objective_value": float(diagnostics.get("target_metric_value", np.nan)),
            "fit_q0_trials": tuple(float(v) for v in diagnostics.get("fit_q0_trials", [])),
            "fit_metric_trials": tuple(float(v) for v in diagnostics.get("fit_metric_trials", [])),
            "diagnostics": diagnostics,
            "success": bool(diagnostics.get("fit_success", False)),
        }


def _save_ab_scan_h5(
    out_h5: Path,
    *,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
    a_values: np.ndarray,
    b_values: np.ndarray,
    best_q0: np.ndarray,
    objective_values: np.ndarray,
    chi2: np.ndarray,
    rho2: np.ndarray,
    eta2: np.ndarray,
    success: np.ndarray,
    point_payloads: dict[tuple[int, int], dict[str, Any]],
    run_history: list[dict[str, Any]] | None = None,
) -> None:
    save_rectangular_scan_file(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=wcs_header,
        diagnostics=diagnostics,
        a_values=a_values,
        b_values=b_values,
        best_q0=best_q0,
        objective_values=objective_values,
        chi2=chi2,
        rho2=rho2,
        eta2=eta2,
        success=success,
        point_payloads=point_payloads,
        slice_key=slice_descriptor_from_diagnostics(diagnostics)["key"],
        run_history=run_history,
    )


@dataclass(frozen=True)
class _RectangularWorkerBootstrap:
    model_path: str
    ebtel_path: str
    geometry_data: dict[str, float | int]
    observer_override_data: dict[str, float] | None
    pixel_scale_arcsec: float
    frequency_ghz: float
    tbase: float
    nbase: float
    observed: np.ndarray
    sigma: np.ndarray
    psf_kernel: np.ndarray | None


@dataclass(frozen=True)
class _RectangularWorkerState:
    context: GXRenderMWContext
    frequency_ghz: float
    tbase: float
    nbase: float
    observed: np.ndarray
    sigma: np.ndarray
    psf_kernel: np.ndarray | None


@dataclass(frozen=True)
class _RectangularPointEvaluationRequest:
    task: Any
    q0_start: float | None
    hard_q0_min: float | None
    hard_q0_max: float | None
    target_metric: str
    adaptive_bracketing: bool
    q0_step: float
    max_bracket_steps: int


@dataclass(frozen=True)
class _RectangularPointEvaluationResult:
    task: Any
    success: bool
    q0: float
    objective_value: float
    chi2: float
    rho2: float
    eta2: float
    message: str
    nfev: int
    nit: int
    used_adaptive_bracketing: bool
    bracket_found: bool
    bracket: tuple[float, float, float] | None
    trial_q0: tuple[float, ...]
    trial_objective_values: tuple[float, ...]
    trial_chi2_values: tuple[float, ...]
    trial_rho2_values: tuple[float, ...]
    trial_eta2_values: tuple[float, ...]
    modeled_best: np.ndarray | None
    raw_modeled_best: np.ndarray | None
    residual: np.ndarray | None
    trial_render_count: int
    total_render_calls: int
    elapsed_seconds: float
    error_message: str | None = None


def _geometry_snapshot(geometry: Any) -> dict[str, float | int]:
    return {
        "xc": float(geometry.xc),
        "yc": float(geometry.yc),
        "dx": float(geometry.dx),
        "dy": float(geometry.dy),
        "nx": int(geometry.nx),
        "ny": int(geometry.ny),
    }


def _observer_override_snapshot(observer_overrides: Any | None) -> dict[str, float] | None:
    if observer_overrides is None:
        return None
    return {
        "dsun_cm": float(getattr(observer_overrides, "dsun_cm")),
        "lonc_deg": float(getattr(observer_overrides, "lonc_deg")),
        "b0sun_deg": float(getattr(observer_overrides, "b0sun_deg")),
    }


def _bootstrap_rectangular_worker(payload: _RectangularWorkerBootstrap) -> _RectangularWorkerState:
    sdk = import_module("gxrender.sdk")
    observer = None
    if payload.observer_override_data is not None:
        observer = sdk.ObserverOverrides(
            dsun_cm=float(payload.observer_override_data["dsun_cm"]),
            lonc_deg=float(payload.observer_override_data["lonc_deg"]),
            b0sun_deg=float(payload.observer_override_data["b0sun_deg"]),
        )
    geometry = sdk.MapGeometry(
        xc=float(payload.geometry_data["xc"]),
        yc=float(payload.geometry_data["yc"]),
        dx=float(payload.geometry_data["dx"]),
        dy=float(payload.geometry_data["dy"]),
        nx=int(payload.geometry_data["nx"]),
        ny=int(payload.geometry_data["ny"]),
    )
    context = GXRenderMWContext(
        model_path=str(payload.model_path),
        ebtel_path=str(payload.ebtel_path),
        geometry=geometry,
        observer=observer,
        pixel_scale_arcsec=float(payload.pixel_scale_arcsec),
    )
    return _RectangularWorkerState(
        context=context,
        frequency_ghz=float(payload.frequency_ghz),
        tbase=float(payload.tbase),
        nbase=float(payload.nbase),
        observed=np.asarray(payload.observed, dtype=float),
        sigma=np.asarray(payload.sigma, dtype=float),
        psf_kernel=None if payload.psf_kernel is None else np.asarray(payload.psf_kernel, dtype=float),
    )


def _evaluate_rectangular_point_request(
    request: _RectangularPointEvaluationRequest,
    worker_state: _RectangularWorkerState,
) -> _RectangularPointEvaluationResult:
    started = time.perf_counter()
    task = request.task
    try:
        base_renderer = _SharedContextPointRenderer(
            worker_state.context,
            frequency_ghz=float(worker_state.frequency_ghz),
            tbase=float(worker_state.tbase),
            nbase=float(worker_state.nbase),
            a=float(task.a),
            b=float(task.b),
        )
        renderer = base_renderer if worker_state.psf_kernel is None else PSFConvolvedRenderer(base_renderer, worker_state.psf_kernel)

        render_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

        if isinstance(renderer, PSFConvolvedRenderer):
            class _CachedObservedRenderer:
                def render(self_inner, q0: float) -> np.ndarray:
                    raw_arr, modeled_arr = renderer.render_pair(q0)
                    render_cache[float(q0)] = (raw_arr, modeled_arr)
                    return modeled_arr
        else:
            class _CachedObservedRenderer:
                def render(self_inner, q0: float) -> np.ndarray:
                    modeled_arr = base_renderer.render(q0)
                    render_cache[float(q0)] = (modeled_arr, modeled_arr)
                    return modeled_arr

        result = fit_q0_to_observation(
            renderer=_CachedObservedRenderer(),
            observed=worker_state.observed,
            sigma=worker_state.sigma,
            q0_min=float(task.q0_min),
            q0_max=float(task.q0_max),
            hard_q0_min=request.hard_q0_min,
            hard_q0_max=request.hard_q0_max,
            target_metric=str(request.target_metric),
            adaptive_bracketing=bool(request.adaptive_bracketing),
            q0_start=request.q0_start,
            q0_step=float(request.q0_step),
            max_bracket_steps=int(request.max_bracket_steps),
        )

        cached_best_pair = _lookup_cached_render_pair(render_cache, result.q0)
        if cached_best_pair is not None:
            raw_modeled_best, modeled_best = cached_best_pair
        elif isinstance(renderer, PSFConvolvedRenderer):
            raw_modeled_best, modeled_best = renderer.render_pair(result.q0)
        else:
            modeled_best = base_renderer.render(result.q0)
            raw_modeled_best = modeled_best

        residual = modeled_best - worker_state.observed
        return _RectangularPointEvaluationResult(
            task=task,
            success=bool(result.success),
            q0=float(result.q0),
            objective_value=float(result.objective_value),
            chi2=float(result.metrics.chi2),
            rho2=float(result.metrics.rho2),
            eta2=float(result.metrics.eta2),
            message=str(result.message),
            nfev=int(result.nfev),
            nit=int(result.nit),
            used_adaptive_bracketing=bool(result.used_adaptive_bracketing),
            bracket_found=bool(result.bracket_found),
            bracket=result.bracket,
            trial_q0=tuple(float(v) for v in result.trial_q0),
            trial_objective_values=tuple(float(v) for v in result.trial_objective_values),
            trial_chi2_values=tuple(float(v) for v in result.trial_chi2_values),
            trial_rho2_values=tuple(float(v) for v in result.trial_rho2_values),
            trial_eta2_values=tuple(float(v) for v in result.trial_eta2_values),
            modeled_best=np.asarray(modeled_best, dtype=float),
            raw_modeled_best=np.asarray(raw_modeled_best, dtype=float),
            residual=np.asarray(residual, dtype=float),
            trial_render_count=len(result.trial_q0),
            total_render_calls=int(base_renderer.render_call_count),
            elapsed_seconds=float(time.perf_counter() - started),
        )
    except Exception as exc:
        return _RectangularPointEvaluationResult(
            task=task,
            success=False,
            q0=np.nan,
            objective_value=np.nan,
            chi2=np.nan,
            rho2=np.nan,
            eta2=np.nan,
            message=str(exc),
            nfev=-1,
            nit=-1,
            used_adaptive_bracketing=False,
            bracket_found=False,
            bracket=None,
            trial_q0=tuple(),
            trial_objective_values=tuple(),
            trial_chi2_values=tuple(),
            trial_rho2_values=tuple(),
            trial_eta2_values=tuple(),
            modeled_best=None,
            raw_modeled_best=None,
            residual=None,
            trial_render_count=0,
            total_render_calls=0,
            elapsed_seconds=float(time.perf_counter() - started),
            error_message=str(exc),
        )


def parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    p = argparse.ArgumentParser(
        description="Scan a rectangular `(a,b)` grid against a real observational map.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("fits_file", type=Path, nargs="?", help="Path to EOVSA FITS file")
    p.add_argument("model_h5", type=Path, nargs="?", help="Path to model H5 file")
    p.add_argument("--ebtel-path", type=Path, default=None, help="Path to EBTEL .sav file")

    p.add_argument("--a-values", default=None, help="Comma-separated list of a values to scan.")
    p.add_argument("--b-values", default=None, help="Comma-separated list of b values to scan.")
    p.add_argument("--ab-pairs", default=None, help="Comma-separated explicit point list in a:b form, e.g. 0.0:2.1,0.3:2.4")
    p.add_argument(
        "--grid-point",
        action="append",
        default=[],
        help="Sparse point in a:b[:q0_min[:q0_max]] form. May be repeated.",
    )
    p.add_argument(
        "--grid-file",
        action="append",
        type=Path,
        default=[],
        help="CSV or JSON file containing sparse grid points with columns/keys a,b,q0_min,q0_max. May be repeated.",
    )
    p.add_argument("--a-start", type=float, default=None, help="Grid start for a when --a-values is omitted.")
    p.add_argument("--a-stop", type=float, default=None, help="Grid stop for a when --a-values is omitted.")
    p.add_argument("--a-step", type=float, default=None, help="Grid step for a when --a-values is omitted.")
    p.add_argument("--b-start", type=float, default=None, help="Grid start for b when --b-values is omitted.")
    p.add_argument("--b-stop", type=float, default=None, help="Grid stop for b when --b-values is omitted.")
    p.add_argument("--b-step", type=float, default=None, help="Grid step for b when --b-values is omitted.")

    p.add_argument("--q0-min", type=float, default=0.01, help="Lower edge of the initial Q0 search interval.")
    p.add_argument("--q0-max", type=float, default=2.5, help="Upper edge of the initial Q0 search interval.")
    p.add_argument("--hard-q0-min", type=float, default=None, help="Optional hard lower Q0 boundary.")
    p.add_argument("--hard-q0-max", type=float, default=None, help="Optional hard upper Q0 boundary.")
    p.add_argument("--target-metric", choices=METRIC_CHOICES, default="chi2", help="Metric minimized for each `(a,b)` point.")
    p.add_argument("--adaptive-bracketing", action=argparse.BooleanOptionalAction, default=True, help="Enable adaptive Q0 bracketing for each `(a,b)` point.")
    p.add_argument("--q0-start-scalar", type=float, default=None, help="Fixed Q0 start used for every `(a,b)` point.")
    p.add_argument("--use-idl-q0-start-heuristic", action=argparse.BooleanOptionalAction, default=False, help="Use the IDL empirical Q0_start(a,b) heuristic if no scalar start is given.")
    p.add_argument("--q0-step", type=float, default=1.61803398875, help="Multiplicative Q0 step for adaptive bracketing.")
    p.add_argument("--max-bracket-steps", type=int, default=12, help="Maximum adaptive bracketing expansion steps.")

    p.add_argument("--tbase", type=float, default=None, help="Override base temperature (K).")
    p.add_argument("--nbase", type=float, default=None, help="Override base density (cm^-3).")
    p.add_argument("--observer", default=None, help="Observer name (earth, stereo-a, stereo-b).")
    p.add_argument("--dsun-cm", type=float, default=None, help="Observer-Sun distance override in cm.")
    p.add_argument("--lonc-deg", type=float, default=None, help="Observer heliographic Carrington longitude override in deg.")
    p.add_argument("--b0sun-deg", type=float, default=None, help="Observer heliographic latitude override in deg.")
    p.add_argument("--xc", type=float, default=None, help="Map center X in arcsec (exact override).")
    p.add_argument("--yc", type=float, default=None, help="Map center Y in arcsec (exact override).")
    p.add_argument("--dx", type=float, default=None, help="Map pixel scale X in arcsec/pixel.")
    p.add_argument("--dy", type=float, default=None, help="Map pixel scale Y in arcsec/pixel.")
    p.add_argument("--nx", type=int, default=None, help="Map width in pixels.")
    p.add_argument("--ny", type=int, default=None, help="Map height in pixels.")
    p.add_argument("--pixel-scale-arcsec", type=float, default=2.0, help="Pixel scale used only when no explicit geometry overrides are provided.")

    p.add_argument("--psf-bmaj-arcsec", type=float, default=None, help="PSF major axis FWHM.")
    p.add_argument("--psf-bmin-arcsec", type=float, default=None, help="PSF minor axis FWHM.")
    p.add_argument("--psf-bpa-deg", type=float, default=None, help="PSF position angle in degrees.")
    p.add_argument("--fallback-psf-bmaj-arcsec", type=float, default=None, help="Fallback PSF major axis FWHM used only when the FITS header has no beam and no explicit PSF override is supplied.")
    p.add_argument("--fallback-psf-bmin-arcsec", type=float, default=None, help="Fallback PSF minor axis FWHM used only when the FITS header has no beam and no explicit PSF override is supplied.")
    p.add_argument("--fallback-psf-bpa-deg", type=float, default=None, help="Fallback PSF position angle used only when the FITS header has no beam and no explicit PSF override is supplied.")
    p.add_argument("--psf-ref-frequency-ghz", type=float, default=None, help="Reference frequency for PSF axes values.")
    p.add_argument("--psf-scale-inverse-frequency", action="store_true", help="Scale PSF axes by (ref_freq / active_freq).")

    p.add_argument("--artifacts-dir", type=Path, default=None, help="Directory to write the consolidated H5 and optional PNGs.")
    p.add_argument("--artifacts-stem", default=None, help="Base filename stem (no extension) for outputs.")
    p.add_argument("--artifact-h5", type=Path, default=None, help="Explicit consolidated artifact path to create/update.")
    p.add_argument(
        "--recompute-existing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recompute matching existing records instead of skipping them. Failed points are always retried by default.",
    )
    p.add_argument("--point-timeout-s", type=float, default=900.0, help="Per-point worker timeout in seconds. Enforced for sparse explicit-point mode.")
    p.add_argument("--preflight-render", action=argparse.BooleanOptionalAction, default=True, help="Run an isolated single-render preflight before launching a full sparse explicit-point fit.")
    p.add_argument("--preflight-timeout-s", type=float, default=45.0, help="Timeout in seconds for the isolated preflight render worker.")
    p.add_argument("--preflight-q0", type=float, default=None, help="Optional q0 used for sparse preflight renders. Defaults to q0_start or sqrt(q0_min*q0_max).")
    p.add_argument("--keep-point-artifacts", action="store_true", help="Keep per-point fit_q0_obs_map H5 artifacts instead of deleting them after consolidation.")
    p.add_argument("--selected-a-index", type=int, default=None, help="Optional A-grid index used for the selected-point diagnostic plot. If omitted, defaults to the best point for the target metric.")
    p.add_argument("--selected-b-index", type=int, default=None, help="Optional B-grid index used for the selected-point diagnostic plot. If omitted, defaults to the best point for the target metric.")
    p.add_argument("--no-grid-png", action="store_true", help="Skip the grid-summary PNG.")
    p.add_argument("--no-point-png", action="store_true", help="Skip the selected-point PNG.")
    p.add_argument("--show-plot", action="store_true", help="Display generated plots interactively.")
    p.add_argument("--no-viewer", action="store_true", help="Do not launch pychmp-view automatically when the scan starts.")
    p.add_argument(
        "--execution-policy",
        choices=("auto", "serial", "process-pool"),
        default="serial",
        help=(
            "Per-point execution policy for a single-slice scan: 'serial' keeps one worker, "
            "'process-pool' uses the requested process pool, and 'auto' stays serial for very small scans "
            "but otherwise chooses a conservative worker count from task count and system limits."
        ),
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of worker processes used when the resolved execution policy is process-pool, including auto mode.",
    )
    p.add_argument("--worker-chunksize", type=int, default=1, help="Task chunksize passed to the process pool executor.")
    p.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True, help="Print per-trial and per-point progress diagnostics.")
    p.add_argument("--spinner", action=argparse.BooleanOptionalAction, default=True, help="Show a spinner during long-running stages.")
    p.add_argument("--defaults", action="store_true", help="Print assumed defaults and exit.")
    return p, p.parse_args()


def main() -> int:
    import sys

    parser, args = parse_args()
    suppress_auto_viewer = os.environ.get("PYCHMP_NO_AUTO_VIEWER", "").strip().lower() in {"1", "true", "yes", "on"}
    scan_start = time.perf_counter()
    if args.defaults:
        defaults = {
            "fits_file": "/path/to/eovsa_map.fits",
            "model_h5": "/path/to/model.h5",
            "ebtel_path": "/path/to/ebtel.sav",
            "a_values": "0.0,0.3,0.6",
            "b_values": "2.1,2.4,2.7",
            "ab_pairs": None,
            "grid_point": [],
            "grid_file": [],
            "q0_min": 0.01,
            "q0_max": 2.5,
            "hard_q0_min": None,
            "hard_q0_max": None,
            "target_metric": "chi2",
            "adaptive_bracketing": True,
            "q0_start_scalar": None,
            "use_idl_q0_start_heuristic": False,
            "q0_step": 1.61803398875,
            "max_bracket_steps": 12,
            "execution_policy": "serial",
            "max_workers": None,
            "worker_chunksize": 1,
            "tbase": DEFAULT_TBASE,
            "nbase": DEFAULT_NBASE,
            "pixel_scale_arcsec": 2.0,
            "artifacts_dir": None,
            "artifacts_stem": None,
            "artifact_h5": None,
            "recompute_existing": False,
            "point_timeout_s": 900.0,
            "preflight_render": True,
            "preflight_timeout_s": 45.0,
            "preflight_q0": None,
            "keep_point_artifacts": False,
            "selected_a_index": None,
            "selected_b_index": None,
            "no_grid_png": False,
            "no_point_png": False,
            "show_plot": False,
            "progress": True,
            "spinner": True,
        }
        print("Assumed defaults for scan_ab_obs_map.py:")
        for key, value in defaults.items():
            print(f"  {key}: {value}")
        return 0

    if args.fits_file is None or args.model_h5 is None or args.ebtel_path is None:
        parser.error("fits_file, model_h5, and --ebtel-path are required unless --defaults is used")

    file_specs: list[GridPointSpec] = []
    for grid_file in args.grid_file:
        if not Path(grid_file).exists():
            parser.error(f"--grid-file does not exist: {grid_file}")
        file_specs.extend(_load_grid_file(Path(grid_file)))
    cli_specs = [_parse_grid_point_token(text) for text in args.grid_point]
    legacy_specs = _parse_ab_pairs(args.ab_pairs) if args.ab_pairs else []
    explicit_points = _merge_sparse_point_specs(file_specs, cli_specs, legacy_specs)
    if explicit_points:
        a_values = np.asarray(sorted({float(point.a) for point in explicit_points}), dtype=float)
        b_values = np.asarray(sorted({float(point.b) for point in explicit_points}), dtype=float)
    else:
        a_values = _parse_grid_values(
            values_text=args.a_values,
            start=args.a_start,
            stop=args.a_stop,
            step=args.a_step,
            name="a",
        )
        b_values = _parse_grid_values(
            values_text=args.b_values,
            start=args.b_start,
            stop=args.b_stop,
            step=args.b_step,
            name="b",
        )

    if not args.fits_file.exists() or not args.model_h5.exists() or not args.ebtel_path.exists():
        parser.error("fits_file, model_h5, and --ebtel-path must all exist")

    print(f"\n{'=' * 70}")
    print("SCANNING (a, b) GRID AGAINST OBSERVATIONAL MAP")
    print(f"{'=' * 70}\n")
    print(f"Process PID: {os.getpid()}")
    print(f"If the process seems inactive, you can stop it from another terminal with: kill -9 {os.getpid()}\n")

    print(f"Loading FITS file: {args.fits_file.name}")
    observed, header, freq_ghz = load_eovsa_map(args.fits_file)
    print(f"  Shape: {observed.shape}")
    print(f"  Frequency: {freq_ghz:.3f} GHz")
    print(f"  Data range: [{observed.min():.2f}, {observed.max():.2f}]")

    print("\nEstimating noise from map...")
    noise_result = estimate_map_noise(observed, method="histogram_clip")
    if noise_result is None:
        sigma_map = np.full_like(observed, observed.std())
        noise_diag = None
        print(f"  Noise estimate unavailable; using sigma={float(observed.std()):.2f} K")
    else:
        sigma_map = noise_result.sigma_map
        noise_diag = noise_result.diagnostics
        print(f"  Estimated sigma: {noise_result.sigma:.2f} K")
        print(f"  Background fraction: {noise_result.mask_fraction:.1%}")

    sdk = import_module("gxrender.sdk")

    tbase = float(args.tbase) if args.tbase is not None else DEFAULT_TBASE
    nbase = float(args.nbase) if args.nbase is not None else DEFAULT_NBASE

    header_psf, header_psf_source = _extract_psf_from_header(header)
    psf_bmaj_arcsec = float(args.psf_bmaj_arcsec) if args.psf_bmaj_arcsec is not None else None
    psf_bmin_arcsec = float(args.psf_bmin_arcsec) if args.psf_bmin_arcsec is not None else None
    psf_bpa_deg = float(args.psf_bpa_deg) if args.psf_bpa_deg is not None else None
    fallback_psf_bmaj_arcsec = float(args.fallback_psf_bmaj_arcsec) if args.fallback_psf_bmaj_arcsec is not None else None
    fallback_psf_bmin_arcsec = float(args.fallback_psf_bmin_arcsec) if args.fallback_psf_bmin_arcsec is not None else None
    fallback_psf_bpa_deg = float(args.fallback_psf_bpa_deg) if args.fallback_psf_bpa_deg is not None else None
    has_cli_psf_override = any(value is not None for value in (args.psf_bmaj_arcsec, args.psf_bmin_arcsec, args.psf_bpa_deg))
    has_cli_psf_fallback = any(
        value is not None for value in (args.fallback_psf_bmaj_arcsec, args.fallback_psf_bmin_arcsec, args.fallback_psf_bpa_deg)
    )
    if header_psf is not None and not has_cli_psf_override:
        psf_bmaj_arcsec = float(header_psf["psf_bmaj_arcsec"])
        psf_bmin_arcsec = float(header_psf["psf_bmin_arcsec"])
        psf_bpa_deg = float(header_psf["psf_bpa_deg"])
        psf_source = header_psf_source
    elif has_cli_psf_override:
        psf_source = "cli_override"
    elif has_cli_psf_fallback:
        psf_bmaj_arcsec = fallback_psf_bmaj_arcsec
        psf_bmin_arcsec = fallback_psf_bmin_arcsec
        psf_bpa_deg = fallback_psf_bpa_deg
        psf_source = "cli_fallback"
    else:
        psf_source = "none"

    observer_overrides, observer_source = _resolve_observer_overrides(
        sdk,
        model_path=args.model_h5,
        observer_name=args.observer,
        dsun_cm=args.dsun_cm,
        lonc_deg=args.lonc_deg,
        b0sun_deg=args.b0sun_deg,
    )
    model_observer_meta = _load_model_observer_metadata(args.model_h5)

    geometry_overrides_requested = any(v is not None for v in (args.xc, args.yc, args.dx, args.dy, args.nx, args.ny))
    saved_fov = None
    if geometry_overrides_requested:
        geometry = sdk.MapGeometry(xc=args.xc, yc=args.yc, dx=args.dx, dy=args.dy, nx=args.nx, ny=args.ny)
        geometry_mode = "explicit"
    else:
        saved_fov = _load_saved_fov_from_model(args.model_h5)
        if saved_fov is None:
            parser.error("model does not expose a saved FOV; provide explicit geometry overrides")
        dx_eff = float(args.pixel_scale_arcsec)
        dy_eff = float(args.pixel_scale_arcsec)
        nx_eff = max(16, int(round(float(saved_fov["xsize_arcsec"]) / abs(dx_eff))))
        ny_eff = max(16, int(round(float(saved_fov["ysize_arcsec"]) / abs(dy_eff))))
        geometry = sdk.MapGeometry(
            xc=float(saved_fov["xc_arcsec"]),
            yc=float(saved_fov["yc_arcsec"]),
            dx=dx_eff,
            dy=dy_eff,
            nx=nx_eff,
            ny=ny_eff,
        )
        geometry_mode = "saved_fov"

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

    print("\nPreparing model-aligned observational submap...")
    print(f"  Observer mode: {'saved metadata' if observer_overrides is None else 'overrides'} ({observer_source})")
    print(f"  Geometry mode: {geometry_mode} xc={float(geometry.xc):.3f} yc={float(geometry.yc):.3f} dx={float(geometry.dx):.3f} dy={float(geometry.dy):.3f} nx={int(geometry.nx)} ny={int(geometry.ny)}")
    print(f"  Observed submap grid: Ny={observed_cropped.shape[0]} Nx={observed_cropped.shape[1]}")
    print(f"  Model render grid: Ny={int(geometry.ny)} Nx={int(geometry.nx)}")
    print(f"  Pixel scale: dx={float(geometry.dx):.1f} dy={float(geometry.dy):.1f} arcsec/pixel")
    print(f"  FOV: {abs(float(geometry.dx)) * int(geometry.nx):.0f} x {abs(float(geometry.dy)) * int(geometry.ny):.0f} arcsec")
    print(f"  A-grid: {', '.join(f'{float(v):.3f}' for v in a_values)}")
    print(f"  B-grid: {', '.join(f'{float(v):.3f}' for v in b_values)}")
    print(
        "  "
        + _format_psf_report(
            source=str(psf_source),
            bmaj_arcsec=psf_bmaj_arcsec,
            bmin_arcsec=psf_bmin_arcsec,
            bpa_deg=psf_bpa_deg,
            active_frequency_ghz=float(freq_ghz),
            ref_frequency_ghz=float(args.psf_ref_frequency_ghz) if args.psf_ref_frequency_ghz is not None else None,
            scale_inverse_frequency=bool(args.psf_scale_inverse_frequency),
        )
    )
    print(f"  Plasma baseline: tbase={tbase:.3e} nbase={nbase:.3e}")

    artifacts_dir = args.artifacts_dir or (Path(".").resolve() / "ab_scan_artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stem = args.artifacts_stem or f"{args.fits_file.stem}_ab_scan_{args.target_metric}"
    out_h5 = args.artifact_h5 if args.artifact_h5 is not None else artifacts_dir / f"{stem}.h5"
    out_h5 = Path(out_h5)
    artifacts_dir = out_h5.parent
    out_log = Path(f"{out_h5}.log")
    current_slice_key = str(
        slice_descriptor_from_diagnostics(
            {
                "spectral_domain": "mw",
                "spectral_label": f"{float(freq_ghz):.3f} GHz",
                "frequency_ghz": float(freq_ghz),
            }
        )["key"]
    )
    current_slice_descriptor = ABSliceTaskDescriptor(
        key=current_slice_key,
        domain="mw",
        label=f"{float(freq_ghz):.3f} GHz",
        display_label=f"MW: {float(freq_ghz):.3f} GHz",
    )

    live_log_handle = None
    try:
        out_log.parent.mkdir(parents=True, exist_ok=True)
        live_log_handle = out_log.open("a", encoding="utf-8", buffering=1)
        sys.stdout = _TeeStream(sys.stdout, live_log_handle)
        sys.stderr = _TeeStream(sys.stderr, live_log_handle)
        print(f"Live log file: {out_log}")
    except Exception as exc:
        print(_colorize(f"WARNING: failed to initialize live log sidecar: {exc}", "yellow"))

    existing_format = detect_scan_artifact_format(out_h5, slice_key=current_slice_key) if out_h5.exists() else None
    use_sparse_mode = bool(explicit_points or existing_format == "sparse")
    target_points = explicit_points or [GridPointSpec(a=float(a), b=float(b)) for a in a_values for b in b_values]
    if use_sparse_mode:
        target_tasks = compile_sparse_point_tasks(
            point_specs=[
                (float(point.a), float(point.b), point.q0_min, point.q0_max)
                for point in target_points
            ],
            a_values=a_values,
            b_values=b_values,
            slice_descriptor=current_slice_descriptor,
            default_q0_min=float(args.q0_min),
            default_q0_max=float(args.q0_max),
            target_metric=str(args.target_metric),
        )
    else:
        target_tasks = compile_rectangular_point_tasks(
            a_values=a_values,
            b_values=b_values,
            slice_descriptor=current_slice_descriptor,
            q0_min=float(args.q0_min),
            q0_max=float(args.q0_max),
            target_metric=str(args.target_metric),
        )
    point_artifacts_dir = artifacts_dir / f"{stem}_point_runs" if (args.keep_point_artifacts or use_sparse_mode) else None
    if point_artifacts_dir is not None:
        point_artifacts_dir.mkdir(parents=True, exist_ok=True)
    if args.progress:
        print("  Shared observational preprocessing: in-memory reuse across all points")
    if (not use_sparse_mode) and args.point_timeout_s is not None and float(args.point_timeout_s) > 0:
        print(
            _colorize(
                "  Note: --point-timeout-s is not enforced in the current in-process scan mode.",
                "yellow",
            )
        )
    viewer_script = Path(__file__).with_name("pychmp_view.py")
    viewer_cmd = [sys.executable, str(viewer_script), str(out_h5)]
    viewer_cmd_text = shlex.join(viewer_cmd)
    artifact_preexisting = out_h5.exists()
    if artifact_preexisting:
        try:
            existing_run_history = load_run_history(out_h5, slice_key=current_slice_key)
        except KeyError:
            existing_run_history = []
    else:
        existing_run_history = []
    current_run_entry = _build_run_history_entry(
        artifact_h5=out_h5,
        viewer_cmd_text=viewer_cmd_text,
        action=(
            "recompute"
            if bool(args.recompute_existing) and artifact_preexisting
            else ("resume" if artifact_preexisting and not bool(args.recompute_existing) else "create")
        ),
        target_metric=str(args.target_metric),
        recompute_existing=bool(args.recompute_existing),
    )
    current_run_history = [*existing_run_history, current_run_entry]
    viewer_process = None
    viewer_refresh_signal = Path(f"{out_h5}.refresh")

    def _notify_viewer_refresh(phase: str) -> None:
        try:
            viewer_refresh_signal.write_text(f"{time.time():.6f} {phase}\n", encoding="utf-8")
        except Exception:
            pass

    def _maybe_launch_viewer(phase: str) -> None:
        nonlocal viewer_process
        if bool(args.no_viewer) or suppress_auto_viewer:
            return
        if viewer_process is not None and viewer_process.poll() is None:
            return
        try:
            proc = subprocess.Popen(
                viewer_cmd,
                start_new_session=True,
            )
            if proc.poll() is not None:
                print(
                    _colorize(
                        f"WARNING: pychmp-view exited immediately after auto-launch ({phase}). "
                        f"Try running manually: {viewer_cmd_text}",
                        "yellow",
                    )
                )
                return
            viewer_process = proc
            print(f"✓ Launched pychmp-view ({phase}) pid={proc.pid}")
        except Exception as exc:
            print(_colorize(f"WARNING: failed to launch pychmp-view automatically ({phase}): {exc}", "yellow"))

    print(f"  Live viewer command: {viewer_cmd_text}")
    print("  You may run this command while the scan is still in progress and use Refresh Artifact in the viewer.")
    if use_sparse_mode:
        print("  Artifact storage mode: sparse point-record H5")
    else:
        print("  Artifact storage mode: rectangular grid H5")

    psf_kernel = None
    if not use_sparse_mode:
        if psf_bmaj_arcsec is not None and psf_bmin_arcsec is not None and psf_bpa_deg is not None:
            psf_meta = _effective_psf_parameters(
                bmaj_arcsec=psf_bmaj_arcsec,
                bmin_arcsec=psf_bmin_arcsec,
                bpa_deg=psf_bpa_deg,
                active_frequency_ghz=float(freq_ghz),
                ref_frequency_ghz=float(args.psf_ref_frequency_ghz) if args.psf_ref_frequency_ghz is not None else None,
                scale_inverse_frequency=bool(args.psf_scale_inverse_frequency),
            )
            assert psf_meta is not None
            psf_kernel = _elliptical_gaussian_kernel(
                bmaj_arcsec=float(psf_meta["active_bmaj_arcsec"]),
                bmin_arcsec=float(psf_meta["active_bmin_arcsec"]),
                bpa_deg=float(psf_bpa_deg),
                dx_arcsec=float(geometry.dx),
                dy_arcsec=float(geometry.dy),
            )

    execution_plan = resolve_execution_plan(
        task_count=int(len(target_tasks)),
        requested_policy=str(args.execution_policy),
        max_workers=args.max_workers,
    )
    if args.progress:
        print(
            "  Execution plan: "
            f"requested={execution_plan.requested_policy} "
            f"resolved={execution_plan.policy} "
            f"workers={execution_plan.max_workers} "
            f"cpus={execution_plan.available_cpus} "
            f"tasks={execution_plan.task_count}"
        )
        if execution_plan.policy != "serial":
            print("  Note: process-pool mode reports progress per completed point; per-trial progress remains serial-only")

    best_q0 = np.full((a_values.size, b_values.size), np.nan, dtype=float)
    objective_values = np.full((a_values.size, b_values.size), np.nan, dtype=float)
    chi2 = np.full((a_values.size, b_values.size), np.nan, dtype=float)
    rho2 = np.full((a_values.size, b_values.size), np.nan, dtype=float)
    eta2 = np.full((a_values.size, b_values.size), np.nan, dtype=float)
    success = np.zeros((a_values.size, b_values.size), dtype=bool)
    point_payloads: dict[tuple[int, int], dict[str, Any]] = {
        (int(i), int(j)): _pending_point_payload(
            a_value=float(a_value),
            b_value=float(b_value),
            a_index=int(i),
            b_index=int(j),
            observed_template=observed_cropped,
            target_metric=str(args.target_metric),
            status="pending",
            message="point not yet computed",
        )
        for i, a_value in enumerate(a_values)
        for j, b_value in enumerate(b_values)
    }

    model_sha256 = _compute_file_sha256(args.model_h5)
    fits_sha256 = _compute_file_sha256(args.fits_file)
    ebtel_sha256 = _compute_file_sha256(args.ebtel_path)
    root_diag = {
        "artifact_kind": "pychmp_ab_scan",
        "spectral_domain": "mw",
        "spectral_label": f"{float(freq_ghz):.3f} GHz",
        "model_path": str(args.model_h5),
        "model_id": str(_load_model_identity(args.model_h5)),
        "model_sha256": str(model_sha256),
        "fits_file": str(args.fits_file),
        "fits_sha256": str(fits_sha256),
        "ebtel_path": str(args.ebtel_path),
        "ebtel_sha256": str(ebtel_sha256),
        "target_metric": str(args.target_metric),
        "frequency_ghz": float(freq_ghz),
        "map_xc_arcsec": float(geometry.xc),
        "map_yc_arcsec": float(geometry.yc),
        "map_dx_arcsec": float(geometry.dx),
        "map_dy_arcsec": float(geometry.dy),
        "map_nx": int(geometry.nx),
        "map_ny": int(geometry.ny),
        "noise_diagnostics": noise_diag,
        "observer_name": str(model_observer_meta.get("observer_name", args.observer or "earth")),
        "observer_lonc_deg": float(model_observer_meta.get("observer_lonc_deg", 0.0)),
        "observer_b0sun_deg": float(model_observer_meta.get("observer_b0sun_deg", 0.0)),
        "observer_dsun_cm": float(model_observer_meta.get("observer_dsun_cm", 1.495978707e13)),
        "observer_obs_time": target_header.get("DATE-OBS", ""),
        "execution_policy_requested": str(args.execution_policy),
        "execution_policy_resolved": str(execution_plan.policy),
        "execution_max_workers": int(execution_plan.max_workers),
        "execution_available_cpus": int(execution_plan.available_cpus),
        "psf_source": str(psf_source),
        "psf_bmaj_arcsec": None if psf_bmaj_arcsec is None else float(psf_bmaj_arcsec),
        "psf_bmin_arcsec": None if psf_bmin_arcsec is None else float(psf_bmin_arcsec),
        "psf_bpa_deg": None if psf_bpa_deg is None else float(psf_bpa_deg),
        "psf_ref_frequency_ghz": None if args.psf_ref_frequency_ghz is None else float(args.psf_ref_frequency_ghz),
        "psf_scale_inverse_frequency": bool(args.psf_scale_inverse_frequency),
    }
    compatibility_signature = _build_physical_compatibility_signature(
        {
            "artifact_kind": root_diag["artifact_kind"],
            "target_metric": root_diag["target_metric"],
            "model_sha256": root_diag["model_sha256"],
            "fits_sha256": root_diag["fits_sha256"],
            "ebtel_sha256": root_diag["ebtel_sha256"],
            "frequency_ghz": root_diag["frequency_ghz"],
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
            "psf_source": root_diag["psf_source"],
            "psf_bmaj_arcsec": root_diag["psf_bmaj_arcsec"],
            "psf_bmin_arcsec": root_diag["psf_bmin_arcsec"],
            "psf_bpa_deg": root_diag["psf_bpa_deg"],
            "psf_ref_frequency_ghz": root_diag["psf_ref_frequency_ghz"],
            "psf_scale_inverse_frequency": root_diag["psf_scale_inverse_frequency"],
        }
    )
    root_diag[COMPATIBILITY_SIGNATURE_KEY] = compatibility_signature
    current_run_entry["compatibility_signature"] = compatibility_signature

    if use_sparse_mode:
        assert point_artifacts_dir is not None
        if out_h5.exists() and existing_format != "sparse":
            parser.error(
                "Explicit sparse point-list mode requires a sparse artifact at --artifact-h5. "
                "Convert the existing rectangular artifact first."
            )

        if not out_h5.exists():
            write_sparse_scan_file(
                out_h5,
                observed=observed_cropped,
                sigma_map=sigma_cropped,
                wcs_header=target_header,
                diagnostics=root_diag,
                point_records=[],
                run_history=current_run_history,
            )
            _notify_viewer_refresh("sparse artifact initialized")

        if out_h5.exists():
            try:
                existing_payload = load_scan_file(out_h5, slice_key=current_slice_key)
                validate_scan_artifact_compatibility(
                    existing_payload,
                    observed=observed_cropped,
                    sigma_map=sigma_cropped,
                    wcs_header=target_header,
                    diagnostics=root_diag,
                    artifact_path=out_h5,
                )
            except ScanArtifactCompatibilityError as exc:
                parser.error(str(exc))
            except Exception as exc:
                parser.error(f"existing sparse artifact at {out_h5} could not be read safely: {exc}")
        else:
            existing_payload = None
        existing_points = {
            (float(record["a"]), float(record["b"])): record
            for record in (existing_payload.get("point_records", []) if existing_payload is not None else [])
            if point_record_matches_compatibility_signature(
                record,
                compatibility_signature=compatibility_signature,
            )
        }
        _maybe_launch_viewer("scan start")
        prepared_observation_h5 = point_artifacts_dir / f"{stem}_prepared_observation.h5"
        save_prepared_observation_bundle(
            prepared_observation_h5,
            observed_cropped=observed_cropped,
            sigma_cropped=sigma_cropped,
            target_header=target_header,
            frequency_ghz=float(freq_ghz),
            geometry=geometry,
            observer_source=str(observer_source),
            observer_overrides=observer_overrides,
            model_observer_meta=model_observer_meta,
            header_psf=header_psf,
            header_psf_source=str(header_psf_source),
            noise_diagnostics=noise_diag,
        )

        classified_sparse_tasks, skipped_sparse_points, recompute_sparse_points = _build_sparse_pending_tasks(
            target_tasks=target_tasks,
            existing_points=existing_points,
            recompute_existing=bool(args.recompute_existing),
        )
        run_success_count = 0
        for skipped_a, skipped_b in skipped_sparse_points:
            existing_point = existing_points[(float(skipped_a), float(skipped_b))]
            existing_status = str(existing_point.get("status", "computed"))
            print(
                f"    skipping point already present in sparse artifact "
                f"(status={existing_status})"
            )
            run_success_count += int(bool(existing_point.get("success", False)))
        pending_sparse_requests: list[_SparsePointEvaluationRequest] = []
        fit_script = Path(__file__).with_name("fit_q0_obs_map.py")
        for point_task in classified_sparse_tasks:
            a_value = float(point_task.a)
            b_value = float(point_task.b)
            point_q0_min = float(point_task.q0_min)
            point_q0_max = float(point_task.q0_max)
            if not point_q0_min < point_q0_max:
                raise ValueError(
                    f"invalid q0 interval for point a={a_value:.3f}, b={b_value:.3f}: "
                    f"q0_min={point_q0_min} q0_max={point_q0_max}"
                )

            existing_point = existing_points.get((float(a_value), float(b_value)))
            if existing_point is not None and (float(a_value), float(b_value)) in recompute_sparse_points:
                existing_status = str(existing_point.get("status", "computed"))
                if existing_status == "failed":
                    print("    recomputing failed point already present in sparse artifact")
                else:
                    print(
                        f"    recomputing point already present in sparse artifact "
                        f"(status={existing_status})"
                    )

            q0_start = None
            if args.q0_start_scalar is not None:
                q0_start = float(args.q0_start_scalar)
            elif args.use_idl_q0_start_heuristic:
                q0_start = float(idl_q0_start_heuristic(float(a_value), float(b_value)))
            point_stem = f"{stem}_a{int(point_task.a_index):03d}_b{int(point_task.b_index):03d}"
            pending_sparse_requests.append(
                _SparsePointEvaluationRequest(
                    task=point_task,
                    fits_file=str(args.fits_file),
                    model_h5=str(args.model_h5),
                    ebtel_path=str(args.ebtel_path),
                    prepared_observation_h5=str(prepared_observation_h5),
                    point_artifacts_dir=str(point_artifacts_dir),
                    fit_script=str(fit_script),
                    point_stem=point_stem,
                    q0_start=q0_start,
                    hard_q0_min=args.hard_q0_min,
                    hard_q0_max=args.hard_q0_max,
                    target_metric=str(args.target_metric),
                    adaptive_bracketing=bool(args.adaptive_bracketing),
                    q0_step=float(args.q0_step),
                    max_bracket_steps=int(args.max_bracket_steps),
                    psf_bmaj_arcsec=psf_bmaj_arcsec,
                    psf_bmin_arcsec=psf_bmin_arcsec,
                    psf_bpa_deg=psf_bpa_deg,
                    psf_ref_frequency_ghz=args.psf_ref_frequency_ghz,
                    psf_scale_inverse_frequency=bool(args.psf_scale_inverse_frequency),
                    preflight_render=bool(args.preflight_render),
                    preflight_q0=args.preflight_q0,
                    preflight_timeout_s=float(args.preflight_timeout_s),
                    point_timeout_s=None if args.point_timeout_s is None else float(args.point_timeout_s),
                    progress=bool(args.progress),
                    spinner=bool(args.spinner),
                    keep_point_artifacts=bool(args.keep_point_artifacts),
                )
            )

        stage_total = int(len(pending_sparse_requests))
        if stage_total == 0:
            print("\nNo pending sparse/manual points remain after resume/skip checks.")
        execution_settings = ABExecutionSettings(
            policy=execution_plan.policy,
            max_workers=int(execution_plan.max_workers),
            chunksize=int(args.worker_chunksize),
            yield_completion_order=True,
        )
        result_iter = iter_execute_tasks(
            pending_sparse_requests,
            bootstrap_worker=_bootstrap_sparse_worker,
            bootstrap_payload=None,
            evaluate_task=_evaluate_sparse_point_request,
            settings=execution_settings,
        )
        request_lookup = {
            (float(request.task.a), float(request.task.b)): request
            for request in pending_sparse_requests
        }
        for point_counter, result in enumerate(result_iter, start=1):
            point_task = result.task
            a_value = float(point_task.a)
            b_value = float(point_task.b)
            request = request_lookup[(a_value, b_value)]
            q0_start_text = _format_q0_value(request.q0_start) if request.q0_start is not None else "auto"
            point_elapsed = float(result.elapsed_seconds)

            print(f"\nPoint {point_counter}/{stage_total}: a={a_value:.3f} b={b_value:.3f}")
            if args.progress:
                timeout_text = (
                    f", timeout={float(request.point_timeout_s):.0f}s"
                    if request.point_timeout_s is not None and float(request.point_timeout_s) > 0
                    else ""
                )
                preflight_text = ", preflight=on" if bool(request.preflight_render) else ""
                print(
                    f"    completed via {execution_plan.policy} "
                    f"(workers={execution_plan.max_workers}, q0_start={q0_start_text}{timeout_text}{preflight_text})",
                    flush=True,
                )

            if result.error_message is not None or result.point_payload is None:
                print(_colorize(f"  WARNING: point evaluation failed and was skipped: {result.error_message}", "red"))
                point_payload = _failed_sparse_point_payload(
                    observed_template=observed_cropped,
                    a_value=a_value,
                    b_value=b_value,
                    target_metric=str(args.target_metric),
                    message=f"sparse point worker failed: {result.error_message}",
                )
                point_payload["diagnostics"]["psf_source"] = str(psf_source)
            else:
                point_payload = result.point_payload
                run_success_count += int(bool(point_payload["success"]))
            point_payload["diagnostics"][COMPATIBILITY_SIGNATURE_KEY] = compatibility_signature

            append_point_record(
                out_h5,
                observed=observed_cropped,
                sigma_map=sigma_cropped,
                wcs_header=target_header,
                diagnostics=root_diag,
                point_payload=point_payload,
            )
            _notify_viewer_refresh(f"point {point_counter}/{stage_total} saved")
            existing_points[(float(a_value), float(b_value))] = point_payload
            point_elapsed = time.perf_counter() - point_start
            print(
                f"{'=' * 70}\n"
                f"POINT COMPLETE: success={bool(point_payload['success'])} total={point_elapsed:.3f}s\n"
                f"{'=' * 70}\n"
            )
            print(f"Viewer command: {viewer_cmd_text}")
            _maybe_launch_viewer(f"point {point_counter}/{stage_total} complete")

        payload = load_scan_file(out_h5)
        final_a_values = np.asarray(payload["a_values"], dtype=float)
        final_b_values = np.asarray(payload["b_values"], dtype=float)
        metric_arr = np.asarray(payload[str(args.target_metric)], dtype=float)
        has_finite_metric = bool(np.any(np.isfinite(metric_arr)))
        if args.selected_a_index is None or args.selected_b_index is None:
            if has_finite_metric:
                selected_flat = int(np.nanargmin(metric_arr))
                selected_a_index, selected_b_index = np.unravel_index(selected_flat, metric_arr.shape)
                selected_a_index = int(selected_a_index)
                selected_b_index = int(selected_b_index)
            else:
                selected_a_index = 0
                selected_b_index = 0
        else:
            selected_a_index = int(np.clip(int(args.selected_a_index), 0, max(0, final_a_values.size - 1)))
            selected_b_index = int(np.clip(int(args.selected_b_index), 0, max(0, final_b_values.size - 1)))

        grid_png = None if args.no_grid_png else artifacts_dir / f"{stem}_grid.png"
        point_png = None if args.no_point_png else artifacts_dir / f"{stem}_point.png"
        if grid_png is not None or point_png is not None or args.show_plot:
            try:
                plot_ab_scan_file(
                    out_h5,
                    a_index=selected_a_index,
                    b_index=selected_b_index,
                    out_grid_png=grid_png,
                    out_point_png=point_png,
                    show_plot=bool(args.show_plot),
                )
            except ValueError as exc:
                if "No finite values available for metric" not in str(exc):
                    raise
                print(_colorize(f"WARNING: skipping best-point/grid plotting: {exc}", "yellow"))

        print(f"\n✓ Saved consolidated scan file: {out_h5}")
        if grid_png is not None and has_finite_metric:
            print(f"✓ Grid summary PNG: {grid_png}")
        elif grid_png is not None:
            print("⚠ Grid summary PNG skipped because no finite metric values are available yet")
        if point_png is not None and final_a_values.size and final_b_values.size:
            print(
                f"✓ Selected point: a_index={selected_a_index} b_index={selected_b_index} "
                f"a={float(final_a_values[selected_a_index]):.3f} b={float(final_b_values[selected_b_index]):.3f} "
                f"metric[{args.target_metric}]={float(metric_arr[selected_a_index, selected_b_index]):.6e}"
            )
            print(f"✓ Selected-point PNG: {point_png}")
        if not has_finite_metric:
            print(
                _colorize(
                    f"⚠ No finite {args.target_metric} values are present in the artifact yet; "
                    "all attempted points failed or remain pending.",
                    "yellow",
                )
            )
        print(f"Viewer command: {shlex.join(viewer_cmd)}")
        _notify_viewer_refresh("scan complete")
        _maybe_launch_viewer("scan complete")
        total_elapsed = time.perf_counter() - scan_start
        print(f"\n{'=' * 70}")
        print(f"SCAN COMPLETE: success={run_success_count}/{len(target_points)} total={total_elapsed:.3f}s")
        print(f"{'=' * 70}")
        return 0

    if out_h5.exists():
        try:
            existing = load_scan_file(out_h5, slice_key=current_slice_key)
            validate_scan_artifact_compatibility(
                existing,
                observed=observed_cropped,
                sigma_map=sigma_cropped,
                wcs_header=target_header,
                diagnostics=root_diag,
                artifact_path=out_h5,
            )
            point_payloads, existing_points, reused_points = _merge_existing_rectangular_payload(
                existing_payload=existing,
                a_values=a_values,
                b_values=b_values,
                point_payloads=point_payloads,
                best_q0=best_q0,
                objective_values=objective_values,
                chi2=chi2,
                rho2=rho2,
                eta2=eta2,
                success=success,
                target_metric=str(args.target_metric),
            )
            if reused_points:
                print(
                    f"  Resume: loaded {reused_points} overlapping point(s) from existing artifact {out_h5.name} "
                    f"(existing points in file: {existing_points})"
                )
        except KeyError:
            existing = None
        except ScanArtifactCompatibilityError as exc:
            parser.error(str(exc))
        except Exception as exc:
            parser.error(f"existing artifact at {out_h5} could not be read safely: {exc}")

    _save_ab_scan_h5(
        out_h5,
        observed=observed_cropped,
        sigma_map=sigma_cropped,
        wcs_header=target_header,
        diagnostics=root_diag,
        a_values=a_values,
        b_values=b_values,
        best_q0=best_q0,
        objective_values=objective_values,
        chi2=chi2,
        rho2=rho2,
        eta2=eta2,
        success=success,
        point_payloads=point_payloads,
        run_history=current_run_history,
    )
    _notify_viewer_refresh("rectangular artifact initialized")
    _maybe_launch_viewer("scan start")

    pending_requests, skipped_points = _build_rectangular_pending_requests(
        target_tasks=target_tasks,
        point_payloads=point_payloads,
        recompute_existing=bool(args.recompute_existing),
        q0_start_scalar=args.q0_start_scalar,
        use_idl_q0_start_heuristic=bool(args.use_idl_q0_start_heuristic),
        hard_q0_min=args.hard_q0_min,
        hard_q0_max=args.hard_q0_max,
        target_metric=str(args.target_metric),
        adaptive_bracketing=bool(args.adaptive_bracketing),
        q0_step=float(args.q0_step),
        max_bracket_steps=int(args.max_bracket_steps),
    )
    for skipped_a, skipped_b in skipped_points:
        existing_payload = point_payloads[(
            int(_match_existing_index(a_values, float(skipped_a))),
            int(_match_existing_index(b_values, float(skipped_b))),
        )]
        print(
            f"\nSkipping existing point: a={float(skipped_a):.3f} b={float(skipped_b):.3f} "
            f"(status={existing_payload.get('status', 'computed')})"
        )
    if bool(args.recompute_existing):
        for request in pending_requests:
            existing_payload = point_payloads[(int(request.task.a_index), int(request.task.b_index))]
            existing_status = str(existing_payload.get("status", "pending"))
            if existing_status != "pending":
                print(
                    f"\nPoint queued for recompute: a={float(request.task.a):.3f} b={float(request.task.b):.3f} "
                    f"(status={existing_status})"
                )
    else:
        for request in pending_requests:
            existing_status = str(point_payloads[(int(request.task.a_index), int(request.task.b_index))].get("status", "pending"))
            if existing_status == "failed":
                print(
                    f"\nPoint queued for recompute: a={float(request.task.a):.3f} b={float(request.task.b):.3f} (existing failed)"
                )

    stage_total = int(len(pending_requests))
    if stage_total == 0:
        print("\nNo pending rectangular points remain after resume/skip checks.")
    else:
        worker_bootstrap = _RectangularWorkerBootstrap(
            model_path=str(args.model_h5),
            ebtel_path=str(args.ebtel_path),
            geometry_data=_geometry_snapshot(geometry),
            observer_override_data=_observer_override_snapshot(observer_overrides),
            pixel_scale_arcsec=float(args.pixel_scale_arcsec),
            frequency_ghz=float(freq_ghz),
            tbase=float(tbase),
            nbase=float(nbase),
            observed=np.asarray(observed_cropped, dtype=float),
            sigma=np.asarray(sigma_cropped, dtype=float),
            psf_kernel=None if psf_kernel is None else np.asarray(psf_kernel, dtype=float),
        )
        execution_settings = ABExecutionSettings(
            policy=execution_plan.policy,
            max_workers=int(execution_plan.max_workers),
            chunksize=int(args.worker_chunksize),
            yield_completion_order=True,
        )
        result_iter = iter_execute_tasks(
            pending_requests,
            bootstrap_worker=_bootstrap_rectangular_worker,
            bootstrap_payload=worker_bootstrap,
            evaluate_task=_evaluate_rectangular_point_request,
            settings=execution_settings,
        )
        request_lookup = {
            (int(request.task.a_index), int(request.task.b_index)): request
            for request in pending_requests
        }
        for point_counter, response in enumerate(result_iter, start=1):
            point_task = response.task
            i = int(point_task.a_index)
            j = int(point_task.b_index)
            a_value = float(point_task.a)
            b_value = float(point_task.b)
            request = request_lookup[(int(i), int(j))]
            q0_start_text = _format_q0_value(request.q0_start) if request.q0_start is not None else "auto"
            point_stem = f"{stem}_a{i:03d}_b{j:03d}"

            print(f"\nPoint {point_counter}/{stage_total}: a={a_value:.3f} b={b_value:.3f}")
            if args.progress:
                print(
                    f"    completed via {execution_plan.policy} "
                    f"(workers={execution_plan.max_workers}, q0_start={q0_start_text})",
                    flush=True,
                )

            if response.error_message is not None or response.modeled_best is None or response.raw_modeled_best is None or response.residual is None:
                print(_colorize(f"  WARNING: point evaluation failed and was skipped: {response.error_message}", "red"))
                point_diag = {
                    "a": a_value,
                    "b": b_value,
                    "target_metric": str(args.target_metric),
                    "optimizer_message": f"executor point failed: {response.error_message}",
                    "fit_success": False,
                    "psf_source": str(psf_source),
                    "point_status": "failed",
                }
                point_payloads[(int(i), int(j))] = _pending_point_payload(
                    a_value=a_value,
                    b_value=b_value,
                    a_index=int(i),
                    b_index=int(j),
                    observed_template=observed_cropped,
                    target_metric=str(args.target_metric),
                    status="failed",
                    message=f"executor point failed: {response.error_message}",
                )
                point_payloads[(int(i), int(j))]["diagnostics"] = point_diag
            else:
                if response.success:
                    print(_colorize("  ✓ Fitting converged", "green"))
                else:
                    print(_colorize("  ⚠ Fitting stopped without an interior bracket", "yellow"))
                print(f"  Fitted Q0: {response.q0:.6f}")
                print(f"  Objective value ({args.target_metric}): {response.objective_value:.6e}")
                print(
                    f"  Metrics: chi2={response.chi2:.6e}, "
                    f"rho2={response.rho2:.6e}, eta2={response.eta2:.6e}"
                )
                print(f"  Trials: nfev={response.nfev} nit={response.nit} saved_trials={len(response.trial_q0)}")
                print(f"  Optimizer message: {response.message}")
                if response.used_adaptive_bracketing:
                    bracket_text = (
                        f"({_format_q0_value(response.bracket[0])}, {_format_q0_value(response.bracket[1])}, {_format_q0_value(response.bracket[2])})"
                        if response.bracket is not None
                        else "<none>"
                    )
                    print(f"  Adaptive bracketing: found={response.bracket_found} bracket={bracket_text}")

                hard_left = args.hard_q0_min
                hard_right = args.hard_q0_max
                near_left = False
                near_right = False
                if hard_left is not None and hard_right is not None:
                    span = max(abs(hard_right - hard_left), 1.0)
                    near_left = abs(response.q0 - hard_left) <= 0.01 * span
                    near_right = abs(response.q0 - hard_right) <= 0.01 * span
                elif hard_left is not None:
                    near_left = abs(response.q0 - hard_left) <= 0.01 * max(abs(hard_left), 1.0)
                elif hard_right is not None:
                    near_right = abs(response.q0 - hard_right) <= 0.01 * max(abs(hard_right), 1.0)
                if (near_left or near_right) and response.bracket is not None:
                    near_left = near_left and response.q0 <= response.bracket[1]
                    near_right = near_right and response.q0 >= response.bracket[1]
                if near_left or near_right:
                    side = "lower" if near_left else "upper"
                    print(
                        _colorize(
                            f"  WARNING: best q0 lies near the {side} hard search boundary; "
                            "this run did not demonstrate a well-bracketed interior minimum.",
                            "red",
                        )
                    )

                final_render_calls = max(0, int(response.total_render_calls) - int(response.trial_render_count))
                print(
                    f"  Render diagnostics: trial_renders={int(response.trial_render_count)} "
                    f"final_renders={final_render_calls} total_gxrender_calls={int(response.total_render_calls)}"
                )

                point_diag = {
                    "model_path": str(args.model_h5),
                    "observer_name_effective": str(args.observer or "saved_metadata"),
                    "observer_name": str(model_observer_meta.get("observer_name", args.observer or "earth")),
                    "observer_lonc_deg": float(model_observer_meta.get("observer_lonc_deg", 0.0)),
                    "observer_b0sun_deg": float(model_observer_meta.get("observer_b0sun_deg", 0.0)),
                    "observer_dsun_cm": float(model_observer_meta.get("observer_dsun_cm", 1.495978707e13)),
                    "target_metric": str(args.target_metric),
                    "target_metric_value": float(response.objective_value),
                    "chi2": float(response.chi2),
                    "rho2": float(response.rho2),
                    "eta2": float(response.eta2),
                    "q0_recovered": float(response.q0),
                    "fit_success": bool(response.success),
                    "optimizer_message": str(response.message),
                    "nfev": int(response.nfev),
                    "nit": int(response.nit),
                    "used_adaptive_bracketing": bool(response.used_adaptive_bracketing),
                    "bracket_found": bool(response.bracket_found),
                    "bracket": [float(v) for v in response.bracket] if response.bracket is not None else None,
                    "fit_q0_trials": [float(v) for v in response.trial_q0],
                    "fit_metric_trials": [float(v) for v in response.trial_objective_values],
                    "fit_chi2_trials": [float(v) for v in response.trial_chi2_values],
                    "fit_rho2_trials": [float(v) for v in response.trial_rho2_values],
                    "fit_eta2_trials": [float(v) for v in response.trial_eta2_values],
                    "map_xc_arcsec": float(geometry.xc),
                    "map_yc_arcsec": float(geometry.yc),
                    "map_dx_arcsec": float(geometry.dx),
                    "map_dy_arcsec": float(geometry.dy),
                    "active_frequency_ghz": float(freq_ghz),
                    "a": a_value,
                    "b": b_value,
                    "psf_bmaj_arcsec": float(psf_bmaj_arcsec) if psf_bmaj_arcsec is not None else None,
                    "psf_bmin_arcsec": float(psf_bmin_arcsec) if psf_bmin_arcsec is not None else None,
                    "psf_bpa_deg": float(psf_bpa_deg) if psf_bpa_deg is not None else None,
                    "psf_source": str(psf_source),
                    COMPATIBILITY_SIGNATURE_KEY: compatibility_signature,
                    "observer_obs_time": target_header.get("DATE-OBS", ""),
                    "point_status": "computed",
                }

                if point_artifacts_dir is not None:
                    point_h5 = point_artifacts_dir / f"{point_stem}.h5"
                    save_q0_artifact(
                        point_h5,
                        observed=observed_cropped,
                        sigma_map=sigma_cropped,
                        modeled_best=response.modeled_best,
                        raw_modeled_best=response.raw_modeled_best,
                        residual=response.residual,
                        frequency_ghz=freq_ghz,
                        q0_fitted=response.q0,
                        metrics_dict={
                            "chi2": response.chi2,
                            "rho2": response.rho2,
                            "eta2": response.eta2,
                        },
                        diagnostics=point_diag,
                        noise_diagnostics=noise_diag,
                    )
                    print(f"  ✓ Saved to: {point_h5}")

                best_q0[i, j] = float(response.q0)
                objective_values[i, j] = float(response.objective_value)
                chi2[i, j] = float(response.chi2)
                rho2[i, j] = float(response.rho2)
                eta2[i, j] = float(response.eta2)
                success[i, j] = bool(response.success)
                point_payloads[(int(i), int(j))] = build_computed_point_payload(
                    a_value=a_value,
                    b_value=b_value,
                    a_index=int(i),
                    b_index=int(j),
                    q0=float(response.q0),
                    success=bool(response.success),
                    status="computed",
                    modeled_best=response.modeled_best,
                    raw_modeled_best=response.raw_modeled_best,
                    residual=response.residual,
                    fit_q0_trials=tuple(float(v) for v in response.trial_q0),
                    fit_metric_trials=tuple(float(v) for v in response.trial_objective_values),
                    fit_chi2_trials=tuple(float(v) for v in response.trial_chi2_values),
                    fit_rho2_trials=tuple(float(v) for v in response.trial_rho2_values),
                    fit_eta2_trials=tuple(float(v) for v in response.trial_eta2_values),
                    nfev=int(response.nfev),
                    nit=int(response.nit),
                    message=str(response.message),
                    used_adaptive_bracketing=bool(response.used_adaptive_bracketing),
                    bracket_found=bool(response.bracket_found),
                    bracket=None if response.bracket is None else tuple(float(v) for v in response.bracket),
                    target_metric=str(args.target_metric),
                    diagnostics=point_diag,
                )

            append_point_record(
                out_h5,
                observed=observed_cropped,
                sigma_map=sigma_cropped,
                wcs_header=target_header,
                diagnostics=root_diag,
                point_payload=point_payloads[(int(i), int(j))],
            )
            _notify_viewer_refresh(f"point {point_counter}/{stage_total} saved")
            print(
                f"{'=' * 70}\n"
                f"POINT COMPLETE: success={bool(response.success and response.error_message is None)} total={float(response.elapsed_seconds):.3f}s\n"
                f"{'=' * 70}\n"
            )
            print(f"Viewer command: {viewer_cmd_text}")
            _maybe_launch_viewer(f"point {point_counter}/{stage_total} complete")

    grid_png = None if args.no_grid_png else artifacts_dir / f"{stem}_grid.png"
    point_png = None if args.no_point_png else artifacts_dir / f"{stem}_point.png"
    metric_arr = {"chi2": chi2, "rho2": rho2, "eta2": eta2}[str(args.target_metric)]
    if args.selected_a_index is None or args.selected_b_index is None:
        if np.any(np.isfinite(metric_arr)):
            selected_flat = int(np.nanargmin(metric_arr))
            selected_a_index, selected_b_index = np.unravel_index(selected_flat, metric_arr.shape)
            selected_a_index = int(selected_a_index)
            selected_b_index = int(selected_b_index)
        else:
            selected_a_index = 0
            selected_b_index = 0
    else:
        selected_a_index = int(np.clip(int(args.selected_a_index), 0, max(0, a_values.size - 1)))
        selected_b_index = int(np.clip(int(args.selected_b_index), 0, max(0, b_values.size - 1)))

    if grid_png is not None or point_png is not None or args.show_plot:
        plot_ab_scan_file(
            out_h5,
            a_index=selected_a_index,
            b_index=selected_b_index,
            out_grid_png=grid_png,
            out_point_png=point_png,
            show_plot=bool(args.show_plot),
        )

    print(f"\n✓ Saved consolidated scan file: {out_h5}")
    if grid_png is not None:
        print(f"✓ Grid summary PNG: {grid_png}")
    if point_png is not None:
        print(
            f"✓ Selected point: a_index={selected_a_index} b_index={selected_b_index} "
            f"a={float(a_values[selected_a_index]):.3f} b={float(b_values[selected_b_index]):.3f} "
            f"metric[{args.target_metric}]={float(metric_arr[selected_a_index, selected_b_index]):.6e}"
        )
        print(f"✓ Selected-point PNG: {point_png}")
    print(f"Viewer command: {shlex.join(viewer_cmd)}")
    _notify_viewer_refresh("scan complete")
    _maybe_launch_viewer("scan complete")
    total_elapsed = time.perf_counter() - scan_start
    success_count = int(np.count_nonzero(success))
    total_points = int(success.size)
    print(f"\n{'=' * 70}")
    print(f"SCAN COMPLETE: success={success_count}/{total_points} total={total_elapsed:.3f}s")
    print(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
