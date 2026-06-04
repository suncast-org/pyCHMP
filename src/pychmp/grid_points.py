"""Grid-point header + append-only trial storage for unified search artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from astropy.io import fits

from .ab_scan_artifacts import (
    ACTIVE_SEARCH_ID_DATASET,
    MAP_REFS_DATASET,
    SEARCHES_GROUP,
    SEARCH_LIFECYCLE_DATASET,
    SEARCH_REQUEST_DATASET,
    SLICE_CONTAINER_GROUP,
    TRIAL_HISTORY_DATASET,
    UNIFIED_ARTIFACT_KIND,
    _H5PY_FILE,
    _SPARSE_APPEND_RETRY_ATTEMPTS,
    _SPARSE_APPEND_RETRY_DELAY_S,
    is_h5_transient_read_error,
    _create_text_dataset,
    _replace_text_dataset,
    _derive_display_maps_from_raw,
    _diagnostics_from_slice_group,
    _json_dumps,
    _json_loads_or_empty,
    _map_store_identity,
    viewer_record_metrics,
    _matching_search_id_for_request,
    _normalize_point_status,
    _read_map_store_ref_array,
    _read_common_group,
    _resolve_slice_group,
    _search_id_from_diagnostics,
    _selected_search_id,
    _search_lifecycle_payload,
    _search_request_from_diagnostics,
    _search_should_remain_in_progress,
    _search_status_counts_from_records,
    _search_status_from_counts,
    _set_slice_group_attrs,
    _validate_new_slice_geometry_compatibility,
    _write_auxiliary_slice_shells,
    _write_common_group,
    _write_map_store_array,
    _write_reference_map_group,
    _write_search_lifecycle_dataset,
    _write_search_status_attrs,
    append_scan_point_record,
    decode_scalar,
    target_slice_descriptor_from_diagnostics,
)
from .ab_scan_artifacts import _read_point_group_sparse

GRID_POINTS_GROUP = "grid_points"
GRID_POINTS_TRIALS_GROUP = "trials"
GRID_POINT_STORAGE_CORRUPT_ATTR = "storage_corrupt"
GRID_POINTS_CONTRACT_VERSION = "2026-06-header-grid-points-v1"
GRID_POINT_AB_TOLERANCE = 1e-6


def grid_point_storage_corrupt(point_group: h5py.Group) -> bool:
    raw = point_group.attrs.get(GRID_POINT_STORAGE_CORRUPT_ATTR, 0)
    return raw in (1, True, "1", b"1")


class GridPointStatus(str, Enum):
    ASSIGNED = "ASSIGNED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def format_grid_point_id(order: int) -> str:
    return f"p{int(order):06d}"


def format_trial_id(trial_index: int) -> str:
    return f"t{int(trial_index):06d}"


def parse_grid_point_order(point_id: str) -> int:
    text = str(point_id).strip()
    if text.startswith("p"):
        return int(text[1:])
    raise ValueError(f"invalid grid point id: {point_id}")


def _read_optional_float_attr(group: h5py.Group, name: str) -> float | None:
    if name not in group.attrs:
        return None
    try:
        value = float(group.attrs[name])
    except Exception:
        return None
    return value if np.isfinite(value) else None


def read_grid_point_header(group: h5py.Group) -> dict[str, Any]:
    point_id = str(group.name.rsplit("/", 1)[-1])
    if "point_id" in group.attrs:
        point_id = decode_scalar(group.attrs["point_id"])
    header: dict[str, Any] = {
        "point_id": str(point_id),
        "a": float(group.attrs["a"]),
        "b": float(group.attrs["b"]),
        "status": decode_scalar(group.attrs.get("status", GridPointStatus.ASSIGNED.value)),
        "q0_start": float(group.attrs["q0_start"]),
        "best_trial_index": int(group.attrs.get("best_trial_index", -1)),
        "n_trials": int(group.attrs.get("n_trials", 0)),
        "metric_name": decode_scalar(group.attrs.get("metric_name", "chi2")),
        "created_utc": decode_scalar(group.attrs.get("created_utc", "")),
        "updated_utc": decode_scalar(group.attrs.get("updated_utc", "")),
    }
    next_q0 = _read_optional_float_attr(group, "next_q0")
    if next_q0 is not None:
        header["next_q0"] = float(next_q0)
    for optional in ("completed_utc", "failed_utc", "error_message"):
        if optional in group.attrs:
            header[optional] = decode_scalar(group.attrs[optional])
    return header


def classify_grid_point_state(header: dict[str, Any]) -> str:
    status = str(header.get("status", "")).strip().upper()
    if status == GridPointStatus.COMPLETED.value:
        return "complete"
    if status == GridPointStatus.FAILED.value:
        return "failed"
    n_trials = int(header.get("n_trials", 0))
    if "next_q0" in header:
        if n_trials <= 0:
            return "assigned_no_trials"
        return "running_partial"
    if status == GridPointStatus.ASSIGNED.value:
        return "assigned_no_trials"
    return "unknown"


def _write_header_attrs(group: h5py.Group, header: dict[str, Any]) -> None:
    group.attrs["point_id"] = np.bytes_(str(header["point_id"]))
    group.attrs["a"] = float(header["a"])
    group.attrs["b"] = float(header["b"])
    group.attrs["status"] = np.bytes_(str(header["status"]))
    group.attrs["q0_start"] = float(header["q0_start"])
    group.attrs["best_trial_index"] = int(header.get("best_trial_index", -1))
    group.attrs["n_trials"] = int(header.get("n_trials", 0))
    group.attrs["metric_name"] = np.bytes_(str(header.get("metric_name", "chi2")))
    group.attrs["created_utc"] = np.bytes_(str(header.get("created_utc", _utc_now())))
    group.attrs["updated_utc"] = np.bytes_(str(header.get("updated_utc", _utc_now())))
    if "next_q0" in header and header["next_q0"] is not None:
        group.attrs["next_q0"] = float(header["next_q0"])
    elif "next_q0" in group.attrs:
        del group.attrs["next_q0"]
    for optional in ("completed_utc", "failed_utc", "error_message"):
        if optional in header and header[optional] is not None and str(header[optional]).strip():
            group.attrs[optional] = np.bytes_(str(header[optional]))
        elif optional in group.attrs:
            del group.attrs[optional]


def find_grid_point_group(
    search_group: h5py.Group,
    *,
    a: float,
    b: float,
    tolerance: float = GRID_POINT_AB_TOLERANCE,
) -> tuple[str, h5py.Group] | None:
    if GRID_POINTS_GROUP not in search_group:
        return None
    grid_group = search_group[GRID_POINTS_GROUP]
    best: tuple[str, h5py.Group] | None = None
    best_order = -1
    for name in grid_group.keys():
        candidate = grid_group[name]
        if grid_point_storage_corrupt(candidate):
            continue
        try:
            cand_a = float(candidate.attrs["a"])
            cand_b = float(candidate.attrs["b"])
        except Exception:
            continue
        if not (np.isclose(cand_a, float(a), rtol=0.0, atol=tolerance) and np.isclose(cand_b, float(b), rtol=0.0, atol=tolerance)):
            continue
        try:
            order = parse_grid_point_order(str(name))
        except Exception:
            order = int(candidate.attrs.get("grid_point_order", -1))
        if best is None or order >= best_order:
            best = (str(name), candidate)
            best_order = order
    return best


def next_grid_point_order(search_group: h5py.Group) -> int:
    if GRID_POINTS_GROUP not in search_group:
        return 0
    orders: list[int] = []
    for name in search_group[GRID_POINTS_GROUP].keys():
        try:
            orders.append(parse_grid_point_order(str(name)))
        except Exception:
            continue
    return max(orders, default=-1) + 1


def list_grid_point_headers(search_group: h5py.Group) -> list[dict[str, Any]]:
    if GRID_POINTS_GROUP not in search_group:
        return []
    out: list[dict[str, Any]] = []
    for name in sorted(search_group[GRID_POINTS_GROUP].keys()):
        out.append(read_grid_point_header(search_group[GRID_POINTS_GROUP][name]))
    return out


def _ensure_contract_version(search_group: h5py.Group) -> None:
    diagnostics = _json_loads_or_empty(search_group["diagnostics_json"][()]) if "diagnostics_json" in search_group else {}
    if str(diagnostics.get("contract_version", "")).strip() != GRID_POINTS_CONTRACT_VERSION:
        diagnostics = dict(diagnostics)
        diagnostics["contract_version"] = GRID_POINTS_CONTRACT_VERSION
        if "diagnostics_json" in search_group:
            del search_group["diagnostics_json"]
        _create_text_dataset(search_group, "diagnostics_json", _json_dumps(diagnostics))


def _grid_points_count_for_search(search_group: h5py.Group) -> int:
    if GRID_POINTS_GROUP not in search_group:
        return 0
    return int(len(search_group[GRID_POINTS_GROUP]))


def _resolve_current_search_id(
    searches_group: h5py.Group,
    slice_group: h5py.Group,
    *,
    diagnostics: dict[str, Any],
    request_payload: dict[str, Any],
    layout_payload: dict[str, Any],
) -> str:
    """Pick the search group for grid writes (explicit runner id wins over contract matching)."""
    requested = str(diagnostics.get("selected_search_id") or diagnostics.get("search_id") or "").strip()
    if requested:
        return requested
    matched = _matching_search_id_for_request(searches_group, request_payload)
    if matched:
        return str(matched)
    if ACTIVE_SEARCH_ID_DATASET in slice_group:
        active = decode_scalar(slice_group[ACTIVE_SEARCH_ID_DATASET][()]).strip()
        if active and active in searches_group:
            return active
    return _search_id_from_diagnostics(diagnostics, layout=layout_payload)


def _ensure_search_context(
    h5_file: h5py.File,
    *,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
    blos_reference: tuple[np.ndarray, fits.Header] | None,
    psf_kernel: np.ndarray | None,
    ensure_slice_common: bool = True,
) -> tuple[h5py.Group, h5py.Group, str]:
    diagnostics_out = dict(diagnostics)
    diagnostics_out["artifact_kind"] = UNIFIED_ARTIFACT_KIND
    if "mask_type" not in diagnostics_out:
        diagnostics_out["mask_type"] = diagnostics.get("mask_type", "union")
    descriptor = target_slice_descriptor_from_diagnostics(diagnostics_out, fallback_key="default")
    resolved_slice_key = str(descriptor["key"] or "default")
    slices_group = h5_file.require_group(SLICE_CONTAINER_GROUP)
    if resolved_slice_key not in slices_group:
        _validate_new_slice_geometry_compatibility(
            slices_group,
            diagnostics=diagnostics_out,
            slice_key=resolved_slice_key,
            artifact_path=Path(h5_file.filename),
        )
        slice_group = slices_group.create_group(resolved_slice_key)
        _set_slice_group_attrs(slice_group, descriptor)
        if "common" in h5_file:
            from .ab_scan_artifacts import _copy_legacy_root_layout_to_slice

            _copy_legacy_root_layout_to_slice(h5_file, slice_group)
    else:
        slice_group = slices_group[resolved_slice_key]
        _set_slice_group_attrs(slice_group, descriptor)
    if "common" not in slice_group:
        if ensure_slice_common:
            common = slice_group.create_group("common")
            _write_common_group(
                common,
                observed=observed,
                sigma_map=sigma_map,
                wcs_header=wcs_header,
                diagnostics=diagnostics_out,
                blos_reference=blos_reference,
                psf_kernel=psf_kernel,
                run_history=None,
            )
            _write_auxiliary_slice_shells(
                slices_group,
                observed_template=observed,
                sigma_template=sigma_map,
                wcs_header=wcs_header,
                diagnostics=diagnostics_out,
                blos_reference=blos_reference,
                existing_names=set(slices_group.keys()),
            )
    elif ensure_slice_common and blos_reference is not None and "refmaps" not in slice_group["common"]:
        refmaps = slice_group["common"].create_group("refmaps")
        blos_data, blos_header = blos_reference
        _write_reference_map_group(
            refmaps,
            group_name="Bz_reference",
            data=np.asarray(blos_data, dtype=float),
            wcs_header=blos_header,
        )
    searches_group = slice_group.require_group(SEARCHES_GROUP)
    layout_payload = {"kind": "point_list"}
    request_payload = _search_request_from_diagnostics(diagnostics_out, layout=layout_payload)
    current_search_id = _resolve_current_search_id(
        searches_group,
        slice_group,
        diagnostics=diagnostics_out,
        request_payload=request_payload,
        layout_payload=layout_payload,
    )
    if ACTIVE_SEARCH_ID_DATASET in slice_group:
        del slice_group[ACTIVE_SEARCH_ID_DATASET]
    _create_text_dataset(slice_group, ACTIVE_SEARCH_ID_DATASET, current_search_id)
    search_group = searches_group.require_group(current_search_id)
    if "diagnostics_json" not in search_group:
        search_group.attrs["search_id"] = np.bytes_(current_search_id)
        search_group.attrs["target_metric"] = np.bytes_(str(diagnostics_out.get("target_metric", "chi2")))
        _create_text_dataset(search_group, "diagnostics_json", _json_dumps(diagnostics_out))
        _create_text_dataset(search_group, "layout_json", _json_dumps(layout_payload))
        _create_text_dataset(search_group, "run_history_json", _json_dumps([]))
        _create_text_dataset(search_group, SEARCH_REQUEST_DATASET, _json_dumps(request_payload))
    _ensure_contract_version(search_group)
    search_group.require_group(GRID_POINTS_GROUP)
    return slice_group, search_group, current_search_id


def _update_search_counts(search_group: h5py.Group, *, diagnostics: dict[str, Any]) -> None:
    grid_headers = list_grid_point_headers(search_group)
    records: list[dict[str, Any]] = []
    for header in grid_headers:
        status = classify_grid_point_state(header)
        if status == "complete":
            records.append({"status": "computed"})
        elif status == "failed":
            records.append({"status": "failed"})
        else:
            records.append({"status": "pending"})
    counts = _search_status_counts_from_records(records)
    existing_lifecycle = (
        _json_loads_or_empty(search_group[SEARCH_LIFECYCLE_DATASET][()])
        if SEARCH_LIFECYCLE_DATASET in search_group
        else {}
    )
    status = _write_search_status_attrs(
        search_group,
        counts,
        diagnostics=diagnostics,
        remain_in_progress=_search_should_remain_in_progress(
            diagnostics=diagnostics,
            existing_lifecycle=existing_lifecycle,
        ),
    )
    lifecycle = _search_lifecycle_payload(
        status=status,
        diagnostics=diagnostics,
        existing=existing_lifecycle,
    )
    _write_search_lifecycle_dataset(search_group, lifecycle=lifecycle)


@dataclass(frozen=True)
class GridPointAssignedEvent:
    a: float
    b: float
    q0_start: float
    next_q0: float
    metric_name: str
    point_id: str | None = None
    force_new_point_id: bool = False


def _resolve_trial_metric_values(
    *,
    target_metric: str,
    metric: float,
    chi2: float | None = None,
    rho2: float | None = None,
    eta2: float | None = None,
) -> tuple[float, float, float]:
    """Resolve per-trial chi2/rho2/eta2, falling back to the target metric value."""
    resolved: dict[str, float | None] = {
        "chi2": chi2,
        "rho2": rho2,
        "eta2": eta2,
    }
    out: dict[str, float] = {}
    target = str(target_metric or "chi2").strip().lower()
    for name in ("chi2", "rho2", "eta2"):
        value = resolved[name]
        if value is not None and np.isfinite(float(value)):
            out[name] = float(value)
        elif name == target and np.isfinite(float(metric)):
            out[name] = float(metric)
        else:
            out[name] = float("nan")
    return out["chi2"], out["rho2"], out["eta2"]


def _assemble_trial_metric_arrays(
    trials: list[dict[str, Any]],
    *,
    target_metric: str,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    fit_metric_trials = tuple(float(item.get("target_metric_value", np.nan)) for item in trials)
    target = str(target_metric or "chi2").strip().lower()
    chi2_trials: list[float] = []
    rho2_trials: list[float] = []
    eta2_trials: list[float] = []
    for item in trials:
        chi2_value = item.get("chi2")
        rho2_value = item.get("rho2")
        eta2_value = item.get("eta2")
        target_value = float(item.get("target_metric_value", np.nan))
        if chi2_value is not None and np.isfinite(float(chi2_value)):
            chi2_trials.append(float(chi2_value))
        elif target == "chi2" and np.isfinite(target_value):
            chi2_trials.append(target_value)
        else:
            chi2_trials.append(float("nan"))
        if rho2_value is not None and np.isfinite(float(rho2_value)):
            rho2_trials.append(float(rho2_value))
        elif target == "rho2" and np.isfinite(target_value):
            rho2_trials.append(target_value)
        else:
            rho2_trials.append(float("nan"))
        if eta2_value is not None and np.isfinite(float(eta2_value)):
            eta2_trials.append(float(eta2_value))
        elif target == "eta2" and np.isfinite(target_value):
            eta2_trials.append(target_value)
        else:
            eta2_trials.append(float("nan"))
    return fit_metric_trials, tuple(chi2_trials), tuple(rho2_trials), tuple(eta2_trials)


@dataclass(frozen=True)
class GridTrialCommittedEvent:
    point_id: str
    trial_index: int
    q0: float
    metric: float
    next_q0: float | None
    best_trial_index: int
    best_metric: float
    raw_modeled_map: np.ndarray | None = None
    raw_map_ref: str | None = None
    trial_metadata: dict[str, Any] | None = None
    shift_x: float | None = None
    shift_y: float | None = None
    shift_valid: bool | None = None
    chi2: float | None = None
    rho2: float | None = None
    eta2: float | None = None


@dataclass(frozen=True)
class GridPointCompletedEvent:
    point_id: str
    best_trial_index: int
    best_metric: float
    best_q0: float | None = None


@dataclass(frozen=True)
class GridPointFailedEvent:
    point_id: str
    error_message: str
    terminal: bool = False


def _validate_linked_map_store_ref(
    h5_file: h5py.File,
    linked_ref: str,
    *,
    map_store_artifact: Path | None,
) -> None:
    if map_store_artifact is not None:
        with _H5PY_FILE(map_store_artifact, "r") as ref_store:
            if _read_map_store_ref_array(ref_store, linked_ref) is None:
                raise ValueError(f"map_store reference not found: {linked_ref}")
        return
    if _read_map_store_ref_array(h5_file, linked_ref) is None:
        raise ValueError(f"map_store reference not found: {linked_ref}")


def apply_grid_point_active_q0(
    h5_path: Path,
    *,
    point_id: str,
    next_q0: float,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
    blos_reference: tuple[np.ndarray, fits.Header] | None = None,
    psf_kernel: np.ndarray | None = None,
    map_store_artifact: Path | None = None,
    ensure_slice_common: bool = True,
) -> None:
    with _H5PY_FILE(h5_path, "a") as f:
        _slice_group, search_group, _search_id = _ensure_search_context(
            f,
            observed=observed,
            sigma_map=sigma_map,
            wcs_header=wcs_header,
            diagnostics=diagnostics,
            blos_reference=blos_reference,
            psf_kernel=psf_kernel,
            ensure_slice_common=ensure_slice_common,
        )
        point_group = _grid_point_group(search_group, str(point_id))
        header = read_grid_point_header(point_group)
        header["next_q0"] = float(next_q0)
        if int(header.get("n_trials", 0)) > 0:
            header["status"] = GridPointStatus.RUNNING.value
        header["updated_utc"] = _utc_now()
        _write_header_attrs(point_group, header)


@dataclass(frozen=True)
class GridPointActiveQ0Event:
    point_id: str
    next_q0: float


def apply_grid_point_assigned(
    h5_path: Path,
    event: GridPointAssignedEvent,
    *,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
    blos_reference: tuple[np.ndarray, fits.Header] | None = None,
    psf_kernel: np.ndarray | None = None,
    map_store_artifact: Path | None = None,
    ensure_slice_common: bool = True,
) -> str:
    _ = map_store_artifact
    with _H5PY_FILE(h5_path, "a") as f:
        _slice_group, search_group, _search_id = _ensure_search_context(
            f,
            observed=observed,
            sigma_map=sigma_map,
            wcs_header=wcs_header,
            diagnostics=diagnostics,
            blos_reference=blos_reference,
            psf_kernel=psf_kernel,
            ensure_slice_common=ensure_slice_common,
        )
        grid_root = search_group[GRID_POINTS_GROUP]
        if not event.force_new_point_id:
            existing = find_grid_point_group(search_group, a=float(event.a), b=float(event.b))
            if existing is not None:
                point_id, group = existing
                header = read_grid_point_header(group)
                header["status"] = GridPointStatus.ASSIGNED.value
                header["q0_start"] = float(event.q0_start)
                header["next_q0"] = float(event.next_q0)
                header["updated_utc"] = _utc_now()
                _write_header_attrs(group, header)
                return point_id
        order = next_grid_point_order(search_group)
        point_id = event.point_id or format_grid_point_id(order)
        if point_id in grid_root:
            raise KeyError(f"grid point already exists: {point_id}")
        group = grid_root.create_group(point_id)
        now = _utc_now()
        header = {
            "point_id": point_id,
            "a": float(event.a),
            "b": float(event.b),
            "status": GridPointStatus.ASSIGNED.value,
            "q0_start": float(event.q0_start),
            "next_q0": float(event.next_q0),
            "best_trial_index": -1,
            "n_trials": 0,
            "metric_name": str(event.metric_name),
            "created_utc": now,
            "updated_utc": now,
        }
        _write_header_attrs(group, header)
        group.create_group(GRID_POINTS_TRIALS_GROUP)
        _update_search_counts(search_group, diagnostics=diagnostics)
        return point_id


def _grid_point_group(search_group: h5py.Group, point_id: str) -> h5py.Group:
    if GRID_POINTS_GROUP not in search_group or point_id not in search_group[GRID_POINTS_GROUP]:
        raise KeyError(f"grid point not found: {point_id}")
    return search_group[GRID_POINTS_GROUP][point_id]


def resolve_trial_shift_commit_fields(
    *,
    trial_index: int,
    shift_x_trials: Any = None,
    shift_y_trials: Any = None,
    shift_valid_trials: Any = None,
    stage: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], float | None, float | None, bool | None]:
    """Build trial metadata and GridTrialCommittedEvent shift fields for one trial."""
    trial_metadata = dict(extra_metadata or {})
    if stage is not None and str(stage).strip():
        trial_metadata["stage"] = str(stage)
    shift_x = None
    shift_y = None
    shift_valid = None
    if shift_x_trials is not None and shift_y_trials is not None:
        x_vals = np.asarray(shift_x_trials, dtype=float)
        y_vals = np.asarray(shift_y_trials, dtype=float)
        idx = int(trial_index)
        if 0 <= idx < min(x_vals.size, y_vals.size):
            shift_x = float(x_vals[idx])
            shift_y = float(y_vals[idx])
            if np.isfinite(shift_x) and np.isfinite(shift_y):
                if shift_valid_trials is not None:
                    valid_vals = np.asarray(shift_valid_trials, dtype=bool)
                    if 0 <= idx < valid_vals.size:
                        shift_valid = bool(valid_vals[idx])
            else:
                shift_x = None
                shift_y = None
    return trial_metadata, shift_x, shift_y, shift_valid


def apply_grid_trial_committed(
    h5_path: Path,
    event: GridTrialCommittedEvent,
    *,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
    blos_reference: tuple[np.ndarray, fits.Header] | None = None,
    psf_kernel: np.ndarray | None = None,
    map_store_artifact: Path | None = None,
    ensure_slice_common: bool = True,
) -> None:
    with _H5PY_FILE(h5_path, "a") as f:
        _slice_group, search_group, _search_id = _ensure_search_context(
            f,
            observed=observed,
            sigma_map=sigma_map,
            wcs_header=wcs_header,
            diagnostics=diagnostics,
            blos_reference=blos_reference,
            psf_kernel=psf_kernel,
            ensure_slice_common=ensure_slice_common,
        )
        point_group = _grid_point_group(search_group, str(event.point_id))
        header = read_grid_point_header(point_group)
        trials_group = point_group.require_group(GRID_POINTS_TRIALS_GROUP)
        trial_name = format_trial_id(int(event.trial_index))
        if trial_name in trials_group:
            trial_group = trials_group[trial_name]
        else:
            trial_group = trials_group.create_group(trial_name)
        trial_group.attrs["trial_index"] = int(event.trial_index)
        trial_group.attrs["q0"] = float(event.q0)
        trial_group.attrs["metric"] = float(event.metric)
        trial_group.attrs["target_metric"] = np.bytes_(str(header.get("metric_name", "chi2")))
        chi2_value, rho2_value, eta2_value = _resolve_trial_metric_values(
            target_metric=str(header.get("metric_name", "chi2")),
            metric=float(event.metric),
            chi2=event.chi2,
            rho2=event.rho2,
            eta2=event.eta2,
        )
        trial_group.attrs["chi2"] = float(chi2_value)
        trial_group.attrs["rho2"] = float(rho2_value)
        trial_group.attrs["eta2"] = float(eta2_value)
        map_refs = _json_loads_or_empty(trial_group[MAP_REFS_DATASET][()]) if MAP_REFS_DATASET in trial_group else {}
        linked_ref = str(event.raw_map_ref or "").strip()
        if linked_ref:
            _validate_linked_map_store_ref(f, linked_ref, map_store_artifact=map_store_artifact)
            map_refs["raw_modeled"] = linked_ref
        elif event.raw_modeled_map is not None:
            identity_source = {
                **dict(diagnostics),
                "a": float(header["a"]),
                "b": float(header["b"]),
                "q0": float(event.q0),
                "target_metric": str(header.get("metric_name", "chi2")),
            }
            map_refs["raw_modeled"] = _write_map_store_array(
                f,
                identity=_map_store_identity(
                    name=f"grid_points/{event.point_id}/trials/{trial_name}/raw_modeled",
                    normalized=identity_source,
                ),
                data=np.asarray(event.raw_modeled_map, dtype=float),
            )
        existing_metadata = (
            _json_loads_or_empty(trial_group["trial_metadata_json"][()])
            if "trial_metadata_json" in trial_group
            else {}
        )
        trial_metadata = {**existing_metadata, **dict(event.trial_metadata or {})}
        if event.shift_x is not None:
            trial_metadata["shift_x_arcsec"] = float(event.shift_x)
        if event.shift_y is not None:
            trial_metadata["shift_y_arcsec"] = float(event.shift_y)
        if event.shift_valid is not None:
            trial_metadata["shift_valid"] = bool(event.shift_valid)
        _replace_text_dataset(trial_group, "trial_metadata_json", _json_dumps(trial_metadata))
        _replace_text_dataset(trial_group, MAP_REFS_DATASET, _json_dumps(map_refs))
        q0_value = float(event.q0)
        has_map = bool(str(map_refs.get("raw_modeled", "")).strip())
        if np.isfinite(q0_value) and q0_value > 0.0 and not has_map:
            raise ValueError(
                "GridTrialCommittedEvent requires a stored map for finite Q0 trials "
                f"(point_id={event.point_id}, trial_index={int(event.trial_index)}, q0={q0_value:g})"
            )
        header["n_trials"] = max(int(header.get("n_trials", 0)), int(event.trial_index) + 1)
        header["best_trial_index"] = int(event.best_trial_index)
        header["status"] = GridPointStatus.RUNNING.value
        header["updated_utc"] = _utc_now()
        if event.next_q0 is None:
            header.pop("next_q0", None)
        else:
            header["next_q0"] = float(event.next_q0)
        _write_header_attrs(point_group, header)
        _update_search_counts(search_group, diagnostics=diagnostics)


def apply_grid_point_completed(
    h5_path: Path,
    event: GridPointCompletedEvent,
    *,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
    blos_reference: tuple[np.ndarray, fits.Header] | None = None,
    psf_kernel: np.ndarray | None = None,
    map_store_artifact: Path | None = None,
    ensure_slice_common: bool = True,
) -> None:
    _ = map_store_artifact
    with _H5PY_FILE(h5_path, "a") as f:
        _slice_group, search_group, _search_id = _ensure_search_context(
            f,
            observed=observed,
            sigma_map=sigma_map,
            wcs_header=wcs_header,
            diagnostics=diagnostics,
            blos_reference=blos_reference,
            psf_kernel=psf_kernel,
            ensure_slice_common=ensure_slice_common,
        )
        point_group = _grid_point_group(search_group, str(event.point_id))
        header = read_grid_point_header(point_group)
        header["status"] = GridPointStatus.COMPLETED.value
        header["best_trial_index"] = int(event.best_trial_index)
        header.pop("next_q0", None)
        header["updated_utc"] = _utc_now()
        header["completed_utc"] = _utc_now()
        _write_header_attrs(point_group, header)
        _update_search_counts(search_group, diagnostics=diagnostics)


def apply_grid_point_failed(
    h5_path: Path,
    event: GridPointFailedEvent,
    *,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
    blos_reference: tuple[np.ndarray, fits.Header] | None = None,
    psf_kernel: np.ndarray | None = None,
    map_store_artifact: Path | None = None,
    ensure_slice_common: bool = True,
) -> None:
    _ = map_store_artifact
    with _H5PY_FILE(h5_path, "a") as f:
        _slice_group, search_group, _search_id = _ensure_search_context(
            f,
            observed=observed,
            sigma_map=sigma_map,
            wcs_header=wcs_header,
            diagnostics=diagnostics,
            blos_reference=blos_reference,
            psf_kernel=psf_kernel,
            ensure_slice_common=ensure_slice_common,
        )
        point_group = _grid_point_group(search_group, str(event.point_id))
        header = read_grid_point_header(point_group)
        header["status"] = GridPointStatus.FAILED.value
        header["error_message"] = str(event.error_message)
        header["failed_utc"] = _utc_now()
        header["updated_utc"] = _utc_now()
        if event.terminal:
            header.pop("next_q0", None)
        _write_header_attrs(point_group, header)
        _update_search_counts(search_group, diagnostics=diagnostics)


def select_fit_trials_for_viewer(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only optimizer Q0 trials (finite q0); drop mask/diagnostic rows with NaN q0."""
    selected: list[dict[str, Any]] = []
    for item in trials:
        try:
            q0_value = float(item["q0"])
        except Exception:
            continue
        if np.isfinite(q0_value) and q0_value > 0.0:
            selected.append(item)
    return selected


def _grid_point_finite_q0_trials_have_map_store_links_on_group(
    root_file: h5py.File,
    point_group: h5py.Group,
) -> bool:
    """True when every finite-Q0 trial row resolves to a readable map_store dataset."""
    trials = select_fit_trials_for_viewer(_load_grid_point_trials(point_group, include_maps=False))
    if not trials:
        return True
    for trial in trials:
        raw_ref = str(trial.get("raw_map_ref", "") or "").strip()
        if not raw_ref:
            return False
        if _read_map_store_ref_array(root_file, raw_ref) is None:
            return False
    return True


def grid_point_finite_q0_trials_have_map_store_links(
    h5_path: Path,
    *,
    slice_key: str,
    search_id: str,
    point_id: str,
) -> bool:
    """True when every finite-Q0 trial row resolves to a readable map_store dataset."""
    last_exc: BaseException | None = None
    for attempt in range(1, _SPARSE_APPEND_RETRY_ATTEMPTS + 1):
        try:
            with _H5PY_FILE(h5_path, "r") as f:
                if SLICE_CONTAINER_GROUP not in f or slice_key not in f[SLICE_CONTAINER_GROUP]:
                    return True
                slice_group = f[SLICE_CONTAINER_GROUP][slice_key]
                if SEARCHES_GROUP not in slice_group or search_id not in slice_group[SEARCHES_GROUP]:
                    return True
                search_group = slice_group[SEARCHES_GROUP][search_id]
                if GRID_POINTS_GROUP not in search_group or point_id not in search_group[GRID_POINTS_GROUP]:
                    return True
                point_group = search_group[GRID_POINTS_GROUP][point_id]
                return _grid_point_finite_q0_trials_have_map_store_links_on_group(f, point_group)
        except (OSError, RuntimeError, KeyError) as exc:
            if not is_h5_transient_read_error(exc):
                raise
            last_exc = exc
            import time

            time.sleep(_SPARSE_APPEND_RETRY_DELAY_S)
    if last_exc is not None:
        raise last_exc
    return True


def _mark_grid_point_storage_corrupt(point_group: h5py.Group, *, point_id: str) -> None:
    """Quarantine a grid point whose HDF5 trial subtree cannot be read or deleted."""
    point_group.attrs[GRID_POINT_STORAGE_CORRUPT_ATTR] = 1
    point_group.attrs["storage_corrupt_utc"] = _utc_now()
    point_group.attrs["storage_corrupt_point_id"] = str(point_id)


def _reset_grid_point_on_group(
    point_group: h5py.Group,
    *,
    point_id: str,
    q0_start: float,
    next_q0: float,
    metric_name: str,
) -> None:
    """Clear invalid trial rows and reset header (caller holds the artifact file open)."""
    if GRID_POINTS_TRIALS_GROUP in point_group:
        try:
            del point_group[GRID_POINTS_TRIALS_GROUP]
        except (OSError, RuntimeError, KeyError):
            _mark_grid_point_storage_corrupt(point_group, point_id=str(point_id))
            return
    point_group.create_group(GRID_POINTS_TRIALS_GROUP)
    existing = read_grid_point_header(point_group)
    now = _utc_now()
    header = {
        "point_id": str(point_id),
        "a": float(existing["a"]),
        "b": float(existing["b"]),
        "status": GridPointStatus.ASSIGNED.value,
        "q0_start": float(q0_start),
        "next_q0": float(next_q0),
        "best_trial_index": -1,
        "n_trials": 0,
        "metric_name": str(metric_name),
        "created_utc": str(existing.get("created_utc", now)),
        "updated_utc": now,
    }
    _write_header_attrs(point_group, header)


def reset_grid_point_for_rerun(
    h5_path: Path,
    *,
    slice_key: str,
    search_id: str,
    point_id: str,
    q0_start: float,
    next_q0: float,
    metric_name: str,
) -> None:
    """Clear invalid trial rows and reset header so the point is recomputed with map-linked trials only."""
    with _H5PY_FILE(h5_path, "r+") as f:
        point_group = f[SLICE_CONTAINER_GROUP][slice_key][SEARCHES_GROUP][search_id][GRID_POINTS_GROUP][point_id]
        _reset_grid_point_on_group(
            point_group,
            point_id=str(point_id),
            q0_start=float(q0_start),
            next_q0=float(next_q0),
            metric_name=str(metric_name),
        )


def repair_invalid_grid_points_in_search(
    h5_path: Path,
    *,
    slice_key: str,
    search_id: str,
) -> int:
    """Reset only grid points whose finite-Q0 trials lack valid map_store links."""
    reset_count = 0
    with _H5PY_FILE(h5_path, "r+") as f:
        if SLICE_CONTAINER_GROUP not in f or slice_key not in f[SLICE_CONTAINER_GROUP]:
            return 0
        slice_group = f[SLICE_CONTAINER_GROUP][slice_key]
        if SEARCHES_GROUP not in slice_group or search_id not in slice_group[SEARCHES_GROUP]:
            return 0
        search_group = slice_group[SEARCHES_GROUP][search_id]
        if GRID_POINTS_GROUP not in search_group:
            return 0
        for point_name in sorted(search_group[GRID_POINTS_GROUP].keys()):
            point_group = search_group[GRID_POINTS_GROUP][point_name]
            try:
                valid_links = _grid_point_finite_q0_trials_have_map_store_links_on_group(f, point_group)
            except (OSError, RuntimeError):
                valid_links = False
            if valid_links or grid_point_storage_corrupt(point_group):
                continue
            existing = read_grid_point_header(point_group)
            resume_q0 = float(existing.get("next_q0", existing.get("q0_start", np.nan)))
            if not np.isfinite(resume_q0):
                resume_q0 = float(existing.get("q0_start", np.nan))
            if not np.isfinite(resume_q0):
                continue
            _reset_grid_point_on_group(
                point_group,
                point_id=str(point_name),
                q0_start=float(resume_q0),
                next_q0=float(resume_q0),
                metric_name=str(existing.get("metric_name", "chi2")),
            )
            reset_count += 1
    return reset_count


def _load_grid_point_trials_once(point_group: h5py.Group, *, include_maps: bool) -> list[dict[str, Any]]:
    if GRID_POINTS_TRIALS_GROUP not in point_group:
        return []
    trials: list[dict[str, Any]] = []
    trials_group = point_group[GRID_POINTS_TRIALS_GROUP]
    for name in sorted(trials_group.keys()):
        trial = trials_group[name]
        trial_metadata = _json_loads_or_empty(trial["trial_metadata_json"][()]) if "trial_metadata_json" in trial else {}
        map_refs = _json_loads_or_empty(trial[MAP_REFS_DATASET][()]) if MAP_REFS_DATASET in trial else {}
        entry = {
            "trial_index": int(trial.attrs.get("trial_index", 0)),
            "q0": float(trial.attrs["q0"]),
            "target_metric_value": float(trial.attrs.get("metric", np.nan)),
            "chi2": float(trial.attrs.get("chi2", np.nan)),
            "rho2": float(trial.attrs.get("rho2", np.nan)),
            "eta2": float(trial.attrs.get("eta2", np.nan)),
            "raw_map_ref": str(map_refs.get("raw_modeled", "")),
            "trial_metadata": trial_metadata,
        }
        if include_maps and entry["raw_map_ref"]:
            from .ab_scan_artifacts import _read_map_store_ref_array

            raw_map = _read_map_store_ref_array(trial.file, entry["raw_map_ref"])
            entry["raw_modeled_map"] = raw_map
        trials.append(entry)
    return trials


def _load_grid_point_trials(point_group: h5py.Group, *, include_maps: bool) -> list[dict[str, Any]]:
    last_exc: BaseException | None = None
    for attempt in range(1, _SPARSE_APPEND_RETRY_ATTEMPTS + 1):
        try:
            return _load_grid_point_trials_once(point_group, include_maps=include_maps)
        except (OSError, RuntimeError, KeyError) as exc:
            if not is_h5_transient_read_error(exc):
                raise
            last_exc = exc
            import time

            time.sleep(_SPARSE_APPEND_RETRY_DELAY_S)
    if last_exc is not None:
        raise last_exc
    return []


def grid_point_header_to_viewer_record(
    header: dict[str, Any],
    trials: list[dict[str, Any]],
    *,
    include_maps: bool,
) -> dict[str, Any]:
    fit_trials = select_fit_trials_for_viewer(trials)
    fit_q0_trials = tuple(float(item["q0"]) for item in fit_trials)
    target_metric = str(header.get("metric_name", "chi2"))
    fit_metric_trials, fit_chi2_trials, fit_rho2_trials, fit_eta2_trials = _assemble_trial_metric_arrays(
        fit_trials,
        target_metric=target_metric,
    )
    trial_history = [
        {
            "trial_index": int(item["trial_index"]),
            "q0": float(item["q0"]),
            "target_metric_value": float(item.get("target_metric_value", np.nan)),
            "raw_map_ref": str(item.get("raw_map_ref", "")),
        }
        for item in fit_trials
    ]
    best_trial_index = int(header.get("best_trial_index", -1))
    if best_trial_index < 0 and fit_trials:
        best_trial_index = int(max(fit_trials, key=lambda item: int(item["trial_index"]))["trial_index"])
    best_q0 = float(header.get("q0_start", np.nan))
    best_array_index = next(
        (idx for idx, item in enumerate(fit_trials) if int(item["trial_index"]) == int(best_trial_index)),
        None,
    )
    if best_array_index is not None:
        best_q0 = float(fit_q0_trials[int(best_array_index)])
    status = "computed" if classify_grid_point_state(header) == "complete" else "pending"
    if str(header.get("status", "")).upper() == GridPointStatus.FAILED.value:
        status = "failed"
    record_preview = {
        "fit_q0_trials": fit_q0_trials,
        "fit_metric_trials": fit_metric_trials,
        "fit_chi2_trials": fit_chi2_trials,
        "fit_rho2_trials": fit_rho2_trials,
        "fit_eta2_trials": fit_eta2_trials,
        "target_metric": target_metric,
    }
    metrics = viewer_record_metrics(record_preview)
    raw_modeled_best = None
    trial_raw_maps = []
    if include_maps:
        for item in fit_trials:
            raw_map = item.get("raw_modeled_map")
            if raw_map is not None:
                trial_raw_maps.append(np.asarray(raw_map, dtype=float))
        if best_array_index is not None and 0 <= int(best_array_index) < len(trial_raw_maps):
            raw_modeled_best = trial_raw_maps[int(best_array_index)]
    trial_raw_stack = None
    if trial_raw_maps:
        trial_raw_stack = np.stack(trial_raw_maps, axis=0)
    return {
        "record_order": parse_grid_point_order(str(header["point_id"])),
        "a": float(header["a"]),
        "b": float(header["b"]),
        "q0": float(best_q0),
        "success": status == "computed",
        "status": status,
        "fit_q0_trials": fit_q0_trials,
        "fit_metric_trials": fit_metric_trials,
        "fit_chi2_trials": fit_chi2_trials,
        "fit_rho2_trials": fit_rho2_trials,
        "fit_eta2_trials": fit_eta2_trials,
        "metrics": metrics,
        "fit_shift_x_trials": tuple(
            float(item.get("trial_metadata", {}).get("shift_x_arcsec", np.nan)) for item in fit_trials
        ),
        "fit_shift_y_trials": tuple(
            float(item.get("trial_metadata", {}).get("shift_y_arcsec", np.nan)) for item in fit_trials
        ),
        "fit_find_shift_valid_trials": tuple(
            bool(item.get("trial_metadata", {}).get("shift_valid", False)) for item in fit_trials
        ),
        "fit_trial_mask_stages": tuple(
            str(item.get("trial_metadata", {}).get("stage", "")) for item in fit_trials
        ),
        "trial_history": trial_history,
        "best_trial_index": None if best_trial_index < 0 else int(best_trial_index),
        "target_metric": target_metric,
        "raw_modeled_best": raw_modeled_best,
        "trial_raw_modeled_maps": trial_raw_stack,
        "diagnostics": {
            "grid_point_id": str(header["point_id"]),
            "grid_point_status": str(header.get("status", "")),
            target_metric: metrics[target_metric],
            "target_metric_value": metrics[target_metric],
        },
        "nfev": int(len(trials)),
        "nit": max(0, int(len(trials) - 1)),
        "message": str(header.get("error_message", "")),
        "used_adaptive_bracketing": True,
        "bracket_found": False,
        "bracket": None,
        "map_refs": {},
    }


def load_grid_points_as_viewer_records(
    search_group: h5py.Group,
    *,
    include_maps: bool = True,
) -> list[dict[str, Any]]:
    if GRID_POINTS_GROUP not in search_group:
        return []
    records: list[dict[str, Any]] = []
    grid_group = search_group[GRID_POINTS_GROUP]
    for name in sorted(grid_group.keys()):
        point_group = grid_group[name]
        if grid_point_storage_corrupt(point_group):
            continue
        try:
            header = read_grid_point_header(point_group)
            trials = _load_grid_point_trials(point_group, include_maps=include_maps)
            records.append(grid_point_header_to_viewer_record(header, trials, include_maps=include_maps))
        except (OSError, RuntimeError, KeyError) as exc:
            if is_h5_transient_read_error(exc):
                raise
            continue
    return records


def _resolve_grid_point_trial_raw_map_ref(
    trials: list[dict[str, Any]],
    *,
    requested_trial_index: int | None,
    header: dict[str, Any],
    use_array_index: bool = False,
) -> tuple[str, int | None]:
    fit_trials = select_fit_trials_for_viewer(trials)
    indexed: list[tuple[int, str]] = []
    for item in fit_trials:
        ref = str(item.get("raw_map_ref", "") or "").strip()
        if ref:
            indexed.append((int(item["trial_index"]), ref))
    if not indexed:
        return "", None
    if requested_trial_index is not None:
        requested = int(requested_trial_index)
        if use_array_index:
            if 0 <= requested < len(fit_trials):
                ref = str(fit_trials[requested].get("raw_map_ref", "") or "").strip()
                if ref:
                    return ref, int(fit_trials[requested]["trial_index"])
        else:
            for trial_idx, ref in indexed:
                if trial_idx == requested:
                    return ref, requested
    best_trial_index = int(header.get("best_trial_index", -1))
    if best_trial_index >= 0:
        for trial_idx, ref in indexed:
            if trial_idx == best_trial_index:
                return ref, best_trial_index
    trial_idx, ref = indexed[-1]
    return ref, trial_idx


def load_grid_point_trial_plot_payload(
    h5_path: Path,
    *,
    a: float,
    b: float,
    trial_index: int | None = None,
    slice_key: str | None = None,
    search_id: str | None = None,
) -> dict[str, Any] | None:
    with _H5PY_FILE(h5_path, "r") as f:
        group, _descriptors, selected_key = _resolve_slice_group(
            f,
            slice_key=slice_key,
            allow_missing=slice_key is not None,
        )
        if group is None:
            return None
        selected_search_id = _selected_search_id(group, requested_search_id=search_id)
        if selected_search_id is None or SEARCHES_GROUP not in group:
            return None
        searches_group = group[SEARCHES_GROUP]
        if selected_search_id not in searches_group:
            return None
        search_group = searches_group[selected_search_id]
        if GRID_POINTS_GROUP not in search_group:
            return None
        found = find_grid_point_group(search_group, a=float(a), b=float(b))
        if found is None:
            return None
        _point_id, point_group = found
        header = read_grid_point_header(point_group)
        trials = _load_grid_point_trials(point_group, include_maps=False)
        fit_trials = select_fit_trials_for_viewer(trials)
        fit_q0_trials = np.asarray([float(item["q0"]) for item in fit_trials], dtype=float)
        if fit_q0_trials.size == 0:
            return None
        chosen_array_index = None if trial_index is None else int(trial_index)
        if chosen_array_index is None:
            best_trial_index = header.get("best_trial_index")
            if best_trial_index is not None:
                best_array_index = next(
                    (
                        idx
                        for idx, item in enumerate(fit_trials)
                        if int(item["trial_index"]) == int(best_trial_index)
                    ),
                    None,
                )
                chosen_array_index = 0 if best_array_index is None else int(best_array_index)
            else:
                chosen_array_index = int(fit_q0_trials.size - 1)
        else:
            chosen_array_index = int(np.clip(chosen_array_index, 0, int(fit_q0_trials.size) - 1))
        raw_map_ref, resolved_trial_index = _resolve_grid_point_trial_raw_map_ref(
            trials,
            requested_trial_index=chosen_array_index,
            header=header,
            use_array_index=True,
        )
        if not raw_map_ref or resolved_trial_index is None:
            return None
        storage_trial_index = int(resolved_trial_index)
        raw_modeled = _read_map_store_ref_array(f, raw_map_ref)
        if raw_modeled is None:
            return None
        common_payload = _read_common_group(group["common"])
        observed = np.asarray(common_payload.get("observed"), dtype=float)
        psf_kernel = common_payload.get("psf_kernel")
        raw_display, modeled, residual, _has_raw = _derive_display_maps_from_raw(
            raw_modeled,
            observed_template=observed,
            psf_kernel=psf_kernel,
        )
        if raw_display is None or modeled is None or residual is None:
            return None
        return {
            "a": float(a),
            "b": float(b),
            "trial_index": int(chosen_array_index),
            "storage_trial_index": storage_trial_index,
            "fit_q0_trials": fit_q0_trials,
            "raw_modeled_best": np.asarray(raw_display, dtype=float),
            "modeled_best": np.asarray(modeled, dtype=float),
            "residual": np.asarray(residual, dtype=float),
            "observed": observed,
            "wcs_header": common_payload["wcs_header"],
            "psf_kernel": psf_kernel,
            "selected_slice_key": str(selected_key),
            "selected_search_id": str(selected_search_id),
        }


def hydrate_render_maps_from_grid_point(
    h5_path: Path,
    *,
    slice_key: str | None,
    search_id: str | None,
    a: float,
    b: float,
    raw_modeled_by_q0: dict[str, Any],
    modeled_by_q0: dict[str, Any],
) -> int:
    """Load committed grid-point trial maps into q0-keyed render caches."""
    if not slice_key or not str(slice_key).strip():
        return 0
    hydrated = 0
    with _H5PY_FILE(h5_path, "r") as f:
        slice_group, _descriptors, _selected_key = _resolve_slice_group(
            f,
            slice_key=str(slice_key).strip(),
            allow_missing=True,
        )
        if slice_group is None or SEARCHES_GROUP not in slice_group:
            return 0
        resolved_search_id = _selected_search_id(slice_group, requested_search_id=search_id)
        if resolved_search_id is None or resolved_search_id not in slice_group[SEARCHES_GROUP]:
            return 0
        found = find_grid_point_group(
            slice_group[SEARCHES_GROUP][resolved_search_id],
            a=float(a),
            b=float(b),
        )
        if found is None:
            return 0
        _point_id, point_group = found
        if "common" not in slice_group:
            return 0
        common_payload = _read_common_group(slice_group["common"])
        observed = np.asarray(common_payload.get("observed"), dtype=float)
        psf_kernel = common_payload.get("psf_kernel")
        for trial in _load_grid_point_trials(point_group, include_maps=False):
            q0_value = float(trial["q0"])
            key = f"{float(q0_value):.17g}"
            if key in raw_modeled_by_q0:
                continue
            raw_map_ref = str(trial.get("raw_map_ref", "") or "").strip()
            if not raw_map_ref:
                continue
            raw_modeled = _read_map_store_ref_array(f, raw_map_ref)
            if raw_modeled is None:
                continue
            raw_display, modeled, _residual, _has_raw = _derive_display_maps_from_raw(
                raw_modeled,
                observed_template=observed,
                psf_kernel=psf_kernel,
            )
            if raw_display is None or modeled is None:
                continue
            raw_modeled_by_q0[key] = np.asarray(raw_display, dtype=np.float32)
            modeled_by_q0[key] = np.asarray(modeled, dtype=np.float32)
            hydrated += 1
    return hydrated


def load_grid_point_live_state(
    h5_path: Path,
    *,
    slice_key: str,
    search_id: str | None,
    point_id: str,
) -> dict[str, Any] | None:
    with _H5PY_FILE(h5_path, "r") as f:
        if SLICE_CONTAINER_GROUP not in f or slice_key not in f[SLICE_CONTAINER_GROUP]:
            return None
        slice_group = f[SLICE_CONTAINER_GROUP][slice_key]
        if SEARCHES_GROUP not in slice_group:
            return None
        searches_group = slice_group[SEARCHES_GROUP]
        resolved_search_id = str(search_id or "").strip()
        if resolved_search_id and resolved_search_id not in searches_group:
            resolved_search_id = ""
        if not resolved_search_id:
            resolved_search_id = str(_selected_search_id(slice_group) or "").strip()
        if not resolved_search_id or resolved_search_id not in searches_group:
            return None
        search_group = searches_group[resolved_search_id]
        if GRID_POINTS_GROUP not in search_group or point_id not in search_group[GRID_POINTS_GROUP]:
            return None
        point_group = search_group[GRID_POINTS_GROUP][point_id]
        header = read_grid_point_header(point_group)
        trials = _load_grid_point_trials(point_group, include_maps=False)
        fit_q0_trials = [float(item["q0"]) for item in trials]
        fit_metric_trials, fit_chi2_trials, fit_rho2_trials, fit_eta2_trials = _assemble_trial_metric_arrays(
            trials,
            target_metric=str(header.get("metric_name", "chi2")),
        )
        active_trial_index = int(header.get("n_trials", 0))
        active_trial_q0 = header.get("next_q0")
        record_preview = grid_point_header_to_viewer_record(header, trials, include_maps=False)
        return {
            "slice_key": str(slice_key),
            "search_id": str(resolved_search_id),
            "point_id": str(point_id),
            "a": float(header["a"]),
            "b": float(header["b"]),
            "metric_name": str(header.get("metric_name", "chi2")),
            "updated_utc": str(header.get("updated_utc", "")),
            "fit_q0_trials": fit_q0_trials,
            "fit_metric_trials": list(fit_metric_trials),
            "fit_chi2_trials": list(fit_chi2_trials),
            "fit_rho2_trials": list(fit_rho2_trials),
            "fit_eta2_trials": list(fit_eta2_trials),
            "fit_shift_x_trials": list(record_preview.get("fit_shift_x_trials") or ()),
            "fit_shift_y_trials": list(record_preview.get("fit_shift_y_trials") or ()),
            "fit_find_shift_valid_trials": list(record_preview.get("fit_find_shift_valid_trials") or ()),
            "q0": None if active_trial_q0 is None else float(active_trial_q0),
            "trial_index": None if active_trial_q0 is None else active_trial_index,
        }


def apply_grid_point_event_with_retry(
    h5_path: Path,
    event: Any,
    *,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
    blos_reference: tuple[np.ndarray, fits.Header] | None = None,
    psf_kernel: np.ndarray | None = None,
    map_store_artifact: Path | None = None,
    ensure_slice_common: bool = True,
) -> str | None:
    last_exc: Exception | None = None
    for attempt in range(1, _SPARSE_APPEND_RETRY_ATTEMPTS + 1):
        try:
            context_kwargs = {
                "observed": observed,
                "sigma_map": sigma_map,
                "wcs_header": wcs_header,
                "diagnostics": diagnostics,
                "blos_reference": blos_reference,
                "psf_kernel": psf_kernel,
                "map_store_artifact": map_store_artifact,
                "ensure_slice_common": ensure_slice_common,
            }
            if isinstance(event, GridPointAssignedEvent):
                return apply_grid_point_assigned(h5_path, event, **context_kwargs)
            if isinstance(event, GridTrialCommittedEvent):
                apply_grid_trial_committed(h5_path, event, **context_kwargs)
                return None
            if isinstance(event, GridPointCompletedEvent):
                apply_grid_point_completed(h5_path, event, **context_kwargs)
                return None
            if isinstance(event, GridPointFailedEvent):
                apply_grid_point_failed(h5_path, event, **context_kwargs)
                return None
            if isinstance(event, GridPointActiveQ0Event):
                apply_grid_point_active_q0(
                    h5_path,
                    point_id=str(event.point_id),
                    next_q0=float(event.next_q0),
                    **context_kwargs,
                )
                return None
            raise TypeError(f"unsupported grid point event: {type(event)!r}")
        except (BlockingIOError, PermissionError, OSError) as exc:
            last_exc = exc
            if attempt >= _SPARSE_APPEND_RETRY_ATTEMPTS:
                break
            import time

            time.sleep(_SPARSE_APPEND_RETRY_DELAY_S)
    if last_exc is not None:
        raise OSError(f"unable to apply grid point event: {h5_path}") from last_exc
    return None
