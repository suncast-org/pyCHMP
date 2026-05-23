from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time
from datetime import datetime, timezone
from typing import Any

import h5py
import numpy as np
from astropy.io import fits


METRICS = ("chi2", "rho2", "eta2")
UNIFIED_ARTIFACT_KIND = "pychmp_ab_scan_unified"
RECTANGULAR_ARTIFACT_KIND = "pychmp_ab_scan"
SPARSE_ARTIFACT_KIND = "pychmp_ab_scan_sparse_points"
SLICE_CONTAINER_GROUP = "slices"
RUN_HISTORY_DATASET = "run_history_json"
COMMON_SLICE_DESCRIPTORS_DATASET = "slice_descriptors_json"
COMMON_TARGET_SLICE_KEY_DATASET = "target_slice_key"
COMMON_TRIAL_LOGGING_POLICY_DATASET = "trial_logging_policy_json"
COMMON_ARTIFACT_CONTRACT_VERSION_DATASET = "artifact_contract_version"
SEARCHES_GROUP = "searches"
ACTIVE_SEARCH_ID_DATASET = "active_search_id"
SEARCH_REQUEST_DATASET = "request_json"
SEARCH_LIFECYCLE_DATASET = "lifecycle_json"
MAP_STORE_GROUP = "map_store"
MAP_STORE_MAPS_GROUP = "maps"
MAP_REFS_DATASET = "map_refs_json"
CANONICAL_ARTIFACT_CONTRACT_VERSION = "2026-05-21-unified-searches"
REQUIRED_COMPATIBILITY_DIAGNOSTIC_KEYS = (
    "artifact_kind",
    "target_metric",
    "model_sha256",
    "fits_sha256",
    "ebtel_sha256",
    "frequency_ghz",
    "map_xc_arcsec",
    "map_yc_arcsec",
    "map_dx_arcsec",
    "map_dy_arcsec",
    "map_nx",
    "map_ny",
    "observer_name",
    "observer_lonc_deg",
    "observer_b0sun_deg",
    "observer_dsun_cm",
    "observer_obs_time",
)
GEOMETRY_COMPATIBILITY_DIAGNOSTIC_KEYS = (
    "map_xc_arcsec",
    "map_yc_arcsec",
    "map_dx_arcsec",
    "map_dy_arcsec",
    "map_nx",
    "map_ny",
    "observer_name",
    "observer_lonc_deg",
    "observer_b0sun_deg",
    "observer_dsun_cm",
    "observer_obs_time",
)
COMPATIBILITY_SIGNATURE_KEY = "compatibility_signature"
SEARCH_SPECIFIC_DIAGNOSTIC_KEYS = {
    COMPATIBILITY_SIGNATURE_KEY,
    "target_metric",
    "metrics_mask_threshold",
    "metrics_mask_source",
    "metrics_mask_fits",
    "mask_type",
    "tr_mask_bmin_gauss",
    "tr_mask_source",
    "search_mode",
}


class ScanArtifactCompatibilityError(ValueError):
    """Raised when an existing scan artifact cannot be safely reused."""


_SPARSE_APPEND_RETRY_ATTEMPTS = 12
_SPARSE_APPEND_RETRY_DELAY_S = 0.20
_H5PY_FILE = h5py.File


def decode_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        item = value.item()
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace")
        return str(item)
    return str(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _json_loads_or_empty(value: Any) -> dict[str, Any]:
    try:
        loaded = json.loads(decode_scalar(value))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _normalize_path_like(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    return os.path.normcase(os.path.normpath(text))


def _compute_file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_candidates_from_text(value: Any) -> list[Path]:
    text = str(value or "").strip()
    if not text:
        return []
    candidates: list[Path] = [Path(text).expanduser()]
    msys_match = re.match(r"^/([a-zA-Z])/(.*)$", text)
    if msys_match:
        drive = msys_match.group(1).upper()
        suffix = msys_match.group(2).replace("/", "\\")
        candidates.append(Path(f"{drive}:\\{suffix}"))
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _resolve_existing_file_from_diagnostics(value: Any) -> Path | None:
    for candidate in _path_candidates_from_text(value):
        try:
            if candidate.is_file():
                return candidate
        except Exception:
            continue
    return None


def _canonical_header_text(header: fits.Header) -> str:
    return header.tostring(sep="\n", endcard=True)


def _coerce_array_for_comparison(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.floating):
        return np.asarray(arr, dtype=np.float32)
    return arr


def _arrays_match_for_reuse(lhs: np.ndarray, rhs: np.ndarray) -> bool:
    lhs_arr = _coerce_array_for_comparison(lhs)
    rhs_arr = _coerce_array_for_comparison(rhs)
    if lhs_arr.shape != rhs_arr.shape:
        return False
    return bool(np.array_equal(lhs_arr, rhs_arr, equal_nan=True))


def _diagnostic_values_match(key: str, existing: Any, current: Any) -> bool:
    if key == "artifact_kind":
        existing_text = str(existing)
        current_text = str(current)
        if UNIFIED_ARTIFACT_KIND in {existing_text, current_text} and {
            existing_text,
            current_text,
        } & {RECTANGULAR_ARTIFACT_KIND, SPARSE_ARTIFACT_KIND}:
            return True
    if isinstance(existing, (int, float, np.integer, np.floating)) or isinstance(current, (int, float, np.integer, np.floating)):
        try:
            existing_value = float(existing)
            current_value = float(current)
        except Exception:
            return str(existing) == str(current)
        if np.isnan(existing_value) and np.isnan(current_value):
            return True
        return bool(np.isclose(existing_value, current_value, rtol=0.0, atol=1e-9))

    return str(existing) == str(current)


def scan_artifact_compatibility_issues(
    payload: dict[str, Any],
    *,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
) -> list[str]:
    issues: list[str] = []

    existing_observed = np.asarray(payload.get("observed"), dtype=float)
    current_observed = np.asarray(observed, dtype=float)
    if not _arrays_match_for_reuse(existing_observed, current_observed):
        issues.append(
            "observed map differs from the stored artifact "
            f"(stored shape={existing_observed.shape}, current shape={current_observed.shape})"
        )

    existing_sigma = np.asarray(payload.get("sigma_map"), dtype=float)
    current_sigma = np.asarray(sigma_map, dtype=float)
    if not _arrays_match_for_reuse(existing_sigma, current_sigma):
        issues.append(
            "sigma map differs from the stored artifact "
            f"(stored shape={existing_sigma.shape}, current shape={current_sigma.shape})"
        )

    existing_header = payload.get("wcs_header")
    if not isinstance(existing_header, fits.Header):
        issues.append("stored artifact is missing a valid WCS header")
    elif _canonical_header_text(existing_header) != _canonical_header_text(wcs_header):
        issues.append("WCS header differs from the stored artifact")

    existing_diagnostics = dict(payload.get("diagnostics", {}))
    artifact_kind = str(existing_diagnostics.get("artifact_kind", ""))
    for key in REQUIRED_COMPATIBILITY_DIAGNOSTIC_KEYS:
        if key not in existing_diagnostics:
            issues.append(f"stored diagnostics are missing required key '{key}'")
            continue
        if key not in diagnostics:
            issues.append(f"current diagnostics are missing required key '{key}'")
            continue
        if not _diagnostic_values_match(key, existing_diagnostics[key], diagnostics[key]):
            issues.append(
                f"diagnostic mismatch for '{key}' "
                f"(stored={existing_diagnostics[key]!r}, current={diagnostics[key]!r})"
            )

    # Rectangular artifacts represent a single coherent run, so require an exact
    # command-signature match when present. Sparse artifacts may intentionally
    # accumulate multiple runs and therefore filter incompatible point records
    # during hydration instead of rejecting the whole file.
    if artifact_kind not in {SPARSE_ARTIFACT_KIND, UNIFIED_ARTIFACT_KIND}:
        existing_signature = str(existing_diagnostics.get(COMPATIBILITY_SIGNATURE_KEY, "")).strip()
        current_signature = str(diagnostics.get(COMPATIBILITY_SIGNATURE_KEY, "")).strip()
        if existing_signature and current_signature and existing_signature != current_signature:
            issues.append(
                f"diagnostic mismatch for '{COMPATIBILITY_SIGNATURE_KEY}' "
                f"(stored={existing_signature!r}, current={current_signature!r})"
            )

    return issues


def validate_scan_artifact_compatibility(
    payload: dict[str, Any],
    *,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
    artifact_path: Path | None = None,
) -> None:
    issues = scan_artifact_compatibility_issues(
        payload,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=wcs_header,
        diagnostics=diagnostics,
    )
    if not issues:
        return

    artifact_label = f"existing artifact {artifact_path}" if artifact_path is not None else "existing artifact"
    raise ScanArtifactCompatibilityError(
        f"{artifact_label} is not compatible with the current run: " + "; ".join(issues)
    )


def _diagnostics_from_slice_group(group: h5py.Group) -> dict[str, Any]:
    common = group.get("common")
    if common is None or "diagnostics_json" not in common:
        return {}
    try:
        return json.loads(decode_scalar(common["diagnostics_json"][()]))
    except Exception:
        return {}


def _validate_new_slice_geometry_compatibility(
    slices_group: h5py.Group,
    *,
    diagnostics: dict[str, Any],
    slice_key: str,
    artifact_path: Path | None = None,
) -> None:
    if slice_key in slices_group:
        return
    issues: list[str] = []
    for existing_key in sorted(slices_group.keys()):
        existing_diag = _diagnostics_from_slice_group(slices_group[existing_key])
        if not existing_diag:
            continue
        for key in GEOMETRY_COMPATIBILITY_DIAGNOSTIC_KEYS:
            if key not in existing_diag or key not in diagnostics:
                continue
            if not _diagnostic_values_match(key, existing_diag[key], diagnostics[key]):
                issues.append(
                    f"new slice '{slice_key}' geometry mismatch against slice '{existing_key}' for '{key}' "
                    f"(stored={existing_diag[key]!r}, current={diagnostics[key]!r})"
                )
        if issues:
            break
    if issues:
        artifact_label = f"existing artifact {artifact_path}" if artifact_path is not None else "existing artifact"
        raise ScanArtifactCompatibilityError(
            f"{artifact_label} cannot accept new slice '{slice_key}': " + "; ".join(issues)
        )


def _sanitize_slice_token(value: str) -> str:
    text = str(value).strip().lower().replace(" ", "_")
    sanitized = []
    for char in text:
        if char.isalnum() or char in {"_", "-"}:
            sanitized.append(char)
        elif char == ".":
            sanitized.append("p")
    token = "".join(sanitized).strip("_")
    return token or "slice"


def _format_frequency_label(frequency_ghz: float) -> str:
    return f"{float(frequency_ghz):.3f} GHz"


def _format_wavelength_label(wavelength_angstrom: float) -> str:
    rounded = round(float(wavelength_angstrom))
    if np.isclose(float(wavelength_angstrom), float(rounded), rtol=0.0, atol=1e-9):
        return f"{int(rounded)} A"
    return f"{float(wavelength_angstrom):.3f} A"


def _display_label_for_slice_descriptor(descriptor: dict[str, Any]) -> str:
    domain = str(descriptor.get("domain", "")).strip().lower()
    label = str(descriptor.get("label", "")).strip() or str(descriptor.get("key", "slice"))
    if domain in {"mw", "euv", "uv"}:
        return f"{domain.upper()}: {label}"
    return label


def _normalize_slice_descriptor(raw: dict[str, Any], *, fallback_key: str) -> dict[str, Any]:
    domain = str(raw.get("domain", raw.get("spectral_domain", "generic"))).strip().lower() or "generic"
    frequency_ghz = _optional_float(raw.get("frequency_ghz", raw.get("active_frequency_ghz")))
    wavelength_angstrom = _optional_float(raw.get("wavelength_angstrom"))
    channel_label_raw = raw.get("channel_label", raw.get("euv_channel", raw.get("channel_name")))
    channel_label = None if channel_label_raw is None else str(channel_label_raw).strip() or None
    label = str(raw.get("label", raw.get("spectral_label", ""))).strip()
    key = str(raw.get("key", raw.get("slice_key", ""))).strip()
    sort_value = _optional_float(raw.get("sort_value"))

    if not label:
        if domain == "mw" and frequency_ghz is not None:
            label = _format_frequency_label(frequency_ghz)
        elif domain in {"euv", "uv"} and wavelength_angstrom is not None:
            label = _format_wavelength_label(wavelength_angstrom)
        elif channel_label:
            label = channel_label
        else:
            label = str(raw.get("slice_label", fallback_key))

    if not key:
        if domain == "mw" and frequency_ghz is not None:
            key = f"mw_{frequency_ghz:.6f}ghz".replace(".", "p")
            if sort_value is None:
                sort_value = frequency_ghz
        elif domain in {"euv", "uv"} and (channel_label or wavelength_angstrom is not None):
            token_source = channel_label or _format_wavelength_label(float(wavelength_angstrom))
            key = f"{domain}_{_sanitize_slice_token(str(token_source))}"
        else:
            key = _sanitize_slice_token(fallback_key)

    role = str(raw.get("role", "")).strip().lower()
    is_target_raw = raw.get("is_target")
    is_target = bool(is_target_raw) if is_target_raw is not None else False
    if role not in {"target", "auxiliary"}:
        role = "target" if is_target else "auxiliary"
    is_target = role == "target"

    descriptor = {
        "key": str(key),
        "domain": domain,
        "label": label,
        "display_label": str(raw.get("display_label", "")).strip(),
        "frequency_ghz": frequency_ghz,
        "wavelength_angstrom": wavelength_angstrom,
        "channel_label": channel_label,
        "sort_value": sort_value,
        "role": role,
        "is_target": bool(is_target),
    }
    if not descriptor["display_label"]:
        descriptor["display_label"] = _display_label_for_slice_descriptor(descriptor)
    return descriptor


def canonical_slice_descriptors_from_diagnostics(
    diagnostics: dict[str, Any],
    *,
    fallback_key: str = "default",
) -> tuple[list[dict[str, Any]], str]:
    raw_descriptors = diagnostics.get("slice_descriptors")
    descriptors: list[dict[str, Any]] = []

    if isinstance(raw_descriptors, list) and raw_descriptors:
        for index, item in enumerate(raw_descriptors):
            if not isinstance(item, dict):
                continue
            descriptors.append(
                _normalize_slice_descriptor(
                    item,
                    fallback_key=str(item.get("key", item.get("slice_key", f"{fallback_key}_{index}"))),
                )
            )

    if not descriptors:
        descriptors = [
            _normalize_slice_descriptor(diagnostics, fallback_key=fallback_key),
        ]

    explicit_target_key = str(diagnostics.get("target_slice_key", "")).strip()
    target_key = explicit_target_key if explicit_target_key else ""
    if not target_key:
        for descriptor in descriptors:
            if bool(descriptor.get("is_target")):
                target_key = str(descriptor["key"])
                break
    if not target_key:
        target_key = str(descriptors[0]["key"])

    normalized_descriptors: list[dict[str, Any]] = []
    for descriptor in descriptors:
        updated = dict(descriptor)
        updated["is_target"] = str(updated["key"]) == target_key
        updated["role"] = "target" if bool(updated["is_target"]) else "auxiliary"
        normalized_descriptors.append(updated)
    return normalized_descriptors, target_key


def canonical_trial_logging_policy_from_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    def _bool(key: str, default: bool) -> bool:
        value = diagnostics.get(key)
        return default if value is None else bool(value)

    return {
        "store_observed_maps": True,
        "store_raw_rendered_cubes": _bool("store_raw_rendered_cubes", False),
        "store_trial_metrics": _bool("store_trial_metrics", True),
        "store_trial_metric_masks": _bool("store_trial_metric_masks", False),
        "store_trial_explicit_metric_masks": _bool("store_trial_explicit_metric_masks", False),
        "store_psf_metadata": _bool("store_psf_metadata", True),
        "store_euv_component_cubes": _bool("store_euv_component_cubes", False),
        "store_euv_tr_mask": _bool("store_euv_tr_mask", False),
        "store_trial_convolved_cubes": _bool("store_trial_convolved_cubes", False),
        "store_trial_residual_cubes": _bool("store_trial_residual_cubes", False),
        "store_final_solution_views": _bool("store_final_solution_views", True),
    }


def slice_descriptor_from_diagnostics(diagnostics: dict[str, Any], *, fallback_key: str = "default") -> dict[str, Any]:
    descriptor = _normalize_slice_descriptor(diagnostics, fallback_key=fallback_key)
    return {
        "key": descriptor["key"],
        "domain": descriptor["domain"],
        "label": descriptor["label"],
        "display_label": descriptor["display_label"],
        "frequency_ghz": descriptor["frequency_ghz"],
        "channel_label": descriptor["channel_label"],
        "wavelength_angstrom": descriptor["wavelength_angstrom"],
        "sort_value": descriptor["sort_value"],
        "role": descriptor["role"],
        "is_target": descriptor["is_target"],
    }


def target_slice_descriptor_from_diagnostics(diagnostics: dict[str, Any], *, fallback_key: str = "default") -> dict[str, Any]:
    descriptors, target_slice_key = canonical_slice_descriptors_from_diagnostics(diagnostics, fallback_key=fallback_key)
    for descriptor in descriptors:
        if str(descriptor.get("key")) == str(target_slice_key):
            return dict(descriptor)
    return slice_descriptor_from_diagnostics(diagnostics, fallback_key=fallback_key)


def _artifact_kind_from_group(f: h5py.Group | h5py.File) -> str:
    common = f.get("common")
    if common is not None and "diagnostics_json" in common:
        try:
            diagnostics = json.loads(decode_scalar(common["diagnostics_json"][()]))
            return str(diagnostics.get("artifact_kind", RECTANGULAR_ARTIFACT_KIND))
        except Exception:
            pass
    if "point_records" in f and "summary" not in f:
        return SPARSE_ARTIFACT_KIND
    return RECTANGULAR_ARTIFACT_KIND


def _run_history_dtype() -> Any:
    return h5py.string_dtype(encoding="utf-8")


def _text_dataset_dtype() -> Any:
    return h5py.string_dtype(encoding="utf-8")


def _create_text_dataset(group: h5py.Group, name: str, text: str) -> h5py.Dataset:
    return group.create_dataset(name, data=str(text), dtype=_text_dataset_dtype())


def _replace_text_dataset(group: h5py.Group, name: str, text: str) -> h5py.Dataset:
    if name in group:
        del group[name]
    return _create_text_dataset(group, name, text)


def _ensure_run_history_dataset(common: h5py.Group) -> h5py.Dataset:
    if RUN_HISTORY_DATASET in common:
        return common[RUN_HISTORY_DATASET]
    return common.create_dataset(
        RUN_HISTORY_DATASET,
        shape=(0,),
        maxshape=(None,),
        dtype=_run_history_dtype(),
    )


def _search_id_from_diagnostics(diagnostics: dict[str, Any], *, fallback: str = "search") -> str:
    signature = str(diagnostics.get(COMPATIBILITY_SIGNATURE_KEY, "")).strip()
    if not signature:
        identity = {
            str(key): value
            for key, value in diagnostics.items()
            if key in SEARCH_SPECIFIC_DIAGNOSTIC_KEYS
            or str(key).startswith("metrics_")
            or str(key).startswith("tr_mask_")
        }
        if not identity:
            identity = {"target_metric": diagnostics.get("target_metric", "chi2")}
        signature = hashlib.sha256(_json_dumps(identity).encode("utf-8")).hexdigest()
    return f"{fallback}_{signature[:16]}"


def _search_status_from_records(point_records: list[dict[str, Any]]) -> str:
    if not point_records:
        return "empty"
    statuses = {_normalize_point_status(record.get("status", "computed")) for record in point_records}
    return _search_status_from_statuses(statuses)


def _normalize_point_status(status: Any) -> str:
    text = str(status if status is not None else "computed").strip().lower()
    return text or "computed"


def _search_status_from_statuses(statuses: set[str]) -> str:
    if not statuses:
        return "empty"
    if any(status in {"pending", "missing"} for status in statuses):
        return "in_progress"
    if all(status == "failed" for status in statuses):
        return "failed"
    if any(status == "failed" for status in statuses):
        return "partial"
    return "complete"


def _search_status_counts_from_records(point_records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"total": 0, "pending": 0, "missing": 0, "failed": 0, "computed": 0, "other": 0}
    for record in point_records:
        status = _normalize_point_status(record.get("status", "computed"))
        counts["total"] += 1
        if status in {"pending", "missing", "failed", "computed"}:
            counts[status] += 1
        else:
            counts["other"] += 1
    return counts


def _search_status_from_counts(counts: dict[str, int]) -> str:
    total = int(counts.get("total", 0))
    if total <= 0:
        return "empty"
    if int(counts.get("pending", 0)) > 0 or int(counts.get("missing", 0)) > 0:
        return "in_progress"
    failed = int(counts.get("failed", 0))
    if failed >= total:
        return "failed"
    if failed > 0:
        return "partial"
    return "complete"


def _write_search_status_attrs(search_group: h5py.Group, counts: dict[str, int], *, diagnostics: dict[str, Any]) -> str:
    status = _search_status_from_counts(counts)
    for key, value in counts.items():
        search_group.attrs[f"{key}_point_count"] = int(value)
    search_group.attrs["status"] = np.bytes_(status)
    search_group.attrs["label"] = np.bytes_(_search_label_from_diagnostics(diagnostics, status=status))
    return status


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _first_present(diagnostics: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in diagnostics and diagnostics[key] not in {None, ""}:
            return diagnostics[key]
    return None


def _search_request_from_diagnostics(
    diagnostics: dict[str, Any],
    *,
    layout: dict[str, Any] | None,
) -> dict[str, Any]:
    layout_payload = dict(layout or {})
    request: dict[str, Any] = {
        "target_metric": str(diagnostics.get("target_metric", "chi2")),
        "layout": layout_payload,
        "metrics_mask": {
            "source": diagnostics.get("metrics_mask_source"),
            "threshold": diagnostics.get("metrics_mask_threshold", diagnostics.get("threshold")),
            "fits": diagnostics.get("metrics_mask_fits"),
            "mask_type": diagnostics.get("mask_type"),
        },
        "tr_mask": {
            "source": diagnostics.get("tr_mask_source"),
            "bmin_gauss": diagnostics.get("tr_mask_bmin_gauss"),
        },
        "optimizer": {
            "q0_min": diagnostics.get("q0_min"),
            "q0_max": diagnostics.get("q0_max"),
            "hard_q0_min": diagnostics.get("hard_q0_min"),
            "hard_q0_max": diagnostics.get("hard_q0_max"),
            "q0_start": diagnostics.get("q0_start"),
            "q0_step": diagnostics.get("q0_step"),
            "adaptive_bracketing": diagnostics.get("adaptive_bracketing", diagnostics.get("used_adaptive_bracketing")),
            "max_bracket_steps": diagnostics.get("max_bracket_steps"),
            "threshold_metric": diagnostics.get("threshold_metric"),
            "no_area": diagnostics.get("no_area"),
        },
        "execution": {
            "policy": _first_present(diagnostics, ("execution_policy", "execution_policy_resolved")),
            "requested_policy": diagnostics.get("execution_policy_requested"),
            "max_workers": diagnostics.get("execution_max_workers"),
        },
    }
    if "requested_points" in diagnostics:
        request["requested_points"] = diagnostics["requested_points"]
    elif layout_payload.get("kind") == "rectangular_grid":
        a_values = [float(v) for v in layout_payload.get("a_values", [])]
        b_values = [float(v) for v in layout_payload.get("b_values", [])]
        request["requested_points"] = [{"a": a, "b": b} for a in a_values for b in b_values]
    return request


def _search_lifecycle_payload(
    *,
    status: str,
    diagnostics: dict[str, Any],
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    previous = dict(existing or {})
    created_at = str(
        diagnostics.get("search_created_at")
        or previous.get("created_at")
        or _utc_now_iso()
    )
    started_at = str(
        diagnostics.get("search_started_at")
        or previous.get("started_at")
        or created_at
    )
    completed_at = diagnostics.get("search_completed_at", previous.get("completed_at"))
    in_progress = status in {"empty", "in_progress", "partial"}
    if not in_progress and not completed_at:
        completed_at = _utc_now_iso()
    return {
        "status": str(status),
        "active": bool(diagnostics.get("search_active", True)),
        "in_progress": bool(in_progress),
        "created_at": created_at,
        "started_at": started_at,
        "completed_at": None if completed_at in {"", None} else str(completed_at),
    }


def _read_search_lifecycle(search_group: h5py.Group, *, status: str) -> dict[str, Any]:
    if SEARCH_LIFECYCLE_DATASET in search_group:
        payload = _json_loads_or_empty(search_group[SEARCH_LIFECYCLE_DATASET][()])
    else:
        payload = {}
    if "status" not in payload:
        payload["status"] = str(status)
    if "active" not in payload:
        payload["active"] = bool(search_group.attrs.get("active", True))
    if "in_progress" not in payload:
        payload["in_progress"] = str(payload.get("status", status)) in {"empty", "in_progress", "partial"}
    return payload


def _write_search_lifecycle_dataset(
    search_group: h5py.Group,
    *,
    lifecycle: dict[str, Any],
) -> None:
    _replace_text_dataset(search_group, SEARCH_LIFECYCLE_DATASET, _json_dumps(lifecycle))
    search_group.attrs["active"] = int(bool(lifecycle.get("active", True)))
    search_group.attrs["in_progress"] = int(bool(lifecycle.get("in_progress", False)))
    for attr_key, payload_key in (
        ("created_at", "created_at"),
        ("started_at", "started_at"),
        ("completed_at", "completed_at"),
    ):
        value = lifecycle.get(payload_key)
        if value not in {None, ""}:
            search_group.attrs[attr_key] = np.bytes_(str(value))


def _search_label_from_diagnostics(diagnostics: dict[str, Any], *, status: str) -> str:
    metric = str(diagnostics.get("target_metric", "chi2"))
    source = str(diagnostics.get("metrics_mask_source", "")).strip().lower()
    if source == "explicit_fits":
        mask_path = str(diagnostics.get("metrics_mask_fits", "")).strip()
        mask_text = f"mask={Path(mask_path).name}" if mask_path else "mask=explicit FITS"
    else:
        try:
            mask_text = f"threshold={float(diagnostics.get('metrics_mask_threshold')):.3f}"
        except Exception:
            mask_text = "threshold=n/a"
    return f"{metric} {mask_text} [{status}]"


def _write_search_group(
    searches_group: h5py.Group,
    *,
    search_id: str,
    diagnostics: dict[str, Any],
    point_records: list[dict[str, Any]],
    run_history: list[dict[str, Any]] | None,
    layout: dict[str, Any] | None = None,
) -> None:
    if search_id in searches_group:
        del searches_group[search_id]
    search_group = searches_group.create_group(search_id)
    counts = _search_status_counts_from_records(point_records)
    status = _search_status_from_counts(counts)
    search_group.attrs["search_id"] = np.bytes_(str(search_id))
    search_group.attrs["target_metric"] = np.bytes_(str(diagnostics.get("target_metric", "chi2")))
    _write_search_status_attrs(search_group, counts, diagnostics=diagnostics)
    _create_text_dataset(search_group, "diagnostics_json", _json_dumps(diagnostics))
    _create_text_dataset(search_group, "layout_json", _json_dumps(layout or {}))
    _create_text_dataset(search_group, "run_history_json", _json_dumps(list(run_history or [])))
    records_group = search_group.create_group("point_records")
    for record_order, payload in enumerate(point_records):
        grp = records_group.create_group(f"r{record_order:06d}")
        _write_point_group(grp, payload, record_order=record_order)
    request = _search_request_from_diagnostics(diagnostics, layout=layout)
    lifecycle = _search_lifecycle_payload(status=status, diagnostics=diagnostics)
    _create_text_dataset(search_group, SEARCH_REQUEST_DATASET, _json_dumps(request))
    _write_search_lifecycle_dataset(search_group, lifecycle=lifecycle)


def _read_search_records(group: h5py.Group) -> list[dict[str, Any]]:
    if SEARCHES_GROUP not in group:
        return []
    searches_group = group[SEARCHES_GROUP]
    records: list[dict[str, Any]] = []
    for name in sorted(searches_group.keys()):
        search_group = searches_group[name]
        diagnostics = _json_loads_or_empty(search_group["diagnostics_json"][()]) if "diagnostics_json" in search_group else {}
        layout = _json_loads_or_empty(search_group["layout_json"][()]) if "layout_json" in search_group else {}
        run_history: list[dict[str, Any]] = []
        if "run_history_json" in search_group:
            try:
                loaded_history = json.loads(decode_scalar(search_group["run_history_json"][()]))
                if isinstance(loaded_history, list):
                    run_history = loaded_history
            except Exception:
                run_history = []
        point_count = len(search_group["point_records"]) if "point_records" in search_group else 0
        status = decode_scalar(search_group.attrs.get("status", "unknown"))
        request = _json_loads_or_empty(search_group[SEARCH_REQUEST_DATASET][()]) if SEARCH_REQUEST_DATASET in search_group else _search_request_from_diagnostics(diagnostics, layout=layout)
        lifecycle = _read_search_lifecycle(search_group, status=status)
        records.append(
            {
                "search_id": decode_scalar(search_group.attrs.get("search_id", name)),
                "label": decode_scalar(search_group.attrs.get("label", name)),
                "status": status,
                "active": bool(lifecycle.get("active", False)),
                "in_progress": bool(lifecycle.get("in_progress", False)),
                "created_at": lifecycle.get("created_at"),
                "started_at": lifecycle.get("started_at"),
                "completed_at": lifecycle.get("completed_at"),
                "target_metric": decode_scalar(search_group.attrs.get("target_metric", diagnostics.get("target_metric", "chi2"))),
                "diagnostics": diagnostics,
                "layout": layout,
                "request": request,
                "lifecycle": lifecycle,
                "run_history": run_history,
                "point_count": int(point_count),
            }
        )
    return records


def _selected_search_id(group: h5py.Group, requested_search_id: str | None = None) -> str | None:
    if SEARCHES_GROUP not in group:
        return None
    searches = group[SEARCHES_GROUP]
    if requested_search_id and str(requested_search_id) in searches:
        return str(requested_search_id)
    active = ""
    if ACTIVE_SEARCH_ID_DATASET in group:
        active = decode_scalar(group[ACTIVE_SEARCH_ID_DATASET][()]).strip()
    if active and active in searches:
        return active
    names = sorted(searches.keys())
    return names[-1] if names else None


def _decode_run_history(common: h5py.Group) -> list[dict[str, Any]]:
    dataset = common.get(RUN_HISTORY_DATASET)
    if dataset is None:
        return []
    entries: list[dict[str, Any]] = []
    for raw in dataset[()]:
        try:
            entries.append(json.loads(decode_scalar(raw)))
        except Exception:
            entries.append({"raw": decode_scalar(raw)})
    return entries


def append_run_history_entry(h5_path: Path, entry: dict[str, Any], *, slice_key: str | None = None) -> None:
    serialized = _json_dumps(entry)
    with _H5PY_FILE(h5_path, "a") as f:
        group, _descriptors, selected_key = _resolve_slice_group(
            f,
            slice_key=slice_key,
            allow_missing=slice_key is not None,
        )
        if group is None:
            raise KeyError(f"slice not found: {slice_key or selected_key}")
        common = group["common"]
        dataset = _ensure_run_history_dataset(common)
        next_index = int(dataset.shape[0])
        dataset.resize((next_index + 1,))
        dataset[next_index] = serialized


def _sorted_slice_descriptors(descriptors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        descriptors,
        key=lambda item: (
            item.get("domain", ""),
            item.get("sort_value") if item.get("sort_value") is not None else float("inf"),
            str(item.get("label", "")),
            str(item.get("key", "")),
        ),
    )


def _resolve_slice_group(
    f: h5py.File,
    *,
    slice_key: str | None = None,
    allow_missing: bool = False,
) -> tuple[h5py.Group | h5py.File | None, list[dict[str, Any]], str | None]:
    if SLICE_CONTAINER_GROUP not in f:
        diagnostics = {}
        common = f.get("common")
        if common is not None and "diagnostics_json" in common:
            diagnostics = json.loads(decode_scalar(common["diagnostics_json"][()]))
        descriptor = slice_descriptor_from_diagnostics(diagnostics, fallback_key="default")
        return f, [descriptor], descriptor["key"]

    slices_group = f[SLICE_CONTAINER_GROUP]
    descriptors: list[dict[str, Any]] = []
    for name in sorted(slices_group.keys()):
        grp = slices_group[name]
        diagnostics = {}
        common = grp.get("common")
        if common is not None and "diagnostics_json" in common:
            diagnostics = json.loads(decode_scalar(common["diagnostics_json"][()]))
        descriptor = slice_descriptor_from_diagnostics(diagnostics, fallback_key=name)
        descriptor["key"] = str(name)
        descriptors.append(descriptor)

    descriptors = _sorted_slice_descriptors(descriptors)
    key_lookup = {str(item["key"]): item for item in descriptors}
    selected_key = slice_key if slice_key in key_lookup else None
    if selected_key is None:
        if allow_missing and slice_key is not None:
            return None, descriptors, None
        selected_key = str(descriptors[0]["key"]) if descriptors else None
    if selected_key is None:
        return None, descriptors, None
    return slices_group[selected_key], descriptors, selected_key


def list_scan_slices(h5_path: Path) -> list[dict[str, Any]]:
    with _H5PY_FILE(h5_path, "r") as f:
        _group, descriptors, _selected_key = _resolve_slice_group(f)
    return descriptors


def detect_scan_artifact_format(h5_path: Path, *, slice_key: str | None = None) -> str | None:
    with _H5PY_FILE(h5_path, "r") as f:
        group, _descriptors, _selected_key = _resolve_slice_group(f, slice_key=slice_key, allow_missing=slice_key is not None)
        if group is None:
            return None
        kind = _artifact_kind_from_group(group)
    if kind == UNIFIED_ARTIFACT_KIND:
        return "unified"
    return "sparse" if kind == SPARSE_ARTIFACT_KIND else "rectangular"


def _axis_edges(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("Axis values must be a non-empty 1D array.")
    if arr.size == 1:
        delta = 0.5
        return np.asarray([arr[0] - delta, arr[0] + delta], dtype=float)
    mids = 0.5 * (arr[:-1] + arr[1:])
    first_delta = mids[0] - arr[0]
    last_delta = arr[-1] - mids[-1]
    return np.concatenate(
        [
            np.asarray([arr[0] - first_delta], dtype=float),
            mids,
            np.asarray([arr[-1] + last_delta], dtype=float),
        ]
    )


def _axis_spans(values: np.ndarray) -> dict[float, tuple[float, float]]:
    arr = np.asarray(values, dtype=float)
    edges = _axis_edges(arr)
    return {float(value): (float(edges[i]), float(edges[i + 1])) for i, value in enumerate(arr)}


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
        "nfev": -1,
        "nit": -1,
        "used_adaptive_bracketing": False,
        "bracket_found": False,
        "bracket": None,
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
        "nfev": -1,
        "nit": -1,
        "message": str(message),
        "used_adaptive_bracketing": False,
        "bracket_found": False,
        "bracket": None,
        "target_metric": str(target_metric),
        "diagnostics": diagnostics,
    }


def build_computed_point_payload(
    *,
    a_value: float,
    b_value: float,
    q0: float,
    success: bool,
    status: str,
    modeled_best: np.ndarray,
    raw_modeled_best: np.ndarray,
    residual: np.ndarray,
    fit_q0_trials: tuple[float, ...],
    fit_metric_trials: tuple[float, ...],
    fit_chi2_trials: tuple[float, ...],
    fit_rho2_trials: tuple[float, ...],
    fit_eta2_trials: tuple[float, ...],
    trial_raw_modeled_maps: np.ndarray | None = None,
    trial_modeled_maps: np.ndarray | None = None,
    trial_residual_maps: np.ndarray | None = None,
    euv_coronal_best: np.ndarray | None = None,
    euv_tr_best: np.ndarray | None = None,
    euv_tr_mask: np.ndarray | None = None,
    trial_euv_coronal_maps: np.ndarray | None = None,
    trial_euv_tr_maps: np.ndarray | None = None,
    map_store_arrays: dict[str, np.ndarray] | None = None,
    nfev: int,
    nit: int,
    message: str,
    used_adaptive_bracketing: bool,
    bracket_found: bool,
    bracket: tuple[float, float, float] | None,
    target_metric: str,
    diagnostics: dict[str, Any],
    a_index: int | None = None,
    b_index: int | None = None,
) -> dict[str, Any]:
    payload = {
        "a": float(a_value),
        "b": float(b_value),
        "q0": float(q0),
        "success": bool(success),
        "status": str(status),
        "modeled_best": np.asarray(modeled_best, dtype=float),
        "raw_modeled_best": np.asarray(raw_modeled_best, dtype=float),
        "residual": np.asarray(residual, dtype=float),
        "fit_q0_trials": tuple(float(v) for v in fit_q0_trials),
        "fit_metric_trials": tuple(float(v) for v in fit_metric_trials),
        "fit_chi2_trials": tuple(float(v) for v in fit_chi2_trials),
        "fit_rho2_trials": tuple(float(v) for v in fit_rho2_trials),
        "fit_eta2_trials": tuple(float(v) for v in fit_eta2_trials),
        "trial_raw_modeled_maps": None if trial_raw_modeled_maps is None else np.asarray(trial_raw_modeled_maps, dtype=float),
        "trial_modeled_maps": None if trial_modeled_maps is None else np.asarray(trial_modeled_maps, dtype=float),
        "trial_residual_maps": None if trial_residual_maps is None else np.asarray(trial_residual_maps, dtype=float),
        "euv_coronal_best": None if euv_coronal_best is None else np.asarray(euv_coronal_best, dtype=float),
        "euv_tr_best": None if euv_tr_best is None else np.asarray(euv_tr_best, dtype=float),
        "euv_tr_mask": None if euv_tr_mask is None else np.asarray(euv_tr_mask, dtype=bool),
        "trial_euv_coronal_maps": None if trial_euv_coronal_maps is None else np.asarray(trial_euv_coronal_maps, dtype=float),
        "trial_euv_tr_maps": None if trial_euv_tr_maps is None else np.asarray(trial_euv_tr_maps, dtype=float),
        "map_store_arrays": {
            str(key): np.asarray(value, dtype=float)
            for key, value in dict(map_store_arrays or {}).items()
        },
        "nfev": int(nfev),
        "nit": int(nit),
        "message": str(message),
        "used_adaptive_bracketing": bool(used_adaptive_bracketing),
        "bracket_found": bool(bracket_found),
        "bracket": None if bracket is None else tuple(float(v) for v in bracket),
        "target_metric": str(target_metric),
        "diagnostics": dict(diagnostics),
    }
    if a_index is not None:
        payload["a_index"] = int(a_index)
    if b_index is not None:
        payload["b_index"] = int(b_index)
    return payload


def _normalize_point_payload(payload: dict[str, Any], *, record_order: int) -> dict[str, Any]:
    diagnostics = dict(payload.get("diagnostics", {}))
    target_metric = str(payload.get("target_metric", diagnostics.get("target_metric", "chi2")))
    return {
        "record_order": int(record_order),
        "a": float(payload["a"]),
        "b": float(payload["b"]),
        "q0": float(payload.get("q0", np.nan)),
        "success": bool(payload.get("success", False)),
        "status": str(payload.get("status", "computed")),
        "modeled_best": np.asarray(payload["modeled_best"], dtype=float),
        "raw_modeled_best": np.asarray(payload["raw_modeled_best"], dtype=float),
        "residual": np.asarray(payload["residual"], dtype=float),
        "fit_q0_trials": tuple(float(v) for v in payload.get("fit_q0_trials", ())),
        "fit_metric_trials": tuple(float(v) for v in payload.get("fit_metric_trials", ())),
        "fit_chi2_trials": tuple(float(v) for v in payload.get("fit_chi2_trials", ())),
        "fit_rho2_trials": tuple(float(v) for v in payload.get("fit_rho2_trials", ())),
        "fit_eta2_trials": tuple(float(v) for v in payload.get("fit_eta2_trials", ())),
        "trial_raw_modeled_maps": (
            None if payload.get("trial_raw_modeled_maps") is None else np.asarray(payload["trial_raw_modeled_maps"], dtype=float)
        ),
        "trial_modeled_maps": (
            None if payload.get("trial_modeled_maps") is None else np.asarray(payload["trial_modeled_maps"], dtype=float)
        ),
        "trial_residual_maps": (
            None if payload.get("trial_residual_maps") is None else np.asarray(payload["trial_residual_maps"], dtype=float)
        ),
        "euv_coronal_best": (
            None if payload.get("euv_coronal_best") is None else np.asarray(payload["euv_coronal_best"], dtype=float)
        ),
        "euv_tr_best": (
            None if payload.get("euv_tr_best") is None else np.asarray(payload["euv_tr_best"], dtype=float)
        ),
        "euv_tr_mask": (
            None if payload.get("euv_tr_mask") is None else np.asarray(payload["euv_tr_mask"], dtype=bool)
        ),
        "trial_euv_coronal_maps": (
            None if payload.get("trial_euv_coronal_maps") is None else np.asarray(payload["trial_euv_coronal_maps"], dtype=float)
        ),
        "trial_euv_tr_maps": (
            None if payload.get("trial_euv_tr_maps") is None else np.asarray(payload["trial_euv_tr_maps"], dtype=float)
        ),
        "map_store_arrays": {
            str(key): np.asarray(value, dtype=float)
            for key, value in dict(payload.get("map_store_arrays") or {}).items()
        },
        "nfev": int(payload.get("nfev", -1)),
        "nit": int(payload.get("nit", -1)),
        "message": str(payload.get("message", "")),
        "used_adaptive_bracketing": bool(payload.get("used_adaptive_bracketing", False)),
        "bracket_found": bool(payload.get("bracket_found", False)),
        "bracket": payload.get("bracket", None),
        "target_metric": target_metric,
        "diagnostics": diagnostics,
    }


def _read_point_group_rectangular(grp: h5py.Group) -> dict[str, Any]:
    target_metric = decode_scalar(
        grp["fit_metric_trials"].attrs["target_metric"]
        if "fit_metric_trials" in grp and "target_metric" in grp["fit_metric_trials"].attrs
        else grp.attrs.get("target_metric", b"chi2")
    )
    fit_metric_trials = np.asarray(grp["fit_metric_trials"], dtype=float) if "fit_metric_trials" in grp else np.asarray([], dtype=float)
    fit_chi2_trials = (
        np.asarray(grp["fit_chi2_trials"], dtype=float)
        if "fit_chi2_trials" in grp
        else (fit_metric_trials if target_metric == "chi2" else np.asarray([], dtype=float))
    )
    fit_rho2_trials = (
        np.asarray(grp["fit_rho2_trials"], dtype=float)
        if "fit_rho2_trials" in grp
        else (fit_metric_trials if target_metric == "rho2" else np.asarray([], dtype=float))
    )
    fit_eta2_trials = (
        np.asarray(grp["fit_eta2_trials"], dtype=float)
        if "fit_eta2_trials" in grp
        else (fit_metric_trials if target_metric == "eta2" else np.asarray([], dtype=float))
    )
    bracket = None
    if "bracket" in grp:
        try:
            bracket_arr = np.asarray(grp["bracket"], dtype=float)
            if bracket_arr.size == 3:
                bracket = tuple(float(v) for v in bracket_arr)
        except Exception:
            bracket = None
    map_refs = _json_loads_or_empty(grp[MAP_REFS_DATASET][()]) if MAP_REFS_DATASET in grp else {}
    return {
        "record_order": int(grp.attrs.get("record_order", 0)),
        "a": float(grp.attrs["a"]),
        "b": float(grp.attrs["b"]),
        "q0": float(grp.attrs["q0"]),
        "success": bool(grp.attrs["success"]),
        "status": decode_scalar(grp.attrs.get("status", b"computed")),
        "modeled_best": np.asarray(_read_point_map_array(grp, "modeled_best", map_refs), dtype=float),
        "raw_modeled_best": np.asarray(_read_point_map_array(grp, "raw_modeled_best", map_refs), dtype=float),
        "residual": np.asarray(_read_point_map_array(grp, "residual", map_refs), dtype=float),
        "fit_q0_trials": tuple(float(v) for v in np.asarray(grp["fit_q0_trials"], dtype=float)),
        "fit_metric_trials": tuple(float(v) for v in fit_metric_trials),
        "fit_chi2_trials": tuple(float(v) for v in fit_chi2_trials),
        "fit_rho2_trials": tuple(float(v) for v in fit_rho2_trials),
        "fit_eta2_trials": tuple(float(v) for v in fit_eta2_trials),
        "trial_raw_modeled_maps": _read_point_map_array(grp, "trial_raw_modeled_maps", map_refs),
        "trial_modeled_maps": _read_point_map_array(grp, "trial_modeled_maps", map_refs),
        "trial_residual_maps": _read_point_map_array(grp, "trial_residual_maps", map_refs),
        "euv_coronal_best": _read_point_map_array(grp, "euv_coronal_best", map_refs),
        "euv_tr_best": _read_point_map_array(grp, "euv_tr_best", map_refs),
        "euv_tr_mask": (
            np.asarray(grp["euv_tr_mask"], dtype=bool) if "euv_tr_mask" in grp else None
        ),
        "trial_euv_coronal_maps": _read_point_map_array(grp, "trial_euv_coronal_maps", map_refs),
        "trial_euv_tr_maps": _read_point_map_array(grp, "trial_euv_tr_maps", map_refs),
        "nfev": int(grp.attrs.get("nfev", -1)),
        "nit": int(grp.attrs.get("nit", -1)),
        "message": decode_scalar(grp.attrs.get("message", b"")),
        "used_adaptive_bracketing": bool(grp.attrs.get("used_adaptive_bracketing", False)),
        "bracket_found": bool(grp.attrs.get("bracket_found", False)),
        "bracket": bracket,
        "target_metric": target_metric,
        "map_refs": map_refs,
        "diagnostics": json.loads(decode_scalar(grp["diagnostics_json"][()])),
    }


def _read_point_group_sparse(grp: h5py.Group) -> dict[str, Any]:
    return _read_point_group_rectangular(grp)


def _load_sparse_point_records(records_group: h5py.Group) -> list[dict[str, Any]]:
    latest_by_coord: dict[tuple[float, float], dict[str, Any]] = {}
    for name in sorted(records_group.keys()):
        record = _read_point_group_sparse(records_group[name])
        coord = (float(record["a"]), float(record["b"]))
        existing = latest_by_coord.get(coord)
        if existing is None or int(record["record_order"]) >= int(existing["record_order"]):
            latest_by_coord[coord] = record
    return sorted(latest_by_coord.values(), key=lambda item: (float(item["a"]), float(item["b"])))


def _load_rectangular_point_records(points_group: h5py.Group) -> list[dict[str, Any]]:
    records = [_read_point_group_rectangular(points_group[name]) for name in sorted(points_group.keys())]
    return sorted(records, key=lambda item: (float(item["a"]), float(item["b"])))


def _load_canonical_point_records(group: h5py.Group) -> list[dict[str, Any]]:
    if "point_records" in group:
        return _load_sparse_point_records(group["point_records"])
    if "points" in group:
        return _load_rectangular_point_records(group["points"])
    return []


def _payload_from_point_records(
    *,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
    point_records: list[dict[str, Any]],
    target_metric: str,
) -> dict[str, Any]:
    unique_a = np.asarray(sorted({float(record["a"]) for record in point_records}), dtype=float)
    unique_b = np.asarray(sorted({float(record["b"]) for record in point_records}), dtype=float)
    if unique_a.size == 0:
        unique_a = np.asarray([], dtype=float)
    if unique_b.size == 0:
        unique_b = np.asarray([], dtype=float)

    shape = (unique_a.size, unique_b.size)
    best_q0 = np.full(shape, np.nan, dtype=float)
    objective_values = np.full(shape, np.nan, dtype=float)
    chi2 = np.full(shape, np.nan, dtype=float)
    rho2 = np.full(shape, np.nan, dtype=float)
    eta2 = np.full(shape, np.nan, dtype=float)
    success = np.zeros(shape, dtype=bool)
    points: dict[tuple[int, int], dict[str, Any]] = {}

    target_metric_name = str(target_metric or diagnostics.get("target_metric", "chi2"))
    for i, a_value in enumerate(unique_a):
        for j, b_value in enumerate(unique_b):
            points[(int(i), int(j))] = _pending_point_payload(
                a_value=float(a_value),
                b_value=float(b_value),
                a_index=int(i),
                b_index=int(j),
                observed_template=observed,
                target_metric=target_metric_name,
                status="missing",
                message="point not stored in artifact",
            )

    a_lookup = {float(v): int(i) for i, v in enumerate(unique_a)}
    b_lookup = {float(v): int(i) for i, v in enumerate(unique_b)}
    normalized_records: list[dict[str, Any]] = []
    for record in point_records:
        a_value = float(record["a"])
        b_value = float(record["b"])
        a_index = a_lookup[a_value]
        b_index = b_lookup[b_value]
        diagnostics_json = dict(record.get("diagnostics", {}))
        metrics = {
            "chi2": float(diagnostics_json.get("chi2", np.nan)),
            "rho2": float(diagnostics_json.get("rho2", np.nan)),
            "eta2": float(diagnostics_json.get("eta2", np.nan)),
        }
        points[(a_index, b_index)] = {
            **record,
            "a_index": int(a_index),
            "b_index": int(b_index),
        }
        best_q0[a_index, b_index] = float(record.get("q0", np.nan))
        objective_values[a_index, b_index] = float(diagnostics_json.get("target_metric_value", np.nan))
        chi2[a_index, b_index] = metrics["chi2"]
        rho2[a_index, b_index] = metrics["rho2"]
        eta2[a_index, b_index] = metrics["eta2"]
        success[a_index, b_index] = bool(record.get("success", False))
        normalized_records.append(
            {
                **record,
                "a_index": int(a_index),
                "b_index": int(b_index),
                "metrics": metrics,
            }
        )

    return {
        "observed": np.asarray(observed, dtype=float),
        "sigma_map": np.asarray(sigma_map, dtype=float),
        "wcs_header": wcs_header,
        "diagnostics": diagnostics,
        "a_values": unique_a,
        "b_values": unique_b,
        "best_q0": best_q0,
        "objective_values": objective_values,
        "chi2": chi2,
        "rho2": rho2,
        "eta2": eta2,
        "success": success,
        "target_metric": target_metric_name,
        "points": points,
        "point_records": normalized_records,
        "artifact_format": (
            "unified"
            if diagnostics.get("artifact_kind") == UNIFIED_ARTIFACT_KIND
            else ("sparse" if diagnostics.get("artifact_kind") == SPARSE_ARTIFACT_KIND else "rectangular")
        ),
    }


def load_scan_file(h5_path: Path, *, slice_key: str | None = None, search_id: str | None = None) -> dict[str, Any]:
    with _H5PY_FILE(h5_path, "r") as f:
        group, descriptors, selected_key = _resolve_slice_group(
            f,
            slice_key=slice_key,
            allow_missing=slice_key is not None,
        )
        if group is None:
            raise KeyError(f"slice not found: {slice_key}")
        common = group["common"]
        common_payload = _read_common_group(common)
        wcs_header = common_payload["wcs_header"]
        diagnostics = common_payload["diagnostics"]
        run_history = common_payload["run_history"]
        # Non-breaking: if mask_type is missing, assume 'union'
        if "mask_type" not in diagnostics:
            diagnostics["mask_type"] = "union"
        search_records = _read_search_records(group)
        selected_search_id = _selected_search_id(group, requested_search_id=search_id)
        selected_search_record = next(
            (record for record in search_records if str(record.get("search_id")) == str(selected_search_id)),
            None,
        )
        if selected_search_id is not None and SEARCHES_GROUP in group:
            search_group = group[SEARCHES_GROUP][selected_search_id]
            point_records = _load_sparse_point_records(search_group["point_records"]) if "point_records" in search_group else []
            search_diagnostics = dict(selected_search_record.get("diagnostics", {}) if selected_search_record else {})
            search_specific_diagnostics = {
                key: value
                for key, value in search_diagnostics.items()
                if key in SEARCH_SPECIFIC_DIAGNOSTIC_KEYS or str(key).startswith("metrics_") or str(key).startswith("tr_mask_")
            }
            diagnostics = {**diagnostics, **search_specific_diagnostics}
            target_metric = str(search_diagnostics.get("target_metric", diagnostics.get("target_metric", "chi2")))
            run_history = list(selected_search_record.get("run_history", run_history) if selected_search_record else run_history)
            kind = str(diagnostics.get("artifact_kind", _artifact_kind_from_group(group)))
        else:
            kind = _artifact_kind_from_group(group)
            if "point_records" in group:
                point_records = _load_sparse_point_records(group["point_records"])
                target_metric = str(diagnostics.get("target_metric", "chi2"))
            else:
                point_records = _load_canonical_point_records(group)
                if "summary" in group:
                    summary = group["summary"]
                    target_metric = decode_scalar(summary.attrs.get("target_metric", diagnostics.get("target_metric", b"chi2")))
                else:
                    target_metric = str(diagnostics.get("target_metric", "chi2"))
        payload = _payload_from_point_records(
            observed=np.asarray(common["observed"], dtype=float),
            sigma_map=np.asarray(common["sigma_map"], dtype=float),
            wcs_header=wcs_header,
            diagnostics=diagnostics,
            point_records=point_records,
            target_metric=target_metric,
        )
        selected_descriptor = next((item for item in descriptors if str(item["key"]) == str(selected_key)), None)
        payload["available_slices"] = descriptors
        payload["selected_slice_key"] = selected_key
        payload["selected_slice"] = selected_descriptor
        payload["run_history"] = run_history
        payload["search_records"] = search_records
        payload["selected_search_id"] = selected_search_id
        payload["selected_search"] = selected_search_record
        if not search_records:
            legacy_status = _search_status_from_records(payload.get("point_records", []))
            legacy_search = {
                "search_id": "legacy_current",
                "label": _search_label_from_diagnostics(diagnostics, status=legacy_status),
                "status": legacy_status,
                "target_metric": str(target_metric),
                "diagnostics": dict(diagnostics),
                "layout": {},
                "run_history": list(run_history),
                "point_count": int(len(payload.get("point_records", []))),
            }
            payload["search_records"] = [legacy_search]
            payload["selected_search_id"] = "legacy_current"
            payload["selected_search"] = legacy_search
        payload["artifact_contract_version"] = common_payload.get("artifact_contract_version")
        payload["canonical_slice_descriptors"] = common_payload.get("slice_descriptors", [])
        payload["target_slice_key"] = common_payload.get("target_slice_key")
        payload["trial_logging_policy"] = common_payload.get("trial_logging_policy", {})
        payload["blos_reference"] = common_payload.get("blos_reference")
        return payload


def load_run_history(h5_path: Path, *, slice_key: str | None = None) -> list[dict[str, Any]]:
    with _H5PY_FILE(h5_path, "r") as f:
        group, _descriptors, selected_key = _resolve_slice_group(
            f,
            slice_key=slice_key,
            allow_missing=slice_key is not None,
        )
        if group is None:
            raise KeyError(f"slice not found: {slice_key or selected_key}")
        common = group["common"]
        return _decode_run_history(common)


def backfill_artifact_diagnostics(
    h5_path: Path,
    *,
    slice_key: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    artifact_path = Path(h5_path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"artifact not found: {artifact_path}")

    report: dict[str, Any] = {
        "artifact_path": str(artifact_path),
        "dry_run": bool(dry_run),
        "slice_count": 0,
        "updated_slice_count": 0,
        "updated_fields": {},
        "skipped_fields": {},
        "slices": [],
    }
    mode = "r" if dry_run else "r+"
    with _H5PY_FILE(artifact_path, mode) as f:
        selected_groups: list[tuple[str, h5py.Group]] = []
        if SLICE_CONTAINER_GROUP in f:
            slices = f[SLICE_CONTAINER_GROUP]
            if slice_key is not None:
                if str(slice_key) not in slices:
                    raise KeyError(f"slice not found: {slice_key}")
                selected_groups.append((str(slice_key), slices[str(slice_key)]))
            else:
                for name in sorted(slices.keys()):
                    selected_groups.append((str(name), slices[name]))
        else:
            selected_groups.append((str(slice_key or "legacy"), f))

        report["slice_count"] = len(selected_groups)
        for selected_key, group in selected_groups:
            common = group.get("common")
            if common is None or "diagnostics_json" not in common:
                report["slices"].append(
                    {
                        "slice_key": selected_key,
                        "updated": False,
                        "updated_fields": {},
                        "skipped_fields": {"diagnostics_json": "missing common diagnostics"},
                    }
                )
                continue

            diagnostics = json.loads(decode_scalar(common["diagnostics_json"][()]))
            updated_fields: dict[str, Any] = {}
            skipped_fields: dict[str, str] = {}

            observed_shape = tuple(int(v) for v in np.asarray(common["observed"]).shape)
            if len(observed_shape) >= 2:
                map_ny = int(observed_shape[-2])
                map_nx = int(observed_shape[-1])
                if "map_nx" not in diagnostics or diagnostics.get("map_nx") in {None, ""}:
                    diagnostics["map_nx"] = map_nx
                    updated_fields["map_nx"] = map_nx
                if "map_ny" not in diagnostics or diagnostics.get("map_ny") in {None, ""}:
                    diagnostics["map_ny"] = map_ny
                    updated_fields["map_ny"] = map_ny

            for path_key, hash_key in (
                ("fits_file", "fits_sha256"),
                ("model_path", "model_sha256"),
                ("ebtel_path", "ebtel_sha256"),
            ):
                if str(diagnostics.get(hash_key, "")).strip():
                    continue
                resolved_path = _resolve_existing_file_from_diagnostics(diagnostics.get(path_key))
                if resolved_path is None:
                    skipped_fields[hash_key] = f"source file unavailable from {path_key}"
                    continue
                diagnostics[hash_key] = _compute_file_sha256(resolved_path)
                updated_fields[hash_key] = diagnostics[hash_key]

            if updated_fields and not dry_run:
                _replace_text_dataset(common, "diagnostics_json", _json_dumps(diagnostics))

            if updated_fields:
                report["updated_slice_count"] = int(report["updated_slice_count"]) + 1
            for key in updated_fields:
                report["updated_fields"][key] = int(report["updated_fields"].get(key, 0)) + 1
            for key in skipped_fields:
                report["skipped_fields"][key] = int(report["skipped_fields"].get(key, 0)) + 1
            report["slices"].append(
                {
                    "slice_key": selected_key,
                    "updated": bool(updated_fields),
                    "updated_fields": updated_fields,
                    "skipped_fields": skipped_fields,
                }
            )
    return report


def point_record_matches_compatibility_signature(
    record: dict[str, Any],
    *,
    compatibility_signature: str | None,
) -> bool:
    expected = str(compatibility_signature or "").strip()
    if not expected:
        return True
    diagnostics = dict(record.get("diagnostics", {}))
    actual = str(diagnostics.get(COMPATIBILITY_SIGNATURE_KEY, "")).strip()
    if not actual:
        return False
    return actual == expected


def _write_reference_map_group(
    parent: h5py.Group,
    *,
    group_name: str,
    data: np.ndarray,
    wcs_header: fits.Header,
) -> None:
    ref_group = parent.create_group(group_name)
    ref_group.create_dataset("data", data=np.asarray(data, dtype=np.float32), compression="gzip", compression_opts=4)
    _create_text_dataset(ref_group, "wcs_header", wcs_header.tostring(sep="\n", endcard=True))


def _read_reference_map_group(parent: h5py.Group, group_name: str) -> tuple[np.ndarray, fits.Header] | None:
    if group_name not in parent:
        return None
    ref_group = parent[group_name]
    if "data" not in ref_group or "wcs_header" not in ref_group:
        return None
    data = np.asarray(ref_group["data"], dtype=float)
    header = fits.Header.fromstring(decode_scalar(ref_group["wcs_header"][()]), sep="\n")
    return data, header


def _map_store_identity(
    *,
    name: str,
    normalized: dict[str, Any],
) -> dict[str, Any]:
    diagnostics = dict(normalized.get("diagnostics", {}))
    physical_keys = (
        "model_sha256",
        "ebtel_sha256",
        "spectral_domain",
        "spectral_label",
        "frequency_ghz",
        "wavelength_angstrom",
        "euv_channel",
        "euv_instrument",
        "euv_response_sav",
        "map_xc_arcsec",
        "map_yc_arcsec",
        "map_dx_arcsec",
        "map_dy_arcsec",
        "map_nx",
        "map_ny",
        "observer_name",
        "observer_lonc_deg",
        "observer_b0sun_deg",
        "observer_dsun_cm",
        "observer_obs_time",
        "psf_source",
        "resolved_psf",
        "psf_bmaj_arcsec",
        "psf_bmin_arcsec",
        "psf_bpa_deg",
        "psf_ref_frequency_ghz",
        "psf_scale_inverse_frequency",
        "render_channels",
        "render_frequencies_ghz",
    )
    identity = {
        "array_name": str(name),
        "a": float(normalized["a"]),
        "b": float(normalized["b"]),
        "q0": float(normalized.get("q0", np.nan)),
    }
    for key in physical_keys:
        if key in diagnostics:
            identity[key] = diagnostics[key]
    return identity


def _write_map_store_array(
    h5_file: h5py.File,
    *,
    identity: dict[str, Any],
    data: np.ndarray,
) -> str:
    arr = np.asarray(data, dtype=np.float32)
    digest_payload = {
        "identity": identity,
        "shape": [int(v) for v in arr.shape],
        "dtype": "float32",
    }
    map_id = hashlib.sha256(_json_dumps(digest_payload).encode("utf-8")).hexdigest()
    maps_group = h5_file.require_group(MAP_STORE_GROUP).require_group(MAP_STORE_MAPS_GROUP)
    if map_id not in maps_group:
        map_group = maps_group.create_group(map_id)
        map_group.create_dataset("data", data=arr, compression="gzip", compression_opts=4)
        _create_text_dataset(map_group, "identity_json", _json_dumps(identity))
    return f"/{MAP_STORE_GROUP}/{MAP_STORE_MAPS_GROUP}/{map_id}"


def _write_point_map_ref(
    grp: h5py.Group,
    *,
    normalized: dict[str, Any],
    name: str,
    data: np.ndarray | None,
    map_refs: dict[str, str],
) -> None:
    if data is None:
        return
    map_refs[name] = _write_map_store_array(
        grp.file,
        identity=_map_store_identity(name=name, normalized=normalized),
        data=np.asarray(data, dtype=float),
    )


def _read_point_map_array(grp: h5py.Group, name: str, refs: dict[str, Any]) -> np.ndarray | None:
    if name in grp:
        return np.asarray(grp[name], dtype=float)
    ref_path = refs.get(name)
    if ref_path:
        ref_text = str(ref_path)
        if ref_text in grp.file and "data" in grp.file[ref_text]:
            return np.asarray(grp.file[ref_text]["data"], dtype=float)
    return None


def _read_map_store_ref_array(h5_file: h5py.File, ref_path: Any) -> np.ndarray | None:
    ref_text = str(ref_path or "").strip()
    if not ref_text:
        return None
    if ref_text in h5_file and "data" in h5_file[ref_text]:
        return np.asarray(h5_file[ref_text]["data"], dtype=float)
    return None


def _auxiliary_map_ref_prefixes_for_descriptor(descriptor: dict[str, Any]) -> tuple[str, ...]:
    domain = str(descriptor.get("domain", "")).strip().lower()
    if domain == "mw":
        frequency = _optional_float(descriptor.get("frequency_ghz"))
        if frequency is None:
            return tuple()
        return (f"extra/mw/{float(frequency):.6f}ghz",)
    if domain in {"euv", "uv"}:
        channel = str(descriptor.get("channel_label") or "").strip()
        if not channel:
            wavelength = _optional_float(descriptor.get("wavelength_angstrom"))
            if wavelength is not None:
                rounded = round(float(wavelength))
                channel = str(int(rounded)) if np.isclose(float(wavelength), float(rounded), rtol=0.0, atol=1e-9) else f"{float(wavelength):.6g}"
        if not channel:
            label = str(descriptor.get("label", "")).strip()
            channel = label.split()[0] if label else ""
        if not channel:
            return tuple()
        return (f"extra/{domain}/{channel}", f"extra/euv/{channel}") if domain == "uv" else (f"extra/euv/{channel}",)
    return tuple()


def load_auxiliary_map_store_point_records(
    h5_path: Path,
    *,
    slice_key: str,
    source_search_id: str | None = None,
) -> list[dict[str, Any]]:
    """Load point records reconstructed from auxiliary maps stored for a slice.

    The returned records are not written to ``slice_key``. They expose stored
    trial maps so callers can rescore them for a new metric/mask and then append
    promoted point records under a real search for the selected slice.
    """

    out: list[dict[str, Any]] = []
    with _H5PY_FILE(h5_path, "r") as h5_file:
        if SLICE_CONTAINER_GROUP not in h5_file:
            return out
        _group, descriptors, _selected_key = _resolve_slice_group(
            h5_file,
            slice_key=slice_key,
            allow_missing=False,
        )
        descriptor = next((item for item in descriptors if str(item.get("key")) == str(slice_key)), None)
        if descriptor is None:
            return out
        prefixes = _auxiliary_map_ref_prefixes_for_descriptor(descriptor)
        if not prefixes:
            return out

        slices_group = h5_file[SLICE_CONTAINER_GROUP]
        for source_slice_key in sorted(str(key) for key in slices_group.keys()):
            source_slice = slices_group[source_slice_key]
            searches = source_slice.get(SEARCHES_GROUP)
            if searches is None:
                continue
            search_ids = [str(source_search_id)] if source_search_id is not None else sorted(str(key) for key in searches.keys())
            for search_id in search_ids:
                if search_id not in searches:
                    continue
                records_group = searches[search_id].get("point_records")
                if records_group is None:
                    continue
                for record_name in sorted(str(key) for key in records_group.keys()):
                    record_group = records_group[record_name]
                    if MAP_REFS_DATASET not in record_group:
                        continue
                    map_refs = _json_loads_or_empty(record_group[MAP_REFS_DATASET][()])
                    matching_prefix = next(
                        (
                            prefix
                            for prefix in prefixes
                            if any(str(name).startswith(f"{prefix}/") for name in map_refs.keys())
                        ),
                        None,
                    )
                    if matching_prefix is None:
                        continue

                    base_record = _read_point_group_sparse(record_group)
                    trial_by_index: dict[int, np.ndarray] = {}
                    best_map: np.ndarray | None = None
                    best_keys = (
                        f"{matching_prefix}/rendered_best",
                        f"{matching_prefix}/raw_modeled_best",
                    )
                    for key in best_keys:
                        if key in map_refs:
                            best_map = _read_map_store_ref_array(h5_file, map_refs[key])
                            if best_map is not None:
                                break
                    trial_pattern = re.compile(
                        re.escape(f"{matching_prefix}/") + r"trial_(\d+)/(?:rendered|raw_modeled)$"
                    )
                    for key, ref_path in map_refs.items():
                        match = trial_pattern.match(str(key))
                        if match is None:
                            continue
                        arr = _read_map_store_ref_array(h5_file, ref_path)
                        if arr is not None:
                            trial_by_index[int(match.group(1))] = arr
                    if not trial_by_index and best_map is None:
                        continue

                    promoted = dict(base_record)
                    if trial_by_index:
                        ordered_items = sorted(trial_by_index.items(), key=lambda item: item[0])
                        trial_maps = np.stack([np.asarray(arr, dtype=float) for _index, arr in ordered_items], axis=0)
                        trial_q0 = tuple(float(v) for v in promoted.get("fit_q0_trials", ()))
                        if len(trial_q0) != len(ordered_items):
                            source_q0 = list(float(v) for v in promoted.get("fit_q0_trials", ()))
                            trial_q0 = tuple(source_q0[index] for index, _arr in ordered_items if index < len(source_q0))
                        if len(trial_q0) == trial_maps.shape[0]:
                            promoted["fit_q0_trials"] = trial_q0
                            promoted["trial_modeled_maps"] = trial_maps
                            promoted["trial_raw_modeled_maps"] = trial_maps.copy()
                    if best_map is not None:
                        promoted["modeled_best"] = np.asarray(best_map, dtype=float)
                        promoted["raw_modeled_best"] = np.asarray(best_map, dtype=float)
                    promoted["source_slice_key"] = source_slice_key
                    promoted["source_search_id"] = search_id
                    promoted["source_record_name"] = record_name
                    promoted["source_auxiliary_map_prefix"] = matching_prefix
                    out.append(promoted)
    return out


def _write_common_group(
    common: h5py.Group,
    *,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
    blos_reference: tuple[np.ndarray, fits.Header] | None = None,
    run_history: list[dict[str, Any]] | None = None,
) -> None:
    slice_descriptors, target_slice_key = canonical_slice_descriptors_from_diagnostics(diagnostics)
    trial_logging_policy = canonical_trial_logging_policy_from_diagnostics(diagnostics)
    common.create_dataset("observed", data=np.asarray(observed, dtype=np.float32), compression="gzip", compression_opts=4)
    common.create_dataset("sigma_map", data=np.asarray(sigma_map, dtype=np.float32), compression="gzip", compression_opts=4)
    _create_text_dataset(common, "wcs_header", wcs_header.tostring(sep="\n", endcard=True))
    _create_text_dataset(common, "diagnostics_json", _json_dumps(diagnostics))
    _create_text_dataset(common, COMMON_ARTIFACT_CONTRACT_VERSION_DATASET, CANONICAL_ARTIFACT_CONTRACT_VERSION)
    _create_text_dataset(common, COMMON_SLICE_DESCRIPTORS_DATASET, _json_dumps(slice_descriptors))
    _create_text_dataset(common, COMMON_TARGET_SLICE_KEY_DATASET, str(target_slice_key))
    _create_text_dataset(common, COMMON_TRIAL_LOGGING_POLICY_DATASET, _json_dumps(trial_logging_policy))
    if blos_reference is not None:
        refmaps = common.create_group("refmaps")
        blos_data, blos_header = blos_reference
        _write_reference_map_group(
            refmaps,
            group_name="Bz_reference",
            data=np.asarray(blos_data, dtype=float),
            wcs_header=blos_header,
        )
    dataset = _ensure_run_history_dataset(common)
    for entry in list(run_history or []):
        next_index = int(dataset.shape[0])
        dataset.resize((next_index + 1,))
        dataset[next_index] = _json_dumps(entry)


def _read_common_group(common: h5py.Group) -> dict[str, Any]:
    observed = np.asarray(common["observed"], dtype=float)
    sigma_map = np.asarray(common["sigma_map"], dtype=float)
    wcs_header = fits.Header.fromstring(decode_scalar(common["wcs_header"][()]), sep="\n")
    diagnostics = json.loads(decode_scalar(common["diagnostics_json"][()]))
    run_history = _decode_run_history(common)
    artifact_contract_version = (
        decode_scalar(common[COMMON_ARTIFACT_CONTRACT_VERSION_DATASET][()])
        if COMMON_ARTIFACT_CONTRACT_VERSION_DATASET in common
        else CANONICAL_ARTIFACT_CONTRACT_VERSION
    )
    if COMMON_SLICE_DESCRIPTORS_DATASET in common:
        slice_descriptors = json.loads(decode_scalar(common[COMMON_SLICE_DESCRIPTORS_DATASET][()]))
    else:
        slice_descriptors, _target_slice_key = canonical_slice_descriptors_from_diagnostics(diagnostics)
    if COMMON_TARGET_SLICE_KEY_DATASET in common:
        target_slice_key = decode_scalar(common[COMMON_TARGET_SLICE_KEY_DATASET][()])
    else:
        _slice_descriptors_fallback, target_slice_key = canonical_slice_descriptors_from_diagnostics(diagnostics)
    if COMMON_TRIAL_LOGGING_POLICY_DATASET in common:
        trial_logging_policy = json.loads(decode_scalar(common[COMMON_TRIAL_LOGGING_POLICY_DATASET][()]))
    else:
        trial_logging_policy = canonical_trial_logging_policy_from_diagnostics(diagnostics)
    blos_reference = None
    if "refmaps" in common:
        blos_reference = _read_reference_map_group(common["refmaps"], "Bz_reference")
    return {
        "observed": observed,
        "sigma_map": sigma_map,
        "wcs_header": wcs_header,
        "diagnostics": diagnostics,
        "run_history": run_history,
        "artifact_contract_version": artifact_contract_version,
        "slice_descriptors": slice_descriptors,
        "target_slice_key": target_slice_key,
        "trial_logging_policy": trial_logging_policy,
        "blos_reference": blos_reference,
    }


def _set_slice_group_attrs(slice_group: h5py.Group, descriptor: dict[str, Any]) -> None:
    slice_group.attrs["domain"] = np.bytes_(str(descriptor["domain"]))
    slice_group.attrs["label"] = np.bytes_(str(descriptor["label"]))
    if descriptor.get("frequency_ghz") is not None:
        slice_group.attrs["frequency_ghz"] = float(descriptor["frequency_ghz"])
    if descriptor.get("channel_label") is not None:
        slice_group.attrs["channel_label"] = np.bytes_(str(descriptor["channel_label"]))


def _diagnostics_for_slice_descriptor(
    diagnostics: dict[str, Any],
    descriptor: dict[str, Any],
    *,
    target_slice_key: str,
) -> dict[str, Any]:
    out = dict(diagnostics)
    out["slice_key"] = str(descriptor["key"])
    out["target_slice_key"] = str(target_slice_key)
    out["spectral_domain"] = str(descriptor["domain"])
    out["spectral_label"] = str(descriptor["label"])
    out["frequency_ghz"] = descriptor.get("frequency_ghz")
    out["active_frequency_ghz"] = descriptor.get("frequency_ghz")
    out["wavelength_angstrom"] = descriptor.get("wavelength_angstrom")
    out["euv_channel"] = descriptor.get("channel_label")
    out["slice_role"] = str(descriptor.get("role", "auxiliary"))
    out["is_target_slice"] = bool(descriptor.get("is_target", False))
    return out


def _write_auxiliary_slice_shells(
    slices_group: h5py.Group,
    *,
    observed_template: np.ndarray,
    sigma_template: np.ndarray,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
    blos_reference: tuple[np.ndarray, fits.Header] | None,
    existing_names: set[str] | None = None,
) -> None:
    descriptors, target_slice_key = canonical_slice_descriptors_from_diagnostics(diagnostics)
    existing = set(existing_names or set(slices_group.keys()))
    for descriptor in descriptors:
        key = str(descriptor["key"])
        if key == str(target_slice_key) or key in existing:
            continue
        slice_group = slices_group.create_group(key)
        _set_slice_group_attrs(slice_group, descriptor)
        common = slice_group.create_group("common")
        aux_diag = _diagnostics_for_slice_descriptor(
            diagnostics,
            descriptor,
            target_slice_key=str(target_slice_key),
        )
        aux_diag["artifact_kind"] = UNIFIED_ARTIFACT_KIND
        aux_diag["render_only_slice"] = True
        aux_diag["search_mode"] = str(diagnostics.get("search_mode", "")) + "_render_only"
        aux_observed = np.full_like(np.asarray(observed_template, dtype=float), np.nan, dtype=float)
        aux_sigma = np.full_like(np.asarray(sigma_template, dtype=float), np.nan, dtype=float)
        _write_common_group(
            common,
            observed=aux_observed,
            sigma_map=aux_sigma,
            wcs_header=wcs_header,
            diagnostics=aux_diag,
            blos_reference=blos_reference,
            run_history=None,
        )
        slice_group.create_group(SEARCHES_GROUP)
        existing.add(key)


def _copy_legacy_root_layout_to_slice(src: h5py.File, dst_slice: h5py.Group) -> None:
    for name in ("common", "grid", "summary", "points", "point_records", SEARCHES_GROUP, ACTIVE_SEARCH_ID_DATASET):
        if name in src and name not in dst_slice:
            src.copy(src[name], dst_slice, name=name)


def write_grid_scan_artifact(
    out_h5: Path,
    *,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
    blos_reference: tuple[np.ndarray, fits.Header] | None = None,
    a_values: np.ndarray,
    b_values: np.ndarray,
    best_q0: np.ndarray,
    objective_values: np.ndarray,
    chi2: np.ndarray,
    rho2: np.ndarray,
    eta2: np.ndarray,
    success: np.ndarray,
    point_payloads: dict[tuple[int, int], dict[str, Any]],
    slice_key: str | None = None,
    run_history: list[dict[str, Any]] | None = None,
) -> None:
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_out = dict(diagnostics)
    diagnostics_out["artifact_kind"] = UNIFIED_ARTIFACT_KIND
    descriptor = target_slice_descriptor_from_diagnostics(diagnostics_out, fallback_key=slice_key or "default")
    resolved_slice_key = str(slice_key or descriptor["key"] or "default")
    descriptor["key"] = resolved_slice_key
    tmp_h5 = out_h5.with_suffix(out_h5.suffix + ".tmp")
    with _H5PY_FILE(tmp_h5, "w") as dst:
        slices_dst = dst.create_group(SLICE_CONTAINER_GROUP)
        if out_h5.exists():
            with _H5PY_FILE(out_h5, "r") as src:
                if SLICE_CONTAINER_GROUP in src:
                    _validate_new_slice_geometry_compatibility(
                        src[SLICE_CONTAINER_GROUP],
                        diagnostics=diagnostics_out,
                        slice_key=resolved_slice_key,
                        artifact_path=out_h5,
                    )
                    for name in src[SLICE_CONTAINER_GROUP].keys():
                        if str(name) == resolved_slice_key:
                            continue
                        src.copy(src[SLICE_CONTAINER_GROUP][name], slices_dst, name=name)
                elif "common" in src:
                    existing_diag = json.loads(decode_scalar(src["common"]["diagnostics_json"][()]))
                    existing_descriptor = slice_descriptor_from_diagnostics(existing_diag, fallback_key="legacy")
                    existing_key = str(existing_descriptor["key"])
                    if existing_key != resolved_slice_key:
                        legacy_dst = slices_dst.create_group(existing_key)
                        for name in src.keys():
                            src.copy(name, legacy_dst, name=name)

        slice_group = slices_dst.create_group(resolved_slice_key)
        _set_slice_group_attrs(slice_group, descriptor)

        common = slice_group.create_group("common")
        _write_common_group(
            common,
            observed=observed,
            sigma_map=sigma_map,
            wcs_header=wcs_header,
            diagnostics=diagnostics_out,
            blos_reference=blos_reference,
            run_history=run_history,
        )
        _write_auxiliary_slice_shells(
            slices_dst,
            observed_template=observed,
            sigma_template=sigma_map,
            wcs_header=wcs_header,
            diagnostics=diagnostics_out,
            blos_reference=blos_reference,
            existing_names=set(slices_dst.keys()),
        )

        current_point_records: list[dict[str, Any]] = []
        for record_order, ((_ai, _bi), payload) in enumerate(sorted(point_payloads.items())):
            current_point_records.append(payload)
        searches_group = slice_group.create_group(SEARCHES_GROUP)
        if out_h5.exists():
            with _H5PY_FILE(out_h5, "r") as src:
                src_slice = None
                if SLICE_CONTAINER_GROUP in src and resolved_slice_key in src[SLICE_CONTAINER_GROUP]:
                    src_slice = src[SLICE_CONTAINER_GROUP][resolved_slice_key]
                elif "common" in src:
                    src_slice = src
                if src_slice is not None and SEARCHES_GROUP in src_slice:
                    for name in src_slice[SEARCHES_GROUP].keys():
                        src_slice[SEARCHES_GROUP].copy(name, searches_group, name=name)
        current_search_id = _search_id_from_diagnostics(diagnostics_out)
        _write_search_group(
            searches_group,
            search_id=current_search_id,
            diagnostics=diagnostics_out,
            point_records=current_point_records,
            run_history=run_history,
            layout={
                "kind": "rectangular_grid",
                "a_values": [float(v) for v in np.asarray(a_values, dtype=float)],
                "b_values": [float(v) for v in np.asarray(b_values, dtype=float)],
            },
        )
        _create_text_dataset(slice_group, ACTIVE_SEARCH_ID_DATASET, current_search_id)
    os.replace(tmp_h5, out_h5)


def save_rectangular_scan_file(
    out_h5: Path,
    *,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
    blos_reference: tuple[np.ndarray, fits.Header] | None = None,
    a_values: np.ndarray,
    b_values: np.ndarray,
    best_q0: np.ndarray,
    objective_values: np.ndarray,
    chi2: np.ndarray,
    rho2: np.ndarray,
    eta2: np.ndarray,
    success: np.ndarray,
    point_payloads: dict[tuple[int, int], dict[str, Any]],
    slice_key: str | None = None,
    run_history: list[dict[str, Any]] | None = None,
) -> None:
    write_grid_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=wcs_header,
        diagnostics=diagnostics,
        blos_reference=blos_reference,
        a_values=a_values,
        b_values=b_values,
        best_q0=best_q0,
        objective_values=objective_values,
        chi2=chi2,
        rho2=rho2,
        eta2=eta2,
        success=success,
        point_payloads=point_payloads,
        slice_key=slice_key,
        run_history=run_history,
    )


def _write_point_group(grp: h5py.Group, payload: dict[str, Any], *, record_order: int) -> None:
    normalized = _normalize_point_payload(payload, record_order=record_order)
    map_refs: dict[str, str] = {}
    grp.attrs["record_order"] = int(record_order)
    grp.attrs["a"] = float(normalized["a"])
    grp.attrs["b"] = float(normalized["b"])
    grp.attrs["q0"] = float(normalized["q0"])
    grp.attrs["success"] = int(bool(normalized["success"]))
    grp.attrs["status"] = np.bytes_(str(normalized["status"]))
    grp.attrs["target_metric"] = np.bytes_(str(normalized["target_metric"]))
    grp.attrs["nfev"] = int(normalized["nfev"])
    grp.attrs["nit"] = int(normalized["nit"])
    grp.attrs["message"] = np.bytes_(str(normalized["message"]))
    grp.attrs["used_adaptive_bracketing"] = int(bool(normalized["used_adaptive_bracketing"]))
    grp.attrs["bracket_found"] = int(bool(normalized["bracket_found"]))
    if normalized["bracket"] is not None:
        grp.create_dataset("bracket", data=np.asarray(normalized["bracket"], dtype=np.float64))
    _write_point_map_ref(grp, normalized=normalized, name="modeled_best", data=normalized["modeled_best"], map_refs=map_refs)
    _write_point_map_ref(grp, normalized=normalized, name="raw_modeled_best", data=normalized["raw_modeled_best"], map_refs=map_refs)
    _write_point_map_ref(grp, normalized=normalized, name="residual", data=normalized["residual"], map_refs=map_refs)
    grp.create_dataset("fit_q0_trials", data=np.asarray(normalized["fit_q0_trials"], dtype=np.float64))
    fit_metric_ds = grp.create_dataset("fit_metric_trials", data=np.asarray(normalized["fit_metric_trials"], dtype=np.float64))
    fit_metric_ds.attrs["target_metric"] = np.bytes_(str(normalized["target_metric"]))
    grp.create_dataset("fit_chi2_trials", data=np.asarray(normalized["fit_chi2_trials"], dtype=np.float64))
    grp.create_dataset("fit_rho2_trials", data=np.asarray(normalized["fit_rho2_trials"], dtype=np.float64))
    grp.create_dataset("fit_eta2_trials", data=np.asarray(normalized["fit_eta2_trials"], dtype=np.float64))
    if normalized["trial_raw_modeled_maps"] is not None:
        _write_point_map_ref(grp, normalized=normalized, name="trial_raw_modeled_maps", data=normalized["trial_raw_modeled_maps"], map_refs=map_refs)
    if normalized["trial_modeled_maps"] is not None:
        _write_point_map_ref(grp, normalized=normalized, name="trial_modeled_maps", data=normalized["trial_modeled_maps"], map_refs=map_refs)
    if normalized["trial_residual_maps"] is not None:
        _write_point_map_ref(grp, normalized=normalized, name="trial_residual_maps", data=normalized["trial_residual_maps"], map_refs=map_refs)
    if normalized["euv_coronal_best"] is not None:
        _write_point_map_ref(grp, normalized=normalized, name="euv_coronal_best", data=normalized["euv_coronal_best"], map_refs=map_refs)
    if normalized["euv_tr_best"] is not None:
        _write_point_map_ref(grp, normalized=normalized, name="euv_tr_best", data=normalized["euv_tr_best"], map_refs=map_refs)
    if normalized["euv_tr_mask"] is not None:
        grp.create_dataset(
            "euv_tr_mask",
            data=np.asarray(normalized["euv_tr_mask"], dtype=np.uint8),
            compression="gzip",
            compression_opts=4,
        )
    if normalized["trial_euv_coronal_maps"] is not None:
        _write_point_map_ref(grp, normalized=normalized, name="trial_euv_coronal_maps", data=normalized["trial_euv_coronal_maps"], map_refs=map_refs)
    if normalized["trial_euv_tr_maps"] is not None:
        _write_point_map_ref(grp, normalized=normalized, name="trial_euv_tr_maps", data=normalized["trial_euv_tr_maps"], map_refs=map_refs)
    for extra_name, extra_data in dict(normalized.get("map_store_arrays") or {}).items():
        _write_point_map_ref(
            grp,
            normalized=normalized,
            name=f"extra/{extra_name}",
            data=np.asarray(extra_data, dtype=float),
            map_refs=map_refs,
        )
    _create_text_dataset(grp, MAP_REFS_DATASET, _json_dumps(map_refs))
    _create_text_dataset(grp, "diagnostics_json", _json_dumps(normalized["diagnostics"]))


def write_point_scan_artifact(
    out_h5: Path,
    *,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
    blos_reference: tuple[np.ndarray, fits.Header] | None = None,
    point_records: list[dict[str, Any]],
    run_history: list[dict[str, Any]] | None = None,
) -> None:
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    tmp_h5 = out_h5.with_suffix(out_h5.suffix + ".tmp")
    diagnostics_out = dict(diagnostics)
    diagnostics_out["artifact_kind"] = UNIFIED_ARTIFACT_KIND
    # Always record mask_type if present, else default to 'union'
    if "mask_type" not in diagnostics_out:
        diagnostics_out["mask_type"] = diagnostics.get("mask_type", "union")
    with _H5PY_FILE(tmp_h5, "w") as f:
        descriptor = target_slice_descriptor_from_diagnostics(diagnostics_out, fallback_key="default")
        resolved_slice_key = str(descriptor["key"] or "default")
        slices_group = f.create_group(SLICE_CONTAINER_GROUP)
        slice_group = slices_group.create_group(resolved_slice_key)
        _set_slice_group_attrs(slice_group, descriptor)
        if out_h5.exists():
            with _H5PY_FILE(out_h5, "r") as src:
                if SLICE_CONTAINER_GROUP in src:
                    _validate_new_slice_geometry_compatibility(
                        src[SLICE_CONTAINER_GROUP],
                        diagnostics=diagnostics_out,
                        slice_key=resolved_slice_key,
                        artifact_path=out_h5,
                    )
                    for name in src[SLICE_CONTAINER_GROUP].keys():
                        if str(name) == resolved_slice_key:
                            continue
                        src.copy(src[SLICE_CONTAINER_GROUP][name], slices_group, name=name)
        common = slice_group.create_group("common")
        _write_common_group(
            common,
            observed=observed,
            sigma_map=sigma_map,
            wcs_header=wcs_header,
            diagnostics=diagnostics_out,
            blos_reference=blos_reference,
            run_history=run_history,
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
        searches_group = slice_group.create_group(SEARCHES_GROUP)
        if out_h5.exists():
            with _H5PY_FILE(out_h5, "r") as src:
                src_slice = None
                if SLICE_CONTAINER_GROUP in src and resolved_slice_key in src[SLICE_CONTAINER_GROUP]:
                    src_slice = src[SLICE_CONTAINER_GROUP][resolved_slice_key]
                elif "common" in src:
                    src_slice = src
                if src_slice is not None and SEARCHES_GROUP in src_slice:
                    for name in src_slice[SEARCHES_GROUP].keys():
                        src_slice[SEARCHES_GROUP].copy(name, searches_group, name=name)
        current_search_id = _search_id_from_diagnostics(diagnostics_out)
        _write_search_group(
            searches_group,
            search_id=current_search_id,
            diagnostics=diagnostics_out,
            point_records=list(point_records),
            run_history=run_history,
            layout={"kind": "point_list"},
        )
        _create_text_dataset(slice_group, ACTIVE_SEARCH_ID_DATASET, current_search_id)
    os.replace(tmp_h5, out_h5)


def write_sparse_scan_file(
    out_h5: Path,
    *,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
    blos_reference: tuple[np.ndarray, fits.Header] | None = None,
    point_records: list[dict[str, Any]],
    run_history: list[dict[str, Any]] | None = None,
) -> None:
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=wcs_header,
        diagnostics=diagnostics,
        blos_reference=blos_reference,
        point_records=point_records,
        run_history=run_history,
    )


def write_single_point_scan_file(
    out_h5: Path,
    *,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
    point_payload: dict[str, Any],
    blos_reference: tuple[np.ndarray, fits.Header] | None = None,
    run_history: list[dict[str, Any]] | None = None,
) -> None:
    diagnostics_out = dict(diagnostics)
    diagnostics_out["artifact_kind"] = UNIFIED_ARTIFACT_KIND
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=wcs_header,
        diagnostics=diagnostics_out,
        blos_reference=blos_reference,
        point_records=[point_payload],
        run_history=run_history,
    )


def append_scan_point_record(
    out_h5: Path,
    *,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
    blos_reference: tuple[np.ndarray, fits.Header] | None = None,
    point_payload: dict[str, Any],
) -> None:
    diagnostics_out = dict(diagnostics)
    diagnostics_out["artifact_kind"] = UNIFIED_ARTIFACT_KIND
    if "mask_type" not in diagnostics_out:
        diagnostics_out["mask_type"] = diagnostics.get("mask_type", "union")
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if out_h5.exists() else "w"
    last_exc: Exception | None = None
    for attempt in range(1, _SPARSE_APPEND_RETRY_ATTEMPTS + 1):
        try:
            with h5py.File(out_h5, mode) as f:
                descriptor = target_slice_descriptor_from_diagnostics(diagnostics_out, fallback_key="default")
                resolved_slice_key = str(descriptor["key"] or "default")
                slices_group = f.require_group(SLICE_CONTAINER_GROUP)
                if resolved_slice_key not in slices_group:
                    _validate_new_slice_geometry_compatibility(
                        slices_group,
                        diagnostics=diagnostics_out,
                        slice_key=resolved_slice_key,
                        artifact_path=out_h5,
                    )
                    slice_group = slices_group.create_group(resolved_slice_key)
                    _set_slice_group_attrs(slice_group, descriptor)
                    if "common" in f:
                        _copy_legacy_root_layout_to_slice(f, slice_group)
                else:
                    slice_group = slices_group[resolved_slice_key]
                    _set_slice_group_attrs(slice_group, descriptor)
                if "common" not in slice_group:
                    common = slice_group.create_group("common")
                    _write_common_group(
                        common,
                        observed=observed,
                        sigma_map=sigma_map,
                        wcs_header=wcs_header,
                        diagnostics=diagnostics_out,
                        blos_reference=blos_reference,
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
                elif blos_reference is not None and "refmaps" not in slice_group["common"]:
                    refmaps = slice_group["common"].create_group("refmaps")
                    blos_data, blos_header = blos_reference
                    _write_reference_map_group(
                        refmaps,
                        group_name="Bz_reference",
                        data=np.asarray(blos_data, dtype=float),
                        wcs_header=blos_header,
                    )
                if "common" in slice_group:
                    existing_diagnostics = _diagnostics_from_slice_group(slice_group)
                    if bool(existing_diagnostics.get("render_only_slice", False)):
                        del slice_group["common"]
                        common = slice_group.create_group("common")
                        _write_common_group(
                            common,
                            observed=observed,
                            sigma_map=sigma_map,
                            wcs_header=wcs_header,
                            diagnostics=diagnostics_out,
                            blos_reference=blos_reference,
                            run_history=None,
                        )
                searches_group = slice_group.require_group(SEARCHES_GROUP)
                current_search_id = _search_id_from_diagnostics(diagnostics_out)
                if ACTIVE_SEARCH_ID_DATASET in slice_group:
                    del slice_group[ACTIVE_SEARCH_ID_DATASET]
                _create_text_dataset(slice_group, ACTIVE_SEARCH_ID_DATASET, current_search_id)
                search_group = searches_group.require_group(current_search_id)
                if "diagnostics_json" not in search_group:
                    search_group.attrs["search_id"] = np.bytes_(current_search_id)
                    search_group.attrs["target_metric"] = np.bytes_(str(diagnostics_out.get("target_metric", "chi2")))
                    _create_text_dataset(search_group, "diagnostics_json", _json_dumps(diagnostics_out))
                    layout_payload = {"kind": "point_list"}
                    _create_text_dataset(search_group, "layout_json", _json_dumps(layout_payload))
                    _create_text_dataset(search_group, "run_history_json", _json_dumps([]))
                    _create_text_dataset(
                        search_group,
                        SEARCH_REQUEST_DATASET,
                        _json_dumps(_search_request_from_diagnostics(diagnostics_out, layout=layout_payload)),
                    )
                search_records = search_group.require_group("point_records")
                search_orders = [int(search_records[name].attrs.get("record_order", -1)) for name in search_records.keys()]
                search_next_order = max(search_orders, default=-1) + 1
                search_point_group = search_records.create_group(f"r{search_next_order:06d}")
                _write_point_group(search_point_group, point_payload, record_order=search_next_order)
                if "total_point_count" in search_group.attrs:
                    counts = {
                        "total": int(search_group.attrs.get("total_point_count", 0)),
                        "pending": int(search_group.attrs.get("pending_point_count", 0)),
                        "missing": int(search_group.attrs.get("missing_point_count", 0)),
                        "failed": int(search_group.attrs.get("failed_point_count", 0)),
                        "computed": int(search_group.attrs.get("computed_point_count", 0)),
                        "other": int(search_group.attrs.get("other_point_count", 0)),
                    }
                else:
                    counts = _search_status_counts_from_records(_load_sparse_point_records(search_records))
                    counts["total"] = max(0, counts["total"] - 1)
                    status = _normalize_point_status(point_payload.get("status", "computed"))
                    if status in {"pending", "missing", "failed", "computed"}:
                        counts[status] = max(0, counts[status] - 1)
                    else:
                        counts["other"] = max(0, counts["other"] - 1)
                new_status = _normalize_point_status(point_payload.get("status", "computed"))
                counts["total"] += 1
                if new_status in {"pending", "missing", "failed", "computed"}:
                    counts[new_status] += 1
                else:
                    counts["other"] += 1
                status = _write_search_status_attrs(search_group, counts, diagnostics=diagnostics_out)
                existing_lifecycle = (
                    _json_loads_or_empty(search_group[SEARCH_LIFECYCLE_DATASET][()])
                    if SEARCH_LIFECYCLE_DATASET in search_group
                    else {}
                )
                lifecycle = _search_lifecycle_payload(
                    status=status,
                    diagnostics=diagnostics_out,
                    existing=existing_lifecycle,
                )
                _write_search_lifecycle_dataset(search_group, lifecycle=lifecycle)
            return
        except (BlockingIOError, PermissionError, OSError) as exc:
            last_exc = exc
            if attempt >= _SPARSE_APPEND_RETRY_ATTEMPTS:
                break
            time.sleep(_SPARSE_APPEND_RETRY_DELAY_S)
    if last_exc is not None:
        raise OSError(
            f"unable to append sparse point record after {_SPARSE_APPEND_RETRY_ATTEMPTS} attempts: {out_h5}"
        ) from last_exc


def append_sparse_point_record(
    out_h5: Path,
    *,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
    blos_reference: tuple[np.ndarray, fits.Header] | None = None,
    point_payload: dict[str, Any],
) -> None:
    append_scan_point_record(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=wcs_header,
        diagnostics=diagnostics,
        blos_reference=blos_reference,
        point_payload=point_payload,
    )


def append_point_record(
    out_h5: Path,
    *,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
    blos_reference: tuple[np.ndarray, fits.Header] | None = None,
    point_payload: dict[str, Any],
) -> None:
    artifact_kind = str(diagnostics.get("artifact_kind", "")).strip()
    if not out_h5.exists():
        raise FileNotFoundError(
            f"artifact must be initialized before appending point records: {out_h5}"
        )

    with _H5PY_FILE(out_h5, "r") as f:
        uses_slice_container = SLICE_CONTAINER_GROUP in f
        selected_group, _descriptors, _selected_key = _resolve_slice_group(f)
        has_rectangular_summary = bool(selected_group is not None and "summary" in selected_group)

    if artifact_kind == SPARSE_ARTIFACT_KIND or (
        artifact_kind == UNIFIED_ARTIFACT_KIND and not has_rectangular_summary
    ):
        append_scan_point_record(
            out_h5,
            observed=observed,
            sigma_map=sigma_map,
            wcs_header=wcs_header,
            diagnostics=diagnostics,
            blos_reference=blos_reference,
            point_payload=point_payload,
        )
        return

    payload = load_scan_file(out_h5)
    if str(payload.get("artifact_format", "")) in {"sparse", "unified"} and not uses_slice_container:
        append_scan_point_record(
            out_h5,
            observed=observed,
            sigma_map=sigma_map,
            wcs_header=wcs_header,
            diagnostics=diagnostics,
            blos_reference=blos_reference,
            point_payload=point_payload,
        )
        return

    a_values = np.asarray(payload["a_values"], dtype=float)
    b_values = np.asarray(payload["b_values"], dtype=float)
    points = dict(payload["points"])
    best_q0 = np.asarray(payload["best_q0"], dtype=float)
    objective_values = np.asarray(payload["objective_values"], dtype=float)
    chi2 = np.asarray(payload["chi2"], dtype=float)
    rho2 = np.asarray(payload["rho2"], dtype=float)
    eta2 = np.asarray(payload["eta2"], dtype=float)
    success = np.asarray(payload["success"], dtype=bool)

    if "a_index" in point_payload and "b_index" in point_payload:
        a_index = int(point_payload["a_index"])
        b_index = int(point_payload["b_index"])
    else:
        a_index = int(_match_existing_index(a_values, float(point_payload["a"])))
        b_index = int(_match_existing_index(b_values, float(point_payload["b"])))

    diagnostics_json = dict(point_payload.get("diagnostics", {}))
    points[(a_index, b_index)] = {
        **point_payload,
        "a_index": int(a_index),
        "b_index": int(b_index),
    }
    best_q0[a_index, b_index] = float(point_payload.get("q0", np.nan))
    objective_values[a_index, b_index] = float(diagnostics_json.get("target_metric_value", np.nan))
    chi2[a_index, b_index] = float(diagnostics_json.get("chi2", np.nan))
    rho2[a_index, b_index] = float(diagnostics_json.get("rho2", np.nan))
    eta2[a_index, b_index] = float(diagnostics_json.get("eta2", np.nan))
    success[a_index, b_index] = bool(point_payload.get("success", False))

    write_grid_scan_artifact(
        out_h5,
        observed=np.asarray(observed, dtype=float),
        sigma_map=np.asarray(sigma_map, dtype=float),
        wcs_header=wcs_header,
        diagnostics=dict(diagnostics),
        blos_reference=blos_reference if blos_reference is not None else payload.get("blos_reference"),
        a_values=a_values,
        b_values=b_values,
        best_q0=best_q0,
        objective_values=objective_values,
        chi2=chi2,
        rho2=rho2,
        eta2=eta2,
        success=success,
        point_payloads=points,
        run_history=list(payload.get("run_history", [])),
    )


def convert_rectangular_artifact_to_sparse(src_h5: Path, dst_h5: Path, *, overwrite: bool = False) -> None:
    if dst_h5.exists() and not overwrite:
        raise FileExistsError(f"destination already exists: {dst_h5}")
    payload = load_scan_file(src_h5)
    diagnostics = dict(payload["diagnostics"])
    diagnostics["artifact_kind"] = SPARSE_ARTIFACT_KIND
    point_records = [
        {key: value for key, value in record.items() if key not in {"a_index", "b_index", "metrics"}}
        for record in payload.get("point_records", [])
    ]
    write_point_scan_artifact(
        dst_h5,
        observed=np.asarray(payload["observed"], dtype=float),
        sigma_map=np.asarray(payload["sigma_map"], dtype=float),
        wcs_header=payload["wcs_header"],
        diagnostics=diagnostics,
        blos_reference=payload.get("blos_reference"),
        point_records=point_records,
    )


def build_patch_grid_model(payload: dict[str, Any]) -> dict[str, Any]:
    source_records = payload.get("point_records")
    if source_records is None:
        source_records = list(payload["points"].values())
    records = [record for record in source_records if str(record.get("status", "computed")) != "missing"]
    if not records:
        return {
            "records": [],
            "a_min": 0.0,
            "a_max": 1.0,
            "b_min": 0.0,
            "b_max": 1.0,
        }

    a_coords = np.unique([float(record["a"]) for record in records])
    b_coords = np.unique([float(record["b"]) for record in records])
    a_spans = _axis_spans(a_coords)
    b_spans = _axis_spans(b_coords)
    display_records: list[dict[str, Any]] = []
    for record in records:
        a_value = float(record["a"])
        b_value = float(record["b"])
        a0, a1 = a_spans[a_value]
        b0, b1 = b_spans[b_value]
        diagnostics = dict(record.get("diagnostics", {}))
        metrics = record.get(
            "metrics",
            {
                "chi2": float(diagnostics.get("chi2", np.nan)),
                "rho2": float(diagnostics.get("rho2", np.nan)),
                "eta2": float(diagnostics.get("eta2", np.nan)),
            },
        )
        display_records.append(
            {
                "key": (int(record.get("a_index", 0)), int(record.get("b_index", 0))),
                "a_index": int(record.get("a_index", 0)),
                "b_index": int(record.get("b_index", 0)),
                "a": a_value,
                "b": b_value,
                "a0": a0,
                "a1": a1,
                "b0": b0,
                "b1": b1,
                "a_center": 0.5 * (a0 + a1),
                "b_center": 0.5 * (b0 + b1),
                "metrics": metrics,
                "status": str(record.get("status", "computed")),
                "q0": float(record.get("q0", np.nan)),
                "success": bool(record.get("success", False)),
            }
        )
    return {
        "records": display_records,
        "a_min": float(min(record["a0"] for record in display_records)),
        "a_max": float(max(record["a1"] for record in display_records)),
        "b_min": float(min(record["b0"] for record in display_records)),
        "b_max": float(max(record["b1"] for record in display_records)),
    }


def find_record_for_point(model: dict[str, Any], x: float, y: float) -> dict[str, Any] | None:
    for record in model.get("records", []):
        if float(record["b0"]) <= float(x) <= float(record["b1"]) and float(record["a0"]) <= float(y) <= float(record["a1"]):
            return record
    return None


def best_grid_index(payload: dict[str, Any], metric: str) -> tuple[int, int]:
    metric_name = str(metric).strip().lower()
    if metric_name not in METRICS:
        raise ValueError(f"Unsupported best-of-grid metric: {metric_name}")
    arr = np.asarray(payload[metric_name], dtype=float)
    good = np.isfinite(arr)
    if not np.any(good):
        raise ValueError(f"No finite values available for metric {metric_name}")
    idx = np.nanargmin(arr)
    a_index, b_index = np.unravel_index(idx, arr.shape)
    return int(a_index), int(b_index)


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(np.asarray(values, dtype=float) - float(target))))


def with_observer_metadata(header: fits.Header, source_header: fits.Header, diagnostics: dict[str, Any]) -> fits.Header:
    out = header.copy()
    for key in (
        "OBSERVER",
        "DATE-OBS",
        "DSUN_OBS",
        "HGLN_OBS",
        "HGLT_OBS",
        "CRLN_OBS",
        "CRLT_OBS",
        "HGLN-OBS",
        "HGLT-OBS",
        "CRLN-OBS",
        "CRLT-OBS",
    ):
        if key in source_header and key not in out:
            out[key] = source_header[key]

    if "OBSERVER" not in out and diagnostics.get("observer_name"):
        out["OBSERVER"] = str(diagnostics["observer_name"])
    if "DATE-OBS" not in out and diagnostics.get("observer_obs_time"):
        out["DATE-OBS"] = str(diagnostics["observer_obs_time"])

    scalar_fallbacks = {
        "DSUN_OBS": ("observer_dsun_cm", 0.01),
        "HGLN_OBS": ("observer_lonc_deg", 1.0),
        "HGLT_OBS": ("observer_b0sun_deg", 1.0),
        "CRLN_OBS": ("observer_lonc_deg", 1.0),
        "CRLT_OBS": ("observer_b0sun_deg", 1.0),
        "HGLN-OBS": ("observer_lonc_deg", 1.0),
        "HGLT-OBS": ("observer_b0sun_deg", 1.0),
        "CRLN-OBS": ("observer_lonc_deg", 1.0),
        "CRLT-OBS": ("observer_b0sun_deg", 1.0),
    }
    for key, (diag_key, scale) in scalar_fallbacks.items():
        if key in out:
            continue
        value = diagnostics.get(diag_key)
        if value is None:
            continue
        try:
            out[key] = float(value) * float(scale)
        except Exception:
            continue
    return out
