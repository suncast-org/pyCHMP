from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import h5py
import numpy as np
from astropy.io import fits


METRICS = ("chi2", "rho2", "eta2")
RECTANGULAR_ARTIFACT_KIND = "pychmp_ab_scan"
SPARSE_ARTIFACT_KIND = "pychmp_ab_scan_sparse_points"
SLICE_CONTAINER_GROUP = "slices"
RUN_HISTORY_DATASET = "run_history_json"
COMMON_SLICE_DESCRIPTORS_DATASET = "slice_descriptors_json"
COMMON_TARGET_SLICE_KEY_DATASET = "target_slice_key"
COMMON_TRIAL_LOGGING_POLICY_DATASET = "trial_logging_policy_json"
COMMON_ARTIFACT_CONTRACT_VERSION_DATASET = "artifact_contract_version"
CANONICAL_ARTIFACT_CONTRACT_VERSION = "2026-04-23-a"
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
COMPATIBILITY_SIGNATURE_KEY = "compatibility_signature"


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
    if artifact_kind != SPARSE_ARTIFACT_KIND:
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
    return {
        "record_order": int(grp.attrs.get("record_order", 0)),
        "a": float(grp.attrs["a"]),
        "b": float(grp.attrs["b"]),
        "q0": float(grp.attrs["q0"]),
        "success": bool(grp.attrs["success"]),
        "status": decode_scalar(grp.attrs.get("status", b"computed")),
        "modeled_best": np.asarray(grp["modeled_best"], dtype=float),
        "raw_modeled_best": np.asarray(grp["raw_modeled_best"], dtype=float),
        "residual": np.asarray(grp["residual"], dtype=float),
        "fit_q0_trials": tuple(float(v) for v in np.asarray(grp["fit_q0_trials"], dtype=float)),
        "fit_metric_trials": tuple(float(v) for v in fit_metric_trials),
        "fit_chi2_trials": tuple(float(v) for v in fit_chi2_trials),
        "fit_rho2_trials": tuple(float(v) for v in fit_rho2_trials),
        "fit_eta2_trials": tuple(float(v) for v in fit_eta2_trials),
        "trial_raw_modeled_maps": (
            np.asarray(grp["trial_raw_modeled_maps"], dtype=float) if "trial_raw_modeled_maps" in grp else None
        ),
        "trial_modeled_maps": (
            np.asarray(grp["trial_modeled_maps"], dtype=float) if "trial_modeled_maps" in grp else None
        ),
        "trial_residual_maps": (
            np.asarray(grp["trial_residual_maps"], dtype=float) if "trial_residual_maps" in grp else None
        ),
        "euv_coronal_best": (
            np.asarray(grp["euv_coronal_best"], dtype=float) if "euv_coronal_best" in grp else None
        ),
        "euv_tr_best": (
            np.asarray(grp["euv_tr_best"], dtype=float) if "euv_tr_best" in grp else None
        ),
        "euv_tr_mask": (
            np.asarray(grp["euv_tr_mask"], dtype=bool) if "euv_tr_mask" in grp else None
        ),
        "trial_euv_coronal_maps": (
            np.asarray(grp["trial_euv_coronal_maps"], dtype=float) if "trial_euv_coronal_maps" in grp else None
        ),
        "trial_euv_tr_maps": (
            np.asarray(grp["trial_euv_tr_maps"], dtype=float) if "trial_euv_tr_maps" in grp else None
        ),
        "nfev": int(grp.attrs.get("nfev", -1)),
        "nit": int(grp.attrs.get("nit", -1)),
        "message": decode_scalar(grp.attrs.get("message", b"")),
        "used_adaptive_bracketing": bool(grp.attrs.get("used_adaptive_bracketing", False)),
        "bracket_found": bool(grp.attrs.get("bracket_found", False)),
        "bracket": bracket,
        "target_metric": target_metric,
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
                message="point not stored in sparse artifact",
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
        "artifact_format": "sparse" if diagnostics.get("artifact_kind") == SPARSE_ARTIFACT_KIND else "rectangular",
    }


def load_scan_file(h5_path: Path, *, slice_key: str | None = None) -> dict[str, Any]:
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
        kind = _artifact_kind_from_group(group)
        if kind == SPARSE_ARTIFACT_KIND:
            point_records = _load_sparse_point_records(group["point_records"])
            target_metric = str(diagnostics.get("target_metric", "chi2"))
        else:
            point_records = _load_canonical_point_records(group)
            summary = group["summary"]
            target_metric = decode_scalar(summary.attrs.get("target_metric", diagnostics.get("target_metric", b"chi2")))
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
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    descriptor = slice_descriptor_from_diagnostics(diagnostics, fallback_key=slice_key or "default")
    resolved_slice_key = str(slice_key or "default")
    tmp_h5 = out_h5.with_suffix(out_h5.suffix + ".tmp")
    with _H5PY_FILE(tmp_h5, "w") as dst:
        slices_dst = dst.create_group(SLICE_CONTAINER_GROUP)
        if out_h5.exists():
            with _H5PY_FILE(out_h5, "r") as src:
                if SLICE_CONTAINER_GROUP in src:
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
        slice_group.attrs["domain"] = np.bytes_(str(descriptor["domain"]))
        slice_group.attrs["label"] = np.bytes_(str(descriptor["label"]))
        if descriptor.get("frequency_ghz") is not None:
            slice_group.attrs["frequency_ghz"] = float(descriptor["frequency_ghz"])
        if descriptor.get("channel_label") is not None:
            slice_group.attrs["channel_label"] = np.bytes_(str(descriptor["channel_label"]))

        common = slice_group.create_group("common")
        _write_common_group(
            common,
            observed=observed,
            sigma_map=sigma_map,
            wcs_header=wcs_header,
            diagnostics=diagnostics,
            blos_reference=blos_reference,
            run_history=run_history,
        )

        grid = slice_group.create_group("grid")
        grid.create_dataset("a_values", data=np.asarray(a_values, dtype=np.float64))
        grid.create_dataset("b_values", data=np.asarray(b_values, dtype=np.float64))

        summary = slice_group.create_group("summary")
        summary.create_dataset("best_q0", data=np.asarray(best_q0, dtype=np.float64))
        summary.create_dataset("objective_values", data=np.asarray(objective_values, dtype=np.float64))
        summary.create_dataset("chi2", data=np.asarray(chi2, dtype=np.float64))
        summary.create_dataset("rho2", data=np.asarray(rho2, dtype=np.float64))
        summary.create_dataset("eta2", data=np.asarray(eta2, dtype=np.float64))
        summary.create_dataset("success", data=np.asarray(success, dtype=np.uint8))
        summary.attrs["target_metric"] = np.bytes_(str(diagnostics.get("target_metric", "chi2")))

        points = slice_group.create_group("points")
        point_records = slice_group.create_group("point_records")
        for record_order, ((_ai, _bi), payload) in enumerate(sorted(point_payloads.items())):
            name = f"a{payload['a_index']:03d}_b{payload['b_index']:03d}"
            grp = points.create_group(name)
            grp.attrs["a"] = float(payload["a"])
            grp.attrs["b"] = float(payload["b"])
            grp.attrs["a_index"] = int(payload["a_index"])
            grp.attrs["b_index"] = int(payload["b_index"])
            grp.attrs["q0"] = float(payload["q0"])
            grp.attrs["success"] = int(bool(payload["success"]))
            grp.attrs["status"] = np.bytes_(str(payload.get("status", "computed")))
            grp.attrs["target_metric"] = np.bytes_(str(payload["target_metric"]))
            grp.create_dataset("modeled_best", data=np.asarray(payload["modeled_best"], dtype=np.float32), compression="gzip", compression_opts=4)
            grp.create_dataset("raw_modeled_best", data=np.asarray(payload["raw_modeled_best"], dtype=np.float32), compression="gzip", compression_opts=4)
            grp.create_dataset("residual", data=np.asarray(payload["residual"], dtype=np.float32), compression="gzip", compression_opts=4)
            grp.create_dataset("fit_q0_trials", data=np.asarray(payload["fit_q0_trials"], dtype=np.float64))
            fit_metric_ds = grp.create_dataset("fit_metric_trials", data=np.asarray(payload["fit_metric_trials"], dtype=np.float64))
            fit_metric_ds.attrs["target_metric"] = np.bytes_(str(payload["target_metric"]))
            grp.create_dataset("fit_chi2_trials", data=np.asarray(payload.get("fit_chi2_trials", ()), dtype=np.float64))
            grp.create_dataset("fit_rho2_trials", data=np.asarray(payload.get("fit_rho2_trials", ()), dtype=np.float64))
            grp.create_dataset("fit_eta2_trials", data=np.asarray(payload.get("fit_eta2_trials", ()), dtype=np.float64))
            _create_text_dataset(grp, "diagnostics_json", _json_dumps(payload["diagnostics"]))
            canonical_grp = point_records.create_group(f"r{record_order:06d}")
            _write_point_group(canonical_grp, payload, record_order=record_order)
    os.replace(tmp_h5, out_h5)


def _write_point_group(grp: h5py.Group, payload: dict[str, Any], *, record_order: int) -> None:
    normalized = _normalize_point_payload(payload, record_order=record_order)
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
    grp.create_dataset("modeled_best", data=np.asarray(normalized["modeled_best"], dtype=np.float32), compression="gzip", compression_opts=4)
    grp.create_dataset("raw_modeled_best", data=np.asarray(normalized["raw_modeled_best"], dtype=np.float32), compression="gzip", compression_opts=4)
    grp.create_dataset("residual", data=np.asarray(normalized["residual"], dtype=np.float32), compression="gzip", compression_opts=4)
    grp.create_dataset("fit_q0_trials", data=np.asarray(normalized["fit_q0_trials"], dtype=np.float64))
    fit_metric_ds = grp.create_dataset("fit_metric_trials", data=np.asarray(normalized["fit_metric_trials"], dtype=np.float64))
    fit_metric_ds.attrs["target_metric"] = np.bytes_(str(normalized["target_metric"]))
    grp.create_dataset("fit_chi2_trials", data=np.asarray(normalized["fit_chi2_trials"], dtype=np.float64))
    grp.create_dataset("fit_rho2_trials", data=np.asarray(normalized["fit_rho2_trials"], dtype=np.float64))
    grp.create_dataset("fit_eta2_trials", data=np.asarray(normalized["fit_eta2_trials"], dtype=np.float64))
    if normalized["trial_raw_modeled_maps"] is not None:
        grp.create_dataset(
            "trial_raw_modeled_maps",
            data=np.asarray(normalized["trial_raw_modeled_maps"], dtype=np.float32),
            compression="gzip",
            compression_opts=4,
        )
    if normalized["trial_modeled_maps"] is not None:
        grp.create_dataset(
            "trial_modeled_maps",
            data=np.asarray(normalized["trial_modeled_maps"], dtype=np.float32),
            compression="gzip",
            compression_opts=4,
        )
    if normalized["trial_residual_maps"] is not None:
        grp.create_dataset(
            "trial_residual_maps",
            data=np.asarray(normalized["trial_residual_maps"], dtype=np.float32),
            compression="gzip",
            compression_opts=4,
        )
    if normalized["euv_coronal_best"] is not None:
        grp.create_dataset(
            "euv_coronal_best",
            data=np.asarray(normalized["euv_coronal_best"], dtype=np.float32),
            compression="gzip",
            compression_opts=4,
        )
    if normalized["euv_tr_best"] is not None:
        grp.create_dataset(
            "euv_tr_best",
            data=np.asarray(normalized["euv_tr_best"], dtype=np.float32),
            compression="gzip",
            compression_opts=4,
        )
    if normalized["euv_tr_mask"] is not None:
        grp.create_dataset(
            "euv_tr_mask",
            data=np.asarray(normalized["euv_tr_mask"], dtype=np.uint8),
            compression="gzip",
            compression_opts=4,
        )
    if normalized["trial_euv_coronal_maps"] is not None:
        grp.create_dataset(
            "trial_euv_coronal_maps",
            data=np.asarray(normalized["trial_euv_coronal_maps"], dtype=np.float32),
            compression="gzip",
            compression_opts=4,
        )
    if normalized["trial_euv_tr_maps"] is not None:
        grp.create_dataset(
            "trial_euv_tr_maps",
            data=np.asarray(normalized["trial_euv_tr_maps"], dtype=np.float32),
            compression="gzip",
            compression_opts=4,
        )
    _create_text_dataset(grp, "diagnostics_json", _json_dumps(normalized["diagnostics"]))


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
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    tmp_h5 = out_h5.with_suffix(out_h5.suffix + ".tmp")
    diagnostics_out = dict(diagnostics)
    diagnostics_out["artifact_kind"] = SPARSE_ARTIFACT_KIND
    # Always record mask_type if present, else default to 'union'
    if "mask_type" not in diagnostics_out:
        diagnostics_out["mask_type"] = diagnostics.get("mask_type", "union")
    with _H5PY_FILE(tmp_h5, "w") as f:
        common = f.create_group("common")
        _write_common_group(
            common,
            observed=observed,
            sigma_map=sigma_map,
            wcs_header=wcs_header,
            diagnostics=diagnostics_out,
            blos_reference=blos_reference,
            run_history=run_history,
        )
        records_group = f.create_group("point_records")
        for record_order, payload in enumerate(point_records):
            grp = records_group.create_group(f"r{record_order:06d}")
            _write_point_group(grp, payload, record_order=record_order)
    os.replace(tmp_h5, out_h5)


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
    diagnostics_out["artifact_kind"] = SPARSE_ARTIFACT_KIND
    write_sparse_scan_file(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=wcs_header,
        diagnostics=diagnostics_out,
        blos_reference=blos_reference,
        point_records=[point_payload],
        run_history=run_history,
    )


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
    diagnostics_out = dict(diagnostics)
    diagnostics_out["artifact_kind"] = SPARSE_ARTIFACT_KIND
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if out_h5.exists() else "w"
    last_exc: Exception | None = None
    for attempt in range(1, _SPARSE_APPEND_RETRY_ATTEMPTS + 1):
        try:
            with h5py.File(out_h5, mode) as f:
                if "common" not in f:
                    common = f.create_group("common")
                    _write_common_group(
                        common,
                        observed=observed,
                        sigma_map=sigma_map,
                        wcs_header=wcs_header,
                        diagnostics=diagnostics_out,
                        blos_reference=blos_reference,
                        run_history=None,
                    )
                elif blos_reference is not None and "refmaps" not in f["common"]:
                    refmaps = f["common"].create_group("refmaps")
                    blos_data, blos_header = blos_reference
                    _write_reference_map_group(
                        refmaps,
                        group_name="Bz_reference",
                        data=np.asarray(blos_data, dtype=float),
                        wcs_header=blos_header,
                    )
                records_group = f.require_group("point_records")
                existing_orders = [int(records_group[name].attrs.get("record_order", -1)) for name in records_group.keys()]
                next_order = max(existing_orders, default=-1) + 1
                grp = records_group.create_group(f"r{next_order:06d}")
                _write_point_group(grp, point_payload, record_order=next_order)
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
    if artifact_kind == SPARSE_ARTIFACT_KIND:
        append_sparse_point_record(
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
    if str(payload.get("artifact_format", "")) == "sparse":
        append_sparse_point_record(
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

    save_rectangular_scan_file(
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
    write_sparse_scan_file(
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
