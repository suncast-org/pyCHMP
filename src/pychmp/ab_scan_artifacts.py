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
from scipy.signal import fftconvolve

from .search_contract import (
    OBSERVATION_REF_GROUP,
    _observation_ref_diagnostics_from_full,
    build_search_evaluation_config,
    search_evaluation_signature,
    search_id_from_evaluation_config,
)
from .obs_preprocessing import compute_array_content_sha256


METRICS = ("chi2", "rho2", "eta2")


def viewer_record_metrics(record: dict[str, Any]) -> dict[str, float]:
    """Derive per-metric best values from stored trial histories."""
    metrics = {name: float("nan") for name in METRICS}
    q0_trials = tuple(record.get("fit_q0_trials", ()))
    q0_size = len(q0_trials)
    if q0_size <= 0:
        return metrics
    for name in METRICS:
        trials = np.asarray(record.get(f"fit_{name}_trials", ()), dtype=float)
        if trials.size == q0_size and np.any(np.isfinite(trials)):
            metrics[name] = float(np.nanmin(trials))
    target_metric = str(record.get("target_metric", "") or "").strip().lower()
    if target_metric in METRICS:
        fit_metric_trials = np.asarray(record.get("fit_metric_trials", ()), dtype=float)
        if fit_metric_trials.size == q0_size and np.any(np.isfinite(fit_metric_trials)):
            metrics[target_metric] = float(np.nanmin(fit_metric_trials))
    return metrics
UNIFIED_ARTIFACT_KIND = "pychmp_ab_scan_unified"
RECTANGULAR_ARTIFACT_KIND = "pychmp_ab_scan"
SPARSE_ARTIFACT_KIND = "pychmp_ab_scan_sparse_points"
SLICE_CONTAINER_GROUP = "slices"
RUN_HISTORY_DATASET = "run_history_json"
COMMON_SLICE_DESCRIPTORS_DATASET = "slice_descriptors_json"
COMMON_TARGET_SLICE_KEY_DATASET = "target_slice_key"
COMMON_TRIAL_LOGGING_POLICY_DATASET = "trial_logging_policy_json"
COMMON_ARTIFACT_CONTRACT_VERSION_DATASET = "artifact_contract_version"
COMMON_PSF_KERNEL_DATASET = "psf_kernel"
COMMON_PSF_KERNEL_META_DATASET = "psf_kernel_meta_json"
SEARCHES_GROUP = "searches"
ACTIVE_SEARCH_ID_DATASET = "active_search_id"
ACTIVE_POINT_SNAPSHOT_GROUP = "active_point_snapshot"
LIVE_TRIAL_POINT_GROUP = "live_trial_point"
SEARCH_REQUEST_DATASET = "request_json"
SEARCH_LIFECYCLE_DATASET = "lifecycle_json"
MAP_STORE_GROUP = "map_store"
MAP_STORE_MAPS_GROUP = "maps"
MAP_STORE_SYNTHETIC_REGISTRY_GROUP = "synthetic_registry"
MAP_REFS_DATASET = "map_refs_json"
TRIAL_HISTORY_DATASET = "trial_history_json"
POINT_SYNTHETIC_MAP_MACHINE_KEYS_DATASET = "synthetic_map_machine_keys_json"
CANONICAL_ARTIFACT_CONTRACT_VERSION = "2026-05-28-slice-shared-canvas"
MAP_IDENTITY_SCHEMA = "pychmp.map_identity.v1"
ARTIFACT_GEOMETRY_SCHEMA = "pychmp.artifact_geometry.v1"
FORWARD_MODEL_IDENTITY_VERSION = "pychmp.forward_model.file_sha256.v0"
COMMON_ARTIFACT_GEOMETRY_DATASET = "artifact_geometry_json"
COMMON_ARTIFACT_GEOMETRY_SHA256_DATASET = "artifact_geometry_sha256"
REQUIRED_COMPATIBILITY_DIAGNOSTIC_KEYS = (
    "artifact_kind",
    "target_metric",
    "model_sha256",
    "forward_model_sha256",
    "forward_model_identity_version",
    "artifact_geometry_sha256",
    "fits_sha256",
    "ebtel_sha256",
    "euv_response_identity_version",
    "euv_response_sha256",
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
PREFLIGHT_COMPATIBILITY_DIAGNOSTIC_KEYS = (
    "artifact_kind",
    "model_sha256",
    "forward_model_sha256",
    "forward_model_identity_version",
    "artifact_geometry_sha256",
    "ebtel_sha256",
    "euv_response_identity_version",
    "euv_response_sha256",
    "spectral_domain",
    "spectral_label",
    "frequency_ghz",
    "wavelength_angstrom",
    "euv_channel",
    "euv_instrument",
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
)
COMPATIBILITY_SIGNATURE_KEY = "compatibility_signature"
SEARCH_SPECIFIC_DIAGNOSTIC_KEYS = {
    COMPATIBILITY_SIGNATURE_KEY,
    "search_instance_id",
    "target_metric",
    "metrics_mask_threshold",
    "metrics_mask_source",
    "metrics_mask_fits",
    "mask_type",
    "tr_mask_bmin_gauss",
    "tr_mask_source",
    "search_mode",
    "shift_policy",
    "max_shift_arcsec",
    "xy_shift_arcsec",
    "use_smoothed_obs_max",
    "use_emthreshold",
    "emthreshold",
    "q0_search_stages",
}


class ScanArtifactCompatibilityError(ValueError):
    """Raised when an existing scan artifact cannot be safely reused."""


_SPARSE_APPEND_RETRY_ATTEMPTS = 40
_SPARSE_APPEND_RETRY_DELAY_S = 0.25


def _is_h5_locking_flag_mismatch(exc: BaseException) -> bool:
    if not isinstance(exc, OSError):
        return False
    message = str(exc).lower()
    return "locking" in message and ("don't match" in message or "do not match" in message)


def is_h5_transient_read_error(exc: BaseException) -> bool:
    """True for HDF5 read races (viewer + writer, Dropbox) that may succeed on retry."""
    if isinstance(exc, RuntimeError):
        message = str(exc).lower()
        return "bad symbol table" in message or "symbol table node" in message
    if isinstance(exc, OSError):
        message = str(exc).lower()
        return (
            "unable to synchronously" in message
            or "resource temporarily unavailable" in message
            or _is_h5_locking_flag_mismatch(exc)
        )
    return False


def _open_h5_with_lock_tolerance(path: Path | str, mode: str = "r", *args: Any, **kwargs: Any) -> h5py.File:
    if "locking" in kwargs:
        return h5py.File(path, mode, *args, **kwargs)

    text_mode = str(mode)
    if text_mode in {"r", "r+"}:
        last_exc: OSError | None = None
        for locking in (False, True):
            try:
                return h5py.File(path, mode, *args, locking=locking, **kwargs)
            except TypeError:
                return h5py.File(path, mode, *args, **kwargs)
            except OSError as exc:
                if _is_h5_locking_flag_mismatch(exc):
                    last_exc = exc
                    continue
                raise
        if last_exc is not None:
            raise last_exc
    return h5py.File(path, mode, *args, **kwargs)


_H5PY_FILE = _open_h5_with_lock_tolerance


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


def build_artifact_geometry_block(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Canonical observer/FOV/WCS block stored once per artifact."""

    return {
        "schema": ARTIFACT_GEOMETRY_SCHEMA,
        "map_xc_arcsec": float(diagnostics.get("map_xc_arcsec", np.nan)),
        "map_yc_arcsec": float(diagnostics.get("map_yc_arcsec", np.nan)),
        "map_dx_arcsec": float(diagnostics.get("map_dx_arcsec", np.nan)),
        "map_dy_arcsec": float(diagnostics.get("map_dy_arcsec", np.nan)),
        "map_nx": int(diagnostics.get("map_nx", 0)),
        "map_ny": int(diagnostics.get("map_ny", 0)),
        "observer_name": diagnostics.get("observer_name"),
        "observer_lonc_deg": diagnostics.get("observer_lonc_deg"),
        "observer_b0sun_deg": diagnostics.get("observer_b0sun_deg"),
        "observer_dsun_cm": diagnostics.get("observer_dsun_cm"),
        "observer_obs_time": diagnostics.get("observer_obs_time"),
    }


def artifact_geometry_sha256(geometry_block: dict[str, Any]) -> str:
    canonical = json.dumps(geometry_block, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_map_identity(
    *,
    a: float,
    b: float,
    q0: float,
    domain: str,
    channel_or_frequency: str,
    component: str,
    forward_model_sha256: str,
    ebtel_sha256: str,
    artifact_geometry_sha256: str,
    forward_model_identity_version: str = FORWARD_MODEL_IDENTITY_VERSION,
    euv_response_sha256: str | None = None,
    euv_response_identity_version: str | None = None,
    array_name: str | None = None,
) -> dict[str, Any]:
    identity = {
        "schema": MAP_IDENTITY_SCHEMA,
        "forward_model_sha256": str(forward_model_sha256),
        "forward_model_identity_version": str(forward_model_identity_version),
        "ebtel_sha256": str(ebtel_sha256),
        "artifact_geometry_sha256": str(artifact_geometry_sha256),
        "a": float(a),
        "b": float(b),
        "q0": float(q0),
        "domain": str(domain).strip().lower(),
        "channel_or_frequency": str(channel_or_frequency),
        "component": str(component).strip().lower(),
    }
    if array_name is not None:
        identity["array_name"] = str(array_name)
    if euv_response_sha256 is not None:
        identity["euv_response_sha256"] = str(euv_response_sha256)
    if euv_response_identity_version is not None:
        identity["euv_response_identity_version"] = str(euv_response_identity_version)
    return identity


def _recombine_euv_raw_maps(
    flux_corona: np.ndarray,
    flux_tr: np.ndarray,
    *,
    tr_region_mask: np.ndarray | None,
) -> np.ndarray:
    cor = np.asarray(flux_corona, dtype=float)
    tr = np.asarray(flux_tr, dtype=float)
    if tr_region_mask is None:
        return cor + tr
    mask = np.asarray(tr_region_mask, dtype=bool)
    if mask.shape != cor.shape:
        return cor + tr
    return cor + (tr * mask.astype(float))


def _derive_euv_raw_best_from_components(record: dict[str, Any]) -> np.ndarray | None:
    corona = record.get("euv_coronal_best")
    tr_flux = record.get("euv_tr_best")
    if corona is None or tr_flux is None:
        return None
    return _recombine_euv_raw_maps(
        np.asarray(corona, dtype=float),
        np.asarray(tr_flux, dtype=float),
        tr_region_mask=record.get("euv_tr_mask"),
    )


def _derive_euv_trial_raw_from_components(record: dict[str, Any]) -> np.ndarray | None:
    corona = record.get("trial_euv_coronal_maps")
    tr_flux = record.get("trial_euv_tr_maps")
    if corona is None or tr_flux is None:
        return None
    cor = np.asarray(corona, dtype=float)
    tr = np.asarray(tr_flux, dtype=float)
    if cor.ndim != 3 or tr.ndim != 3 or cor.shape != tr.shape:
        return None
    mask = record.get("euv_tr_mask")
    if mask is None:
        return cor + tr
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.ndim != 2 or mask_arr.shape != cor.shape[1:]:
        return cor + tr
    return cor + (tr * mask_arr.astype(float)[None, :, :])


def _component_from_array_name(name: str) -> str | None:
    lowered = str(name).strip().lower()
    mapping = {
        "euv_coronal_best": "corona",
        "trial_euv_coronal_maps": "corona",
        "euv_tr_best": "tr",
        "trial_euv_tr_maps": "tr",
        "stokes_v_best": "stokes_v",
        "trial_stokes_v_maps": "stokes_v",
        "raw_modeled_best": "stokes_i",
        "trial_raw_modeled_maps": "stokes_i",
    }
    if lowered in mapping:
        return mapping[lowered]
    if lowered.startswith("trial_raw_modeled_maps/"):
        return "stokes_i"
    return None


def _domain_from_diagnostics(diagnostics: dict[str, Any]) -> str:
    domain = str(diagnostics.get("spectral_domain", "")).strip().lower()
    if domain in {"mw", "euv", "uv"}:
        return domain
    return "unknown"


def _channel_or_frequency_from_diagnostics(diagnostics: dict[str, Any]) -> str:
    domain = _domain_from_diagnostics(diagnostics)
    if domain == "mw":
        freq = _optional_float(diagnostics.get("frequency_ghz"))
        if freq is not None:
            return f"{float(freq):.6f}ghz"
    channel = str(diagnostics.get("euv_channel") or diagnostics.get("spectral_label") or "").strip()
    if channel:
        return channel
    wavelength = _optional_float(diagnostics.get("wavelength_angstrom"))
    if wavelength is not None:
        rounded = round(float(wavelength))
        if np.isclose(float(wavelength), float(rounded), rtol=0.0, atol=1e-9):
            return str(int(rounded))
        return f"{float(wavelength):.6g}"
    return str(diagnostics.get("spectral_label") or "unknown")


def _map_identity_physical_keys_from_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in (
        "forward_model_sha256",
        "forward_model_identity_version",
        "artifact_geometry_sha256",
        "ebtel_sha256",
        "euv_response_sha256",
        "euv_response_identity_version",
    ):
        if key in diagnostics and str(diagnostics.get(key, "")).strip():
            out[key] = diagnostics[key]
    return out


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


def _normalize_compatibility_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    out = dict(diagnostics)
    if not str(out.get("artifact_geometry_sha256", "")).strip():
        geometry_block = build_artifact_geometry_block(out)
        out["artifact_geometry_sha256"] = artifact_geometry_sha256(geometry_block)
    if not str(out.get("forward_model_sha256", "")).strip() and str(out.get("model_sha256", "")).strip():
        out["forward_model_sha256"] = str(out["model_sha256"])
    if not str(out.get("forward_model_identity_version", "")).strip():
        out["forward_model_identity_version"] = FORWARD_MODEL_IDENTITY_VERSION
    return out


def _normalize_path_like(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    return os.path.normcase(os.path.normpath(text))
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


def _normalize_observation_time_unix(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    from datetime import datetime, timezone

    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%d-%b-%Y %H:%M:%S.%f",
        "%d-%b-%Y %H:%M:%S",
    ):
        try:
            parsed = datetime.strptime(text, fmt)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return float(parsed.timestamp())
        except ValueError:
            continue
    try:
        from dateutil.parser import parse as parse_datetime

        parsed = parse_datetime(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return float(parsed.timestamp())
    except Exception:
        return None


def _diagnostic_values_match(key: str, existing: Any, current: Any) -> bool:
    if key == "artifact_kind":
        existing_text = str(existing)
        current_text = str(current)
        if UNIFIED_ARTIFACT_KIND in {existing_text, current_text} and {
            existing_text,
            current_text,
        } & {RECTANGULAR_ARTIFACT_KIND, SPARSE_ARTIFACT_KIND}:
            return True
    if key == "observer_obs_time":
        existing_ts = _normalize_observation_time_unix(existing)
        current_ts = _normalize_observation_time_unix(current)
        if existing_ts is not None and current_ts is not None:
            return bool(np.isclose(existing_ts, current_ts, rtol=0.0, atol=1.0))
        return str(existing) == str(current)
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


def _effective_preprocessed_content_sha256(stored_sha: str, array: np.ndarray) -> str:
    """Use array content when stored metadata hashes drift from common map datasets."""
    from .obs_preprocessing import artifact_storage_content_sha256

    array_sha = artifact_storage_content_sha256(array)
    stored_text = str(stored_sha or "").strip()
    if stored_text and stored_text == array_sha:
        return stored_text
    return array_sha


def _sync_preprocessed_content_identity_diagnostics(
    diagnostics: dict[str, Any],
    *,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    observation_canvas: np.ndarray | None = None,
    sigma_canvas: np.ndarray | None = None,
) -> dict[str, Any]:
    from .obs_preprocessing import artifact_storage_content_sha256

    diagnostics_out = dict(diagnostics)
    observed_store = np.asarray(observed, dtype=np.float32)
    sigma_store = np.asarray(sigma_map, dtype=np.float32)
    diagnostics_out["preprocessed_observation_sha256"] = artifact_storage_content_sha256(observed_store)
    diagnostics_out["preprocessed_sigma_sha256"] = artifact_storage_content_sha256(sigma_store)
    diagnostics_out["preprocessed_observation_shape"] = [int(v) for v in observed_store.shape]
    diagnostics_out["preprocessed_sigma_shape"] = [int(v) for v in sigma_store.shape]
    if observation_canvas is not None and sigma_canvas is not None:
        canvas_obs = np.asarray(observation_canvas, dtype=np.float32)
        canvas_sigma = np.asarray(sigma_canvas, dtype=np.float32)
        diagnostics_out["observation_canvas_sha256"] = artifact_storage_content_sha256(canvas_obs)
        diagnostics_out["sigma_canvas_sha256"] = artifact_storage_content_sha256(canvas_sigma)
        diagnostics_out["observation_canvas_shape"] = [int(v) for v in canvas_obs.shape]
        diagnostics_out["sigma_canvas_shape"] = [int(v) for v in canvas_sigma.shape]
    return diagnostics_out


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
    existing_sigma = np.asarray(payload.get("sigma_map"), dtype=float)
    current_sigma = np.asarray(sigma_map, dtype=float)
    existing_diagnostics = _normalize_compatibility_diagnostics(dict(payload.get("diagnostics", {})))
    current_diagnostics = _normalize_compatibility_diagnostics(dict(diagnostics))
    existing_obs_sha = str(existing_diagnostics.get("preprocessed_observation_sha256", "")).strip()
    current_obs_sha = str(current_diagnostics.get("preprocessed_observation_sha256", "")).strip()
    existing_sigma_sha = str(existing_diagnostics.get("preprocessed_sigma_sha256", "")).strip()
    current_sigma_sha = str(current_diagnostics.get("preprocessed_sigma_sha256", "")).strip()
    if existing_obs_sha and current_obs_sha and existing_sigma_sha and current_sigma_sha:
        existing_obs_sha = _effective_preprocessed_content_sha256(existing_obs_sha, existing_observed)
        existing_sigma_sha = _effective_preprocessed_content_sha256(existing_sigma_sha, existing_sigma)
        current_obs_sha = _effective_preprocessed_content_sha256(current_obs_sha, current_observed)
        current_sigma_sha = _effective_preprocessed_content_sha256(current_sigma_sha, current_sigma)
        if existing_obs_sha != current_obs_sha:
            issues.append(
                "preprocessed observation identity differs from the stored artifact "
                f"(stored={existing_obs_sha!r}, current={current_obs_sha!r})"
            )
        if existing_sigma_sha != current_sigma_sha:
            issues.append(
                "preprocessed sigma identity differs from the stored artifact "
                f"(stored={existing_sigma_sha!r}, current={current_sigma_sha!r})"
            )
    else:
        if not _arrays_match_for_reuse(existing_observed, current_observed):
            issues.append(
                "observed map differs from the stored artifact "
                f"(stored shape={existing_observed.shape}, current shape={current_observed.shape})"
            )
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

    artifact_kind = str(existing_diagnostics.get("artifact_kind", ""))
    # Only require observer/FOV fields if present in both stored and current diagnostics
    for key in REQUIRED_COMPATIBILITY_DIAGNOSTIC_KEYS:
        if artifact_kind in {SPARSE_ARTIFACT_KIND, UNIFIED_ARTIFACT_KIND} and key in SEARCH_SPECIFIC_DIAGNOSTIC_KEYS:
            continue
        if key not in existing_diagnostics and key not in current_diagnostics:
            continue  # treat as optional if missing in both
        if key not in existing_diagnostics:
            issues.append(f"stored artifact is missing required diagnostic '{key}'")
            continue
        if key not in current_diagnostics:
            issues.append(f"current run is missing required diagnostic '{key}'")
            continue
        if not _diagnostic_values_match(key, existing_diagnostics[key], current_diagnostics[key]):
            issues.append(
                f"diagnostic mismatch for '{key}' "
                f"(stored={existing_diagnostics[key]!r}, current={current_diagnostics[key]!r})"
            )

    # Rectangular artifacts represent a single coherent run, so require an exact
    # command-signature match when present. Sparse artifacts may intentionally
    # accumulate multiple runs and therefore filter incompatible point records
    # during hydration instead of rejecting the whole file.
    if artifact_kind not in {SPARSE_ARTIFACT_KIND, UNIFIED_ARTIFACT_KIND}:
        existing_signature = str(existing_diagnostics.get(COMPATIBILITY_SIGNATURE_KEY, "")).strip()
        current_signature = str(current_diagnostics.get(COMPATIBILITY_SIGNATURE_KEY, "")).strip()
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


def scan_artifact_reuse_preflight_issues(
    payload: dict[str, Any],
    *,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
) -> list[str]:
    issues: list[str] = []

    existing_header = payload.get("wcs_header")
    if not isinstance(existing_header, fits.Header):
        issues.append("stored artifact is missing a valid WCS header")
    elif _canonical_header_text(existing_header) != _canonical_header_text(wcs_header):
        issues.append("WCS header differs from the stored artifact")

    existing_diagnostics = _normalize_compatibility_diagnostics(dict(payload.get("diagnostics", {})))
    current_diagnostics = _normalize_compatibility_diagnostics(dict(diagnostics))
    for key in PREFLIGHT_COMPATIBILITY_DIAGNOSTIC_KEYS:
        if key not in existing_diagnostics and key not in current_diagnostics:
            continue
        if key not in existing_diagnostics:
            issues.append(f"stored artifact is missing required diagnostic '{key}'")
            continue
        if key not in current_diagnostics:
            issues.append(f"current run is missing required diagnostic '{key}'")
            continue
        if not _diagnostic_values_match(key, existing_diagnostics[key], current_diagnostics[key]):
            issues.append(
                f"diagnostic mismatch for '{key}' "
                f"(stored={existing_diagnostics[key]!r}, current={current_diagnostics[key]!r})"
            )

    return issues


def validate_scan_artifact_reuse_preflight(
    payload: dict[str, Any],
    *,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
    artifact_path: Path | None = None,
) -> None:
    issues = scan_artifact_reuse_preflight_issues(
        payload,
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


def _search_id_from_diagnostics(
    diagnostics: dict[str, Any],
    *,
    fallback: str = "search",
    layout: dict[str, Any] | None = None,
) -> str:
    return search_id_from_evaluation_config(diagnostics, fallback=fallback, layout=layout)


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


def _search_should_remain_in_progress(
    *,
    diagnostics: dict[str, Any],
    existing_lifecycle: dict[str, Any] | None = None,
) -> bool:
    if bool(diagnostics.get("search_active")):
        return True
    existing = dict(existing_lifecycle or {})
    return bool(existing.get("active")) and not str(existing.get("completed_at") or "").strip()


def _write_search_status_attrs(
    search_group: h5py.Group,
    counts: dict[str, int],
    *,
    diagnostics: dict[str, Any],
    remain_in_progress: bool = False,
) -> str:
    status = _search_status_from_counts(counts)
    if remain_in_progress and status == "complete":
        status = "in_progress"
    for key, value in counts.items():
        search_group.attrs[f"{key}_point_count"] = int(value)
    search_group.attrs["status"] = np.bytes_(status)
    search_group.attrs["label"] = np.bytes_(_search_label_from_diagnostics(diagnostics, status=status))
    search_group.attrs["in_progress"] = int(status in {"empty", "in_progress", "partial"})
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
    return build_search_evaluation_config(diagnostics, layout=layout)


def _search_request_from_group(search_group: h5py.Group) -> dict[str, Any]:
    if SEARCH_REQUEST_DATASET in search_group:
        payload = _json_loads_or_empty(search_group[SEARCH_REQUEST_DATASET][()])
        if payload:
            return payload
    diagnostics = _json_loads_or_empty(search_group["diagnostics_json"][()]) if "diagnostics_json" in search_group else {}
    layout = _json_loads_or_empty(search_group["layout_json"][()]) if "layout_json" in search_group else {}
    return _search_request_from_diagnostics(diagnostics, layout=layout)


def _matching_search_id_for_request(searches_group: h5py.Group, request: dict[str, Any]) -> str | None:
    target_signature = search_evaluation_signature(request)
    matches: list[tuple[int, str]] = []
    for search_id in searches_group.keys():
        search_group = searches_group[search_id]
        existing_request = _search_request_from_group(search_group)
        if search_evaluation_signature(existing_request) != target_signature:
            continue
        point_count = len(search_group["point_records"]) if "point_records" in search_group else 0
        from .grid_points import GRID_POINTS_GROUP

        if GRID_POINTS_GROUP in search_group:
            point_count = max(int(point_count), int(len(search_group[GRID_POINTS_GROUP])))
        matches.append((int(point_count), str(search_id)))
    if not matches:
        return None
    # Prefer the most complete existing search when duplicates already exist.
    matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return matches[0][1]


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
    if bool(diagnostics.get("search_active")):
        completed_at = None
    else:
        completed_at = diagnostics.get("search_completed_at", previous.get("completed_at"))
    in_progress = status in {"empty", "in_progress", "partial"}
    if "search_active" in diagnostics:
        active = bool(diagnostics.get("search_active"))
    elif "active" in previous:
        # Preserve the existing active state across incremental point appends.
        # The runner sets search_active=False explicitly when it exits; we must
        # not infer completion just because all current grid points are computed
        # (the adaptive search may continue exploring new regions).
        active = bool(previous["active"])
    elif existing is not None:
        # Incremental (append) write, first record: the runner is creating this
        # search so it is active regardless of in_progress status.
        active = True
    else:
        # Batch write (_write_search_group): infer activity from completion
        # status.  A fully-computed batch-written search has no live runner.
        active = bool(in_progress)
    if active and status == "complete":
        status = "in_progress"
        in_progress = True
    if not in_progress and not active and not completed_at:
        completed_at = _utc_now_iso()
    return {
        "status": str(status),
        "active": bool(active),
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


def _write_observation_ref_group(
    ref_group: h5py.Group,
    *,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
    observation_canvas: np.ndarray | None = None,
    sigma_canvas: np.ndarray | None = None,
    canvas_wcs_header: fits.Header | None = None,
) -> None:
    ref_group.create_dataset(
        "observed",
        data=np.asarray(observed, dtype=np.float32),
        compression="gzip",
        compression_opts=4,
    )
    ref_group.create_dataset(
        "sigma_map",
        data=np.asarray(sigma_map, dtype=np.float32),
        compression="gzip",
        compression_opts=4,
    )
    if observation_canvas is not None and sigma_canvas is not None and canvas_wcs_header is not None:
        ref_group.create_dataset(
            "observation_canvas",
            data=np.asarray(observation_canvas, dtype=np.float32),
            compression="gzip",
            compression_opts=4,
        )
        ref_group.create_dataset(
            "sigma_canvas",
            data=np.asarray(sigma_canvas, dtype=np.float32),
            compression="gzip",
            compression_opts=4,
        )
        _create_text_dataset(
            ref_group,
            "canvas_wcs_header",
            canvas_wcs_header.tostring(sep="\n", endcard=True),
        )
    _create_text_dataset(ref_group, "wcs_header", wcs_header.tostring(sep="\n", endcard=True))
    _create_text_dataset(ref_group, "diagnostics_json", _json_dumps(dict(diagnostics)))


def _read_observation_ref_group(ref_group: h5py.Group) -> dict[str, Any]:
    observed = np.asarray(ref_group["observed"], dtype=float) if "observed" in ref_group else None
    sigma_map = np.asarray(ref_group["sigma_map"], dtype=float) if "sigma_map" in ref_group else None
    wcs_header = (
        fits.Header.fromstring(decode_scalar(ref_group["wcs_header"][()]), sep="\n")
        if "wcs_header" in ref_group
        else None
    )
    observation_canvas = (
        np.asarray(ref_group["observation_canvas"], dtype=float)
        if "observation_canvas" in ref_group
        else None
    )
    sigma_canvas = (
        np.asarray(ref_group["sigma_canvas"], dtype=float) if "sigma_canvas" in ref_group else None
    )
    canvas_wcs_header = (
        fits.Header.fromstring(decode_scalar(ref_group["canvas_wcs_header"][()]), sep="\n")
        if "canvas_wcs_header" in ref_group
        else None
    )
    diagnostics = (
        json.loads(decode_scalar(ref_group["diagnostics_json"][()]))
        if "diagnostics_json" in ref_group
        else {}
    )
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    return {
        "observed": observed,
        "sigma_map": sigma_map,
        "wcs_header": wcs_header,
        "observation_canvas": observation_canvas,
        "sigma_canvas": sigma_canvas,
        "canvas_wcs_header": canvas_wcs_header,
        "diagnostics": diagnostics,
    }


def _observation_ref_payload_from_common(common: h5py.Group) -> dict[str, Any] | None:
    payload = _read_common_group(common)
    has_model_fov = payload.get("observed") is not None and payload.get("sigma_map") is not None
    has_canvas = payload.get("observation_canvas") is not None and payload.get("sigma_canvas") is not None
    if not has_model_fov and not has_canvas:
        return None
    return {
        "observed": payload.get("observed"),
        "sigma_map": payload.get("sigma_map"),
        "wcs_header": payload.get("wcs_header"),
        "observation_canvas": payload.get("observation_canvas"),
        "sigma_canvas": payload.get("sigma_canvas"),
        "canvas_wcs_header": payload.get("canvas_wcs_header"),
        "diagnostics": dict(payload.get("diagnostics") or {}),
    }


def _resolve_observation_reference_payload(
    common: h5py.Group,
    *,
    search_group: h5py.Group | None = None,
) -> dict[str, Any] | None:
    if search_group is not None and OBSERVATION_REF_GROUP in search_group:
        search_payload = _read_observation_ref_group(search_group[OBSERVATION_REF_GROUP])
        if search_payload.get("observed") is not None or search_payload.get("observation_canvas") is not None:
            return search_payload
    payload = _observation_ref_payload_from_common(common)
    if payload is not None and (
        payload.get("observed") is not None
        or payload.get("observation_canvas") is not None
    ):
        return payload
    return payload

def load_slice_observation_reference_payload(
    h5_path: Path,
    *,
    slice_key: str | None = None,
    search_id: str | None = None,
) -> dict[str, Any] | None:
    """Load slice-shared observation reference maps, with legacy per-search fallback."""
    with _H5PY_FILE(h5_path, "r") as f:
        group, _descriptors, _selected_key = _resolve_slice_group(
            f,
            slice_key=slice_key,
            allow_missing=slice_key is not None,
        )
        if group is None:
            return None
        common = group.get("common")
        if common is None:
            return None
        search_group = None
        if search_id and SEARCHES_GROUP in group and search_id in group[SEARCHES_GROUP]:
            search_group = group[SEARCHES_GROUP][search_id]
        payload = _resolve_observation_reference_payload(common, search_group=search_group)
        if payload is not None and (
            payload.get("observed") is not None or payload.get("observation_canvas") is not None
        ):
            return payload
        if SEARCHES_GROUP in group:
            for name in sorted(group[SEARCHES_GROUP].keys()):
                candidate = group[SEARCHES_GROUP][name]
                payload = _resolve_observation_reference_payload(common, search_group=candidate)
                if payload is not None and (
                    payload.get("observed") is not None or payload.get("observation_canvas") is not None
                ):
                    return payload
        return _observation_ref_payload_from_common(common)


def load_search_observation_reference_payload(
    h5_path: Path,
    *,
    slice_key: str | None = None,
    search_id: str | None = None,
) -> dict[str, Any] | None:
    """Deprecated alias for :func:`load_slice_observation_reference_payload`."""
    return load_slice_observation_reference_payload(
        h5_path,
        slice_key=slice_key,
        search_id=search_id,
    )


def _write_search_group(
    searches_group: h5py.Group,
    *,
    search_id: str,
    diagnostics: dict[str, Any],
    point_records: list[dict[str, Any]],
    run_history: list[dict[str, Any]] | None,
    layout: dict[str, Any] | None = None,
    observation_ref: dict[str, Any] | None = None,
) -> None:
    if search_id in searches_group:
        del searches_group[search_id]
    search_group = searches_group.create_group(search_id)
    diagnostics_out = dict(diagnostics)
    diagnostics_out["search_id"] = str(search_id)
    diagnostics_out["selected_search_id"] = str(search_id)
    counts = _search_status_counts_from_records(point_records)
    status = _search_status_from_counts(counts)
    search_group.attrs["search_id"] = np.bytes_(str(search_id))
    search_group.attrs["target_metric"] = np.bytes_(str(diagnostics.get("target_metric", "chi2")))
    _write_search_status_attrs(search_group, counts, diagnostics=diagnostics_out)
    _create_text_dataset(search_group, "diagnostics_json", _json_dumps(diagnostics_out))
    _create_text_dataset(search_group, "layout_json", _json_dumps(layout or {}))
    _create_text_dataset(search_group, "run_history_json", _json_dumps(list(run_history or [])))
    records_group = search_group.create_group("point_records")
    for record_order, payload in enumerate(point_records):
        grp = records_group.create_group(f"r{record_order:06d}")
        _write_point_group(grp, payload, record_order=record_order)
    request = _search_request_from_diagnostics(diagnostics_out, layout=layout)
    lifecycle = _search_lifecycle_payload(status=status, diagnostics=diagnostics_out)
    _create_text_dataset(search_group, SEARCH_REQUEST_DATASET, _json_dumps(request))
    _write_search_lifecycle_dataset(search_group, lifecycle=lifecycle)
    if observation_ref is not None:
        ref_group = search_group.create_group(OBSERVATION_REF_GROUP)
        ref_diag = dict(observation_ref.get("diagnostics") or diagnostics_out)
        ref_diag.update(_observation_ref_diagnostics_from_full(ref_diag))
        _write_observation_ref_group(
            ref_group,
            observed=np.asarray(observation_ref["observed"], dtype=float),
            sigma_map=np.asarray(observation_ref["sigma_map"], dtype=float),
            wcs_header=observation_ref["wcs_header"],
            diagnostics=ref_diag,
            observation_canvas=observation_ref.get("observation_canvas"),
            sigma_canvas=observation_ref.get("sigma_canvas"),
            canvas_wcs_header=observation_ref.get("canvas_wcs_header"),
        )


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


def read_slice_active_search_id(h5_path: Path, *, slice_key: str) -> str | None:
    """Return the active search id recorded for a slice, if present."""
    with _H5PY_FILE(h5_path, "r") as f:
        if SLICE_CONTAINER_GROUP not in f or str(slice_key) not in f[SLICE_CONTAINER_GROUP]:
            return None
        return _selected_search_id(f[SLICE_CONTAINER_GROUP][str(slice_key)])


def purge_search_from_artifact(
    h5_path: Path,
    *,
    slice_key: str,
    search_id: str,
) -> dict[str, Any]:
    """Remove one search metadata branch from a slice without touching ``map_store``.

    Deletes ``slices/<slice>/searches/<search_id>`` and clears ``active_search_id`` when
    it pointed at the purged search. Shared slice ``common/`` data is left intact.
    """
    resolved_slice = str(slice_key).strip()
    resolved_search = str(search_id).strip()
    if not resolved_slice or not resolved_search:
        raise ValueError("slice_key and search_id are required")
    with _H5PY_FILE(h5_path, "a") as f:
        if SLICE_CONTAINER_GROUP not in f or resolved_slice not in f[SLICE_CONTAINER_GROUP]:
            raise KeyError(f"slice not found: {resolved_slice}")
        slice_group = f[SLICE_CONTAINER_GROUP][resolved_slice]
        if SEARCHES_GROUP not in slice_group or resolved_search not in slice_group[SEARCHES_GROUP]:
            raise KeyError(f"search not found: {resolved_search}")
        del slice_group[SEARCHES_GROUP][resolved_search]
        cleared_active = False
        if ACTIVE_SEARCH_ID_DATASET in slice_group:
            active = decode_scalar(slice_group[ACTIVE_SEARCH_ID_DATASET][()]).strip()
            if active == resolved_search:
                del slice_group[ACTIVE_SEARCH_ID_DATASET]
                cleared_active = True
        remaining = (
            sorted(slice_group[SEARCHES_GROUP].keys())
            if SEARCHES_GROUP in slice_group
            else []
        )
    return {
        "slice_key": resolved_slice,
        "purged_search_id": resolved_search,
        "cleared_active_search_id": cleared_active,
        "remaining_search_ids": remaining,
    }


def matching_search_id_for_slice(
    h5_path: Path,
    *,
    slice_key: str,
    request: dict[str, Any],
) -> str | None:
    """Return an existing search id whose evaluation request matches ``request``."""
    with _H5PY_FILE(h5_path, "r") as f:
        if SLICE_CONTAINER_GROUP not in f or str(slice_key) not in f[SLICE_CONTAINER_GROUP]:
            return None
        slice_group = f[SLICE_CONTAINER_GROUP][str(slice_key)]
        if SEARCHES_GROUP not in slice_group:
            return None
        return _matching_search_id_for_request(slice_group[SEARCHES_GROUP], request)


def resolve_search_location(
    h5_path: Path,
    *,
    search_id: str,
) -> tuple[str, str]:
    """Return ``(slice_key, search_id)`` when ``search_id`` exists in ``h5_path``."""
    resolved_search_id = str(search_id or "").strip()
    if not resolved_search_id:
        raise ValueError("search_id is required")
    with _H5PY_FILE(h5_path, "r") as f:
        if SLICE_CONTAINER_GROUP not in f:
            raise KeyError(f"No slice container in artifact {h5_path}")
        for slice_key in f[SLICE_CONTAINER_GROUP].keys():
            slice_group = f[SLICE_CONTAINER_GROUP][str(slice_key)]
            if SEARCHES_GROUP not in slice_group:
                continue
            if resolved_search_id in slice_group[SEARCHES_GROUP]:
                return str(slice_key), resolved_search_id
    raise KeyError(f"Search {resolved_search_id!r} not found in {h5_path}")


RECOMPUTE_SEARCH_FORBIDDEN_CLI_FLAGS: frozenset[str] = frozenset(
    {
        "--target-metric",
        "--metrics-mask-threshold",
        "--threshold",
        "--metrics-mask-fits",
        "--mask-type",
        "--tr-mask-bmin-gauss",
        "--shift-policy",
        "--max-shift-arcsec",
        "--use-smoothed-obs-max",
        "--no-use-smoothed-obs-max",
        "--use-emthreshold",
        "--no-use-emthreshold",
        "--emthreshold",
        "--observation-time",
        "--obs-domain",
        "--obs-frequency-ghz",
        "--obs-wavelength-angstrom",
        "--recompute-existing",
        "--new-search-identity",
        "--a-start",
        "--b-start",
        "--da",
        "--db",
        "--a-min",
        "--a-max",
        "--b-min",
        "--b-max",
        "--q0-min",
        "--q0-max",
        "--q0-start",
        "--hard-q0-min",
        "--hard-q0-max",
        "--q0-step",
        "--xatol",
        "--maxiter",
        "--max-bracket-steps",
        "--threshold-metric",
        "--no-area",
        "--adaptive-bracketing",
        "--no-adaptive-bracketing",
        "--pixel-scale-arcsec",
        "--override-header-psf",
        "--no-override-header-psf",
        "--psf-bmaj-arcsec",
        "--psf-bmin-arcsec",
        "--psf-bpa-deg",
        "--psf-ref-frequency-ghz",
        "--psf-scale-inverse-frequency",
        "--no-psf-scale-inverse-frequency",
        "--tbase",
        "--nbase",
        "--observer",
        "--dsun-cm",
        "--lonc-deg",
        "--b0sun-deg",
        "--all-channels",
        "--render-channels",
        "--render-frequencies-ghz",
        "--render-obs-fits-dir",
        "--euv-instrument",
        "--euv-response-sav",
    }
)


EXPAND_GRID_SEARCH_BOUNDS_CLI_FLAGS: frozenset[str] = frozenset(
    {"--a-min", "--a-max", "--b-min", "--b-max"}
)

RECOMPUTE_SEARCH_ALLOWED_CLI_FLAGS: frozenset[str] = frozenset(
    {
        "--artifact-h5",
        "--recompute-search-id",
        "--no-viewer",
        "--dry-run",
    }
)

EXPAND_GRID_SEARCH_ALLOWED_CLI_FLAGS: frozenset[str] = (
    RECOMPUTE_SEARCH_ALLOWED_CLI_FLAGS
    - {"--recompute-search-id"}
    | {"--expand-grid-search-id"}
    | EXPAND_GRID_SEARCH_BOUNDS_CLI_FLAGS
)


def _argv_token_is_numeric_value(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _cli_flags_not_in_allowlist(argv: list[str], allowed: frozenset[str]) -> list[str]:
    hits: list[str] = []
    index = 1
    while index < len(argv):
        token = str(argv[index])
        if token in {"-h", "--help"}:
            index += 1
            continue
        if not token.startswith("--") or _argv_token_is_numeric_value(token):
            index += 1
            continue
        flag = token.split("=", 1)[0].strip().lower()
        if flag not in allowed:
            hits.append(flag)
        index += 1
    return hits


def assert_recompute_search_cli_argv_allowed(argv: list[str]) -> None:
    """Only --artifact-h5, --recompute-search-id, and optional --no-viewer/--dry-run."""
    disallowed = _cli_flags_not_in_allowlist(argv, RECOMPUTE_SEARCH_ALLOWED_CLI_FLAGS)
    if disallowed:
        joined = ", ".join(sorted(set(disallowed)))
        raise SystemExit(
            "--recompute-search-id uses the stored scoring recipe from artifact metadata only. "
            f"Allowed flags: --artifact-h5, --recompute-search-id, --no-viewer. Disallowed: {joined}"
        )


def build_recompute_search_guard_argv(
    *,
    artifact_h5: Path | str,
    recompute_search_id: str,
    no_viewer: bool = False,
    dry_run: bool = False,
) -> list[str]:
    """Synthetic argv for CLI allowlist checks (avoids pytest runner flags in ``sys.argv``)."""
    argv = [
        "adaptive_ab_search_single_observation.py",
        "--artifact-h5",
        str(artifact_h5),
        "--recompute-search-id",
        str(recompute_search_id),
    ]
    if no_viewer:
        argv.append("--no-viewer")
    if dry_run:
        argv.append("--dry-run")
    return argv


def build_expand_grid_search_guard_argv(
    *,
    artifact_h5: Path | str,
    expand_search_id: str,
    bounds_overrides: dict[str, float],
    no_viewer: bool = False,
    dry_run: bool = False,
) -> list[str]:
    """Synthetic argv for expand-mode CLI allowlist checks."""
    argv = [
        "adaptive_ab_search_single_observation.py",
        "--artifact-h5",
        str(artifact_h5),
        "--expand-grid-search-id",
        str(expand_search_id),
    ]
    field_flags = {
        "a_min": "--a-min",
        "a_max": "--a-max",
        "b_min": "--b-min",
        "b_max": "--b-max",
    }
    for field_name, flag in field_flags.items():
        if field_name in bounds_overrides:
            argv.extend([flag, str(bounds_overrides[field_name])])
    if no_viewer:
        argv.append("--no-viewer")
    if dry_run:
        argv.append("--dry-run")
    return argv


def parse_expand_grid_bounds_from_argv(argv: list[str]) -> dict[str, float]:
    """Return explicit a/b bound overrides present on the command line."""
    overrides: dict[str, float] = {}
    key_map = {
        "--a-min": "a_min",
        "--a-max": "a_max",
        "--b-min": "b_min",
        "--b-max": "b_max",
    }
    index = 1
    while index < len(argv):
        token = str(argv[index]).split("=", 1)[0].strip().lower()
        if token not in key_map:
            index += 1
            continue
        if "=" in str(argv[index]):
            raw_value = str(argv[index]).split("=", 1)[1]
        elif index + 1 < len(argv):
            next_token = str(argv[index + 1])
            try:
                float(next_token)
            except ValueError:
                raise SystemExit(f"{token} requires a numeric value") from None
            raw_value = next_token
            index += 1
        else:
            raise SystemExit(f"{token} requires a numeric value")
        try:
            overrides[key_map[token]] = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"{token} requires a numeric value") from exc
        index += 1
    return overrides


def validate_expanded_ab_bounds(
    *,
    stored_a_range: tuple[float, float],
    stored_b_range: tuple[float, float],
    new_a_range: tuple[float, float],
    new_b_range: tuple[float, float],
) -> None:
    """Require new bounds to be a strict superset (expansion only) of the stored search footprint."""
    stored_a_min, stored_a_max = (float(stored_a_range[0]), float(stored_a_range[1]))
    stored_b_min, stored_b_max = (float(stored_b_range[0]), float(stored_b_range[1]))
    new_a_min, new_a_max = (float(new_a_range[0]), float(new_a_range[1]))
    new_b_min, new_b_max = (float(new_b_range[0]), float(new_b_range[1]))
    if new_a_min > stored_a_min + 1e-12:
        raise SystemExit(
            f"--a-min ({new_a_min:g}) must not shrink below stored search minimum ({stored_a_min:g})"
        )
    if new_a_max < stored_a_max - 1e-12:
        raise SystemExit(
            f"--a-max ({new_a_max:g}) must not shrink below stored search maximum ({stored_a_max:g})"
        )
    if new_b_min > stored_b_min + 1e-12:
        raise SystemExit(
            f"--b-min ({new_b_min:g}) must not shrink below stored search minimum ({stored_b_min:g})"
        )
    if new_b_max < stored_b_max - 1e-12:
        raise SystemExit(
            f"--b-max ({new_b_max:g}) must not shrink below stored search maximum ({stored_b_max:g})"
        )
    if new_a_min >= new_a_max - 1e-12:
        raise SystemExit(f"Invalid a bounds: --a-min ({new_a_min:g}) must be < --a-max ({new_a_max:g})")
    if new_b_min >= new_b_max - 1e-12:
        raise SystemExit(f"Invalid b bounds: --b-min ({new_b_min:g}) must be < --b-max ({new_b_max:g})")
    expanded = (
        new_a_min < stored_a_min - 1e-12
        or new_a_max > stored_a_max + 1e-12
        or new_b_min < stored_b_min - 1e-12
        or new_b_max > stored_b_max + 1e-12
    )
    if not expanded:
        raise SystemExit(
            "Expanded bounds must widen at least one edge of the stored search footprint "
            f"(stored a=({stored_a_min:g}, {stored_a_max:g}) b=({stored_b_min:g}, {stored_b_max:g}); "
            f"requested a=({new_a_min:g}, {new_a_max:g}) b=({new_b_min:g}, {new_b_max:g}))"
        )


def assert_expand_grid_search_cli_argv_allowed(argv: list[str]) -> None:
    """Stored recipe plus widened a/b bounds and optional --no-viewer/--dry-run only."""
    disallowed = _cli_flags_not_in_allowlist(argv, EXPAND_GRID_SEARCH_ALLOWED_CLI_FLAGS)
    if disallowed:
        joined = ", ".join(sorted(set(disallowed)))
        raise SystemExit(
            "--expand-grid-search-id uses the stored scoring recipe plus widened "
            "--a-min/--a-max/--b-min/--b-max only. "
            f"Also allowed: --artifact-h5, --expand-grid-search-id, --no-viewer. Disallowed: {joined}"
        )


def validate_pinned_search_evaluation_recipe(
    profile: dict[str, Any],
    *,
    compatibility_signature: str,
) -> None:
    """Refuse expand/recompute when the live CLI recipe drifts from the stored request."""
    request = dict(profile.get("request") or {})
    if not request:
        return
    stored_signature = search_evaluation_signature(request)
    expected = str(compatibility_signature or "").strip()
    if not expected:
        return
    if stored_signature == expected:
        return
    search_id = str(profile.get("search_id") or "").strip() or "<search_id>"
    stored_q0 = request.get("q0_search_stages")
    raise SystemExit(
        "Pinned search recipe mismatch for "
        f"{search_id}: stored evaluation signature "
        f"{stored_signature[:16]}… does not match the resolved CLI recipe "
        f"{expected[:16]}…. "
        f"Stored q0_search_stages={stored_q0!r}. "
        "Use --expand-grid-search-id / --recompute-search-id only (no recipe overrides), "
        "or start a new identity with --new-search-identity."
    )


def load_search_run_profile(
    h5_path: Path,
    *,
    search_id: str,
) -> dict[str, Any]:
    """Load stored diagnostics and evaluation request for a specific search identity."""
    slice_key, resolved_search_id = resolve_search_location(h5_path, search_id=search_id)
    with _H5PY_FILE(h5_path, "r") as f:
        search_group = f[SLICE_CONTAINER_GROUP][slice_key][SEARCHES_GROUP][resolved_search_id]
        diagnostics = (
            _json_loads_or_empty(search_group["diagnostics_json"][()])
            if "diagnostics_json" in search_group
            else {}
        )
        request = _search_request_from_group(search_group)
    return {
        "search_id": resolved_search_id,
        "slice_key": slice_key,
        "diagnostics": diagnostics,
        "request": request,
    }


def _profile_optional_float(diagnostics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = diagnostics.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _profile_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def apply_search_run_profile_to_namespace(
    args: Any,
    profile: dict[str, Any],
) -> None:
    """Patch argparse ``Namespace`` fields from a stored search profile."""
    diagnostics = dict(profile.get("diagnostics") or {})
    request = dict(profile.get("request") or {})
    optimizer = dict(request.get("optimizer") or {})

    def _opt_or_diag(name: str) -> Any:
        value = optimizer.get(name)
        if value is not None:
            return value
        return diagnostics.get(name)

    fits_path = str(diagnostics.get("fits_file") or diagnostics.get("observation_source_path") or "").strip()
    if fits_path:
        args.fits_file = Path(fits_path)
    source_mode = str(diagnostics.get("observation_source_mode") or "").strip().lower()
    if source_mode in {"external_fits", "model_refmap"}:
        args.obs_source = source_mode
    if source_mode == "external_fits" and fits_path:
        args.obs_path = Path(fits_path)
    map_id = diagnostics.get("observation_source_map_id")
    if map_id not in {None, ""}:
        args.obs_map_id = str(map_id)

    model_path = str(diagnostics.get("model_path") or "").strip()
    if model_path:
        model_path_obj = Path(model_path)
        args.model_h5 = model_path_obj
        if hasattr(args, "model_h5_override"):
            args.model_h5_override = model_path_obj

    ebtel_path = str(diagnostics.get("ebtel_path") or "").strip()
    if ebtel_path:
        args.ebtel_path = Path(ebtel_path)

    spectral_domain = str(diagnostics.get("spectral_domain") or "").strip().lower()
    if spectral_domain in {"mw", "euv", "uv", "generic"}:
        args.obs_domain = spectral_domain
    frequency_ghz = _profile_optional_float(diagnostics, "frequency_ghz", "active_frequency_ghz")
    if frequency_ghz is not None:
        args.obs_frequency_ghz = frequency_ghz
    wavelength = diagnostics.get("wavelength_angstrom")
    if wavelength is not None:
        try:
            args.obs_wavelength_angstrom = float(wavelength)
        except (TypeError, ValueError):
            pass

    for field in ("a_start", "b_start", "da", "db"):
        value = diagnostics.get(field)
        if value is not None:
            setattr(args, field, float(value))
    a_range = diagnostics.get("a_range")
    if isinstance(a_range, (list, tuple)) and len(a_range) == 2:
        args.a_min = float(a_range[0])
        args.a_max = float(a_range[1])
    b_range = diagnostics.get("b_range")
    if isinstance(b_range, (list, tuple)) and len(b_range) == 2:
        args.b_min = float(b_range[0])
        args.b_max = float(b_range[1])

    target_metric = str(diagnostics.get("target_metric") or "").strip().lower()
    if target_metric in {"chi2", "rho2", "eta2"}:
        args.target_metric = target_metric

    threshold = diagnostics.get("metrics_mask_threshold", diagnostics.get("threshold"))
    if threshold is not None:
        args.metrics_mask_threshold = float(threshold)
    mask_fits = diagnostics.get("metrics_mask_fits")
    if mask_fits not in {None, ""}:
        args.metrics_mask_fits = Path(str(mask_fits))
    threshold_metric = _opt_or_diag("threshold_metric")
    if threshold_metric is not None:
        args.threshold_metric = float(threshold_metric)
    tr_mask = diagnostics.get("tr_mask_bmin_gauss")
    if tr_mask is not None:
        args.tr_mask_bmin_gauss = float(tr_mask)
    no_area = _profile_optional_bool(_opt_or_diag("no_area"))
    if no_area is not None:
        args.no_area = no_area

    execution_policy = str(diagnostics.get("execution_policy") or "").strip()
    if execution_policy in {"serial", "process-pool", "auto"}:
        args.execution_policy = execution_policy
    max_workers = diagnostics.get("execution_max_workers")
    if max_workers is not None:
        args.max_workers = int(max_workers)

    render_freqs = diagnostics.get("render_frequencies_ghz")
    if isinstance(render_freqs, (list, tuple)) and render_freqs:
        target_freq = frequency_ghz
        extras: list[float] = []
        for item in render_freqs:
            try:
                freq = float(item)
            except (TypeError, ValueError):
                continue
            if target_freq is not None and np.isclose(freq, float(target_freq), rtol=0.0, atol=1e-12):
                continue
            extras.append(freq)
        if extras:
            args.render_frequencies_ghz = ",".join(f"{value:g}" for value in extras)

    render_channels = diagnostics.get("render_channels")
    if isinstance(render_channels, (list, tuple)) and render_channels:
        args.render_channels = ",".join(str(item) for item in render_channels)
    render_obs_fits_dir = diagnostics.get("render_obs_fits_dir")
    if render_obs_fits_dir not in {None, ""}:
        args.render_obs_fits_dir = Path(str(render_obs_fits_dir))
    euv_channel = diagnostics.get("euv_channel")
    if euv_channel not in {None, ""}:
        args.euv_channel = str(euv_channel)
    euv_instrument = diagnostics.get("euv_instrument")
    if euv_instrument not in {None, ""}:
        args.euv_instrument = str(euv_instrument)
    euv_response = diagnostics.get("euv_response_override_path")
    if euv_response not in {None, ""}:
        args.euv_response_sav = Path(str(euv_response))

    pixel_scale = _profile_optional_float(diagnostics, "map_dx_arcsec")
    if pixel_scale is not None and hasattr(args, "pixel_scale_arcsec"):
        args.pixel_scale_arcsec = pixel_scale

    for field in (
        "q0_min",
        "q0_max",
        "hard_q0_min",
        "hard_q0_max",
        "q0_start",
        "q0_step",
        "max_bracket_steps",
    ):
        value = _opt_or_diag(field)
        if value is not None and hasattr(args, field):
            setattr(args, field, float(value) if field != "max_bracket_steps" else int(value))
    adaptive_bracketing = _profile_optional_bool(_opt_or_diag("adaptive_bracketing"))
    if adaptive_bracketing is not None and hasattr(args, "adaptive_bracketing"):
        args.adaptive_bracketing = adaptive_bracketing

    shift_policy = str(diagnostics.get("shift_policy") or "").strip().lower()
    if shift_policy in {"auto", "fixed"} and hasattr(args, "shift_policy"):
        args.shift_policy = shift_policy
    max_shift = diagnostics.get("max_shift_arcsec")
    if max_shift is not None and hasattr(args, "max_shift_arcsec"):
        args.max_shift_arcsec = float(max_shift)
    xy_shift = diagnostics.get("xy_shift_arcsec")
    if (
        isinstance(xy_shift, (list, tuple))
        and len(xy_shift) == 2
        and hasattr(args, "xy_shift_arcsec")
        and any(abs(float(item)) > 0.0 for item in xy_shift)
    ):
        args.xy_shift_arcsec = f"{float(xy_shift[0]):g},{float(xy_shift[1]):g}"

    use_smoothed = _profile_optional_bool(diagnostics.get("use_smoothed_obs_max"))
    if use_smoothed is not None and hasattr(args, "use_smoothed_obs_max"):
        args.use_smoothed_obs_max = use_smoothed
    use_emthreshold = _profile_optional_bool(diagnostics.get("use_emthreshold"))
    if use_emthreshold is not None and hasattr(args, "use_emthreshold"):
        args.use_emthreshold = use_emthreshold
    emthreshold = diagnostics.get("emthreshold")
    if emthreshold is not None and hasattr(args, "emthreshold"):
        args.emthreshold = float(emthreshold)
    from .q0_search import canonical_q0_search_stages_from_profile

    q0_stages = canonical_q0_search_stages_from_profile(
        {"request": request, "diagnostics": diagnostics}
    )
    if q0_stages and hasattr(args, "q0_search_stages"):
        args.q0_search_stages = ",".join(str(item) for item in q0_stages)


def register_sparse_search_in_artifact(
    out_h5: Path,
    *,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
    search_id: str | None = None,
    reset_search_points: bool = False,
    point_records: list[dict[str, Any]] | None = None,
    observation_canvas: np.ndarray | None = None,
    sigma_canvas: np.ndarray | None = None,
    canvas_wcs_header: fits.Header | None = None,
    blos_reference: tuple[np.ndarray, fits.Header] | None = None,
    psf_kernel: np.ndarray | None = None,
) -> str:
    """Register or reset a sparse search without rewriting other slice searches or ``common``."""
    diagnostics_out = dict(diagnostics)
    diagnostics_out["artifact_kind"] = UNIFIED_ARTIFACT_KIND
    if "mask_type" not in diagnostics_out:
        diagnostics_out["mask_type"] = diagnostics.get("mask_type", "union")
    layout_payload = {"kind": "point_list"}
    request_payload = _search_request_from_diagnostics(diagnostics_out, layout=layout_payload)
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if out_h5.exists() else "w"
    resolved_search_id = ""
    with _H5PY_FILE(out_h5, mode) as f:
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
                psf_kernel=psf_kernel,
                run_history=None,
                observation_canvas=observation_canvas,
                sigma_canvas=sigma_canvas,
                canvas_wcs_header=canvas_wcs_header,
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
        searches_group = slice_group.require_group(SEARCHES_GROUP)
        resolved_search_id = str(search_id or "").strip()
        if not resolved_search_id:
            resolved_search_id = _matching_search_id_for_request(searches_group, request_payload) or ""
        if not resolved_search_id:
            resolved_search_id = _search_id_from_diagnostics(diagnostics_out, layout=layout_payload)
        ref_diag = dict(diagnostics_out)
        ref_diag.update(_observation_ref_diagnostics_from_full(diagnostics_out))
        observation_ref_payload = {
            "observed": observed,
            "sigma_map": sigma_map,
            "wcs_header": wcs_header,
            "diagnostics": ref_diag,
            "observation_canvas": observation_canvas,
            "sigma_canvas": sigma_canvas,
            "canvas_wcs_header": canvas_wcs_header,
        }
        records = list(point_records or [])
        if bool(reset_search_points):
            records = []
        if resolved_search_id in searches_group and not bool(reset_search_points):
            search_group = searches_group[resolved_search_id]
            if OBSERVATION_REF_GROUP in search_group:
                del search_group[OBSERVATION_REF_GROUP]
            ref_group = search_group.create_group(OBSERVATION_REF_GROUP)
            _write_observation_ref_group(
                ref_group,
                observed=np.asarray(observed, dtype=float),
                sigma_map=np.asarray(sigma_map, dtype=float),
                wcs_header=wcs_header,
                diagnostics=ref_diag,
                observation_canvas=observation_canvas,
                sigma_canvas=sigma_canvas,
                canvas_wcs_header=canvas_wcs_header,
            )
        else:
            _write_search_group(
                searches_group,
                search_id=resolved_search_id,
                diagnostics=diagnostics_out,
                point_records=records,
                run_history=[],
                layout=layout_payload,
                observation_ref=observation_ref_payload,
            )
        if ACTIVE_SEARCH_ID_DATASET in slice_group:
            del slice_group[ACTIVE_SEARCH_ID_DATASET]
        _create_text_dataset(slice_group, ACTIVE_SEARCH_ID_DATASET, resolved_search_id)
    return str(resolved_search_id)


def _iter_slice_search_lifecycles(h5_path: Path) -> list[tuple[str, str, dict[str, Any]]]:
    records: list[tuple[str, str, dict[str, Any]]] = []
    with _H5PY_FILE(h5_path, "r") as f:
        if SLICE_CONTAINER_GROUP not in f:
            return records
        slices_group = f[SLICE_CONTAINER_GROUP]
        for slice_name in sorted(slices_group.keys()):
            slice_group = slices_group[slice_name]
            if SEARCHES_GROUP not in slice_group:
                continue
            searches_group = slice_group[SEARCHES_GROUP]
            candidate_ids: list[str] = []
            if ACTIVE_SEARCH_ID_DATASET in slice_group:
                active_id = decode_scalar(slice_group[ACTIVE_SEARCH_ID_DATASET][()]).strip()
                if active_id:
                    candidate_ids.append(active_id)
            for search_name in sorted(searches_group.keys()):
                if search_name not in candidate_ids:
                    candidate_ids.append(search_name)
            for search_id in candidate_ids:
                if search_id not in searches_group:
                    continue
                search_group = searches_group[search_id]
                status = decode_scalar(search_group.attrs.get("status", "unknown"))
                lifecycle = _read_search_lifecycle(search_group, status=status)
                records.append((str(slice_name), str(search_id), dict(lifecycle)))
    return records


def search_lifecycle_is_in_progress(lifecycle: dict[str, Any] | None) -> bool:
    """True only for searches that are genuinely running, not stale active flags."""
    data = dict(lifecycle or {})
    status = str(data.get("status", "") or "").strip().lower()
    if status in {"complete", "completed", "failed", "aborted", "interrupted"}:
        return False
    if str(data.get("completed_at") or "").strip():
        return False
    if bool(data.get("in_progress", False)):
        return True
    return bool(data.get("active", False))


def find_in_progress_slice_search(h5_path: Path) -> tuple[str | None, str | None]:
    """Return the first slice/search pair whose lifecycle is still in progress.

    Search iteration order prefers each slice's ``active_search_id`` entry first.
    """
    for slice_key, search_id, lifecycle in _iter_slice_search_lifecycles(h5_path):
        if search_lifecycle_is_in_progress(lifecycle):
            return slice_key, search_id
    return None, None


def list_artifact_search_catalog(h5_path: Path) -> list[dict[str, Any]]:
    """Return all stored searches with their owning slice keys."""
    catalog: list[dict[str, Any]] = []
    with _H5PY_FILE(h5_path, "r") as f:
        if SLICE_CONTAINER_GROUP not in f:
            return catalog
        slices_group = f[SLICE_CONTAINER_GROUP]
        for slice_name in sorted(slices_group.keys()):
            slice_group = slices_group[slice_name]
            for record in _read_search_records(slice_group):
                catalog.append({**dict(record), "slice_key": str(slice_name)})
    return catalog


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
    target_slice_key = ""
    for name in sorted(slices_group.keys()):
        grp = slices_group[name]
        diagnostics = {}
        common = grp.get("common")
        if common is not None and "diagnostics_json" in common:
            diagnostics = json.loads(decode_scalar(common["diagnostics_json"][()]))
        if not target_slice_key and common is not None and COMMON_TARGET_SLICE_KEY_DATASET in common:
            target_slice_key = decode_scalar(common[COMMON_TARGET_SLICE_KEY_DATASET][()]).strip()
        descriptor = slice_descriptor_from_diagnostics(diagnostics, fallback_key=name)
        descriptor["key"] = str(name)
        descriptors.append(descriptor)

    descriptors = _sorted_slice_descriptors(descriptors)
    key_lookup = {str(item["key"]): item for item in descriptors}
    selected_key = slice_key if slice_key in key_lookup else None
    if selected_key is None:
        if allow_missing and slice_key is not None:
            return None, descriptors, None
        if target_slice_key and target_slice_key in key_lookup:
            selected_key = target_slice_key
        else:
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


def _coerce_loaded_display_map(
    data: np.ndarray | None,
    *,
    observed_template: np.ndarray,
) -> tuple[np.ndarray, bool]:
    template = np.asarray(observed_template, dtype=float)
    if template.ndim != 2:
        raise ValueError(f"observed template must be 2D, got shape {template.shape}")
    if data is None:
        return _blank_map(template), False
    array = np.asarray(data, dtype=float)
    if array.ndim != 2 or array.shape != template.shape:
        return _blank_map(template), False
    return array, True


def _convolve_raw_map(raw_map: np.ndarray, psf_kernel: np.ndarray | None) -> np.ndarray:
    raw = np.asarray(raw_map, dtype=float)
    kernel = None if psf_kernel is None else np.asarray(psf_kernel, dtype=float)
    if kernel is None or kernel.ndim != 2 or kernel.size == 0:
        return raw.copy()
    return np.asarray(fftconvolve(raw, kernel, mode="same"), dtype=float)


def _derive_display_maps_from_raw(
    raw_map: np.ndarray | None,
    *,
    observed_template: np.ndarray,
    psf_kernel: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    raw, has_raw = _coerce_loaded_display_map(raw_map, observed_template=observed_template)
    if not has_raw:
        blank = _blank_map(observed_template)
        return blank, blank.copy(), blank.copy(), False
    modeled = _convolve_raw_map(raw, psf_kernel)
    residual = np.asarray(modeled - np.asarray(observed_template, dtype=float), dtype=float)
    return raw, modeled, residual, True


def _derive_trial_display_maps_from_raw(
    trial_raw_maps: np.ndarray | None,
    *,
    observed_template: np.ndarray,
    psf_kernel: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if trial_raw_maps is None:
        return None, None, None
    raw = np.asarray(trial_raw_maps, dtype=float)
    template = np.asarray(observed_template, dtype=float)
    if raw.ndim != 3 or template.ndim != 2 or raw.shape[1:] != template.shape:
        return None, None, None
    modeled = np.stack([_convolve_raw_map(frame, psf_kernel) for frame in raw], axis=0)
    residual = np.asarray(modeled - template[None, :, :], dtype=float)
    return raw, modeled, residual


def _finite_best_trial_index(
    *,
    fit_metric_trials: tuple[float, ...],
    q0: float,
    fit_q0_trials: tuple[float, ...],
) -> int | None:
    if fit_metric_trials:
        arr = np.asarray(fit_metric_trials, dtype=float)
        finite = np.flatnonzero(np.isfinite(arr))
        if finite.size:
            return int(finite[np.argmin(arr[finite])])
    if fit_q0_trials:
        q0_arr = np.asarray(fit_q0_trials, dtype=float)
        matches = np.flatnonzero(np.isclose(q0_arr, float(q0), rtol=0.0, atol=1e-12))
        if matches.size:
            return int(matches[0])
    return None


def _build_trial_history_entries(
    grp: h5py.Group,
    *,
    normalized: dict[str, Any],
    map_refs: dict[str, str],
) -> tuple[list[dict[str, Any]], int | None]:
    fit_q0_trials = tuple(float(v) for v in normalized.get("fit_q0_trials", ()))
    fit_metric_trials = tuple(float(v) for v in normalized.get("fit_metric_trials", ()))
    fit_chi2_trials = tuple(float(v) for v in normalized.get("fit_chi2_trials", ()))
    fit_rho2_trials = tuple(float(v) for v in normalized.get("fit_rho2_trials", ()))
    fit_eta2_trials = tuple(float(v) for v in normalized.get("fit_eta2_trials", ()))
    trial_raw = normalized.get("trial_raw_modeled_maps")
    trial_raw_arr = None if trial_raw is None else np.asarray(trial_raw, dtype=float)
    best_trial_index = _finite_best_trial_index(
        fit_metric_trials=fit_metric_trials,
        q0=float(normalized.get("q0", np.nan)),
        fit_q0_trials=fit_q0_trials,
    )

    entries: list[dict[str, Any]] = []
    if trial_raw_arr is not None and trial_raw_arr.ndim == 3 and trial_raw_arr.shape[0] == len(fit_q0_trials):
        for trial_index, raw_map in enumerate(trial_raw_arr):
            map_ref_key = f"trial_raw_modeled_maps/{trial_index:03d}"
            _write_point_map_ref(
                grp,
                normalized=normalized,
                name=map_ref_key,
                data=np.asarray(raw_map, dtype=float),
                map_refs=map_refs,
            )
            entries.append(
                {
                    "trial_index": int(trial_index),
                    "q0": float(fit_q0_trials[trial_index]),
                    "target_metric_value": float(fit_metric_trials[trial_index]) if trial_index < len(fit_metric_trials) else float("nan"),
                    "chi2": float(fit_chi2_trials[trial_index]) if trial_index < len(fit_chi2_trials) else float("nan"),
                    "rho2": float(fit_rho2_trials[trial_index]) if trial_index < len(fit_rho2_trials) else float("nan"),
                    "eta2": float(fit_eta2_trials[trial_index]) if trial_index < len(fit_eta2_trials) else float("nan"),
                    "raw_map_ref": str(map_refs.get(map_ref_key, "")),
                }
            )
        return entries, best_trial_index

    raw_best = normalized.get("raw_modeled_best")
    if fit_q0_trials and raw_best is not None:
        fallback_index = 0 if best_trial_index is None else int(best_trial_index)
        map_ref_key = f"trial_raw_modeled_maps/{fallback_index:03d}"
        _write_point_map_ref(
            grp,
            normalized=normalized,
            name=map_ref_key,
            data=np.asarray(raw_best, dtype=float),
            map_refs=map_refs,
        )
        entries.append(
            {
                "trial_index": int(fallback_index),
                "q0": float(fit_q0_trials[fallback_index]),
                "target_metric_value": float(fit_metric_trials[fallback_index]) if fallback_index < len(fit_metric_trials) else float("nan"),
                "chi2": float(fit_chi2_trials[fallback_index]) if fallback_index < len(fit_chi2_trials) else float("nan"),
                "rho2": float(fit_rho2_trials[fallback_index]) if fallback_index < len(fit_rho2_trials) else float("nan"),
                "eta2": float(fit_eta2_trials[fallback_index]) if fallback_index < len(fit_eta2_trials) else float("nan"),
                "raw_map_ref": str(map_refs.get(map_ref_key, "")),
            }
        )
        return entries, fallback_index

    return entries, best_trial_index


def _load_trial_history_entries(
    grp: h5py.Group,
    *,
    map_refs: dict[str, Any],
    include_maps: bool = True,
) -> tuple[list[dict[str, Any]], np.ndarray | None, int | None]:
    if TRIAL_HISTORY_DATASET not in grp:
        return [], None, None
    try:
        parsed = json.loads(decode_scalar(grp[TRIAL_HISTORY_DATASET][()]))
    except Exception:
        return [], None, None
    if not isinstance(parsed, list):
        return [], None, None

    entries: list[dict[str, Any]] = []
    trial_maps: list[np.ndarray] = []
    best_trial_index = None if "best_trial_index" not in grp.attrs else int(grp.attrs.get("best_trial_index"))
    for item in parsed:
        if not isinstance(item, dict):
            continue
        entry = dict(item)
        ref_path = str(entry.get("raw_map_ref") or "").strip()
        if not ref_path:
            ref_key = str(entry.get("map_ref_key") or "").strip()
            ref_path = str(map_refs.get(ref_key, "")).strip()
        entry["raw_map_ref"] = ref_path
        entries.append(entry)
        if not include_maps:
            continue
        raw_map = _read_map_store_ref_array(grp.file, ref_path)
        if raw_map is None:
            continue
        trial_maps.append(np.asarray(raw_map, dtype=float))

    if not include_maps or not trial_maps:
        return entries, None, best_trial_index
    try:
        trial_raw_maps = np.stack(trial_maps, axis=0)
    except Exception:
        trial_raw_maps = None
    return entries, trial_raw_maps, best_trial_index


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
    fit_shift_x_trials: tuple[float, ...] = (),
    fit_shift_y_trials: tuple[float, ...] = (),
    fit_find_shift_valid_trials: tuple[bool, ...] = (),
    fit_trial_mask_stages: tuple[str, ...] = (),
    trial_raw_modeled_maps: np.ndarray | None = None,
    trial_modeled_maps: np.ndarray | None = None,
    trial_residual_maps: np.ndarray | None = None,
    euv_coronal_best: np.ndarray | None = None,
    euv_tr_best: np.ndarray | None = None,
    euv_tr_mask: np.ndarray | None = None,
    trial_euv_coronal_maps: np.ndarray | None = None,
    trial_euv_tr_maps: np.ndarray | None = None,
    stokes_v_best: np.ndarray | None = None,
    trial_stokes_v_maps: np.ndarray | None = None,
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
    best_trial_index = _finite_best_trial_index(
        fit_metric_trials=tuple(float(v) for v in fit_metric_trials),
        q0=float(q0),
        fit_q0_trials=tuple(float(v) for v in fit_q0_trials),
    )
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
        "fit_shift_x_trials": tuple(float(v) for v in fit_shift_x_trials),
        "fit_shift_y_trials": tuple(float(v) for v in fit_shift_y_trials),
        "fit_find_shift_valid_trials": tuple(bool(v) for v in fit_find_shift_valid_trials),
        "fit_trial_mask_stages": tuple(str(v) for v in fit_trial_mask_stages),
        "trial_raw_modeled_maps": None if trial_raw_modeled_maps is None else np.asarray(trial_raw_modeled_maps, dtype=float),
        "trial_modeled_maps": None if trial_modeled_maps is None else np.asarray(trial_modeled_maps, dtype=float),
        "trial_residual_maps": None if trial_residual_maps is None else np.asarray(trial_residual_maps, dtype=float),
        "euv_coronal_best": None if euv_coronal_best is None else np.asarray(euv_coronal_best, dtype=float),
        "euv_tr_best": None if euv_tr_best is None else np.asarray(euv_tr_best, dtype=float),
        "euv_tr_mask": None if euv_tr_mask is None else np.asarray(euv_tr_mask, dtype=bool),
        "trial_euv_coronal_maps": None if trial_euv_coronal_maps is None else np.asarray(trial_euv_coronal_maps, dtype=float),
        "trial_euv_tr_maps": None if trial_euv_tr_maps is None else np.asarray(trial_euv_tr_maps, dtype=float),
        "stokes_v_best": None if stokes_v_best is None else np.asarray(stokes_v_best, dtype=float),
        "trial_stokes_v_maps": None if trial_stokes_v_maps is None else np.asarray(trial_stokes_v_maps, dtype=float),
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
        "best_trial_index": None if best_trial_index is None else int(best_trial_index),
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
        "fit_shift_x_trials": tuple(float(v) for v in payload.get("fit_shift_x_trials", ())),
        "fit_shift_y_trials": tuple(float(v) for v in payload.get("fit_shift_y_trials", ())),
        "fit_find_shift_valid_trials": tuple(bool(v) for v in payload.get("fit_find_shift_valid_trials", ())),
        "fit_trial_mask_stages": tuple(str(v) for v in payload.get("fit_trial_mask_stages", ())),
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
        "stokes_v_best": (
            None if payload.get("stokes_v_best") is None else np.asarray(payload["stokes_v_best"], dtype=float)
        ),
        "trial_stokes_v_maps": (
            None if payload.get("trial_stokes_v_maps") is None else np.asarray(payload["trial_stokes_v_maps"], dtype=float)
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
        "best_trial_index": None if payload.get("best_trial_index") is None else int(payload["best_trial_index"]),
    }


def _read_point_group_rectangular(grp: h5py.Group, *, include_maps: bool = True) -> dict[str, Any]:
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
    diagnostics = json.loads(decode_scalar(grp["diagnostics_json"][()]))
    if POINT_SYNTHETIC_MAP_MACHINE_KEYS_DATASET in grp:
        try:
            parsed_keys = json.loads(decode_scalar(grp[POINT_SYNTHETIC_MAP_MACHINE_KEYS_DATASET][()]))
            if isinstance(parsed_keys, list):
                diagnostics["synthetic_map_machine_keys"] = [str(item) for item in parsed_keys if str(item).strip()]
        except Exception:
            pass
    trial_history, trial_raw_modeled_maps, best_trial_index = _load_trial_history_entries(
        grp,
        map_refs=map_refs,
        include_maps=include_maps,
    )
    raw_modeled_best = None
    if include_maps:
        if best_trial_index is not None:
            matching_entry = next(
                (
                    item
                    for item in trial_history
                    if int(item.get("trial_index", -1)) == int(best_trial_index)
                ),
                None,
            )
            if matching_entry is not None:
                raw_modeled_best = _read_map_store_ref_array(grp.file, matching_entry.get("raw_map_ref"))
        if raw_modeled_best is None and trial_raw_modeled_maps is not None and trial_raw_modeled_maps.ndim == 3 and trial_raw_modeled_maps.shape[0] > 0:
            fallback_index = 0
            if best_trial_index is not None:
                fallback_index = int(np.clip(int(best_trial_index), 0, int(trial_raw_modeled_maps.shape[0]) - 1))
            raw_modeled_best = np.asarray(trial_raw_modeled_maps[fallback_index], dtype=float)
        if raw_modeled_best is None:
            raw_modeled_best = _derive_euv_raw_best_from_components(
                {
                    "euv_coronal_best": _read_point_map_array(grp, "euv_coronal_best", map_refs),
                    "euv_tr_best": _read_point_map_array(grp, "euv_tr_best", map_refs),
                    "euv_tr_mask": np.asarray(grp["euv_tr_mask"], dtype=bool) if "euv_tr_mask" in grp else None,
                }
            )
        if raw_modeled_best is None:
            raw_modeled_best = _read_point_map_array(grp, "raw_modeled_best", map_refs)
    return {
        "record_order": int(grp.attrs.get("record_order", 0)),
        "a": float(grp.attrs["a"]),
        "b": float(grp.attrs["b"]),
        "q0": float(grp.attrs["q0"]),
        "success": bool(grp.attrs["success"]),
        "status": decode_scalar(grp.attrs.get("status", b"computed")),
        "modeled_best": _read_point_map_array(grp, "modeled_best", map_refs) if include_maps else None,
        "raw_modeled_best": raw_modeled_best,
        "residual": _read_point_map_array(grp, "residual", map_refs) if include_maps else None,
        "fit_q0_trials": tuple(float(v) for v in np.asarray(grp["fit_q0_trials"], dtype=float)),
        "fit_metric_trials": tuple(float(v) for v in fit_metric_trials),
        "fit_chi2_trials": tuple(float(v) for v in fit_chi2_trials),
        "fit_rho2_trials": tuple(float(v) for v in fit_rho2_trials),
        "fit_eta2_trials": tuple(float(v) for v in fit_eta2_trials),
        "fit_shift_x_trials": tuple(
            float(v) for v in np.asarray(grp["fit_shift_x_trials"], dtype=float)
        ) if "fit_shift_x_trials" in grp else (),
        "fit_shift_y_trials": tuple(
            float(v) for v in np.asarray(grp["fit_shift_y_trials"], dtype=float)
        ) if "fit_shift_y_trials" in grp else (),
        "fit_find_shift_valid_trials": tuple(
            bool(v) for v in np.asarray(grp["fit_find_shift_valid_trials"], dtype=np.uint8)
        ) if "fit_find_shift_valid_trials" in grp else (),
        "fit_trial_mask_stages": tuple(
            decode_scalar(value)
            for value in np.asarray(grp["fit_trial_mask_stages"][()], dtype=object)
        ) if "fit_trial_mask_stages" in grp else (),
        "trial_raw_modeled_maps": (
            trial_raw_modeled_maps
            if include_maps and trial_raw_modeled_maps is not None
            else (_read_point_map_array(grp, "trial_raw_modeled_maps", map_refs) if include_maps else None)
        ),
        "trial_modeled_maps": _read_point_map_array(grp, "trial_modeled_maps", map_refs) if include_maps else None,
        "trial_residual_maps": _read_point_map_array(grp, "trial_residual_maps", map_refs) if include_maps else None,
        "euv_coronal_best": _read_point_map_array(grp, "euv_coronal_best", map_refs) if include_maps else None,
        "euv_tr_best": _read_point_map_array(grp, "euv_tr_best", map_refs) if include_maps else None,
        "euv_tr_mask": (
            np.asarray(grp["euv_tr_mask"], dtype=bool) if "euv_tr_mask" in grp else None
        ),
        "trial_euv_coronal_maps": _read_point_map_array(grp, "trial_euv_coronal_maps", map_refs) if include_maps else None,
        "trial_euv_tr_maps": _read_point_map_array(grp, "trial_euv_tr_maps", map_refs) if include_maps else None,
        "stokes_v_best": _read_point_map_array(grp, "stokes_v_best", map_refs) if include_maps else None,
        "trial_stokes_v_maps": _read_point_map_array(grp, "trial_stokes_v_maps", map_refs) if include_maps else None,
        "nfev": int(grp.attrs.get("nfev", -1)),
        "nit": int(grp.attrs.get("nit", -1)),
        "message": decode_scalar(grp.attrs.get("message", b"")),
        "used_adaptive_bracketing": bool(grp.attrs.get("used_adaptive_bracketing", False)),
        "bracket_found": bool(grp.attrs.get("bracket_found", False)),
        "bracket": bracket,
        "target_metric": target_metric,
        "map_refs": map_refs,
        "trial_history": trial_history,
        "best_trial_index": best_trial_index,
        "diagnostics": diagnostics,
    }


def _read_point_group_sparse(grp: h5py.Group, *, include_maps: bool = True) -> dict[str, Any]:
    return _read_point_group_rectangular(grp, include_maps=include_maps)


def _load_sparse_point_records(records_group: h5py.Group, *, include_maps: bool = True) -> list[dict[str, Any]]:
    latest_by_coord: dict[tuple[float, float], dict[str, Any]] = {}
    for name in sorted(records_group.keys()):
        record = _read_point_group_sparse(records_group[name], include_maps=include_maps)
        coord = (float(record["a"]), float(record["b"]))
        existing = latest_by_coord.get(coord)
        if existing is None or int(record["record_order"]) >= int(existing["record_order"]):
            latest_by_coord[coord] = record
    return sorted(latest_by_coord.values(), key=lambda item: (float(item["a"]), float(item["b"])))


def _load_rectangular_point_records(points_group: h5py.Group, *, include_maps: bool = True) -> list[dict[str, Any]]:
    records = [_read_point_group_rectangular(points_group[name], include_maps=include_maps) for name in sorted(points_group.keys())]
    return sorted(records, key=lambda item: (float(item["a"]), float(item["b"])))


def _load_canonical_point_records(group: h5py.Group, *, include_maps: bool = True) -> list[dict[str, Any]]:
    if "point_records" in group:
        return _load_sparse_point_records(group["point_records"], include_maps=include_maps)
    if "points" in group:
        return _load_rectangular_point_records(group["points"], include_maps=include_maps)
    return []


def _payload_from_point_records(
    *,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
    point_records: list[dict[str, Any]],
    target_metric: str,
    psf_kernel: np.ndarray | None,
    include_maps: bool = True,
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
        raw_modeled_best = None
        modeled_best = None
        residual_map = None
        raw_trial_maps = None
        trial_modeled_maps = None
        trial_residual_maps = None
        if include_maps:
            raw_modeled_best, modeled_best, residual_map, has_raw_modeled_best = _derive_display_maps_from_raw(
                record.get("raw_modeled_best"),
                observed_template=observed,
                psf_kernel=psf_kernel,
            )
            if not has_raw_modeled_best:
                derived_raw = _derive_euv_raw_best_from_components(record)
                if derived_raw is not None:
                    raw_modeled_best, modeled_best, residual_map, has_raw_modeled_best = _derive_display_maps_from_raw(
                        derived_raw,
                        observed_template=observed,
                        psf_kernel=psf_kernel,
                    )
            legacy_modeled_best, has_modeled_best = _coerce_loaded_display_map(
                record.get("modeled_best"),
                observed_template=observed,
            )
            legacy_residual_map, has_residual = _coerce_loaded_display_map(
                record.get("residual"),
                observed_template=observed,
            )
            if has_modeled_best and has_residual and not has_raw_modeled_best:
                modeled_best = legacy_modeled_best
                residual_map = legacy_residual_map
            raw_trial_maps, trial_modeled_maps, trial_residual_maps = _derive_trial_display_maps_from_raw(
                record.get("trial_raw_modeled_maps"),
                observed_template=observed,
                psf_kernel=psf_kernel,
            )
            if raw_trial_maps is None:
                derived_trials = _derive_euv_trial_raw_from_components(record)
                if derived_trials is not None:
                    raw_trial_maps, trial_modeled_maps, trial_residual_maps = _derive_trial_display_maps_from_raw(
                        derived_trials,
                        observed_template=observed,
                        psf_kernel=psf_kernel,
                    )
            if trial_modeled_maps is None:
                raw_trial_maps = record.get("trial_raw_modeled_maps")
                trial_modeled_maps = record.get("trial_modeled_maps")
                trial_residual_maps = record.get("trial_residual_maps")
            diagnostics_json["stored_display_maps_available"] = bool(has_raw_modeled_best and has_modeled_best and has_residual)
            diagnostics_json["display_maps_derived_from_raw"] = bool(has_raw_modeled_best)
        else:
            diagnostics_json["stored_display_maps_available"] = False
            diagnostics_json["display_maps_derived_from_raw"] = False
        metrics = viewer_record_metrics(record)
        for name in METRICS:
            try:
                diag_value = float(diagnostics_json.get(name, np.nan))
            except Exception:
                diag_value = float("nan")
            if np.isfinite(diag_value):
                metrics[name] = float(diag_value)
        target_value = metrics.get(target_metric_name, float("nan"))
        if np.isfinite(target_value):
            diagnostics_json["target_metric_value"] = float(target_value)
        points[(a_index, b_index)] = {
            **record,
            "metrics": metrics,
            "raw_modeled_best": raw_modeled_best,
            "modeled_best": modeled_best,
            "residual": residual_map,
            "trial_raw_modeled_maps": raw_trial_maps,
            "trial_modeled_maps": trial_modeled_maps,
            "trial_residual_maps": trial_residual_maps,
            "diagnostics": diagnostics_json,
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
                "raw_modeled_best": raw_modeled_best,
                "modeled_best": modeled_best,
                "residual": residual_map,
                "trial_raw_modeled_maps": raw_trial_maps,
                "trial_modeled_maps": trial_modeled_maps,
                "trial_residual_maps": trial_residual_maps,
                "diagnostics": diagnostics_json,
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


def load_scan_file(
    h5_path: Path,
    *,
    slice_key: str | None = None,
    search_id: str | None = None,
    include_maps: bool = True,
) -> dict[str, Any]:
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
        search_group = None
        if selected_search_id is not None and SEARCHES_GROUP in group and selected_search_id in group[SEARCHES_GROUP]:
            search_group = group[SEARCHES_GROUP][selected_search_id]
        obs_ref_payload = _resolve_observation_reference_payload(common, search_group=search_group)
        if obs_ref_payload is None:
            raise KeyError(
                f"observation reference not found for slice={selected_key!r} search={selected_search_id!r}"
            )
        observation_canvas = obs_ref_payload.get("observation_canvas")
        sigma_canvas = obs_ref_payload.get("sigma_canvas")
        canvas_wcs_header = obs_ref_payload.get("canvas_wcs_header")
        if obs_ref_payload.get("observed") is not None and obs_ref_payload.get("sigma_map") is not None:
            observed = np.asarray(obs_ref_payload["observed"], dtype=float)
            sigma_map = np.asarray(obs_ref_payload["sigma_map"], dtype=float)
        elif (
            observation_canvas is not None
            and sigma_canvas is not None
            and canvas_wcs_header is not None
            and wcs_header is not None
        ):
            from .obs_alignment import extract_observation_to_model_fov

            observed, sigma_map = extract_observation_to_model_fov(
                np.asarray(observation_canvas, dtype=float),
                canvas_wcs_header,
                wcs_header,
                shift_x_arcsec=0.0,
                shift_y_arcsec=0.0,
                canvas_sigma=np.asarray(sigma_canvas, dtype=float),
            )
            if sigma_map is None:
                raise KeyError(
                    f"observation reference canvas extraction failed for slice={selected_key!r}"
                )
        else:
            raise KeyError(
                f"observation reference not found for slice={selected_key!r} search={selected_search_id!r}"
            )
        if obs_ref_payload.get("wcs_header") is not None:
            wcs_header = obs_ref_payload["wcs_header"]
        obs_ref_diag = dict(obs_ref_payload.get("diagnostics") or {})
        if selected_search_id is not None and search_group is not None:
            from .grid_points import GRID_POINTS_GROUP, load_grid_points_as_viewer_records

            if GRID_POINTS_GROUP in search_group:
                point_records = load_grid_points_as_viewer_records(search_group, include_maps=include_maps)
            elif "point_records" in search_group:
                point_records = _load_sparse_point_records(search_group["point_records"], include_maps=include_maps)
            else:
                point_records = []
            search_diagnostics = dict(selected_search_record.get("diagnostics", {}) if selected_search_record else {})
            search_specific_diagnostics = {
                key: value
                for key, value in search_diagnostics.items()
                if key in SEARCH_SPECIFIC_DIAGNOSTIC_KEYS or str(key).startswith("metrics_") or str(key).startswith("tr_mask_")
            }
            diagnostics = {**diagnostics, **obs_ref_diag, **search_specific_diagnostics}
            target_metric = str(search_diagnostics.get("target_metric", diagnostics.get("target_metric", "chi2")))
            run_history = list(selected_search_record.get("run_history", run_history) if selected_search_record else run_history)
            kind = str(diagnostics.get("artifact_kind", _artifact_kind_from_group(group)))
        else:
            kind = _artifact_kind_from_group(group)
            if "point_records" in group:
                point_records = _load_sparse_point_records(group["point_records"], include_maps=include_maps)
                target_metric = str(diagnostics.get("target_metric", "chi2"))
            else:
                point_records = _load_canonical_point_records(group, include_maps=include_maps)
                if "summary" in group:
                    summary = group["summary"]
                    target_metric = decode_scalar(summary.attrs.get("target_metric", diagnostics.get("target_metric", b"chi2")))
                else:
                    target_metric = str(diagnostics.get("target_metric", "chi2"))
        payload = _payload_from_point_records(
            observed=observed,
            sigma_map=sigma_map,
            wcs_header=wcs_header,
            diagnostics=diagnostics,
            point_records=point_records,
            target_metric=target_metric,
            psf_kernel=common_payload.get("psf_kernel"),
            include_maps=include_maps,
        )
        selected_descriptor = next((item for item in descriptors if str(item["key"]) == str(selected_key)), None)
        payload["available_slices"] = descriptors
        payload["selected_slice_key"] = selected_key
        payload["selected_slice"] = selected_descriptor
        payload["run_history"] = run_history
        payload["search_records"] = search_records
        payload["selected_search_id"] = selected_search_id
        payload["selected_search"] = selected_search_record
        payload["psf_kernel"] = common_payload.get("psf_kernel")
        payload["psf_kernel_metadata"] = common_payload.get("psf_kernel_metadata")
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
        payload["observation_canvas"] = observation_canvas
        payload["sigma_canvas"] = sigma_canvas
        payload["canvas_wcs_header"] = canvas_wcs_header
        return payload


def load_active_point_snapshot(
    h5_path: Path,
    *,
    slice_key: str | None = None,
    search_id: str | None = None,
    include_maps: bool = True,
) -> dict[str, Any] | None:
    with _H5PY_FILE(h5_path, "r") as f:
        group, _descriptors, selected_key = _resolve_slice_group(
            f,
            slice_key=slice_key,
            allow_missing=slice_key is not None,
        )
        if group is None:
            raise KeyError(f"slice not found: {slice_key or selected_key}")
        selected_search_id = _selected_search_id(group, requested_search_id=search_id)
        if selected_search_id is None or SEARCHES_GROUP not in group:
            return None
        searches_group = group[SEARCHES_GROUP]
        if selected_search_id not in searches_group:
            return None
        search_group = searches_group[selected_search_id]
        if ACTIVE_POINT_SNAPSHOT_GROUP not in search_group:
            return None
        payload = _read_point_group_sparse(search_group[ACTIVE_POINT_SNAPSHOT_GROUP], include_maps=include_maps)
        obs_ref_payload = _resolve_observation_reference_payload(group["common"], search_group=search_group)
        if obs_ref_payload is None or obs_ref_payload.get("observed") is None:
            common_payload = _read_common_group(group["common"])
        else:
            common_payload = {
                **obs_ref_payload,
                "psf_kernel": _read_common_group(group["common"]).get("psf_kernel"),
            }
        if include_maps:
            observed = np.asarray(common_payload.get("observed"), dtype=float)
            psf_kernel = common_payload.get("psf_kernel")
            raw_modeled_best, modeled_best, residual, _has_raw = _derive_display_maps_from_raw(
                payload.get("raw_modeled_best"),
                observed_template=observed,
                psf_kernel=psf_kernel,
            )
            trial_raw_maps, trial_modeled_maps, trial_residual_maps = _derive_trial_display_maps_from_raw(
                payload.get("trial_raw_modeled_maps"),
                observed_template=observed,
                psf_kernel=psf_kernel,
            )
            payload["raw_modeled_best"] = raw_modeled_best
            payload["modeled_best"] = modeled_best
            payload["residual"] = residual
            payload["trial_raw_modeled_maps"] = trial_raw_maps
            payload["trial_modeled_maps"] = trial_modeled_maps
            payload["trial_residual_maps"] = trial_residual_maps
        payload["selected_slice_key"] = str(selected_key)
        payload["selected_search_id"] = str(selected_search_id)
        return payload


def load_live_trial_point(
    h5_path: Path,
    *,
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
            raise KeyError(f"slice not found: {slice_key or selected_key}")
        selected_search_id = _selected_search_id(group, requested_search_id=search_id)
        if selected_search_id is None or SEARCHES_GROUP not in group:
            return None
        searches_group = group[SEARCHES_GROUP]
        if selected_search_id not in searches_group:
            return None
        search_group = searches_group[selected_search_id]
        if LIVE_TRIAL_POINT_GROUP not in search_group:
            return None
        live_group = search_group[LIVE_TRIAL_POINT_GROUP]

        def _load_optional_float(attr_name: str) -> float | None:
            if attr_name not in live_group.attrs:
                return None
            try:
                value = float(live_group.attrs[attr_name])
            except Exception:
                return None
            return value if np.isfinite(value) else None

        def _load_optional_int(attr_name: str) -> int | None:
            if attr_name not in live_group.attrs:
                return None
            try:
                return int(live_group.attrs[attr_name])
            except Exception:
                return None

        fit_q0_trials = np.asarray(live_group.get("fit_q0_trials", ()), dtype=float)
        fit_metric_trials = np.asarray(live_group.get("fit_metric_trials", ()), dtype=float)
        trial_history: list[dict[str, Any]] = []
        if TRIAL_HISTORY_DATASET in live_group:
            try:
                parsed_history = json.loads(decode_scalar(live_group[TRIAL_HISTORY_DATASET][()]))
            except Exception:
                parsed_history = []
            if isinstance(parsed_history, list):
                trial_history = [dict(item) for item in parsed_history if isinstance(item, dict)]
        return {
            "selected_slice_key": str(selected_key),
            "selected_search_id": str(selected_search_id),
            "slice_key": decode_scalar(live_group.attrs.get("slice_key", str(selected_key))),
            "search_id": decode_scalar(live_group.attrs.get("search_id", str(selected_search_id))),
            "metric_name": decode_scalar(live_group.attrs.get("metric_name", "chi2")),
            "updated_utc": decode_scalar(live_group.attrs.get("updated_utc", "")),
            "sequence": _load_optional_int("sequence"),
            "a": _load_optional_float("a"),
            "b": _load_optional_float("b"),
            "q0": _load_optional_float("q0"),
            "trial_index": _load_optional_int("trial_index"),
            "fit_q0_trials": fit_q0_trials,
            "fit_metric_trials": fit_metric_trials,
            "trial_history": trial_history,
        }


def _load_live_trial_history_entries(live_group: h5py.Group) -> list[dict[str, Any]]:
    if TRIAL_HISTORY_DATASET not in live_group:
        return []
    try:
        parsed = json.loads(decode_scalar(live_group[TRIAL_HISTORY_DATASET][()]))
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [dict(item) for item in parsed if isinstance(item, dict)]


def _search_diagnostics_for_live_maps(search_group: h5py.Group, slice_group: h5py.Group) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    if "common" in slice_group:
        diagnostics.update(_diagnostics_from_slice_group(slice_group))
    if "diagnostics_json" in search_group:
        try:
            search_diag = json.loads(decode_scalar(search_group["diagnostics_json"][()]))
        except Exception:
            search_diag = {}
        if isinstance(search_diag, dict):
            diagnostics.update(search_diag)
    return diagnostics


def _merge_live_trial_history_entries(
    h5_file: h5py.File,
    *,
    existing_history: list[dict[str, Any]],
    fit_q0_trials: np.ndarray,
    fit_metric_trials: np.ndarray,
    metric_name: str,
    normalized: dict[str, Any],
    completed_trial_index: int | None,
    completed_trial_raw_map: np.ndarray | None,
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    trial_count = int(fit_q0_trials.size)
    completed_index = None if completed_trial_index is None else int(completed_trial_index)
    completed_raw = None if completed_trial_raw_map is None else np.asarray(completed_trial_raw_map, dtype=float)
    for trial_index in range(trial_count):
        q0_value = float(fit_q0_trials[trial_index])
        metric_value = (
            float(fit_metric_trials[trial_index])
            if trial_index < int(fit_metric_trials.size)
            else float("nan")
        )
        if completed_index is not None and trial_index == completed_index and completed_raw is not None:
            map_ref_key = f"live_trial_raw_modeled_maps/{trial_index:03d}"
            identity_source = dict(normalized)
            identity_source["q0"] = q0_value
            raw_map_ref = _write_map_store_array(
                h5_file,
                identity=_map_store_identity(name=map_ref_key, normalized=identity_source),
                data=completed_raw,
            )
            history.append(
                {
                    "trial_index": int(trial_index),
                    "q0": q0_value,
                    "target_metric": str(metric_name),
                    "target_metric_value": metric_value,
                    "raw_map_ref": str(raw_map_ref),
                }
            )
            continue
        if trial_index < len(existing_history):
            prior = dict(existing_history[trial_index])
            prior.setdefault("trial_index", int(trial_index))
            prior["q0"] = q0_value
            prior["target_metric"] = str(metric_name)
            prior["target_metric_value"] = metric_value
            history.append(prior)
            continue
        history.append(
            {
                "trial_index": int(trial_index),
                "q0": q0_value,
                "target_metric": str(metric_name),
                "target_metric_value": metric_value,
                "raw_map_ref": "",
            }
        )
    return history


def load_live_trial_plot_payload(
    h5_path: Path,
    *,
    a: float,
    b: float,
    trial_index: int | None = None,
    slice_key: str | None = None,
    search_id: str | None = None,
) -> dict[str, Any] | None:
    live_state = load_live_trial_point(h5_path, slice_key=slice_key, search_id=search_id)
    if live_state is None:
        return None
    try:
        live_a = float(live_state.get("a", np.nan))
        live_b = float(live_state.get("b", np.nan))
    except Exception:
        return None
    if not (
        np.isfinite(live_a)
        and np.isfinite(live_b)
        and np.isclose(live_a, float(a), rtol=0.0, atol=1e-6)
        and np.isclose(live_b, float(b), rtol=0.0, atol=1e-6)
    ):
        return None

    fit_q0_trials = np.asarray(live_state.get("fit_q0_trials", ()), dtype=float)
    if fit_q0_trials.size == 0:
        return None
    chosen_trial_index = int(fit_q0_trials.size - 1) if trial_index is None else int(trial_index)
    chosen_trial_index = int(np.clip(chosen_trial_index, 0, int(fit_q0_trials.size) - 1))

    trial_history = list(live_state.get("trial_history") or [])
    raw_map_ref = ""
    matching_entry = next(
        (item for item in trial_history if int(item.get("trial_index", -1)) == int(chosen_trial_index)),
        None,
    )
    if matching_entry is not None:
        raw_map_ref = str(matching_entry.get("raw_map_ref", "")).strip()
    if not raw_map_ref:
        return None

    with _H5PY_FILE(h5_path, "r") as f:
        group, _descriptors, selected_key = _resolve_slice_group(
            f,
            slice_key=slice_key or str(live_state.get("slice_key", "")).strip() or None,
            allow_missing=slice_key is not None,
        )
        if group is None or "common" not in group:
            return None
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
            "trial_index": int(chosen_trial_index),
            "fit_q0_trials": fit_q0_trials,
            "raw_modeled_best": np.asarray(raw_display, dtype=float),
            "modeled_best": np.asarray(modeled, dtype=float),
            "residual": np.asarray(residual, dtype=float),
            "observed": observed,
            "wcs_header": common_payload["wcs_header"],
            "psf_kernel": psf_kernel,
            "selected_slice_key": str(selected_key),
            "selected_search_id": str(live_state.get("selected_search_id", "") or ""),
            "live_trial": True,
        }


def load_selected_trial_plot_payload(
    h5_path: Path,
    *,
    a: float,
    b: float,
    trial_index: int | None = None,
    slice_key: str | None = None,
    search_id: str | None = None,
) -> dict[str, Any] | None:
    from .grid_points import GRID_POINTS_GROUP, load_grid_point_trial_plot_payload

    with _H5PY_FILE(h5_path, "r") as f:
        group, _descriptors, selected_key = _resolve_slice_group(
            f,
            slice_key=slice_key,
            allow_missing=slice_key is not None,
        )
        if group is None:
            raise KeyError(f"slice not found: {slice_key or selected_key}")

        selected_search_id = _selected_search_id(group, requested_search_id=search_id)
        if selected_search_id is not None and SEARCHES_GROUP in group:
            searches_group = group[SEARCHES_GROUP]
            if selected_search_id in searches_group:
                search_group = searches_group[selected_search_id]
                if GRID_POINTS_GROUP in search_group:
                    return load_grid_point_trial_plot_payload(
                        h5_path,
                        a=float(a),
                        b=float(b),
                        trial_index=trial_index,
                        slice_key=slice_key or selected_key,
                        search_id=selected_search_id,
                    )
        records_group: h5py.Group | None = None
        if selected_search_id is not None and SEARCHES_GROUP in group:
            searches_group = group[SEARCHES_GROUP]
            if selected_search_id in searches_group:
                search_group = searches_group[selected_search_id]
                if "point_records" in search_group:
                    records_group = search_group["point_records"]
        if records_group is None and "point_records" in group:
            records_group = group["point_records"]
        if records_group is None and "points" in group:
            records_group = group["points"]
        if records_group is None:
            return None

        selected_record: h5py.Group | None = None
        selected_order = -1
        for name in sorted(records_group.keys()):
            candidate = records_group[name]
            try:
                cand_a = float(candidate.attrs["a"])
                cand_b = float(candidate.attrs["b"])
            except Exception:
                continue
            if not (np.isclose(cand_a, float(a), rtol=0.0, atol=1e-6) and np.isclose(cand_b, float(b), rtol=0.0, atol=1e-6)):
                continue
            order = int(candidate.attrs.get("record_order", 0))
            if selected_record is None or order >= selected_order:
                selected_record = candidate
                selected_order = order
        if selected_record is None:
            return None

        point_payload = _read_point_group_sparse(selected_record, include_maps=False)
        map_refs = dict(point_payload.get("map_refs") or {})
        trial_history = list(point_payload.get("trial_history") or [])
        fit_q0_trials = np.asarray(point_payload.get("fit_q0_trials", ()), dtype=float)

        chosen_trial_index = None if trial_index is None else int(trial_index)
        if fit_q0_trials.size > 0:
            if chosen_trial_index is None:
                best_trial_index = point_payload.get("best_trial_index")
                if best_trial_index is not None:
                    chosen_trial_index = int(np.clip(int(best_trial_index), 0, int(fit_q0_trials.size) - 1))
                else:
                    chosen_trial_index = int(fit_q0_trials.size - 1)
            else:
                chosen_trial_index = int(np.clip(chosen_trial_index, 0, int(fit_q0_trials.size) - 1))

        raw_map_ref = ""
        if chosen_trial_index is not None:
            matching_entry = next(
                (
                    item
                    for item in trial_history
                    if int(item.get("trial_index", -1)) == int(chosen_trial_index)
                ),
                None,
            )
            if matching_entry is not None:
                raw_map_ref = str(matching_entry.get("raw_map_ref", "")).strip()
                if not raw_map_ref:
                    ref_key = str(matching_entry.get("map_ref_key", "")).strip()
                    raw_map_ref = str(map_refs.get(ref_key, "")).strip()
            if not raw_map_ref:
                ref_key = f"trial_raw_modeled_maps/{int(chosen_trial_index):03d}"
                raw_map_ref = str(map_refs.get(ref_key, "")).strip()
        if not raw_map_ref:
            raw_map_ref = str(map_refs.get("raw_modeled_best", "")).strip()

        raw_modeled = _read_map_store_ref_array(f, raw_map_ref)
        if raw_modeled is None:
            raw_modeled = _read_point_map_array(selected_record, "raw_modeled_best", map_refs)
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
            "trial_index": chosen_trial_index,
            "fit_q0_trials": fit_q0_trials,
            "raw_modeled_best": np.asarray(raw_display, dtype=float),
            "modeled_best": np.asarray(modeled, dtype=float),
            "residual": np.asarray(residual, dtype=float),
            "observed": observed,
            "wcs_header": common_payload["wcs_header"],
            "psf_kernel": psf_kernel,
            "selected_slice_key": str(selected_key),
            "selected_search_id": str(selected_search_id) if selected_search_id is not None else None,
        }


def write_active_point_snapshot(
    h5_path: Path,
    *,
    point_payload: dict[str, Any],
    slice_key: str | None = None,
    search_id: str | None = None,
) -> None:
    with _H5PY_FILE(h5_path, "r+") as f:
        group, _descriptors, selected_key = _resolve_slice_group(
            f,
            slice_key=slice_key,
            allow_missing=slice_key is not None,
        )
        if group is None:
            raise KeyError(f"slice not found: {slice_key or selected_key}")
        selected_search_id = _selected_search_id(group, requested_search_id=search_id)
        if selected_search_id is None or SEARCHES_GROUP not in group:
            raise KeyError(f"search not found for slice: {slice_key or selected_key}")
        searches_group = group[SEARCHES_GROUP]
        if selected_search_id not in searches_group:
            raise KeyError(f"search not found: {selected_search_id}")
        search_group = searches_group[selected_search_id]
        if ACTIVE_POINT_SNAPSHOT_GROUP in search_group:
            del search_group[ACTIVE_POINT_SNAPSHOT_GROUP]
        snapshot_group = search_group.create_group(ACTIVE_POINT_SNAPSHOT_GROUP)
        snapshot_group.attrs["updated_utc"] = np.bytes_(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        _write_point_group(snapshot_group, point_payload, record_order=-1)


def write_live_trial_point(
    h5_path: Path,
    *,
    live_state: dict[str, Any],
    slice_key: str | None = None,
    search_id: str | None = None,
) -> None:
    with _H5PY_FILE(h5_path, "r+") as f:
        group, _descriptors, selected_key = _resolve_slice_group(
            f,
            slice_key=slice_key,
            allow_missing=slice_key is not None,
        )
        if group is None:
            raise KeyError(f"slice not found: {slice_key or selected_key}")
        selected_search_id = _selected_search_id(group, requested_search_id=search_id)
        if selected_search_id is None or SEARCHES_GROUP not in group:
            raise KeyError(f"search not found for slice: {slice_key or selected_key}")
        searches_group = group[SEARCHES_GROUP]
        if selected_search_id not in searches_group:
            raise KeyError(f"search not found: {selected_search_id}")
        search_group = searches_group[selected_search_id]
        existing_history: list[dict[str, Any]] = []
        if LIVE_TRIAL_POINT_GROUP in search_group:
            existing_history = _load_live_trial_history_entries(search_group[LIVE_TRIAL_POINT_GROUP])
            del search_group[LIVE_TRIAL_POINT_GROUP]
        live_group = search_group.create_group(LIVE_TRIAL_POINT_GROUP)
        live_group.attrs["updated_utc"] = np.bytes_(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        live_group.attrs["slice_key"] = np.bytes_(str(live_state.get("slice_key") or selected_key))
        live_group.attrs["search_id"] = np.bytes_(str(selected_search_id))
        live_group.attrs["metric_name"] = np.bytes_(str(live_state.get("metric_name") or "chi2"))
        if live_state.get("sequence") is not None:
            live_group.attrs["sequence"] = int(live_state["sequence"])
        for attr_name in ("a", "b"):
            value = live_state.get(attr_name)
            if value is None:
                continue
            numeric = float(value)
            if np.isfinite(numeric):
                live_group.attrs[attr_name] = numeric
        q0_value = live_state.get("q0")
        if q0_value is not None:
            q0_numeric = float(q0_value)
            if np.isfinite(q0_numeric):
                live_group.attrs["q0"] = q0_numeric
        trial_index_value = live_state.get("trial_index")
        if trial_index_value is not None:
            live_group.attrs["trial_index"] = int(trial_index_value)
        fit_q0_trials = np.asarray(live_state.get("fit_q0_trials", ()), dtype=float)
        fit_metric_trials = np.asarray(live_state.get("fit_metric_trials", ()), dtype=float)
        live_group.create_dataset("fit_q0_trials", data=fit_q0_trials.astype(float, copy=False))
        live_group.create_dataset("fit_metric_trials", data=fit_metric_trials.astype(float, copy=False))

        completed_trial_index = live_state.get("completed_trial_index")
        completed_trial_raw_map = live_state.get("completed_trial_raw_map")
        if completed_trial_raw_map is not None:
            completed_trial_raw_map = np.asarray(completed_trial_raw_map, dtype=float)
        identity_diag = _search_diagnostics_for_live_maps(search_group, group)
        identity_diag.update(
            {
                "a": float(live_state.get("a", np.nan)),
                "b": float(live_state.get("b", np.nan)),
                "target_metric": str(live_state.get("metric_name") or "chi2"),
            }
        )
        trial_history = _merge_live_trial_history_entries(
            f,
            existing_history=existing_history,
            fit_q0_trials=fit_q0_trials,
            fit_metric_trials=fit_metric_trials,
            metric_name=str(live_state.get("metric_name") or "chi2"),
            normalized=identity_diag,
            completed_trial_index=None if completed_trial_index is None else int(completed_trial_index),
            completed_trial_raw_map=completed_trial_raw_map,
        )
        _create_text_dataset(live_group, TRIAL_HISTORY_DATASET, _json_dumps(trial_history))


def clear_active_point_snapshot(
    h5_path: Path,
    *,
    slice_key: str | None = None,
    search_id: str | None = None,
) -> None:
    with _H5PY_FILE(h5_path, "r+") as f:
        group, _descriptors, selected_key = _resolve_slice_group(
            f,
            slice_key=slice_key,
            allow_missing=slice_key is not None,
        )
        if group is None:
            return
        selected_search_id = _selected_search_id(group, requested_search_id=search_id)
        if selected_search_id is None or SEARCHES_GROUP not in group:
            return
        searches_group = group[SEARCHES_GROUP]
        if selected_search_id not in searches_group:
            return
        search_group = searches_group[selected_search_id]
        if ACTIVE_POINT_SNAPSHOT_GROUP in search_group:
            del search_group[ACTIVE_POINT_SNAPSHOT_GROUP]


def clear_live_trial_point(
    h5_path: Path,
    *,
    slice_key: str | None = None,
    search_id: str | None = None,
) -> None:
    with _H5PY_FILE(h5_path, "r+") as f:
        group, _descriptors, selected_key = _resolve_slice_group(
            f,
            slice_key=slice_key,
            allow_missing=slice_key is not None,
        )
        if group is None:
            return
        selected_search_id = _selected_search_id(group, requested_search_id=search_id)
        if selected_search_id is None or SEARCHES_GROUP not in group:
            return
        searches_group = group[SEARCHES_GROUP]
        if selected_search_id not in searches_group:
            return
        search_group = searches_group[selected_search_id]
        if LIVE_TRIAL_POINT_GROUP in search_group:
            del search_group[LIVE_TRIAL_POINT_GROUP]


def extract_artifact_identity_summary(
    h5_path: str | Path,
    *,
    slice_key: str | None = None,
    search_id: str | None = None,
) -> dict[str, Any]:
    payload = load_scan_file(Path(h5_path), slice_key=slice_key, search_id=search_id)
    diagnostics = dict(payload.get("diagnostics") or {})
    selected_search = payload.get("selected_search") or {}
    search_diagnostics = dict(selected_search.get("diagnostics") or {})
    readable_keys = (
        "artifact_kind",
        "spectral_domain",
        "spectral_label",
        "model_id",
        "model_sha256",
        "observation_source_mode",
        "observation_source_path",
        "observation_source_map_id",
        "observation_source_sha256",
        "observation_instrument",
        "observation_observer",
        "ebtel_path",
        "ebtel_sha256",
        "frequency_ghz",
        "wavelength_angstrom",
        "euv_channel",
        "euv_instrument",
        "euv_response_sav",
        "euv_response_origin",
        "euv_response_override_path",
        "euv_response_resolver",
        "euv_response_identity_version",
        "euv_response_sha256",
        "euv_response_source",
        "euv_response_mode",
        "euv_response_identity_summary",
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
        "slice_observation_identity_sha256",
        "preprocessed_observation_sha256",
        "preprocessed_sigma_sha256",
        "observation_reference_preprocessed",
        "observation_time_rotation_applied",
        "geometry_policy_mode",
        "geometry_policy_observation_los",
        "geometry_policy_model_los",
        "geometry_policy_los_aligned",
        COMPATIBILITY_SIGNATURE_KEY,
        "psf_source",
        "resolved_psf",
        "psf_bmaj_arcsec",
        "psf_bmin_arcsec",
        "psf_bpa_deg",
        "psf_ref_frequency_ghz",
        "psf_scale_inverse_frequency",
        "render_channels",
        "render_frequencies_ghz",
        "target_metric",
        "metrics_mask_threshold",
        "metrics_mask_fits",
        "metrics_mask_source",
        "mask_type",
        "tr_mask_bmin_gauss",
        "tr_mask_source",
    )
    identity = {key: diagnostics.get(key) for key in readable_keys if key in diagnostics}
    search_identity = {
        key: search_diagnostics.get(key)
        for key in (
            "search_id",
            "target_metric",
            "metrics_mask_threshold",
            "metrics_mask_fits",
            "metrics_mask_source",
            "mask_type",
            "tr_mask_bmin_gauss",
            "tr_mask_source",
            COMPATIBILITY_SIGNATURE_KEY,
        )
        if key in search_diagnostics
    }
    return {
        "artifact_format": payload.get("artifact_format"),
        "selected_slice_key": payload.get("selected_slice_key"),
        "selected_slice": payload.get("selected_slice"),
        "available_slices": payload.get("available_slices", []),
        "selected_search_id": payload.get("selected_search_id"),
        "available_search_ids": [
            str(record.get("search_id"))
            for record in payload.get("search_records", [])
            if record.get("search_id") is not None
        ],
        "point_count": len(payload.get("point_records", [])),
        "identity": identity,
        "search_identity": search_identity,
    }


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

            obs_ref_payload = None
            if SEARCHES_GROUP in group:
                selected_search_id = _selected_search_id(group)
                if selected_search_id and selected_search_id in group[SEARCHES_GROUP]:
                    obs_ref_payload = _resolve_observation_reference_payload(
                        common,
                        search_group=group[SEARCHES_GROUP][selected_search_id],
                    )
            if obs_ref_payload is None:
                obs_ref_payload = _resolve_observation_reference_payload(common, search_group=None)
            observed_arr = obs_ref_payload.get("observed") if obs_ref_payload else common.get("observed")
            if observed_arr is None:
                skipped_fields["map_shape"] = "observation reference unavailable"
                observed_shape = ()
            else:
                observed_shape = tuple(int(v) for v in np.asarray(observed_arr).shape)
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

            if not str(diagnostics.get("forward_model_sha256", "")).strip():
                model_path = _resolve_existing_file_from_diagnostics(diagnostics.get("model_path"))
                if model_path is not None:
                    diagnostics["forward_model_sha256"] = _compute_file_sha256(model_path)
                    diagnostics["forward_model_identity_version"] = FORWARD_MODEL_IDENTITY_VERSION
                    updated_fields["forward_model_sha256"] = diagnostics["forward_model_sha256"]
                    updated_fields["forward_model_identity_version"] = diagnostics["forward_model_identity_version"]
                elif str(diagnostics.get("model_sha256", "")).strip():
                    diagnostics["forward_model_sha256"] = str(diagnostics["model_sha256"])
                    diagnostics["forward_model_identity_version"] = FORWARD_MODEL_IDENTITY_VERSION
                    updated_fields["forward_model_sha256"] = diagnostics["forward_model_sha256"]
                    updated_fields["forward_model_identity_version"] = diagnostics["forward_model_identity_version"]

            if not str(diagnostics.get("artifact_geometry_sha256", "")).strip():
                geometry_block = build_artifact_geometry_block(diagnostics)
                geometry_sha = artifact_geometry_sha256(geometry_block)
                diagnostics["artifact_geometry_sha256"] = geometry_sha
                updated_fields["artifact_geometry_sha256"] = geometry_sha
                if not dry_run:
                    if COMMON_ARTIFACT_GEOMETRY_DATASET not in common:
                        _create_text_dataset(common, COMMON_ARTIFACT_GEOMETRY_DATASET, _json_dumps(geometry_block))
                    if COMMON_ARTIFACT_GEOMETRY_SHA256_DATASET not in common:
                        _create_text_dataset(common, COMMON_ARTIFACT_GEOMETRY_SHA256_DATASET, str(geometry_sha))

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
    physical = _map_identity_physical_keys_from_diagnostics(diagnostics)
    component = _component_from_array_name(name)
    if component is None:
        legacy_identity = {
            "array_name": str(name),
            "a": float(normalized["a"]),
            "b": float(normalized["b"]),
            "q0": float(normalized.get("q0", np.nan)),
        }
        legacy_keys = (
            "model_sha256",
            "ebtel_sha256",
            "euv_response_identity_version",
            "euv_response_sha256",
            "spectral_domain",
            "spectral_label",
            "frequency_ghz",
            "wavelength_angstrom",
            "euv_channel",
            "euv_instrument",
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
        for key in legacy_keys:
            if key in diagnostics:
                legacy_identity[key] = diagnostics[key]
        return legacy_identity

    forward_model_sha256 = str(physical.get("forward_model_sha256") or diagnostics.get("model_sha256") or "")
    artifact_geom_sha = str(physical.get("artifact_geometry_sha256") or "")
    ebtel_sha256 = str(physical.get("ebtel_sha256") or diagnostics.get("ebtel_sha256") or "")
    return build_map_identity(
        a=float(normalized["a"]),
        b=float(normalized["b"]),
        q0=float(normalized.get("q0", np.nan)),
        domain=_domain_from_diagnostics(diagnostics),
        channel_or_frequency=_channel_or_frequency_from_diagnostics(diagnostics),
        component=str(component),
        forward_model_sha256=forward_model_sha256,
        forward_model_identity_version=str(
            physical.get("forward_model_identity_version") or diagnostics.get("forward_model_identity_version") or FORWARD_MODEL_IDENTITY_VERSION
        ),
        ebtel_sha256=ebtel_sha256,
        artifact_geometry_sha256=artifact_geom_sha,
        euv_response_sha256=physical.get("euv_response_sha256"),
        euv_response_identity_version=physical.get("euv_response_identity_version"),
        array_name=str(name),
    )


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


def _synthetic_map_entries_from_diagnostics(diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    raw_entries = diagnostics.get("synthetic_map_keys")
    if not isinstance(raw_entries, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw_entries:
        if not isinstance(item, dict):
            continue
        machine_key = str(item.get("machine_key", "")).strip()
        if not machine_key:
            continue
        out.append(
            {
                "machine_key": machine_key,
                "map_store_array": str(item.get("map_store_array", "")).strip(),
                "label": str(item.get("label", "")).strip(),
                "identity": item.get("identity") if isinstance(item.get("identity"), dict) else None,
            }
        )
    return out


def _register_synthetic_map_registry_entries(
    h5_file: h5py.File,
    *,
    map_refs: dict[str, str],
    diagnostics: dict[str, Any],
) -> list[str]:
    entries = _synthetic_map_entries_from_diagnostics(diagnostics)
    if not entries:
        return []

    registry_group = h5_file.require_group(MAP_STORE_GROUP).require_group(MAP_STORE_SYNTHETIC_REGISTRY_GROUP)
    registered: list[str] = []
    for entry in entries:
        machine_key = str(entry["machine_key"]).strip()
        if not machine_key:
            continue
        map_store_array = str(entry.get("map_store_array", "")).strip()
        map_ref_key_candidates = []
        if map_store_array:
            map_ref_key_candidates.append(map_store_array)
            if not str(map_store_array).startswith("extra/"):
                map_ref_key_candidates.append(f"extra/{map_store_array}")

        resolved_map_ref_path = ""
        for ref_key in map_ref_key_candidates:
            ref_path = str(map_refs.get(ref_key, "")).strip()
            if ref_path:
                resolved_map_ref_path = ref_path
                break
        if not resolved_map_ref_path:
            continue

        if machine_key not in registry_group:
            entry_group = registry_group.create_group(machine_key)
            _create_text_dataset(entry_group, "machine_key", machine_key)
            _create_text_dataset(entry_group, "map_store_array", str(map_store_array))
            _create_text_dataset(entry_group, "map_ref_path", str(resolved_map_ref_path))
            _create_text_dataset(entry_group, "label", str(entry.get("label", "")))
            identity = entry.get("identity") if isinstance(entry.get("identity"), dict) else None
            if identity is not None:
                _create_text_dataset(entry_group, "identity_json", _json_dumps(identity))
        registered.append(machine_key)
    return sorted(set(registered))


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
    if not ref_text or ref_text not in h5_file:
        return None
    node = h5_file[ref_text]
    if isinstance(node, h5py.Dataset):
        return np.asarray(node[()], dtype=float)
    if isinstance(node, h5py.Group) and "data" in node:
        return np.asarray(node["data"][()], dtype=float)
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


def _auxiliary_channel_token_from_descriptor(descriptor: dict[str, Any]) -> str:
    channel = str(descriptor.get("channel_label") or "").strip()
    if channel:
        return channel.lower()
    wavelength = _optional_float(descriptor.get("wavelength_angstrom"))
    if wavelength is not None:
        rounded = round(float(wavelength))
        if np.isclose(float(wavelength), float(rounded), rtol=0.0, atol=1e-9):
            return str(int(rounded)).lower()
        return f"{float(wavelength):.6g}".lower()
    label = str(descriptor.get("label", "")).strip()
    if label:
        return str(label.split()[0]).strip().lower()
    return ""


def _synthetic_registry_entries_for_descriptor(
    h5_file: h5py.File,
    *,
    descriptor: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if MAP_STORE_GROUP not in h5_file:
        return {}
    map_store = h5_file[MAP_STORE_GROUP]
    if MAP_STORE_SYNTHETIC_REGISTRY_GROUP not in map_store:
        return {}

    target_domain = str(descriptor.get("domain", "")).strip().lower()
    target_freq = _optional_float(descriptor.get("frequency_ghz"))
    target_channel = _auxiliary_channel_token_from_descriptor(descriptor)
    out: dict[str, dict[str, Any]] = {}
    registry_group = map_store[MAP_STORE_SYNTHETIC_REGISTRY_GROUP]

    def _dataset_text(group: h5py.Group, name: str) -> str:
        if name not in group:
            return ""
        try:
            return str(decode_scalar(group[name][()])).strip()
        except Exception:
            return ""

    for machine_key in sorted(str(key) for key in registry_group.keys()):
        entry_group = registry_group[machine_key]
        map_ref_path = _dataset_text(entry_group, "map_ref_path")
        if not map_ref_path:
            continue
        identity: dict[str, Any] = {}
        if "identity_json" in entry_group:
            try:
                parsed = json.loads(decode_scalar(entry_group["identity_json"][()]))
                if isinstance(parsed, dict):
                    identity = parsed
            except Exception:
                identity = {}

        identity_domain = str(
            identity.get("domain") or identity.get("domain_label") or identity.get("spectral_domain") or ""
        ).strip().lower()
        if identity_domain and target_domain and identity_domain != target_domain:
            continue
        channel_or_frequency = str(identity.get("channel_or_frequency") or "").strip().lower()
        if target_domain == "mw":
            if target_freq is None:
                continue
            expected_token = f"{float(target_freq):.6f}ghz".lower()
            component = str(identity.get("component") or identity.get("map_role") or "").strip().lower()
            if channel_or_frequency != expected_token:
                continue
            if component and component not in {"stokes_i", "stokes_v", "raw_modeled_best", ""}:
                if component not in {"stokes_i", "stokes_v"}:
                    continue
        elif target_domain in {"euv", "uv"}:
            if not target_channel:
                continue
            if channel_or_frequency != target_channel:
                continue
        else:
            continue

        map_role = str(identity.get("map_role") or identity.get("component") or "").strip()
        out[machine_key] = {
            "machine_key": machine_key,
            "map_ref_path": map_ref_path,
            "map_store_array": _dataset_text(entry_group, "map_store_array"),
            "label": _dataset_text(entry_group, "label"),
            "identity": identity,
            "map_role": map_role,
            "component": str(identity.get("component") or "").strip().lower(),
        }
    return out


def _record_synthetic_machine_keys(base_record: dict[str, Any]) -> list[str]:
    diagnostics = dict(base_record.get("diagnostics") or {})
    out: list[str] = []
    for item in diagnostics.get("synthetic_map_machine_keys", []):
        key = str(item).strip()
        if key:
            out.append(key)
    if out:
        return sorted(set(out))
    for item in diagnostics.get("synthetic_map_keys", []):
        if not isinstance(item, dict):
            continue
        key = str(item.get("machine_key", "")).strip()
        if key:
            out.append(key)
    return sorted(set(out))


def _trial_index_from_synthetic_map_role(map_role: str) -> int | None:
    match = re.match(r"^trial_(\d+)_(?:rendered|raw_modeled|corona|tr|stokes_i|stokes_v)$", str(map_role).strip().lower())
    if match is None:
        return None
    return int(match.group(1))


def load_auxiliary_map_store_point_records(
    h5_path: Path,
    *,
    slice_key: str,
    source_search_id: str | None = None,
    use_synthetic_machine_keys: bool = True,
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
        synthetic_entries = (
            _synthetic_registry_entries_for_descriptor(h5_file, descriptor=descriptor)
            if bool(use_synthetic_machine_keys)
            else {}
        )

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
                    base_record = _read_point_group_sparse(record_group)
                    trial_by_index: dict[int, np.ndarray] = {}
                    trial_corona_by_index: dict[int, np.ndarray] = {}
                    trial_tr_by_index: dict[int, np.ndarray] = {}
                    best_map: np.ndarray | None = None
                    best_corona: np.ndarray | None = None
                    best_tr: np.ndarray | None = None
                    matched_source = ""
                    matched_machine_keys: list[str] = []

                    if synthetic_entries:
                        for machine_key in _record_synthetic_machine_keys(base_record):
                            entry = synthetic_entries.get(machine_key)
                            if entry is None:
                                continue
                            arr = _read_map_store_ref_array(h5_file, entry.get("map_ref_path"))
                            if arr is None:
                                continue
                            map_role = str(entry.get("map_role", "")).strip().lower()
                            component = str(entry.get("component") or map_role).strip().lower()
                            if map_role in {"rendered_best", "raw_modeled_best", "stokes_i"} and best_map is None:
                                best_map = np.asarray(arr, dtype=float)
                            if component == "corona" or map_role in {"corona", "flux_corona_best"}:
                                best_corona = np.asarray(arr, dtype=float)
                            if component == "tr" or map_role in {"tr", "flux_tr_best"}:
                                best_tr = np.asarray(arr, dtype=float)
                            trial_index = _trial_index_from_synthetic_map_role(map_role)
                            if trial_index is not None:
                                if component == "corona" or map_role.endswith("_corona"):
                                    trial_corona_by_index[int(trial_index)] = np.asarray(arr, dtype=float)
                                elif component == "tr" or map_role.endswith("_tr"):
                                    trial_tr_by_index[int(trial_index)] = np.asarray(arr, dtype=float)
                                elif component == "stokes_i" or map_role.endswith("_stokes_i") or map_role.endswith("_raw_modeled"):
                                    trial_by_index[int(trial_index)] = np.asarray(arr, dtype=float)
                                else:
                                    trial_by_index[int(trial_index)] = np.asarray(arr, dtype=float)
                            matched_machine_keys.append(machine_key)
                        if matched_machine_keys:
                            matched_source = f"synthetic_registry/{str(slice_key)}"

                    if not matched_source:
                        matching_prefix = next(
                            (
                                prefix
                                for prefix in prefixes
                                if any(str(name).startswith(f"{prefix}/") for name in map_refs.keys())
                            ),
                            None,
                        )
                        if matching_prefix is not None:
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
                            matched_source = matching_prefix

                    if not trial_by_index and not trial_corona_by_index and best_map is None and best_corona is None:
                        continue

                    tr_mask = base_record.get("euv_tr_mask")
                    promoted = dict(base_record)
                    if trial_corona_by_index and trial_tr_by_index:
                        shared_indices = sorted(set(trial_corona_by_index.keys()) & set(trial_tr_by_index.keys()))
                        if shared_indices:
                            ordered_items = [(idx, trial_corona_by_index[idx]) for idx in shared_indices]
                            trial_maps = np.stack(
                                [
                                    _recombine_euv_raw_maps(
                                        trial_corona_by_index[idx],
                                        trial_tr_by_index[idx],
                                        tr_region_mask=tr_mask,
                                    )
                                    for idx in shared_indices
                                ],
                                axis=0,
                            )
                            trial_q0 = tuple(float(v) for v in promoted.get("fit_q0_trials", ()))
                            if len(trial_q0) != len(shared_indices):
                                source_q0 = list(float(v) for v in promoted.get("fit_q0_trials", ()))
                                trial_q0 = tuple(source_q0[idx] for idx in shared_indices if idx < len(source_q0))
                            if len(trial_q0) == trial_maps.shape[0]:
                                promoted["fit_q0_trials"] = trial_q0
                                promoted["trial_modeled_maps"] = trial_maps
                                promoted["trial_raw_modeled_maps"] = trial_maps.copy()
                                promoted["trial_euv_coronal_maps"] = np.stack(
                                    [trial_corona_by_index[idx] for idx in shared_indices],
                                    axis=0,
                                )
                                promoted["trial_euv_tr_maps"] = np.stack(
                                    [trial_tr_by_index[idx] for idx in shared_indices],
                                    axis=0,
                                )
                    elif trial_by_index:
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
                    if best_corona is not None and best_tr is not None:
                        combined_best = _recombine_euv_raw_maps(best_corona, best_tr, tr_region_mask=tr_mask)
                        promoted["modeled_best"] = np.asarray(combined_best, dtype=float)
                        promoted["raw_modeled_best"] = np.asarray(combined_best, dtype=float)
                        promoted["euv_coronal_best"] = np.asarray(best_corona, dtype=float)
                        promoted["euv_tr_best"] = np.asarray(best_tr, dtype=float)
                    elif best_map is not None:
                        promoted["modeled_best"] = np.asarray(best_map, dtype=float)
                        promoted["raw_modeled_best"] = np.asarray(best_map, dtype=float)
                    promoted["source_slice_key"] = source_slice_key
                    promoted["source_search_id"] = search_id
                    promoted["source_record_name"] = record_name
                    promoted["source_auxiliary_map_prefix"] = matched_source
                    if matched_machine_keys:
                        promoted["source_synthetic_machine_keys"] = sorted(set(matched_machine_keys))
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
    psf_kernel: np.ndarray | None = None,
    run_history: list[dict[str, Any]] | None = None,
    observation_canvas: np.ndarray | None = None,
    sigma_canvas: np.ndarray | None = None,
    canvas_wcs_header: fits.Header | None = None,
    store_observation_maps: bool = True,
) -> None:
    slice_descriptors, target_slice_key = canonical_slice_descriptors_from_diagnostics(diagnostics)
    trial_logging_policy = canonical_trial_logging_policy_from_diagnostics(diagnostics)
    geometry_block = build_artifact_geometry_block(diagnostics)
    geometry_sha = artifact_geometry_sha256(geometry_block)
    diagnostics_out = dict(diagnostics)
    diagnostics_out.setdefault("artifact_geometry_sha256", geometry_sha)
    if store_observation_maps:
        observed_store = np.asarray(observed, dtype=np.float32)
        sigma_store = np.asarray(sigma_map, dtype=np.float32)
        diagnostics_out = _sync_preprocessed_content_identity_diagnostics(
            diagnostics_out,
            observed=observed_store,
            sigma_map=sigma_store,
            observation_canvas=observation_canvas,
            sigma_canvas=sigma_canvas,
        )
        common.create_dataset("observed", data=observed_store, compression="gzip", compression_opts=4)
        common.create_dataset("sigma_map", data=sigma_store, compression="gzip", compression_opts=4)
        if observation_canvas is not None and sigma_canvas is not None and canvas_wcs_header is not None:
            common.create_dataset(
                "observation_canvas",
                data=np.asarray(observation_canvas, dtype=np.float32),
                compression="gzip",
                compression_opts=4,
            )
            common.create_dataset(
                "sigma_canvas",
                data=np.asarray(sigma_canvas, dtype=np.float32),
                compression="gzip",
                compression_opts=4,
            )
            _create_text_dataset(common, "canvas_wcs_header", canvas_wcs_header.tostring(sep="\n", endcard=True))
    _create_text_dataset(common, "wcs_header", wcs_header.tostring(sep="\n", endcard=True))
    _create_text_dataset(common, "diagnostics_json", _json_dumps(diagnostics_out))
    _create_text_dataset(common, COMMON_ARTIFACT_CONTRACT_VERSION_DATASET, CANONICAL_ARTIFACT_CONTRACT_VERSION)
    _create_text_dataset(common, COMMON_ARTIFACT_GEOMETRY_DATASET, _json_dumps(geometry_block))
    _create_text_dataset(common, COMMON_ARTIFACT_GEOMETRY_SHA256_DATASET, str(geometry_sha))
    _create_text_dataset(common, COMMON_SLICE_DESCRIPTORS_DATASET, _json_dumps(slice_descriptors))
    _create_text_dataset(common, COMMON_TARGET_SLICE_KEY_DATASET, str(target_slice_key))
    _create_text_dataset(common, COMMON_TRIAL_LOGGING_POLICY_DATASET, _json_dumps(trial_logging_policy))
    if psf_kernel is not None:
        kernel = np.asarray(psf_kernel, dtype=float)
        if kernel.ndim == 2 and kernel.size > 0:
            kernel_sum = float(np.nansum(kernel))
            if np.isfinite(kernel_sum) and kernel_sum != 0.0:
                kernel = kernel / kernel_sum
            common.create_dataset(
                COMMON_PSF_KERNEL_DATASET,
                data=np.asarray(kernel, dtype=np.float32),
                compression="gzip",
                compression_opts=4,
            )
            kernel_meta = {
                "source": diagnostics.get("psf_source"),
                "resolved_psf": diagnostics.get("resolved_psf"),
                "shape": [int(v) for v in kernel.shape],
                "normalized": True,
            }
            _create_text_dataset(common, COMMON_PSF_KERNEL_META_DATASET, _json_dumps(kernel_meta))
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
    observed = np.asarray(common["observed"], dtype=float) if "observed" in common else None
    sigma_map = np.asarray(common["sigma_map"], dtype=float) if "sigma_map" in common else None
    wcs_header = (
        fits.Header.fromstring(decode_scalar(common["wcs_header"][()]), sep="\n")
        if "wcs_header" in common
        else None
    )
    observation_canvas = (
        np.asarray(common["observation_canvas"], dtype=float) if "observation_canvas" in common else None
    )
    sigma_canvas = np.asarray(common["sigma_canvas"], dtype=float) if "sigma_canvas" in common else None
    canvas_wcs_header = (
        fits.Header.fromstring(decode_scalar(common["canvas_wcs_header"][()]), sep="\n")
        if "canvas_wcs_header" in common
        else None
    )
    diagnostics = json.loads(decode_scalar(common["diagnostics_json"][()]))
    run_history = _decode_run_history(common)
    artifact_contract_version = (
        decode_scalar(common[COMMON_ARTIFACT_CONTRACT_VERSION_DATASET][()])
        if COMMON_ARTIFACT_CONTRACT_VERSION_DATASET in common
        else CANONICAL_ARTIFACT_CONTRACT_VERSION
    )
    artifact_geometry_block = None
    artifact_geometry_sha256_value = None
    if COMMON_ARTIFACT_GEOMETRY_DATASET in common:
        try:
            parsed_geometry = json.loads(decode_scalar(common[COMMON_ARTIFACT_GEOMETRY_DATASET][()]))
            if isinstance(parsed_geometry, dict):
                artifact_geometry_block = parsed_geometry
        except Exception:
            artifact_geometry_block = None
    if COMMON_ARTIFACT_GEOMETRY_SHA256_DATASET in common:
        artifact_geometry_sha256_value = decode_scalar(common[COMMON_ARTIFACT_GEOMETRY_SHA256_DATASET][()])
    elif artifact_geometry_block is not None:
        artifact_geometry_sha256_value = artifact_geometry_sha256(artifact_geometry_block)
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
    psf_kernel = None
    if COMMON_PSF_KERNEL_DATASET in common:
        kernel = np.asarray(common[COMMON_PSF_KERNEL_DATASET], dtype=float)
        if kernel.ndim == 2 and kernel.size > 0:
            psf_kernel = kernel
    psf_kernel_metadata: dict[str, Any] | None = None
    if COMMON_PSF_KERNEL_META_DATASET in common:
        try:
            parsed = json.loads(decode_scalar(common[COMMON_PSF_KERNEL_META_DATASET][()]))
            if isinstance(parsed, dict):
                psf_kernel_metadata = parsed
        except Exception:
            psf_kernel_metadata = None
    return {
        "observed": observed,
        "sigma_map": sigma_map,
        "wcs_header": wcs_header,
        "observation_canvas": observation_canvas,
        "sigma_canvas": sigma_canvas,
        "canvas_wcs_header": canvas_wcs_header,
        "diagnostics": diagnostics,
        "run_history": run_history,
        "artifact_contract_version": artifact_contract_version,
        "artifact_geometry": artifact_geometry_block,
        "artifact_geometry_sha256": artifact_geometry_sha256_value,
        "slice_descriptors": slice_descriptors,
        "target_slice_key": target_slice_key,
        "trial_logging_policy": trial_logging_policy,
        "blos_reference": blos_reference,
        "psf_kernel": psf_kernel,
        "psf_kernel_metadata": psf_kernel_metadata,
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


def _copy_existing_map_store(src: h5py.File, dst: h5py.File) -> None:
    if MAP_STORE_GROUP in src and MAP_STORE_GROUP not in dst:
        src.copy(src[MAP_STORE_GROUP], dst, name=MAP_STORE_GROUP)


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
    psf_kernel: np.ndarray | None = None,
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
                _copy_existing_map_store(src, dst)
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
            psf_kernel=psf_kernel,
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
        rect_layout = {
            "kind": "rectangular_grid",
            "a_values": [float(v) for v in np.asarray(a_values, dtype=float)],
            "b_values": [float(v) for v in np.asarray(b_values, dtype=float)],
        }
        request_payload = _search_request_from_diagnostics(diagnostics_out, layout=rect_layout)
        current_search_id = _matching_search_id_for_request(searches_group, request_payload)
        if not current_search_id:
            current_search_id = _search_id_from_diagnostics(diagnostics_out, layout=rect_layout)
        _write_search_group(
            searches_group,
            search_id=current_search_id,
            diagnostics=diagnostics_out,
            point_records=current_point_records,
            run_history=run_history,
            layout=request_payload.get("layout") if isinstance(request_payload.get("layout"), dict) else None,
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
    diagnostics_out = dict(normalized["diagnostics"])
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
    grp.create_dataset("fit_q0_trials", data=np.asarray(normalized["fit_q0_trials"], dtype=np.float64))
    fit_metric_ds = grp.create_dataset("fit_metric_trials", data=np.asarray(normalized["fit_metric_trials"], dtype=np.float64))
    fit_metric_ds.attrs["target_metric"] = np.bytes_(str(normalized["target_metric"]))
    grp.create_dataset("fit_chi2_trials", data=np.asarray(normalized["fit_chi2_trials"], dtype=np.float64))
    grp.create_dataset("fit_rho2_trials", data=np.asarray(normalized["fit_rho2_trials"], dtype=np.float64))
    grp.create_dataset("fit_eta2_trials", data=np.asarray(normalized["fit_eta2_trials"], dtype=np.float64))
    if normalized["fit_shift_x_trials"]:
        grp.create_dataset("fit_shift_x_trials", data=np.asarray(normalized["fit_shift_x_trials"], dtype=np.float64))
        grp.create_dataset("fit_shift_y_trials", data=np.asarray(normalized["fit_shift_y_trials"], dtype=np.float64))
        grp.create_dataset(
            "fit_find_shift_valid_trials",
            data=np.asarray(normalized["fit_find_shift_valid_trials"], dtype=np.uint8),
        )
    if normalized["fit_trial_mask_stages"]:
        grp.create_dataset(
            "fit_trial_mask_stages",
            data=np.asarray([str(v) for v in normalized["fit_trial_mask_stages"]], dtype=object),
        )
    trial_history_entries, best_trial_index = _build_trial_history_entries(grp, normalized=normalized, map_refs=map_refs)
    if best_trial_index is not None:
        grp.attrs["best_trial_index"] = int(best_trial_index)
    _create_text_dataset(grp, TRIAL_HISTORY_DATASET, _json_dumps(trial_history_entries))
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
    if normalized["stokes_v_best"] is not None:
        _write_point_map_ref(grp, normalized=normalized, name="stokes_v_best", data=normalized["stokes_v_best"], map_refs=map_refs)
    if normalized["trial_stokes_v_maps"] is not None:
        _write_point_map_ref(grp, normalized=normalized, name="trial_stokes_v_maps", data=normalized["trial_stokes_v_maps"], map_refs=map_refs)
    for extra_name, extra_data in dict(normalized.get("map_store_arrays") or {}).items():
        _write_point_map_ref(
            grp,
            normalized=normalized,
            name=f"extra/{extra_name}",
            data=np.asarray(extra_data, dtype=float),
            map_refs=map_refs,
        )
    synthetic_machine_keys = _register_synthetic_map_registry_entries(
        grp.file,
        map_refs=map_refs,
        diagnostics=diagnostics_out,
    )
    if synthetic_machine_keys:
        compact_entries: list[dict[str, Any]] = []
        for item in _synthetic_map_entries_from_diagnostics(diagnostics_out):
            machine_key = str(item.get("machine_key", "")).strip()
            if not machine_key or machine_key not in synthetic_machine_keys:
                continue
            compact_entry = {
                "machine_key": machine_key,
                "map_store_array": str(item.get("map_store_array", "")).strip(),
            }
            if str(item.get("label", "")).strip():
                compact_entry["label"] = str(item.get("label", "")).strip()
            compact_entries.append(compact_entry)
        diagnostics_out["synthetic_map_machine_keys"] = list(synthetic_machine_keys)
        diagnostics_out["synthetic_map_keys"] = compact_entries
        _create_text_dataset(grp, POINT_SYNTHETIC_MAP_MACHINE_KEYS_DATASET, _json_dumps(synthetic_machine_keys))
    _create_text_dataset(grp, MAP_REFS_DATASET, _json_dumps(map_refs))
    _create_text_dataset(grp, "diagnostics_json", _json_dumps(diagnostics_out))


def write_point_scan_artifact(
    out_h5: Path,
    *,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
    blos_reference: tuple[np.ndarray, fits.Header] | None = None,
    point_records: list[dict[str, Any]],
    psf_kernel: np.ndarray | None = None,
    run_history: list[dict[str, Any]] | None = None,
    preserve_existing_searches: bool = True,
    observation_canvas: np.ndarray | None = None,
    sigma_canvas: np.ndarray | None = None,
    canvas_wcs_header: fits.Header | None = None,
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
                _copy_existing_map_store(src, f)
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
            psf_kernel=psf_kernel,
            run_history=run_history,
            observation_canvas=observation_canvas,
            sigma_canvas=sigma_canvas,
            canvas_wcs_header=canvas_wcs_header,
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
        if out_h5.exists() and bool(preserve_existing_searches):
            with _H5PY_FILE(out_h5, "r") as src:
                src_slice = None
                if SLICE_CONTAINER_GROUP in src and resolved_slice_key in src[SLICE_CONTAINER_GROUP]:
                    src_slice = src[SLICE_CONTAINER_GROUP][resolved_slice_key]
                elif "common" in src:
                    src_slice = src
                if src_slice is not None and SEARCHES_GROUP in src_slice:
                    for name in src_slice[SEARCHES_GROUP].keys():
                        src_slice[SEARCHES_GROUP].copy(name, searches_group, name=name)
        layout_payload = {"kind": "point_list"}
        request_payload = _search_request_from_diagnostics(diagnostics_out, layout=layout_payload)
        current_search_id = _matching_search_id_for_request(searches_group, request_payload)
        if not current_search_id:
            current_search_id = _search_id_from_diagnostics(diagnostics_out, layout=layout_payload)
        _write_search_group(
            searches_group,
            search_id=current_search_id,
            diagnostics=diagnostics_out,
            point_records=list(point_records),
            run_history=run_history,
            layout=layout_payload,
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
    psf_kernel: np.ndarray | None = None,
    run_history: list[dict[str, Any]] | None = None,
) -> None:
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=wcs_header,
        diagnostics=diagnostics,
        blos_reference=blos_reference,
        psf_kernel=psf_kernel,
        point_records=point_records,
        run_history=run_history,
        preserve_existing_searches=True,
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
    psf_kernel: np.ndarray | None = None,
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
        psf_kernel=psf_kernel,
        point_records=[point_payload],
        run_history=run_history,
        preserve_existing_searches=True,
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
    psf_kernel: np.ndarray | None = None,
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
            with _H5PY_FILE(out_h5, mode) as f:
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
                            psf_kernel=psf_kernel,
                            run_history=None,
                        )
                searches_group = slice_group.require_group(SEARCHES_GROUP)
                layout_payload = {"kind": "point_list"}
                request_payload = _search_request_from_diagnostics(diagnostics_out, layout=layout_payload)
                current_search_id = _matching_search_id_for_request(searches_group, request_payload)
                if not current_search_id:
                    current_search_id = _search_id_from_diagnostics(diagnostics_out, layout=layout_payload)
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
                    _create_text_dataset(
                        search_group,
                        SEARCH_REQUEST_DATASET,
                        _json_dumps(request_payload),
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
                existing_lifecycle = (
                    _json_loads_or_empty(search_group[SEARCH_LIFECYCLE_DATASET][()])
                    if SEARCH_LIFECYCLE_DATASET in search_group
                    else {}
                )
                status = _write_search_status_attrs(
                    search_group,
                    counts,
                    diagnostics=diagnostics_out,
                    remain_in_progress=_search_should_remain_in_progress(
                        diagnostics=diagnostics_out,
                        existing_lifecycle=existing_lifecycle,
                    ),
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
    psf_kernel: np.ndarray | None = None,
) -> None:
    append_scan_point_record(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=wcs_header,
        diagnostics=diagnostics,
        blos_reference=blos_reference,
        psf_kernel=psf_kernel,
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
    psf_kernel: np.ndarray | None = None,
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
            psf_kernel=psf_kernel,
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
            psf_kernel=psf_kernel,
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
        metrics = dict(record.get("metrics") or viewer_record_metrics(record))
        if not any(np.isfinite(float(metrics.get(name, np.nan))) for name in METRICS):
            metrics = viewer_record_metrics(record)
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


def extend_patch_grid_model_with_pending_point(
    display_model: dict[str, Any],
    *,
    a_value: float,
    b_value: float,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Add a ghost pending cell for a live active point not yet saved in the artifact."""
    records = list(display_model.get("records", []))
    a_target = float(a_value)
    b_target = float(b_value)
    for record in records:
        if np.isclose(float(record["a"]), a_target, rtol=0.0, atol=1e-12) and np.isclose(
            float(record["b"]), b_target, rtol=0.0, atol=1e-12
        ):
            return dict(display_model)
    diag = dict(diagnostics or {})
    a_coords = np.unique([float(record["a"]) for record in records] + [a_target])
    b_coords = np.unique([float(record["b"]) for record in records] + [b_target])
    if a_coords.size == 1:
        half_da = max(abs(float(diag.get("da", 0.3))) * 0.5, 0.05)
        a_spans = {a_target: (a_target - half_da, a_target + half_da)}
    else:
        a_spans = _axis_spans(a_coords)
    if b_coords.size == 1:
        half_db = max(abs(float(diag.get("db", 0.3))) * 0.5, 0.05)
        b_spans = {b_target: (b_target - half_db, b_target + half_db)}
    else:
        b_spans = _axis_spans(b_coords)
    a0, a1 = a_spans[a_target]
    b0, b1 = b_spans[b_target]
    pending_record = {
        "key": ("live", "pending"),
        "a_index": -1,
        "b_index": -1,
        "a": a_target,
        "b": b_target,
        "a0": a0,
        "a1": a1,
        "b0": b0,
        "b1": b1,
        "a_center": 0.5 * (a0 + a1),
        "b_center": 0.5 * (b0 + b1),
        "metrics": {"chi2": np.nan, "rho2": np.nan, "eta2": np.nan},
        "status": "pending",
        "q0": np.nan,
        "success": False,
        "live_pending": True,
    }
    merged_records = records + [pending_record]
    return {
        "records": merged_records,
        "a_min": float(min(record["a0"] for record in merged_records)),
        "a_max": float(max(record["a1"] for record in merged_records)),
        "b_min": float(min(record["b0"] for record in merged_records)),
        "b_max": float(max(record["b1"] for record in merged_records)),
    }


def reopen_search_runner_state(
    h5_path: Path,
    *,
    slice_key: str | None = None,
    search_id: str | None = None,
) -> None:
    """Mark a resumed expand/recompute search active again (clears stale completion metadata)."""
    with _H5PY_FILE(h5_path, "a") as f:
        group, _descriptors, _selected_key = _resolve_slice_group(
            f,
            slice_key=slice_key,
            allow_missing=True,
        )
        if group is None or SEARCHES_GROUP not in group:
            return
        selected_search_id = _selected_search_id(group, requested_search_id=search_id)
        if selected_search_id is None or selected_search_id not in group[SEARCHES_GROUP]:
            return
        search_group = group[SEARCHES_GROUP][selected_search_id]
        counts = {
            "total": int(search_group.attrs.get("total_point_count", 0)),
            "pending": int(search_group.attrs.get("pending_point_count", 0)),
            "missing": int(search_group.attrs.get("missing_point_count", 0)),
            "failed": int(search_group.attrs.get("failed_point_count", 0)),
            "computed": int(search_group.attrs.get("computed_point_count", 0)),
            "other": int(search_group.attrs.get("other_point_count", 0)),
        }
        diagnostics_out = (
            _json_loads_or_empty(search_group["diagnostics_json"][()])
            if "diagnostics_json" in search_group
            else {}
        )
        diagnostics_out["search_active"] = True
        diagnostics_out.pop("search_completed_at", None)
        _replace_text_dataset(search_group, "diagnostics_json", _json_dumps(diagnostics_out))
        existing_lifecycle = (
            _json_loads_or_empty(search_group[SEARCH_LIFECYCLE_DATASET][()])
            if SEARCH_LIFECYCLE_DATASET in search_group
            else {}
        )
        status = _write_search_status_attrs(
            search_group,
            counts,
            diagnostics=diagnostics_out,
            remain_in_progress=True,
        )
        lifecycle = _search_lifecycle_payload(
            status=status,
            diagnostics=diagnostics_out,
            existing=existing_lifecycle,
        )
        lifecycle["active"] = True
        lifecycle["in_progress"] = True
        lifecycle["completed_at"] = None
        _write_search_lifecycle_dataset(search_group, lifecycle=lifecycle)


def finalize_search_runner_state(
    h5_path: Path,
    *,
    slice_key: str | None = None,
    search_id: str | None = None,
    completed: bool = True,
) -> None:
    """Mark an adaptive search inactive once the runner exits."""
    with _H5PY_FILE(h5_path, "a") as f:
        group, _descriptors, _selected_key = _resolve_slice_group(
            f,
            slice_key=slice_key,
            allow_missing=True,
        )
        if group is None or SEARCHES_GROUP not in group:
            return
        selected_search_id = _selected_search_id(group, requested_search_id=search_id)
        if selected_search_id is None or selected_search_id not in group[SEARCHES_GROUP]:
            return
        search_group = group[SEARCHES_GROUP][selected_search_id]
        counts = {
            "total": int(search_group.attrs.get("total_point_count", 0)),
            "pending": int(search_group.attrs.get("pending_point_count", 0)),
            "missing": int(search_group.attrs.get("missing_point_count", 0)),
            "failed": int(search_group.attrs.get("failed_point_count", 0)),
            "computed": int(search_group.attrs.get("computed_point_count", 0)),
            "other": int(search_group.attrs.get("other_point_count", 0)),
        }
        if int(counts["total"]) <= 0 and "point_records" in search_group:
            counts = _search_status_counts_from_records(
                _load_sparse_point_records(search_group["point_records"], include_maps=False)
            )
        diagnostics_out = (
            _json_loads_or_empty(search_group["diagnostics_json"][()])
            if "diagnostics_json" in search_group
            else {}
        )
        diagnostics_out["search_active"] = False
        if completed:
            diagnostics_out["search_completed_at"] = _utc_now_iso()
        _replace_text_dataset(search_group, "diagnostics_json", _json_dumps(diagnostics_out))
        existing_lifecycle = (
            _json_loads_or_empty(search_group[SEARCH_LIFECYCLE_DATASET][()])
            if SEARCH_LIFECYCLE_DATASET in search_group
            else {}
        )
        status = _write_search_status_attrs(
            search_group,
            counts,
            diagnostics=diagnostics_out,
            remain_in_progress=False,
        )
        lifecycle = _search_lifecycle_payload(
            status=status,
            diagnostics=diagnostics_out,
            existing=existing_lifecycle,
        )
        lifecycle["active"] = False
        lifecycle["in_progress"] = False
        if completed and not lifecycle.get("completed_at"):
            lifecycle["completed_at"] = diagnostics_out.get("search_completed_at")
        _write_search_lifecycle_dataset(search_group, lifecycle=lifecycle)


def grid_extents_from_ab_values(a_values: Any, b_values: Any) -> dict[str, float]:
    a_coords = np.unique(np.asarray(a_values, dtype=float))
    b_coords = np.unique(np.asarray(b_values, dtype=float))
    if a_coords.size == 0 or b_coords.size == 0:
        return {"a_min": 0.0, "a_max": 1.0, "b_min": 0.0, "b_max": 1.0}
    a_spans = _axis_spans(a_coords)
    b_spans = _axis_spans(b_coords)
    return {
        "a_min": float(min(bounds[0] for bounds in a_spans.values())),
        "a_max": float(max(bounds[1] for bounds in a_spans.values())),
        "b_min": float(min(bounds[0] for bounds in b_spans.values())),
        "b_max": float(max(bounds[1] for bounds in b_spans.values())),
    }


def grid_extents_from_search_diagnostics(diagnostics: dict[str, Any] | None) -> dict[str, float] | None:
    """Return declared adaptive (a, b) bounds when stored on a search."""
    diag = dict(diagnostics or {})
    a_range = diag.get("a_range")
    b_range = diag.get("b_range")
    if not isinstance(a_range, (list, tuple)) or len(a_range) < 2:
        return None
    if not isinstance(b_range, (list, tuple)) or len(b_range) < 2:
        return None
    try:
        a_lo = float(min(a_range[0], a_range[1]))
        a_hi = float(max(a_range[0], a_range[1]))
        b_lo = float(min(b_range[0], b_range[1]))
        b_hi = float(max(b_range[0], b_range[1]))
    except Exception:
        return None
    if not (np.isfinite(a_lo) and np.isfinite(a_hi) and np.isfinite(b_lo) and np.isfinite(b_hi)):
        return None
    if a_lo >= a_hi or b_lo >= b_hi:
        return None
    return {"a_min": a_lo, "a_max": a_hi, "b_min": b_lo, "b_max": b_hi}


def merge_grid_extents(*candidates: dict[str, float] | None) -> dict[str, float] | None:
    merged: dict[str, float] | None = None
    for extents in candidates:
        if not isinstance(extents, dict):
            continue
        if merged is None:
            merged = dict(extents)
            continue
        merged["a_min"] = float(min(merged["a_min"], extents["a_min"]))
        merged["a_max"] = float(max(merged["a_max"], extents["a_max"]))
        merged["b_min"] = float(min(merged["b_min"], extents["b_min"]))
        merged["b_max"] = float(max(merged["b_max"], extents["b_max"]))
    return merged


def load_slice_grid_extents(
    h5_path: Path,
    *,
    slice_key: str,
    search_id: str | None = None,
) -> dict[str, float] | None:
    with _H5PY_FILE(h5_path, "r") as f:
        group, _descriptors, _selected_key = _resolve_slice_group(
            f,
            slice_key=slice_key,
            allow_missing=True,
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
        diagnostics = (
            _json_loads_or_empty(search_group["diagnostics_json"][()])
            if "diagnostics_json" in search_group
            else {}
        )
        layout = _json_loads_or_empty(search_group["layout_json"][()]) if "layout_json" in search_group else {}
        point_extents: dict[str, float] | None = None
        if str(layout.get("kind", "")).strip().lower() == "rectangular_grid":
            a_values = layout.get("a_values", [])
            b_values = layout.get("b_values", [])
            if a_values and b_values:
                point_extents = grid_extents_from_ab_values(a_values, b_values)
        from .grid_points import GRID_POINTS_GROUP, list_grid_point_headers

        if point_extents is None and GRID_POINTS_GROUP in search_group:
            headers = list_grid_point_headers(search_group)
            if headers:
                unique_a = np.unique([float(item["a"]) for item in headers])
                unique_b = np.unique([float(item["b"]) for item in headers])
                if unique_a.size and unique_b.size:
                    point_extents = grid_extents_from_ab_values(unique_a, unique_b)
        if point_extents is None and "point_records" in search_group:
            records = _load_sparse_point_records(search_group["point_records"], include_maps=False)
            active_records = [
                record for record in records if str(record.get("status", "computed")).strip().lower() != "missing"
            ]
            if active_records:
                unique_a = np.unique([float(record["a"]) for record in active_records])
                unique_b = np.unique([float(record["b"]) for record in active_records])
                if unique_a.size and unique_b.size:
                    point_extents = grid_extents_from_ab_values(unique_a, unique_b)
        if point_extents is None:
            try:
                payload = load_scan_file(
                    h5_path,
                    slice_key=slice_key,
                    search_id=selected_search_id,
                    include_maps=False,
                )
            except Exception:
                payload = None
            if isinstance(payload, dict):
                a_values = np.asarray(payload.get("a_values", ()), dtype=float)
                b_values = np.asarray(payload.get("b_values", ()), dtype=float)
                if a_values.size and b_values.size:
                    point_extents = grid_extents_from_ab_values(a_values, b_values)
        return merge_grid_extents(point_extents, grid_extents_from_search_diagnostics(diagnostics))


def load_shared_grid_extents(
    h5_path: Path,
    *,
    search_id: str | None = None,
    slice_keys: list[str] | None = None,
) -> dict[str, float] | None:
    with _H5PY_FILE(h5_path, "r") as f:
        _group, descriptors, _selected_key = _resolve_slice_group(
            f,
            slice_key=None,
            allow_missing=False,
        )
    keys = [str(key).strip() for key in (slice_keys or []) if str(key).strip()]
    if not keys:
        keys = [str(descriptor.get("key", "")).strip() for descriptor in descriptors]
        keys = [key for key in keys if key]
    if not keys:
        return None
    merged: dict[str, float] | None = None
    for slice_key in keys:
        if search_id:
            slice_extents = load_slice_grid_extents(h5_path, slice_key=slice_key, search_id=search_id)
            candidates = [slice_extents] if slice_extents is not None else []
        else:
            candidates = _slice_grid_extents_for_all_searches(h5_path, slice_key=slice_key)
        for extents in candidates:
            if extents is None:
                continue
            if merged is None:
                merged = dict(extents)
                continue
            merged["a_min"] = float(min(merged["a_min"], extents["a_min"]))
            merged["a_max"] = float(max(merged["a_max"], extents["a_max"]))
            merged["b_min"] = float(min(merged["b_min"], extents["b_min"]))
            merged["b_max"] = float(max(merged["b_max"], extents["b_max"]))
    return merged


def _slice_grid_extents_for_all_searches(h5_path: Path, *, slice_key: str) -> list[dict[str, float]]:
    with _H5PY_FILE(h5_path, "r") as f:
        group, _descriptors, _selected_key = _resolve_slice_group(
            f,
            slice_key=slice_key,
            allow_missing=True,
        )
        if group is None or SEARCHES_GROUP not in group:
            return []
        search_ids = [str(name).strip() for name in group[SEARCHES_GROUP].keys() if str(name).strip()]
    extents: list[dict[str, float]] = []
    for search_id in search_ids:
        slice_extents = load_slice_grid_extents(h5_path, slice_key=slice_key, search_id=search_id)
        if slice_extents is not None:
            extents.append(slice_extents)
    return extents


def point_records_for_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    source_records = payload.get("point_records")
    if source_records is None:
        source_records = list(dict(payload.get("points", {})).values())
    return [record for record in source_records if str(record.get("status", "computed")) != "missing"]


def _point_index_from_record(payload: dict[str, Any], record: dict[str, Any]) -> tuple[int, int]:
    if "a_index" in record and "b_index" in record:
        return int(record["a_index"]), int(record["b_index"])
    a_values = np.asarray(payload.get("a_values", ()), dtype=float)
    b_values = np.asarray(payload.get("b_values", ()), dtype=float)
    return nearest_index(a_values, float(record["a"])), nearest_index(b_values, float(record["b"]))


def default_point_index(payload: dict[str, Any], metric: str | None = None) -> tuple[int, int]:
    records = point_records_for_payload(payload)
    if not records:
        raise ValueError("No stored point records are available")

    metric_name = str(metric or "").strip().lower()
    if metric_name in METRICS:
        ranked: list[tuple[float, dict[str, Any]]] = []
        for record in records:
            diagnostics = dict(record.get("diagnostics", {}))
            metrics = dict(record.get("metrics", {}))
            value = metrics.get(metric_name, diagnostics.get(metric_name, np.nan))
            try:
                numeric = float(value)
            except Exception:
                numeric = float("nan")
            if np.isfinite(numeric):
                ranked.append((numeric, record))
        if ranked:
            return _point_index_from_record(payload, min(ranked, key=lambda item: item[0])[1])

    latest_record = max(
        records,
        key=lambda record: (
            int(record.get("record_order", -1)),
            float(record.get("a", np.nan)),
            float(record.get("b", np.nan)),
        ),
    )
    return _point_index_from_record(payload, latest_record)


def resolve_point_index(
    payload: dict[str, Any],
    *,
    metric: str | None = None,
    a_index: int | None = None,
    b_index: int | None = None,
) -> tuple[int, int]:
    points = dict(payload.get("points", {}))
    if not points:
        raise ValueError("No stored point records are available")

    if a_index is not None and b_index is not None:
        key = (int(a_index), int(b_index))
        if key in points:
            status = str(points[key].get("status", "computed")).strip().lower()
            # The viewer pre-fills the rectangular grid with placeholder
            # "missing" records. Treat those as unresolved so we can fall back
            # to the nearest real computed point.
            if status not in {"missing", "pending"}:
                return key

    records = point_records_for_payload(payload)
    if not records:
        raise ValueError("No stored point records are available")

    fallback_index = default_point_index(payload, metric)
    fallback_record = points.get(fallback_index)
    fallback_a = float(fallback_record["a"]) if fallback_record is not None else float(records[0]["a"])
    fallback_b = float(fallback_record["b"]) if fallback_record is not None else float(records[0]["b"])

    a_values = np.asarray(payload.get("a_values", ()), dtype=float)
    b_values = np.asarray(payload.get("b_values", ()), dtype=float)
    target_a = fallback_a
    target_b = fallback_b
    if a_index is not None and a_values.size:
        target_a = float(a_values[int(np.clip(int(a_index), 0, max(0, a_values.size - 1)))])
    if b_index is not None and b_values.size:
        target_b = float(b_values[int(np.clip(int(b_index), 0, max(0, b_values.size - 1)))])

    a_coords = np.asarray([float(record["a"]) for record in records], dtype=float)
    b_coords = np.asarray([float(record["b"]) for record in records], dtype=float)
    a_scale = max(float(np.nanmax(a_coords) - np.nanmin(a_coords)), 1.0)
    b_scale = max(float(np.nanmax(b_coords) - np.nanmin(b_coords)), 1.0)
    best_record = min(
        records,
        key=lambda record: (
            ((float(record["a"]) - target_a) / a_scale) ** 2 + ((float(record["b"]) - target_b) / b_scale) ** 2,
            -int(record.get("record_order", -1)),
        ),
    )
    return _point_index_from_record(payload, best_record)


def grid_patch_rectangle(record: dict[str, Any]) -> tuple[float, float, float, float]:
    """Matplotlib Rectangle (x, y, width, height) with *a* on x and *b* on y."""
    return (
        float(record["a0"]),
        float(record["b0"]),
        float(record["a1"]) - float(record["a0"]),
        float(record["b1"]) - float(record["b0"]),
    )


def find_record_for_point(model: dict[str, Any], x: float, y: float) -> dict[str, Any] | None:
    """Return the grid cell under plot coordinates (*a*, *b*) = (*x*, *y*)."""
    for record in model.get("records", []):
        if float(record["a0"]) <= float(x) <= float(record["a1"]) and float(record["b0"]) <= float(y) <= float(record["b1"]):
            return record
    return None


def grid_indices_for_coordinates(payload: dict[str, Any], x: float, y: float) -> tuple[int, int] | None:
    """Map heatmap click coordinates to grid indices, including empty/missing cells."""
    a_values = np.asarray(payload.get("a_values", ()), dtype=float)
    b_values = np.asarray(payload.get("b_values", ()), dtype=float)
    if a_values.size == 0 or b_values.size == 0:
        return None
    a_spans = _axis_spans(a_values)
    b_spans = _axis_spans(b_values)
    a_index: int | None = None
    b_index: int | None = None
    for index, a_value in enumerate(a_values):
        a0, a1 = a_spans[float(a_value)]
        if float(a0) <= float(x) <= float(a1):
            a_index = int(index)
            break
    for index, b_value in enumerate(b_values):
        b0, b1 = b_spans[float(b_value)]
        if float(b0) <= float(y) <= float(b1):
            b_index = int(index)
            break
    if a_index is None or b_index is None:
        return None
    return a_index, b_index


def best_grid_index(payload: dict[str, Any], metric: str) -> tuple[int, int]:
    metric_name = str(metric).strip().lower()
    if metric_name not in METRICS:
        raise ValueError(f"Unsupported best-of-grid metric: {metric_name}")
    records = point_records_for_payload(payload)
    ranked: list[tuple[float, dict[str, Any]]] = []
    for record in records:
        diagnostics = dict(record.get("diagnostics", {}))
        metrics = dict(record.get("metrics", {}))
        value = metrics.get(metric_name, diagnostics.get(metric_name, np.nan))
        try:
            numeric = float(value)
        except Exception:
            numeric = float("nan")
        if np.isfinite(numeric):
            ranked.append((numeric, record))
    if not ranked:
        raise ValueError(f"No finite values available for metric {metric_name}")
    return _point_index_from_record(payload, min(ranked, key=lambda item: item[0])[1])


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
