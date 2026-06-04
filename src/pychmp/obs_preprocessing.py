"""Shared observational preprocessing before metric comparison.

Pipeline order (required for ``compute_metrics`` / ``fit_q0_to_observation``):

1. **Epoch alignment** — differential solar rotation when observation and model
   times differ modestly on the same UTC day (``obs_time_alignment``).
2. **FOV regrid** — resample the (possibly rotated) observation onto the render
   grid defined by ``target_header`` (same FOV/resolution as gxrender output).

Neither ``metrics.compute_metrics`` nor ``fitting.fit_q0_to_observation`` apply
rotation or regridding; callers must pass shape-matched arrays.

Slice workflows should call ``resolve_slice_observation_reference`` once when a
spectral slice is initialized. The rotated+regridded canvas observation/sigma maps
(max-padded shift envelope) and model-FOV extracts are stored in slice ``common``
and shared by all searches on that slice; per-search shift policy applies at eval time.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy.io import fits
from scipy.ndimage import map_coordinates

from .obs_time_alignment import (
    DEFAULT_MAX_ROTATION_SECONDS,
    ObsModelTimeAlignment,
    align_observation_to_model_time,
    assess_obs_model_time_alignment,
)
from .search_contract import normalize_shift_policy

SLICE_OBSERVATION_REFERENCE_SCHEMA = "pychmp.slice_observation_reference.v2"
CHMP_EVAL_POLICY_VERSION = "1"

SLICE_OBSERVATION_SOURCE_IDENTITY_KEYS = (
    "slice_observation_reference_schema",
    "observation_source_sha256",
    "artifact_geometry_sha256",
    "model_time_reference",
    "observation_time_original",
    "observation_time_alignment",
    "observation_time_rotation_applied",
    "slice_canvas_max_shift_arcsec",
    "chmp_eval_policy_version",
)

SLICE_OBSERVATION_CONTENT_IDENTITY_KEYS = (
    "preprocessed_observation_sha256",
    "preprocessed_sigma_sha256",
    "preprocessed_observation_shape",
    "preprocessed_sigma_shape",
    "observation_canvas_sha256",
    "sigma_canvas_sha256",
    "observation_canvas_shape",
    "sigma_canvas_shape",
)


class SliceObservationReferenceError(ValueError):
    """Raised when a stored slice observation reference cannot be restored safely."""


@dataclass(frozen=True, slots=True)
class SliceObservationReference:
    """Rotated+regridded observation reference for one search slice."""

    observed: np.ndarray
    sigma: np.ndarray
    source_header: fits.Header
    target_header: fits.Header
    diagnostics: dict[str, Any]
    identity: dict[str, Any]
    restored_from_artifact: bool
    shift_policy: str = "fixed"
    max_shift_arcsec: float | None = None
    xy_shift_arcsec: tuple[float, float] = (0.0, 0.0)
    observation_canvas: np.ndarray | None = None
    sigma_canvas: np.ndarray | None = None
    canvas_header: fits.Header | None = None
    canvas_pad_x: int = 0
    canvas_pad_y: int = 0


def compute_array_content_sha256(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def artifact_storage_content_sha256(array: np.ndarray) -> str:
    """Hash map content using the float32 representation stored in unified artifacts."""
    return compute_array_content_sha256(np.asarray(array, dtype=np.float32))


def slice_observation_identity_sha256(identity: dict[str, Any]) -> str:
    payload = {
        key: identity.get(key)
        for key in (*SLICE_OBSERVATION_SOURCE_IDENTITY_KEYS, *SLICE_OBSERVATION_CONTENT_IDENTITY_KEYS)
    }
    canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_slice_observation_source_identity(
    *,
    observation_source_sha256: str | None,
    artifact_geometry_sha256: str,
    model_time_text: str | None,
    observation_time_text: str | None,
    max_rotation_seconds: float = DEFAULT_MAX_ROTATION_SECONDS,
    slice_canvas_max_shift_arcsec: float | None = None,
    chmp_eval_policy_version: str = CHMP_EVAL_POLICY_VERSION,
) -> dict[str, Any]:
    alignment = assess_obs_model_time_alignment(
        observation_time_text,
        model_time_text,
        max_rotation_seconds=max_rotation_seconds,
    )
    return {
        "slice_observation_reference_schema": SLICE_OBSERVATION_REFERENCE_SCHEMA,
        "observation_source_sha256": str(observation_source_sha256 or ""),
        "artifact_geometry_sha256": str(artifact_geometry_sha256),
        "model_time_reference": alignment.model_time_text,
        "observation_time_original": alignment.observation_time_text,
        "observation_time_alignment": alignment.compatibility,
        "observation_time_rotation_applied": bool(model_time_text and alignment.should_rotate()),
        "slice_canvas_max_shift_arcsec": None
        if slice_canvas_max_shift_arcsec is None
        else float(slice_canvas_max_shift_arcsec),
        "chmp_eval_policy_version": str(chmp_eval_policy_version),
    }


def build_slice_observation_content_identity(
    observed: np.ndarray,
    sigma: np.ndarray,
    *,
    observation_canvas: np.ndarray | None = None,
    sigma_canvas: np.ndarray | None = None,
) -> dict[str, Any]:
    observed_arr = np.asarray(observed, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    if observed_arr.shape != sigma_arr.shape:
        raise ValueError("preprocessed observed and sigma maps must have identical shapes")
    identity = {
        "preprocessed_observation_sha256": compute_array_content_sha256(observed_arr),
        "preprocessed_sigma_sha256": compute_array_content_sha256(sigma_arr),
        "preprocessed_observation_shape": [int(v) for v in observed_arr.shape],
        "preprocessed_sigma_shape": [int(v) for v in sigma_arr.shape],
    }
    if observation_canvas is not None and sigma_canvas is not None:
        canvas_obs = np.asarray(observation_canvas, dtype=float)
        canvas_sigma = np.asarray(sigma_canvas, dtype=float)
        identity.update(
            {
                "observation_canvas_sha256": compute_array_content_sha256(canvas_obs),
                "sigma_canvas_sha256": compute_array_content_sha256(canvas_sigma),
                "observation_canvas_shape": [int(v) for v in canvas_obs.shape],
                "sigma_canvas_shape": [int(v) for v in canvas_sigma.shape],
            }
        )
    else:
        identity.update(
            {
                "observation_canvas_sha256": "",
                "sigma_canvas_sha256": "",
                "observation_canvas_shape": [],
                "sigma_canvas_shape": [],
            }
        )
    return identity


def build_slice_observation_identity(
    *,
    observation_source_sha256: str | None,
    artifact_geometry_sha256: str,
    model_time_text: str | None,
    observation_time_text: str | None,
    observed: np.ndarray,
    sigma: np.ndarray,
    max_rotation_seconds: float = DEFAULT_MAX_ROTATION_SECONDS,
    slice_canvas_max_shift_arcsec: float | None = None,
    observation_canvas: np.ndarray | None = None,
    sigma_canvas: np.ndarray | None = None,
) -> dict[str, Any]:
    identity = {
        **build_slice_observation_source_identity(
            observation_source_sha256=observation_source_sha256,
            artifact_geometry_sha256=artifact_geometry_sha256,
            model_time_text=model_time_text,
            observation_time_text=observation_time_text,
            max_rotation_seconds=max_rotation_seconds,
            slice_canvas_max_shift_arcsec=slice_canvas_max_shift_arcsec,
        ),
        **build_slice_observation_content_identity(
            observed,
            sigma,
            observation_canvas=observation_canvas,
            sigma_canvas=sigma_canvas,
        ),
    }
    identity["slice_observation_identity_sha256"] = slice_observation_identity_sha256(identity)
    return identity


def _canonical_header_text(header: fits.Header) -> str:
    return header.tostring(sep="\n", endcard=True)


def _normalize_slice_source_identity(identity: dict[str, Any]) -> dict[str, Any]:
    out = dict(identity)
    if out.get("slice_canvas_max_shift_arcsec") in {None, ""}:
        legacy_max_shift = out.get("max_shift_arcsec")
        if legacy_max_shift not in {None, ""}:
            out["slice_canvas_max_shift_arcsec"] = legacy_max_shift
    if out.get("chmp_eval_policy_version") in {None, ""}:
        out["chmp_eval_policy_version"] = CHMP_EVAL_POLICY_VERSION
    return out


def _source_identities_match(stored: dict[str, Any], expected: dict[str, Any]) -> bool:
    stored_norm = _normalize_slice_source_identity(stored)
    expected_norm = _normalize_slice_source_identity(expected)
    for key in SLICE_OBSERVATION_SOURCE_IDENTITY_KEYS:
        if str(stored_norm.get(key, "")) != str(expected_norm.get(key, "")):
            return False
    return True


def _content_identity_matches_arrays(content_identity: dict[str, Any], observed: np.ndarray, sigma: np.ndarray) -> bool:
    expected_obs_sha = str(content_identity.get("preprocessed_observation_sha256", "")).strip()
    expected_sigma_sha = str(content_identity.get("preprocessed_sigma_sha256", "")).strip()
    if not expected_obs_sha or not expected_sigma_sha:
        return False
    return (
        expected_obs_sha == compute_array_content_sha256(observed)
        and expected_sigma_sha == compute_array_content_sha256(sigma)
    )


def _stored_slice_observation_identity(diagnostics: dict[str, Any]) -> dict[str, Any] | None:
    if not diagnostics:
        return None
    content_keys_present = all(str(diagnostics.get(key, "")).strip() for key in (
        "preprocessed_observation_sha256",
        "preprocessed_sigma_sha256",
    ))
    source_keys_present = all(key in diagnostics for key in (
        "slice_observation_reference_schema",
        "observation_source_sha256",
        "artifact_geometry_sha256",
    ))
    if not content_keys_present or not source_keys_present:
        return None
    keys = (*SLICE_OBSERVATION_SOURCE_IDENTITY_KEYS, *SLICE_OBSERVATION_CONTENT_IDENTITY_KEYS)
    identity = {key: diagnostics.get(key) for key in keys}
    identity["slice_observation_identity_sha256"] = str(
        diagnostics.get("slice_observation_identity_sha256")
        or slice_observation_identity_sha256(identity)
    )
    return identity


def _legacy_source_identity_from_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any] | None:
    observation_source_sha256 = str(
        diagnostics.get("observation_source_sha256") or diagnostics.get("fits_sha256") or ""
    ).strip()
    artifact_geometry_sha256_value = str(diagnostics.get("artifact_geometry_sha256") or "").strip()
    if not observation_source_sha256 and not artifact_geometry_sha256_value:
        return None
    return build_slice_observation_source_identity(
        observation_source_sha256=observation_source_sha256 or None,
        artifact_geometry_sha256=artifact_geometry_sha256_value,
        model_time_text=str(diagnostics.get("model_time_reference") or diagnostics.get("observer_obs_time") or "").strip() or None,
        observation_time_text=str(diagnostics.get("observation_time_original") or "").strip() or None,
    )


def _try_restore_slice_observation_reference(
    *,
    target_header: fits.Header,
    expected_source_identity: dict[str, Any],
    stored_slice_payload: dict[str, Any],
    shift_policy: str = "fixed",
    max_shift_arcsec: float | None = None,
    xy_shift_arcsec: tuple[float, float] = (0.0, 0.0),
) -> SliceObservationReference | None:
    stored_observed = stored_slice_payload.get("observed")
    stored_sigma = stored_slice_payload.get("sigma_map")
    stored_header = stored_slice_payload.get("wcs_header")
    stored_canvas = stored_slice_payload.get("observation_canvas")
    stored_sigma_canvas = stored_slice_payload.get("sigma_canvas")
    stored_canvas_header = stored_slice_payload.get("canvas_wcs_header")
    stored_diag = dict(stored_slice_payload.get("diagnostics") or {})
    if not isinstance(stored_header, fits.Header):
        return None

    canvas_header = stored_canvas_header.copy() if isinstance(stored_canvas_header, fits.Header) else None
    canvas_arr = None if stored_canvas is None else np.asarray(stored_canvas, dtype=float)
    sigma_canvas_arr = None if stored_sigma_canvas is None else np.asarray(stored_sigma_canvas, dtype=float)

    if stored_observed is None or stored_sigma is None:
        if canvas_arr is None or sigma_canvas_arr is None or canvas_header is None:
            return None
        from .obs_alignment import extract_observation_to_model_fov

        stored_observed, stored_sigma = extract_observation_to_model_fov(
            canvas_arr,
            canvas_header,
            target_header,
            shift_x_arcsec=0.0,
            shift_y_arcsec=0.0,
            canvas_sigma=sigma_canvas_arr,
        )
        if stored_sigma is None:
            return None

    observed_arr = np.asarray(stored_observed, dtype=float)
    sigma_arr = np.asarray(stored_sigma, dtype=float)
    if observed_arr.ndim != 2 or sigma_arr.shape != observed_arr.shape:
        return None
    if _canonical_header_text(stored_header) != _canonical_header_text(target_header):
        return None

    stored_identity = _stored_slice_observation_identity(stored_diag)
    if stored_identity is not None:
        if not _source_identities_match(stored_identity, expected_source_identity):
            return None
        if not _content_identity_matches_arrays(stored_identity, observed_arr, sigma_arr):
            # Metadata content hashes can drift when common maps are rewritten without
            # updating diagnostics_json. Trust the stored arrays when source identity matches.
            pass
    else:
        legacy_source = _legacy_source_identity_from_diagnostics(stored_diag)
        if legacy_source is None or not _source_identities_match(legacy_source, expected_source_identity):
            return None

    content_identity = build_slice_observation_content_identity(
        observed_arr,
        sigma_arr,
        observation_canvas=None if stored_canvas is None else np.asarray(stored_canvas, dtype=float),
        sigma_canvas=None if stored_sigma_canvas is None else np.asarray(stored_sigma_canvas, dtype=float),
    )
    identity = {
        **expected_source_identity,
        **content_identity,
    }
    identity["slice_observation_identity_sha256"] = slice_observation_identity_sha256(identity)
    diagnostics = dict(stored_diag)
    diagnostics.update(identity)
    if stored_identity is not None and not _content_identity_matches_arrays(
        stored_identity,
        observed_arr,
        sigma_arr,
    ):
        diagnostics["observation_content_identity_repaired"] = True
    diagnostics.setdefault("observation_regridded_to_render_fov", True)
    diagnostics.setdefault(
        "observation_reference_preprocessed",
        True,
    )
    policy = str(shift_policy).strip().lower()
    pad_x = int(diagnostics.get("canvas_pad_x", 0) or 0)
    pad_y = int(diagnostics.get("canvas_pad_y", 0) or 0)
    return SliceObservationReference(
        observed=observed_arr,
        sigma=sigma_arr,
        source_header=stored_header.copy(),
        target_header=target_header.copy(),
        diagnostics=diagnostics,
        identity=identity,
        restored_from_artifact=True,
        shift_policy=policy,
        max_shift_arcsec=max_shift_arcsec,
        xy_shift_arcsec=(float(xy_shift_arcsec[0]), float(xy_shift_arcsec[1])),
        observation_canvas=canvas_arr,
        sigma_canvas=sigma_canvas_arr,
        canvas_header=canvas_header,
        canvas_pad_x=pad_x,
        canvas_pad_y=pad_y,
    )


def regrid_observation_to_target_fov(
    data: np.ndarray,
    source_header: fits.Header,
    target_header: fits.Header,
) -> np.ndarray:
    """Bilinearly resample *data* from *source_header* onto the *target_header* grid."""

    ny = int(target_header["NAXIS2"])
    nx = int(target_header["NAXIS1"])

    target_x = (
        (np.arange(nx, dtype=float) + 1.0 - float(target_header["CRPIX1"])) * float(target_header["CDELT1"])
        + float(target_header["CRVAL1"])
    )
    target_y = (
        (np.arange(ny, dtype=float) + 1.0 - float(target_header["CRPIX2"])) * float(target_header["CDELT2"])
        + float(target_header["CRVAL2"])
    )
    world_x, world_y = np.meshgrid(target_x, target_y)

    src_x = (
        (world_x - float(source_header["CRVAL1"])) / float(source_header["CDELT1"])
        + float(source_header["CRPIX1"])
        - 1.0
    )
    src_y = (
        (world_y - float(source_header["CRVAL2"])) / float(source_header["CDELT2"])
        + float(source_header["CRPIX2"])
        - 1.0
    )
    sampled = map_coordinates(
        np.asarray(data, dtype=float),
        [np.asarray(src_y, dtype=float), np.asarray(src_x, dtype=float)],
        order=1,
        mode="constant",
        cval=np.nan,
    )
    return np.asarray(sampled, dtype=float)


def _fill_observed_nans(observed: np.ndarray) -> np.ndarray:
    arr = np.asarray(observed, dtype=float)
    if not np.isnan(arr).any():
        return arr
    fill_value = float(np.nanmedian(arr))
    return np.asarray(np.nan_to_num(arr, nan=fill_value), dtype=float)


def _fill_sigma_nans(sigma: np.ndarray, *, fallback: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(sigma, dtype=float)
    if not np.isnan(arr).any():
        return arr
    fill_sigma = float(np.nanmedian(arr))
    if not np.isfinite(fill_sigma) or fill_sigma <= 0:
        if fallback is not None:
            fill_sigma = float(np.nanmedian(fallback))
        if not np.isfinite(fill_sigma) or fill_sigma <= 0:
            fill_sigma = 1.0
    return np.asarray(np.nan_to_num(arr, nan=fill_sigma), dtype=float)


def prepare_observation_for_metrics(
    observed: np.ndarray,
    header: fits.Header,
    target_header: fits.Header,
    *,
    model_time_text: str | None = None,
    observation_time_text: str | None = None,
    sigma: np.ndarray | None = None,
    alignment: ObsModelTimeAlignment | None = None,
    max_rotation_seconds: float = DEFAULT_MAX_ROTATION_SECONDS,
    fill_observed_nans: bool = True,
    fill_sigma_nans: bool = True,
    sigma_nan_fallback: np.ndarray | None = None,
    xy_shift_arcsec: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray | None, fits.Header, dict[str, Any]]:
    """Rotate (when applicable) then regrid observation data to the render FOV."""

    obs_text = str(
        observation_time_text
        or header.get("DATE-OBS", header.get("DATE_OBS", ""))
        or ""
    ).strip()
    model_text = str(model_time_text or "").strip() or None
    alignment = alignment or assess_obs_model_time_alignment(
        obs_text,
        model_text,
        max_rotation_seconds=max_rotation_seconds,
    )
    diagnostics: dict[str, Any] = {
        "observation_time_original": alignment.observation_time_text,
        "model_time_reference": alignment.model_time_text,
        "observation_time_delta_s": alignment.delta_seconds,
        "observation_time_alignment": alignment.compatibility,
        "observation_time_rotation_applied": False,
        "observation_time_alignment_message": alignment.message,
        "observation_time_warning_lines": alignment.warning_lines(),
        "observation_regridded_to_render_fov": True,
        "observation_reference_preprocessed": True,
        "observation_render_grid_ny": int(target_header["NAXIS2"]),
        "observation_render_grid_nx": int(target_header["NAXIS1"]),
    }

    aligned_data = np.asarray(observed, dtype=float)
    aligned_header = header.copy()
    if model_text and alignment.should_rotate():
        aligned_data, aligned_header, rotate_diag = align_observation_to_model_time(
            aligned_data,
            aligned_header,
            model_time_text=model_text,
            alignment=alignment,
        )
        diagnostics.update(rotate_diag)
    elif model_text and alignment.compatibility not in {"exact", "unknown"}:
        diagnostics["observation_time_alignment_message"] = alignment.message

    regrid_header = aligned_header.copy()
    if float(xy_shift_arcsec[0]) != 0.0 or float(xy_shift_arcsec[1]) != 0.0:
        regrid_header["CRVAL1"] = float(regrid_header["CRVAL1"]) - float(xy_shift_arcsec[0])
        regrid_header["CRVAL2"] = float(regrid_header["CRVAL2"]) - float(xy_shift_arcsec[1])
        diagnostics["observation_xy_shift_applied"] = True
        diagnostics["observation_xy_shift_x_arcsec"] = float(xy_shift_arcsec[0])
        diagnostics["observation_xy_shift_y_arcsec"] = float(xy_shift_arcsec[1])
    else:
        diagnostics["observation_xy_shift_applied"] = False

    observed_cropped = regrid_observation_to_target_fov(aligned_data, regrid_header, target_header)
    if fill_observed_nans:
        observed_cropped = _fill_observed_nans(observed_cropped)

    sigma_cropped = None
    if sigma is not None:
        sigma_cropped = regrid_observation_to_target_fov(
            np.asarray(sigma, dtype=float),
            regrid_header,
            target_header,
        )
        if fill_sigma_nans:
            sigma_cropped = _fill_sigma_nans(
                sigma_cropped,
                fallback=sigma_nan_fallback if sigma_nan_fallback is not None else np.asarray(sigma, dtype=float),
            )

    return observed_cropped, sigma_cropped, aligned_header, diagnostics


def resolve_slice_observation_reference(
    observed: np.ndarray,
    header: fits.Header,
    target_header: fits.Header,
    *,
    sigma: np.ndarray | None = None,
    observation_source_sha256: str | None,
    artifact_geometry_sha256: str,
    model_time_text: str | None = None,
    observation_time_text: str | None = None,
    stored_slice_payload: dict[str, Any] | None = None,
    force_recompute: bool = False,
    max_rotation_seconds: float = DEFAULT_MAX_ROTATION_SECONDS,
    shift_policy: str = "auto",
    max_shift_arcsec: float | None = None,
    xy_shift_arcsec: tuple[float, float] = (0.0, 0.0),
    slice_canvas_max_shift_arcsec: float | None = None,
) -> SliceObservationReference:
    """Prepare or restore the slice observation reference used for all metric work."""
    from .obs_alignment import (
        DEFAULT_MAX_SHIFT_ARSEC,
        build_padded_canvas_header,
        extract_observation_to_model_fov,
    )

    policy = str(shift_policy).strip().lower()
    if policy not in {"auto", "fixed"}:
        raise ValueError("shift_policy must be 'auto' or 'fixed'")
    resolved_search_max_shift = float(DEFAULT_MAX_SHIFT_ARSEC if max_shift_arcsec is None else max_shift_arcsec)
    resolved_canvas_max_shift = float(
        resolved_search_max_shift
        if slice_canvas_max_shift_arcsec is None
        else slice_canvas_max_shift_arcsec
    )

    expected_source_identity = build_slice_observation_source_identity(
        observation_source_sha256=observation_source_sha256,
        artifact_geometry_sha256=artifact_geometry_sha256,
        model_time_text=model_time_text,
        observation_time_text=observation_time_text,
        max_rotation_seconds=max_rotation_seconds,
        slice_canvas_max_shift_arcsec=resolved_canvas_max_shift,
    )

    if stored_slice_payload is not None and not force_recompute:
        restored = _try_restore_slice_observation_reference(
            target_header=target_header,
            expected_source_identity=expected_source_identity,
            stored_slice_payload=stored_slice_payload,
            shift_policy=policy,
            max_shift_arcsec=resolved_search_max_shift if policy == "auto" else None,
            xy_shift_arcsec=xy_shift_arcsec,
        )
        if restored is not None:
            return restored
        stored_has_reference = (
            stored_slice_payload.get("observed") is not None
            or stored_slice_payload.get("sigma_map") is not None
            or stored_slice_payload.get("observation_canvas") is not None
        )
        if stored_has_reference:
            raise SliceObservationReferenceError(
                "Existing slice observation reference is present but incompatible with the current "
                "observation source, geometry, epoch-alignment settings, or canvas shift envelope."
            )

    if sigma is None:
        raise ValueError("sigma map is required when preparing a new slice observation reference")

    canvas_header, pad_x, pad_y = build_padded_canvas_header(
        target_header,
        max_shift_arcsec=resolved_canvas_max_shift,
    )
    observation_canvas, sigma_canvas, aligned_header, preprocess_diag = prepare_observation_for_metrics(
        observed,
        header,
        canvas_header,
        model_time_text=model_time_text,
        observation_time_text=observation_time_text,
        sigma=sigma,
        max_rotation_seconds=max_rotation_seconds,
        sigma_nan_fallback=sigma,
    )
    if sigma_canvas is None:
        raise ValueError("sigma canvas is required when preparing a new slice observation reference")

    model_shift_x = float(xy_shift_arcsec[0]) if policy == "fixed" else 0.0
    model_shift_y = float(xy_shift_arcsec[1]) if policy == "fixed" else 0.0
    observed_cropped, sigma_cropped = extract_observation_to_model_fov(
        observation_canvas,
        canvas_header,
        target_header,
        shift_x_arcsec=model_shift_x,
        shift_y_arcsec=model_shift_y,
        canvas_sigma=sigma_canvas,
    )
    if sigma_cropped is None:
        raise ValueError("sigma canvas extraction failed")

    identity = build_slice_observation_identity(
        observation_source_sha256=observation_source_sha256,
        artifact_geometry_sha256=artifact_geometry_sha256,
        model_time_text=model_time_text,
        observation_time_text=observation_time_text,
        observed=observed_cropped,
        sigma=sigma_cropped,
        max_rotation_seconds=max_rotation_seconds,
        slice_canvas_max_shift_arcsec=resolved_canvas_max_shift,
        observation_canvas=observation_canvas,
        sigma_canvas=sigma_canvas,
    )
    diagnostics = dict(preprocess_diag)
    diagnostics.update(identity)
    diagnostics["shift_policy"] = policy
    diagnostics["slice_canvas_max_shift_arcsec"] = resolved_canvas_max_shift
    diagnostics["canvas_pad_x"] = int(pad_x)
    diagnostics["canvas_pad_y"] = int(pad_y)
    return SliceObservationReference(
        observed=np.asarray(observed_cropped, dtype=float),
        sigma=np.asarray(sigma_cropped, dtype=float),
        source_header=aligned_header.copy(),
        target_header=target_header.copy(),
        diagnostics=diagnostics,
        identity=identity,
        restored_from_artifact=False,
        shift_policy=policy,
        max_shift_arcsec=resolved_search_max_shift if policy == "auto" else None,
        xy_shift_arcsec=(float(xy_shift_arcsec[0]), float(xy_shift_arcsec[1])),
        observation_canvas=observation_canvas,
        sigma_canvas=sigma_canvas,
        canvas_header=canvas_header.copy(),
        canvas_pad_x=int(pad_x),
        canvas_pad_y=int(pad_y),
    )


def resolve_trial_shift_arcsec(
    *,
    diagnostics: dict[str, Any] | None,
    trial_index: int | None = None,
    fit_shift_x_trials: tuple[float, ...] | list[float] | np.ndarray | None = None,
    fit_shift_y_trials: tuple[float, ...] | list[float] | np.ndarray | None = None,
) -> tuple[float, float] | None:
    """Return the observation shift applied for a trial, or ``None`` when unset/zero."""
    diag = dict(diagnostics or {})
    policy = normalize_shift_policy(diag.get("shift_policy"))
    if policy == "auto":
        if trial_index is None or fit_shift_x_trials is None or fit_shift_y_trials is None:
            return None
        x_vals = np.asarray(fit_shift_x_trials, dtype=float)
        y_vals = np.asarray(fit_shift_y_trials, dtype=float)
        if x_vals.size == 0 or y_vals.size == 0:
            return None
        idx = int(np.clip(int(trial_index), 0, min(x_vals.size, y_vals.size) - 1))
        shift_x = float(x_vals[idx])
        shift_y = float(y_vals[idx])
        if not (np.isfinite(shift_x) and np.isfinite(shift_y)):
            return None
        return (shift_x, shift_y)

    xy = diag.get("xy_shift_arcsec")
    if isinstance(xy, (list, tuple)) and len(xy) >= 2:
        shift_x = float(xy[0])
        shift_y = float(xy[1])
    else:
        shift_x = float(diag.get("observation_xy_shift_x_arcsec", 0.0) or 0.0)
        shift_y = float(diag.get("observation_xy_shift_y_arcsec", 0.0) or 0.0)
    if shift_x == 0.0 and shift_y == 0.0:
        return None
    return (shift_x, shift_y)


def format_search_shift_policy_label(diagnostics: dict[str, Any] | None) -> str:
    """Summarize the search-level observation shift policy for status panels."""
    diag = dict(diagnostics or {})
    policy = normalize_shift_policy(diag.get("shift_policy"))
    if policy == "auto":
        max_shift = diag.get("max_shift_arcsec")
        try:
            max_value = float(max_shift)
        except Exception:
            max_value = float("nan")
        if np.isfinite(max_value):
            return f"Shift policy: auto (max {max_value:.1f} arcsec)"
        return "Shift policy: auto"
    shift = resolve_trial_shift_arcsec(diagnostics=diag)
    if shift is None:
        return "Shift policy: fixed (none)"
    shift_x, shift_y = shift
    return f"Shift policy: fixed ({shift_x:+.2f}, {shift_y:+.2f}) arcsec"


def format_observation_shift_label(
    *,
    diagnostics: dict[str, Any] | None,
    trial_index: int | None = None,
    fit_shift_x_trials: tuple[float, ...] | list[float] | np.ndarray | None = None,
    fit_shift_y_trials: tuple[float, ...] | list[float] | np.ndarray | None = None,
    fit_find_shift_valid_trials: tuple[bool, ...] | list[bool] | np.ndarray | None = None,
) -> str:
    """Format a compact shift label for observed-map panels."""
    diag = dict(diagnostics or {})
    policy = normalize_shift_policy(diag.get("shift_policy"))
    shift = resolve_trial_shift_arcsec(
        diagnostics=diag,
        trial_index=trial_index,
        fit_shift_x_trials=fit_shift_x_trials,
        fit_shift_y_trials=fit_shift_y_trials,
    )
    if policy == "auto":
        if shift is None:
            return "shift: auto (per trial)"
        shift_x, shift_y = shift
        suffix = ""
        if trial_index is not None and fit_find_shift_valid_trials is not None:
            valid_vals = np.asarray(fit_find_shift_valid_trials, dtype=bool)
            if valid_vals.size > 0:
                idx = int(np.clip(int(trial_index), 0, valid_vals.size - 1))
                if not bool(valid_vals[idx]):
                    suffix = " [FindShift invalid]"
        return f"shift: ({shift_x:+.2f}, {shift_y:+.2f}) arcsec{suffix}"
    if shift is None:
        return ""
    shift_x, shift_y = shift
    return f"shift: fixed ({shift_x:+.2f}, {shift_y:+.2f}) arcsec"


def resolve_trial_observation_for_display(
    *,
    observed: np.ndarray,
    sigma: np.ndarray | None,
    model_header: fits.Header | None,
    diagnostics: dict[str, Any] | None,
    observation_canvas: np.ndarray | None = None,
    sigma_canvas: np.ndarray | None = None,
    canvas_header: fits.Header | None = None,
    trial_index: int | None = None,
    fit_shift_x_trials: tuple[float, ...] | list[float] | np.ndarray | None = None,
    fit_shift_y_trials: tuple[float, ...] | list[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return trial-true observed/sigma maps for display (canvas + shift + crop)."""
    diag = dict(diagnostics or {})
    shift_policy = normalize_shift_policy(diag.get("shift_policy"))
    if shift_policy != "auto" or observation_canvas is None or canvas_header is None or model_header is None:
        sigma_arr = None if sigma is None else np.asarray(sigma, dtype=float)
        return np.asarray(observed, dtype=float), sigma_arr

    resolved_shift = resolve_trial_shift_arcsec(
        diagnostics=diag,
        trial_index=trial_index,
        fit_shift_x_trials=fit_shift_x_trials,
        fit_shift_y_trials=fit_shift_y_trials,
    )
    shift_x = 0.0 if resolved_shift is None else float(resolved_shift[0])
    shift_y = 0.0 if resolved_shift is None else float(resolved_shift[1])
    if not np.isfinite(shift_x):
        shift_x = 0.0
    if not np.isfinite(shift_y):
        shift_y = 0.0

    from .obs_alignment import extract_observation_to_model_fov

    obs, sig = extract_observation_to_model_fov(
        observation_canvas,
        canvas_header,
        model_header,
        shift_x_arcsec=shift_x,
        shift_y_arcsec=shift_y,
        canvas_sigma=sigma_canvas,
    )
    return np.asarray(obs, dtype=float), None if sig is None else np.asarray(sig, dtype=float)
