#!/usr/bin/env python
"""Fit Q0 to real observational maps (e.g., EOVSA).

This example demonstrates:
1. Loading observed solar maps from FITS files
2. Estimating noise using map_noise utilities
3. Fitting Q0 using the gxrender adapter
4. Saving artifacts and visualizations

Usage:
    python fit_q0_obs_map.py /path/to/eovsa_map.fits /path/to/model.h5 --ebtel-path /path/to/ebtel.sav
    python fit_q0_obs_map.py /path/to/eovsa_map.fits /path/to/model.h5 --ebtel-path /path/to/ebtel.sav --artifacts-dir /tmp/artifacts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import itertools
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Any

import h5py
import numpy as np
from astropy.io import fits
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve

from pychmp import (
    estimate_obs_map_noise,
    GXRenderEUVAdapter,
    GXRenderMWAdapter,
    build_tr_region_mask_from_blos,
    fit_q0_to_observation,
    load_obs_map,
    resolve_default_testdata_fixture_paths,
    validate_obs_map_identity,
)

try:
    from q0_artifact_plot import plot_q0_artifact_panel
except ModuleNotFoundError:
    from examples.q0_artifact_plot import plot_q0_artifact_panel

from pychmp.q0_artifact_panel import load_blos_reference_for_fov
from pychmp.ab_scan_artifacts import build_computed_point_payload, write_single_point_scan_file


DEFAULT_TBASE = 1.0e6
DEFAULT_NBASE = 1.0e8
DEFAULT_A = 0.3
DEFAULT_B = 2.7
@dataclass(frozen=True)
class _ObservationRequest:
    source_mode: str
    obs_path: Path | None
    obs_map_id: str | None
    model_h5: Path
    ebtel_path: Path | None


@dataclass(frozen=True)
class _RenderSelection:
    domain: str
    spectral_label: str
    active_frequency_ghz: float | None
    euv_channel: str | None
    euv_instrument: str | None
    euv_response_sav: Path | None

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
    model_h5 = _coerce_path(getattr(args, "model_h5_override", None) or args.model_h5)
    ebtel_path = _coerce_path(args.ebtel_path)
    testdata_repo = _coerce_path(getattr(args, "testdata_repo", None))
    obs_map_id = None if args.obs_map_id is None else str(args.obs_map_id).strip() or None
    explicit_source = None if args.obs_source is None else str(args.obs_source).strip().lower() or None

    if positional_fits is not None and explicit_obs_path is not None and positional_fits != explicit_obs_path:
        raise SystemExit(
            f"Conflicting observation path selectors: positional fits_file={positional_fits} "
            f"and --obs-path={explicit_obs_path}"
        )
    obs_path = explicit_obs_path or positional_fits

    if explicit_source is None:
        explicit_source = "model_refmap" if obs_map_id is not None else "external_fits"
    if explicit_source not in {"external_fits", "model_refmap"}:
        raise SystemExit(f"Unsupported --obs-source value: {explicit_source}")

    if explicit_source == "external_fits" and obs_map_id is not None:
        raise SystemExit("Conflicting observation selectors: --obs-map-id requires --obs-source=model_refmap")
    if explicit_source == "model_refmap" and obs_path is not None:
        raise SystemExit("Conflicting observation selectors: external FITS paths cannot be used with --obs-source=model_refmap")

    eovsa_root, model_root, ebtel_root = _default_testdata_roots(repo_root, testdata_repo=testdata_repo)
    default_eovsa_fits, default_model_h5, default_ebtel_path = resolve_default_testdata_fixture_paths(
        repo_root=repo_root,
        testdata_repo=testdata_repo,
    )

    if explicit_source == "external_fits" and obs_path is None:
        if default_eovsa_fits is None:
            raise SystemExit(
                f"Default EOVSA test-data FITS not found under {eovsa_root}; "
                "install the 2020-11-26 CHR/EOVSA fixture set or pass an explicit FITS path"
            )
        obs_path = default_eovsa_fits
    if explicit_source == "model_refmap" and obs_map_id is None:
        raise SystemExit("--obs-map-id is required when --obs-source=model_refmap")
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


def _default_observation_stem(obs_request: _ObservationRequest) -> str:
    if obs_request.obs_path is not None:
        return obs_request.obs_path.stem
    return str(obs_request.obs_map_id or "observation")


def _format_euv_channel(wavelength_angstrom: float) -> str:
    rounded = round(float(wavelength_angstrom))
    if np.isclose(float(wavelength_angstrom), float(rounded), rtol=0.0, atol=1e-9):
        return str(int(rounded))
    return f"{float(wavelength_angstrom):.6g}"


def _spectral_label_for_obs_map(obs_map: Any) -> str:
    label = str(getattr(obs_map, "spectral_label", "") or "").strip()
    if label:
        return label
    frequency_ghz = getattr(obs_map, "frequency_ghz", None)
    if frequency_ghz is not None:
        return f"{float(frequency_ghz):.3f} GHz"
    wavelength_angstrom = getattr(obs_map, "wavelength_angstrom", None)
    if wavelength_angstrom is not None:
        return f"{_format_euv_channel(float(wavelength_angstrom))} A"
    return "selected slice"


def _resolve_render_selection(args: argparse.Namespace, obs_map: Any) -> _RenderSelection:
    domain = str(obs_map.domain or "").strip().lower()
    spectral_label = _spectral_label_for_obs_map(obs_map)
    active_frequency_ghz = None if obs_map.frequency_ghz is None else float(obs_map.frequency_ghz)

    if domain == "mw":
        if args.euv_instrument is not None or args.euv_response_sav is not None:
            raise ValueError(
                "EUV-specific CLI options (--euv-instrument/--euv-response-sav) cannot be used with an MW observation"
            )
        return _RenderSelection(
            domain=domain,
            spectral_label=spectral_label,
            active_frequency_ghz=active_frequency_ghz,
            euv_channel=None,
            euv_instrument=None,
            euv_response_sav=None,
        )

    if domain not in {"euv", "uv"}:
        raise ValueError(
            f"unsupported observation domain={domain!r}; fit_q0_obs_map.py currently supports MW and one-point EUV/UV"
        )

    if obs_map.wavelength_angstrom is None:
        raise ValueError("selected EUV/UV observation is missing wavelength metadata")

    instrument = str(args.euv_instrument or obs_map.instrument or "AIA").strip()
    if not instrument:
        raise ValueError("could not resolve an EUV instrument name for the selected observation")
    if args.euv_instrument is not None and obs_map.instrument is not None:
        cli_instrument = str(args.euv_instrument).strip().lower()
        obs_instrument = str(obs_map.instrument).strip().lower()
        if cli_instrument and obs_instrument and cli_instrument != obs_instrument:
            raise ValueError(
                f"conflicting EUV instrument request: observation resolves to instrument={obs_map.instrument!r} "
                f"but --euv-instrument={args.euv_instrument!r} was supplied"
            )

    return _RenderSelection(
        domain=domain,
        spectral_label=spectral_label,
        active_frequency_ghz=active_frequency_ghz,
        euv_channel=_format_euv_channel(float(obs_map.wavelength_angstrom)),
        euv_instrument=instrument,
        euv_response_sav=None if args.euv_response_sav is None else Path(args.euv_response_sav).expanduser().resolve(),
    )


def _resolve_existing_file(path_text: str | None) -> Path | None:
    if path_text is None:
        return None
    candidate = Path(path_text).expanduser()
    try:
        resolved = candidate.resolve()
    except Exception:
        resolved = candidate
    return resolved if resolved.exists() else None


def _open_path_hint(path: Path) -> str:
    quoted = f'"{path}"'
    if sys.platform == "darwin":
        return f"open {quoted}"
    if os.name == "nt":
        return f'start "" {quoted}'
    return f"xdg-open {quoted}"


def _load_explicit_metric_mask(mask_path: str | Path, *, expected_shape: tuple[int, int]) -> np.ndarray:
    path = Path(mask_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"metrics mask FITS file not found: {path}")
    data = fits.getdata(path)
    arr = np.asarray(data)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"metrics mask FITS must contain a 2D image after squeeze(), got shape {arr.shape}")
    if tuple(arr.shape) != tuple(expected_shape):
        raise ValueError(
            f"metrics mask FITS shape mismatch: expected {expected_shape}, got {tuple(arr.shape)}"
        )
    return np.asarray(np.isfinite(arr) & (arr != 0), dtype=bool)


def _first_header_value(header: fits.Header, keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in header:
            return header[key]
    return None


def _extract_psf_from_header(header: fits.Header) -> tuple[dict[str, float] | None, str]:
    bmaj_raw = _first_header_value(header, ("BMAJ", "BMAJ_DEG", "BMAJDEG", "BEAM_MAJ", "PSF_BMAJ"))
    bmin_raw = _first_header_value(header, ("BMIN", "BMIN_DEG", "BMINDEG", "BEAM_MIN", "PSF_BMIN"))
    bpa_raw = _first_header_value(header, ("BPA", "BPA_DEG", "BEAM_PA", "PSF_BPA"))

    if bmaj_raw is None or bmin_raw is None:
        return None, "none"

    try:
        bmaj = float(bmaj_raw)
        bmin = float(bmin_raw)
        bpa = float(bpa_raw) if bpa_raw is not None else 0.0
    except Exception:
        return None, "none"

    # Standard FITS BMAJ/BMIN are in degrees; arcsec-style custom keys tend to
    # say so explicitly and are usually much larger than 1.
    if abs(bmaj) <= 1.0 and abs(bmin) <= 1.0:
        bmaj_arcsec = bmaj * 3600.0
        bmin_arcsec = bmin * 3600.0
    else:
        bmaj_arcsec = bmaj
        bmin_arcsec = bmin

    if not (np.isfinite(bmaj_arcsec) and np.isfinite(bmin_arcsec) and bmaj_arcsec > 0 and bmin_arcsec > 0):
        return None, "none"

    return {
        "psf_bmaj_arcsec": float(bmaj_arcsec),
        "psf_bmin_arcsec": float(bmin_arcsec),
        "psf_bpa_deg": float(bpa),
    }, "fits_header"


def _effective_psf_parameters(
    *,
    bmaj_arcsec: float | None,
    bmin_arcsec: float | None,
    bpa_deg: float | None,
    active_frequency_ghz: float,
    ref_frequency_ghz: float | None,
    scale_inverse_frequency: bool,
) -> dict[str, float | bool] | None:
    if bmaj_arcsec is None or bmin_arcsec is None or bpa_deg is None:
        return None
    psf_scale = (
        float(ref_frequency_ghz) / float(active_frequency_ghz)
        if scale_inverse_frequency and ref_frequency_ghz is not None
        else 1.0
    )
    return {
        "reference_bmaj_arcsec": float(bmaj_arcsec),
        "reference_bmin_arcsec": float(bmin_arcsec),
        "reference_bpa_deg": float(bpa_deg),
        "active_bmaj_arcsec": float(bmaj_arcsec) * float(psf_scale),
        "active_bmin_arcsec": float(bmin_arcsec) * float(psf_scale),
        "active_bpa_deg": float(bpa_deg),
        "reference_frequency_ghz": float(ref_frequency_ghz) if ref_frequency_ghz is not None else float(active_frequency_ghz),
        "active_frequency_ghz": float(active_frequency_ghz),
        "scaled": bool(scale_inverse_frequency and ref_frequency_ghz is not None and not np.isclose(psf_scale, 1.0)),
        "scale_factor": float(psf_scale),
    }


def _format_psf_report(
    *,
    source: str,
    bmaj_arcsec: float | None,
    bmin_arcsec: float | None,
    bpa_deg: float | None,
    active_frequency_ghz: float,
    ref_frequency_ghz: float | None,
    scale_inverse_frequency: bool,
) -> str:
    if source == "none" or bmaj_arcsec is None or bmin_arcsec is None or bpa_deg is None:
        return "PSF source: none"
    psf = _effective_psf_parameters(
        bmaj_arcsec=bmaj_arcsec,
        bmin_arcsec=bmin_arcsec,
        bpa_deg=bpa_deg,
        active_frequency_ghz=active_frequency_ghz,
        ref_frequency_ghz=ref_frequency_ghz,
        scale_inverse_frequency=scale_inverse_frequency,
    )
    assert psf is not None
    if bool(psf["scaled"]):
        return (
            f"PSF source: {source} "
            f"reference beam: bmaj={float(psf['reference_bmaj_arcsec']):.3f} "
            f"bmin={float(psf['reference_bmin_arcsec']):.3f} "
            f"bpa={float(psf['reference_bpa_deg']):.3f} @ {float(psf['reference_frequency_ghz']):.3f} GHz"
            f"\n    rescaled beam: bmaj={float(psf['active_bmaj_arcsec']):.3f} "
            f"bmin={float(psf['active_bmin_arcsec']):.3f} "
            f"bpa={float(psf['active_bpa_deg']):.3f} @ {float(psf['active_frequency_ghz']):.3f} GHz"
        )
    return (
        f"PSF source: {source} "
        f"beam: bmaj={float(psf['active_bmaj_arcsec']):.3f} "
        f"bmin={float(psf['active_bmin_arcsec']):.3f} "
        f"bpa={float(psf['active_bpa_deg']):.3f} @ {float(psf['active_frequency_ghz']):.3f} GHz"
    )


def _resolve_selected_psf(
    *,
    header_psf: dict[str, float] | None,
    header_psf_source: str,
    cli_psf_bmaj_arcsec: float | None,
    cli_psf_bmin_arcsec: float | None,
    cli_psf_bpa_deg: float | None,
    fallback_psf_bmaj_arcsec: float | None,
    fallback_psf_bmin_arcsec: float | None,
    fallback_psf_bpa_deg: float | None,
    override_header_psf: bool,
) -> tuple[float | None, float | None, float | None, str, bool]:
    has_cli_psf_override = any(
        value is not None for value in (cli_psf_bmaj_arcsec, cli_psf_bmin_arcsec, cli_psf_bpa_deg)
    )
    has_cli_psf_fallback = any(
        value is not None for value in (fallback_psf_bmaj_arcsec, fallback_psf_bmin_arcsec, fallback_psf_bpa_deg)
    )

    if header_psf is not None and not override_header_psf:
        return (
            float(header_psf["psf_bmaj_arcsec"]),
            float(header_psf["psf_bmin_arcsec"]),
            float(header_psf["psf_bpa_deg"]),
            header_psf_source,
            False,
        )
    if has_cli_psf_override:
        return (
            cli_psf_bmaj_arcsec,
            cli_psf_bmin_arcsec,
            cli_psf_bpa_deg,
            "cli_override",
            True,
        )
    if header_psf is not None:
        return (
            float(header_psf["psf_bmaj_arcsec"]),
            float(header_psf["psf_bmin_arcsec"]),
            float(header_psf["psf_bpa_deg"]),
            header_psf_source,
            False,
        )
    if has_cli_psf_fallback:
        return (
            fallback_psf_bmaj_arcsec,
            fallback_psf_bmin_arcsec,
            fallback_psf_bpa_deg,
            "cli_fallback",
            True,
        )
    return None, None, None, "none", False


def load_eovsa_map(fits_path: Path) -> tuple[np.ndarray, fits.Header, float]:
    """Compatibility wrapper around the package-owned observation loader."""

    obs_map = load_obs_map(obs_path=fits_path, domain="mw", instrument="EOVSA")
    if obs_map.frequency_ghz is None:
        raise ValueError(f"Could not extract MW observing frequency from {fits_path}")
    return np.asarray(obs_map.data, dtype=float), obs_map.header.copy(), float(obs_map.frequency_ghz)


def _elliptical_gaussian_kernel(
    bmaj_arcsec: float,
    bmin_arcsec: float,
    bpa_deg: float,
    dx_arcsec: float,
    dy_arcsec: float,
    size: int = 41,
) -> np.ndarray:
    fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_x = (bmaj_arcsec * fwhm_to_sigma) / dx_arcsec
    sigma_y = (bmin_arcsec * fwhm_to_sigma) / dy_arcsec

    half = size // 2
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1]

    theta = np.deg2rad(bpa_deg)
    ct = np.cos(theta)
    st = np.sin(theta)

    x_rot = ct * xx + st * yy
    y_rot = -st * xx + ct * yy
    kernel = np.exp(-0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2))
    kernel /= np.sum(kernel)
    return kernel


class PSFConvolvedRenderer:
    def __init__(self, base_renderer: Any, kernel: np.ndarray) -> None:
        self._base = base_renderer
        self._kernel = kernel

    def render_pair(self, q0: float) -> tuple[np.ndarray, np.ndarray]:
        raw = self._base.render(q0)
        convolved = fftconvolve(raw, self._kernel, mode="same")
        return raw, convolved

    def render(self, q0: float) -> np.ndarray:
        _raw, convolved = self.render_pair(q0)
        return convolved


def _lookup_cached_render_pair(
    render_cache: dict[float, Any],
    q0: float,
    *,
    atol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray] | None:
    q0_value = float(q0)
    cached = render_cache.get(q0_value)
    if cached is not None:
        if isinstance(cached, dict):
            raw = cached.get("raw")
            modeled = cached.get("modeled")
            if raw is not None and modeled is not None:
                return np.asarray(raw, dtype=float), np.asarray(modeled, dtype=float)
        else:
            return cached
    for cached_q0, cached_pair in render_cache.items():
        if np.isclose(cached_q0, q0_value, rtol=0.0, atol=atol):
            if isinstance(cached_pair, dict):
                raw = cached_pair.get("raw")
                modeled = cached_pair.get("modeled")
                if raw is not None and modeled is not None:
                    return np.asarray(raw, dtype=float), np.asarray(modeled, dtype=float)
            else:
                return cached_pair
    return None


def _lookup_cached_render_payload(
    render_cache: dict[float, Any],
    q0: float,
    *,
    atol: float = 1e-12,
) -> dict[str, Any] | None:
    q0_value = float(q0)
    cached = render_cache.get(q0_value)
    if isinstance(cached, dict):
        return cached
    for cached_q0, cached_payload in render_cache.items():
        if np.isclose(cached_q0, q0_value, rtol=0.0, atol=atol) and isinstance(cached_payload, dict):
            return cached_payload
    return None


def _decode_h5_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        return value
    return str(value)


def _artifact_frequency_ghz(
    *,
    frequency_ghz: float | None,
    diagnostics: dict[str, Any] | None = None,
) -> float | None:
    diag = diagnostics or {}
    domain = str(diag.get("spectral_domain", "")).strip().lower()
    if domain in {"euv", "uv"}:
        return None

    candidates = (
        frequency_ghz,
        diag.get("frequency_ghz"),
        diag.get("active_frequency_ghz"),
        diag.get("mw_frequency_ghz"),
    )
    for value in candidates:
        if value is None:
            continue
        try:
            numeric = float(value)
        except Exception:
            continue
        if np.isfinite(numeric):
            return numeric
    return None


def _compute_file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_model_obs_time_iso(model_path: Path) -> str | None:
    try:
        with h5py.File(model_path, "r") as f:
            for key in ("observer/pb0r/obs_date", "metadata/date_obs", "metadata/date-obs"):
                if key in f:
                    txt = _decode_h5_scalar(np.asarray(f[key]).reshape(-1)[0]).strip()
                    if txt:
                        return txt
    except Exception:
        return None
    return None


def _load_model_identity(model_path: Path) -> str:
    try:
        with h5py.File(model_path, "r") as f:
            for key in ("metadata/id", "meta/id", "model/id"):
                if key in f:
                    txt = _decode_h5_scalar(np.asarray(f[key]).reshape(-1)[0]).strip()
                    if txt:
                        return txt
    except Exception:
        pass
    return model_path.stem


def _load_saved_fov_from_model(model_path: Path) -> dict[str, float] | None:
    try:
        with h5py.File(model_path, "r") as f:
            fov = None
            for candidate in ("fov", "observer/fov", "metadata/fov", "meta/fov"):
                if candidate in f:
                    fov = f[candidate]
                    break
            if fov is None:
                return None

            out = {
                "xc_arcsec": float(np.asarray(fov["xc_arcsec"]).reshape(-1)[0]),
                "yc_arcsec": float(np.asarray(fov["yc_arcsec"]).reshape(-1)[0]),
                "xsize_arcsec": float(np.asarray(fov["xsize_arcsec"]).reshape(-1)[0]),
                "ysize_arcsec": float(np.asarray(fov["ysize_arcsec"]).reshape(-1)[0]),
            }
            if "grid" in f and "dx" in f["grid"]:
                dx = float(np.asarray(f["grid"]["dx"]).reshape(-1)[0])
                if np.isfinite(dx) and dx > 0:
                    out["dx_arcsec"] = dx
            if "grid" in f and "dy" in f["grid"]:
                dy = float(np.asarray(f["grid"]["dy"]).reshape(-1)[0])
                if np.isfinite(dy) and dy > 0:
                    out["dy_arcsec"] = dy
            return out
    except Exception:
        return None


def _resolve_observer_overrides(
    sdk: Any,
    *,
    model_path: Path,
    observer_name: str | None,
    dsun_cm: float | None,
    lonc_deg: float | None,
    b0sun_deg: float | None,
) -> tuple[Any | None, str]:
    lonc = lonc_deg
    b0 = b0sun_deg
    dsun = dsun_cm
    source = "saved_observer_metadata"

    if observer_name:
        try:
            from astropy.time import Time
            from gxrender.geometry import normalize_observer_name
            from gxrender.geometry.observer_geometry import _observer_from_sunpy

            model_time = Time(_load_model_obs_time_iso(model_path) or "2025-01-01T00:00:00")
            norm_name = normalize_observer_name(observer_name) or observer_name
            l0, b0_resolved, dsun_resolved = _observer_from_sunpy(str(norm_name), model_time)
            lonc = float(l0) if lonc is None else lonc
            b0 = float(b0_resolved) if b0 is None else b0
            dsun = float(dsun_resolved) if dsun is None else dsun
            source = f"observer_name:{norm_name}"
        except Exception as exc:
            raise ValueError(f"unable to resolve --observer '{observer_name}': {exc}") from exc

    if lonc is None and b0 is None and dsun is None:
        return None, source

    return sdk.ObserverOverrides(dsun_cm=dsun, lonc_deg=lonc, b0sun_deg=b0), source


def _load_model_observer_metadata(model_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        with h5py.File(model_path, "r") as f:
            scalar_map = {
                "observer_name": "observer/name",
                "observer_label": "observer/label",
                "observer_obs_time": "observer/pb0r/obs_date",
                "observer_lonc_deg": "observer/ephemeris/hgln_obs_deg",
                "observer_b0sun_deg": "observer/ephemeris/hglt_obs_deg",
                "observer_dsun_cm": "observer/ephemeris/dsun_cm",
            }
            for out_key, h5_key in scalar_map.items():
                if h5_key in f:
                    value = np.asarray(f[h5_key]).reshape(-1)[0]
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", errors="ignore")
                    out[out_key] = value
    except Exception:
        return out
    return out


def _with_observer_wcs_keywords(
    header: fits.Header,
    *,
    observer_name: str,
    hgln_obs_deg: float,
    hglt_obs_deg: float,
    dsun_obs_m: float,
) -> fits.Header:
    out = header.copy()
    out["OBSERVER"] = str(observer_name)
    out["DSUN_OBS"] = float(dsun_obs_m)
    out["HGLN_OBS"] = float(hgln_obs_deg)
    out["HGLT_OBS"] = float(hglt_obs_deg)
    out["CRLN_OBS"] = float(hgln_obs_deg)
    out["CRLT_OBS"] = float(hglt_obs_deg)
    out["HGLN-OBS"] = float(hgln_obs_deg)
    out["HGLT-OBS"] = float(hglt_obs_deg)
    out["CRLN-OBS"] = float(hgln_obs_deg)
    out["CRLT-OBS"] = float(hglt_obs_deg)
    return out


def _build_target_header(
    *,
    nx: int,
    ny: int,
    xc_arcsec: float,
    yc_arcsec: float,
    dx_arcsec: float,
    dy_arcsec: float,
    template_header: fits.Header,
) -> fits.Header:
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = int(nx)
    header["NAXIS2"] = int(ny)
    header["CTYPE1"] = template_header.get("CTYPE1", "HPLN-TAN")
    header["CTYPE2"] = template_header.get("CTYPE2", "HPLT-TAN")
    header["CUNIT1"] = template_header.get("CUNIT1", "arcsec")
    header["CUNIT2"] = template_header.get("CUNIT2", "arcsec")
    header["CDELT1"] = float(dx_arcsec)
    header["CDELT2"] = float(dy_arcsec)
    header["CRPIX1"] = (float(nx) + 1.0) / 2.0
    header["CRPIX2"] = (float(ny) + 1.0) / 2.0
    header["CRVAL1"] = float(xc_arcsec)
    header["CRVAL2"] = float(yc_arcsec)
    header["DATE-OBS"] = template_header.get("DATE-OBS", template_header.get("DATE_OBS", ""))
    header["BUNIT"] = template_header.get("BUNIT", "K")
    if "RSUN_REF" in template_header:
        header["RSUN_REF"] = template_header["RSUN_REF"]
    return header


def _regrid_full_disk_to_target(
    data: np.ndarray,
    source_header: fits.Header,
    target_header: fits.Header,
) -> np.ndarray:
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


def create_gxrender_adapter(model_path: Path, frequency_ghz: float) -> GXRenderMWAdapter:
    """Create GXRenderMWAdapter for the given explicit model path and frequency."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if model_path.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError(f"Model file must be H5/HDF5: {model_path}")

    return GXRenderMWAdapter(model_path=model_path, frequency_ghz=frequency_ghz)


def save_q0_artifact(
    h5_path: Path,
    *,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    modeled_best: np.ndarray,
    raw_modeled_best: np.ndarray,
    residual: np.ndarray,
    frequency_ghz: float | None,
    q0_fitted: float,
    metrics_dict: dict[str, float],
    diagnostics: dict[str, Any] | None = None,
    noise_diagnostics: dict[str, Any] | None = None,
    wcs_header: fits.Header | None = None,
    model_path: Path | None = None,
    blos_reference: tuple[np.ndarray, fits.Header] | None = None,
    trial_raw_modeled_maps: np.ndarray | None = None,
    trial_modeled_maps: np.ndarray | None = None,
    trial_residual_maps: np.ndarray | None = None,
    euv_coronal_best: np.ndarray | None = None,
    euv_tr_best: np.ndarray | None = None,
    euv_tr_mask: np.ndarray | None = None,
    trial_euv_coronal_maps: np.ndarray | None = None,
    trial_euv_tr_maps: np.ndarray | None = None,
) -> None:
    """Save Q0 fitting results as a canonical viewer-compatible sparse artifact."""
    diagnostics_payload = dict(diagnostics or {})
    artifact_frequency_ghz = _artifact_frequency_ghz(
        frequency_ghz=frequency_ghz,
        diagnostics=diagnostics_payload,
    )
    observed_arr = np.asarray(observed, dtype=np.float32)
    sigma_arr = np.asarray(sigma_map, dtype=np.float32)
    modeled_arr = np.asarray(modeled_best, dtype=np.float32)
    raw_modeled_arr = np.asarray(raw_modeled_best, dtype=np.float32)
    residual_arr = np.asarray(residual, dtype=np.float32)
    if model_path is not None and not str(diagnostics_payload.get("model_path", "")).strip():
        diagnostics_payload["model_path"] = str(Path(model_path))
    if artifact_frequency_ghz is not None:
        diagnostics_payload["frequency_ghz"] = float(artifact_frequency_ghz)
        diagnostics_payload.setdefault("active_frequency_ghz", float(artifact_frequency_ghz))
    diagnostics_payload["q0_recovered"] = float(q0_fitted)
    diagnostics_payload["point_status"] = str(
        diagnostics_payload.get("point_status", diagnostics_payload.get("status", "computed"))
    )
    diagnostics_payload["fit_success"] = bool(diagnostics_payload.get("fit_success", True))
    diagnostics_payload["chi2"] = float(metrics_dict["chi2"])
    diagnostics_payload["rho2"] = float(metrics_dict["rho2"])
    diagnostics_payload["eta2"] = float(metrics_dict["eta2"])
    target_metric = str(diagnostics_payload.get("target_metric", "chi2"))
    diagnostics_payload["target_metric"] = target_metric
    if "target_metric_value" not in diagnostics_payload and target_metric in metrics_dict:
        diagnostics_payload["target_metric_value"] = float(metrics_dict[target_metric])
    if noise_diagnostics is not None:
        diagnostics_payload["noise_diagnostics"] = dict(noise_diagnostics)
    diagnostics_payload.setdefault("store_raw_rendered_cubes", trial_raw_modeled_maps is not None)
    diagnostics_payload.setdefault("store_trial_convolved_cubes", trial_modeled_maps is not None)
    diagnostics_payload.setdefault("store_trial_residual_cubes", trial_residual_maps is not None)
    diagnostics_payload.setdefault("store_euv_component_cubes", trial_euv_coronal_maps is not None and trial_euv_tr_maps is not None)
    diagnostics_payload.setdefault("store_euv_tr_mask", euv_tr_mask is not None)

    def _trial_tuple(key: str) -> tuple[float, ...]:
        values = diagnostics_payload.get(key, ())
        return tuple(float(v) for v in values)

    bracket_values = diagnostics_payload.get("bracket")
    bracket: tuple[float, float, float] | None = None
    if bracket_values is not None:
        try:
            bracket_arr = np.asarray(bracket_values, dtype=float).reshape(-1)
            if bracket_arr.size == 3:
                bracket = tuple(float(v) for v in bracket_arr)
        except Exception:
            bracket = None

    point_payload = build_computed_point_payload(
        a_value=float(diagnostics_payload.get("a", 0.0)),
        b_value=float(diagnostics_payload.get("b", 0.0)),
        a_index=0,
        b_index=0,
        q0=float(q0_fitted),
        success=bool(diagnostics_payload.get("fit_success", True)),
        status=str(diagnostics_payload.get("point_status", "computed")),
        modeled_best=modeled_arr,
        raw_modeled_best=raw_modeled_arr,
        residual=residual_arr,
        fit_q0_trials=_trial_tuple("fit_q0_trials"),
        fit_metric_trials=_trial_tuple("fit_metric_trials"),
        fit_chi2_trials=_trial_tuple("fit_chi2_trials"),
        fit_rho2_trials=_trial_tuple("fit_rho2_trials"),
        fit_eta2_trials=_trial_tuple("fit_eta2_trials"),
        trial_raw_modeled_maps=trial_raw_modeled_maps,
        trial_modeled_maps=trial_modeled_maps,
        trial_residual_maps=trial_residual_maps,
        euv_coronal_best=euv_coronal_best,
        euv_tr_best=euv_tr_best,
        euv_tr_mask=euv_tr_mask,
        trial_euv_coronal_maps=trial_euv_coronal_maps,
        trial_euv_tr_maps=trial_euv_tr_maps,
        nfev=int(diagnostics_payload.get("nfev", -1)),
        nit=int(diagnostics_payload.get("nit", -1)),
        message=str(diagnostics_payload.get("optimizer_message", diagnostics_payload.get("message", ""))),
        used_adaptive_bracketing=bool(diagnostics_payload.get("used_adaptive_bracketing", False)),
        bracket_found=bool(diagnostics_payload.get("bracket_found", False)),
        bracket=bracket,
        target_metric=target_metric,
        diagnostics=diagnostics_payload,
    )

    write_single_point_scan_file(
        h5_path,
        observed=observed_arr,
        sigma_map=sigma_arr,
        wcs_header=wcs_header if wcs_header is not None else fits.Header(),
        diagnostics=diagnostics_payload,
        point_payload=point_payload,
        blos_reference=blos_reference,
        run_history=None,
    )


def save_prepared_observation_bundle(
    h5_path: Path,
    *,
    observed_cropped: np.ndarray,
    sigma_cropped: np.ndarray,
    target_header: fits.Header,
    frequency_ghz: float,
    geometry: Any,
    observer_source: str,
    observer_overrides: Any | None,
    model_observer_meta: dict[str, Any],
    header_psf: dict[str, float] | None,
    header_psf_source: str,
    noise_diagnostics: dict[str, Any] | None,
) -> None:
    geometry_dict = {
        "xc": float(geometry.xc),
        "yc": float(geometry.yc),
        "dx": float(geometry.dx),
        "dy": float(geometry.dy),
        "nx": int(geometry.nx),
        "ny": int(geometry.ny),
    }
    observer_override_dict = None
    if observer_overrides is not None:
        observer_override_dict = {
            "dsun_cm": None if getattr(observer_overrides, "dsun_cm", None) is None else float(observer_overrides.dsun_cm),
            "lonc_deg": None if getattr(observer_overrides, "lonc_deg", None) is None else float(observer_overrides.lonc_deg),
            "b0sun_deg": None if getattr(observer_overrides, "b0sun_deg", None) is None else float(observer_overrides.b0sun_deg),
        }
    metadata = {
        "frequency_ghz": float(frequency_ghz),
        "geometry": geometry_dict,
        "observer_source": str(observer_source),
        "observer_overrides": observer_override_dict,
        "model_observer_meta": dict(model_observer_meta or {}),
        "header_psf": dict(header_psf or {}) if header_psf is not None else None,
        "header_psf_source": str(header_psf_source),
        "noise_diagnostics": dict(noise_diagnostics or {}) if noise_diagnostics is not None else None,
    }
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("observed", data=np.asarray(observed_cropped, dtype=np.float32), compression="gzip")
        f.create_dataset("sigma_map", data=np.asarray(sigma_cropped, dtype=np.float32), compression="gzip")
        f.create_dataset("wcs_header", data=np.bytes_(target_header.tostring(sep="\n", endcard=True)))
        f.create_dataset("metadata_json", data=np.bytes_(json.dumps(metadata, sort_keys=True)))


def load_prepared_observation_bundle(h5_path: Path) -> dict[str, Any]:
    with h5py.File(h5_path, "r") as f:
        metadata = json.loads(_decode_h5_scalar(f["metadata_json"][()]))
        wcs_header = fits.Header.fromstring(_decode_h5_scalar(f["wcs_header"][()]), sep="\n")
        return {
            "observed": np.asarray(f["observed"], dtype=float),
            "sigma_map": np.asarray(f["sigma_map"], dtype=float),
            "wcs_header": wcs_header,
            "metadata": metadata,
        }


def _clear_terminal_status_line(width: int = 120) -> None:
    print("\r" + (" " * width) + "\r", end="", flush=True)


def _run_stage(
    name: str,
    fn,
    *,
    spinner: bool,
    stage_index: int | None = None,
    stage_total: int | None = None,
    running_label: str | None = None,
):
    if stage_index is not None and stage_total is not None and stage_total > 0:
        label = f"[stage {stage_index}/{stage_total}] {name}"
    else:
        label = f"[stage] {name}"
    running_text = running_label if running_label is not None else f"{label}: running"
    print(f"{label}: started")
    start = time.perf_counter()
    stop = threading.Event()
    thread: threading.Thread | None = None

    if spinner:
        def _spin() -> None:
            for ch in itertools.cycle("|/-\\"):
                if stop.wait(0.12):
                    break
                print(f"\r{running_text} {ch}", end="", flush=True)

        thread = threading.Thread(target=_spin, daemon=True)
        thread.start()

    try:
        result = fn()
    finally:
        if thread is not None:
            stop.set()
            thread.join(timeout=1.0)
            _clear_terminal_status_line()

    elapsed = time.perf_counter() - start
    print(f"{label}: done in {elapsed:.3f}s")
    return result, elapsed


def _make_trial_progress_reporter(*, target_metric: str):
    spinner_stop = threading.Event()
    spinner_thread: threading.Thread | None = None
    active_trial: int | None = None

    def _stop_spinner() -> None:
        nonlocal spinner_thread, active_trial
        if spinner_thread is not None:
            spinner_stop.set()
            spinner_thread.join(timeout=1.0)
            spinner_thread = None
            spinner_stop.clear()
            _clear_terminal_status_line()
        active_trial = None

    def _start(trial_index: int, q0: float) -> None:
        del q0
        nonlocal spinner_thread, active_trial
        _stop_spinner()
        active_trial = int(trial_index)

        def _spin() -> None:
            assert active_trial is not None
            for ch in itertools.cycle("|/-\\"):
                print(f"\r    trial {active_trial:02d}: rendering... {ch}", end="", flush=True)
                if spinner_stop.wait(0.12):
                    break

        spinner_thread = threading.Thread(target=_spin, daemon=True)
        spinner_thread.start()

    def _report(q0: float, objective_value: float, is_valid: bool, message: str, elapsed_s: float) -> None:
        nonlocal active_trial
        trial_index = 1 if active_trial is None else active_trial
        _stop_spinner()
        validity = "ok" if is_valid else "invalid"
        extra = f" msg={message}" if message else ""
        print(
            f"    trial {trial_index:02d}: q0={q0:.6f} {target_metric}={objective_value:.6e} "
            f"{validity} dt={elapsed_s:.3f}s{extra}",
            flush=True,
        )

    return _start, _report


def _format_q0_value(value: float) -> str:
    value = float(value)
    if value == 0.0:
        return "0.000000"
    abs_value = abs(value)
    if abs_value < 1.0e-3 or abs_value >= 1.0e3:
        return f"{value:.6e}"
    return f"{value:.6f}"


def _supports_color() -> bool:
    return os.getenv("NO_COLOR") is None


def _colorize(text: str, color: str) -> str:
    if not _supports_color():
        return text
    codes = {
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "reset": "\033[0m",
    }
    return f"{codes[color]}{text}{codes['reset']}"


def _resolve_preflight_q0(
    *,
    q0_min: float,
    q0_max: float,
    q0_start: float | None,
    preflight_q0: float | None,
) -> float:
    if preflight_q0 is not None:
        return float(preflight_q0)
    if q0_start is not None:
        return float(q0_start)
    return float(math.sqrt(float(q0_min) * float(q0_max)))



def main():
    import sys
    parser = argparse.ArgumentParser(
        description="Fit Q0 to real observational maps (EOVSA).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    %(prog)s /path/to/eovsa_map.fits /path/to/model.h5 --ebtel-path /path/to/ebtel.sav
    %(prog)s /path/to/eovsa_map.fits /path/to/model.h5 --ebtel-path /path/to/ebtel.sav --q0-min 0.5 --q0-max 2.5
    %(prog)s /path/to/eovsa_map.fits /path/to/model.h5 --ebtel-path /path/to/ebtel.sav --artifacts-dir /tmp/q0_artifacts
    %(prog)s /path/to/eovsa_map.fits /path/to/model.h5 --ebtel-path /path/to/ebtel.sav --defaults
""",
    )

    parser.add_argument("fits_file", type=Path, nargs="?", help="Path to the observational FITS map")
    parser.add_argument("model_h5", type=Path, nargs="?", help="Path to the model H5 file")
    parser.add_argument("--obs-source", choices=("external_fits", "model_refmap"), default=None, help="Select whether the observation comes from an external FITS product or an internal model refmap")
    parser.add_argument("--obs-path", type=Path, default=None, help="Explicit path to an external observational FITS map")
    parser.add_argument("--obs-map-id", default=None, help="Internal model refmap identifier, for example AIA_171")
    parser.add_argument("--obs-domain", choices=("mw", "euv", "uv", "generic"), default=None, help="Optional observation-domain hint used only for validation or to fill missing metadata")
    parser.add_argument("--obs-frequency-ghz", type=float, default=None, help="Optional MW frequency hint used only when the selected observation is missing frequency metadata")
    parser.add_argument("--obs-wavelength-angstrom", type=float, default=None, help="Optional EUV/UV wavelength hint used only when the selected observation is missing wavelength metadata")
    parser.add_argument("--model-h5", dest="model_h5_override", type=Path, default=None, help="Explicit model H5 path used when positional model_h5 is omitted")
    parser.add_argument("--ebtel-path", type=Path, default=None, help="Path to EBTEL .sav file (required)")
    parser.add_argument("--testdata-repo", type=Path, default=None, help="Optional sibling pyGXrender-test-data checkout used for default input resolution")
    parser.add_argument(
        "--prepared-observation-h5",
        type=Path,
        default=None,
        help="Optional prepared observation bundle H5; if provided, reuse precomputed cropped observation/sigma/WCS instead of reloading and regridding the FITS map.",
    )
    parser.add_argument("--euv-instrument", type=str, default=None, help="Optional EUV/UV instrument override. Must agree with the selected observation if that observation already declares an instrument.")
    parser.add_argument("--euv-response-sav", type=Path, default=None, help="Optional gxresponse SAV used for EUV/UV rendering. If omitted, the adapter will try the environment/test-data discovery path.")
    parser.add_argument("--tr-mask-bmin-gauss", type=float, default=1000.0, help="For EUV/UV, build the default TR-region mask from abs(B_los) >= Bmin [G]. Negative inputs are treated as abs(Bmin).")
    parser.add_argument("--tr-mask-threshold-gauss", dest="tr_mask_bmin_gauss", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--metrics-mask-threshold", type=float, default=0.1, help="Relative threshold used by the default union metrics mask.")
    parser.add_argument("--metrics-mask-fits", type=Path, default=None, help="Optional FITS bit mask used for metrics evaluation. Non-zero finite pixels are treated as in-mask and override --metrics-mask-threshold.")

    # Q0 fitting controls
    parser.add_argument("--q0-min", type=float, default=0.01, help="Lower edge of the initial Q0 search interval")
    parser.add_argument("--q0-max", type=float, default=2.5, help="Upper edge of the initial Q0 search interval")
    parser.add_argument("--hard-q0-min", type=float, default=None, help="Optional hard lower Q0 boundary")
    parser.add_argument("--hard-q0-max", type=float, default=None, help="Optional hard upper Q0 boundary")
    parser.add_argument("--adaptive-bracketing", action=argparse.BooleanOptionalAction, default=True, help="Enable adaptive Q0 bracketing")
    parser.add_argument("--q0-start", type=float, default=None, help="Explicit starting Q0 for adaptive bracketing")
    parser.add_argument("--q0-step", type=float, default=1.61803398875, help="Multiplicative Q0 step for adaptive bracketing")
    parser.add_argument("--max-bracket-steps", type=int, default=12, help="Maximum adaptive bracketing expansion steps")
    parser.add_argument("--target-metric", choices=["chi2", "rho2", "eta2"], default="chi2", help="Target metric for optimization (default: chi2)")

    # Plasma/geometry/observer overrides
    parser.add_argument("--tbase", type=float, default=None, help="Override base temperature (K)")
    parser.add_argument("--nbase", type=float, default=None, help="Override base density (cm^-3)")
    parser.add_argument("--a", type=float, default=None, help="Override model parameter a")
    parser.add_argument("--b", type=float, default=None, help="Override model parameter b")
    parser.add_argument("--observer", default=None, help="Observer name (earth, stereo-a, stereo-b)")
    parser.add_argument("--dsun-cm", type=float, default=None, help="Observer-Sun distance override in cm")
    parser.add_argument("--lonc-deg", type=float, default=None, help="Observer heliographic Carrington longitude override in deg")
    parser.add_argument("--b0sun-deg", type=float, default=None, help="Observer heliographic latitude override in deg")
    parser.add_argument("--xc", type=float, default=None, help="Map center X in arcsec (exact override)")
    parser.add_argument("--yc", type=float, default=None, help="Map center Y in arcsec (exact override)")
    parser.add_argument("--dx", type=float, default=None, help="Map pixel scale X in arcsec/pixel")
    parser.add_argument("--dy", type=float, default=None, help="Map pixel scale Y in arcsec/pixel")
    parser.add_argument("--nx", type=int, default=None, help="Map width in pixels")
    parser.add_argument("--ny", type=int, default=None, help="Map height in pixels")
    parser.add_argument("--pixel-scale-arcsec", type=float, default=2.0, help="Pixel scale used if no explicit geometry overrides")

    # PSF/beam options
    parser.add_argument("--psf-bmaj-arcsec", type=float, default=None, help="PSF major axis FWHM")
    parser.add_argument("--psf-bmin-arcsec", type=float, default=None, help="PSF minor axis FWHM")
    parser.add_argument("--psf-bpa-deg", type=float, default=None, help="PSF position angle in degrees")
    parser.add_argument(
        "--override-header-psf",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use user-supplied PSF or fallback reference-beam parameters even when the FITS header already contains a PSF beam",
    )
    parser.add_argument("--fallback-psf-bmaj-arcsec", type=float, default=None, help="Fallback PSF major axis FWHM used only when the FITS header has no beam and no explicit PSF override is supplied")
    parser.add_argument("--fallback-psf-bmin-arcsec", type=float, default=None, help="Fallback PSF minor axis FWHM used only when the FITS header has no beam and no explicit PSF override is supplied")
    parser.add_argument("--fallback-psf-bpa-deg", type=float, default=None, help="Fallback PSF position angle used only when the FITS header has no beam and no explicit PSF override is supplied")
    parser.add_argument("--psf-ref-frequency-ghz", type=float, default=None, help="Reference frequency for PSF axes values")
    parser.add_argument("--psf-scale-inverse-frequency", action="store_true", help="Scale PSF axes by (ref_freq / active_freq)")

    # Artifacts/outputs
    parser.add_argument("--artifacts-dir", type=Path, default=None, help="Directory to save H5/PNG artifacts")
    parser.add_argument("--artifacts-stem", default=None, help="Base filename stem for artifact files")
    parser.add_argument("--save-raw-h5", default=None, help="If set, write raw rendered map to this .h5 path")
    parser.add_argument("--no-artifacts-png", action="store_true", help="Skip PNG panel output even if --artifacts-dir is set")
    parser.add_argument("--show-plot", action="store_true", help="Display the PNG panel interactively after saving")
    parser.add_argument("--no-artifacts", action="store_true", help="Skip artifact saving")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True, help="Print stage timing/progress diagnostics.")
    parser.add_argument("--spinner", action=argparse.BooleanOptionalAction, default=True, help="Show a spinner during long-running stages.")
    parser.add_argument(
        "--preflight-render-only",
        action="store_true",
        help="Render exactly one modeled map at the initial optimizer q0 and exit without fitting.",
    )
    parser.add_argument(
        "--preflight-q0",
        type=float,
        default=None,
        help="Override the q0 used by --preflight-render-only. Defaults to --q0-start or sqrt(q0_min*q0_max).",
    )

    # Utility
    parser.add_argument("--defaults", action="store_true", help="Print assumed defaults and exit")

    args = parser.parse_args()

    if args.defaults:
        defaults = {
            "fits_file": "/path/to/eovsa_map.fits",
            "model_h5": "/path/to/model.h5",
            "obs_source": None,
            "obs_path": None,
            "obs_map_id": None,
            "obs_domain": None,
            "obs_frequency_ghz": None,
            "obs_wavelength_angstrom": None,
            "model_h5_override": None,
            "ebtel_path": "/path/to/ebtel.sav",
            "testdata_repo": None,
            "prepared_observation_h5": None,
            "euv_instrument": None,
            "euv_response_sav": None,
            "tr_mask_bmin_gauss": 1000.0,
            "metrics_mask_threshold": 0.1,
            "metrics_mask_fits": None,
            "q0_min": 0.01,
            "q0_max": 2.5,
            "hard_q0_min": None,
            "hard_q0_max": None,
            "adaptive_bracketing": True,
            "q0_start": None,
            "q0_step": 1.61803398875,
            "max_bracket_steps": 12,
            "target_metric": "chi2",
            "tbase": DEFAULT_TBASE,
            "nbase": DEFAULT_NBASE,
            "a": DEFAULT_A,
            "b": DEFAULT_B,
            "observer": None,
            "dsun_cm": None,
            "lonc_deg": None,
            "b0sun_deg": None,
            "xc": None,
            "yc": None,
            "dx": None,
            "dy": None,
            "nx": None,
            "ny": None,
            "pixel_scale_arcsec": 2.0,
            "psf_bmaj_arcsec": None,
            "psf_bmin_arcsec": None,
            "psf_bpa_deg": None,
            "fallback_psf_bmaj_arcsec": None,
            "fallback_psf_bmin_arcsec": None,
            "fallback_psf_bpa_deg": None,
            "psf_ref_frequency_ghz": None,
            "psf_scale_inverse_frequency": False,
            "artifacts_dir": None,
            "artifacts_stem": None,
            "save_raw_h5": None,
            "no_artifacts_png": False,
            "show_plot": False,
            "no_artifacts": False,
            "progress": True,
            "spinner": True,
            "preflight_render_only": False,
            "preflight_q0": None,
        }
        print("Assumed defaults for fit_q0_obs_map.py:")
        for k, v in defaults.items():
            print(f"  {k}: {v}")
        sys.exit(0)

    repo_root = Path(__file__).resolve().parents[1]
    obs_request = _resolve_observation_request(args, repo_root=repo_root)
    args.model_h5 = obs_request.model_h5
    args.ebtel_path = obs_request.ebtel_path

    if obs_request.obs_path is not None and not obs_request.obs_path.exists():
        print(f"ERROR: Observational FITS file not found: {obs_request.obs_path}")
        exit(1)
    if not args.model_h5.exists():
        print(f"ERROR: Model file not found: {args.model_h5}")
        exit(1)
    if args.ebtel_path is None or not args.ebtel_path.exists():
        print(f"ERROR: EBTEL .sav file not found: {args.ebtel_path}")
        exit(1)

    try:
        obs_map = load_obs_map(
            obs_path=obs_request.obs_path,
            model_h5=args.model_h5,
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
        print(f"ERROR: {exc}")
        exit(1)

    try:
        render_selection = _resolve_render_selection(args, obs_map)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        exit(1)
    if render_selection.domain == "mw" and render_selection.active_frequency_ghz is None:
        print("ERROR: Could not extract MW observing frequency from the selected observation")
        exit(1)
    if args.prepared_observation_h5 is not None and render_selection.domain != "mw":
        print(
            "ERROR: --prepared-observation-h5 currently reuses the MW prepared-bundle path only. "
            "Please run the one-point EUV/UV path directly from the selected observation for now."
        )
        exit(1)

    total_start = time.perf_counter()
    obs_source_detail = (
        str(obs_request.obs_path)
        if obs_request.obs_path is not None
        else f"{args.model_h5}::{obs_request.obs_map_id}"
    )
    observation_stem = _default_observation_stem(obs_request)

    print(f"\n{'=' * 70}")
    print("FITTING Q0 TO OBSERVATIONAL MAP")
    print(f"{'=' * 70}\n")
    print(f"Observation selection: {obs_source_detail}")
    print(
        "Resolved observation: "
        f"domain={obs_map.domain} "
        f"label={render_selection.spectral_label} "
        f"instrument={obs_map.instrument or 'n/a'}"
    )

    # Load map
    sdk = None
    try:
        from importlib import import_module

        sdk = import_module("gxrender.sdk")
    except Exception as exc:
        print(f"  ✗ Failed to import gxrender.sdk: {exc}")
        exit(1)

    tbase = float(args.tbase) if args.tbase is not None else DEFAULT_TBASE
    nbase = float(args.nbase) if args.nbase is not None else DEFAULT_NBASE
    a_param = float(args.a) if args.a is not None else DEFAULT_A
    b_param = float(args.b) if args.b is not None else DEFAULT_B
    if args.prepared_observation_h5 is not None:
        if not args.prepared_observation_h5.exists():
            print(f"ERROR: Prepared observation bundle not found: {args.prepared_observation_h5}")
            exit(1)
        print(f"Loading prepared observation bundle: {args.prepared_observation_h5.name}")
        prepared = load_prepared_observation_bundle(args.prepared_observation_h5)
        prepared_meta = dict(prepared.get("metadata") or {})
        observed_cropped = np.asarray(prepared["observed"], dtype=float)
        sigma_cropped = np.asarray(prepared["sigma_map"], dtype=float)
        target_header = prepared["wcs_header"].copy()
        freq_ghz = (
            None
            if prepared_meta.get("frequency_ghz") is None
            else float(prepared_meta.get("frequency_ghz"))
        )
        print(f"  Prepared submap grid: Ny={observed_cropped.shape[0]} Nx={observed_cropped.shape[1]}")
        print(f"  Slice: {render_selection.spectral_label}")
        geometry_data = dict(prepared_meta.get("geometry") or {})
        geometry = sdk.MapGeometry(
            xc=float(geometry_data["xc"]),
            yc=float(geometry_data["yc"]),
            dx=float(geometry_data["dx"]),
            dy=float(geometry_data["dy"]),
            nx=int(geometry_data["nx"]),
            ny=int(geometry_data["ny"]),
        )
        observer_source = str(prepared_meta.get("observer_source", "prepared_bundle"))
        override_data = prepared_meta.get("observer_overrides")
        observer_overrides = None
        if override_data:
            observer_overrides = sdk.ObserverOverrides(
                dsun_cm=override_data.get("dsun_cm"),
                lonc_deg=override_data.get("lonc_deg"),
                b0sun_deg=override_data.get("b0sun_deg"),
            )
        model_observer_meta = dict(prepared_meta.get("model_observer_meta") or {})
        noise_diagnostics = prepared_meta.get("noise_diagnostics")
        header_psf = prepared_meta.get("header_psf")
        header_psf_source = str(prepared_meta.get("header_psf_source", "prepared_bundle"))
        geometry_mode = "prepared_bundle"
    else:
        print("Loading selected observational map payload")
        observed = np.asarray(obs_map.data, dtype=float)
        header = obs_map.header.copy()
        freq_ghz = render_selection.active_frequency_ghz
        print(f"  Shape: {observed.shape}")
        print(f"  Slice: {render_selection.spectral_label}")
        print(f"  Data range: [{observed.min():.2f}, {observed.max():.2f}]")

        print(f"\nEstimating noise from map...")
        noise_result = estimate_obs_map_noise(obs_map, method="histogram_clip")
        if str(noise_result.method_used) == "fallback_std":
            print("  ⚠️  Noise estimation failed (map quality issues)")
            print(f"  Falling back to fixed sigma = {int(noise_result.sigma)}K")
        else:
            print(f"  Estimated sigma: {noise_result.sigma:.2f} K")
            print(f"  Background fraction: {noise_result.mask_fraction:.1%}")
        sigma_map = np.asarray(noise_result.sigma_map, dtype=float)
        noise_diagnostics = noise_result.diagnostics

        header_psf, header_psf_source = _extract_psf_from_header(header)
        try:
            observer_overrides, observer_source = _resolve_observer_overrides(
                sdk,
                model_path=args.model_h5,
                observer_name=args.observer,
                dsun_cm=args.dsun_cm,
                lonc_deg=args.lonc_deg,
                b0sun_deg=args.b0sun_deg,
            )
        except ValueError as exc:
            print(f"  ✗ Failed to resolve observer settings: {exc}")
            exit(1)
        model_observer_meta = _load_model_observer_metadata(args.model_h5)

        geometry_overrides_requested = any(v is not None for v in (args.xc, args.yc, args.dx, args.dy, args.nx, args.ny))
        saved_fov = None
        if geometry_overrides_requested:
            geometry = sdk.MapGeometry(
                xc=args.xc,
                yc=args.yc,
                dx=args.dx,
                dy=args.dy,
                nx=args.nx,
                ny=args.ny,
            )
            geometry_mode = "explicit"
        else:
            saved_fov = _load_saved_fov_from_model(args.model_h5)
            if saved_fov is None:
                print("  ✗ Model does not expose a saved FOV. Provide --xc/--yc/--dx/--dy/--nx/--ny explicitly.")
                exit(1)
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
            nan_fraction = float(np.isnan(observed_cropped).sum()) / float(observed_cropped.size)
            if nan_fraction > 0.05:
                print(f"  ✗ Regridded observed submap contains too many NaNs ({nan_fraction:.1%}). Check FITS WCS or requested geometry.")
                exit(1)
            fill_value = float(np.nanmedian(observed_cropped))
            observed_cropped = np.nan_to_num(observed_cropped, nan=fill_value)
        if np.isnan(sigma_cropped).any():
            fill_sigma = float(np.nanmedian(sigma_cropped))
            if not np.isfinite(fill_sigma) or fill_sigma <= 0:
                fill_sigma = float(np.nanmedian(sigma_map))
            sigma_cropped = np.nan_to_num(sigma_cropped, nan=fill_sigma)

        print(f"\nPreparing model-aligned observational submap...")
        print(f"  Observer mode: {'saved metadata' if observer_overrides is None else 'overrides'} ({observer_source})")
        if geometry_mode == "explicit":
            print(
                f"  Geometry mode: explicit xc={float(geometry.xc):.3f} yc={float(geometry.yc):.3f} "
                f"dx={float(geometry.dx):.3f} dy={float(geometry.dy):.3f} nx={int(geometry.nx)} ny={int(geometry.ny)}"
            )
        else:
            raw_model_dx = saved_fov.get("dx_arcsec") if saved_fov is not None else None
            raw_model_dy = saved_fov.get("dy_arcsec") if saved_fov is not None else None
            raw_model_dx_str = f"{float(raw_model_dx):.6f}" if raw_model_dx is not None else "<not present>"
            raw_model_dy_str = f"{float(raw_model_dy):.6f}" if raw_model_dy is not None else "<not present>"
            print(
                f"  Saved model FOV: xc={float(saved_fov['xc_arcsec']):.3f} yc={float(saved_fov['yc_arcsec']):.3f} "
                f"xsize={float(saved_fov['xsize_arcsec']):.3f} ysize={float(saved_fov['ysize_arcsec']):.3f} "
                f"model_dx={raw_model_dx_str} model_dy={raw_model_dy_str}"
            )
            print(
                f"  Geometry mode: saved_fov xc={float(geometry.xc):.3f} yc={float(geometry.yc):.3f} "
                f"dx={float(geometry.dx):.3f} dy={float(geometry.dy):.3f} nx={int(geometry.nx)} ny={int(geometry.ny)}"
            )
        print(f"  Observed submap grid: Ny={observed_cropped.shape[0]} Nx={observed_cropped.shape[1]}")
        print(f"  Model render grid: Ny={int(geometry.ny)} Nx={int(geometry.nx)}")
        print(f"  Pixel scale: dx={float(geometry.dx):.1f} dy={float(geometry.dy):.1f} arcsec/pixel")
        print(
            f"  FOV: {abs(float(geometry.dx)) * int(geometry.nx):.0f} x "
            f"{abs(float(geometry.dy)) * int(geometry.ny):.0f} arcsec"
        )

    blos_reference_for_fov = None
    euv_tr_mask = None
    if render_selection.domain != "mw":
        blos_reference_for_fov = load_blos_reference_for_fov(
            args.model_h5,
            header=target_header,
            shape=np.asarray(observed_cropped, dtype=float).shape,
            wcs_header_transform=None,
        )
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

    if render_selection.domain != "mw":
        if any(
            value is not None
            for value in (
                args.psf_bmaj_arcsec,
                args.psf_bmin_arcsec,
                args.psf_bpa_deg,
                args.fallback_psf_bmaj_arcsec,
                args.fallback_psf_bmin_arcsec,
                args.fallback_psf_bpa_deg,
                args.psf_ref_frequency_ghz,
            )
        ) or bool(args.psf_scale_inverse_frequency) or bool(args.override_header_psf):
            print(
                "ERROR: MW beam/PSF CLI options are not supported on the one-point EUV/UV path yet."
            )
            exit(1)
        psf_bmaj_arcsec = None
        psf_bmin_arcsec = None
        psf_bpa_deg = None
        psf_source = "none"
        psf_allows_frequency_scaling = False

    if render_selection.domain == "mw":
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
    else:
        print("  PSF source: none (one-point EUV/UV path currently compares the direct rendered map)")
    print(f"  Plasma/heating: a={a_param:.3f} b={b_param:.3f} tbase={tbase:.3e} nbase={nbase:.3e}")


    # Create gxrender adapter from explicit user-provided model path, passing all relevant overrides
    print(f"\nInitializing gxrender adapter for {render_selection.spectral_label}...")
    print(f"  Model file: {args.model_h5}")
    try:
        if render_selection.domain == "mw":
            adapter_kwargs = dict(
                model_path=args.model_h5,
                frequency_ghz=float(freq_ghz),
                ebtel_path=str(args.ebtel_path),
                tbase=tbase,
                nbase=nbase,
                a=a_param,
                b=b_param,
                geometry=geometry,
                observer=observer_overrides,
                pixel_scale_arcsec=float(args.pixel_scale_arcsec),
            )
            base_adapter = GXRenderMWAdapter(**adapter_kwargs)
        else:
            adapter_kwargs = dict(
                model_path=args.model_h5,
                channel=str(render_selection.euv_channel),
                instrument=str(render_selection.euv_instrument),
                response_sav=render_selection.euv_response_sav,
                ebtel_path=str(args.ebtel_path),
                tbase=tbase,
                nbase=nbase,
                a=a_param,
                b=b_param,
                geometry=geometry,
                observer=observer_overrides,
                tr_region_mask=euv_tr_mask,
                pixel_scale_arcsec=float(args.pixel_scale_arcsec),
            )
            base_adapter = GXRenderEUVAdapter(**adapter_kwargs)
        renderer = base_adapter
        if render_selection.domain == "mw" and psf_bmaj_arcsec is not None and psf_bmin_arcsec is not None and psf_bpa_deg is not None:
            psf_meta = _effective_psf_parameters(
                bmaj_arcsec=psf_bmaj_arcsec,
                bmin_arcsec=psf_bmin_arcsec,
                bpa_deg=psf_bpa_deg,
                active_frequency_ghz=float(freq_ghz),
                ref_frequency_ghz=float(args.psf_ref_frequency_ghz) if args.psf_ref_frequency_ghz is not None else None,
                scale_inverse_frequency=bool(args.psf_scale_inverse_frequency and psf_allows_frequency_scaling),
            )
            assert psf_meta is not None
            kernel = _elliptical_gaussian_kernel(
                bmaj_arcsec=float(psf_meta["active_bmaj_arcsec"]),
                bmin_arcsec=float(psf_meta["active_bmin_arcsec"]),
                bpa_deg=float(psf_bpa_deg),
                dx_arcsec=float(geometry.dx),
                dy_arcsec=float(geometry.dy),
            )
            renderer = PSFConvolvedRenderer(base_adapter, kernel)
        print("  ✓ Model loaded successfully")
    except Exception as e:
        print(f"  ✗ Failed to initialize gxrender adapter: {e}")
        exit(1)

    if bool(args.preflight_render_only):
        preflight_q0 = _resolve_preflight_q0(
            q0_min=float(args.q0_min),
            q0_max=float(args.q0_max),
            q0_start=None if args.q0_start is None else float(args.q0_start),
            preflight_q0=None if args.preflight_q0 is None else float(args.preflight_q0),
        )
        print(f"\nRunning single-render preflight at q0={_format_q0_value(preflight_q0)}...")
        try:
            if isinstance(renderer, PSFConvolvedRenderer):
                (raw_modeled_best, modeled_best), render_elapsed = _run_stage(
                    "Preflight render",
                    lambda: renderer.render_pair(preflight_q0),
                    spinner=bool(args.spinner),
                    stage_index=1,
                    stage_total=1,
                )
            else:
                modeled_best, render_elapsed = _run_stage(
                    "Preflight render",
                    lambda: base_adapter.render(preflight_q0),
                    spinner=bool(args.spinner),
                    stage_index=1,
                    stage_total=1,
                )
                raw_modeled_best = modeled_best
            print("  ✓ Preflight render completed")
            print(f"  q0={_format_q0_value(preflight_q0)} dt={render_elapsed:.3f}s")
            print(
                f"  modeled map range: "
                f"[{float(np.nanmin(modeled_best)):.6e}, {float(np.nanmax(modeled_best)):.6e}]"
            )
            print(
                f"  raw modeled map range: "
                f"[{float(np.nanmin(raw_modeled_best)):.6e}, {float(np.nanmax(raw_modeled_best)):.6e}]"
            )
            total_elapsed = time.perf_counter() - total_start
            print(f"\n{'=' * 70}")
            print(f"PREFLIGHT COMPLETE: success=True total={total_elapsed:.3f}s")
            print(f"{'=' * 70}\n")
            return
        except Exception as e:
            print(f"  ✗ Preflight render failed: {e}")
            exit(1)

    # Fit Q0
    result = None
    render_cache: dict[float, dict[str, Any]] = {}
    print(f"\nFitting Q0 using {args.target_metric} metric...")
    print(f"  Q0 initial interval: [{_format_q0_value(args.q0_min)}, {_format_q0_value(args.q0_max)}]")
    if args.hard_q0_min is not None or args.hard_q0_max is not None:
        hard_left = _format_q0_value(args.hard_q0_min) if args.hard_q0_min is not None else "<unbounded>"
        hard_right = _format_q0_value(args.hard_q0_max) if args.hard_q0_max is not None else "<unbounded>"
        print(f"  Q0 hard bounds: [{hard_left}, {hard_right}]")
    stage_spinner = bool(args.spinner)
    stage_total = 2 if args.no_artifacts else 3

    try:
        progress_start_callback = None
        progress_callback = None
        if args.progress:
            progress_start_callback, progress_callback = _make_trial_progress_reporter(target_metric=args.target_metric)
        if isinstance(renderer, PSFConvolvedRenderer):
            class _CachedObservedRenderer:
                def render(self_inner, q0: float) -> np.ndarray:
                    raw_arr, modeled_arr = renderer.render_pair(q0)
                    render_cache[float(q0)] = {"raw": raw_arr, "modeled": modeled_arr}
                    return modeled_arr

            optimization_renderer = _CachedObservedRenderer()
        else:
            class _CachedObservedRenderer:
                def render(self_inner, q0: float) -> np.ndarray:
                    if hasattr(base_adapter, "render_components"):
                        components = base_adapter.render_components(q0)
                        modeled_arr = np.asarray(components["rendered"], dtype=float)
                        render_cache[float(q0)] = {
                            "raw": modeled_arr,
                            "modeled": modeled_arr,
                            "euv_coronal": np.asarray(components.get("flux_corona"), dtype=float),
                            "euv_tr": np.asarray(components.get("flux_tr"), dtype=float),
                            "euv_tr_mask": (
                                None
                                if components.get("tr_region_mask") is None
                                else np.asarray(components.get("tr_region_mask"), dtype=bool)
                            ),
                        }
                    else:
                        modeled_arr = base_adapter.render(q0)
                        render_cache[float(q0)] = {"raw": modeled_arr, "modeled": modeled_arr}
                    return modeled_arr

            optimization_renderer = _CachedObservedRenderer()
        result, _fit_elapsed = _run_stage(
            "Optimize q0",
            lambda: fit_q0_to_observation(
                renderer=optimization_renderer,
                observed=observed_cropped,
                sigma=sigma_cropped,
                q0_min=args.q0_min,
                q0_max=args.q0_max,
                hard_q0_min=args.hard_q0_min,
                hard_q0_max=args.hard_q0_max,
                threshold=float(args.metrics_mask_threshold),
                explicit_mask=explicit_metric_mask,
                target_metric=args.target_metric,
                adaptive_bracketing=bool(args.adaptive_bracketing),
                q0_start=args.q0_start,
                q0_step=float(args.q0_step),
                max_bracket_steps=int(args.max_bracket_steps),
                progress_start_callback=progress_start_callback,
                progress_callback=progress_callback,
            ),
            spinner=stage_spinner and not bool(args.progress),
            stage_index=1,
            stage_total=stage_total,
        )

        if result.success:
            print(_colorize("  ✓ Fitting converged", "green"))
        else:
            print(_colorize("  ⚠ Fitting stopped without an interior bracket", "yellow"))
        print(f"  Fitted Q0: {result.q0:.6f}")
        print(f"  Objective value ({result.target_metric}): {result.objective_value:.6e}")
        print(
            f"  Metrics: chi2={result.metrics.chi2:.6e}, "
            f"rho2={result.metrics.rho2:.6e}, eta2={result.metrics.eta2:.6e}"
        )
        print(f"  Trials: nfev={result.nfev} nit={result.nit} saved_trials={len(result.trial_q0)}")
        print(f"  Optimizer message: {result.message}")
        if result.used_adaptive_bracketing:
            bracket_text = (
                f"({_format_q0_value(result.bracket[0])}, {_format_q0_value(result.bracket[1])}, {_format_q0_value(result.bracket[2])})"
                if result.bracket is not None
                else "<none>"
            )
            print(f"  Adaptive bracketing: found={result.bracket_found} bracket={bracket_text}")
        hard_left = args.hard_q0_min
        hard_right = args.hard_q0_max
        near_left = False
        near_right = False
        if hard_left is not None and hard_right is not None:
            span = max(abs(hard_right - hard_left), 1.0)
            near_left = abs(result.q0 - hard_left) <= 0.01 * span
            near_right = abs(result.q0 - hard_right) <= 0.01 * span
        elif hard_left is not None:
            near_left = abs(result.q0 - hard_left) <= 0.01 * max(abs(hard_left), 1.0)
        elif hard_right is not None:
            near_right = abs(result.q0 - hard_right) <= 0.01 * max(abs(hard_right), 1.0)
        if (near_left or near_right) and result.bracket is not None:
            near_left = near_left and result.q0 <= result.bracket[1]
            near_right = near_right and result.q0 >= result.bracket[1]
        if near_left or near_right:
            side = "lower" if near_left else "upper"
            print(
                _colorize(
                    f"  WARNING: best q0 lies near the {side} hard search boundary; "
                    "this run did not demonstrate a well-bracketed interior minimum.",
                    "red",
                )
            )

    except Exception as e:
        print(f"  ✗ Fitting failed: {e}")
        exit(1)

    cached_best_pair = _lookup_cached_render_pair(render_cache, result.q0)
    cached_best_payload = _lookup_cached_render_payload(render_cache, result.q0)
    if cached_best_pair is not None:
        stage_label = f"[stage 2/{stage_total}] Render best-fit map"
        print(f"{stage_label}: started")
        raw_modeled_best, modeled_best = cached_best_pair
        _render_elapsed = 0.0
        print(f"{stage_label}: done in 0.000s")
    elif isinstance(renderer, PSFConvolvedRenderer):
        (raw_modeled_best, modeled_best), _render_elapsed = _run_stage(
            "Render best-fit map",
            lambda: renderer.render_pair(result.q0),
            spinner=stage_spinner,
            stage_index=2,
            stage_total=stage_total,
        )
    else:
        if hasattr(base_adapter, "render_components"):
            component_payload, _render_elapsed = _run_stage(
                "Render best-fit map",
                lambda: base_adapter.render_components(result.q0),
                spinner=stage_spinner,
                stage_index=2,
                stage_total=stage_total,
            )
            modeled_best = np.asarray(component_payload["rendered"], dtype=float)
            raw_modeled_best = modeled_best
            cached_best_payload = {
                "raw": modeled_best,
                "modeled": modeled_best,
                "euv_coronal": np.asarray(component_payload.get("flux_corona"), dtype=float),
                "euv_tr": np.asarray(component_payload.get("flux_tr"), dtype=float),
                "euv_tr_mask": (
                    None
                    if component_payload.get("tr_region_mask") is None
                    else np.asarray(component_payload.get("tr_region_mask"), dtype=bool)
                ),
            }
            render_cache[float(result.q0)] = cached_best_payload
        else:
            modeled_best, _render_elapsed = _run_stage(
                "Render best-fit map",
                lambda: base_adapter.render(result.q0),
                spinner=stage_spinner,
                stage_index=2,
                stage_total=stage_total,
            )
            raw_modeled_best = modeled_best
    residual = modeled_best - observed_cropped
    trial_render_count = len(result.trial_q0)
    total_render_calls = int(base_adapter.render_call_count)
    final_render_calls = max(0, total_render_calls - trial_render_count)
    print(
        f"  Render diagnostics: trial_renders={trial_render_count} "
        f"final_renders={final_render_calls} total_gxrender_calls={total_render_calls}"
    )

    # Save artifacts
    if not args.no_artifacts and result is not None:
        if args.artifacts_dir is None:
            args.artifacts_dir = Path(".").resolve() / "q0_observation_artifacts"

        args.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Create filename from FITS file and frequency
        stem = args.artifacts_stem or f"{observation_stem}_q0_fitted_{result.q0:.6f}"
        h5_name = f"{stem}.h5"
        h5_path = args.artifacts_dir / h5_name
        png_path = args.artifacts_dir / f"{stem}.png"

        print(f"\nSaving artifacts...")
        try:
            observation_source_path = obs_map.source_path
            observation_source_file = _resolve_existing_file(observation_source_path)
            observation_source_sha256 = (
                _compute_file_sha256(observation_source_file)
                if observation_source_file is not None and observation_source_file.is_file()
                else None
            )
            diagnostics = {
                "model_path": str(args.model_h5),
                "model_id": str(_load_model_identity(args.model_h5)),
                "model_sha256": str(_compute_file_sha256(args.model_h5)),
                "fits_file": str(observation_source_path or ""),
                "fits_sha256": str(observation_source_sha256 or ""),
                "observation_source_mode": str(obs_map.source_mode),
                "observation_source_path": (None if observation_source_path is None else str(observation_source_path)),
                "observation_source_map_id": obs_map.source_map_id,
                "observation_source_sha256": observation_source_sha256,
                "observation_instrument": obs_map.instrument,
                "observation_observer": obs_map.observer,
                "ebtel_path": str(args.ebtel_path),
                "ebtel_sha256": str(_compute_file_sha256(args.ebtel_path)),
                "spectral_domain": str(obs_map.domain),
                "spectral_label": str(render_selection.spectral_label),
                "observer_name_effective": str(args.observer or "saved_metadata"),
                "observer_name": str(model_observer_meta.get("observer_name", args.observer or "earth")),
                "observer_lonc_deg": float(model_observer_meta.get("observer_lonc_deg", 0.0)),
                "observer_b0sun_deg": float(model_observer_meta.get("observer_b0sun_deg", 0.0)),
                "observer_dsun_cm": float(model_observer_meta.get("observer_dsun_cm", 1.495978707e13)),
                "target_metric": str(result.target_metric),
                "target_metric_value": float(result.objective_value),
                "chi2": float(result.metrics.chi2),
                "rho2": float(result.metrics.rho2),
                "eta2": float(result.metrics.eta2),
                "q0_recovered": float(result.q0),
                "fit_success": bool(result.success),
                "optimizer_message": str(result.message),
                "nfev": int(result.nfev),
                "nit": int(result.nit),
                "used_adaptive_bracketing": bool(result.used_adaptive_bracketing),
                "bracket_found": bool(result.bracket_found),
                "bracket": [float(v) for v in result.bracket] if result.bracket is not None else None,
                "fit_q0_trials": [float(q) for q in result.trial_q0],
                "fit_metric_trials": [float(v) for v in result.trial_objective_values],
                "fit_chi2_trials": [float(v) for v in getattr(result, "trial_chi2_values", ())],
                "fit_rho2_trials": [float(v) for v in getattr(result, "trial_rho2_values", ())],
                "fit_eta2_trials": [float(v) for v in getattr(result, "trial_eta2_values", ())],
                "map_xc_arcsec": float(geometry.xc),
                "map_yc_arcsec": float(geometry.yc),
                "map_dx_arcsec": float(geometry.dx),
                "map_dy_arcsec": float(geometry.dy),
                "active_frequency_ghz": (
                    None if render_selection.active_frequency_ghz is None else float(render_selection.active_frequency_ghz)
                ),
                "frequency_ghz": (
                    None if render_selection.active_frequency_ghz is None else float(render_selection.active_frequency_ghz)
                ),
                "wavelength_angstrom": None if obs_map.wavelength_angstrom is None else float(obs_map.wavelength_angstrom),
                "euv_channel": render_selection.euv_channel,
                "euv_instrument": render_selection.euv_instrument,
                "tr_mask_bmin_gauss": (
                    abs(float(args.tr_mask_bmin_gauss)) if render_selection.domain != "mw" else None
                ),
                "tr_mask_source": (
                    "abs_blos_ge_bmin" if euv_tr_mask is not None else ("unavailable" if render_selection.domain != "mw" else None)
                ),
                "metrics_mask_threshold": float(args.metrics_mask_threshold),
                "metrics_mask_fits": (
                    None if args.metrics_mask_fits is None else str(Path(args.metrics_mask_fits).expanduser())
                ),
                "metrics_mask_source": (
                    "explicit_fits" if explicit_metric_mask is not None else "union_threshold"
                ),
                "mask_type": (
                    "explicit_fits" if explicit_metric_mask is not None else "union"
                ),
                "a": float(a_param),
                "b": float(b_param),
                "psf_bmaj_arcsec": float(psf_bmaj_arcsec) if psf_bmaj_arcsec is not None else None,
                "psf_bmin_arcsec": float(psf_bmin_arcsec) if psf_bmin_arcsec is not None else None,
                "psf_bpa_deg": float(psf_bpa_deg) if psf_bpa_deg is not None else None,
                "psf_source": str(psf_source),
                "observer_obs_time": target_header.get("DATE-OBS", ""),
                "point_status": "computed",
            }

            def _save_outputs() -> None:
                blos_reference = blos_reference_for_fov
                trial_raw_modeled_maps = None
                trial_modeled_maps = None
                trial_residual_maps = None
                trial_euv_coronal_maps = None
                trial_euv_tr_maps = None
                trial_q0_values = [float(q) for q in result.trial_q0]
                if trial_q0_values:
                    raw_trials: list[np.ndarray] = []
                    modeled_trials: list[np.ndarray] = []
                    residual_trials: list[np.ndarray] = []
                    euv_coronal_trials: list[np.ndarray] = []
                    euv_tr_trials: list[np.ndarray] = []
                    for q0_value in trial_q0_values:
                        cached_payload = _lookup_cached_render_payload(render_cache, q0_value)
                        cached_pair = _lookup_cached_render_pair(render_cache, q0_value)
                        if cached_pair is None:
                            raw_trials = []
                            modeled_trials = []
                            residual_trials = []
                            euv_coronal_trials = []
                            euv_tr_trials = []
                            break
                        raw_trial, modeled_trial = cached_pair
                        raw_trials.append(np.asarray(raw_trial, dtype=np.float32))
                        modeled_trials.append(np.asarray(modeled_trial, dtype=np.float32))
                        residual_trials.append(np.asarray(modeled_trial, dtype=np.float32) - np.asarray(observed_cropped, dtype=np.float32))
                        if cached_payload is not None and cached_payload.get("euv_coronal") is not None and cached_payload.get("euv_tr") is not None:
                            euv_coronal_trials.append(np.asarray(cached_payload["euv_coronal"], dtype=np.float32))
                            euv_tr_trials.append(np.asarray(cached_payload["euv_tr"], dtype=np.float32))
                    if raw_trials and len(raw_trials) == len(trial_q0_values):
                        trial_raw_modeled_maps = np.stack(raw_trials, axis=0)
                        trial_modeled_maps = np.stack(modeled_trials, axis=0)
                        trial_residual_maps = np.stack(residual_trials, axis=0)
                    if euv_coronal_trials and len(euv_coronal_trials) == len(trial_q0_values):
                        trial_euv_coronal_maps = np.stack(euv_coronal_trials, axis=0)
                        trial_euv_tr_maps = np.stack(euv_tr_trials, axis=0)
                euv_coronal_best = None
                euv_tr_best = None
                euv_tr_mask_payload = None if euv_tr_mask is None else np.asarray(euv_tr_mask, dtype=bool)
                if cached_best_payload is not None:
                    if cached_best_payload.get("euv_coronal") is not None:
                        euv_coronal_best = np.asarray(cached_best_payload["euv_coronal"], dtype=np.float32)
                    if cached_best_payload.get("euv_tr") is not None:
                        euv_tr_best = np.asarray(cached_best_payload["euv_tr"], dtype=np.float32)
                    if cached_best_payload.get("euv_tr_mask") is not None:
                        euv_tr_mask_payload = np.asarray(cached_best_payload["euv_tr_mask"], dtype=bool)
                save_q0_artifact(
                    h5_path,
                    observed=observed_cropped,
                    sigma_map=sigma_cropped,
                    modeled_best=modeled_best,
                    raw_modeled_best=raw_modeled_best,
                    residual=residual,
                    frequency_ghz=(
                        float(render_selection.active_frequency_ghz)
                        if render_selection.active_frequency_ghz is not None
                        else None
                    ),
                    q0_fitted=result.q0,
                    metrics_dict={
                        "chi2": result.metrics.chi2,
                        "rho2": result.metrics.rho2,
                        "eta2": result.metrics.eta2,
                    },
                    diagnostics=diagnostics,
                    noise_diagnostics=noise_diagnostics,
                    wcs_header=target_header,
                    model_path=args.model_h5,
                    blos_reference=blos_reference,
                    trial_raw_modeled_maps=trial_raw_modeled_maps,
                    trial_modeled_maps=trial_modeled_maps,
                    trial_residual_maps=trial_residual_maps,
                    euv_coronal_best=euv_coronal_best,
                    euv_tr_best=euv_tr_best,
                    euv_tr_mask=euv_tr_mask_payload,
                    trial_euv_coronal_maps=trial_euv_coronal_maps,
                    trial_euv_tr_maps=trial_euv_tr_maps,
                )
                if not args.no_artifacts_png:
                    plot_q0_artifact_panel(
                        png_path,
                        model_path=args.model_h5,
                        observed_noisy=observed_cropped,
                        raw_modeled_best=raw_modeled_best,
                        modeled_best=modeled_best,
                        residual=residual,
                        wcs_header=target_header,
                        frequency_ghz=(
                            float(render_selection.active_frequency_ghz)
                            if render_selection.active_frequency_ghz is not None
                            else None
                        ),
                        diagnostics=diagnostics,
                        show_plot=bool(args.show_plot),
                        blos_reference=blos_reference,
                    )

            _, _save_elapsed = _run_stage(
                "Save artifacts",
                _save_outputs,
                spinner=stage_spinner,
                stage_index=3,
                stage_total=stage_total,
            )
            print(f"  ✓ Saved to: {h5_path}")
            if not args.no_artifacts_png:
                print(f"  ✓ PNG panel: {png_path}")
                print(f"  Open PNG: {_open_path_hint(png_path)}")
            print(f"  Replot PNG: {sys.executable} examples/replot_q0_artifacts.py \"{h5_path}\"")
            print(f"  Interactive viewer: pychmp-view \"{h5_path}\"")
        except Exception as e:
            print(f"  ✗ Failed to save artifacts: {e}")

    total_elapsed = time.perf_counter() - total_start
    fit_success = bool(result.success) if result is not None else False
    print(f"\n{'=' * 70}")
    print(f"FITTING COMPLETE: success={fit_success} total={total_elapsed:.3f}s")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
