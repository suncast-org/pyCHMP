"""Core PSF helpers for pyCHMP.

This module centralizes PSF resolution so workflow scripts can rely on a single
package-owned contract instead of re-implementing beam parsing and kernel
construction locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np
from astropy.io import fits
from scipy.signal import fftconvolve

_RESPONSE_SAMPLING_FWHM_FACTORS: dict[str, float] = {
    "aia": 2.5,
    "sdoaia": 2.5,
}
@dataclass(frozen=True)
class PSFMetadata:
    source: str
    kind: str
    bmaj_arcsec: float | None = None
    bmin_arcsec: float | None = None
    bpa_deg: float | None = None
    kernel: np.ndarray | None = None
    allows_frequency_scaling: bool = False

    def as_dict(self) -> dict[str, Any]:
        out = {
            "source": str(self.source),
            "kind": str(self.kind),
            "allows_frequency_scaling": bool(self.allows_frequency_scaling),
        }
        if self.bmaj_arcsec is not None:
            out["psf_bmaj_arcsec"] = float(self.bmaj_arcsec)
        if self.bmin_arcsec is not None:
            out["psf_bmin_arcsec"] = float(self.bmin_arcsec)
        if self.bpa_deg is not None:
            out["psf_bpa_deg"] = float(self.bpa_deg)
        if self.kernel is not None:
            out["psf_kernel_shape"] = tuple(int(value) for value in np.asarray(self.kernel, dtype=float).shape)
        return out


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


def _normalize_instrument_key(instrument_name: str | None) -> str:
    return "".join(ch for ch in str(instrument_name or "").strip().lower() if ch.isalnum())


def _channel_label_from_wavelength(wavelength_angstrom: float | None) -> str | None:
    numeric = _optional_float(wavelength_angstrom)
    if numeric is None:
        return None
    rounded = round(float(numeric))
    if np.isclose(float(numeric), float(rounded), rtol=0.0, atol=1e-6):
        return str(int(rounded))
    return f"{float(numeric):.6g}"


def _build_aiapy_aia_kernel(
    *,
    instrument_name: str | None,
    wavelength_angstrom: float | None,
) -> PSFMetadata | None:
    instrument_key = _normalize_instrument_key(instrument_name)
    if instrument_key not in {"aia", "sdoaia"}:
        return None
    channel_label = _channel_label_from_wavelength(wavelength_angstrom)
    if channel_label is None:
        return None

    try:
        units = import_module("astropy.units")
        aiapy_psf = import_module("aiapy.psf")
    except ImportError:
        return None

    try:
        kernel = aiapy_psf.calculate_psf(float(channel_label) * units.angstrom)
    except Exception:
        return None

    kernel_arr = np.asarray(getattr(kernel, "value", kernel), dtype=float)
    if kernel_arr.ndim != 2 or kernel_arr.size == 0:
        return None

    return PSFMetadata(
        source=f"aiapy_psf:{channel_label}",
        kind="kernel",
        kernel=kernel_arr,
        allows_frequency_scaling=False,
    )


def _lookup_response_sampling_pixel_arcsec(instrument_name: str | None) -> tuple[float, str] | None:
    instrument_key = _normalize_instrument_key(instrument_name)
    if not instrument_key:
        return None

    try:
        if instrument_key in {"aia", "sdoaia"}:
            response_aia = import_module("pyeuvtools.response.aia")
            ds_arcsec2 = float(getattr(response_aia, "_GX_AIA_DS_ARCSEC"))
            return float(np.sqrt(ds_arcsec2)), instrument_key
        if instrument_key in {"euifsi", "fsi", "soloorbitereuifsi", "soloeuifsi"}:
            response_eui = import_module("pyeuvtools.response.eui")
            pixel_arcsec = float(getattr(response_eui, "_EUI_PIXEL_ARCSEC")["fsi"])
            return pixel_arcsec, "euifsi"
        if instrument_key in {"euihri", "hri", "soloorbitereuihri", "soloeuihri"}:
            response_eui = import_module("pyeuvtools.response.eui")
            pixel_arcsec = float(getattr(response_eui, "_EUI_PIXEL_ARCSEC")["hri"])
            return pixel_arcsec, "euihri"
        if instrument_key in {"stereoaeuvi", "euvia", "ahead", "stereoaeuvi"}:
            response_euvi = import_module("pyeuvtools.response.euvi")
            pixel_arcsec = float(getattr(response_euvi, "_EUVI_PIXEL_ARCSEC")["ahead"])
            return pixel_arcsec, "stereoaeuvi"
        if instrument_key in {"stereobeuvi", "euvib", "behind", "stereobeuvi"}:
            response_euvi = import_module("pyeuvtools.response.euvi")
            pixel_arcsec = float(getattr(response_euvi, "_EUVI_PIXEL_ARCSEC")["behind"])
            return pixel_arcsec, "stereobeuvi"
        if instrument_key in {"trace"}:
            response_trace = import_module("pyeuvtools.response.trace")
            return float(getattr(response_trace, "TRACE_PIXEL_ARCSEC")), instrument_key
        if instrument_key in {"sxt", "yohkohsxt"}:
            response_sxt = import_module("pyeuvtools.response.sxt")
            return float(getattr(response_sxt, "SXT_PIXEL_ARCSEC")), "sxt"
    except Exception:
        return None

    return None


def _response_sampling_default_psf(
    *,
    instrument_name: str | None,
) -> PSFMetadata | None:
    resolved_sampling = _lookup_response_sampling_pixel_arcsec(instrument_name)
    if resolved_sampling is None:
        return None

    pixel_arcsec, normalized_key = resolved_sampling
    fwhm_factor = float(_RESPONSE_SAMPLING_FWHM_FACTORS.get(normalized_key, 1.0))
    beam_arcsec = float(pixel_arcsec) * fwhm_factor
    return PSFMetadata(
        source=f"response_sampling_default:{normalized_key}",
        kind="gaussian",
        bmaj_arcsec=beam_arcsec,
        bmin_arcsec=beam_arcsec,
        bpa_deg=0.0,
        allows_frequency_scaling=False,
    )


def extract_psf_metadata_from_header(header: fits.Header) -> PSFMetadata | None:
    def first_header_value(keys: tuple[str, ...]) -> Any | None:
        for key in keys:
            if key in header:
                return header[key]
        return None

    bmaj_raw = first_header_value(("BMAJ", "BMAJ_DEG", "BMAJDEG", "BEAM_MAJ", "PSF_BMAJ"))
    bmin_raw = first_header_value(("BMIN", "BMIN_DEG", "BMINDEG", "BEAM_MIN", "PSF_BMIN"))
    bpa_raw = first_header_value(("BPA", "BPA_DEG", "BEAM_PA", "PSF_BPA"))

    if bmaj_raw is None or bmin_raw is None:
        return None

    try:
        bmaj = float(bmaj_raw)
        bmin = float(bmin_raw)
        bpa = float(bpa_raw) if bpa_raw is not None else 0.0
    except Exception:
        return None

    if abs(bmaj) <= 1.0 and abs(bmin) <= 1.0:
        bmaj_arcsec = bmaj * 3600.0
        bmin_arcsec = bmin * 3600.0
    else:
        bmaj_arcsec = bmaj
        bmin_arcsec = bmin

    if not (np.isfinite(bmaj_arcsec) and np.isfinite(bmin_arcsec) and bmaj_arcsec > 0 and bmin_arcsec > 0):
        return None

    return PSFMetadata(
        source="fits_header",
        kind="gaussian",
        bmaj_arcsec=float(bmaj_arcsec),
        bmin_arcsec=float(bmin_arcsec),
        bpa_deg=float(bpa),
        allows_frequency_scaling=False,
    )


def default_psf_metadata(
    *,
    domain: str | None,
    instrument_name: str | None,
    wavelength_angstrom: float | None = None,
    date_obs: str | None = None,
) -> PSFMetadata | None:
    if str(domain or "").strip().lower() not in {"euv", "uv"}:
        return None
    _ = date_obs
    aiapy_default = _build_aiapy_aia_kernel(
        instrument_name=instrument_name,
        wavelength_angstrom=wavelength_angstrom,
    )
    if aiapy_default is not None:
        return aiapy_default
    return _response_sampling_default_psf(instrument_name=instrument_name)


def resolve_psf_metadata(
    *,
    header_psf: PSFMetadata | None,
    domain: str | None,
    instrument_name: str | None,
    wavelength_angstrom: float | None = None,
    date_obs: str | None = None,
    cli_psf_kernel: np.ndarray | None = None,
    cli_psf_bmaj_arcsec: float | None,
    cli_psf_bmin_arcsec: float | None,
    cli_psf_bpa_deg: float | None,
    fallback_psf_bmaj_arcsec: float | None,
    fallback_psf_bmin_arcsec: float | None,
    fallback_psf_bpa_deg: float | None,
    override_header_psf: bool,
) -> PSFMetadata | None:
    has_cli_psf_override = cli_psf_kernel is not None or any(
        value is not None for value in (cli_psf_bmaj_arcsec, cli_psf_bmin_arcsec, cli_psf_bpa_deg)
    )
    has_cli_psf_fallback = any(
        value is not None for value in (fallback_psf_bmaj_arcsec, fallback_psf_bmin_arcsec, fallback_psf_bpa_deg)
    )

    if cli_psf_kernel is not None:
        kernel = np.asarray(cli_psf_kernel, dtype=float)
        if kernel.ndim != 2 or kernel.size == 0:
            raise ValueError("cli_psf_kernel must be a non-empty 2D array")
        return PSFMetadata(source="cli_kernel", kind="kernel", kernel=kernel, allows_frequency_scaling=False)
    if header_psf is not None and not override_header_psf:
        return header_psf
    if has_cli_psf_override:
        return PSFMetadata(
            source="cli_override",
            kind="gaussian",
            bmaj_arcsec=_optional_float(cli_psf_bmaj_arcsec),
            bmin_arcsec=_optional_float(cli_psf_bmin_arcsec),
            bpa_deg=_optional_float(cli_psf_bpa_deg),
            allows_frequency_scaling=True,
        )
    if header_psf is not None:
        return header_psf
    if has_cli_psf_fallback:
        return PSFMetadata(
            source="cli_fallback",
            kind="gaussian",
            bmaj_arcsec=_optional_float(fallback_psf_bmaj_arcsec),
            bmin_arcsec=_optional_float(fallback_psf_bmin_arcsec),
            bpa_deg=_optional_float(fallback_psf_bpa_deg),
            allows_frequency_scaling=True,
        )
    return default_psf_metadata(
        domain=domain,
        instrument_name=instrument_name,
        wavelength_angstrom=wavelength_angstrom,
        date_obs=date_obs,
    )


def effective_psf_parameters(
    *,
    metadata: PSFMetadata | None,
    active_frequency_ghz: float,
    ref_frequency_ghz: float | None,
    scale_inverse_frequency: bool,
) -> dict[str, float | bool] | None:
    if metadata is None or metadata.kind != "gaussian":
        return None
    if metadata.bmaj_arcsec is None or metadata.bmin_arcsec is None or metadata.bpa_deg is None:
        return None
    psf_scale = (
        float(ref_frequency_ghz) / float(active_frequency_ghz)
        if scale_inverse_frequency and ref_frequency_ghz is not None and bool(metadata.allows_frequency_scaling)
        else 1.0
    )
    return {
        "reference_bmaj_arcsec": float(metadata.bmaj_arcsec),
        "reference_bmin_arcsec": float(metadata.bmin_arcsec),
        "reference_bpa_deg": float(metadata.bpa_deg),
        "active_bmaj_arcsec": float(metadata.bmaj_arcsec) * float(psf_scale),
        "active_bmin_arcsec": float(metadata.bmin_arcsec) * float(psf_scale),
        "active_bpa_deg": float(metadata.bpa_deg),
        "reference_frequency_ghz": float(ref_frequency_ghz) if ref_frequency_ghz is not None else float(active_frequency_ghz),
        "active_frequency_ghz": float(active_frequency_ghz),
        "scaled": bool(scale_inverse_frequency and ref_frequency_ghz is not None and not np.isclose(psf_scale, 1.0)),
        "scale_factor": float(psf_scale),
    }


def elliptical_gaussian_kernel(
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


def build_psf_kernel(
    *,
    metadata: PSFMetadata | None,
    dx_arcsec: float,
    dy_arcsec: float,
    active_frequency_ghz: float | None = None,
    ref_frequency_ghz: float | None = None,
    scale_inverse_frequency: bool = False,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    if metadata is None:
        return None, None
    if metadata.kind == "kernel":
        kernel = np.asarray(metadata.kernel, dtype=float) if metadata.kernel is not None else None
        if kernel is None or kernel.ndim != 2 or kernel.size == 0:
            return None, None
        kernel_sum = float(np.sum(kernel))
        if np.isfinite(kernel_sum) and kernel_sum != 0.0:
            kernel = kernel / kernel_sum
        return kernel, {**metadata.as_dict(), "normalized": True}
    if metadata.kind != "gaussian":
        return None, None
    if metadata.bmaj_arcsec is None or metadata.bmin_arcsec is None or metadata.bpa_deg is None:
        return None, None

    resolved_meta: dict[str, Any]
    kernel_bmaj_arcsec = float(metadata.bmaj_arcsec)
    kernel_bmin_arcsec = float(metadata.bmin_arcsec)
    if active_frequency_ghz is not None:
        effective = effective_psf_parameters(
            metadata=metadata,
            active_frequency_ghz=float(active_frequency_ghz),
            ref_frequency_ghz=ref_frequency_ghz,
            scale_inverse_frequency=bool(scale_inverse_frequency),
        )
        if effective is not None:
            kernel_bmaj_arcsec = float(effective["active_bmaj_arcsec"])
            kernel_bmin_arcsec = float(effective["active_bmin_arcsec"])
            resolved_meta = {**metadata.as_dict(), **effective}
        else:
            resolved_meta = metadata.as_dict()
    else:
        resolved_meta = metadata.as_dict()

    kernel = elliptical_gaussian_kernel(
        bmaj_arcsec=kernel_bmaj_arcsec,
        bmin_arcsec=kernel_bmin_arcsec,
        bpa_deg=float(metadata.bpa_deg),
        dx_arcsec=float(dx_arcsec),
        dy_arcsec=float(dy_arcsec),
    )
    return kernel, resolved_meta


def format_psf_report(
    *,
    metadata: PSFMetadata | None,
    active_frequency_ghz: float | None = None,
    ref_frequency_ghz: float | None = None,
    scale_inverse_frequency: bool = False,
) -> str:
    if metadata is None:
        return "PSF source: none"
    if metadata.kind == "kernel":
        kernel = np.asarray(metadata.kernel, dtype=float) if metadata.kernel is not None else None
        if kernel is None:
            return "PSF source: none"
        return f"PSF source: {metadata.source} kernel: shape={tuple(int(v) for v in kernel.shape)}"
    if metadata.bmaj_arcsec is None or metadata.bmin_arcsec is None or metadata.bpa_deg is None:
        return "PSF source: none"
    if active_frequency_ghz is None:
        return (
            f"PSF source: {metadata.source} "
            f"beam: bmaj={float(metadata.bmaj_arcsec):.3f} "
            f"bmin={float(metadata.bmin_arcsec):.3f} "
            f"bpa={float(metadata.bpa_deg):.3f}"
        )
    psf = effective_psf_parameters(
        metadata=metadata,
        active_frequency_ghz=float(active_frequency_ghz),
        ref_frequency_ghz=ref_frequency_ghz,
        scale_inverse_frequency=bool(scale_inverse_frequency),
    )
    if psf is None:
        return "PSF source: none"
    if bool(psf["scaled"]):
        return (
            f"PSF source: {metadata.source} "
            f"reference beam: bmaj={float(psf['reference_bmaj_arcsec']):.3f} "
            f"bmin={float(psf['reference_bmin_arcsec']):.3f} "
            f"bpa={float(psf['reference_bpa_deg']):.3f} @ {float(psf['reference_frequency_ghz']):.3f} GHz"
            f"\n    rescaled beam: bmaj={float(psf['active_bmaj_arcsec']):.3f} "
            f"bmin={float(psf['active_bmin_arcsec']):.3f} "
            f"bpa={float(psf['active_bpa_deg']):.3f} @ {float(psf['active_frequency_ghz']):.3f} GHz"
        )
    return (
        f"PSF source: {metadata.source} "
        f"beam: bmaj={float(psf['active_bmaj_arcsec']):.3f} "
        f"bmin={float(psf['active_bmin_arcsec']):.3f} "
        f"bpa={float(psf['active_bpa_deg']):.3f} @ {float(psf['active_frequency_ghz']):.3f} GHz"
    )


class KernelConvolvedRenderer:
    def __init__(self, base_renderer: Any, kernel: np.ndarray) -> None:
        self._base = base_renderer
        self._kernel = np.asarray(kernel, dtype=float)
        self._last_stokes_v: np.ndarray | None = None

    def render_stokes_raw(self, q0: float) -> tuple[np.ndarray, np.ndarray | None]:
        base = self._base
        if hasattr(base, "render_stokes_raw"):
            stokes_i, stokes_v = base.render_stokes_raw(float(q0))
            self._last_stokes_v = None if stokes_v is None else np.asarray(stokes_v, dtype=float)
            return np.asarray(stokes_i, dtype=float), self._last_stokes_v
        raw = np.asarray(base.render(q0), dtype=float)
        self._last_stokes_v = None
        return raw, None

    def render_pair(self, q0: float) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(self._base, "render_stokes_raw"):
            raw, self._last_stokes_v = self.render_stokes_raw(q0)
            raw = np.asarray(raw, dtype=float)
        else:
            raw = np.asarray(self._base.render(q0), dtype=float)
            self._last_stokes_v = None
        convolved = fftconvolve(raw, self._kernel, mode="same")
        return raw, convolved

    def render(self, q0: float) -> np.ndarray:
        _raw, convolved = self.render_pair(q0)
        return convolved
