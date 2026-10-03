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


# FWHM = σ * 2√(2 ln 2). Radio σ headers are stored in the existing FWHM fields.
_SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


def sigma_arcsec_to_fwhm(sigma_arcsec: float) -> float:
    """Convert Gaussian σ (arcsec) to FWHM (arcsec)."""
    return float(sigma_arcsec) * _SIGMA_TO_FWHM


def fwhm_arcsec_to_sigma(fwhm_arcsec: float) -> float:
    """Convert Gaussian FWHM (arcsec) to σ (arcsec)."""
    return float(fwhm_arcsec) / _SIGMA_TO_FWHM


def _normalize_pa_deg(pa_deg: float) -> float:
    """Wrap position angle to (-90, 90] degrees."""
    pa = float(pa_deg) % 180.0
    if pa > 90.0:
        pa -= 180.0
    if pa <= -90.0:
        pa += 180.0
    return pa


def ordered_major_minor_sigma(
    sigma_a: float,
    sigma_b: float,
    *,
    pa_of_a_deg: float,
) -> tuple[float, float, float]:
    """Return (σ_maj, σ_min, PA_maj_deg) with σ_maj ≥ σ_min."""
    sa = float(sigma_a)
    sb = float(sigma_b)
    if sa >= sb:
        return sa, sb, _normalize_pa_deg(pa_of_a_deg)
    return sb, sa, _normalize_pa_deg(pa_of_a_deg + 90.0)


def correlated_sigma_to_rotated(
    sx: float,
    sy: float,
    rho: float,
) -> tuple[float, float, float]:
    """Convert gx ``MakeSRHbeam`` (sx, sy, rho) to rotated (σ_maj, σ_min, PA_deg).

    ``MakeSRHbeam`` uses the standard bivariate Gaussian with σ_x=sx, σ_y=sy and
    correlation ``rho`` (FITS ``beam_sx`` / ``beam_sy`` / ``beam_rho``). Eigenvalues
    of the covariance are the ellipse σ axes; PA is CCW from +X to the major axis
    and matches the angle consumed by ``elliptical_gaussian_kernel`` / ``Gauss2Drot``.
    """
    sx = float(sx)
    sy = float(sy)
    rho = float(np.clip(rho, -0.999999, 0.999999))
    if sx <= 0.0 or sy <= 0.0:
        raise ValueError("beam_sx/beam_sy must be positive")

    var_x = sx * sx
    var_y = sy * sy
    cov_xy = rho * sx * sy
    mid = 0.5 * (var_x + var_y)
    diff = 0.5 * (var_x - var_y)
    disc = np.sqrt(diff * diff + cov_xy * cov_xy)
    lam_maj = mid + disc
    lam_min = mid - disc
    smaj = float(np.sqrt(max(lam_maj, 0.0)))
    smin = float(np.sqrt(max(lam_min, 0.0)))

    if abs(cov_xy) > 1e-15 or abs(lam_maj - var_x) > 1e-15:
        if abs(cov_xy) >= abs(lam_maj - var_x):
            vx, vy = cov_xy, lam_maj - var_x
        else:
            vx, vy = lam_maj - var_y, cov_xy
        pa = np.rad2deg(np.arctan2(vy, vx))
    else:
        pa = 0.0 if var_x >= var_y else 90.0
    return smaj, smin, _normalize_pa_deg(pa)


def make_srh_correlated_beam(
    sx: float,
    sy: float,
    rho: float,
    *,
    nx: int,
    ny: int,
    dx_arcsec: float,
    dy_arcsec: float,
) -> np.ndarray:
    """IDL ``MakeSRHbeam`` (unnormalized, peak=1 at center).

    ``sx``/``sy`` are Gaussian σ in arcsec (IDL comments call these 1/e widths;
    the formula is the standard bivariate normal with those σ parameters).
    """
    sx = float(sx)
    sy = float(sy)
    rho = float(np.clip(rho, -0.999999, 0.999999))
    x = (np.arange(nx, dtype=float) - 0.5 * nx + 0.5) * float(dx_arcsec)
    y = (np.arange(ny, dtype=float) - 0.5 * ny + 0.5) * float(dy_arcsec)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    denom = 1.0 - rho * rho
    beam = np.exp(
        -0.5
        / denom
        * (xx * xx / (sx * sx) + yy * yy / (sy * sy) - 2.0 * rho * xx * yy / (sx * sy))
    )
    return beam


def _header_first(header: fits.Header, keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in header:
            return header[key]
        for existing in header.keys():
            if str(existing).upper() == key.upper():
                return header[existing]
    return None


def _psf_from_sigma_axes(
    *,
    sigma_a: float,
    sigma_b: float,
    pa_of_a_deg: float,
    source: str,
    allows_frequency_scaling: bool,
) -> PSFMetadata | None:
    if sigma_a <= 0.0 or sigma_b <= 0.0:
        return None
    smaj, smin, bpa = ordered_major_minor_sigma(sigma_a, sigma_b, pa_of_a_deg=pa_of_a_deg)
    return PSFMetadata(
        source=source,
        kind="gaussian",
        bmaj_arcsec=sigma_arcsec_to_fwhm(smaj),
        bmin_arcsec=sigma_arcsec_to_fwhm(smin),
        bpa_deg=float(bpa),
        allows_frequency_scaling=bool(allows_frequency_scaling),
    )


def extract_psf_metadata_from_header(header: fits.Header) -> PSFMetadata | None:
    """Resolve beam metadata from FITS header keywords.

    Resolution order:
    1. Viktor/srhimages σ axes: ``BEAM_SA`` / ``BEAM_SB`` / ``BEAM_PHI`` (``BEAM_P`` must be 1)
    2. gx_simulator SRH correlated σ: ``BEAM_SX`` / ``BEAM_SY`` / ``BEAM_RHO``
    3. Standard FWHM ellipse: ``BMAJ`` / ``BMIN`` / ``BPA`` (degrees if |value|≤1)

    Radio σ headers are converted to FWHM for storage in ``PSFMetadata`` so the
    existing ``elliptical_gaussian_kernel`` path stays unchanged (FWHM→σ internally).
    """
    # --- SRH σ + PA (Viktor / srhimages) ---
    sa_raw = _header_first(header, ("BEAM_SA", "beam_sa"))
    sb_raw = _header_first(header, ("BEAM_SB", "beam_sb"))
    if sa_raw is not None and sb_raw is not None:
        sa = _optional_float(sa_raw)
        sb = _optional_float(sb_raw)
        phi = _optional_float(_header_first(header, ("BEAM_PHI", "beam_phi"))) or 0.0
        beam_p = _optional_float(_header_first(header, ("BEAM_P", "beam_p")))
        if beam_p is not None and abs(beam_p - 1.0) > 1e-6:
            # Super-Gaussian p≠1 not supported in the first slice.
            return None
        if sa is not None and sb is not None:
            # Match Viktor get_psf: gaussian2d(..., theta=-beam_phi)
            return _psf_from_sigma_axes(
                sigma_a=sa,
                sigma_b=sb,
                pa_of_a_deg=-float(phi),
                source="fits_header:srh_sigma",
                allows_frequency_scaling=True,
            )

    # --- gx MakeSRHbeam correlated σ ---
    sx_raw = _header_first(header, ("BEAM_SX", "beam_sx"))
    sy_raw = _header_first(header, ("BEAM_SY", "beam_sy"))
    if sx_raw is not None and sy_raw is not None:
        sx = _optional_float(sx_raw)
        sy = _optional_float(sy_raw)
        rho = _optional_float(_header_first(header, ("BEAM_RHO", "beam_rho"))) or 0.0
        if sx is not None and sy is not None:
            try:
                smaj, smin, pa = correlated_sigma_to_rotated(sx, sy, rho)
            except ValueError:
                return None
            return PSFMetadata(
                source="fits_header:srh_correlated",
                kind="gaussian",
                bmaj_arcsec=sigma_arcsec_to_fwhm(smaj),
                bmin_arcsec=sigma_arcsec_to_fwhm(smin),
                bpa_deg=float(pa),
                allows_frequency_scaling=True,
            )

    # --- Standard FWHM beam ---
    bmaj_raw = _header_first(header, ("BMAJ", "BMAJ_DEG", "BMAJDEG", "BEAM_MAJ", "PSF_BMAJ"))
    bmin_raw = _header_first(header, ("BMIN", "BMIN_DEG", "BMINDEG", "BEAM_MIN", "PSF_BMIN"))
    bpa_raw = _header_first(header, ("BPA", "BPA_DEG", "BEAM_PA", "PSF_BPA"))

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
        if instrument_key in {
            "stereoaeuvi",
            "euvia",
            "ahead",
            "stereoa",
            "stereo",
            "stereobeuvi",
            "euvib",
            "behind",
        }:
            response_euvi = import_module("pyeuvtools.response.euvi")
            spacecraft = "behind" if instrument_key in {"stereobeuvi", "euvib", "behind"} else "ahead"
            pixel_arcsec = float(getattr(response_euvi, "_EUVI_PIXEL_ARCSEC")[spacecraft])
            label = "stereoaeuvi" if spacecraft == "ahead" else "stereobeuvi"
            return pixel_arcsec, label
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


def beam_fwhm_from_kernel(
    kernel: np.ndarray,
    *,
    dx_arcsec: float,
    dy_arcsec: float,
) -> dict[str, float] | None:
    """Estimate elliptical Gaussian FWHM (arcsec) from a 2D PSF kernel."""
    arr = np.asarray(kernel, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return None
    weights = np.where(np.isfinite(arr) & (arr > 0.0), arr, 0.0)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        return None

    ny, nx = arr.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    x0 = float(np.sum(xx * weights) / total)
    y0 = float(np.sum(yy * weights) / total)
    dx_pix = float(np.sum((xx - x0) ** 2 * weights) / total)
    dy_pix = float(np.sum((yy - y0) ** 2 * weights) / total)
    dxy_pix = float(np.sum((xx - x0) * (yy - y0) * weights) / total)
    if not all(np.isfinite(value) for value in (dx_pix, dy_pix, dxy_pix)):
        return None

    scale_x = max(abs(float(dx_arcsec)), 1e-12)
    scale_y = max(abs(float(dy_arcsec)), 1e-12)
    cov_xx = dx_pix * scale_x**2
    cov_yy = dy_pix * scale_y**2
    cov_xy = dxy_pix * scale_x * scale_y
    cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=float)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    order = np.argsort(eigvals)
    bmin_arcsec = float(_SIGMA_TO_FWHM * np.sqrt(eigvals[order[0]]))
    bmaj_arcsec = float(_SIGMA_TO_FWHM * np.sqrt(eigvals[order[1]]))
    if bmaj_arcsec <= 0.0 or bmin_arcsec <= 0.0:
        return None
    major_vec = eigvecs[:, order[1]]
    bpa_deg = float(np.degrees(np.arctan2(major_vec[1], major_vec[0])))
    return {
        "bmaj_arcsec": bmaj_arcsec,
        "bmin_arcsec": max(bmin_arcsec, 1e-6),
        "bpa_deg": bpa_deg,
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


def _first_finite_float(*values: Any) -> float | None:
    for value in values:
        numeric = _optional_float(value)
        if numeric is not None:
            return float(numeric)
    return None


def psf_metadata_from_diagnostics(diagnostics: dict[str, Any] | None) -> PSFMetadata | None:
    """Reconstruct PSF metadata from persisted slice/search diagnostics."""
    diag = dict(diagnostics or {})
    resolved = dict(diag.get("resolved_psf") or {})
    kind = str(resolved.get("kind") or "").strip().lower()
    source = str(diag.get("psf_source") or resolved.get("source") or "diagnostics").strip() or "diagnostics"
    if kind == "gaussian" or resolved.get("active_bmaj_arcsec") is not None:
        bmaj = _first_finite_float(
            resolved.get("active_bmaj_arcsec"),
            resolved.get("psf_bmaj_arcsec"),
            resolved.get("reference_bmaj_arcsec"),
        )
        bmin = _first_finite_float(
            resolved.get("active_bmin_arcsec"),
            resolved.get("psf_bmin_arcsec"),
            resolved.get("reference_bmin_arcsec"),
        )
        bpa = _first_finite_float(
            resolved.get("active_bpa_deg"),
            resolved.get("psf_bpa_deg"),
            resolved.get("reference_bpa_deg"),
            0.0,
        )
        if bmaj is not None and bmin is not None and bpa is not None:
            return PSFMetadata(
                source=source,
                kind="gaussian",
                bmaj_arcsec=float(bmaj),
                bmin_arcsec=float(bmin),
                bpa_deg=float(bpa),
                allows_frequency_scaling=bool(resolved.get("allows_frequency_scaling", False)),
            )
    return None


def _normalized_psf_kernel_array(psf_kernel: np.ndarray | None) -> np.ndarray | None:
    if psf_kernel is None:
        return None
    kernel = np.asarray(psf_kernel, dtype=float)
    if kernel.ndim != 2 or kernel.size == 0:
        return None
    kernel_sum = float(np.nansum(kernel))
    if not np.isfinite(kernel_sum) or kernel_sum == 0.0:
        return None
    return np.asarray(kernel / kernel_sum, dtype=np.float32)


_PSF_KERNEL_BUILD_CACHE: dict[tuple[Any, ...], np.ndarray | None] = {}


def _psf_kernel_build_cache_key(
    metadata: PSFMetadata,
    *,
    dx_arcsec: float,
    dy_arcsec: float,
    active_frequency_ghz: float | None,
) -> tuple[Any, ...]:
    return (
        str(metadata.source),
        str(metadata.kind),
        None if metadata.bmaj_arcsec is None else float(metadata.bmaj_arcsec),
        None if metadata.bmin_arcsec is None else float(metadata.bmin_arcsec),
        None if metadata.bpa_deg is None else float(metadata.bpa_deg),
        float(dx_arcsec),
        float(dy_arcsec),
        None if active_frequency_ghz is None else float(active_frequency_ghz),
    )


def clear_psf_kernel_build_cache() -> None:
    """Clear the in-process PSF kernel build cache (mainly for tests)."""
    _PSF_KERNEL_BUILD_CACHE.clear()


def resolve_slice_psf_kernel(
    *,
    stored_kernel: np.ndarray | None,
    diagnostics: dict[str, Any] | None = None,
    dx_arcsec: float,
    dy_arcsec: float,
    active_frequency_ghz: float | None = None,
) -> np.ndarray | None:
    """Return the slice-common PSF kernel from storage or diagnostics."""
    normalized = _normalized_psf_kernel_array(stored_kernel)
    if normalized is not None:
        return normalized

    metadata = psf_metadata_from_diagnostics(diagnostics)
    if metadata is None:
        return None

    resolved_frequency = active_frequency_ghz
    if resolved_frequency is None and diagnostics is not None:
        for key in ("frequency_ghz", "active_frequency_ghz", "mw_frequency_ghz"):
            candidate = _optional_float(diagnostics.get(key))
            if candidate is not None:
                resolved_frequency = float(candidate)
                break
        if resolved_frequency is None:
            resolved_block = dict(diagnostics.get("resolved_psf") or {})
            resolved_frequency = _optional_float(resolved_block.get("active_frequency_ghz"))

    cache_key = _psf_kernel_build_cache_key(
        metadata,
        dx_arcsec=float(dx_arcsec),
        dy_arcsec=float(dy_arcsec),
        active_frequency_ghz=resolved_frequency,
    )
    if cache_key in _PSF_KERNEL_BUILD_CACHE:
        return _PSF_KERNEL_BUILD_CACHE[cache_key]

    kernel, _resolved = build_psf_kernel(
        metadata=metadata,
        dx_arcsec=float(dx_arcsec),
        dy_arcsec=float(dy_arcsec),
        active_frequency_ghz=resolved_frequency,
        ref_frequency_ghz=None,
        scale_inverse_frequency=False,
    )
    normalized_kernel = _normalized_psf_kernel_array(kernel)
    _PSF_KERNEL_BUILD_CACHE[cache_key] = normalized_kernel
    return normalized_kernel


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
