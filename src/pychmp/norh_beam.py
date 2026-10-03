"""Nobeyama Radioheliograph (NORH) clean-beam helpers.

In-tree port of the SSW / gx_simulator NORH beam chain used by CHMP:

* ``norh_prog2pinf`` — dirty-map pixel scale + e-folding length from PROGNAME/freq
* ``norh_beam`` — projected clean beam on a marx×marx grid
* ``BeamFitNoRH`` / ``FitBeam`` — rotated Gaussian fit (σ axes + PA)

No SSW/IDL runtime dependency. Header SOLP is used in degrees (NORH FITS
convention); IDL ``norh_beam`` calls ``get_rb0p(/pangle)`` (radians) which
matches header SOLP to ≲0.01° on standard products.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit

_NS2EW = 1.000112494
_FREQ_RATIO_17_TO_33P8 = 17.0 / 33.8
_DEFAULT_EFL = 2.0654677
_DEFAULT_SEC_PER_PIX_DTY_17 = 4.64947


@dataclass(frozen=True)
class NorhBeamParams:
    """Clean-beam geometry at the observation frequency (Gaussian σ, arcsec)."""

    sigma_x_arcsec: float
    sigma_y_arcsec: float
    theta_rad: float
    frequency_ghz: float
    sec_per_pix_arcsec: float
    efl_in_pix_dty: float
    sec_per_pix_dty: float
    solp_rad: float
    pmat: tuple[float, float, float, float]
    marx: int

    @property
    def theta_deg(self) -> float:
        return float(np.rad2deg(self.theta_rad))


def norh_prog2pinf(
    progname: str | None,
    obs_freq: str | None,
    *,
    cellsize: float | None = None,
) -> tuple[float, float, float]:
    """Return ``(sec_per_pix_dty, efl_in_pix_dty, roll)`` like IDL ``norh_prog2pinf``."""
    name = str(progname or "").strip()
    freq = str(obs_freq or "").strip()
    roll = 0.0

    if name == "snap2d17suw":
        sec_per_pix_dty = _DEFAULT_SEC_PER_PIX_DTY_17 * 0.5
        cell = int(cellsize) if cellsize is not None and np.isfinite(cellsize) else -1
        if cell == 3:
            xa = 2.5
        elif cell == 5:
            xa = 2.3
        elif cell == 7:
            xa = 2.0
        else:
            xa = 1.8
        return float(sec_per_pix_dty), float(xa * np.sqrt(2.0)), roll

    # snap2d17 v5.1 / koshix / else (incl. snap2d34 …) share the same table.
    if freq == "17GHz":
        return float(_DEFAULT_SEC_PER_PIX_DTY_17), float(_DEFAULT_EFL), roll
    return float(_DEFAULT_SEC_PER_PIX_DTY_17 * _FREQ_RATIO_17_TO_33P8), float(_DEFAULT_EFL), roll


def _parse_obs_freq_ghz(obs_freq: Any) -> float | None:
    if obs_freq is None:
        return None
    text = str(obs_freq).strip().lower().replace(" ", "")
    if text.endswith("ghz"):
        text = text[:-3]
    try:
        value = float(text)
    except Exception:
        return None
    if not np.isfinite(value) or value <= 0.0:
        return None
    return float(value)


def is_norh_header(header: fits.Header) -> bool:
    """True when the FITS header looks like a Nobeyama Radioheliograph image."""
    origin = str(header.get("ORIGIN") or "").lower()
    telescop = str(header.get("TELESCOP") or "").lower()
    instrume = str(header.get("INSTRUME") or "").lower()
    progname = str(header.get("PROGNAME") or "").lower()
    obs_freq = header.get("OBS-FREQ", header.get("OBSFREQ"))
    has_pmat = all(k in header for k in ("PMAT1", "PMAT2", "PMAT3", "PMAT4"))
    if not has_pmat or _parse_obs_freq_ghz(obs_freq) is None:
        return False
    if "nobeyama" in origin or "radioheliograph" in telescop or "norh" in telescop or "norh" in instrume:
        return True
    if progname.startswith("snap2d"):
        return True
    return False


def _header_float(header: fits.Header, *keys: str, default: float | None = None) -> float | None:
    for key in keys:
        if key not in header:
            continue
        try:
            value = float(header[key])
        except Exception:
            continue
        if np.isfinite(value):
            return float(value)
    return default


def norh_index_from_header(header: fits.Header) -> dict[str, Any]:
    """Minimal NORH index fields needed by ``norh_beam`` / ``BeamFitNoRH``."""
    obs_freq = header.get("OBS-FREQ", header.get("OBSFREQ"))
    progname = str(header.get("PROGNAME") or "").strip()
    cdelt1 = _header_float(header, "CDELT1", "cdelt1")
    if cdelt1 is None or cdelt1 == 0.0:
        raise ValueError("NORH header requires CDELT1 (sec_per_pix)")
    solp_deg = _header_float(header, "SOLP", "solp", default=0.0)
    assert solp_deg is not None
    pmat = (
        float(_header_float(header, "PMAT1", default=1.0) or 1.0),
        float(_header_float(header, "PMAT2", default=0.0) or 0.0),
        float(_header_float(header, "PMAT3", default=0.0) or 0.0),
        float(_header_float(header, "PMAT4", default=1.0) or 1.0),
    )
    cellsize = _header_float(header, "CELLSIZE", "cellsize")
    sec_dty, efl, roll = norh_prog2pinf(progname, str(obs_freq) if obs_freq is not None else None, cellsize=cellsize)
    freq_ghz = _parse_obs_freq_ghz(obs_freq)
    if freq_ghz is None:
        raise ValueError("NORH header requires OBS-FREQ")
    return {
        "obs_freq": str(obs_freq).strip(),
        "frequency_ghz": float(freq_ghz),
        "progname": progname,
        "sec_per_pix": float(abs(cdelt1)),
        "sec_per_pix_dty": float(sec_dty),
        "efl_in_pix_dty": float(efl),
        "solp_rad": float(np.deg2rad(solp_deg)),
        "pmat": pmat,
        "roll": float(roll),
        "ns2ew": float(_NS2EW),
    }


def make_norh_beam(
    *,
    sec_per_pix: float,
    sec_per_pix_dty: float,
    efl_in_pix_dty: float,
    solp_rad: float,
    pmat: tuple[float, float, float, float],
    marx: int = 21,
    ns2ew: float = _NS2EW,
) -> np.ndarray:
    """IDL ``norh_beam`` (marx×marx, peak=1). Layout matches IDL ``beam[i,j]`` (x,y)."""
    marx = int(marx)
    if marx < 3:
        raise ValueError("marx must be >= 3")
    rbeam = float(efl_in_pix_dty)
    if rbeam <= 0.0:
        raise ValueError("efl_in_pix_dty must be positive")
    ratio_c2d = float(sec_per_pix) / float(sec_per_pix_dty)
    delta = 1.0 / (rbeam * rbeam)
    half = int(marx) // 2
    idx = np.arange(marx, dtype=float) - float(half)
    # IDL: ii[i,j]=i-half, jj[i,j]=j-half  → beam[i,j]
    ii = idx[:, None] + np.zeros((1, marx), dtype=float)
    jj = np.zeros((marx, 1), dtype=float) + idx[None, :]
    cos_p = np.cos(float(solp_rad))
    sin_p = np.sin(float(solp_rad))
    x = (cos_p * ii - sin_p * jj) * ratio_c2d
    y = (sin_p * ii + cos_p * jj) * ratio_c2d
    p0, p1, p2, p3 = (float(v) for v in pmat)
    xx = p0 * x + p2 * y
    yy = (p1 * x + p3 * y) * float(ns2ew)
    return np.exp(-delta * (xx * xx + yy * yy))


def _gauss2drot_flat(u: np.ndarray, sx: float, sy: float, theta: float) -> np.ndarray:
    """IDL ``Gauss2Drot`` on concatenated 1D x,y (IDL beam layout)."""
    n = u.size // 2
    x = u[:n]
    y = u[n:]
    xx = x[:, None] + np.zeros((1, n), dtype=float)
    yy = np.zeros((n, 1), dtype=float) + y[None, :]
    ct = np.cos(theta)
    st = np.sin(theta)
    xr = xx * ct + yy * st
    yr = -xx * st + yy * ct
    return np.exp(-(xr * xr) / (2.0 * sx * sx) - (yr * yr) / (2.0 * sy * sy)).ravel(order="C")


def fit_norh_beam_ellipse(
    beam: np.ndarray,
    *,
    sec_per_pix: float,
) -> tuple[float, float, float]:
    """IDL ``FitBeam`` → ``(sx, sy, theta_rad)`` with σ in arcsec."""
    arr = np.asarray(beam, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("beam must be a square 2D array")
    n = int(arr.shape[0])
    half = n // 2
    coords_1d = (np.arange(n, dtype=float) - float(half)) * float(sec_per_pix)
    u = np.concatenate([coords_1d, coords_1d])
    # IDL FitBeam stores beam as (N,N) with first index = x; ravel matches Gauss2Drot reform.
    z = arr.ravel(order="C")
    span = float(np.hypot(coords_1d.max() - coords_1d.min(), coords_1d.max() - coords_1d.min()))
    s0 = max(span / 4.0, abs(float(sec_per_pix)))
    p0 = (s0, s0, 0.0)
    bounds = ([1e-6, 1e-6, -np.pi], [np.inf, np.inf, np.pi])
    popt, _ = curve_fit(_gauss2drot_flat, u, z, p0=p0, bounds=bounds, maxfev=5000)
    sx, sy, theta = (float(v) for v in popt)
    theta = float(np.arctan2(np.sin(theta), np.cos(theta)))
    if theta > 0.5 * np.pi:
        theta -= np.pi
    if theta < -0.5 * np.pi:
        theta += np.pi
    return sx, sy, theta


def beam_fit_norh_from_header(
    header: fits.Header,
    *,
    marx: int = 51,
    solp_rad: float | None = None,
) -> NorhBeamParams:
    """``BeamFitNoRH`` equivalent from a NORH FITS header (no image data required)."""
    index = norh_index_from_header(header)
    solp = float(index["solp_rad"] if solp_rad is None else solp_rad)
    beam = make_norh_beam(
        sec_per_pix=float(index["sec_per_pix"]),
        sec_per_pix_dty=float(index["sec_per_pix_dty"]),
        efl_in_pix_dty=float(index["efl_in_pix_dty"]),
        solp_rad=solp,
        pmat=tuple(index["pmat"]),  # type: ignore[arg-type]
        marx=int(marx),
        ns2ew=float(index["ns2ew"]),
    )
    sx, sy, theta = fit_norh_beam_ellipse(beam, sec_per_pix=float(index["sec_per_pix"]))
    return NorhBeamParams(
        sigma_x_arcsec=float(sx),
        sigma_y_arcsec=float(sy),
        theta_rad=float(theta),
        frequency_ghz=float(index["frequency_ghz"]),
        sec_per_pix_arcsec=float(index["sec_per_pix"]),
        efl_in_pix_dty=float(index["efl_in_pix_dty"]),
        sec_per_pix_dty=float(index["sec_per_pix_dty"]),
        solp_rad=solp,
        pmat=tuple(index["pmat"]),  # type: ignore[arg-type]
        marx=int(marx),
    )
