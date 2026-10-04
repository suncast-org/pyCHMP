"""Siberian Solar Radio Telescope (SSRT) clean-beam helpers.

In-tree port of gx_simulator SSRT beam routines:

* ``GetSSRTangles`` — time → EW/NS half-widths (arcsec) and scan angles (deg)
* ``MakeSSRTbeam`` — 2D restoring beam from those angles
* ``BeamFitSSRT`` / ``FitSSRTbeam`` — rotated Gaussian fit (σ axes + PA)

No SSW/IDL runtime dependency for the Python path. Ephemeris uses Astropy /
SunPy and is intended to track IDL closely; for bit-level beam parity tests,
pass the IDL ``dEW/dNS/pEW/pNS`` into ``make_ssrt_beam`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
from astropy.coordinates import EarthLocation, get_sun
from astropy.time import Time
from astropy import units as u
from scipy.optimize import curve_fit
from sunpy.coordinates import sun as sunpy_sun

# Badary observatory (matches GetSSRTangles.pro)
_BADARY_LON_DEG = 102.0 + 13.0 / 60.0
_BADARY_LAT_RAD = 0.903338787600965
_LAMBDA_ND = 0.052 / (127.0 * 4.9)
_W_RAD_PER_SEC = 15.0 * np.pi / 180.0 / 3600.0
_DF_SSRT_HZ = 250000.0 * 2.0
_F0_SSRT_HZ = 5.73e9
_SSRT_OBS_FREQ_GHZ = 5.7


@dataclass(frozen=True)
class SsrtAngles:
    d_ew_arcsec: float
    d_ns_arcsec: float
    p_ew_deg: float
    p_ns_deg: float
    noon_sec: float
    p_angle_deg: float
    declination_deg: float


@dataclass(frozen=True)
class SsrtBeamParams:
    """Clean-beam geometry at ~5.7 GHz (Gaussian σ, arcsec)."""

    sigma_x_arcsec: float
    sigma_y_arcsec: float
    theta_rad: float
    frequency_ghz: float
    angles: SsrtAngles
    marx: int
    dx_arcsec: float
    dy_arcsec: float

    @property
    def theta_deg(self) -> float:
        return float(np.rad2deg(self.theta_rad))


def make_ssrt_beam(
    d_ew: float,
    d_ns: float,
    p_ew_deg: float,
    p_ns_deg: float,
    *,
    nx: int,
    ny: int,
    dx_arcsec: float,
    dy_arcsec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IDL ``MakeSSRTbeam`` → ``(x, y, beam)`` with IDL ``beam[i,j]`` layout.

    ``d_ew``/``d_ns`` are half-widths at half-maximum (arcsec). ``p_*`` are
    scan angles in degrees, counterclockwise from +Y.
    """
    nx = int(nx)
    ny = int(ny)
    x = (np.arange(nx, dtype=float) - 0.5 * nx + 0.5) * float(dx_arcsec)
    y = (np.arange(ny, dtype=float) - 0.5 * ny + 0.5) * float(dy_arcsec)
    c_ew = np.cos(np.deg2rad(float(p_ew_deg)))
    s_ew = np.sin(np.deg2rad(float(p_ew_deg)))
    c_ns = np.cos(np.deg2rad(float(p_ns_deg)))
    s_ns = np.sin(np.deg2rad(float(p_ns_deg)))
    a_ew = float(d_ew) ** 2 / np.log(2.0) / 4.0
    a_ns = float(d_ns) ** 2 / np.log(2.0) / 4.0
    xx = x[:, None]
    yy = y[None, :]
    r_ew = np.abs(xx * c_ew + yy * s_ew)
    r_ns = np.abs(xx * c_ns + yy * s_ns)
    beam = np.exp(-(r_ew * r_ew) / a_ew - (r_ns * r_ns) / a_ns)
    return x, y, beam


def _parse_time(time: Any) -> Time:
    if isinstance(time, Time):
        return time
    if isinstance(time, datetime):
        return Time(time)
    text = str(time).strip()
    # IDL atime / map.time often looks like " 2-Oct-2012 03:16:23.917"
    for fmt in (
        "%d-%b-%Y %H:%M:%S.%f",
        "%d-%b-%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return Time(datetime.strptime(text, fmt).replace(tzinfo=timezone.utc))
        except ValueError:
            continue
    return Time(text)


def get_ssrt_angles(time: Any) -> SsrtAngles:
    """IDL ``GetSSRTangles`` using Astropy/SunPy ephemeris (Badary site)."""
    t = _parse_time(time)
    # Declination / P-angle: SunPy tracks SSW get_sun closely for SSRT use.
    decl_deg = float(get_sun(t).dec.deg)
    p_angle_deg = float(sunpy_sun.P(t).to(u.deg).value)
    d = np.deg2rad(decl_deg)

    # Local solar geometry around Badary noon (matches GetSSRTangles.pro).
    loc = EarthLocation(lon=_BADARY_LON_DEG * u.deg, lat=np.rad2deg(_BADARY_LAT_RAD) * u.deg)
    sun = get_sun(t)
    ra_hours = float(sun.ra.hour)
    lst_hours = float(t.sidereal_time("apparent", longitude=loc.lon).hour)
    # UTC hour + fractional minute like IDL anytim(/external) path
    utc = t.utc.datetime
    hour = utc.hour
    minute = utc.minute + (utc.second + utc.microsecond * 1e-6) / 60.0
    h = float(hour) + float(minute) / 60.0
    noon = h + (ra_hours - lst_hours) * 0.99726958
    if noon < 0.0:
        noon += 24.0
    if noon >= 24.0:
        noon -= 24.0
    noon_sec = noon * 3600.0

    n = 1000
    t_grid = np.linspace(0.0, 10.0 * 3600.0, n)
    h_arr = _W_RAD_PER_SEC * (t_grid - noon_sec)
    phi = _BADARY_LAT_RAD

    # EW / NS scanning angles on the hour-angle grid (IDL gU / gQ branch).
    base_ew = np.array([0.0, 1.0, 0.0])
    base_phi = np.array(
        [
            -np.sin(phi) * base_ew[0] + np.cos(phi) * base_ew[2],
            base_ew[1],
            np.cos(phi) * base_ew[0] + np.sin(phi) * base_ew[2],
        ]
    )
    uu = np.sin(h_arr) * base_phi[0] + np.cos(h_arr) * base_phi[1]
    vv = (
        -np.sin(d) * np.cos(h_arr) * base_phi[0]
        + np.sin(d) * np.sin(h_arr) * base_phi[1]
        + np.cos(d) * base_phi[2]
    )
    g_u = np.pi - np.arctan2(uu, vv)

    g_q = np.arctan(-(np.sin(d) / np.tan(h_arr) + np.cos(d) / (np.sin(h_arr) * np.tan(phi)))) + np.pi / 2.0
    g_q = np.array(g_q, dtype=float, copy=True)
    g_q[h_arr > 0.0] = np.pi + g_q[h_arr > 0.0]

    given_t = hour * 3600.0 + minute * 60.0
    if given_t > 10.0 * 3600.0 - 360.0:
        given_t = noon_sec
    t0_ind = np.where(t_grid < given_t - 180.0)[0]
    t1_ind = np.where(t_grid > given_t + 180.0)[0]
    if t0_ind.size == 0 or t1_ind.size == 0:
        t_i = n // 2
    else:
        t_i = int((t0_ind[-1] + t1_ind[0]) // 2)

    ew_angle = -(g_u[t_i] * 180.0 / np.pi - 90.0)
    ns_angle = -(g_q[t_i] * 180.0 / np.pi - 90.0)
    if given_t > noon_sec:
        ns_angle = 180.0 + ns_angle

    cos_h = np.cos(_W_RAD_PER_SEC * (given_t - noon_sec))
    sin_h = np.sin(_W_RAD_PER_SEC * (given_t - noon_sec))
    cos_p = sin_h * np.cos(d)
    cos_q = cos_h * np.cos(d) * np.sin(phi) - np.sin(d) * np.cos(phi)
    sin_p = np.sqrt(max(1.0 - cos_p * cos_p, 0.0))
    sin_q = np.sqrt(max(1.0 - cos_q * cos_q, 0.0))

    beam_ew = _LAMBDA_ND / sin_p * (180.0 / np.pi) * 3600.0
    beam_ns = _LAMBDA_ND / sin_q * (180.0 / np.pi) * 3600.0
    return SsrtAngles(
        d_ew_arcsec=float(beam_ew),
        d_ns_arcsec=float(beam_ns),
        p_ew_deg=float(ew_angle - p_angle_deg),
        p_ns_deg=float(ns_angle - p_angle_deg),
        noon_sec=float(noon_sec),
        p_angle_deg=float(p_angle_deg),
        declination_deg=float(decl_deg),
    )


def _gauss2drot_flat(u: np.ndarray, sx: float, sy: float, theta: float) -> np.ndarray:
    n = u.size // 2
    x = u[:n]
    y = u[n:]
    xx = x[:, None] + np.zeros((1, n), dtype=float)
    yy = np.zeros((n, 1), dtype=float) + y[None, :]
    ct = np.cos(theta)
    st = np.sin(theta)
    xr = xx * ct + yy * st
    yr = -xx * st + yy * ct
    return np.exp(-(xr * xr) / (2.0 * sx * sx) - (yr * yr) / (2.0 * sy * sy)).ravel()


def fit_ssrt_beam_ellipse(
    beam: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float]:
    """IDL ``FitBeam`` on an SSRT beam → ``(sx, sy, theta_rad)``."""
    arr = np.asarray(beam, dtype=float)
    n = int(arr.shape[0])
    if arr.shape != (n, n):
        raise ValueError("beam must be square")
    u = np.concatenate([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    z = arr.ravel(order="C")
    span = float(np.hypot(np.ptp(x), np.ptp(y)))
    s0 = max(span / 4.0, 1e-3)
    popt, _ = curve_fit(
        _gauss2drot_flat,
        u,
        z,
        p0=(s0, s0, 0.0),
        bounds=([1e-6, 1e-6, -np.pi], [np.inf, np.inf, np.pi]),
        maxfev=5000,
    )
    sx, sy, theta = (float(v) for v in popt)
    theta = float(np.arctan2(np.sin(theta), np.cos(theta)))
    if theta > 0.5 * np.pi:
        theta -= np.pi
    if theta < -0.5 * np.pi:
        theta += np.pi
    return sx, sy, theta


def beam_fit_ssrt(
    time: Any,
    *,
    marx: int = 50,
    dx_arcsec: float = 1.0,
    dy_arcsec: float | None = None,
    angles: SsrtAngles | None = None,
) -> tuple[SsrtBeamParams, np.ndarray]:
    """IDL ``BeamFitSSRT`` → ``(params, beam)``."""
    dy = float(dx_arcsec if dy_arcsec is None else dy_arcsec)
    ang = angles if angles is not None else get_ssrt_angles(time)
    x, y, beam = make_ssrt_beam(
        ang.d_ew_arcsec,
        ang.d_ns_arcsec,
        ang.p_ew_deg,
        ang.p_ns_deg,
        nx=int(marx),
        ny=int(marx),
        dx_arcsec=float(dx_arcsec),
        dy_arcsec=dy,
    )
    sx, sy, theta = fit_ssrt_beam_ellipse(beam, x, y)
    params = SsrtBeamParams(
        sigma_x_arcsec=float(sx),
        sigma_y_arcsec=float(sy),
        theta_rad=float(theta),
        frequency_ghz=float(_SSRT_OBS_FREQ_GHZ),
        angles=ang,
        marx=int(marx),
        dx_arcsec=float(dx_arcsec),
        dy_arcsec=dy,
    )
    return params, beam
