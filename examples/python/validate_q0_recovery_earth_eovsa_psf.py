from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from pychmp import GXRenderMWAdapter, fit_q0_to_observation


@dataclass(frozen=True)
class ValidationProfile:
    noise_frac: float
    q0_abs_tol: float
    chi2_max: float


DEFAULT_PROFILES: tuple[ValidationProfile, ...] = (
    ValidationProfile(noise_frac=0.02, q0_abs_tol=3e-3, chi2_max=2.0),
    ValidationProfile(noise_frac=0.05, q0_abs_tol=6e-3, chi2_max=5.0),
)

DEFAULT_Q0_TRUE = 0.0217
DEFAULT_PSF_BMAJ_ARCSEC = 5.77
DEFAULT_PSF_BMIN_ARCSEC = 5.77
DEFAULT_PSF_BPA_DEG = -17.5
DEFAULT_MAP_XC_ARCSEC = -257.0
DEFAULT_MAP_YC_ARCSEC = -233.0
DEFAULT_MAP_DX_ARCSEC = 2.5
DEFAULT_MAP_DY_ARCSEC = 2.5
DEFAULT_MAP_NX = 64
DEFAULT_MAP_NY = 64
DEFAULT_PSF_KERNEL_SIZE = 41
DEFAULT_NOISE_SEED = 12345


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate q0 recovery against a real gxrender model using an Earth-observer "
            "geometry, a PSF convolution, and synthetic noise."
        )
    )
    parser.add_argument("--model-path", type=Path, required=True, help="path to the real GX model file")
    parser.add_argument("--ebtel-path", type=Path, required=True, help="path to the matching EBTEL file")
    parser.add_argument("--frequency-ghz", type=float, default=17.0, help="render frequency in GHz")
    parser.add_argument("--tbase", type=float, default=1e6, help="base temperature")
    parser.add_argument("--nbase", type=float, default=1e8, help="base density")
    parser.add_argument("--a", type=float, default=0.3, help="heating slope a")
    parser.add_argument("--b", type=float, default=2.7, help="heating slope b")
    parser.add_argument("--q0-true", type=float, default=DEFAULT_Q0_TRUE, help="true q0 used to generate the synthetic observation")
    parser.add_argument("--q0-min", type=float, default=0.005, help="minimum q0 for recovery")
    parser.add_argument("--q0-max", type=float, default=0.05, help="maximum q0 for recovery")
    parser.add_argument("--threshold", type=float, default=0.1, help="metric threshold used by the fitter")
    parser.add_argument("--xatol", type=float, default=1e-3, help="absolute q0 tolerance for the fitter")
    parser.add_argument("--maxiter", type=int, default=60, help="maximum fitter iterations")
    parser.add_argument("--map-xc-arcsec", type=float, default=DEFAULT_MAP_XC_ARCSEC, help="map center x in arcsec")
    parser.add_argument("--map-yc-arcsec", type=float, default=DEFAULT_MAP_YC_ARCSEC, help="map center y in arcsec")
    parser.add_argument("--map-dx-arcsec", type=float, default=DEFAULT_MAP_DX_ARCSEC, help="map pixel scale x in arcsec")
    parser.add_argument("--map-dy-arcsec", type=float, default=DEFAULT_MAP_DY_ARCSEC, help="map pixel scale y in arcsec")
    parser.add_argument("--map-nx", type=int, default=DEFAULT_MAP_NX, help="map width in pixels")
    parser.add_argument("--map-ny", type=int, default=DEFAULT_MAP_NY, help="map height in pixels")
    parser.add_argument("--psf-bmaj-arcsec", type=float, default=DEFAULT_PSF_BMAJ_ARCSEC, help="PSF major axis FWHM in arcsec")
    parser.add_argument("--psf-bmin-arcsec", type=float, default=DEFAULT_PSF_BMIN_ARCSEC, help="PSF minor axis FWHM in arcsec")
    parser.add_argument("--psf-bpa-deg", type=float, default=DEFAULT_PSF_BPA_DEG, help="PSF position angle in degrees")
    parser.add_argument("--psf-kernel-size", type=int, default=DEFAULT_PSF_KERNEL_SIZE, help="odd PSF kernel size in pixels")
    parser.add_argument(
        "--noise-frac",
        type=float,
        action="append",
        help="noise fraction relative to the clean map peak; repeat to run multiple profiles",
    )
    parser.add_argument(
        "--q0-abs-tol",
        type=float,
        action="append",
        help="allowed absolute q0 error for the corresponding noise profile; repeat to match --noise-frac",
    )
    parser.add_argument(
        "--chi2-max",
        type=float,
        action="append",
        help="allowed chi2 ceiling for the corresponding noise profile; repeat to match --noise-frac",
    )
    parser.add_argument("--noise-seed", type=int, default=DEFAULT_NOISE_SEED, help="seed used for synthetic noise")
    return parser.parse_args()


def _validation_profiles(args: argparse.Namespace) -> tuple[ValidationProfile, ...]:
    if args.noise_frac is None and args.q0_abs_tol is None and args.chi2_max is None:
        return DEFAULT_PROFILES

    if args.noise_frac is None or args.q0_abs_tol is None or args.chi2_max is None:
        raise SystemExit("--noise-frac, --q0-abs-tol, and --chi2-max must be provided together")
    if not (len(args.noise_frac) == len(args.q0_abs_tol) == len(args.chi2_max)):
        raise SystemExit("--noise-frac, --q0-abs-tol, and --chi2-max must have the same number of values")
    return tuple(
        ValidationProfile(noise_frac=noise_frac, q0_abs_tol=q0_abs_tol, chi2_max=chi2_max)
        for noise_frac, q0_abs_tol, chi2_max in zip(args.noise_frac, args.q0_abs_tol, args.chi2_max, strict=True)
    )


def _elliptical_gaussian_kernel(
    bmaj_arcsec: float,
    bmin_arcsec: float,
    bpa_deg: float,
    dx_arcsec: float,
    dy_arcsec: float,
    size: int,
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


class _PSFConvolvedRenderer:
    def __init__(self, base_renderer: GXRenderMWAdapter, kernel: np.ndarray) -> None:
        self._base = base_renderer
        self._kernel = kernel

    def render(self, q0: float) -> np.ndarray:
        raw = self._base.render(q0)
        return fftconvolve(raw, self._kernel, mode="same")


def _load_sdk() -> object:
    try:
        return import_module("gxrender.sdk")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "gxrender.sdk is not importable. Install gxrender in the active environment before running this example."
        ) from exc


def _build_renderer(args: argparse.Namespace, *, kernel: np.ndarray) -> _PSFConvolvedRenderer:
    sdk = _load_sdk()
    geometry = sdk.MapGeometry(
        xc=float(args.map_xc_arcsec),
        yc=float(args.map_yc_arcsec),
        dx=float(args.map_dx_arcsec),
        dy=float(args.map_dy_arcsec),
        nx=int(args.map_nx),
        ny=int(args.map_ny),
    )
    base_renderer = GXRenderMWAdapter(
        model_path=args.model_path,
        ebtel_path=str(args.ebtel_path),
        frequency_ghz=float(args.frequency_ghz),
        tbase=float(args.tbase),
        nbase=float(args.nbase),
        a=float(args.a),
        b=float(args.b),
        geometry=geometry,
    )
    return _PSFConvolvedRenderer(base_renderer, kernel)


def _validate_inputs(args: argparse.Namespace) -> None:
    if not args.model_path.exists():
        raise SystemExit(f"Model file does not exist: {args.model_path}")
    if not args.ebtel_path.exists():
        raise SystemExit(f"EBTEL file does not exist: {args.ebtel_path}")
    if args.psf_kernel_size < 3 or args.psf_kernel_size % 2 == 0:
        raise SystemExit("--psf-kernel-size must be an odd integer >= 3")


def _run_profile(
    renderer: _PSFConvolvedRenderer,
    profile: ValidationProfile,
    *,
    q0_true: float,
    q0_min: float,
    q0_max: float,
    threshold: float,
    xatol: float,
    maxiter: int,
    noise_seed: int,
) -> tuple[bool, str]:
    observed_clean = renderer.render(q0_true)
    noise_std = profile.noise_frac * float(np.max(observed_clean))
    rng = np.random.default_rng(noise_seed)
    noise = rng.normal(loc=0.0, scale=noise_std, size=observed_clean.shape)
    observed = np.clip(observed_clean + noise, a_min=0.0, a_max=None)

    sigma_level = max(noise_std, 1.0)
    sigma = sigma_level * np.ones_like(observed)

    result = fit_q0_to_observation(
        renderer,
        observed,
        sigma,
        q0_min=q0_min,
        q0_max=q0_max,
        threshold=threshold,
        target_metric="chi2",
        xatol=xatol,
        maxiter=maxiter,
    )
    q0_abs_err = abs(result.q0 - q0_true)
    ok = bool(result.success and q0_abs_err <= profile.q0_abs_tol and result.metrics.chi2 <= profile.chi2_max)
    summary = (
        f"noise_frac={profile.noise_frac:.3f} q0={result.q0:.6f} true_q0={q0_true:.6f} "
        f"abs_err={q0_abs_err:.6e} chi2={result.metrics.chi2:.6e} status={'PASS' if ok else 'FAIL'}"
    )
    if not result.success:
        summary += f" optimizer_message={result.message!r}"
    return ok, summary


def main() -> int:
    args = _parse_args()
    _validate_inputs(args)
    profiles = _validation_profiles(args)
    kernel = _elliptical_gaussian_kernel(
        bmaj_arcsec=float(args.psf_bmaj_arcsec),
        bmin_arcsec=float(args.psf_bmin_arcsec),
        bpa_deg=float(args.psf_bpa_deg),
        dx_arcsec=float(args.map_dx_arcsec),
        dy_arcsec=float(args.map_dy_arcsec),
        size=int(args.psf_kernel_size),
    )
    renderer = _build_renderer(args, kernel=kernel)

    any_failed = False
    for index, profile in enumerate(profiles, start=1):
        ok, summary = _run_profile(
            renderer,
            profile,
            q0_true=float(args.q0_true),
            q0_min=float(args.q0_min),
            q0_max=float(args.q0_max),
            threshold=float(args.threshold),
            xatol=float(args.xatol),
            maxiter=int(args.maxiter),
            noise_seed=int(args.noise_seed) + index - 1,
        )
        print(summary)
        any_failed = any_failed or not ok

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())