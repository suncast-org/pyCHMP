from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import fftconvolve

from pychmp import GXRenderMWAdapter, fit_q0_to_observation


MODEL_PATH = Path(
    "/Users/gelu/Library/CloudStorage/Dropbox/@Projects/@SUNCAST-ORG/pyGXrender-test-data/raw/models/models_20251126T153431/test.chr.sav"
)
EBTEL_PATH = Path(
    "/Users/gelu/Library/CloudStorage/Dropbox/@Projects/@SUNCAST-ORG/pyGXrender-test-data/raw/ebtel/ebtel_gxsimulator_euv/ebtel.sav"
)

Q0_TRUE = 0.0217
PSF_BMAJ_ARCSEC = 5.77
PSF_BMIN_ARCSEC = 5.77
PSF_BPA_DEG = -17.5
MAP_DX_ARCSEC = 2.5
MAP_DY_ARCSEC = 2.5
PSF_KERNEL_SIZE = 41
NOISE_SEED = 12345


def _elliptical_gaussian_kernel(
    bmaj_arcsec: float,
    bmin_arcsec: float,
    bpa_deg: float,
    dx_arcsec: float,
    dy_arcsec: float,
    size: int = 41,
) -> np.ndarray:
    # Convert FWHM beam to sigma in pixels.
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


@pytest.mark.skipif(
    os.environ.get("PYCHMP_RUN_GXRENDER_INTEGRATION") != "1",
    reason="Set PYCHMP_RUN_GXRENDER_INTEGRATION=1 to run gxrender integration smoke test",
)
@pytest.mark.parametrize(
    ("noise_frac", "q0_abs_tol", "chi2_max"),
    [
        (0.02, 3e-3, 2.0),
        (0.05, 6e-3, 5.0),
    ],
)
def test_q0_recovery_earth_observer_eovsa_psf(
    noise_frac: float,
    q0_abs_tol: float,
    chi2_max: float,
) -> None:
    """Smoke integration: real gxrender render + PSF + noise + Q0 recovery asserts."""
    if not MODEL_PATH.exists() or not EBTEL_PATH.exists():
        pytest.skip("Real fixture files are not available on this machine")

    sdk = pytest.importorskip("gxrender.sdk", reason="gxrender SDK is required for integration smoke")

    geometry = sdk.MapGeometry(
        xc=-257.0,
        yc=-233.0,
        dx=MAP_DX_ARCSEC,
        dy=MAP_DY_ARCSEC,
        nx=64,
        ny=64,
    )
    base_renderer = GXRenderMWAdapter(
        model_path=MODEL_PATH,
        ebtel_path=str(EBTEL_PATH),
        frequency_ghz=17.0,
        tbase=1e6,
        nbase=1e8,
        a=0.3,
        b=2.7,
        geometry=geometry,
    )

    kernel = _elliptical_gaussian_kernel(
        bmaj_arcsec=PSF_BMAJ_ARCSEC,
        bmin_arcsec=PSF_BMIN_ARCSEC,
        bpa_deg=PSF_BPA_DEG,
        dx_arcsec=MAP_DX_ARCSEC,
        dy_arcsec=MAP_DY_ARCSEC,
        size=PSF_KERNEL_SIZE,
    )
    renderer = _PSFConvolvedRenderer(base_renderer, kernel)

    observed_clean = renderer.render(Q0_TRUE)
    noise_std = noise_frac * float(np.max(observed_clean))
    rng = np.random.default_rng(NOISE_SEED)
    noise = rng.normal(loc=0.0, scale=noise_std, size=observed_clean.shape)
    observed = np.clip(observed_clean + noise, a_min=0.0, a_max=None)

    # Use known noise scale with a floor to keep sigma positive and stable.
    sigma_level = max(noise_std, 1.0)
    sigma = sigma_level * np.ones_like(observed)

    result = fit_q0_to_observation(
        renderer,
        observed,
        sigma,
        q0_min=0.005,
        q0_max=0.05,
        threshold=0.1,
        target_metric="chi2",
        xatol=1e-3,
        maxiter=60,
    )

    q0_abs_err = abs(result.q0 - Q0_TRUE)

    assert result.success, f"optimizer failed: {result.message}"
    assert q0_abs_err <= q0_abs_tol, (
        f"q0 recovery error too large: |{result.q0:.6f} - {Q0_TRUE:.6f}| = {q0_abs_err:.6e} > {q0_abs_tol:.6e}"
    )
    assert result.metrics.chi2 <= chi2_max, (
        f"chi2 too large: {result.metrics.chi2:.6e} > {chi2_max:.6e}"
    )
