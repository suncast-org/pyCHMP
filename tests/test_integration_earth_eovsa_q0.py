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
    reason="Set PYCHMP_RUN_GXRENDER_INTEGRATION=1 to run gxrender integration test",
)
def test_q0_recovery_earth_observer_eovsa_psf() -> None:
    if not MODEL_PATH.exists() or not EBTEL_PATH.exists():
        pytest.skip("Real fixture files are not available on this machine")

    sdk = pytest.importorskip("gxrender.sdk")

    geometry = sdk.MapGeometry(
        xc=-257.0,
        yc=-233.0,
        dx=2.5,
        dy=2.5,
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
        bmaj_arcsec=5.77,
        bmin_arcsec=5.77,
        bpa_deg=-17.5,
        dx_arcsec=2.5,
        dy_arcsec=2.5,
    )
    renderer = _PSFConvolvedRenderer(base_renderer, kernel)

    q0_true = 0.0217
    observed = renderer.render(q0_true)

    # Use a floor to keep sigma positive and stable in low-intensity pixels.
    sigma = np.maximum(0.05 * np.max(observed), 1.0) * np.ones_like(observed)

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

    assert result.success
    assert result.q0 == pytest.approx(q0_true, abs=3e-3)
