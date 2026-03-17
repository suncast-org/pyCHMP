from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from pychmp import GXRenderMWAdapter, fit_q0_to_observation


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
    def __init__(self, base_renderer: GXRenderMWAdapter, kernel: np.ndarray) -> None:
        self._base = base_renderer
        self._kernel = kernel

    def render(self, q0: float) -> np.ndarray:
        raw = self._base.render(q0)
        return fftconvolve(raw, self._kernel, mode="same")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Q0 recovery demo in Earth frame with EOVSA-like PSF")
    p.add_argument(
        "--model-path",
        default=(
            "/Users/gelu/Library/CloudStorage/Dropbox/@Projects/@SUNCAST-ORG/pyGXrender-test-data/raw/models/"
            "models_20251126T153431/test.chr.sav"
        ),
    )
    p.add_argument("--ebtel-path", required=True)
    p.add_argument("--q0-true", type=float, default=0.0217)
    p.add_argument("--q0-min", type=float, default=0.005)
    p.add_argument("--q0-max", type=float, default=0.05)
    p.add_argument("--save-raw-h5", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from gxrender import sdk
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("gxrender is required for this demo") from exc

    output_dir = None
    output_name = None
    if args.save_raw_h5:
        output_path = Path(args.save_raw_h5)
        output_dir = str(output_path.parent)
        output_name = output_path.stem if output_path.suffix == ".h5" else output_path.name

    geometry = sdk.MapGeometry(
        xc=-257.0,
        yc=-233.0,
        dx=2.5,
        dy=2.5,
        nx=64,
        ny=64,
    )
    base_renderer = GXRenderMWAdapter(
        model_path=args.model_path,
        ebtel_path=args.ebtel_path,
        frequency_ghz=17.0,
        tbase=1e6,
        nbase=1e8,
        a=0.3,
        b=2.7,
        geometry=geometry,
        output_dir=output_dir,
        output_name=output_name,
        output_format="h5",
    )

    kernel = _elliptical_gaussian_kernel(
        bmaj_arcsec=5.77,
        bmin_arcsec=5.77,
        bpa_deg=-17.5,
        dx_arcsec=2.5,
        dy_arcsec=2.5,
    )
    renderer = PSFConvolvedRenderer(base_renderer, kernel)

    observed = renderer.render(args.q0_true)
    sigma = np.maximum(0.05 * np.max(observed), 1.0) * np.ones_like(observed)

    result = fit_q0_to_observation(
        renderer,
        observed,
        sigma,
        q0_min=args.q0_min,
        q0_max=args.q0_max,
        threshold=0.1,
        target_metric="chi2",
        xatol=1e-3,
        maxiter=60,
    )

    print(f"truth q0: {args.q0_true:.6f}")
    print(f"fit q0:   {result.q0:.6f}")
    print(f"chi2:     {result.metrics.chi2:.6e}")
    print(f"success:  {result.success}")

    if args.save_raw_h5:
        target = Path(args.save_raw_h5)
        if target.suffix == ".h5":
            print(f"raw rendered map saved to: {target}")
            print(f"view with: gxrender-map-view {target}")
        else:
            print(f"raw rendered map saved to: {target.with_suffix('.h5')}")
            print(f"view with: gxrender-map-view {target.with_suffix('.h5')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
