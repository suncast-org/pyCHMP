from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

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


def _save_artifacts(
    out_dir: Path,
    *,
    observed_clean: np.ndarray,
    observed_noisy: np.ndarray,
    modeled_best: np.ndarray,
    residual: np.ndarray,
    diagnostics: dict[str, Any],
    save_png: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "earth_eovsa_q0_artifacts.npz",
        observed_clean=observed_clean,
        observed_noisy=observed_noisy,
        modeled_best=modeled_best,
        residual=residual,
        diagnostics=diagnostics,
    )

    if not save_png:
        return

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    panels = [
        ("Observed (Noisy)", observed_noisy),
        ("Modeled (Best Q0)", modeled_best),
        ("Residual (Model-Obs)", residual),
    ]
    for ax, (title, data) in zip(axes, panels):
        im = ax.imshow(data, origin="lower", cmap="inferno")
        ax.set_title(title)
        ax.set_xlabel("X [pix]")
        ax.set_ylabel("Y [pix]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_dir / "earth_eovsa_q0_artifacts.png", dpi=140)
    plt.close(fig)


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
    p.add_argument("--noise-frac", type=float, default=0.02)
    p.add_argument("--noise-seed", type=int, default=12345)
    p.add_argument("--save-raw-h5", default=None)
    p.add_argument("--artifacts-dir", default=None)
    p.add_argument("--no-artifacts-png", action="store_true")
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

    observed_clean = renderer.render(args.q0_true)
    noise_std = max(0.0, float(args.noise_frac)) * float(np.max(observed_clean))
    rng = np.random.default_rng(args.noise_seed)
    noise = rng.normal(loc=0.0, scale=noise_std, size=observed_clean.shape)
    observed = np.clip(observed_clean + noise, a_min=0.0, a_max=None)
    sigma = max(noise_std, 1.0) * np.ones_like(observed)

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
    modeled_best = renderer.render(result.q0)
    residual = modeled_best - observed

    print(f"truth q0: {args.q0_true:.6f}")
    print(f"fit q0:   {result.q0:.6f}")
    print(f"chi2:     {result.metrics.chi2:.6e}")
    print(f"noise:    gaussian frac={args.noise_frac:.4f} seed={args.noise_seed} std={noise_std:.6e}")
    print(f"success:  {result.success}")

    if args.artifacts_dir:
        out_dir = Path(args.artifacts_dir)
        _save_artifacts(
            out_dir,
            observed_clean=observed_clean,
            observed_noisy=observed,
            modeled_best=modeled_best,
            residual=residual,
            diagnostics={
                "q0_truth": float(args.q0_true),
                "q0_recovered": float(result.q0),
                "q0_abs_error": float(abs(result.q0 - args.q0_true)),
                "chi2": float(result.metrics.chi2),
                "rho2": float(result.metrics.rho2),
                "eta2": float(result.metrics.eta2),
                "noise_frac": float(args.noise_frac),
                "noise_seed": int(args.noise_seed),
                "noise_std": float(noise_std),
                "psf_bmaj_arcsec": 5.77,
                "psf_bmin_arcsec": 5.77,
                "psf_bpa_deg": -17.5,
            },
            save_png=not args.no_artifacts_png,
        )
        print(f"artifacts dir: {out_dir}")
        print(f"data file:      {out_dir / 'earth_eovsa_q0_artifacts.npz'}")
        print(f"png file:       {out_dir / 'earth_eovsa_q0_artifacts.png'}")

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
