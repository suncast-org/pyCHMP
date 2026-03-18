from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from astropy.io import fits
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


def _build_common_wcs_header(
    ny: int,
    nx: int,
    *,
    xc_arcsec: float,
    yc_arcsec: float,
    dx_arcsec: float,
    dy_arcsec: float,
    date_obs: str,
    bunit: str,
) -> fits.Header:
    return fits.Header(
        {
            "NAXIS": 2,
            "NAXIS1": int(nx),
            "NAXIS2": int(ny),
            "CTYPE1": "HPLN-TAN",
            "CTYPE2": "HPLT-TAN",
            "CUNIT1": "arcsec",
            "CUNIT2": "arcsec",
            "CDELT1": float(dx_arcsec),
            "CDELT2": float(dy_arcsec),
            "CRPIX1": (nx + 1.0) / 2.0,
            "CRPIX2": (ny + 1.0) / 2.0,
            "CRVAL1": float(xc_arcsec),
            "CRVAL2": float(yc_arcsec),
            "DATE-OBS": str(date_obs or "2025-01-01T00:00:00"),
            "BUNIT": str(bunit),
            "OBSERVER": "Earth",
            "HGLN_OBS": 0.0,
            "HGLT_OBS": 0.0,
            "DSUN_OBS": 1.495978707e11,
            "RSUN_REF": 6.957e8,
        }
    )


def _decode_h5_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        return value
    return str(value)


def _load_blos_reference_map(model_path: Path) -> tuple[np.ndarray, fits.Header] | None:
    if model_path.suffix.lower() not in {".h5", ".hdf5"}:
        return None
    if not model_path.exists():
        return None

    candidates = [
        ("refmaps", "Bz_reference"),
        ("reference_maps", "B_los"),
        ("reference_maps", "Bz_reference"),
    ]
    try:
        with h5py.File(model_path, "r") as f:
            for root, key in candidates:
                path = f"{root}/{key}"
                if path not in f:
                    continue
                grp = f[path]
                if "data" not in grp or "wcs_header" not in grp:
                    continue
                data = np.asarray(grp["data"], dtype=float)
                wcs_text = _decode_h5_scalar(grp["wcs_header"][()])
                header = fits.Header.fromstring(wcs_text, sep="\n")
                return data, header
    except Exception:
        return None
    return None


def _save_viewer_h5(
    out_h5: Path,
    *,
    observed_noisy: np.ndarray,
    modeled_best: np.ndarray,
    residual: np.ndarray,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
) -> None:
    out_h5.parent.mkdir(parents=True, exist_ok=True)

    ti = np.stack([observed_noisy, modeled_best, residual], axis=-1).astype(np.float32)  # (ny, nx, 3)
    tv = np.zeros_like(ti, dtype=np.float32)
    cube = np.stack([np.transpose(ti, (1, 0, 2)), np.transpose(tv, (1, 0, 2))], axis=-1)  # (nx, ny, 3, 2)

    header_text = wcs_header.tostring(sep="\n", endcard=True)
    with h5py.File(out_h5, "w") as f:
        maps = f.create_group("maps")
        maps.create_dataset("data", data=cube, compression="gzip", compression_opts=4)
        maps.create_dataset("freqlist_ghz", data=np.asarray([17.0, 17.1, 17.2], dtype=np.float64))
        maps.create_dataset("stokes_ids", data=np.asarray(["TI", "TV"], dtype="S8"))
        maps.create_dataset(
            "map_ids",
            data=np.asarray(["Observed", "Modeled", "Residual", "Observed", "Modeled", "Residual"], dtype="S32"),
        )
        maps.create_dataset("artifact_labels", data=np.asarray(["Observed", "Modeled", "Residual"], dtype="S32"))

        meta = f.create_group("metadata")
        meta.create_dataset("wcs_header", data=np.bytes_(header_text))
        meta.create_dataset("index_header", data=np.bytes_(header_text))
        meta.create_dataset("date_obs", data=np.bytes_(str(wcs_header.get("DATE-OBS", ""))))
        meta.create_dataset("observer_name", data=np.bytes_("Earth"))
        meta.create_dataset("artifact_kind", data=np.bytes_("pychmp_q0_recovery"))
        meta.create_dataset("diagnostics_json", data=np.bytes_(json.dumps(diagnostics, sort_keys=True)))


def _save_png_panel(
    out_png: Path,
    *,
    model_path: Path,
    observed_noisy: np.ndarray,
    modeled_best: np.ndarray,
    residual: np.ndarray,
    wcs_header: fits.Header,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import sunpy.map
        from astropy import units as u
        from astropy.coordinates import SkyCoord
    except ModuleNotFoundError:
        return

    m_obs = sunpy.map.Map(observed_noisy, wcs_header)
    m_mod = sunpy.map.Map(modeled_best, wcs_header)
    m_res = sunpy.map.Map(residual, wcs_header)

    blos = _load_blos_reference_map(model_path)

    fig = plt.figure(figsize=(12.8, 9.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    if blos is not None:
        blos_data, blos_hdr = blos
        m_blos = sunpy.map.Map(blos_data, blos_hdr)
        ax0 = fig.add_subplot(gs[0, 0], projection=m_blos)
        try:
            xc = float(wcs_header.get("CRVAL1", 0.0))
            yc = float(wcs_header.get("CRVAL2", 0.0))
            nx = int(wcs_header.get("NAXIS1", observed_noisy.shape[1]))
            ny = int(wcs_header.get("NAXIS2", observed_noisy.shape[0]))
            dx = float(wcs_header.get("CDELT1", 1.0))
            dy = float(wcs_header.get("CDELT2", 1.0))
            half_x = 0.5 * nx * abs(dx)
            half_y = 0.5 * ny * abs(dy)
            bl = SkyCoord((xc - half_x) * u.arcsec, (yc - half_y) * u.arcsec, frame=m_blos.coordinate_frame)
            tr = SkyCoord((xc + half_x) * u.arcsec, (yc + half_y) * u.arcsec, frame=m_blos.coordinate_frame)
            m_blos = m_blos.submap(bl, top_right=tr)
            ax0.remove()
            ax0 = fig.add_subplot(gs[0, 0], projection=m_blos)
        except Exception:
            pass
        im0 = m_blos.plot(axes=ax0, cmap="gray")
        ax0.set_title("B_los Reference")
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    else:
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.set_title("B_los Reference (Unavailable)")
        ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1], projection=m_obs)
    im1 = m_obs.plot(axes=ax1, cmap="inferno")
    ax1.set_title("Observed (Noisy)")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(gs[1, 0], projection=m_mod)
    im2 = m_mod.plot(axes=ax2, cmap="inferno")
    ax2.set_title("Modeled (Best Q0)")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(gs[1, 1], projection=m_res)
    im3 = m_res.plot(axes=ax3, cmap="coolwarm")
    ax3.set_title("Residual (Model-Obs)")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    fig.savefig(out_png, dpi=140, facecolor="white")
    plt.close(fig)


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
    ny, nx = observed_noisy.shape
    wcs_header = _build_common_wcs_header(
        ny,
        nx,
        xc_arcsec=-257.0,
        yc_arcsec=-233.0,
        dx_arcsec=2.5,
        dy_arcsec=2.5,
        date_obs="",
        bunit="K",
    )

    out_h5 = out_dir / "earth_eovsa_q0_artifacts.h5"
    _save_viewer_h5(
        out_h5,
        observed_noisy=observed_noisy,
        modeled_best=modeled_best,
        residual=residual,
        wcs_header=wcs_header,
        diagnostics=diagnostics,
    )

    if save_png:
        _save_png_panel(
            out_dir / "earth_eovsa_q0_artifacts.png",
            model_path=Path(str(diagnostics.get("model_path", ""))),
            observed_noisy=observed_noisy,
            modeled_best=modeled_best,
            residual=residual,
            wcs_header=wcs_header,
        )


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
                "model_path": str(args.model_path),
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
        print(f"data file:      {out_dir / 'earth_eovsa_q0_artifacts.h5'}")
        print(f"view with:      gxrender-map-view {out_dir / 'earth_eovsa_q0_artifacts.h5'}")
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
