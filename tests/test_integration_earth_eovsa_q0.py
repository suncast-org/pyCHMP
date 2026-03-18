from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from astropy.io import fits
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
            data=np.asarray(["OBSERVED", "MODELED", "RESIDUAL", "OBSERVED", "MODELED", "RESIDUAL"], dtype="S32"),
        )
        maps.create_dataset("artifact_labels", data=np.asarray(["OBSERVED", "MODELED", "RESIDUAL"], dtype="S32"))

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
        dx_arcsec=MAP_DX_ARCSEC,
        dy_arcsec=MAP_DY_ARCSEC,
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
    if not MODEL_PATH.exists() or not EBTEL_PATH.exists():
        pytest.skip("Real fixture files are not available on this machine")

    sdk = pytest.importorskip("gxrender.sdk")

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

    modeled_best = renderer.render(result.q0)
    residual = modeled_best - observed

    q0_abs_err = abs(result.q0 - Q0_TRUE)

    if os.environ.get("PYCHMP_VERBOSE_INTEGRATION") == "1":
        print("integration diagnostics:")
        print(f"  model_path:      {MODEL_PATH}")
        print(f"  frequency_ghz:   17.0")
        print(f"  map_shape:       {observed.shape}")
        print("  psf_applied:     yes")
        print(f"  psf_bmaj_arcsec: {PSF_BMAJ_ARCSEC:.2f}")
        print(f"  psf_bmin_arcsec: {PSF_BMIN_ARCSEC:.2f}")
        print(f"  psf_bpa_deg:     {PSF_BPA_DEG:.1f}")
        print(f"  psf_kernel_size: {PSF_KERNEL_SIZE}x{PSF_KERNEL_SIZE}")
        print(f"  map_dx_arcsec:   {MAP_DX_ARCSEC:.2f}")
        print(f"  map_dy_arcsec:   {MAP_DY_ARCSEC:.2f}")
        print("  noise_applied:   yes")
        print(f"  noise_model:     gaussian")
        print(f"  noise_frac:      {noise_frac:.4f}")
        print(f"  noise_seed:      {NOISE_SEED}")
        print(f"  noise_std:       {noise_std:.6e}")
        print(f"  q0_truth:        {Q0_TRUE:.6f}")
        print(f"  q0_recovered:    {result.q0:.6f}")
        print(f"  q0_abs_error:    {q0_abs_err:.6e}")
        print(f"  q0_abs_tol:      {q0_abs_tol:.6e}")
        print(f"  chi2:            {result.metrics.chi2:.6e}")
        print(f"  rho2:            {result.metrics.rho2:.6e}")
        print(f"  eta2:            {result.metrics.eta2:.6e}")
        print(f"  chi2_max:        {chi2_max:.6e}")
        print(f"  optimizer_nit:   {result.nit}")
        print(f"  optimizer_nfev:  {result.nfev}")
        print(f"  optimizer_ok:    {result.success}")

    if os.environ.get("PYCHMP_SAVE_INTEGRATION_ARTIFACTS") == "1":
        out_dir = Path(os.environ.get("PYCHMP_ARTIFACTS_DIR", "/tmp/pychmp_artifacts"))
        save_png = os.environ.get("PYCHMP_SAVE_INTEGRATION_PNG", "1") == "1"
        _save_artifacts(
            out_dir,
            observed_clean=observed_clean,
            observed_noisy=observed,
            modeled_best=modeled_best,
            residual=residual,
            diagnostics={
                "q0_truth": Q0_TRUE,
                "q0_recovered": float(result.q0),
                "q0_abs_error": float(q0_abs_err),
                "q0_abs_tol": float(q0_abs_tol),
                "chi2": float(result.metrics.chi2),
                "rho2": float(result.metrics.rho2),
                "eta2": float(result.metrics.eta2),
                "chi2_max": float(chi2_max),
                "noise_frac": float(noise_frac),
                "noise_seed": int(NOISE_SEED),
                "noise_std": float(noise_std),
                "psf_bmaj_arcsec": float(PSF_BMAJ_ARCSEC),
                "psf_bmin_arcsec": float(PSF_BMIN_ARCSEC),
                "psf_bpa_deg": float(PSF_BPA_DEG),
                "psf_kernel_size": int(PSF_KERNEL_SIZE),
                "model_path": str(MODEL_PATH),
            },
            save_png=save_png,
        )
        if os.environ.get("PYCHMP_VERBOSE_INTEGRATION") == "1":
            print(f"  artifacts_saved: yes")
            print(f"  artifacts_dir:   {out_dir}")

    assert result.success, f"optimizer failed: {result.message}"
    assert q0_abs_err <= q0_abs_tol, (
        f"q0 recovery error too large: |{result.q0:.6f} - {Q0_TRUE:.6f}| = {q0_abs_err:.6e} > {q0_abs_tol:.6e}"
    )
    assert result.metrics.chi2 <= chi2_max, (
        f"chi2 too large: {result.metrics.chi2:.6e} > {chi2_max:.6e}"
    )
