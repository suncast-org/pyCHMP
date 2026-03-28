from __future__ import annotations

import argparse
import itertools
import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from astropy.io import fits
from scipy.signal import fftconvolve

from pychmp import GXRenderMWAdapter
from pychmp.metrics import MetricValues, compute_metrics, threshold_union_mask
from pychmp.optimize import MetricName, Q0MetricEvaluation, find_best_q0
from q0_artifact_plot import plot_q0_artifact_panel


METRIC_CHOICES: tuple[MetricName, ...] = ("chi2", "rho2", "eta2")


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

    def render_pair(self, q0: float) -> tuple[np.ndarray, np.ndarray]:
        raw = self._base.render(q0)
        convolved = fftconvolve(raw, self._kernel, mode="same")
        return raw, convolved

    def render(self, q0: float) -> np.ndarray:
        _raw, convolved = self.render_pair(q0)
        return convolved


def _clear_terminal_status_line(width: int = 120) -> None:
    print("\r" + (" " * width) + "\r", end="", flush=True)


def _lookup_cached_render_pair(
    render_cache: dict[float, tuple[np.ndarray, np.ndarray]],
    q0: float,
    *,
    atol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray] | None:
    q0_value = float(q0)
    cached = render_cache.get(q0_value)
    if cached is not None:
        return cached
    for cached_q0, cached_pair in render_cache.items():
        if np.isclose(cached_q0, q0_value, rtol=0.0, atol=atol):
            return cached_pair
    return None


def _metric_value(metrics: MetricValues, name: MetricName) -> float:
    return float(getattr(metrics, name))


def _ordered_metric_names(target_metric: MetricName) -> tuple[MetricName, ...]:
    return (target_metric,) + tuple(name for name in METRIC_CHOICES if name != target_metric)


def _format_metric_report(metrics: MetricValues, target_metric: MetricName) -> str:
    parts: list[str] = []
    for name in _ordered_metric_names(target_metric):
        value = _metric_value(metrics, name)
        label = f"target[{name}]" if name == target_metric else name
        parts.append(f"{label}={value:.6e}")
    return " ".join(parts)

def _stage_label(name: str, *, stage_index: int | None, stage_total: int | None) -> str:
    if stage_index is not None and stage_total is not None and stage_total > 0:
        return f"[stage {stage_index}/{stage_total}] {name}"
    return f"[stage] {name}"


def _run_stage(
    name: str,
    fn,
    *,
    spinner: bool,
    stage_index: int | None = None,
    stage_total: int | None = None,
    running_label: str | None = None,
) -> tuple[Any, float]:
    label = _stage_label(name, stage_index=stage_index, stage_total=stage_total)
    running_text = running_label if running_label is not None else f"{label}: running"
    print(f"{label}: started")
    start = time.perf_counter()
    stop = threading.Event()
    thread: threading.Thread | None = None

    if spinner:
        def _spin() -> None:
            for ch in itertools.cycle("|/-\\"):
                if stop.wait(0.12):
                    break
                print(f"\r{running_text} {ch}", end="", flush=True)

        thread = threading.Thread(target=_spin, daemon=True)
        thread.start()

    try:
        result = fn()
    finally:
        if thread is not None:
            stop.set()
            thread.join(timeout=1.0)
            _clear_terminal_status_line()

    elapsed = time.perf_counter() - start
    print(f"{label}: done in {elapsed:.3f}s")
    return result, elapsed


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
    observer_name: str = "observer",
    hgln_obs_deg: float = 0.0,
    hglt_obs_deg: float = 0.0,
    dsun_obs_m: float = 1.495978707e11,
) -> fits.Header:
    header = fits.Header(
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
            "RSUN_REF": 6.957e8,
        }
    )
    return _with_observer_wcs_keywords(
        header,
        observer_name=observer_name,
        hgln_obs_deg=hgln_obs_deg,
        hglt_obs_deg=hglt_obs_deg,
        dsun_obs_m=dsun_obs_m,
    )


def _with_observer_wcs_keywords(
    header: fits.Header,
    *,
    observer_name: str,
    hgln_obs_deg: float,
    hglt_obs_deg: float,
    dsun_obs_m: float,
) -> fits.Header:
    out = header.copy()
    lon = float(hgln_obs_deg)
    lat = float(hglt_obs_deg)
    dsun = float(dsun_obs_m)

    out["OBSERVER"] = str(observer_name)

    # Populate both hyphen and underscore variants so SunPy metadata parsing
    # for Stonyhurst/Carrington frames always finds observer keys.
    out["DSUN_OBS"] = dsun
    out["HGLN_OBS"] = lon
    out["HGLT_OBS"] = lat
    out["CRLN_OBS"] = lon
    out["CRLT_OBS"] = lat
    out["HGLN-OBS"] = lon
    out["HGLT-OBS"] = lat
    out["CRLN-OBS"] = lon
    out["CRLT-OBS"] = lat
    return out


def _decode_h5_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        return value
    return str(value)


def _load_model_obs_time_iso(model_path: Path) -> str | None:
    try:
        with h5py.File(model_path, "r") as f:
            for key in ("observer/pb0r/obs_date", "metadata/date_obs", "metadata/date-obs"):
                if key in f:
                    txt = _decode_h5_scalar(np.asarray(f[key]).reshape(-1)[0])
                    txt = txt.strip()
                    if txt:
                        return txt
    except Exception:
        return None
    return None


def _resolve_observer_overrides(
    sdk: Any,
    *,
    model_path: Path,
    observer_name: str | None,
    dsun_cm: float | None,
    lonc_deg: float | None,
    b0sun_deg: float | None,
) -> tuple[Any | None, str]:
    lonc = lonc_deg
    b0 = b0sun_deg
    dsun = dsun_cm
    source = "saved_observer_metadata"

    if observer_name:
        try:
            from astropy.time import Time
            from gxrender.geometry import normalize_observer_name
            from gxrender.geometry.observer_geometry import _observer_from_sunpy

            model_time = Time(_load_model_obs_time_iso(model_path) or "2025-01-01T00:00:00")
            norm_name = normalize_observer_name(observer_name) or observer_name
            l0, b0_resolved, dsun_resolved = _observer_from_sunpy(str(norm_name), model_time)
            lonc = float(l0) if lonc is None else lonc
            b0 = float(b0_resolved) if b0 is None else b0
            dsun = float(dsun_resolved) if dsun is None else dsun
            source = f"observer_name:{norm_name}"
        except Exception as exc:
            raise ValueError(f"unable to resolve --observer '{observer_name}': {exc}") from exc

    if lonc is None and b0 is None and dsun is None:
        return None, source

    return sdk.ObserverOverrides(dsun_cm=dsun, lonc_deg=lonc, b0sun_deg=b0), source


def _ensure_refmap_h5_for_model(model_path: Path) -> Path | None:
    suffix = model_path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        return model_path if model_path.exists() else None
    if suffix != ".sav" or not model_path.exists():
        return None

    try:
        from gxrender.io import build_h5_from_sav
    except Exception:
        return None

    src = model_path.resolve()
    stat = src.stat()
    stamp = f"{int(stat.st_mtime)}_{int(stat.st_size)}"
    out_h5 = Path(tempfile.gettempdir()) / f"pychmp_refmaps_{src.stem}_{stamp}.h5"
    if out_h5.exists():
        return out_h5

    try:
        build_h5_from_sav(src, out_h5, template_h5=None)
        return out_h5
    except Exception:
        return None


def _load_saved_fov_from_model(model_path: Path) -> dict[str, float] | None:
    def _normalize_fov_dict(data: dict[str, Any]) -> dict[str, float] | None:
        try:
            xc = float(data.get("xc_arcsec", data.get("xc")))
            yc = float(data.get("yc_arcsec", data.get("yc")))
            xsize = float(data.get("xsize_arcsec", data.get("xsize")))
            ysize = float(data.get("ysize_arcsec", data.get("ysize")))
        except Exception:
            return None
        if not (np.isfinite(xc) and np.isfinite(yc) and np.isfinite(xsize) and np.isfinite(ysize)):
            return None
        if xsize <= 0 or ysize <= 0:
            return None

        out = {
            "xc_arcsec": xc,
            "yc_arcsec": yc,
            "xsize_arcsec": xsize,
            "ysize_arcsec": ysize,
        }
        try:
            dx = float(data.get("dx_arcsec", data.get("dx")))
            if np.isfinite(dx) and dx > 0:
                out["dx_arcsec"] = dx
        except Exception:
            pass
        try:
            dy = float(data.get("dy_arcsec", data.get("dy")))
            if np.isfinite(dy) and dy > 0:
                out["dy_arcsec"] = dy
        except Exception:
            pass
        return out

    try:
        with h5py.File(model_path, "r") as f:
            fov = None
            for candidate in ("fov", "observer/fov", "metadata/fov", "meta/fov"):
                if candidate in f:
                    fov = f[candidate]
                    break
            if fov is None:
                return None
            required = ("xc_arcsec", "yc_arcsec", "xsize_arcsec", "ysize_arcsec")
            if any(key not in fov for key in required):
                # Try metadata fallback below for non-canonical group layouts.
                raise KeyError("non-canonical fov layout")

            xc = float(np.asarray(fov["xc_arcsec"]).reshape(-1)[0])
            yc = float(np.asarray(fov["yc_arcsec"]).reshape(-1)[0])
            xsize = float(np.asarray(fov["xsize_arcsec"]).reshape(-1)[0])
            ysize = float(np.asarray(fov["ysize_arcsec"]).reshape(-1)[0])
            if not (np.isfinite(xc) and np.isfinite(yc) and np.isfinite(xsize) and np.isfinite(ysize)):
                return None
            if xsize <= 0 or ysize <= 0:
                return None

            dx = None
            dy = None
            if "grid" in f and "dx" in f["grid"]:
                dx_val = float(np.asarray(f["grid"]["dx"]).reshape(-1)[0])
                if np.isfinite(dx_val) and dx_val > 0:
                    dx = dx_val
            if "grid" in f and "dy" in f["grid"]:
                dy_val = float(np.asarray(f["grid"]["dy"]).reshape(-1)[0])
                if np.isfinite(dy_val) and dy_val > 0:
                    dy = dy_val

            out = {
                "xc_arcsec": xc,
                "yc_arcsec": yc,
                "xsize_arcsec": xsize,
                "ysize_arcsec": ysize,
            }
            if dx is not None:
                out["dx_arcsec"] = dx
            if dy is not None:
                out["dy_arcsec"] = dy
            return out
    except Exception:
        pass

    # Fallback: let gxrender's model I/O resolve observer metadata and pull fov from there.
    try:
        from gxrender.io.model import load_model_hdf_with_observer, load_model_sav_with_observer

        suffix = model_path.suffix.lower()
        if suffix in {".h5", ".hdf5"}:
            _m, _dt, _meta, observer_metadata = load_model_hdf_with_observer(str(model_path))
        elif suffix == ".sav":
            _m, _dt, _meta, observer_metadata = load_model_sav_with_observer(str(model_path))
        else:
            return None

        if isinstance(observer_metadata, dict):
            fov_meta = observer_metadata.get("fov")
            if isinstance(fov_meta, dict):
                return _normalize_fov_dict(fov_meta)
    except Exception:
        return None

    return None


def _load_blos_reference_map(model_path: Path) -> tuple[np.ndarray, fits.Header] | None:
    refmap_h5 = _ensure_refmap_h5_for_model(model_path)
    if refmap_h5 is None:
        return None

    candidates = [
        ("refmaps", "Bz_reference"),
        ("reference_maps", "B_los"),
        ("reference_maps", "Bz_reference"),
    ]
    try:
        with h5py.File(refmap_h5, "r") as f:
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
    raw_modeled_best: np.ndarray,
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
    freqlist = diagnostics.get("freqlist_ghz_used")
    if freqlist is None:
        freqlist = [float(diagnostics.get("mw_frequency_ghz", 17.0))]
    freqlist_arr = np.asarray(freqlist, dtype=np.float64).reshape(-1)
    observer_name = str(diagnostics.get("observer_name_effective", "observer"))
    with h5py.File(out_h5, "w") as f:
        maps = f.create_group("maps")
        maps.create_dataset("data", data=cube, compression="gzip", compression_opts=4)
        maps.create_dataset("freqlist_ghz", data=freqlist_arr)
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
        meta.create_dataset("observer_name", data=np.bytes_(observer_name))
        meta.create_dataset("artifact_kind", data=np.bytes_("pychmp_q0_recovery"))
        meta.create_dataset("diagnostics_json", data=np.bytes_(json.dumps(diagnostics, sort_keys=True)))

        analysis = f.create_group("analysis")
        analysis.create_dataset("raw_modeled_best_ti", data=np.asarray(raw_modeled_best, dtype=np.float32), compression="gzip", compression_opts=4)
        q0_trials = diagnostics.get("fit_q0_trials")
        metric_trials = diagnostics.get("fit_metric_trials")
        target_metric = str(diagnostics.get("target_metric", "chi2"))
        if q0_trials is not None and metric_trials is not None:
            try:
                q0_arr = np.asarray(q0_trials, dtype=np.float64)
                metric_arr = np.asarray(metric_trials, dtype=np.float64)
                if q0_arr.ndim == 1 and metric_arr.ndim == 1 and q0_arr.size == metric_arr.size and q0_arr.size > 0:
                    analysis.create_dataset("fit_q0_trials", data=q0_arr)
                    fit_metric_ds = analysis.create_dataset("fit_metric_trials", data=metric_arr)
                    fit_metric_ds.attrs["target_metric"] = np.bytes_(target_metric)
            except Exception:
                pass


def _save_png_panel(
    out_png: Path,
    *,
    model_path: Path,
    observed_noisy: np.ndarray,
    raw_modeled_best: np.ndarray,
    modeled_best: np.ndarray,
    residual: np.ndarray,
    wcs_header: fits.Header,
    frequency_ghz: float = 17.0,
    diagnostics: dict[str, Any] | None = None,
    log_metrics: bool = False,
    log_q0: bool = False,
    zoom2best: int | None = None,
    defer_show: bool = False,
    show_plot: bool = False,
) -> None:
    diag = diagnostics or {}
    observer_name = str(diag.get("observer_name_effective", wcs_header.get("OBSERVER", "observer")))
    observer_lonc = float(
        diag.get(
            "observer_lonc_deg",
            wcs_header.get("CRLN-OBS", wcs_header.get("CRLN_OBS", wcs_header.get("HGLN_OBS", 0.0))),
        )
    )
    observer_b0 = float(
        diag.get(
            "observer_b0sun_deg",
            wcs_header.get("CRLT-OBS", wcs_header.get("CRLT_OBS", wcs_header.get("HGLT_OBS", 0.0))),
        )
    )
    observer_dsun_m = float(diag.get("observer_dsun_cm", float(wcs_header.get("DSUN_OBS", 1.495978707e11) * 100.0))) / 100.0

    def _apply_observer_keywords(header: fits.Header) -> fits.Header:
        return _with_observer_wcs_keywords(
            header,
            observer_name=observer_name,
            hgln_obs_deg=observer_lonc,
            hglt_obs_deg=observer_b0,
            dsun_obs_m=observer_dsun_m,
        )

    plot_q0_artifact_panel(
        out_png,
        model_path=model_path,
        observed_noisy=observed_noisy,
        raw_modeled_best=raw_modeled_best,
        modeled_best=modeled_best,
        residual=residual,
        wcs_header=wcs_header,
        frequency_ghz=frequency_ghz,
        diagnostics=diag,
        log_metrics=log_metrics,
        log_q0=log_q0,
        zoom2best=zoom2best,
        show_plot=show_plot,
        defer_show=defer_show,
        wcs_header_transform=_apply_observer_keywords,
    )


def _save_artifacts(
    out_dir: Path,
    *,
    stem: str = "q0_recovery_artifacts",
    observed_clean: np.ndarray,
    observed_noisy: np.ndarray,
    raw_modeled_best: np.ndarray,
    modeled_best: np.ndarray,
    residual: np.ndarray,
    diagnostics: dict[str, Any],
    save_png: bool,
    show_plot: bool = False,
    defer_show: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ny, nx = observed_noisy.shape
    xc_arcsec = float(diagnostics.get("map_xc_arcsec", -257.0))
    yc_arcsec = float(diagnostics.get("map_yc_arcsec", -233.0))
    dx_arcsec = float(diagnostics.get("map_dx_arcsec", 2.5))
    dy_arcsec = float(diagnostics.get("map_dy_arcsec", 2.5))
    observer_name = str(diagnostics.get("observer_name_effective", "observer"))
    hgln_obs = float(diagnostics.get("observer_lonc_deg", 0.0))
    hglt_obs = float(diagnostics.get("observer_b0sun_deg", 0.0))
    dsun_obs_m = float(diagnostics.get("observer_dsun_cm", 1.495978707e13)) / 100.0
    wcs_header = _build_common_wcs_header(
        ny,
        nx,
        xc_arcsec=xc_arcsec,
        yc_arcsec=yc_arcsec,
        dx_arcsec=dx_arcsec,
        dy_arcsec=dy_arcsec,
        date_obs=str(diagnostics.get("observer_obs_time", "")),
        bunit="K",
        observer_name=observer_name,
        hgln_obs_deg=hgln_obs,
        hglt_obs_deg=hglt_obs,
        dsun_obs_m=dsun_obs_m,
    )

    out_h5 = out_dir / f"{stem}.h5"
    _save_viewer_h5(
        out_h5,
        observed_noisy=observed_noisy,
        raw_modeled_best=raw_modeled_best,
        modeled_best=modeled_best,
        residual=residual,
        wcs_header=wcs_header,
        diagnostics=diagnostics,
    )

    if save_png:
        _save_png_panel(
            out_dir / f"{stem}.png",
            show_plot=show_plot,
            defer_show=defer_show,
            model_path=Path(str(diagnostics.get("model_path", ""))),
            observed_noisy=observed_noisy,
            raw_modeled_best=raw_modeled_best,
            modeled_best=modeled_best,
            residual=residual,
            wcs_header=wcs_header,
            frequency_ghz=float(diagnostics.get("active_frequency_ghz", diagnostics.get("mw_frequency_ghz", 17.0))),
            diagnostics=diagnostics,
            log_metrics=bool(diagnostics.get("log_metrics", False)),
            log_q0=bool(diagnostics.get("log_q0", False)),
            zoom2best=int(diagnostics.get("zoom2best", 0)) or None,
        )


def parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    p = argparse.ArgumentParser(
        description="Q0 recovery demo with configurable observer, single frequency, and PSF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model-path",
        default=None,
        help="Path to coronal model file (.h5 preferred; .sav accepted and auto-converted to .h5).",
    )
    p.add_argument(
        "--ebtel-path",
        default="",
        help=(
            "Path to the ebtel.sav file. Use an empty string to disable DEM/DDM tables "
            "and use isothermal fallback mode."
        ),
    )
    p.add_argument("--q0-true", type=float, default=0.0217, help="Injected true Q0 value.")
    p.add_argument("--q0-min", type=float, default=0.005, help="Lower bound of Q0 search interval.")
    p.add_argument("--q0-max", type=float, default=0.05, help="Upper bound of Q0 search interval.")
    p.add_argument("--adaptive-bracketing", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable adaptive multiplicative Q0 bracketing before bounded refinement.")
    p.add_argument("--q0-start", type=float, default=None,
                   help="Starting Q0 for adaptive bracketing (default: geometric mean of q0_min and q0_max).")
    p.add_argument("--q0-step", type=float, default=1.61803398875,
                   help="Multiplicative step factor for adaptive bracketing.")
    p.add_argument("--max-bracket-steps", type=int, default=12,
                   help="Maximum number of adaptive bracketing expansion steps.")
    p.add_argument("--noise-frac", type=float, default=0.02, help="Gaussian noise level as fraction of peak.")
    p.add_argument("--noise-seed", type=int, default=12345, help="RNG seed for reproducible noise.")
    p.add_argument(
        "--target-metric",
        choices=METRIC_CHOICES,
        default="chi2",
        help="Metric minimized during Q0 fitting.",
    )
    p.add_argument("--frequency-ghz", type=float, default=17.0, help="Single MW frequency used for fitting/rendering.")
    p.add_argument("--observer", default=None, help="Optional observer name (e.g., earth, stereo-a, stereo-b).")
    p.add_argument("--dsun-cm", type=float, default=None, help="Observer-Sun distance override in cm.")
    p.add_argument("--lonc-deg", type=float, default=None, help="Observer heliographic Carrington longitude override in deg.")
    p.add_argument("--b0sun-deg", type=float, default=None, help="Observer heliographic latitude override in deg.")
    p.add_argument("--xc", type=float, default=None, help="Optional map center X in arcsec (exact override).")
    p.add_argument("--yc", type=float, default=None, help="Optional map center Y in arcsec (exact override).")
    p.add_argument("--dx", type=float, default=None, help="Optional map pixel scale X in arcsec/pixel.")
    p.add_argument("--dy", type=float, default=None, help="Optional map pixel scale Y in arcsec/pixel.")
    p.add_argument("--nx", type=int, default=None, help="Optional map width in pixels.")
    p.add_argument("--ny", type=int, default=None, help="Optional map height in pixels.")
    p.add_argument(
        "--pixel-scale-arcsec",
        type=float,
        default=2.0,
        help="Pixel scale used only when no explicit geometry overrides are provided.",
    )
    p.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print iterative optimizer progress and timing diagnostics.",
    )
    p.add_argument(
        "--spinner",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show a rotating spinner while long stages are running.",
    )
    p.add_argument(
        "--log-metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use logarithmic y-axis for the metric-vs-q0 trials panel in PNG artifacts.",
    )
    p.add_argument(
        "--log-q0",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use logarithmic x-axis (q0 axis) for the metric-vs-q0 trials panel in PNG artifacts.",
    )
    p.add_argument(
        "--zoom2best",
        type=int,
        default=None,
        metavar="N",
        help="Restrict trials panel x/y axes to ±N trials (sorted by q0) around the best-metric trial.",
    )
    p.add_argument("--psf-bmaj-arcsec", type=float, default=5.77, help="PSF major axis FWHM at --psf-ref-frequency-ghz.")
    p.add_argument("--psf-bmin-arcsec", type=float, default=5.77, help="PSF minor axis FWHM at --psf-ref-frequency-ghz.")
    p.add_argument("--psf-bpa-deg", type=float, default=-17.5, help="PSF position angle in degrees.")
    p.add_argument("--psf-ref-frequency-ghz", type=float, default=17.0, help="Reference frequency for PSF axes values.")
    p.add_argument(
        "--psf-scale-inverse-frequency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale PSF axes by (ref_freq / active_freq).",
    )
    p.add_argument("--save-raw-h5", default=None, help="If set, write raw rendered map to this .h5 path.")
    p.add_argument("--artifacts-dir", default=None, help="Directory to write H5 viewer + PNG artifacts.")
    p.add_argument("--artifacts-stem", default="q0_recovery_artifacts",
                   help="Base filename stem (no extension) for artifact files inside --artifacts-dir.")
    p.add_argument("--no-artifacts-png", action="store_true", help="Skip PNG panel output even if --artifacts-dir is set.")
    p.add_argument("--show-plot", action="store_true", help="Display the PNG panel interactively after saving (calls plt.show()).")
    p.add_argument("--defaults", action="store_true", help="Print assumed defaults and exit.")
    return p, p.parse_args()


def main() -> int:
    import sys
    t_global = time.perf_counter()
    parser, args = parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    if args.defaults:
        defaults = {
            "cli": {
                "model_path": None,
                "ebtel_path": "",
                "q0_true": 0.0217,
                "q0_min": 0.005,
                "q0_max": 0.05,
                "adaptive_bracketing": False,
                "q0_start": None,
                "q0_step": 1.61803398875,
                "max_bracket_steps": 12,
                "noise_frac": 0.02,
                "noise_seed": 12345,
                "target_metric": "chi2",
                "frequency_ghz": 17.0,
                "observer": None,
                "dsun_cm": None,
                "lonc_deg": None,
                "b0sun_deg": None,
                "xc": None,
                "yc": None,
                "dx": None,
                "dy": None,
                "nx": None,
                "ny": None,
                "pixel_scale_arcsec": 2.0,
                "progress": True,
                "spinner": True,
                "log_metrics": False,
                "log_q0": False,
                "zoom2best": 0,
                "psf_bmaj_arcsec": 5.77,
                "psf_bmin_arcsec": 5.77,
                "psf_bpa_deg": -17.5,
                "psf_ref_frequency_ghz": 17.0,
                "psf_scale_inverse_frequency": True,
                "save_raw_h5": None,
                "artifacts_dir": None,
                "no_artifacts_png": False,
            },
            "internal": {
                "geometry": {
                    "mode": "explicit_if_cli_else_saved_fov_with_default_pixel_scale_else_gxrender_auto",
                    "saved_fov_keys": ["xc_arcsec", "yc_arcsec", "xsize_arcsec", "ysize_arcsec"],
                    "fallback_pixel_scale_arcsec": 2.0,
                },
                "psf": {"bmaj_arcsec": 5.77, "bmin_arcsec": 5.77, "bpa_deg": -17.5, "kernel_size": 41, "scale_inverse_frequency": True},
                "optimizer": {
                    "threshold": 0.1,
                    "target_metric": "chi2",
                    "xatol": 1e-3,
                    "maxiter": 60,
                    "adaptive_bracketing": False,
                    "q0_start": "geometric_mean(q0_min, q0_max)",
                    "q0_step": 1.61803398875,
                    "max_bracket_steps": 12,
                },
            },
        }
        print(json.dumps(defaults, indent=2, sort_keys=True))
        return 0

    if not args.model_path:
        parser.error("--model-path is required (use --defaults to inspect defaults without running)")

    model_path_input = Path(args.model_path).expanduser()
    model_suffix = model_path_input.suffix.lower()
    if model_suffix not in {".h5", ".hdf5", ".sav"}:
        parser.error("--model-path must point to a .h5/.hdf5 file (preferred) or a .sav file")

    try:
        from gxrender import sdk
    except ModuleNotFoundError:
        parser.error(
            "gxrender is required for this demo. Install gximagecomputing/gxrender into this environment, "
            "or run with the Python environment where gxrender is already available."
        )

    model_path_for_render = _ensure_refmap_h5_for_model(model_path_input)
    if model_path_for_render is None:
        parser.error(f"unable to open/prepare model file: {model_path_input}")
    if model_suffix == ".sav":
        print(f"converted SAV model to temporary HDF5: {model_path_for_render}")

    ebtel_path_for_render = str(args.ebtel_path or "").strip()
    if ebtel_path_for_render:
        ebtel_candidate = Path(ebtel_path_for_render).expanduser()
        if not ebtel_candidate.exists():
            parser.error(f"--ebtel-path does not exist: {ebtel_candidate}")
        ebtel_path_for_render = str(ebtel_candidate)

    output_dir = None
    output_name = None
    a_param = 0.3
    b_param = 2.7
    active_frequency_ghz = float(args.frequency_ghz)
    freqlist_ghz = [active_frequency_ghz]

    try:
        observer_overrides, observer_source = _resolve_observer_overrides(
            sdk,
            model_path=model_path_for_render,
            observer_name=args.observer,
            dsun_cm=args.dsun_cm,
            lonc_deg=args.lonc_deg,
            b0sun_deg=args.b0sun_deg,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.save_raw_h5:
        output_path = Path(args.save_raw_h5)
        output_dir = str(output_path.parent)
        output_name = output_path.stem if output_path.suffix == ".h5" else output_path.name

    geometry_overrides_requested = any(v is not None for v in (args.xc, args.yc, args.dx, args.dy, args.nx, args.ny))
    geometry = None
    saved_fov = None
    if geometry_overrides_requested:
        geometry = sdk.MapGeometry(
            xc=args.xc,
            yc=args.yc,
            dx=args.dx,
            dy=args.dy,
            nx=args.nx,
            ny=args.ny,
        )
    else:
        saved_fov = _load_saved_fov_from_model(model_path_for_render)
        if saved_fov is not None:
            dx_saved = float(args.pixel_scale_arcsec)
            dy_saved = float(args.pixel_scale_arcsec)
            nx_saved = max(16, int(round(float(saved_fov["xsize_arcsec"]) / abs(dx_saved))))
            ny_saved = max(16, int(round(float(saved_fov["ysize_arcsec"]) / abs(dy_saved))))
            geometry = sdk.MapGeometry(
                xc=float(saved_fov["xc_arcsec"]),
                yc=float(saved_fov["yc_arcsec"]),
                dx=dx_saved,
                dy=dy_saved,
                nx=nx_saved,
                ny=ny_saved,
            )

    print(f"[setup] model: {model_path_for_render}")
    print(f"[setup] ebtel: {ebtel_path_for_render if ebtel_path_for_render else '<isothermal fallback>'}")
    print(f"[setup] frequency [GHz]: {active_frequency_ghz:.3f}")
    if observer_overrides is None:
        print(f"[setup] observer mode: saved metadata ({observer_source})")
    else:
        print(
            "[setup] observer mode: overrides "
            f"lonc={observer_overrides.lonc_deg} b0={observer_overrides.b0sun_deg} dsun_cm={observer_overrides.dsun_cm} "
            f"source={observer_source}"
        )
    if geometry_overrides_requested:
        print(
            "[setup] geometry mode: explicit "
            f"xc={args.xc} yc={args.yc} dx={args.dx} dy={args.dy} nx={args.nx} ny={args.ny}"
        )
    elif saved_fov is not None and geometry is not None:
        raw_model_dx = saved_fov.get("dx_arcsec")
        raw_model_dy = saved_fov.get("dy_arcsec")
        raw_model_dx_str = f"{float(raw_model_dx):.6f}" if raw_model_dx is not None else "<not present>"
        raw_model_dy_str = f"{float(raw_model_dy):.6f}" if raw_model_dy is not None else "<not present>"
        print(
            "[setup] saved model fov read: "
            f"xc={saved_fov['xc_arcsec']:.3f} yc={saved_fov['yc_arcsec']:.3f} "
            f"xsize={saved_fov['xsize_arcsec']:.3f} ysize={saved_fov['ysize_arcsec']:.3f} "
            f"model_dx={raw_model_dx_str} model_dy={raw_model_dy_str}"
        )
        print(
            "[setup] geometry mode: saved_fov "
            f"dx={float(geometry.dx):.3f} dy={float(geometry.dy):.3f} nx={int(geometry.nx)} ny={int(geometry.ny)}"
        )
        print(
            "[setup] note: using --pixel-scale-arcsec for image sampling when explicit --dx/--dy are not provided"
        )
    else:
        print(
            "[setup] geometry mode: auto (gxrender-derived center/FOV) "
            f"pixel_scale_arcsec={float(args.pixel_scale_arcsec):.3f}"
        )
    psf_scale = float(args.psf_ref_frequency_ghz) / float(active_frequency_ghz) if args.psf_scale_inverse_frequency else 1.0
    psf_bmaj_eff = float(args.psf_bmaj_arcsec) * psf_scale
    psf_bmin_eff = float(args.psf_bmin_arcsec) * psf_scale
    print(f"[setup] plasma/heating: a={a_param:.3f} b={b_param:.3f} tbase=1.000e+06 nbase=1.000e+08")
    if bool(args.psf_scale_inverse_frequency):
        print(
            "[setup] psf: "
            f"reference beam: bmaj={float(args.psf_bmaj_arcsec):.3f} "
            f"bmin={float(args.psf_bmin_arcsec):.3f} "
            f"bpa={float(args.psf_bpa_deg):.3f} @ {float(args.psf_ref_frequency_ghz):.3f} GHz"
            f"\n    rescaled beam: bmaj={psf_bmaj_eff:.3f} bmin={psf_bmin_eff:.3f} "
            f"bpa={float(args.psf_bpa_deg):.3f} @ {float(active_frequency_ghz):.3f} GHz"
        )
    else:
        print(
            "[setup] psf: "
            f"beam: bmaj={psf_bmaj_eff:.3f} bmin={psf_bmin_eff:.3f} "
            f"bpa={float(args.psf_bpa_deg):.3f} @ {float(active_frequency_ghz):.3f} GHz"
        )
    print(
        "[setup] optimizer: "
        f"q0 in [{args.q0_min:.6f}, {args.q0_max:.6f}] "
        f"target={args.target_metric} xatol=0.001 maxiter=60 adaptive={bool(args.adaptive_bracketing)} "
        f"q0_start={args.q0_start if args.q0_start is not None else '<auto>'} "
        f"q0_step={float(args.q0_step):.6f} max_bracket_steps={int(args.max_bracket_steps)}"
    )

    stage_spinner = bool(args.spinner)
    base_renderer = GXRenderMWAdapter(
        model_path=model_path_for_render,
        ebtel_path=ebtel_path_for_render,
        frequency_ghz=active_frequency_ghz,
        tbase=1e6,
        nbase=1e8,
        a=a_param,
        b=b_param,
        geometry=geometry,
        observer=observer_overrides,
        pixel_scale_arcsec=float(args.pixel_scale_arcsec),
        output_dir=output_dir,
        output_name=output_name,
        output_format="h5",
    )

    kernel_dx = float(geometry.dx) if geometry is not None and geometry.dx is not None else float(args.pixel_scale_arcsec)
    kernel_dy = float(geometry.dy) if geometry is not None and geometry.dy is not None else float(args.pixel_scale_arcsec)
    kernel = _elliptical_gaussian_kernel(
        bmaj_arcsec=psf_bmaj_eff,
        bmin_arcsec=psf_bmin_eff,
        bpa_deg=float(args.psf_bpa_deg),
        dx_arcsec=kernel_dx,
        dy_arcsec=kernel_dy,
    )
    renderer = PSFConvolvedRenderer(base_renderer, kernel)

    stage_total = 4 if args.artifacts_dir else 3
    stage_index = 1

    observed_clean, _obs_elapsed = _run_stage(
        "Render synthetic observation",
        lambda: renderer.render(args.q0_true),
        spinner=stage_spinner,
        stage_index=stage_index,
        stage_total=stage_total,
    )
    stage_index += 1

    ny_eff, nx_eff = observed_clean.shape
    dx_eff = float(geometry.dx) if geometry is not None and geometry.dx is not None else float(args.pixel_scale_arcsec)
    dy_eff = float(geometry.dy) if geometry is not None and geometry.dy is not None else float(args.pixel_scale_arcsec)
    xc_eff = float(geometry.xc) if geometry is not None and geometry.xc is not None else float(saved_fov["xc_arcsec"]) if saved_fov is not None else 0.0
    yc_eff = float(geometry.yc) if geometry is not None and geometry.yc is not None else float(saved_fov["yc_arcsec"]) if saved_fov is not None else 0.0
    print(
        "[setup] computation grid: "
        f"nx={nx_eff} ny={ny_eff} dx={dx_eff:.3f} dy={dy_eff:.3f} "
        f"fov={nx_eff * abs(dx_eff):.2f}x{ny_eff * abs(dy_eff):.2f} arcsec"
    )

    peak_clean = float(np.nanmax(observed_clean)) if observed_clean.size else float("nan")
    if (not np.isfinite(peak_clean)) or peak_clean <= 0.0:
        if ebtel_path_for_render:
            parser.error(
                "rendered map is empty/non-positive for q0_true even with the provided EBTEL table. "
                "This usually indicates the model/frequency/plasma setup produces no MW emission in this configuration."
            )
        parser.error(
            "rendered map is empty/non-positive for q0_true; cannot build a valid fitting mask. "
            "Try providing --ebtel-path to a real ebtel.sav table, or use a model/frequency that produces emission."
        )

    noise_std = max(0.0, float(args.noise_frac)) * float(np.max(observed_clean))
    rng = np.random.default_rng(args.noise_seed)
    noise = rng.normal(loc=0.0, scale=noise_std, size=observed_clean.shape)
    observed = np.clip(observed_clean + noise, a_min=0.0, a_max=None)
    sigma = max(noise_std, 1.0) * np.ones_like(observed)

    eval_counter = 0
    fit_q0_trials: list[float] = []
    fit_metric_trials: list[float] = []
    render_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    def metric_function(q0: float) -> Q0MetricEvaluation:
        nonlocal eval_counter
        t_eval = time.perf_counter()
        q0_value = float(q0)
        raw_arr, modeled_arr = renderer.render_pair(q0_value)
        raw_arr = np.asarray(raw_arr, dtype=float)
        modeled_arr = np.asarray(modeled_arr, dtype=float)
        render_cache[q0_value] = (raw_arr, modeled_arr)
        mask = threshold_union_mask(observed, modeled_arr, 0.1)
        mask = mask & (observed > 0)  # exclude noise-clipped zeros (safe for rho2 = mod/obs)
        metrics = compute_metrics(observed, modeled_arr, sigma, mask)
        eval_counter += 1
        fit_q0_trials.append(q0_value)
        fit_metric_trials.append(_metric_value(metrics, args.target_metric))
        if args.progress:
            _clear_terminal_status_line()
            selected = int(np.count_nonzero(mask))
            total = int(mask.size)
            dt = time.perf_counter() - t_eval
            print(
                f"trial={eval_counter:03d} "
                f"q0={q0_value:.6f} "
                f"{_format_metric_report(metrics, args.target_metric)} "
                f"mask={selected}/{total} "
                f"dt={dt:.3f}s"
            )
        return Q0MetricEvaluation(
            metrics=metrics,
            total_observed_flux=float(np.sum(observed[mask], dtype=float)),
            total_modeled_flux=float(np.sum(modeled_arr[mask], dtype=float)),
        )

    try:
        result, fit_elapsed = _run_stage(
            "Optimize q0",
            lambda: find_best_q0(
                metric_function,
                q0_min=args.q0_min,
                q0_max=args.q0_max,
                target_metric=args.target_metric,
                xatol=1e-3,
                maxiter=60,
                adaptive_bracketing=bool(args.adaptive_bracketing),
                q0_start=args.q0_start,
                q0_step=float(args.q0_step),
                max_bracket_steps=int(args.max_bracket_steps),
            ),
            spinner=stage_spinner,
            stage_index=stage_index,
            stage_total=stage_total,
            running_label="running",
        )
        stage_index += 1
    except ValueError as exc:
        if "mask selects no elements" in str(exc):
            parser.error(
                "fitting mask is empty for the requested settings. "
                "This usually means rendered intensity is near zero in both observed/modeled maps. "
                "Try a different model, wider q0 bounds, or provide --ebtel-path to physical EBTEL tables."
            )
        raise

    cached_best_pair = _lookup_cached_render_pair(render_cache, result.q0)

    (modeled_best_raw, modeled_best), _best_elapsed = _run_stage(
        "Render best-fit map",
        lambda: cached_best_pair if cached_best_pair is not None else renderer.render_pair(result.q0),
        spinner=stage_spinner,
        stage_index=stage_index,
        stage_total=stage_total,
    )
    stage_index += 1
    residual = modeled_best - observed

    print(f"truth q0: {args.q0_true:.6f}")
    print(f"fit q0:   {result.q0:.6f}")
    print(f"metrics:  {_format_metric_report(result.metrics, result.target_metric)}")
    print(f"noise:    gaussian frac={args.noise_frac:.4f} seed={args.noise_seed} std={noise_std:.6e}")
    print(f"success:  {result.success}")
    print(f"adaptive: {result.used_adaptive_bracketing}")
    print(f"bracket:  {result.bracket if result.bracket is not None else '<none>'}")
    print(f"nfev:     {result.nfev}")
    print(f"nit:      {result.nit}")
    print(f"message:  {result.message}")
    print(f"fit time: {fit_elapsed:.3f}s")
    print(f"total:    {time.perf_counter() - t_global:.3f}s")

    fit_q0_trials = list(result.trial_q0) if result.trial_q0 else fit_q0_trials
    fit_metric_trials = list(result.trial_objective_values) if result.trial_objective_values else fit_metric_trials

    if args.artifacts_dir:
        out_dir = Path(args.artifacts_dir)
        _, _artifact_elapsed = _run_stage(
            "Save artifacts",
            lambda: _save_artifacts(
                out_dir,
                stem=args.artifacts_stem,
                show_plot=False,
                defer_show=args.show_plot,
                observed_clean=observed_clean,
                observed_noisy=observed,
                raw_modeled_best=modeled_best_raw,
                modeled_best=modeled_best,
                residual=residual,
                diagnostics={
                    "model_path": str(args.model_path),
                    "model_path_effective": str(model_path_for_render),
                    "ebtel_path_effective": ebtel_path_for_render,
                    "q0_truth": float(args.q0_true),
                    "q0_recovered": float(result.q0),
                    "q0_abs_error": float(abs(result.q0 - args.q0_true)),
                    "target_metric": str(result.target_metric),
                    "chi2": float(result.metrics.chi2),
                    "target_metric_value": _metric_value(result.metrics, result.target_metric),
                    "fit_q0_trials": fit_q0_trials,
                    "fit_metric_trials": fit_metric_trials,
                    "optimizer_message": str(result.message),
                    "optimizer_used_adaptive_bracketing": bool(result.used_adaptive_bracketing),
                    "optimizer_bracket_found": bool(result.bracket_found),
                    "optimizer_bracket": list(result.bracket) if result.bracket is not None else None,
                    "rho2": float(result.metrics.rho2),
                    "eta2": float(result.metrics.eta2),
                    "noise_frac": float(args.noise_frac),
                    "noise_seed": int(args.noise_seed),
                    "noise_std": float(noise_std),
                    "mw_frequency_ghz": float(active_frequency_ghz),
                    "active_frequency_ghz": float(active_frequency_ghz),
                    "freqlist_ghz_used": [float(x) for x in freqlist_ghz],
                    "log_metrics": bool(args.log_metrics),
                    "log_q0": bool(args.log_q0),
                    "zoom2best": int(args.zoom2best) if args.zoom2best else 0,
                    "a": float(a_param),
                    "b": float(b_param),
                    "observer_name_effective": str(args.observer) if args.observer else "saved-metadata",
                    "observer_source": observer_source,
                    "observer_lonc_deg": float(observer_overrides.lonc_deg) if observer_overrides and observer_overrides.lonc_deg is not None else 0.0,
                    "observer_b0sun_deg": float(observer_overrides.b0sun_deg) if observer_overrides and observer_overrides.b0sun_deg is not None else 0.0,
                    "observer_dsun_cm": float(observer_overrides.dsun_cm) if observer_overrides and observer_overrides.dsun_cm is not None else 1.495978707e13,
                    "observer_obs_time": str(_load_model_obs_time_iso(model_path_for_render) or ""),
                    "map_xc_arcsec": float(xc_eff),
                    "map_yc_arcsec": float(yc_eff),
                    "map_dx_arcsec": float(dx_eff),
                    "map_dy_arcsec": float(dy_eff),
                    "psf_bmaj_arcsec": float(psf_bmaj_eff),
                    "psf_bmin_arcsec": float(psf_bmin_eff),
                    "psf_bpa_deg": float(args.psf_bpa_deg),
                },
                save_png=not args.no_artifacts_png,
            ),
            spinner=stage_spinner,
            stage_index=stage_index,
            stage_total=stage_total,
        )
        stage_index += 1
        print(f"artifacts dir: {out_dir}")
        if args.show_plot and not args.no_artifacts_png:
            import matplotlib.pyplot as _plt
            _plt.show()
            _plt.close("all")
        print(f"data file:      {out_dir / (args.artifacts_stem + '.h5')}")
        print(f"png file:       {out_dir / (args.artifacts_stem + '.png')}")

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
