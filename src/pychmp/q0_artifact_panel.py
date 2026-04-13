from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import warnings

import h5py
import numpy as np
from astropy.io import fits
from astropy.wcs import FITSFixedWarning, WCS
from matplotlib.figure import Figure


METRIC_CHOICES = ("chi2", "rho2", "eta2")

_REFMAP_CACHE: dict[str, Path | None] = {}
_BLOS_CACHE: dict[str, tuple[np.ndarray, fits.Header] | None] = {}
_BLOS_FOV_CACHE: dict[str, tuple[np.ndarray, fits.Header] | None] = {}


def _safe_wcs(header: fits.Header) -> WCS:
    """Build a WCS while suppressing benign DATE-OBS->MJD-OBS fix warnings."""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FITSFixedWarning)
        return WCS(header)


def _decode_h5_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        item = value.item()
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace")
        return str(item)
    return str(value)


def _ensure_refmap_h5_for_model(model_path: Path) -> Path | None:
    model_key = str(model_path.expanduser())
    if model_key in _REFMAP_CACHE:
        return _REFMAP_CACHE[model_key]

    suffix = model_path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        result = model_path if model_path.exists() else None
        _REFMAP_CACHE[model_key] = result
        return result
    if suffix != ".sav" or not model_path.exists():
        _REFMAP_CACHE[model_key] = None
        return None

    try:
        from gxrender.io import build_h5_from_sav
    except Exception:
        _REFMAP_CACHE[model_key] = None
        return None

    out_h5 = Path("/tmp") / f"pychmp_refmaps_{model_path.stem}.h5"
    if out_h5.exists():
        _REFMAP_CACHE[model_key] = out_h5
        return out_h5
    try:
        build_h5_from_sav(model_path, out_h5, template_h5=None)
        _REFMAP_CACHE[model_key] = out_h5
        return out_h5
    except Exception:
        _REFMAP_CACHE[model_key] = None
        return None


def _load_blos_reference_map(model_path: Path) -> tuple[np.ndarray, fits.Header] | None:
    model_key = str(model_path.expanduser())
    if model_key in _BLOS_CACHE:
        return _BLOS_CACHE[model_key]

    refmap_h5 = _ensure_refmap_h5_for_model(model_path)
    if refmap_h5 is None:
        _BLOS_CACHE[model_key] = None
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
                result = (data, header)
                _BLOS_CACHE[model_key] = result
                return result
    except Exception:
        _BLOS_CACHE[model_key] = None
        return None

    _BLOS_CACHE[model_key] = None
    return None


def _ordered_metric_names(target_metric: str) -> tuple[str, ...]:
    return (target_metric,) + tuple(name for name in METRIC_CHOICES if name != target_metric)


def _metric_summary_lines(diag: dict[str, Any], target_metric: str, fmt: Callable[[Any, str], str]) -> list[str]:
    lines: list[str] = []
    for name in _ordered_metric_names(target_metric):
        label = f"target[{name}]" if name == target_metric else name
        lines.append(f"{label:<12}: {fmt(diag.get(name), '.6e')}")
    return lines


def _header_token(header: fits.Header, shape: tuple[int, int]) -> str:
    return f"{shape[0]}x{shape[1]}::{header.tostring(sep='\n', endcard=True)}"


def _color_limits(data: np.ndarray, *, symmetric: bool = False) -> tuple[float, float]:
    finite = np.asarray(data, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    if symmetric:
        vmax = float(np.nanmax(np.abs(finite)))
        vmax = max(vmax, 1e-12)
        return -vmax, vmax
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if vmin == vmax:
        pad = max(abs(vmin) * 0.05, 1.0)
        return vmin - pad, vmax + pad
    return vmin, vmax


def _format_scalar(value: Any, pattern: str) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "n/a"
    if not np.isfinite(numeric):
        return "nan"
    if pattern.endswith("f") and numeric != 0.0 and abs(numeric) < 1e-4:
        precision_text = pattern[1:-1] if pattern.startswith(".") else ""
        try:
            precision = max(1, int(precision_text))
        except Exception:
            precision = 6
        return format(numeric, f".{precision}e")
    return format(numeric, pattern)


def _crop_blos_to_fov(
    model_path: Path,
    *,
    header: fits.Header,
    shape: tuple[int, int],
    wcs_header_transform: Callable[[fits.Header], fits.Header] | None,
) -> tuple[np.ndarray, fits.Header] | None:
    fov_key = _header_token(header, shape)
    cache_key = f"{str(model_path.expanduser())}::{fov_key}"
    if cache_key in _BLOS_FOV_CACHE:
        return _BLOS_FOV_CACHE[cache_key]

    blos = _load_blos_reference_map(model_path)
    if blos is None:
        _BLOS_FOV_CACHE[cache_key] = None
        return None

    blos_data, blos_hdr = blos
    if wcs_header_transform is not None:
        blos_hdr = wcs_header_transform(blos_hdr)

    try:
        import sunpy.map
        from astropy import units as u
        from astropy.coordinates import SkyCoord

        m_blos = sunpy.map.Map(blos_data, blos_hdr)
        xc = float(header.get("CRVAL1", 0.0))
        yc = float(header.get("CRVAL2", 0.0))
        nx = int(header.get("NAXIS1", shape[1]))
        ny = int(header.get("NAXIS2", shape[0]))
        dx = float(header.get("CDELT1", 1.0))
        dy = float(header.get("CDELT2", 1.0))
        half_x = 0.5 * nx * abs(dx)
        half_y = 0.5 * ny * abs(dy)
        bottom_left = SkyCoord((xc - half_x) * u.arcsec, (yc - half_y) * u.arcsec, frame=m_blos.coordinate_frame)
        top_right = SkyCoord((xc + half_x) * u.arcsec, (yc + half_y) * u.arcsec, frame=m_blos.coordinate_frame)
        submap = m_blos.submap(bottom_left, top_right=top_right)
        result = (np.asarray(submap.data, dtype=float), submap.fits_header)
    except Exception:
        result = (np.asarray(blos_data, dtype=float), blos_hdr)

    _BLOS_FOV_CACHE[cache_key] = result
    return result


class Q0ArtifactPanelFigure:
    def __init__(self, figure: Figure | None = None) -> None:
        self.figure = figure or Figure(figsize=(14.8, 9.2), constrained_layout=True)
        self._gs: Any = None
        self._header_token: str | None = None
        self._shape: tuple[int, int] | None = None
        self._common_axes: dict[str, Any] = {}
        self._common_images: dict[str, Any] = {}
        self._common_colorbars: dict[str, Any] = {}
        self._common_notes: dict[str, Any] = {}
        self._common_titles: dict[str, str] = {}
        self._trials_ax: Any = None
        self._blos_ax: Any = None
        self._blos_image: Any = None
        self._blos_colorbar: Any = None
        self._blos_note: Any = None
        self._blos_cache_key: str | None = None
        self._blos_loaded = False
        self._show_mask_contours = True

    def set_mask_contours_visible(self, visible: bool) -> None:
        self._show_mask_contours = bool(visible)

    def mask_contours_visible(self) -> bool:
        return bool(self._show_mask_contours)

    def _get_mask(self, observed, modeled, diagnostics):
        # Get mask type and threshold from diagnostics
        mask_type = diagnostics.get("mask_type", "union")
        threshold = float(diagnostics.get("threshold", 0.1))
        # Compute mask
        from pychmp.metrics import threshold_union_mask, threshold_data_mask, threshold_model_mask, threshold_and_mask
        mask_fn = {
            "union": threshold_union_mask,
            "data": threshold_data_mask,
            "model": threshold_model_mask,
            "and": threshold_and_mask,
        }.get(mask_type, threshold_union_mask)
        return mask_fn(observed, modeled, threshold)

    def _draw_mask_contours(self, show, observed, modeled, diagnostics):
        # Overlay mask contours on observed, modeled, and residual panels
        mask = self._get_mask(observed, modeled, diagnostics)
        for name in ("observed", "modeled", "residual"):
            ax = self._common_axes.get(name)
            if ax is None:
                continue
            # Remove previous contours
            previous = getattr(ax, "_mask_contours", ())
            if hasattr(previous, "remove"):
                previous.remove()
            else:
                for coll in previous:
                    coll.remove()
            if show:
                cs = ax.contour(mask.astype(float), levels=[0.5], colors="lime", linewidths=1.5, alpha=0.8)
                ax._mask_contours = cs
            else:
                ax._mask_contours = ()
    def _build_layout(self, header: fits.Header, shape: tuple[int, int]) -> None:
        self.figure.clear()
        gs = self.figure.add_gridspec(2, 3)
        self._gs = gs

        common_specs = [
            ("observed", gs[0, 1], "inferno"),
            ("raw_modeled", gs[0, 2], "inferno"),
            ("modeled", gs[1, 0], "inferno"),
            ("residual", gs[1, 1], "coolwarm"),
        ]
        self._common_axes = {}
        self._common_images = {}
        self._common_colorbars = {}
        self._common_notes = {}

        for name, slot, cmap in common_specs:
            axis = self.figure.add_subplot(slot, projection=_safe_wcs(header))
            image = axis.imshow(np.zeros(shape, dtype=float), origin="lower", cmap=cmap)
            colorbar = self.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            note = axis.text(
                0.02,
                0.02,
                "",
                transform=axis.transAxes,
                va="bottom",
                ha="left",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85},
            )
            self._common_axes[name] = axis
            self._common_images[name] = image
            self._common_colorbars[name] = colorbar
            self._common_notes[name] = note
            self._common_notes[name] = note

        self._blos_ax = self.figure.add_subplot(gs[0, 0])
        self._blos_image = None
        self._blos_colorbar = None
        self._blos_note = self._blos_ax.text(
            0.02,
            0.02,
            "",
            transform=self._blos_ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85},
        )
        self._trials_ax = self.figure.add_subplot(gs[1, 2])
        self._trials_ax.set_box_aspect(1)
        self._header_token = _header_token(header, shape)
        self._shape = shape
        self._blos_cache_key = None
        self._blos_loaded = False

    def _ensure_layout(self, header: fits.Header, shape: tuple[int, int]) -> None:
        token = _header_token(header, shape)
        if token != self._header_token:
            self._build_layout(header, shape)

    def _update_common_panel(self, name: str, data: np.ndarray, *, title: str, note: str) -> None:
        image = self._common_images[name]
        axis = self._common_axes[name]
        image.set_data(np.asarray(data, dtype=float))
        vmin, vmax = _color_limits(np.asarray(data, dtype=float), symmetric=name == "residual")
        image.set_clim(vmin, vmax)
        self._common_colorbars[name].update_normal(image)
        axis.set_title(title)
        self._common_notes[name].set_text(note)

    def _show_blos_placeholder(self, *, a_text: str, b_text: str, message: str) -> None:
        if self._blos_colorbar is not None:
            self._blos_colorbar.remove()
            self._blos_colorbar = None
        if self._blos_ax is not None:
            self._blos_ax.remove()
        self._blos_ax = self.figure.add_subplot(self._gs[0, 0])
        self._blos_image = None
        self._blos_ax.set_title("B_los Reference")
        self._blos_ax.text(0.5, 0.58, message, transform=self._blos_ax.transAxes, ha="center", va="center")
        self._blos_ax.axis("off")
        self._blos_note = self._blos_ax.text(
            0.02,
            0.02,
            f"a={a_text}  b={b_text}\nq0=n/a",
            transform=self._blos_ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85},
        )
        self._blos_cache_key = None
        self._blos_loaded = False

    def _update_blos_panel(
        self,
        *,
        model_path: Path,
        header: fits.Header,
        shape: tuple[int, int],
        wcs_header_transform: Callable[[fits.Header], fits.Header] | None,
        a_text: str,
        b_text: str,
        load_blos: bool,
    ) -> None:
        if not load_blos:
            self._show_blos_placeholder(a_text=a_text, b_text=b_text, message="Reference map not loaded.\nUse Load B_los to fetch it on demand.")
            return

        display_payload = _crop_blos_to_fov(
            model_path,
            header=header,
            shape=shape,
            wcs_header_transform=wcs_header_transform,
        )
        if display_payload is None:
            self._show_blos_placeholder(a_text=a_text, b_text=b_text, message="B_los reference unavailable.")
            return

        blos_data, blos_header = display_payload
        blos_token = _header_token(blos_header, np.asarray(blos_data, dtype=float).shape)
        if self._blos_cache_key != blos_token or self._blos_image is None:
            if self._blos_colorbar is not None:
                self._blos_colorbar.remove()
                self._blos_colorbar = None
            self._blos_ax.remove()
            self._blos_ax = self.figure.add_subplot(self._gs[0, 0], projection=_safe_wcs(blos_header))
            self._blos_image = self._blos_ax.imshow(np.asarray(blos_data, dtype=float), origin="lower", cmap="gray")
            self._blos_colorbar = self.figure.colorbar(self._blos_image, ax=self._blos_ax, fraction=0.046, pad=0.04)
            self._blos_note = self._blos_ax.text(
                0.02,
                0.02,
                "",
                transform=self._blos_ax.transAxes,
                va="bottom",
                ha="left",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85},
            )
            self._blos_cache_key = blos_token

        self._blos_image.set_data(np.asarray(blos_data, dtype=float))
        vmin, vmax = _color_limits(np.asarray(blos_data, dtype=float), symmetric=True)
        self._blos_image.set_clim(vmin, vmax)
        if self._blos_colorbar is not None:
            self._blos_colorbar.update_normal(self._blos_image)
        self._blos_ax.set_title("B_los Reference")
        if self._blos_note is not None:
            self._blos_note.set_text(f"a={a_text}  b={b_text}\nq0=n/a")
        self._blos_loaded = True

    def _update_trials(
        self,
        *,
        q0_trials: Any,
        metric_trials: Any,
        target_metric: str,
        target_metric_val: Any,
        q0_best: Any,
        q0_true: Any,
        a_text: str,
        b_text: str,
        frequency_ghz: float,
        log_metrics: bool,
        log_q0: bool,
        zoom2best: int | None,
        fmt: Callable[[Any, str], str],
        diagnostics: dict[str, Any],
    ) -> None:
        ax = self._trials_ax
        ax.clear()
        ax.set_box_aspect(1)

        plotted_trials = False
        try:
            q0_arr = np.asarray(q0_trials, dtype=float)
            metric_arr = np.asarray(metric_trials, dtype=float)
            ok = q0_arr.ndim == 1 and metric_arr.ndim == 1 and q0_arr.size == metric_arr.size and q0_arr.size > 0
            if ok:
                order = np.argsort(q0_arr)
                ax.plot(q0_arr[order], metric_arr[order], "-o", ms=4, lw=1.2, color="#2b6cb0", alpha=0.9, label=f"trial {target_metric}")
                ax.scatter(q0_arr, metric_arr, s=22, color="#2b6cb0", alpha=0.55)
                if q0_best is not None:
                    ax.axvline(float(q0_best), color="#d62728", ls="--", lw=1.6)
                ymin = float(np.nanmin(metric_arr))
                ymax = float(np.nanmax(metric_arr))
                yref = float(target_metric_val) if target_metric_val is not None else ymin
                if q0_best is not None and np.isfinite(yref):
                    ax.scatter([float(q0_best)], [yref], color="#d62728", s=40, zorder=4)
                title_parts: list[str] = []
                if log_q0:
                    title_parts.append("log x")
                if log_metrics:
                    title_parts.append("log y")
                if zoom2best is not None and zoom2best > 0:
                    title_parts.append(f"zoom±{zoom2best}")
                title_suffix = f" ({', '.join(title_parts)})" if title_parts else ""
                ax.set_title(f"{target_metric} vs q0 Trials{title_suffix}")
                ax.set_xlabel("q0")
                ax.set_ylabel(target_metric)
                ax.grid(alpha=0.25)
                if log_metrics and np.all(np.asarray(metric_arr, dtype=float) > 0.0):
                    ax.set_yscale("log")
                if log_q0 and np.all(np.asarray(q0_arr, dtype=float) > 0.0):
                    ax.set_xscale("log")
                if zoom2best is not None and zoom2best > 0 and q0_arr.size > 1:
                    q0_sorted = q0_arr[order]
                    best_sorted_pos = int(np.argmin(metric_arr[order]))
                    lo = max(0, best_sorted_pos - zoom2best)
                    hi = min(len(order) - 1, best_sorted_pos + zoom2best)
                    q0_lo = float(q0_sorted[lo])
                    q0_hi = float(q0_sorted[hi])
                    margin = (q0_hi - q0_lo) * 0.08 if q0_hi > q0_lo else max(abs(q0_lo) * 0.05, 1e-9)
                    ax.set_xlim(q0_lo - margin, q0_hi + margin)
                    visible_metric = (metric_arr[order])[lo : hi + 1]
                    finite_m = visible_metric[np.isfinite(visible_metric)]
                    if finite_m.size > 0:
                        vy_min = float(np.nanmin(finite_m))
                        vy_max = float(np.nanmax(finite_m))
                        if ax.get_yscale() == "log" and vy_min > 0:
                            ax.set_ylim(vy_min / 1.5, vy_max * 1.5)
                        else:
                            ypad = (vy_max - vy_min) * 0.12 if vy_max > vy_min else max(abs(vy_min) * 0.1, 1e-12)
                            ax.set_ylim(vy_min - ypad, vy_max + ypad)
                legend_label = (
                    f"q0={fmt(q0_best, '.6f')}\n"
                    f"target[{target_metric}]={fmt(target_metric_val, '.6e')}\n"
                    f"a={a_text} b={b_text}\n"
                    f"q0_true={fmt(q0_true, '.6f')}\n"
                    f"trials={int(q0_arr.size)}"
                )
                ax.plot([], [], " ", label=legend_label)
                ax.legend(loc="best", fontsize=9, framealpha=0.9)
                if np.isfinite(ymin) and np.isfinite(ymax) and ymin == ymax:
                    pad = max(1.0, abs(ymin) * 0.05)
                    ax.set_ylim(ymin - pad, ymax + pad)
                plotted_trials = True
        except Exception:
            plotted_trials = False

        if not plotted_trials:
            summary_lines = [
                "Run Summary",
                f"freq [GHz] : {fmt(frequency_ghz, '.2f')}",
                f"a          : {a_text}",
                f"b          : {b_text}",
                f"q0 true    : {fmt(q0_true, '.6f')}",
                f"q0 best    : {fmt(q0_best, '.6f')}",
                *_metric_summary_lines(diagnostics, target_metric, fmt),
                "trials     : unavailable",
            ]
            ax.axis("off")
            ax.text(
                0.02,
                0.98,
                "\n".join(summary_lines),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=11,
                family="monospace",
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f6f6f6", "edgecolor": "#cccccc"},
            )

    def update(
        self,
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
        wcs_header_transform: Callable[[fits.Header], fits.Header] | None = None,
        load_blos: bool = False,
        out_png: Path | None = None,
    ) -> None:
        diag = diagnostics or {}
        header = wcs_header.copy()
        if wcs_header_transform is not None:
            header = wcs_header_transform(header)

        observed_arr = np.asarray(observed_noisy, dtype=float)
        shape = observed_arr.shape
        self._ensure_layout(header, shape)

        def fmt(value: Any, pattern: str) -> str:
            return _format_scalar(value, pattern)

        q0_true = diag.get("q0_truth")
        q0_best = diag.get("q0_recovered")
        a_text = fmt(diag.get("a"), ".3f")
        b_text = fmt(diag.get("b"), ".3f")
        target_metric = str(diag.get("target_metric", "chi2"))
        target_metric_val = diag.get("target_metric_value", diag.get(target_metric))

        psf_bmaj_val = diag.get("psf_bmaj_arcsec")
        psf_bmin_val = diag.get("psf_bmin_arcsec")
        psf_bpa_val = diag.get("psf_bpa_deg")
        noise_frac_val = diag.get("noise_frac")
        noise_std_val = diag.get("noise_std")

        if psf_bmaj_val is not None and psf_bmin_val is not None:
            bpa_part = f"  PA={fmt(psf_bpa_val, '.1f')}°" if psf_bpa_val is not None else ""
            psf_legend = f"\nPSF: {fmt(psf_bmaj_val, '.1f')}\"×{fmt(psf_bmin_val, '.1f')}\"{bpa_part}"
        else:
            psf_legend = ""

        if noise_frac_val is not None:
            noise_legend = f"\nnoise: {fmt(float(noise_frac_val) * 100.0, '.1f')}%"
            if noise_std_val is not None:
                noise_legend += f"  std={fmt(noise_std_val, '.2e')}"
        else:
            noise_legend = ""

        self._update_common_panel(
            "observed",
            observed_arr,
            title=f"Observed (Noisy) @ {float(frequency_ghz):.2f} GHz",
            note=f"a={a_text}  b={b_text}\nq0={fmt(q0_true, '.6f')}{psf_legend}{noise_legend}",
        )
        self._update_common_panel(
            "raw_modeled",
            np.asarray(raw_modeled_best, dtype=float),
            title=f"Modeled Raw (Best Q0) @ {float(frequency_ghz):.2f} GHz",
            note=f"a={a_text}  b={b_text}\nq0={fmt(q0_best, '.6f')}",
        )
        self._update_common_panel(
            "modeled",
            np.asarray(modeled_best, dtype=float),
            title=f"Modeled (Best Q0) @ {float(frequency_ghz):.2f} GHz",
            note=f"a={a_text}  b={b_text}\nq0={fmt(q0_best, '.6f')}{psf_legend}",
        )
        self._update_common_panel(
            "residual",
            np.asarray(residual, dtype=float),
            title=f"Residual (Model-Obs) @ {float(frequency_ghz):.2f} GHz",
            note=f"a={a_text}  b={b_text}\nq0=true:{fmt(q0_true, '.6f')} best:{fmt(q0_best, '.6f')}{psf_legend}",
        )

        self._update_blos_panel(
            model_path=model_path,
            header=header,
            shape=shape,
            wcs_header_transform=wcs_header_transform,
            a_text=a_text,
            b_text=b_text,
            load_blos=load_blos,
        )

        self._update_trials(
            q0_trials=diag.get("fit_q0_trials", ()),
            metric_trials=diag.get(f"fit_{target_metric}_trials", ()),
            target_metric=target_metric,
            target_metric_val=target_metric_val,
            q0_best=q0_best,
            q0_true=q0_true,
            a_text=a_text,
            b_text=b_text,
            frequency_ghz=frequency_ghz,
            log_metrics=log_metrics,
            log_q0=log_q0,
            zoom2best=zoom2best,
            fmt=fmt,
            diagnostics=diag,
        )

        observed_arr = np.asarray(observed_noisy, dtype=float)
        modeled_arr = np.asarray(modeled_best, dtype=float)
        self._draw_mask_contours(self._show_mask_contours, observed_arr, modeled_arr, diag)

        if out_png is not None:
            self.figure.savefig(str(out_png), dpi=180)


def plot_q0_artifact_panel(
    out_png: Path | str | None,
    *,
    model_path: Path | str,
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
    show_plot: bool = False,
    defer_show: bool = False,
    wcs_header_transform: Callable[[fits.Header], fits.Header] | None = None,
    load_blos: bool = False,
) -> Figure:
    """Render the legacy Q0 artifact panel and optionally save/show it.

    This compatibility wrapper preserves the public function used by the
    example scripts while delegating the actual panel construction to
    ``Q0ArtifactPanelFigure``.
    """

    panel = Q0ArtifactPanelFigure()
    output_path = None if out_png is None else Path(out_png)
    panel.update(
        model_path=Path(model_path),
        observed_noisy=np.asarray(observed_noisy, dtype=float),
        raw_modeled_best=np.asarray(raw_modeled_best, dtype=float),
        modeled_best=np.asarray(modeled_best, dtype=float),
        residual=np.asarray(residual, dtype=float),
        wcs_header=wcs_header,
        frequency_ghz=float(frequency_ghz),
        diagnostics=diagnostics,
        log_metrics=bool(log_metrics),
        log_q0=bool(log_q0),
        zoom2best=zoom2best,
        wcs_header_transform=wcs_header_transform,
        load_blos=bool(load_blos),
        out_png=output_path,
    )

    if show_plot:
        import matplotlib.pyplot as plt

        plt.figure(panel.figure.number)
        if defer_show:
            panel.figure.canvas.draw_idle()
        else:
            plt.show()

    return panel.figure
