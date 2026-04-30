from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import warnings

import h5py
import numpy as np
from astropy.io import fits
from astropy.wcs import FITSFixedWarning, WCS
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm, Normalize, SymLogNorm


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


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _format_frequency_label(frequency_ghz: float) -> str:
    return f"{float(frequency_ghz):.2f} GHz"


def _format_wavelength_label(wavelength_angstrom: float) -> str:
    rounded = round(float(wavelength_angstrom))
    if np.isclose(float(wavelength_angstrom), float(rounded), rtol=0.0, atol=1e-9):
        return f"{int(rounded)} A"
    return f"{float(wavelength_angstrom):.3f} A"


def _spectral_label(diagnostics: dict[str, Any] | None, frequency_ghz: float | None) -> str:
    diag = diagnostics or {}
    label = str(diag.get("spectral_label", "")).strip()
    if label:
        return label
    domain = str(diag.get("spectral_domain", "")).strip().lower()
    wavelength_angstrom = _optional_float(diag.get("wavelength_angstrom"))
    if domain in {"euv", "uv"} and wavelength_angstrom is not None:
        return _format_wavelength_label(wavelength_angstrom)
    resolved_frequency_ghz = (
        _optional_float(diag.get("frequency_ghz"))
        or _optional_float(diag.get("active_frequency_ghz"))
        or _optional_float(diag.get("mw_frequency_ghz"))
        or _optional_float(frequency_ghz)
    )
    if resolved_frequency_ghz is not None:
        return _format_frequency_label(resolved_frequency_ghz)
    channel_label = str(diag.get("euv_channel", "")).strip()
    if channel_label:
        return channel_label
    return "selected slice"


def _load_embedded_blos_reference(f: h5py.File) -> tuple[np.ndarray, fits.Header] | None:
    candidate_suffixes = (
        "refmaps/Bz_reference",
        "reference_maps/B_los",
        "reference_maps/Bz_reference",
    )

    def _group_payload(grp: h5py.Group) -> tuple[np.ndarray, fits.Header] | None:
        if "data" not in grp or "wcs_header" not in grp:
            return None
        data = np.asarray(grp["data"], dtype=float)
        wcs_text = _decode_h5_scalar(grp["wcs_header"][()])
        header = fits.Header.fromstring(wcs_text, sep="\n")
        return data, header

    for suffix in candidate_suffixes:
        if suffix in f:
            payload = _group_payload(f[suffix])
            if payload is not None:
                return payload

    found: tuple[np.ndarray, fits.Header] | None = None

    def _visitor(name: str, obj: Any) -> None:
        nonlocal found
        if found is not None or not isinstance(obj, h5py.Group):
            return
        if not any(str(name).endswith(suffix) for suffix in candidate_suffixes):
            return
        found = _group_payload(obj)

    f.visititems(_visitor)
    if found is not None:
        return found
    return None


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


def load_blos_reference_map(model_path: Path) -> tuple[np.ndarray, fits.Header] | None:
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


def load_blos_reference_from_artifact(artifact_h5: Path | str) -> tuple[np.ndarray, fits.Header] | None:
    try:
        with h5py.File(Path(artifact_h5).expanduser(), "r") as f:
            return _load_embedded_blos_reference(f)
    except Exception:
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
    header_text = header.tostring(sep="\n", endcard=True)
    return f"{shape[0]}x{shape[1]}::{header_text}"


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


def _sanitize_blos_display_data(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=float).copy()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr
    abs_finite = np.abs(finite)
    robust = float(np.nanpercentile(abs_finite, 99.5))
    if not np.isfinite(robust) or robust <= 0.0:
        robust = float(np.nanmax(abs_finite))
    sentinel_cut = max(1.0e6, robust * 100.0)
    arr[np.abs(arr) > sentinel_cut] = np.nan
    return arr


def _positive_color_limits(data: np.ndarray) -> tuple[float, float] | None:
    finite = np.asarray(data, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if finite.size == 0:
        return None
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if vmin == vmax:
        vmax = max(vmax * 1.05, vmin * 1.05, vmin + 1e-12)
    return vmin, vmax


def _resolve_image_render_state(
    data: np.ndarray,
    *,
    scale: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = False,
) -> tuple[np.ndarray | np.ma.MaskedArray, Normalize, tuple[float, float], str]:
    arr = np.asarray(data, dtype=float)
    masked = np.ma.masked_invalid(arr)
    scale_name = str(scale or ("linear" if not symmetric else "linear")).strip().lower()

    if symmetric:
        if scale_name not in {"linear", "symlog"}:
            scale_name = "linear"
        auto_vmin, auto_vmax = _color_limits(arr, symmetric=True)
        resolved_vmin = auto_vmin if vmin is None else float(vmin)
        resolved_vmax = auto_vmax if vmax is None else float(vmax)
        if resolved_vmin >= resolved_vmax:
            resolved_vmin, resolved_vmax = auto_vmin, auto_vmax
        if scale_name == "symlog":
            nonzero = np.abs(arr[np.isfinite(arr) & (arr != 0.0)])
            if nonzero.size:
                linthresh = float(np.nanpercentile(nonzero, 10.0))
                linthresh = max(linthresh, max(abs(resolved_vmin), abs(resolved_vmax)) * 1e-3, 1e-6)
            else:
                linthresh = 1.0
            norm: Normalize = SymLogNorm(
                linthresh=min(linthresh, max(abs(resolved_vmin), abs(resolved_vmax))),
                vmin=resolved_vmin,
                vmax=resolved_vmax,
            )
        else:
            norm = Normalize(vmin=resolved_vmin, vmax=resolved_vmax)
        return masked, norm, (resolved_vmin, resolved_vmax), scale_name

    if scale_name not in {"linear", "log"}:
        scale_name = "linear"
    if scale_name == "log":
        auto_limits = _positive_color_limits(arr)
        if auto_limits is None:
            scale_name = "linear"
        else:
            auto_vmin, auto_vmax = auto_limits
            resolved_vmin = auto_vmin if vmin is None or float(vmin) <= 0.0 else float(vmin)
            resolved_vmax = auto_vmax if vmax is None or float(vmax) <= 0.0 else float(vmax)
            if resolved_vmin >= resolved_vmax:
                resolved_vmin, resolved_vmax = auto_vmin, auto_vmax
            norm = LogNorm(vmin=resolved_vmin, vmax=resolved_vmax)
            return np.ma.masked_less_equal(masked, 0.0), norm, (resolved_vmin, resolved_vmax), scale_name

    auto_vmin, auto_vmax = _color_limits(arr, symmetric=False)
    resolved_vmin = auto_vmin if vmin is None else float(vmin)
    resolved_vmax = auto_vmax if vmax is None else float(vmax)
    if resolved_vmin >= resolved_vmax:
        resolved_vmin, resolved_vmax = auto_vmin, auto_vmax
    return masked, Normalize(vmin=resolved_vmin, vmax=resolved_vmax), (resolved_vmin, resolved_vmax), "linear"


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


def _coerce_axis_limit(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        numeric = float(text)
    except Exception:
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def load_blos_reference_for_fov(
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

    blos = load_blos_reference_map(model_path)
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
        target_dimensions = u.Quantity([int(shape[1]), int(shape[0])], u.pixel)
        if tuple(np.asarray(submap.data).shape) != tuple(shape):
            submap = submap.resample(target_dimensions)
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
        mask_source = str(diagnostics.get("metrics_mask_source", "")).strip().lower()
        if mask_source == "explicit_fits":
            return None
        mask_type = diagnostics.get("mask_type", "union")
        threshold = float(diagnostics.get("metrics_mask_threshold", diagnostics.get("threshold", 0.1)))
        from pychmp.metrics import resolve_threshold_mask
        mask_fn = resolve_threshold_mask(mask_type)
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
            if show and mask is not None:
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

    def _update_common_panel(
        self,
        name: str,
        data: np.ndarray,
        *,
        title: str,
        note: str,
        scale: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        image = self._common_images[name]
        axis = self._common_axes[name]
        display_data, norm, _limits, _applied_scale = _resolve_image_render_state(
            np.asarray(data, dtype=float),
            scale=scale,
            vmin=vmin,
            vmax=vmax,
            symmetric=name == "residual",
        )
        image.set_data(display_data)
        image.set_norm(norm)
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
        blos_reference: tuple[np.ndarray, fits.Header] | None = None,
    ) -> None:
        if blos_reference is not None:
            display_payload = (
                np.asarray(blos_reference[0], dtype=float),
                blos_reference[1].copy(),
            )
        elif not load_blos:
            self._show_blos_placeholder(a_text=a_text, b_text=b_text, message="Reference map not loaded.\nUse Load B_los to fetch it on demand.")
            return
        else:
            display_payload = load_blos_reference_for_fov(
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

        display_data = _sanitize_blos_display_data(np.asarray(blos_data, dtype=float))
        self._blos_image.set_data(display_data)
        vmin, vmax = _color_limits(display_data, symmetric=True)
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
        frequency_ghz: float | None,
        log_metrics: bool,
        log_q0: bool,
        zoom2best: int | None,
        trials_xmin: float | None,
        trials_xmax: float | None,
        trials_ymin: float | None,
        trials_ymax: float | None,
        trials_xscale: str | None,
        trials_yscale: str | None,
        fmt: Callable[[Any, str], str],
        diagnostics: dict[str, Any],
    ) -> None:
        ax = self._trials_ax
        ax.clear()
        ax.set_box_aspect(1)
        selected_trial_index = diagnostics.get("selected_trial_index")
        try:
            selected_trial_index = None if selected_trial_index is None else int(selected_trial_index)
        except Exception:
            selected_trial_index = None

        plotted_trials = False
        try:
            q0_arr = np.asarray(q0_trials, dtype=float)
            metric_arr = np.asarray(metric_trials, dtype=float)
            ok = q0_arr.ndim == 1 and metric_arr.ndim == 1 and q0_arr.size == metric_arr.size and q0_arr.size > 0
            if ok:
                order = np.argsort(q0_arr)
                best_trial_index = int(np.nanargmin(metric_arr))
                if selected_trial_index is None or selected_trial_index < 0 or selected_trial_index >= q0_arr.size:
                    selected_trial_index = best_trial_index
                ax.plot(
                    q0_arr[order],
                    metric_arr[order],
                    "-o",
                    ms=4,
                    lw=1.2,
                    color="#2b6cb0",
                    alpha=0.9,
                    label=f"trial {target_metric}",
                )
                ax.scatter(q0_arr, metric_arr, s=22, color="#2b6cb0", alpha=0.55)
                selected_q0 = float(q0_arr[selected_trial_index])
                selected_metric_value = float(metric_arr[selected_trial_index])
                ax.axvline(selected_q0, color="#d62728", ls="--", lw=1.6)
                ax.scatter(
                    [selected_q0],
                    [selected_metric_value],
                    color="#d62728",
                    s=40,
                    zorder=5,
                    label="selected trial",
                )
                ax.scatter(
                    [float(q0_arr[best_trial_index])],
                    [float(metric_arr[best_trial_index])],
                    facecolor="none",
                    edgecolor="#f08c00",
                    linewidth=1.8,
                    s=76,
                    zorder=4,
                    label="best trial",
                )
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

                xscale_override = str(trials_xscale or "").strip().lower()
                if xscale_override in {"linear", "log"}:
                    if xscale_override != "log" or np.all(np.asarray(q0_arr, dtype=float) > 0.0):
                        ax.set_xscale(xscale_override)
                yscale_override = str(trials_yscale or "").strip().lower()
                if yscale_override in {"linear", "log"}:
                    if yscale_override != "log" or np.all(np.asarray(metric_arr, dtype=float) > 0.0):
                        ax.set_yscale(yscale_override)

                xmin = trials_xmin
                xmax = trials_xmax
                ymin = trials_ymin
                ymax = trials_ymax
                if ax.get_xscale() == "log":
                    if xmin is not None and xmin <= 0:
                        xmin = None
                    if xmax is not None and xmax <= 0:
                        xmax = None
                if ax.get_yscale() == "log":
                    if ymin is not None and ymin <= 0:
                        ymin = None
                    if ymax is not None and ymax <= 0:
                        ymax = None
                if xmin is not None and xmax is not None and xmin < xmax:
                    ax.set_xlim(xmin, xmax)
                elif xmin is not None:
                    ax.set_xlim(left=xmin)
                elif xmax is not None:
                    ax.set_xlim(right=xmax)
                if ymin is not None and ymax is not None and ymin < ymax:
                    ax.set_ylim(ymin, ymax)
                elif ymin is not None:
                    ax.set_ylim(bottom=ymin)
                elif ymax is not None:
                    ax.set_ylim(top=ymax)

                if (
                    ymin is not None
                    and ymax is not None
                    and np.isfinite(float(ymin))
                    and np.isfinite(float(ymax))
                    and float(ymin) == float(ymax)
                ):
                    pad = max(1.0, abs(ymin) * 0.05)
                    ax.set_ylim(ymin - pad, ymax + pad)
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    unique_handles: list[Any] = []
                    unique_labels: list[str] = []
                    seen: set[str] = set()
                    for handle, label in zip(handles, labels):
                        if not label or label in seen:
                            continue
                        seen.add(label)
                        unique_handles.append(handle)
                        unique_labels.append(label)
                    if unique_handles:
                        ax.legend(
                            unique_handles,
                            unique_labels,
                            loc="upper left",
                            fontsize=8,
                            framealpha=0.9,
                            handlelength=1.6,
                            borderpad=0.35,
                        )
                plotted_trials = True
        except Exception:
            plotted_trials = False

        if not plotted_trials:
            spectral_label = _spectral_label(diagnostics, frequency_ghz)
            selected_trial_display = "n/a"
            try:
                selected_trial_index_display = diagnostics.get("selected_trial_index")
                selected_trial_count_display = diagnostics.get("selected_trial_count")
                if selected_trial_index_display is not None and selected_trial_count_display is not None:
                    selected_trial_display = f"{int(selected_trial_index_display) + 1}/{int(selected_trial_count_display)}"
            except Exception:
                selected_trial_display = "n/a"
            summary_lines = [
                "Run Summary",
                f"slice      : {spectral_label}",
                f"a          : {a_text}",
                f"b          : {b_text}",
                f"q0 true    : {fmt(q0_true, '.6f')}",
                f"trial      : {selected_trial_display}",
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
        frequency_ghz: float | None = 17.0,
        diagnostics: dict[str, Any] | None = None,
        log_metrics: bool = False,
        log_q0: bool = False,
        zoom2best: int | None = None,
        trials_xmin: float | None = None,
        trials_xmax: float | None = None,
        trials_ymin: float | None = None,
        trials_ymax: float | None = None,
        trials_xscale: str | None = None,
        trials_yscale: str | None = None,
        common_map_scale: str | None = None,
        common_map_vmin: float | None = None,
        common_map_vmax: float | None = None,
        residual_map_scale: str | None = None,
        residual_map_vmin: float | None = None,
        residual_map_vmax: float | None = None,
        wcs_header_transform: Callable[[fits.Header], fits.Header] | None = None,
        load_blos: bool = False,
        blos_reference: tuple[np.ndarray, fits.Header] | None = None,
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
        q0_best = diag.get("best_q0_recovered", diag.get("q0_recovered"))
        display_q0 = diag.get("selected_trial_q0", diag.get("q0_recovered"))
        a_text = fmt(diag.get("a"), ".3f")
        b_text = fmt(diag.get("b"), ".3f")
        target_metric = str(diag.get("target_metric", "chi2"))
        target_metric_val = diag.get("target_metric_value", diag.get(target_metric))
        spectral_label = _spectral_label(diag, frequency_ghz)
        selected_trial_index = diag.get("selected_trial_index")
        selected_trial_count = diag.get("selected_trial_count")
        selected_trial_is_best = bool(diag.get("selected_trial_is_best", False))
        selected_trial_maps_available = bool(diag.get("selected_trial_maps_available", True))
        try:
            trial_title = (
                f"Trial {int(selected_trial_index) + 1}"
                if selected_trial_index is not None
                else "Best Q0"
            )
        except Exception:
            trial_title = "Best Q0"
        if selected_trial_index is not None and selected_trial_count is not None:
            trial_title = f"Trial {int(selected_trial_index) + 1}/{int(selected_trial_count)}"
        if selected_trial_is_best and selected_trial_index is not None:
            trial_title = f"{trial_title} / Best"
        # Keep panel geometry stable across search modes. Long fallback text in
        # per-panel annotation boxes causes constrained_layout to shrink the map
        # axes aggressively, especially for adaptive artifacts that lack stored
        # selected-trial maps. The shared single-point panel should therefore
        # stay layout-identical and simply fall back silently to best-fit maps.
        map_suffix = ""

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
            title=f"Observed (Noisy) @ {spectral_label}",
            note=f"a={a_text}  b={b_text}\nq0={fmt(q0_true, '.6f')}{psf_legend}{noise_legend}",
            scale=common_map_scale,
            vmin=_coerce_axis_limit(common_map_vmin),
            vmax=_coerce_axis_limit(common_map_vmax),
        )
        self._update_common_panel(
            "raw_modeled",
            np.asarray(raw_modeled_best, dtype=float),
            title=f"Modeled Raw ({trial_title}) @ {spectral_label}",
            note=f"a={a_text}  b={b_text}\nq0={fmt(display_q0, '.6f')}{map_suffix}",
            scale=common_map_scale,
            vmin=_coerce_axis_limit(common_map_vmin),
            vmax=_coerce_axis_limit(common_map_vmax),
        )
        self._update_common_panel(
            "modeled",
            np.asarray(modeled_best, dtype=float),
            title=f"Modeled ({trial_title}) @ {spectral_label}",
            note=f"a={a_text}  b={b_text}\nq0={fmt(display_q0, '.6f')}{psf_legend}{map_suffix}",
            scale=common_map_scale,
            vmin=_coerce_axis_limit(common_map_vmin),
            vmax=_coerce_axis_limit(common_map_vmax),
        )
        self._update_common_panel(
            "residual",
            np.asarray(residual, dtype=float),
            title=f"Residual ({trial_title}-Obs) @ {spectral_label}",
            note=f"a={a_text}  b={b_text}\nq0=true:{fmt(q0_true, '.6f')} selected:{fmt(display_q0, '.6f')} best:{fmt(q0_best, '.6f')}{psf_legend}{map_suffix}",
            scale=residual_map_scale,
            vmin=_coerce_axis_limit(residual_map_vmin),
            vmax=_coerce_axis_limit(residual_map_vmax),
        )

        self._update_blos_panel(
            model_path=model_path,
            header=header,
            shape=shape,
            wcs_header_transform=wcs_header_transform,
            a_text=a_text,
            b_text=b_text,
            load_blos=load_blos,
            blos_reference=blos_reference,
        )

        self._update_trials(
            q0_trials=diag.get("fit_q0_trials", ()),
            metric_trials=diag.get(
                f"fit_{target_metric}_trials",
                diag.get("fit_metric_trials", diag.get("fit_chi2_trials", ())),
            ),
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
            trials_xmin=_coerce_axis_limit(trials_xmin),
            trials_xmax=_coerce_axis_limit(trials_xmax),
            trials_ymin=_coerce_axis_limit(trials_ymin),
            trials_ymax=_coerce_axis_limit(trials_ymax),
            trials_xscale=trials_xscale,
            trials_yscale=trials_yscale,
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
    frequency_ghz: float | None = 17.0,
    diagnostics: dict[str, Any] | None = None,
    log_metrics: bool = False,
    log_q0: bool = False,
    zoom2best: int | None = None,
    trials_xmin: float | None = None,
    trials_xmax: float | None = None,
    trials_ymin: float | None = None,
    trials_ymax: float | None = None,
    trials_xscale: str | None = None,
    trials_yscale: str | None = None,
    common_map_scale: str | None = None,
    common_map_vmin: float | None = None,
    common_map_vmax: float | None = None,
    residual_map_scale: str | None = None,
    residual_map_vmin: float | None = None,
    residual_map_vmax: float | None = None,
    show_plot: bool = False,
    defer_show: bool = False,
    wcs_header_transform: Callable[[fits.Header], fits.Header] | None = None,
    load_blos: bool = False,
    blos_reference: tuple[np.ndarray, fits.Header] | None = None,
) -> Figure:
    """Render the legacy Q0 artifact panel and optionally save/show it.

    This compatibility wrapper preserves the public function used by the
    example scripts while delegating the actual panel construction to
    ``Q0ArtifactPanelFigure``.
    """

    if show_plot:
        import matplotlib.pyplot as plt

        # Interactive display needs a pyplot-managed figure so attributes such
        # as ``number`` and the GUI canvas exist consistently across backends.
        panel = Q0ArtifactPanelFigure(
            figure=plt.figure(figsize=(14.8, 9.2), constrained_layout=True)
        )
    else:
        panel = Q0ArtifactPanelFigure()
    output_path = None if out_png is None else Path(out_png)
    panel.update(
        model_path=Path(model_path),
        observed_noisy=np.asarray(observed_noisy, dtype=float),
        raw_modeled_best=np.asarray(raw_modeled_best, dtype=float),
        modeled_best=np.asarray(modeled_best, dtype=float),
        residual=np.asarray(residual, dtype=float),
        wcs_header=wcs_header,
        frequency_ghz=_optional_float(frequency_ghz),
        diagnostics=diagnostics,
        log_metrics=bool(log_metrics),
        log_q0=bool(log_q0),
        zoom2best=zoom2best,
        trials_xmin=trials_xmin,
        trials_xmax=trials_xmax,
        trials_ymin=trials_ymin,
        trials_ymax=trials_ymax,
        trials_xscale=trials_xscale,
        trials_yscale=trials_yscale,
        common_map_scale=common_map_scale,
        common_map_vmin=common_map_vmin,
        common_map_vmax=common_map_vmax,
        residual_map_scale=residual_map_scale,
        residual_map_vmin=residual_map_vmin,
        residual_map_vmax=residual_map_vmax,
        wcs_header_transform=wcs_header_transform,
        load_blos=bool(load_blos),
        blos_reference=blos_reference,
        out_png=output_path,
    )

    if show_plot:
        plt.figure(panel.figure.number)
        if defer_show:
            panel.figure.canvas.draw_idle()
        else:
            plt.show()

    return panel.figure
