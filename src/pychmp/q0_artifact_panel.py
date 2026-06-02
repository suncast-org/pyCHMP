from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import warnings

import h5py
import numpy as np
from astropy.io import fits
from astropy.wcs import FITSFixedWarning, WCS
from matplotlib.patches import Ellipse
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from .metrics import resolve_metrics_threshold_mask
from .obs_preprocessing import format_observation_shift_label
from .viewer_plot_style import (
    apply_q0_panel_trials_axis_style,
    apply_q0_solution_panel_layout,
    autoscale_trials_from_data,
    limits_frame_finite_data,
    normalize_axis_scale_choice,
    resolve_trial_metric_arrays,
    reserve_q0_trials_subplot,
)


METRIC_CHOICES = ("chi2", "rho2", "eta2")
_Q0_PANEL_TITLE_KW = {"fontsize": 9, "pad": 8}
_Q0_PANEL_BOTTOM_ROW_TITLE_KW = {"fontsize": 9, "pad": 10}
_Q0_PANEL_GRID_KW = {
    "left": 0.06,
    "right": 0.99,
    "top": 0.90,
    "bottom": 0.12,
    "hspace": 0.58,
    "wspace": 0.38,
    "height_ratios": (1.0, 1.05),
}
_Q0_COLORBAR_SHRINK = 0.88
_Q0_COLORBAR_PAD = 0.04
_Q0_COLORBAR_ASPECT = 22.0


def _attach_panel_colorbar(figure: Figure, axis: Any, mappable: Any) -> Any:
    colorbar = figure.colorbar(
        mappable,
        ax=axis,
        shrink=_Q0_COLORBAR_SHRINK,
        pad=_Q0_COLORBAR_PAD,
        aspect=_Q0_COLORBAR_ASPECT,
    )
    colorbar.ax.tick_params(length=2, labelsize=6, pad=1)
    return colorbar


def _compact_map_title(kind: str, spectral_label: str, *, trial_title: str = "") -> str:
    label = str(spectral_label or "").strip() or "map"
    title = f"{kind} @ {label}"
    trial = str(trial_title or "").strip()
    if trial:
        return f"{title}\n{trial}"
    return title


def _style_map_axis(axis: Any, *, title: str | None = None, grid_row: int = 0, grid_col: int = 0) -> None:
    if title is not None:
        title_kw = _Q0_PANEL_BOTTOM_ROW_TITLE_KW if int(grid_row) == 1 else _Q0_PANEL_TITLE_KW
        axis.set_title(title, **title_kw)
    show_x = int(grid_row) == 1
    show_y = int(grid_col) == 0
    try:
        x_coord = axis.coords[0]
        y_coord = axis.coords[1]
        x_coord.set_axislabel("Solar X" if show_x else "", minpad=1.2)
        y_coord.set_axislabel("Solar Y" if show_y else "", minpad=1.2)
        x_coord.set_ticklabel_visible(show_x)
        x_coord.set_ticks_visible(show_x)
        y_coord.set_ticklabel_visible(show_y)
        y_coord.set_ticks_visible(show_y)
        if hasattr(x_coord, "set_axislabel_visible"):
            x_coord.set_axislabel_visible(show_x)
        if hasattr(y_coord, "set_axislabel_visible"):
            y_coord.set_axislabel_visible(show_y)
        if show_x:
            x_coord.set_ticklabel_position("b")
            x_coord.set_axislabel_position("b")
            if hasattr(x_coord, "set_ticks_position"):
                x_coord.set_ticks_position("b")
        if show_y:
            y_coord.set_ticklabel_position("l")
            y_coord.set_axislabel_position("l")
            if hasattr(y_coord, "set_ticks_position"):
                y_coord.set_ticks_position("l")
        x_coord.set_ticklabel(size=6)
        y_coord.set_ticklabel(size=6)
    except Exception:
        axis.tick_params(
            labelsize=6,
            bottom=show_x,
            labelbottom=show_x,
            top=False,
            labeltop=False,
            left=show_y,
            labelleft=show_y,
            right=False,
            labelright=False,
        )
        axis.set_xlabel("Solar X" if show_x else "")
        axis.set_ylabel("Solar Y" if show_y else "")


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


def _euv_channel_token(channel: str) -> str:
    token = str(channel or "").strip().upper().replace(" ", "")
    if token.startswith("A") and token[1:].isdigit():
        return token[1:]
    return token.lower()


def _euv_channel_from_diagnostics(diagnostics: dict[str, Any] | None) -> str:
    diag = diagnostics or {}
    channel = str(diag.get("euv_channel", "")).strip()
    if channel:
        return channel
    wavelength_angstrom = _optional_float(diag.get("wavelength_angstrom"))
    if wavelength_angstrom is not None:
        rounded = round(float(wavelength_angstrom))
        if np.isclose(float(wavelength_angstrom), float(rounded), rtol=0.0, atol=1e-9):
            return str(int(rounded))
        return f"{float(wavelength_angstrom):.3f}".rstrip("0").rstrip(".")
    return ""


def _sunpy_colormap_registry() -> dict[str, Any]:
    try:
        from sunpy.visualization import colormaps as sunpy_colormaps
    except Exception:
        return {}
    return dict(getattr(sunpy_colormaps, "cmlist", {}) or {})


def _sunpy_euv_colormap_name(diagnostics: dict[str, Any] | None) -> str | None:
    diag = diagnostics or {}
    domain = str(diag.get("spectral_domain", "")).strip().lower()
    if domain not in {"euv", "uv"}:
        return None
    channel = _euv_channel_from_diagnostics(diag)
    if not channel:
        return None
    instrument = str(diag.get("euv_instrument") or diag.get("observation_instrument") or "AIA").strip()
    instrument_key = instrument.lower()
    channel_key = _euv_channel_token(channel)
    cmap_candidates: list[str] = []

    if instrument_key in {"aia", "sdo/aia", "sdoaia"}:
        cmap_candidates.append(f"sdoaia{channel_key}")
    elif instrument_key in {"euvia", "euvib", "stereo-a", "stereo-b", "stereo-a/euvi", "stereo-b/euvi"}:
        cmap_candidates.append(f"euvi{channel_key}")
    elif instrument_key in {"eui/fsi", "solo-fsi", "solar orbiterfsi", "solar orbiter/fsi"}:
        cmap_candidates.append(f"solar orbiterfsi{channel_key}")
    elif instrument_key in {"eui/hri", "solo-hri", "solar orbiterhri", "solar orbiter/hri"}:
        if channel_key == "1216":
            cmap_candidates.append("solar orbiterhri_lya1216")
        else:
            cmap_candidates.append(f"solar orbiterhri_euv{channel_key}")
    elif instrument_key in {"trace"}:
        cmap_candidates.append(f"trace{channel_key}")
    elif instrument_key in {"eit", "soho/eit", "sohoeit"}:
        cmap_candidates.append(f"sohoeit{channel_key}")
    elif instrument_key in {"sxt", "yohkoh/sxt", "yohkohsxt"}:
        if channel_key in {"a", "al", "openal", "thinal"}:
            cmap_candidates.append("yohkohsxtal")
        if channel_key in {"w", "wh", "openwh", "thinwh"}:
            cmap_candidates.append("yohkohsxtwh")

    available = _sunpy_colormap_registry()
    for candidate in cmap_candidates:
        if candidate in available:
            return candidate
    return None


def _intensity_colormap_for_panel(panel_name: str, diagnostics: dict[str, Any] | None) -> str:
    if panel_name == "residual":
        return "coolwarm"
    sunpy_name = _sunpy_euv_colormap_name(diagnostics)
    if sunpy_name is not None:
        return sunpy_name
    return "inferno"


def _resolved_psf_summary(diagnostics: dict[str, Any] | None) -> tuple[str, dict[str, float] | None]:
    diag = diagnostics or {}
    resolved_psf = dict(diag.get("resolved_psf") or {})
    source = str(resolved_psf.get("source") or diag.get("psf_source") or "").strip()
    kind = str(resolved_psf.get("kind") or "").strip().lower()

    def _beam_payload(source_name: str, payload: dict[str, Any]) -> tuple[str, dict[str, float] | None]:
        bmaj = _optional_float(payload.get("active_bmaj_arcsec"))
        bmin = _optional_float(payload.get("active_bmin_arcsec"))
        bpa = _optional_float(payload.get("active_bpa_deg"))
        if bmaj is None or bmin is None:
            bmaj = _optional_float(payload.get("psf_bmaj_arcsec"))
            bmin = _optional_float(payload.get("psf_bmin_arcsec"))
            bpa = _optional_float(payload.get("psf_bpa_deg")) if bpa is None else bpa
        if bmaj is None or bmin is None:
            return "", None
        source_prefix = f"{source_name} " if source_name else ""
        bpa_text = f" PA={bpa:.1f}°" if bpa is not None else ""
        return (
            f"\nPSF: {source_prefix}{bmaj:.1f}\"x{bmin:.1f}\"{bpa_text}",
            {
                "bmaj_arcsec": float(bmaj),
                "bmin_arcsec": float(bmin),
                "bpa_deg": 0.0 if bpa is None else float(bpa),
            },
        )

    if kind == "kernel":
        kernel_shape = resolved_psf.get("psf_kernel_shape")
        kernel_text = ""
        if isinstance(kernel_shape, (list, tuple)) and len(kernel_shape) == 2:
            try:
                kernel_text = f" {int(kernel_shape[0])}x{int(kernel_shape[1])}"
            except Exception:
                kernel_text = ""
        source_prefix = f"{source} " if source else ""
        return f"\nPSF: {source_prefix}kernel{kernel_text}", None
    beam_text, beam_payload = _beam_payload(source, resolved_psf)
    if beam_payload is not None:
        return beam_text, beam_payload
    legacy_payload = {
        "psf_bmaj_arcsec": diag.get("psf_bmaj_arcsec"),
        "psf_bmin_arcsec": diag.get("psf_bmin_arcsec"),
        "psf_bpa_deg": diag.get("psf_bpa_deg"),
    }
    return _beam_payload(source, legacy_payload)


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
        self.figure = figure or Figure(figsize=(15.0, 10.5))
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
        self._map_diagnostics: dict[str, Any] | None = None

    def set_mask_contours_visible(self, visible: bool) -> None:
        self._show_mask_contours = bool(visible)

    def mask_contours_visible(self) -> bool:
        return bool(self._show_mask_contours)

    def apply_autolayout(self) -> None:
        """Apply fixed subplot spacing tuned for WCS tick labels and titles."""
        apply_q0_solution_panel_layout(self.figure)
        if self._trials_ax is not None:
            reserve_q0_trials_subplot(self._trials_ax)

    def _get_mask(self, observed, modeled, diagnostics):
        header = getattr(self, "_map_wcs_header", None)
        return resolve_metrics_threshold_mask(
            observed,
            modeled,
            diagnostics,
            wcs_header=header,
        )

    def _clear_axis_mask_contours(self, ax: Any) -> None:
        previous = getattr(ax, "_mask_contours", ())
        if hasattr(previous, "remove"):
            previous.remove()
        else:
            for coll in previous:
                coll.remove()
        ax._mask_contours = ()

    def _draw_mask_contour_on_axis(self, ax: Any, mask: np.ndarray | None, *, show: bool) -> None:
        self._clear_axis_mask_contours(ax)
        if show and mask is not None:
            cs = ax.contour(mask.astype(float), levels=[0.5], colors="lime", linewidths=1.5, alpha=0.8)
            ax._mask_contours = cs

    def _draw_mask_contours(self, show, observed, modeled, diagnostics):
        mask = self._get_mask(observed, modeled, diagnostics)
        for name in ("observed", "raw_modeled", "modeled", "residual"):
            ax = self._common_axes.get(name)
            if ax is None:
                continue
            self._draw_mask_contour_on_axis(ax, mask, show=bool(show))
        blos_ax = self._blos_ax
        if blos_ax is not None and self._blos_image is not None and mask is not None and self._shape == mask.shape:
            self._draw_mask_contour_on_axis(blos_ax, mask, show=bool(show))
        elif blos_ax is not None:
            self._clear_axis_mask_contours(blos_ax)
    def _build_layout(self, header: fits.Header, shape: tuple[int, int]) -> None:
        self.figure.clear()
        grid_kw = dict(_Q0_PANEL_GRID_KW)
        height_ratios = grid_kw.pop("height_ratios")
        gs = self.figure.add_gridspec(
            2,
            3,
            figure=self.figure,
            height_ratios=height_ratios,
            hspace=grid_kw.pop("hspace"),
            wspace=grid_kw.pop("wspace"),
        )
        self._gs = gs
        self.figure.subplots_adjust(**grid_kw)

        common_specs = [
            ("observed", gs[0, 1], 0, 1),
            ("raw_modeled", gs[0, 2], 0, 2),
            ("modeled", gs[1, 0], 1, 0),
            ("residual", gs[1, 1], 1, 1),
        ]
        self._common_axes = {}
        self._common_images = {}
        self._common_colorbars = {}
        self._common_notes = {}
        self._common_titles = {}

        for name, slot, grid_row, grid_col in common_specs:
            axis = self.figure.add_subplot(slot, projection=_safe_wcs(header))
            axis._q0_grid_row = int(grid_row)
            axis._q0_grid_col = int(grid_col)
            cmap = _intensity_colormap_for_panel(name, self._map_diagnostics)
            image = axis.imshow(np.zeros(shape, dtype=float), origin="lower", cmap=cmap)
            colorbar = _attach_panel_colorbar(self.figure, axis, image)
            _style_map_axis(axis, grid_row=grid_row, grid_col=grid_col)
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
            axis._psf_overlay = None

        self._blos_ax = self.figure.add_subplot(gs[0, 0], projection=_safe_wcs(header))
        self._blos_ax._q0_grid_row = 0
        self._blos_ax._q0_grid_col = 0
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
        self._trials_ax.set_title("Trials", **_Q0_PANEL_TITLE_KW)
        self._trials_ax.tick_params(labelsize=7)
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
        image.set_cmap(_intensity_colormap_for_panel(name, self._map_diagnostics))
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
        grid_row = int(getattr(axis, "_q0_grid_row", 0))
        grid_col = int(getattr(axis, "_q0_grid_col", 0))
        self._common_titles[name] = title
        _style_map_axis(axis, title=title, grid_row=grid_row, grid_col=grid_col)
        self._common_notes[name].set_text(note)

    def _update_psf_overlay(self, *, name: str, diagnostics: dict[str, Any], header: fits.Header) -> None:
        axis = self._common_axes.get(name)
        if axis is None:
            return
        previous = getattr(axis, "_psf_overlay", None)
        if previous is not None:
            try:
                previous.remove()
            except Exception:
                pass
            axis._psf_overlay = None
        if name != "modeled":
            return
        _legend, beam_payload = _resolved_psf_summary(diagnostics)
        if beam_payload is None:
            return
        dx_arcsec = abs(_optional_float(header.get("CDELT1")) or 0.0)
        dy_arcsec = abs(_optional_float(header.get("CDELT2")) or 0.0)
        if dx_arcsec <= 0.0 or dy_arcsec <= 0.0:
            return
        bmaj_px = float(beam_payload["bmaj_arcsec"]) / dx_arcsec
        bmin_px = float(beam_payload["bmin_arcsec"]) / dy_arcsec
        if not np.isfinite(bmaj_px) or not np.isfinite(bmin_px) or bmaj_px <= 0.0 or bmin_px <= 0.0:
            return
        ny, nx = self._shape
        center_x = max(0.15 * float(nx), 0.5 * bmaj_px + 4.0)
        center_y = max(0.15 * float(ny), 0.5 * bmin_px + 4.0)
        overlay = Ellipse(
            (center_x, center_y),
            width=bmaj_px,
            height=bmin_px,
            angle=float(beam_payload["bpa_deg"]),
            facecolor="none",
            edgecolor="white",
            linewidth=1.6,
            alpha=0.95,
        )
        axis.add_patch(overlay)
        axis._psf_overlay = overlay

    def _show_blos_placeholder(self, *, a_text: str, b_text: str, message: str) -> None:
        if self._blos_colorbar is not None:
            self._blos_colorbar.remove()
            self._blos_colorbar = None
        if self._blos_ax is not None:
            self._blos_ax.remove()
        self._blos_ax = self.figure.add_subplot(self._gs[0, 0])
        self._blos_image = None
        self._blos_ax.set_title("B_los Reference", **_Q0_PANEL_TITLE_KW)
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
            self._blos_colorbar = _attach_panel_colorbar(self.figure, self._blos_ax, self._blos_image)
            _style_map_axis(self._blos_ax, title="B_los Reference", grid_row=0, grid_col=0)
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
        _style_map_axis(self._blos_ax, title="B_los Reference", grid_row=0, grid_col=0)
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
        trials_xlim: tuple[float, float] | None = None,
        trials_ylim: tuple[float, float] | None = None,
        trials_match_parent_view: bool = False,
        fmt: Callable[[Any, str], str],
        diagnostics: dict[str, Any],
    ) -> None:
        ax = self._trials_ax
        ax.clear()
        selected_trial_index = diagnostics.get("selected_trial_index")
        try:
            selected_trial_index = None if selected_trial_index is None else int(selected_trial_index)
        except Exception:
            selected_trial_index = None

        plotted_trials = False
        try:
            q0_arr, metric_arr = resolve_trial_metric_arrays(diagnostics, target_metric)
            if q0_arr.ndim != 1 or metric_arr.ndim != 1:
                q0_arr = np.asarray(q0_trials, dtype=float).reshape(-1)
                metric_arr = np.asarray(metric_trials, dtype=float).reshape(-1)
            if q0_arr.size and (metric_arr.size != q0_arr.size or not np.any(np.isfinite(metric_arr))):
                q0_fallback = np.asarray(q0_trials, dtype=float).reshape(-1)
                metric_fallback = np.asarray(metric_trials, dtype=float).reshape(-1)
                if metric_fallback.size == q0_fallback.size and np.any(np.isfinite(metric_fallback)):
                    q0_arr, metric_arr = q0_fallback, metric_fallback
            ok = q0_arr.ndim == 1 and metric_arr.ndim == 1 and q0_arr.size == metric_arr.size and q0_arr.size > 0
            if ok and not np.any(np.isfinite(metric_arr)):
                ok = False
            if ok:
                finite = np.isfinite(q0_arr) & np.isfinite(metric_arr)
                if not np.any(finite):
                    ok = False
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
                ax.grid(alpha=0.25)
                xscale_choice = normalize_axis_scale_choice(trials_xscale)
                yscale_choice = normalize_axis_scale_choice(trials_yscale)
                if xscale_choice == "log" and np.all(np.asarray(q0_arr, dtype=float) > 0.0):
                    ax.set_xscale("log")
                else:
                    ax.set_xscale("linear")
                if yscale_choice == "log" and np.all(np.asarray(metric_arr, dtype=float) > 0.0):
                    ax.set_yscale("log")
                else:
                    ax.set_yscale("linear")
                if log_metrics and yscale_choice != "log" and np.all(np.asarray(metric_arr, dtype=float) > 0.0):
                    ax.set_yscale("log")
                if log_q0 and xscale_choice != "log" and np.all(np.asarray(q0_arr, dtype=float) > 0.0):
                    ax.set_xscale("log")
                limits_locked = False
                if (
                    not trials_match_parent_view
                    and zoom2best is not None
                    and zoom2best > 0
                    and q0_arr.size > 1
                ):
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
                    limits_locked = True

                used_parent_limits = False
                if trials_match_parent_view and trials_xlim is not None and trials_ylim is not None:
                    x0, x1 = float(trials_xlim[0]), float(trials_xlim[1])
                    y0, y1 = float(trials_ylim[0]), float(trials_ylim[1])
                    if limits_frame_finite_data((x0, x1), (y0, y1), q0_arr, metric_arr):
                        if ax.get_xscale() == "log":
                            if x0 > 0 and x1 > 0 and x0 < x1:
                                ax.set_xlim(x0, x1)
                        elif x0 < x1:
                            ax.set_xlim(x0, x1)
                        if ax.get_yscale() == "log":
                            if y0 > 0 and y1 > 0 and y0 < y1:
                                ax.set_ylim(y0, y1)
                        elif y0 < y1:
                            ax.set_ylim(y0, y1)
                        used_parent_limits = True
                        limits_locked = True
                if not limits_locked:
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
                    manual_limits = (
                        (trials_xmin is not None and trials_xmax is not None and trials_xmin < trials_xmax)
                        or trials_xmin is not None
                        or trials_xmax is not None
                        or (trials_ymin is not None and trials_ymax is not None and trials_ymin < trials_ymax)
                        or trials_ymin is not None
                        or trials_ymax is not None
                    )
                    if not manual_limits:
                        autoscale_trials_from_data(ax, q0_arr, metric_arr)
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
                            loc="upper right",
                            fontsize=7,
                            framealpha=0.9,
                            handlelength=1.4,
                            borderpad=0.3,
                        )
                apply_q0_panel_trials_axis_style(ax, metric_name=target_metric)
                reserve_q0_trials_subplot(ax)
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
        trials_xlim: tuple[float, float] | None = None,
        trials_ylim: tuple[float, float] | None = None,
        trials_match_parent_view: bool = False,
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
        self._map_diagnostics = dict(diag)
        header = wcs_header.copy()
        if wcs_header_transform is not None:
            header = wcs_header_transform(header)
        self._map_wcs_header = header.copy()

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
        target_metric = str(diag.get("trials_display_metric") or diag.get("target_metric", "chi2"))
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

        psf_legend, _beam_payload = _resolved_psf_summary(diag)
        noise_frac_val = diag.get("noise_frac")
        noise_std_val = diag.get("noise_std")

        if noise_frac_val is not None:
            noise_legend = f"\nnoise: {fmt(float(noise_frac_val) * 100.0, '.1f')}%"
            if noise_std_val is not None:
                noise_legend += f"  std={fmt(noise_std_val, '.2e')}"
        else:
            noise_legend = ""

        try:
            selected_trial_idx = None if selected_trial_index is None else int(selected_trial_index)
        except Exception:
            selected_trial_idx = None
        shift_label = format_observation_shift_label(
            diagnostics=diag,
            trial_index=selected_trial_idx,
            fit_shift_x_trials=diag.get("fit_shift_x_trials"),
            fit_shift_y_trials=diag.get("fit_shift_y_trials"),
            fit_find_shift_valid_trials=diag.get("fit_find_shift_valid_trials"),
        )
        observed_title = _compact_map_title("Observed", spectral_label)
        observed_note = f"a={a_text}  b={b_text}\nq0={fmt(q0_true, '.6f')}{psf_legend}{noise_legend}"
        if shift_label:
            observed_note = f"{observed_note}\n{shift_label}"

        self._update_common_panel(
            "observed",
            observed_arr,
            title=observed_title,
            note=observed_note,
            scale=common_map_scale,
            vmin=_coerce_axis_limit(common_map_vmin),
            vmax=_coerce_axis_limit(common_map_vmax),
        )
        self._update_common_panel(
            "raw_modeled",
            np.asarray(raw_modeled_best, dtype=float),
            title=_compact_map_title("Modeled Raw", spectral_label, trial_title=trial_title),
            note=f"a={a_text}  b={b_text}\nq0={fmt(display_q0, '.6f')}{map_suffix}",
            scale=common_map_scale,
            vmin=_coerce_axis_limit(common_map_vmin),
            vmax=_coerce_axis_limit(common_map_vmax),
        )
        self._update_common_panel(
            "modeled",
            np.asarray(modeled_best, dtype=float),
            title=_compact_map_title("Modeled", spectral_label, trial_title=trial_title),
            note=f"a={a_text}  b={b_text}\nq0={fmt(display_q0, '.6f')}{psf_legend}{map_suffix}",
            scale=common_map_scale,
            vmin=_coerce_axis_limit(common_map_vmin),
            vmax=_coerce_axis_limit(common_map_vmax),
        )
        self._update_common_panel(
            "residual",
            np.asarray(residual, dtype=float),
            title=_compact_map_title("Residual", spectral_label, trial_title=trial_title),
            note=f"a={a_text}  b={b_text}\nq0=true:{fmt(q0_true, '.6f')} selected:{fmt(display_q0, '.6f')} best:{fmt(q0_best, '.6f')}{psf_legend}{map_suffix}",
            scale=residual_map_scale,
            vmin=_coerce_axis_limit(residual_map_vmin),
            vmax=_coerce_axis_limit(residual_map_vmax),
        )
        for panel_name in ("observed", "raw_modeled", "modeled", "residual"):
            self._update_psf_overlay(name=panel_name, diagnostics=diag, header=header)

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

        trials_q0, trials_metric = resolve_trial_metric_arrays(diag, target_metric)
        self._update_trials(
            q0_trials=trials_q0,
            metric_trials=trials_metric,
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
            trials_xlim=trials_xlim,
            trials_ylim=trials_ylim,
            trials_match_parent_view=trials_match_parent_view,
            fmt=fmt,
            diagnostics=diag,
        )

        observed_arr = np.asarray(observed_noisy, dtype=float)
        modeled_arr = np.asarray(modeled_best, dtype=float)
        self._draw_mask_contours(self._show_mask_contours, observed_arr, modeled_arr, diag)
        self.apply_autolayout()

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
        panel = Q0ArtifactPanelFigure(figure=plt.figure(figsize=(14.8, 9.2)))
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
