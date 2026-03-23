from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
from astropy.io import fits


METRIC_CHOICES = ("chi2", "rho2", "eta2")


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
    suffix = model_path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        return model_path if model_path.exists() else None
    if suffix != ".sav" or not model_path.exists():
        return None

    try:
        from gxrender.io import build_h5_from_sav
    except Exception:
        return None

    out_h5 = Path("/tmp") / f"pychmp_refmaps_{model_path.stem}.h5"
    if out_h5.exists():
        return out_h5
    try:
        build_h5_from_sav(model_path, out_h5, template_h5=None)
        return out_h5
    except Exception:
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


def _ordered_metric_names(target_metric: str) -> tuple[str, ...]:
    return (target_metric,) + tuple(name for name in METRIC_CHOICES if name != target_metric)


def _metric_summary_lines(diag: dict[str, Any], target_metric: str, fmt: Callable[[Any, str], str]) -> list[str]:
    lines: list[str] = []
    for name in _ordered_metric_names(target_metric):
        label = f"target[{name}]" if name == target_metric else name
        lines.append(f"{label:<12}: {fmt(diag.get(name), '.6e')}")
    return lines


def plot_q0_artifact_panel(
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
    show_plot: bool = False,
    defer_show: bool = False,
    wcs_header_transform: Callable[[fits.Header], fits.Header] | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        import sunpy.map
        from astropy import units as u
        from astropy.coordinates import SkyCoord
    except ModuleNotFoundError:
        return

    diag = diagnostics or {}
    header = wcs_header.copy()
    if wcs_header_transform is not None:
        header = wcs_header_transform(header)

    m_obs = sunpy.map.Map(observed_noisy, header)
    m_raw_mod = sunpy.map.Map(raw_modeled_best, header)
    m_mod = sunpy.map.Map(modeled_best, header)
    m_res = sunpy.map.Map(residual, header)

    blos = _load_blos_reference_map(model_path)

    q0_true = diag.get("q0_truth")
    q0_best = diag.get("q0_recovered")
    a_val = diag.get("a")
    b_val = diag.get("b")
    target_metric = str(diag.get("target_metric", "chi2"))
    target_metric_val = diag.get("target_metric_value", diag.get(target_metric))
    q0_trials = diag.get("fit_q0_trials")
    metric_trials = diag.get("fit_metric_trials", diag.get("fit_chi2_trials"))

    def _fmt(v: Any, fmt: str) -> str:
        try:
            return format(float(v), fmt)
        except Exception:
            return "n/a"

    def _add_panel_legend(ax, text: str) -> None:
        ax.legend(
            handles=[Line2D([], [], linestyle="none")],
            labels=[text],
            loc="lower left",
            handlelength=0,
            handletextpad=0,
            borderpad=0.35,
            fontsize=8,
            framealpha=0.85,
            facecolor="white",
            edgecolor="#cccccc",
        )

    q0_true_txt = _fmt(q0_true, ".6f")
    q0_best_txt = _fmt(q0_best, ".6f")
    a_txt = _fmt(a_val, ".3f")
    b_txt = _fmt(b_val, ".3f")

    psf_bmaj_val = diag.get("psf_bmaj_arcsec")
    psf_bmin_val = diag.get("psf_bmin_arcsec")
    psf_bpa_val = diag.get("psf_bpa_deg")
    noise_frac_val = diag.get("noise_frac")
    noise_std_val = diag.get("noise_std")

    if psf_bmaj_val is not None and psf_bmin_val is not None:
        bpa_part = f"  PA={_fmt(psf_bpa_val, '.1f')}\u00b0" if psf_bpa_val is not None else ""
        psf_legend = f"\nPSF: {_fmt(psf_bmaj_val, '.1f')}\"\u00d7{_fmt(psf_bmin_val, '.1f')}\"{bpa_part}"
    else:
        psf_legend = ""

    if noise_frac_val is not None:
        noise_legend = f"\nnoise: {_fmt(float(noise_frac_val) * 100.0, '.1f')}%"
        if noise_std_val is not None:
            noise_legend += f"  std={_fmt(noise_std_val, '.2e')}"
    else:
        noise_legend = ""

    fig = plt.figure(figsize=(14.8, 9.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    if blos is not None:
        blos_data, blos_hdr = blos
        if wcs_header_transform is not None:
            blos_hdr = wcs_header_transform(blos_hdr)
        m_blos = sunpy.map.Map(blos_data, blos_hdr)
        ax0 = fig.add_subplot(gs[0, 0], projection=m_blos)
        try:
            xc = float(header.get("CRVAL1", 0.0))
            yc = float(header.get("CRVAL2", 0.0))
            nx = int(header.get("NAXIS1", observed_noisy.shape[1]))
            ny = int(header.get("NAXIS2", observed_noisy.shape[0]))
            dx = float(header.get("CDELT1", 1.0))
            dy = float(header.get("CDELT2", 1.0))
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
        _add_panel_legend(ax0, f"a={a_txt}  b={b_txt}\nq0=n/a")
    else:
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.set_title("B_los Reference (Unavailable)")
        _add_panel_legend(ax0, f"a={a_txt}  b={b_txt}\nq0=n/a")
        ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1], projection=m_obs)
    im1 = m_obs.plot(axes=ax1, cmap="inferno")
    ax1.set_title(f"Observed (Noisy) @ {float(frequency_ghz):.2f} GHz")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    _add_panel_legend(ax1, f"a={a_txt}  b={b_txt}\nq0={q0_true_txt}{psf_legend}{noise_legend}")

    ax4 = fig.add_subplot(gs[0, 2], projection=m_raw_mod)
    im4 = m_raw_mod.plot(axes=ax4, cmap="inferno")
    ax4.set_title(f"Modeled Raw (Best Q0) @ {float(frequency_ghz):.2f} GHz")
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    _add_panel_legend(ax4, f"a={a_txt}  b={b_txt}\nq0={q0_best_txt}")

    ax2 = fig.add_subplot(gs[1, 0], projection=m_mod)
    im2 = m_mod.plot(axes=ax2, cmap="inferno")
    ax2.set_title(f"Modeled (Best Q0) @ {float(frequency_ghz):.2f} GHz")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    _add_panel_legend(ax2, f"a={a_txt}  b={b_txt}\nq0={q0_best_txt}{psf_legend}")

    ax3 = fig.add_subplot(gs[1, 1], projection=m_res)
    im3 = m_res.plot(axes=ax3, cmap="coolwarm")
    ax3.set_title(f"Residual (Model-Obs) @ {float(frequency_ghz):.2f} GHz")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    _add_panel_legend(ax3, f"a={a_txt}  b={b_txt}\nq0=true:{q0_true_txt} best:{q0_best_txt}{psf_legend}")

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_box_aspect(1)

    plotted_trials = False
    try:
        q0_arr = np.asarray(q0_trials, dtype=float)
        metric_arr = np.asarray(metric_trials, dtype=float)
        ok = q0_arr.ndim == 1 and metric_arr.ndim == 1 and q0_arr.size == metric_arr.size and q0_arr.size > 0
        if ok:
            order = np.argsort(q0_arr)
            ax5.plot(q0_arr[order], metric_arr[order], "-o", ms=4, lw=1.2, color="#2b6cb0", alpha=0.9, label=f"trial {target_metric}")
            ax5.scatter(q0_arr, metric_arr, s=22, color="#2b6cb0", alpha=0.55)
            if q0_best is not None:
                ax5.axvline(float(q0_best), color="#d62728", ls="--", lw=1.6)
            ymin = float(np.nanmin(metric_arr))
            ymax = float(np.nanmax(metric_arr))
            yref = float(target_metric_val) if target_metric_val is not None else ymin
            if q0_best is not None and np.isfinite(yref):
                ax5.scatter([float(q0_best)], [yref], color="#d62728", s=40, zorder=4)
            _title_parts = []
            if log_q0:
                _title_parts.append("log x")
            if log_metrics:
                _title_parts.append("log y")
            if zoom2best is not None and zoom2best > 0:
                _title_parts.append(f"zoom\u00b1{zoom2best}")
            title_suffix = f" ({', '.join(_title_parts)})" if _title_parts else ""
            ax5.set_title(f"{target_metric} vs q0 Trials{title_suffix}")
            ax5.set_xlabel("q0")
            ax5.set_ylabel(target_metric)
            ax5.grid(alpha=0.25)
            if log_metrics:
                positive = np.all(np.asarray(metric_arr, dtype=float) > 0.0)
                if positive:
                    ax5.set_yscale("log")
            if log_q0:
                positive_x = np.all(np.asarray(q0_arr, dtype=float) > 0.0)
                if positive_x:
                    ax5.set_xscale("log")
            if zoom2best is not None and zoom2best > 0 and q0_arr.size > 1:
                q0_sorted = q0_arr[order]
                best_sorted_pos = int(np.argmin(metric_arr[order]))
                lo = max(0, best_sorted_pos - zoom2best)
                hi = min(len(order) - 1, best_sorted_pos + zoom2best)
                q0_lo = float(q0_sorted[lo])
                q0_hi = float(q0_sorted[hi])
                margin = (q0_hi - q0_lo) * 0.08 if q0_hi > q0_lo else max(abs(q0_lo) * 0.05, 1e-9)
                ax5.set_xlim(q0_lo - margin, q0_hi + margin)
                visible_metric = (metric_arr[order])[lo : hi + 1]
                finite_m = visible_metric[np.isfinite(visible_metric)]
                if finite_m.size > 0:
                    vy_min = float(np.nanmin(finite_m))
                    vy_max = float(np.nanmax(finite_m))
                    if ax5.get_yscale() == "log" and vy_min > 0:
                        ax5.set_ylim(vy_min / 1.5, vy_max * 1.5)
                    else:
                        ypad = (vy_max - vy_min) * 0.12 if vy_max > vy_min else max(abs(vy_min) * 0.1, 1e-12)
                        ax5.set_ylim(vy_min - ypad, vy_max + ypad)
            legend_label = (
                f"q0={_fmt(q0_best, '.6f')}\n"
                f"target[{target_metric}]={_fmt(target_metric_val, '.6e')}\n"
                f"a={_fmt(a_val, '.3f')} b={_fmt(b_val, '.3f')}\n"
                f"q0_true={_fmt(q0_true, '.6f')}\n"
                f"trials={int(q0_arr.size)}"
            )
            ax5.plot([], [], " ", label=legend_label)
            ax5.legend(loc="best", fontsize=9, framealpha=0.9)
            if np.isfinite(ymin) and np.isfinite(ymax) and ymin == ymax:
                pad = max(1.0, abs(ymin) * 0.05)
                ax5.set_ylim(ymin - pad, ymax + pad)
            plotted_trials = True
    except Exception:
        plotted_trials = False

    if not plotted_trials:
        ax5.axis("off")
        summary_lines = [
            "Run Summary",
            f"freq [GHz] : {_fmt(frequency_ghz, '.2f')}",
            f"a          : {_fmt(a_val, '.3f')}",
            f"b          : {_fmt(b_val, '.3f')}",
            f"q0 true    : {_fmt(q0_true, '.6f')}",
            f"q0 best    : {_fmt(q0_best, '.6f')}",
            *_metric_summary_lines(diag, target_metric, _fmt),
            "trials     : unavailable",
        ]
        ax5.text(
            0.02,
            0.98,
            "\n".join(summary_lines),
            transform=ax5.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            family="monospace",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f6f6f6", "edgecolor": "#cccccc"},
        )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, facecolor="white")
    if show_plot and not defer_show:
        plt.show()
    if not defer_show:
        plt.close(fig)
