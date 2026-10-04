#!/usr/bin/env python3
"""Compare NORH clean beams: pyCHMP Python path vs SSW IDL ``norh_beam``.

Pass any Nobeyama IFZ/FITS product with a standard NORH header:

  python compare_norh_ifz_beam_python_vs_idl.py --ifz /path/to/ifzYYMMDD_HHMMSS

Or set ``PYCHMP_NORH_IFZ``. P-angle for the Python beam is taken from:

* ``--solp idl`` (default): live ``get_rb0p(/pangle)`` returned by the IDL run
* ``--solp header``: ``SOLP`` from the FITS header (degrees → radians)

Requires local ``sswidl`` and gx_simulator ``FitBeam`` / ``Gauss2Drot`` on
``IDL_PATH`` (override package root with ``GX_SIMULATOR``).

Writes a comparison PNG and JSON metrics summary under ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

# Allow running from a source checkout without install.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from pychmp.norh_beam import (  # noqa: E402
    beam_fit_norh_from_header,
    fit_norh_beam_ellipse,
    make_norh_beam,
    norh_index_from_header,
)
from pychmp.psf import sigma_arcsec_to_fwhm  # noqa: E402


def _resolve_ifz(path: Path | None) -> Path:
    env = os.environ.get("PYCHMP_NORH_IFZ")
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path).expanduser())
    if env:
        candidates.append(Path(env).expanduser())
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "NORH IFZ/FITS not found. Pass --ifz /path/to/file or set PYCHMP_NORH_IFZ."
    )


def _find_sswidl() -> Path:
    for candidate in (
        Path(os.environ["SSWIDL"]) if os.environ.get("SSWIDL") else None,
        Path.home() / "scripts" / "sswidl",
    ):
        if candidate is not None and candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    which = subprocess.run(["bash", "-lc", "command -v sswidl"], capture_output=True, text=True)
    text = which.stdout.strip()
    if text:
        return Path(text)
    raise FileNotFoundError("sswidl not found on PATH; set SSWIDL=/path/to/sswidl")


def _gx_beams_dir() -> Path:
    env_root = os.environ.get("GX_SIMULATOR") or os.environ.get("SSW_GX_SIMULATOR")
    if env_root:
        root = Path(env_root).expanduser()
    elif os.environ.get("SSW"):
        root = Path(os.environ["SSW"]).expanduser() / "packages" / "gx_simulator"
    else:
        raise FileNotFoundError(
            "Set GX_SIMULATOR=/path/to/gx_simulator (or SSW) so FitBeam.pro can be found"
        )
    beams = root / "beams"
    if not (beams / "FitBeam.pro").is_file():
        raise FileNotFoundError(
            f"gx_simulator FitBeam.pro not found under {beams}. "
            "Set GX_SIMULATOR=/path/to/gx_simulator"
        )
    return beams


def run_idl_norh_beam(ifz: Path, marx: int, out_fits: Path) -> dict[str, float]:
    """Run IDL ``norh_beam`` + ``FitBeam`` and write the beam to ``out_fits``."""
    sswidl = _find_sswidl()
    gx_beams = _gx_beams_dir()
    pro = out_fits.with_suffix(".pro")
    log = out_fits.with_suffix(".idl.log")
    metrics_txt = out_fits.with_suffix(".idl_metrics.txt")
    pro.write_text(
        "\n".join(
            [
                "pro norh_ifz_dump, fname, outf, metf, marx",
                "  forward_function norh_beam, get_rb0p",
                "  norh_rd_img, fname, index, data",
                "  solp = get_rb0p(index, /pangle, /quiet)",
                "  beam = norh_beam(index, marx=marx)",
                "  spp = index.norh.sec_per_pix",
                "  x = (dindgen(marx) - long(marx)/2) * spp",
                "  y = (dindgen(marx) - long(marx)/2) * spp",
                "  FitBeam, x, y, beam, sx, sy, theta",
                "  writefits, outf, double(beam)",
                "  openw, 1, metf",
                "  printf, 1, 'solp_rad=', solp",
                "  printf, 1, 'sec_per_pix=', spp",
                "  printf, 1, 'sec_per_pix_dty=', index.norh.sec_per_pix_dty",
                "  printf, 1, 'efl=', index.norh.efl_in_pix_dty",
                "  printf, 1, 'sx=', sx",
                "  printf, 1, 'sy=', sy",
                "  printf, 1, 'theta_rad=', theta",
                "  printf, 1, 'theta_deg=', theta/!dtor",
                "  printf, 1, 'beam_max=', max(beam)",
                "  printf, 1, 'beam_sum=', total(double(beam))",
                "  close, 1",
                "end",
                "",
            ]
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    existing = env.get("IDL_PATH", "")
    env["IDL_PATH"] = f"+{gx_beams}:+{gx_beams}/NORH" + (f":{existing}" if existing else "")
    cmd = (
        f".compile {gx_beams}/FitBeam.pro\n"
        f".compile {gx_beams}/gauss2drot.pro\n"
        f".compile {pro}\n"
        f"norh_ifz_dump, '{ifz}', '{out_fits}', '{metrics_txt}', {int(marx)}L\n"
        "exit\n"
    )
    proc = subprocess.run(
        [str(sswidl)],
        input=cmd,
        text=True,
        capture_output=True,
        env=env,
        cwd=str(out_fits.parent),
    )
    log.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0 or not out_fits.is_file() or not metrics_txt.is_file():
        raise RuntimeError(
            f"IDL norh_beam dump failed (rc={proc.returncode}). See {log}"
        )
    metrics: dict[str, float] = {}
    for line in metrics_txt.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, raw = line.split("=", 1)
        metrics[key.strip()] = float(raw.strip())
    return metrics


def _ordered_major_minor(sx: float, sy: float, theta_deg: float) -> tuple[float, float, float]:
    if sx >= sy:
        smaj, smin, pa = sx, sy, theta_deg
    else:
        smaj, smin, pa = sy, sx, theta_deg + 90.0
    pa = ((pa + 90.0) % 180.0) - 90.0
    if pa <= -90.0:
        pa += 180.0
    return float(smaj), float(smin), float(pa)


def _beam_metrics(python_beam: np.ndarray, idl_beam: np.ndarray) -> dict[str, float]:
    a = np.asarray(python_beam, dtype=float)
    b = np.asarray(idl_beam, dtype=float)
    diff = a - b
    absdiff = np.abs(diff)
    denom = float(np.sqrt(np.mean(a * a) * np.mean(b * b)))
    corr = float(np.mean(a * b) / denom) if denom > 0 else float("nan")
    peak = max(float(np.max(a)), float(np.max(b)), 1e-30)
    return {
        "max_abs_diff": float(np.max(absdiff)),
        "rms_diff": float(np.sqrt(np.mean(diff * diff))),
        "mean_abs_diff": float(np.mean(absdiff)),
        "max_rel_to_peak": float(np.max(absdiff) / peak),
        "rms_rel_to_peak": float(np.sqrt(np.mean(diff * diff)) / peak),
        "corr": corr,
        "python_sum": float(np.sum(a)),
        "idl_sum": float(np.sum(b)),
        "python_max": float(np.max(a)),
        "idl_max": float(np.max(b)),
    }


def make_comparison_plot(
    *,
    python_beam: np.ndarray,
    idl_beam: np.ndarray,
    spp: float,
    marx: int,
    metrics: dict[str, float],
    py_fit: tuple[float, float, float],
    idl_fit: tuple[float, float, float],
    out_png: Path,
    title_prefix: str,
) -> None:
    import matplotlib.pyplot as plt

    half = marx // 2
    extent = [
        (-half - 0.5) * spp,
        (half + 0.5) * spp,
        (-half - 0.5) * spp,
        (half + 0.5) * spp,
    ]
    diff = python_beam - idl_beam
    vmax = max(float(np.max(python_beam)), float(np.max(idl_beam)))
    dmax = max(float(np.max(np.abs(diff))), 1e-30)

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.2), constrained_layout=True)
    fig.suptitle(
        f"{title_prefix}\n"
        f"max|Δ|={metrics['max_abs_diff']:.3e}  "
        f"rms={metrics['rms_diff']:.3e}  "
        f"corr={metrics['corr']:.8f}",
        fontsize=12,
    )

    im0 = axes[0, 0].imshow(
        python_beam.T, origin="lower", extent=extent, cmap="magma", vmin=0.0, vmax=vmax
    )
    axes[0, 0].set_title("Python ``make_norh_beam``")
    axes[0, 0].set_xlabel("X [arcsec]")
    axes[0, 0].set_ylabel("Y [arcsec]")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(
        idl_beam.T, origin="lower", extent=extent, cmap="magma", vmin=0.0, vmax=vmax
    )
    axes[0, 1].set_title("IDL ``norh_beam``")
    axes[0, 1].set_xlabel("X [arcsec]")
    axes[0, 1].set_ylabel("Y [arcsec]")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[0, 2].imshow(
        diff.T, origin="lower", extent=extent, cmap="coolwarm", vmin=-dmax, vmax=dmax
    )
    axes[0, 2].set_title("Python − IDL")
    axes[0, 2].set_xlabel("X [arcsec]")
    axes[0, 2].set_ylabel("Y [arcsec]")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    cx = cy = half
    x_arc = (np.arange(marx) - half) * spp
    axes[1, 0].plot(x_arc, python_beam[:, cy], label="Python", lw=2)
    axes[1, 0].plot(x_arc, idl_beam[:, cy], "--", label="IDL", lw=2)
    axes[1, 0].set_title("Cut through center (Y=0)")
    axes[1, 0].set_xlabel("X [arcsec]")
    axes[1, 0].set_ylabel("beam")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(x_arc, python_beam[cx, :], label="Python", lw=2)
    axes[1, 1].plot(x_arc, idl_beam[cx, :], "--", label="IDL", lw=2)
    axes[1, 1].set_title("Cut through center (X=0)")
    axes[1, 1].set_xlabel("Y [arcsec]")
    axes[1, 1].set_ylabel("beam")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].axis("off")
    py_smaj, py_smin, py_pa = _ordered_major_minor(*py_fit)
    id_smaj, id_smin, id_pa = _ordered_major_minor(*idl_fit)
    text = (
        "FitBeam ellipse (σ arcsec → ordered major/minor)\n"
        f"Python: σmaj={py_smaj:.6f}\"  σmin={py_smin:.6f}\"  PA={py_pa:.4f}°\n"
        f"IDL:    σmaj={id_smaj:.6f}\"  σmin={id_smin:.6f}\"  PA={id_pa:.4f}°\n"
        f"Δσmaj={py_smaj - id_smaj:.3e}\"  Δσmin={py_smin - id_smin:.3e}\"  "
        f"ΔPA={py_pa - id_pa:.4f}°\n\n"
        "Array metrics\n"
        f"max|Δ| = {metrics['max_abs_diff']:.6e}\n"
        f"rms(Δ) = {metrics['rms_diff']:.6e}\n"
        f"max|Δ|/peak = {metrics['max_rel_to_peak']:.6e}\n"
        f"corr = {metrics['corr']:.10f}\n"
        f"Σ Python={metrics['python_sum']:.6f}  Σ IDL={metrics['idl_sum']:.6f}\n"
        f"FWHM maj Python={sigma_arcsec_to_fwhm(py_smaj):.4f}\"  "
        f"IDL={sigma_arcsec_to_fwhm(id_smaj):.4f}\""
    )
    axes[1, 2].text(
        0.02,
        0.98,
        text,
        transform=axes[1, 2].transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ifz",
        type=Path,
        default=None,
        help="Path to a NORH IFZ/FITS file (or set PYCHMP_NORH_IFZ)",
    )
    parser.add_argument("--marx", type=int, default=51, help="Beam array size (odd)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for PNG/JSON (default: tempfile)",
    )
    parser.add_argument(
        "--solp",
        choices=("idl", "header"),
        default="idl",
        help=(
            "P-angle for the Python beam: live IDL get_rb0p (default) "
            "or FITS header SOLP"
        ),
    )
    args = parser.parse_args(argv)

    ifz = _resolve_ifz(args.ifz)
    marx = int(args.marx)
    if marx % 2 == 0:
        raise SystemExit("--marx must be odd")

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="norh_ifz_compare_"))
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    header = fits.getheader(ifz)
    index = norh_index_from_header(header)

    print(f"IFZ: {ifz}")
    print(f"marx={marx}  solp_source={args.solp}")
    print(f"out_dir: {out_dir}")

    # --- IDL path first (also supplies live get_rb0p when --solp idl) ---
    idl_fits = out_dir / f"norh_beam_idl_marx{marx}.fits"
    print(f"Running sswidl norh_beam → {idl_fits}")
    idl_metrics = run_idl_norh_beam(ifz, marx=marx, out_fits=idl_fits)
    idl_beam = np.asarray(fits.getdata(idl_fits), dtype=float)
    # Astropy returns FITS arrays as (NAXIS2, NAXIS1); IDL wrote beam[i,j] with
    # i=NAXIS1 (x). Transpose back to IDL (x, y) layout to match make_norh_beam.
    if idl_beam.ndim == 2:
        idl_beam = idl_beam.T

    if args.solp == "idl":
        if "solp_rad" not in idl_metrics:
            raise RuntimeError("IDL metrics missing solp_rad")
        solp = float(idl_metrics["solp_rad"])
    else:
        solp = float(index["solp_rad"])
    print(f"solp_rad={solp:.8f} (source={args.solp})")

    # --- Python path (same P-angle choice as above) ---
    python_beam = make_norh_beam(
        sec_per_pix=float(index["sec_per_pix"]),
        sec_per_pix_dty=float(index["sec_per_pix_dty"]),
        efl_in_pix_dty=float(index["efl_in_pix_dty"]),
        solp_rad=solp,
        pmat=tuple(index["pmat"]),  # type: ignore[arg-type]
        marx=marx,
        ns2ew=float(index["ns2ew"]),
    )
    if python_beam.shape != idl_beam.shape:
        raise RuntimeError(
            f"shape mismatch: python {python_beam.shape} vs idl {idl_beam.shape}"
        )
    py_sx, py_sy, py_theta = fit_norh_beam_ellipse(
        python_beam, sec_per_pix=float(index["sec_per_pix"])
    )
    py_params = beam_fit_norh_from_header(header, marx=marx, solp_rad=solp)
    print(
        f"Python FitBeam: sx={py_sx:.7f} sy={py_sy:.7f} "
        f"theta_rad={py_theta:.7f} (header-path sx={py_params.sigma_x_arcsec:.7f})"
    )
    print(
        f"IDL FitBeam: sx={idl_metrics['sx']:.7f} sy={idl_metrics['sy']:.7f} "
        f"theta_rad={idl_metrics['theta_rad']:.7f}"
    )

    metrics = _beam_metrics(python_beam, idl_beam)
    print("Array comparison:")
    for key, value in metrics.items():
        print(f"  {key}={value}")

    out_png = out_dir / f"norh_ifz_beam_python_vs_idl_marx{marx}.png"
    make_comparison_plot(
        python_beam=python_beam,
        idl_beam=idl_beam,
        spp=float(index["sec_per_pix"]),
        marx=marx,
        metrics=metrics,
        py_fit=(py_sx, py_sy, float(np.rad2deg(py_theta))),
        idl_fit=(
            float(idl_metrics["sx"]),
            float(idl_metrics["sy"]),
            float(idl_metrics["theta_deg"]),
        ),
        out_png=out_png,
        title_prefix=f"NORH beam parity (marx={marx}) — {ifz.name}",
    )
    print(f"Wrote plot: {out_png}")

    summary = {
        "ifz": str(ifz),
        "marx": marx,
        "solp_source": args.solp,
        "solp_rad_python": solp,
        "solp_rad_idl": idl_metrics.get("solp_rad"),
        "solp_rad_header": float(index["solp_rad"]),
        "python_fit": {
            "sx": py_sx,
            "sy": py_sy,
            "theta_rad": py_theta,
            "theta_deg": float(np.rad2deg(py_theta)),
        },
        "idl_fit": {
            "sx": idl_metrics["sx"],
            "sy": idl_metrics["sy"],
            "theta_rad": idl_metrics["theta_rad"],
            "theta_deg": idl_metrics["theta_deg"],
        },
        "array_metrics": metrics,
        "plot": str(out_png),
        "idl_beam_fits": str(idl_fits),
    }
    out_json = out_dir / f"norh_ifz_beam_python_vs_idl_marx{marx}.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote metrics: {out_json}")

    if metrics["max_abs_diff"] > 1e-6 or metrics["corr"] < 1.0 - 1e-10:
        print(
            "WARNING: larger-than-expected residual "
            f"(max|Δ|={metrics['max_abs_diff']:.3e}, corr={metrics['corr']:.10f})"
        )
        return 2
    print("PASS: Python and IDL beams agree within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
