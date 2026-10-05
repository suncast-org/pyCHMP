#!/usr/bin/env python3
"""Compare SSRT restoring beams: pyCHMP Python path vs SSW IDL ``MakeSSRTbeam``.

Supply an observation time, a classic SSRT map FITS (``DATE-OBS`` / ``TIME-OBS``),
and/or a classic SSRT map ``.sav`` (``map`` variable). Optionally compare against
a precomputed IDL beam FITS (e.g. map-object index 2 exported with ``writefits``):

  python compare_ssrt_beam_python_vs_idl.py \\
    --fits /path/to/I20110801_0313.fit

  python compare_ssrt_beam_python_vs_idl.py \\
    --time '2-Oct-2012 03:16:23.917' \\
    --idl-beam-fits /path/to/ssrt_idl_beam.fits

  python compare_ssrt_beam_python_vs_idl.py \\
    --map-sav /path/to/ssrt_map.sav \\
    --solp-angles idl

``--solp-angles idl`` (default for array parity): use live IDL ``GetSSRTangles``
for both beams. ``--solp-angles python``: Python ``get_ssrt_angles`` vs IDL beam
built with IDL angles (documents ephemeris residual).

Requires ``sswidl`` and gx_simulator (``GX_SIMULATOR`` or ``SSW``).
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
from scipy.io import readsav

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from pychmp.psf import sigma_arcsec_to_fwhm  # noqa: E402
from pychmp.ssrt_beam import (  # noqa: E402
    _parse_time,
    fit_ssrt_beam_ellipse,
    get_ssrt_angles,
    make_ssrt_beam,
    ssrt_time_from_header,
)


def _find_sswidl() -> Path:
    if os.environ.get("SSWIDL"):
        candidate = Path(os.environ["SSWIDL"]).expanduser()
        if candidate.is_file():
            return candidate
    which = subprocess.run(["bash", "-lc", "command -v sswidl"], capture_output=True, text=True)
    text = which.stdout.strip()
    if text:
        return Path(text)
    home = Path.home() / "scripts" / "sswidl"
    if home.is_file():
        return home
    raise FileNotFoundError("sswidl not found; set SSWIDL=/path/to/sswidl")


def _gx_beams_dir() -> Path:
    env_root = os.environ.get("GX_SIMULATOR") or os.environ.get("SSW_GX_SIMULATOR")
    if env_root:
        root = Path(env_root).expanduser()
    elif os.environ.get("SSW"):
        root = Path(os.environ["SSW"]).expanduser() / "packages" / "gx_simulator"
    else:
        raise FileNotFoundError("Set GX_SIMULATOR or SSW to locate MakeSSRTbeam.pro")
    beams = root / "beams"
    support = root / "support"
    if not (support / "MakeSSRTbeam.pro").is_file():
        raise FileNotFoundError(f"MakeSSRTbeam.pro not found under {support}")
    return beams


def _decode_bytes(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, np.ndarray) and value.dtype.kind in {"S", "O", "U"}:
        item = value.reshape(-1)[0]
        return _decode_bytes(item)
    return str(value).strip()


def time_from_map_sav(path: Path) -> str:
    payload = readsav(str(path), python_dict=True)
    if "map" not in payload:
        raise KeyError(f"{path} has no 'map' variable")
    m = payload["map"]
    return _decode_bytes(m["time"])


def time_from_ssrt_fits(path: Path) -> str:
    """Observation time from classic SSRT FITS (DATE-OBS + TIME-OBS), IDL anytim form."""
    with fits.open(path) as hdul:
        raw = ssrt_time_from_header(hdul[0].header)
    if raw is None:
        raise KeyError(f"{path} has no DATE-OBS/TIME-OBS (or equivalent)")
    t = _parse_time(raw).utc.datetime
    # Match prior compare artifacts: "1-Aug-2011 03:13:32.673"
    return (
        f"{t.day}-{t.strftime('%b')}-{t.year} "
        f"{t.hour:02d}:{t.minute:02d}:{t.second:02d}."
        f"{int(t.microsecond / 1000):03d}"
    )


def run_idl_ssrt_beam(time: str, marx: int, dx: float, out_fits: Path) -> dict[str, float]:
    sswidl = _find_sswidl()
    gx_beams = _gx_beams_dir()
    gx_root = gx_beams.parent
    pro = out_fits.with_suffix(".pro")
    log = out_fits.with_suffix(".idl.log")
    metrics = out_fits.with_suffix(".idl_metrics.txt")
    pro.write_text(
        "\n".join(
            [
                "pro ssrt_beam_dump, tstr, outf, metf, marx, dx",
                "  forward_function BeamFitSSRT",
                "  GetSSRTangles, tstr, dEW, dNS, pEW, pNS",
                "  MakeSSRTbeam, dEW, dNS, pEW, pNS, marx, marx, dx, dx, x, y, beam",
                "  FitBeam, x, y, beam, sx, sy, theta",
                "  writefits, outf, double(beam)",
                "  openw, 1, metf",
                "  printf, 1, 'dEW=', dEW",
                "  printf, 1, 'dNS=', dNS",
                "  printf, 1, 'pEW=', pEW",
                "  printf, 1, 'pNS=', pNS",
                "  printf, 1, 'sx=', sx",
                "  printf, 1, 'sy=', sy",
                "  printf, 1, 'theta_rad=', theta",
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
    env["IDL_PATH"] = (
        f"+{gx_beams}:+{gx_beams}/SSRT:+{gx_root}/support"
        + (f":{existing}" if existing else "")
    )
    cmd = (
        f".compile {gx_beams}/FitBeam.pro\n"
        f".compile {gx_beams}/gauss2drot.pro\n"
        f".compile {gx_root}/support/MakeSSRTbeam.pro\n"
        f".compile {gx_beams}/SSRT/GetSSRTangles.pro\n"
        f".compile {gx_beams}/SSRT/BeamFitSSRT.pro\n"
        f".compile {pro}\n"
        f"ssrt_beam_dump, '{time}', '{out_fits}', '{metrics}', {int(marx)}L, {float(dx)}\n"
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
    if proc.returncode != 0 or not out_fits.is_file() or not metrics.is_file():
        raise RuntimeError(f"IDL SSRT dump failed (rc={proc.returncode}). See {log}")
    out: dict[str, float] = {}
    for line in metrics.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, raw = line.split("=", 1)
        out[key.strip()] = float(raw.strip())
    return out


def _ordered(sx: float, sy: float, theta_deg: float) -> tuple[float, float, float]:
    if sx >= sy:
        smaj, smin, pa = sx, sy, theta_deg
    else:
        smaj, smin, pa = sy, sx, theta_deg + 90.0
    pa = ((pa + 90.0) % 180.0) - 90.0
    if pa <= -90.0:
        pa += 180.0
    return float(smaj), float(smin), float(pa)


def _metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = a - b
    peak = max(float(np.max(a)), float(np.max(b)), 1e-30)
    denom = float(np.sqrt(np.mean(a * a) * np.mean(b * b)))
    return {
        "max_abs_diff": float(np.max(np.abs(diff))),
        "rms_diff": float(np.sqrt(np.mean(diff * diff))),
        "corr": float(np.mean(a * b) / denom) if denom > 0 else float("nan"),
        "max_rel_to_peak": float(np.max(np.abs(diff)) / peak),
        "python_sum": float(np.sum(a)),
        "idl_sum": float(np.sum(b)),
    }


def make_plot(
    *,
    python_beam: np.ndarray,
    idl_beam: np.ndarray,
    dx: float,
    metrics: dict[str, float],
    py_fit: tuple[float, float, float],
    idl_fit: tuple[float, float, float],
    out_png: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    marx = python_beam.shape[0]
    half = marx // 2
    extent = [(-half - 0.5) * dx, (half + 0.5) * dx, (-half - 0.5) * dx, (half + 0.5) * dx]
    diff = python_beam - idl_beam
    vmax = max(float(np.max(python_beam)), float(np.max(idl_beam)))
    dmax = max(float(np.max(np.abs(diff))), 1e-30)
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.2), constrained_layout=True)
    fig.suptitle(
        f"{title}\nmax|Δ|={metrics['max_abs_diff']:.3e}  rms={metrics['rms_diff']:.3e}  "
        f"corr={metrics['corr']:.8f}"
    )
    for ax, arr, name, cmap, vmin, vmax_ in (
        (axes[0, 0], python_beam, "Python MakeSSRTbeam", "magma", 0.0, vmax),
        (axes[0, 1], idl_beam, "IDL MakeSSRTbeam", "magma", 0.0, vmax),
        (axes[0, 2], diff, "Python − IDL", "coolwarm", -dmax, dmax),
    ):
        im = ax.imshow(arr.T, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax_)
        ax.set_title(name)
        ax.set_xlabel("X [arcsec]")
        ax.set_ylabel("Y [arcsec]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    x_arc = (np.arange(marx) - half) * dx
    axes[1, 0].plot(x_arc, python_beam[:, half], label="Python", lw=2)
    axes[1, 0].plot(x_arc, idl_beam[:, half], "--", label="IDL", lw=2)
    axes[1, 0].set_title("Cut Y=0")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(x_arc, python_beam[half, :], label="Python", lw=2)
    axes[1, 1].plot(x_arc, idl_beam[half, :], "--", label="IDL", lw=2)
    axes[1, 1].set_title("Cut X=0")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 2].axis("off")
    py = _ordered(*py_fit)
    idl = _ordered(*idl_fit)
    axes[1, 2].text(
        0.02,
        0.98,
        "FitBeam ellipse (σ)\n"
        f"Python: σmaj={py[0]:.5f}\" σmin={py[1]:.5f}\" PA={py[2]:.3f}°\n"
        f"IDL:    σmaj={idl[0]:.5f}\" σmin={idl[1]:.5f}\" PA={idl[2]:.3f}°\n"
        f"Δσmaj={py[0]-idl[0]:.3e}\" Δσmin={py[1]-idl[1]:.3e}\" ΔPA={py[2]-idl[2]:.4f}°\n\n"
        f"max|Δ|={metrics['max_abs_diff']:.6e}\n"
        f"rms={metrics['rms_diff']:.6e}\n"
        f"corr={metrics['corr']:.10f}\n"
        f"FWHM maj Python={sigma_arcsec_to_fwhm(py[0]):.4f}\" "
        f"IDL={sigma_arcsec_to_fwhm(idl[0]):.4f}\"",
        transform=axes[1, 2].transAxes,
        va="top",
        family="monospace",
        fontsize=9,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--time", default=None, help="Observation time (anytim-like string)")
    p.add_argument("--map-sav", type=Path, default=None, help="Classic SSRT map .sav with 'map'")
    p.add_argument(
        "--fits",
        type=Path,
        default=None,
        help="Classic SSRT map FITS; derive time from DATE-OBS (+ TIME-OBS)",
    )
    p.add_argument("--idl-beam-fits", type=Path, default=None, help="Optional IDL beam FITS to compare")
    p.add_argument("--marx", type=int, default=50)
    p.add_argument("--dx", type=float, default=1.0, help="Beam pixel size arcsec (BeamFitSSRT uses 1)")
    p.add_argument(
        "--angles",
        choices=("idl", "python"),
        default="idl",
        help="Angles for the Python MakeSSRTbeam call (default: live IDL GetSSRTangles)",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args(argv)

    time = args.time
    if time is None and args.fits is not None:
        time = time_from_ssrt_fits(Path(args.fits).expanduser())
    if time is None and args.map_sav is not None:
        time = time_from_map_sav(Path(args.map_sav).expanduser())
    if not time:
        raise SystemExit("Provide --time, --fits, or --map-sav")

    out_dir = args.out_dir or Path(tempfile.mkdtemp(prefix="ssrt_beam_compare_"))
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"time={time!r}")
    print(f"marx={args.marx} dx={args.dx} angles={args.angles}")
    idl_fits = out_dir / f"ssrt_beam_idl_marx{args.marx}.fits"
    print(f"Running sswidl MakeSSRTbeam → {idl_fits}")
    idl_meta = run_idl_ssrt_beam(time, marx=int(args.marx), dx=float(args.dx), out_fits=idl_fits)
    idl_beam = np.asarray(fits.getdata(idl_fits), dtype=float).T

    if args.angles == "idl":
        d_ew, d_ns, p_ew, p_ns = (
            idl_meta["dEW"],
            idl_meta["dNS"],
            idl_meta["pEW"],
            idl_meta["pNS"],
        )
    else:
        ang = get_ssrt_angles(time)
        d_ew, d_ns, p_ew, p_ns = ang.d_ew_arcsec, ang.d_ns_arcsec, ang.p_ew_deg, ang.p_ns_deg
    print(f"angles used: dEW={d_ew} dNS={d_ns} pEW={p_ew} pNS={p_ns}")
    print(
        f"IDL angles:   dEW={idl_meta['dEW']} dNS={idl_meta['dNS']} "
        f"pEW={idl_meta['pEW']} pNS={idl_meta['pNS']}"
    )

    x, y, python_beam = make_ssrt_beam(
        d_ew, d_ns, p_ew, p_ns, nx=int(args.marx), ny=int(args.marx), dx_arcsec=float(args.dx), dy_arcsec=float(args.dx)
    )
    py_sx, py_sy, py_th = fit_ssrt_beam_ellipse(python_beam, x, y)
    metrics = _metrics(python_beam, idl_beam)
    print("Array comparison:")
    for k, v in metrics.items():
        print(f"  {k}={v}")

    # Optional extra compare to a stored restoring-beam FITS (e.g. ref map index 2).
    if args.idl_beam_fits is not None:
        stored = np.asarray(fits.getdata(Path(args.idl_beam_fits).expanduser()), dtype=float)
        if stored.ndim == 2:
            stored = stored.T
        extra = _metrics(python_beam, stored)
        print("vs --idl-beam-fits:")
        for k, v in extra.items():
            print(f"  {k}={v}")

    out_png = out_dir / f"ssrt_beam_python_vs_idl_marx{args.marx}.png"
    make_plot(
        python_beam=python_beam,
        idl_beam=idl_beam,
        dx=float(args.dx),
        metrics=metrics,
        py_fit=(py_sx, py_sy, float(np.rad2deg(py_th))),
        idl_fit=(idl_meta["sx"], idl_meta["sy"], float(np.rad2deg(idl_meta["theta_rad"]))),
        out_png=out_png,
        title=f"SSRT beam parity (marx={args.marx}) — {time}",
    )
    summary = {
        "time": time,
        "angles_mode": args.angles,
        "angles_python_used": {"dEW": d_ew, "dNS": d_ns, "pEW": p_ew, "pNS": p_ns},
        "angles_idl": {
            "dEW": idl_meta["dEW"],
            "dNS": idl_meta["dNS"],
            "pEW": idl_meta["pEW"],
            "pNS": idl_meta["pNS"],
        },
        "python_fit": {"sx": py_sx, "sy": py_sy, "theta_rad": py_th},
        "idl_fit": {
            "sx": idl_meta["sx"],
            "sy": idl_meta["sy"],
            "theta_rad": idl_meta["theta_rad"],
        },
        "array_metrics": metrics,
        "plot": str(out_png),
    }
    out_json = out_dir / f"ssrt_beam_python_vs_idl_marx{args.marx}.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_png}")
    print(f"Wrote {out_json}")

    # When using IDL angles, expect near-machine beam agreement.
    if args.angles == "idl" and (metrics["max_abs_diff"] > 1e-6 or metrics["corr"] < 1.0 - 1e-10):
        print("WARNING: unexpected residual for IDL-angle parity mode")
        return 2
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
