from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkRow:
    mode: str
    workers: int
    repeat: int
    elapsed_seconds: float
    exit_code: int
    artifact_h5: str


def _portable_artifact_path(artifact_h5: Path, *, bundle_root: Path | None) -> str:
    if bundle_root is None:
        return str(artifact_h5)
    try:
        return Path(os.path.relpath(artifact_h5, start=bundle_root)).as_posix()
    except Exception:
        return str(artifact_h5)


def _parse_worker_counts(text: str) -> list[int]:
    values = [item.strip() for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("expected a comma-separated list of worker counts")
    parsed = [int(value) for value in values]
    if any(value <= 0 for value in parsed):
        raise ValueError("worker counts must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark real-data 3x3 scan_ab_obs_map runs using pyGXrender-test-data inputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("fits_file", type=Path, nargs="?")
    parser.add_argument("model_h5", type=Path, nargs="?")
    parser.add_argument("--ebtel-path", type=Path, required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--worker-counts", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--artifacts-root", type=Path, default=Path(tempfile.gettempdir()) / "pychmp_benchmarks")
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--scan-arg", action="append", default=[], help="Additional argument forwarded to examples/scan_ab_obs_map.py. May be repeated.")
    parser.add_argument("--obs-source", choices=("external_fits", "model_refmap"), default="external_fits")
    parser.add_argument("--obs-path", type=Path, default=None)
    parser.add_argument("--obs-map-id", default=None)
    parser.add_argument("--euv-instrument", default=None)
    parser.add_argument("--euv-response-sav", type=Path, default=None)
    parser.add_argument("--model-h5", dest="model_h5_override", type=Path, default=None)
    parser.add_argument("--tr-mask-bmin-gauss", type=float, default=None)
    parser.add_argument("--metrics-mask-threshold", type=float, default=None)
    parser.add_argument("--metrics-mask-fits", type=Path, default=None)
    parser.add_argument("--a-values", default="0.0,0.3,0.6")
    parser.add_argument("--b-values", default="2.1,2.4,2.7")
    parser.add_argument("--q0-min", type=float, default=0.00001)
    parser.add_argument("--q0-max", type=float, default=0.001)
    parser.add_argument("--target-metric", default="chi2")
    parser.add_argument("--psf-bmaj-arcsec", type=float, default=5.77)
    parser.add_argument("--psf-bmin-arcsec", type=float, default=5.77)
    parser.add_argument("--psf-bpa-deg", type=float, default=-17.5)
    parser.add_argument("--psf-ref-frequency-ghz", type=float, default=17.0)
    parser.add_argument(
        "--psf-scale-inverse-frequency",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def _resolve_model_h5(args: argparse.Namespace) -> Path:
    model_h5 = args.model_h5_override or args.model_h5
    if model_h5 is None:
        raise ValueError("model_h5 is required; provide it positionally or via --model-h5")
    return Path(model_h5)


def _resolve_obs_path(args: argparse.Namespace) -> Path:
    obs_path = args.obs_path or args.fits_file
    if obs_path is None:
        raise ValueError("fits_file/--obs-path is required for --obs-source=external_fits")
    return Path(obs_path)


def _build_scan_command(args: argparse.Namespace, *, execution_policy: str, max_workers: int, repeat_index: int) -> tuple[list[str], Path]:
    artifact_dir = Path(args.artifacts_root)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    domain_tag = "euv" if str(args.obs_source) == "model_refmap" else "mw"
    artifact_h5 = artifact_dir / f"scan_ab_obs_map_{domain_tag}_{execution_policy}_{max_workers:02d}_r{repeat_index:02d}.h5"
    model_h5 = _resolve_model_h5(args)
    cmd = [str(args.python_bin), "examples/scan_ab_obs_map.py"]
    if str(args.obs_source) == "model_refmap":
        cmd.extend(
            [
                "--model-h5",
                str(model_h5),
                "--obs-source",
                "model_refmap",
            ]
        )
        if args.obs_map_id:
            cmd.extend(["--obs-map-id", str(args.obs_map_id)])
        if args.euv_instrument:
            cmd.extend(["--euv-instrument", str(args.euv_instrument)])
        if args.euv_response_sav is not None:
            cmd.extend(["--euv-response-sav", str(args.euv_response_sav)])
    else:
        obs_path = _resolve_obs_path(args)
        cmd.extend([str(obs_path), str(model_h5)])
    cmd.extend(
        [
            "--ebtel-path",
            str(args.ebtel_path),
            "--a-values",
            str(args.a_values),
            "--b-values",
            str(args.b_values),
            "--q0-min",
            str(float(args.q0_min)),
            "--q0-max",
            str(float(args.q0_max)),
            "--target-metric",
            str(args.target_metric),
            "--adaptive-bracketing",
            "--artifact-h5",
            str(artifact_h5),
            "--execution-policy",
            str(execution_policy),
            "--max-workers",
            str(int(max_workers)),
            "--worker-chunksize",
            "1",
            "--no-viewer",
            "--no-grid-png",
            "--no-point-png",
            "--no-progress",
            "--no-spinner",
        ]
    )
    if str(args.obs_source) == "external_fits":
        cmd.extend(
            [
                "--psf-bmaj-arcsec",
                str(float(args.psf_bmaj_arcsec)),
                "--psf-bmin-arcsec",
                str(float(args.psf_bmin_arcsec)),
                "--psf-bpa-deg",
                str(float(args.psf_bpa_deg)),
                "--psf-ref-frequency-ghz",
                str(float(args.psf_ref_frequency_ghz)),
            ]
        )
        if bool(args.psf_scale_inverse_frequency):
            cmd.append("--psf-scale-inverse-frequency")
    if args.tr_mask_bmin_gauss is not None:
        cmd.extend(["--tr-mask-bmin-gauss", str(float(args.tr_mask_bmin_gauss))])
    if args.metrics_mask_threshold is not None:
        cmd.extend(["--metrics-mask-threshold", str(float(args.metrics_mask_threshold))])
    if args.metrics_mask_fits is not None:
        cmd.extend(["--metrics-mask-fits", str(args.metrics_mask_fits)])
    for extra_arg in args.scan_arg:
        cmd.append(str(extra_arg))
    return cmd, artifact_h5


def _run_once(cmd: list[str], *, cwd: Path) -> tuple[float, int]:
    env = os.environ.copy()
    env["PYCHMP_NO_AUTO_VIEWER"] = "1"
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, check=False)
    elapsed = time.perf_counter() - started
    return float(elapsed), int(proc.returncode)


def main() -> int:
    args = parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    worker_counts = _parse_worker_counts(args.worker_counts)
    repo_root = Path(__file__).resolve().parents[1]
    rows: list[BenchmarkRow] = []
    model_h5 = _resolve_model_h5(args)
    if str(args.obs_source) == "model_refmap":
        if not args.obs_map_id:
            raise ValueError("--obs-map-id is required for --obs-source=model_refmap")
        if args.euv_response_sav is None:
            raise ValueError("--euv-response-sav is required for --obs-source=model_refmap")
    else:
        obs_path = _resolve_obs_path(args)
        if not obs_path.is_file():
            raise FileNotFoundError(f"observational FITS file not found: {obs_path}")
    if not model_h5.is_file():
        raise FileNotFoundError(f"model H5 file not found: {model_h5}")
    if args.metrics_mask_fits is not None and not Path(args.metrics_mask_fits).is_file():
        raise FileNotFoundError(f"metrics-mask FITS file not found: {args.metrics_mask_fits}")
    if args.euv_response_sav is not None and not Path(args.euv_response_sav).is_file():
        raise FileNotFoundError(f"EUV response SAV file not found: {args.euv_response_sav}")

    run_matrix: list[tuple[str, int]] = [("serial", 1)] + [("process-pool", worker_count) for worker_count in worker_counts]

    print(f"Benchmark root: {repo_root}")
    if str(args.obs_source) == "model_refmap":
        print(
            "Inputs: "
            f"obs_source=model_refmap obs_map_id={args.obs_map_id} "
            f"model={model_h5} ebtel={args.ebtel_path} response={args.euv_response_sav}"
        )
    else:
        print(f"Inputs: fits={_resolve_obs_path(args)} model={model_h5} ebtel={args.ebtel_path}")
    print(f"Matrix: repeats={args.repeats} runs={', '.join(f'{mode}:{workers}' for mode, workers in run_matrix)}")

    for mode, workers in run_matrix:
        for repeat_index in range(1, int(args.repeats) + 1):
            cmd, artifact_h5 = _build_scan_command(
                args,
                execution_policy=mode,
                max_workers=workers,
                repeat_index=repeat_index,
            )
            print("Command:", " ".join(f'\"{part}\"' if " " in part else part for part in cmd))
            if args.dry_run:
                continue
            elapsed, exit_code = _run_once(cmd, cwd=repo_root)
            rows.append(
                BenchmarkRow(
                    mode=mode,
                    workers=int(workers),
                    repeat=int(repeat_index),
                    elapsed_seconds=float(elapsed),
                    exit_code=int(exit_code),
                    artifact_h5=_portable_artifact_path(artifact_h5, bundle_root=args.csv_out.parent if args.csv_out is not None else None),
                )
            )
            print(
                f"Result: mode={mode} workers={workers} repeat={repeat_index} "
                f"elapsed={elapsed:.3f}s exit_code={exit_code} artifact={artifact_h5}"
            )
            if exit_code != 0:
                raise RuntimeError(f"benchmark run failed for mode={mode} workers={workers} repeat={repeat_index}")

    if args.dry_run:
        return 0

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["mode", "workers", "repeat", "elapsed_seconds", "exit_code", "artifact_h5"])
            for row in rows:
                writer.writerow([row.mode, row.workers, row.repeat, f"{row.elapsed_seconds:.6f}", row.exit_code, row.artifact_h5])
        print(f"CSV report: {args.csv_out}")

    print("mode,workers,repeat,elapsed_seconds,exit_code,artifact_h5")
    for row in rows:
        print(f"{row.mode},{row.workers},{row.repeat},{row.elapsed_seconds:.6f},{row.exit_code},{row.artifact_h5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
