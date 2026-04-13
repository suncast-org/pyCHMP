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
    parser.add_argument("fits_file", type=Path)
    parser.add_argument("model_h5", type=Path)
    parser.add_argument("--ebtel-path", type=Path, required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--worker-counts", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--artifacts-root", type=Path, default=Path(tempfile.gettempdir()) / "pychmp_benchmarks")
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--scan-arg", action="append", default=[], help="Additional argument forwarded to examples/scan_ab_obs_map.py. May be repeated.")
    parser.add_argument("--a-values", default="0.0,0.3,0.6")
    parser.add_argument("--b-values", default="2.1,2.4,2.7")
    parser.add_argument("--q0-min", type=float, default=0.00001)
    parser.add_argument("--q0-max", type=float, default=0.001)
    parser.add_argument("--target-metric", default="chi2")
    parser.add_argument("--psf-bmaj-arcsec", type=float, default=5.77)
    parser.add_argument("--psf-bmin-arcsec", type=float, default=5.77)
    parser.add_argument("--psf-bpa-deg", type=float, default=-17.5)
    parser.add_argument("--psf-ref-frequency-ghz", type=float, default=17.0)
    parser.add_argument("--psf-scale-inverse-frequency", action="store_true", default=True)
    return parser.parse_args()


def _build_scan_command(args: argparse.Namespace, *, execution_policy: str, max_workers: int, repeat_index: int) -> tuple[list[str], Path]:
    artifact_dir = Path(args.artifacts_root)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_h5 = artifact_dir / f"scan_ab_obs_map_{execution_policy}_{max_workers:02d}_r{repeat_index:02d}.h5"
    cmd = [
        str(args.python_bin),
        "examples/scan_ab_obs_map.py",
        str(args.fits_file),
        str(args.model_h5),
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
        "--psf-bmaj-arcsec",
        str(float(args.psf_bmaj_arcsec)),
        "--psf-bmin-arcsec",
        str(float(args.psf_bmin_arcsec)),
        "--psf-bpa-deg",
        str(float(args.psf_bpa_deg)),
        "--psf-ref-frequency-ghz",
        str(float(args.psf_ref_frequency_ghz)),
        "--psf-scale-inverse-frequency",
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

    run_matrix: list[tuple[str, int]] = [("serial", 1)] + [("process-pool", worker_count) for worker_count in worker_counts]

    print(f"Benchmark root: {repo_root}")
    print(f"Inputs: fits={args.fits_file} model={args.model_h5} ebtel={args.ebtel_path}")
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
                    artifact_h5=str(artifact_h5),
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
