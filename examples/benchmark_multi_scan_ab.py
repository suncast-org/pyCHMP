from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import numpy as np

from pychmp.ab_search import multi_scan_ab


@dataclass(frozen=True)
class BenchmarkRow:
    workers: int
    execution_policy: str
    run_seconds: float
    median_seconds: float
    speedup_vs_serial: float


class BenchmarkRenderer:
    def __init__(self, observed: np.ndarray, true_q0: float, work_array_size: int, work_iterations: int) -> None:
        self._observed = observed
        self._true_q0 = true_q0
        self._work_array_size = int(work_array_size)
        self._work_iterations = int(work_iterations)

    def render(self, q0: float) -> np.ndarray:
        delta = float(q0) - self._true_q0
        scratch = np.full((self._work_array_size, self._work_array_size), delta, dtype=float)
        for _ in range(self._work_iterations):
            scratch = np.tanh(scratch + 0.001) + 0.5 * scratch
        correction = float(np.mean(scratch, dtype=float))
        return self._observed + delta + correction * 1.0e-9


class BenchmarkRendererFactory:
    def __init__(self, observed: np.ndarray, work_array_size: int, work_iterations: int) -> None:
        self._observed = observed
        self._work_array_size = int(work_array_size)
        self._work_iterations = int(work_iterations)

    @staticmethod
    def true_q0(a: float, b: float) -> float:
        return 2.0 + 0.5 * float(a) - 0.25 * float(b)

    def __call__(self, a: float, b: float) -> BenchmarkRenderer:
        return BenchmarkRenderer(
            self._observed,
            self.true_q0(a, b),
            self._work_array_size,
            self._work_iterations,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark 3x3 multi_scan_ab serial versus process-pool execution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--grid-size", type=int, default=3, help="Grid dimension for both a and b axes.")
    parser.add_argument("--repeats", type=int, default=3, help="Benchmark repeats per worker count.")
    parser.add_argument("--max-workers", type=int, default=9, help="Maximum process-pool worker count to benchmark.")
    parser.add_argument("--work-array-size", type=int, default=96, help="Synthetic render scratch-array size.")
    parser.add_argument("--work-iterations", type=int, default=240, help="Synthetic CPU work iterations per render.")
    parser.add_argument("--q0-min", type=float, default=0.1)
    parser.add_argument("--q0-max", type=float, default=10.0)
    return parser.parse_args()


def _run_once(*, grid_size: int, execution_policy: str, max_workers: int | None, work_array_size: int, work_iterations: int, q0_min: float, q0_max: float) -> float:
    observed = np.array([[10.0, 12.0], [14.0, 16.0]], dtype=float)
    sigma = np.ones_like(observed)
    factory = BenchmarkRendererFactory(observed, work_array_size=work_array_size, work_iterations=work_iterations)
    a_values = np.linspace(0.0, 1.0, int(grid_size), dtype=float)
    b_values = np.linspace(0.0, 2.0, int(grid_size), dtype=float)

    started = time.perf_counter()
    result = multi_scan_ab(
        factory,
        observed,
        sigma,
        a_values=a_values,
        b_values=b_values,
        q0_min=float(q0_min),
        q0_max=float(q0_max),
        target_metric="chi2",
        q0_start_grid=2.0,
        execution_policy=execution_policy,
        max_workers=max_workers,
        worker_chunksize=1,
    )
    elapsed = time.perf_counter() - started
    if not np.all(result.success):
        raise RuntimeError("benchmark scan did not converge for all points")
    return float(elapsed)


def main() -> int:
    args = parse_args()
    if args.grid_size <= 0:
        raise ValueError("--grid-size must be positive")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be positive")

    rows: list[BenchmarkRow] = []

    serial_runs = [
        _run_once(
            grid_size=int(args.grid_size),
            execution_policy="serial",
            max_workers=1,
            work_array_size=int(args.work_array_size),
            work_iterations=int(args.work_iterations),
            q0_min=float(args.q0_min),
            q0_max=float(args.q0_max),
        )
        for _ in range(int(args.repeats))
    ]
    serial_median = float(statistics.median(serial_runs))
    rows.append(
        BenchmarkRow(
            workers=1,
            execution_policy="serial",
            run_seconds=float(serial_runs[-1]),
            median_seconds=serial_median,
            speedup_vs_serial=1.0,
        )
    )

    for workers in range(1, int(args.max_workers) + 1):
        parallel_runs = [
            _run_once(
                grid_size=int(args.grid_size),
                execution_policy="process-pool",
                max_workers=int(workers),
                work_array_size=int(args.work_array_size),
                work_iterations=int(args.work_iterations),
                q0_min=float(args.q0_min),
                q0_max=float(args.q0_max),
            )
            for _ in range(int(args.repeats))
        ]
        parallel_median = float(statistics.median(parallel_runs))
        rows.append(
            BenchmarkRow(
                workers=int(workers),
                execution_policy="process-pool",
                run_seconds=float(parallel_runs[-1]),
                median_seconds=parallel_median,
                speedup_vs_serial=float(serial_median / parallel_median),
            )
        )

    print(f"Benchmark: {args.grid_size}x{args.grid_size} grid, repeats={args.repeats}, work_array_size={args.work_array_size}, work_iterations={args.work_iterations}")
    print("mode,workers,last_run_s,median_s,speedup_vs_serial")
    for row in rows:
        print(
            f"{row.execution_policy},{row.workers},{row.run_seconds:.6f},{row.median_seconds:.6f},{row.speedup_vs_serial:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
