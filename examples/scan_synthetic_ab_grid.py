from __future__ import annotations

import numpy as np

from pychmp.ab_search import multi_scan_ab


class SyntheticABRenderer:
    def __init__(self, observed: np.ndarray, true_q0: float) -> None:
        self._observed = observed
        self._true_q0 = true_q0

    def render(self, q0: float) -> np.ndarray:
        return self._observed + (float(q0) - self._true_q0)


class SyntheticABRendererFactory:
    def __init__(self, observed: np.ndarray) -> None:
        self._observed = observed

    @staticmethod
    def true_q0(a: float, b: float) -> float:
        return 2.0 + 0.5 * float(a) - 0.25 * float(b)

    def __call__(self, a: float, b: float) -> SyntheticABRenderer:
        return SyntheticABRenderer(self._observed, self.true_q0(a, b))


def main() -> int:
    observed = np.array([[10.0, 12.0], [14.0, 16.0]], dtype=float)
    sigma = np.ones_like(observed)
    factory = SyntheticABRendererFactory(observed)

    a_values = np.array([0.0, 0.5, 1.0], dtype=float)
    b_values = np.array([0.0, 1.0, 2.0], dtype=float)

    result = multi_scan_ab(
        factory,
        observed,
        sigma,
        a_values=a_values,
        b_values=b_values,
        q0_min=0.1,
        q0_max=10.0,
        target_metric="chi2",
        q0_start_grid=2.0,
    )

    np.set_printoptions(precision=4, suppress=True)
    print("a_values:", result.a_values)
    print("b_values:", result.b_values)
    print("best_q0 grid:\n", result.best_q0)
    print("objective grid:\n", result.objective_values)
    print("success grid:\n", result.success.astype(int))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
