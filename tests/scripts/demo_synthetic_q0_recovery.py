from __future__ import annotations

import numpy as np

from pychmp.fitting import fit_q0_to_observation


class SyntheticRenderer:
    def __init__(self, observed: np.ndarray, true_q0: float) -> None:
        self._observed = observed
        self._true_q0 = true_q0

    def render(self, q0: float) -> np.ndarray:
        return self._observed + (q0 - self._true_q0)


def main() -> int:
    true_q0 = 3.7
    observed = np.array([[10.0, 12.0], [14.0, 16.0]], dtype=float)
    sigma = np.ones_like(observed)

    result = fit_q0_to_observation(
        SyntheticRenderer(observed, true_q0=true_q0),
        observed,
        sigma,
        q0_min=0.1,
        q0_max=10.0,
        threshold=0.1,
        target_metric="chi2",
    )

    print(f"true_q0={true_q0:.4f}")
    print(f"fit_q0={result.q0:.4f}")
    print(f"objective={result.objective_value:.6f}")
    print(f"success={result.success}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
