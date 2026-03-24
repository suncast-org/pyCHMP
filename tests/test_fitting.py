import numpy as np
import pytest

from pychmp.fitting import fit_q0_to_observation


class SyntheticRenderer:
    def __init__(self, observed: np.ndarray, true_q0: float) -> None:
        self._observed = observed
        self._true_q0 = true_q0

    def render(self, q0: float) -> np.ndarray:
        # Global offset model with minimum mismatch at q0 == true_q0.
        return self._observed + (q0 - self._true_q0)


class WrongShapeRenderer:
    def render(self, q0: float) -> np.ndarray:  # noqa: ARG002
        return np.ones((2, 2))


def test_fit_q0_to_observation_recovers_true_q0() -> None:
    observed = np.array([[10.0, 12.0], [14.0, 16.0]])
    sigma = np.ones_like(observed)
    renderer = SyntheticRenderer(observed, true_q0=3.7)

    result = fit_q0_to_observation(
        renderer,
        observed,
        sigma,
        q0_min=0.1,
        q0_max=10.0,
        threshold=0.1,
        target_metric="chi2",
    )

    assert result.success
    assert result.q0 == pytest.approx(3.7, abs=1e-2)


def test_fit_q0_to_observation_validates_shape_mismatch() -> None:
    observed = np.ones((3, 3))
    sigma = np.ones((3, 3))

    with pytest.raises(ValueError, match="renderer output shape"):
        fit_q0_to_observation(
            WrongShapeRenderer(),
            observed,
            sigma,
            q0_min=0.1,
            q0_max=10.0,
        )


def test_fit_q0_to_observation_validates_observed_sigma_shapes() -> None:
    observed = np.ones((3, 3))
    sigma = np.ones((2, 2))

    with pytest.raises(ValueError, match="observed and sigma"):
        fit_q0_to_observation(
            SyntheticRenderer(observed, true_q0=2.0),
            observed,
            sigma,
            q0_min=0.1,
            q0_max=10.0,
        )
