import numpy as np
import pytest

from pychmp.fitting import fit_q0_to_observation
from pychmp.metrics import MetricValues


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


class RecordingRenderer(SyntheticRenderer):
    def __init__(self, observed: np.ndarray, true_q0: float) -> None:
        super().__init__(observed, true_q0)
        self.calls: list[float] = []

    def render(self, q0: float) -> np.ndarray:
        self.calls.append(float(q0))
        return super().render(q0)


def _localized_peak_observed(*, size: int = 32, peak: float = 100.0) -> np.ndarray:
    observed = np.zeros((size, size), dtype=float)
    center = size // 2
    observed[center - 2 : center + 2, center - 2 : center + 2] = peak
    return observed


def test_fit_q0_to_observation_recovers_true_q0() -> None:
    """Recover the true q0 from a synthetic observation."""
    observed = _localized_peak_observed()
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
    """Reject renderer outputs whose shape differs from the observation."""
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
    """Reject observed and sigma arrays with incompatible shapes."""
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


def test_fit_q0_to_observation_seeds_saved_metric_evaluations() -> None:
    observed = _localized_peak_observed()
    sigma = np.ones_like(observed)
    renderer = RecordingRenderer(observed, true_q0=1.0)

    result = fit_q0_to_observation(
        renderer,
        observed,
        sigma,
        q0_min=0.5,
        q0_max=2.0,
        threshold=0.1,
        target_metric="chi2",
        adaptive_bracketing=True,
        q0_start=1.0,
        initial_evaluations={
            0.5: MetricValues(chi2=0.25, rho2=0.01, eta2=0.01),
            1.0: MetricValues(chi2=0.0, rho2=0.0, eta2=0.0),
            2.0: MetricValues(chi2=1.0, rho2=0.02, eta2=0.02),
        },
    )

    assert result.trial_q0[:3] == (0.5, 1.0, 2.0)
    assert not any(q0 in {0.5, 1.0, 2.0} for q0 in renderer.calls)
