import pytest

from pychmp.metrics import MetricValues
from pychmp.optimize import find_best_q0


def test_find_best_q0_chi2_target() -> None:
    def metric_function(q0: float) -> MetricValues:
        return MetricValues(
            chi2=(q0 - 2.5) ** 2 + 1.0,
            rho2=(q0 - 5.0) ** 2 + 2.0,
            eta2=(q0 - 8.0) ** 2 + 3.0,
        )

    result = find_best_q0(metric_function, q0_min=0.1, q0_max=10.0, target_metric="chi2")

    assert result.success
    assert result.q0 == pytest.approx(2.5, abs=1e-2)
    assert result.objective_value == pytest.approx(1.0, abs=1e-3)


def test_find_best_q0_target_metric_switch() -> None:
    def metric_function(q0: float) -> MetricValues:
        return MetricValues(
            chi2=(q0 - 9.0) ** 2,
            rho2=(q0 - 1.2) ** 2,
            eta2=(q0 - 4.0) ** 2,
        )

    result = find_best_q0(metric_function, q0_min=0.1, q0_max=10.0, target_metric="rho2")

    assert result.success
    assert result.q0 == pytest.approx(1.2, abs=1e-2)


def test_find_best_q0_validates_bounds() -> None:
    def metric_function(_: float) -> MetricValues:
        return MetricValues(chi2=1.0, rho2=1.0, eta2=1.0)

    with pytest.raises(ValueError, match="must be positive"):
        find_best_q0(metric_function, q0_min=0.0, q0_max=1.0)

    with pytest.raises(ValueError, match="must be less than"):
        find_best_q0(metric_function, q0_min=2.0, q0_max=1.0)
