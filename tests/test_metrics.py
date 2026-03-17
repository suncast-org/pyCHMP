import numpy as np
import pytest

from pychmp.metrics import compute_metrics, threshold_union_mask


def test_threshold_union_mask_selects_union() -> None:
    observed = np.array([0.0, 1.0, 2.0, 3.0])
    modeled = np.array([4.0, 0.1, 0.2, 0.3])

    mask = threshold_union_mask(observed, modeled, threshold=0.5)
    # observed threshold picks elements 2 and 3; modeled threshold picks element 0
    assert mask.tolist() == [True, False, True, True]


def test_compute_metrics_expected_values() -> None:
    observed = np.array([10.0, 20.0, 30.0, 40.0])
    modeled = np.array([11.0, 18.0, 33.0, 37.0])
    sigma = np.array([1.0, 2.0, 3.0, 4.0])
    mask = np.array([True, True, True, False])

    metrics = compute_metrics(observed, modeled, sigma, mask)

    assert metrics.chi2 == pytest.approx((1.0**2 + 1.0**2 + 1.0**2) / 3.0)
    expected_rho2 = ((11 / 10 - 1) ** 2 + (18 / 20 - 1) ** 2 + (33 / 30 - 1) ** 2) / 3
    assert metrics.rho2 == pytest.approx(expected_rho2)
    expected_eta2 = (((11 - 10) / 20) ** 2 + ((18 - 20) / 20) ** 2 + ((33 - 30) / 20) ** 2) / 3
    assert metrics.eta2 == pytest.approx(expected_eta2)


def test_compute_metrics_requires_non_empty_mask() -> None:
    observed = np.ones((2, 2))
    modeled = np.ones((2, 2))
    sigma = np.ones((2, 2))
    mask = np.zeros((2, 2), dtype=bool)

    with pytest.raises(ValueError, match="mask selects no elements"):
        compute_metrics(observed, modeled, sigma, mask)


def test_compute_metrics_rejects_zero_sigma_or_observed() -> None:
    observed = np.array([1.0, 0.0])
    modeled = np.array([1.2, 0.3])
    sigma = np.array([1.0, 1.0])
    mask = np.array([True, True])

    with pytest.raises(ValueError, match="observed contains zero"):
        compute_metrics(observed, modeled, sigma, mask)

    observed = np.array([1.0, 2.0])
    sigma = np.array([1.0, 0.0])
    with pytest.raises(ValueError, match="sigma contains zero"):
        compute_metrics(observed, modeled, sigma, mask)
