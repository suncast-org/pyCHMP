from __future__ import annotations

import numpy as np
import pytest

from pychmp.ab_search import ABPointResult, idl_q0_start_heuristic, multi_scan_ab


class SyntheticABRenderer:
    def __init__(self, observed: np.ndarray, true_q0: float, render_log: list[float]) -> None:
        self._observed = observed
        self._true_q0 = true_q0
        self._render_log = render_log

    def render(self, q0: float) -> np.ndarray:
        self._render_log.append(float(q0))
        return self._observed + (float(q0) - self._true_q0)


class CountingRendererFactory:
    def __init__(self, observed: np.ndarray) -> None:
        self.observed = observed
        self.render_log: list[float] = []
        self.calls: list[tuple[float, float]] = []

    def true_q0(self, a: float, b: float) -> float:
        return 2.0 + 0.5 * float(a) - 0.25 * float(b)

    def __call__(self, a: float, b: float) -> SyntheticABRenderer:
        self.calls.append((float(a), float(b)))
        return SyntheticABRenderer(self.observed, self.true_q0(a, b), self.render_log)


def test_idl_q0_start_heuristic_matches_formula() -> None:
    value = idl_q0_start_heuristic(0.3, 2.7)
    expected = np.exp(-10.1 - 0.193 * 0.3 + 2.17 * 2.7)
    assert value == pytest.approx(expected)


def test_multi_scan_ab_recovers_expected_q0_grid() -> None:
    observed = np.array([[10.0, 12.0], [14.0, 16.0]], dtype=float)
    sigma = np.ones_like(observed)
    factory = CountingRendererFactory(observed)

    a_values = np.array([0.0, 1.0], dtype=float)
    b_values = np.array([0.0, 2.0], dtype=float)

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

    expected = np.array(
        [
            [factory.true_q0(0.0, 0.0), factory.true_q0(0.0, 2.0)],
            [factory.true_q0(1.0, 0.0), factory.true_q0(1.0, 2.0)],
        ],
        dtype=float,
    )
    assert result.best_q0.shape == (2, 2)
    assert np.all(result.success)
    assert np.allclose(result.best_q0, expected, atol=1e-2)
    assert np.allclose(result.objective_values, 0.0, atol=1e-6)
    assert len(result.points) == 4


def test_multi_scan_ab_reuses_cached_points() -> None:
    observed = np.array([[10.0, 12.0], [14.0, 16.0]], dtype=float)
    sigma = np.ones_like(observed)
    factory = CountingRendererFactory(observed)
    cache: dict[tuple[float, float], ABPointResult] = {}

    first = multi_scan_ab(
        factory,
        observed,
        sigma,
        a_values=[0.0, 1.0],
        b_values=[0.0, 2.0],
        q0_min=0.1,
        q0_max=10.0,
        q0_start_grid=2.0,
        cache=cache,
    )
    render_count_after_first = len(factory.render_log)
    factory.calls.clear()

    second = multi_scan_ab(
        factory,
        observed,
        sigma,
        a_values=[0.0, 1.0],
        b_values=[0.0, 2.0],
        q0_min=0.1,
        q0_max=10.0,
        q0_start_grid=2.0,
        cache=cache,
    )

    assert len(cache) == 4
    assert len(factory.calls) == 0
    assert len(factory.render_log) == render_count_after_first
    assert np.allclose(first.best_q0, second.best_q0)


def test_multi_scan_ab_validates_q0_start_grid_shape() -> None:
    observed = np.array([[10.0, 12.0], [14.0, 16.0]], dtype=float)
    sigma = np.ones_like(observed)
    factory = CountingRendererFactory(observed)

    with pytest.raises(ValueError, match="q0_start_grid"):
        multi_scan_ab(
            factory,
            observed,
            sigma,
            a_values=[0.0, 1.0],
            b_values=[0.0, 2.0],
            q0_min=0.1,
            q0_max=10.0,
            q0_start_grid=np.ones((3, 1), dtype=float),
        )
