from __future__ import annotations

import numpy as np
import pytest

from pychmp.ab_search import (
    ABPointResult,
    ExpandResumeContext,
    _append_adaptive_request_if_needed,
    _cache_point_blocks_rerun,
    idl_q0_start_heuristic,
    multi_scan_ab,
    search_local_minimum_ab,
    select_expand_frontier_seed,
    widened_boundary_axes,
)
from pychmp.metrics import MetricValues


def _localized_peak_observed(*, size: int = 32, peak: float = 100.0) -> np.ndarray:
    observed = np.zeros((size, size), dtype=float)
    center = size // 2
    observed[center - 2 : center + 2, center - 2 : center + 2] = peak
    return observed


def _adaptive_penalty_pattern(shape: tuple[int, ...]) -> np.ndarray:
    pattern = np.zeros(shape, dtype=float)
    cy, cx = shape[0] // 2, shape[1] // 2
    pattern[cy - 1 : cy + 1, cx - 1 : cx + 1] = [[1.0, -1.0], [-1.0, 1.0]]
    return pattern


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


class PicklableSyntheticABRenderer:
    def __init__(self, observed: np.ndarray, true_q0: float) -> None:
        self._observed = observed
        self._true_q0 = true_q0

    def render(self, q0: float) -> np.ndarray:
        return self._observed + (float(q0) - self._true_q0)


class PicklableRendererFactory:
    def __init__(self, observed: np.ndarray) -> None:
        self.observed = observed

    def true_q0(self, a: float, b: float) -> float:
        return 2.0 + 0.5 * float(a) - 0.25 * float(b)

    def __call__(self, a: float, b: float) -> PicklableSyntheticABRenderer:
        return PicklableSyntheticABRenderer(self.observed, self.true_q0(a, b))


class AdaptiveSyntheticABRenderer:
    def __init__(self, observed: np.ndarray, true_q0: float, penalty: float, render_log: list[float]) -> None:
        self._observed = observed
        self._true_q0 = true_q0
        self._penalty = penalty
        self._render_log = render_log
        self._pattern = _adaptive_penalty_pattern(observed.shape)

    def render(self, q0: float) -> np.ndarray:
        self._render_log.append(float(q0))
        return self._observed + (float(q0) - self._true_q0) + self._penalty * self._pattern


class AdaptiveRendererFactory:
    def __init__(self, observed: np.ndarray, *, optimum_a: float, optimum_b: float) -> None:
        self.observed = observed
        self.optimum_a = float(optimum_a)
        self.optimum_b = float(optimum_b)
        self.render_log: list[float] = []
        self.calls: list[tuple[float, float]] = []

    def true_q0(self, a: float, b: float) -> float:
        return 2.0 + 0.1 * float(a) - 0.05 * float(b)

    def penalty(self, a: float, b: float) -> float:
        return (float(a) - self.optimum_a) ** 2 + (float(b) - self.optimum_b) ** 2

    def __call__(self, a: float, b: float) -> AdaptiveSyntheticABRenderer:
        self.calls.append((float(a), float(b)))
        return AdaptiveSyntheticABRenderer(
            self.observed,
            self.true_q0(a, b),
            self.penalty(a, b),
            self.render_log,
        )


class PicklableAdaptiveSyntheticABRenderer:
    def __init__(self, observed: np.ndarray, true_q0: float, penalty: float) -> None:
        self._observed = observed
        self._true_q0 = true_q0
        self._penalty = penalty
        self._pattern = _adaptive_penalty_pattern(observed.shape)

    def render(self, q0: float) -> np.ndarray:
        return self._observed + (float(q0) - self._true_q0) + self._penalty * self._pattern


class PicklableAdaptiveRendererFactory:
    def __init__(self, observed: np.ndarray, *, optimum_a: float, optimum_b: float) -> None:
        self.observed = observed
        self.optimum_a = float(optimum_a)
        self.optimum_b = float(optimum_b)

    def true_q0(self, a: float, b: float) -> float:
        return 2.0 + 0.1 * float(a) - 0.05 * float(b)

    def penalty(self, a: float, b: float) -> float:
        return (float(a) - self.optimum_a) ** 2 + (float(b) - self.optimum_b) ** 2

    def __call__(self, a: float, b: float) -> PicklableAdaptiveSyntheticABRenderer:
        return PicklableAdaptiveSyntheticABRenderer(self.observed, self.true_q0(a, b), self.penalty(a, b))


class LookupAdaptiveRendererFactory:
    def __init__(
        self,
        observed: np.ndarray,
        *,
        penalty_map: dict[tuple[float, float], float],
        default_penalty: float = 9.0,
    ) -> None:
        self.observed = observed
        self.penalty_map = {(float(a), float(b)): float(value) for (a, b), value in penalty_map.items()}
        self.default_penalty = float(default_penalty)
        self.render_log: list[float] = []
        self.calls: list[tuple[float, float]] = []

    def true_q0(self, a: float, b: float) -> float:
        return 2.0 + 0.1 * float(a) - 0.05 * float(b)

    def penalty(self, a: float, b: float) -> float:
        return self.penalty_map.get((float(a), float(b)), self.default_penalty)

    def __call__(self, a: float, b: float) -> AdaptiveSyntheticABRenderer:
        self.calls.append((float(a), float(b)))
        return AdaptiveSyntheticABRenderer(
            self.observed,
            self.true_q0(a, b),
            self.penalty(a, b),
            self.render_log,
        )


class RecordingAdaptiveCache(dict[tuple[float, float], ABPointResult]):
    def __init__(self) -> None:
        super().__init__()
        self.pending_history: list[tuple[tuple[float, float], ...]] = []

    def set_pending_points(self, points: list[tuple[float, float]] | tuple[tuple[float, float], ...]) -> None:
        normalized = tuple((float(a), float(b)) for a, b in points)
        self.pending_history.append(normalized)

    def clear_pending_points(self) -> None:
        self.pending_history.append(())


def test_idl_q0_start_heuristic_matches_formula() -> None:
    """Match the legacy IDL q0-start heuristic formula exactly."""
    value = idl_q0_start_heuristic(0.3, 2.7)
    expected = np.exp(-10.1 - 0.193 * 0.3 + 2.17 * 2.7)
    assert value == pytest.approx(expected)


def test_multi_scan_ab_recovers_expected_q0_grid() -> None:
    """Recover the expected q0 grid for a full rectangular scan."""
    observed = _localized_peak_observed()
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


def test_multi_scan_ab_points_preserve_all_metric_trial_histories() -> None:
    """Keep chi2/rho2/eta2 trial histories on stored AB point results."""
    observed = _localized_peak_observed()
    sigma = np.ones_like(observed)
    factory = CountingRendererFactory(observed)

    result = multi_scan_ab(
        factory,
        observed,
        sigma,
        a_values=[0.0],
        b_values=[0.0],
        q0_min=0.1,
        q0_max=10.0,
        target_metric="chi2",
        q0_start_grid=2.0,
    )

    point = result.points[0]
    assert len(point.trial_q0) > 0
    assert len(point.trial_objective_values) == len(point.trial_q0)
    assert len(point.trial_chi2_values) == len(point.trial_q0)
    assert len(point.trial_rho2_values) == len(point.trial_q0)
    assert len(point.trial_eta2_values) == len(point.trial_q0)


def test_multi_scan_ab_reuses_cached_points() -> None:
    """Reuse cached rectangular scan points without re-rendering them."""
    observed = _localized_peak_observed()
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
    """Reject q0 start grids whose shape does not match the scan grid."""
    observed = _localized_peak_observed()
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


def test_multi_scan_ab_supports_process_pool_execution() -> None:
    """Support rectangular scan execution through the process pool."""
    observed = _localized_peak_observed()
    sigma = np.ones_like(observed)
    factory = PicklableRendererFactory(observed)

    result = multi_scan_ab(
        factory,
        observed,
        sigma,
        a_values=[0.0, 1.0],
        b_values=[0.0, 2.0],
        q0_min=0.1,
        q0_max=10.0,
        q0_start_grid=2.0,
        execution_policy="process-pool",
        max_workers=1,
        worker_chunksize=1,
    )

    expected = np.array(
        [
            [factory.true_q0(0.0, 0.0), factory.true_q0(0.0, 2.0)],
            [factory.true_q0(1.0, 0.0), factory.true_q0(1.0, 2.0)],
        ],
        dtype=float,
    )
    assert np.all(result.success)
    assert np.allclose(result.best_q0, expected, atol=1e-2)


def test_multi_scan_ab_supports_auto_execution_policy() -> None:
    """Support rectangular scan execution through the auto policy."""
    observed = _localized_peak_observed()
    sigma = np.ones_like(observed)
    factory = PicklableRendererFactory(observed)

    result = multi_scan_ab(
        factory,
        observed,
        sigma,
        a_values=[0.0, 1.0],
        b_values=[0.0, 2.0],
        q0_min=0.1,
        q0_max=10.0,
        q0_start_grid=2.0,
        execution_policy="auto",
        max_workers=4,
        worker_chunksize=1,
    )

    expected = np.array(
        [
            [factory.true_q0(0.0, 0.0), factory.true_q0(0.0, 2.0)],
            [factory.true_q0(1.0, 0.0), factory.true_q0(1.0, 2.0)],
        ],
        dtype=float,
    )
    assert np.all(result.success)
    assert np.allclose(result.best_q0, expected, atol=1e-2)


def test_adaptive_serial_reports_only_one_active_point_at_a_time() -> None:
    """Serial adaptive refresh state should advertise one in-flight point at a time."""
    observed = _localized_peak_observed()
    sigma = np.ones_like(observed)
    factory = AdaptiveRendererFactory(observed, optimum_a=0.0, optimum_b=0.0)
    cache = RecordingAdaptiveCache()

    search_local_minimum_ab(
        factory,
        observed,
        sigma,
        a_start=0.0,
        b_start=0.0,
        da=1.0,
        db=1.0,
        a_range=(-1.0, 1.0),
        b_range=(-1.0, 1.0),
        q0_min=0.1,
        q0_max=10.0,
        q0_start=2.0,
        execution_policy="serial",
        cache=cache,
    )

    non_empty_pending = [points for points in cache.pending_history if points]
    assert non_empty_pending
    assert all(len(points) == 1 for points in non_empty_pending)


def test_multi_scan_ab_rejects_progress_callback_in_process_pool_mode() -> None:
    """Reject progress callbacks when scans run in process-pool mode."""
    observed = _localized_peak_observed()
    sigma = np.ones_like(observed)
    factory = PicklableRendererFactory(observed)

    with pytest.raises(ValueError, match="progress_callback"):
        multi_scan_ab(
            factory,
            observed,
            sigma,
            a_values=[0.0, 1.0],
            b_values=[0.0, 2.0],
            q0_min=0.1,
            q0_max=10.0,
            execution_policy="process-pool",
            progress_callback=lambda _q0, _metric, _ctx: None,
        )


def test_search_local_minimum_ab_converges_on_known_minimum() -> None:
    """Converge on a known local minimum during phase 1 search."""
    observed = _localized_peak_observed()
    sigma = np.ones_like(observed)
    factory = AdaptiveRendererFactory(observed, optimum_a=1.0, optimum_b=-1.0)

    result = search_local_minimum_ab(
        factory,
        observed,
        sigma,
        a_start=0.0,
        b_start=0.0,
        da=1.0,
        db=1.0,
        a_range=(-2.0, 2.0),
        b_range=(-2.0, 2.0),
        q0_min=0.1,
        q0_max=10.0,
        no_area=True,
    )

    assert result.best_a == pytest.approx(1.0)
    assert result.best_b == pytest.approx(-1.0)
    assert np.isclose(np.nanmin(result.objective_values), 0.0, atol=1e-6)
    assert result.n_phase1_iters >= 1
    assert 1.0 in result.a_values
    assert -1.0 in result.b_values
    assert result.best_is_interior is True
    assert result.best_boundary_axes == ()
    assert result.minimum_certified is False
    assert result.termination_reason == "phase1_only"


def test_search_local_minimum_ab_expands_when_minimum_is_on_edge() -> None:
    """Expand the sampled domain when the current minimum sits on an edge."""
    observed = _localized_peak_observed()
    sigma = np.ones_like(observed)
    factory = AdaptiveRendererFactory(observed, optimum_a=2.0, optimum_b=0.0)

    result = search_local_minimum_ab(
        factory,
        observed,
        sigma,
        a_start=0.0,
        b_start=0.0,
        da=1.0,
        db=1.0,
        a_range=(-2.0, 2.0),
        b_range=(-1.0, 1.0),
        q0_min=0.1,
        q0_max=10.0,
        no_area=True,
    )

    assert result.best_a == pytest.approx(2.0)
    assert 2.0 in result.a_values
    assert result.n_phase1_iters >= 2
    assert result.best_is_interior is False
    assert result.best_boundary_axes == ("a_max",)
    assert result.minimum_certified is False
    assert result.termination_reason == "phase1_only"


def test_search_local_minimum_ab_reuses_cached_points() -> None:
    """Reuse cached adaptive-search points across repeated runs."""
    observed = _localized_peak_observed()
    sigma = np.ones_like(observed)
    factory = AdaptiveRendererFactory(observed, optimum_a=1.0, optimum_b=-1.0)
    cache: dict[tuple[float, float], ABPointResult] = {}

    first = search_local_minimum_ab(
        factory,
        observed,
        sigma,
        a_start=0.0,
        b_start=0.0,
        da=1.0,
        db=1.0,
        a_range=(-2.0, 2.0),
        b_range=(-2.0, 2.0),
        q0_min=0.1,
        q0_max=10.0,
        cache=cache,
        no_area=True,
    )
    render_count_after_first = len(factory.render_log)
    factory.calls.clear()

    second = search_local_minimum_ab(
        factory,
        observed,
        sigma,
        a_start=0.0,
        b_start=0.0,
        da=1.0,
        db=1.0,
        a_range=(-2.0, 2.0),
        b_range=(-2.0, 2.0),
        q0_min=0.1,
        q0_max=10.0,
        cache=cache,
        no_area=True,
    )

    assert len(factory.calls) == 0
    assert len(factory.render_log) == render_count_after_first
    assert first.best_a == pytest.approx(second.best_a)
    assert first.best_b == pytest.approx(second.best_b)
    assert second.evaluated_point_count == first.evaluated_point_count


def test_search_local_minimum_ab_supports_process_pool_neighbor_batches() -> None:
    """Evaluate adaptive-search neighbor batches through the process pool."""
    observed = _localized_peak_observed()
    sigma = np.ones_like(observed)
    factory = PicklableAdaptiveRendererFactory(observed, optimum_a=1.0, optimum_b=-1.0)

    result = search_local_minimum_ab(
        factory,
        observed,
        sigma,
        a_start=0.0,
        b_start=0.0,
        da=1.0,
        db=1.0,
        a_range=(-2.0, 2.0),
        b_range=(-2.0, 2.0),
        q0_min=0.1,
        q0_max=10.0,
        execution_policy="process-pool",
        max_workers=2,
        worker_chunksize=1,
        no_area=True,
    )

    assert result.best_a == pytest.approx(1.0)
    assert result.best_b == pytest.approx(-1.0)
    assert result.minimum_certified is False


def test_search_local_minimum_ab_phase2_expands_threshold_region() -> None:
    """Expand phase 2 across the requested threshold region."""
    observed = _localized_peak_observed()
    sigma = np.ones_like(observed)
    factory = AdaptiveRendererFactory(observed, optimum_a=0.5, optimum_b=0.5)

    result = search_local_minimum_ab(
        factory,
        observed,
        sigma,
        a_start=0.0,
        b_start=0.0,
        da=1.0,
        db=1.0,
        a_range=(-1.0, 2.0),
        b_range=(-1.0, 2.0),
        q0_min=0.1,
        q0_max=10.0,
        threshold_metric=1.01,
    )

    assert 2.0 in result.a_values
    assert 2.0 in result.b_values
    assert result.n_phase2_iters >= 1
    assert np.isfinite(result.objective_values[result.a_values.index(2.0), result.b_values.index(1.0)])
    assert result.minimum_certified is True
    assert result.termination_reason == "certified_local_minimum"


def test_search_local_minimum_ab_no_area_stops_after_phase1() -> None:
    """Skip phase 2 completely when no-area mode is enabled."""
    observed = _localized_peak_observed()
    sigma = np.ones_like(observed)
    factory = AdaptiveRendererFactory(observed, optimum_a=0.5, optimum_b=0.5)

    result = search_local_minimum_ab(
        factory,
        observed,
        sigma,
        a_start=0.0,
        b_start=0.0,
        da=1.0,
        db=1.0,
        a_range=(-1.0, 2.0),
        b_range=(-1.0, 2.0),
        q0_min=0.1,
        q0_max=10.0,
        threshold_metric=1.01,
        no_area=True,
    )

    assert result.a_values == pytest.approx((-1.0, 0.0, 1.0))
    assert result.b_values == pytest.approx((-1.0, 0.0, 1.0))
    assert result.n_phase2_iters == 0
    assert result.termination_reason == "phase1_only"


def test_search_local_minimum_ab_warm_start_does_not_change_recovered_minimum() -> None:
    """Preserve the recovered minimum when a warm-start q0 is supplied."""
    observed = _localized_peak_observed()
    sigma = np.ones_like(observed)
    default_factory = AdaptiveRendererFactory(observed, optimum_a=1.0, optimum_b=-1.0)
    explicit_factory = AdaptiveRendererFactory(observed, optimum_a=1.0, optimum_b=-1.0)

    default_result = search_local_minimum_ab(
        default_factory,
        observed,
        sigma,
        a_start=0.0,
        b_start=0.0,
        da=1.0,
        db=1.0,
        a_range=(-2.0, 2.0),
        b_range=(-2.0, 2.0),
        q0_min=0.1,
        q0_max=10.0,
        threshold_metric=1.01,
    )
    explicit_result = search_local_minimum_ab(
        explicit_factory,
        observed,
        sigma,
        a_start=0.0,
        b_start=0.0,
        da=1.0,
        db=1.0,
        a_range=(-2.0, 2.0),
        b_range=(-2.0, 2.0),
        q0_min=0.1,
        q0_max=10.0,
        q0_start=0.25,
        threshold_metric=1.01,
    )

    assert default_result.best_a == pytest.approx(explicit_result.best_a)
    assert default_result.best_b == pytest.approx(explicit_result.best_b)
    assert default_result.a_values == pytest.approx(explicit_result.a_values)
    assert default_result.b_values == pytest.approx(explicit_result.b_values)
    assert np.nanmin(default_result.objective_values) == pytest.approx(np.nanmin(explicit_result.objective_values))
    assert default_result.minimum_certified == explicit_result.minimum_certified


def test_search_local_minimum_ab_phase2_prefers_connected_threshold_basin() -> None:
    """Expand the connected threshold basin around the current best instead of chasing disconnected islands."""
    observed = _localized_peak_observed()
    sigma = np.ones_like(observed)
    factory = LookupAdaptiveRendererFactory(
        observed,
        penalty_map={
            (0.0, 0.0): 2.0,
            (1.0, -1.0): 0.0,
            (1.0, 0.0): 3.0,
            (1.0, 1.0): 0.0,
            (2.0, -1.0): 1.0,
            (2.0, 1.0): 1.0,
        },
        default_penalty=8.0,
    )

    result = search_local_minimum_ab(
        factory,
        observed,
        sigma,
        a_start=0.0,
        b_start=0.0,
        da=1.0,
        db=1.0,
        a_range=(-1.0, 2.0),
        b_range=(-2.0, 2.0),
        q0_min=0.1,
        q0_max=10.0,
        threshold_metric=1.01,
    )

    a_index = result.a_values.index(2.0)
    lower_b_index = result.b_values.index(-1.0)
    upper_b_index = result.b_values.index(1.0)

    assert result.best_a == pytest.approx(1.0)
    assert result.best_b == pytest.approx(-1.0) or result.best_b == pytest.approx(1.0)
    assert 2.0 in result.a_values
    assert np.isfinite(result.objective_values[a_index, lower_b_index])
    assert not np.isfinite(result.objective_values[a_index, upper_b_index])
    assert result.minimum_certified is True


def test_search_local_minimum_ab_matches_bruteforce_scan_on_small_domain() -> None:
    """Match brute-force scanning on a small discrete a-b domain."""
    observed = _localized_peak_observed()
    sigma = np.ones_like(observed)
    factory = AdaptiveRendererFactory(observed, optimum_a=1.0, optimum_b=-1.0)
    a_values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)
    b_values = np.array([-3.0, -2.0, -1.0, 0.0, 1.0], dtype=float)

    brute_force = multi_scan_ab(
        factory,
        observed,
        sigma,
        a_values=a_values,
        b_values=b_values,
        q0_min=0.1,
        q0_max=10.0,
        target_metric="chi2",
    )
    local_search = search_local_minimum_ab(
        factory,
        observed,
        sigma,
        a_start=0.0,
        b_start=0.0,
        da=1.0,
        db=1.0,
        a_range=(float(a_values[0]), float(a_values[-1])),
        b_range=(float(b_values[0]), float(b_values[-1])),
        q0_min=0.1,
        q0_max=10.0,
        threshold_metric=1.01,
    )

    brute_force_index = np.unravel_index(np.nanargmin(brute_force.objective_values), brute_force.objective_values.shape)
    brute_force_best_a = float(a_values[brute_force_index[0]])
    brute_force_best_b = float(b_values[brute_force_index[1]])
    brute_force_best_q0 = float(brute_force.best_q0[brute_force_index])
    brute_force_best_objective = float(brute_force.objective_values[brute_force_index])

    assert local_search.best_a == pytest.approx(brute_force_best_a)
    assert local_search.best_b == pytest.approx(brute_force_best_b)
    assert np.nanmin(local_search.objective_values) == pytest.approx(brute_force_best_objective)
    assert local_search.best_q0[local_search.a_values.index(local_search.best_a), local_search.b_values.index(local_search.best_b)] == pytest.approx(brute_force_best_q0, abs=1e-2)
    assert local_search.minimum_certified is True


def test_widened_boundary_axes_detects_superset_edges() -> None:
    axes = widened_boundary_axes(
        prior_a_range=(-1.0, 1.0),
        prior_b_range=(0.0, 5.0),
        new_a_range=(-1.5, 1.0),
        new_b_range=(0.0, 6.0),
    )
    assert axes == ("a_min", "b_max")


def test_select_expand_frontier_seed_prefers_prior_wall_over_interior() -> None:
    interior = ABPointResult(
        a=0.0,
        b=2.0,
        q0=1.0,
        objective_value=0.01,
        metrics=MetricValues(chi2=0.01, rho2=0.0, eta2=0.0),
        target_metric="chi2",
        success=True,
        nfev=1,
        nit=0,
        message="",
        used_adaptive_bracketing=False,
        bracket_found=False,
        bracket=None,
        trial_q0=(1.0,),
        trial_objective_values=(0.01,),
    )
    wall = ABPointResult(
        a=0.0,
        b=5.0,
        q0=1.0,
        objective_value=0.5,
        metrics=MetricValues(chi2=0.5, rho2=0.0, eta2=0.0),
        target_metric="chi2",
        success=True,
        nfev=1,
        nit=0,
        message="",
        used_adaptive_bracketing=False,
        bracket_found=False,
        bracket=None,
        trial_q0=(1.0,),
        trial_objective_values=(0.5,),
    )
    context = ExpandResumeContext(
        prior_a_range=(-1.0, 1.0),
        prior_b_range=(0.0, 5.0),
        widened_axes=("b_max",),
    )
    seed = select_expand_frontier_seed(
        {(0.0, 2.0): interior, (0.0, 5.0): wall},
        context=context,
        da=0.25,
        db=0.25,
    )
    assert seed == (0.0, 5.0)


def test_search_local_minimum_ab_resume_expands_cached_frontier_when_bounds_widen() -> None:
    """Resume from cached sparse points and expand further when wider bounds are requested."""
    observed = _localized_peak_observed()
    sigma = np.ones_like(observed)
    factory = AdaptiveRendererFactory(observed, optimum_a=2.0, optimum_b=-1.0)
    cache: dict[tuple[float, float], ABPointResult] = {}

    first = search_local_minimum_ab(
        factory,
        observed,
        sigma,
        a_start=0.0,
        b_start=0.0,
        da=1.0,
        db=1.0,
        a_range=(-1.0, 1.0),
        b_range=(-2.0, 1.0),
        q0_min=0.1,
        q0_max=10.0,
        threshold_metric=1.01,
        cache=cache,
    )
    first_call_count = len(factory.calls)
    factory.calls.clear()

    expand_context = ExpandResumeContext(
        prior_a_range=(-1.0, 1.0),
        prior_b_range=(-2.0, 1.0),
        widened_axes=("a_max",),
    )
    resumed = search_local_minimum_ab(
        factory,
        observed,
        sigma,
        a_start=0.0,
        b_start=0.0,
        da=1.0,
        db=1.0,
        a_range=(-1.0, 2.0),
        b_range=(-2.0, 1.0),
        q0_min=0.1,
        q0_max=10.0,
        threshold_metric=1.01,
        cache=cache,
        expand_resume_context=expand_context,
    )

    assert first.best_a == pytest.approx(1.0)
    assert first.minimum_certified is False
    assert resumed.best_a == pytest.approx(2.0)
    assert 2.0 in resumed.a_values
    assert len(factory.calls) > 0
    assert resumed.evaluated_point_count > first.evaluated_point_count
    assert first_call_count > 0


class _IncompleteResumeAdaptiveCache(RecordingAdaptiveCache):
    def __init__(self, *, partial_point: ABPointResult) -> None:
        super().__init__()
        self._partial_point = partial_point
        self._resume_keys = {(float(partial_point.a), float(partial_point.b))}

    def point_needs_completion(self, a_value: float, b_value: float) -> bool:
        return (float(a_value), float(b_value)) in self._resume_keys

    def partial_point_results(self) -> dict[tuple[float, float], ABPointResult]:
        key = (float(self._partial_point.a), float(self._partial_point.b))
        return {key: self._partial_point}


def test_cache_point_blocks_rerun_allows_incomplete_resume_points() -> None:
    partial = ABPointResult(
        a=0.45,
        b=3.75,
        q0=0.04,
        objective_value=0.37,
        metrics=MetricValues(chi2=1.0, rho2=0.5, eta2=0.37),
        target_metric="eta2",
        success=False,
        nfev=3,
        nit=2,
        message="partial",
        used_adaptive_bracketing=False,
        bracket_found=False,
        bracket=None,
        trial_q0=(0.03, 0.04, 0.05),
        trial_objective_values=(0.39, 0.37, 0.38),
        trial_chi2_values=(1.1, 1.0, 1.05),
        trial_rho2_values=(0.49, 0.5, 0.51),
        trial_eta2_values=(0.39, 0.37, 0.38),
    )
    cache = _IncompleteResumeAdaptiveCache(partial_point=partial)
    key = (0.45, 3.75)
    assert _cache_point_blocks_rerun(cache, key) is False


def test_append_adaptive_request_if_needed_queues_incomplete_partial_topology() -> None:
    partial = ABPointResult(
        a=-0.05,
        b=4.0,
        q0=0.07,
        objective_value=0.35,
        metrics=MetricValues(chi2=1.0, rho2=0.5, eta2=0.35),
        target_metric="eta2",
        success=False,
        nfev=4,
        nit=3,
        message="partial",
        used_adaptive_bracketing=False,
        bracket_found=False,
        bracket=None,
        trial_q0=(0.05, 0.07, 0.09),
        trial_objective_values=(0.36, 0.35, 0.37),
        trial_chi2_values=(1.1, 1.0, 1.05),
        trial_rho2_values=(0.49, 0.5, 0.51),
        trial_eta2_values=(0.36, 0.35, 0.37),
    )
    cache = _IncompleteResumeAdaptiveCache(partial_point=partial)
    a_values = np.asarray([-0.05], dtype=float)
    b_values = np.asarray([4.0], dtype=float)
    point_results = {(float(partial.a), float(partial.b)): partial}
    pending_requests = []
    pending_keys: set[tuple[float, float]] = set()
    _append_adaptive_request_if_needed(
        pending_requests,
        pending_keys,
        a_values=a_values,
        b_values=b_values,
        a_index=0,
        b_index=0,
        q0_min=1e-5,
        q0_max=1e-1,
        hard_q0_min=None,
        hard_q0_max=None,
        threshold=0.1,
        mask_type="union",
        explicit_mask=None,
        target_metric="eta2",
        xatol=1e-3,
        maxiter=50,
        adaptive_bracketing=False,
        q0_seed=0.07,
        q0_step=1.618,
        max_bracket_steps=8,
        point_results=point_results,
        cache_map=cache,
    )
    assert len(pending_requests) == 1
    assert pending_requests[0].q0_start == pytest.approx(0.07)


def test_evaluate_ab_search_request_finalizes_resume_without_rendering() -> None:
    from types import SimpleNamespace

    from pychmp.ab_search import (
        ABPointEvaluationRequest,
        ABPointTask,
        ABSearchWorkerPayload,
        _evaluate_ab_search_request,
    )

    partial = ABPointResult(
        a=-0.05,
        b=4.0,
        q0=0.072105,
        objective_value=0.337719,
        metrics=MetricValues(chi2=1.0, rho2=0.5, eta2=0.337719),
        target_metric="eta2",
        success=True,
        nfev=84,
        nit=83,
        message="restored from grid point trials",
        used_adaptive_bracketing=False,
        bracket_found=True,
        bracket=None,
        trial_q0=(0.05, 0.072105, 0.09),
        trial_objective_values=(0.36, 0.337719, 0.37),
        trial_chi2_values=(1.1, 1.0, 1.05),
        trial_rho2_values=(0.49, 0.5, 0.51),
        trial_eta2_values=(0.36, 0.337719, 0.37),
    )
    render_log: list[float] = []

    class _ResumeFinalizeCache(_IncompleteResumeAdaptiveCache):
        def try_finalize_resume_point_from_stored_trials(
            self,
            a_value: float,
            b_value: float,
        ) -> ABPointResult | None:
            if (float(a_value), float(b_value)) == (-0.05, 4.0):
                return partial
            return None

    class _FinalizeRenderer:
        def prepare_stored_trial_maps(self) -> int:
            return 0

        def build_artifact_payload(self, point: ABPointResult) -> dict[str, object]:
            return {"q0": float(point.q0), "best_metric": float(point.objective_value)}

        def render(self, q0: float) -> np.ndarray:
            render_log.append(float(q0))
            return np.zeros((4, 4), dtype=float)

    request = ABPointEvaluationRequest(
        task=ABPointTask(
            slice_key="mw_test",
            slice_domain="mw",
            slice_label="1.418 GHz",
            slice_display_label="MW: 1.418 GHz",
            a=-0.05,
            b=4.0,
            a_index=0,
            b_index=0,
            q0_min=1e-4,
            q0_max=0.2,
            target_metric="eta2",
            source_kind="adaptive",
        ),
        hard_q0_min=None,
        hard_q0_max=None,
        threshold=0.2,
        mask_type="union",
        explicit_mask=None,
        target_metric="eta2",
        xatol=1e-3,
        maxiter=200,
        adaptive_bracketing=True,
        q0_start=0.072105,
        q0_step=1.618,
        max_bracket_steps=12,
    )
    payload = ABSearchWorkerPayload(
        renderer_factory=lambda _a, _b: _FinalizeRenderer(),
        observed=np.ones((4, 4), dtype=float),
        sigma=np.ones((4, 4), dtype=float),
        cache_map=_ResumeFinalizeCache(partial_point=partial),
    )
    result = _evaluate_ab_search_request(request, payload)
    assert render_log == []
    assert result.q0 == pytest.approx(0.072105)
    assert result.artifact_payload is not None
    assert float(result.artifact_payload["best_metric"]) == pytest.approx(0.337719)


@pytest.mark.parametrize('frequency,wavelength,instrument,label', [
    (None, 211., 'AIA', 'AIA 211 Å'),
    (17., None, None, '17 GHz'),
])
def test_point_start_log_identifies_target_spectrum(capsys, frequency, wavelength, instrument, label) -> None:
    observed = _localized_peak_observed()
    factory = CountingRendererFactory(observed)
    factory.frequency_ghz = frequency
    factory.wavelength_angstrom = wavelength
    factory.euv_instrument = instrument
    factory.render_channels = ('94', '131', '171', '193', '211', '304', '335')
    search_local_minimum_ab(
        factory, observed, np.ones_like(observed), a_start=0., b_start=0., da=.1, db=.1,
        a_range=(0., 0.), b_range=(0., 0.), q0_min=.1, q0_max=10., maxiter=2,
    )
    starts = [line for line in capsys.readouterr().out.splitlines() if 'Starting point:' in line]
    assert starts
    assert all(f'Starting point: {label}; a=' in line for line in starts)
