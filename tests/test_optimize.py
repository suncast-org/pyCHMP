import pytest

from pychmp.metrics import MetricValues
from pychmp.optimize import Q0MetricEvaluation, find_best_q0


def test_find_best_q0_chi2_target() -> None:
    """Minimize chi2 when it is the selected target metric."""
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
    """Switch optimization to the requested target metric."""
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
    """Validate basic q0 interval bounds before optimization."""
    def metric_function(_: float) -> MetricValues:
        return MetricValues(chi2=1.0, rho2=1.0, eta2=1.0)

    with pytest.raises(ValueError, match="must be positive"):
        find_best_q0(metric_function, q0_min=0.0, q0_max=1.0)

    with pytest.raises(ValueError, match="must be less than"):
        find_best_q0(metric_function, q0_min=2.0, q0_max=1.0)


def test_find_best_q0_validates_adaptive_inputs() -> None:
    """Validate adaptive-bracketing configuration before running."""
    def metric_function(_: float) -> MetricValues:
        return MetricValues(chi2=1.0, rho2=1.0, eta2=1.0)

    with pytest.raises(ValueError, match="greater than 1"):
        find_best_q0(metric_function, q0_min=0.1, q0_max=1.0, q0_step=1.0)

    with pytest.raises(ValueError, match="at least 1"):
        find_best_q0(metric_function, q0_min=0.1, q0_max=1.0, max_bracket_steps=0)

    with pytest.raises(ValueError, match="q0_start must lie within"):
        find_best_q0(metric_function, q0_min=0.1, q0_max=1.0, q0_start=2.0)

    with pytest.raises(ValueError, match="must not lie below hard_q0_min"):
        find_best_q0(metric_function, q0_min=0.1, q0_max=1.0, hard_q0_min=0.2)

    with pytest.raises(ValueError, match="must not lie above hard_q0_max"):
        find_best_q0(metric_function, q0_min=0.1, q0_max=1.0, hard_q0_max=0.9, q0_start=0.95)


def test_find_best_q0_adaptive_bracketing_moves_right_from_flux_deficit() -> None:
    """Grow the adaptive bracket upward when modeled flux is too low."""
    def metric_function(q0: float) -> Q0MetricEvaluation:
        return Q0MetricEvaluation(
            metrics=MetricValues(
                chi2=(q0 - 4.0) ** 2 + 0.5,
                rho2=(q0 - 4.0) ** 2 + 1.0,
                eta2=(q0 - 4.0) ** 2 + 2.0,
            ),
            total_observed_flux=40.0,
            total_modeled_flux=10.0 * q0,
        )

    result = find_best_q0(
        metric_function,
        q0_min=0.25,
        q0_max=10.0,
        adaptive_bracketing=True,
        q0_start=1.0,
        q0_step=1.6,
        max_bracket_steps=8,
    )

    assert result.success
    assert result.used_adaptive_bracketing
    assert result.bracket_found
    assert result.bracket is not None
    assert result.bracket[0] < 4.0 < result.bracket[2]
    assert result.q0 == pytest.approx(4.0, abs=1e-2)
    assert result.message.startswith("adaptive bracketing")


def test_find_best_q0_adaptive_bracketing_moves_left_from_flux_excess() -> None:
    """Grow the adaptive bracket downward when modeled flux is too high."""
    def metric_function(q0: float) -> Q0MetricEvaluation:
        return Q0MetricEvaluation(
            metrics=MetricValues(
                chi2=(q0 - 2.0) ** 2 + 0.25,
                rho2=(q0 - 2.0) ** 2 + 1.25,
                eta2=(q0 - 2.0) ** 2 + 2.25,
            ),
            total_observed_flux=20.0,
            total_modeled_flux=10.0 * q0,
        )

    result = find_best_q0(
        metric_function,
        q0_min=0.25,
        q0_max=12.0,
        adaptive_bracketing=True,
        q0_start=8.0,
        q0_step=1.7,
        max_bracket_steps=8,
    )

    assert result.success
    assert result.used_adaptive_bracketing
    assert result.bracket_found
    assert result.bracket is not None
    assert result.bracket[0] < 2.0 < result.bracket[2]
    assert result.q0 == pytest.approx(2.0, abs=1e-2)


def test_find_best_q0_adaptive_corrects_initial_flux_direction_from_metric_trend() -> None:
    """Correct the initial adaptive search direction from metric behavior."""
    def metric_function(q0: float) -> Q0MetricEvaluation:
        return Q0MetricEvaluation(
            metrics=MetricValues(
                chi2=(q0 - 0.001) ** 2,
                rho2=(q0 - 0.001) ** 2,
                eta2=(q0 - 0.001) ** 2,
            ),
            total_observed_flux=1.0,
            total_modeled_flux=0.01,
        )

    result = find_best_q0(
        metric_function,
        q0_min=0.0001,
        q0_max=0.01,
        adaptive_bracketing=True,
        q0_start=0.001,
        q0_step=1.61803398875,
        max_bracket_steps=12,
        target_metric="eta2",
    )

    assert result.used_adaptive_bracketing
    assert result.success
    assert result.q0 == pytest.approx(0.001, abs=1e-4)
    assert result.trial_q0[:3] == pytest.approx((0.0001, 0.001, 0.01))


def test_find_best_q0_adaptive_failure_falls_back_to_bounded_refinement() -> None:
    """Fall back to bounded refinement when adaptive expansion stalls."""
    def metric_function(q0: float) -> Q0MetricEvaluation:
        return Q0MetricEvaluation(
            metrics=MetricValues(
                chi2=(q0 - 0.97) ** 2,
                rho2=(q0 - 0.97) ** 2,
                eta2=(q0 - 0.97) ** 2,
            ),
            total_observed_flux=1.0,
            total_modeled_flux=0.1 * q0,
        )

    result = find_best_q0(
        metric_function,
        q0_min=0.9,
        q0_max=1.0,
        adaptive_bracketing=True,
        q0_start=0.91,
        q0_step=1.01,
        max_bracket_steps=1,
    )

    assert result.success
    assert result.used_adaptive_bracketing
    assert result.bracket_found
    assert result.bracket == pytest.approx((0.91, 1.0, 1.01))
    assert "adaptive bracketing found a valid interior minimum" in result.message
    assert result.q0 == pytest.approx(0.97, abs=1e-2)


def test_find_best_q0_soft_interval_expands_beyond_initial_upper_edge() -> None:
    """Allow adaptive search to expand beyond the initial soft upper bound."""
    def metric_function(q0: float) -> Q0MetricEvaluation:
        return Q0MetricEvaluation(
            metrics=MetricValues(
                chi2=(q0 - 1.5) ** 2,
                rho2=(q0 - 1.5) ** 2,
                eta2=(q0 - 1.5) ** 2,
            ),
            total_observed_flux=1.0,
            total_modeled_flux=0.1 * q0,
        )

    result = find_best_q0(
        metric_function,
        q0_min=0.1,
        q0_max=1.0,
        adaptive_bracketing=True,
        q0_start=0.31622776601683794,
        q0_step=1.61803398875,
        max_bracket_steps=20,
    )

    assert result.success
    assert result.used_adaptive_bracketing
    assert result.bracket_found
    assert result.q0 == pytest.approx(1.5, abs=1e-2)
    assert max(result.trial_q0) > 1.0


def test_find_best_q0_hard_upper_bound_stops_expansion() -> None:
    """Stop adaptive expansion when the hard upper limit is reached."""
    def metric_function(q0: float) -> Q0MetricEvaluation:
        return Q0MetricEvaluation(
            metrics=MetricValues(
                chi2=(q0 - 1.5) ** 2,
                rho2=(q0 - 1.5) ** 2,
                eta2=(q0 - 1.5) ** 2,
            ),
            total_observed_flux=1.0,
            total_modeled_flux=0.1 * q0,
        )

    result = find_best_q0(
        metric_function,
        q0_min=0.1,
        q0_max=1.0,
        hard_q0_max=1.0,
        adaptive_bracketing=True,
        q0_start=0.31622776601683794,
        q0_step=1.61803398875,
        max_bracket_steps=20,
    )

    assert not result.success
    assert result.used_adaptive_bracketing
    assert not result.bracket_found
    assert "upper safety bound" in result.message
    assert result.q0 == pytest.approx(1.0, abs=1e-12)


def test_find_best_q0_tracks_unique_trials_and_unique_evaluations() -> None:
    """Track each unique objective evaluation in the result metadata."""
    calls: list[float] = []

    def metric_function(q0: float) -> MetricValues:
        calls.append(float(q0))
        return MetricValues(
            chi2=(q0 - 3.0) ** 2,
            rho2=(q0 - 3.0) ** 2,
            eta2=(q0 - 3.0) ** 2,
        )

    result = find_best_q0(metric_function, q0_min=0.1, q0_max=10.0, target_metric="chi2")

    assert result.success
    assert result.nfev == len(result.trial_q0)
    assert result.nfev == len(calls)
    assert result.trial_q0 == pytest.approx(tuple(calls))
