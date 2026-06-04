import pytest

from pychmp.metrics import MetricValues
from pychmp.optimize import Q0OptimizationResult, find_best_q0
from pychmp.q0_search import merge_q0_stage_results, parse_q0_search_stages, resolve_q0_search_stages
from pychmp.search_options import parse_xy_shift, resolve_chmp_search_settings, resolve_shift_policy_from_args


def test_parse_q0_search_stages_accepts_data_union() -> None:
    assert parse_q0_search_stages("data,union") == ("data", "union")


def test_parse_q0_search_stages_rejects_unknown_stage() -> None:
    with pytest.raises(ValueError, match="unsupported q0 search stage"):
        parse_q0_search_stages("data,foo")


def test_resolve_q0_search_stages_defaults_to_mask_type() -> None:
    assert resolve_q0_search_stages(q0_search_stages=None, mask_type="union", explicit_mask=None) == ("union",)


def test_resolve_q0_search_stages_rejects_two_stage_with_explicit_mask() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        resolve_q0_search_stages(
            q0_search_stages=("data", "union"),
            mask_type="union",
            explicit_mask=object(),
        )


def test_parse_xy_shift_parses_pair() -> None:
    assert parse_xy_shift("1.5,-2.0") == (1.5, -2.0)


def test_resolve_shift_policy_from_args_xy_shift_forces_fixed() -> None:
    class Args:
        shift_policy = "auto"
        max_shift_arcsec = None
        xy_shift_arcsec = "3,4"

    policy, max_shift, xy = resolve_shift_policy_from_args(Args())
    assert policy == "fixed"
    assert max_shift is None
    assert xy == (3.0, 4.0)


def test_resolve_chmp_search_settings_two_stage() -> None:
    class Args:
        shift_policy = "auto"
        max_shift_arcsec = None
        xy_shift_arcsec = None
        use_smoothed_obs_max = True
        use_emthreshold = True
        emthreshold = 0.1
        q0_search_stages = "data,union"

    settings = resolve_chmp_search_settings(Args(), mask_type="union", explicit_mask=None)
    assert settings.q0_search_stages == ("data", "union")


def test_merge_q0_stage_results_concatenates_trials() -> None:
    stage_one = Q0OptimizationResult(
        q0=0.2,
        objective_value=1.0,
        metrics=MetricValues(chi2=1.0, rho2=2.0, eta2=3.0),
        target_metric="chi2",
        success=True,
        nfev=2,
        nit=1,
        message="stage one",
        trial_q0=(0.1, 0.2),
        trial_objective_values=(2.0, 1.0),
        trial_chi2_values=(2.0, 1.0),
        trial_rho2_values=(2.0, 1.0),
        trial_eta2_values=(2.0, 1.0),
        trial_mask_stages=("data", "data"),
    )
    stage_two = Q0OptimizationResult(
        q0=0.15,
        objective_value=0.5,
        metrics=MetricValues(chi2=0.5, rho2=0.6, eta2=0.7),
        target_metric="chi2",
        success=True,
        nfev=3,
        nit=2,
        message="stage two",
        trial_q0=(0.15,),
        trial_objective_values=(0.5,),
        trial_chi2_values=(0.5,),
        trial_rho2_values=(0.6,),
        trial_eta2_values=(0.7,),
        trial_mask_stages=("union",),
    )
    merged = merge_q0_stage_results((stage_one, stage_two), ("data", "union"))
    assert merged.q0 == pytest.approx(0.15)
    assert merged.trial_q0 == (0.1, 0.2, 0.15)
    assert merged.trial_mask_stages == ("data", "data", "union")
    assert merged.nfev == 5
    assert merged.q0_search_stages == ("data", "union")


def test_fit_q0_two_stage_invokes_both_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    from pychmp import fitting

    calls: list[tuple[str, float | None]] = []

    class Renderer:
        def render(self, q0: float):
            import numpy as np

            arr = np.full((4, 4), float(q0), dtype=float)
            return arr

    def fake_run_single_stage_q0_fit(*_args, stage_mask_type: str, q0_start, **_kwargs):
        calls.append((stage_mask_type, q0_start))
        q0 = 0.2 if stage_mask_type == "data" else 0.15
        return Q0OptimizationResult(
            q0=q0,
            objective_value=float(q0),
            metrics=MetricValues(chi2=float(q0), rho2=float(q0), eta2=float(q0)),
            target_metric="chi2",
            success=True,
            nfev=1,
            nit=1,
            message=stage_mask_type,
            trial_q0=(q0,),
            trial_objective_values=(float(q0),),
            trial_chi2_values=(float(q0),),
            trial_rho2_values=(float(q0),),
            trial_eta2_values=(float(q0),),
            trial_mask_stages=(stage_mask_type,),
        )

    monkeypatch.setattr(fitting, "_run_single_stage_q0_fit", fake_run_single_stage_q0_fit)

    import numpy as np

    result = fitting.fit_q0_to_observation(
        Renderer(),
        np.ones((4, 4), dtype=float),
        np.ones((4, 4), dtype=float),
        q0_min=0.01,
        q0_max=1.0,
        q0_search_stages=("data", "union"),
    )
    assert calls == [("data", None), ("union", pytest.approx(0.2))]
    assert result.q0 == pytest.approx(0.15)
    assert result.q0_search_stages == ("data", "union")


def test_fit_q0_two_stage_clamps_inter_stage_q0_start(monkeypatch: pytest.MonkeyPatch) -> None:
    from pychmp import fitting

    calls: list[tuple[str, float | None]] = []

    class Renderer:
        def render(self, q0: float):
            import numpy as np

            return np.full((4, 4), float(q0), dtype=float)

    def fake_run_single_stage_q0_fit(*_args, stage_mask_type: str, q0_start, **_kwargs):
        calls.append((stage_mask_type, q0_start))
        if stage_mask_type == "data":
            q0 = 0.001618
        else:
            q0 = 0.0005
        return Q0OptimizationResult(
            q0=q0,
            objective_value=float(q0),
            metrics=MetricValues(chi2=float(q0), rho2=float(q0), eta2=float(q0)),
            target_metric="eta2",
            success=True,
            nfev=1,
            nit=1,
            message=stage_mask_type,
            trial_q0=(q0,),
            trial_objective_values=(float(q0),),
            trial_chi2_values=(float(q0),),
            trial_rho2_values=(float(q0),),
            trial_eta2_values=(float(q0),),
            trial_mask_stages=(stage_mask_type,),
        )

    monkeypatch.setattr(fitting, "_run_single_stage_q0_fit", fake_run_single_stage_q0_fit)

    import numpy as np

    result = fitting.fit_q0_to_observation(
        Renderer(),
        np.ones((4, 4), dtype=float),
        np.ones((4, 4), dtype=float),
        q0_min=1.0e-5,
        q0_max=0.001,
        q0_search_stages=("data", "union"),
    )
    assert calls == [("data", None), ("union", pytest.approx(0.001))]
    assert result.q0 == pytest.approx(0.0005)
