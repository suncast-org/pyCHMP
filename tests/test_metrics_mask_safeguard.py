import numpy as np
import pytest

from pychmp.chmp_evaluation import ObservationEvaluationContext, evaluate_modeled_trial
from pychmp.metrics import should_prefer_data_mask_over_union


def test_should_prefer_data_mask_when_union_inflated_by_underestimation() -> None:
    assert should_prefer_data_mask_over_union(
        mask_obs_fraction=0.05,
        mask_mod_fraction=0.25,
        total_observed_flux=100.0,
        total_modeled_flux=20.0,
    )


def test_evaluate_modeled_trial_falls_back_to_data_mask() -> None:
    from astropy.io import fits

    observed = np.zeros((16, 16), dtype=float)
    observed[6:10, 6:10] = 100.0
    modeled = np.zeros((16, 16), dtype=float)
    modeled[:, :] = 2.0
    sigma = np.full((16, 16), 1.0, dtype=float)
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 16
    header["NAXIS2"] = 16
    header["CDELT1"] = 1.0
    header["CDELT2"] = 1.0
    context = ObservationEvaluationContext(
        model_header=header,
        shift_policy="fixed",
        observed=observed,
        sigma=sigma,
        use_smoothed_obs_max=False,
    )
    result = evaluate_modeled_trial(
        modeled,
        context,
        threshold=0.1,
        mask_type="union",
        use_emthreshold=False,
    )
    assert result.mask_stage == "data"
    assert result.is_valid
