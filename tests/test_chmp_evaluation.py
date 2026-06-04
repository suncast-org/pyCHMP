from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from pychmp.chmp_evaluation import ObservationEvaluationContext, evaluate_modeled_trial
from pychmp.metrics import chmp_mask_valid


def _header(nx: int, ny: int) -> fits.Header:
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = nx
    header["NAXIS2"] = ny
    header["CDELT1"] = 1.0
    header["CDELT2"] = 1.0
    header["CRPIX1"] = (float(nx) + 1.0) / 2.0
    header["CRPIX2"] = (float(ny) + 1.0) / 2.0
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    return header


def test_chmp_mask_valid_rejects_high_model_fraction() -> None:
    valid, message = chmp_mask_valid(mask_obs_fraction=0.1, mask_mod_fraction=0.995)
    assert not valid
    assert "0.99" in message


def test_evaluate_modeled_trial_fixed_policy_computes_metrics() -> None:
    observed = np.zeros((8, 8), dtype=float)
    observed[3:5, 3:5] = 10.0
    modeled = observed.copy()
    modeled[3:5, 3:5] = 11.0
    sigma = np.full((8, 8), 0.1, dtype=float)
    context = ObservationEvaluationContext(
        model_header=_header(8, 8),
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
    assert result.is_valid
    assert np.isfinite(result.metrics.chi2)
    assert result.shift_x_arcsec == 0.0
    assert result.shift_y_arcsec == 0.0


def test_evaluate_modeled_trial_invalid_mask_still_reports_flux_totals() -> None:
    observed = np.zeros((8, 8), dtype=float)
    observed[3:5, 3:5] = 10.0
    modeled = np.zeros((8, 8), dtype=float)
    modeled[:, :] = 1.0
    sigma = np.full((8, 8), 0.1, dtype=float)
    context = ObservationEvaluationContext(
        model_header=_header(8, 8),
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
    assert result.is_valid
    assert result.mask_stage == "data"
    assert result.total_observed_flux is not None
    assert result.total_modeled_flux is not None
