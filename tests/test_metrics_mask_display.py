from __future__ import annotations

import numpy as np
from astropy.io import fits

from pychmp.metrics import (
    format_metrics_mask_label,
    resolve_metrics_mask_type,
    resolve_metrics_threshold_mask,
    threshold_data_mask,
    threshold_union_mask,
)


def _make_header(*, cdelt: float = 2.0) -> fits.Header:
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 5
    header["NAXIS2"] = 5
    header["CDELT1"] = cdelt
    header["CDELT2"] = cdelt
    header["CRPIX1"] = 3
    header["CRPIX2"] = 3
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    return header


def test_resolve_metrics_mask_type_uses_selected_trial_stage() -> None:
    diagnostics = {
        "mask_type": "union",
        "fit_trial_mask_stages": ["data", "union", "union"],
        "selected_trial_index": 0,
    }
    assert resolve_metrics_mask_type(diagnostics) == "data"


def test_resolve_metrics_threshold_mask_uses_smoothed_obs_peak() -> None:
    observed = np.zeros((5, 5), dtype=float)
    observed[2, 2] = 100.0
    observed[0, 0] = 90.0
    modeled = np.full((5, 5), 1.0, dtype=float)
    header = _make_header()
    diagnostics = {
        "mask_type": "data",
        "metrics_mask_threshold": 0.1,
        "use_smoothed_obs_max": True,
    }
    mask = resolve_metrics_threshold_mask(observed, modeled, diagnostics, wcs_header=header)
    raw_mask = threshold_data_mask(observed, modeled, 0.1)
    assert mask is not None
    assert int(np.count_nonzero(mask)) <= int(np.count_nonzero(raw_mask))


def test_union_mask_can_be_larger_than_data_mask_at_low_heating() -> None:
    observed = np.zeros((7, 7), dtype=float)
    observed[3, 3] = 100.0
    modeled = np.full((7, 7), 0.01, dtype=float)
    modeled[3, 3] = 0.02
    data_mask = threshold_data_mask(observed, modeled, 0.1)
    union_mask = threshold_union_mask(observed, modeled, 0.1)
    assert int(np.count_nonzero(union_mask)) >= int(np.count_nonzero(data_mask))


def test_format_metrics_mask_label_includes_stage_and_smoothed_note() -> None:
    label = format_metrics_mask_label(
        {
            "mask_type": "union",
            "metrics_mask_threshold": 0.1,
            "fit_trial_mask_stages": ["data"],
            "selected_trial_index": 0,
            "use_smoothed_obs_max": True,
        }
    )
    assert "data" in label
    assert "0.100" in label
    assert "smoothed obs peak" in label
