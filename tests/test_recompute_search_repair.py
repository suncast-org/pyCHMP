"""Tests for --recompute-search-id repair semantics."""

from __future__ import annotations

import sys
from argparse import Namespace

import numpy as np
import pytest

from pychmp.ab_scan_artifacts import assert_recompute_search_cli_argv_allowed
from pychmp.warm_q0 import _evaluation_from_stored_grid_trial, initial_evaluations_from_grid_trials


def test_assert_recompute_search_cli_rejects_recipe_overrides() -> None:
    argv = ["prog", "--artifact-h5", "a.h5", "--recompute-search-id", "search_x", "--target-metric", "eta2"]
    with pytest.raises(SystemExit, match="Disallowed"):
        assert_recompute_search_cli_argv_allowed(argv)


def test_assert_recompute_search_cli_allows_viewer_flags() -> None:
    argv = ["prog", "--artifact-h5", "a.h5", "--recompute-search-id", "search_x", "--no-viewer"]
    assert_recompute_search_cli_argv_allowed(argv)


def test_restore_stored_slice_identity_uses_artifact_geometry() -> None:
    from astropy.io import fits

    from examples.python.adaptive_ab_search_single_observation import (
        _restore_stored_slice_identity_from_artifact,
    )

    stored_header = fits.Header({"NAXIS": 2, "CRVAL1": 0.0, "CRVAL2": 180.0})
    resume_diag = {
        "map_xc_arcsec": 1.0,
        "map_yc_arcsec": 181.0,
        "map_dx_arcsec": 2.5,
        "map_dy_arcsec": 2.5,
        "map_nx": 128,
        "map_ny": 128,
        "observer_name": "earth",
        "observer_lonc_deg": 0.0,
        "observer_b0sun_deg": 0.0,
        "observer_dsun_cm": 1.49e11,
        "observer_obs_time": "2026-04-03T19:59:33",
        "artifact_geometry_sha256": "abc123",
    }
    geometry = type("G", (), {"xc": 0.0, "yc": 0.0, "dx": 2.0, "dy": 2.0, "nx": 200, "ny": 200})()
    header, geometry_out, _block, sha = _restore_stored_slice_identity_from_artifact(
        resume_slice_payload={"wcs_header": stored_header},
        resume_slice_diagnostics=resume_diag,
        target_header=fits.Header({"NAXIS": 2}),
        geometry=geometry,
    )
    assert header["CRVAL2"] == 180.0
    assert float(geometry_out.xc) == pytest.approx(1.0)
    assert int(geometry_out.nx) == 128
    assert sha == "abc123"


def test_initial_evaluations_from_grid_trials_without_rescore() -> None:
    trials = [
        {
            "trial_index": 0,
            "q0": 1.0e-4,
            "target_metric_value": 0.42,
            "chi2": 1.0,
            "rho2": 0.5,
            "eta2": 0.42,
            "raw_map_ref": "/map_store/maps/m000001",
            "trial_metadata": {"shift_x_arcsec": -10.0, "shift_y_arcsec": 1.0, "shift_valid": True},
        }
    ]
    evaluations = initial_evaluations_from_grid_trials(
        __import__("pathlib").Path("/nonexistent.h5"),
        trials,
        slice_key="mw_test",
        target_metric="eta2",
        context=Namespace(),
        threshold=0.1,
        explicit_mask=None,
        rescore=False,
    )
    assert evaluations is not None
    assert float(evaluations[1.0e-4].metrics.eta2) == pytest.approx(0.42)
    assert evaluations[1.0e-4].message == "restored from stored grid trial"


def test_evaluation_from_stored_grid_trial_requires_map_ref() -> None:
    assert _evaluation_from_stored_grid_trial({"q0": 1e-4}, target_metric="eta2") is None
