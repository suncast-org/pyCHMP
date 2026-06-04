"""Tests for grid trial map_store repair utility."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits

from pychmp.ab_scan_artifacts import MAP_REFS_DATASET, _create_text_dataset, _json_dumps
from pychmp.grid_points import (
    GridPointAssignedEvent,
    GridTrialCommittedEvent,
    apply_grid_point_event_with_retry,
    grid_point_finite_q0_trials_have_map_store_links,
    read_grid_point_header,
)
from pychmp.repair_grid_trial_maps import (
    artifact_grid_trials_are_clean,
    grid_trial_row_should_purge,
    repair_grid_trial_maps_in_artifact,
)


def _make_header() -> fits.Header:
    return fits.Header(
        {
            "NAXIS": 2,
            "NAXIS1": 2,
            "NAXIS2": 2,
            "CDELT1": 1.0,
            "CDELT2": 1.0,
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "CRPIX1": 1.0,
            "CRPIX2": 1.0,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
        }
    )


def _make_diagnostics() -> dict:
    return {
        "search_id": "search0001",
        "target_metric": "eta2",
        "slice_key": "mw_5p700000ghz",
        "artifact_kind": "unified_ab_scan",
        "q0_start": 1e-4,
        "q0_min": 1e-5,
        "q0_max": 1e-3,
    }


def test_repair_purges_metrics_only_and_phantom_trials(tmp_path: Path) -> None:
    artifact_h5 = tmp_path / "repair.h5"
    observed = np.ones((2, 2), dtype=float)
    sigma = np.ones((2, 2), dtype=float)
    header = _make_header()
    diagnostics = _make_diagnostics()
    raw_map = np.full((2, 2), 3.0, dtype=np.float32)

    point_id = apply_grid_point_event_with_retry(
        artifact_h5,
        GridPointAssignedEvent(a=-0.3, b=2.7, q0_start=1e-4, next_q0=1e-4, metric_name="eta2"),
        observed=observed,
        sigma_map=sigma,
        wcs_header=header,
        diagnostics=diagnostics,
    )
    apply_grid_point_event_with_retry(
        artifact_h5,
        GridTrialCommittedEvent(
            point_id=str(point_id),
            trial_index=0,
            q0=1.0e-4,
            metric=0.42,
            next_q0=1.618e-4,
            best_trial_index=0,
            best_metric=0.42,
            raw_modeled_map=raw_map,
            eta2=0.42,
        ),
        observed=observed,
        sigma_map=sigma,
        wcs_header=header,
        diagnostics=diagnostics,
    )

    with h5py.File(artifact_h5, "a") as f:
        trials_group = f["slices"]["mw_5p700000ghz"]["searches"]["search0001"]["grid_points"][str(point_id)][
            "trials"
        ]
        broken = trials_group.create_group("t000001")
        broken.attrs["trial_index"] = 1
        broken.attrs["q0"] = 1.618e-4
        broken.attrs["metric"] = 0.40
        broken.attrs["target_metric"] = np.bytes_("eta2")
        broken.attrs["chi2"] = 1.0
        broken.attrs["rho2"] = 0.5
        broken.attrs["eta2"] = 0.40
        _create_text_dataset(broken, MAP_REFS_DATASET, _json_dumps({}))

        phantom = trials_group.create_group("t0123")
        phantom.attrs["trial_index"] = 123
        phantom.attrs["q0"] = float("nan")
        phantom.attrs["metric"] = 0.1
        _create_text_dataset(
            phantom,
            MAP_REFS_DATASET,
            _json_dumps({"raw_modeled": "map_store/phantom"}),
        )

    assert not artifact_grid_trials_are_clean(artifact_h5)
    report = repair_grid_trial_maps_in_artifact(artifact_h5)
    assert report.purged_trial_count == 2
    assert report.clean
    assert artifact_grid_trials_are_clean(artifact_h5)
    assert grid_point_finite_q0_trials_have_map_store_links(
        artifact_h5,
        slice_key="mw_5p700000ghz",
        search_id="search0001",
        point_id=str(point_id),
    )
    with h5py.File(artifact_h5, "r") as f:
        trials_group = f["slices"]["mw_5p700000ghz"]["searches"]["search0001"]["grid_points"][str(point_id)][
            "trials"
        ]
        assert sorted(trials_group.keys()) == ["t000000"]
        point_header = read_grid_point_header(trials_group.parent)
        assert int(point_header["n_trials"]) == 1
        assert int(point_header["best_trial_index"]) == 0


def test_grid_trial_row_should_purge_detects_broken_ref(tmp_path: Path) -> None:
    artifact_h5 = tmp_path / "broken_ref.h5"
    with h5py.File(artifact_h5, "w") as f:
        trial = f.create_group("trial")
        trial.attrs["q0"] = 1e-4
        _create_text_dataset(trial, MAP_REFS_DATASET, _json_dumps({"raw_modeled": "map_store/missing"}))
        should_purge, reason = grid_trial_row_should_purge(f, trial)
    assert should_purge
    assert reason == "broken_map_ref"
