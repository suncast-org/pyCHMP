from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy.io import fits

from pychmp.ab_scan_artifacts import (
    _read_map_store_ref_array,
    _write_map_store_array,
    decode_scalar,
    write_point_scan_artifact,
)
from pychmp.chmp_evaluation import ObservationEvaluationContext
from pychmp.grid_points import (
    ACTIVE_SEARCH_ID_DATASET,
    GRID_POINTS_GROUP,
    GRID_POINTS_TRIALS_GROUP,
    SEARCHES_GROUP,
    SLICE_CONTAINER_GROUP,
    GridPointAssignedEvent,
    GridTrialCommittedEvent,
    apply_grid_point_assigned,
    apply_grid_trial_committed,
)
from pychmp.slice_map_index import SliceMapIndex, build_slice_map_index
from pychmp.warm_q0 import build_warm_grid_trial_commit_events


def _header() -> fits.Header:
    header = fits.Header()
    header["SIMPLE"] = True
    header["BITPIX"] = -32
    header["NAXIS"] = 2
    header["NAXIS1"] = 2
    header["NAXIS2"] = 2
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRPIX1"] = 1.0
    header["CRPIX2"] = 1.0
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CDELT1"] = 2.0
    header["CDELT2"] = 2.0
    header["DATE-OBS"] = "2020-11-26T20:00:00"
    return header


def test_grid_trial_commit_links_existing_map_store_ref(tmp_path: Path) -> None:
    observed = np.array([[0.0, 2.0], [0.0, 0.0]], dtype=float)
    sigma = np.ones((2, 2), dtype=float)
    header = _header()
    diagnostics = {
        "artifact_kind": "pychmp_ab_scan_sparse_points",
        "target_slice_key": "mw_5p700000ghz",
        "slice_key": "mw_5p700000ghz",
        "target_metric": "eta2",
        "spectral_domain": "mw",
        "spectral_label": "5.700 GHz",
        "frequency_ghz": 5.7,
        "metrics_mask_threshold": 0.1,
        "shift_policy": "fixed",
        "selected_search_id": "search_test",
        "search_id": "search_test",
    }
    artifact_h5 = tmp_path / "warm_ref.h5"
    write_point_scan_artifact(
        artifact_h5,
        observed=observed,
        sigma_map=sigma,
        wcs_header=header,
        diagnostics=diagnostics,
        point_records=[],
    )
    stored_map = np.array([[0.0, 1.8], [0.0, 0.0]], dtype=float)
    with h5py.File(artifact_h5, "a") as f:
        ref_path = _write_map_store_array(
            f,
            identity={"name": "test/warm_ref"},
            data=stored_map,
        )

    index = SliceMapIndex(slice_key="mw_5p700000ghz", descriptor={"key": "mw_5p700000ghz", "domain": "mw", "frequency_ghz": 5.7})
    index.register(a=0.2, b=2.5, q0=5e-4, raw_map_ref=ref_path)

    point_id = apply_grid_point_assigned(
        artifact_h5,
        GridPointAssignedEvent(a=0.2, b=2.5, q0_start=5e-4, next_q0=5e-4, metric_name="eta2"),
        observed=observed,
        sigma_map=sigma,
        wcs_header=header,
        diagnostics=diagnostics,
    )
    context = ObservationEvaluationContext(
        model_header=header,
        shift_policy="fixed",
        observed=observed,
        sigma=sigma,
        use_smoothed_obs_max=True,
    )
    events = build_warm_grid_trial_commit_events(
        artifact_h5,
        point_id=str(point_id),
        slice_map_index=index,
        a_value=0.2,
        b_value=2.5,
        context=context,
        threshold=0.1,
        explicit_mask=None,
        target_metric="eta2",
        use_emthreshold=False,
    )
    assert len(events) == 1
    apply_grid_trial_committed(
        artifact_h5,
        events[0],
        observed=observed,
        sigma_map=sigma,
        wcs_header=header,
        diagnostics=diagnostics,
    )

    with h5py.File(artifact_h5, "r") as f:
        slice_group = f[SLICE_CONTAINER_GROUP]["mw_5p700000ghz"]
        search_id = decode_scalar(slice_group[ACTIVE_SEARCH_ID_DATASET][()])
        trial = f[
            f"{SLICE_CONTAINER_GROUP}/mw_5p700000ghz/{SEARCHES_GROUP}/{search_id}/"
            f"{GRID_POINTS_GROUP}/{point_id}/{GRID_POINTS_TRIALS_GROUP}/t000000"
        ]
        assert np.isfinite(float(trial.attrs["chi2"]))
        assert np.isfinite(float(trial.attrs["eta2"]))
        assert np.isfinite(float(trial.attrs["rho2"]))
        map_refs = trial["map_refs_json"][()].decode()
        assert ref_path in map_refs
        assert "map_store_rescore" not in map_refs
        stored = _read_map_store_ref_array(f, ref_path)
        assert stored is not None
        assert np.allclose(stored, stored_map)
