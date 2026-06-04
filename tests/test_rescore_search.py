from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy.io import fits

from pychmp.ab_scan_artifacts import (
    SEARCHES_GROUP,
    SLICE_CONTAINER_GROUP,
    _write_map_store_array,
    decode_scalar,
    write_point_scan_artifact,
)
from pychmp.ab_scan_artifacts import _json_dumps, _replace_text_dataset, SEARCH_LIFECYCLE_DATASET
from pychmp.grid_points import (
    ACTIVE_SEARCH_ID_DATASET,
    GRID_POINTS_GROUP,
    GridPointAssignedEvent,
    GridPointCompletedEvent,
    GridPointFailedEvent,
    GridTrialCommittedEvent,
    apply_grid_point_assigned,
    apply_grid_point_completed,
    apply_grid_point_failed,
    apply_grid_trial_committed,
)
from pychmp.rescore_search import (
    RESCORE_META_DATASET,
    RESCORE_SEARCH_MODE,
    RescoreBuildError,
    RescoreCommitError,
    build_rescore_sidecar,
    commit_rescore_sidecar,
    derive_rescore_search_id,
    rescore_root_search_id,
)
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


def _seed_search_with_map_and_grid(tmp_path: Path) -> tuple[Path, str, str]:
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
        "selected_search_id": "search_source",
        "search_id": "search_source",
        "search_active": False,
    }
    artifact_h5 = tmp_path / "main.h5"
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
            identity={
                "name": "test/rescore",
                "a": 0.2,
                "b": 2.5,
                "q0": 5e-4,
                "domain": "mw",
                "frequency_ghz": 5.7,
                "component": "stokes_i",
            },
            data=stored_map,
        )
        search_id = decode_scalar(f[f"{SLICE_CONTAINER_GROUP}/mw_5p700000ghz/{ACTIVE_SEARCH_ID_DATASET}"][()])
        point_id = apply_grid_point_assigned(
            artifact_h5,
            GridPointAssignedEvent(a=0.2, b=2.5, q0_start=5e-4, next_q0=5e-4, metric_name="eta2"),
            observed=observed,
            sigma_map=sigma,
            wcs_header=header,
            diagnostics=diagnostics,
        )
        apply_grid_trial_committed(
            artifact_h5,
            GridTrialCommittedEvent(
                point_id=str(point_id),
                trial_index=0,
                q0=5e-4,
                metric=0.5,
                next_q0=None,
                best_trial_index=0,
                best_metric=0.5,
                raw_map_ref=ref_path,
                chi2=0.5,
                rho2=0.4,
                eta2=0.3,
            ),
            observed=observed,
            sigma_map=sigma,
            wcs_header=header,
            diagnostics=diagnostics,
        )
        apply_grid_point_completed(
            artifact_h5,
            GridPointCompletedEvent(point_id=str(point_id), best_trial_index=0, best_metric=0.5, best_q0=5e-4),
            observed=observed,
            sigma_map=sigma,
            wcs_header=header,
            diagnostics=diagnostics,
        )
        search_group = f[f"{SLICE_CONTAINER_GROUP}/mw_5p700000ghz/{SEARCHES_GROUP}/{search_id}"]
        lifecycle = {
            "status": "complete",
            "active": False,
            "in_progress": False,
            "completed_at": "2020-11-26T21:00:00Z",
        }
        _replace_text_dataset(search_group, SEARCH_LIFECYCLE_DATASET, _json_dumps(lifecycle))
        search_group.attrs["active"] = 0
        search_group.attrs["in_progress"] = 0
        search_group.attrs["status"] = np.bytes_("complete")
    with h5py.File(artifact_h5, "r") as f:
        resolved_id = decode_scalar(f[f"{SLICE_CONTAINER_GROUP}/mw_5p700000ghz/{ACTIVE_SEARCH_ID_DATASET}"][()])
    return artifact_h5, str(resolved_id), "mw_5p700000ghz"


def test_derive_rescore_search_id_increments(tmp_path: Path) -> None:
    artifact_h5, source_id, slice_key = _seed_search_with_map_and_grid(tmp_path)
    assert derive_rescore_search_id(artifact_h5, source_search_id=source_id, slice_key=slice_key) == "search_source_r1"
    with h5py.File(artifact_h5, "a") as f:
        searches = f[f"{SLICE_CONTAINER_GROUP}/{slice_key}/{SEARCHES_GROUP}"]
        searches.create_group("search_source_r1")
    assert derive_rescore_search_id(artifact_h5, source_search_id=source_id, slice_key=slice_key) == "search_source_r2"
    assert rescore_root_search_id("search_source_r2") == "search_source"


def test_build_and_commit_rescore_sidecar(tmp_path: Path) -> None:
    artifact_h5, source_id, slice_key = _seed_search_with_map_and_grid(tmp_path)
    report = build_rescore_sidecar(artifact_h5, source_search_id=source_id, footprint="source")
    assert report.target_search_id == "search_source_r1"
    assert report.sidecar_path.exists()
    assert report.points_rescored == 1

    with h5py.File(report.sidecar_path, "r") as f:
        meta = f[RESCORE_META_DATASET][()].decode()
        assert "ready" in meta
        diag = f[f"{SLICE_CONTAINER_GROUP}/{slice_key}/{SEARCHES_GROUP}/search_source_r1/diagnostics_json"][()].decode()
        assert RESCORE_SEARCH_MODE in diag

    commit = commit_rescore_sidecar(artifact_h5, sidecar_path=report.sidecar_path)
    assert commit.grid_point_count == 1
    with h5py.File(artifact_h5, "r") as f:
        assert "search_source_r1" in f[f"{SLICE_CONTAINER_GROUP}/{slice_key}/{SEARCHES_GROUP}"]


def test_commit_refuses_when_search_active(tmp_path: Path) -> None:
    artifact_h5, source_id, slice_key = _seed_search_with_map_and_grid(tmp_path)
    report = build_rescore_sidecar(artifact_h5, source_search_id=source_id)
    with h5py.File(artifact_h5, "a") as f:
        search_group = f[f"{SLICE_CONTAINER_GROUP}/{slice_key}/{SEARCHES_GROUP}/{source_id}"]
        lifecycle = {"status": "in_progress", "active": True, "in_progress": True}
        _replace_text_dataset(search_group, SEARCH_LIFECYCLE_DATASET, _json_dumps(lifecycle))
        search_group.attrs["active"] = 1
        search_group.attrs["in_progress"] = 1
    with pytest.raises(RescoreCommitError, match="still active"):
        commit_rescore_sidecar(artifact_h5, sidecar_path=report.sidecar_path)


def test_build_fails_when_source_point_has_no_maps(tmp_path: Path) -> None:
    artifact_h5, source_id, _slice_key = _seed_search_with_map_and_grid(tmp_path)
    observed = np.ones((2, 2), dtype=float)
    sigma = np.ones((2, 2), dtype=float)
    header = _header()
    diagnostics = {
        "artifact_kind": "pychmp_ab_scan_sparse_points",
        "target_slice_key": "mw_5p700000ghz",
        "target_metric": "eta2",
        "shift_policy": "fixed",
        "selected_search_id": source_id,
        "search_id": source_id,
    }
    with h5py.File(artifact_h5, "a") as f:
        search_group = f[f"{SLICE_CONTAINER_GROUP}/mw_5p700000ghz/{SEARCHES_GROUP}/{source_id}"]
        point_id = apply_grid_point_assigned(
            artifact_h5,
            GridPointAssignedEvent(a=0.5, b=3.0, q0_start=1e-3, next_q0=1e-3, metric_name="eta2", force_new_point_id=True),
            observed=observed,
            sigma_map=sigma,
            wcs_header=header,
            diagnostics=diagnostics,
        )
        apply_grid_point_failed(
            artifact_h5,
            GridPointFailedEvent(
                point_id=str(point_id),
                error_message="missing maps",
            ),
            observed=observed,
            sigma_map=sigma,
            wcs_header=header,
            diagnostics=diagnostics,
        )
    with pytest.raises(RescoreBuildError, match="incomplete"):
        build_rescore_sidecar(artifact_h5, source_search_id=source_id, footprint="source")
