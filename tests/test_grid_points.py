from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy.io import fits

from pychmp.ab_scan_artifacts import load_scan_file
from pychmp.grid_points import (
    GRID_POINTS_CONTRACT_VERSION,
    GRID_POINTS_GROUP,
    GridPointAssignedEvent,
    GridPointCompletedEvent,
    GridTrialCommittedEvent,
    apply_grid_point_event_with_retry,
    classify_grid_point_state,
    list_grid_point_headers,
    GRID_POINT_STORAGE_CORRUPT_ATTR,
    find_grid_point_group,
    grid_point_finite_q0_trials_have_map_store_links,
    load_grid_point_live_state,
    load_grid_point_trial_plot_payload,
    load_grid_points_as_viewer_records,
    read_grid_point_header,
    reset_grid_point_for_rerun,
)
from pychmp.refresh_signal import REFRESH_V2_VERSION, RefreshSignalWriter
from pychmp.search_contract import search_id_from_evaluation_config


def _make_header() -> fits.Header:
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
    header["OBSERVER"] = "earth"
    return header


def _make_diagnostics(*, slice_key: str = "mw_5p700000ghz") -> dict[str, object]:
    return {
        "artifact_kind": "pychmp_ab_scan_sparse_points",
        "target_metric": "chi2",
        "target_slice_key": slice_key,
        "slice_key": slice_key,
        "metrics_mask_threshold": 0.1,
        "metrics_mask_source": "union_threshold",
        "model_path": "/tmp/model.h5",
        "model_id": "model-123",
        "model_sha256": "a" * 64,
        "fits_file": "/tmp/obs.fits",
        "fits_sha256": "b" * 64,
        "ebtel_path": "/tmp/ebtel.bin",
        "ebtel_sha256": "c" * 64,
        "frequency_ghz": 5.7,
        "map_xc_arcsec": 0.0,
        "map_yc_arcsec": 0.0,
        "map_dx_arcsec": 2.0,
        "map_dy_arcsec": 2.0,
        "map_nx": 2,
        "map_ny": 2,
        "observer_name": "earth",
        "observer_lonc_deg": 0.0,
        "observer_b0sun_deg": 0.0,
        "observer_dsun_cm": 1.495978707e13,
        "observer_obs_time": "2020-11-26T20:00:00",
        "contract_version": GRID_POINTS_CONTRACT_VERSION,
        "search_id": "search0001",
        "selected_search_id": "search0001",
    }


def test_grid_point_assign_commit_complete_cycle(tmp_path: Path) -> None:
    artifact_h5 = tmp_path / "grid_test.h5"
    observed = np.ones((2, 2), dtype=float)
    sigma = np.ones((2, 2), dtype=float)
    header = _make_header()
    diagnostics = _make_diagnostics()
    raw_map = np.full((2, 2), 3.0, dtype=np.float32)

    point_id = apply_grid_point_event_with_retry(
        artifact_h5,
        GridPointAssignedEvent(
            a=-0.3,
            b=2.1,
            q0_start=1e-4,
            next_q0=1e-4,
            metric_name="chi2",
        ),
        observed=observed,
        sigma_map=sigma,
        wcs_header=header,
        diagnostics=diagnostics,
    )
    assert point_id == "p000000"

    apply_grid_point_event_with_retry(
        artifact_h5,
        GridTrialCommittedEvent(
            point_id=str(point_id),
            trial_index=0,
            q0=1e-4,
            metric=0.42,
            next_q0=1.618e-4,
            best_trial_index=0,
            best_metric=0.42,
            raw_modeled_map=raw_map,
            trial_metadata={"stage": "data"},
            shift_x=1.25,
            shift_y=-0.75,
            shift_valid=True,
        ),
        observed=observed,
        sigma_map=sigma,
        wcs_header=header,
        diagnostics=diagnostics,
    )
    apply_grid_point_event_with_retry(
        artifact_h5,
        GridPointCompletedEvent(
            point_id=str(point_id),
            best_trial_index=0,
            best_metric=0.42,
            best_q0=1e-4,
        ),
        observed=observed,
        sigma_map=sigma,
        wcs_header=header,
        diagnostics=diagnostics,
    )

    with h5py.File(artifact_h5, "r") as f:
        slice_group = f["slices"]["mw_5p700000ghz"]
        search_id = slice_group["active_search_id"][()].decode("utf-8")
        search_group = slice_group["searches"][search_id]
        point_group = search_group[GRID_POINTS_GROUP]["p000000"]
        header_state = read_grid_point_header(point_group)
        assert header_state["status"] == "COMPLETED"
        assert "next_q0" not in header_state
        assert int(header_state["n_trials"]) == 1
        assert int(header_state["best_trial_index"]) == 0
        diagnostics_json = search_group["diagnostics_json"][()].decode("utf-8")
        assert GRID_POINTS_CONTRACT_VERSION in diagnostics_json

    payload = load_scan_file(artifact_h5, slice_key="mw_5p700000ghz", search_id=search_id, include_maps=False)
    assert len(payload["point_records"]) == 1
    record = payload["point_records"][0]
    assert float(record["a"]) == pytest.approx(-0.3)
    assert float(record["b"]) == pytest.approx(2.1)
    assert record["status"] == "computed"
    assert str(record["diagnostics"]["grid_point_id"]) == "p000000"
    assert float(record["fit_shift_x_trials"][0]) == pytest.approx(1.25)
    assert float(record["fit_shift_y_trials"][0]) == pytest.approx(-0.75)
    assert bool(record["fit_find_shift_valid_trials"][0]) is True

    live_state = load_grid_point_live_state(
        artifact_h5,
        slice_key="mw_5p700000ghz",
        search_id=search_id,
        point_id="p000000",
    )
    assert live_state is not None
    assert float(live_state["a"]) == pytest.approx(-0.3)
    assert live_state["q0"] is None

    wrong_search_id = search_id_from_evaluation_config(diagnostics)
    assert wrong_search_id != search_id
    fallback_state = load_grid_point_live_state(
        artifact_h5,
        slice_key="mw_5p700000ghz",
        search_id=wrong_search_id,
        point_id="p000000",
    )
    assert fallback_state is not None
    assert fallback_state["search_id"] == search_id


def test_grid_trial_commits_store_all_metric_values(tmp_path: Path) -> None:
    artifact_h5 = tmp_path / "grid_metrics_test.h5"
    observed = np.ones((2, 2), dtype=float)
    sigma = np.ones((2, 2), dtype=float)
    header = _make_header()
    diagnostics = {**_make_diagnostics(), "target_metric": "eta2"}
    raw_map = np.full((2, 2), 3.0, dtype=np.float32)

    point_id = apply_grid_point_event_with_retry(
        artifact_h5,
        GridPointAssignedEvent(
            a=0.1,
            b=1.9,
            q0_start=1e-4,
            next_q0=1e-4,
            metric_name="eta2",
        ),
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
            q0=1e-4,
            metric=0.8,
            next_q0=2e-4,
            best_trial_index=0,
            best_metric=0.8,
            raw_modeled_map=raw_map,
            chi2=1.2,
            rho2=0.55,
            eta2=0.8,
        ),
        observed=observed,
        sigma_map=sigma,
        wcs_header=header,
        diagnostics=diagnostics,
    )
    apply_grid_point_event_with_retry(
        artifact_h5,
        GridPointCompletedEvent(
            point_id=str(point_id),
            best_trial_index=0,
            best_metric=0.8,
            best_q0=1e-4,
        ),
        observed=observed,
        sigma_map=sigma,
        wcs_header=header,
        diagnostics=diagnostics,
    )

    with h5py.File(artifact_h5, "r") as f:
        slice_group = f["slices"]["mw_5p700000ghz"]
        search_id = slice_group["active_search_id"][()].decode("utf-8")
        trial_group = slice_group["searches"][search_id][GRID_POINTS_GROUP]["p000000"]["trials"]["t000000"]
        assert float(trial_group.attrs["chi2"]) == pytest.approx(1.2)
        assert float(trial_group.attrs["rho2"]) == pytest.approx(0.55)
        assert float(trial_group.attrs["eta2"]) == pytest.approx(0.8)

    payload = load_scan_file(artifact_h5, slice_key="mw_5p700000ghz", search_id=search_id, include_maps=False)
    record = payload["point_records"][0]
    assert tuple(record["fit_chi2_trials"]) == pytest.approx((1.2,))
    assert tuple(record["fit_rho2_trials"]) == pytest.approx((0.55,))
    assert tuple(record["fit_eta2_trials"]) == pytest.approx((0.8,))
    assert float(record["metrics"]["chi2"]) == pytest.approx(1.2)
    assert float(record["metrics"]["rho2"]) == pytest.approx(0.55)
    assert float(record["metrics"]["eta2"]) == pytest.approx(0.8)


def test_refresh_signal_v2_writer(tmp_path: Path) -> None:
    signal_path = tmp_path / "artifact.h5.refresh"
    writer = RefreshSignalWriter(signal_path)
    writer.set_routing(slice_key="mw_5p700000ghz", search_id="search0001")
    writer.write_event(
        "trial_committed",
        point_id="p000001",
        trial_index=2,
        legacy_phase="trial 02 complete",
    )
    payload = json.loads(signal_path.read_text(encoding="utf-8"))
    assert payload["version"] == REFRESH_V2_VERSION
    assert payload["event"] == "trial_committed"
    assert payload["point_id"] == "p000001"
    assert payload["trial_index"] == 2
    assert payload["slice_key"] == "mw_5p700000ghz"
    assert payload["search_id"] == "search0001"


def test_grid_writes_use_explicit_selected_search_id_over_contract_match(tmp_path: Path) -> None:
    """Parallel searches with the same contract must not redirect grid events to another search."""
    observed = np.array([[0.0, 2.0], [0.0, 0.0]], dtype=float)
    sigma = np.ones((2, 2), dtype=float)
    header = _make_header()
    slice_key = "mw_5p700000ghz"
    base_diag = _make_diagnostics(slice_key=slice_key)
    layout = {"kind": "point_list"}
    legacy_id = search_id_from_evaluation_config(base_diag, layout=layout)
    parallel_diag = {
        **base_diag,
        "search_instance_id": "parallel_test_instance",
        "search_id": search_id_from_evaluation_config(
            {**base_diag, "search_instance_id": "parallel_test_instance"},
            layout=layout,
        ),
        "selected_search_id": search_id_from_evaluation_config(
            {**base_diag, "search_instance_id": "parallel_test_instance"},
            layout=layout,
        ),
    }
    parallel_id = str(parallel_diag["selected_search_id"])
    assert parallel_id != legacy_id

    artifact_h5 = tmp_path / "parallel_search_routing.h5"
    for search_id, diag in ((legacy_id, base_diag), (parallel_id, parallel_diag)):
        apply_grid_point_event_with_retry(
            artifact_h5,
            GridPointAssignedEvent(a=0.1, b=1.0, q0_start=1e-4, next_q0=1e-4, metric_name="chi2"),
            observed=observed,
            sigma_map=sigma,
            wcs_header=header,
            diagnostics={**diag, "search_id": search_id, "selected_search_id": search_id},
        )

    apply_grid_point_event_with_retry(
        artifact_h5,
        GridTrialCommittedEvent(
            point_id="p000000",
            trial_index=0,
            q0=1e-4,
            metric=0.5,
            next_q0=1e-4,
            best_trial_index=0,
            best_metric=0.5,
            raw_modeled_map=observed.copy(),
            chi2=1.0,
            rho2=0.5,
            eta2=0.5,
        ),
        observed=observed,
        sigma_map=sigma,
        wcs_header=header,
        diagnostics=parallel_diag,
    )

    with h5py.File(artifact_h5, "r") as f:
        parallel_group = f["slices"][slice_key]["searches"][parallel_id]
        legacy_group = f["slices"][slice_key]["searches"][legacy_id]
        assert GRID_POINTS_GROUP in parallel_group
        assert "p000000" in parallel_group[GRID_POINTS_GROUP]
        assert "trials" in parallel_group[GRID_POINTS_GROUP]["p000000"]
        assert "t000000" in parallel_group[GRID_POINTS_GROUP]["p000000"]["trials"]
        if GRID_POINTS_GROUP in legacy_group and "p000000" in legacy_group[GRID_POINTS_GROUP]:
            legacy_point = legacy_group[GRID_POINTS_GROUP]["p000000"]
            assert "trials" not in legacy_point or len(legacy_point["trials"]) == 0


def test_load_grid_point_trial_plot_payload_falls_back_when_trial_has_no_map(tmp_path: Path) -> None:
    artifact_h5 = tmp_path / "grid_trial_map_fallback.h5"
    observed = np.ones((2, 2), dtype=float)
    sigma = np.ones((2, 2), dtype=float)
    header = _make_header()
    diagnostics = _make_diagnostics()
    raw_map = np.full((2, 2), 3.0, dtype=np.float32)

    point_id = apply_grid_point_event_with_retry(
        artifact_h5,
        GridPointAssignedEvent(a=-0.3, b=2.1, q0_start=1e-4, next_q0=1e-4, metric_name="chi2"),
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
        ),
        observed=observed,
        sigma_map=sigma,
        wcs_header=header,
        diagnostics=diagnostics,
    )
    apply_grid_point_event_with_retry(
        artifact_h5,
        GridPointCompletedEvent(
            point_id=str(point_id),
            best_trial_index=0,
            best_metric=0.42,
            best_q0=1.0e-4,
        ),
        observed=observed,
        sigma_map=sigma,
        wcs_header=header,
        diagnostics=diagnostics,
    )

    payload = load_grid_point_trial_plot_payload(
        artifact_h5,
        a=-0.3,
        b=2.1,
        trial_index=1,
        slice_key="mw_5p700000ghz",
        search_id="search0001",
    )

    assert payload is not None
    assert int(payload["trial_index"]) == 0
    assert int(payload["storage_trial_index"]) == 0
    np.testing.assert_allclose(np.asarray(payload["raw_modeled_best"], dtype=float), raw_map)


def test_reset_grid_point_when_restored_trials_lack_map_store_links(tmp_path: Path) -> None:
    artifact_h5 = tmp_path / "contract_reset.h5"
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
        ),
        observed=observed,
        sigma_map=sigma,
        wcs_header=header,
        diagnostics=diagnostics,
    )
    from pychmp.ab_scan_artifacts import MAP_REFS_DATASET, _create_text_dataset, _json_dumps

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

    assert not grid_point_finite_q0_trials_have_map_store_links(
        artifact_h5,
        slice_key="mw_5p700000ghz",
        search_id="search0001",
        point_id=str(point_id),
    )

    reset_grid_point_for_rerun(
        artifact_h5,
        slice_key="mw_5p700000ghz",
        search_id="search0001",
        point_id=str(point_id),
        q0_start=1.0e-4,
        next_q0=1.0e-4,
        metric_name="eta2",
    )

    assert grid_point_finite_q0_trials_have_map_store_links(
        artifact_h5,
        slice_key="mw_5p700000ghz",
        search_id="search0001",
        point_id=str(point_id),
    )
    with h5py.File(artifact_h5, "r") as f:
        trials_group = f["slices"]["mw_5p700000ghz"]["searches"]["search0001"]["grid_points"][str(point_id)]["trials"]
        assert len(trials_group.keys()) == 0


def test_find_grid_point_ignores_storage_corrupt_point(tmp_path: Path) -> None:
    artifact_h5 = tmp_path / "corrupt_skip.h5"
    observed = np.ones((2, 2), dtype=float)
    sigma = np.ones((2, 2), dtype=float)
    header = _make_header()
    diagnostics = _make_diagnostics()
    point_id = apply_grid_point_event_with_retry(
        artifact_h5,
        GridPointAssignedEvent(a=0.25, b=4.5, q0_start=1e-4, next_q0=1e-4, metric_name="eta2"),
        observed=observed,
        sigma_map=sigma,
        wcs_header=header,
        diagnostics=diagnostics,
    )
    with h5py.File(artifact_h5, "a") as f:
        group = f["slices"]["mw_5p700000ghz"]["searches"]["search0001"]["grid_points"][str(point_id)]
        group.attrs[GRID_POINT_STORAGE_CORRUPT_ATTR] = 1
        search_group = f["slices"]["mw_5p700000ghz"]["searches"]["search0001"]
        assert find_grid_point_group(search_group, a=0.25, b=4.5) is None
    new_id = apply_grid_point_event_with_retry(
        artifact_h5,
        GridPointAssignedEvent(a=0.25, b=4.5, q0_start=2e-4, next_q0=2e-4, metric_name="eta2"),
        observed=observed,
        sigma_map=sigma,
        wcs_header=header,
        diagnostics=diagnostics,
    )
    assert str(new_id) != str(point_id)


def test_classify_grid_point_state_completed_with_stale_next_q0() -> None:
    assert (
        classify_grid_point_state({"status": "COMPLETED", "n_trials": 2, "next_q0": 1e-4})
        == "complete"
    )


def test_classify_grid_point_state() -> None:
    assert classify_grid_point_state({"status": "COMPLETED", "n_trials": 2}) == "complete"
    assert classify_grid_point_state({"status": "COMPLETED", "n_trials": 1, "next_q0": 1.0}) == "complete"
    assert classify_grid_point_state({"status": "ASSIGNED", "n_trials": 0, "next_q0": 1e-4}) == "assigned_no_trials"
    assert classify_grid_point_state({"status": "RUNNING", "n_trials": 2, "next_q0": 1e-4}) == "running_partial"
    assert classify_grid_point_state({"status": "FAILED", "next_q0": 1e-4}) == "failed"
