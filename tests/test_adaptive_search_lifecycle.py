from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits

from pychmp.ab_scan_artifacts import (
    COMPATIBILITY_SIGNATURE_KEY,
    append_scan_point_record,
    extend_patch_grid_model_with_pending_point,
    finalize_search_runner_state,
    reopen_search_runner_state,
    write_point_scan_artifact,
)


def _make_header() -> fits.Header:
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 4
    header["NAXIS2"] = 4
    header["CDELT1"] = 2.0
    header["CDELT2"] = 2.0
    header["CRPIX1"] = 2
    header["CRPIX2"] = 2
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    return header


def _make_point_payload(a_value: float, b_value: float) -> dict[str, object]:
    arr = np.ones((4, 4), dtype=float)
    return {
        "a": float(a_value),
        "b": float(b_value),
        "q0": 0.001,
        "success": True,
        "status": "computed",
        "modeled_best": arr,
        "raw_modeled_best": arr,
        "residual": np.zeros_like(arr),
        "fit_q0_trials": (0.0001, 0.001),
        "fit_metric_trials": (1.0, 0.5),
        "fit_chi2_trials": (1.0, 0.5),
        "fit_rho2_trials": (1.0, 0.5),
        "fit_eta2_trials": (1.0, 0.5),
        "fit_trial_mask_stages": ("data", "union"),
        "nfev": 2,
        "nit": 1,
        "message": "ok",
        "used_adaptive_bracketing": True,
        "bracket_found": True,
        "bracket": None,
        "target_metric": "eta2",
        "diagnostics": {"target_metric_value": 0.5, "eta2": 0.5},
    }


def test_active_adaptive_search_stays_in_progress_after_point_save(tmp_path: Path) -> None:
    out_h5 = tmp_path / "adaptive_lifecycle.h5"
    observed = np.ones((4, 4), dtype=float)
    sigma_map = np.ones((4, 4), dtype=float)
    header = _make_header()
    diagnostics = {
        "artifact_kind": "pychmp_ab_scan_sparse_points",
        COMPATIBILITY_SIGNATURE_KEY: "sig-adaptive-test",
        "target_slice_key": "default",
        "target_metric": "eta2",
        "search_mode": "adaptive_local_single_observation",
        "search_active": True,
        "metrics_mask_threshold": 0.1,
        "mask_type": "union",
        "q0_search_stages": ["data", "union"],
    }
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_records=[],
        run_history=[],
        preserve_existing_searches=True,
    )
    append_scan_point_record(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_payload=_make_point_payload(0.3, 2.7),
    )
    with h5py.File(out_h5, "r") as handle:
        search_id = handle["slices/default/active_search_id"][()].decode()
        search = handle["slices/default/searches"][search_id]
        assert search.attrs["status"].decode() == "in_progress"
        lifecycle = search["lifecycle_json"][()].decode()
        assert '"in_progress": true' in lifecycle
        assert '"active": true' in lifecycle


def test_finalize_search_runner_state_marks_search_complete(tmp_path: Path) -> None:
    out_h5 = tmp_path / "adaptive_finalize.h5"
    observed = np.ones((4, 4), dtype=float)
    sigma_map = np.ones((4, 4), dtype=float)
    header = _make_header()
    diagnostics = {
        "artifact_kind": "pychmp_ab_scan_sparse_points",
        COMPATIBILITY_SIGNATURE_KEY: "sig-adaptive-finalize",
        "target_slice_key": "default",
        "target_metric": "eta2",
        "search_mode": "adaptive_local_single_observation",
        "search_active": True,
    }
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_records=[],
        run_history=[],
        preserve_existing_searches=True,
    )
    append_scan_point_record(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_payload=_make_point_payload(0.3, 2.7),
    )
    with h5py.File(out_h5, "r") as handle:
        search_id = handle["slices/default/active_search_id"][()].decode()
    finalize_search_runner_state(out_h5, slice_key="default", search_id=search_id, completed=True)
    with h5py.File(out_h5, "r") as handle:
        search = handle["slices/default/searches"][search_id]
        assert search.attrs["status"].decode() == "complete"
        lifecycle = search["lifecycle_json"][()].decode()
        assert '"active": false' in lifecycle


def test_reopen_search_runner_state_clears_completed_at(tmp_path: Path) -> None:
    out_h5 = tmp_path / "adaptive_reopen.h5"
    observed = np.ones((4, 4), dtype=float)
    sigma_map = np.ones((4, 4), dtype=float)
    header = _make_header()
    diagnostics = {
        "artifact_kind": "pychmp_ab_scan_sparse_points",
        COMPATIBILITY_SIGNATURE_KEY: "sig-adaptive-reopen",
        "target_slice_key": "default",
        "target_metric": "eta2",
        "search_mode": "adaptive_local_single_observation",
        "search_active": True,
    }
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_records=[],
        run_history=[],
        preserve_existing_searches=True,
    )
    with h5py.File(out_h5, "r") as handle:
        search_id = handle["slices/default/active_search_id"][()].decode()
    finalize_search_runner_state(out_h5, slice_key="default", search_id=search_id, completed=True)
    reopen_search_runner_state(out_h5, slice_key="default", search_id=search_id)
    with h5py.File(out_h5, "r") as handle:
        search = handle["slices/default/searches"][search_id]
        lifecycle = search["lifecycle_json"][()].decode()
        assert '"active": true' in lifecycle
        assert '"completed_at": null' in lifecycle or "completed_at" not in lifecycle
        assert search.attrs["status"].decode() in {"in_progress", "partial", "empty"}


def test_extend_patch_grid_model_with_pending_point_adds_ghost_cell() -> None:
    model = {
        "records": [
            {
                "a": 0.3,
                "b": 2.7,
                "a0": 0.15,
                "a1": 0.45,
                "b0": 2.55,
                "b1": 2.85,
                "a_index": 0,
                "b_index": 0,
                "metrics": {"eta2": 0.5},
                "status": "computed",
            }
        ],
        "a_min": 0.15,
        "a_max": 0.45,
        "b_min": 2.55,
        "b_max": 2.85,
    }
    extended = extend_patch_grid_model_with_pending_point(
        model,
        a_value=0.0,
        b_value=2.7,
        diagnostics={"da": 0.3, "db": 0.3},
    )
    assert len(extended["records"]) == 2
    assert extended["records"][-1]["status"] == "pending"
