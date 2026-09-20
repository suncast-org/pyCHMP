from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from examples.python.adaptive_ab_search_single_observation import _PersistentPointCache
from pychmp.ab_scan_artifacts import UNIFIED_ARTIFACT_KIND, write_point_scan_artifact
from pychmp.ab_search import ABPointResult
from pychmp.grid_points import (
    GRID_POINTS_CONTRACT_VERSION,
    GridPointAssignedEvent,
    GridTrialCommittedEvent,
    apply_grid_point_event_with_retry,
    read_grid_point_header,
)
from pychmp.metrics import MetricValues


def _grid_diagnostics(*, slice_key: str = "mw_1p418ghz", search_id: str = "search_commit") -> dict[str, object]:
    return {
        "artifact_kind": UNIFIED_ARTIFACT_KIND,
        "slice_key": slice_key,
        "target_slice_key": slice_key,
        "selected_search_id": search_id,
        "search_id": search_id,
        "target_metric": "eta2",
        "contract_version": GRID_POINTS_CONTRACT_VERSION,
        "q0_search_stages": ["data", "union"],
        "mask_type": "union",
        "metrics_mask_threshold": 0.2,
        "model_path": "/tmp/model.h5",
        "model_id": "model-123",
        "model_sha256": "a" * 64,
        "fits_file": "/tmp/obs.fits",
        "fits_sha256": "b" * 64,
        "ebtel_path": "/tmp/ebtel.bin",
        "ebtel_sha256": "c" * 64,
        "frequency_ghz": 1.418,
        "map_xc_arcsec": 0.0,
        "map_yc_arcsec": 180.0,
        "map_dx_arcsec": 2.0,
        "map_dy_arcsec": 2.0,
        "map_nx": 2,
        "map_ny": 2,
        "observer_name": "earth",
        "observer_lonc_deg": 0.0,
        "observer_b0sun_deg": 0.0,
        "observer_dsun_cm": 1.495978707e13,
        "observer_obs_time": "2026-04-03T19:30:00",
    }


def test_commit_completed_point_resolves_live_map_by_q0_when_trial_index_differs(
    tmp_path: Path,
) -> None:
    """Merged evaluation_order indices can disagree with live grid trial indices."""
    artifact_h5 = tmp_path / "adaptive.h5"
    diagnostics = _grid_diagnostics()
    observed = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma = np.ones_like(observed)
    wcs_header = fits.Header()
    write_point_scan_artifact(
        artifact_h5,
        observed=observed,
        sigma_map=sigma,
        wcs_header=wcs_header,
        diagnostics=diagnostics,
        point_records=[],
    )
    point_id = apply_grid_point_event_with_retry(
        artifact_h5,
        GridPointAssignedEvent(
            a=-0.55,
            b=3.25,
            q0_start=1.0e-4,
            next_q0=1.0e-4,
            metric_name="eta2",
        ),
        observed=observed,
        sigma_map=sigma,
        wcs_header=wcs_header,
        diagnostics=diagnostics,
    )
    first_map = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    live_map = np.array([[10.0, 11.0], [12.0, 13.0]], dtype=np.float32)
    apply_grid_point_event_with_retry(
        artifact_h5,
        GridTrialCommittedEvent(
            point_id=str(point_id),
            trial_index=3,
            q0=0.006854,
            metric=0.4439,
            next_q0=0.006854,
            best_trial_index=3,
            best_metric=0.4439,
            raw_modeled_map=first_map,
            trial_metadata={"stage": "data"},
        ),
        observed=observed,
        sigma_map=sigma,
        wcs_header=wcs_header,
        diagnostics=diagnostics,
    )
    apply_grid_point_event_with_retry(
        artifact_h5,
        GridTrialCommittedEvent(
            point_id=str(point_id),
            trial_index=15,
            q0=0.0110902,
            metric=0.3655,
            next_q0=0.0110902,
            best_trial_index=15,
            best_metric=0.3655,
            raw_modeled_map=live_map,
            trial_metadata={"stage": "union"},
        ),
        observed=observed,
        sigma_map=sigma,
        wcs_header=wcs_header,
        diagnostics=diagnostics,
    )

    merged_q0 = (0.006854, 0.0110902)
    merged_metrics = (0.4439, 0.3655)
    merged_mask_stages = ("data", "union")
    target_index = 1

    result = ABPointResult(
        a=-0.55,
        b=3.25,
        q0=0.0110902,
        objective_value=float(merged_metrics[target_index]),
        metrics=MetricValues(
            chi2=float("nan"),
            rho2=float("nan"),
            eta2=float(merged_metrics[target_index]),
        ),
        target_metric="eta2",
        success=True,
        nfev=len(merged_q0),
        nit=1,
        message="ok",
        used_adaptive_bracketing=True,
        bracket_found=True,
        bracket=(1.0e-5, 0.0110902, 1.0e-3),
        trial_q0=merged_q0,
        trial_objective_values=merged_metrics,
        trial_chi2_values=tuple(float("nan") for _ in merged_q0),
        trial_rho2_values=tuple(float("nan") for _ in merged_q0),
        trial_eta2_values=merged_metrics,
        trial_mask_stages=merged_mask_stages,
        elapsed_seconds=816.0,
    )
    payload = {
        "q0": float(result.q0),
        "fit_q0_trials": list(result.trial_q0),
        "fit_metric_trials": list(result.trial_objective_values),
        "fit_chi2_trials": list(result.trial_chi2_values),
        "fit_rho2_trials": list(result.trial_rho2_values),
        "fit_eta2_trials": list(result.trial_eta2_values),
        "fit_trial_mask_stages": list(result.trial_mask_stages),
        "trial_raw_modeled_maps": None,
    }

    cache = _PersistentPointCache(
        artifact_h5=artifact_h5,
        observed=observed,
        sigma_map=sigma,
        target_header=wcs_header,
        diagnostics=diagnostics,
        blos_reference=None,
        renderer_factory=lambda a_value, b_value: None,
        target_metric="eta2",
        psf_source="none",
        psf_kernel=None,
        compatibility_signature="sig-commit",
        viewer_heartbeat=None,
    )
    cache._point_ids[(-0.55, 3.25)] = str(point_id)
    cache._trials_committed[(-0.55, 3.25)] = 16

    cache._commit_completed_point_from_payload(
        a_value=-0.55,
        b_value=3.25,
        payload=payload,
        result=result,
    )
    cache.close()

    import h5py

    from pychmp.grid_points import GRID_POINTS_GROUP, SEARCHES_GROUP, SLICE_CONTAINER_GROUP, _load_grid_point_trials

    with h5py.File(artifact_h5, "r") as f:
        point_group = f[SLICE_CONTAINER_GROUP]["mw_1p418ghz"][SEARCHES_GROUP]["search_commit"][GRID_POINTS_GROUP][point_id]
        header = read_grid_point_header(point_group)
        trials = _load_grid_point_trials(point_group, include_maps=False)
    assert header["status"] == "COMPLETED"
    trials_by_index = {int(item["trial_index"]): item for item in trials}
    assert float(trials_by_index[target_index]["q0"]) == pytest.approx(0.0110902)
    assert str(trials_by_index[target_index].get("raw_map_ref", "")).strip()
    assert float(trials_by_index[0]["q0"]) == pytest.approx(0.006854)
    assert str(trials_by_index[0].get("raw_map_ref", "")).strip()
