from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from examples.python.adaptive_ab_search_single_observation import (
    _PointRenderStream,
    _TrackedRendererProxy,
)
from pychmp.ab_scan_artifacts import UNIFIED_ARTIFACT_KIND, write_point_scan_artifact
from pychmp.ab_search import ABPointResult
from pychmp.grid_points import (
    GRID_POINTS_CONTRACT_VERSION,
    GridPointAssignedEvent,
    GridTrialCommittedEvent,
    apply_grid_point_event_with_retry,
)
from pychmp.metrics import MetricValues


def _grid_diagnostics(*, slice_key: str = "euv_193", search_id: str = "search_live") -> dict[str, object]:
    header = fits.Header()
    header["SIMPLE"] = True
    header["BITPIX"] = -32
    header["NAXIS"] = 2
    header["NAXIS1"] = 2
    header["NAXIS2"] = 2
    return {
        "artifact_kind": UNIFIED_ARTIFACT_KIND,
        "slice_key": slice_key,
        "target_slice_key": slice_key,
        "selected_search_id": search_id,
        "search_id": search_id,
        "target_metric": "eta2",
        "contract_version": GRID_POINTS_CONTRACT_VERSION,
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
    }


def test_build_artifact_payload_hydrates_best_maps_from_grid_point(tmp_path: Path) -> None:
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
            a=0.3,
            b=2.7,
            q0_start=1.0e-5,
            next_q0=1.0e-5,
            metric_name="eta2",
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
            trial_index=0,
            q0=1.0e-5,
            metric=0.8,
            next_q0=4.92816e-4,
            best_trial_index=0,
            best_metric=0.8,
            raw_modeled_map=np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
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
            trial_index=1,
            q0=4.92816e-4,
            metric=0.502,
            next_q0=4.92816e-4,
            best_trial_index=1,
            best_metric=0.502,
            raw_modeled_map=np.array([[10.0, 11.0], [12.0, 13.0]], dtype=np.float32),
        ),
        observed=observed,
        sigma_map=sigma,
        wcs_header=wcs_header,
        diagnostics=diagnostics,
    )

    point = ABPointResult(
        a=0.3,
        b=2.7,
        q0=4.92816e-4,
        objective_value=0.502,
        metrics=MetricValues(chi2=float("nan"), rho2=float("nan"), eta2=0.502),
        target_metric="eta2",
        success=True,
        nfev=2,
        nit=1,
        message="ok",
        used_adaptive_bracketing=True,
        bracket_found=True,
        bracket=(1.0e-5, 4.92816e-4, 1.0e-3),
        trial_q0=(1.0e-5, 4.92816e-4),
        trial_objective_values=(0.8, 0.502),
        trial_chi2_values=(float("nan"), float("nan")),
        trial_rho2_values=(float("nan"), float("nan")),
        trial_eta2_values=(0.8, 0.502),
        elapsed_seconds=1.0,
    )

    stream = _PointRenderStream()
    proxy = _TrackedRendererProxy(
        SimpleNamespace(render=lambda _q0: np.zeros((2, 2), dtype=float)),
        stream=stream,
        a_value=0.3,
        b_value=2.7,
        renderer_factory=lambda a, b: SimpleNamespace(),
        observed_template=observed,
        target_metric="eta2",
        psf_source="none",
        compatibility_signature="sig",
    )
    proxy._artifact_h5 = artifact_h5
    proxy._slice_key = "euv_193"
    proxy._search_id = "search_live"

    payload = proxy.build_artifact_payload(point)

    np.testing.assert_allclose(
        np.asarray(payload["raw_modeled_best"], dtype=float),
        np.array([[10.0, 11.0], [12.0, 13.0]], dtype=float),
    )
    assert float(payload["q0"]) == pytest.approx(4.92816e-4)
