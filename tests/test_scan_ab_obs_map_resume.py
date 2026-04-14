from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from examples.scan_ab_obs_map import (
    _build_rectangular_pending_requests,
    _build_sparse_pending_tasks,
    _merge_existing_rectangular_payload,
    _pending_point_payload,
)
from pychmp.ab_scan_artifacts import append_sparse_point_record, load_scan_file, save_rectangular_scan_file, write_sparse_scan_file
from pychmp.ab_scan_tasks import ABSliceTaskDescriptor, compile_rectangular_point_tasks, compile_sparse_point_tasks


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


def _make_root_diag() -> dict[str, object]:
    return {
        "artifact_kind": "pychmp_ab_scan",
        "spectral_domain": "mw",
        "spectral_label": "5.700 GHz",
        "model_path": "model.h5",
        "fits_file": "obs.fits",
        "ebtel_path": "ebtel.bin",
        "target_metric": "chi2",
        "frequency_ghz": 5.7,
        "map_xc_arcsec": 0.0,
        "map_yc_arcsec": 0.0,
        "map_dx_arcsec": 2.0,
        "map_dy_arcsec": 2.0,
        "observer_name": "earth",
        "observer_lonc_deg": 0.0,
        "observer_b0sun_deg": 0.0,
        "observer_dsun_cm": 1.495978707e13,
        "observer_obs_time": "2020-11-26T20:00:00",
    }


def _make_point_payload(a_value: float, b_value: float, *, a_index: int, b_index: int, q0: float, objective: float) -> dict[str, object]:
    modeled = np.full((2, 2), q0, dtype=float)
    return {
        "a": float(a_value),
        "b": float(b_value),
        "a_index": int(a_index),
        "b_index": int(b_index),
        "q0": float(q0),
        "success": True,
        "status": "computed",
        "modeled_best": modeled,
        "raw_modeled_best": modeled.copy(),
        "residual": np.zeros_like(modeled),
        "fit_q0_trials": (float(q0) - 0.1, float(q0)),
        "fit_metric_trials": (float(objective) + 0.5, float(objective)),
        "fit_chi2_trials": (float(objective) + 0.5, float(objective)),
        "fit_rho2_trials": (float(objective) + 0.6, float(objective) + 0.1),
        "fit_eta2_trials": (float(objective) + 0.7, float(objective) + 0.2),
        "target_metric": "chi2",
        "diagnostics": {
            "a": float(a_value),
            "b": float(b_value),
            "target_metric": "chi2",
            "target_metric_value": float(objective),
            "chi2": float(objective),
            "rho2": float(objective) + 0.1,
            "eta2": float(objective) + 0.2,
            "fit_success": True,
            "point_status": "computed",
        },
    }


def _write_rectangular_artifact(
    out_h5: Path,
    *,
    a_values: np.ndarray,
    b_values: np.ndarray,
    q0_offset: float,
) -> tuple[np.ndarray, np.ndarray, fits.Header, dict[str, object]]:
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_root_diag()
    best_q0 = np.full((a_values.size, b_values.size), np.nan, dtype=float)
    objective_values = np.full((a_values.size, b_values.size), np.nan, dtype=float)
    chi2 = np.full((a_values.size, b_values.size), np.nan, dtype=float)
    rho2 = np.full((a_values.size, b_values.size), np.nan, dtype=float)
    eta2 = np.full((a_values.size, b_values.size), np.nan, dtype=float)
    success = np.zeros((a_values.size, b_values.size), dtype=bool)
    point_payloads: dict[tuple[int, int], dict[str, object]] = {}

    for i, a_value in enumerate(a_values):
        for j, b_value in enumerate(b_values):
            q0 = float(q0_offset + 10.0 * i + j)
            objective = float(i + j / 10.0)
            payload = _make_point_payload(float(a_value), float(b_value), a_index=i, b_index=j, q0=q0, objective=objective)
            point_payloads[(i, j)] = payload
            best_q0[i, j] = q0
            objective_values[i, j] = objective
            chi2[i, j] = objective
            rho2[i, j] = objective + 0.1
            eta2[i, j] = objective + 0.2
            success[i, j] = True

    save_rectangular_scan_file(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        a_values=a_values,
        b_values=b_values,
        best_q0=best_q0,
        objective_values=objective_values,
        chi2=chi2,
        rho2=rho2,
        eta2=eta2,
        success=success,
        point_payloads=point_payloads,
    )
    return observed, sigma_map, header, diagnostics


def _make_pending_grid(
    *,
    a_values: np.ndarray,
    b_values: np.ndarray,
    observed_template: np.ndarray,
) -> tuple[dict[tuple[int, int], dict[str, object]], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    point_payloads = {
        (int(i), int(j)): _pending_point_payload(
            a_value=float(a_value),
            b_value=float(b_value),
            a_index=int(i),
            b_index=int(j),
            observed_template=observed_template,
            target_metric="chi2",
            status="pending",
            message="point not yet computed",
        )
        for i, a_value in enumerate(a_values)
        for j, b_value in enumerate(b_values)
    }
    shape = (a_values.size, b_values.size)
    return (
        point_payloads,
        np.full(shape, np.nan, dtype=float),
        np.full(shape, np.nan, dtype=float),
        np.full(shape, np.nan, dtype=float),
        np.full(shape, np.nan, dtype=float),
        np.full(shape, np.nan, dtype=float),
        np.zeros(shape, dtype=bool),
    )


def test_rectangular_partial_overlap_resume_merges_existing_and_queues_only_new_points(tmp_path: Path) -> None:
    """Resume a partially overlapping rectangular scan without recomputing matches."""
    out_h5 = tmp_path / "scan.h5"
    existing_a = np.asarray([0.3, 0.6], dtype=float)
    existing_b = np.asarray([2.1, 2.4, 2.7], dtype=float)
    observed, sigma_map, header, diagnostics = _write_rectangular_artifact(
        out_h5,
        a_values=existing_a,
        b_values=existing_b,
        q0_offset=1.0,
    )
    existing_payload = load_scan_file(out_h5)

    current_a = np.asarray([0.3, 0.6, 0.9], dtype=float)
    current_b = existing_b.copy()
    point_payloads, best_q0, objective_values, chi2, rho2, eta2, success = _make_pending_grid(
        a_values=current_a,
        b_values=current_b,
        observed_template=observed,
    )

    point_payloads, existing_points, reused_points = _merge_existing_rectangular_payload(
        existing_payload=existing_payload,
        a_values=current_a,
        b_values=current_b,
        point_payloads=point_payloads,
        best_q0=best_q0,
        objective_values=objective_values,
        chi2=chi2,
        rho2=rho2,
        eta2=eta2,
        success=success,
        target_metric="chi2",
    )

    assert existing_points == 6
    assert reused_points == 6
    assert np.all(np.isfinite(best_q0[:2, :]))
    assert np.all(np.isnan(best_q0[2, :]))

    slice_descriptor = ABSliceTaskDescriptor(
        key="mw_5p700000ghz",
        domain="mw",
        label="5.700 GHz",
        display_label="MW: 5.700 GHz",
    )
    target_tasks = compile_rectangular_point_tasks(
        a_values=current_a,
        b_values=current_b,
        slice_descriptor=slice_descriptor,
        q0_min=0.1,
        q0_max=10.0,
        target_metric="chi2",
    )
    pending_requests, skipped_points = _build_rectangular_pending_requests(
        target_tasks=target_tasks,
        point_payloads=point_payloads,
        recompute_existing=False,
        q0_start_scalar=None,
        use_idl_q0_start_heuristic=False,
        hard_q0_min=None,
        hard_q0_max=None,
        target_metric="chi2",
        adaptive_bracketing=False,
        q0_step=1.61803398875,
        max_bracket_steps=12,
    )

    assert len(skipped_points) == 6
    assert [(float(request.task.a), float(request.task.b)) for request in pending_requests] == [
        (0.9, 2.1),
        (0.9, 2.4),
        (0.9, 2.7),
    ]

    for request in pending_requests:
        i = int(request.task.a_index)
        j = int(request.task.b_index)
        payload = _make_point_payload(float(request.task.a), float(request.task.b), a_index=i, b_index=j, q0=90.0 + j, objective=9.0 + j)
        point_payloads[(i, j)] = payload
        best_q0[i, j] = float(payload["q0"])
        objective_values[i, j] = float(payload["diagnostics"]["target_metric_value"])
        chi2[i, j] = float(payload["diagnostics"]["chi2"])
        rho2[i, j] = float(payload["diagnostics"]["rho2"])
        eta2[i, j] = float(payload["diagnostics"]["eta2"])
        success[i, j] = bool(payload["success"])

    save_rectangular_scan_file(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        a_values=current_a,
        b_values=current_b,
        best_q0=best_q0,
        objective_values=objective_values,
        chi2=chi2,
        rho2=rho2,
        eta2=eta2,
        success=success,
        point_payloads=point_payloads,
    )
    merged_payload = load_scan_file(out_h5)

    assert len(merged_payload["points"]) == 9
    assert np.all(np.isfinite(np.asarray(merged_payload["best_q0"], dtype=float)))
    assert float(merged_payload["best_q0"][0, 0]) == 1.0
    assert float(merged_payload["best_q0"][2, 0]) == 90.0


def test_rectangular_recompute_existing_requeues_overlapping_points_and_persists_recomputed_values(tmp_path: Path) -> None:
    """Requeue overlapping rectangular points when recompute-existing is enabled."""
    out_h5 = tmp_path / "scan.h5"
    a_values = np.asarray([0.3, 0.6], dtype=float)
    b_values = np.asarray([2.1, 2.4], dtype=float)
    observed, sigma_map, header, diagnostics = _write_rectangular_artifact(
        out_h5,
        a_values=a_values,
        b_values=b_values,
        q0_offset=5.0,
    )
    existing_payload = load_scan_file(out_h5)

    point_payloads, best_q0, objective_values, chi2, rho2, eta2, success = _make_pending_grid(
        a_values=a_values,
        b_values=b_values,
        observed_template=observed,
    )
    point_payloads, existing_points, reused_points = _merge_existing_rectangular_payload(
        existing_payload=existing_payload,
        a_values=a_values,
        b_values=b_values,
        point_payloads=point_payloads,
        best_q0=best_q0,
        objective_values=objective_values,
        chi2=chi2,
        rho2=rho2,
        eta2=eta2,
        success=success,
        target_metric="chi2",
    )

    assert existing_points == 4
    assert reused_points == 4

    slice_descriptor = ABSliceTaskDescriptor(
        key="mw_5p700000ghz",
        domain="mw",
        label="5.700 GHz",
        display_label="MW: 5.700 GHz",
    )
    target_tasks = compile_rectangular_point_tasks(
        a_values=a_values,
        b_values=b_values,
        slice_descriptor=slice_descriptor,
        q0_min=0.1,
        q0_max=10.0,
        target_metric="chi2",
    )
    pending_requests, skipped_points = _build_rectangular_pending_requests(
        target_tasks=target_tasks,
        point_payloads=point_payloads,
        recompute_existing=True,
        q0_start_scalar=7.5,
        use_idl_q0_start_heuristic=False,
        hard_q0_min=None,
        hard_q0_max=None,
        target_metric="chi2",
        adaptive_bracketing=False,
        q0_step=1.61803398875,
        max_bracket_steps=12,
    )

    assert skipped_points == []
    assert len(pending_requests) == 4
    assert all(request.q0_start == 7.5 for request in pending_requests)

    for index, request in enumerate(pending_requests, start=1):
        i = int(request.task.a_index)
        j = int(request.task.b_index)
        payload = _make_point_payload(float(request.task.a), float(request.task.b), a_index=i, b_index=j, q0=70.0 + index, objective=7.0 + index)
        point_payloads[(i, j)] = payload
        best_q0[i, j] = float(payload["q0"])
        objective_values[i, j] = float(payload["diagnostics"]["target_metric_value"])
        chi2[i, j] = float(payload["diagnostics"]["chi2"])
        rho2[i, j] = float(payload["diagnostics"]["rho2"])
        eta2[i, j] = float(payload["diagnostics"]["eta2"])
        success[i, j] = bool(payload["success"])

    save_rectangular_scan_file(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        a_values=a_values,
        b_values=b_values,
        best_q0=best_q0,
        objective_values=objective_values,
        chi2=chi2,
        rho2=rho2,
        eta2=eta2,
        success=success,
        point_payloads=point_payloads,
    )
    recomputed_payload = load_scan_file(out_h5)

    assert np.allclose(np.asarray(recomputed_payload["best_q0"], dtype=float), np.asarray([[71.0, 72.0], [73.0, 74.0]], dtype=float))
    assert np.allclose(np.asarray(recomputed_payload["objective_values"], dtype=float), np.asarray([[8.0, 9.0], [10.0, 11.0]], dtype=float))


def test_rectangular_fresh_scan_writes_all_points(tmp_path: Path) -> None:
    """Write a complete rectangular artifact for a brand-new scan."""
    out_h5 = tmp_path / "fresh_scan.h5"
    a_values = np.asarray([0.0, 0.3], dtype=float)
    b_values = np.asarray([2.1, 2.4], dtype=float)
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_root_diag()

    point_payloads, best_q0, objective_values, chi2, rho2, eta2, success = _make_pending_grid(
        a_values=a_values,
        b_values=b_values,
        observed_template=observed,
    )
    slice_descriptor = ABSliceTaskDescriptor(
        key="mw_5p700000ghz",
        domain="mw",
        label="5.700 GHz",
        display_label="MW: 5.700 GHz",
    )
    target_tasks = compile_rectangular_point_tasks(
        a_values=a_values,
        b_values=b_values,
        slice_descriptor=slice_descriptor,
        q0_min=0.1,
        q0_max=10.0,
        target_metric="chi2",
    )
    pending_requests, skipped_points = _build_rectangular_pending_requests(
        target_tasks=target_tasks,
        point_payloads=point_payloads,
        recompute_existing=False,
        q0_start_scalar=None,
        use_idl_q0_start_heuristic=False,
        hard_q0_min=None,
        hard_q0_max=None,
        target_metric="chi2",
        adaptive_bracketing=False,
        q0_step=1.61803398875,
        max_bracket_steps=12,
    )

    assert skipped_points == []
    assert len(pending_requests) == 4

    for index, request in enumerate(pending_requests, start=1):
        i = int(request.task.a_index)
        j = int(request.task.b_index)
        payload = _make_point_payload(float(request.task.a), float(request.task.b), a_index=i, b_index=j, q0=20.0 + index, objective=2.0 + index)
        point_payloads[(i, j)] = payload
        best_q0[i, j] = float(payload["q0"])
        objective_values[i, j] = float(payload["diagnostics"]["target_metric_value"])
        chi2[i, j] = float(payload["diagnostics"]["chi2"])
        rho2[i, j] = float(payload["diagnostics"]["rho2"])
        eta2[i, j] = float(payload["diagnostics"]["eta2"])
        success[i, j] = bool(payload["success"])

    save_rectangular_scan_file(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        a_values=a_values,
        b_values=b_values,
        best_q0=best_q0,
        objective_values=objective_values,
        chi2=chi2,
        rho2=rho2,
        eta2=eta2,
        success=success,
        point_payloads=point_payloads,
    )
    payload = load_scan_file(out_h5)

    assert len(payload["points"]) == 4
    assert np.all(np.isfinite(np.asarray(payload["best_q0"], dtype=float)))
    assert np.all(np.isfinite(np.asarray(payload["objective_values"], dtype=float)))


def test_rectangular_full_resume_with_no_new_points_yields_no_pending_work(tmp_path: Path) -> None:
    """Yield no pending rectangular work when the artifact already covers the grid."""
    out_h5 = tmp_path / "fully_resumed_scan.h5"
    a_values = np.asarray([0.3, 0.6], dtype=float)
    b_values = np.asarray([2.1, 2.4], dtype=float)
    observed, _sigma_map, _header, _diagnostics = _write_rectangular_artifact(
        out_h5,
        a_values=a_values,
        b_values=b_values,
        q0_offset=12.0,
    )
    existing_payload = load_scan_file(out_h5)

    point_payloads, best_q0, objective_values, chi2, rho2, eta2, success = _make_pending_grid(
        a_values=a_values,
        b_values=b_values,
        observed_template=observed,
    )
    point_payloads, existing_points, reused_points = _merge_existing_rectangular_payload(
        existing_payload=existing_payload,
        a_values=a_values,
        b_values=b_values,
        point_payloads=point_payloads,
        best_q0=best_q0,
        objective_values=objective_values,
        chi2=chi2,
        rho2=rho2,
        eta2=eta2,
        success=success,
        target_metric="chi2",
    )

    assert existing_points == 4
    assert reused_points == 4
    assert np.all(np.isfinite(best_q0))
    assert np.all(success)

    slice_descriptor = ABSliceTaskDescriptor(
        key="mw_5p700000ghz",
        domain="mw",
        label="5.700 GHz",
        display_label="MW: 5.700 GHz",
    )
    target_tasks = compile_rectangular_point_tasks(
        a_values=a_values,
        b_values=b_values,
        slice_descriptor=slice_descriptor,
        q0_min=0.1,
        q0_max=10.0,
        target_metric="chi2",
    )
    pending_requests, skipped_points = _build_rectangular_pending_requests(
        target_tasks=target_tasks,
        point_payloads=point_payloads,
        recompute_existing=False,
        q0_start_scalar=None,
        use_idl_q0_start_heuristic=False,
        hard_q0_min=None,
        hard_q0_max=None,
        target_metric="chi2",
        adaptive_bracketing=False,
        q0_step=1.61803398875,
        max_bracket_steps=12,
    )

    assert pending_requests == []
    assert sorted(skipped_points) == [(0.3, 2.1), (0.3, 2.4), (0.6, 2.1), (0.6, 2.4)]


def test_sparse_resume_smoke_skips_existing_and_appends_only_new_points(tmp_path: Path) -> None:
    """Skip existing sparse points and append only newly requested ones."""
    out_h5 = tmp_path / "sparse_scan.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_root_diag()
    diagnostics["artifact_kind"] = "pychmp_ab_scan_sparse_points"
    existing_payload = _make_point_payload(0.3, 2.1, a_index=0, b_index=0, q0=11.0, objective=1.1)
    write_sparse_scan_file(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_records=[existing_payload],
    )

    loaded_payload = load_scan_file(out_h5)
    existing_points = {
        (float(record["a"]), float(record["b"])): record
        for record in loaded_payload.get("point_records", [])
    }
    slice_descriptor = ABSliceTaskDescriptor(
        key="mw_5p700000ghz",
        domain="mw",
        label="5.700 GHz",
        display_label="MW: 5.700 GHz",
    )
    target_tasks = compile_sparse_point_tasks(
        point_specs=[(0.3, 2.1, None, None), (0.6, 2.1, None, None)],
        a_values=np.asarray([0.3, 0.6], dtype=float),
        b_values=np.asarray([2.1], dtype=float),
        slice_descriptor=slice_descriptor,
        default_q0_min=0.1,
        default_q0_max=10.0,
        target_metric="chi2",
    )

    pending_tasks, skipped_points, recompute_points = _build_sparse_pending_tasks(
        target_tasks=target_tasks,
        existing_points=existing_points,
        recompute_existing=False,
    )

    assert skipped_points == [(0.3, 2.1)]
    assert recompute_points == []
    assert [(float(task.a), float(task.b)) for task in pending_tasks] == [(0.6, 2.1)]

    append_sparse_point_record(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_payload=_make_point_payload(0.6, 2.1, a_index=1, b_index=0, q0=22.0, objective=2.2),
    )
    merged_payload = load_scan_file(out_h5)

    assert sorted((float(record["a"]), float(record["b"])) for record in merged_payload["point_records"]) == [(0.3, 2.1), (0.6, 2.1)]


def test_sparse_recompute_existing_smoke_keeps_latest_record_per_coordinate(tmp_path: Path) -> None:
    """Keep the latest sparse record when recomputing existing coordinates."""
    out_h5 = tmp_path / "sparse_recompute.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_root_diag()
    diagnostics["artifact_kind"] = "pychmp_ab_scan_sparse_points"
    original_payload = _make_point_payload(0.3, 2.1, a_index=0, b_index=0, q0=11.0, objective=1.1)
    write_sparse_scan_file(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_records=[original_payload],
    )

    loaded_payload = load_scan_file(out_h5)
    existing_points = {
        (float(record["a"]), float(record["b"])): record
        for record in loaded_payload.get("point_records", [])
    }
    slice_descriptor = ABSliceTaskDescriptor(
        key="mw_5p700000ghz",
        domain="mw",
        label="5.700 GHz",
        display_label="MW: 5.700 GHz",
    )
    target_tasks = compile_sparse_point_tasks(
        point_specs=[(0.3, 2.1, None, None), (0.6, 2.1, None, None)],
        a_values=np.asarray([0.3, 0.6], dtype=float),
        b_values=np.asarray([2.1], dtype=float),
        slice_descriptor=slice_descriptor,
        default_q0_min=0.1,
        default_q0_max=10.0,
        target_metric="chi2",
    )

    pending_tasks, skipped_points, recompute_points = _build_sparse_pending_tasks(
        target_tasks=target_tasks,
        existing_points=existing_points,
        recompute_existing=True,
    )

    assert skipped_points == []
    assert recompute_points == [(0.3, 2.1)]
    assert sorted((float(task.a), float(task.b)) for task in pending_tasks) == [(0.3, 2.1), (0.6, 2.1)]

    append_sparse_point_record(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_payload=_make_point_payload(0.3, 2.1, a_index=0, b_index=0, q0=33.0, objective=3.3),
    )
    append_sparse_point_record(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_payload=_make_point_payload(0.6, 2.1, a_index=1, b_index=0, q0=44.0, objective=4.4),
    )
    recomputed_payload = load_scan_file(out_h5)
    point_map = {
        (float(record["a"]), float(record["b"])): record
        for record in recomputed_payload["point_records"]
    }

    assert len(point_map) == 2
    assert float(point_map[(0.3, 2.1)]["q0"]) == 33.0
    assert float(point_map[(0.6, 2.1)]["q0"]) == 44.0