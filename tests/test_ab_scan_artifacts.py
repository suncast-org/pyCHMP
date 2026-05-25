from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy.io import fits

from pychmp.ab_scan_artifacts import (
    COMPATIBILITY_SIGNATURE_KEY,
    CANONICAL_ARTIFACT_CONTRACT_VERSION,
    UNIFIED_ARTIFACT_KIND,
    ScanArtifactCompatibilityError,
    append_point_record,
    append_scan_point_record,
    backfill_artifact_diagnostics,
    build_computed_point_payload,
    default_point_index,
    load_auxiliary_map_store_point_records,
    list_scan_slices,
    load_scan_file,
    point_record_matches_compatibility_signature,
    resolve_point_index,
    scan_artifact_reuse_preflight_issues,
    write_grid_scan_artifact,
    scan_artifact_compatibility_issues,
    validate_scan_artifact_compatibility,
    validate_scan_artifact_reuse_preflight,
    write_single_point_scan_file,
    write_point_scan_artifact,
)
from pychmp import ab_scan_artifacts


def _make_header(*, crval1: float = 0.0) -> fits.Header:
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
    header["CRVAL1"] = float(crval1)
    header["CRVAL2"] = 0.0
    header["CDELT1"] = 2.0
    header["CDELT2"] = 2.0
    header["DATE-OBS"] = "2020-11-26T20:00:00"
    header["OBSERVER"] = "earth"
    return header


def _make_diagnostics(*, artifact_kind: str = "pychmp_ab_scan", model_id: str = "model-123") -> dict[str, object]:
    return {
        "artifact_kind": artifact_kind,
        COMPATIBILITY_SIGNATURE_KEY: "sig-123",
        "target_metric": "chi2",
        "metrics_mask_threshold": 0.1,
        "metrics_mask_source": "union_threshold",
        "model_path": "C:/tmp/model.h5",
        "model_id": model_id,
        "model_sha256": "a" * 64,
        "fits_file": "C:/tmp/obs.fits",
        "fits_sha256": "b" * 64,
        "ebtel_path": "C:/tmp/ebtel.bin",
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


def _make_point_payload(a_value: float, b_value: float, *, a_index: int = 0, b_index: int = 0) -> dict[str, object]:
    modeled = np.ones((2, 2), dtype=float)
    diagnostics = {
        "chi2": 0.1,
        "rho2": 0.2,
        "eta2": 0.3,
        "target_metric_value": 0.1,
        "target_metric": "chi2",
    }
    return {
        "a": float(a_value),
        "b": float(b_value),
        "a_index": int(a_index),
        "b_index": int(b_index),
        "q0": 2.5,
        "success": True,
        "status": "computed",
        "modeled_best": modeled,
        "raw_modeled_best": modeled.copy(),
        "residual": np.zeros_like(modeled),
        "fit_q0_trials": (2.0, 2.5),
        "fit_metric_trials": (0.4, 0.1),
        "fit_chi2_trials": (0.4, 0.1),
        "fit_rho2_trials": (0.5, 0.2),
        "fit_eta2_trials": (0.6, 0.3),
        "nfev": 2,
        "nit": 1,
        "message": "ok",
        "used_adaptive_bracketing": False,
        "bracket_found": False,
        "bracket": None,
        "target_metric": "chi2",
        "diagnostics": diagnostics,
    }


def _make_blos_reference() -> tuple[np.ndarray, fits.Header]:
    header = _make_header(crval1=12.0)
    data = np.asarray([[10.0, -10.0], [5.0, -5.0]], dtype=float)
    return data, header


def _write_rectangular_artifact(out_h5: Path) -> tuple[np.ndarray, np.ndarray, fits.Header, dict[str, object]]:
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_diagnostics()
    a_values = np.asarray([0.0], dtype=float)
    b_values = np.asarray([1.0], dtype=float)
    write_grid_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        a_values=a_values,
        b_values=b_values,
        best_q0=np.asarray([[2.5]], dtype=float),
        objective_values=np.asarray([[0.1]], dtype=float),
        chi2=np.asarray([[0.1]], dtype=float),
        rho2=np.asarray([[0.2]], dtype=float),
        eta2=np.asarray([[0.3]], dtype=float),
        success=np.asarray([[True]], dtype=bool),
        point_payloads={(0, 0): _make_point_payload(0.0, 1.0)},
    )
    return observed, sigma_map, header, diagnostics


def test_validate_scan_artifact_compatibility_accepts_matching_rectangular_artifact(tmp_path: Path) -> None:
    """Accept reuse when rectangular artifact inputs match exactly."""
    out_h5 = tmp_path / "scan.h5"
    observed, sigma_map, header, diagnostics = _write_rectangular_artifact(out_h5)

    payload = load_scan_file(out_h5)

    validate_scan_artifact_compatibility(
        payload,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        artifact_path=out_h5,
    )


def test_rectangular_artifact_persists_selectable_search_records(tmp_path: Path) -> None:
    out_h5 = tmp_path / "scan.h5"
    observed, sigma_map, header, diagnostics = _write_rectangular_artifact(out_h5)
    first_payload = load_scan_file(out_h5)
    first_search_id = str(first_payload["selected_search_id"])

    second_diagnostics = dict(diagnostics)
    second_diagnostics[COMPATIBILITY_SIGNATURE_KEY] = "sig-456"
    second_diagnostics["metrics_mask_threshold"] = 0.2
    second_point = _make_point_payload(0.0, 1.0)
    second_point["diagnostics"] = {
        **dict(second_point["diagnostics"]),
        COMPATIBILITY_SIGNATURE_KEY: "sig-456",
        "metrics_mask_threshold": 0.2,
        "target_metric_value": 0.05,
        "chi2": 0.05,
    }
    write_grid_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=second_diagnostics,
        a_values=np.asarray([0.0], dtype=float),
        b_values=np.asarray([1.0], dtype=float),
        best_q0=np.asarray([[3.0]], dtype=float),
        objective_values=np.asarray([[0.05]], dtype=float),
        chi2=np.asarray([[0.05]], dtype=float),
        rho2=np.asarray([[0.2]], dtype=float),
        eta2=np.asarray([[0.3]], dtype=float),
        success=np.asarray([[True]], dtype=bool),
        point_payloads={(0, 0): second_point},
    )

    latest_payload = load_scan_file(out_h5)
    assert len(latest_payload["search_records"]) == 2
    assert latest_payload["diagnostics"]["metrics_mask_threshold"] == pytest.approx(0.2)

    first_search_payload = load_scan_file(out_h5, search_id=first_search_id)
    assert first_search_payload["diagnostics"]["metrics_mask_threshold"] == pytest.approx(0.1)
    assert first_search_payload["chi2"][0, 0] == pytest.approx(0.1)


def test_point_artifact_start_over_reuses_same_search_id_and_replaces_history(tmp_path: Path) -> None:
    out_h5 = tmp_path / "scan.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_diagnostics(artifact_kind="pychmp_ab_scan_sparse_points")

    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_records=[_make_point_payload(0.0, 1.0)],
    )
    first_payload = load_scan_file(out_h5)
    first_search_id = str(first_payload["selected_search_id"])

    second_diagnostics = dict(diagnostics)
    second_point = _make_point_payload(0.3, 1.3)
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=second_diagnostics,
        point_records=[second_point],
        preserve_existing_searches=False,
    )

    latest_payload = load_scan_file(out_h5)
    latest_search_id = str(latest_payload["selected_search_id"])
    assert latest_search_id == first_search_id
    assert len(latest_payload["search_records"]) == 1
    assert len(latest_payload["point_records"]) == 1
    assert latest_payload["point_records"][0]["a"] == pytest.approx(0.3)


def test_point_artifact_start_over_rewrite_preserves_existing_map_store_refs(tmp_path: Path) -> None:
    out_h5 = tmp_path / "scan.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_diagnostics(artifact_kind="pychmp_ab_scan_sparse_points")

    first_point = _make_point_payload(0.0, 1.0)
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_records=[first_point],
    )

    first_payload = load_scan_file(out_h5)
    first_record = first_payload["point_records"][0]
    first_search_id = str(first_payload["selected_search_id"])
    with h5py.File(out_h5, "r") as handle:
        refs = json.loads(
            handle[f"slices/default/searches/{first_search_id}/point_records/r000000/map_refs_json"][()].decode()
        )
    expected_ref_paths = tuple(str(path) for path in refs.values())

    second_diagnostics = dict(diagnostics)
    second_point = _make_point_payload(0.3, 1.3)
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=second_diagnostics,
        point_records=[second_point],
        preserve_existing_searches=False,
    )

    with h5py.File(out_h5, "r") as handle:
        for ref_path in expected_ref_paths:
            assert ref_path in handle


def test_point_artifact_persists_single_common_psf_kernel(tmp_path: Path) -> None:
    out_h5 = tmp_path / "scan.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_diagnostics(artifact_kind="pychmp_ab_scan_sparse_points")
    kernel = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [1.0, 4.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )

    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        psf_kernel=kernel,
        point_records=[_make_point_payload(0.0, 1.0)],
    )

    payload = load_scan_file(out_h5)
    loaded_kernel = np.asarray(payload.get("psf_kernel"), dtype=float)
    assert loaded_kernel.shape == (3, 3)
    np.testing.assert_allclose(loaded_kernel.sum(), 1.0, rtol=0.0, atol=1e-6)

    selected_slice_key = str(payload.get("selected_slice_key"))
    with h5py.File(out_h5, "r") as handle:
        assert f"slices/{selected_slice_key}/common/psf_kernel" in handle
        assert f"slices/{selected_slice_key}/common/psf_kernel_meta_json" in handle


def test_validate_scan_artifact_compatibility_rejects_header_mismatch(tmp_path: Path) -> None:
    """Reject reuse when the persisted WCS header differs."""
    out_h5 = tmp_path / "scan.h5"
    observed, sigma_map, _header, diagnostics = _write_rectangular_artifact(out_h5)
    payload = load_scan_file(out_h5)
    changed_header = _make_header(crval1=12.0)

    with pytest.raises(ScanArtifactCompatibilityError, match="WCS header differs"):
        validate_scan_artifact_compatibility(
            payload,
            observed=observed,
            sigma_map=sigma_map,
            wcs_header=changed_header,
            diagnostics=diagnostics,
            artifact_path=out_h5,
        )


def test_validate_scan_artifact_compatibility_rejects_required_diagnostic_mismatch(tmp_path: Path) -> None:
    """Reject reuse when required diagnostic identity fields change."""
    out_h5 = tmp_path / "scan.h5"
    observed, sigma_map, header, diagnostics = _write_rectangular_artifact(out_h5)
    payload = load_scan_file(out_h5)
    changed_diagnostics = dict(diagnostics)
    changed_diagnostics["model_sha256"] = "d" * 64

    issues = scan_artifact_compatibility_issues(
        payload,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=changed_diagnostics,
    )

    assert any("model_sha256" in issue for issue in issues)
    with pytest.raises(ScanArtifactCompatibilityError, match="model_sha256"):
        validate_scan_artifact_compatibility(
            payload,
            observed=observed,
            sigma_map=sigma_map,
            wcs_header=header,
            diagnostics=changed_diagnostics,
            artifact_path=out_h5,
        )


def test_validate_scan_artifact_compatibility_allows_new_search_signature(tmp_path: Path) -> None:
    out_h5 = tmp_path / "scan.h5"
    observed, sigma_map, header, diagnostics = _write_rectangular_artifact(out_h5)
    payload = load_scan_file(out_h5)
    changed_diagnostics = dict(diagnostics)
    changed_diagnostics[COMPATIBILITY_SIGNATURE_KEY] = "sig-other"

    validate_scan_artifact_compatibility(
        payload,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=changed_diagnostics,
        artifact_path=out_h5,
    )


def test_validate_scan_artifact_compatibility_rejects_sparse_observation_mismatch(tmp_path: Path) -> None:
    """Reject sparse reuse when the observed map no longer matches."""
    out_h5 = tmp_path / "sparse_scan.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_diagnostics(artifact_kind="pychmp_ab_scan_sparse_points")
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_records=[_make_point_payload(0.0, 1.0)],
    )

    payload = load_scan_file(out_h5)
    changed_observed = observed.copy()
    changed_observed[0, 0] = 99.0

    with pytest.raises(ScanArtifactCompatibilityError, match="observed map differs"):
        validate_scan_artifact_compatibility(
            payload,
            observed=changed_observed,
            sigma_map=sigma_map,
            wcs_header=header,
            diagnostics=diagnostics,
            artifact_path=out_h5,
        )


def test_validate_scan_artifact_reuse_preflight_rejects_sparse_geometry_mismatch(tmp_path: Path) -> None:
    out_h5 = tmp_path / "sparse_scan.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_diagnostics(artifact_kind="pychmp_ab_scan_sparse_points")
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_records=[_make_point_payload(0.0, 1.0)],
    )

    payload = load_scan_file(out_h5)
    changed_header = _make_header(crval1=12.0)
    changed_diagnostics = dict(diagnostics)
    changed_diagnostics["map_dx_arcsec"] = 3.0

    issues = scan_artifact_reuse_preflight_issues(
        payload,
        wcs_header=changed_header,
        diagnostics=changed_diagnostics,
    )

    assert any("WCS header differs" in issue for issue in issues)
    assert any("map_dx_arcsec" in issue for issue in issues)
    with pytest.raises(ScanArtifactCompatibilityError, match="map_dx_arcsec"):
        validate_scan_artifact_reuse_preflight(
            payload,
            wcs_header=changed_header,
            diagnostics=changed_diagnostics,
            artifact_path=out_h5,
        )


def test_validate_scan_artifact_reuse_preflight_allows_sparse_search_specific_changes(tmp_path: Path) -> None:
    out_h5 = tmp_path / "sparse_scan.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_diagnostics(artifact_kind="pychmp_ab_scan_sparse_points")
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_records=[_make_point_payload(0.0, 1.0)],
    )

    payload = load_scan_file(out_h5)
    changed_diagnostics = dict(diagnostics)
    changed_diagnostics["target_metric"] = "eta2"
    changed_diagnostics["metrics_mask_threshold"] = 0.5
    changed_diagnostics[COMPATIBILITY_SIGNATURE_KEY] = "sig-eta2-threshold-0p5"

    validate_scan_artifact_reuse_preflight(
        payload,
        wcs_header=header,
        diagnostics=changed_diagnostics,
        artifact_path=out_h5,
    )


def test_validate_scan_artifact_compatibility_allows_sparse_target_metric_change(tmp_path: Path) -> None:
    out_h5 = tmp_path / "sparse_scan.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_diagnostics(artifact_kind="pychmp_ab_scan_sparse_points")
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_records=[_make_point_payload(0.0, 1.0)],
    )

    payload = load_scan_file(out_h5)
    changed_diagnostics = dict(diagnostics)
    changed_diagnostics["target_metric"] = "eta2"
    changed_diagnostics["metrics_mask_threshold"] = 0.5
    changed_diagnostics[COMPATIBILITY_SIGNATURE_KEY] = "sig-eta2-threshold-0p5"

    validate_scan_artifact_compatibility(
        payload,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=changed_diagnostics,
        artifact_path=out_h5,
    )


def test_sparse_artifact_round_trip_preserves_point_elapsed_seconds(tmp_path: Path) -> None:
    """Persist per-point elapsed_seconds in sparse point diagnostics when present."""
    out_h5 = tmp_path / "sparse_elapsed.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_diagnostics(artifact_kind="pychmp_ab_scan_sparse_points")
    point = _make_point_payload(0.0, 1.0)
    point["diagnostics"] = dict(point["diagnostics"])
    point["diagnostics"]["elapsed_seconds"] = 12.345
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_records=[point],
    )

    payload = load_scan_file(out_h5)
    record = payload["point_records"][0]

    assert float(record["diagnostics"]["elapsed_seconds"]) == pytest.approx(12.345)


def test_sparse_artifact_rewrite_preserves_selectable_search_records(tmp_path: Path) -> None:
    out_h5 = tmp_path / "sparse_searches.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    first_diagnostics = _make_diagnostics(artifact_kind="pychmp_ab_scan_sparse_points")
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=first_diagnostics,
        point_records=[_make_point_payload(0.0, 1.0)],
    )
    first_payload = load_scan_file(out_h5)
    first_search_id = str(first_payload["selected_search_id"])

    second_diagnostics = dict(first_diagnostics)
    second_diagnostics[COMPATIBILITY_SIGNATURE_KEY] = "sig-789"
    second_diagnostics["metrics_mask_threshold"] = 0.2
    second_point = _make_point_payload(0.0, 1.0)
    second_point["diagnostics"] = {
        **dict(second_point["diagnostics"]),
        COMPATIBILITY_SIGNATURE_KEY: "sig-789",
        "metrics_mask_threshold": 0.2,
        "target_metric_value": 0.07,
        "chi2": 0.07,
    }
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=second_diagnostics,
        point_records=[second_point],
    )

    latest_payload = load_scan_file(out_h5)
    assert len(latest_payload["search_records"]) == 2
    assert latest_payload["diagnostics"]["metrics_mask_threshold"] == pytest.approx(0.2)

    first_search_payload = load_scan_file(out_h5, search_id=first_search_id)
    assert first_search_payload["diagnostics"]["metrics_mask_threshold"] == pytest.approx(0.1)
    assert first_search_payload["chi2"][0, 0] == pytest.approx(0.1)


def test_write_single_point_scan_file_round_trip_is_unified_and_viewer_compatible(tmp_path: Path) -> None:
    out_h5 = tmp_path / "single_point.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_diagnostics(artifact_kind="pychmp_q0_recovery")
    diagnostics.update(
        {
            "a": 0.3,
            "b": 2.7,
            "fit_q0_trials": [2.0, 2.5],
            "fit_metric_trials": [0.4, 0.1],
            "fit_chi2_trials": [0.4, 0.1],
            "fit_rho2_trials": [0.5, 0.2],
            "fit_eta2_trials": [0.6, 0.3],
        }
    )
    point_payload = build_computed_point_payload(
        a_value=0.3,
        b_value=2.7,
        a_index=0,
        b_index=0,
        q0=2.5,
        success=True,
        status="computed",
        modeled_best=np.ones((2, 2), dtype=float),
        raw_modeled_best=np.full((2, 2), 2.0, dtype=float),
        residual=np.zeros((2, 2), dtype=float),
        fit_q0_trials=(2.0, 2.5),
        fit_metric_trials=(0.4, 0.1),
        fit_chi2_trials=(0.4, 0.1),
        fit_rho2_trials=(0.5, 0.2),
        fit_eta2_trials=(0.6, 0.3),
        trial_raw_modeled_maps=np.stack(
            [
                np.full((2, 2), 2.0, dtype=float),
                np.full((2, 2), 2.5, dtype=float),
            ],
            axis=0,
        ),
        trial_modeled_maps=np.stack(
            [
                np.full((2, 2), 1.0, dtype=float),
                np.full((2, 2), 1.5, dtype=float),
            ],
            axis=0,
        ),
        trial_residual_maps=np.stack(
            [
                np.zeros((2, 2), dtype=float),
                np.full((2, 2), 0.5, dtype=float),
            ],
            axis=0,
        ),
        nfev=2,
        nit=1,
        message="ok",
        used_adaptive_bracketing=False,
        bracket_found=False,
        bracket=None,
        target_metric="chi2",
        diagnostics=diagnostics,
    )

    write_single_point_scan_file(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_payload=point_payload,
        blos_reference=_make_blos_reference(),
    )

    payload = load_scan_file(out_h5)

    assert payload["artifact_format"] == "unified"
    assert payload["diagnostics"]["artifact_kind"] == UNIFIED_ARTIFACT_KIND
    assert payload["target_metric"] == "chi2"
    assert payload["a_values"].shape == (1,)
    assert payload["b_values"].shape == (1,)
    assert len(payload["point_records"]) == 1
    assert float(payload["point_records"][0]["q0"]) == pytest.approx(2.5)
    assert payload["point_records"][0]["trial_raw_modeled_maps"] is not None
    assert payload["point_records"][0]["trial_modeled_maps"] is not None
    assert payload["point_records"][0]["trial_residual_maps"] is not None
    assert payload["blos_reference"] is not None


def test_single_point_artifact_stores_auxiliary_maps_in_map_store(tmp_path: Path) -> None:
    out_h5 = tmp_path / "single_point_aux_maps.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_diagnostics(artifact_kind="pychmp_q0_recovery")
    aux_map = np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=float)
    point_payload = build_computed_point_payload(
        a_value=0.3,
        b_value=2.7,
        a_index=0,
        b_index=0,
        q0=2.5,
        success=True,
        status="computed",
        modeled_best=np.ones((2, 2), dtype=float),
        raw_modeled_best=np.full((2, 2), 2.0, dtype=float),
        residual=np.zeros((2, 2), dtype=float),
        fit_q0_trials=(2.0, 2.5),
        fit_metric_trials=(0.4, 0.1),
        fit_chi2_trials=(0.4, 0.1),
        fit_rho2_trials=(0.5, 0.2),
        fit_eta2_trials=(0.6, 0.3),
        nfev=2,
        nit=1,
        message="ok",
        used_adaptive_bracketing=False,
        bracket_found=False,
        bracket=None,
        target_metric="chi2",
        diagnostics=diagnostics,
        map_store_arrays={"euv/193/rendered_best": aux_map},
    )

    write_single_point_scan_file(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_payload=point_payload,
    )

    payload = load_scan_file(out_h5)
    np.testing.assert_allclose(payload["point_records"][0]["modeled_best"], np.ones((2, 2), dtype=float))

    with h5py.File(out_h5, "r") as handle:
        search_id = handle["slices/default/active_search_id"][()].decode()
        record = handle[f"slices/default/searches/{search_id}/point_records/r000000"]
        refs = json.loads(record["map_refs_json"][()].decode())
        assert "extra/euv/193/rendered_best" in refs
        ref_path = refs["extra/euv/193/rendered_best"]
        np.testing.assert_allclose(handle[ref_path]["data"][()], aux_map)
        assert "euv" not in record


def test_load_scan_file_replaces_missing_display_map_refs_with_blank_maps(tmp_path: Path) -> None:
    out_h5 = tmp_path / "missing_display_maps.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_diagnostics(artifact_kind="pychmp_ab_scan_sparse_points")
    point_payload = _make_point_payload(0.0, 1.0)

    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_records=[point_payload],
    )

    with h5py.File(out_h5, "r+") as handle:
        search_id = handle["slices/default/active_search_id"][()].decode()
        record = handle[f"slices/default/searches/{search_id}/point_records/r000000"]
        refs = json.loads(record["map_refs_json"][()].decode())
        for key in ("raw_modeled_best", "modeled_best", "residual"):
            ref_path = refs.pop(key)
            del handle[ref_path]
        record["map_refs_json"][()] = np.bytes_(json.dumps(refs, sort_keys=True))

    payload = load_scan_file(out_h5)
    point = payload["point_records"][0]

    assert point["raw_modeled_best"].shape == observed.shape
    assert point["modeled_best"].shape == observed.shape
    assert point["residual"].shape == observed.shape
    assert np.isnan(point["raw_modeled_best"]).all()
    assert np.isnan(point["modeled_best"]).all()
    assert np.isnan(point["residual"]).all()
    assert point["diagnostics"]["stored_display_maps_available"] is False


def test_auxiliary_map_store_records_can_seed_render_only_slice(tmp_path: Path) -> None:
    out_h5 = tmp_path / "aux_seed.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_diagnostics(artifact_kind="pychmp_ab_scan_sparse_points")
    diagnostics.update(
        {
            "spectral_domain": "euv",
            "spectral_label": "171 A",
            "wavelength_angstrom": 171.0,
            "target_slice_key": "euv_171",
            "slice_descriptors": [
                {
                    "key": "euv_171",
                    "domain": "euv",
                    "label": "171 A",
                    "channel_label": "171",
                    "wavelength_angstrom": 171.0,
                    "role": "target",
                    "is_target": True,
                },
                {
                    "key": "euv_193",
                    "domain": "euv",
                    "label": "193 A",
                    "channel_label": "193",
                    "wavelength_angstrom": 193.0,
                    "role": "auxiliary",
                    "is_target": False,
                },
            ],
        }
    )
    point_payload = build_computed_point_payload(
        a_value=0.3,
        b_value=2.7,
        a_index=0,
        b_index=0,
        q0=2.5,
        success=True,
        status="computed",
        modeled_best=np.ones((2, 2), dtype=float),
        raw_modeled_best=np.ones((2, 2), dtype=float),
        residual=np.zeros((2, 2), dtype=float),
        fit_q0_trials=(2.0, 2.5),
        fit_metric_trials=(0.4, 0.1),
        fit_chi2_trials=(0.4, 0.1),
        fit_rho2_trials=(0.5, 0.2),
        fit_eta2_trials=(0.6, 0.3),
        nfev=2,
        nit=1,
        message="ok",
        used_adaptive_bracketing=False,
        bracket_found=False,
        bracket=None,
        target_metric="chi2",
        diagnostics=diagnostics,
        map_store_arrays={
            "euv/193/trial_000/rendered": np.full((2, 2), 2.0, dtype=float),
            "euv/193/trial_001/rendered": np.full((2, 2), 2.5, dtype=float),
            "euv/193/rendered_best": np.full((2, 2), 2.5, dtype=float),
        },
    )
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_records=[point_payload],
    )

    records = load_auxiliary_map_store_point_records(out_h5, slice_key="euv_193")

    assert len(records) == 1
    assert records[0]["source_slice_key"] == "euv_171"
    assert records[0]["source_auxiliary_map_prefix"] == "extra/euv/193"
    assert records[0]["fit_q0_trials"] == (2.0, 2.5)
    np.testing.assert_allclose(records[0]["trial_modeled_maps"][0], np.full((2, 2), 2.0, dtype=float))
    np.testing.assert_allclose(records[0]["modeled_best"], np.full((2, 2), 2.5, dtype=float))


def test_auxiliary_map_store_records_can_use_synthetic_machine_keys(tmp_path: Path) -> None:
    out_h5 = tmp_path / "aux_seed_synthetic_keys.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_diagnostics(artifact_kind="pychmp_ab_scan_sparse_points")
    diagnostics.update(
        {
            "spectral_domain": "euv",
            "spectral_label": "171 A",
            "wavelength_angstrom": 171.0,
            "target_slice_key": "euv_171",
            "slice_descriptors": [
                {
                    "key": "euv_171",
                    "domain": "euv",
                    "label": "171 A",
                    "channel_label": "171",
                    "wavelength_angstrom": 171.0,
                    "role": "target",
                    "is_target": True,
                },
                {
                    "key": "euv_193",
                    "domain": "euv",
                    "label": "193 A",
                    "channel_label": "193",
                    "wavelength_angstrom": 193.0,
                    "role": "auxiliary",
                    "is_target": False,
                },
            ],
        }
    )
    synthetic_entries = [
        {
            "machine_key": "syn-193-best",
            "map_store_array": "synthetic/syn-193-best",
            "label": "EUV 193 best",
            "identity": {
                "schema": "pychmp.synthetic_map_db.v1",
                "domain_label": "euv",
                "channel_or_frequency": "193",
                "map_role": "rendered_best",
            },
        },
        {
            "machine_key": "syn-193-trial0",
            "map_store_array": "synthetic/syn-193-trial0",
            "label": "EUV 193 trial 0",
            "identity": {
                "schema": "pychmp.synthetic_map_db.v1",
                "domain_label": "euv",
                "channel_or_frequency": "193",
                "map_role": "trial_000_rendered",
            },
        },
        {
            "machine_key": "syn-193-trial1",
            "map_store_array": "synthetic/syn-193-trial1",
            "label": "EUV 193 trial 1",
            "identity": {
                "schema": "pychmp.synthetic_map_db.v1",
                "domain_label": "euv",
                "channel_or_frequency": "193",
                "map_role": "trial_001_rendered",
            },
        },
    ]
    point_payload = build_computed_point_payload(
        a_value=0.3,
        b_value=2.7,
        a_index=0,
        b_index=0,
        q0=2.5,
        success=True,
        status="computed",
        modeled_best=np.ones((2, 2), dtype=float),
        raw_modeled_best=np.ones((2, 2), dtype=float),
        residual=np.zeros((2, 2), dtype=float),
        fit_q0_trials=(2.0, 2.5),
        fit_metric_trials=(0.4, 0.1),
        fit_chi2_trials=(0.4, 0.1),
        fit_rho2_trials=(0.5, 0.2),
        fit_eta2_trials=(0.6, 0.3),
        nfev=2,
        nit=1,
        message="ok",
        used_adaptive_bracketing=False,
        bracket_found=False,
        bracket=None,
        target_metric="chi2",
        diagnostics={
            **diagnostics,
            "synthetic_map_db_version": 1,
            "synthetic_map_keys": synthetic_entries,
        },
        map_store_arrays={
            "synthetic/syn-193-best": np.full((2, 2), 2.5, dtype=float),
            "synthetic/syn-193-trial0": np.full((2, 2), 2.0, dtype=float),
            "synthetic/syn-193-trial1": np.full((2, 2), 2.5, dtype=float),
        },
    )
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_records=[point_payload],
    )

    records = load_auxiliary_map_store_point_records(out_h5, slice_key="euv_193")

    assert len(records) == 1
    assert records[0]["source_slice_key"] == "euv_171"
    assert records[0]["source_auxiliary_map_prefix"] == "synthetic_registry/euv_193"
    assert records[0]["source_synthetic_machine_keys"] == ["syn-193-best", "syn-193-trial0", "syn-193-trial1"]
    assert records[0]["fit_q0_trials"] == (2.0, 2.5)
    np.testing.assert_allclose(records[0]["trial_modeled_maps"][0], np.full((2, 2), 2.0, dtype=float))
    np.testing.assert_allclose(records[0]["modeled_best"], np.full((2, 2), 2.5, dtype=float))


def test_auxiliary_map_store_records_disable_synthetic_key_lookup_uses_legacy_prefix(tmp_path: Path) -> None:
    out_h5 = tmp_path / "aux_seed_disable_synthetic_keys.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_diagnostics(artifact_kind="pychmp_ab_scan_sparse_points")
    diagnostics.update(
        {
            "spectral_domain": "euv",
            "spectral_label": "171 A",
            "wavelength_angstrom": 171.0,
            "target_slice_key": "euv_171",
            "slice_descriptors": [
                {
                    "key": "euv_171",
                    "domain": "euv",
                    "label": "171 A",
                    "channel_label": "171",
                    "wavelength_angstrom": 171.0,
                    "role": "target",
                    "is_target": True,
                },
                {
                    "key": "euv_193",
                    "domain": "euv",
                    "label": "193 A",
                    "channel_label": "193",
                    "wavelength_angstrom": 193.0,
                    "role": "auxiliary",
                    "is_target": False,
                },
            ],
        }
    )
    point_payload = build_computed_point_payload(
        a_value=0.3,
        b_value=2.7,
        a_index=0,
        b_index=0,
        q0=2.5,
        success=True,
        status="computed",
        modeled_best=np.ones((2, 2), dtype=float),
        raw_modeled_best=np.ones((2, 2), dtype=float),
        residual=np.zeros((2, 2), dtype=float),
        fit_q0_trials=(2.0, 2.5),
        fit_metric_trials=(0.4, 0.1),
        fit_chi2_trials=(0.4, 0.1),
        fit_rho2_trials=(0.5, 0.2),
        fit_eta2_trials=(0.6, 0.3),
        nfev=2,
        nit=1,
        message="ok",
        used_adaptive_bracketing=False,
        bracket_found=False,
        bracket=None,
        target_metric="chi2",
        diagnostics={
            **diagnostics,
            "synthetic_map_db_version": 1,
            "synthetic_map_keys": [
                {
                    "machine_key": "syn-193-best",
                    "map_store_array": "synthetic/syn-193-best",
                    "label": "EUV 193 best",
                    "identity": {
                        "schema": "pychmp.synthetic_map_db.v1",
                        "domain_label": "euv",
                        "channel_or_frequency": "193",
                        "map_role": "rendered_best",
                    },
                },
                {
                    "machine_key": "syn-193-trial0",
                    "map_store_array": "synthetic/syn-193-trial0",
                    "label": "EUV 193 trial 0",
                    "identity": {
                        "schema": "pychmp.synthetic_map_db.v1",
                        "domain_label": "euv",
                        "channel_or_frequency": "193",
                        "map_role": "trial_000_rendered",
                    },
                },
                {
                    "machine_key": "syn-193-trial1",
                    "map_store_array": "synthetic/syn-193-trial1",
                    "label": "EUV 193 trial 1",
                    "identity": {
                        "schema": "pychmp.synthetic_map_db.v1",
                        "domain_label": "euv",
                        "channel_or_frequency": "193",
                        "map_role": "trial_001_rendered",
                    },
                },
            ],
        },
        map_store_arrays={
            # Legacy prefix-based auxiliary maps (expected when synthetic lookup is disabled).
            "euv/193/trial_000/rendered": np.full((2, 2), 2.0, dtype=float),
            "euv/193/trial_001/rendered": np.full((2, 2), 2.5, dtype=float),
            "euv/193/rendered_best": np.full((2, 2), 2.5, dtype=float),
            # Synthetic-key maps with deliberately different values to catch accidental use.
            "synthetic/syn-193-best": np.full((2, 2), 8.5, dtype=float),
            "synthetic/syn-193-trial0": np.full((2, 2), 8.0, dtype=float),
            "synthetic/syn-193-trial1": np.full((2, 2), 8.5, dtype=float),
        },
    )
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_records=[point_payload],
    )

    records = load_auxiliary_map_store_point_records(
        out_h5,
        slice_key="euv_193",
        use_synthetic_machine_keys=False,
    )

    assert len(records) == 1
    assert records[0]["source_slice_key"] == "euv_171"
    assert records[0]["source_auxiliary_map_prefix"] == "extra/euv/193"
    assert "source_synthetic_machine_keys" not in records[0]
    assert records[0]["fit_q0_trials"] == (2.0, 2.5)
    np.testing.assert_allclose(records[0]["trial_modeled_maps"][0], np.full((2, 2), 2.0, dtype=float))
    np.testing.assert_allclose(records[0]["modeled_best"], np.full((2, 2), 2.5, dtype=float))


def test_auxiliary_map_store_records_disable_synthetic_key_lookup_skips_synthetic_only_records(tmp_path: Path) -> None:
    out_h5 = tmp_path / "aux_seed_disable_synthetic_keys_only.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_diagnostics(artifact_kind="pychmp_ab_scan_sparse_points")
    diagnostics.update(
        {
            "spectral_domain": "euv",
            "spectral_label": "171 A",
            "wavelength_angstrom": 171.0,
            "target_slice_key": "euv_171",
            "slice_descriptors": [
                {
                    "key": "euv_171",
                    "domain": "euv",
                    "label": "171 A",
                    "channel_label": "171",
                    "wavelength_angstrom": 171.0,
                    "role": "target",
                    "is_target": True,
                },
                {
                    "key": "euv_193",
                    "domain": "euv",
                    "label": "193 A",
                    "channel_label": "193",
                    "wavelength_angstrom": 193.0,
                    "role": "auxiliary",
                    "is_target": False,
                },
            ],
        }
    )
    point_payload = build_computed_point_payload(
        a_value=0.3,
        b_value=2.7,
        a_index=0,
        b_index=0,
        q0=2.5,
        success=True,
        status="computed",
        modeled_best=np.ones((2, 2), dtype=float),
        raw_modeled_best=np.ones((2, 2), dtype=float),
        residual=np.zeros((2, 2), dtype=float),
        fit_q0_trials=(2.0, 2.5),
        fit_metric_trials=(0.4, 0.1),
        fit_chi2_trials=(0.4, 0.1),
        fit_rho2_trials=(0.5, 0.2),
        fit_eta2_trials=(0.6, 0.3),
        nfev=2,
        nit=1,
        message="ok",
        used_adaptive_bracketing=False,
        bracket_found=False,
        bracket=None,
        target_metric="chi2",
        diagnostics={
            **diagnostics,
            "synthetic_map_db_version": 1,
            "synthetic_map_keys": [
                {
                    "machine_key": "syn-193-best",
                    "map_store_array": "synthetic/syn-193-best",
                    "label": "EUV 193 best",
                    "identity": {
                        "schema": "pychmp.synthetic_map_db.v1",
                        "domain_label": "euv",
                        "channel_or_frequency": "193",
                        "map_role": "rendered_best",
                    },
                },
                {
                    "machine_key": "syn-193-trial0",
                    "map_store_array": "synthetic/syn-193-trial0",
                    "label": "EUV 193 trial 0",
                    "identity": {
                        "schema": "pychmp.synthetic_map_db.v1",
                        "domain_label": "euv",
                        "channel_or_frequency": "193",
                        "map_role": "trial_000_rendered",
                    },
                },
            ],
        },
        map_store_arrays={
            # Synthetic-only maps; no legacy prefix keys present.
            "synthetic/syn-193-best": np.full((2, 2), 8.5, dtype=float),
            "synthetic/syn-193-trial0": np.full((2, 2), 8.0, dtype=float),
        },
    )
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_records=[point_payload],
    )

    records = load_auxiliary_map_store_point_records(
        out_h5,
        slice_key="euv_193",
        use_synthetic_machine_keys=False,
    )

    assert records == []


def test_synthetic_map_registry_is_persisted_and_point_diagnostics_are_compact(tmp_path: Path) -> None:
    out_h5 = tmp_path / "synthetic_registry.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_diagnostics(artifact_kind="pychmp_ab_scan_sparse_points")
    synthetic_machine_key = "synthetic-key-001"
    point_payload = build_computed_point_payload(
        a_value=0.3,
        b_value=2.7,
        a_index=0,
        b_index=0,
        q0=2.5,
        success=True,
        status="computed",
        modeled_best=np.ones((2, 2), dtype=float),
        raw_modeled_best=np.ones((2, 2), dtype=float),
        residual=np.zeros((2, 2), dtype=float),
        fit_q0_trials=(2.0, 2.5),
        fit_metric_trials=(0.4, 0.1),
        fit_chi2_trials=(0.4, 0.1),
        fit_rho2_trials=(0.5, 0.2),
        fit_eta2_trials=(0.6, 0.3),
        nfev=2,
        nit=1,
        message="ok",
        used_adaptive_bracketing=False,
        bracket_found=False,
        bracket=None,
        target_metric="chi2",
        diagnostics={
            **diagnostics,
            "synthetic_map_keys": [
                {
                    "machine_key": synthetic_machine_key,
                    "map_store_array": "synthetic/synthetic-key-001",
                    "label": "Synthetic test map",
                    "identity": {"schema": "pychmp.synthetic_map_db.v1", "a": 0.3, "b": 2.7, "q0": 2.5},
                }
            ],
        },
        map_store_arrays={"synthetic/synthetic-key-001": np.full((2, 2), 9.0, dtype=float)},
    )
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_records=[point_payload],
    )

    with h5py.File(out_h5, "r") as handle:
        search_id = handle["slices/default/active_search_id"][()].decode()
        record = handle[f"slices/default/searches/{search_id}/point_records/r000000"]
        stored_diag = json.loads(record["diagnostics_json"][()].decode())
        assert stored_diag["synthetic_map_machine_keys"] == [synthetic_machine_key]
        assert "identity" not in stored_diag["synthetic_map_keys"][0]
        assert "synthetic_map_machine_keys_json" in record

        registry_entry = handle[f"map_store/synthetic_registry/{synthetic_machine_key}"]
        assert registry_entry["machine_key"][()].decode() == synthetic_machine_key
        ref_path = registry_entry["map_ref_path"][()].decode()
        np.testing.assert_allclose(handle[ref_path]["data"][()], np.full((2, 2), 9.0, dtype=float))


def test_synthetic_map_registry_dedupes_entries_across_repeated_writes(tmp_path: Path) -> None:
    out_h5 = tmp_path / "synthetic_registry_dedupe.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_diagnostics(artifact_kind="pychmp_ab_scan_sparse_points")
    synthetic_machine_key = "synthetic-key-dedupe"

    def _payload() -> dict[str, object]:
        return build_computed_point_payload(
            a_value=0.3,
            b_value=2.7,
            a_index=0,
            b_index=0,
            q0=2.5,
            success=True,
            status="computed",
            modeled_best=np.ones((2, 2), dtype=float),
            raw_modeled_best=np.ones((2, 2), dtype=float),
            residual=np.zeros((2, 2), dtype=float),
            fit_q0_trials=(2.0, 2.5),
            fit_metric_trials=(0.4, 0.1),
            fit_chi2_trials=(0.4, 0.1),
            fit_rho2_trials=(0.5, 0.2),
            fit_eta2_trials=(0.6, 0.3),
            nfev=2,
            nit=1,
            message="ok",
            used_adaptive_bracketing=False,
            bracket_found=False,
            bracket=None,
            target_metric="chi2",
            diagnostics={
                **diagnostics,
                "synthetic_map_keys": [
                    {
                        "machine_key": synthetic_machine_key,
                        "map_store_array": "synthetic/synthetic-key-dedupe",
                        "label": "Synthetic dedupe map",
                        "identity": {"schema": "pychmp.synthetic_map_db.v1", "a": 0.3, "b": 2.7, "q0": 2.5},
                    }
                ],
            },
            map_store_arrays={"synthetic/synthetic-key-dedupe": np.full((2, 2), 7.0, dtype=float)},
        )

    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_records=[_payload()],
    )
    append_point_record(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_payload=_payload(),
    )

    with h5py.File(out_h5, "r") as handle:
        registry = handle["map_store/synthetic_registry"]
        assert sorted(registry.keys()) == [synthetic_machine_key]


def test_append_point_record_updates_rectangular_artifact(tmp_path: Path) -> None:
    out_h5 = tmp_path / "scan.h5"
    observed, sigma_map, header, diagnostics = _write_rectangular_artifact(out_h5)
    point_payload = _make_point_payload(0.0, 1.0)
    point_payload["diagnostics"] = dict(point_payload["diagnostics"])
    point_payload["diagnostics"]["chi2"] = 0.05
    point_payload["diagnostics"]["target_metric_value"] = 0.05
    point_payload["q0"] = 3.5

    append_point_record(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_payload=point_payload,
    )

    payload = load_scan_file(out_h5)
    assert payload["artifact_format"] == "unified"
    assert float(payload["best_q0"][0, 0]) == pytest.approx(3.5)
    assert float(payload["chi2"][0, 0]) == pytest.approx(0.05)


def test_new_rectangular_artifact_uses_search_point_records_as_canonical_store(tmp_path: Path) -> None:
    out_h5 = tmp_path / "scan.h5"
    _write_rectangular_artifact(out_h5)

    with h5py.File(out_h5, "r") as handle:
        slice_group = handle["slices/default"]
        assert "point_records" not in slice_group
        assert "points" not in slice_group
        assert "summary" not in slice_group
        search_id = slice_group["active_search_id"][()].decode()
        record = slice_group["searches"][search_id]["point_records/r000000"]
        assert "point_records" in slice_group["searches"][search_id]
        assert "map_refs_json" in record
        assert "modeled_best" not in record
        assert "raw_modeled_best" not in record
        assert "residual" not in record
        assert "map_store" in handle
        assert "common" not in handle
        assert "point_records" not in handle
        assert "searches" not in handle


def test_new_sparse_artifact_uses_slice_layout(tmp_path: Path) -> None:
    out_h5 = tmp_path / "sparse_scan.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_diagnostics(artifact_kind="pychmp_ab_scan_sparse_points")
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_records=[_make_point_payload(0.0, 1.0)],
    )

    with h5py.File(out_h5, "r") as handle:
        assert "slices" in handle
        slice_group = handle["slices/default"]
        assert "common" in slice_group
        assert "point_records" not in slice_group
        assert "searches" in slice_group
        search_id = slice_group["active_search_id"][()].decode()
        record = slice_group["searches"][search_id]["point_records/r000000"]
        assert "point_records" in slice_group["searches"][search_id]
        assert "map_refs_json" in record
        assert "modeled_best" not in record
        assert "raw_modeled_best" not in record
        assert "residual" not in record
        assert "map_store" in handle
        assert "common" not in handle
        assert "point_records" not in handle
        assert "searches" not in handle


def test_legacy_writer_names_remain_compatibility_wrappers() -> None:
    assert ab_scan_artifacts.save_rectangular_scan_file is not write_grid_scan_artifact
    assert ab_scan_artifacts.write_sparse_scan_file is not write_point_scan_artifact
    assert ab_scan_artifacts.append_sparse_point_record is not append_scan_point_record


def test_append_point_record_updates_sparse_artifact(tmp_path: Path) -> None:
    out_h5 = tmp_path / "sparse_scan.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_diagnostics(artifact_kind="pychmp_ab_scan_sparse_points")
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_records=[],
    )
    point_payload = _make_point_payload(0.0, 1.0)

    append_point_record(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_payload=point_payload,
    )

    payload = load_scan_file(out_h5)
    assert payload["artifact_format"] == "unified"
    assert len(payload["point_records"]) == 1
    assert float(payload["point_records"][0]["q0"]) == pytest.approx(2.5)


def test_append_point_record_requires_initialized_artifact(tmp_path: Path) -> None:
    out_h5 = tmp_path / "missing.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_diagnostics()
    point_payload = _make_point_payload(0.0, 1.0)

    with pytest.raises(FileNotFoundError, match="initialized before appending"):
        append_point_record(
            out_h5,
            observed=observed,
            sigma_map=sigma_map,
            wcs_header=header,
            diagnostics=diagnostics,
            point_payload=point_payload,
        )


def test_sparse_point_record_signature_filtering() -> None:
    record = _make_point_payload(0.0, 1.0)
    assert not point_record_matches_compatibility_signature(
        record,
        compatibility_signature="sig-123",
    )
    record["diagnostics"] = dict(record["diagnostics"])
    record["diagnostics"][COMPATIBILITY_SIGNATURE_KEY] = "sig-123"
    assert point_record_matches_compatibility_signature(
        record,
        compatibility_signature="sig-123",
    )
    assert not point_record_matches_compatibility_signature(
        record,
        compatibility_signature="sig-other",
    )


def test_search_status_normalizes_point_status_text() -> None:
    from pychmp.ab_scan_artifacts import _search_status_from_records

    assert _search_status_from_records([{"status": " Computed "}]) == "complete"
    assert _search_status_from_records([{"status": " PENDING "}]) == "in_progress"
    assert _search_status_from_records([{"status": " FAILED "}, {"status": "computed"}]) == "partial"


def test_append_scan_point_record_maintains_incremental_status_counts(tmp_path: Path) -> None:
    out_h5 = tmp_path / "sparse_counts.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_diagnostics(artifact_kind="pychmp_ab_scan_sparse_points")
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_records=[],
    )
    failed_point = _make_point_payload(0.0, 1.0)
    failed_point["status"] = " failed "
    append_scan_point_record(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_payload=failed_point,
    )
    computed_point = _make_point_payload(0.3, 1.0)
    append_scan_point_record(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_payload=computed_point,
    )

    with h5py.File(out_h5, "r") as handle:
        slice_group = handle["slices/default"]
        search_id = slice_group["active_search_id"][()].decode()
        search = slice_group["searches"][search_id]
        assert int(search.attrs["total_point_count"]) == 2
        assert int(search.attrs["failed_point_count"]) == 1
        assert int(search.attrs["computed_point_count"]) == 1
        assert search.attrs["status"].decode() == "partial"
        assert search.attrs["in_progress"] == 1
        assert "request_json" in search
        assert "lifecycle_json" in search


def test_search_records_expose_request_and_lifecycle_metadata(tmp_path: Path) -> None:
    out_h5 = tmp_path / "scan_lifecycle.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_diagnostics()
    diagnostics.update(
        {
            "target_metric": "eta2",
            "metrics_mask_threshold": 0.25,
            "q0_min": 1.0e-5,
            "q0_max": 1.0e-3,
            "max_bracket_steps": 15,
            "search_created_at": "2026-05-22T00:00:00Z",
            "search_started_at": "2026-05-22T00:00:01Z",
        }
    )
    write_grid_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        a_values=np.asarray([0.0, 0.3], dtype=float),
        b_values=np.asarray([2.4], dtype=float),
        best_q0=np.asarray([[1.0], [2.0]], dtype=float),
        objective_values=np.asarray([[3.0], [4.0]], dtype=float),
        chi2=np.asarray([[5.0], [6.0]], dtype=float),
        rho2=np.asarray([[7.0], [8.0]], dtype=float),
        eta2=np.asarray([[9.0], [10.0]], dtype=float),
        success=np.asarray([[True], [True]], dtype=bool),
        point_payloads={
            (0, 0): _make_point_payload(0.0, 2.4, a_index=0, b_index=0),
            (1, 0): _make_point_payload(0.3, 2.4, a_index=1, b_index=0),
        },
    )

    payload = load_scan_file(out_h5)
    search = payload["selected_search"]

    assert search["status"] == "complete"
    assert search["active"] is True
    assert search["in_progress"] is False
    assert search["created_at"] == "2026-05-22T00:00:00Z"
    assert search["started_at"] == "2026-05-22T00:00:01Z"
    assert search["completed_at"]
    assert search["request"]["target_metric"] == "eta2"
    assert search["request"]["metrics_mask"]["threshold"] == 0.25
    assert search["request"]["optimizer"]["q0_min"] == 1.0e-5
    assert search["request"]["optimizer"]["q0_max"] == 1.0e-3
    assert search["request"]["optimizer"]["max_bracket_steps"] == 15
    assert search["request"]["requested_points"] == [{"a": 0.0, "b": 2.4}, {"a": 0.3, "b": 2.4}]


def test_new_slice_reuse_rejects_geometry_mismatch(tmp_path: Path) -> None:
    out_h5 = tmp_path / "slices.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    first_diag = _make_diagnostics(artifact_kind="pychmp_ab_scan_sparse_points")
    first_diag["slice_key"] = "mw_5p7"
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=first_diag,
        point_records=[],
    )

    incompatible_diag = dict(first_diag)
    incompatible_diag["slice_key"] = "mw_8p2"
    incompatible_diag["frequency_ghz"] = 8.2
    incompatible_diag["map_nx"] = 3

    with pytest.raises(ScanArtifactCompatibilityError, match="geometry mismatch"):
        write_point_scan_artifact(
            out_h5,
            observed=observed,
            sigma_map=sigma_map,
            wcs_header=header,
            diagnostics=incompatible_diag,
            point_records=[],
        )


def test_new_slice_reuse_accepts_matching_geometry(tmp_path: Path) -> None:
    out_h5 = tmp_path / "slices.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    first_diag = _make_diagnostics(artifact_kind="pychmp_ab_scan_sparse_points")
    first_diag["slice_key"] = "mw_5p7"
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=first_diag,
        point_records=[],
    )

    compatible_diag = dict(first_diag)
    compatible_diag["slice_key"] = "mw_8p2"
    compatible_diag["frequency_ghz"] = 8.2
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=compatible_diag,
        point_records=[],
    )

    assert [item["key"] for item in list_scan_slices(out_h5)] == ["mw_5p7", "mw_8p2"]


def test_rectangular_artifact_round_trip_preserves_run_history(tmp_path: Path) -> None:
    out_h5 = tmp_path / "scan.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_diagnostics()
    write_grid_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        a_values=np.asarray([0.0], dtype=float),
        b_values=np.asarray([1.0], dtype=float),
        best_q0=np.asarray([[2.5]], dtype=float),
        objective_values=np.asarray([[0.1]], dtype=float),
        chi2=np.asarray([[0.1]], dtype=float),
        rho2=np.asarray([[0.2]], dtype=float),
        eta2=np.asarray([[0.3]], dtype=float),
        success=np.asarray([[True]], dtype=bool),
        point_payloads={(0, 0): _make_point_payload(0.0, 1.0)},
        run_history=[{"timestamp_utc": "2026-04-13T21:00:00Z", "action": "create"}],
    )

    payload = load_scan_file(out_h5)

    assert len(payload["run_history"]) == 1
    assert payload["run_history"][0]["action"] == "create"


def test_rectangular_artifact_round_trip_preserves_shared_blos_reference(tmp_path: Path) -> None:
    out_h5 = tmp_path / "scan_blos.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_diagnostics()
    blos_reference = _make_blos_reference()
    write_grid_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        blos_reference=blos_reference,
        a_values=np.asarray([0.0], dtype=float),
        b_values=np.asarray([1.0], dtype=float),
        best_q0=np.asarray([[2.5]], dtype=float),
        objective_values=np.asarray([[0.1]], dtype=float),
        chi2=np.asarray([[0.1]], dtype=float),
        rho2=np.asarray([[0.2]], dtype=float),
        eta2=np.asarray([[0.3]], dtype=float),
        success=np.asarray([[True]], dtype=bool),
        point_payloads={(0, 0): _make_point_payload(0.0, 1.0)},
    )

    payload = load_scan_file(out_h5)

    assert payload["blos_reference"] is not None
    blos_data, blos_header = payload["blos_reference"]
    np.testing.assert_allclose(blos_data, blos_reference[0])
    assert float(blos_header["CRVAL1"]) == pytest.approx(float(blos_reference[1]["CRVAL1"]))


def test_sparse_artifact_round_trip_preserves_shared_blos_reference(tmp_path: Path) -> None:
    out_h5 = tmp_path / "sparse_blos.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_diagnostics(artifact_kind="pychmp_ab_scan_sparse_points")
    blos_reference = _make_blos_reference()
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        blos_reference=blos_reference,
        point_records=[_make_point_payload(0.0, 1.0)],
    )

    payload = load_scan_file(out_h5)

    assert payload["blos_reference"] is not None
    blos_data, blos_header = payload["blos_reference"]
    np.testing.assert_allclose(blos_data, blos_reference[0])
    assert float(blos_header["CRVAL1"]) == pytest.approx(float(blos_reference[1]["CRVAL1"]))


def test_sparse_artifact_round_trip_exposes_canonical_slice_metadata_and_trial_logging_policy(tmp_path: Path) -> None:
    out_h5 = tmp_path / "sparse_contract.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_diagnostics(artifact_kind="pychmp_ab_scan_sparse_points")
    diagnostics.update(
        {
            "spectral_domain": "euv",
            "spectral_label": "171 A",
            "wavelength_angstrom": 171.0,
            "euv_channel": "171",
            "slice_descriptors": [
                {
                    "key": "euv_171",
                    "domain": "euv",
                    "label": "171 A",
                    "wavelength_angstrom": 171.0,
                    "channel_label": "171",
                    "role": "target",
                },
                {
                    "key": "euv_193",
                    "domain": "euv",
                    "label": "193 A",
                    "wavelength_angstrom": 193.0,
                    "channel_label": "193",
                    "role": "auxiliary",
                },
            ],
            "target_slice_key": "euv_171",
            "store_raw_rendered_cubes": True,
            "store_trial_metric_masks": True,
            "store_euv_component_cubes": True,
            "store_euv_tr_mask": True,
        }
    )
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_records=[_make_point_payload(0.0, 1.0)],
    )

    payload = load_scan_file(out_h5)

    assert payload["artifact_contract_version"] == CANONICAL_ARTIFACT_CONTRACT_VERSION
    assert payload["target_slice_key"] == "euv_171"
    assert len(payload["canonical_slice_descriptors"]) == 2
    assert payload["canonical_slice_descriptors"][0]["key"] == "euv_171"
    assert payload["canonical_slice_descriptors"][0]["is_target"] is True
    assert payload["canonical_slice_descriptors"][1]["key"] == "euv_193"
    assert payload["canonical_slice_descriptors"][1]["is_target"] is False
    assert payload["trial_logging_policy"]["store_observed_maps"] is True
    assert payload["trial_logging_policy"]["store_trial_metrics"] is True
    assert payload["trial_logging_policy"]["store_raw_rendered_cubes"] is True
    assert payload["trial_logging_policy"]["store_trial_metric_masks"] is True
    assert payload["trial_logging_policy"]["store_euv_component_cubes"] is True
    assert payload["trial_logging_policy"]["store_euv_tr_mask"] is True

    slices = list_scan_slices(out_h5)
    assert [item["key"] for item in slices] == ["euv_171", "euv_193"]
    aux_payload = load_scan_file(out_h5, slice_key="euv_193")
    assert aux_payload["selected_slice_key"] == "euv_193"
    assert aux_payload["diagnostics"]["render_only_slice"] is True
    assert aux_payload["point_records"] == []
    assert np.isnan(aux_payload["observed"]).all()


def test_load_scan_file_prefers_target_slice_key_when_unspecified(tmp_path: Path) -> None:
    out_h5 = tmp_path / "target_slice_default.h5"
    observed = np.ones((2, 2), dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_diagnostics(artifact_kind=UNIFIED_ARTIFACT_KIND)
    diagnostics.update(
        {
            "spectral_domain": "euv",
            "spectral_label": "193 A",
            "wavelength_angstrom": 193.0,
            "euv_channel": "193",
            "slice_descriptors": [
                {
                    "key": "euv_171",
                    "domain": "euv",
                    "label": "171 A",
                    "wavelength_angstrom": 171.0,
                    "channel_label": "171",
                    "role": "auxiliary",
                },
                {
                    "key": "euv_193",
                    "domain": "euv",
                    "label": "193 A",
                    "wavelength_angstrom": 193.0,
                    "channel_label": "193",
                    "role": "target",
                },
            ],
            "target_slice_key": "euv_193",
        }
    )
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_records=[_make_point_payload(0.3, 2.1)],
    )

    payload = load_scan_file(out_h5)

    assert payload["selected_slice_key"] == "euv_193"


def test_point_selection_helpers_use_existing_sparse_point_records() -> None:
    point = {
        **_make_point_payload(0.3, 2.1, a_index=1, b_index=0),
        "record_order": 2,
        "metrics": {"chi2": 1.0, "rho2": 2.0, "eta2": 3.0},
        "diagnostics": {"chi2": 1.0, "rho2": 2.0, "eta2": 3.0},
    }
    payload = {
        "a_values": np.asarray([0.0, 0.3], dtype=float),
        "b_values": np.asarray([2.1, 2.4], dtype=float),
        "points": {(1, 0): point},
        "point_records": [point],
    }

    assert default_point_index(payload, "chi2") == (1, 0)
    assert resolve_point_index(payload, metric="chi2", a_index=0, b_index=0) == (1, 0)


def test_single_point_artifact_round_trips_euv_components_and_tr_mask(tmp_path: Path) -> None:
    out_h5 = tmp_path / "single_point_euv_components.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_diagnostics(artifact_kind="pychmp_ab_scan_sparse_points")
    diagnostics.update(
        {
            "spectral_domain": "euv",
            "spectral_label": "171 A",
            "wavelength_angstrom": 171.0,
            "slice_descriptors": [
                {
                    "key": "euv_171",
                    "domain": "euv",
                    "label": "171 A",
                    "wavelength_angstrom": 171.0,
                    "role": "target",
                    "is_target": True,
                }
            ],
        }
    )
    point_payload = build_computed_point_payload(
        a_value=0.3,
        b_value=2.7,
        a_index=0,
        b_index=0,
        q0=2.5,
        success=True,
        status="computed",
        modeled_best=np.ones((2, 2), dtype=float),
        raw_modeled_best=np.ones((2, 2), dtype=float),
        residual=np.zeros((2, 2), dtype=float),
        fit_q0_trials=(2.0, 2.5),
        fit_metric_trials=(0.4, 0.1),
        fit_chi2_trials=(0.4, 0.1),
        fit_rho2_trials=(0.5, 0.2),
        fit_eta2_trials=(0.6, 0.3),
        trial_raw_modeled_maps=np.ones((2, 2, 2), dtype=float),
        trial_modeled_maps=np.ones((2, 2, 2), dtype=float) * 2.0,
        trial_residual_maps=np.ones((2, 2, 2), dtype=float) * -1.0,
        euv_coronal_best=np.full((2, 2), 3.0, dtype=float),
        euv_tr_best=np.full((2, 2), 4.0, dtype=float),
        euv_tr_mask=np.asarray([[True, False], [False, True]], dtype=bool),
        trial_euv_coronal_maps=np.full((2, 2, 2), 5.0, dtype=float),
        trial_euv_tr_maps=np.full((2, 2, 2), 6.0, dtype=float),
        nfev=2,
        nit=1,
        message="ok",
        used_adaptive_bracketing=False,
        bracket_found=False,
        bracket=None,
        target_metric="chi2",
        diagnostics={"target_metric": "chi2"},
    )

    write_single_point_scan_file(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_payload=point_payload,
        blos_reference=None,
        run_history=None,
    )

    payload = load_scan_file(out_h5)
    point = payload["point_records"][0]

    np.testing.assert_allclose(point["euv_coronal_best"], np.full((2, 2), 3.0, dtype=float))
    np.testing.assert_allclose(point["euv_tr_best"], np.full((2, 2), 4.0, dtype=float))
    np.testing.assert_array_equal(point["euv_tr_mask"], np.asarray([[True, False], [False, True]], dtype=bool))
    np.testing.assert_allclose(point["trial_euv_coronal_maps"], np.full((2, 2, 2), 5.0, dtype=float))
    np.testing.assert_allclose(point["trial_euv_tr_maps"], np.full((2, 2, 2), 6.0, dtype=float))


def test_append_scan_point_record_retries_transient_file_lock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Retry transient HDF5 open failures so viewer/read contention does not abort the run."""
    out_h5 = tmp_path / "sparse_retry.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_diagnostics(artifact_kind="pychmp_ab_scan_sparse_points")
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_records=[],
    )

    import pychmp.ab_scan_artifacts as artifacts

    real_h5py_file = artifacts.h5py.File
    state = {"calls": 0}

    def flaky_file(*args: object, **kwargs: object):
        state["calls"] += 1
        if state["calls"] <= 2:
            raise OSError("simulated lock")
        return real_h5py_file(*args, **kwargs)

    monkeypatch.setattr(artifacts.h5py, "File", flaky_file)
    monkeypatch.setattr(artifacts.time, "sleep", lambda _seconds: None)

    append_scan_point_record(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_payload=_make_point_payload(0.0, 1.0),
    )

    payload = load_scan_file(out_h5)

    assert state["calls"] == 3
    assert len(payload["point_records"]) == 1


def test_backfill_artifact_diagnostics_populates_missing_hashes_and_map_shape(tmp_path: Path) -> None:
    out_h5 = tmp_path / "legacy_scan.h5"
    observed, _sigma_map, _header, _diagnostics = _write_rectangular_artifact(out_h5)

    model_file = tmp_path / "model.h5"
    fits_file = tmp_path / "obs.fits"
    ebtel_file = tmp_path / "ebtel.bin"
    model_file.write_bytes(b"model-bytes")
    fits_file.write_bytes(b"fits-bytes")
    ebtel_file.write_bytes(b"ebtel-bytes")

    payload = load_scan_file(out_h5)
    legacy_diagnostics = dict(payload["diagnostics"])
    legacy_diagnostics["model_path"] = str(model_file)
    legacy_diagnostics["fits_file"] = str(fits_file)
    legacy_diagnostics["ebtel_path"] = str(ebtel_file)
    for key in ("model_sha256", "fits_sha256", "ebtel_sha256", "map_nx", "map_ny"):
        legacy_diagnostics.pop(key, None)

    with h5py.File(out_h5, "r+") as f:
        group = f["slices"][list(f["slices"].keys())[0]]
        group["common"]["diagnostics_json"][()] = np.bytes_(json.dumps(legacy_diagnostics, sort_keys=True))

    report = backfill_artifact_diagnostics(out_h5)

    assert report["updated_slice_count"] == 1
    assert report["updated_fields"]["map_nx"] == 1
    assert report["updated_fields"]["map_ny"] == 1
    assert report["updated_fields"]["model_sha256"] == 1
    assert report["updated_fields"]["fits_sha256"] == 1
    assert report["updated_fields"]["ebtel_sha256"] == 1

    refreshed = load_scan_file(out_h5)
    refreshed_diag = dict(refreshed["diagnostics"])
    assert int(refreshed_diag["map_nx"]) == observed.shape[1]
    assert int(refreshed_diag["map_ny"]) == observed.shape[0]
    assert len(str(refreshed_diag["model_sha256"])) == 64
    assert len(str(refreshed_diag["fits_sha256"])) == 64
    assert len(str(refreshed_diag["ebtel_sha256"])) == 64


def test_backfill_artifact_diagnostics_skips_missing_sources(tmp_path: Path) -> None:
    out_h5 = tmp_path / "legacy_scan.h5"
    _write_rectangular_artifact(out_h5)
    payload = load_scan_file(out_h5)
    legacy_diagnostics = dict(payload["diagnostics"])
    legacy_diagnostics["model_path"] = str(tmp_path / "missing-model.h5")
    legacy_diagnostics["fits_file"] = str(tmp_path / "missing-obs.fits")
    legacy_diagnostics["ebtel_path"] = str(tmp_path / "missing-ebtel.bin")
    for key in ("model_sha256", "fits_sha256", "ebtel_sha256", "map_nx", "map_ny"):
        legacy_diagnostics.pop(key, None)

    with h5py.File(out_h5, "r+") as f:
        group = f["slices"][list(f["slices"].keys())[0]]
        group["common"]["diagnostics_json"][()] = np.bytes_(json.dumps(legacy_diagnostics, sort_keys=True))

    report = backfill_artifact_diagnostics(out_h5, dry_run=True)

    assert report["updated_slice_count"] == 1
    assert report["updated_fields"]["map_nx"] == 1
    assert report["updated_fields"]["map_ny"] == 1
    assert "model_sha256" in report["skipped_fields"]
    assert "fits_sha256" in report["skipped_fields"]
    assert "ebtel_sha256" in report["skipped_fields"]

    unchanged = load_scan_file(out_h5)
    unchanged_diag = dict(unchanged["diagnostics"])
    assert "model_sha256" not in unchanged_diag
    assert "fits_sha256" not in unchanged_diag
    assert "ebtel_sha256" not in unchanged_diag
