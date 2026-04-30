from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy.io import fits

from pychmp.ab_scan_artifacts import (
    COMPATIBILITY_SIGNATURE_KEY,
    ScanArtifactCompatibilityError,
    append_point_record,
    append_sparse_point_record,
    backfill_artifact_diagnostics,
    build_computed_point_payload,
    load_scan_file,
    point_record_matches_compatibility_signature,
    save_rectangular_scan_file,
    scan_artifact_compatibility_issues,
    validate_scan_artifact_compatibility,
    write_single_point_scan_file,
    write_sparse_scan_file,
)


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
    save_rectangular_scan_file(
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


def test_validate_scan_artifact_compatibility_rejects_rectangular_signature_mismatch(tmp_path: Path) -> None:
    out_h5 = tmp_path / "scan.h5"
    observed, sigma_map, header, diagnostics = _write_rectangular_artifact(out_h5)
    payload = load_scan_file(out_h5)
    changed_diagnostics = dict(diagnostics)
    changed_diagnostics[COMPATIBILITY_SIGNATURE_KEY] = "sig-other"

    with pytest.raises(ScanArtifactCompatibilityError, match=COMPATIBILITY_SIGNATURE_KEY):
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
    write_sparse_scan_file(
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
    write_sparse_scan_file(
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


def test_write_single_point_scan_file_round_trip_is_sparse_and_viewer_compatible(tmp_path: Path) -> None:
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

    assert payload["artifact_format"] == "sparse"
    assert payload["target_metric"] == "chi2"
    assert payload["a_values"].shape == (1,)
    assert payload["b_values"].shape == (1,)
    assert len(payload["point_records"]) == 1
    assert float(payload["point_records"][0]["q0"]) == pytest.approx(2.5)
    assert payload["point_records"][0]["trial_raw_modeled_maps"] is not None
    assert payload["point_records"][0]["trial_modeled_maps"] is not None
    assert payload["point_records"][0]["trial_residual_maps"] is not None
    assert payload["blos_reference"] is not None


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
    assert payload["artifact_format"] == "rectangular"
    assert float(payload["best_q0"][0, 0]) == pytest.approx(3.5)
    assert float(payload["chi2"][0, 0]) == pytest.approx(0.05)


def test_new_rectangular_artifact_writes_canonical_point_records(tmp_path: Path) -> None:
    out_h5 = tmp_path / "scan.h5"
    _write_rectangular_artifact(out_h5)

    with h5py.File(out_h5, "r") as handle:
        slice_group = handle["slices/default"]
        assert "point_records" in slice_group
        assert "points" in slice_group


def test_append_point_record_updates_sparse_artifact(tmp_path: Path) -> None:
    out_h5 = tmp_path / "sparse_scan.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_diagnostics(artifact_kind="pychmp_ab_scan_sparse_points")
    write_sparse_scan_file(
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
    assert payload["artifact_format"] == "sparse"
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


def test_rectangular_artifact_round_trip_preserves_run_history(tmp_path: Path) -> None:
    out_h5 = tmp_path / "scan.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_diagnostics()
    save_rectangular_scan_file(
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
    save_rectangular_scan_file(
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
    write_sparse_scan_file(
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
    write_sparse_scan_file(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_records=[_make_point_payload(0.0, 1.0)],
    )

    payload = load_scan_file(out_h5)

    assert payload["artifact_contract_version"] == "2026-04-23-a"
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


def test_append_sparse_point_record_retries_transient_file_lock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Retry transient HDF5 open failures so viewer/read contention does not abort the run."""
    out_h5 = tmp_path / "sparse_retry.h5"
    observed = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma_map = np.ones_like(observed)
    header = _make_header()
    diagnostics = _make_diagnostics(artifact_kind="pychmp_ab_scan_sparse_points")
    write_sparse_scan_file(
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

    append_sparse_point_record(
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
