from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from pychmp.ab_scan_artifacts import (
    ScanArtifactCompatibilityError,
    append_sparse_point_record,
    load_scan_file,
    save_rectangular_scan_file,
    scan_artifact_compatibility_issues,
    validate_scan_artifact_compatibility,
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


def _make_diagnostics(*, artifact_kind: str = "pychmp_ab_scan", model_path: str = "model.h5") -> dict[str, object]:
    return {
        "artifact_kind": artifact_kind,
        "target_metric": "chi2",
        "model_path": model_path,
        "fits_file": "obs.fits",
        "ebtel_path": "ebtel.bin",
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
    changed_diagnostics["model_path"] = "other_model.h5"

    issues = scan_artifact_compatibility_issues(
        payload,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=changed_diagnostics,
    )

    assert any("model_path" in issue for issue in issues)
    with pytest.raises(ScanArtifactCompatibilityError, match="model_path"):
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
