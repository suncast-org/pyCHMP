from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits

from pychmp.ab_scan_artifacts import detect_scan_artifact_format, load_scan_file
from examples.fit_q0_obs_map import save_q0_artifact
from examples.replot_q0_artifacts import _parse_artifact_h5
from pychmp.q0_artifact_panel import load_blos_reference_from_artifact


def _sample_header() -> fits.Header:
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 4
    header["NAXIS2"] = 3
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRPIX1"] = 2.0
    header["CRPIX2"] = 2.0
    header["CRVAL1"] = -600.0
    header["CRVAL2"] = -300.0
    header["CDELT1"] = 2.0
    header["CDELT2"] = 2.0
    header["DATE-OBS"] = "2020-11-26T20:00:00"
    return header


def test_save_q0_artifact_writes_replot_contract_with_embedded_blos(tmp_path: Path) -> None:
    h5_path = tmp_path / "artifact.h5"
    header = _sample_header()
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    diagnostics = {
        "model_path": "/tmp/model.h5",
        "target_metric": "chi2",
        "fit_q0_trials": [1.0e-4, 2.0e-4, 3.0e-4],
        "fit_metric_trials": [400.0, 350.0, 360.0],
        "active_frequency_ghz": 2.874,
        "a": 0.3,
        "b": 2.7,
    }

    save_q0_artifact(
        h5_path,
        observed=data,
        sigma_map=np.ones_like(data),
        modeled_best=data + 1.0,
        raw_modeled_best=data + 2.0,
        residual=np.full_like(data, -1.0),
        frequency_ghz=2.874,
        q0_fitted=3.64e-4,
        metrics_dict={"chi2": 350.0, "rho2": 0.5, "eta2": 0.7},
        diagnostics=diagnostics,
        noise_diagnostics={"method": "histogram_clip"},
        wcs_header=header,
        model_path=Path("/tmp/model.h5"),
        blos_reference=(data * 10.0, header),
        trial_raw_modeled_maps=np.stack([data + 2.0, data + 3.0, data + 4.0], axis=0),
        trial_modeled_maps=np.stack([data + 1.0, data + 1.5, data + 2.0], axis=0),
        trial_residual_maps=np.stack([np.full_like(data, -1.0), np.full_like(data, -0.5), np.zeros_like(data)], axis=0),
    )

    assert detect_scan_artifact_format(h5_path) == "sparse"
    payload = load_scan_file(h5_path)
    point = payload["point_records"][0]
    assert payload["artifact_format"] == "sparse"
    assert payload["target_metric"] == "chi2"
    assert point["target_metric"] == "chi2"
    np.testing.assert_allclose(point["fit_q0_trials"], [1.0e-4, 2.0e-4, 3.0e-4])
    np.testing.assert_allclose(point["fit_metric_trials"], [400.0, 350.0, 360.0])
    assert point["trial_raw_modeled_maps"] is not None
    assert point["trial_raw_modeled_maps"].shape == (3, 3, 4)
    assert point["trial_modeled_maps"] is not None
    assert point["trial_residual_maps"] is not None
    assert payload["blos_reference"] is not None

    blos_reference = load_blos_reference_from_artifact(h5_path)
    assert blos_reference is not None
    blos_data, blos_header = blos_reference
    np.testing.assert_allclose(blos_data, data * 10.0)
    assert blos_header["CRVAL1"] == header["CRVAL1"]

    parsed = _parse_artifact_h5(h5_path)
    np.testing.assert_allclose(parsed["observed"], data)
    np.testing.assert_allclose(parsed["modeled"], data + 1.0)
    np.testing.assert_allclose(parsed["raw_modeled"], data + 2.0)
    assert parsed["blos_reference"] is not None


def test_save_q0_artifact_keeps_euv_artifact_generic_without_forced_frequency(tmp_path: Path) -> None:
    h5_path = tmp_path / "artifact_euv.h5"
    header = _sample_header()
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    diagnostics = {
        "model_path": "/tmp/model.h5",
        "target_metric": "chi2",
        "fit_q0_trials": [1.0e-5, 2.0e-5, 3.0e-5],
        "fit_metric_trials": [14.0, 13.5, 13.8],
        "spectral_domain": "euv",
        "spectral_label": "171 A",
        "wavelength_angstrom": 171.0,
        "euv_channel": "171",
        "a": 0.3,
        "b": 2.7,
    }

    save_q0_artifact(
        h5_path,
        observed=data,
        sigma_map=np.ones_like(data),
        modeled_best=data + 1.0,
        raw_modeled_best=data + 2.0,
        residual=np.full_like(data, -1.0),
        frequency_ghz=None,
        q0_fitted=3.0e-5,
        metrics_dict={"chi2": 13.5, "rho2": 0.8, "eta2": 1.1},
        diagnostics=diagnostics,
        noise_diagnostics=None,
        wcs_header=header,
        model_path=Path("/tmp/model.h5"),
        blos_reference=None,
    )

    payload = load_scan_file(h5_path)
    diagnostics_payload = payload["diagnostics"]
    assert diagnostics_payload["spectral_domain"] == "euv"
    assert diagnostics_payload["spectral_label"] == "171 A"
    assert diagnostics_payload["wavelength_angstrom"] == 171.0
    assert diagnostics_payload["euv_channel"] == "171"
    assert payload["canonical_slice_descriptors"][0]["domain"] == "euv"
    assert payload["canonical_slice_descriptors"][0]["wavelength_angstrom"] == 171.0

    parsed = _parse_artifact_h5(h5_path)
    assert parsed["diagnostics"]["spectral_domain"] == "euv"
    assert parsed["diagnostics"]["spectral_label"] == "171 A"
