from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits
import pytest

from examples import estimate_map_noise_cli
from examples import fit_q0_obs_map
from examples import scan_ab_obs_map


def _make_resolution_args(tmp_path: Path, **overrides: object) -> Namespace:
    values: dict[str, object] = {
        "fits_file": None,
        "model_h5": tmp_path / "model.h5",
        "model_h5_override": None,
        "obs_source": None,
        "obs_path": None,
        "obs_map_id": None,
        "ebtel_path": tmp_path / "ebtel.sav",
        "testdata_repo": None,
        "euv_instrument": None,
        "euv_response_sav": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_estimate_map_noise_cli_loads_map_via_shared_observation_loader(tmp_path: Path) -> None:
    fits_path = tmp_path / "mw_map.fits"
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    header = fits.Header()
    header["CUNIT3"] = "Hz"
    header["CRVAL3"] = 2.874e9
    header["INSTRUME"] = "EOVSA"
    fits.PrimaryHDU(data=data, header=header).writeto(fits_path)

    obs_map = estimate_map_noise_cli.load_observational_map(fits_path)

    assert obs_map.source_mode == "external_fits"
    assert obs_map.source_path == str(fits_path.resolve())
    assert obs_map.frequency_ghz == 2.874
    assert obs_map.wcs_metadata is not None
    assert obs_map.wcs_metadata["hdu_name"] == "PRIMARY"
    np.testing.assert_allclose(obs_map.data, data)


def test_fit_observation_request_defaults_to_model_refmap_when_map_id_is_supplied(tmp_path: Path) -> None:
    args = _make_resolution_args(tmp_path, obs_map_id="AIA_171")

    request = fit_q0_obs_map._resolve_observation_request(args, repo_root=tmp_path)

    assert request.source_mode == "model_refmap"
    assert request.obs_path is None
    assert request.obs_map_id == "AIA_171"
    assert request.model_h5 == (tmp_path / "model.h5").resolve()
    assert request.ebtel_path == (tmp_path / "ebtel.sav").resolve()


def test_fit_observation_request_rejects_conflicting_path_selectors(tmp_path: Path) -> None:
    args = _make_resolution_args(
        tmp_path,
        fits_file=tmp_path / "obs_a.fits",
        obs_path=tmp_path / "obs_b.fits",
    )

    with pytest.raises(SystemExit, match="Conflicting observation path selectors"):
        fit_q0_obs_map._resolve_observation_request(args, repo_root=tmp_path)


def test_scan_and_fit_observation_request_resolution_stay_aligned(tmp_path: Path) -> None:
    args = _make_resolution_args(tmp_path, obs_map_id="AIA_171")

    fit_request = fit_q0_obs_map._resolve_observation_request(args, repo_root=tmp_path)
    scan_request = scan_ab_obs_map._resolve_observation_request(args, repo_root=tmp_path)

    assert scan_request.source_mode == fit_request.source_mode
    assert scan_request.obs_path == fit_request.obs_path
    assert scan_request.obs_map_id == fit_request.obs_map_id
    assert scan_request.model_h5 == fit_request.model_h5
    assert scan_request.ebtel_path == fit_request.ebtel_path


def test_fit_render_selection_derives_euv_channel_and_instrument(tmp_path: Path) -> None:
    args = _make_resolution_args(tmp_path)
    model_path = tmp_path / "model.h5"
    with h5py.File(model_path, "w") as h5f:
        group = h5f.create_group("refmaps/AIA_171")
        group.create_dataset("data", data=np.ones((4, 4), dtype=np.float32))
        header = fits.Header()
        header["WAVELNTH"] = 171
        header["INSTRUME"] = "AIA"
        group.create_dataset("wcs_header", data=np.bytes_(header.tostring(sep="\n", endcard=True)))
    obs_map = fit_q0_obs_map.validate_obs_map_identity(
        fit_q0_obs_map.load_obs_map(
            model_h5=model_path,
            map_id="AIA_171",
            source_mode="model_refmap",
        ),
        wavelength_angstrom_hint=171.0,
    )

    selection = fit_q0_obs_map._resolve_render_selection(args, obs_map)

    assert selection.domain == "euv"
    assert selection.spectral_label == "171 A"
    assert selection.euv_channel == "171"
    assert selection.euv_instrument == "AIA"
    assert selection.active_frequency_ghz is None


def test_fit_render_selection_rejects_conflicting_euv_instrument_override(tmp_path: Path) -> None:
    header = fits.Header()
    header["INSTRUME"] = "AIA"
    header["WAVELNTH"] = 171
    data = np.ones((4, 4), dtype=float)
    fits_path = tmp_path / "aia_171.fits"
    fits.PrimaryHDU(data=data, header=header).writeto(fits_path)
    obs_map = fit_q0_obs_map.validate_obs_map_identity(
        fit_q0_obs_map.load_obs_map(obs_path=fits_path, source_mode="external_fits")
    )
    args = _make_resolution_args(tmp_path, euv_instrument="SUVI")

    with pytest.raises(ValueError, match="conflicting EUV instrument request"):
        fit_q0_obs_map._resolve_render_selection(args, obs_map)
