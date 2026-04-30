from __future__ import annotations

import h5py
import numpy as np
from astropy.io import fits
import pytest

from pychmp import estimate_obs_map_noise, load_obs_map, validate_obs_map_identity


def test_load_obs_map_mw_external_fits_extracts_frequency(tmp_path) -> None:
    fits_path = tmp_path / "mw_map.fits"
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    header = fits.Header()
    header["CUNIT3"] = "Hz"
    header["CRVAL3"] = 2.874e9
    header["INSTRUME"] = "EOVSA"
    header["DATE-OBS"] = "2020-11-26T20:00:00"
    fits.PrimaryHDU(data=data, header=header).writeto(fits_path)

    obs_map = load_obs_map(obs_path=fits_path, domain="mw")

    assert obs_map.domain == "mw"
    assert obs_map.instrument == "EOVSA"
    assert obs_map.frequency_ghz == 2.874
    assert obs_map.wavelength_angstrom is None
    assert obs_map.spectral_label == "2.874 GHz"
    assert obs_map.source_mode == "external_fits"
    assert obs_map.source_path == str(fits_path.resolve())
    np.testing.assert_allclose(obs_map.data, data)


def test_load_obs_map_euv_external_fits_extracts_wavelength(tmp_path) -> None:
    fits_path = tmp_path / "aia_171.fits"
    data = np.ones((3, 5), dtype=np.float32)
    header = fits.Header()
    header["WAVELNTH"] = 171
    header["WAVEUNIT"] = "angstrom"
    header["INSTRUME"] = "AIA"
    header["DATE-OBS"] = "2020-11-26T20:00:00"
    fits.PrimaryHDU(data=data, header=header).writeto(fits_path)

    obs_map = load_obs_map(obs_path=fits_path, domain="euv")

    assert obs_map.domain == "euv"
    assert obs_map.instrument == "AIA"
    assert obs_map.frequency_ghz is None
    assert obs_map.wavelength_angstrom == 171.0
    assert obs_map.spectral_label == "171 A"
    np.testing.assert_allclose(obs_map.data, data)


def test_load_obs_map_model_refmap_reads_internal_aia_map(tmp_path) -> None:
    model_h5 = tmp_path / "model.h5"
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    header = fits.Header()
    header["DATE-OBS"] = "2025-11-26T15:34:33.350"
    header["RSUN_OBS"] = 972.5
    header["HGLT_OBS"] = 1.44
    header["HGLN_OBS"] = 44.92
    header["INSTRUME"] = "AIA"

    with h5py.File(model_h5, "w") as h5f:
        group = h5f.create_group("refmaps").create_group("AIA_171")
        group.create_dataset("data", data=data)
        group.create_dataset("wcs_header", data=np.bytes_(header.tostring(sep="\n", endcard=True)))

    obs_map = load_obs_map(model_h5=model_h5, map_id="AIA_171", source_mode="model_refmap")

    assert obs_map.domain == "euv"
    assert obs_map.instrument == "AIA"
    assert obs_map.frequency_ghz is None
    assert obs_map.wavelength_angstrom == 171.0
    assert obs_map.spectral_label == "171 A"
    assert obs_map.source_mode == "model_refmap"
    assert obs_map.source_map_id == "AIA_171"
    assert obs_map.source_path == str(model_h5.resolve())
    assert obs_map.wcs_metadata is not None
    assert obs_map.wcs_metadata["group_path"] == "refmaps/AIA_171"
    np.testing.assert_allclose(obs_map.data, data)


def test_load_obs_map_model_refmap_requires_map_id(tmp_path) -> None:
    model_h5 = tmp_path / "model.h5"
    with h5py.File(model_h5, "w") as h5f:
        group = h5f.create_group("refmaps").create_group("AIA_193")
        group.create_dataset("data", data=np.ones((2, 2), dtype=np.float32))
        group.create_dataset("wcs_header", data=np.bytes_(fits.Header().tostring(sep="\n", endcard=True)))

    with pytest.raises(ValueError, match="map_id is required"):
        load_obs_map(model_h5=model_h5, source_mode="model_refmap")


def test_estimate_obs_map_noise_falls_back_to_uniform_std_for_invalid_map(tmp_path) -> None:
    fits_path = tmp_path / "invalid_map.fits"
    data = np.full((4, 4), 7.0, dtype=np.float32)
    header = fits.Header()
    header["CUNIT3"] = "Hz"
    header["CRVAL3"] = 2.874e9
    fits.PrimaryHDU(data=data, header=header).writeto(fits_path)

    obs_map = load_obs_map(obs_path=fits_path, domain="mw")
    with pytest.warns(UserWarning, match="Map data quality check failed"):
        noise = estimate_obs_map_noise(obs_map, method="histogram_clip")

    assert noise.method_used == "fallback_std"
    assert noise.sigma == 0.0
    np.testing.assert_allclose(noise.sigma_map, np.zeros_like(data, dtype=float))
    assert noise.diagnostics is not None
    assert noise.diagnostics["source_method"] == "histogram_clip"


def test_validate_obs_map_identity_infers_euv_domain_from_wavelength_metadata(tmp_path) -> None:
    fits_path = tmp_path / "aia_193.fits"
    data = np.ones((3, 3), dtype=np.float32)
    header = fits.Header()
    header["WAVELNTH"] = 193
    header["WAVEUNIT"] = "angstrom"
    fits.PrimaryHDU(data=data, header=header).writeto(fits_path)

    obs_map = load_obs_map(obs_path=fits_path)
    validated = validate_obs_map_identity(obs_map)

    assert validated.domain == "euv"
    assert validated.wavelength_angstrom == 193.0
    assert validated.spectral_label == "193 A"


def test_validate_obs_map_identity_rejects_mw_map_with_euv_hint(tmp_path) -> None:
    fits_path = tmp_path / "mw_map.fits"
    data = np.arange(9, dtype=np.float32).reshape(3, 3)
    header = fits.Header()
    header["CUNIT3"] = "Hz"
    header["CRVAL3"] = 2.874e9
    fits.PrimaryHDU(data=data, header=header).writeto(fits_path)

    obs_map = load_obs_map(obs_path=fits_path, domain="mw")

    with pytest.raises(ValueError, match="requests EUV/UV"):
        validate_obs_map_identity(obs_map, wavelength_angstrom_hint=171.0)


def test_validate_obs_map_identity_rejects_euv_map_with_mw_hint(tmp_path) -> None:
    fits_path = tmp_path / "aia_171.fits"
    data = np.ones((4, 4), dtype=np.float32)
    header = fits.Header()
    header["WAVELNTH"] = 171
    header["WAVEUNIT"] = "angstrom"
    fits.PrimaryHDU(data=data, header=header).writeto(fits_path)

    obs_map = load_obs_map(obs_path=fits_path, domain="euv")

    with pytest.raises(ValueError, match="requests MW"):
        validate_obs_map_identity(obs_map, frequency_ghz_hint=5.7)
