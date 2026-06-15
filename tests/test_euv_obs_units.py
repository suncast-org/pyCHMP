from __future__ import annotations

import numpy as np
from astropy.io import fits
import pytest

from pychmp.euv_obs_units import (
    MODELED_EUV_BUNIT,
    convert_euv_observation_to_rate,
    euv_observation_appears_integrated_dn,
    euv_observation_declares_dn_rate,
    resolve_euv_exposure_seconds,
)


def test_euv_observation_declares_dn_rate_from_bunit() -> None:
    header = fits.Header()
    header["BUNIT"] = "DN s^-1 pix^-1"
    assert euv_observation_declares_dn_rate(header)


def test_euv_observation_appears_integrated_from_pixlunit() -> None:
    header = fits.Header()
    header["PIXLUNIT"] = "DN"
    assert euv_observation_appears_integrated_dn(header)
    assert not euv_observation_declares_dn_rate(header)


def test_convert_integrated_dn_to_rate() -> None:
    data = np.array([[100.0, 200.0]], dtype=float)
    header = fits.Header()
    header["PIXLUNIT"] = "DN"
    header["EXPTIME"] = 2.0
    rate, out_header, diag = convert_euv_observation_to_rate(data, header)
    np.testing.assert_allclose(rate, [[50.0, 100.0]])
    assert out_header["BUNIT"] == MODELED_EUV_BUNIT
    assert out_header["PIXLUNIT"] == MODELED_EUV_BUNIT
    assert diag["euv_unit_conversion"] == "integrated_dn_to_rate"
    assert diag["euv_exposure_seconds"] == 2.0


def test_convert_skips_when_already_rate() -> None:
    data = np.array([[3.0, 4.0]], dtype=float)
    header = fits.Header()
    header["BUNIT"] = "DN/s"
    rate, _, diag = convert_euv_observation_to_rate(data, header)
    np.testing.assert_allclose(rate, data)
    assert diag["euv_unit_conversion"] == "already_rate"


def test_resolve_exposure_from_source_fits(tmp_path) -> None:
    source = tmp_path / "aia_171.fits"
    header = fits.Header()
    header["EXPTIME"] = 2.5
    fits.PrimaryHDU(data=np.ones((2, 2), dtype=np.float32), header=header).writeto(source)

    ref_header = fits.Header()
    assert resolve_euv_exposure_seconds(ref_header, source_path=source) == 2.5
