from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from pychmp.ssrt_beam import (
    beam_fit_ssrt,
    beam_fit_ssrt_from_header,
    fit_ssrt_beam_ellipse,
    is_ssrt_header,
    make_ssrt_beam,
    ssrt_time_from_header,
)
from pychmp.psf import build_psf_kernel, extract_psf_metadata_from_header, sigma_arcsec_to_fwhm


def _idl_beam_path() -> Path | None:
    env = os.environ.get("PYCHMP_SSRT_IDL_BEAM")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env).expanduser())
    repo = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            repo.parent / "SRH-NORH-4CHMP" / "ssrt2oct2012_idl_beam.fits",
            Path("/Users/gelu/code/SUNCAST-ORG/SRH-NORH-4CHMP/ssrt2oct2012_idl_beam.fits"),
        ]
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


_IDL_BEAM = _idl_beam_path()
# GetSSRTangles output for map time "2-Oct-2012 03:16:23.917"
_IDL_ANGLES = (19.157769, 25.826949, -27.850994, -85.062927)
_IDL_FIT = (14.500631, 7.3190091, 0.79709549)

requires_idl_beam = pytest.mark.skipif(
    _IDL_BEAM is None, reason="SSRT IDL beam FITS fixture not found"
)


@requires_idl_beam
def test_make_ssrt_beam_matches_idl_fixture() -> None:
    assert _IDL_BEAM is not None
    d_ew, d_ns, p_ew, p_ns = _IDL_ANGLES
    _x, _y, beam = make_ssrt_beam(
        d_ew, d_ns, p_ew, p_ns, nx=50, ny=50, dx_arcsec=1.0, dy_arcsec=1.0
    )
    idl = np.asarray(fits.getdata(_IDL_BEAM), dtype=float).T
    assert beam.shape == idl.shape
    assert float(np.max(np.abs(beam - idl))) < 1e-7
    assert float(np.corrcoef(beam.ravel(), idl.ravel())[0, 1]) > 1.0 - 1e-12


@requires_idl_beam
def test_fit_ssrt_beam_matches_idl_beamfitssrt() -> None:
    d_ew, d_ns, p_ew, p_ns = _IDL_ANGLES
    x, y, beam = make_ssrt_beam(
        d_ew, d_ns, p_ew, p_ns, nx=50, ny=50, dx_arcsec=1.0, dy_arcsec=1.0
    )
    sx, sy, theta = fit_ssrt_beam_ellipse(beam, x, y)
    py_axes = sorted([sx, sy], reverse=True)
    id_axes = sorted([_IDL_FIT[0], _IDL_FIT[1]], reverse=True)
    assert py_axes[0] == pytest.approx(id_axes[0], rel=1e-4)
    assert py_axes[1] == pytest.approx(id_axes[1], rel=1e-4)
    if sx >= sy:
        pa = theta
    else:
        pa = theta + 0.5 * np.pi
    # wrap to (-pi/2, pi/2]
    pa = float(np.arctan2(np.sin(pa), np.cos(pa)))
    id_pa = float(_IDL_FIT[2]) if _IDL_FIT[0] >= _IDL_FIT[1] else float(_IDL_FIT[2] + 0.5 * np.pi)
    id_pa = float(np.arctan2(np.sin(id_pa), np.cos(id_pa)))
    assert pa == pytest.approx(id_pa, abs=1e-3)


def test_is_ssrt_header_detects_telescop_and_badary() -> None:
    header = fits.Header()
    assert is_ssrt_header(header) is False
    header["TELESCOP"] = "SSRT"
    assert is_ssrt_header(header) is True
    header2 = fits.Header()
    header2["INSTRUME"] = "AOR BADARY"
    header2["ORIGIN"] = "Siberian Solar Radio Telescope"
    assert is_ssrt_header(header2) is True


def test_ssrt_time_from_header_reads_date_obs_and_split_cards() -> None:
    header = fits.Header()
    header["DATE-OBS"] = "2012-07-12T02:49:22.341"
    assert ssrt_time_from_header(header) == "2012-07-12T02:49:22.341"
    split = fits.Header()
    split["DATE-OBS"] = "2012-07-12"
    split["TIME-OBS"] = "02:49:22.341"
    assert ssrt_time_from_header(split) == "2012-07-12 02:49:22.341"


def test_extract_psf_metadata_from_ssrt_header_uses_time_beam() -> None:
    """Standard SSRT FITS (TELESCOP + DATE-OBS) routes through GetSSRTangles/BeamFitSSRT."""
    header = fits.Header()
    header["TELESCOP"] = "SSRT"
    header["INSTRUME"] = "AOR"
    header["DATE-OBS"] = "12-Jul-2012 02:49:22.341"
    header["CDELT1"] = 4.9110398
    header["CDELT2"] = 4.9110398
    # Even with a stale BMAJ present, SSRT time path wins (matches ConvolveSSRT).
    header["BMAJ"] = 1.0 / 3600.0
    header["BMIN"] = 0.5 / 3600.0
    header["BPA"] = 0.0

    metadata = extract_psf_metadata_from_header(header)
    assert metadata is not None
    assert metadata.source == "fits_header:ssrt_beam"
    assert metadata.kind == "gaussian"
    assert metadata.allows_frequency_scaling is True
    assert metadata.bmaj_arcsec is not None and metadata.bmin_arcsec is not None
    assert metadata.bmaj_arcsec > metadata.bmin_arcsec > 0.0

    params = beam_fit_ssrt_from_header(header)
    assert metadata.bmaj_arcsec == pytest.approx(sigma_arcsec_to_fwhm(max(params.sigma_x_arcsec, params.sigma_y_arcsec)))
    assert metadata.bmin_arcsec == pytest.approx(sigma_arcsec_to_fwhm(min(params.sigma_x_arcsec, params.sigma_y_arcsec)))

    # Same time through the direct BeamFitSSRT path must agree.
    direct, _ = beam_fit_ssrt("12-Jul-2012 02:49:22.341", marx=50, dx_arcsec=1.0)
    assert params.sigma_x_arcsec == pytest.approx(direct.sigma_x_arcsec, rel=1e-6)
    assert params.sigma_y_arcsec == pytest.approx(direct.sigma_y_arcsec, rel=1e-6)

    kernel, kernel_meta = build_psf_kernel(
        metadata=metadata,
        dx_arcsec=4.9110398,
        dy_arcsec=4.9110398,
        active_frequency_ghz=5.7,
        ref_frequency_ghz=5.7,
        scale_inverse_frequency=True,
    )
    assert kernel is not None and kernel_meta is not None
    assert float(kernel.sum()) == pytest.approx(1.0)


def test_ssrt_header_without_time_does_not_claim_psf() -> None:
    header = fits.Header()
    header["TELESCOP"] = "SSRT"
    assert extract_psf_metadata_from_header(header) is None


def test_eovsa_like_bmaj_header_still_uses_standard_path() -> None:
    header = fits.Header()
    header["TELESCOP"] = "EOVSA"
    header["BMAJ"] = 30.0 / 3600.0
    header["BMIN"] = 20.0 / 3600.0
    header["BPA"] = 15.0
    metadata = extract_psf_metadata_from_header(header)
    assert metadata is not None
    assert metadata.source == "fits_header"
    assert metadata.bmaj_arcsec == pytest.approx(30.0)
    assert metadata.bmin_arcsec == pytest.approx(20.0)
