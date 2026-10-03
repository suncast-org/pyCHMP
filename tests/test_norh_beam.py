from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from pychmp.norh_beam import (
    beam_fit_norh_from_header,
    is_norh_header,
    make_norh_beam,
    norh_index_from_header,
    norh_prog2pinf,
)
from pychmp.psf import build_psf_kernel, extract_psf_metadata_from_header, sigma_arcsec_to_fwhm


def _ifz_path() -> Path | None:
    env = os.environ.get("PYCHMP_NORH_IFZ")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env).expanduser())
    repo_root = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            repo_root.parent / "SRH-NORH-4CHMP" / "ifz140202_022005_corrected",
            Path("/Users/gelu/code/SUNCAST-ORG/SRH-NORH-4CHMP/ifz140202_022005_corrected"),
        ]
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


IFZ_PATH = _ifz_path()
requires_ifz = pytest.mark.skipif(IFZ_PATH is None, reason="NORH IFZ fixture not found")


def test_norh_prog2pinf_snap2d34_uses_scaled_dirty_pixel() -> None:
    sec_dty, efl, roll = norh_prog2pinf("snap2d34 v6.2  Y. Hanaoka", "34GHz")
    assert roll == pytest.approx(0.0)
    assert efl == pytest.approx(2.0654677)
    assert sec_dty == pytest.approx(4.64947 * 17.0 / 33.8)


def test_is_norh_header_requires_pmat_and_freq() -> None:
    header = fits.Header()
    header["ORIGIN"] = "nobeyama radio obs"
    header["TELESCOP"] = "radioheliograph"
    assert is_norh_header(header) is False
    header["OBS-FREQ"] = "34GHz"
    header["PMAT1"] = 1.0
    header["PMAT2"] = 0.0
    header["PMAT3"] = 0.0
    header["PMAT4"] = 1.0
    assert is_norh_header(header) is True


@requires_ifz
def test_norh_beam_pixels_match_idl_reference() -> None:
    """Spot-check marx=21 beam samples against IDL ``norh_beam`` on the IFZ file."""
    assert IFZ_PATH is not None
    header = fits.getheader(IFZ_PATH)
    index = norh_index_from_header(header)
    # IDL norh_beam uses get_rb0p(/pangle); recorded for this IFZ timestamp.
    solp_idl = -0.21937722
    beam = make_norh_beam(
        sec_per_pix=index["sec_per_pix"],
        sec_per_pix_dty=index["sec_per_pix_dty"],
        efl_in_pix_dty=index["efl_in_pix_dty"],
        solp_rad=solp_idl,
        pmat=index["pmat"],
        marx=21,
    )
    assert beam[10, 10] == pytest.approx(1.0, rel=0, abs=1e-12)
    assert beam[11, 10] == pytest.approx(0.772464, rel=1e-5)
    assert beam[10, 11] == pytest.approx(0.909542, rel=1e-5)
    assert beam[15, 10] == pytest.approx(0.00157380, rel=1e-4)
    assert beam[10, 15] == pytest.approx(0.0934483, rel=1e-4)
    assert beam[0, 0] == pytest.approx(1.19841e-16, rel=1e-3)


@requires_ifz
def test_beam_fit_norh_matches_idl_beamfitnorh() -> None:
    """IDL ``BeamFitNoRH`` on IFZ @ marx=51: sx=5.6473555, sy=3.4153717, theta=-1.5292061."""
    assert IFZ_PATH is not None
    header = fits.getheader(IFZ_PATH)
    params = beam_fit_norh_from_header(header, marx=51, solp_rad=-0.21937722)
    # FitBeam may swap axes; compare ordered major/minor + PA of major.
    axes = sorted([params.sigma_x_arcsec, params.sigma_y_arcsec], reverse=True)
    assert axes[0] == pytest.approx(5.6473555, rel=1e-4)
    assert axes[1] == pytest.approx(3.4153717, rel=1e-4)
    if params.sigma_x_arcsec >= params.sigma_y_arcsec:
        pa_maj = params.theta_deg
    else:
        pa_maj = params.theta_deg + 90.0
    # Wrap to (-90, 90]
    pa_maj = ((pa_maj + 90.0) % 180.0) - 90.0
    if pa_maj <= -90.0:
        pa_maj += 180.0
    assert pa_maj == pytest.approx(-87.617057, abs=0.05)


@requires_ifz
def test_extract_psf_metadata_from_ifz_header() -> None:
    assert IFZ_PATH is not None
    header = fits.getheader(IFZ_PATH)
    metadata = extract_psf_metadata_from_header(header)
    assert metadata is not None
    assert metadata.source == "fits_header:norh_beam"
    assert metadata.kind == "gaussian"
    assert metadata.allows_frequency_scaling is True
    assert metadata.bmaj_arcsec == pytest.approx(sigma_arcsec_to_fwhm(5.6473555), rel=1e-3)
    assert metadata.bmin_arcsec == pytest.approx(sigma_arcsec_to_fwhm(3.4153717), rel=1e-3)
    assert metadata.bpa_deg == pytest.approx(-87.617057, abs=0.1)

    kernel, kernel_meta = build_psf_kernel(
        metadata=metadata,
        dx_arcsec=float(header["CDELT1"]),
        dy_arcsec=float(header["CDELT2"]),
        active_frequency_ghz=17.0,
        ref_frequency_ghz=34.0,
        scale_inverse_frequency=True,
    )
    assert kernel is not None and kernel_meta is not None
    assert float(kernel.sum()) == pytest.approx(1.0)
    assert bool(kernel_meta["scaled"]) is True
    assert float(kernel_meta["scale_factor"]) == pytest.approx(2.0)
