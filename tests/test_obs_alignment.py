from __future__ import annotations

import numpy as np
from astropy.io import fits

from pychmp.obs_alignment import (
    build_padded_canvas_header,
    extract_observation_to_model_fov,
    find_shift,
)
from pychmp.obs_preprocessing import regrid_observation_to_target_fov


def _make_header(*, nx: int, ny: int, crval1: float = 0.0, crval2: float = 0.0, cdelt: float = 1.0) -> fits.Header:
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = nx
    header["NAXIS2"] = ny
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CDELT1"] = cdelt
    header["CDELT2"] = cdelt
    header["CRPIX1"] = (float(nx) + 1.0) / 2.0
    header["CRPIX2"] = (float(ny) + 1.0) / 2.0
    header["CRVAL1"] = crval1
    header["CRVAL2"] = crval2
    return header


def test_build_padded_canvas_header_centers_model_fov() -> None:
    model_header = _make_header(nx=4, ny=3)
    canvas_header, pad_x, pad_y = build_padded_canvas_header(model_header, max_shift_arcsec=2.0)
    assert int(canvas_header["NAXIS1"]) == 4 + 2 * pad_x
    assert int(canvas_header["NAXIS2"]) == 3 + 2 * pad_y
    assert pad_x >= 2
    assert pad_y >= 2


def test_extract_observation_round_trip_at_zero_shift() -> None:
    model_header = _make_header(nx=5, ny=4)
    canvas_header, _pad_x, _pad_y = build_padded_canvas_header(model_header, max_shift_arcsec=3.0)
    source = np.arange(100, dtype=float).reshape(10, 10)
    source_header = _make_header(nx=10, ny=10, crval1=-5.0, crval2=-5.0)
    canvas = regrid_observation_to_target_fov(source, source_header, canvas_header)
    extracted, _sigma = extract_observation_to_model_fov(
        canvas,
        canvas_header,
        model_header,
        shift_x_arcsec=0.0,
        shift_y_arcsec=0.0,
    )
    assert extracted.shape == (4, 5)


def test_find_shift_recovers_known_offset() -> None:
    model_header = _make_header(nx=6, ny=6, cdelt=1.0)
    canvas_header, _pad_x, _pad_y = build_padded_canvas_header(model_header, max_shift_arcsec=5.0)
    yy, xx = np.mgrid[0:canvas_header["NAXIS2"], 0:canvas_header["NAXIS1"]]
    canvas = np.exp(-((xx - canvas_header["CRPIX1"]) ** 2 + (yy - canvas_header["CRPIX2"]) ** 2) / 8.0)
    modeled = extract_observation_to_model_fov(canvas, canvas_header, model_header, shift_x_arcsec=2.0, shift_y_arcsec=-1.0)[0]
    shifted = find_shift(canvas, canvas_header, model_header, modeled, max_shift_arcsec=5.0)
    assert shifted.valid
    assert abs(float(shifted.shift_x_arcsec) - 2.0) <= 1.0
    assert abs(float(shifted.shift_y_arcsec) + 1.0) <= 1.0
