"""Helpers for reading observational FITS products."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits


def _as_2d_image(data: object) -> np.ndarray | None:
    """Return a float 2D array when the HDU stores an image, else ``None``."""
    if data is None:
        return None
    arr = np.array(data, dtype=float, copy=True)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        return None
    return arr


def load_2d_fits_image(fits_path: Path) -> tuple[np.ndarray, fits.Header, str]:
    """Load the first FITS HDU whose data can be interpreted as a 2D image."""
    with fits.open(fits_path) as hdul:
        for hdu in hdul:
            data_arr = _as_2d_image(hdu.data)
            if data_arr is not None:
                return data_arr, hdu.header.copy(), hdu.name
    raise ValueError(
        f"Could not find a 2D image HDU in FITS file: {fits_path}"
    )


def extract_frequency_ghz(header: fits.Header) -> float:
    """Extract observing frequency in GHz from a FITS header."""
    if "CRVAL3" in header and "CUNIT3" in header:
        if str(header["CUNIT3"]).strip() == "Hz":
            return float(header["CRVAL3"]) / 1e9
    raise ValueError("Could not extract frequency from FITS header")
