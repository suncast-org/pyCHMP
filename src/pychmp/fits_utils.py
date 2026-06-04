"""Helpers for reading observational FITS products."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


_FREQUENCY_UNIT_TO_GHZ = {
    "hz": 1.0e-9,
    "s-1": 1.0e-9,
    "1/s": 1.0e-9,
    "khz": 1.0e-6,
    "mhz": 1.0e-3,
    "ghz": 1.0,
}

_SCALAR_FREQUENCY_KEYS = (
    "FREQ",
    "FREQUENCY",
    "OBSFREQ",
    "OBS_FREQ",
    "RESTFRQ",
    "RESTFREQ",
    "REST_FREQ",
    "SKYFREQ",
    "SKY_FREQ",
)


def _optional_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _frequency_unit_scale_to_ghz(unit: Any) -> float | None:
    text = str(unit or "").strip().lower()
    text = text.replace(" ", "")
    return _FREQUENCY_UNIT_TO_GHZ.get(text)


def _header_frequency_value_to_ghz(value: Any, *, unit: Any | None) -> float | None:
    numeric = _optional_float(value)
    if numeric is None:
        return None
    scale = _frequency_unit_scale_to_ghz(unit)
    if scale is not None:
        return numeric * scale

    # FITS radio convention keys such as RESTFRQ are commonly stored in Hz
    # even when the unit card is absent.
    if abs(numeric) >= 1.0e6:
        return numeric * 1.0e-9

    # If an unqualified radio frequency is already a small floating-point
    # value, treat it as GHz rather than rejecting useful metadata.
    if 0.0 < abs(numeric) < 1.0e6:
        return numeric
    return None


def _axis_is_frequency(header: fits.Header, axis: int) -> bool:
    ctype = str(header.get(f"CTYPE{axis}", "")).strip().upper()
    unit = header.get(f"CUNIT{axis}")
    return ctype.startswith("FREQ") or _frequency_unit_scale_to_ghz(unit) is not None


def _axis_reference_frequency_ghz(header: fits.Header, axis: int) -> float | None:
    if f"CRVAL{axis}" not in header:
        return None
    if not _axis_is_frequency(header, axis):
        return None
    return _header_frequency_value_to_ghz(header[f"CRVAL{axis}"], unit=header.get(f"CUNIT{axis}"))


def extract_frequency_ghz(header: fits.Header) -> float:
    """Extract an observing frequency in GHz from common radio FITS metadata.

    Supports the existing EOVSA-style ``CRVAL3``/``CUNIT3=Hz`` convention,
    frequency WCS axes on any FITS axis, and common scalar radio keywords such
    as ``RESTFRQ``, ``RESTFREQ``, ``OBSFREQ``, and ``FREQ``.
    """
    for axis in range(1, int(header.get("NAXIS", 0) or 0) + 1):
        frequency = _axis_reference_frequency_ghz(header, axis)
        if frequency is not None:
            return float(frequency)

    # Some valid headers omit NAXIS in test or metadata-only contexts. Keep
    # checking the conventional first four WCS axes so legacy EOVSA-like maps
    # without NAXIS still work.
    for axis in range(1, 5):
        frequency = _axis_reference_frequency_ghz(header, axis)
        if frequency is not None:
            return float(frequency)

    for key in _SCALAR_FREQUENCY_KEYS:
        if key not in header:
            continue
        unit = header.get(f"{key}U") or header.get(f"{key}_UNIT") or header.get("FREQUNIT")
        frequency = _header_frequency_value_to_ghz(header[key], unit=unit)
        if frequency is not None:
            return float(frequency)

    raise ValueError("Could not extract frequency from FITS header")
