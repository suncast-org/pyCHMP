"""Convert EUV/UV observational maps to gxrender-compatible rate units.

Modeled EUV maps from pyGXrender are documented as ``DN s^-1 pix^-1``. Reference
observations from FITS or embedded pyAMPP refmaps are often exposure-integrated
``DN`` (AIA ``PIXLUNIT`` or EUVI ``BUNIT``). This module normalizes loaded
observation payloads before regridding and metric evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

MODELED_EUV_BUNIT = "DN s^-1 pix^-1"


def _normalized_unit_token(header: fits.Header) -> str:
    bunit = str(header.get("BUNIT", "")).strip()
    if bunit:
        return bunit.upper().replace(" ", "")
    pixlunit = str(header.get("PIXLUNIT", "")).strip()
    if pixlunit:
        return pixlunit.upper().replace(" ", "")
    return ""


def euv_observation_declares_dn_rate(header: fits.Header) -> bool:
    token = _normalized_unit_token(header)
    if not token:
        return False
    return "S^-1" in token or "S-1" in token or "/S" in token


def euv_observation_appears_integrated_dn(header: fits.Header) -> bool:
    if euv_observation_declares_dn_rate(header):
        return False
    token = _normalized_unit_token(header)
    if not token:
        return True
    if token == "DN" or token.endswith("/DN"):
        return True
    return token.endswith("DN") and "S" not in token


def resolve_euv_exposure_seconds(
    header: fits.Header,
    *,
    source_path: Path | None = None,
) -> float | None:
    try:
        exptime = float(header.get("EXPTIME", 0.0) or 0.0)
        if np.isfinite(exptime) and exptime > 0.0:
            return exptime
    except (TypeError, ValueError):
        pass

    if source_path is None:
        return None
    resolved = Path(source_path).expanduser()
    if not resolved.is_file():
        return None
    try:
        with fits.open(resolved) as hdul:
            for hdu in hdul:
                if "EXPTIME" not in hdu.header:
                    continue
                exptime = float(hdu.header["EXPTIME"])
                if np.isfinite(exptime) and exptime > 0.0:
                    return exptime
    except (OSError, TypeError, ValueError):
        return None
    return None


def convert_euv_observation_to_rate(
    data: np.ndarray,
    header: fits.Header,
    *,
    source_path: Path | None = None,
) -> tuple[np.ndarray, fits.Header, dict[str, Any]]:
    """Return observation data/header in ``DN s^-1 pix^-1`` when conversion applies."""
    out_header = header.copy()
    diagnostics: dict[str, Any] = {
        "euv_target_bunit": MODELED_EUV_BUNIT,
        "euv_source_unit_token": _normalized_unit_token(header) or None,
    }

    if euv_observation_declares_dn_rate(out_header):
        if not str(out_header.get("BUNIT", "")).strip():
            out_header["BUNIT"] = MODELED_EUV_BUNIT
        diagnostics["euv_unit_conversion"] = "already_rate"
        return np.asarray(data, dtype=float), out_header, diagnostics

    if not euv_observation_appears_integrated_dn(out_header):
        diagnostics["euv_unit_conversion"] = "skipped_unknown_units"
        return np.asarray(data, dtype=float), out_header, diagnostics

    exptime = resolve_euv_exposure_seconds(out_header, source_path=source_path)
    if exptime is None:
        diagnostics["euv_unit_conversion"] = "skipped_missing_exptime"
        return np.asarray(data, dtype=float), out_header, diagnostics

    rate = np.asarray(data, dtype=float) / float(exptime)
    out_header["BUNIT"] = MODELED_EUV_BUNIT
    if "PIXLUNIT" in out_header:
        out_header["PIXLUNIT"] = MODELED_EUV_BUNIT
    diagnostics["euv_unit_conversion"] = "integrated_dn_to_rate"
    diagnostics["euv_exposure_seconds"] = float(exptime)
    if source_path is not None:
        diagnostics["euv_exposure_source_path"] = str(Path(source_path).expanduser().resolve())
    return rate, out_header, diagnostics


def maybe_convert_euv_domain_observation(
    data: np.ndarray,
    header: fits.Header,
    *,
    domain: str,
    source_path: Path | None = None,
) -> tuple[np.ndarray, fits.Header, dict[str, Any]]:
    normalized_domain = str(domain or "").strip().lower()
    if normalized_domain not in {"euv", "uv"}:
        return np.asarray(data, dtype=float), header.copy(), {}
    return convert_euv_observation_to_rate(data, header, source_path=source_path)
