"""Noise estimation utilities for full-disk observed maps.

This module provides robust methods for estimating signal noise from full-disk
solar maps, primarily using off-limb regions where the true signal is zero.

Returns None if input map is unsuitable for reliable noise estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class MapNoiseEstimate:
    """Result container for map noise estimation.

    Attributes:
        sigma: Scalar noise standard deviation estimate.
        sigma_map: Full-resolution noise map (same shape as input data).
        method_used: Name of estimation method ('offlimb_mad' or 'histogram_clip').
        n_pixels_used: Number of background pixels used in estimation.
        mask_fraction: Fraction of input pixels classified as background.
        diagnostics: Optional dict with per-method diagnostic data.
    """

    sigma: float
    sigma_map: np.ndarray
    method_used: str
    n_pixels_used: int
    mask_fraction: float
    diagnostics: dict | None = None


def estimate_map_noise(
    data: np.ndarray,
    wcs=None,
    method: Literal["offlimb_mad", "histogram_clip"] = "offlimb_mad",
    offlimb_annulus_deg: float = 15.0,
    histogram_clip_sigma: float = 3.0,
    histogram_clip_percentile: float = 50.0,
) -> MapNoiseEstimate | None:
    """Estimate noise in a full-disk observed map.

    Parameters:
        data: Map data array (any 2D shape).
        wcs: WCS object for the map (optional, required for offlimb_mad method).
               If None with method='offlimb_mad', falls back to histogram_clip.
        method: Estimation method ('offlimb_mad' or 'histogram_clip').
        offlimb_annulus_deg: Thickness of off-limb annulus in degrees
                            (used for offlimb_mad method).
        histogram_clip_sigma: Sigma-clipping threshold for histogram method.
        histogram_clip_percentile: Percentile for histogram-based background
                                   identification (default 50th = median).

    Returns:
        MapNoiseEstimate with sigma, sigma_map, and diagnostic metadata.
        None if the input map is not suitable for reliable estimation.

    Raises:
        ValueError: If data is not 2D, or if requested method cannot be applied.
    """
    data_arr = np.asarray(data, dtype=float)

    if data_arr.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data_arr.shape}")

    # Validate data quality before proceeding
    validation_result = _validate_map_data(data_arr)
    if validation_result['is_valid'] is False:
        import warnings
        warnings.warn(
            f"Map data quality check failed: {validation_result['message']}. "
            "Noise estimation unreliable; returning None.",
            UserWarning
        )
        return None

    # Try primary method; fall back if WCS unavailable
    if method == "offlimb_mad":
        if wcs is None:
            import warnings

            warnings.warn(
                "WCS required for offlimb_mad; falling back to histogram_clip",
                UserWarning,
            )
            method = "histogram_clip"
        else:
            try:
                return _estimate_offlimb_mad(
                    data_arr, wcs, offlimb_annulus_deg=offlimb_annulus_deg
                )
            except Exception as e:
                import warnings

                warnings.warn(
                    f"offlimb_mad failed ({e}); falling back to histogram_clip",
                    UserWarning,
                )
                method = "histogram_clip"

    if method == "histogram_clip":
        return _estimate_histogram_clip(
            data_arr,
            sigma=histogram_clip_sigma,
            percentile=histogram_clip_percentile,
        )

    raise ValueError(f"Unknown method: {method}")


def _validate_map_data(data: np.ndarray) -> dict:
    """Validate that map data is suitable for noise estimation.

    Checks for:
    - Excessive NaN pixels (threshold: 10%)
    - Sufficient data coverage (minimum: 1000 valid pixels)
    - Reasonable data value ranges (not constant or extremely corrupted)

    Returns:
        dict with keys: is_valid (bool), message (str)
    """
    # Check for NaN or masked pixels
    nan_fraction = np.isnan(data).sum() / data.size
    if nan_fraction > 0.1:  # More than 10% NaN
        return {
            'is_valid': False,
            'message': f'Data contains {nan_fraction*100:.1f}% NaN pixels (threshold: 10%)',
        }

    # Remove NaN for further analysis
    valid_data = data[~np.isnan(data)]

    if len(valid_data) < 1000:  # At least 1000 valid pixels
        return {
            'is_valid': False,
            'message': f'Only {len(valid_data)} valid pixels available (minimum: 1000)',
        }

    # Check for unreasonable data ranges (potential data corruption)
    data_range = np.ptp(valid_data)  # peak-to-peak
    data_std = np.std(valid_data)

    if data_std == 0:
        return {
            'is_valid': False,
            'message': 'Data has zero standard deviation (constant values)',
        }

    # Flag if range is extremely large relative to std (potential outliers/corruption)
    if data_range > 100 * data_std:
        return {
            'is_valid': False,
            'message': (
                f'Data range ({data_range:.2e}) is {data_range/data_std:.1f}x std dev. '
                'Potential data corruption or extreme outliers.'
            ),
        }

    # All checks passed
    return {
        'is_valid': True,
        'message': 'Data validation passed',
    }


def _estimate_offlimb_mad(
    data: np.ndarray,
    wcs,
    offlimb_annulus_deg: float = 15.0,
) -> MapNoiseEstimate:
    """Estimate noise using off-limb MAD (Median Absolute Deviation).

    Off-limb regions (beyond solar disk) should contain only noise, since the
    true solar brightness is zero there. We use the MAD-based robust sigma
    estimate on these pixels.

    Parameters:
        data: 2D map array.
        wcs: WCS object defining pixel-to-world transformation.
        offlimb_annulus_deg: Thickness of annulus beyond solar limb (degrees).

    Returns:
        MapNoiseEstimate with sigma from off-limb MAD.
    """
    try:
        from astropy.coordinates import SkyCoord
        from sunpy.map import Map
    except ImportError:
        raise ImportError(
            "SunPy and Astropy required for off-limb noise estimation. "
            "Install with: pip install sunpy astropy"
        )

    # Create dummy map to use SunPy's solar coordinate machinery
    # (we only need the reference_pixel and scale for distance calculation)
    dummy_map = Map(data, wcs)

    # Create pixel coordinate grids
    ny, nx = data.shape
    yy, xx = np.mgrid[0:ny, 0:nx]

    # Convert pixel coordinates to world coordinates
    # This is the expensive part, but necessary for accurate off-limb masking
    coords = wcs.pixel_to_world(xx.ravel(), yy.ravel())

    # Compute distance from solar center for each pixel
    # Use coordinate distance in the plane of the sky
    if hasattr(coords, "Tx") and hasattr(coords, "Ty"):
        # Helioprojective coordinates
        distances_arcsec = np.sqrt(coords.Tx.deg**2 + coords.Ty.deg**2) * 3600.0
    else:
        # Fallback: assume coordinates are in arcsec
        distances_arcsec = np.sqrt(
            coords.lon.deg**2 + coords.lat.deg**2
        ) * 3600.0

    distances_arcsec = distances_arcsec.reshape(data.shape)

    # Solar radius in arcsec (1 degree ~= 60 arcsec on average)
    # We use the instrument's plate scale from WCS if available
    solar_radius_arcsec = 900.0  # ~15 degrees, standard assumption

    # Off-limb annulus: from solar_radius to solar_radius + annulus_thickness
    annulus_thickness_arcsec = offlimb_annulus_deg * 60.0
    min_dist = solar_radius_arcsec
    max_dist = solar_radius_arcsec + annulus_thickness_arcsec

    offlimb_mask = (distances_arcsec >= min_dist) & (distances_arcsec <= max_dist)
    offlimb_pixels = data[offlimb_mask]

    if len(offlimb_pixels) < 10:
        raise ValueError(
            f"Too few off-limb pixels ({len(offlimb_pixels)}); "
            "cannot estimate noise reliably"
        )

    # Compute sigma from MAD (Median Absolute Deviation)
    # sigma_MAD = 1.4826 * MAD (for normally distributed data)
    median_val = np.median(offlimb_pixels)
    mad = np.median(np.abs(offlimb_pixels - median_val))
    sigma = 1.4826 * mad if mad > 0 else np.std(offlimb_pixels)

    # Create uniform sigma map
    sigma_map = np.full_like(data, sigma)

    n_pixels = len(offlimb_pixels)
    mask_fraction = n_pixels / data.size

    diagnostics = {
        "method": "offlimb_mad",
        "median_offlimb": float(median_val),
        "mad_offlimb": float(mad),
        "solar_radius_arcsec": float(solar_radius_arcsec),
        "annulus_thickness_arcsec": float(annulus_thickness_arcsec),
        "offlimb_mask": offlimb_mask.copy(),
    }

    return MapNoiseEstimate(
        sigma=float(sigma),
        sigma_map=sigma_map,
        method_used="offlimb_mad",
        n_pixels_used=n_pixels,
        mask_fraction=float(mask_fraction),
        diagnostics=diagnostics,
    )


def _estimate_histogram_clip(
    data: np.ndarray,
    sigma: float = 3.0,
    percentile: float = 50.0,
) -> MapNoiseEstimate:
    """Estimate noise using histogram-based percentile and sigma clipping.

    This method assumes the histogram mode corresponds to background/noise,
    and clips pixels above the specified percentile level. It is more robust
    to extended signal contamination than simple percentile-based thresholding.

    Parameters:
        data: 2D map array.
        sigma: Sigma-clipping threshold (number of sigma above percentile).
        percentile: Percentile for initial background estimate (default 50 = median).

    Returns:
        MapNoiseEstimate with sigma from clipped background.
    """
    if not (0 <= percentile <= 100):
        raise ValueError("percentile must be in [0, 100]")

    # Use percentile as initial background estimate
    background_level = np.percentile(data.ravel(), percentile)

    # Pixels "near" background: within sigma_threshold of background_level
    # (allows some noise fluctuation in the background region)
    clipped_pixels = data[data <= background_level + sigma * np.std(data)]

    if len(clipped_pixels) < 10:
        # Fallback: use all pixels below 75th percentile
        clipped_pixels = data[data <= np.percentile(data.ravel(), 75)]

    # Compute sigma from clipped background
    sigma_est = np.std(clipped_pixels)
    if sigma_est <= 0:
        sigma_est = np.median(np.abs(clipped_pixels - np.median(clipped_pixels)))

    # Create uniform sigma map
    sigma_map = np.full_like(data, sigma_est)

    n_pixels = len(clipped_pixels)
    mask_fraction = n_pixels / data.size

    diagnostics = {
        "method": "histogram_clip",
        "background_level": float(background_level),
        "percentile": float(percentile),
        "sigma_clip_threshold": float(sigma),
        "data_min": float(np.min(data)),
        "data_max": float(np.max(data)),
        "data_median": float(np.median(data)),
    }

    return MapNoiseEstimate(
        sigma=float(sigma_est),
        sigma_map=sigma_map,
        method_used="histogram_clip",
        n_pixels_used=n_pixels,
        mask_fraction=float(mask_fraction),
        diagnostics=diagnostics,
    )
