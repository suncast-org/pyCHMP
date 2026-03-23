#!/usr/bin/env python
"""Test map_noise on real EOVSA FITS data."""

from pathlib import Path

import numpy as np
from astropy.io import fits

from pychmp import estimate_map_noise

# EOVSA FITS file path
fits_path = Path(
    "/Users/gelu/Library/CloudStorage/Dropbox/@Projects/@SUNCAST-ORG/"
    "test-data/eovsa_maps/eovsa.synoptic_daily.20201126T200000Z.f2.874GHz.tb.disk.fits"
)

print(f"Loading FITS file: {fits_path}")
print(f"File exists: {fits_path.exists()}")

if not fits_path.exists():
    print("ERROR: File not found!")
    exit(1)

# Open FITS file
with fits.open(fits_path) as hdul:
    print(f"\nFITS file structure:")
    hdul.info()

    # Get primary HDU
    hdu = hdul[0]
    data = hdu.data
    header = hdu.header

    print(f"\nData shape: {data.shape}")
    print(f"Data dtype: {data.dtype}")
    print(f"Data range: [{np.min(data):.2f}, {np.max(data):.2f}]")
    print(f"Data mean: {np.mean(data):.2f}")
    print(f"Data std: {np.std(data):.2f}")

    # Extract frequency from header
    if "CRVAL3" in header:
        freq_hz = header["CRVAL3"]
        if "CUNIT3" in header and header["CUNIT3"].strip() == "Hz":
            freq_ghz = freq_hz / 1e9
            print(f"Frequency from FITS: {freq_ghz:.3f} GHz")
    
    # Print relevant header keys
    print("\nRelevant FITS header keys:")
    for key in ["CRVAL3", "CUNIT3", "CTYPE3", "CRPIX3", "CDELT3"]:
        if key in header:
            print(f"  {key}: {header[key]}")

# Test histogram_clip method (no WCS required)
print("\n" + "=" * 70)
print("TEST: histogram_clip method (no WCS required)")
print("=" * 70)
result_hc = estimate_map_noise(data, method="histogram_clip")
print(f"Estimated sigma: {result_hc.sigma:.4f}")
print(f"Method: {result_hc.method_used}")
print(f"Pixels used: {result_hc.n_pixels_used}")
print(f"Mask fraction: {result_hc.mask_fraction:.4f}")
if result_hc.diagnostics:
    print(f"Background level: {result_hc.diagnostics.get('background_level'):.4f}")
    print(f"Percentile: {result_hc.diagnostics.get('percentile')}")
    print(f"Data min/max: {result_hc.diagnostics.get('data_min'):.2f} / {result_hc.diagnostics.get('data_max'):.2f}")

# Try offlimb_mad with WCS from FITS
print("\n" + "=" * 70)
print("TEST: offlimb_mad method (with WCS from FITS)")
print("=" * 70)

try:
    from astropy.wcs import WCS

    # Create WCS from FITS header
    wcs = WCS(header)
    print(f"WCS created successfully")
    print(f"WCS projection: {wcs.wcs.ctype}")

    result_om = estimate_map_noise(data, wcs=wcs, method="offlimb_mad")
    print(f"Estimated sigma: {result_om.sigma:.4f}")
    print(f"Method: {result_om.method_used}")
    print(f"Pixels used: {result_om.n_pixels_used}")
    print(f"Mask fraction: {result_om.mask_fraction:.4f}")
    if result_om.diagnostics:
        print(f"Solar radius (arcsec): {result_om.diagnostics.get('solar_radius_arcsec'):.1f}")
        print(f"Annulus thickness (arcsec): {result_om.diagnostics.get('annulus_thickness_arcsec'):.1f}")

except Exception as e:
    print(f"Error creating WCS or running offlimb_mad: {e}")

# Comparison
print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"histogram_clip sigma: {result_hc.sigma:.4f}")
try:
    print(f"offlimb_mad sigma:   {result_om.sigma:.4f}")
    print(f"Difference: {abs(result_hc.sigma - result_om.sigma):.4f}")
except:
    print("offlimb_mad: Not available")

print("\n✓ Testing complete!")
