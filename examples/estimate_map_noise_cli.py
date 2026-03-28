#!/usr/bin/env python
"""CLI tool to estimate noise from observational maps using different methods.

Usage:
    python estimate_map_noise_cli.py /path/to/map.fits
    python estimate_map_noise_cli.py /path/to/map.fits --all-methods
    python estimate_map_noise_cli.py /path/to/map.fits --method histogram_clip
"""

import argparse
from pathlib import Path

import numpy as np
from astropy.wcs import WCS

from pychmp import estimate_map_noise, extract_frequency_ghz, load_2d_fits_image


def load_fits_map(fits_path: Path) -> tuple:
    """Load FITS map and extract data, header, frequency."""
    data, header, hdu_name = load_2d_fits_image(fits_path)
    print(f"Loaded image HDU: {hdu_name}")
    try:
        freq_ghz = extract_frequency_ghz(header)
    except ValueError:
        freq_ghz = None
    return data, header, freq_ghz


def print_data_stats(data: np.ndarray, freq_ghz: float | None = None):
    """Print summary statistics of the map data."""
    print("\n" + "=" * 70)
    print("MAP DATA STATISTICS")
    print("=" * 70)
    print(f"Shape:           {data.shape}")
    print(f"Data type:       {data.dtype}")
    print(f"Min value:       {np.min(data):15.2f}")
    print(f"Max value:       {np.max(data):15.2f}")
    print(f"Mean:            {np.mean(data):15.2f}")
    print(f"Median:          {np.median(data):15.2f}")
    print(f"Std dev:         {np.std(data):15.2f}")
    print(f"Total pixels:    {data.size:15,d}")
    if freq_ghz is not None:
        print(f"Frequency:       {freq_ghz:15.3f} GHz")


def test_histogram_clip(data: np.ndarray) -> dict:
    """Test histogram_clip method."""
    print("\n" + "=" * 70)
    print("METHOD: histogram_clip (percentile + sigma-clipping)")
    print("=" * 70)

    result = estimate_map_noise(data, method="histogram_clip")

    print(f"Estimated sigma:     {result.sigma:15.4f}")
    print(f"Pixels in background: {result.n_pixels_used:15,d}")
    print(f"Background fraction: {result.mask_fraction:15.4f}")

    if result.diagnostics:
        print(f"\nDiagnostics:")
        print(f"  Background level:  {result.diagnostics.get('background_level', 'N/A'):15.4f}")
        print(f"  Percentile:        {result.diagnostics.get('percentile', 'N/A'):15.1f}")
        print(f"  Sigma-clip value:  {result.diagnostics.get('sigma_clip_threshold', 'N/A'):15.1f}")

    return {"method": "histogram_clip", "sigma": result.sigma, "result": result}


def test_offlimb_mad(data: np.ndarray, header: dict) -> dict | None:
    """Test offlimb_mad method with WCS from FITS header."""
    print("\n" + "=" * 70)
    print("METHOD: offlimb_mad (off-limb Median Absolute Deviation)")
    print("=" * 70)

    try:
        wcs = WCS(header)
        print("WCS loaded successfully from FITS header")
        print(f"WCS projection: {wcs.wcs.ctype}")

        result = estimate_map_noise(data, wcs=wcs, method="offlimb_mad")

        print(f"Estimated sigma:     {result.sigma:15.4f}")
        print(f"Pixels in annulus:   {result.n_pixels_used:15,d}")
        print(f"Annulus fraction:    {result.mask_fraction:15.4f}")

        if result.diagnostics:
            print(f"\nDiagnostics:")
            print(f"  Solar radius:      {result.diagnostics.get('solar_radius_arcsec', 'N/A'):15.1f} arcsec")
            print(f"  Annulus thickness: {result.diagnostics.get('annulus_thickness_arcsec', 'N/A'):15.1f} arcsec")
            print(f"  Off-limb median:   {result.diagnostics.get('median_offlimb', 'N/A'):15.4f}")
            print(f"  Off-limb MAD:      {result.diagnostics.get('mad_offlimb', 'N/A'):15.4f}")

        return {"method": "offlimb_mad", "sigma": result.sigma, "result": result}

    except Exception as e:
        print(f"⚠️  Could not use offlimb_mad: {e}")
        print("   (WCS may not be properly formatted or SunPy may not be installed)")
        return None


def compare_methods(results: list[dict]):
    """Compare results from different methods."""
    if len(results) < 2:
        return

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    for i, r in enumerate(results):
        print(f"{i + 1}. {r['method']:20s}: σ = {r['sigma']:12.4f}")

    # Calculate percent difference
    if len(results) == 2:
        diff_percent = abs(results[0]["sigma"] - results[1]["sigma"]) / results[0]["sigma"] * 100
        print(f"\nDifference: {diff_percent:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate noise in observational maps using different methods.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s /path/to/eovsa_map.fits
  %(prog)s /path/to/eovsa_map.fits --all-methods
  %(prog)s /path/to/eovsa_map.fits --method histogram_clip
  %(prog)s /path/to/eovsa_map.fits --method offlimb_mad
""",
    )

    parser.add_argument(
        "fits_file",
        type=Path,
        help="Path to FITS file containing the observational map",
    )

    parser.add_argument(
        "--method",
        choices=["histogram_clip", "offlimb_mad", "all"],
        default="all",
        help="Which noise estimation method to use (default: all)",
    )

    parser.add_argument(
        "--all-methods",
        action="store_const",
        const="all",
        dest="method",
        help="Run all available methods (same as --method all)",
    )

    args = parser.parse_args()

    # Validate file
    if not args.fits_file.exists():
        print(f"ERROR: File not found: {args.fits_file}")
        exit(1)

    if not args.fits_file.suffix.lower() in [".fits", ".fit"]:
        print(f"WARNING: File does not have .fits extension: {args.fits_file}")

    # Load data
    print(f"\n{'=' * 70}")
    print("LOADING FITS FILE")
    print(f"{'=' * 70}")
    print(f"File: {args.fits_file}")

    data, header, freq_ghz = load_fits_map(args.fits_file)

    # Print statistics
    print_data_stats(data, freq_ghz)

    # Run selected methods
    results = []

    if args.method in ["histogram_clip", "all"]:
        results.append(test_histogram_clip(data))

    if args.method in ["offlimb_mad", "all"]:
        result = test_offlimb_mad(data, header)
        if result:
            results.append(result)

    # Compare if multiple methods ran
    if len(results) > 1:
        compare_methods(results)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if results:
        best_result = results[0]  # First method
        print(f"✓ Recommended sigma for fitting: {best_result['sigma']:.4f}")
        print(f"  (from {best_result['method']} method)")
    else:
        print("✗ No noise estimation methods succeeded")
        exit(1)


if __name__ == "__main__":
    main()
