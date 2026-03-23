#!/usr/bin/env python
"""Fit Q0 to real observational maps (e.g., EOVSA).

This example demonstrates:
1. Loading observed solar maps from FITS files
2. Estimating noise using map_noise utilities
3. Fitting Q0 using the gxrender adapter
4. Saving artifacts and visualizations

Usage:
    python fit_q0_obs_map.py /path/to/eovsa_map.fits
    python fit_q0_obs_map.py /path/to/eovsa_map.fits --gxrender-path /path/to/gxrender_models
    python fit_q0_obs_map.py /path/to/eovsa_map.fits --artifacts-dir /tmp/artifacts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from astropy.io import fits

from pychmp import GXRenderMWAdapter, estimate_map_noise, fit_q0_to_observation


def load_eovsa_map(fits_path: Path) -> tuple[np.ndarray, fits.Header, float]:
    """Load EOVSA FITS map and extract data, header, frequency.
    
    Returns:
        (data, header, frequency_ghz)
    """
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    # Extract frequency from header
    freq_ghz = None
    if "CRVAL3" in header and "CUNIT3" in header:
        if header["CUNIT3"].strip() == "Hz":
            freq_ghz = header["CRVAL3"] / 1e9

    if freq_ghz is None:
        raise ValueError("Could not extract frequency from FITS header")

    return np.asarray(data, dtype=float), header, freq_ghz


def create_gxrender_adapter(
    frequency_ghz: float,
    gxrender_base: Path | None = None,
) -> GXRenderMWAdapter | None:
    """Create GXRenderMWAdapter for the given frequency.
    
    If gxrender_base is not provided, uses default model paths.
    Returns None if models are not available.
    """
    # Default model paths (adjust based on your setup)
    if gxrender_base is None:
        gxrender_base = Path(
            "/Users/gelu/ssw/packages/gx_simulator"
        )

    # Infer model path from frequency
    # This is a simplified approach; real implementation would have frequency-specific models
    model_path = gxrender_base / "beams" / "model.h5"

    if not model_path.exists():
        # Try alternative path
        model_path = gxrender_base / "models" / f"freq_{frequency_ghz:.1f}ghz.h5"

    if not model_path.exists():
        return None

    try:
        return GXRenderMWAdapter(model_path=model_path, frequency_ghz=frequency_ghz)
    except (ImportError, Exception):
        return None


def save_q0_artifact(
    h5_path: Path,
    *,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    frequency_ghz: float,
    q0_fitted: float,
    metrics_dict: dict[str, float],
    noise_diagnostics: dict[str, Any] | None = None,
) -> None:
    """Save Q0 fitting results to H5 file."""
    with h5py.File(h5_path, "w") as f:
        # Data
        f.create_dataset("observed", data=observed, compression="gzip")
        f.create_dataset("sigma_map", data=sigma_map, compression="gzip")

        # Results
        f.attrs["frequency_ghz"] = float(frequency_ghz)
        f.attrs["q0_fitted"] = float(q0_fitted)

        # Metrics
        metrics_grp = f.create_group("metrics")
        for name, value in metrics_dict.items():
            metrics_grp.attrs[name] = float(value)

        # Noise estimation diagnostics
        if noise_diagnostics:
            diag_grp = f.create_group("noise_diagnostics")
            for key, value in noise_diagnostics.items():
                if isinstance(value, (str, int, float)):
                    diag_grp.attrs[key] = value
                elif isinstance(value, bool):
                    diag_grp.attrs[key] = int(value)  # HDF5 doesn't support bool


def main():
    parser = argparse.ArgumentParser(
        description="Fit Q0 to real observational maps (EOVSA).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s /path/to/eovsa_map.fits
  %(prog)s /path/to/eovsa_map.fits --q0-min 0.5 --q0-max 2.5
  %(prog)s /path/to/eovsa_map.fits --artifacts-dir /tmp/q0_artifacts
""",
    )

    parser.add_argument("fits_file", type=Path, help="Path to FITS file (EOVSA map)")
    parser.add_argument(
        "--q0-min", type=float, default=0.01, help="Minimum Q0 to search (default: 0.01)"
    )
    parser.add_argument(
        "--q0-max", type=float, default=2.5, help="Maximum Q0 to search (default: 2.5)"
    )
    parser.add_argument(
        "--target-metric",
        choices=["chi2", "rho2", "eta2"],
        default="chi2",
        help="Target metric for optimization (default: chi2)",
    )
    parser.add_argument(
        "--gxrender-path",
        type=Path,
        default=None,
        help="Path to gxrender models directory",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Directory to save H5/PNG artifacts",
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Skip artifact saving",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode (skip gxrender fitting, show noise estimation only)",
    )

    args = parser.parse_args()

    # Validate input file
    if not args.fits_file.exists():
        print(f"ERROR: FITS file not found: {args.fits_file}")
        exit(1)

    print(f"\n{'=' * 70}")
    print("FITTING Q0 TO OBSERVATIONAL MAP")
    print(f"{'=' * 70}\n")

    # Load map
    print(f"Loading FITS file: {args.fits_file.name}")
    observed, header, freq_ghz = load_eovsa_map(args.fits_file)
    print(f"  Shape: {observed.shape}")
    print(f"  Frequency: {freq_ghz:.3f} GHz")
    print(f"  Data range: [{observed.min():.2f}, {observed.max():.2f}]")

    # Estimate noise
    print(f"\nEstimating noise from map...")
    noise_result = estimate_map_noise(observed, method="histogram_clip")

    if noise_result is None:
        print("  ⚠️  Noise estimation failed (map quality issues)")
        print(f"  Falling back to fixed sigma = {int(observed.std())}K")
        sigma_map = np.full_like(observed, observed.std())
        noise_diagnostics = None
    else:
        print(f"  Estimated sigma: {noise_result.sigma:.2f} K")
        print(f"  Background fraction: {noise_result.mask_fraction:.1%}")
        sigma_map = noise_result.sigma_map
        noise_diagnostics = noise_result.diagnostics

    # Create gxrender adapter
    print(f"\nInitializing gxrender adapter for {freq_ghz:.3f} GHz...")
    adapter = None
    
    if not args.demo:
        adapter = create_gxrender_adapter(freq_ghz, args.gxrender_path)
        if adapter is None:
            print(f"  ⚠️  Could not load gxrender model for {freq_ghz:.3f} GHz")
            if args.gxrender_path:
                print(f"     Checked: {args.gxrender_path}")
            else:
                print(f"     Default path: /Users/gelu/ssw/packages/gx_simulator")
            print(f"\n  To enable fitting:")
            print(f"    1. Set 'gxrender_base' in create_gxrender_adapter()")
            print(f"    2. Or provide --gxrender-path /path/to/models")
            print(f"    3. Or run with --demo to see noise estimation only\n")

    # Fitting (optional, requires gxrender)
    result = None
    if adapter is None and not args.demo:
        print(f"Skipping Q0 fitting (models unavailable)")
    elif adapter is not None:
        print(f"  ✓ Model loaded successfully")

        # Fit Q0
        print(f"\nFitting Q0 using {args.target_metric} metric...")
        print(f"  Q0 range: [{args.q0_min:.4f}, {args.q0_max:.4f}]")

        try:
            result = fit_q0_to_observation(
                renderer=adapter,
                observed=observed,
                sigma=sigma_map,
                q0_min=args.q0_min,
                q0_max=args.q0_max,
                target_metric=args.target_metric,
                adaptive_bracketing=True,  # Use adaptive bracketing for efficiency
            )

            print(f"  ✓ Fitting converged")
            print(f"  Fitted Q0: {result.q0:.6f}")
            print(f"  Objective value ({result.target_metric}): {result.objective_value:.6e}")
            print(f"  Metrics: chi2={result.metrics.chi2:.6e}, "
                  f"rho2={result.metrics.rho2:.6e}, eta2={result.metrics.eta2:.6e}")

        except Exception as e:
            print(f"  ✗ Fitting failed: {e}")
            # Continue anyway to show noise estimation

    # Save artifacts
    if not args.no_artifacts and result is not None:
        if args.artifacts_dir is None:
            args.artifacts_dir = Path(".").resolve() / "q0_observation_artifacts"

        args.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Create filename from FITS file and frequency
        stem = args.fits_file.stem
        h5_name = f"{stem}_q0_fitted_{result.q0:.6f}.h5"
        h5_path = args.artifacts_dir / h5_name

        print(f"\nSaving artifacts...")
        try:
            save_q0_artifact(
                h5_path,
                observed=observed,
                sigma_map=sigma_map,
                frequency_ghz=freq_ghz,
                q0_fitted=result.q0,
                metrics_dict={
                    "chi2": result.metrics.chi2,
                    "rho2": result.metrics.rho2,
                    "eta2": result.metrics.eta2,
                },
                noise_diagnostics=noise_diagnostics,
            )
            print(f"  ✓ Saved to: {h5_path}")
        except Exception as e:
            print(f"  ✗ Failed to save artifacts: {e}")

    print(f"\n{'=' * 70}")
    print("FITTING COMPLETE")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
