#!/usr/bin/env python
"""Manual testing script for map_noise module."""

import numpy as np
import matplotlib.pyplot as plt

from pychmp import estimate_map_noise


def test_histogram_clip():
    """Test histogram_clip method on synthetic data."""
    print("=" * 70)
    print("TEST 1: Histogram Clip - Pure Noise")
    print("=" * 70)

    np.random.seed(42)
    true_sigma = 1.5
    noise = np.random.normal(0, true_sigma, size=(100, 100))

    result = estimate_map_noise(noise, method="histogram_clip")

    print(f"True sigma: {true_sigma}")
    print(f"Estimated sigma: {result.sigma:.4f}")
    print(f"Error: {abs(result.sigma - true_sigma):.4f}")
    print(f"Method: {result.method_used}")
    print(f"Pixels used: {result.n_pixels_used}/{noise.size}")
    print(f"Mask fraction: {result.mask_fraction:.3f}")
    if result.diagnostics:
        print(f"Background level: {result.diagnostics.get('background_level', 'N/A')}")
        print(f"Percentile used: {result.diagnostics.get('percentile', 'N/A')}")


def test_histogram_clip_with_signal():
    """Test histogram_clip on realistic signal + noise."""
    print("\n" + "=" * 70)
    print("TEST 2: Histogram Clip - Signal + Noise (Gaussian peak)")
    print("=" * 70)

    np.random.seed(42)
    true_sigma = 2.0

    # Create synthetic map: Gaussian peak + noise
    yy, xx = np.mgrid[0:100, 0:100]
    signal = 50.0 * np.exp(-((xx - 50) ** 2 + (yy - 50) ** 2) / 500.0)
    noise = np.random.normal(0, true_sigma, size=(100, 100))
    data = signal + noise

    result = estimate_map_noise(data, method="histogram_clip")

    print(f"True sigma: {true_sigma}")
    print(f"Estimated sigma: {result.sigma:.4f}")
    print(f"Signal peak: {signal.max():.2f}")
    print(f"Signal std: {signal.std():.4f}")
    print(f"Method: {result.method_used}")
    print(f"Pixels used: {result.n_pixels_used}/{data.size}")
    print(f"Mask fraction: {result.mask_fraction:.3f}")
    if result.diagnostics:
        print(f"Data min/max: {result.diagnostics.get('data_min', 'N/A'):.2f} / "
              f"{result.diagnostics.get('data_max', 'N/A'):.2f}")
        print(f"Data median: {result.diagnostics.get('data_median', 'N/A'):.2f}")


def test_histogram_clip_custom_params():
    """Test histogram_clip with custom parameters."""
    print("\n" + "=" * 70)
    print("TEST 3: Histogram Clip - Custom Parameters")
    print("=" * 70)

    np.random.seed(42)
    data = np.random.normal(5, 2.0, size=(100, 100))

    # Test with different percentiles
    for percentile in [25, 50, 75]:
        result = estimate_map_noise(
            data, method="histogram_clip", histogram_clip_percentile=percentile
        )
        print(f"  Percentile {percentile:2d}: sigma={result.sigma:.4f}, "
              f"n_pixels={result.n_pixels_used}")


def test_sigma_map_properties():
    """Test that sigma_map has correct properties."""
    print("\n" + "=" * 70)
    print("TEST 4: Sigma Map Properties")
    print("=" * 70)

    data = np.random.normal(0, 1.5, size=(50, 50))
    result = estimate_map_noise(data, method="histogram_clip")

    print(f"Input shape: {data.shape}")
    print(f"Sigma map shape: {result.sigma_map.shape}")
    print(f"Sigma map uniform: {np.allclose(result.sigma_map, result.sigma)}")
    print(f"Sigma scalar: {result.sigma:.6f}")
    print(f"Sigma map min/max: {result.sigma_map.min():.6f} / {result.sigma_map.max():.6f}")


def test_fallback_behavior():
    """Test fallback from offlimb_mad to histogram_clip."""
    print("\n" + "=" * 70)
    print("TEST 5: Fallback Behavior (offlimb_mad -> histogram_clip)")
    print("=" * 70)

    data = np.random.normal(0, 1.5, size=(100, 100))

    # Try offlimb_mad without WCS (should fall back)
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = estimate_map_noise(data, wcs=None, method="offlimb_mad")

        if w:
            print(f"Warning raised: {w[0].message}")
        print(f"Method used: {result.method_used}")
        print(f"Sigma: {result.sigma:.4f}")


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "=" * 70)
    print("TEST 6: Edge Cases")
    print("=" * 70)

    # Constant data
    print("Constant data (all 5.0):")
    data_const = np.full((50, 50), 5.0)
    result = estimate_map_noise(data_const, method="histogram_clip")
    print(f"  Sigma: {result.sigma:.6f}")

    # Small array
    print("\nSmall array (5x5):")
    data_small = np.random.normal(0, 1, size=(5, 5))
    result = estimate_map_noise(data_small, method="histogram_clip")
    print(f"  Sigma: {result.sigma:.4f}")

    # Negative values
    print("\nNegative values (mean=-10):")
    data_neg = np.random.normal(-10, 2.0, size=(100, 100))
    result = estimate_map_noise(data_neg, method="histogram_clip")
    print(f"  Sigma: {result.sigma:.4f}")
    print(f"  Data range: [{data_neg.min():.2f}, {data_neg.max():.2f}]")


def visualize_results():
    """Create a visualization of map_noise on sample data."""
    print("\n" + "=" * 70)
    print("TEST 7: Visualization")
    print("=" * 70)

    np.random.seed(42)
    true_sigma = 2.0

    # Create synthetic map
    yy, xx = np.mgrid[0:100, 0:100]
    signal = 50.0 * np.exp(-((xx - 50) ** 2 + (yy - 50) ** 2) / 500.0)
    noise = np.random.normal(0, true_sigma, size=(100, 100))
    data = signal + noise

    result = estimate_map_noise(data, method="histogram_clip")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Data
    im1 = axes[0].imshow(data, cmap="viridis")
    axes[0].set_title("Synthetic Map (Signal + Noise)")
    axes[0].set_xlabel("X pixel")
    axes[0].set_ylabel("Y pixel")
    plt.colorbar(im1, ax=axes[0], label="Intensity")

    # Plot 2: Sigma map
    im2 = axes[1].imshow(result.sigma_map, cmap="hot")
    axes[1].set_title(f"Noise Map (uniform, σ={result.sigma:.3f})")
    axes[1].set_xlabel("X pixel")
    axes[1].set_ylabel("Y pixel")
    plt.colorbar(im2, ax=axes[1], label="σ")

    fig.suptitle(
        f"Map Noise Estimation: {result.method_used}\n"
        f"True σ={true_sigma}, Est. σ={result.sigma:.4f}, "
        f"Error={abs(result.sigma - true_sigma):.4f}",
        fontsize=12,
    )

    plt.tight_layout()
    plt.savefig("map_noise_test.png", dpi=100, bbox_inches="tight")
    print(f"✓ Saved visualization to: map_noise_test.png")
    print(f"\nResults:")
    print(f"  True sigma: {true_sigma}")
    print(f"  Estimated sigma: {result.sigma:.4f}")
    print(f"  Error: {abs(result.sigma - true_sigma):.4f}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MANUAL TESTING OF map_noise.py")
    print("=" * 70 + "\n")

    test_histogram_clip()
    test_histogram_clip_with_signal()
    test_histogram_clip_custom_params()
    test_sigma_map_properties()
    test_fallback_behavior()
    test_edge_cases()

    try:
        visualize_results()
    except ImportError:
        print("\n(Skipping visualization - matplotlib not available)")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)
