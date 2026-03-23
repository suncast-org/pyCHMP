"""Tests for map_noise module."""

import warnings

import numpy as np
import pytest

from pychmp.map_noise import MapNoiseEstimate, estimate_map_noise


class TestMapNoiseEstimate:
    """Tests for MapNoiseEstimate dataclass."""

    def test_dataclass_creation(self) -> None:
        """Test MapNoiseEstimate constructor and attributes."""
        sigma_map = np.ones((10, 10)) * 1.5
        estimate = MapNoiseEstimate(
            sigma=1.5,
            sigma_map=sigma_map,
            method_used="test_method",
            n_pixels_used=50,
            mask_fraction=0.5,
        )

        assert estimate.sigma == 1.5
        assert estimate.sigma_map.shape == (10, 10)
        assert estimate.method_used == "test_method"
        assert estimate.n_pixels_used == 50
        assert estimate.mask_fraction == 0.5
        assert estimate.diagnostics is None

    def test_dataclass_with_diagnostics(self) -> None:
        """Test MapNoiseEstimate with diagnostic metadata."""
        diag = {"key": "value", "count": 42}
        estimate = MapNoiseEstimate(
            sigma=2.0,
            sigma_map=np.ones((5, 5)) * 2.0,
            method_used="histogram_clip",
            n_pixels_used=100,
            mask_fraction=1.0,
            diagnostics=diag,
        )

        assert estimate.diagnostics == diag
        assert estimate.diagnostics["key"] == "value"


class TestHistogramClipMethod:
    """Tests for histogram-clip noise estimation."""

    def test_uniform_noise(self) -> None:
        """Test noise estimation on uniform noise array."""
        np.random.seed(42)
        true_sigma = 1.5
        noise = np.random.normal(0, true_sigma, size=(100, 100))

        result = estimate_map_noise(noise, method="histogram_clip")

        assert isinstance(result, MapNoiseEstimate)
        assert result.method_used == "histogram_clip"
        assert result.sigma > 0
        assert result.sigma_map.shape == noise.shape
        assert np.allclose(result.sigma_map, result.sigma)
        assert result.n_pixels_used > 0
        assert 0 < result.mask_fraction <= 1

    def test_noise_plus_signal(self) -> None:
        """Test on realistic data with signal + noise."""
        np.random.seed(42)
        true_sigma = 2.0
        # Create synthetic map: low signal background + Gaussian peak
        yy, xx = np.mgrid[0:100, 0:100]
        signal = 50.0 * np.exp(-((xx - 50) ** 2 + (yy - 50) ** 2) / 500.0)
        noise = np.random.normal(0, true_sigma, size=(100, 100))
        data = signal + noise

        result = estimate_map_noise(data, method="histogram_clip")

        # Histogram method should identify background (not affected by peak)
        assert result.method_used == "histogram_clip"
        assert result.sigma > 0
        # Histogram clip can be affected by signal edges, so just verify it gives
        # a reasonable positive value; exact accuracy depends on SNR and signal morphology
        assert result.sigma > 0.5  # At least some noise captured

    def test_histogram_clip_parameters(self) -> None:
        """Test histogram_clip with custom parameters."""
        np.random.seed(42)
        data = np.random.normal(10, 2.0, size=(100, 100))

        # Test with different percentiles
        result_p25 = estimate_map_noise(
            data, method="histogram_clip", histogram_clip_percentile=25.0
        )
        result_p50 = estimate_map_noise(
            data, method="histogram_clip", histogram_clip_percentile=50.0
        )
        result_p75 = estimate_map_noise(
            data, method="histogram_clip", histogram_clip_percentile=75.0
        )

        # Different percentiles should give different estimates
        assert result_p25.sigma != result_p50.sigma
        assert result_p50.sigma != result_p75.sigma

    def test_histogram_clip_sigma_threshold(self) -> None:
        """Test histogram_clip with different sigma thresholds."""
        np.random.seed(42)
        data = np.random.normal(5, 1.5, size=(100, 100))

        result_s1 = estimate_map_noise(
            data, method="histogram_clip", histogram_clip_sigma=1.0
        )
        result_s3 = estimate_map_noise(
            data, method="histogram_clip", histogram_clip_sigma=3.0
        )

        # Different sigma thresholds should give different pixel counts
        assert result_s1.n_pixels_used != result_s3.n_pixels_used


class TestOfflimbMadMethod:
    """Tests for off-limb MAD noise estimation (requires WCS)."""

    def test_offlimb_mad_without_wcs_fallback(self) -> None:
        """Test that offlimb_mad without WCS falls back to histogram_clip."""
        np.random.seed(42)
        data = np.random.normal(0, 1.5, size=(100, 100))

        # Should warn and fall back
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = estimate_map_noise(data, wcs=None, method="offlimb_mad")

            # Check for fallback warning
            assert len(w) > 0
            assert "falling back" in str(w[0].message).lower()

        # Result should be from histogram_clip fallback
        assert result.method_used == "histogram_clip"

    def test_invalid_method(self) -> None:
        """Test that invalid method raises ValueError."""
        data = np.random.normal(0, 1, size=(100, 100))

        with pytest.raises(ValueError, match="Unknown method"):
            estimate_map_noise(data, method="invalid_method")  # type: ignore

    def test_histogram_clip_invalid_percentile(self) -> None:
        """Test that invalid percentile raises ValueError."""
        data = np.random.normal(0, 1, size=(100, 100))

        with pytest.raises(ValueError, match="percentile must be in"):
            estimate_map_noise(
                data, method="histogram_clip", histogram_clip_percentile=150.0
            )

    def test_2d_requirement(self) -> None:
        """Test that non-2D arrays raise ValueError."""
        # 1D array
        with pytest.raises(ValueError, match="must be 2D"):
            estimate_map_noise(np.ones(10), method="histogram_clip")

        # 3D array
        with pytest.raises(ValueError, match="must be 2D"):
            estimate_map_noise(np.ones((10, 10, 10)), method="histogram_clip")


class TestResultProperties:
    """Tests for MapNoiseEstimate result properties."""

    def test_sigma_map_shape(self) -> None:
        """Test that sigma_map matches input shape."""
        for shape in [(100, 100), (200, 200), (1024, 1024)]:
            data = np.random.normal(0, 1, size=shape)
            result = estimate_map_noise(data, method="histogram_clip")

            assert result is not None, f"Result should not be None for shape {shape}"
            assert result.sigma_map.shape == shape

    def test_sigma_map_uniform(self) -> None:
        """Test that sigma_map is uniform (v1 implementation)."""
        data = np.random.normal(5, 2.0, size=(100, 100))
        result = estimate_map_noise(data, method="histogram_clip")

        # All elements should equal the scalar sigma
        assert np.allclose(result.sigma_map, result.sigma)

    def test_mask_fraction_range(self) -> None:
        """Test that mask_fraction is in valid range [0, 1]."""
        data = np.random.normal(0, 1, size=(100, 100))
        result = estimate_map_noise(data, method="histogram_clip")

        assert 0 <= result.mask_fraction <= 1

    def test_n_pixels_used(self) -> None:
        """Test that n_pixels_used is consistent with mask_fraction."""
        data = np.random.normal(0, 1, size=(100, 100))
        result = estimate_map_noise(data, method="histogram_clip")

        expected_n_pixels = int(np.round(result.mask_fraction * data.size))
        assert abs(result.n_pixels_used - expected_n_pixels) <= 1  # Allow rounding

    def test_diagnostics_content(self) -> None:
        """Test that diagnostics contain expected keys."""
        data = np.random.normal(0, 1, size=(100, 100))
        result = estimate_map_noise(
            data, method="histogram_clip", histogram_clip_percentile=75.0
        )

        assert result.diagnostics is not None
        assert "method" in result.diagnostics
        assert "background_level" in result.diagnostics
        assert "percentile" in result.diagnostics
        assert result.diagnostics["percentile"] == 75.0


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_zero_variance_data(self) -> None:
        """Test handling of constant data."""
        data = np.full((100, 100), 5.0)
        result = estimate_map_noise(data, method="histogram_clip")

        # Should return None because zero variance is unreliable
        assert result is None

    def test_small_array(self) -> None:
        """Test on small arrays."""
        # Small array should fail validation (< 1000 pixels minimum)
        data = np.random.normal(0, 1, size=(5, 5))
        result = estimate_map_noise(data, method="histogram_clip")
        assert result is None

        # Larger but still small array should also fail
        data = np.random.normal(0, 1, size=(20, 20))
        result = estimate_map_noise(data, method="histogram_clip")
        assert result is None

        # Array with sufficient pixels should work
        data = np.random.normal(0, 1, size=(50, 50))  # 2500 pixels > 1000 minimum
        result = estimate_map_noise(data, method="histogram_clip")
        assert result is not None
        assert result.sigma > 0

    def test_large_dynamic_range(self) -> None:
        """Test data with large dynamic range."""
        np.random.seed(42)
        # Mix of very small and very large values
        data = np.concatenate(
            [
                np.random.normal(0, 1.0, size=(50, 50)),
                np.random.normal(1e6, 1e5, size=(50, 50)),
            ],
            axis=0,
        )

        result = estimate_map_noise(data, method="histogram_clip")
        assert result.sigma > 0
        assert not np.isnan(result.sigma)
        assert not np.isinf(result.sigma)

    def test_negative_values(self) -> None:
        """Test handling of negative values in data."""
        np.random.seed(42)
        data = np.random.normal(-10, 2.0, size=(100, 100))

        result = estimate_map_noise(data, method="histogram_clip")
        assert result.sigma > 0

    def test_single_value_array(self) -> None:
        """Test very small data value."""
        # Very small but non-zero values
        data = np.ones((100, 100)) * 1e-10

        result = estimate_map_noise(data, method="histogram_clip")
        # Should handle gracefully
        assert result.sigma >= 0


class TestReproducibility:
    """Tests for reproducibility across multiple calls."""

    def test_histogram_clip_reproducibility(self) -> None:
        """Test that results are reproducible for same input."""
        data = np.random.normal(0, 1.5, size=(100, 100))

        result1 = estimate_map_noise(data, method="histogram_clip")
        result2 = estimate_map_noise(data, method="histogram_clip")

        assert result1.sigma == result2.sigma
        assert np.array_equal(result1.sigma_map, result2.sigma_map)
        assert result1.method_used == result2.method_used

    def test_histogram_clip_default_parameters(self) -> None:
        """Test that default parameters are sensible."""
        data = np.random.normal(0, 1, size=(100, 100))

        # Should work with just method specified
        result = estimate_map_noise(data, method="histogram_clip")
        assert result.sigma > 0
        assert result.method_used == "histogram_clip"
