from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from astropy.io import fits
import pytest

from pychmp import build_psf_kernel, default_psf_metadata, extract_psf_metadata_from_header, resolve_psf_metadata
from pychmp import psf as psf_module
from pychmp.psf import beam_fwhm_from_kernel, elliptical_gaussian_kernel


def test_extract_psf_metadata_from_header_converts_degree_beam_to_arcsec() -> None:
    header = fits.Header()
    header["BMAJ"] = 2.0 / 3600.0
    header["BMIN"] = 1.0 / 3600.0
    header["BPA"] = 33.0

    metadata = extract_psf_metadata_from_header(header)

    assert metadata is not None
    assert metadata.source == "fits_header"
    assert metadata.kind == "gaussian"
    assert metadata.bmaj_arcsec == pytest.approx(2.0)
    assert metadata.bmin_arcsec == pytest.approx(1.0)
    assert metadata.bpa_deg == pytest.approx(33.0)


def test_beam_fwhm_from_kernel_recovers_gaussian_beam() -> None:
    kernel = elliptical_gaussian_kernel(
        bmaj_arcsec=40.0,
        bmin_arcsec=28.0,
        bpa_deg=35.0,
        dx_arcsec=2.0,
        dy_arcsec=2.0,
        size=61,
    )
    payload = beam_fwhm_from_kernel(kernel, dx_arcsec=2.0, dy_arcsec=2.0)
    assert payload is not None
    assert payload["bmaj_arcsec"] == pytest.approx(40.0, rel=0.08)
    assert payload["bmin_arcsec"] == pytest.approx(28.0, rel=0.08)


def test_build_psf_kernel_normalizes_direct_kernel() -> None:
    metadata = resolve_psf_metadata(
        header_psf=None,
        domain="mw",
        instrument_name="EOVSA",
        cli_psf_kernel=np.array([[0.0, 1.0], [1.0, 2.0]], dtype=float),
        cli_psf_bmaj_arcsec=None,
        cli_psf_bmin_arcsec=None,
        cli_psf_bpa_deg=None,
        fallback_psf_bmaj_arcsec=None,
        fallback_psf_bmin_arcsec=None,
        fallback_psf_bpa_deg=None,
        override_header_psf=False,
    )

    kernel, kernel_meta = build_psf_kernel(metadata=metadata, dx_arcsec=1.0, dy_arcsec=1.0)

    assert metadata is not None
    assert metadata.kind == "kernel"
    assert kernel is not None
    assert kernel_meta is not None
    assert kernel.shape == (2, 2)
    assert float(kernel.sum()) == pytest.approx(1.0)
    np.testing.assert_allclose(kernel, np.array([[0.0, 0.25], [0.25, 0.5]], dtype=float))


def test_build_psf_kernel_applies_frequency_scaling_for_gaussian_override() -> None:
    metadata = resolve_psf_metadata(
        header_psf=None,
        domain="mw",
        instrument_name="EOVSA",
        cli_psf_kernel=None,
        cli_psf_bmaj_arcsec=4.0,
        cli_psf_bmin_arcsec=2.0,
        cli_psf_bpa_deg=15.0,
        fallback_psf_bmaj_arcsec=None,
        fallback_psf_bmin_arcsec=None,
        fallback_psf_bpa_deg=None,
        override_header_psf=False,
    )

    kernel, kernel_meta = build_psf_kernel(
        metadata=metadata,
        dx_arcsec=1.0,
        dy_arcsec=1.0,
        active_frequency_ghz=4.0,
        ref_frequency_ghz=8.0,
        scale_inverse_frequency=True,
    )

    assert metadata is not None
    assert kernel is not None
    assert kernel_meta is not None
    assert kernel.shape == (41, 41)
    assert float(kernel.sum()) == pytest.approx(1.0)
    assert float(kernel_meta["active_bmaj_arcsec"]) == pytest.approx(8.0)
    assert float(kernel_meta["active_bmin_arcsec"]) == pytest.approx(4.0)
    assert bool(kernel_meta["scaled"]) is True


def test_default_psf_metadata_prefers_aiapy_kernel_for_aia(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        psf_module,
        "_build_aiapy_aia_kernel",
        lambda **kwargs: psf_module.PSFMetadata(
            source="aiapy_psf:171",
            kind="kernel",
            kernel=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            allows_frequency_scaling=False,
        ),
    )
    monkeypatch.setattr(psf_module, "_response_sampling_default_psf", lambda **kwargs: None)

    metadata = default_psf_metadata(domain="euv", instrument_name="AIA", wavelength_angstrom=171.0)

    assert metadata is not None
    assert metadata.kind == "kernel"
    assert metadata.source == "aiapy_psf:171"
    assert metadata.kernel is not None
    assert metadata.as_dict()["psf_kernel_shape"] == (2, 2)


def test_default_psf_metadata_uses_response_sampling_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(psf_module, "_build_aiapy_aia_kernel", lambda **kwargs: None)
    monkeypatch.setattr(psf_module, "_lookup_response_sampling_pixel_arcsec", lambda instrument_name: (1.25, "trace"))

    metadata = default_psf_metadata(domain="euv", instrument_name="TRACE")

    assert metadata is not None
    assert metadata.kind == "gaussian"
    assert metadata.source == "response_sampling_default:trace"
    assert metadata.bmaj_arcsec == pytest.approx(1.25)
    assert metadata.bmin_arcsec == pytest.approx(1.25)
    assert metadata.bpa_deg == pytest.approx(0.0)


def test_aiapy_psf_kernel_recomputes_without_persistent_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def _calculate_psf(_wavelength):
        calls["count"] += 1
        return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)

    def _fake_import_module(name: str):
        if name == "astropy.units":
            return SimpleNamespace(angstrom=1.0)
        if name == "aiapy.psf":
            return SimpleNamespace(calculate_psf=_calculate_psf)
        raise ImportError(name)

    monkeypatch.setattr(psf_module, "import_module", _fake_import_module)

    first = default_psf_metadata(domain="euv", instrument_name="AIA", wavelength_angstrom=171.0)
    second = default_psf_metadata(domain="euv", instrument_name="AIA", wavelength_angstrom=171.0)

    assert first is not None and second is not None
    assert first.kernel is not None and second.kernel is not None
    np.testing.assert_allclose(np.asarray(first.kernel), np.asarray(second.kernel))
    assert calls["count"] == 2
