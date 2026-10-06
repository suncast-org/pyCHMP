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


def test_sigma_fwhm_roundtrip() -> None:
    sigma = 14.057080473472078
    assert psf_module.fwhm_arcsec_to_sigma(psf_module.sigma_arcsec_to_fwhm(sigma)) == pytest.approx(sigma)


def test_extract_viktor_srh_sigma_header_from_pdf_example() -> None:
    """Viktor PDF example (~21 GHz SRH1224): beam_sa/sb are σ; PA uses -beam_phi."""
    header = fits.Header()
    header["BEAM_SA"] = 2.819550429330291
    header["BEAM_SB"] = 14.057080473472078
    header["BEAM_PHI"] = 11.203765115667782
    header["BEAM_P"] = 1.0

    metadata = extract_psf_metadata_from_header(header)

    assert metadata is not None
    assert metadata.source == "fits_header:srh_sigma"
    assert metadata.kind == "gaussian"
    assert metadata.allows_frequency_scaling is True
    # sa < sb → major is sb; PA of major = -phi + 90°
    assert metadata.bmaj_arcsec == pytest.approx(psf_module.sigma_arcsec_to_fwhm(14.057080473472078))
    assert metadata.bmin_arcsec == pytest.approx(psf_module.sigma_arcsec_to_fwhm(2.819550429330291))
    expected_pa = psf_module._normalize_pa_deg(-11.203765115667782 + 90.0)
    assert metadata.bpa_deg == pytest.approx(expected_pa)


def test_extract_viktor_srh_rejects_super_gaussian_p_without_fallback() -> None:
    """Unsupported BEAM_P≠1 yields None only when no later route exists."""
    header = fits.Header()
    header["beam_sa"] = 3.0
    header["beam_sb"] = 5.0
    header["beam_phi"] = 10.0
    header["beam_p"] = 2.0

    assert extract_psf_metadata_from_header(header) is None


def test_srh_super_gaussian_p_falls_through_to_bmaj() -> None:
    header = fits.Header()
    header["BEAM_SA"] = 3.0
    header["BEAM_SB"] = 5.0
    header["BEAM_PHI"] = 10.0
    header["BEAM_P"] = 2.0
    header["BMAJ"] = 30.0 / 3600.0
    header["BMIN"] = 20.0 / 3600.0
    header["BPA"] = 15.0

    metadata = extract_psf_metadata_from_header(header)
    assert metadata is not None
    assert metadata.source == "fits_header"
    assert metadata.bmaj_arcsec == pytest.approx(30.0)
    assert metadata.bmin_arcsec == pytest.approx(20.0)


def test_srh_super_gaussian_p_falls_through_to_correlated_sx() -> None:
    header = fits.Header()
    header["BEAM_SA"] = 3.0
    header["BEAM_SB"] = 5.0
    header["BEAM_PHI"] = 10.0
    header["BEAM_P"] = 2.0
    header["BEAM_SX"] = 5.0
    header["BEAM_SY"] = 12.0
    header["BEAM_RHO"] = 0.35

    metadata = extract_psf_metadata_from_header(header)
    smaj, smin, pa = psf_module.correlated_sigma_to_rotated(5.0, 12.0, 0.35)
    assert metadata is not None
    assert metadata.source == "fits_header:srh_correlated"
    assert metadata.bmaj_arcsec == pytest.approx(psf_module.sigma_arcsec_to_fwhm(smaj))
    assert metadata.bmin_arcsec == pytest.approx(psf_module.sigma_arcsec_to_fwhm(smin))
    assert metadata.bpa_deg == pytest.approx(pa)


def test_srh_zero_sa_falls_through_to_bmaj() -> None:
    header = fits.Header()
    header["BEAM_SA"] = 0.0
    header["BEAM_SB"] = 5.0
    header["BEAM_PHI"] = 10.0
    header["BEAM_P"] = 1.0
    header["BMAJ"] = 25.0 / 3600.0
    header["BMIN"] = 18.0 / 3600.0
    header["BPA"] = -5.0

    metadata = extract_psf_metadata_from_header(header)
    assert metadata is not None
    assert metadata.source == "fits_header"
    assert metadata.bmaj_arcsec == pytest.approx(25.0)
    assert metadata.bmin_arcsec == pytest.approx(18.0)


def test_srh_negative_sx_falls_through_to_bmaj() -> None:
    header = fits.Header()
    header["BEAM_SX"] = -1.0
    header["BEAM_SY"] = 12.0
    header["BEAM_RHO"] = 0.0
    header["BMAJ"] = 22.0 / 3600.0
    header["BMIN"] = 16.0 / 3600.0
    header["BPA"] = 0.0

    metadata = extract_psf_metadata_from_header(header)
    assert metadata is not None
    assert metadata.source == "fits_header"
    assert metadata.bmaj_arcsec == pytest.approx(22.0)
    assert metadata.bmin_arcsec == pytest.approx(16.0)


def test_eovsa_bmaj_extract_does_not_import_sunpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """EOVSA/BMAJ resolve must not pull sunpy (lazy SSRT-only import).

    Do not evict ``pychmp.psf`` from ``sys.modules``: reloading it mid-suite
    leaves stale ``from pychmp.psf import …`` bindings in other tests and
    breaks monkeypatch-based cache assertions.
    """
    import builtins
    import sys

    for key in list(sys.modules):
        if key == "sunpy" or key.startswith("sunpy."):
            monkeypatch.delitem(sys.modules, key, raising=False)

    real_import = builtins.__import__

    def _guard(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        root = name.split(".", 1)[0]
        if root == "sunpy":
            raise AssertionError(f"unexpected sunpy import during EOVSA resolve: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guard)

    header = fits.Header()
    header["TELESCOP"] = "EOVSA"
    header["BMAJ"] = 30.0 / 3600.0
    header["BMIN"] = 20.0 / 3600.0
    header["BPA"] = 15.0
    metadata = extract_psf_metadata_from_header(header)
    assert metadata is not None
    assert metadata.source == "fits_header"
    assert not any(k == "sunpy" or k.startswith("sunpy.") for k in sys.modules)


def test_extract_gx_srh_correlated_header() -> None:
    header = fits.Header()
    header["beam_sx"] = 5.0
    header["beam_sy"] = 12.0
    header["beam_rho"] = 0.35

    metadata = extract_psf_metadata_from_header(header)
    smaj, smin, pa = psf_module.correlated_sigma_to_rotated(5.0, 12.0, 0.35)

    assert metadata is not None
    assert metadata.source == "fits_header:srh_correlated"
    assert metadata.allows_frequency_scaling is True
    assert metadata.bmaj_arcsec == pytest.approx(psf_module.sigma_arcsec_to_fwhm(smaj))
    assert metadata.bmin_arcsec == pytest.approx(psf_module.sigma_arcsec_to_fwhm(smin))
    assert metadata.bpa_deg == pytest.approx(pa)


def test_make_srh_correlated_beam_matches_rotated_gaussian_kernel() -> None:
    """MakeSRHbeam → σ axes/PA → elliptical_gaussian_kernel should recover the beam shape."""
    sx, sy, rho = 6.0, 14.0, 0.4
    dx = dy = 1.0
    nx = ny = 81

    beam = psf_module.make_srh_correlated_beam(sx, sy, rho, nx=nx, ny=ny, dx_arcsec=dx, dy_arcsec=dy)
    beam = beam / float(np.sum(beam))

    smaj, smin, pa = psf_module.correlated_sigma_to_rotated(sx, sy, rho)
    kernel = elliptical_gaussian_kernel(
        bmaj_arcsec=psf_module.sigma_arcsec_to_fwhm(smaj),
        bmin_arcsec=psf_module.sigma_arcsec_to_fwhm(smin),
        bpa_deg=pa,
        dx_arcsec=dx,
        dy_arcsec=dy,
        size=nx,
    )

    beam_n = beam / float(np.max(beam))
    kernel_n = kernel / float(np.max(kernel))
    residual = float(np.max(np.abs(beam_n - kernel_n)))
    assert residual < 0.05

    beam_fwhm = beam_fwhm_from_kernel(beam, dx_arcsec=dx, dy_arcsec=dy)
    kernel_fwhm = beam_fwhm_from_kernel(kernel, dx_arcsec=dx, dy_arcsec=dy)
    assert beam_fwhm is not None and kernel_fwhm is not None
    assert beam_fwhm["bmaj_arcsec"] == pytest.approx(kernel_fwhm["bmaj_arcsec"], rel=0.02)
    assert beam_fwhm["bmin_arcsec"] == pytest.approx(kernel_fwhm["bmin_arcsec"], rel=0.02)
    # PA may differ by 180°; compare wrapped absolute difference.
    dpa = abs(psf_module._normalize_pa_deg(beam_fwhm["bpa_deg"] - kernel_fwhm["bpa_deg"]))
    assert dpa == pytest.approx(0.0, abs=2.0) or dpa == pytest.approx(180.0, abs=2.0)


def test_viktor_header_builds_frequency_scalable_kernel() -> None:
    header = fits.Header()
    header["BEAM_SA"] = 2.819550429330291
    header["BEAM_SB"] = 14.057080473472078
    header["BEAM_PHI"] = 11.203765115667782
    header["BEAM_P"] = 1.0
    metadata = extract_psf_metadata_from_header(header)

    kernel, kernel_meta = build_psf_kernel(
        metadata=metadata,
        dx_arcsec=4.9,
        dy_arcsec=4.9,
        active_frequency_ghz=10.66,
        ref_frequency_ghz=21.32,
        scale_inverse_frequency=True,
    )
    assert kernel is not None
    assert kernel_meta is not None
    assert float(kernel.sum()) == pytest.approx(1.0)
    assert bool(kernel_meta["scaled"]) is True
    assert float(kernel_meta["scale_factor"]) == pytest.approx(2.0)
