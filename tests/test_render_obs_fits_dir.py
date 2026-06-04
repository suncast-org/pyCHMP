from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from pychmp.obs_maps import infer_spectral_domain_from_header, spectral_domains_compatible
from pychmp.render_obs_fits_dir import (
    RenderObsTargetContext,
    build_render_obs_target_context,
    discover_render_channels_from_dir,
    discover_render_frequencies_ghz_from_dir,
    euv_channel_from_obs_fits,
    frequency_ghz_from_obs_fits,
    infer_spectral_domain_from_obs_fits,
    list_obs_fits_files,
    scan_render_obs_fits_directory,
)
from pychmp.obs_maps import load_obs_map


def _write_fits(path: Path, header: fits.Header, *, data: np.ndarray | None = None) -> None:
    image = np.ones((4, 4), dtype=float) if data is None else data
    hdu = fits.PrimaryHDU(image, header=header)
    hdu.writeto(path, overwrite=True)


def test_list_obs_fits_files_sorted(tmp_path: Path) -> None:
    _write_fits(tmp_path / "z_map.fits", fits.Header({"RESTFRQ": 3.0e9, "NAXIS": 2, "NAXIS1": 4, "NAXIS2": 4}))
    _write_fits(tmp_path / "a_map.fits", fits.Header({"RESTFRQ": 1.4e9, "NAXIS": 2, "NAXIS1": 4, "NAXIS2": 4}))
    _write_fits(tmp_path / "notes.txt", fits.Header())

    files = list_obs_fits_files(tmp_path)

    assert [item.name for item in files] == ["a_map.fits", "z_map.fits"]


def test_frequency_ghz_from_obs_fits_reads_restfrq(tmp_path: Path) -> None:
    fits_path = tmp_path / "mw.fits"
    _write_fits(
        fits_path,
        fits.Header({"RESTFRQ": 2.874e9, "NAXIS": 2, "NAXIS1": 4, "NAXIS2": 4}),
    )

    assert frequency_ghz_from_obs_fits(fits_path, target_domain="mw") == pytest.approx(2.874)


def test_euv_channel_from_obs_fits_reads_wavelength(tmp_path: Path) -> None:
    fits_path = tmp_path / "aia193.fits"
    _write_fits(
        fits_path,
        fits.Header(
            {
                "WAVELNTH": 193.0,
                "INSTRUME": "AIA",
                "TELESCOP": "SDO",
                "NAXIS": 2,
                "NAXIS1": 4,
                "NAXIS2": 4,
            }
        ),
    )

    context = RenderObsTargetContext(domain="euv", instrument="aia", observer_los="earth")
    assert euv_channel_from_obs_fits(fits_path, target_context=context) == "193"


def test_discover_render_frequencies_ghz_from_dir_excludes_target(tmp_path: Path) -> None:
    target = tmp_path / "target.fits"
    extra_a = tmp_path / "extra_a.fits"
    extra_b = tmp_path / "extra_b.fits"
    for path, hz in (
        (target, 2.874e9),
        (extra_a, 3.2e9),
        (extra_b, 5.8e9),
    ):
        _write_fits(path, fits.Header({"RESTFRQ": hz, "NAXIS": 2, "NAXIS1": 4, "NAXIS2": 4}))

    freqs = discover_render_frequencies_ghz_from_dir(
        tmp_path,
        target_domain="mw",
        exclude_paths=(target,),
        exclude_frequency_ghz=2.874,
    )

    assert freqs == pytest.approx((3.2, 5.8))


def test_discover_render_channels_from_dir_excludes_target_channel(tmp_path: Path) -> None:
    target = tmp_path / "aia171.fits"
    extra = tmp_path / "aia193.fits"
    _write_fits(
        target,
        fits.Header({"WAVELNTH": 171.0, "NAXIS": 2, "NAXIS1": 4, "NAXIS2": 4}),
    )
    _write_fits(
        extra,
        fits.Header({"WAVELNTH": 193.0, "NAXIS": 2, "NAXIS1": 4, "NAXIS2": 4}),
    )

    channels = discover_render_channels_from_dir(
        tmp_path,
        target_domain="euv",
        exclude_paths=(target,),
        exclude_channel="171",
    )

    assert channels == ("193",)


def test_infer_spectral_domain_from_header_mw_and_euv() -> None:
    mw_header = fits.Header({"RESTFRQ": 2.874e9, "NAXIS": 2})
    euv_header = fits.Header({"WAVELNTH": 193.0, "NAXIS": 2})

    assert infer_spectral_domain_from_header(mw_header) == "mw"
    assert infer_spectral_domain_from_header(euv_header) == "euv"
    assert spectral_domains_compatible("mw", "mw")
    assert not spectral_domains_compatible("mw", "euv")


def test_scan_render_obs_fits_directory_filters_mixed_domains(tmp_path: Path) -> None:
    _write_fits(
        tmp_path / "aia193.fits",
        fits.Header({"WAVELNTH": 193.0, "NAXIS": 2, "NAXIS1": 4, "NAXIS2": 4}),
    )
    _write_fits(
        tmp_path / "mw.fits",
        fits.Header({"RESTFRQ": 3.2e9, "NAXIS": 2, "NAXIS1": 4, "NAXIS2": 4}),
    )

    scan = scan_render_obs_fits_directory(tmp_path, target_domain="mw")

    assert scan.candidates_scanned == 2
    assert len(scan.compatible) == 1
    assert scan.compatible[0].frequency_ghz == pytest.approx(3.2)
    assert scan.skipped_incompatible_domain == ("aia193.fits",)


def test_discover_render_frequencies_skips_euv_and_keeps_mw(tmp_path: Path) -> None:
    _write_fits(
        tmp_path / "aia193.fits",
        fits.Header({"WAVELNTH": 193.0, "NAXIS": 2, "NAXIS1": 4, "NAXIS2": 4}),
    )
    _write_fits(
        tmp_path / "mw.fits",
        fits.Header({"RESTFRQ": 3.2e9, "NAXIS": 2, "NAXIS1": 4, "NAXIS2": 4}),
    )

    freqs = discover_render_frequencies_ghz_from_dir(tmp_path, target_domain="mw")

    assert freqs == pytest.approx((3.2,))


def test_discover_render_frequencies_errors_when_only_incompatible_files(tmp_path: Path) -> None:
    _write_fits(
        tmp_path / "aia193.fits",
        fits.Header({"WAVELNTH": 193.0, "NAXIS": 2, "NAXIS1": 4, "NAXIS2": 4}),
    )

    with pytest.raises(ValueError, match="Reconsider --render-obs-fits-dir"):
        discover_render_frequencies_ghz_from_dir(tmp_path, target_domain="mw")


def test_discover_render_channels_skips_mw_and_keeps_euv(tmp_path: Path) -> None:
    _write_fits(
        tmp_path / "mw.fits",
        fits.Header({"RESTFRQ": 2.874e9, "NAXIS": 2, "NAXIS1": 4, "NAXIS2": 4}),
    )
    _write_fits(
        tmp_path / "aia193.fits",
        fits.Header({"WAVELNTH": 193.0, "NAXIS": 2, "NAXIS1": 4, "NAXIS2": 4}),
    )

    channels = discover_render_channels_from_dir(tmp_path, target_domain="euv")

    assert channels == ("193",)


def test_discover_render_channels_skips_uv_when_target_is_euv(tmp_path: Path) -> None:
    _write_fits(
        tmp_path / "aia1600.fits",
        fits.Header({"WAVELNTH": 1600.0, "NAXIS": 2, "NAXIS1": 4, "NAXIS2": 4}),
    )
    _write_fits(
        tmp_path / "aia193.fits",
        fits.Header({"WAVELNTH": 193.0, "NAXIS": 2, "NAXIS1": 4, "NAXIS2": 4}),
    )

    channels = discover_render_channels_from_dir(tmp_path, target_domain="euv")

    assert channels == ("193",)


def test_scan_requires_matching_instrument_when_target_declares_it(tmp_path: Path) -> None:
    target = tmp_path / "aia171.fits"
    aux_missing_instr = tmp_path / "aux_no_instr.fits"
    aux_matching = tmp_path / "aia193.fits"
    _write_fits(
        target,
        fits.Header({"WAVELNTH": 171.0, "INSTRUME": "AIA", "NAXIS": 2, "NAXIS1": 4, "NAXIS2": 4}),
    )
    _write_fits(
        aux_missing_instr,
        fits.Header({"WAVELNTH": 193.0, "NAXIS": 2, "NAXIS1": 4, "NAXIS2": 4}),
    )
    _write_fits(
        aux_matching,
        fits.Header({"WAVELNTH": 193.0, "INSTRUME": "AIA", "NAXIS": 2, "NAXIS1": 4, "NAXIS2": 4}),
    )
    target_map = load_obs_map(obs_path=target, source_mode="external_fits", domain="euv")
    context = build_render_obs_target_context(
        target_map,
        domain="euv",
        euv_instrument="AIA",
    )

    scan = scan_render_obs_fits_directory(
        tmp_path,
        target_domain="euv",
        exclude_paths=(target,),
        target_context=context,
    )

    assert [entry.channel for entry in scan.compatible] == ["193"]
    assert scan.skipped_incompatible_render_context == ("aux_no_instr.fits",)


def test_scan_filters_euvi_when_target_is_aia(tmp_path: Path) -> None:
    target = tmp_path / "aia171.fits"
    aux_aia = tmp_path / "aia193.fits"
    aux_euvi = tmp_path / "euvi195.fits"
    _write_fits(
        target,
        fits.Header({"WAVELNTH": 171.0, "INSTRUME": "AIA", "NAXIS": 2, "NAXIS1": 4, "NAXIS2": 4}),
    )
    _write_fits(
        aux_aia,
        fits.Header({"WAVELNTH": 193.0, "INSTRUME": "AIA", "NAXIS": 2, "NAXIS1": 4, "NAXIS2": 4}),
    )
    _write_fits(
        aux_euvi,
        fits.Header({"WAVELNTH": 195.0, "INSTRUME": "EUVI", "NAXIS": 2, "NAXIS1": 4, "NAXIS2": 4}),
    )
    target_map = load_obs_map(obs_path=target, source_mode="external_fits", domain="euv")
    context = build_render_obs_target_context(
        target_map,
        domain="euv",
        euv_instrument="AIA",
    )

    scan = scan_render_obs_fits_directory(
        tmp_path,
        target_domain="euv",
        exclude_paths=(target,),
        target_context=context,
    )
    channels = discover_render_channels_from_dir(
        tmp_path,
        target_domain="euv",
        exclude_paths=(target,),
        exclude_channel="171",
        scan=scan,
        target_context=context,
    )

    assert channels == ("193",)
    assert scan.skipped_incompatible_render_context == ("euvi195.fits",)


def test_build_render_obs_target_context_from_obs_map(tmp_path: Path) -> None:
    fits_path = tmp_path / "aia171.fits"
    _write_fits(
        fits_path,
        fits.Header({"WAVELNTH": 171.0, "INSTRUME": "AIA", "NAXIS": 2, "NAXIS1": 4, "NAXIS2": 4}),
    )
    obs_map = load_obs_map(obs_path=fits_path, source_mode="external_fits", domain="euv")

    context = build_render_obs_target_context(obs_map, domain="euv", euv_instrument="AIA")

    assert context.domain == "euv"
    assert context.instrument == "aia"
    assert context.observer_los == "earth"


def test_infer_spectral_domain_from_obs_fits(tmp_path: Path) -> None:
    fits_path = tmp_path / "mw.fits"
    _write_fits(
        fits_path,
        fits.Header({"RESTFRQ": 1.4e9, "NAXIS": 2, "NAXIS1": 4, "NAXIS2": 4}),
    )

    assert infer_spectral_domain_from_obs_fits(fits_path) == "mw"
