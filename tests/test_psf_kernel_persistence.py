from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy.io import fits

from pychmp.ab_scan_artifacts import (
    COMMON_PSF_KERNEL_DATASET,
    SLICE_CONTAINER_GROUP,
    _resolved_psf_kernel_from_common_payload,
    ensure_slice_common_psf_kernel_in_file,
    load_scan_file,
    register_sparse_search_in_artifact,
    write_point_scan_artifact,
)
from pychmp.grid_points import (
    GRID_POINTS_CONTRACT_VERSION,
    GridPointAssignedEvent,
    GridTrialCommittedEvent,
    apply_grid_point_event_with_retry,
    load_grid_point_trial_plot_payload,
)
from pychmp.psf import build_psf_kernel, clear_psf_kernel_build_cache, resolve_slice_psf_kernel


def _make_header() -> fits.Header:
    header = fits.Header()
    header["SIMPLE"] = True
    header["BITPIX"] = -32
    header["NAXIS"] = 2
    header["NAXIS1"] = 4
    header["NAXIS2"] = 4
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRPIX1"] = 2.0
    header["CRPIX2"] = 2.0
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CDELT1"] = 2.0
    header["CDELT2"] = 2.0
    return header


def _eovsa_diagnostics(*, slice_key: str = "mw_1p418ghz") -> dict[str, object]:
    return {
        "artifact_kind": "pychmp_ab_scan_sparse_points",
        "target_metric": "eta2",
        "target_slice_key": slice_key,
        "slice_key": slice_key,
        "metrics_mask_threshold": 0.2,
        "model_path": "/tmp/model.h5",
        "model_id": "model-123",
        "model_sha256": "a" * 64,
        "fits_file": "/tmp/obs.fits",
        "fits_sha256": "b" * 64,
        "ebtel_path": "/tmp/ebtel.bin",
        "ebtel_sha256": "c" * 64,
        "frequency_ghz": 1.418335,
        "active_frequency_ghz": 1.418335,
        "map_xc_arcsec": 0.0,
        "map_yc_arcsec": 180.0,
        "map_dx_arcsec": 2.0,
        "map_dy_arcsec": 2.0,
        "map_nx": 4,
        "map_ny": 4,
        "psf_source": "fits_header",
        "resolved_psf": {
            "kind": "gaussian",
            "source": "fits_header",
            "active_bmaj_arcsec": 35.0,
            "active_bmin_arcsec": 35.0,
            "active_bpa_deg": 0.0,
            "allows_frequency_scaling": False,
        },
        "contract_version": GRID_POINTS_CONTRACT_VERSION,
        "search_id": "search_psf",
        "selected_search_id": "search_psf",
    }


def _compact_test_kernel(*, dx_arcsec: float = 2.0) -> np.ndarray:
    from pychmp.psf import PSFMetadata

    metadata = PSFMetadata(
        source="test",
        kind="gaussian",
        bmaj_arcsec=35.0,
        bmin_arcsec=35.0,
        bpa_deg=0.0,
        allows_frequency_scaling=False,
    )
    kernel, _resolved = build_psf_kernel(
        metadata=metadata,
        dx_arcsec=float(dx_arcsec),
        dy_arcsec=float(dx_arcsec),
        active_frequency_ghz=1.418335,
    )
    assert kernel is not None
    return np.asarray(kernel, dtype=float)


def test_resolve_slice_psf_kernel_rebuilds_from_diagnostics() -> None:
    kernel = resolve_slice_psf_kernel(
        stored_kernel=None,
        diagnostics=_eovsa_diagnostics(),
        dx_arcsec=2.0,
        dy_arcsec=2.0,
        active_frequency_ghz=1.418335,
    )
    assert kernel is not None
    assert kernel.ndim == 2
    assert kernel.size > 0
    assert np.isfinite(float(np.nansum(kernel)))


def test_ensure_slice_common_psf_kernel_backfills_missing_dataset(tmp_path: Path) -> None:
    artifact_h5 = tmp_path / "missing_psf_kernel.h5"
    observed = np.ones((4, 4), dtype=float)
    sigma = np.ones((4, 4), dtype=float)
    header = _make_header()
    diagnostics = _eovsa_diagnostics()
    psf_kernel = _compact_test_kernel()

    write_point_scan_artifact(
        artifact_h5,
        observed=observed,
        sigma_map=sigma,
        wcs_header=header,
        diagnostics=diagnostics,
        psf_kernel=psf_kernel,
        point_records=[],
    )

    slice_key = "mw_1p418ghz"
    with h5py.File(artifact_h5, "a") as handle:
        common = handle[f"{SLICE_CONTAINER_GROUP}/{slice_key}/common"]
        assert COMMON_PSF_KERNEL_DATASET in common
        del common[COMMON_PSF_KERNEL_DATASET]
        if "psf_kernel_meta_json" in common:
            del common["psf_kernel_meta_json"]

    changed = ensure_slice_common_psf_kernel_in_file(
        artifact_h5,
        slice_key=slice_key,
        psf_kernel=psf_kernel,
        diagnostics=diagnostics,
    )
    assert changed is True
    with h5py.File(artifact_h5, "r") as handle:
        assert f"{SLICE_CONTAINER_GROUP}/{slice_key}/common/{COMMON_PSF_KERNEL_DATASET}" in handle


def test_register_sparse_search_backfills_missing_psf_kernel(tmp_path: Path) -> None:
    artifact_h5 = tmp_path / "register_backfill.h5"
    observed = np.ones((4, 4), dtype=float)
    sigma = np.ones((4, 4), dtype=float)
    header = _make_header()
    diagnostics = _eovsa_diagnostics()
    psf_kernel = _compact_test_kernel()

    write_point_scan_artifact(
        artifact_h5,
        observed=observed,
        sigma_map=sigma,
        wcs_header=header,
        diagnostics=diagnostics,
        psf_kernel=psf_kernel,
        point_records=[],
    )

    slice_key = "mw_1p418ghz"
    with h5py.File(artifact_h5, "a") as handle:
        common = handle[f"{SLICE_CONTAINER_GROUP}/{slice_key}/common"]
        del common[COMMON_PSF_KERNEL_DATASET]
        if "psf_kernel_meta_json" in common:
            del common["psf_kernel_meta_json"]

    register_sparse_search_in_artifact(
        artifact_h5,
        observed=observed,
        sigma_map=sigma,
        wcs_header=header,
        diagnostics=diagnostics,
        search_id="search_psf",
        psf_kernel=psf_kernel,
    )

    with h5py.File(artifact_h5, "r") as handle:
        assert f"{SLICE_CONTAINER_GROUP}/{slice_key}/common/{COMMON_PSF_KERNEL_DATASET}" in handle


def test_load_grid_point_trial_plot_payload_derives_modeled_from_diagnostics_psf(
    tmp_path: Path,
) -> None:
    artifact_h5 = tmp_path / "viewer_psf.h5"
    observed = np.ones((4, 4), dtype=float)
    sigma = np.ones((4, 4), dtype=float)
    header = _make_header()
    diagnostics = _eovsa_diagnostics()
    psf_kernel = _compact_test_kernel()
    raw_map = np.zeros((4, 4), dtype=float)
    raw_map[1:3, 1:3] = 10.0

    write_point_scan_artifact(
        artifact_h5,
        observed=observed,
        sigma_map=sigma,
        wcs_header=header,
        diagnostics=diagnostics,
        psf_kernel=psf_kernel,
        point_records=[],
    )

    slice_key = "mw_1p418ghz"
    with h5py.File(artifact_h5, "a") as handle:
        common = handle[f"{SLICE_CONTAINER_GROUP}/{slice_key}/common"]
        del common[COMMON_PSF_KERNEL_DATASET]
        if "psf_kernel_meta_json" in common:
            del common["psf_kernel_meta_json"]

    point_id = apply_grid_point_event_with_retry(
        artifact_h5,
        GridPointAssignedEvent(
            a=-0.8,
            b=2.5,
            q0_start=1e-4,
            next_q0=1e-4,
            metric_name="eta2",
        ),
        observed=observed,
        sigma_map=sigma,
        wcs_header=header,
        diagnostics=diagnostics,
        psf_kernel=psf_kernel,
    )
    apply_grid_point_event_with_retry(
        artifact_h5,
        GridTrialCommittedEvent(
            point_id=str(point_id),
            trial_index=0,
            q0=1.0e-4,
            metric=0.42,
            next_q0=1.0e-4,
            best_trial_index=0,
            best_metric=0.42,
            raw_modeled_map=raw_map.astype(np.float32),
        ),
        observed=observed,
        sigma_map=sigma,
        wcs_header=header,
        diagnostics=diagnostics,
        psf_kernel=psf_kernel,
    )

    payload = load_grid_point_trial_plot_payload(
        artifact_h5,
        a=-0.8,
        b=2.5,
        slice_key=slice_key,
        search_id="search_psf",
    )
    assert payload is not None
    raw_display = np.asarray(payload["raw_modeled_best"], dtype=float)
    modeled = np.asarray(payload["modeled_best"], dtype=float)
    assert float(np.nanmax(np.abs(modeled - raw_display))) > 0.0
    assert payload["psf_kernel"] is not None


def test_load_scan_file_resolves_psf_kernel_from_diagnostics_when_missing(tmp_path: Path) -> None:
    artifact_h5 = tmp_path / "load_scan_psf.h5"
    observed = np.ones((4, 4), dtype=float)
    sigma = np.ones((4, 4), dtype=float)
    header = _make_header()
    diagnostics = _eovsa_diagnostics()
    psf_kernel = _compact_test_kernel()

    write_point_scan_artifact(
        artifact_h5,
        observed=observed,
        sigma_map=sigma,
        wcs_header=header,
        diagnostics=diagnostics,
        psf_kernel=psf_kernel,
        point_records=[],
    )

    slice_key = "mw_1p418ghz"
    with h5py.File(artifact_h5, "a") as handle:
        common = handle[f"{SLICE_CONTAINER_GROUP}/{slice_key}/common"]
        del common[COMMON_PSF_KERNEL_DATASET]

    payload = load_scan_file(artifact_h5, slice_key=slice_key, include_maps=False)
    resolved = np.asarray(payload.get("psf_kernel"), dtype=float)
    assert resolved.ndim == 2
    assert resolved.size > 0

    with h5py.File(artifact_h5, "r") as handle:
        common = handle[f"{SLICE_CONTAINER_GROUP}/{slice_key}/common"]
        common_payload = {
            "diagnostics": json.loads(common["diagnostics_json"][()].decode("utf-8")),
            "psf_kernel": None,
        }
    rebuilt = _resolved_psf_kernel_from_common_payload(common_payload)
    assert rebuilt is not None
    np.testing.assert_allclose(rebuilt.sum(), 1.0, rtol=0.0, atol=1e-6)


def test_resolve_slice_psf_kernel_reuses_build_cache_when_stored_kernel_missing() -> None:
    clear_psf_kernel_build_cache()
    diagnostics = _eovsa_diagnostics()
    calls = {"count": 0}
    original = build_psf_kernel

    def _counting_build(**kwargs: object) -> tuple[np.ndarray | None, dict[str, object] | None]:
        calls["count"] += 1
        return original(**kwargs)

    import pychmp.psf as psf_module

    psf_module.build_psf_kernel = _counting_build
    try:
        kwargs = {
            "stored_kernel": None,
            "diagnostics": diagnostics,
            "dx_arcsec": 2.0,
            "dy_arcsec": 2.0,
            "active_frequency_ghz": 1.418335,
        }
        first = resolve_slice_psf_kernel(**kwargs)
        second = resolve_slice_psf_kernel(**kwargs)
    finally:
        psf_module.build_psf_kernel = original
        clear_psf_kernel_build_cache()

    assert first is not None
    assert second is not None
    np.testing.assert_array_equal(first, second)
    assert calls["count"] == 1
