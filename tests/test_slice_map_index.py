from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy.io import fits

from pychmp.ab_scan_artifacts import COMPATIBILITY_SIGNATURE_KEY, write_point_scan_artifact
from pychmp.grid_points import (
    GridPointAssignedEvent,
    GridTrialCommittedEvent,
    apply_grid_point_assigned,
    apply_grid_trial_committed,
)
from pychmp.slice_map_index import (
    SliceMapIndex,
    build_slice_map_index,
    hydrate_render_caches_from_index,
    load_render_pair_from_index,
)


def _header() -> fits.Header:
    header = fits.Header()
    header["SIMPLE"] = True
    header["BITPIX"] = -32
    header["NAXIS"] = 2
    header["NAXIS1"] = 2
    header["NAXIS2"] = 2
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRPIX1"] = 1.0
    header["CRPIX2"] = 1.0
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CDELT1"] = 2.0
    header["CDELT2"] = 2.0
    header["DATE-OBS"] = "2020-11-26T20:00:00"
    return header


def test_build_slice_map_index_from_grid_trial(tmp_path: Path) -> None:
    observed = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma = np.ones((2, 2), dtype=float)
    header = _header()
    diagnostics = {
        "artifact_kind": "pychmp_ab_scan_sparse_points",
        COMPATIBILITY_SIGNATURE_KEY: "sig-index",
        "target_metric": "chi2",
        "target_slice_key": "mw_5p700000ghz",
        "spectral_domain": "mw",
        "spectral_label": "5.700 GHz",
        "frequency_ghz": 5.7,
        "metrics_mask_threshold": 0.1,
    }
    artifact_h5 = tmp_path / "index.h5"
    write_point_scan_artifact(
        artifact_h5,
        observed=observed,
        sigma_map=sigma,
        wcs_header=header,
        diagnostics=diagnostics,
        point_records=[],
    )
    apply_grid_point_assigned(
        artifact_h5,
        GridPointAssignedEvent(a=0.3, b=2.7, q0_start=1e-3, next_q0=1e-3, metric_name="chi2"),
        observed=observed,
        sigma_map=sigma,
        wcs_header=header,
        diagnostics=diagnostics,
    )
    apply_grid_trial_committed(
        artifact_h5,
        GridTrialCommittedEvent(
            point_id="p000000",
            trial_index=0,
            q0=1e-3,
            metric=0.5,
            next_q0=2e-3,
            best_trial_index=0,
            best_metric=0.5,
            raw_modeled_map=observed * 0.9,
        ),
        observed=observed,
        sigma_map=sigma,
        wcs_header=header,
        diagnostics=diagnostics,
    )

    index = build_slice_map_index(artifact_h5, slice_key="mw_5p700000ghz")
    assert index.point_count() >= 1
    assert index.trial_count() >= 1
    assert index.raw_map_ref(0.3, 2.7, 1e-3) is not None

    pair = load_render_pair_from_index(
        artifact_h5,
        index=index,
        a=0.3,
        b=2.7,
        q0=1e-3,
        observed_template=observed,
        psf_kernel=None,
    )
    assert pair is not None
    raw_arr, modeled_arr = pair
    assert raw_arr.shape == (2, 2)
    assert modeled_arr.shape == (2, 2)

    raw_by_q0: dict[str, np.ndarray] = {}
    modeled_by_q0: dict[str, np.ndarray] = {}
    hydrated = hydrate_render_caches_from_index(
        artifact_h5,
        index=index,
        a=0.3,
        b=2.7,
        raw_modeled_by_q0=raw_by_q0,
        modeled_by_q0=modeled_by_q0,
        observed_template=observed,
        psf_kernel=None,
    )
    assert hydrated >= 1
    assert len(raw_by_q0) >= 1


def test_slice_map_index_register_dedupes_by_q0() -> None:
    index = SliceMapIndex(slice_key="mw_test", descriptor={"key": "mw_test", "domain": "mw"})
    index.register(a=0.1, b=1.0, q0=0.001, raw_map_ref="/map_store/maps/aaa")
    index.register(a=0.1, b=1.0, q0=0.001, raw_map_ref="/map_store/maps/bbb")
    assert index.trial_count() == 1
    assert index.raw_map_ref(0.1, 1.0, 0.001) == "/map_store/maps/bbb"
    assert index.summary() == "1 map(s) across 1 (a,b) point(s)"
    assert index.point_keys() == ((0.1, 1.0),)


def test_euv_slice_map_index_requires_explicit_channel_identity(tmp_path: Path) -> None:
    observed = np.ones((2, 2), dtype=float)
    sigma = np.ones((2, 2), dtype=float)
    diagnostics = {
        "artifact_kind": "pychmp_ab_scan_sparse_points",
        "target_slice_key": "euv_171",
        "spectral_domain": "euv",
        "spectral_label": "171 A",
        "wavelength_angstrom": 171.0,
        "euv_channel": "171",
        "target_metric": "eta2",
    }
    artifact_h5 = tmp_path / "euv_index.h5"
    write_point_scan_artifact(
        artifact_h5,
        observed=observed,
        sigma_map=sigma,
        wcs_header=_header(),
        diagnostics=diagnostics,
        point_records=[],
    )
    with h5py.File(artifact_h5, "r+") as handle:
        maps = handle.require_group("map_store").require_group("maps")
        ambiguous = maps.create_group("ambiguous")
        ambiguous.create_dataset("data", data=np.ones((2, 2), dtype=np.float32))
        ambiguous.create_dataset(
            "identity_json",
            data=np.bytes_(
                json.dumps(
                    {
                        "schema": "pychmp.map_identity.v0",
                        "domain": "euv",
                        "component": "stokes_i",
                        "a": 0.6,
                        "b": 1.8,
                        "q0": 0.003,
                    }
                )
            ),
        )

    index = build_slice_map_index(artifact_h5, slice_key="euv_171")
    assert index.trial_count() == 0

    with h5py.File(artifact_h5, "r+") as handle:
        explicit = handle["map_store/maps"].create_group("explicit")
        explicit.create_dataset("data", data=np.full((2, 2), 2.0, dtype=np.float32))
        explicit.create_dataset(
            "identity_json",
            data=np.bytes_(
                json.dumps(
                    {
                        "schema": "pychmp.map_identity.v2",
                        "domain": "euv",
                        "channel_or_frequency": "171",
                        "component": "stokes_i",
                        "a": 0.6,
                        "b": 1.8,
                        "q0": 0.003,
                    }
                )
            ),
        )

    index = build_slice_map_index(artifact_h5, slice_key="euv_171")
    assert index.trial_count() == 1
    assert index.raw_map_ref(0.6, 1.8, 0.003) == "/map_store/maps/explicit"
