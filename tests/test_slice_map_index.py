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
from pychmp.map_store import build_map_identity
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


def test_grid_trial_auxiliary_maps_are_channel_safe(tmp_path: Path) -> None:
    observed = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma = np.ones((2, 2), dtype=float)
    header = _header()
    artifact_h5 = tmp_path / "euv_auxiliary_index.h5"
    base = {
        "artifact_kind": "pychmp_ab_scan_sparse_points",
        COMPATIBILITY_SIGNATURE_KEY: "sig-euv-aux",
        "target_metric": "eta2",
        "spectral_domain": "euv",
        "metrics_mask_threshold": 0.1,
        "forward_model_sha256": "forward",
        "forward_model_identity_version": "v-test",
        "ebtel_sha256": "ebtel",
        "artifact_geometry_sha256": "geometry",
        "euv_response_sha256": "response",
        "euv_response_identity_version": "response-v-test",
    }
    diagnostics_94 = {
        **base,
        "target_slice_key": "euv_94",
        "slice_key": "euv_94",
        "spectral_label": "94 A",
        "euv_channel": "94",
        "wavelength_angstrom": 94.0,
    }
    diagnostics_171 = {
        **base,
        "target_slice_key": "euv_171",
        "slice_key": "euv_171",
        "spectral_label": "171 A",
        "euv_channel": "171",
        "wavelength_angstrom": 171.0,
    }
    write_point_scan_artifact(
        artifact_h5,
        observed=observed,
        sigma_map=sigma,
        wcs_header=header,
        diagnostics=diagnostics_94,
        point_records=[],
    )
    write_point_scan_artifact(
        artifact_h5,
        observed=observed,
        sigma_map=sigma,
        wcs_header=header,
        diagnostics=diagnostics_171,
        point_records=[],
    )
    apply_grid_point_assigned(
        artifact_h5,
        GridPointAssignedEvent(a=0.6, b=1.8, q0_start=1e-3, next_q0=1e-3, metric_name="eta2"),
        observed=observed,
        sigma_map=sigma,
        wcs_header=header,
        diagnostics=diagnostics_94,
    )
    identity_94 = build_map_identity(
        a=0.6,
        b=1.8,
        q0=1e-3,
        domain="euv",
        channel_or_frequency="94",
        component="stokes_i",
        forward_model_sha256="forward",
        forward_model_identity_version="v-test",
        ebtel_sha256="ebtel",
        artifact_geometry_sha256="geometry",
        euv_response_sha256="response",
        euv_response_identity_version="response-v-test",
    )
    identity_171 = build_map_identity(
        a=0.6,
        b=1.8,
        q0=1e-3,
        domain="euv",
        channel_or_frequency="171",
        component="stokes_i",
        forward_model_sha256="forward",
        forward_model_identity_version="v-test",
        ebtel_sha256="ebtel",
        artifact_geometry_sha256="geometry",
        euv_response_sha256="response",
        euv_response_identity_version="response-v-test",
    )
    apply_grid_trial_committed(
        artifact_h5,
        GridTrialCommittedEvent(
            point_id="p000000",
            trial_index=0,
            q0=1e-3,
            metric=0.3,
            next_q0=2e-3,
            best_trial_index=0,
            best_metric=0.3,
            raw_modeled_map=np.full((2, 2), 94.0, dtype=np.float32),
            raw_map_identity=identity_94,
            map_store_arrays={"synthetic/euv171": np.full((2, 2), 171.0, dtype=np.float32)},
            map_store_identities={"synthetic/euv171": identity_171},
        ),
        observed=observed,
        sigma_map=sigma,
        wcs_header=header,
        diagnostics=diagnostics_94,
    )

    index_94 = build_slice_map_index(artifact_h5, slice_key="euv_94")
    index_171 = build_slice_map_index(artifact_h5, slice_key="euv_171")
    ref_94 = index_94.raw_map_ref(0.6, 1.8, 1e-3)
    ref_171 = index_171.raw_map_ref(0.6, 1.8, 1e-3)
    assert ref_94 is not None
    assert ref_171 is not None
    assert ref_94 != ref_171

    with h5py.File(artifact_h5, "r") as h5:
        data_94 = np.asarray(h5[ref_94]["data"][()], dtype=float)
        data_171 = np.asarray(h5[ref_171]["data"][()], dtype=float)
        identity_json_94 = json.loads(h5[ref_94]["identity_json"][()].decode("utf-8"))
        identity_json_171 = json.loads(h5[ref_171]["identity_json"][()].decode("utf-8"))
    np.testing.assert_allclose(data_94, np.full((2, 2), 94.0))
    np.testing.assert_allclose(data_171, np.full((2, 2), 171.0))
    assert identity_json_94["channel_or_frequency"] == "94"
    assert identity_json_171["channel_or_frequency"] == "171"


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


@pytest.mark.parametrize("complete_layer", [True, False])
def test_index_prefers_layer_metadata_with_legacy_fallback(tmp_path, complete_layer):
    from pychmp.slice_map_index import _index_map_store_maps

    path = tmp_path / "layers.h5"
    identity = dict(domain="euv", channel_or_frequency="94", component="stokes_i",
                    a=0.6, b=1.8, q0=0.001)
    layer = dict(identity) if complete_layer else {"channel_or_frequency": "94"}
    with h5py.File(path, "w") as f:
        group = f.create_group("map_store/maps/raw")
        group.create_dataset("map_layer_json", data=json.dumps(layer))
        # A complete layer must not need the full identity dataset.
        if not complete_layer:
            group.create_dataset("identity_json", data=json.dumps(identity))
        group.create_dataset("data", data=np.ones((2, 2)))
    index = SliceMapIndex(slice_key="euv_94", descriptor={"domain": "euv", "channel_label": "94"})
    with h5py.File(path) as f:
        assert _index_map_store_maps(f, index, index.descriptor) == 1
    assert index.raw_map_ref(0.6, 1.8, 0.001) == "/map_store/maps/raw"


def test_map_store_index_progress_hook_fires_a_handful_of_times(tmp_path, capsys):
    from pychmp.slice_map_index import _index_map_store_maps, _report_map_index_progress

    path = tmp_path / "progress.h5"
    n_maps = 11
    matching = {0, 5, 10}
    with h5py.File(path, "w") as f:
        maps = f.create_group("map_store/maps")
        for i in range(n_maps):
            channel = "94" if i in matching else "171"
            layer = {
                "domain": "euv",
                "channel_or_frequency": channel,
                "component": "stokes_i",
                "a": 0.6,
                "b": 1.8,
                "q0": 0.001 * (i + 1),
            }
            group = maps.create_group(f"m{i:02d}")
            group.create_dataset("map_layer_json", data=json.dumps(layer))
    descriptor = {"domain": "euv", "channel_label": "94"}
    index = SliceMapIndex(slice_key="euv_94", descriptor=descriptor)
    calls: list[tuple[int, int]] = []
    with h5py.File(path, "r") as f:
        added = _index_map_store_maps(
            f,
            index,
            descriptor,
            progress=lambda visited, total: calls.append((visited, total)),
            progress_every=3,
            progress_interval_s=1.0e9,
        )
    assert added == len(matching)
    assert calls == [(3, n_maps), (6, n_maps), (9, n_maps), (11, n_maps)]
    assert index.raw_map_ref(0.6, 1.8, 0.001) == "/map_store/maps/m00"

    stamps = iter([0.0, 6.0, 6.2, 12.0, *([12.1] * 8)])

    def clock() -> float:
        return next(stamps)

    timed: list[tuple[int, int]] = []
    timed_index = SliceMapIndex(slice_key="euv_94", descriptor=descriptor)
    with h5py.File(path, "r") as f:
        _index_map_store_maps(
            f,
            timed_index,
            descriptor,
            progress=lambda visited, total: timed.append((visited, total)),
            progress_every=10**9,
            progress_interval_s=5.0,
            _clock=clock,
        )
    assert timed == [(1, n_maps), (3, n_maps), (11, n_maps)]

    _report_map_index_progress(5000, n_maps, None)
    captured = capsys.readouterr()
    assert captured.out == "Map store index: visited 5000/11 maps\n"


def test_current_slice_index_reads_one_slice_without_walking_maps(tmp_path, monkeypatch):
    from pychmp.ab_scan_artifacts import _write_map_store_array
    from pychmp.slice_map_index import MAP_STORE_SLICE_INDEX_GROUP, _index_map_store_maps

    path = tmp_path / "shared-index.h5"
    with h5py.File(path, "w") as f:
        maps = f.create_group("map_store/maps")
        for name, channel, q0 in (("a94", "94", 0.002), ("a171", "171", 0.004), ("other", "335", 0.006)):
            layer = {
                "domain": "euv",
                "channel_or_frequency": channel,
                "component": "stokes_i",
                "a": 0.5,
                "b": 1.5,
                "q0": q0,
            }
            group = maps.create_group(name)
            group.create_dataset("map_layer_json", data=json.dumps(layer))

    map_walks: list[str] = []
    original_keys = h5py.Group.keys

    def counting_keys(self):
        if str(self.name).endswith("/maps"):
            map_walks.append(str(self.name))
        return original_keys(self)

    monkeypatch.setattr(h5py.Group, "keys", counting_keys)
    descriptor_94 = {"domain": "euv", "channel_label": "94"}
    descriptor_171 = {"domain": "euv", "channel_label": "171"}
    first = SliceMapIndex(slice_key="euv_94", descriptor=descriptor_94)
    with h5py.File(path, "r+") as f:
        assert _index_map_store_maps(f, first, descriptor_94) == 1
        assert MAP_STORE_SLICE_INDEX_GROUP in f["map_store"]
        assert len(f["map_store"][MAP_STORE_SLICE_INDEX_GROUP]["slice_id"]) == 3
    assert map_walks == ["/map_store/maps"]
    assert first.raw_map_ref(0.5, 1.5, 0.002) == "/map_store/maps/a94"
    assert first.raw_map_ref(0.5, 1.5, 0.004) is None

    map_walks.clear()
    second = SliceMapIndex(slice_key="euv_171", descriptor=descriptor_171)
    with h5py.File(path, "r+") as f:
        assert _index_map_store_maps(f, second, descriptor_171) == 1
    assert map_walks == []
    assert second.raw_map_ref(0.5, 1.5, 0.004) == "/map_store/maps/a171"
    assert second.raw_map_ref(0.5, 1.5, 0.002) is None

    with h5py.File(path, "r+") as f:
        _write_map_store_array(
            f,
            identity={
                "domain": "euv",
                "channel_or_frequency": "193",
                "component": "stokes_i",
                "a": 0.4,
                "b": 1.2,
                "q0": 0.005,
            },
            data=np.ones((2, 2), dtype=np.float32),
        )
        assert len(f["map_store"][MAP_STORE_SLICE_INDEX_GROUP]["slice_id"]) == len(f["map_store/maps"])
    map_walks.clear()
    third = SliceMapIndex(slice_key="euv_193", descriptor={"domain": "euv", "channel_label": "193"})
    with h5py.File(path, "r+") as f:
        assert _index_map_store_maps(f, third, third.descriptor) == 1
    assert map_walks == []
    assert third.raw_map_ref(0.4, 1.2, 0.005) is not None


def test_slice_index_ab_points_reads_columns_without_opening_maps(tmp_path, monkeypatch) -> None:
    from pychmp.slice_map_index import _write_slice_index_rows, slice_index_ab_points, slice_index_ab_snapshot

    path = tmp_path / "index-only.h5"
    rows = [
        ("euv_193", 0.0, 1.0, 0.002, "sibling-low-q0", "stokes_i"),
        ("euv_193", 0.0, 1.0, 0.2, "sibling-other-q0", "stokes_i"),
        ("euv_193", 4.0, 4.0, 1.0e-6, "any-positive-q0", "stokes_i"),
        ("euv_193", 1.0, 1.0, 0.002, "corona-only", "corona"),
        ("euv_193", 2.0, 2.0, 0.0, "nonpositive-q0", "stokes_i"),
        ("euv_193", 3.0, 3.0, -0.1, "negative-q0", "stokes_i"),
        ("euv_94", 0.0, 1.0, 0.002, "other-wavelength", "stokes_i"),
    ]
    with h5py.File(path, "w") as handle:
        _write_slice_index_rows(handle, rows)
        maps = handle.create_group("map_store/maps")
        data = maps.create_group("sibling-low-q0").create_dataset("data", data=np.ones((2, 2), dtype=np.float32))
        assert data.name.endswith("/data")

    opened_map_arrays: list[str] = []
    original_getitem = h5py.Dataset.__getitem__

    def tracking_getitem(self, item):
        if str(self.name).endswith("/data"):
            opened_map_arrays.append(str(self.name))
        return original_getitem(self, item)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", tracking_getitem)
    shared = {(0.0, 1.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)}
    assert slice_index_ab_points(path, "euv_193") == shared
    assert slice_index_ab_points(path, "euv_94") == shared
    assert slice_index_ab_points(path, "euv_335") == shared
    assert slice_index_ab_points(tmp_path / "missing.h5", "euv_193") == set()
    assert opened_map_arrays == []
    column_reads: list[str] = []
    index_kinds: list[str] = []

    def tracking_columns(self, item):
        column_reads.append(str(self.name))
        index_kinds.append(type(item).__name__)
        return original_getitem(self, item)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", tracking_columns)
    _points, nrows = slice_index_ab_snapshot(path, "euv_193", after_row=0)
    assert (0.0, 1.0) in _points
    assert "ndarray" not in index_kinds
    column_reads.clear()
    unchanged, same_nrows = slice_index_ab_snapshot(path, "euv_193", after_row=nrows)
    assert unchanged == set()
    assert same_nrows == nrows
    assert column_reads == []


def test_slice_index_ab_snapshot_keeps_points_when_the_tail_is_torn(tmp_path, monkeypatch) -> None:
    from pychmp import slice_map_index as index_mod
    from pychmp.slice_map_index import _write_slice_index_rows, slice_index_ab_snapshot

    path = tmp_path / "torn-tail.h5"
    rows = [
        ("euv_131", 0.0, 0.1, 0.002, "row-0", "stokes_i"),
        ("euv_211", 0.2, 0.2, 0.002, "sibling-other", "stokes_i"),
        ("euv_131", 0.4, 0.4, 0.003, "row-2", "stokes_i"),
        ("euv_131", 0.5, 0.5, 0.003, "row-3", "corona"),
        ("euv_131", 0.6, 0.6, 0.004, "row-4", "stokes_i"),
        ("euv_131", 0.7, 0.7, 0.004, "row-5", "stokes_i"),
        ("euv_131", 9.0, 9.0, 0.004, "torn-tail", "stokes_i"),
    ]
    with h5py.File(path, "w") as handle:
        _write_slice_index_rows(handle, rows)
    monkeypatch.setattr(index_mod, "_INDEX_READ_CHUNK", 3)
    original_getitem = h5py.Dataset.__getitem__

    def torn_tail(self, item):
        if (
            str(self.name).endswith("/a")
            and isinstance(item, slice)
            and item.start is not None
            and int(item.start) >= 6
        ):
            raise OSError("address of object past end of allocation")
        return original_getitem(self, item)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", torn_tail)
    points, nrows = slice_index_ab_snapshot(path, "euv_131")
    assert points == {(0.0, 0.1), (0.2, 0.2), (0.4, 0.4), (0.5, 0.5), (0.6, 0.6), (0.7, 0.7)}
    assert (9.0, 9.0) not in points
    assert nrows == 6
