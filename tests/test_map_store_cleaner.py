"""Tests for map_store unvisited-point cleaner (report-only default)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from pychmp.ab_scan_artifacts import _create_text_dataset, _json_dumps
from pychmp.map_store_cleaner import (
    clean_unvisited_map_store_points,
    main,
    report_unvisited_map_store_points,
)
from pychmp.slice_map_index import MAP_STORE_SLICE_INDEX_GROUP, _write_slice_index_rows


def _write_map(group: h5py.Group, map_key: str, *, a: float, b: float, q0: float = 1e-3) -> None:
    map_group = group.create_group(map_key)
    map_group.create_dataset("data", data=np.ones((2, 2), dtype=np.float32))
    identity = {
        "a": float(a),
        "b": float(b),
        "q0": float(q0),
        "domain": "euv",
        "channel_or_frequency": "171",
        "component": "stokes_i",
    }
    _create_text_dataset(map_group, "identity_json", _json_dumps(identity))
    _create_text_dataset(map_group, "map_layer_json", _json_dumps(identity))


def _write_grid_point(search_group: h5py.Group, point_id: str, *, a: float, b: float) -> None:
    point = search_group.require_group("grid_points").create_group(point_id)
    point.attrs["point_id"] = np.bytes_(point_id)
    point.attrs["a"] = float(a)
    point.attrs["b"] = float(b)
    point.attrs["status"] = np.bytes_("COMPLETED")
    point.attrs["q0_start"] = 1.0e-3
    point.attrs["best_trial_index"] = 0
    point.attrs["n_trials"] = 1
    point.attrs["metric_name"] = np.bytes_("chi2")
    point.attrs["created_utc"] = np.bytes_("2026-10-02T00:00:00Z")
    point.attrs["updated_utc"] = np.bytes_("2026-10-02T00:00:00Z")


def _seed_artifact(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        maps = handle.create_group("map_store/maps")
        _write_map(maps, "visited_map", a=0.1, b=1.0)
        _write_map(maps, "orphan_map", a=0.9, b=2.0)
        _write_slice_index_rows(
            handle,
            [
                ("euv_171", 0.1, 1.0, 1e-3, "visited_map", "stokes_i"),
                ("euv_171", 0.9, 2.0, 1e-3, "orphan_map", "stokes_i"),
            ],
        )
        search = handle.create_group("slices/euv_171/searches/search_a")
        _write_grid_point(search, "p000000", a=0.1, b=1.0)


def test_report_unvisited_map_store_points_counts_orphans(tmp_path: Path) -> None:
    path = tmp_path / "artifact.h5"
    _seed_artifact(path)

    report = report_unvisited_map_store_points(path)

    assert report.dry_run is True
    assert report.store_ab_count == 2
    assert report.visited_ab_count == 1
    assert report.orphan_ab_count == 1
    assert report.orphan_map_count == 1
    assert report.orphan_map_keys == ["orphan_map"]
    assert report.per_search[0].unvisited_store_ab_count == 1
    with h5py.File(path, "r") as handle:
        assert "orphan_map" in handle["map_store/maps"]


def test_clean_without_delete_is_dry_run(tmp_path: Path) -> None:
    path = tmp_path / "artifact.h5"
    _seed_artifact(path)

    report = clean_unvisited_map_store_points(path, delete=False)

    assert report.dry_run is True
    assert report.deleted_map_count == 0
    with h5py.File(path, "r") as handle:
        assert set(handle["map_store/maps"].keys()) == {"visited_map", "orphan_map"}


def test_clean_with_delete_removes_orphans_and_rewrites_index(tmp_path: Path) -> None:
    path = tmp_path / "artifact.h5"
    _seed_artifact(path)

    report = clean_unvisited_map_store_points(path, delete=True)

    assert report.dry_run is False
    assert report.deleted_map_count == 1
    assert report.deleted_map_keys == ["orphan_map"]
    with h5py.File(path, "r") as handle:
        assert set(handle["map_store/maps"].keys()) == {"visited_map"}
        index = handle["map_store"][MAP_STORE_SLICE_INDEX_GROUP]
        keys = [str(v) if not isinstance(v, bytes) else v.decode() for v in index["map_key"][()]]
        assert keys == ["visited_map"]


def test_cli_defaults_to_dry_run(tmp_path: Path, capsys) -> None:
    path = tmp_path / "artifact.h5"
    _seed_artifact(path)

    code = main([str(path)])

    assert code == 0
    out = capsys.readouterr().out
    assert "dry-run" in out.lower()
    assert "Re-run with --delete" in out
    with h5py.File(path, "r") as handle:
        assert "orphan_map" in handle["map_store/maps"]


def _seed_multi_search_artifact(path: Path) -> None:
    """Two searches share one slice; each visits a distinct store map + one orphan."""
    with h5py.File(path, "w") as handle:
        maps = handle.create_group("map_store/maps")
        _write_map(maps, "a_map", a=0.1, b=1.0)
        _write_map(maps, "b_map", a=0.5, b=1.5)
        _write_map(maps, "orphan_map", a=0.9, b=2.0)
        _write_slice_index_rows(
            handle,
            [
                ("euv_171", 0.1, 1.0, 1e-3, "a_map", "stokes_i"),
                ("euv_171", 0.5, 1.5, 1e-3, "b_map", "stokes_i"),
                ("euv_171", 0.9, 2.0, 1e-3, "orphan_map", "stokes_i"),
            ],
        )
        search_a = handle.create_group("slices/euv_171/searches/search_a")
        _write_grid_point(search_a, "p000000", a=0.1, b=1.0)
        search_b = handle.create_group("slices/euv_171/searches/search_b")
        _write_grid_point(search_b, "p000000", a=0.5, b=1.5)


def test_report_with_search_id_filter_still_uses_global_visited(tmp_path: Path) -> None:
    path = tmp_path / "artifact.h5"
    _seed_multi_search_artifact(path)

    report = report_unvisited_map_store_points(path, search_id="search_a")

    assert [item.search_id for item in report.per_search] == ["search_a"]
    assert report.per_search[0].unvisited_store_ab_count == 2  # b_map + orphan vs search_a only
    # Orphans must not include b_map — search_b visited it.
    assert report.visited_ab_count == 2
    assert report.orphan_ab_count == 1
    assert report.orphan_map_keys == ["orphan_map"]


def test_delete_with_search_id_filter_preserves_other_search_maps(tmp_path: Path) -> None:
    """Regression: scoped --delete must not wipe maps another search still uses."""
    path = tmp_path / "artifact.h5"
    _seed_multi_search_artifact(path)

    report = clean_unvisited_map_store_points(path, search_id="search_a", delete=True)

    assert report.dry_run is False
    assert report.deleted_map_keys == ["orphan_map"]
    assert report.deleted_map_count == 1
    with h5py.File(path, "r") as handle:
        assert set(handle["map_store/maps"].keys()) == {"a_map", "b_map"}
        index = handle["map_store"][MAP_STORE_SLICE_INDEX_GROUP]
        keys = [str(v) if not isinstance(v, bytes) else v.decode() for v in index["map_key"][()]]
        assert set(keys) == {"a_map", "b_map"}


def test_delete_with_slice_key_filter_preserves_other_search_maps(tmp_path: Path) -> None:
    path = tmp_path / "artifact.h5"
    _seed_multi_search_artifact(path)
    # Add a second slice with its own visited map so slice filter is meaningful for reporting.
    with h5py.File(path, "a") as handle:
        maps = handle["map_store/maps"]
        _write_map(maps, "other_slice_map", a=0.2, b=0.3)
        search_other = handle.create_group("slices/euv_94/searches/search_other")
        _write_grid_point(search_other, "p000000", a=0.2, b=0.3)

    report = clean_unvisited_map_store_points(path, slice_key="euv_171", delete=True)

    assert "orphan_map" in report.deleted_map_keys
    assert "b_map" not in report.deleted_map_keys
    assert "other_slice_map" not in report.deleted_map_keys
    with h5py.File(path, "r") as handle:
        remaining = set(handle["map_store/maps"].keys())
        assert "a_map" in remaining
        assert "b_map" in remaining
        assert "other_slice_map" in remaining
        assert "orphan_map" not in remaining
