"""Report (and optionally delete) map_store (a, b) points no search visited."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

from .ab_scan_artifacts import (
    MAP_STORE_GROUP,
    MAP_STORE_MAPS_GROUP,
    SEARCHES_GROUP,
    SLICE_CONTAINER_GROUP,
    _H5PY_FILE,
)
from .grid_points import GRID_POINT_AB_TOLERANCE, GRID_POINTS_GROUP, list_grid_point_headers
from .slice_map_index import (
    MAP_STORE_SLICE_INDEX_GROUP,
    _optional_float,
    _read_map_group_identity,
    _write_slice_index_rows,
)


def _ab_token(a: float, b: float, *, decimals: int = 6) -> tuple[float, float]:
    return (round(float(a), decimals), round(float(b), decimals))


def _near(a: float, b: float, other_a: float, other_b: float, *, tol: float = GRID_POINT_AB_TOLERANCE) -> bool:
    return abs(float(a) - float(other_a)) <= tol and abs(float(b) - float(other_b)) <= tol


def _point_in_set(a: float, b: float, points: set[tuple[float, float]], *, tol: float = GRID_POINT_AB_TOLERANCE) -> bool:
    token = _ab_token(a, b)
    if token in points:
        return True
    for other_a, other_b in points:
        if _near(a, b, other_a, other_b, tol=tol):
            return True
    return False


@dataclass
class SearchVisitReport:
    slice_key: str
    search_id: str
    visited_ab_count: int
    store_ab_count: int
    unvisited_store_ab_count: int
    unvisited_store_abs: list[tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slice_key": self.slice_key,
            "search_id": self.search_id,
            "visited_ab_count": self.visited_ab_count,
            "store_ab_count": self.store_ab_count,
            "unvisited_store_ab_count": self.unvisited_store_ab_count,
            "unvisited_store_abs": [[float(a), float(b)] for a, b in self.unvisited_store_abs],
        }


@dataclass
class MapStoreCleanerReport:
    artifact_path: str
    dry_run: bool
    store_ab_count: int = 0
    visited_ab_count: int = 0
    orphan_ab_count: int = 0
    orphan_map_count: int = 0
    deleted_map_count: int = 0
    orphan_abs: list[tuple[float, float]] = field(default_factory=list)
    orphan_map_keys: list[str] = field(default_factory=list)
    deleted_map_keys: list[str] = field(default_factory=list)
    per_search: list[SearchVisitReport] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_path": self.artifact_path,
            "dry_run": self.dry_run,
            "store_ab_count": self.store_ab_count,
            "visited_ab_count": self.visited_ab_count,
            "orphan_ab_count": self.orphan_ab_count,
            "orphan_map_count": self.orphan_map_count,
            "deleted_map_count": self.deleted_map_count,
            "orphan_abs": [[float(a), float(b)] for a, b in self.orphan_abs],
            "orphan_map_keys": list(self.orphan_map_keys),
            "deleted_map_keys": list(self.deleted_map_keys),
            "per_search": [item.to_dict() for item in self.per_search],
        }


def _iter_search_groups(
    h5_file: h5py.File,
    *,
    slice_key: str | None = None,
    search_id: str | None = None,
) -> list[tuple[str, str, h5py.Group]]:
    if SLICE_CONTAINER_GROUP not in h5_file:
        return []
    slices = h5_file[SLICE_CONTAINER_GROUP]
    out: list[tuple[str, str, h5py.Group]] = []
    for name in sorted(str(key) for key in slices.keys()):
        if slice_key and name != str(slice_key):
            continue
        slice_group = slices[name]
        if SEARCHES_GROUP not in slice_group:
            continue
        searches = slice_group[SEARCHES_GROUP]
        for sid in sorted(str(key) for key in searches.keys()):
            if search_id and sid != str(search_id):
                continue
            out.append((name, sid, searches[sid]))
    return out


def _visited_ab_for_search(search_group: h5py.Group) -> set[tuple[float, float]]:
    points: set[tuple[float, float]] = set()
    if GRID_POINTS_GROUP not in search_group:
        return points
    for header in list_grid_point_headers(search_group):
        a_value = _optional_float(header.get("a"))
        b_value = _optional_float(header.get("b"))
        if a_value is None or b_value is None:
            continue
        points.add(_ab_token(a_value, b_value))
    return points


def _map_store_entries(h5_file: h5py.File) -> list[tuple[str, float, float]]:
    """Return ``(map_key, a, b)`` for every readable map_store entry."""
    store = h5_file.get(MAP_STORE_GROUP)
    if store is None or MAP_STORE_MAPS_GROUP not in store:
        return []
    maps_group = store[MAP_STORE_MAPS_GROUP]
    entries: list[tuple[str, float, float]] = []
    for map_key in sorted(str(key) for key in maps_group.keys()):
        identity = _read_map_group_identity(maps_group[map_key])
        if not isinstance(identity, dict):
            continue
        a_value = _optional_float(identity.get("a"))
        b_value = _optional_float(identity.get("b"))
        if a_value is None or b_value is None:
            continue
        entries.append((str(map_key), float(a_value), float(b_value)))
    return entries


def _rewrite_slice_index_without_keys(h5_file: h5py.File, deleted_keys: set[str]) -> None:
    store = h5_file.get(MAP_STORE_GROUP)
    if store is None or MAP_STORE_SLICE_INDEX_GROUP not in store:
        return
    group = store[MAP_STORE_SLICE_INDEX_GROUP]
    if "map_key" not in group or "a" not in group:
        del store[MAP_STORE_SLICE_INDEX_GROUP]
        return
    map_keys = [str(decode) for decode in _decode_index_strings(group["map_key"])]
    keep_rows: list[tuple[str, float, float, float, str, str]] = []
    for idx, map_key in enumerate(map_keys):
        if map_key in deleted_keys:
            continue
        slice_id = _index_string_at(group, "slice_id", idx)
        component = _index_string_at(group, "component", idx)
        a_value = float(np.asarray(group["a"][idx], dtype=float))
        b_value = float(np.asarray(group["b"][idx], dtype=float))
        q0_value = float(np.asarray(group["q0"][idx], dtype=float)) if "q0" in group else float("nan")
        keep_rows.append((slice_id, a_value, b_value, q0_value, map_key, component))
    del store[MAP_STORE_SLICE_INDEX_GROUP]
    if keep_rows:
        _write_slice_index_rows(h5_file, keep_rows)


def _decode_index_strings(dataset: h5py.Dataset) -> list[str]:
    values = dataset[()]
    if getattr(values, "shape", ()) == ():
        values = [values]
    out: list[str] = []
    for raw in values:
        if isinstance(raw, bytes):
            out.append(raw.decode("utf-8", errors="replace"))
        else:
            out.append(str(raw))
    return out


def _index_string_at(group: h5py.Group, name: str, idx: int) -> str:
    if name not in group:
        return ""
    raw = group[name][idx]
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def report_unvisited_map_store_points(
    artifact_h5: Path,
    *,
    slice_key: str | None = None,
    search_id: str | None = None,
) -> MapStoreCleanerReport:
    """Report map_store (a, b) points that searches never visited (dry-run friendly)."""
    path = Path(artifact_h5)
    report = MapStoreCleanerReport(artifact_path=str(path), dry_run=True)
    with _H5PY_FILE(path, "r") as h5_file:
        entries = _map_store_entries(h5_file)
        store_abs = {_ab_token(a, b) for _key, a, b in entries}
        report.store_ab_count = int(len(store_abs))

        visited_union: set[tuple[float, float]] = set()
        search_visits: list[tuple[str, str, set[tuple[float, float]]]] = []
        for sk, sid, search_group in _iter_search_groups(h5_file, slice_key=slice_key, search_id=search_id):
            visited = _visited_ab_for_search(search_group)
            visited_union.update(visited)
            search_visits.append((sk, sid, visited))

        # When filtering to one search, still gather global visited for orphan safety
        # unless the caller also scoped the slice. Orphans always use the full union
        # of matching searches in this pass.
        report.visited_ab_count = int(len(visited_union))

        orphan_abs = sorted(
            ab for ab in store_abs if not _point_in_set(ab[0], ab[1], visited_union)
        )
        report.orphan_abs = orphan_abs
        report.orphan_ab_count = int(len(orphan_abs))
        orphan_keys = [
            key
            for key, a, b in entries
            if not _point_in_set(a, b, visited_union)
        ]
        report.orphan_map_keys = orphan_keys
        report.orphan_map_count = int(len(orphan_keys))

        for sk, sid, visited in search_visits:
            unvisited = sorted(ab for ab in store_abs if not _point_in_set(ab[0], ab[1], visited))
            report.per_search.append(
                SearchVisitReport(
                    slice_key=sk,
                    search_id=sid,
                    visited_ab_count=int(len(visited)),
                    store_ab_count=int(len(store_abs)),
                    unvisited_store_ab_count=int(len(unvisited)),
                    unvisited_store_abs=unvisited,
                )
            )
    return report


def clean_unvisited_map_store_points(
    artifact_h5: Path,
    *,
    slice_key: str | None = None,
    search_id: str | None = None,
    delete: bool = False,
) -> MapStoreCleanerReport:
    """Report unvisited map_store points; delete orphans only when ``delete`` is True.

    Dry-run is the default: ``delete=False`` never modifies the artifact.
    """
    report = report_unvisited_map_store_points(
        artifact_h5,
        slice_key=slice_key,
        search_id=search_id,
    )
    report.dry_run = not bool(delete)
    if not delete or not report.orphan_map_keys:
        return report

    path = Path(artifact_h5)
    deleted_keys = set(report.orphan_map_keys)
    with _H5PY_FILE(path, "a") as h5_file:
        store = h5_file.get(MAP_STORE_GROUP)
        if store is None or MAP_STORE_MAPS_GROUP not in store:
            return report
        maps_group = store[MAP_STORE_MAPS_GROUP]
        for map_key in list(report.orphan_map_keys):
            if map_key in maps_group:
                del maps_group[map_key]
                report.deleted_map_keys.append(map_key)
        report.deleted_map_count = int(len(report.deleted_map_keys))
        _rewrite_slice_index_without_keys(h5_file, deleted_keys)
    return report


def _print_report(report: MapStoreCleanerReport) -> None:
    print(f"Artifact: {report.artifact_path}")
    print(f"Mode: {'dry-run (report only)' if report.dry_run else 'delete'}")
    print(f"Map-store distinct (a,b): {report.store_ab_count}")
    print(f"Visited distinct (a,b) across searches: {report.visited_ab_count}")
    print(f"Orphan (a,b) not visited by any matched search: {report.orphan_ab_count}")
    print(f"Orphan map_store entries: {report.orphan_map_count}")
    if report.deleted_map_count:
        print(f"Deleted map_store entries: {report.deleted_map_count}")
    if report.per_search:
        print("")
        print("Per search:")
        for item in report.per_search:
            print(
                f"  [{item.slice_key}/{item.search_id}] "
                f"visited={item.visited_ab_count} "
                f"store_unvisited={item.unvisited_store_ab_count}"
            )
    if report.orphan_abs:
        print("")
        print("Orphan (a,b) sample:")
        for a_value, b_value in report.orphan_abs[:20]:
            print(f"  a={a_value:g} b={b_value:g}")
        if len(report.orphan_abs) > 20:
            print(f"  ... and {len(report.orphan_abs) - 20} more")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pychmp-clean-map-store",
        description=(
            "Report map_store (a,b) points that no search grid visited. "
            "Dry-run by default; pass --delete to remove orphan map entries."
        ),
    )
    parser.add_argument("artifact_h5", type=Path, help="Unified artifact H5 path")
    parser.add_argument("--slice-key", default=None, help="Limit to one slice key")
    parser.add_argument("--search-id", default=None, help="Limit visit accounting to one search id")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete orphan map_store entries (off by default; dry-run otherwise)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = clean_unvisited_map_store_points(
        Path(args.artifact_h5),
        slice_key=args.slice_key,
        search_id=args.search_id,
        delete=bool(args.delete),
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        _print_report(report)
        if report.dry_run and report.orphan_map_count:
            print("")
            print("Re-run with --delete to remove orphan map_store entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
