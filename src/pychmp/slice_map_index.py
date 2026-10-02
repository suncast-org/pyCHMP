"""In-memory index of map_store entries for one spectral slice (a, b, q0 → ref)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .ab_scan_artifacts import (
    MAP_STORE_GROUP,
    MAP_STORE_MAPS_GROUP,
    MAP_STORE_SYNTHETIC_REGISTRY_GROUP,
    SEARCHES_GROUP,
    SLICE_CONTAINER_GROUP,
    _H5PY_FILE,
    _auxiliary_channel_token_from_descriptor,
    _derive_display_maps_from_raw,
    _optional_float,
    _read_map_store_ref_array,
    _resolve_slice_group,
    _sanitize_slice_token,
    _synthetic_registry_entries_for_descriptor,
    decode_scalar,
)
from .grid_points import GRID_POINTS_GROUP, _load_grid_point_trials

_STOKES_I_COMPONENTS = frozenset({"stokes_i", "stokes_i_raw", ""})
_MAP_INDEX_PROGRESS_EVERY = 5000
_MAP_INDEX_PROGRESS_INTERVAL_S = 5.0
MAP_STORE_SLICE_INDEX_GROUP = "slice_index"
_SLICE_INDEX_SCHEMA = "pychmp.map_store_slice_index.v1"
# One full variable-length read of a half-million-row index takes tens of seconds
# and holds the HDF5 lock the viewer also needs. Chunks stay on the order of a
# single dataset read.
_INDEX_READ_CHUNK = 16384


@dataclass(frozen=True)
class MapStoreEntry:
    """One stored simulation map for a grid point and q0."""

    a: float
    b: float
    q0: float
    raw_map_ref: str
    component: str = "stokes_i"
    source_search_id: str | None = None
    source_point_id: str | None = None


def _q0_key(q0: float) -> float:
    return float(q0)


def _ab_key(a: float, b: float) -> tuple[float, float]:
    return (float(a), float(b))


def _identity_matches_slice(identity: dict[str, Any], descriptor: dict[str, Any]) -> bool:
    target_domain = str(descriptor.get("domain", "")).strip().lower()
    identity_domain = str(
        identity.get("domain") or identity.get("spectral_domain") or ""
    ).strip().lower()
    if target_domain and identity_domain and identity_domain != target_domain:
        return False

    if target_domain == "mw":
        target_freq = _optional_float(descriptor.get("frequency_ghz"))
        if target_freq is None:
            return True
        identity_channel = str(identity.get("channel_or_frequency") or "").strip().lower()
        if not identity_channel:
            identity_freq = _optional_float(identity.get("frequency_ghz"))
            if identity_freq is None:
                return True
            return np.isclose(float(identity_freq), float(target_freq), rtol=0.0, atol=1e-6)
        try:
            if "ghz" in identity_channel:
                parsed = float(identity_channel.replace("ghz", "").strip())
                return np.isclose(parsed, float(target_freq), rtol=0.0, atol=1e-6)
        except Exception:
            pass
        return True

    target_channel = _auxiliary_channel_token_from_descriptor(descriptor)
    identity_channel = str(identity.get("channel_or_frequency") or "").strip().lower()
    if target_channel and identity_channel:
        return identity_channel == target_channel
    if target_domain in {"euv", "uv"}:
        return False
    return True


def _component_accepts_warm_start(component: str) -> bool:
    lowered = str(component or "").strip().lower()
    if lowered in _STOKES_I_COMPONENTS:
        return True
    return lowered == "stokes_i" or "raw" in lowered or lowered.endswith("_raw_modeled")


class SliceMapIndex:
    """Read-only (a, b) → q0 → map_store ref index for one slice."""

    def __init__(self, *, slice_key: str, descriptor: dict[str, Any]) -> None:
        self.slice_key = str(slice_key).strip()
        self.descriptor = dict(descriptor)
        self._entries: dict[tuple[float, float], dict[float, MapStoreEntry]] = {}
        self._component_entries: dict[tuple[float, float], dict[float, dict[str, MapStoreEntry]]] = {}

    def register(
        self,
        *,
        a: float,
        b: float,
        q0: float,
        raw_map_ref: str,
        component: str = "stokes_i",
        source_search_id: str | None = None,
        source_point_id: str | None = None,
    ) -> None:
        ref = str(raw_map_ref or "").strip()
        if not ref or not np.isfinite(float(q0)) or float(q0) <= 0.0:
            return
        if not _component_accepts_warm_start(component):
            return
        key = _ab_key(a, b)
        q0_norm = _q0_key(q0)
        bucket = self._entries.setdefault(key, {})
        existing = bucket.get(q0_norm)
        entry = MapStoreEntry(
            a=float(a),
            b=float(b),
            q0=q0_norm,
            raw_map_ref=ref,
            component=str(component),
            source_search_id=source_search_id,
            source_point_id=source_point_id,
        )
        component_key = str(component or "").strip().lower()
        component_bucket = self._component_entries.setdefault(key, {}).setdefault(q0_norm, {})
        existing_component = component_bucket.get(component_key)
        if existing_component is None or len(ref) >= len(existing_component.raw_map_ref):
            component_bucket[component_key] = entry
        if _component_accepts_warm_start(component_key):
            if existing is None or len(ref) >= len(existing.raw_map_ref):
                bucket[q0_norm] = entry

    def has_point(self, a: float, b: float) -> bool:
        return _ab_key(a, b) in self._entries

    def q0_values(self, a: float, b: float) -> tuple[float, ...]:
        bucket = self._entries.get(_ab_key(a, b))
        if not bucket:
            return ()
        return tuple(sorted(bucket.keys()))

    def raw_map_ref(self, a: float, b: float, q0: float) -> str | None:
        bucket = self._entries.get(_ab_key(a, b))
        if not bucket:
            return None
        entry = bucket.get(_q0_key(q0))
        if entry is None:
            for candidate_q0, candidate in bucket.items():
                if np.isclose(candidate_q0, float(q0), rtol=0.0, atol=1e-12):
                    return candidate.raw_map_ref
            return None
        return entry.raw_map_ref

    def entries_for_point(self, a: float, b: float) -> tuple[MapStoreEntry, ...]:
        bucket = self._entries.get(_ab_key(a, b))
        if not bucket:
            return ()
        return tuple(bucket[q0] for q0 in sorted(bucket.keys()))

    def component_ref(self, a: float, b: float, q0: float, component: str) -> str | None:
        bucket = self._component_entries.get(_ab_key(a, b))
        if not bucket:
            return None
        q0_bucket = bucket.get(_q0_key(q0))
        if q0_bucket is None:
            for candidate_q0, candidate_bucket in bucket.items():
                if np.isclose(candidate_q0, float(q0), rtol=0.0, atol=1e-12):
                    q0_bucket = candidate_bucket
                    break
        if not q0_bucket:
            return None
        entry = q0_bucket.get(str(component or "").strip().lower())
        return None if entry is None else entry.raw_map_ref

    def point_count(self) -> int:
        return len(self._entries)

    def trial_count(self) -> int:
        return sum(len(bucket) for bucket in self._entries.values())

    def summary(self) -> str:
        return f"{self.trial_count()} map(s) across {self.point_count()} (a,b) point(s)"

    def point_keys(self) -> tuple[tuple[float, float], ...]:
        return tuple(sorted(self._entries.keys()))


def _load_slice_descriptor(h5_file: h5py.File, slice_key: str) -> dict[str, Any]:
    slice_group, descriptors, _selected = _resolve_slice_group(
        h5_file,
        slice_key=str(slice_key).strip(),
        allow_missing=False,
    )
    descriptor = next(
        (dict(item) for item in descriptors if str(item.get("key")) == str(slice_key)),
        None,
    )
    if descriptor is not None:
        return descriptor
    if "common" in slice_group and "diagnostics_json" in slice_group["common"]:
        try:
            diagnostics = json.loads(decode_scalar(slice_group["common"]["diagnostics_json"][()]))
            if isinstance(diagnostics, dict):
                return {
                    "key": str(slice_key),
                    "domain": diagnostics.get("spectral_domain", "unknown"),
                    "label": diagnostics.get("spectral_label", slice_key),
                    "frequency_ghz": diagnostics.get("frequency_ghz"),
                    "wavelength_angstrom": diagnostics.get("wavelength_angstrom"),
                    "channel_label": diagnostics.get("euv_channel"),
                }
        except Exception:
            pass
    return {"key": str(slice_key), "domain": "unknown", "label": str(slice_key)}


def _report_map_index_progress(
    visited: int,
    total: int,
    progress: Callable[[int, int], None] | None,
) -> None:
    """Report walk progress. The default line is flushed so a redirected log updates."""
    if progress is not None:
        progress(visited, total)
        return
    print(f"Map store index: visited {visited}/{total} maps", flush=True)


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if value is None:
        return ""
    return str(value)


def _h5_writable(h5_file: h5py.File) -> bool:
    return str(getattr(h5_file, "mode", "r")) not in {"r"}


def _slice_id_from_map_identity(identity: dict[str, Any]) -> str:
    """Canonical slice key for one map, derived from domain and channel only."""
    domain = str(identity.get("domain") or identity.get("spectral_domain") or "").strip().lower()
    channel = str(identity.get("channel_or_frequency") or "").strip().lower()
    if domain == "mw":
        freq = _optional_float(channel[:-3]) if channel.endswith("ghz") else None
        if freq is None:
            freq = _optional_float(identity.get("frequency_ghz"))
        if freq is None:
            return ""
        return f"mw_{float(freq):.6f}ghz".replace(".", "p")
    if domain in {"euv", "uv"} and channel:
        return f"{domain}_{_sanitize_slice_token(channel)}"
    return ""


def _component_for_index(identity: dict[str, Any]) -> str:
    component = str(identity.get("component") or "").strip().lower()
    if component:
        return component
    array_name = str(identity.get("array_name") or "").lower()
    if "raw_modeled" in array_name:
        return "stokes_i"
    return ""


def _read_map_group_identity(map_group: h5py.Group) -> dict[str, Any] | None:
    layer_keys = {"domain", "channel_or_frequency", "component", "a", "b", "q0"}
    layer: dict[str, Any] = {}
    if "map_layer_json" in map_group:
        try:
            parsed = json.loads(decode_scalar(map_group["map_layer_json"][()]))
            if isinstance(parsed, dict):
                layer = {k: v for k, v in parsed.items() if k in layer_keys}
        except Exception:
            pass
    if layer_keys <= layer.keys():
        return layer
    if "identity_json" not in map_group:
        return None
    try:
        identity = json.loads(decode_scalar(map_group["identity_json"][()]))
    except Exception:
        return None
    if not isinstance(identity, dict):
        return None
    return {**identity, **layer}


def _index_row(
    identity: dict[str, Any] | None,
    map_key: str,
    *,
    descriptor: dict[str, Any] | None,
    requested_slice_key: str,
) -> tuple[str, float, float, float, str, str]:
    """One compact record: slice id, a, b, q0, map key, component."""
    if not isinstance(identity, dict):
        return ("", np.nan, np.nan, np.nan, str(map_key), "")
    a_value = _optional_float(identity.get("a"))
    b_value = _optional_float(identity.get("b"))
    q0_value = _optional_float(identity.get("q0"))
    component = _component_for_index(identity)
    slice_id = ""
    if component and a_value is not None and b_value is not None and q0_value is not None:
        slice_id = _slice_id_from_map_identity(identity)
        if (
            descriptor is not None
            and _identity_matches_slice(identity, descriptor)
            and slice_id in {"", requested_slice_key}
        ):
            slice_id = requested_slice_key
    return (
        slice_id,
        np.nan if a_value is None else float(a_value),
        np.nan if b_value is None else float(b_value),
        np.nan if q0_value is None else float(q0_value),
        str(map_key),
        component,
    )


def _slice_index_group(h5_file: h5py.File) -> h5py.Group | None:
    store = h5_file.get(MAP_STORE_GROUP)
    if store is None or MAP_STORE_SLICE_INDEX_GROUP not in store:
        return None
    group = store[MAP_STORE_SLICE_INDEX_GROUP]
    if not isinstance(group, h5py.Group):
        return None
    if _decode_text(group.attrs.get("schema", "")) != _SLICE_INDEX_SCHEMA:
        return None
    if "slice_id" not in group or "map_key" not in group:
        return None
    return group


def _map_store_count(h5_file: h5py.File) -> int | None:
    store = h5_file.get(MAP_STORE_GROUP)
    if store is None or MAP_STORE_MAPS_GROUP not in store:
        return None
    return int(len(store[MAP_STORE_MAPS_GROUP]))


def _slice_index_is_current(h5_file: h5py.File) -> bool:
    count = _map_store_count(h5_file)
    group = _slice_index_group(h5_file)
    if count is None or group is None:
        return False
    return int(group["slice_id"].shape[0]) == count


def _column_text(dataset: h5py.Dataset) -> list[str]:
    if int(dataset.shape[0]) == 0:
        return []
    values = dataset.asstr()[()] if hasattr(dataset, "asstr") else dataset[()]
    if np.ndim(values) == 0:
        return [_decode_text(values)]
    return [_decode_text(value) for value in values]


def _write_slice_index_rows(h5_file: h5py.File, rows: list[tuple[str, float, float, float, str, str]]) -> None:
    store = h5_file.require_group(MAP_STORE_GROUP)
    if MAP_STORE_SLICE_INDEX_GROUP in store:
        del store[MAP_STORE_SLICE_INDEX_GROUP]
    group = store.create_group(MAP_STORE_SLICE_INDEX_GROUP)
    group.attrs["schema"] = _SLICE_INDEX_SCHEMA
    text_dtype = h5py.string_dtype(encoding="utf-8")
    if not rows:
        for name in ("slice_id", "map_key", "component"):
            group.create_dataset(name, shape=(0,), maxshape=(None,), dtype=text_dtype)
        for name in ("a", "b", "q0"):
            group.create_dataset(name, shape=(0,), maxshape=(None,), dtype=np.float64)
        return
    columns: dict[str, Any] = {
        "slice_id": np.array([row[0] for row in rows], dtype=object),
        "a": np.array([row[1] for row in rows], dtype=np.float64),
        "b": np.array([row[2] for row in rows], dtype=np.float64),
        "q0": np.array([row[3] for row in rows], dtype=np.float64),
        "map_key": np.array([row[4] for row in rows], dtype=object),
        "component": np.array([row[5] for row in rows], dtype=object),
    }
    for name, values in columns.items():
        if name in {"a", "b", "q0"}:
            group.create_dataset(name, data=values, maxshape=(None,), dtype=np.float64)
        else:
            group.create_dataset(name, data=values, maxshape=(None,), dtype=text_dtype)


def _append_slice_index_row(
    h5_file: h5py.File,
    row: tuple[str, float, float, float, str, str],
) -> None:
    group = _slice_index_group(h5_file)
    if group is None:
        return
    n = int(group["slice_id"].shape[0])
    values = {
        "slice_id": row[0],
        "a": row[1],
        "b": row[2],
        "q0": row[3],
        "map_key": row[4],
        "component": row[5],
    }
    for name, value in values.items():
        dataset = group[name]
        dataset.resize((n + 1,))
        dataset[n] = value


def append_map_store_slice_index(
    h5_file: h5py.File,
    *,
    identity: dict[str, Any],
    map_key: str,
) -> None:
    """Append one record after a new map is saved, when the index was already current."""
    count = _map_store_count(h5_file)
    group = _slice_index_group(h5_file)
    if count is None or group is None:
        return
    if int(group["slice_id"].shape[0]) != count - 1:
        return
    _append_slice_index_row(
        h5_file,
        _index_row(identity, map_key, descriptor=None, requested_slice_key=""),
    )


def _register_index_row(index: SliceMapIndex, row: tuple[str, float, float, float, str, str]) -> bool:
    slice_id, a_value, b_value, q0_value, map_key, component = row
    if str(slice_id) != index.slice_key or not component or not map_key:
        return False
    before = index.trial_count()
    index.register(
        a=float(a_value),
        b=float(b_value),
        q0=float(q0_value),
        raw_map_ref=f"/{MAP_STORE_GROUP}/{MAP_STORE_MAPS_GROUP}/{map_key}",
        component=str(component),
    )
    return index.trial_count() > before


def _register_slice_from_saved_index(h5_file: h5py.File, index: SliceMapIndex) -> int:
    group = _slice_index_group(h5_file)
    if group is None:
        return 0
    slice_ids = _column_text(group["slice_id"])
    selected = [i for i, slice_id in enumerate(slice_ids) if slice_id == index.slice_key]
    if not selected:
        return 0
    picker = np.asarray(selected, dtype=np.int64)
    a_values = np.asarray(group["a"][picker], dtype=float)
    b_values = np.asarray(group["b"][picker], dtype=float)
    q0_values = np.asarray(group["q0"][picker], dtype=float)
    map_keys = [_decode_text(value) for value in group["map_key"].asstr()[picker]]
    components = [_decode_text(value) for value in group["component"].asstr()[picker]]
    added = 0
    for offset, row_index in enumerate(selected):
        if _register_index_row(
            index,
            (
                slice_ids[row_index],
                float(a_values[offset]),
                float(b_values[offset]),
                float(q0_values[offset]),
                map_keys[offset],
                components[offset],
            ),
        ):
            added += 1
    return added


def _index_map_store_maps(
    h5_file: h5py.File,
    index: SliceMapIndex,
    descriptor: dict[str, Any],
    *,
    progress: Callable[[int, int], None] | None = None,
    progress_every: int = _MAP_INDEX_PROGRESS_EVERY,
    progress_interval_s: float = _MAP_INDEX_PROGRESS_INTERVAL_S,
    _clock: Callable[[], float] | None = None,
) -> int:
    if MAP_STORE_GROUP not in h5_file or MAP_STORE_MAPS_GROUP not in h5_file[MAP_STORE_GROUP]:
        return 0
    if _slice_index_is_current(h5_file):
        return _register_slice_from_saved_index(h5_file, index)
    added = 0
    maps_group = h5_file[MAP_STORE_GROUP][MAP_STORE_MAPS_GROUP]
    # Group length is the stored link count, not a second walk of every map.
    total = int(len(maps_group))
    visited = 0
    now = _clock or time.monotonic
    last_report_at = now()
    last_reported = 0
    rows: list[tuple[str, float, float, float, str, str]] = []
    if progress is None and total:
        print(
            f"Building map metadata index ({index.slice_key}); no maps are being rescored...",
            flush=True,
        )

    def _maybe_report() -> None:
        nonlocal last_report_at, last_reported
        stamp = now()
        count_due = progress_every > 0 and visited % progress_every == 0
        time_due = progress_interval_s > 0 and (stamp - last_report_at) >= progress_interval_s
        if not count_due and not time_due:
            return
        _report_map_index_progress(visited, total, progress)
        last_report_at = stamp
        last_reported = visited

    for map_id in maps_group.keys():
        visited += 1
        try:
            identity = _read_map_group_identity(maps_group[map_id])
            row = _index_row(
                identity,
                str(map_id),
                descriptor=descriptor,
                requested_slice_key=index.slice_key,
            )
            rows.append(row)
            registered = _register_index_row(index, row)
            if (
                not registered
                and isinstance(identity, dict)
                and _identity_matches_slice(identity, descriptor)
            ):
                registered = _register_index_row(
                    index,
                    (index.slice_key, row[1], row[2], row[3], row[4], row[5]),
                )
            if registered:
                added += 1
        finally:
            _maybe_report()
    if visited and visited != last_reported:
        _report_map_index_progress(visited, total, progress)
    if _h5_writable(h5_file):
        _write_slice_index_rows(h5_file, rows)
    return added


def _index_synthetic_registry(h5_file: h5py.File, index: SliceMapIndex, descriptor: dict[str, Any]) -> int:
    added = 0
    entries = _synthetic_registry_entries_for_descriptor(h5_file, descriptor=descriptor)
    for machine_key, entry in entries.items():
        ref_path = str(entry.get("map_ref_path") or "").strip()
        if not ref_path:
            continue
        identity = entry.get("identity") if isinstance(entry.get("identity"), dict) else {}
        a_value = _optional_float(identity.get("a"))
        b_value = _optional_float(identity.get("b"))
        q0_value = _optional_float(identity.get("q0"))
        if a_value is None or b_value is None or q0_value is None:
            continue
        map_role = str(entry.get("map_role") or "").strip().lower()
        component = str(entry.get("component") or map_role or "stokes_i").strip().lower()
        before = index.trial_count()
        index.register(
            a=float(a_value),
            b=float(b_value),
            q0=float(q0_value),
            raw_map_ref=ref_path,
            component=component,
        )
        if index.trial_count() > before:
            added += 1
    return added


def _index_grid_trials_on_slice(h5_file: h5py.File, index: SliceMapIndex, slice_key: str) -> int:
    if SLICE_CONTAINER_GROUP not in h5_file or slice_key not in h5_file[SLICE_CONTAINER_GROUP]:
        return 0
    slice_group = h5_file[SLICE_CONTAINER_GROUP][slice_key]
    if SEARCHES_GROUP not in slice_group:
        return 0
    added = 0
    searches = slice_group[SEARCHES_GROUP]
    for search_id in sorted(str(key) for key in searches.keys()):
        search_group = searches[search_id]
        if GRID_POINTS_GROUP not in search_group:
            continue
        grid_group = search_group[GRID_POINTS_GROUP]
        for point_id in sorted(str(key) for key in grid_group.keys()):
            point_group = grid_group[point_id]
            try:
                header_a = float(point_group.attrs["a"])
                header_b = float(point_group.attrs["b"])
            except Exception:
                continue
            for trial in _load_grid_point_trials(point_group, include_maps=False):
                raw_ref = str(trial.get("raw_map_ref", "") or "").strip()
                if not raw_ref:
                    continue
                before = index.trial_count()
                index.register(
                    a=float(header_a),
                    b=float(header_b),
                    q0=float(trial["q0"]),
                    raw_map_ref=raw_ref,
                    component="stokes_i",
                    source_search_id=str(search_id),
                    source_point_id=str(point_id),
                )
                if index.trial_count() > before:
                    added += 1
    return added


def _finite_ab_points(a_values: np.ndarray, b_values: np.ndarray) -> set[tuple[float, float]]:
    """Every finite ``(a, b)`` in these rows. Does not touch map arrays."""
    a_arr = np.asarray(a_values, dtype=float).reshape(-1)
    b_arr = np.asarray(b_values, dtype=float).reshape(-1)
    if a_arr.size == 0 or b_arr.size != a_arr.size:
        return set()
    keep = np.isfinite(a_arr) & np.isfinite(b_arr)
    if not np.any(keep):
        return set()
    pairs = np.unique(np.stack((a_arr[keep], b_arr[keep]), axis=1), axis=0)
    return {(float(pair[0]), float(pair[1])) for pair in pairs}


def _readable_float_prefix(dataset: h5py.Dataset, start: int, stop: int) -> tuple[int, np.ndarray | None]:
    """Float values for the readable prefix of ``dataset[start:stop]``."""
    if stop <= start:
        return start, None
    try:
        values = np.asarray(dataset[slice(start, stop)], dtype=float).reshape(-1)
    except OSError:
        values = None
    if values is not None and int(values.size) == stop - start:
        return stop, values
    if stop <= start + 1:
        return start, None
    mid = start + (stop - start) // 2
    lower_end, lower_values = _readable_float_prefix(dataset, start, mid)
    if lower_end < mid:
        return lower_end, lower_values
    upper_end, upper_values = _readable_float_prefix(dataset, mid, stop)
    if upper_end <= mid or upper_values is None:
        return lower_end, lower_values
    if lower_values is None:
        return upper_end, upper_values
    return upper_end, np.concatenate((lower_values, upper_values))


def slice_index_ab_snapshot(
    artifact_h5: Path,
    slice_key: str = "",
    *,
    after_row: int = 0,
) -> tuple[set[tuple[float, float]], int]:
    """Every finite ``(a, b)`` in the slice index, plus the row count.

    ``slice_key`` is ignored. One render stores every channel, so the set is
    shared by every heatmap. Reads ``a`` and ``b`` only, one stored chunk at a
    time. ``after_row`` skips rows already cached. Does not walk ``map_store/maps``
    or open map arrays.
    """
    del slice_key
    start = max(0, int(after_row))
    points: set[tuple[float, float]] = set()
    pos = start
    nrows = start
    while True:
        try:
            h5_file = _H5PY_FILE(Path(artifact_h5), "r")
        except OSError:
            return points, pos
        try:
            with h5_file:
                group = _slice_index_group(h5_file)
                if group is None or "a" not in group or "b" not in group:
                    return points, 0
                nrows = int(group["a"].shape[0])
                if pos >= nrows:
                    return points, nrows
                stored_chunk = group["a"].chunks
                step = max(1, int(_INDEX_READ_CHUNK))
                if stored_chunk:
                    step = min(step, max(1, int(stored_chunk[0])))
                stop = min(pos + step, nrows)
                trusted, a_values = _readable_float_prefix(group["a"], pos, stop)
                if trusted <= pos or a_values is None:
                    return points, pos
                try:
                    b_values = np.asarray(group["b"][slice(pos, trusted)], dtype=float).reshape(-1)
                except OSError:
                    return points, pos
                points.update(_finite_ab_points(a_values, b_values))
                pos = trusted
                if trusted < stop or pos >= nrows:
                    return points, nrows if pos >= nrows else pos
        except OSError:
            return points, pos


def slice_index_ab_points(artifact_h5: Path, slice_key: str = "") -> set[tuple[float, float]]:
    """Every finite ``(a, b)`` already indexed. ``slice_key`` is ignored."""
    points, _nrows = slice_index_ab_snapshot(artifact_h5, slice_key, after_row=0)
    return points


def build_slice_map_index(
    artifact_h5: Path,
    *,
    slice_key: str,
    progress: Callable[[int, int], None] | None = None,
    progress_every: int = _MAP_INDEX_PROGRESS_EVERY,
    progress_interval_s: float = _MAP_INDEX_PROGRESS_INTERVAL_S,
) -> SliceMapIndex:
    """Scan the artifact once and build (a, b, q0) → map_store ref for one slice."""
    path = Path(artifact_h5)
    try:
        h5_file = _H5PY_FILE(path, "r+")
    except OSError:
        h5_file = _H5PY_FILE(path, "r")
    with h5_file:
        descriptor = _load_slice_descriptor(h5_file, str(slice_key))
        index = SliceMapIndex(slice_key=str(slice_key), descriptor=descriptor)
        _index_map_store_maps(
            h5_file,
            index,
            descriptor,
            progress=progress,
            progress_every=progress_every,
            progress_interval_s=progress_interval_s,
        )
        _index_synthetic_registry(h5_file, index, descriptor)
        _index_grid_trials_on_slice(h5_file, index, str(slice_key))
    return index


def load_render_pair_from_index(
    artifact_h5: Path,
    *,
    index: SliceMapIndex,
    a: float,
    b: float,
    q0: float,
    observed_template: np.ndarray,
    psf_kernel: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load one (a, b, q0) pair from map_store without gxrender."""
    ref = index.raw_map_ref(float(a), float(b), float(q0))
    if not ref:
        return None
    with _H5PY_FILE(Path(artifact_h5), "r") as h5_file:
        raw_modeled = _read_map_store_ref_array(h5_file, ref)
        if raw_modeled is None:
            return None
        raw_display, modeled, _residual, _has_raw = _derive_display_maps_from_raw(
            raw_modeled,
            observed_template=np.asarray(observed_template, dtype=float),
            psf_kernel=psf_kernel,
        )
        if modeled is None:
            return None
        raw_arr = np.asarray(raw_display if raw_display is not None else raw_modeled, dtype=np.float32)
        return raw_arr, np.asarray(modeled, dtype=np.float32)


def hydrate_render_caches_from_index(
    artifact_h5: Path,
    *,
    index: SliceMapIndex,
    a: float,
    b: float,
    raw_modeled_by_q0: dict[str, Any],
    modeled_by_q0: dict[str, Any],
    observed_template: np.ndarray,
    psf_kernel: np.ndarray | None,
) -> int:
    """Load all indexed q0 maps for (a, b) into render caches without gxrender."""
    if not index.has_point(a, b):
        return 0
    hydrated = 0
    with _H5PY_FILE(Path(artifact_h5), "r") as h5_file:
        for entry in index.entries_for_point(a, b):
            key = f"{float(entry.q0):.17g}"
            if key in raw_modeled_by_q0:
                continue
            raw_modeled = _read_map_store_ref_array(h5_file, entry.raw_map_ref)
            if raw_modeled is None:
                continue
            _raw_display, modeled, _residual, _has_raw = _derive_display_maps_from_raw(
                raw_modeled,
                observed_template=np.asarray(observed_template, dtype=float),
                psf_kernel=psf_kernel,
            )
            if modeled is None:
                continue
            raw_arr = np.asarray(_raw_display if _raw_display is not None else raw_modeled, dtype=np.float32)
            modeled_arr = np.asarray(modeled, dtype=np.float32)
            raw_modeled_by_q0[key] = raw_arr
            modeled_by_q0[key] = modeled_arr
            hydrated += 1
    return hydrated
