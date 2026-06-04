"""In-memory index of map_store entries for one spectral slice (a, b, q0 → ref)."""

from __future__ import annotations

import json
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
    _synthetic_registry_entries_for_descriptor,
    decode_scalar,
)
from .grid_points import GRID_POINTS_GROUP, _load_grid_point_trials

_STOKES_I_COMPONENTS = frozenset({"stokes_i", "stokes_i_raw", ""})


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


def _index_map_store_maps(h5_file: h5py.File, index: SliceMapIndex, descriptor: dict[str, Any]) -> int:
    if MAP_STORE_GROUP not in h5_file or MAP_STORE_MAPS_GROUP not in h5_file[MAP_STORE_GROUP]:
        return 0
    added = 0
    maps_group = h5_file[MAP_STORE_GROUP][MAP_STORE_MAPS_GROUP]
    for map_id in maps_group.keys():
        map_group = maps_group[map_id]
        if "identity_json" not in map_group:
            continue
        try:
            identity = json.loads(decode_scalar(map_group["identity_json"][()]))
        except Exception:
            continue
        if not isinstance(identity, dict) or not _identity_matches_slice(identity, descriptor):
            continue
        a_value = _optional_float(identity.get("a"))
        b_value = _optional_float(identity.get("b"))
        q0_value = _optional_float(identity.get("q0"))
        if a_value is None or b_value is None or q0_value is None:
            continue
        component = str(identity.get("component") or "").strip().lower()
        if not component:
            array_name = str(identity.get("array_name") or "").lower()
            if "raw_modeled" in array_name:
                component = "stokes_i"
            else:
                continue
        if not _component_accepts_warm_start(component):
            continue
        ref_path = f"/{MAP_STORE_GROUP}/{MAP_STORE_MAPS_GROUP}/{map_id}"
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
        if not _component_accepts_warm_start(component) and "raw" not in map_role:
            continue
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


def build_slice_map_index(
    artifact_h5: Path,
    *,
    slice_key: str,
) -> SliceMapIndex:
    """Scan the artifact once and build (a, b, q0) → map_store ref for one slice."""
    path = Path(artifact_h5)
    with _H5PY_FILE(path, "r") as h5_file:
        descriptor = _load_slice_descriptor(h5_file, str(slice_key))
        index = SliceMapIndex(slice_key=str(slice_key), descriptor=descriptor)
        _index_map_store_maps(h5_file, index, descriptor)
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
