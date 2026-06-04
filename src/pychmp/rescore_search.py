"""Map-store rescore utility: build a sidecar search, then commit into a live artifact."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
from astropy.io import fits

from .ab_scan_artifacts import (
    OBSERVATION_REF_GROUP,
    SEARCH_REQUEST_DATASET,
    SEARCHES_GROUP,
    SLICE_CONTAINER_GROUP,
    _H5PY_FILE,
    _create_text_dataset,
    _json_dumps,
    _json_loads_or_empty,
    _read_common_group,
    _set_slice_group_attrs,
    decode_scalar,
    _search_request_from_diagnostics,
    _search_lifecycle_payload,
    _write_search_lifecycle_dataset,
    find_in_progress_slice_search,
    load_search_run_profile,
    resolve_search_location,
    target_slice_descriptor_from_diagnostics,
)
from .fitting import observation_reference_to_evaluation_context
from .grid_points import (
    GRID_POINTS_CONTRACT_VERSION,
    GRID_POINTS_GROUP,
    GridPointAssignedEvent,
    GridPointCompletedEvent,
    GridPointFailedEvent,
    apply_grid_point_assigned,
    apply_grid_point_completed,
    apply_grid_point_event_with_retry,
    apply_grid_point_failed,
    list_grid_point_headers,
)
from .obs_preprocessing import SliceObservationReference
from .slice_map_index import SliceMapIndex, build_slice_map_index
from .warm_q0 import build_warm_grid_trial_commit_events

RESCORE_SIDECAR_SCHEMA = "pychmp.rescore_sidecar.v1"
RESCORE_META_DATASET = "rescore_meta_json"
RESCORE_SEARCH_MODE = "map_store_rescore"
RESCORE_SUFFIX_RE = re.compile(r"^(.+)_r(\d+)$")
FootprintKind = Literal["source", "map_store_union"]


class RescoreError(RuntimeError):
    """Base error for rescore build/commit failures."""


class RescoreBuildError(RescoreError):
    """Raised when the sidecar build fails validation."""


class RescoreCommitError(RescoreError):
    """Raised when committing a sidecar into the main artifact is unsafe or invalid."""


@dataclass(frozen=True)
class RescoreBuildReport:
    source_artifact: Path
    sidecar_path: Path
    source_search_id: str
    target_search_id: str
    slice_key: str
    rescore_pass: int
    footprint: FootprintKind
    points_requested: int
    points_rescored: int
    points_failed: int
    trial_count: int


@dataclass(frozen=True)
class RescoreCommitReport:
    artifact_h5: Path
    sidecar_path: Path
    target_search_id: str
    slice_key: str
    grid_point_count: int


@dataclass
class _SliceWriteContext:
    observed: np.ndarray
    sigma_map: np.ndarray
    wcs_header: fits.Header
    diagnostics: dict[str, Any]
    blos_reference: tuple[np.ndarray, fits.Header] | None
    psf_kernel: np.ndarray | None
    map_store_artifact: Path


def rescore_root_search_id(search_id: str) -> str:
    """Return the adaptive root id for a search or rescore clone (``…_rN``)."""
    resolved = str(search_id or "").strip()
    match = RESCORE_SUFFIX_RE.match(resolved)
    if match:
        return str(match.group(1))
    return resolved


def derive_rescore_search_id(
    artifact_h5: Path,
    *,
    source_search_id: str,
    slice_key: str | None = None,
) -> str:
    """Allocate the next ``{root}_rN`` id on ``slice_key``."""
    root = rescore_root_search_id(source_search_id)
    if slice_key is None:
        slice_key, _ = resolve_search_location(artifact_h5, search_id=source_search_id)
    max_pass = 0
    with _H5PY_FILE(artifact_h5, "r") as f:
        if SLICE_CONTAINER_GROUP not in f or str(slice_key) not in f[SLICE_CONTAINER_GROUP]:
            return f"{root}_r1"
        searches = f[SLICE_CONTAINER_GROUP][str(slice_key)].get(SEARCHES_GROUP)
        if searches is None:
            return f"{root}_r1"
        for name in searches.keys():
            match = RESCORE_SUFFIX_RE.match(str(name))
            if match and str(match.group(1)) == root:
                max_pass = max(max_pass, int(match.group(2)))
    return f"{root}_r{max_pass + 1}"


def default_sidecar_path(artifact_h5: Path, *, target_search_id: str) -> Path:
    return artifact_h5.with_name(f"{artifact_h5.stem}.{target_search_id}.rescore.h5")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_sidecar_meta(sidecar_path: Path) -> dict[str, Any]:
    with _H5PY_FILE(sidecar_path, "r") as f:
        if RESCORE_META_DATASET not in f:
            raise RescoreCommitError(f"Sidecar missing {RESCORE_META_DATASET}: {sidecar_path}")
        meta = _json_loads_or_empty(f[RESCORE_META_DATASET][()])
    if str(meta.get("schema", "")).strip() != RESCORE_SIDECAR_SCHEMA:
        raise RescoreCommitError(f"Unsupported sidecar schema in {sidecar_path}")
    return meta


def _write_sidecar_meta(sidecar_path: Path, meta: dict[str, Any]) -> None:
    with _H5PY_FILE(sidecar_path, "a") as f:
        if RESCORE_META_DATASET in f:
            del f[RESCORE_META_DATASET]
        _create_text_dataset(f, RESCORE_META_DATASET, _json_dumps(meta))


def _load_slice_write_context(
    artifact_h5: Path,
    *,
    slice_key: str,
    source_search_id: str,
    profile_diagnostics: dict[str, Any],
) -> _SliceWriteContext:
    with _H5PY_FILE(artifact_h5, "r") as f:
        if SLICE_CONTAINER_GROUP not in f or str(slice_key) not in f[SLICE_CONTAINER_GROUP]:
            raise RescoreBuildError(f"Slice {slice_key!r} not found in {artifact_h5}")
        slice_group = f[SLICE_CONTAINER_GROUP][str(slice_key)]
        if "common" not in slice_group:
            raise RescoreBuildError(f"Slice {slice_key!r} is missing common observation maps")
        common_payload = _read_common_group(slice_group["common"])
    observed = common_payload["observed"]
    sigma_map = common_payload["sigma_map"]
    wcs_header = common_payload["wcs_header"]
    if observed is None or sigma_map is None or wcs_header is None:
        raise RescoreBuildError(f"Slice {slice_key!r} common maps are incomplete")
    diagnostics = dict(profile_diagnostics)
    diagnostics.update(dict(common_payload.get("diagnostics") or {}))
    blos_reference = common_payload.get("blos_reference")
    psf_kernel = common_payload.get("psf_kernel")
    diagnostics["target_slice_key"] = str(slice_key)
    return _SliceWriteContext(
        observed=np.asarray(observed, dtype=float),
        sigma_map=np.asarray(sigma_map, dtype=float),
        wcs_header=wcs_header,
        diagnostics=diagnostics,
        blos_reference=blos_reference,
        psf_kernel=None if psf_kernel is None else np.asarray(psf_kernel, dtype=float),
        map_store_artifact=artifact_h5.resolve(),
    )


def _evaluation_context_from_diagnostics(
    write_ctx: _SliceWriteContext,
    *,
    obs_ref_payload: dict[str, Any] | None,
) -> Any:
    from .chmp_evaluation import ObservationEvaluationContext

    if obs_ref_payload is not None:
        observed = np.asarray(obs_ref_payload.get("observed"), dtype=float)
        sigma = np.asarray(obs_ref_payload.get("sigma_map"), dtype=float)
        target_header = obs_ref_payload.get("wcs_header")
        if not isinstance(target_header, fits.Header):
            raise RescoreBuildError("Observation reference payload is missing a valid WCS header")
        source_header = obs_ref_payload.get("source_header", target_header)
        if not isinstance(source_header, fits.Header):
            source_header = target_header
        reference = SliceObservationReference(
            observed=observed,
            sigma=sigma,
            source_header=source_header.copy(),
            target_header=target_header.copy(),
            diagnostics=dict(obs_ref_payload.get("diagnostics") or {}),
            identity=dict(obs_ref_payload.get("identity") or obs_ref_payload.get("diagnostics") or {}),
            restored_from_artifact=True,
            shift_policy=str(write_ctx.diagnostics.get("shift_policy") or "auto"),
            max_shift_arcsec=write_ctx.diagnostics.get("max_shift_arcsec"),
            xy_shift_arcsec=(
                float(write_ctx.diagnostics.get("xy_shift_arcsec", [0.0, 0.0])[0]),
                float(write_ctx.diagnostics.get("xy_shift_arcsec", [0.0, 0.0])[1]),
            ),
            observation_canvas=(
                None
                if obs_ref_payload.get("observation_canvas") is None
                else np.asarray(obs_ref_payload["observation_canvas"], dtype=float)
            ),
            sigma_canvas=(
                None
                if obs_ref_payload.get("sigma_canvas") is None
                else np.asarray(obs_ref_payload["sigma_canvas"], dtype=float)
            ),
            canvas_header=(
                obs_ref_payload["canvas_wcs_header"].copy()
                if isinstance(obs_ref_payload.get("canvas_wcs_header"), fits.Header)
                else None
            ),
        )
        return observation_reference_to_evaluation_context(
            reference,
            use_smoothed_obs_max=bool(write_ctx.diagnostics.get("use_smoothed_obs_max", True)),
            emthreshold=float(write_ctx.diagnostics.get("emthreshold", 0.1)),
        )

    header = write_ctx.wcs_header.copy()
    return ObservationEvaluationContext(
        model_header=header,
        shift_policy=str(write_ctx.diagnostics.get("shift_policy") or "auto"),
        max_shift_arcsec=float(write_ctx.diagnostics.get("max_shift_arcsec", 20.0) or 20.0),
        xy_shift_arcsec=(
            float(write_ctx.diagnostics.get("xy_shift_arcsec", [0.0, 0.0])[0]),
            float(write_ctx.diagnostics.get("xy_shift_arcsec", [0.0, 0.0])[1]),
        ),
        observed=write_ctx.observed,
        sigma=write_ctx.sigma_map,
        use_smoothed_obs_max=bool(write_ctx.diagnostics.get("use_smoothed_obs_max", True)),
        emthreshold=float(write_ctx.diagnostics.get("emthreshold", 0.1)),
    )


def _footprint_ab_points(
    artifact_h5: Path,
    *,
    slice_key: str,
    source_search_id: str,
    footprint: FootprintKind,
    slice_map_index: SliceMapIndex,
) -> list[tuple[float, float]]:
    if footprint == "map_store_union":
        return list(slice_map_index.point_keys())
    with _H5PY_FILE(artifact_h5, "r") as f:
        search_group = f[SLICE_CONTAINER_GROUP][str(slice_key)][SEARCHES_GROUP][str(source_search_id)]
        headers = list_grid_point_headers(search_group)
    points: list[tuple[float, float]] = []
    for header in headers:
        points.append((float(header["a"]), float(header["b"])))
    return points


def _rescore_diagnostics(
    profile_diagnostics: dict[str, Any],
    *,
    source_search_id: str,
    target_search_id: str,
    rescore_pass: int,
    footprint: FootprintKind,
) -> dict[str, Any]:
    diagnostics = dict(profile_diagnostics)
    diagnostics["search_mode"] = RESCORE_SEARCH_MODE
    diagnostics["search_id"] = str(target_search_id)
    diagnostics["selected_search_id"] = str(target_search_id)
    diagnostics["rescore_source_search_id"] = rescore_root_search_id(source_search_id)
    diagnostics["rescore_pass"] = int(rescore_pass)
    diagnostics["rescore_footprint"] = str(footprint)
    diagnostics["search_active"] = False
    diagnostics["search_completed_at"] = _utc_now_iso()
    diagnostics["contract_version"] = GRID_POINTS_CONTRACT_VERSION
    diagnostics["search_instance_id"] = f"rescore_r{int(rescore_pass)}"
    return diagnostics


def _init_sidecar_shell(
    sidecar_path: Path,
    *,
    slice_key: str,
    target_search_id: str,
    diagnostics: dict[str, Any],
) -> None:
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    if sidecar_path.exists():
        sidecar_path.unlink()
    descriptor = target_slice_descriptor_from_diagnostics(diagnostics, fallback_key=str(slice_key))
    with _H5PY_FILE(sidecar_path, "w") as f:
        slices = f.create_group(SLICE_CONTAINER_GROUP)
        slice_group = slices.create_group(str(slice_key))
        _set_slice_group_attrs(slice_group, descriptor)
        searches = slice_group.create_group(SEARCHES_GROUP)
        search_group = searches.create_group(str(target_search_id))
        search_group.attrs["search_id"] = np.bytes_(str(target_search_id))
        search_group.attrs["target_metric"] = np.bytes_(str(diagnostics.get("target_metric", "chi2")))
        _create_text_dataset(search_group, "diagnostics_json", _json_dumps(diagnostics))
        layout = {"kind": "point_list"}
        _create_text_dataset(search_group, "layout_json", _json_dumps(layout))
        _create_text_dataset(search_group, "run_history_json", _json_dumps([]))
        request = _search_request_from_diagnostics(diagnostics, layout=layout)
        _create_text_dataset(search_group, SEARCH_REQUEST_DATASET, _json_dumps(request))
        lifecycle = _search_lifecycle_payload(status="empty", diagnostics=diagnostics)
        _write_search_lifecycle_dataset(search_group, lifecycle=lifecycle)
        search_group.create_group(GRID_POINTS_GROUP)


def _load_observation_ref_payload(
    artifact_h5: Path,
    *,
    slice_key: str,
    search_id: str,
) -> dict[str, Any] | None:
    from .ab_scan_artifacts import load_slice_observation_reference_payload

    return load_slice_observation_reference_payload(
        artifact_h5,
        slice_key=str(slice_key),
        search_id=str(search_id),
    )


def build_rescore_sidecar(
    artifact_h5: Path,
    *,
    source_search_id: str,
    footprint: FootprintKind = "source",
    sidecar_path: Path | None = None,
    target_search_id: str | None = None,
    dry_run: bool = False,
) -> RescoreBuildReport:
    """Rescore map_store trials into a sidecar search (read-only on the main artifact)."""
    artifact_h5 = Path(artifact_h5).expanduser().resolve()
    if not artifact_h5.exists():
        raise RescoreBuildError(f"Artifact not found: {artifact_h5}")

    profile = load_search_run_profile(artifact_h5, search_id=str(source_search_id))
    slice_key = str(profile["slice_key"])
    source_search_id = str(profile["search_id"])
    resolved_target = str(target_search_id or derive_rescore_search_id(artifact_h5, source_search_id=source_search_id, slice_key=slice_key))
    match = RESCORE_SUFFIX_RE.match(resolved_target)
    rescore_pass = int(match.group(2)) if match else 1

    write_ctx = _load_slice_write_context(
        artifact_h5,
        slice_key=slice_key,
        source_search_id=source_search_id,
        profile_diagnostics=dict(profile["diagnostics"]),
    )
    slice_map_index = build_slice_map_index(artifact_h5, slice_key=slice_key)
    evaluation_context = _evaluation_context_from_diagnostics(
        write_ctx,
        obs_ref_payload=_load_observation_ref_payload(artifact_h5, slice_key=slice_key, search_id=source_search_id),
    )
    target_metric = str(write_ctx.diagnostics.get("target_metric") or "chi2")
    threshold = float(write_ctx.diagnostics.get("metrics_mask_threshold", write_ctx.diagnostics.get("threshold", 0.1)))
    use_emthreshold = bool(write_ctx.diagnostics.get("use_emthreshold", True))
    diagnostics = _rescore_diagnostics(
        write_ctx.diagnostics,
        source_search_id=source_search_id,
        target_search_id=resolved_target,
        rescore_pass=rescore_pass,
        footprint=footprint,
    )
    diagnostics["selected_search_id"] = resolved_target
    diagnostics["search_id"] = resolved_target

    ab_points = _footprint_ab_points(
        artifact_h5,
        slice_key=slice_key,
        source_search_id=source_search_id,
        footprint=footprint,
        slice_map_index=slice_map_index,
    )
    if not ab_points:
        raise RescoreBuildError("No grid points selected for rescore footprint")

    resolved_sidecar = (
        default_sidecar_path(artifact_h5, target_search_id=resolved_target)
        if sidecar_path is None
        else Path(sidecar_path).expanduser().resolve()
    )

    if dry_run:
        rescored = sum(
            1
            for a_value, b_value in ab_points
            if slice_map_index.entries_for_point(float(a_value), float(b_value))
        )
        return RescoreBuildReport(
            source_artifact=artifact_h5,
            sidecar_path=resolved_sidecar,
            source_search_id=source_search_id,
            target_search_id=resolved_target,
            slice_key=slice_key,
            rescore_pass=rescore_pass,
            footprint=footprint,
            points_requested=len(ab_points),
            points_rescored=rescored,
            points_failed=len(ab_points) - rescored,
            trial_count=slice_map_index.trial_count(),
        )

    _init_sidecar_shell(
        resolved_sidecar,
        slice_key=slice_key,
        target_search_id=resolved_target,
        diagnostics=diagnostics,
    )

    points_rescored = 0
    points_failed = 0
    trial_count = 0
    context_kwargs = {
        "observed": write_ctx.observed,
        "sigma_map": write_ctx.sigma_map,
        "wcs_header": write_ctx.wcs_header,
        "diagnostics": diagnostics,
        "blos_reference": write_ctx.blos_reference,
        "psf_kernel": write_ctx.psf_kernel,
        "map_store_artifact": write_ctx.map_store_artifact,
        "ensure_slice_common": False,
    }

    for index, (a_value, b_value) in enumerate(ab_points, start=1):
        q0_values = slice_map_index.q0_values(float(a_value), float(b_value))
        if not q0_values:
            points_failed += 1
            print(f"[{index}/{len(ab_points)}] skip (a,b)=({a_value:.3f},{b_value:.3f}): no map_store trials")
            continue
        q0_start = float(min(q0_values))
        point_id = apply_grid_point_assigned(
            resolved_sidecar,
            GridPointAssignedEvent(
                a=float(a_value),
                b=float(b_value),
                q0_start=q0_start,
                next_q0=q0_start,
                metric_name=target_metric,
                force_new_point_id=True,
            ),
            **context_kwargs,
        )
        events = build_warm_grid_trial_commit_events(
            artifact_h5,
            point_id=str(point_id),
            slice_map_index=slice_map_index,
            a_value=float(a_value),
            b_value=float(b_value),
            context=evaluation_context,
            threshold=float(threshold),
            explicit_mask=None,
            target_metric=target_metric,
            use_emthreshold=use_emthreshold,
        )
        if not events:
            apply_grid_point_failed(
                resolved_sidecar,
                GridPointFailedEvent(point_id=str(point_id), error_message="no rescored trials"),
                **context_kwargs,
            )
            points_failed += 1
            print(f"[{index}/{len(ab_points)}] failed (a,b)=({a_value:.3f},{b_value:.3f}): no valid rescore")
            continue
        for event in events:
            apply_grid_point_event_with_retry(resolved_sidecar, event, **context_kwargs)
        best_event = events[-1]
        apply_grid_point_completed(
            resolved_sidecar,
            GridPointCompletedEvent(
                point_id=str(point_id),
                best_trial_index=int(best_event.best_trial_index),
                best_metric=float(best_event.best_metric),
                best_q0=float(best_event.q0),
            ),
            **context_kwargs,
        )
        points_rescored += 1
        trial_count += len(events)
        print(
            f"[{index}/{len(ab_points)}] rescored (a,b)=({a_value:.3f},{b_value:.3f}) "
            f"trials={len(events)} best_{target_metric}={float(best_event.best_metric):.6e}"
        )

    if footprint == "source" and points_failed > 0:
        raise RescoreBuildError(
            f"Rescore incomplete for source footprint: {points_failed} of {len(ab_points)} points failed"
        )
    if points_rescored == 0:
        raise RescoreBuildError("No grid points were rescored successfully")

    meta = {
        "schema": RESCORE_SIDECAR_SCHEMA,
        "built_utc": _utc_now_iso(),
        "source_artifact": str(artifact_h5),
        "sidecar_path": str(resolved_sidecar),
        "slice_key": slice_key,
        "source_search_id": source_search_id,
        "target_search_id": resolved_target,
        "rescore_pass": rescore_pass,
        "footprint": footprint,
        "points_requested": len(ab_points),
        "points_rescored": points_rescored,
        "points_failed": points_failed,
        "trial_count": trial_count,
        "status": "ready",
    }
    _write_sidecar_meta(resolved_sidecar, meta)
    print(f"Sidecar ready: {resolved_sidecar}")
    print(f"Target search id: {resolved_target} ({points_rescored} points, {trial_count} trials)")

    return RescoreBuildReport(
        source_artifact=artifact_h5,
        sidecar_path=resolved_sidecar,
        source_search_id=source_search_id,
        target_search_id=resolved_target,
        slice_key=slice_key,
        rescore_pass=rescore_pass,
        footprint=footprint,
        points_requested=len(ab_points),
        points_rescored=points_rescored,
        points_failed=points_failed,
        trial_count=trial_count,
    )


def artifact_has_active_search(artifact_h5: Path) -> tuple[str | None, str | None]:
    return find_in_progress_slice_search(Path(artifact_h5))


def commit_rescore_sidecar(
    artifact_h5: Path,
    *,
    sidecar_path: Path,
    force: bool = False,
    remove_sidecar: bool = False,
    touch_refresh: bool = False,
) -> RescoreCommitReport:
    """Merge a validated sidecar search into the main artifact."""
    artifact_h5 = Path(artifact_h5).expanduser().resolve()
    sidecar_path = Path(sidecar_path).expanduser().resolve()
    if not artifact_h5.exists():
        raise RescoreCommitError(f"Artifact not found: {artifact_h5}")
    if not sidecar_path.exists():
        raise RescoreCommitError(f"Sidecar not found: {sidecar_path}")

    meta = _read_sidecar_meta(sidecar_path)
    if str(meta.get("status", "")).strip().lower() != "ready":
        raise RescoreCommitError(f"Sidecar status is not ready: {meta.get('status')!r}")
    source_artifact = Path(str(meta.get("source_artifact", ""))).expanduser().resolve()
    if source_artifact != artifact_h5:
        raise RescoreCommitError(
            f"Sidecar source artifact {source_artifact} does not match commit target {artifact_h5}"
        )

    slice_key = str(meta["slice_key"])
    target_search_id = str(meta["target_search_id"])
    source_search_id = str(meta.get("source_search_id", ""))

    active_slice, active_search = artifact_has_active_search(artifact_h5)
    if active_search and not force:
        raise RescoreCommitError(
            f"Refusing commit: search {active_search!r} on slice {active_slice!r} is still active. "
            "Stop the runner or pass --force."
        )

    src_search_path = f"{SLICE_CONTAINER_GROUP}/{slice_key}/{SEARCHES_GROUP}/{target_search_id}"
    with _H5PY_FILE(sidecar_path, "r") as src_file:
        if src_search_path not in src_file:
            raise RescoreCommitError(f"Sidecar missing search group: {src_search_path}")
        with _H5PY_FILE(artifact_h5, "a") as dst_file:
            if SLICE_CONTAINER_GROUP not in dst_file or slice_key not in dst_file[SLICE_CONTAINER_GROUP]:
                raise RescoreCommitError(f"Target slice {slice_key!r} missing in {artifact_h5}")
            slice_group = dst_file[SLICE_CONTAINER_GROUP][slice_key]
            searches_group = slice_group.require_group(SEARCHES_GROUP)
            if str(target_search_id) in searches_group:
                del searches_group[str(target_search_id)]
            src_file.copy(src_file[src_search_path], searches_group, name=str(target_search_id))
            if source_search_id:
                main_ref = (
                    f"{SLICE_CONTAINER_GROUP}/{slice_key}/{SEARCHES_GROUP}/"
                    f"{source_search_id}/{OBSERVATION_REF_GROUP}"
                )
                if main_ref in dst_file and OBSERVATION_REF_GROUP not in searches_group[str(target_search_id)]:
                    dst_search = searches_group[str(target_search_id)]
                    dst_file.copy(dst_file[main_ref], dst_search, name=OBSERVATION_REF_GROUP)

    with _H5PY_FILE(artifact_h5, "r") as f:
        if src_search_path not in f:
            raise RescoreCommitError("Commit verification failed: search group not present in artifact")
        grid_count = len(f[src_search_path][GRID_POINTS_GROUP])

    if touch_refresh:
        refresh_path = Path(f"{artifact_h5}.refresh")
        payload = {
            "phase": "rescore complete",
            "slice_key": slice_key,
            "search_id": target_search_id,
            "utc": _utc_now_iso(),
        }
        refresh_path.write_text(json.dumps(payload), encoding="utf-8")

    if remove_sidecar:
        sidecar_path.unlink(missing_ok=True)

    print(f"Committed {target_search_id} to {artifact_h5} ({grid_count} grid points)")
    if touch_refresh:
        print(f"Refresh signal: {artifact_h5}.refresh")

    return RescoreCommitReport(
        artifact_h5=artifact_h5,
        sidecar_path=sidecar_path,
        target_search_id=target_search_id,
        slice_key=slice_key,
        grid_point_count=int(grid_count),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pychmp-rescore",
        description=(
            "Build a map-store rescore sidecar from an existing search, then commit it into the "
            "main artifact when no adaptive search is active."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser_cmd = subparsers.add_parser("build", help="Rescore into a sidecar H5 (read-only on main artifact)")
    build_parser_cmd.add_argument("--artifact-h5", type=Path, required=True, help="Main unified artifact H5")
    build_parser_cmd.add_argument("--source-search-id", required=True, help="Search identity to rescore from")
    build_parser_cmd.add_argument(
        "--footprint",
        choices=("source", "map_store_union"),
        default="source",
        help="Grid footprint: original search grid or all indexed (a,b) on the slice",
    )
    build_parser_cmd.add_argument("--sidecar", type=Path, default=None, help="Optional sidecar output path")
    build_parser_cmd.add_argument("--target-search-id", default=None, help="Override derived {root}_rN id")
    build_parser_cmd.add_argument("--dry-run", action="store_true", help="Report coverage without writing a sidecar")

    commit_parser_cmd = subparsers.add_parser("commit", help="Merge sidecar search into the main artifact")
    commit_parser_cmd.add_argument("--artifact-h5", type=Path, required=True, help="Main unified artifact H5")
    commit_parser_cmd.add_argument("--sidecar", type=Path, required=True, help="Sidecar H5 from build")
    commit_parser_cmd.add_argument(
        "--force",
        action="store_true",
        help="Commit even if a search lifecycle is still marked active",
    )
    commit_parser_cmd.add_argument(
        "--remove-sidecar",
        action="store_true",
        help="Delete the sidecar file after a successful commit",
    )
    commit_parser_cmd.add_argument(
        "--touch-refresh",
        action="store_true",
        help="Write artifact.h5.refresh (optional; relaunch viewer is recommended)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            build_rescore_sidecar(
                args.artifact_h5,
                source_search_id=str(args.source_search_id),
                footprint=args.footprint,
                sidecar_path=args.sidecar,
                target_search_id=args.target_search_id,
                dry_run=bool(args.dry_run),
            )
            return 0
        if args.command == "commit":
            commit_rescore_sidecar(
                args.artifact_h5,
                sidecar_path=args.sidecar,
                force=bool(args.force),
                remove_sidecar=bool(args.remove_sidecar),
                touch_refresh=bool(args.touch_refresh),
            )
            return 0
    except (RescoreError, ValueError, KeyError, OSError) as exc:
        print(f"ERROR: {exc}")
        return 1
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
