"""Repair utility: remove grid trial rows that violate the map_store contract."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

from .ab_scan_artifacts import (
    MAP_REFS_DATASET,
    SEARCHES_GROUP,
    SLICE_CONTAINER_GROUP,
    _H5PY_FILE,
    _json_loads_or_empty,
    _read_map_store_ref_array,
)
from .grid_points import (
    GRID_POINTS_GROUP,
    GRID_POINTS_TRIALS_GROUP,
    GridPointStatus,
    _load_grid_point_trials,
    _update_search_counts,
    _write_header_attrs,
    classify_grid_point_state,
    grid_point_finite_q0_trials_have_map_store_links,
    read_grid_point_header,
    select_fit_trials_for_viewer,
)
from .viewer_navigation import search_metric_best_trial_index


def grid_trial_row_should_purge(h5_file: h5py.File, trial_group: h5py.Group) -> tuple[bool, str]:
    """Return (purge, reason) for one grid-point trial HDF5 group."""
    try:
        q0_value = float(trial_group.attrs["q0"])
    except Exception:
        return True, "missing_q0"
    map_refs = _json_loads_or_empty(trial_group[MAP_REFS_DATASET][()]) if MAP_REFS_DATASET in trial_group else {}
    raw_ref = str(map_refs.get("raw_modeled", "") or "").strip()
    if not (np.isfinite(q0_value) and q0_value > 0.0):
        return True, "non_finite_q0"
    if not raw_ref:
        return True, "missing_map_ref"
    if _read_map_store_ref_array(h5_file, raw_ref) is None:
        return True, "broken_map_ref"
    return False, ""


def _rebuild_grid_point_header_after_purge(point_group: h5py.Group) -> None:
    header = read_grid_point_header(point_group)
    trials = _load_grid_point_trials(point_group, include_maps=False)
    fit_trials = select_fit_trials_for_viewer(trials)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    header["updated_utc"] = now

    if not fit_trials:
        header["n_trials"] = 0
        header["best_trial_index"] = -1
        status = str(header.get("status", "")).strip().upper()
        if status in {GridPointStatus.COMPLETED.value, GridPointStatus.RUNNING.value}:
            header["status"] = GridPointStatus.ASSIGNED.value
            header.pop("completed_utc", None)
            if "next_q0" not in header:
                header["next_q0"] = float(header.get("q0_start", np.nan))
        _write_header_attrs(point_group, header)
        return

    header["n_trials"] = int(max(int(item["trial_index"]) for item in fit_trials) + 1)
    preview = {
        "fit_q0_trials": tuple(float(item["q0"]) for item in fit_trials),
        "fit_metric_trials": tuple(float(item.get("target_metric_value", np.nan)) for item in fit_trials),
        "fit_chi2_trials": tuple(float(item.get("chi2", np.nan)) for item in fit_trials),
        "fit_rho2_trials": tuple(float(item.get("rho2", np.nan)) for item in fit_trials),
        "fit_eta2_trials": tuple(float(item.get("eta2", np.nan)) for item in fit_trials),
        "target_metric": str(header.get("metric_name", "chi2")),
    }
    best_array_index = search_metric_best_trial_index(preview, str(header.get("metric_name", "chi2")))
    if best_array_index is not None and 0 <= int(best_array_index) < len(fit_trials):
        header["best_trial_index"] = int(fit_trials[int(best_array_index)]["trial_index"])
    else:
        header["best_trial_index"] = int(fit_trials[-1]["trial_index"])

    state = classify_grid_point_state(header)
    if state == "complete":
        header["status"] = GridPointStatus.COMPLETED.value
        header.pop("next_q0", None)
    elif int(header.get("n_trials", 0)) > 0 and "next_q0" in header:
        header["status"] = GridPointStatus.RUNNING.value

    _write_header_attrs(point_group, header)


@dataclass
class GridTrialPurgeDetail:
    slice_key: str
    search_id: str
    point_id: str
    trial_name: str
    trial_index: int
    q0: float
    reason: str


@dataclass
class RepairGridTrialMapsReport:
    artifact_path: Path
    dry_run: bool
    purged_trial_count: int = 0
    points_touched: int = 0
    slices_scanned: int = 0
    searches_scanned: int = 0
    points_scanned: int = 0
    clean: bool = False
    details: list[GridTrialPurgeDetail] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_path": str(self.artifact_path),
            "dry_run": self.dry_run,
            "purged_trial_count": self.purged_trial_count,
            "points_touched": self.points_touched,
            "slices_scanned": self.slices_scanned,
            "searches_scanned": self.searches_scanned,
            "points_scanned": self.points_scanned,
            "clean": self.clean,
            "details": [
                {
                    "slice_key": item.slice_key,
                    "search_id": item.search_id,
                    "point_id": item.point_id,
                    "trial_name": item.trial_name,
                    "trial_index": item.trial_index,
                    "q0": item.q0,
                    "reason": item.reason,
                }
                for item in self.details
            ],
        }


def artifact_grid_trials_are_clean(
    artifact_h5: Path,
    *,
    slice_key: str | None = None,
    search_id: str | None = None,
) -> bool:
    """True when no purgeable trial rows remain and finite-Q0 trials all link map_store."""
    with _H5PY_FILE(artifact_h5, "r") as h5_file:
        if SLICE_CONTAINER_GROUP not in h5_file:
            return True
        slice_keys = [slice_key] if slice_key else sorted(h5_file[SLICE_CONTAINER_GROUP].keys())
        for key in slice_keys:
            if key not in h5_file[SLICE_CONTAINER_GROUP]:
                continue
            slice_group = h5_file[SLICE_CONTAINER_GROUP][key]
            if SEARCHES_GROUP not in slice_group:
                continue
            search_keys = [search_id] if search_id else sorted(slice_group[SEARCHES_GROUP].keys())
            for sid in search_keys:
                if sid not in slice_group[SEARCHES_GROUP]:
                    continue
                search_group = slice_group[SEARCHES_GROUP][sid]
                if GRID_POINTS_GROUP not in search_group:
                    continue
                for point_name in search_group[GRID_POINTS_GROUP].keys():
                    point_group = search_group[GRID_POINTS_GROUP][point_name]
                    if GRID_POINTS_TRIALS_GROUP in point_group:
                        for trial_name in point_group[GRID_POINTS_TRIALS_GROUP].keys():
                            should_purge, _reason = grid_trial_row_should_purge(
                                h5_file,
                                point_group[GRID_POINTS_TRIALS_GROUP][trial_name],
                            )
                            if should_purge:
                                return False
                    if not grid_point_finite_q0_trials_have_map_store_links(
                        artifact_h5,
                        slice_key=str(key),
                        search_id=str(sid),
                        point_id=str(point_name),
                    ):
                        return False
    return True


def repair_grid_trial_maps_in_artifact(
    artifact_h5: Path,
    *,
    slice_key: str | None = None,
    search_id: str | None = None,
    dry_run: bool = False,
) -> RepairGridTrialMapsReport:
    """Delete grid trial rows without a readable map_store link; drop non-finite-Q0 rows."""
    report = RepairGridTrialMapsReport(artifact_path=Path(artifact_h5), dry_run=bool(dry_run))
    artifact_h5 = Path(artifact_h5)
    if not artifact_h5.exists():
        raise FileNotFoundError(artifact_h5)

    with _H5PY_FILE(artifact_h5, "a" if not dry_run else "r") as h5_file:
        if SLICE_CONTAINER_GROUP not in h5_file:
            report.clean = True
            return report

        slice_keys = [slice_key] if slice_key else sorted(h5_file[SLICE_CONTAINER_GROUP].keys())
        report.slices_scanned = len(slice_keys)

        for key in slice_keys:
            if key not in h5_file[SLICE_CONTAINER_GROUP]:
                continue
            slice_group = h5_file[SLICE_CONTAINER_GROUP][key]
            if SEARCHES_GROUP not in slice_group:
                continue
            search_keys = [search_id] if search_id else sorted(slice_group[SEARCHES_GROUP].keys())
            for sid in search_keys:
                if sid not in slice_group[SEARCHES_GROUP]:
                    continue
                report.searches_scanned += 1
                search_group = slice_group[SEARCHES_GROUP][sid]
                if GRID_POINTS_GROUP not in search_group:
                    continue
                diagnostics = (
                    _json_loads_or_empty(search_group["diagnostics_json"][()])
                    if "diagnostics_json" in search_group
                    else {}
                )
                search_touched = False
                for point_name in sorted(search_group[GRID_POINTS_GROUP].keys()):
                    report.points_scanned += 1
                    point_group = search_group[GRID_POINTS_GROUP][point_name]
                    if GRID_POINTS_TRIALS_GROUP not in point_group:
                        continue
                    trials_group = point_group[GRID_POINTS_TRIALS_GROUP]
                    purge_names: list[tuple[str, str, int, float]] = []
                    for trial_name in sorted(trials_group.keys()):
                        trial_group = trials_group[trial_name]
                        should_purge, reason = grid_trial_row_should_purge(h5_file, trial_group)
                        if not should_purge:
                            continue
                        purge_names.append(
                            (
                                trial_name,
                                reason,
                                int(trial_group.attrs.get("trial_index", -1)),
                                float(trial_group.attrs.get("q0", np.nan)),
                            )
                        )
                    if not purge_names:
                        continue
                    search_touched = True
                    report.points_touched += 1
                    for trial_name, reason, trial_index, q0_value in purge_names:
                        report.purged_trial_count += 1
                        report.details.append(
                            GridTrialPurgeDetail(
                                slice_key=str(key),
                                search_id=str(sid),
                                point_id=str(point_name),
                                trial_name=str(trial_name),
                                trial_index=int(trial_index),
                                q0=float(q0_value),
                                reason=str(reason),
                            )
                        )
                        if dry_run:
                            continue
                        del trials_group[trial_name]
                    if not dry_run:
                        _rebuild_grid_point_header_after_purge(point_group)
                if not dry_run and search_touched:
                    _update_search_counts(search_group, diagnostics=diagnostics)

    report.clean = artifact_grid_trials_are_clean(
        artifact_h5,
        slice_key=slice_key,
        search_id=search_id,
    )
    return report


def _print_report(report: RepairGridTrialMapsReport) -> None:
    print(f"Artifact: {report.artifact_path}")
    print(f"Dry run: {'yes' if report.dry_run else 'no'}")
    print(f"Slices scanned: {report.slices_scanned}")
    print(f"Searches scanned: {report.searches_scanned}")
    print(f"Grid points scanned: {report.points_scanned}")
    print(f"Grid points touched: {report.points_touched}")
    print(f"Trials purged: {report.purged_trial_count}")
    print(f"Clean (ready for viewer / new runs): {'yes' if report.clean else 'no'}")
    if report.details:
        print("")
        print("Purged trials:")
        for item in report.details[:50]:
            print(
                f"  [{item.slice_key}/{item.search_id}/{item.point_id}/{item.trial_name}] "
                f"trial_index={item.trial_index} q0={item.q0:g} reason={item.reason}"
            )
        if len(report.details) > 50:
            print(f"  ... and {len(report.details) - 50} more")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pychmp-repair-grid-trial-maps",
        description=(
            "Remove grid trial HDF5 rows that lack a readable map_store link, and drop "
            "non-finite-Q0 phantom rows. Rebuilds affected grid-point headers. Does not "
            "delete map_store datasets."
        ),
    )
    parser.add_argument("artifact_h5", type=Path, help="Unified artifact H5 path")
    parser.add_argument("--slice-key", default=None, help="Limit repair to one slice key")
    parser.add_argument("--search-id", default=None, help="Limit repair to one search id")
    parser.add_argument("--dry-run", action="store_true", help="Report purge actions without modifying the artifact")
    parser.add_argument("--json", action="store_true", help="Emit JSON report")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only verify the map_store contract; do not modify the artifact",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact_h5 = Path(args.artifact_h5)
    if args.check_only:
        clean = artifact_grid_trials_are_clean(
            artifact_h5,
            slice_key=args.slice_key,
            search_id=args.search_id,
        )
        if args.json:
            print(json.dumps({"artifact_path": str(artifact_h5), "clean": clean}, indent=2))
        else:
            print(f"Artifact: {artifact_h5}")
            print(f"Clean (ready for viewer / new runs): {'yes' if clean else 'no'}")
        return 0 if clean else 1

    report = repair_grid_trial_maps_in_artifact(
        artifact_h5,
        slice_key=args.slice_key,
        search_id=args.search_id,
        dry_run=bool(args.dry_run),
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        _print_report(report)
        if report.clean:
            print("")
            print("Artifact is clean and ready to be viewed or used for new runs.")
        elif args.dry_run:
            print("")
            print("Re-run without --dry-run to apply repairs.")
    return 0 if report.clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
