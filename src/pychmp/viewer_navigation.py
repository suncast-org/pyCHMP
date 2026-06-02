"""Navigation helpers for pychmp-view refresh-driven grid focus."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pathlib import Path

from .ab_scan_artifacts import METRICS, load_scan_file, point_records_for_payload

REFRESH_EVENTS_REQUIRING_SLICE_RELOAD = frozenset(
    {
        "search_initialized",
        "point_assigned",
        "trial_committed",
        "point_completed",
        "point_failed",
        "search_completed",
    }
)


def execution_is_serial(diagnostics: dict[str, Any] | None) -> bool:
    diag = dict(diagnostics or {})
    policy = str(
        diag.get("execution_policy_resolved") or diag.get("execution_policy") or "serial"
    ).strip().lower()
    return policy == "serial"


def should_follow_active_refresh_event(
    event: str,
    *,
    serial: bool,
) -> bool:
    name = str(event or "").strip().lower()
    if not name or name in {"search_initialized", "search_completed"}:
        return False
    if serial:
        return name in {"point_assigned", "trial_committed", "point_completed", "point_failed"}
    return name == "point_completed"


def metric_value_from_record(record: dict[str, Any], metric_name: str) -> float:
    metric = str(metric_name or "").strip().lower()
    diagnostics = dict(record.get("diagnostics", {}))
    metrics = dict(record.get("metrics", {}))
    trial_key = {
        "chi2": "fit_chi2_trials",
        "rho2": "fit_rho2_trials",
        "eta2": "fit_eta2_trials",
    }.get(metric, "fit_metric_trials")
    trials = record.get(trial_key, record.get("fit_metric_trials", ()))
    if trials:
        arr = np.asarray(trials, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            return float(np.min(finite))
    try:
        return float(metrics.get(metric, diagnostics.get(metric, np.nan)))
    except Exception:
        return float("nan")


def search_metric_best_trial_index(record: dict[str, Any], search_metric: str) -> int | None:
    metric = str(search_metric or "").strip().lower()
    if metric not in METRICS:
        return None
    trial_key = {
        "chi2": "fit_chi2_trials",
        "rho2": "fit_rho2_trials",
        "eta2": "fit_eta2_trials",
    }.get(metric, "fit_metric_trials")
    trials = np.asarray(record.get(trial_key, record.get("fit_metric_trials", ())), dtype=float)
    if trials.size == 0:
        return None
    finite_mask = np.isfinite(trials)
    if not np.any(finite_mask):
        return None
    return int(np.nanargmin(np.where(finite_mask, trials, np.inf)))


def _point_index_from_record(payload: dict[str, Any], record: dict[str, Any]) -> tuple[int, int]:
    a_values = np.asarray(payload.get("a_values", ()), dtype=float)
    b_values = np.asarray(payload.get("b_values", ()), dtype=float)
    if "a_index" in record and "b_index" in record:
        return int(record["a_index"]), int(record["b_index"])
    a_value = float(record["a"])
    b_value = float(record["b"])
    a_index = int(np.argmin(np.abs(a_values - a_value))) if a_values.size else 0
    b_index = int(np.argmin(np.abs(b_values - b_value))) if b_values.size else 0
    return a_index, b_index


def _completion_sort_key(record: dict[str, Any]) -> str:
    diagnostics = dict(record.get("diagnostics", {}))
    for key in ("completed_utc", "updated_utc", "created_utc"):
        text = str(diagnostics.get(key) or record.get(key) or "").strip()
        if text:
            return text
    order = diagnostics.get("grid_point_id", record.get("record_order", ""))
    return str(order)


@dataclass(frozen=True)
class BestPointSelection:
    a_index: int
    b_index: int
    record: dict[str, Any]
    tied_records: tuple[dict[str, Any], ...]
    search_metric_best_trial_index: int | None


def best_point_selection(payload: dict[str, Any], search_metric: str) -> BestPointSelection | None:
    metric_name = str(search_metric or "").strip().lower()
    if metric_name not in METRICS:
        return None
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for record in point_records_for_payload(payload):
        status = str(record.get("status", "computed")).strip().lower()
        if status not in {"computed", "complete", "success"} and not bool(record.get("success", False)):
            continue
        value = metric_value_from_record(record, metric_name)
        if not np.isfinite(value):
            continue
        ranked.append((float(value), _completion_sort_key(record), record))
    if not ranked:
        return None
    best_value = float(min(item[0] for item in ranked))
    tolerance = max(1e-12, abs(best_value) * 1e-9)
    tied = [record for value, _key, record in ranked if abs(float(value) - best_value) <= tolerance]
    tied.sort(key=_completion_sort_key)
    winner = tied[-1]
    a_index, b_index = _point_index_from_record(payload, winner)
    return BestPointSelection(
        a_index=int(a_index),
        b_index=int(b_index),
        record=winner,
        tied_records=tuple(tied),
        search_metric_best_trial_index=search_metric_best_trial_index(winner, metric_name),
    )


def find_global_best_domain(
    h5_path: Path,
    *,
    catalog: list[dict[str, Any]],
    target_metric: str,
) -> tuple[str | None, str | None]:
    """Return the slice/search pair with the best run-target metric across the artifact."""
    metric_name = str(target_metric or "").strip().lower()
    if metric_name not in METRICS:
        return None, None
    best_value: float | None = None
    best_slice: str | None = None
    best_search: str | None = None
    for entry in catalog:
        slice_key = str(entry.get("slice_key", "")).strip()
        search_id = str(entry.get("search_id", "")).strip()
        if not slice_key or not search_id:
            continue
        try:
            payload = load_scan_file(
                h5_path,
                slice_key=slice_key,
                search_id=search_id,
                include_maps=False,
            )
        except Exception:
            continue
        selection = best_point_selection(payload, metric_name)
        if selection is None:
            continue
        value = metric_value_from_record(selection.record, metric_name)
        if not np.isfinite(value):
            continue
        tolerance = max(1e-12, abs(float(value)) * 1e-9)
        if best_value is None or float(value) < float(best_value) - tolerance:
            best_value = float(value)
            best_slice = slice_key
            best_search = search_id
    return best_slice, best_search


def assigned_only_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record in point_records_for_payload(payload):
        status = str(record.get("status", "computed")).strip().lower()
        if status != "pending":
            continue
        diagnostics = dict(record.get("diagnostics", {}))
        grid_status = str(diagnostics.get("grid_point_status", "")).strip().upper()
        n_trials = int(record.get("nfev", len(tuple(record.get("fit_q0_trials", ())))))
        if grid_status == "ASSIGNED" or n_trials <= 0:
            out.append(record)
    return out
