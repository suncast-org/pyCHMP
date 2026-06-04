"""Warm-start Q0 trial loading and rescoring for a fixed (a, b) grid point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .ab_scan_artifacts import _derive_display_maps_from_raw, _read_map_store_ref_array
from .chmp_evaluation import ObservationEvaluationContext, evaluate_modeled_trial
from .grid_points import (
    GRID_POINTS_GROUP,
    GRID_POINTS_TRIALS_GROUP,
    SEARCHES_GROUP,
    SLICE_CONTAINER_GROUP,
    GridTrialCommittedEvent,
    _H5PY_FILE,
    _load_grid_point_trials,
    _resolve_slice_group,
    find_grid_point_group,
    reset_grid_point_for_rerun,
    grid_point_finite_q0_trials_have_map_store_links,
    resolve_trial_shift_commit_fields,
    select_fit_trials_for_viewer,
)
from .metrics import MetricValues
from .optimize import InitialQ0Evaluations, MetricName, Q0MetricEvaluation
from .slice_map_index import SliceMapIndex


def load_warm_q0_evaluations_for_grid_point(
    artifact_h5: Path,
    *,
    slice_key: str,
    search_id: str | None,
    a_value: float,
    b_value: float,
    context: ObservationEvaluationContext,
    threshold: float,
    explicit_mask: np.ndarray | None,
    target_metric: str,
    use_emthreshold: bool = True,
    ebtel_miss_ratio_fn: Any = None,
    slice_map_index: SliceMapIndex | None = None,
) -> InitialQ0Evaluations | None:
    """Rescore stored trial maps for one (a, b); prefer slice map index when provided."""
    if slice_map_index is not None and slice_map_index.has_point(float(a_value), float(b_value)):
        return _warm_evaluations_from_slice_index(
            artifact_h5,
            slice_map_index=slice_map_index,
            a_value=float(a_value),
            b_value=float(b_value),
            context=context,
            threshold=float(threshold),
            explicit_mask=explicit_mask,
            target_metric=str(target_metric),
            use_emthreshold=bool(use_emthreshold),
            ebtel_miss_ratio_fn=ebtel_miss_ratio_fn,
        )

    evaluations: dict[float, Q0MetricEvaluation] = {}
    with _H5PY_FILE(artifact_h5, "r") as h5_file:
        if SLICE_CONTAINER_GROUP not in h5_file:
            return None
        slice_group, _descriptors, _selected_key = _resolve_slice_group(
            h5_file,
            slice_key=str(slice_key).strip(),
            allow_missing=True,
        )
        if slice_group is None or SEARCHES_GROUP not in slice_group:
            return None
        searches = slice_group[SEARCHES_GROUP]
        preferred = str(search_id or "").strip()
        search_ids = [preferred] if preferred and preferred in searches else []
        search_ids.extend(sorted(str(key) for key in searches.keys() if str(key) not in search_ids))
        point_group = None
        for candidate_id in search_ids:
            found = find_grid_point_group(searches[candidate_id], a=float(a_value), b=float(b_value))
            if found is not None:
                _point_id, point_group = found
                break
        if point_group is None:
            return None
        common = slice_group.get("common")
        if common is None:
            return None
        observed_template = np.asarray(common["observed"][()], dtype=float)
        psf_kernel = common["psf_kernel"][()] if "psf_kernel" in common else None

        trials = select_fit_trials_for_viewer(_load_grid_point_trials(point_group, include_maps=False))
        if not trials:
            return None
        for trial in trials:
            q0_value = float(trial["q0"])
            raw_ref = str(trial.get("raw_map_ref", "") or "").strip()
            if not raw_ref:
                continue
            evaluation = _rescore_raw_map_ref(
                h5_file,
                raw_ref,
                q0_value=q0_value,
                observed_template=observed_template,
                psf_kernel=psf_kernel,
                context=context,
                threshold=float(threshold),
                explicit_mask=explicit_mask,
                target_metric=target_metric,
                use_emthreshold=use_emthreshold,
                ebtel_miss_ratio_fn=ebtel_miss_ratio_fn,
            )
            if evaluation is not None:
                evaluations[float(q0_value)] = evaluation

    return evaluations or None


def _rescore_raw_map_ref(
    h5_file: Any,
    raw_ref: str,
    *,
    q0_value: float,
    observed_template: np.ndarray,
    psf_kernel: Any,
    context: ObservationEvaluationContext,
    threshold: float,
    explicit_mask: np.ndarray | None,
    target_metric: str,
    use_emthreshold: bool,
    ebtel_miss_ratio_fn: Any = None,
) -> Q0MetricEvaluation | None:
    raw_modeled = _read_map_store_ref_array(h5_file, raw_ref)
    if raw_modeled is None:
        return None
    _raw_display, modeled, _residual, _has_raw = _derive_display_maps_from_raw(
        raw_modeled,
        observed_template=observed_template,
        psf_kernel=psf_kernel,
    )
    if modeled is None:
        return None
    evaluation = evaluate_modeled_trial(
        modeled,
        context,
        threshold=float(threshold),
        mask_type="union",
        explicit_mask=explicit_mask,
        ebtel_miss_ratio_fn=ebtel_miss_ratio_fn,
        use_emthreshold=use_emthreshold,
    )
    if not evaluation.is_valid:
        return None
    return evaluation


def _warm_evaluations_from_slice_index(
    artifact_h5: Path,
    *,
    slice_map_index: SliceMapIndex,
    a_value: float,
    b_value: float,
    context: ObservationEvaluationContext,
    threshold: float,
    explicit_mask: np.ndarray | None,
    target_metric: str,
    use_emthreshold: bool = True,
    ebtel_miss_ratio_fn: Any = None,
) -> InitialQ0Evaluations | None:
    entries = slice_map_index.entries_for_point(float(a_value), float(b_value))
    if not entries:
        return None
    evaluations: dict[float, Q0MetricEvaluation] = {}
    with _H5PY_FILE(artifact_h5, "r") as h5_file:
        slice_group, _descriptors, _selected_key = _resolve_slice_group(
            h5_file,
            slice_key=str(slice_map_index.slice_key).strip(),
            allow_missing=True,
        )
        if slice_group is None:
            return None
        common = slice_group.get("common")
        if common is None:
            return None
        observed_template = np.asarray(common["observed"][()], dtype=float)
        psf_kernel = common["psf_kernel"][()] if "psf_kernel" in common else None
        for entry in entries:
            evaluation = _rescore_raw_map_ref(
                h5_file,
                entry.raw_map_ref,
                q0_value=float(entry.q0),
                observed_template=observed_template,
                psf_kernel=psf_kernel,
                context=context,
                threshold=float(threshold),
                explicit_mask=explicit_mask,
                target_metric=target_metric,
                use_emthreshold=use_emthreshold,
                ebtel_miss_ratio_fn=ebtel_miss_ratio_fn,
            )
            if evaluation is not None:
                evaluations[float(entry.q0)] = evaluation
    return evaluations or None


def _evaluation_from_stored_grid_trial(
    trial: dict[str, Any],
    *,
    target_metric: str,
) -> Q0MetricEvaluation | None:
    """Build optimizer warm-start payload from committed trial scalars (no map rescore)."""
    raw_ref = str(trial.get("raw_map_ref", "") or "").strip()
    if not raw_ref:
        return None
    try:
        q0_value = float(trial["q0"])
    except Exception:
        return None
    if not (np.isfinite(q0_value) and q0_value > 0.0):
        return None
    metadata = dict(trial.get("trial_metadata") or {})
    chi2 = float(trial.get("chi2", np.nan))
    rho2 = float(trial.get("rho2", np.nan))
    eta2 = float(trial.get("eta2", np.nan))
    target_value = float(trial.get("target_metric_value", np.nan))
    metric_name = str(target_metric or "chi2").strip().lower()
    if metric_name == "chi2" and not np.isfinite(chi2) and np.isfinite(target_value):
        chi2 = target_value
    elif metric_name == "rho2" and not np.isfinite(rho2) and np.isfinite(target_value):
        rho2 = target_value
    elif metric_name == "eta2" and not np.isfinite(eta2) and np.isfinite(target_value):
        eta2 = target_value
    if not any(np.isfinite(value) for value in (chi2, rho2, eta2)):
        return None
    return Q0MetricEvaluation(
        metrics=MetricValues(
            chi2=float(chi2),
            rho2=float(rho2),
            eta2=float(eta2),
        ),
        is_valid=True,
        message="restored from stored grid trial",
        shift_x_arcsec=float(metadata.get("shift_x_arcsec", 0.0)),
        shift_y_arcsec=float(metadata.get("shift_y_arcsec", 0.0)),
        find_shift_valid=bool(metadata.get("shift_valid", True)),
        mask_stage=str(metadata.get("stage", "")),
    )


def _metric_value(metrics: MetricValues, target_metric: str) -> float:
    metric_name = str(target_metric).strip().lower()
    if metric_name == "chi2":
        return float(metrics.chi2)
    if metric_name == "rho2":
        return float(metrics.rho2)
    if metric_name == "eta2":
        return float(metrics.eta2)
    raise ValueError(f"unsupported target_metric: {target_metric!r}")


def initial_evaluations_from_grid_trials(
    artifact_h5: Path,
    trials: list[dict[str, Any]],
    *,
    slice_key: str,
    target_metric: str,
    context: ObservationEvaluationContext,
    threshold: float,
    explicit_mask: np.ndarray | None,
    use_emthreshold: bool = True,
    rescore: bool = True,
) -> InitialQ0Evaluations | None:
    """Build optimizer warm-start from grid trials (rescored maps or stored trial scalars)."""
    fit_trials = select_fit_trials_for_viewer(trials)
    if not fit_trials:
        return None
    evaluations: dict[float, Q0MetricEvaluation] = {}
    if not rescore:
        for trial in sorted(fit_trials, key=lambda item: float(item["q0"])):
            evaluation = _evaluation_from_stored_grid_trial(trial, target_metric=str(target_metric))
            if evaluation is not None:
                evaluations[float(trial["q0"])] = evaluation
        return evaluations or None
    with _H5PY_FILE(artifact_h5, "r") as h5_file:
        slice_group, _descriptors, _selected_key = _resolve_slice_group(
            h5_file,
            slice_key=str(slice_key).strip(),
            allow_missing=True,
        )
        if slice_group is None:
            return None
        common = slice_group.get("common")
        if common is None:
            return None
        observed_template = np.asarray(common["observed"][()], dtype=float)
        psf_kernel = common["psf_kernel"][()] if "psf_kernel" in common else None
        for trial in sorted(fit_trials, key=lambda item: float(item["q0"])):
            q0_value = float(trial["q0"])
            raw_ref = str(trial.get("raw_map_ref", "") or "").strip()
            if not raw_ref:
                continue
            evaluation = _rescore_raw_map_ref(
                h5_file,
                raw_ref,
                q0_value=q0_value,
                observed_template=observed_template,
                psf_kernel=psf_kernel,
                context=context,
                threshold=float(threshold),
                explicit_mask=explicit_mask,
                target_metric=str(target_metric),
                use_emthreshold=bool(use_emthreshold),
            )
            if evaluation is not None:
                evaluations[q0_value] = evaluation
    return evaluations or None


def build_warm_grid_trial_commit_events(
    artifact_h5: Path,
    *,
    point_id: str,
    slice_map_index: SliceMapIndex,
    a_value: float,
    b_value: float,
    context: ObservationEvaluationContext,
    threshold: float,
    explicit_mask: np.ndarray | None,
    target_metric: str,
    use_emthreshold: bool = True,
) -> list[GridTrialCommittedEvent]:
    """Rescore compatible map_store maps and return grid commit events (linked refs, all metrics)."""
    entries = slice_map_index.entries_for_point(float(a_value), float(b_value))
    if not entries:
        return []
    scored: list[tuple[float, str, Q0MetricEvaluation]] = []
    with _H5PY_FILE(artifact_h5, "r") as h5_file:
        slice_group, _descriptors, _selected_key = _resolve_slice_group(
            h5_file,
            slice_key=str(slice_map_index.slice_key).strip(),
            allow_missing=True,
        )
        if slice_group is None:
            return []
        common = slice_group.get("common")
        if common is None:
            return []
        observed_template = np.asarray(common["observed"][()], dtype=float)
        psf_kernel = common["psf_kernel"][()] if "psf_kernel" in common else None
        for entry in entries:
            evaluation = _rescore_raw_map_ref(
                h5_file,
                entry.raw_map_ref,
                q0_value=float(entry.q0),
                observed_template=observed_template,
                psf_kernel=psf_kernel,
                context=context,
                threshold=float(threshold),
                explicit_mask=explicit_mask,
                target_metric=str(target_metric),
                use_emthreshold=bool(use_emthreshold),
            )
            if evaluation is None:
                continue
            scored.append((float(entry.q0), str(entry.raw_map_ref), evaluation))
    if not scored:
        return []
    scored.sort(key=lambda item: item[0])
    metric_name = str(target_metric).strip().lower()
    shift_x_trials = [float(item[2].shift_x_arcsec) for item in scored]
    shift_y_trials = [float(item[2].shift_y_arcsec) for item in scored]
    shift_valid_trials = [bool(item[2].find_shift_valid) for item in scored]
    metric_trials = [_metric_value(item[2].metrics, metric_name) for item in scored]
    finite = [(idx, value) for idx, value in enumerate(metric_trials) if np.isfinite(float(value))]
    best_index = int(finite[0][0]) if finite else 0
    best_metric = float(metric_trials[best_index]) if metric_trials else float("nan")
    if finite:
        best_index, best_metric = min(finite, key=lambda item: item[1])
    events: list[GridTrialCommittedEvent] = []
    for trial_index, (q0_value, raw_ref, evaluation) in enumerate(scored):
        shift_metadata, shift_x, shift_y, shift_valid = resolve_trial_shift_commit_fields(
            trial_index=int(trial_index),
            shift_x_trials=shift_x_trials,
            shift_y_trials=shift_y_trials,
            shift_valid_trials=shift_valid_trials,
        )
        events.append(
            GridTrialCommittedEvent(
                point_id=str(point_id),
                trial_index=int(trial_index),
                q0=float(q0_value),
                metric=float(metric_trials[trial_index]),
                next_q0=float(q0_value),
                best_trial_index=int(best_index),
                best_metric=float(best_metric),
                raw_modeled_map=None,
                raw_map_ref=str(raw_ref),
                trial_metadata=shift_metadata,
                shift_x=shift_x,
                shift_y=shift_y,
                shift_valid=shift_valid,
                chi2=float(evaluation.metrics.chi2),
                rho2=float(evaluation.metrics.rho2),
                eta2=float(evaluation.metrics.eta2),
            )
        )
    return events


def count_warm_rescorable_points(
    artifact_h5: Path,
    *,
    slice_map_index: SliceMapIndex,
    context: ObservationEvaluationContext,
    threshold: float,
    explicit_mask: np.ndarray | None,
    target_metric: str,
    use_emthreshold: bool = True,
) -> int:
    """Count (a,b) with at least one valid rescore (full metric pass per map; not for preload)."""
    count = 0
    for a_value, b_value in slice_map_index.point_keys():
        evaluations = _warm_evaluations_from_slice_index(
            artifact_h5,
            slice_map_index=slice_map_index,
            a_value=float(a_value),
            b_value=float(b_value),
            context=context,
            threshold=float(threshold),
            explicit_mask=explicit_mask,
            target_metric=str(target_metric),
            use_emthreshold=bool(use_emthreshold),
        )
        if evaluations:
            count += 1
    return count
