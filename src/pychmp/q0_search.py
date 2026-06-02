"""Q0 mask-stage orchestration for CHMP-faithful and opt-in two-stage search."""

from __future__ import annotations

from dataclasses import replace

from .optimize import Q0OptimizationResult

Q0_MASK_STAGE_NAMES: frozenset[str] = frozenset({"union", "data", "model", "and"})


def parse_q0_search_stages(text: str | None) -> tuple[str, ...] | None:
    """Parse a comma-separated stage list such as ``data,union``."""
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    stages = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    if not stages:
        return None
    unknown = [stage for stage in stages if stage not in Q0_MASK_STAGE_NAMES]
    if unknown:
        allowed = ", ".join(sorted(Q0_MASK_STAGE_NAMES))
        raise ValueError(
            f"unsupported q0 search stage(s): {', '.join(unknown)}; allowed values are {allowed}"
        )
    return stages


def resolve_q0_search_stages(
    *,
    q0_search_stages: tuple[str, ...] | None,
    mask_type: str,
    explicit_mask: object | None,
) -> tuple[str, ...]:
    """Resolve the effective mask stages for one Q0 optimization."""
    if explicit_mask is not None:
        if q0_search_stages is not None and len(q0_search_stages) > 1:
            raise ValueError(
                "two-stage q0 search (--q0-search-stages with multiple stages) is incompatible "
                "with an explicit FITS metrics mask"
            )
        return ("explicit",)

    if q0_search_stages is not None:
        return q0_search_stages

    normalized_mask_type = str(mask_type).strip().lower()
    if normalized_mask_type in {"explicit", "explicit_fits"}:
        return ("explicit",)
    if normalized_mask_type in Q0_MASK_STAGE_NAMES:
        return (normalized_mask_type,)
    return ("union",)


def merge_q0_stage_results(
    stage_results: tuple[Q0OptimizationResult, ...],
    q0_search_stages: tuple[str, ...],
) -> Q0OptimizationResult:
    """Merge per-stage optimizer outputs into one combined trial history."""
    if not stage_results:
        raise ValueError("stage_results must be non-empty")
    if len(stage_results) != len(q0_search_stages):
        raise ValueError("stage_results and q0_search_stages must have the same length")

    final = stage_results[-1]
    if len(stage_results) == 1:
        return replace(final, q0_search_stages=q0_search_stages)

    trial_q0: list[float] = []
    trial_objective_values: list[float] = []
    trial_chi2_values: list[float] = []
    trial_rho2_values: list[float] = []
    trial_eta2_values: list[float] = []
    trial_shift_x: list[float] = []
    trial_shift_y: list[float] = []
    trial_shift_valid: list[bool] = []
    trial_mask_stages: list[str] = []

    messages: list[str] = []
    total_nfev = 0
    total_nit = 0
    for stage_index, (stage_mask, stage_result) in enumerate(
        zip(q0_search_stages, stage_results, strict=True),
        start=1,
    ):
        trial_q0.extend(float(v) for v in stage_result.trial_q0)
        trial_objective_values.extend(float(v) for v in stage_result.trial_objective_values)
        trial_chi2_values.extend(float(v) for v in stage_result.trial_chi2_values)
        trial_rho2_values.extend(float(v) for v in stage_result.trial_rho2_values)
        trial_eta2_values.extend(float(v) for v in stage_result.trial_eta2_values)
        trial_shift_x.extend(float(v) for v in stage_result.trial_shift_x_arcsec)
        trial_shift_y.extend(float(v) for v in stage_result.trial_shift_y_arcsec)
        trial_shift_valid.extend(bool(v) for v in stage_result.trial_find_shift_valid)
        stage_trial_masks = stage_result.trial_mask_stages
        if stage_trial_masks and len(stage_trial_masks) == len(stage_result.trial_q0):
            trial_mask_stages.extend(str(v) for v in stage_trial_masks)
        else:
            trial_mask_stages.extend([stage_mask] * len(stage_result.trial_q0))
        total_nfev += int(stage_result.nfev)
        total_nit += int(stage_result.nit)
        messages.append(
            f"stage {stage_index}/{len(stage_results)} ({stage_mask}): {stage_result.message}"
        )

    return Q0OptimizationResult(
        q0=float(final.q0),
        objective_value=float(final.objective_value),
        metrics=final.metrics,
        target_metric=final.target_metric,
        success=bool(final.success),
        nfev=int(total_nfev),
        nit=int(total_nit),
        message="; ".join(messages),
        used_adaptive_bracketing=bool(final.used_adaptive_bracketing),
        bracket_found=bool(final.bracket_found),
        bracket=final.bracket,
        boundary_constrained=bool(final.boundary_constrained),
        trial_q0=tuple(trial_q0),
        trial_objective_values=tuple(trial_objective_values),
        trial_chi2_values=tuple(trial_chi2_values),
        trial_rho2_values=tuple(trial_rho2_values),
        trial_eta2_values=tuple(trial_eta2_values),
        trial_shift_x_arcsec=tuple(trial_shift_x),
        trial_shift_y_arcsec=tuple(trial_shift_y),
        trial_find_shift_valid=tuple(trial_shift_valid),
        q0_search_stages=q0_search_stages,
        trial_mask_stages=tuple(trial_mask_stages),
    )
