"""High-level fitting entry points that connect renderers to optimization."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .chmp_evaluation import ObservationEvaluationContext, evaluate_modeled_trial
from .metrics import MetricValues
from .obs_preprocessing import SliceObservationReference
from .optimize import (
    InitialQ0Evaluations,
    MetricName,
    ProgressCallback,
    ProgressStartCallback,
    Q0MetricEvaluation,
    Q0OptimizationResult,
    find_best_q0,
)
from .q0_search import merge_q0_stage_results, resolve_q0_search_stages


class Q0MapRenderer(Protocol):
    """Protocol for render adapters that synthesize a map for a given Q0."""

    def render(self, q0: float) -> np.ndarray:
        """Return modeled map corresponding to Q0."""


def observation_reference_to_evaluation_context(
    reference: SliceObservationReference,
    *,
    use_smoothed_obs_max: bool = True,
    emthreshold: float = 0.1,
) -> ObservationEvaluationContext:
    policy = str(reference.shift_policy).strip().lower()
    return ObservationEvaluationContext(
        model_header=reference.target_header.copy(),
        shift_policy=policy,
        max_shift_arcsec=float(reference.max_shift_arcsec or 0.0),
        xy_shift_arcsec=tuple(reference.xy_shift_arcsec),
        observed=np.asarray(reference.observed, dtype=float),
        sigma=np.asarray(reference.sigma, dtype=float),
        observation_canvas=(
            None if reference.observation_canvas is None else np.asarray(reference.observation_canvas, dtype=float)
        ),
        sigma_canvas=None if reference.sigma_canvas is None else np.asarray(reference.sigma_canvas, dtype=float),
        canvas_header=reference.canvas_header.copy() if reference.canvas_header is not None else None,
        use_smoothed_obs_max=use_smoothed_obs_max,
        emthreshold=emthreshold,
    )


def _build_evaluation_context(
    *,
    observed_arr: np.ndarray,
    sigma_arr: np.ndarray,
    observation_reference: SliceObservationReference | None,
    evaluation_context: ObservationEvaluationContext | None,
    use_smoothed_obs_max: bool,
    emthreshold: float,
) -> ObservationEvaluationContext:
    context = evaluation_context
    if context is None and observation_reference is not None:
        context = observation_reference_to_evaluation_context(
            observation_reference,
            use_smoothed_obs_max=use_smoothed_obs_max,
            emthreshold=emthreshold,
        )
    if context is None:
        from astropy.io import fits

        model_header = fits.Header()
        model_header["NAXIS"] = 2
        model_header["NAXIS1"] = int(observed_arr.shape[1])
        model_header["NAXIS2"] = int(observed_arr.shape[0])
        model_header["CDELT1"] = 1.0
        model_header["CDELT2"] = 1.0
        model_header["CRPIX1"] = (float(observed_arr.shape[1]) + 1.0) / 2.0
        model_header["CRPIX2"] = (float(observed_arr.shape[0]) + 1.0) / 2.0
        model_header["CRVAL1"] = 0.0
        model_header["CRVAL2"] = 0.0
        context = ObservationEvaluationContext(
            model_header=model_header,
            shift_policy="fixed",
            observed=observed_arr,
            sigma=sigma_arr,
            use_smoothed_obs_max=use_smoothed_obs_max,
            emthreshold=emthreshold,
        )
    return context


def _run_single_stage_q0_fit(
    renderer: Q0MapRenderer,
    observed_arr: np.ndarray,
    *,
    context: ObservationEvaluationContext,
    stage_mask_type: str,
    mask_stage_label: str,
    explicit_mask_arr: np.ndarray | None,
    threshold: float,
    use_emthreshold: bool,
    q0_min: float,
    q0_max: float,
    hard_q0_min: float | None,
    hard_q0_max: float | None,
    target_metric: MetricName,
    xatol: float,
    maxiter: int,
    adaptive_bracketing: bool,
    q0_start: float | None,
    q0_step: float,
    max_bracket_steps: int,
    progress_start_callback: ProgressStartCallback | None,
    progress_callback: ProgressCallback | None,
    initial_evaluations: InitialQ0Evaluations | None,
    emthreshold: float = 0.1,
) -> Q0OptimizationResult:
    def metric_function(q0: float) -> Q0MetricEvaluation:
        modeled_arr = np.asarray(renderer.render(float(q0)), dtype=float)
        if modeled_arr.shape != observed_arr.shape:
            raise ValueError("renderer output shape must match observed shape")

        def miss_fn() -> float | None:
            fn = getattr(renderer, "ebtel_miss_ratio", None)
            if fn is None:
                return None
            try:
                return fn(float(q0))
            except TypeError:
                try:
                    return fn()
                except Exception:
                    return None

        return evaluate_modeled_trial(
            modeled_arr,
            context,
            threshold=threshold,
            mask_type=stage_mask_type,
            explicit_mask=explicit_mask_arr,
            ebtel_miss_ratio_fn=miss_fn if hasattr(renderer, "ebtel_miss_ratio") else None,
            use_emthreshold=use_emthreshold,
            mask_stage=mask_stage_label,
        )

    return find_best_q0(
        metric_function,
        q0_min=q0_min,
        q0_max=q0_max,
        hard_q0_min=hard_q0_min,
        hard_q0_max=hard_q0_max,
        target_metric=target_metric,
        xatol=xatol,
        maxiter=maxiter,
        adaptive_bracketing=adaptive_bracketing,
        q0_start=q0_start,
        q0_step=q0_step,
        max_bracket_steps=max_bracket_steps,
        progress_start_callback=progress_start_callback,
        progress_callback=progress_callback,
        initial_evaluations=initial_evaluations,
        emthreshold=emthreshold,
    )


def _clamp_q0_start_for_interval(
    q0_value: float,
    *,
    q0_min: float,
    q0_max: float,
    hard_q0_min: float | None = None,
    hard_q0_max: float | None = None,
) -> float:
    """Clamp a warm-start q0 into the user initialization interval."""
    lower = float(q0_min)
    upper = float(q0_max)
    if hard_q0_min is not None:
        lower = max(lower, float(hard_q0_min))
    if hard_q0_max is not None:
        upper = min(upper, float(hard_q0_max))
    return float(min(max(float(q0_value), lower), upper))


def fit_q0_to_observation(
    renderer: Q0MapRenderer,
    observed: np.ndarray,
    sigma: np.ndarray | None,
    *,
    q0_min: float,
    q0_max: float,
    hard_q0_min: float | None = None,
    hard_q0_max: float | None = None,
    threshold: float = 0.1,
    mask_type: str = "union",
    explicit_mask: np.ndarray | None = None,
    target_metric: MetricName = "chi2",
    xatol: float = 1e-3,
    maxiter: int = 200,
    adaptive_bracketing: bool = False,
    q0_start: float | None = None,
    q0_step: float = 1.61803398875,
    max_bracket_steps: int = 12,
    progress_start_callback: ProgressStartCallback | None = None,
    progress_callback: ProgressCallback | None = None,
    initial_evaluations: InitialQ0Evaluations | None = None,
    observation_reference: SliceObservationReference | None = None,
    evaluation_context: ObservationEvaluationContext | None = None,
    use_smoothed_obs_max: bool = True,
    use_emthreshold: bool = True,
    emthreshold: float = 0.1,
    q0_search_stages: tuple[str, ...] | None = None,
) -> Q0OptimizationResult:
    """Optimize Q0 by comparing rendered maps against observed maps.

    When ``observation_reference`` or ``evaluation_context`` is supplied, CHMP-style
    per-trial shifting and safeguards are applied. Otherwise a fixed model-FOV
    reference is used (legacy path).

    Pass ``q0_search_stages=("data", "union")`` for the opt-in two-stage mask
    search. The default resolves to a single stage using ``mask_type``.
    """
    observed_arr = np.asarray(observed, dtype=float)

    actual_target_metric = target_metric
    if sigma is None:
        import warnings

        warnings.warn(
            "sigma is None (map noise estimation failed). "
            f"Falling back from {target_metric} to eta2 metric.",
            UserWarning,
        )
        actual_target_metric = "eta2"
        sigma_arr = np.ones_like(observed_arr)
    else:
        sigma_arr = np.asarray(sigma, dtype=float)
        if observed_arr.shape != sigma_arr.shape:
            raise ValueError("observed and sigma must have identical shapes")

    explicit_mask_arr = None if explicit_mask is None else np.asarray(explicit_mask, dtype=bool)
    if explicit_mask_arr is not None and explicit_mask_arr.shape != observed_arr.shape:
        raise ValueError("explicit_mask must have identical shape to observed")

    context = _build_evaluation_context(
        observed_arr=observed_arr,
        sigma_arr=sigma_arr,
        observation_reference=observation_reference,
        evaluation_context=evaluation_context,
        use_smoothed_obs_max=use_smoothed_obs_max,
        emthreshold=emthreshold,
    )

    resolved_stages = resolve_q0_search_stages(
        q0_search_stages=q0_search_stages,
        mask_type=mask_type,
        explicit_mask=explicit_mask_arr,
    )

    stage_results: list[Q0OptimizationResult] = []
    stage_q0_start = q0_start
    stage_initial_evaluations = initial_evaluations
    for stage_index, stage_mask in enumerate(resolved_stages):
        if stage_mask == "explicit":
            stage_mask_type = str(mask_type)
            stage_label = "explicit"
        else:
            stage_mask_type = stage_mask
            stage_label = stage_mask

        stage_result = _run_single_stage_q0_fit(
            renderer,
            observed_arr,
            context=context,
            stage_mask_type=stage_mask_type,
            mask_stage_label=stage_label,
            explicit_mask_arr=explicit_mask_arr,
            threshold=threshold,
            use_emthreshold=use_emthreshold,
            q0_min=q0_min,
            q0_max=q0_max,
            hard_q0_min=hard_q0_min,
            hard_q0_max=hard_q0_max,
            target_metric=actual_target_metric,
            xatol=xatol,
            maxiter=maxiter,
            adaptive_bracketing=adaptive_bracketing,
            q0_start=stage_q0_start,
            q0_step=q0_step,
            max_bracket_steps=max_bracket_steps,
            progress_start_callback=progress_start_callback,
            progress_callback=progress_callback,
            initial_evaluations=stage_initial_evaluations if stage_index == 0 else None,
            emthreshold=emthreshold,
        )
        stage_results.append(stage_result)
        stage_q0_start = _clamp_q0_start_for_interval(
            float(stage_result.q0),
            q0_min=q0_min,
            q0_max=q0_max,
            hard_q0_min=hard_q0_min,
            hard_q0_max=hard_q0_max,
        )
        stage_initial_evaluations = None

    return merge_q0_stage_results(tuple(stage_results), resolved_stages)
