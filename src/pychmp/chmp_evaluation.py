"""CHMP-style trial evaluation with shift, mask safeguards, and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from astropy.io import fits

from .metrics import (
    MetricValues,
    chmp_mask_valid,
    compute_metrics,
    mask_area_fractions,
    resolve_threshold_mask,
    should_prefer_data_mask_over_union,
    smoothed_observation_max,
    threshold_data_mask,
    union_mask_flux_totals,
)
from .obs_alignment import (
    DEFAULT_MAX_SHIFT_ARSEC,
    FIND_SHIFT_VERSION,
    extract_observation_to_model_fov,
    find_shift,
)
from .optimize import Q0MetricEvaluation

CHMP_EVAL_POLICY_VERSION = "1"
DEFAULT_EMTHRESHOLD = 0.1


@dataclass(frozen=True, slots=True)
class ObservationEvaluationContext:
    """Inputs required to evaluate one modeled map against observations."""

    model_header: fits.Header
    shift_policy: str = "auto"
    max_shift_arcsec: float = DEFAULT_MAX_SHIFT_ARSEC
    xy_shift_arcsec: tuple[float, float] = (0.0, 0.0)
    observed: np.ndarray | None = None
    sigma: np.ndarray | None = None
    observation_canvas: np.ndarray | None = None
    sigma_canvas: np.ndarray | None = None
    canvas_header: fits.Header | None = None
    use_smoothed_obs_max: bool = True
    emthreshold: float = DEFAULT_EMTHRESHOLD


EbtelMissRatioFn = Callable[[], float | None]


def _invalid_metrics() -> MetricValues:
    return MetricValues(chi2=float("nan"), rho2=float("nan"), eta2=float("nan"))


def _pixel_scales_from_header(header: fits.Header) -> tuple[float, float]:
    return abs(float(header["CDELT1"])), abs(float(header["CDELT2"]))


def resolve_trial_observation_pair(
    context: ObservationEvaluationContext,
    modeled: np.ndarray,
    *,
    shift_x_arcsec: float | None = None,
    shift_y_arcsec: float | None = None,
    find_shift_valid: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, bool, str]:
    policy = str(context.shift_policy).strip().lower()
    if policy == "fixed":
        if context.observed is None or context.sigma is None:
            raise ValueError("fixed shift_policy requires observed and sigma arrays")
        return (
            np.asarray(context.observed, dtype=float),
            np.asarray(context.sigma, dtype=float),
            float(context.xy_shift_arcsec[0]),
            float(context.xy_shift_arcsec[1]),
            True,
            "",
        )

    if context.observation_canvas is None or context.sigma_canvas is None or context.canvas_header is None:
        raise ValueError("auto shift_policy requires observation canvas reference")

    if shift_x_arcsec is None or shift_y_arcsec is None:
        shift = find_shift(
            context.observation_canvas,
            context.canvas_header,
            context.model_header,
            modeled,
            max_shift_arcsec=float(context.max_shift_arcsec),
        )
        shift_x = float(shift.shift_x_arcsec)
        shift_y = float(shift.shift_y_arcsec)
        valid = bool(shift.valid)
        message = str(shift.message)
    else:
        shift_x = float(shift_x_arcsec)
        shift_y = float(shift_y_arcsec)
        valid = bool(find_shift_valid) if find_shift_valid is not None else True
        message = "" if valid else "stored shift marked invalid"

    observed, sigma = extract_observation_to_model_fov(
        context.observation_canvas,
        context.canvas_header,
        context.model_header,
        shift_x_arcsec=shift_x,
        shift_y_arcsec=shift_y,
        canvas_sigma=context.sigma_canvas,
    )
    if sigma is None:
        raise ValueError("sigma canvas is required for auto shift_policy")
    return observed, sigma, shift_x, shift_y, valid, message


def evaluate_modeled_trial(
    modeled: np.ndarray,
    context: ObservationEvaluationContext,
    *,
    threshold: float,
    mask_type: str,
    explicit_mask: np.ndarray | None = None,
    ebtel_miss_ratio_fn: EbtelMissRatioFn | None = None,
    use_emthreshold: bool = True,
    mask_stage: str = "",
    shift_x_arcsec: float | None = None,
    shift_y_arcsec: float | None = None,
    find_shift_valid: bool | None = None,
) -> Q0MetricEvaluation:
    modeled_arr = np.asarray(modeled, dtype=float)
    resolved_mask_stage = str(mask_stage or mask_type).strip().lower()
    try:
        observed_arr, trial_sigma, shift_x, shift_y, shift_valid, shift_message = resolve_trial_observation_pair(
            context,
            modeled_arr,
            shift_x_arcsec=shift_x_arcsec,
            shift_y_arcsec=shift_y_arcsec,
            find_shift_valid=find_shift_valid,
        )
    except Exception as exc:
        return Q0MetricEvaluation(
            metrics=_invalid_metrics(),
            is_valid=False,
            message=str(exc),
            shift_x_arcsec=float("nan"),
            shift_y_arcsec=float("nan"),
            find_shift_valid=False,
            mask_stage=resolved_mask_stage,
        )

    if not shift_valid:
        return Q0MetricEvaluation(
            metrics=_invalid_metrics(),
            is_valid=False,
            message=shift_message or "FindShift failed",
            shift_x_arcsec=shift_x,
            shift_y_arcsec=shift_y,
            find_shift_valid=False,
            mask_stage=resolved_mask_stage,
        )

    if use_emthreshold and ebtel_miss_ratio_fn is not None:
        miss_ratio = ebtel_miss_ratio_fn()
        if miss_ratio is not None and float(miss_ratio) > float(context.emthreshold):
            return Q0MetricEvaluation(
                metrics=_invalid_metrics(),
                is_valid=False,
                message=f"EBTEL miss ratio {float(miss_ratio):.3g} exceeds EMthreshold {float(context.emthreshold):.3g}",
                shift_x_arcsec=shift_x,
                shift_y_arcsec=shift_y,
                find_shift_valid=True,
                mask_stage=resolved_mask_stage,
            )

    scale_x, scale_y = _pixel_scales_from_header(context.model_header)
    obs_peak = (
        smoothed_observation_max(observed_arr, pixel_scale_x_arcsec=scale_x, pixel_scale_y_arcsec=scale_y)
        if context.use_smoothed_obs_max
        else float(np.max(observed_arr))
    )

    resolved_mask_type = mask_type.strip().lower()
    mask_fn = resolve_threshold_mask(mask_type)
    if explicit_mask is not None:
        mask = np.asarray(explicit_mask, dtype=bool)
        effective_mask_stage = resolved_mask_stage
    elif resolved_mask_type in {"union", "data", "and"}:
        mask = mask_fn(observed_arr, modeled_arr, threshold, obs_max=obs_peak)
        effective_mask_stage = resolved_mask_stage
        mask_obs_frac, mask_mod_frac = mask_area_fractions(
            observed_arr,
            modeled_arr,
            threshold=threshold,
            obs_max=obs_peak,
        )
        obs_flux, mod_flux = union_mask_flux_totals(
            observed_arr,
            modeled_arr,
            mask,
            pixel_scale_x_arcsec=scale_x,
            pixel_scale_y_arcsec=scale_y,
        )
        valid_mask, mask_message = chmp_mask_valid(mask_obs_frac, mask_mod_frac)
        if resolved_mask_type == "union" and (
            not valid_mask
            or should_prefer_data_mask_over_union(
                mask_obs_fraction=mask_obs_frac,
                mask_mod_fraction=mask_mod_frac,
                total_observed_flux=obs_flux,
                total_modeled_flux=mod_flux,
            )
        ):
            mask = threshold_data_mask(observed_arr, modeled_arr, threshold, obs_max=obs_peak)
            effective_mask_stage = "data"
            mask_obs_frac, mask_mod_frac = mask_area_fractions(
                observed_arr,
                modeled_arr,
                threshold=threshold,
                obs_max=obs_peak,
            )
            obs_flux, mod_flux = union_mask_flux_totals(
                observed_arr,
                modeled_arr,
                mask,
                pixel_scale_x_arcsec=scale_x,
                pixel_scale_y_arcsec=scale_y,
            )
            valid_mask, mask_message = chmp_mask_valid(mask_obs_frac, mask_mod_frac)
            if effective_mask_stage == "data":
                valid_mask = mask_obs_frac <= 0.99
                mask_message = "" if valid_mask else "observation mask fraction > 0.99"
    else:
        mask = mask_fn(observed_arr, modeled_arr, threshold)
        effective_mask_stage = resolved_mask_stage
        mask_obs_frac, mask_mod_frac = mask_area_fractions(
            observed_arr,
            modeled_arr,
            threshold=threshold,
            obs_max=obs_peak,
        )
        obs_flux, mod_flux = union_mask_flux_totals(
            observed_arr,
            modeled_arr,
            mask,
            pixel_scale_x_arcsec=scale_x,
            pixel_scale_y_arcsec=scale_y,
        )
        valid_mask, mask_message = chmp_mask_valid(mask_obs_frac, mask_mod_frac)
    if not valid_mask:
        return Q0MetricEvaluation(
            metrics=_invalid_metrics(),
            is_valid=False,
            message=mask_message,
            shift_x_arcsec=shift_x,
            shift_y_arcsec=shift_y,
            find_shift_valid=True,
            total_observed_flux=obs_flux,
            total_modeled_flux=mod_flux,
            mask_obs_fraction=mask_obs_frac,
            mask_mod_fraction=mask_mod_frac,
            mask_stage=effective_mask_stage,
        )

    try:
        metrics = compute_metrics(observed_arr, modeled_arr, trial_sigma, mask)
    except Exception as exc:
        return Q0MetricEvaluation(
            metrics=_invalid_metrics(),
            is_valid=False,
            message=str(exc),
            shift_x_arcsec=shift_x,
            shift_y_arcsec=shift_y,
            find_shift_valid=True,
            total_observed_flux=obs_flux,
            total_modeled_flux=mod_flux,
            mask_obs_fraction=mask_obs_frac,
            mask_mod_fraction=mask_mod_frac,
            mask_stage=effective_mask_stage,
        )

    return Q0MetricEvaluation(
        metrics=metrics,
        total_observed_flux=obs_flux,
        total_modeled_flux=mod_flux,
        is_valid=True,
        shift_x_arcsec=shift_x,
        shift_y_arcsec=shift_y,
        find_shift_valid=True,
        mask_obs_fraction=mask_obs_frac,
        mask_mod_fraction=mask_mod_frac,
        find_shift_version=FIND_SHIFT_VERSION,
        chmp_eval_policy_version=CHMP_EVAL_POLICY_VERSION,
        mask_stage=effective_mask_stage,
    )
