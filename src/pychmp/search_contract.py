"""Canonical search evaluation contract for artifact search identity."""

from __future__ import annotations

import hashlib
import json
from typing import Any

SEARCH_EVALUATION_CONTRACT_VERSION = "pychmp.search_evaluation.v1"
OBSERVATION_REF_GROUP = "observation_ref"


def _observation_ref_diagnostics_from_full(diagnostics: dict[str, Any]) -> dict[str, Any]:
    prefixes = (
        "observation_",
        "preprocessed_",
        "canvas_",
        "shift_",
        "find_shift_",
        "noise_",
        "slice_observation_",
    )
    exact_keys = {
        "shift_policy",
        "max_shift_arcsec",
        "xy_shift_arcsec",
        "chmp_eval_policy_version",
    }
    out: dict[str, Any] = {}
    for key, value in diagnostics.items():
        text = str(key)
        if text in exact_keys or any(text.startswith(prefix) for prefix in prefixes):
            out[text] = value
    return out


def _first_present(diagnostics: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in diagnostics and diagnostics[key] not in {None, ""}:
            return diagnostics[key]
    return None


def normalize_shift_policy(value: Any) -> str:
    """Normalize legacy/null shift policy values to the runtime contract."""
    text = str(value or "fixed").strip().lower()
    if text in {"", "none", "null", "fixed"}:
        return "fixed"
    if text == "auto":
        return "auto"
    return "fixed"


def search_shift_fields_from_request(request: dict[str, Any] | None) -> dict[str, Any]:
    """Extract search-scoped shift settings from a stored search request payload."""
    request_payload = dict(request or {})
    policy = normalize_shift_policy(request_payload.get("shift_policy"))
    fields: dict[str, Any] = {"shift_policy": policy}
    if policy == "auto":
        max_shift = request_payload.get("max_shift_arcsec")
        if max_shift not in {None, ""}:
            fields["max_shift_arcsec"] = float(max_shift)
        return fields
    xy = request_payload.get("xy_shift_arcsec")
    if isinstance(xy, (list, tuple)) and len(xy) >= 2:
        fields["xy_shift_arcsec"] = [float(xy[0]), float(xy[1])]
    else:
        fields["xy_shift_arcsec"] = [0.0, 0.0]
    return fields


def apply_search_shift_diagnostics(
    diagnostics: dict[str, Any],
    *,
    search_request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Override slice-common shift fields with the selected search identity."""
    merged = dict(diagnostics)
    merged.update(search_shift_fields_from_request(search_request))
    return merged


def build_search_evaluation_config(
    diagnostics: dict[str, Any],
    *,
    layout: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical search evaluation contract payload."""
    layout_payload = dict(layout or {})
    config: dict[str, Any] = {
        "schema": SEARCH_EVALUATION_CONTRACT_VERSION,
        "target_metric": str(diagnostics.get("target_metric", "chi2")),
        "layout": layout_payload,
        "model_sha256": diagnostics.get("model_sha256"),
        "forward_model_sha256": diagnostics.get("forward_model_sha256"),
        "forward_model_identity_version": diagnostics.get("forward_model_identity_version"),
        "ebtel_sha256": diagnostics.get("ebtel_sha256"),
        "euv_response_identity_version": diagnostics.get("euv_response_identity_version"),
        "euv_response_sha256": diagnostics.get("euv_response_sha256"),
        "shift_policy": diagnostics.get("shift_policy"),
        "max_shift_arcsec": diagnostics.get("max_shift_arcsec"),
        "xy_shift_arcsec": diagnostics.get("xy_shift_arcsec"),
        "metrics_mask": {
            "source": diagnostics.get("metrics_mask_source"),
            "threshold": diagnostics.get("metrics_mask_threshold", diagnostics.get("threshold")),
            "fits": diagnostics.get("metrics_mask_fits"),
            "mask_type": diagnostics.get("mask_type"),
        },
        "tr_mask": {
            "source": diagnostics.get("tr_mask_source"),
            "bmin_gauss": diagnostics.get("tr_mask_bmin_gauss"),
        },
        "use_smoothed_obs_max": diagnostics.get("use_smoothed_obs_max"),
        "use_emthreshold": diagnostics.get("use_emthreshold"),
        "emthreshold": diagnostics.get("emthreshold"),
        "q0_search_stages": diagnostics.get("q0_search_stages"),
        "optimizer": {
            "q0_min": diagnostics.get("q0_min"),
            "q0_max": diagnostics.get("q0_max"),
            "hard_q0_min": diagnostics.get("hard_q0_min"),
            "hard_q0_max": diagnostics.get("hard_q0_max"),
            "q0_start": diagnostics.get("q0_start"),
            "q0_step": diagnostics.get("q0_step"),
            "adaptive_bracketing": diagnostics.get(
                "adaptive_bracketing",
                diagnostics.get("used_adaptive_bracketing"),
            ),
            "max_bracket_steps": diagnostics.get("max_bracket_steps"),
            "threshold_metric": diagnostics.get("threshold_metric"),
            "no_area": diagnostics.get("no_area"),
        },
        "execution": {
            "policy": _first_present(diagnostics, ("execution_policy", "execution_policy_resolved")),
            "requested_policy": diagnostics.get("execution_policy_requested"),
            "max_workers": diagnostics.get("execution_max_workers"),
        },
        "no_area": diagnostics.get("no_area"),
        "psf_source": diagnostics.get("psf_source"),
        "resolved_psf": diagnostics.get("resolved_psf"),
        "observation": {
            "fits_sha256": diagnostics.get("fits_sha256") or diagnostics.get("observation_source_sha256"),
            "observation_source_sha256": diagnostics.get("observation_source_sha256"),
            "fits_file": diagnostics.get("fits_file"),
            "observer_obs_time": diagnostics.get("observer_obs_time"),
        },
    }
    if "requested_points" in diagnostics:
        config["requested_points"] = diagnostics["requested_points"]
    elif layout_payload.get("kind") == "rectangular_grid":
        a_values = [float(v) for v in layout_payload.get("a_values", [])]
        b_values = [float(v) for v in layout_payload.get("b_values", [])]
        config["requested_points"] = [{"a": a, "b": b} for a in a_values for b in b_values]
    return config


def search_evaluation_signature(config: dict[str, Any]) -> str:
    normalized = json.dumps(config, separators=(",", ":"), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compatibility_signature_from_diagnostics(
    diagnostics: dict[str, Any],
    *,
    layout: dict[str, Any] | None = None,
) -> str:
    return search_evaluation_signature(build_search_evaluation_config(diagnostics, layout=layout))


def search_id_from_evaluation_config(
    diagnostics: dict[str, Any],
    *,
    fallback: str = "search",
    layout: dict[str, Any] | None = None,
) -> str:
    config = build_search_evaluation_config(diagnostics, layout=layout)
    signature = search_evaluation_signature(config)
    search_instance_id = str(diagnostics.get("search_instance_id", "")).strip()
    if search_instance_id:
        suffix = hashlib.sha256(search_instance_id.encode("utf-8")).hexdigest()[:8]
        return f"{fallback}_{signature[:16]}_{suffix}"
    return f"{fallback}_{signature[:16]}"
