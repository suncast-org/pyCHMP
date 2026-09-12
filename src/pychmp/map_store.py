"""Map-store provenance helpers for reusable synthetic render products."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

MAP_IDENTITY_SCHEMA = "pychmp.map_identity.v2"
RENDER_PRODUCT_SCHEMA = "pychmp.render_product.v1"
MAP_LAYER_SCHEMA = "pychmp.map_layer.v1"

REUSABLE_COMPONENTS = frozenset({"stokes_i", "stokes_v", "corona", "tr"})


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_channel_or_frequency(value: Any) -> str:
    return str(value or "").strip().lower()


def is_reusable_component(component: Any) -> bool:
    return str(component or "").strip().lower() in REUSABLE_COMPONENTS


def build_render_product_identity(layer_identity: dict[str, Any]) -> dict[str, Any]:
    """Return product-level provenance shared by spectral/component layers."""

    identity = dict(layer_identity)
    product = {
        "schema": RENDER_PRODUCT_SCHEMA,
        "forward_model_sha256": str(identity.get("forward_model_sha256", "")),
        "forward_model_identity_version": str(identity.get("forward_model_identity_version", "")),
        "ebtel_sha256": str(identity.get("ebtel_sha256", "")),
        "artifact_geometry_sha256": str(identity.get("artifact_geometry_sha256", "")),
        "a": float(identity.get("a", np.nan)),
        "b": float(identity.get("b", np.nan)),
        "q0": float(identity.get("q0", np.nan)),
        "domain": str(identity.get("domain", "")).strip().lower(),
    }
    for key in ("euv_response_sha256", "euv_response_identity_version"):
        value = identity.get(key)
        if value not in (None, ""):
            product[key] = str(value)
    product["render_product_id"] = canonical_json_sha256(product)
    return product


def build_map_layer_provenance(layer_identity: dict[str, Any]) -> dict[str, Any]:
    """Return self-contained provenance for one reusable map layer."""

    identity = dict(layer_identity)
    product = build_render_product_identity(identity)
    layer = {
        "schema": MAP_LAYER_SCHEMA,
        "render_product_id": product["render_product_id"],
        "domain": str(identity.get("domain", "")).strip().lower(),
        "channel_or_frequency": normalize_channel_or_frequency(identity.get("channel_or_frequency")),
        "component": str(identity.get("component", "")).strip().lower(),
        "a": float(identity.get("a", np.nan)),
        "b": float(identity.get("b", np.nan)),
        "q0": float(identity.get("q0", np.nan)),
    }
    if identity.get("array_name") not in (None, ""):
        layer["array_name"] = str(identity["array_name"])
    layer["layer_id"] = canonical_json_sha256(layer)
    return layer


def build_map_identity(
    *,
    a: float,
    b: float,
    q0: float,
    domain: str,
    channel_or_frequency: str,
    component: str,
    forward_model_sha256: str,
    ebtel_sha256: str,
    artifact_geometry_sha256: str,
    forward_model_identity_version: str,
    euv_response_sha256: str | None = None,
    euv_response_identity_version: str | None = None,
    array_name: str | None = None,
) -> dict[str, Any]:
    component_norm = str(component or "").strip().lower()
    if not is_reusable_component(component_norm):
        raise ValueError(f"component is not reusable map-store evidence: {component!r}")
    identity = {
        "schema": MAP_IDENTITY_SCHEMA,
        "forward_model_sha256": str(forward_model_sha256),
        "forward_model_identity_version": str(forward_model_identity_version),
        "ebtel_sha256": str(ebtel_sha256),
        "artifact_geometry_sha256": str(artifact_geometry_sha256),
        "a": float(a),
        "b": float(b),
        "q0": float(q0),
        "domain": str(domain).strip().lower(),
        "channel_or_frequency": normalize_channel_or_frequency(channel_or_frequency),
        "component": component_norm,
    }
    if array_name is not None:
        identity["array_name"] = str(array_name)
    if euv_response_sha256 is not None:
        identity["euv_response_sha256"] = str(euv_response_sha256)
    if euv_response_identity_version is not None:
        identity["euv_response_identity_version"] = str(euv_response_identity_version)
    product = build_render_product_identity(identity)
    layer = build_map_layer_provenance(identity)
    identity["render_product_id"] = product["render_product_id"]
    identity["map_layer_id"] = layer["layer_id"]
    return identity
