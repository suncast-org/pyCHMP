from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from astropy.io import fits


METRICS = ("chi2", "rho2", "eta2")
RECTANGULAR_ARTIFACT_KIND = "pychmp_ab_scan"
SPARSE_ARTIFACT_KIND = "pychmp_ab_scan_sparse_points"
SLICE_CONTAINER_GROUP = "slices"


def decode_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        item = value.item()
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace")
        return str(item)
    return str(value)


def _sanitize_slice_token(value: str) -> str:
    text = str(value).strip().lower().replace(" ", "_")
    sanitized = []
    for char in text:
        if char.isalnum() or char in {"_", "-"}:
            sanitized.append(char)
        elif char == ".":
            sanitized.append("p")
    token = "".join(sanitized).strip("_")
    return token or "slice"


def slice_descriptor_from_diagnostics(diagnostics: dict[str, Any], *, fallback_key: str = "default") -> dict[str, Any]:
    domain = str(diagnostics.get("spectral_domain", "mw")).strip().lower()
    frequency_ghz = diagnostics.get("frequency_ghz", diagnostics.get("active_frequency_ghz"))
    channel_label = diagnostics.get("channel_label", diagnostics.get("euv_channel", diagnostics.get("channel_name")))
    label = str(diagnostics.get("spectral_label", "")).strip()
    sort_value: float | None = None

    if domain == "mw" and frequency_ghz is not None:
        frequency_ghz = float(frequency_ghz)
        label = label or f"{frequency_ghz:.3f} GHz"
        key = f"mw_{frequency_ghz:.6f}ghz".replace(".", "p")
        sort_value = frequency_ghz
    elif domain == "euv" and channel_label:
        label = label or str(channel_label)
        key = f"euv_{_sanitize_slice_token(str(channel_label))}"
    else:
        if channel_label and not label:
            label = str(channel_label)
        key = _sanitize_slice_token(str(diagnostics.get("slice_key", fallback_key)))
        if not label:
            label = str(diagnostics.get("slice_label", fallback_key))

    display_label = f"{domain.upper()}: {label}" if domain in {"mw", "euv"} else label
    return {
        "key": key,
        "domain": domain,
        "label": label,
        "display_label": display_label,
        "frequency_ghz": float(frequency_ghz) if frequency_ghz is not None else None,
        "channel_label": None if channel_label is None else str(channel_label),
        "sort_value": sort_value,
    }


def _artifact_kind_from_group(f: h5py.Group | h5py.File) -> str:
    if "point_records" in f:
        return SPARSE_ARTIFACT_KIND
    common = f.get("common")
    if common is not None and "diagnostics_json" in common:
        try:
            diagnostics = json.loads(decode_scalar(common["diagnostics_json"][()]))
            return str(diagnostics.get("artifact_kind", RECTANGULAR_ARTIFACT_KIND))
        except Exception:
            pass
    return RECTANGULAR_ARTIFACT_KIND


def _sorted_slice_descriptors(descriptors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        descriptors,
        key=lambda item: (
            item.get("domain", ""),
            item.get("sort_value") if item.get("sort_value") is not None else float("inf"),
            str(item.get("label", "")),
            str(item.get("key", "")),
        ),
    )


def _resolve_slice_group(
    f: h5py.File,
    *,
    slice_key: str | None = None,
    allow_missing: bool = False,
) -> tuple[h5py.Group | h5py.File | None, list[dict[str, Any]], str | None]:
    if SLICE_CONTAINER_GROUP not in f:
        diagnostics = {}
        common = f.get("common")
        if common is not None and "diagnostics_json" in common:
            diagnostics = json.loads(decode_scalar(common["diagnostics_json"][()]))
        descriptor = slice_descriptor_from_diagnostics(diagnostics, fallback_key="default")
        return f, [descriptor], descriptor["key"]

    slices_group = f[SLICE_CONTAINER_GROUP]
    descriptors: list[dict[str, Any]] = []
    for name in sorted(slices_group.keys()):
        grp = slices_group[name]
        diagnostics = {}
        common = grp.get("common")
        if common is not None and "diagnostics_json" in common:
            diagnostics = json.loads(decode_scalar(common["diagnostics_json"][()]))
        descriptor = slice_descriptor_from_diagnostics(diagnostics, fallback_key=name)
        descriptor["key"] = str(name)
        descriptors.append(descriptor)

    descriptors = _sorted_slice_descriptors(descriptors)
    key_lookup = {str(item["key"]): item for item in descriptors}
    selected_key = slice_key if slice_key in key_lookup else None
    if selected_key is None:
        if allow_missing and slice_key is not None:
            return None, descriptors, None
        selected_key = str(descriptors[0]["key"]) if descriptors else None
    if selected_key is None:
        return None, descriptors, None
    return slices_group[selected_key], descriptors, selected_key


def list_scan_slices(h5_path: Path) -> list[dict[str, Any]]:
    with h5py.File(h5_path, "r") as f:
        _group, descriptors, _selected_key = _resolve_slice_group(f)
    return descriptors


def detect_scan_artifact_format(h5_path: Path, *, slice_key: str | None = None) -> str | None:
    with h5py.File(h5_path, "r") as f:
        group, _descriptors, _selected_key = _resolve_slice_group(f, slice_key=slice_key, allow_missing=slice_key is not None)
        if group is None:
            return None
        kind = _artifact_kind_from_group(group)
    return "sparse" if kind == SPARSE_ARTIFACT_KIND else "rectangular"


def _axis_edges(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("Axis values must be a non-empty 1D array.")
    if arr.size == 1:
        delta = 0.5
        return np.asarray([arr[0] - delta, arr[0] + delta], dtype=float)
    mids = 0.5 * (arr[:-1] + arr[1:])
    first_delta = mids[0] - arr[0]
    last_delta = arr[-1] - mids[-1]
    return np.concatenate(
        [
            np.asarray([arr[0] - first_delta], dtype=float),
            mids,
            np.asarray([arr[-1] + last_delta], dtype=float),
        ]
    )


def _axis_spans(values: np.ndarray) -> dict[float, tuple[float, float]]:
    arr = np.asarray(values, dtype=float)
    edges = _axis_edges(arr)
    return {float(value): (float(edges[i]), float(edges[i + 1])) for i, value in enumerate(arr)}


def _blank_map(template: np.ndarray) -> np.ndarray:
    return np.full_like(np.asarray(template, dtype=float), np.nan, dtype=float)


def _pending_point_payload(
    *,
    a_value: float,
    b_value: float,
    a_index: int,
    b_index: int,
    observed_template: np.ndarray,
    target_metric: str,
    status: str,
    message: str,
) -> dict[str, Any]:
    blank = _blank_map(observed_template)
    diagnostics = {
        "a": float(a_value),
        "b": float(b_value),
        "target_metric": str(target_metric),
        "optimizer_message": str(message),
        "fit_success": False,
        "nfev": -1,
        "nit": -1,
        "used_adaptive_bracketing": False,
        "bracket_found": False,
        "bracket": None,
        "point_status": str(status),
    }
    return {
        "a": float(a_value),
        "b": float(b_value),
        "a_index": int(a_index),
        "b_index": int(b_index),
        "q0": np.nan,
        "success": False,
        "status": str(status),
        "modeled_best": blank,
        "raw_modeled_best": blank.copy(),
        "residual": blank.copy(),
        "fit_q0_trials": tuple(),
        "fit_metric_trials": tuple(),
        "fit_chi2_trials": tuple(),
        "fit_rho2_trials": tuple(),
        "fit_eta2_trials": tuple(),
        "nfev": -1,
        "nit": -1,
        "message": str(message),
        "used_adaptive_bracketing": False,
        "bracket_found": False,
        "bracket": None,
        "target_metric": str(target_metric),
        "diagnostics": diagnostics,
    }


def _normalize_point_payload(payload: dict[str, Any], *, record_order: int) -> dict[str, Any]:
    diagnostics = dict(payload.get("diagnostics", {}))
    target_metric = str(payload.get("target_metric", diagnostics.get("target_metric", "chi2")))
    return {
        "record_order": int(record_order),
        "a": float(payload["a"]),
        "b": float(payload["b"]),
        "q0": float(payload.get("q0", np.nan)),
        "success": bool(payload.get("success", False)),
        "status": str(payload.get("status", "computed")),
        "modeled_best": np.asarray(payload["modeled_best"], dtype=float),
        "raw_modeled_best": np.asarray(payload["raw_modeled_best"], dtype=float),
        "residual": np.asarray(payload["residual"], dtype=float),
        "fit_q0_trials": tuple(float(v) for v in payload.get("fit_q0_trials", ())),
        "fit_metric_trials": tuple(float(v) for v in payload.get("fit_metric_trials", ())),
        "fit_chi2_trials": tuple(float(v) for v in payload.get("fit_chi2_trials", ())),
        "fit_rho2_trials": tuple(float(v) for v in payload.get("fit_rho2_trials", ())),
        "fit_eta2_trials": tuple(float(v) for v in payload.get("fit_eta2_trials", ())),
        "nfev": int(payload.get("nfev", -1)),
        "nit": int(payload.get("nit", -1)),
        "message": str(payload.get("message", "")),
        "used_adaptive_bracketing": bool(payload.get("used_adaptive_bracketing", False)),
        "bracket_found": bool(payload.get("bracket_found", False)),
        "bracket": payload.get("bracket", None),
        "target_metric": target_metric,
        "diagnostics": diagnostics,
    }


def _read_point_group_rectangular(grp: h5py.Group) -> dict[str, Any]:
    target_metric = decode_scalar(
        grp["fit_metric_trials"].attrs["target_metric"]
        if "fit_metric_trials" in grp and "target_metric" in grp["fit_metric_trials"].attrs
        else grp.attrs.get("target_metric", b"chi2")
    )
    fit_metric_trials = np.asarray(grp["fit_metric_trials"], dtype=float) if "fit_metric_trials" in grp else np.asarray([], dtype=float)
    fit_chi2_trials = (
        np.asarray(grp["fit_chi2_trials"], dtype=float)
        if "fit_chi2_trials" in grp
        else (fit_metric_trials if target_metric == "chi2" else np.asarray([], dtype=float))
    )
    fit_rho2_trials = (
        np.asarray(grp["fit_rho2_trials"], dtype=float)
        if "fit_rho2_trials" in grp
        else (fit_metric_trials if target_metric == "rho2" else np.asarray([], dtype=float))
    )
    fit_eta2_trials = (
        np.asarray(grp["fit_eta2_trials"], dtype=float)
        if "fit_eta2_trials" in grp
        else (fit_metric_trials if target_metric == "eta2" else np.asarray([], dtype=float))
    )
    bracket = None
    if "bracket" in grp:
        try:
            bracket_arr = np.asarray(grp["bracket"], dtype=float)
            if bracket_arr.size == 3:
                bracket = tuple(float(v) for v in bracket_arr)
        except Exception:
            bracket = None
    return {
        "record_order": int(grp.attrs.get("record_order", 0)),
        "a": float(grp.attrs["a"]),
        "b": float(grp.attrs["b"]),
        "q0": float(grp.attrs["q0"]),
        "success": bool(grp.attrs["success"]),
        "status": decode_scalar(grp.attrs.get("status", b"computed")),
        "modeled_best": np.asarray(grp["modeled_best"], dtype=float),
        "raw_modeled_best": np.asarray(grp["raw_modeled_best"], dtype=float),
        "residual": np.asarray(grp["residual"], dtype=float),
        "fit_q0_trials": tuple(float(v) for v in np.asarray(grp["fit_q0_trials"], dtype=float)),
        "fit_metric_trials": tuple(float(v) for v in fit_metric_trials),
        "fit_chi2_trials": tuple(float(v) for v in fit_chi2_trials),
        "fit_rho2_trials": tuple(float(v) for v in fit_rho2_trials),
        "fit_eta2_trials": tuple(float(v) for v in fit_eta2_trials),
        "nfev": int(grp.attrs.get("nfev", -1)),
        "nit": int(grp.attrs.get("nit", -1)),
        "message": decode_scalar(grp.attrs.get("message", b"")),
        "used_adaptive_bracketing": bool(grp.attrs.get("used_adaptive_bracketing", False)),
        "bracket_found": bool(grp.attrs.get("bracket_found", False)),
        "bracket": bracket,
        "target_metric": target_metric,
        "diagnostics": json.loads(decode_scalar(grp["diagnostics_json"][()])),
    }


def _read_point_group_sparse(grp: h5py.Group) -> dict[str, Any]:
    return _read_point_group_rectangular(grp)


def _load_sparse_point_records(records_group: h5py.Group) -> list[dict[str, Any]]:
    latest_by_coord: dict[tuple[float, float], dict[str, Any]] = {}
    for name in sorted(records_group.keys()):
        record = _read_point_group_sparse(records_group[name])
        coord = (float(record["a"]), float(record["b"]))
        existing = latest_by_coord.get(coord)
        if existing is None or int(record["record_order"]) >= int(existing["record_order"]):
            latest_by_coord[coord] = record
    return sorted(latest_by_coord.values(), key=lambda item: (float(item["a"]), float(item["b"])))


def _load_rectangular_point_records(points_group: h5py.Group) -> list[dict[str, Any]]:
    records = [_read_point_group_rectangular(points_group[name]) for name in sorted(points_group.keys())]
    return sorted(records, key=lambda item: (float(item["a"]), float(item["b"])))


def _payload_from_point_records(
    *,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
    point_records: list[dict[str, Any]],
    target_metric: str,
) -> dict[str, Any]:
    unique_a = np.asarray(sorted({float(record["a"]) for record in point_records}), dtype=float)
    unique_b = np.asarray(sorted({float(record["b"]) for record in point_records}), dtype=float)
    if unique_a.size == 0:
        unique_a = np.asarray([], dtype=float)
    if unique_b.size == 0:
        unique_b = np.asarray([], dtype=float)

    shape = (unique_a.size, unique_b.size)
    best_q0 = np.full(shape, np.nan, dtype=float)
    objective_values = np.full(shape, np.nan, dtype=float)
    chi2 = np.full(shape, np.nan, dtype=float)
    rho2 = np.full(shape, np.nan, dtype=float)
    eta2 = np.full(shape, np.nan, dtype=float)
    success = np.zeros(shape, dtype=bool)
    points: dict[tuple[int, int], dict[str, Any]] = {}

    target_metric_name = str(target_metric or diagnostics.get("target_metric", "chi2"))
    for i, a_value in enumerate(unique_a):
        for j, b_value in enumerate(unique_b):
            points[(int(i), int(j))] = _pending_point_payload(
                a_value=float(a_value),
                b_value=float(b_value),
                a_index=int(i),
                b_index=int(j),
                observed_template=observed,
                target_metric=target_metric_name,
                status="missing",
                message="point not stored in sparse artifact",
            )

    a_lookup = {float(v): int(i) for i, v in enumerate(unique_a)}
    b_lookup = {float(v): int(i) for i, v in enumerate(unique_b)}
    normalized_records: list[dict[str, Any]] = []
    for record in point_records:
        a_value = float(record["a"])
        b_value = float(record["b"])
        a_index = a_lookup[a_value]
        b_index = b_lookup[b_value]
        diagnostics_json = dict(record.get("diagnostics", {}))
        metrics = {
            "chi2": float(diagnostics_json.get("chi2", np.nan)),
            "rho2": float(diagnostics_json.get("rho2", np.nan)),
            "eta2": float(diagnostics_json.get("eta2", np.nan)),
        }
        points[(a_index, b_index)] = {
            **record,
            "a_index": int(a_index),
            "b_index": int(b_index),
        }
        best_q0[a_index, b_index] = float(record.get("q0", np.nan))
        objective_values[a_index, b_index] = float(diagnostics_json.get("target_metric_value", np.nan))
        chi2[a_index, b_index] = metrics["chi2"]
        rho2[a_index, b_index] = metrics["rho2"]
        eta2[a_index, b_index] = metrics["eta2"]
        success[a_index, b_index] = bool(record.get("success", False))
        normalized_records.append(
            {
                **record,
                "a_index": int(a_index),
                "b_index": int(b_index),
                "metrics": metrics,
            }
        )

    return {
        "observed": np.asarray(observed, dtype=float),
        "sigma_map": np.asarray(sigma_map, dtype=float),
        "wcs_header": wcs_header,
        "diagnostics": diagnostics,
        "a_values": unique_a,
        "b_values": unique_b,
        "best_q0": best_q0,
        "objective_values": objective_values,
        "chi2": chi2,
        "rho2": rho2,
        "eta2": eta2,
        "success": success,
        "target_metric": target_metric_name,
        "points": points,
        "point_records": normalized_records,
        "artifact_format": "sparse" if diagnostics.get("artifact_kind") == SPARSE_ARTIFACT_KIND else "rectangular",
    }


def load_scan_file(h5_path: Path, *, slice_key: str | None = None) -> dict[str, Any]:
    with h5py.File(h5_path, "r") as f:
        group, descriptors, selected_key = _resolve_slice_group(
            f,
            slice_key=slice_key,
            allow_missing=slice_key is not None,
        )
        if group is None:
            raise KeyError(f"slice not found: {slice_key}")
        common = group["common"]
        wcs_header = fits.Header.fromstring(decode_scalar(common["wcs_header"][()]), sep="\n")
        diagnostics = json.loads(decode_scalar(common["diagnostics_json"][()]))
        kind = _artifact_kind_from_group(group)
        if kind == SPARSE_ARTIFACT_KIND:
            point_records = _load_sparse_point_records(group["point_records"])
            target_metric = str(diagnostics.get("target_metric", "chi2"))
        else:
            point_records = _load_rectangular_point_records(group["points"])
            summary = group["summary"]
            target_metric = decode_scalar(summary.attrs.get("target_metric", diagnostics.get("target_metric", b"chi2")))
        payload = _payload_from_point_records(
            observed=np.asarray(common["observed"], dtype=float),
            sigma_map=np.asarray(common["sigma_map"], dtype=float),
            wcs_header=wcs_header,
            diagnostics=diagnostics,
            point_records=point_records,
            target_metric=target_metric,
        )
        selected_descriptor = next((item for item in descriptors if str(item["key"]) == str(selected_key)), None)
        payload["available_slices"] = descriptors
        payload["selected_slice_key"] = selected_key
        payload["selected_slice"] = selected_descriptor
        return payload


def _write_common_group(
    common: h5py.Group,
    *,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
) -> None:
    common.create_dataset("observed", data=np.asarray(observed, dtype=np.float32), compression="gzip", compression_opts=4)
    common.create_dataset("sigma_map", data=np.asarray(sigma_map, dtype=np.float32), compression="gzip", compression_opts=4)
    common.create_dataset("wcs_header", data=np.bytes_(wcs_header.tostring(sep="\n", endcard=True)))
    common.create_dataset("diagnostics_json", data=np.bytes_(json.dumps(diagnostics, sort_keys=True)))


def save_rectangular_scan_file(
    out_h5: Path,
    *,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
    a_values: np.ndarray,
    b_values: np.ndarray,
    best_q0: np.ndarray,
    objective_values: np.ndarray,
    chi2: np.ndarray,
    rho2: np.ndarray,
    eta2: np.ndarray,
    success: np.ndarray,
    point_payloads: dict[tuple[int, int], dict[str, Any]],
    slice_key: str | None = None,
) -> None:
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    descriptor = slice_descriptor_from_diagnostics(diagnostics, fallback_key=slice_key or "default")
    resolved_slice_key = str(slice_key or descriptor["key"])
    tmp_h5 = out_h5.with_suffix(out_h5.suffix + ".tmp")
    with h5py.File(tmp_h5, "w") as dst:
        slices_dst = dst.create_group(SLICE_CONTAINER_GROUP)
        if out_h5.exists():
            with h5py.File(out_h5, "r") as src:
                if SLICE_CONTAINER_GROUP in src:
                    for name in src[SLICE_CONTAINER_GROUP].keys():
                        if str(name) == resolved_slice_key:
                            continue
                        src.copy(src[SLICE_CONTAINER_GROUP][name], slices_dst, name=name)
                elif "common" in src:
                    existing_diag = json.loads(decode_scalar(src["common"]["diagnostics_json"][()]))
                    existing_descriptor = slice_descriptor_from_diagnostics(existing_diag, fallback_key="legacy")
                    existing_key = str(existing_descriptor["key"])
                    if existing_key != resolved_slice_key:
                        legacy_dst = slices_dst.create_group(existing_key)
                        for name in src.keys():
                            src.copy(name, legacy_dst, name=name)

        slice_group = slices_dst.create_group(resolved_slice_key)
        slice_group.attrs["domain"] = np.bytes_(str(descriptor["domain"]))
        slice_group.attrs["label"] = np.bytes_(str(descriptor["label"]))
        if descriptor.get("frequency_ghz") is not None:
            slice_group.attrs["frequency_ghz"] = float(descriptor["frequency_ghz"])
        if descriptor.get("channel_label") is not None:
            slice_group.attrs["channel_label"] = np.bytes_(str(descriptor["channel_label"]))

        common = slice_group.create_group("common")
        _write_common_group(
            common,
            observed=observed,
            sigma_map=sigma_map,
            wcs_header=wcs_header,
            diagnostics=diagnostics,
        )

        grid = slice_group.create_group("grid")
        grid.create_dataset("a_values", data=np.asarray(a_values, dtype=np.float64))
        grid.create_dataset("b_values", data=np.asarray(b_values, dtype=np.float64))

        summary = slice_group.create_group("summary")
        summary.create_dataset("best_q0", data=np.asarray(best_q0, dtype=np.float64))
        summary.create_dataset("objective_values", data=np.asarray(objective_values, dtype=np.float64))
        summary.create_dataset("chi2", data=np.asarray(chi2, dtype=np.float64))
        summary.create_dataset("rho2", data=np.asarray(rho2, dtype=np.float64))
        summary.create_dataset("eta2", data=np.asarray(eta2, dtype=np.float64))
        summary.create_dataset("success", data=np.asarray(success, dtype=np.uint8))
        summary.attrs["target_metric"] = np.bytes_(str(diagnostics.get("target_metric", "chi2")))

        points = slice_group.create_group("points")
        for (_ai, _bi), payload in sorted(point_payloads.items()):
            name = f"a{payload['a_index']:03d}_b{payload['b_index']:03d}"
            grp = points.create_group(name)
            grp.attrs["a"] = float(payload["a"])
            grp.attrs["b"] = float(payload["b"])
            grp.attrs["a_index"] = int(payload["a_index"])
            grp.attrs["b_index"] = int(payload["b_index"])
            grp.attrs["q0"] = float(payload["q0"])
            grp.attrs["success"] = int(bool(payload["success"]))
            grp.attrs["status"] = np.bytes_(str(payload.get("status", "computed")))
            grp.attrs["target_metric"] = np.bytes_(str(payload["target_metric"]))
            grp.create_dataset("modeled_best", data=np.asarray(payload["modeled_best"], dtype=np.float32), compression="gzip", compression_opts=4)
            grp.create_dataset("raw_modeled_best", data=np.asarray(payload["raw_modeled_best"], dtype=np.float32), compression="gzip", compression_opts=4)
            grp.create_dataset("residual", data=np.asarray(payload["residual"], dtype=np.float32), compression="gzip", compression_opts=4)
            grp.create_dataset("fit_q0_trials", data=np.asarray(payload["fit_q0_trials"], dtype=np.float64))
            fit_metric_ds = grp.create_dataset("fit_metric_trials", data=np.asarray(payload["fit_metric_trials"], dtype=np.float64))
            fit_metric_ds.attrs["target_metric"] = np.bytes_(str(payload["target_metric"]))
            grp.create_dataset("fit_chi2_trials", data=np.asarray(payload.get("fit_chi2_trials", ()), dtype=np.float64))
            grp.create_dataset("fit_rho2_trials", data=np.asarray(payload.get("fit_rho2_trials", ()), dtype=np.float64))
            grp.create_dataset("fit_eta2_trials", data=np.asarray(payload.get("fit_eta2_trials", ()), dtype=np.float64))
            grp.create_dataset("diagnostics_json", data=np.bytes_(json.dumps(payload["diagnostics"], sort_keys=True)))
    os.replace(tmp_h5, out_h5)


def _write_point_group(grp: h5py.Group, payload: dict[str, Any], *, record_order: int) -> None:
    normalized = _normalize_point_payload(payload, record_order=record_order)
    grp.attrs["record_order"] = int(record_order)
    grp.attrs["a"] = float(normalized["a"])
    grp.attrs["b"] = float(normalized["b"])
    grp.attrs["q0"] = float(normalized["q0"])
    grp.attrs["success"] = int(bool(normalized["success"]))
    grp.attrs["status"] = np.bytes_(str(normalized["status"]))
    grp.attrs["target_metric"] = np.bytes_(str(normalized["target_metric"]))
    grp.attrs["nfev"] = int(normalized["nfev"])
    grp.attrs["nit"] = int(normalized["nit"])
    grp.attrs["message"] = np.bytes_(str(normalized["message"]))
    grp.attrs["used_adaptive_bracketing"] = int(bool(normalized["used_adaptive_bracketing"]))
    grp.attrs["bracket_found"] = int(bool(normalized["bracket_found"]))
    if normalized["bracket"] is not None:
        grp.create_dataset("bracket", data=np.asarray(normalized["bracket"], dtype=np.float64))
    grp.create_dataset("modeled_best", data=np.asarray(normalized["modeled_best"], dtype=np.float32), compression="gzip", compression_opts=4)
    grp.create_dataset("raw_modeled_best", data=np.asarray(normalized["raw_modeled_best"], dtype=np.float32), compression="gzip", compression_opts=4)
    grp.create_dataset("residual", data=np.asarray(normalized["residual"], dtype=np.float32), compression="gzip", compression_opts=4)
    grp.create_dataset("fit_q0_trials", data=np.asarray(normalized["fit_q0_trials"], dtype=np.float64))
    fit_metric_ds = grp.create_dataset("fit_metric_trials", data=np.asarray(normalized["fit_metric_trials"], dtype=np.float64))
    fit_metric_ds.attrs["target_metric"] = np.bytes_(str(normalized["target_metric"]))
    grp.create_dataset("fit_chi2_trials", data=np.asarray(normalized["fit_chi2_trials"], dtype=np.float64))
    grp.create_dataset("fit_rho2_trials", data=np.asarray(normalized["fit_rho2_trials"], dtype=np.float64))
    grp.create_dataset("fit_eta2_trials", data=np.asarray(normalized["fit_eta2_trials"], dtype=np.float64))
    grp.create_dataset("diagnostics_json", data=np.bytes_(json.dumps(normalized["diagnostics"], sort_keys=True)))


def write_sparse_scan_file(
    out_h5: Path,
    *,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
    point_records: list[dict[str, Any]],
) -> None:
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    tmp_h5 = out_h5.with_suffix(out_h5.suffix + ".tmp")
    diagnostics_out = dict(diagnostics)
    diagnostics_out["artifact_kind"] = SPARSE_ARTIFACT_KIND
    with h5py.File(tmp_h5, "w") as f:
        common = f.create_group("common")
        common.create_dataset("observed", data=np.asarray(observed, dtype=np.float32), compression="gzip", compression_opts=4)
        common.create_dataset("sigma_map", data=np.asarray(sigma_map, dtype=np.float32), compression="gzip", compression_opts=4)
        common.create_dataset("wcs_header", data=np.bytes_(wcs_header.tostring(sep="\n", endcard=True)))
        common.create_dataset("diagnostics_json", data=np.bytes_(json.dumps(diagnostics_out, sort_keys=True)))
        records_group = f.create_group("point_records")
        for record_order, payload in enumerate(point_records):
            grp = records_group.create_group(f"r{record_order:06d}")
            _write_point_group(grp, payload, record_order=record_order)
    os.replace(tmp_h5, out_h5)


def append_sparse_point_record(
    out_h5: Path,
    *,
    observed: np.ndarray,
    sigma_map: np.ndarray,
    wcs_header: fits.Header,
    diagnostics: dict[str, Any],
    point_payload: dict[str, Any],
) -> None:
    diagnostics_out = dict(diagnostics)
    diagnostics_out["artifact_kind"] = SPARSE_ARTIFACT_KIND
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if out_h5.exists() else "w"
    with h5py.File(out_h5, mode) as f:
        if "common" not in f:
            common = f.create_group("common")
            common.create_dataset("observed", data=np.asarray(observed, dtype=np.float32), compression="gzip", compression_opts=4)
            common.create_dataset("sigma_map", data=np.asarray(sigma_map, dtype=np.float32), compression="gzip", compression_opts=4)
            common.create_dataset("wcs_header", data=np.bytes_(wcs_header.tostring(sep="\n", endcard=True)))
            common.create_dataset("diagnostics_json", data=np.bytes_(json.dumps(diagnostics_out, sort_keys=True)))
        records_group = f.require_group("point_records")
        existing_orders = [int(records_group[name].attrs.get("record_order", -1)) for name in records_group.keys()]
        next_order = max(existing_orders, default=-1) + 1
        grp = records_group.create_group(f"r{next_order:06d}")
        _write_point_group(grp, point_payload, record_order=next_order)


def convert_rectangular_artifact_to_sparse(src_h5: Path, dst_h5: Path, *, overwrite: bool = False) -> None:
    if dst_h5.exists() and not overwrite:
        raise FileExistsError(f"destination already exists: {dst_h5}")
    payload = load_scan_file(src_h5)
    diagnostics = dict(payload["diagnostics"])
    diagnostics["artifact_kind"] = SPARSE_ARTIFACT_KIND
    point_records = [
        {key: value for key, value in record.items() if key not in {"a_index", "b_index", "metrics"}}
        for record in payload.get("point_records", [])
    ]
    write_sparse_scan_file(
        dst_h5,
        observed=np.asarray(payload["observed"], dtype=float),
        sigma_map=np.asarray(payload["sigma_map"], dtype=float),
        wcs_header=payload["wcs_header"],
        diagnostics=diagnostics,
        point_records=point_records,
    )


def build_patch_grid_model(payload: dict[str, Any]) -> dict[str, Any]:
    source_records = payload.get("point_records")
    if source_records is None:
        source_records = list(payload["points"].values())
    records = [record for record in source_records if str(record.get("status", "computed")) != "missing"]
    if not records:
        return {
            "records": [],
            "a_min": 0.0,
            "a_max": 1.0,
            "b_min": 0.0,
            "b_max": 1.0,
        }

    a_coords = np.unique([float(record["a"]) for record in records])
    b_coords = np.unique([float(record["b"]) for record in records])
    a_spans = _axis_spans(a_coords)
    b_spans = _axis_spans(b_coords)
    display_records: list[dict[str, Any]] = []
    for record in records:
        a_value = float(record["a"])
        b_value = float(record["b"])
        a0, a1 = a_spans[a_value]
        b0, b1 = b_spans[b_value]
        diagnostics = dict(record.get("diagnostics", {}))
        metrics = record.get(
            "metrics",
            {
                "chi2": float(diagnostics.get("chi2", np.nan)),
                "rho2": float(diagnostics.get("rho2", np.nan)),
                "eta2": float(diagnostics.get("eta2", np.nan)),
            },
        )
        display_records.append(
            {
                "key": (int(record.get("a_index", 0)), int(record.get("b_index", 0))),
                "a_index": int(record.get("a_index", 0)),
                "b_index": int(record.get("b_index", 0)),
                "a": a_value,
                "b": b_value,
                "a0": a0,
                "a1": a1,
                "b0": b0,
                "b1": b1,
                "a_center": 0.5 * (a0 + a1),
                "b_center": 0.5 * (b0 + b1),
                "metrics": metrics,
                "status": str(record.get("status", "computed")),
                "q0": float(record.get("q0", np.nan)),
                "success": bool(record.get("success", False)),
            }
        )
    return {
        "records": display_records,
        "a_min": float(min(record["a0"] for record in display_records)),
        "a_max": float(max(record["a1"] for record in display_records)),
        "b_min": float(min(record["b0"] for record in display_records)),
        "b_max": float(max(record["b1"] for record in display_records)),
    }


def find_record_for_point(model: dict[str, Any], x: float, y: float) -> dict[str, Any] | None:
    for record in model.get("records", []):
        if float(record["b0"]) <= float(x) <= float(record["b1"]) and float(record["a0"]) <= float(y) <= float(record["a1"]):
            return record
    return None


def best_grid_index(payload: dict[str, Any], metric: str) -> tuple[int, int]:
    metric_name = str(metric).strip().lower()
    if metric_name not in METRICS:
        raise ValueError(f"Unsupported best-of-grid metric: {metric_name}")
    arr = np.asarray(payload[metric_name], dtype=float)
    good = np.isfinite(arr)
    if not np.any(good):
        raise ValueError(f"No finite values available for metric {metric_name}")
    idx = np.nanargmin(arr)
    a_index, b_index = np.unravel_index(idx, arr.shape)
    return int(a_index), int(b_index)


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(np.asarray(values, dtype=float) - float(target))))


def with_observer_metadata(header: fits.Header, source_header: fits.Header, diagnostics: dict[str, Any]) -> fits.Header:
    out = header.copy()
    for key in (
        "OBSERVER",
        "DATE-OBS",
        "DSUN_OBS",
        "HGLN_OBS",
        "HGLT_OBS",
        "CRLN_OBS",
        "CRLT_OBS",
        "HGLN-OBS",
        "HGLT-OBS",
        "CRLN-OBS",
        "CRLT-OBS",
    ):
        if key in source_header and key not in out:
            out[key] = source_header[key]

    if "OBSERVER" not in out and diagnostics.get("observer_name"):
        out["OBSERVER"] = str(diagnostics["observer_name"])
    if "DATE-OBS" not in out and diagnostics.get("observer_obs_time"):
        out["DATE-OBS"] = str(diagnostics["observer_obs_time"])

    scalar_fallbacks = {
        "DSUN_OBS": ("observer_dsun_cm", 0.01),
        "HGLN_OBS": ("observer_lonc_deg", 1.0),
        "HGLT_OBS": ("observer_b0sun_deg", 1.0),
        "CRLN_OBS": ("observer_lonc_deg", 1.0),
        "CRLT_OBS": ("observer_b0sun_deg", 1.0),
        "HGLN-OBS": ("observer_lonc_deg", 1.0),
        "HGLT-OBS": ("observer_b0sun_deg", 1.0),
        "CRLN-OBS": ("observer_lonc_deg", 1.0),
        "CRLT-OBS": ("observer_b0sun_deg", 1.0),
    }
    for key, (diag_key, scale) in scalar_fallbacks.items():
        if key in out:
            continue
        value = diagnostics.get(diag_key)
        if value is None:
            continue
        try:
            out[key] = float(value) * float(scale)
        except Exception:
            continue
    return out
