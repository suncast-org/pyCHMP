"""Shared EUV calibration: one response array per channel, never per search."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from .gxrender_adapter import _CachedEUVResponse, compute_euv_response_identity, _normalize_euv_channel_token

GROUP = "calibration/euv_responses"


def _setup(request: dict) -> dict:
    return {k: v for k, v in request.items() if k != "channels"}


def response_request_key(request: dict) -> str:
    return hashlib.sha256(json.dumps(_setup(request), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def load_response(path: Path, request: dict) -> _CachedEUVResponse | None:
    if not Path(path).exists():
        return None
    with h5py.File(path, "r") as f:
        key = f"{GROUP}/{response_request_key(request)}"
        if key not in f:
            return None
        g = f[key]
        if json.loads(g["setup_json"][()]) != _setup(request):
            raise ValueError("Persisted EUV response setup mismatch")
        channels = sorted(set(map(str, request["channels"])))
        if any(channel not in g["channels"] for channel in channels):
            return None
        logte = g["logte"][()]
        if _digest(logte) != str(g["logte"].attrs["sha256"]):
            raise ValueError("Persisted EUV temperature-grid checksum mismatch")
        dtype_spec = json.loads(g["dtype_json"][()])
        dtype = np.dtype([(name, fmt, (len(channels), len(logte))) if name == "all"
                          else (name, fmt, tuple(shape[0])) if shape else (name, fmt)
                          for name, fmt, *shape in dtype_spec])
        arr = np.zeros(1, dtype=dtype)
        arr["ds"] = g.attrs["ds"]
        arr["NT"] = len(logte)
        arr["Nchannels"] = len(channels)
        arr["logte"] = logte
        for i, channel in enumerate(channels):
            data = g["channels"][channel][()]
            if _digest(data) != str(g["channels"][channel].attrs["sha256"]):
                raise ValueError(f"Persisted EUV response checksum mismatch for {channel}")
            arr["all"][0, i] = data
        metadata = json.loads(g["metadata_json"][()])
        metadata["channels"] = channels
        meta = SimpleNamespace(**metadata)
        identity = compute_euv_response_identity(response=arr, response_dt=arr.dtype, response_meta=meta)
    arr.setflags(write=False)
    return _CachedEUVResponse(arr, arr.dtype, meta, identity)


def save_response(path: Path, request: dict, cached: _CachedEUVResponse, *, recovered: bool = False) -> None:
    """Add missing channels; never replace an existing channel calibration."""
    arr = np.asarray(cached.response)
    if arr.size != 1 or not {"ds", "NT", "Nchannels", "logte", "all"} <= set(arr.dtype.names or ()):
        raise ValueError("Expected a single GX structured EUV response")
    entry = arr.reshape(-1)[0]
    channels = [_normalize_euv_channel_token(str(c)) for c in cached.response_meta.channels]
    if len(set(channels)) != len(channels) or len(channels) != int(entry["Nchannels"]):
        raise ValueError("Invalid EUV response channel layout")
    logte = np.asarray(entry["logte"])
    meta = {key: getattr(cached.response_meta, key) for key in ("instrument", "source", "mode")}
    with h5py.File(path, "a") as f:
        g = f.require_group(f"{GROUP}/{response_request_key(request)}")
        if "setup_json" in g:
            if (json.loads(g["setup_json"][()]) != _setup(request)
                    or not np.array_equal(g["logte"][()], logte)
                    or float(g.attrs["ds"]) != float(entry["ds"])
                    or json.loads(g["metadata_json"][()]) != meta):
                raise ValueError("Refusing to replace a different persisted EUV calibration")
            for i, channel in enumerate(channels):
                if channel in g["channels"] and not np.array_equal(g["channels"][channel][()], entry["all"][i]):
                    raise ValueError(f"Refusing to replace persisted EUV response for channel {channel}")
        else:
            g.create_dataset("setup_json", data=json.dumps(_setup(request), sort_keys=True))
            g.create_dataset("metadata_json", data=json.dumps(meta))
            g.create_dataset("dtype_json", data=json.dumps(arr.dtype.descr))
            grid = g.create_dataset("logte", data=logte)
            grid.attrs["sha256"] = _digest(logte)
            g.attrs["ds"] = float(entry["ds"])
            g.attrs["schema"] = "pychmp.euv_response_channels.v1"
            g.attrs["recovered_for_legacy_artifact"] = bool(recovered)
        group = g.require_group("channels")
        for i, channel in enumerate(channels):
            if channel not in group:
                values = np.asarray(entry["all"][i])
                dataset = group.create_dataset(channel, data=values, compression="gzip")
                dataset.attrs["sha256"] = _digest(values)
        f.flush()


def canonical_response(cached: _CachedEUVResponse) -> _CachedEUVResponse:
    """Use stable channel order on first use as well as artifact reload."""
    channels = [_normalize_euv_channel_token(str(c)) for c in cached.response_meta.channels]
    order = sorted(range(len(channels)), key=lambda i: channels[i])
    arr = np.array(cached.response, copy=True)
    arr["all"][0] = np.asarray(cached.response)["all"][0, order]
    meta = SimpleNamespace(**{key: getattr(cached.response_meta, key)
                              for key in ("instrument", "source", "mode")},
                           channels=[channels[i] for i in order])
    identity = compute_euv_response_identity(response=arr, response_dt=arr.dtype, response_meta=meta)
    arr.setflags(write=False)
    return _CachedEUVResponse(arr, arr.dtype, meta, identity)


def bind_legacy_search_responses(path: Path, request: dict, cached: _CachedEUVResponse) -> list[str]:
    """Explicit recovery: bind old unhashed recipes without altering maps/scores.

    This cannot certify historical network responses. Keep the original metadata
    in a shared audit record and label the association as recovered/unverified.
    Search IDs stay stable so interrupted runs can continue.
    """
    from .search_contract import search_evaluation_signature

    identity = cached.response_identity
    if identity is None:
        raise ValueError("Response must have a validated identity")
    changed = []
    with h5py.File(path, "r+") as f:
        calibration = f[f"{GROUP}/{response_request_key(request)}"]
        audit = json.loads(calibration["legacy_bindings_json"][()]) if "legacy_bindings_json" in calibration else {}
        for slice_key, sg in f.get("slices", {}).items():
            for search_id, search in sg.get("searches", {}).items():
                if "request_json" not in search or "diagnostics_json" not in search:
                    continue
                old_request = json.loads(search["request_json"][()])
                diag = json.loads(search["diagnostics_json"][()])
                if (old_request.get("model_sha256") != request["model_sha256"]
                        or old_request.get("euv_response_sha256") is not None
                        or diag.get("euv_response_origin") != "pyEUVTools"
                        or str(diag.get("euv_channel")) not in request["channels"]
                        or request["source"] != "dynamic_evenorm_chiantifix"):
                    continue
                binding_key = f"{slice_key}/{search_id}"
                audit[binding_key] = {"request_before": old_request,
                                      "diagnostics_before": diag,
                                      "historical_response_verified": False,
                                      "recovered_response_sha256": identity.sha256}
                new_request = {**old_request, "euv_response_sha256": identity.sha256,
                               "euv_response_identity_version": identity.version}
                new_diag = {**diag, "euv_response_sha256": identity.sha256,
                            "euv_response_identity_version": identity.version,
                            "euv_response_identity_summary": identity.summary,
                            "euv_response_source": identity.summary["source"],
                            "euv_response_mode": identity.summary["mode"],
                            "euv_response_payload_key": response_request_key(request),
                            "euv_response_recovery": "historical response unverified; recovered for continuation",
                            "compatibility_signature": search_evaluation_signature(new_request)}
                for name, data in (("request_json", new_request), ("diagnostics_json", new_diag)):
                    del search[name]
                    search.create_dataset(name, data=json.dumps(data))
                changed.append(binding_key)
        if changed:
            if "legacy_bindings_json" in calibration:
                del calibration["legacy_bindings_json"]
            calibration.create_dataset("legacy_bindings_json", data=json.dumps(audit))
        f.flush()
    return changed
