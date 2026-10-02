from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from pychmp.calibration_store import GROUP, canonical_response, load_response, response_request_key, save_response
from pychmp.gxrender_adapter import _CachedEUVResponse, GXRenderEUVAdapter


def response():
    dtype = np.dtype([("ds", "f8"), ("NT", "i4"), ("Nchannels", "i4"),
                      ("logte", "f8", (3,)), ("all", "f8", (2, 3))])
    arr = np.zeros(1, dtype=dtype)
    arr["ds"] = 0.36
    arr["NT"] = 3
    arr["Nchannels"] = 2
    arr["logte"] = [5, 6, 7]
    arr["all"] = [[1, 2, 3], [4, 5, 6]]
    return _CachedEUVResponse(arr, dtype, SimpleNamespace(instrument="AIA", channels=["94", "171"], source="test", mode="evenorm_chiantifix"))


def request():
    return dict(model_sha256="model", instrument="AIA", channels=["171", "94"],
                source="dynamic_evenorm_chiantifix", response_sav_sha256=None)


def test_response_shared_across_channel_subsets_and_renderer_instances(tmp_path, monkeypatch):
    path = tmp_path / "artifact.h5"
    original = canonical_response(response())
    save_response(path, request(), original)
    restored = load_response(path, request())
    np.testing.assert_array_equal(restored.response, original.response)
    assert restored.response_identity.sha256 == original.response_identity.sha256
    save_response(path, {**request(), "channels": ["94"]}, restored)
    subset = load_response(path, {**request(), "channels": ["94"]})
    assert subset.response_meta.channels == ["94"]
    np.testing.assert_array_equal(subset.response["all"][0, 0], [1, 2, 3])
    with h5py.File(path) as f:
        assert len(f[GROUP]) == 1
        assert set(f[GROUP][response_request_key(request())]["channels"]) == {"94", "171"}
        assert "slices" not in f  # no search-specific copies
    monkeypatch.setattr(GXRenderEUVAdapter, "_resolve_euv_response_cache",
                        lambda self: pytest.fail("response provider must not run on restart"))
    for channel in ["94", "171"]:
        adapter = GXRenderEUVAdapter(model_path="missing.h5", channel=channel, prebuilt_response=restored)
        assert adapter._ensure_euv_response_cache() is restored
        assert adapter.response_identity().sha256 == original.response_identity.sha256


def test_response_rejects_corruption_or_replacement(tmp_path):
    path = tmp_path / "artifact.h5"
    save_response(path, request(), response())
    changed = response()
    changed.response["all"][0, 0, 0] = 99
    with pytest.raises(ValueError, match="Refusing to replace"):
        save_response(path, request(), changed)
    assert load_response(path, {**request(), "model_sha256": "other"}) is None
    with h5py.File(path, "r+") as f:
        f[GROUP][response_request_key(request())]["channels/94"][0] = 99
    with pytest.raises(ValueError, match="checksum"):
        load_response(path, request())


def test_legacy_binding_preserves_search_id_maps_scores_and_audits_origin(tmp_path):
    import json
    from pychmp.calibration_store import bind_legacy_search_responses
    from pychmp.search_contract import search_evaluation_signature

    path = tmp_path / "legacy.h5"
    original_request = {"model_sha256": "model", "euv_response_sha256": None,
                        "euv_response_identity_version": None, "target_metric": "eta2"}
    with h5py.File(path, "w") as f:
        s = f.create_group("slices/euv_94/searches/original_id")
        s.create_dataset("request_json", data=json.dumps(original_request))
        s.create_dataset("diagnostics_json", data=json.dumps({"euv_response_origin": "pyEUVTools", "euv_channel": "94"}))
        s.create_dataset("scores", data=[0.1, 0.2])
        f.create_dataset("map_store/raw", data=[1., 2.])
    r = canonical_response(response())
    save_response(path, request(), r, recovered=True)
    assert bind_legacy_search_responses(path, request(), r) == ["euv_94/original_id"]
    assert bind_legacy_search_responses(path, request(), r) == []
    with h5py.File(path) as f:
        s = f["slices/euv_94/searches/original_id"]
        req = json.loads(s["request_json"][()]);diag = json.loads(s["diagnostics_json"][()])
        assert req["euv_response_sha256"] == r.response_identity.sha256
        assert diag["compatibility_signature"] == search_evaluation_signature(req)
        np.testing.assert_array_equal(s["scores"], [0.1, 0.2])
        np.testing.assert_array_equal(f["map_store/raw"], [1., 2.])
        audit = json.loads(f[GROUP][response_request_key(request())]["legacy_bindings_json"][()])
        assert audit["euv_94/original_id"]["request_before"] == original_request
        assert not audit["euv_94/original_id"]["historical_response_verified"]
