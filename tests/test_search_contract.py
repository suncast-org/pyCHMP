from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
from astropy.io import fits

from pychmp.ab_scan_artifacts import (
    OBSERVATION_REF_GROUP,
    SEARCHES_GROUP,
    SLICE_CONTAINER_GROUP,
    UNIFIED_ARTIFACT_KIND,
    load_scan_file,
    load_slice_observation_reference_payload,
    write_point_scan_artifact,
)
from pychmp.search_contract import (
    SEARCH_EVALUATION_CONTRACT_VERSION,
    apply_search_shift_diagnostics,
    build_search_evaluation_config,
    compatibility_signature_from_diagnostics,
    search_evaluation_signature,
)


def _make_header() -> fits.Header:
    header = fits.Header()
    header["SIMPLE"] = True
    header["BITPIX"] = -32
    header["NAXIS"] = 2
    header["NAXIS1"] = 2
    header["NAXIS2"] = 2
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRPIX1"] = 1.0
    header["CRPIX2"] = 1.0
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CDELT1"] = 2.0
    header["CDELT2"] = 2.0
    header["DATE-OBS"] = "2020-11-26T20:00:00"
    header["OBSERVER"] = "earth"
    return header


def _make_diagnostics(*, shift_policy: str = "auto", target_metric: str = "chi2") -> dict[str, object]:
    return {
        "artifact_kind": UNIFIED_ARTIFACT_KIND,
        "target_metric": target_metric,
        "model_sha256": "a" * 64,
        "forward_model_sha256": "a" * 64,
        "forward_model_identity_version": "pychmp.forward_model.file_sha256.v0",
        "fits_sha256": "b" * 64,
        "ebtel_sha256": "c" * 64,
        "frequency_ghz": 17.0,
        "spectral_domain": "mw",
        "spectral_label": "17 GHz",
        "map_xc_arcsec": 0.0,
        "map_yc_arcsec": 0.0,
        "map_dx_arcsec": 2.0,
        "map_dy_arcsec": 2.0,
        "map_nx": 2,
        "map_ny": 2,
        "observer_name": "earth",
        "observer_lonc_deg": 0.0,
        "observer_b0sun_deg": 0.0,
        "observer_dsun_cm": 1.495978707e13,
        "observer_obs_time": "2020-11-26T20:00:00",
        "metrics_mask_threshold": 0.1,
        "metrics_mask_source": "union_threshold",
        "mask_type": "union",
        "shift_policy": shift_policy,
        "max_shift_arcsec": 20.0,
        "xy_shift_arcsec": [0.0, 0.0],
        "use_smoothed_obs_max": True,
        "use_emthreshold": True,
        "emthreshold": 0.5,
        "q0_search_stages": ["data"],
    }


def test_write_point_scan_artifact_stores_shared_canvas_in_slice_common(tmp_path) -> None:
    out_h5 = tmp_path / "artifact.h5"
    observed = np.arange(4, dtype=float).reshape(2, 2)
    sigma_map = np.full((2, 2), 0.25, dtype=float)
    canvas = np.full((4, 4), 3.0, dtype=float)
    sigma_canvas = np.full((4, 4), 0.5, dtype=float)
    header = _make_header()
    canvas_header = header.copy()
    diagnostics = _make_diagnostics()
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_records=[],
        observation_canvas=canvas,
        sigma_canvas=sigma_canvas,
        canvas_wcs_header=canvas_header,
    )
    with h5py.File(out_h5, "r") as f:
        slice_key = next(iter(f[SLICE_CONTAINER_GROUP].keys()))
        common = f[SLICE_CONTAINER_GROUP][slice_key]["common"]
        assert "observed" in common
        assert "sigma_map" in common
        assert "observation_canvas" in common
        assert "sigma_canvas" in common
        active_search_id = f[SLICE_CONTAINER_GROUP][slice_key]["active_search_id"][()].decode("utf-8")
        search_group = f[SLICE_CONTAINER_GROUP][slice_key][SEARCHES_GROUP][active_search_id]
        assert OBSERVATION_REF_GROUP not in search_group
        np.testing.assert_allclose(np.asarray(common["observation_canvas"], dtype=float), canvas)

    payload = load_scan_file(out_h5, slice_key=slice_key, search_id=active_search_id)
    np.testing.assert_allclose(payload["observed"], observed)
    np.testing.assert_allclose(payload["observation_canvas"], canvas)


def test_load_slice_observation_reference_payload_reads_slice_common(tmp_path) -> None:
    out_h5 = tmp_path / "artifact.h5"
    observed = np.ones((2, 2), dtype=float)
    sigma_map = np.full((2, 2), 0.5, dtype=float)
    write_point_scan_artifact(
        out_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=_make_header(),
        diagnostics=_make_diagnostics(),
        point_records=[],
    )
    with h5py.File(out_h5, "r") as f:
        slice_key = next(iter(f[SLICE_CONTAINER_GROUP].keys()))
    payload = load_slice_observation_reference_payload(out_h5, slice_key=slice_key)
    assert payload is not None
    np.testing.assert_allclose(payload["observed"], observed)


def test_load_slice_observation_reference_payload_legacy_search_ref_fallback(tmp_path) -> None:
    out_h5 = tmp_path / "legacy.h5"
    observed = np.ones((2, 2), dtype=float)
    sigma_map = np.full((2, 2), 0.5, dtype=float)
    header = _make_header()
    with h5py.File(out_h5, "w") as f:
        slices = f.create_group(SLICE_CONTAINER_GROUP)
        slice_group = slices.create_group("mw_17p000000ghz")
        common = slice_group.create_group("common")
        common.create_dataset("wcs_header", data=header.tostring(sep="\n", endcard=True))
        common.create_dataset("diagnostics_json", data=json.dumps({"artifact_kind": UNIFIED_ARTIFACT_KIND}))
        searches = slice_group.create_group(SEARCHES_GROUP)
        search = searches.create_group("legacy_search")
        ref = search.create_group(OBSERVATION_REF_GROUP)
        ref.create_dataset("observed", data=observed)
        ref.create_dataset("sigma_map", data=sigma_map)
        ref.create_dataset("wcs_header", data=header.tostring(sep="\n", endcard=True))
        ref.create_dataset("diagnostics_json", data="{}")

    payload = load_slice_observation_reference_payload(out_h5, slice_key="mw_17p000000ghz")
    assert payload is not None
    np.testing.assert_allclose(payload["observed"], observed)


def test_search_evaluation_config_excludes_slice_observation_content_hashes() -> None:
    diagnostics = _make_diagnostics()
    diagnostics["slice_observation_identity_sha256"] = "deadbeef"
    diagnostics["preprocessed_observation_sha256"] = "cafebabe"
    config = build_search_evaluation_config(diagnostics, layout={"kind": "point_list"})
    assert "slice_observation_identity_sha256" not in config
    assert "preprocessed_observation_sha256" not in config


def test_search_evaluation_config_includes_shift_policy_and_masks() -> None:
    diagnostics = _make_diagnostics(shift_policy="auto")
    config = build_search_evaluation_config(diagnostics, layout={"kind": "point_list"})
    assert config["schema"] == SEARCH_EVALUATION_CONTRACT_VERSION
    assert config["shift_policy"] == "auto"
    assert config["metrics_mask"]["threshold"] == pytest.approx(0.1)

    changed = dict(diagnostics)
    changed["shift_policy"] = "fixed"
    assert search_evaluation_signature(config) != search_evaluation_signature(
        build_search_evaluation_config(changed, layout={"kind": "point_list"})
    )


def test_compatibility_signature_matches_search_evaluation_signature() -> None:
    diagnostics = _make_diagnostics()
    layout = {"kind": "point_list"}
    expected = search_evaluation_signature(build_search_evaluation_config(diagnostics, layout=layout))
    assert compatibility_signature_from_diagnostics(diagnostics, layout=layout) == expected


def test_apply_search_shift_diagnostics_overrides_slice_common_auto_for_legacy_search() -> None:
    slice_diag = _make_diagnostics(shift_policy="auto")
    merged = apply_search_shift_diagnostics(slice_diag, search_request={"shift_policy": None})
    assert merged["shift_policy"] == "fixed"
    assert merged["xy_shift_arcsec"] == [0.0, 0.0]

    auto_merged = apply_search_shift_diagnostics(
        slice_diag,
        search_request={"shift_policy": "auto", "max_shift_arcsec": 20.0},
    )
    assert auto_merged["shift_policy"] == "auto"
    assert auto_merged["max_shift_arcsec"] == pytest.approx(20.0)
