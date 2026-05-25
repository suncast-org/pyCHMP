from __future__ import annotations

from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from examples.python.adaptive_ab_search_single_observation import (
    _PointRenderRecord,
    _PointRenderStream,
    _PersistentPointCache,
    _point_payload_from_result,
    _rescore_auxiliary_map_record,
    _resolve_observation_request,
    _resolve_render_slice_requests,
)
from pychmp.ab_scan_artifacts import COMPATIBILITY_SIGNATURE_KEY, load_scan_file, write_point_scan_artifact
from pychmp.ab_search import ABPointResult
from pychmp.metrics import MetricValues
from pychmp.obs_maps import load_obs_map


def _make_args(tmp_path: Path, **overrides: object) -> Namespace:
    values: dict[str, object] = {
        "fits_file": None,
        "model_h5": tmp_path / "model.h5",
        "obs_source": None,
        "obs_path": None,
        "obs_map_id": None,
        "ebtel_path": tmp_path / "ebtel.sav",
        "testdata_repo": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_resolve_observation_request_defaults_to_model_refmap_when_map_id_is_supplied(tmp_path: Path) -> None:
    args = _make_args(tmp_path, obs_map_id="AIA_171")

    request = _resolve_observation_request(args, repo_root=tmp_path)

    assert request.source_mode == "model_refmap"
    assert request.obs_path is None
    assert request.obs_map_id == "AIA_171"
    assert request.model_h5 == (tmp_path / "model.h5").resolve()
    assert request.ebtel_path == (tmp_path / "ebtel.sav").resolve()


def test_resolve_observation_request_rejects_external_path_with_model_refmap_source(tmp_path: Path) -> None:
    obs_path = tmp_path / "obs.fits"
    args = _make_args(
        tmp_path,
        obs_source="model_refmap",
        obs_path=obs_path,
        obs_map_id="AIA_171",
    )

    request = _resolve_observation_request(args, repo_root=tmp_path)

    with pytest.raises(ValueError, match="obs_path cannot be used"):
        load_obs_map(
            obs_path=request.obs_path,
            model_h5=request.model_h5,
            map_id=request.obs_map_id,
            source_mode=request.source_mode,
        )


def test_resolve_observation_request_prefers_explicit_obs_path_over_positional_fits(tmp_path: Path) -> None:
    args = _make_args(
        tmp_path,
        fits_file=tmp_path / "obs_a.fits",
        obs_path=tmp_path / "obs_b.fits",
    )

    request = _resolve_observation_request(args, repo_root=tmp_path)

    assert request.obs_path == (tmp_path / "obs_b.fits").resolve()


def test_resolve_observation_request_requires_explicit_external_fits_path(tmp_path: Path) -> None:
    args = _make_args(tmp_path, obs_source="external_fits", fits_file=None, obs_path=None)

    request = _resolve_observation_request(args, repo_root=tmp_path)

    with pytest.raises(ValueError, match="obs_path is required"):
        load_obs_map(
            obs_path=request.obs_path,
            model_h5=request.model_h5,
            map_id=request.obs_map_id,
            source_mode=request.source_mode,
        )


def test_resolve_render_slice_requests_expands_aia_all_channels() -> None:
    descriptors, freqs, channels = _resolve_render_slice_requests(
        domain="euv",
        frequency_ghz=None,
        euv_channel="171",
        euv_instrument="AIA",
        all_channels=True,
        render_channels_csv=None,
        render_frequencies_csv=None,
    )

    assert freqs == tuple()
    assert channels == ("171", "94", "131", "193", "211", "304", "335")
    assert [item["key"] for item in descriptors] == [
        "euv_171",
        "euv_94",
        "euv_131",
        "euv_193",
        "euv_211",
        "euv_304",
        "euv_335",
    ]
    assert descriptors[0]["is_target"] is True
    assert all(item["role"] == "auxiliary" for item in descriptors[1:])


def test_resolve_render_slice_requests_requires_explicit_mw_frequency_list() -> None:
    descriptors, freqs, channels = _resolve_render_slice_requests(
        domain="mw",
        frequency_ghz=2.874,
        euv_channel=None,
        euv_instrument=None,
        all_channels=False,
        render_channels_csv=None,
        render_frequencies_csv="3.2,5.8",
    )

    assert channels == tuple()
    assert freqs == (2.874, 3.2, 5.8)
    assert [item["key"] for item in descriptors] == ["mw_2p874000ghz", "mw_3p200000ghz", "mw_5p800000ghz"]


def test_resolve_render_slice_requests_rejects_all_channels_for_mw() -> None:
    with pytest.raises(SystemExit, match="only defined for fixed-channel"):
        _resolve_render_slice_requests(
            domain="mw",
            frequency_ghz=2.874,
            euv_channel=None,
            euv_instrument=None,
            all_channels=True,
            render_channels_csv=None,
            render_frequencies_csv=None,
        )


class _FakeEUVRenderer:
    def render(self, q0: float) -> np.ndarray:
        return np.full((2, 2), float(q0), dtype=float)

    def render_components(self, q0: float) -> dict[str, np.ndarray]:
        return {
            "flux_corona": np.full((2, 2), float(q0) + 1.0, dtype=float),
            "flux_tr": np.full((2, 2), float(q0) + 2.0, dtype=float),
        }


class _FakePSFRenderer:
    def __init__(self) -> None:
        self._base = _FakeEUVRenderer()

    def render_pair(self, q0: float) -> tuple[np.ndarray, np.ndarray]:
        raw = self._base.render(q0)
        return raw, raw + 10.0

    def render(self, q0: float) -> np.ndarray:
        _raw, modeled = self.render_pair(q0)
        return modeled


def test_adaptive_point_payload_uses_lightweight_storage_by_default() -> None:
    point = ABPointResult(
        a=0.3,
        b=2.7,
        q0=2.0e-4,
        objective_value=1.0,
        metrics=MetricValues(chi2=1.0, rho2=2.0, eta2=3.0),
        target_metric="chi2",
        success=True,
        nfev=3,
        nit=2,
        message="ok",
        used_adaptive_bracketing=True,
        bracket_found=True,
        bracket=(1.0e-4, 2.0e-4, 4.0e-4),
        trial_q0=(1.0e-4, 2.0e-4),
        trial_objective_values=(5.0, 1.0),
        trial_chi2_values=(5.0, 1.0),
        trial_rho2_values=(6.0, 2.0),
        trial_eta2_values=(7.0, 3.0),
        elapsed_seconds=1.5,
    )

    stream_record = _PointRenderRecord()
    q0_key = _PointRenderStream._q0_key(point.q0)
    stream_record.raw_modeled_by_q0[q0_key] = np.full((2, 2), 2.0, dtype=np.float32)
    stream_record.modeled_by_q0[q0_key] = np.full((2, 2), 12.0, dtype=np.float32)

    payload = _point_payload_from_result(
        point,
        renderer_factory=lambda a, b: _FakePSFRenderer(),
        observed_template=np.zeros((2, 2), dtype=float),
        target_metric="chi2",
        psf_source="test",
        compatibility_signature="sig",
        stream_record=stream_record,
    )

    assert payload["trial_raw_modeled_maps"] is None
    assert payload["trial_modeled_maps"] is None
    assert payload["trial_residual_maps"] is None
    assert payload["trial_euv_coronal_maps"] is None
    assert payload["trial_euv_tr_maps"] is None
    diagnostics = payload["diagnostics"]
    assert diagnostics["store_trial_map_cubes"] is False
    assert diagnostics["synthetic_map_db_version"] == 1
    assert isinstance(diagnostics["synthetic_map_keys"], list)


def test_adaptive_point_payload_can_persist_trial_maps_when_enabled() -> None:
    point = ABPointResult(
        a=0.3,
        b=2.7,
        q0=2.0e-4,
        objective_value=1.0,
        metrics=MetricValues(chi2=1.0, rho2=2.0, eta2=3.0),
        target_metric="chi2",
        success=True,
        nfev=3,
        nit=2,
        message="ok",
        used_adaptive_bracketing=True,
        bracket_found=True,
        bracket=(1.0e-4, 2.0e-4, 4.0e-4),
        trial_q0=(1.0e-4, 2.0e-4),
        trial_objective_values=(5.0, 1.0),
        trial_chi2_values=(5.0, 1.0),
        trial_rho2_values=(6.0, 2.0),
        trial_eta2_values=(7.0, 3.0),
        elapsed_seconds=1.5,
    )

    stream_record = _PointRenderRecord()
    q0_key = _PointRenderStream._q0_key(point.q0)
    stream_record.raw_modeled_by_q0[q0_key] = np.full((2, 2), 2.0, dtype=np.float32)
    stream_record.modeled_by_q0[q0_key] = np.full((2, 2), 12.0, dtype=np.float32)
    stream_record.raw_modeled_by_q0[_PointRenderStream._q0_key(1.0e-4)] = np.full((2, 2), 1.0, dtype=np.float32)
    stream_record.modeled_by_q0[_PointRenderStream._q0_key(1.0e-4)] = np.full((2, 2), 11.0, dtype=np.float32)
    stream_record.raw_modeled_by_q0[_PointRenderStream._q0_key(2.0e-4)] = np.full((2, 2), 2.0, dtype=np.float32)
    stream_record.modeled_by_q0[_PointRenderStream._q0_key(2.0e-4)] = np.full((2, 2), 12.0, dtype=np.float32)
    stream_record.components_by_q0[q0_key] = {
        "flux_corona": np.full((2, 2), 20.0, dtype=np.float32),
        "flux_tr": np.full((2, 2), 30.0, dtype=np.float32),
        "rendered_by_channel": {"171": np.full((2, 2), 40.0, dtype=np.float32)},
    }
    stream_record.components_by_q0[_PointRenderStream._q0_key(1.0e-4)] = {
        "flux_corona": np.full((2, 2), 21.0, dtype=np.float32),
        "flux_tr": np.full((2, 2), 31.0, dtype=np.float32),
        "rendered_by_channel": {"171": np.full((2, 2), 41.0, dtype=np.float32)},
    }
    stream_record.components_by_q0[_PointRenderStream._q0_key(2.0e-4)] = {
        "flux_corona": np.full((2, 2), 22.0, dtype=np.float32),
        "flux_tr": np.full((2, 2), 32.0, dtype=np.float32),
        "rendered_by_channel": {"171": np.full((2, 2), 42.0, dtype=np.float32)},
    }

    payload = _point_payload_from_result(
        point,
        renderer_factory=lambda a, b: _FakePSFRenderer(),
        observed_template=np.zeros((2, 2), dtype=float),
        target_metric="chi2",
        psf_source="test",
        compatibility_signature="sig",
        store_trial_map_cubes=True,
        stream_record=stream_record,
    )

    assert payload["trial_raw_modeled_maps"] is not None
    assert payload["trial_modeled_maps"] is not None
    assert payload["trial_residual_maps"] is not None
    assert payload["trial_euv_coronal_maps"] is not None
    assert payload["trial_euv_tr_maps"] is not None


def test_adaptive_point_payload_uses_stream_record_without_renderer_fallback() -> None:
    point = ABPointResult(
        a=0.3,
        b=2.7,
        q0=2.0e-4,
        objective_value=1.0,
        metrics=MetricValues(chi2=1.0, rho2=2.0, eta2=3.0),
        target_metric="chi2",
        success=True,
        nfev=3,
        nit=2,
        message="ok",
        used_adaptive_bracketing=True,
        bracket_found=True,
        bracket=(1.0e-4, 2.0e-4, 4.0e-4),
        trial_q0=(1.0e-4, 2.0e-4),
        trial_objective_values=(5.0, 1.0),
        trial_chi2_values=(5.0, 1.0),
        trial_rho2_values=(6.0, 2.0),
        trial_eta2_values=(7.0, 3.0),
        elapsed_seconds=1.5,
    )
    stream_record = _PointRenderRecord()
    q0_key_0 = _PointRenderStream._q0_key(1.0e-4)
    q0_key_1 = _PointRenderStream._q0_key(2.0e-4)
    stream_record.raw_modeled_by_q0 = {
        q0_key_0: np.full((2, 2), 1.0, dtype=np.float32),
        q0_key_1: np.full((2, 2), 2.0, dtype=np.float32),
    }
    stream_record.modeled_by_q0 = {
        q0_key_0: np.full((2, 2), 11.0, dtype=np.float32),
        q0_key_1: np.full((2, 2), 12.0, dtype=np.float32),
    }

    payload = _point_payload_from_result(
        point,
        renderer_factory=lambda a, b: _FakePSFRenderer(),
        observed_template=np.zeros((2, 2), dtype=float),
        target_metric="chi2",
        psf_source="test",
        compatibility_signature="sig",
        store_trial_map_cubes=True,
        stream_record=stream_record,
    )

    np.testing.assert_allclose(payload["raw_modeled_best"], np.full((2, 2), 2.0, dtype=float))
    np.testing.assert_allclose(payload["modeled_best"], np.full((2, 2), 12.0, dtype=float))
    assert payload["trial_modeled_maps"] is not None
    assert payload["trial_modeled_maps"].shape == (2, 2, 2)


def test_rescore_auxiliary_map_record_builds_promoted_point_payload() -> None:
    observed = np.ones((2, 2), dtype=float)
    sigma = np.ones((2, 2), dtype=float)
    record = {
        "a": 0.3,
        "b": 2.7,
        "a_index": 0,
        "b_index": 0,
        "fit_q0_trials": (1.0, 2.0),
        "trial_modeled_maps": np.stack(
            [
                np.full((2, 2), 1.0, dtype=float),
                np.full((2, 2), 3.0, dtype=float),
            ],
            axis=0,
        ),
        "diagnostics": {"target_metric": "chi2"},
        "source_slice_key": "euv_171",
        "source_search_id": "search_a",
    }

    promoted = _rescore_auxiliary_map_record(
        record,
        observed=observed,
        sigma_map=sigma,
        threshold=0.0,
        explicit_mask=None,
        target_metric="chi2",
    )

    assert promoted is not None
    point, payload = promoted
    assert point.q0 == pytest.approx(1.0)
    assert point.objective_value == pytest.approx(0.0)
    assert payload["diagnostics"]["map_store_reused"] is True
    assert payload["diagnostics"]["map_store_source_slice_key"] == "euv_171"
    np.testing.assert_allclose(payload["modeled_best"], np.ones((2, 2), dtype=float))
    np.testing.assert_allclose(payload["trial_modeled_maps"][1], np.full((2, 2), 3.0, dtype=float))


def test_rescore_auxiliary_map_record_reuses_saved_trial_metrics_without_trial_maps() -> None:
    observed = np.ones((2, 2), dtype=float)
    sigma = np.ones((2, 2), dtype=float)
    record = {
        "a": 0.3,
        "b": 2.7,
        "a_index": 0,
        "b_index": 0,
        "fit_q0_trials": (1.0, 2.0),
        "fit_metric_trials": (0.0, 4.0),
        "fit_chi2_trials": (0.0, 4.0),
        "fit_rho2_trials": (0.1, 4.1),
        "fit_eta2_trials": (0.2, 4.2),
        "modeled_best": np.ones((2, 2), dtype=float),
        "raw_modeled_best": np.ones((2, 2), dtype=float),
        "diagnostics": {"target_metric": "chi2"},
        "source_slice_key": "euv_171",
        "source_search_id": "search_a",
    }

    promoted = _rescore_auxiliary_map_record(
        record,
        observed=observed,
        sigma_map=sigma,
        threshold=0.0,
        explicit_mask=None,
        target_metric="chi2",
    )

    assert promoted is not None
    point, payload = promoted
    assert point.q0 == pytest.approx(1.0)
    assert point.objective_value == pytest.approx(0.0)
    assert payload["diagnostics"]["map_store_reused_without_trial_maps"] is True
    assert payload["trial_modeled_maps"] is None


def test_start_over_promotes_same_signature_current_slice_points(tmp_path: Path) -> None:
    observed = np.ones((2, 2), dtype=float)
    sigma_map = np.ones((2, 2), dtype=float)
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

    diagnostics = {
        "artifact_kind": "pychmp_ab_scan_sparse_points",
        COMPATIBILITY_SIGNATURE_KEY: "sig-123",
        "target_metric": "chi2",
        "metrics_mask_threshold": 0.1,
        "metrics_mask_source": "union_threshold",
        "mask_type": "union",
        "model_sha256": "a" * 64,
        "fits_sha256": "b" * 64,
        "ebtel_sha256": "c" * 64,
        "frequency_ghz": 5.7,
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
        "target_slice_key": "mw_5p700000ghz",
        "spectral_domain": "mw",
        "spectral_label": "5.700 GHz",
    }
    point_record = {
        "a": 0.3,
        "b": 2.7,
        "a_index": 0,
        "b_index": 0,
        "q0": 1.0,
        "success": True,
        "status": "computed",
        "modeled_best": np.ones((2, 2), dtype=float),
        "raw_modeled_best": np.ones((2, 2), dtype=float),
        "residual": np.zeros((2, 2), dtype=float),
        "fit_q0_trials": (0.5, 1.0),
        "fit_metric_trials": (0.4, 0.1),
        "fit_chi2_trials": (0.4, 0.1),
        "fit_rho2_trials": (0.5, 0.2),
        "fit_eta2_trials": (0.6, 0.3),
        "trial_modeled_maps": np.stack(
            [
                np.full((2, 2), 0.9, dtype=float),
                np.full((2, 2), 1.0, dtype=float),
            ],
            axis=0,
        ),
        "trial_raw_modeled_maps": np.stack(
            [
                np.full((2, 2), 0.9, dtype=float),
                np.full((2, 2), 1.0, dtype=float),
            ],
            axis=0,
        ),
        "nfev": 2,
        "nit": 1,
        "message": "ok",
        "used_adaptive_bracketing": False,
        "bracket_found": False,
        "bracket": None,
        "target_metric": "chi2",
        "diagnostics": {
            COMPATIBILITY_SIGNATURE_KEY: "sig-123",
            "target_metric": "chi2",
            "chi2": 0.1,
            "rho2": 0.2,
            "eta2": 0.3,
            "target_metric_value": 0.1,
        },
    }
    artifact_h5 = tmp_path / "adaptive.h5"
    write_point_scan_artifact(
        artifact_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_records=[point_record],
    )

    cache = _PersistentPointCache(
        artifact_h5=artifact_h5,
        observed=observed,
        sigma_map=sigma_map,
        target_header=header,
        diagnostics=diagnostics,
        blos_reference=None,
        renderer_factory=lambda a_value, b_value: None,
        target_metric="chi2",
        psf_source="none",
        psf_kernel=None,
        compatibility_signature="sig-123",
        viewer_heartbeat=None,
    )

    assert cache.promote_current_slice_trial_maps(
        threshold=0.1,
        explicit_mask=None,
        include_matching_signature=False,
    ) == 0
    assert cache.promote_current_slice_trial_maps(
        threshold=0.1,
        explicit_mask=None,
        include_matching_signature=True,
    ) == 1
    assert len(cache) == 1


def test_promote_current_slice_trial_maps_allows_metric_change(tmp_path: Path) -> None:
    observed = np.ones((2, 2), dtype=float)
    sigma_map = np.ones((2, 2), dtype=float)
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

    diagnostics = {
        "artifact_kind": "pychmp_ab_scan_sparse_points",
        COMPATIBILITY_SIGNATURE_KEY: "sig-123",
        "target_metric": "chi2",
        "metrics_mask_threshold": 0.1,
        "metrics_mask_source": "union_threshold",
        "mask_type": "union",
        "model_sha256": "a" * 64,
        "fits_sha256": "b" * 64,
        "ebtel_sha256": "c" * 64,
        "frequency_ghz": 5.7,
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
        "target_slice_key": "mw_5p700000ghz",
        "spectral_domain": "mw",
        "spectral_label": "5.700 GHz",
    }
    point_record = {
        "a": 0.3,
        "b": 2.7,
        "a_index": 0,
        "b_index": 0,
        "q0": 1.0,
        "success": True,
        "status": "computed",
        "modeled_best": np.ones((2, 2), dtype=float),
        "raw_modeled_best": np.ones((2, 2), dtype=float),
        "residual": np.zeros((2, 2), dtype=float),
        "fit_q0_trials": (0.5, 1.0),
        "fit_metric_trials": (0.4, 0.1),
        "fit_chi2_trials": (0.4, 0.1),
        "fit_rho2_trials": (0.5, 0.2),
        "fit_eta2_trials": (0.6, 0.3),
        "trial_modeled_maps": np.stack(
            [
                np.full((2, 2), 0.9, dtype=float),
                np.full((2, 2), 1.0, dtype=float),
            ],
            axis=0,
        ),
        "trial_raw_modeled_maps": np.stack(
            [
                np.full((2, 2), 0.9, dtype=float),
                np.full((2, 2), 1.0, dtype=float),
            ],
            axis=0,
        ),
        "nfev": 2,
        "nit": 1,
        "message": "ok",
        "used_adaptive_bracketing": False,
        "bracket_found": False,
        "bracket": None,
        "target_metric": "chi2",
        "diagnostics": {
            COMPATIBILITY_SIGNATURE_KEY: "sig-123",
            "target_metric": "chi2",
            "chi2": 0.1,
            "rho2": 0.2,
            "eta2": 0.3,
            "target_metric_value": 0.1,
        },
    }
    artifact_h5 = tmp_path / "adaptive_metric_change.h5"
    write_point_scan_artifact(
        artifact_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_records=[point_record],
    )

    rescored_diagnostics = {
        **diagnostics,
        COMPATIBILITY_SIGNATURE_KEY: "sig-eta2-threshold-0p5",
        "target_metric": "eta2",
        "metrics_mask_threshold": 0.5,
    }
    cache = _PersistentPointCache(
        artifact_h5=artifact_h5,
        observed=observed,
        sigma_map=sigma_map,
        target_header=header,
        diagnostics=rescored_diagnostics,
        blos_reference=None,
        renderer_factory=lambda a_value, b_value: None,
        target_metric="eta2",
        psf_source="none",
        psf_kernel=None,
        compatibility_signature="sig-eta2-threshold-0p5",
        viewer_heartbeat=None,
    )

    assert cache.promote_current_slice_trial_maps(
        threshold=0.5,
        explicit_mask=None,
        include_matching_signature=False,
    ) == 1
    assert len(cache) == 1


class _FakeCacheRenderer:
    def render(self, q0: float) -> np.ndarray:
        return np.full((2, 2), float(q0), dtype=float)


def test_cache_setitem_persists_point_via_dispatcher(tmp_path: Path) -> None:
    observed = np.ones((2, 2), dtype=float)
    sigma_map = np.ones((2, 2), dtype=float)
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

    diagnostics = {
        "artifact_kind": "pychmp_ab_scan_sparse_points",
        COMPATIBILITY_SIGNATURE_KEY: "sig-123",
        "target_metric": "chi2",
        "metrics_mask_threshold": 0.1,
        "metrics_mask_source": "union_threshold",
        "mask_type": "union",
        "model_sha256": "a" * 64,
        "fits_sha256": "b" * 64,
        "ebtel_sha256": "c" * 64,
        "frequency_ghz": 5.7,
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
        "target_slice_key": "mw_5p700000ghz",
        "spectral_domain": "mw",
        "spectral_label": "5.700 GHz",
    }
    artifact_h5 = tmp_path / "dispatcher_cache_write.h5"
    write_point_scan_artifact(
        artifact_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_records=[],
    )
    point_template = ABPointResult(
        a=0.3,
        b=2.7,
        q0=1.0,
        objective_value=0.1,
        metrics=MetricValues(chi2=0.1, rho2=0.2, eta2=0.3),
        target_metric="chi2",
        success=True,
        nfev=2,
        nit=1,
        message="ok",
        used_adaptive_bracketing=False,
        bracket_found=False,
        bracket=None,
        trial_q0=(0.5, 1.0),
        trial_objective_values=(0.4, 0.1),
        trial_chi2_values=(0.4, 0.1),
        trial_rho2_values=(0.5, 0.2),
        trial_eta2_values=(0.6, 0.3),
        elapsed_seconds=0.0,
    )
    point_stream_record = _PointRenderRecord()
    point_q0_key = _PointRenderStream._q0_key(point_template.q0)
    point_stream_record.raw_modeled_by_q0[point_q0_key] = np.full((2, 2), point_template.q0, dtype=np.float32)
    point_stream_record.modeled_by_q0[point_q0_key] = np.full((2, 2), point_template.q0, dtype=np.float32)
    point = replace(
        point_template,
        artifact_payload=_point_payload_from_result(
            point_template,
            renderer_factory=lambda a_value, b_value: _FakeCacheRenderer(),
            observed_template=observed,
            target_metric="chi2",
            psf_source="none",
            compatibility_signature="sig-123",
            stream_record=point_stream_record,
        ),
    )
    cache = _PersistentPointCache(
        artifact_h5=artifact_h5,
        observed=observed,
        sigma_map=sigma_map,
        target_header=header,
        diagnostics=diagnostics,
        blos_reference=None,
        renderer_factory=lambda a_value, b_value: _FakeCacheRenderer(),
        target_metric="chi2",
        psf_source="none",
        psf_kernel=None,
        compatibility_signature="sig-123",
        viewer_heartbeat=None,
    )

    try:
        cache[(0.3, 2.7)] = point
        cache.close()
        loaded = load_scan_file(artifact_h5)
        assert len(loaded["point_records"]) == 1
        assert float(loaded["point_records"][0]["a"]) == pytest.approx(0.3)
    finally:
        pass


def test_cache_setitem_enqueues_without_waiting_for_writer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observed = np.ones((2, 2), dtype=float)
    sigma_map = np.ones((2, 2), dtype=float)
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

    diagnostics = {
        "artifact_kind": "pychmp_ab_scan_sparse_points",
        COMPATIBILITY_SIGNATURE_KEY: "sig-123",
        "target_metric": "chi2",
        "metrics_mask_threshold": 0.1,
        "metrics_mask_source": "union_threshold",
        "mask_type": "union",
        "model_sha256": "a" * 64,
        "fits_sha256": "b" * 64,
        "ebtel_sha256": "c" * 64,
        "frequency_ghz": 5.7,
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
        "target_slice_key": "mw_5p700000ghz",
        "spectral_domain": "mw",
        "spectral_label": "5.700 GHz",
    }
    artifact_h5 = tmp_path / "dispatcher_cache_async_write.h5"
    write_point_scan_artifact(
        artifact_h5,
        observed=observed,
        sigma_map=sigma_map,
        wcs_header=header,
        diagnostics=diagnostics,
        point_records=[],
    )

    gate = {"release": False}

    def _slow_append(*args: object, **kwargs: object) -> None:
        if gate["release"]:
            return
        raise RuntimeError("writer intentionally blocked")

    monkeypatch.setattr(
        "examples.python.adaptive_ab_search_single_observation.append_point_record",
        _slow_append,
    )

    point_template = ABPointResult(
        a=0.3,
        b=2.7,
        q0=1.0,
        objective_value=0.1,
        metrics=MetricValues(chi2=0.1, rho2=0.2, eta2=0.3),
        target_metric="chi2",
        success=True,
        nfev=2,
        nit=1,
        message="ok",
        used_adaptive_bracketing=False,
        bracket_found=False,
        bracket=None,
        trial_q0=(0.5, 1.0),
        trial_objective_values=(0.4, 0.1),
        trial_chi2_values=(0.4, 0.1),
        trial_rho2_values=(0.5, 0.2),
        trial_eta2_values=(0.6, 0.3),
        elapsed_seconds=0.0,
    )
    point_stream_record = _PointRenderRecord()
    point_q0_key = _PointRenderStream._q0_key(point_template.q0)
    point_stream_record.raw_modeled_by_q0[point_q0_key] = np.full((2, 2), point_template.q0, dtype=np.float32)
    point_stream_record.modeled_by_q0[point_q0_key] = np.full((2, 2), point_template.q0, dtype=np.float32)
    point = replace(
        point_template,
        artifact_payload=_point_payload_from_result(
            point_template,
            renderer_factory=lambda a_value, b_value: _FakeCacheRenderer(),
            observed_template=observed,
            target_metric="chi2",
            psf_source="none",
            compatibility_signature="sig-123",
            stream_record=point_stream_record,
        ),
    )
    cache = _PersistentPointCache(
        artifact_h5=artifact_h5,
        observed=observed,
        sigma_map=sigma_map,
        target_header=header,
        diagnostics=diagnostics,
        blos_reference=None,
        renderer_factory=lambda a_value, b_value: _FakeCacheRenderer(),
        target_metric="chi2",
        psf_source="none",
        psf_kernel=None,
        compatibility_signature="sig-123",
        viewer_heartbeat=None,
    )

    cache[(0.3, 2.7)] = point
    with pytest.raises(RuntimeError, match="artifact write dispatcher failed"):
        cache.close()
