from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from examples.python import adaptive_ab_search_single_frequency as frequency_wrapper
from examples.python.adaptive_ab_search_single_observation import _point_payload_from_result, _resolve_observation_request
from pychmp.ab_search import ABPointResult
from pychmp.metrics import MetricValues


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

    with pytest.raises(SystemExit, match="external FITS paths cannot be used"):
        _resolve_observation_request(args, repo_root=tmp_path)


def test_resolve_observation_request_rejects_conflicting_path_selectors(tmp_path: Path) -> None:
    args = _make_args(
        tmp_path,
        fits_file=tmp_path / "obs_a.fits",
        obs_path=tmp_path / "obs_b.fits",
    )

    with pytest.raises(SystemExit, match="Conflicting observation path selectors"):
        _resolve_observation_request(args, repo_root=tmp_path)


def test_frequency_wrapper_reexports_generic_entrypoint_helper() -> None:
    assert frequency_wrapper._resolve_observation_request is _resolve_observation_request


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


def test_adaptive_point_payload_persists_trial_maps_and_euv_components() -> None:
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

    payload = _point_payload_from_result(
        point,
        renderer_factory=lambda a, b: _FakePSFRenderer(),
        observed_template=np.zeros((2, 2), dtype=float),
        target_metric="chi2",
        psf_source="test",
        compatibility_signature="sig",
    )

    assert payload["trial_raw_modeled_maps"] is not None
    assert payload["trial_modeled_maps"] is not None
    assert payload["trial_residual_maps"] is not None
    assert payload["trial_euv_coronal_maps"] is not None
    assert payload["trial_euv_tr_maps"] is not None
    assert payload["trial_raw_modeled_maps"].shape == (2, 2, 2)
    assert payload["trial_modeled_maps"].shape == (2, 2, 2)
    assert payload["trial_residual_maps"].shape == (2, 2, 2)
    assert payload["trial_euv_coronal_maps"].shape == (2, 2, 2)
    assert payload["trial_euv_tr_maps"].shape == (2, 2, 2)
    np.testing.assert_allclose(payload["trial_raw_modeled_maps"][0], np.full((2, 2), 1.0e-4, dtype=float))
    np.testing.assert_allclose(payload["trial_modeled_maps"][1], np.full((2, 2), 10.0002, dtype=float))
    np.testing.assert_allclose(payload["trial_euv_coronal_maps"][0], np.full((2, 2), 1.0001, dtype=float))
    np.testing.assert_allclose(payload["trial_euv_tr_maps"][1], np.full((2, 2), 2.0002, dtype=float))
