from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from examples import fit_q0_obs_map
from pychmp.geometry_policy import (
    infer_observation_observer,
    resolve_geometry_policy,
    resolve_renderer_observer_name,
)


def _obs(**overrides: object) -> SimpleNamespace:
    values = {
        "domain": "euv",
        "instrument": "AIA",
        "observer": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _saved_fov() -> dict[str, float]:
    return {
        "xc_arcsec": -600.0,
        "yc_arcsec": -300.0,
        "xsize_arcsec": 300.0,
        "ysize_arcsec": 300.0,
    }


def test_aia_and_radio_observations_are_earth_los() -> None:
    assert infer_observation_observer(_obs(domain="euv", instrument="AIA")) == "earth"
    assert infer_observation_observer(_obs(domain="mw", instrument="EOVSA")) == "earth"


def test_saved_model_fov_is_reused_when_observation_los_matches_model() -> None:
    decision = resolve_geometry_policy(
        obs_map=_obs(domain="mw", instrument="EOVSA"),
        model_observer_meta={"observer_name": "earth", "observer_lonc_deg": 0.0, "observer_b0sun_deg": 0.0},
        saved_fov=_saved_fov(),
        geometry_overrides_requested=False,
        explicit_observer_requested=False,
    )

    assert decision.observation_observer == "earth"
    assert decision.model_observer == "earth"
    assert decision.los_aligned is True
    assert decision.use_model_saved_fov is True
    assert decision.geometry_mode == "saved_fov"


def test_saved_model_fov_is_ignored_when_observation_los_differs_from_model() -> None:
    decision = resolve_geometry_policy(
        obs_map=_obs(domain="euv", instrument="AIA"),
        model_observer_meta={
            "observer_name": "stereo-a",
            "observer_lonc_deg": 35.0,
            "observer_b0sun_deg": 2.0,
            "observer_dsun_cm": 1.4e13,
        },
        saved_fov=_saved_fov(),
        geometry_overrides_requested=False,
        explicit_observer_requested=False,
    )

    assert decision.observation_observer == "earth"
    assert decision.model_observer == "stereo-a"
    assert decision.los_aligned is False
    assert decision.use_model_saved_fov is False
    assert decision.geometry_mode == "observation_inscribed_fov"
    assert decision.observer_lonc_deg == 0.0
    assert decision.observer_b0sun_deg == 0.0


def test_explicit_geometry_override_is_honored_but_observation_los_still_recorded() -> None:
    decision = resolve_geometry_policy(
        obs_map=_obs(domain="mw", instrument="EOVSA"),
        model_observer_meta={"observer_name": "stereo-b"},
        saved_fov=_saved_fov(),
        geometry_overrides_requested=True,
        explicit_observer_requested=False,
    )

    assert decision.geometry_mode == "explicit"
    assert decision.use_model_saved_fov is False
    assert decision.observation_observer == "earth"
    assert decision.model_observer == "stereo-b"


def test_saved_fov_delegates_observer_restoration_to_renderer() -> None:
    decision = resolve_geometry_policy(
        obs_map=_obs(domain="euv", instrument="AIA"),
        model_observer_meta={"observer_name": "earth", "observer_lonc_deg": 0.0},
        saved_fov=_saved_fov(),
        geometry_overrides_requested=False,
        explicit_observer_requested=False,
    )

    assert resolve_renderer_observer_name(
        decision,
        explicit_observer_name=None,
        explicit_observer_requested=False,
    ) is None


def test_observation_driven_render_passes_named_observer_not_ephemeris_longitude() -> None:
    decision = resolve_geometry_policy(
        obs_map=_obs(domain="euv", instrument="AIA"),
        model_observer_meta={"observer_name": "stereo-a", "observer_lonc_deg": 35.0},
        saved_fov=_saved_fov(),
        geometry_overrides_requested=False,
        explicit_observer_requested=False,
    )

    assert decision.observer_lonc_deg == 0.0
    assert resolve_renderer_observer_name(
        decision,
        explicit_observer_name=None,
        explicit_observer_requested=False,
    ) == "earth"


def test_explicit_named_observer_is_forwarded_as_identity() -> None:
    decision = resolve_geometry_policy(
        obs_map=_obs(domain="euv", instrument="AIA"),
        model_observer_meta={"observer_name": "earth"},
        saved_fov=_saved_fov(),
        geometry_overrides_requested=False,
        explicit_observer_requested=True,
    )

    assert resolve_renderer_observer_name(
        decision,
        explicit_observer_name="stereo-a",
        explicit_observer_requested=True,
    ) == "stereo-a"


def test_named_observer_does_not_become_scalar_renderer_override() -> None:
    sdk = SimpleNamespace(
        ObserverOverrides=lambda **values: SimpleNamespace(**values),
    )

    overrides, source = fit_q0_obs_map._resolve_observer_overrides(
        sdk,
        model_path=Path("unused.h5"),
        observer_name="earth",
        dsun_cm=None,
        lonc_deg=None,
        b0sun_deg=None,
    )

    assert overrides is None
    assert source == "observer_name:earth"


def test_explicit_renderer_longitude_remains_available() -> None:
    sdk = SimpleNamespace(
        ObserverOverrides=lambda **values: SimpleNamespace(**values),
    )

    overrides, _ = fit_q0_obs_map._resolve_observer_overrides(
        sdk,
        model_path=Path("unused.h5"),
        observer_name="earth",
        dsun_cm=None,
        lonc_deg=-5.25,
        b0sun_deg=None,
    )

    assert overrides is not None
    assert overrides.lonc_deg == -5.25
