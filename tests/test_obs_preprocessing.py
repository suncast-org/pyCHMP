from __future__ import annotations

import inspect

import numpy as np
from astropy.io import fits

from pychmp.metrics import compute_metrics
from pychmp.obs_preprocessing import (
    build_slice_observation_identity,
    compute_array_content_sha256,
    format_observation_shift_label,
    format_search_shift_policy_label,
    prepare_observation_for_metrics,
    regrid_observation_to_target_fov,
    resolve_slice_observation_reference,
    resolve_trial_shift_arcsec,
    slice_observation_identity_sha256,
)


def _make_header(*, nx: int, ny: int, crval1: float, crval2: float, cdelt: float = 1.0) -> fits.Header:
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = nx
    header["NAXIS2"] = ny
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CDELT1"] = cdelt
    header["CDELT2"] = cdelt
    header["CRPIX1"] = (float(nx) + 1.0) / 2.0
    header["CRPIX2"] = (float(ny) + 1.0) / 2.0
    header["CRVAL1"] = crval1
    header["CRVAL2"] = crval2
    header["DATE-OBS"] = "2020-11-26T20:00:00"
    return header


def test_regrid_observation_to_target_fov_changes_shape_to_render_grid() -> None:
    source = np.ones((4, 4), dtype=float)
    source_header = _make_header(nx=4, ny=4, crval1=0.0, crval2=0.0)
    target_header = _make_header(nx=6, ny=5, crval1=10.0, crval2=-5.0, cdelt=2.0)

    cropped = regrid_observation_to_target_fov(source, source_header, target_header)

    assert cropped.shape == (5, 6)


def test_prepare_observation_for_metrics_rotates_then_regrids_in_order() -> None:
    observed = np.arange(16, dtype=float).reshape(4, 4)
    sigma = np.full((4, 4), 0.5, dtype=float)
    source_header = _make_header(nx=4, ny=4, crval1=0.0, crval2=0.0)
    target_header = _make_header(nx=3, ny=3, crval1=0.0, crval2=0.0)

    observed_cropped, sigma_cropped, _header, diagnostics = prepare_observation_for_metrics(
        observed,
        source_header,
        target_header,
        model_time_text="2020-11-26T20:00:00",
        observation_time_text="2020-11-26T20:00:00",
        sigma=sigma,
    )

    assert observed_cropped.shape == (3, 3)
    assert sigma_cropped is not None and sigma_cropped.shape == (3, 3)
    assert diagnostics["observation_regridded_to_render_fov"] is True
    assert diagnostics["observation_time_rotation_applied"] is False
    assert diagnostics["observation_time_alignment"] == "exact"


def test_compute_metrics_does_not_rotate_or_regrid() -> None:
    source = inspect.getsource(compute_metrics)
    assert "differential_rotate" not in source
    assert "regrid" not in source
    assert "map_coordinates" not in source


def test_build_slice_observation_identity_pairs_observed_and_sigma() -> None:
    observed = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    sigma = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
    identity = build_slice_observation_identity(
        observation_source_sha256="abc123",
        artifact_geometry_sha256="geom456",
        model_time_text="2020-11-26T20:00:00",
        observation_time_text="2020-11-26T20:00:00",
        observed=observed,
        sigma=sigma,
        slice_canvas_max_shift_arcsec=20.0,
    )
    assert identity["preprocessed_observation_sha256"] == compute_array_content_sha256(observed)
    assert identity["preprocessed_sigma_sha256"] == compute_array_content_sha256(sigma)
    assert identity["slice_observation_identity_sha256"] == slice_observation_identity_sha256(identity)


def test_resolve_slice_observation_reference_restores_stored_maps() -> None:
    source_header = _make_header(nx=4, ny=4, crval1=0.0, crval2=0.0)
    target_header = _make_header(nx=3, ny=3, crval1=0.0, crval2=0.0)
    observed = np.arange(16, dtype=float).reshape(4, 4)
    sigma = np.full((4, 4), 0.25, dtype=float)
    prepared = resolve_slice_observation_reference(
        observed,
        source_header,
        target_header,
        sigma=sigma,
        observation_source_sha256="source-sha",
        artifact_geometry_sha256="geometry-sha",
        model_time_text="2020-11-26T20:00:00",
        observation_time_text="2020-11-26T20:00:00",
        stored_slice_payload=None,
        force_recompute=True,
    )
    stored_payload = {
        "observed": prepared.observed,
        "sigma_map": prepared.sigma,
        "wcs_header": target_header,
        "diagnostics": prepared.diagnostics,
    }
    restored = resolve_slice_observation_reference(
        observed + 999.0,
        source_header,
        target_header,
        sigma=None,
        observation_source_sha256="source-sha",
        artifact_geometry_sha256="geometry-sha",
        model_time_text="2020-11-26T20:00:00",
        observation_time_text="2020-11-26T20:00:00",
        stored_slice_payload=stored_payload,
        force_recompute=False,
    )
    assert restored.restored_from_artifact is True
    assert np.array_equal(restored.observed, prepared.observed)
    assert np.array_equal(restored.sigma, prepared.sigma)


def test_resolve_slice_observation_reference_repairs_stale_content_hashes() -> None:
    source_header = _make_header(nx=4, ny=4, crval1=0.0, crval2=0.0)
    target_header = _make_header(nx=3, ny=3, crval1=0.0, crval2=0.0)
    observed = np.arange(16, dtype=float).reshape(4, 4)
    sigma = np.full((4, 4), 0.25, dtype=float)
    prepared = resolve_slice_observation_reference(
        observed,
        source_header,
        target_header,
        sigma=sigma,
        observation_source_sha256="source-sha",
        artifact_geometry_sha256="geometry-sha",
        model_time_text="2020-11-26T20:00:00",
        observation_time_text="2020-11-26T20:00:00",
        stored_slice_payload=None,
        force_recompute=True,
    )
    stale_diagnostics = dict(prepared.diagnostics)
    stale_diagnostics["preprocessed_observation_sha256"] = "deadbeef"
    stale_diagnostics["preprocessed_sigma_sha256"] = "deadbeef"
    stored_payload = {
        "observed": prepared.observed,
        "sigma_map": prepared.sigma,
        "wcs_header": target_header,
        "diagnostics": stale_diagnostics,
    }
    restored = resolve_slice_observation_reference(
        observed + 999.0,
        source_header,
        target_header,
        sigma=None,
        observation_source_sha256="source-sha",
        artifact_geometry_sha256="geometry-sha",
        model_time_text="2020-11-26T20:00:00",
        observation_time_text="2020-11-26T20:00:00",
        stored_slice_payload=stored_payload,
        force_recompute=False,
    )
    assert restored.restored_from_artifact is True
    assert restored.diagnostics.get("observation_content_identity_repaired") is True
    assert np.array_equal(restored.observed, prepared.observed)


def test_format_search_shift_policy_label_auto_and_fixed() -> None:
    assert format_search_shift_policy_label({"shift_policy": "auto", "max_shift_arcsec": 20.0}) == (
        "Shift policy: auto (max 20.0 arcsec)"
    )
    assert format_search_shift_policy_label({"shift_policy": "fixed"}) == "Shift policy: fixed (none)"
    assert format_search_shift_policy_label({"shift_policy": "fixed", "xy_shift_arcsec": [1.5, -2.0]}) == (
        "Shift policy: fixed (+1.50, -2.00) arcsec"
    )


def test_format_observation_shift_label_uses_trial_shifts_for_auto_policy() -> None:
    label = format_observation_shift_label(
        diagnostics={"shift_policy": "auto"},
        trial_index=1,
        fit_shift_x_trials=(0.0, 3.25),
        fit_shift_y_trials=(0.0, -1.75),
        fit_find_shift_valid_trials=(True, True),
    )
    assert label == "shift: (+3.25, -1.75) arcsec"
    assert resolve_trial_shift_arcsec(
        diagnostics={"shift_policy": "auto"},
        trial_index=1,
        fit_shift_x_trials=(0.0, 3.25),
        fit_shift_y_trials=(0.0, -1.75),
    ) == (3.25, -1.75)


def test_resolve_trial_shift_arcsec_rejects_non_finite_auto_shifts() -> None:
    assert resolve_trial_shift_arcsec(
        diagnostics={"shift_policy": "auto"},
        trial_index=0,
        fit_shift_x_trials=(float("nan"),),
        fit_shift_y_trials=(float("nan"),),
    ) is None


def test_resolve_trial_observation_for_display_ignores_non_finite_auto_shifts() -> None:
    from pychmp.obs_preprocessing import resolve_trial_observation_for_display

    model_header = _make_header(nx=4, ny=4, crval1=100.0, crval2=200.0)
    canvas_header = _make_header(nx=8, ny=8, crval1=96.0, crval2=196.0)
    canvas = np.ones((8, 8), dtype=float)
    observed = np.ones((4, 4), dtype=float)
    display_observed, _sigma = resolve_trial_observation_for_display(
        observed=observed,
        sigma=None,
        model_header=model_header,
        diagnostics={"shift_policy": "auto"},
        observation_canvas=canvas,
        canvas_header=canvas_header,
        trial_index=0,
        fit_shift_x_trials=(float("nan"),),
        fit_shift_y_trials=(float("nan"),),
    )
    assert display_observed.shape == (4, 4)


def test_viewer_record_metrics_uses_target_metric_trials() -> None:
    from pychmp.ab_scan_artifacts import viewer_record_metrics

    metrics = viewer_record_metrics(
        {
            "target_metric": "eta2",
            "fit_q0_trials": (1e-5, 1e-4),
            "fit_metric_trials": (0.9, 0.4),
            "fit_chi2_trials": (float("nan"), float("nan")),
            "fit_eta2_trials": (0.9, 0.4),
        }
    )
    assert abs(float(metrics["eta2"]) - 0.4) < 1e-12
    assert not np.isfinite(metrics["chi2"])
