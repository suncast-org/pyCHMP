from __future__ import annotations

from pychmp import assess_obs_model_time_alignment
from pychmp.obs_time_alignment import _astropy_time_from_text, normalize_observation_time_unix


def test_astropy_time_from_text_parses_fits_style_model_time() -> None:
    text = "26-Nov-2020 19:58:30.620"
    parsed = _astropy_time_from_text(text)
    expected_unix = normalize_observation_time_unix(text)
    assert expected_unix is not None
    assert abs(float(parsed.unix) - float(expected_unix)) < 0.5


def test_assess_obs_model_time_alignment_marks_same_day_shift_rotatable() -> None:
    alignment = assess_obs_model_time_alignment(
        "2020-11-26T20:00:00",
        "26-Nov-2020 19:58:30.620",
    )
    assert alignment.compatibility == "rotatable"
    assert alignment.same_utc_day is True
    assert alignment.delta_seconds is not None
    assert abs(abs(float(alignment.delta_seconds)) - 89.38) < 0.5
    assert alignment.should_rotate() is True


def test_assess_obs_model_time_alignment_warns_on_different_days() -> None:
    alignment = assess_obs_model_time_alignment(
        "2020-11-26T20:00:00",
        "2020-11-27T20:00:00",
    )
    assert alignment.compatibility == "incompatible"
    assert alignment.same_utc_day is False
    assert alignment.should_rotate() is False
    assert alignment.warning_lines()


def test_assess_obs_model_time_alignment_exact_match() -> None:
    alignment = assess_obs_model_time_alignment(
        "2020-11-26T20:00:00",
        "2020-11-26T20:00:00",
    )
    assert alignment.compatibility == "exact"
    assert alignment.should_rotate() is False
    assert alignment.warning_lines() == []
