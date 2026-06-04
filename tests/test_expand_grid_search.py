"""Tests for --expand-grid-search-id CLI contract."""

from __future__ import annotations

import sys

import pytest

from pychmp.ab_scan_artifacts import (
    assert_expand_grid_search_cli_argv_allowed,
    parse_expand_grid_bounds_from_argv,
    validate_expanded_ab_bounds,
)


def test_expand_grid_search_rejects_metric_override() -> None:
    argv = [
        "prog",
        "--artifact-h5",
        "a.h5",
        "--expand-grid-search-id",
        "search_x",
        "--a-max",
        "4",
        "--target-metric",
        "eta2",
    ]
    with pytest.raises(SystemExit, match="widened"):
        assert_expand_grid_search_cli_argv_allowed(argv)


def test_expand_grid_search_allows_bound_flags_only() -> None:
    argv = ["prog", "--artifact-h5", "a.h5", "--expand-grid-search-id", "search_x", "--a-min", "-1.5"]
    assert_expand_grid_search_cli_argv_allowed(argv)


def test_parse_expand_grid_bounds_from_argv() -> None:
    argv = ["prog", "--a-min", "-1", "--b-max=4.5"]
    assert parse_expand_grid_bounds_from_argv(argv) == {"a_min": -1.0, "b_max": 4.5}


def test_validate_expanded_ab_bounds_requires_superset() -> None:
    with pytest.raises(SystemExit, match="shrink"):
        validate_expanded_ab_bounds(
            stored_a_range=(-1.0, 3.0),
            stored_b_range=(0.0, 4.0),
            new_a_range=(-0.5, 3.0),
            new_b_range=(0.0, 4.0),
        )
    with pytest.raises(SystemExit, match="widen"):
        validate_expanded_ab_bounds(
            stored_a_range=(-1.0, 3.0),
            stored_b_range=(0.0, 4.0),
            new_a_range=(-1.0, 3.0),
            new_b_range=(0.0, 4.0),
        )
    validate_expanded_ab_bounds(
        stored_a_range=(-1.0, 3.0),
        stored_b_range=(0.0, 4.0),
        new_a_range=(-1.5, 3.0),
        new_b_range=(0.0, 4.0),
    )
