"""Shared CLI helpers for CHMP observation alignment and Q0 search options."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from .obs_alignment import DEFAULT_MAX_SHIFT_ARSEC
from .q0_search import parse_q0_search_stages, resolve_q0_search_stages


def parse_xy_shift(value: str | None) -> tuple[float, float] | None:
    """Parse ``dx,dy`` arcsec shift values from CLI input."""
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("--xy-shift requires two comma-separated arcsec values: dx,dy")
    return float(parts[0]), float(parts[1])


@dataclass(frozen=True)
class ChmpSearchSettings:
    """Resolved CHMP evaluation settings from CLI arguments."""

    shift_policy: str
    max_shift_arcsec: float | None
    xy_shift_arcsec: tuple[float, float]
    use_smoothed_obs_max: bool
    use_emthreshold: bool
    emthreshold: float
    q0_search_stages: tuple[str, ...]


def resolve_shift_policy_from_args(args: argparse.Namespace) -> tuple[str, float | None, tuple[float, float]]:
    """Resolve shift policy and padding from parsed CLI arguments."""
    xy_shift = parse_xy_shift(getattr(args, "xy_shift_arcsec", None))
    if xy_shift is not None:
        return "fixed", None, xy_shift

    policy = str(getattr(args, "shift_policy", "auto")).strip().lower()
    if policy not in {"auto", "fixed"}:
        raise ValueError("shift_policy must be 'auto' or 'fixed'")
    max_shift = getattr(args, "max_shift_arcsec", None)
    if policy == "auto":
        return policy, None if max_shift is None else float(max_shift), (0.0, 0.0)
    return policy, None, (0.0, 0.0)


def resolve_chmp_search_settings(
    args: argparse.Namespace,
    *,
    mask_type: str = "union",
    explicit_mask: object | None = None,
) -> ChmpSearchSettings:
    """Resolve CHMP search settings from argparse namespace values."""
    shift_policy, max_shift_arcsec, xy_shift_arcsec = resolve_shift_policy_from_args(args)
    parsed_stages = parse_q0_search_stages(getattr(args, "q0_search_stages", None))
    q0_search_stages = resolve_q0_search_stages(
        q0_search_stages=parsed_stages,
        mask_type=mask_type,
        explicit_mask=explicit_mask,
    )
    return ChmpSearchSettings(
        shift_policy=shift_policy,
        max_shift_arcsec=max_shift_arcsec,
        xy_shift_arcsec=xy_shift_arcsec,
        use_smoothed_obs_max=bool(getattr(args, "use_smoothed_obs_max", True)),
        use_emthreshold=bool(getattr(args, "use_emthreshold", True)),
        emthreshold=float(getattr(args, "emthreshold", 0.1)),
        q0_search_stages=q0_search_stages,
    )


def add_chmp_search_cli_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_q0_stages: bool = True,
) -> argparse._ArgumentGroup:
    """Register shared CHMP observation-alignment and evaluation flags."""
    group = parser.add_argument_group(
        "CHMP observation alignment and evaluation",
        "Defaults follow CHMP reference behavior with opt-in two-stage mask search.",
    )
    group.add_argument(
        "--shift-policy",
        choices=("auto", "fixed"),
        default="auto",
        help="Per-trial observation shift policy. auto runs FindShift on the padded canvas.",
    )
    group.add_argument(
        "--max-shift-arcsec",
        type=float,
        default=None,
        help=f"Maximum FindShift search radius in arcsec (auto mode; default {DEFAULT_MAX_SHIFT_ARSEC:g}).",
    )
    group.add_argument(
        "--xy-shift",
        dest="xy_shift_arcsec",
        default=None,
        metavar="DX,DY",
        help="Fixed observation shift in arcsec. Implies --shift-policy fixed.",
    )
    group.add_argument(
        "--plain-obs-max",
        dest="use_smoothed_obs_max",
        action="store_false",
        help="Use the raw observed maximum instead of the CHMP smoothed maximum for mask thresholds.",
    )
    group.add_argument(
        "--no-smoothed-obs-max",
        dest="use_smoothed_obs_max",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    group.add_argument(
        "--no-emthreshold-gate",
        dest="use_emthreshold",
        action="store_false",
        help="Disable the EBTEL miss-ratio invalidation gate.",
    )
    group.add_argument(
        "--emthreshold",
        type=float,
        default=0.1,
        help="Maximum allowed EBTEL miss ratio before a Q0 trial is invalidated.",
    )
    if include_q0_stages:
        group.add_argument(
            "--q0-search-stages",
            default=None,
            help=(
                "Comma-separated mask stages for Q0 optimization. "
                "Default is a single stage using the active mask type (union for CHMP). "
                "Use data,union for the opt-in two-stage search."
            ),
        )
    return group
