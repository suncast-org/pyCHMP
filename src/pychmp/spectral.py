from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


AIA_EUV_CHANNELS_ANGSTROM: tuple[str, ...] = ("94", "131", "171", "193", "211", "304", "335")
EUVI_CHANNELS_ANGSTROM: tuple[str, ...] = ("171", "195", "284", "304")
SOLO_EUI_CHANNELS_ANGSTROM: tuple[str, ...] = ("174",)


@dataclass(frozen=True, slots=True)
class RenderSliceRequest:
    key: str
    domain: str
    label: str
    display_label: str
    frequency_ghz: float | None = None
    wavelength_angstrom: float | None = None
    channel_label: str | None = None
    is_target: bool = False

    def as_descriptor(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "domain": self.domain,
            "label": self.label,
            "display_label": self.display_label,
            "frequency_ghz": self.frequency_ghz,
            "wavelength_angstrom": self.wavelength_angstrom,
            "channel_label": self.channel_label,
            "sort_value": self.frequency_ghz if self.frequency_ghz is not None else self.wavelength_angstrom,
            "role": "target" if self.is_target else "auxiliary",
            "is_target": bool(self.is_target),
        }


def _sanitize_token(value: str) -> str:
    text = str(value).strip().lower().replace(" ", "_")
    chars = []
    for char in text:
        if char.isalnum() or char in {"_", "-"}:
            chars.append(char)
        elif char == ".":
            chars.append("p")
    return "".join(chars).strip("_") or "slice"


def normalize_euv_channel(value: str | float | int) -> str:
    numeric = float(value)
    rounded = round(numeric)
    if np.isclose(numeric, float(rounded), rtol=0.0, atol=1e-9):
        return str(int(rounded))
    return f"{numeric:.6g}"


def parse_csv_floats(value: str | None, *, option_name: str) -> tuple[float, ...]:
    if value is None or not str(value).strip():
        return tuple()
    parsed: list[float] = []
    for raw in str(value).split(","):
        token = raw.strip()
        if not token:
            continue
        try:
            parsed.append(float(token))
        except ValueError as exc:
            raise ValueError(f"{option_name} must be a comma-separated list of numbers; got {token!r}") from exc
    return tuple(parsed)


def parse_csv_tokens(value: str | None, *, option_name: str) -> tuple[str, ...]:
    if value is None or not str(value).strip():
        return tuple()
    tokens = tuple(token.strip() for token in str(value).split(",") if token.strip())
    if not tokens:
        raise ValueError(f"{option_name} must contain at least one channel")
    return tokens


def default_euv_channels_for_instrument(instrument: str | None) -> tuple[str, ...]:
    token = str(instrument or "").strip().upper().replace("-", "_")
    if token == "AIA" or token == "SDO_AIA":
        return AIA_EUV_CHANNELS_ANGSTROM
    if token in {"EUVI", "STEREO", "STEREO_A", "STEREO_B", "SECCHI_EUVI"}:
        return EUVI_CHANNELS_ANGSTROM
    if token in {"SOLO", "SOLAR_ORBITER", "EUI", "SOLO_EUI"}:
        return SOLO_EUI_CHANNELS_ANGSTROM
    return tuple()


def unique_preserve_order(values: Iterable[Any]) -> tuple[Any, ...]:
    out: list[Any] = []
    for value in values:
        if value is None:
            continue
        if not any(str(existing) == str(value) for existing in out):
            out.append(value)
    return tuple(out)


def mw_slice_request(frequency_ghz: float, *, is_target: bool) -> RenderSliceRequest:
    label = f"{float(frequency_ghz):.3f} GHz"
    return RenderSliceRequest(
        key=f"mw_{float(frequency_ghz):.6f}ghz".replace(".", "p"),
        domain="mw",
        label=label,
        display_label=f"MW: {label}",
        frequency_ghz=float(frequency_ghz),
        is_target=bool(is_target),
    )


def euv_slice_request(channel: str | float | int, *, domain: str, is_target: bool) -> RenderSliceRequest:
    normalized = normalize_euv_channel(channel)
    wavelength = float(normalized)
    label = f"{normalized} A"
    resolved_domain = str(domain or "euv").strip().lower()
    return RenderSliceRequest(
        key=f"{resolved_domain}_{_sanitize_token(normalized)}",
        domain=resolved_domain,
        label=label,
        display_label=f"{resolved_domain.upper()}: {label}",
        wavelength_angstrom=wavelength,
        channel_label=normalized,
        is_target=bool(is_target),
    )

