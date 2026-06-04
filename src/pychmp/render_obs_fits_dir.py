"""Discover auxiliary render channels/frequencies from observational FITS products."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from .fits_utils import extract_frequency_ghz, load_2d_fits_image
from .geometry_policy import infer_observation_observer
from .obs_maps import (
    ObserverGeometrySnapshot,
    _extract_wavelength_angstrom,
    extract_observer_geometry_from_header,
    infer_spectral_domain_from_header,
    normalize_render_instrument,
    spectral_domains_compatible,
)
from .spectral import normalize_euv_channel, unique_preserve_order

_FITS_SUFFIXES = {".fits", ".fit", ".fts", ".fz"}


@dataclass(frozen=True)
class RenderObsTargetContext:
    """Target observation identity used to filter auxiliary render FITS products."""

    domain: str
    instrument: str | None = None
    observer_los: str | None = None
    lonc_deg: float | None = None
    b0sun_deg: float | None = None
    dsun_cm: float | None = None
    lonc_atol_deg: float = 0.05
    b0sun_atol_deg: float = 0.05
    dsun_rtol: float = 0.01


@dataclass(frozen=True)
class CompatibleRenderObsFits:
    path: Path
    domain: str
    frequency_ghz: float | None = None
    channel: str | None = None


@dataclass(frozen=True)
class RenderObsFitsScan:
    directory: Path
    target_domain: str
    compatible: tuple[CompatibleRenderObsFits, ...]
    skipped_incompatible_domain: tuple[str, ...]
    skipped_incompatible_render_context: tuple[str, ...]
    skipped_unreadable: tuple[str, ...]
    candidates_scanned: int

    @property
    def skipped_incompatible(self) -> tuple[str, ...]:
        return self.skipped_incompatible_domain + self.skipped_incompatible_render_context


def build_render_obs_target_context(
    obs_map: Any,
    *,
    domain: str,
    euv_instrument: str | None = None,
) -> RenderObsTargetContext:
    """Build the render filter identity from the fitted target observation."""
    resolved_domain = _normalize_target_domain(domain)
    observer_los = infer_observation_observer(obs_map)
    header = getattr(obs_map, "header", None)
    geom = (
        extract_observer_geometry_from_header(header, domain=resolved_domain)
        if isinstance(header, fits.Header)
        else ObserverGeometrySnapshot(
            observer_los=observer_los,
            instrument=None,
            lonc_deg=None,
            b0sun_deg=None,
            dsun_cm=None,
        )
    )
    if resolved_domain == "mw":
        return RenderObsTargetContext(
            domain="mw",
            instrument=normalize_render_instrument(getattr(obs_map, "instrument", None))
            or geom.instrument,
            observer_los=observer_los or geom.observer_los,
            lonc_deg=geom.lonc_deg,
            b0sun_deg=geom.b0sun_deg,
            dsun_cm=geom.dsun_cm,
        )
    instrument = normalize_render_instrument(
        euv_instrument or getattr(obs_map, "instrument", None) or geom.instrument
    )
    return RenderObsTargetContext(
        domain=resolved_domain,
        instrument=instrument,
        observer_los=observer_los or geom.observer_los,
        lonc_deg=geom.lonc_deg,
        b0sun_deg=geom.b0sun_deg,
        dsun_cm=geom.dsun_cm,
    )


def _normalize_fits_path(path: Path) -> Path:
    try:
        return path.expanduser().resolve()
    except Exception:
        return path.expanduser()


def _normalize_target_domain(domain: str) -> str:
    text = str(domain or "").strip().lower()
    if text not in {"mw", "euv", "uv"}:
        raise ValueError(
            f"render observation FITS directory requires a resolved target domain "
            f"(mw, euv, or uv); got {domain!r}"
        )
    return text


def list_obs_fits_files(directory: Path) -> tuple[Path, ...]:
    """Return sorted FITS files in *directory* (non-recursive)."""
    root = _normalize_fits_path(directory)
    if not root.exists():
        raise ValueError(f"render observation FITS directory not found: {root}")
    if not root.is_dir():
        raise ValueError(f"render observation FITS path is not a directory: {root}")
    files = [
        item
        for item in root.iterdir()
        if item.is_file() and item.suffix.lower() in _FITS_SUFFIXES
    ]
    if not files:
        raise ValueError(f"no FITS files found in render observation directory: {root}")
    return tuple(sorted(files, key=lambda item: item.name.lower()))


def _header_for_render_discovery(fits_path: Path) -> fits.Header:
    resolved = _normalize_fits_path(fits_path)
    if not resolved.is_file():
        raise ValueError(f"render observation FITS file not found: {resolved}")
    _data, header, _hdu_name = load_2d_fits_image(resolved)
    return header


def infer_spectral_domain_from_obs_fits(fits_path: Path) -> str:
    """Infer mw/euv/uv spectral domain for one observational FITS product."""
    return infer_spectral_domain_from_header(_header_for_render_discovery(fits_path))


def _target_identity_token_matches(
    target_value: str | None,
    file_value: str | None,
) -> bool:
    if target_value is None:
        return True
    return file_value == target_value


def _observer_geometry_numeric_compatible(
    target: RenderObsTargetContext,
    file_geom: ObserverGeometrySnapshot,
) -> bool:
    field_specs = (
        ("lonc_deg", float(target.lonc_atol_deg), 0.0),
        ("b0sun_deg", float(target.b0sun_atol_deg), 0.0),
        ("dsun_cm", 0.0, float(target.dsun_rtol)),
    )
    for attr, atol, rtol in field_specs:
        target_value = getattr(target, attr)
        file_value = getattr(file_geom, attr)
        if target_value is None or not np.isfinite(float(target_value)):
            continue
        if file_value is None or not np.isfinite(float(file_value)):
            return False
        tolerance = max(float(atol), abs(float(target_value)) * float(rtol))
        if abs(float(file_value) - float(target_value)) > tolerance:
            return False
    return True


def render_obs_context_compatible(
    header: fits.Header,
    *,
    target: RenderObsTargetContext,
) -> bool:
    """Return whether a FITS product matches the full target observation render context."""
    file_domain = infer_spectral_domain_from_header(header)
    if not spectral_domains_compatible(file_domain, target.domain):
        return False
    file_geom = extract_observer_geometry_from_header(header, domain=file_domain)
    if not _target_identity_token_matches(target.instrument, file_geom.instrument):
        return False
    if not _target_identity_token_matches(target.observer_los, file_geom.observer_los):
        return False
    return _observer_geometry_numeric_compatible(target, file_geom)


def euv_render_context_compatible(
    header: fits.Header,
    *,
    target: RenderObsTargetContext,
) -> bool:
    """Backward-compatible alias for :func:`render_obs_context_compatible`."""
    return render_obs_context_compatible(header, target=target)


def scan_render_obs_fits_directory(
    directory: Path,
    *,
    target_domain: str,
    exclude_paths: tuple[Path, ...] = (),
    target_context: RenderObsTargetContext | None = None,
) -> RenderObsFitsScan:
    """Classify FITS files, keeping entries compatible with the target spectral/render context."""
    expected_domain = _normalize_target_domain(target_domain)
    context = target_context or RenderObsTargetContext(domain=expected_domain)
    root = _normalize_fits_path(directory)
    excluded = {_normalize_fits_path(path) for path in exclude_paths}
    compatible: list[CompatibleRenderObsFits] = []
    skipped_incompatible_domain: list[str] = []
    skipped_incompatible_render_context: list[str] = []
    skipped_unreadable: list[str] = []
    candidates_scanned = 0
    for fits_path in list_obs_fits_files(root):
        resolved = _normalize_fits_path(fits_path)
        if resolved in excluded:
            continue
        candidates_scanned += 1
        try:
            header = _header_for_render_discovery(resolved)
            file_domain = infer_spectral_domain_from_header(header)
        except ValueError:
            skipped_unreadable.append(resolved.name)
            continue
        if not spectral_domains_compatible(file_domain, expected_domain):
            skipped_incompatible_domain.append(resolved.name)
            continue
        if not render_obs_context_compatible(header, target=context):
            skipped_incompatible_render_context.append(resolved.name)
            continue
        if file_domain == "mw":
            try:
                frequency_ghz = float(extract_frequency_ghz(header))
            except ValueError:
                skipped_unreadable.append(resolved.name)
                continue
            compatible.append(
                CompatibleRenderObsFits(
                    path=resolved,
                    domain=file_domain,
                    frequency_ghz=frequency_ghz,
                )
            )
            continue
        wavelength_angstrom = _extract_wavelength_angstrom(header)
        if wavelength_angstrom is None:
            skipped_unreadable.append(resolved.name)
            continue
        compatible.append(
            CompatibleRenderObsFits(
                path=resolved,
                domain=file_domain,
                channel=normalize_euv_channel(wavelength_angstrom),
            )
        )
    return RenderObsFitsScan(
        directory=root,
        target_domain=expected_domain,
        compatible=tuple(compatible),
        skipped_incompatible_domain=tuple(skipped_incompatible_domain),
        skipped_incompatible_render_context=tuple(skipped_incompatible_render_context),
        skipped_unreadable=tuple(skipped_unreadable),
        candidates_scanned=int(candidates_scanned),
    )


def _empty_auxiliary_render_error(
    *,
    scan: RenderObsFitsScan,
    kind_label: str,
    exclude_note: str,
) -> ValueError:
    return ValueError(
        "render observation FITS directory contains no auxiliary "
        f"{kind_label} compatible with target domain {scan.target_domain!r} "
        f"({scan.directory}); scanned {scan.candidates_scanned} candidate file(s), "
        f"skipped {len(scan.skipped_incompatible_domain)} other-domain file(s), "
        f"{len(scan.skipped_incompatible_render_context)} instrument/LOS mismatch(es), and "
        f"{len(scan.skipped_unreadable)} unreadable file(s)"
        f"{exclude_note}. Reconsider --render-obs-fits-dir or add matching FITS maps."
    )


def frequency_ghz_from_obs_fits(
    fits_path: Path,
    *,
    target_domain: str = "mw",
) -> float:
    """Read observing frequency in GHz from a compatible radio FITS product."""
    header = _header_for_render_discovery(fits_path)
    if infer_spectral_domain_from_header(header) != _normalize_target_domain(target_domain):
        raise ValueError(f"FITS file is not a compatible MW render observation: {fits_path}")
    return float(extract_frequency_ghz(header))


def euv_channel_from_obs_fits(
    fits_path: Path,
    *,
    target_context: RenderObsTargetContext,
) -> str:
    """Read EUV/UV channel label from a FITS product compatible with *target_context*."""
    header = _header_for_render_discovery(fits_path)
    if not render_obs_context_compatible(header, target=target_context):
        raise ValueError(f"FITS file is not a compatible EUV/UV render observation: {fits_path}")
    wavelength_angstrom = _extract_wavelength_angstrom(header)
    if wavelength_angstrom is None:
        raise ValueError(
            f"could not extract wavelength/channel metadata from FITS header: {fits_path}"
        )
    return normalize_euv_channel(wavelength_angstrom)


def discover_render_frequencies_ghz_from_dir(
    directory: Path,
    *,
    target_domain: str,
    exclude_paths: tuple[Path, ...] = (),
    exclude_frequency_ghz: float | None = None,
    frequency_atol_ghz: float = 1e-9,
    scan: RenderObsFitsScan | None = None,
    target_context: RenderObsTargetContext | None = None,
) -> tuple[float, ...]:
    """Collect unique MW frequencies (GHz) from compatible FITS headers in *directory*."""
    expected_domain = _normalize_target_domain(target_domain)
    if expected_domain != "mw":
        raise ValueError(
            f"cannot discover MW render frequencies for target domain {expected_domain!r}"
        )
    context = target_context or RenderObsTargetContext(domain=expected_domain)
    scan = scan or scan_render_obs_fits_directory(
        directory,
        target_domain=expected_domain,
        exclude_paths=exclude_paths,
        target_context=context,
    )
    discovered: list[float] = []
    for entry in scan.compatible:
        if entry.frequency_ghz is None:
            continue
        frequency_ghz = float(entry.frequency_ghz)
        if exclude_frequency_ghz is not None and abs(
            float(frequency_ghz) - float(exclude_frequency_ghz)
        ) <= float(frequency_atol_ghz):
            continue
        discovered.append(frequency_ghz)
    if not discovered:
        exclude_note = ""
        if exclude_frequency_ghz is not None:
            exclude_note = f" after excluding the target frequency ({float(exclude_frequency_ghz):g} GHz)"
        raise _empty_auxiliary_render_error(
            scan=scan,
            kind_label="MW frequencies",
            exclude_note=exclude_note,
        )
    return tuple(float(value) for value in unique_preserve_order(discovered))


def discover_render_channels_from_dir(
    directory: Path,
    *,
    target_domain: str,
    exclude_paths: tuple[Path, ...] = (),
    exclude_channel: str | None = None,
    scan: RenderObsFitsScan | None = None,
    target_context: RenderObsTargetContext | None = None,
) -> tuple[str, ...]:
    """Collect unique EUV/UV channel labels from compatible FITS headers in *directory*."""
    expected_domain = _normalize_target_domain(target_domain)
    if expected_domain not in {"euv", "uv"}:
        raise ValueError(
            f"cannot discover EUV/UV render channels for target domain {expected_domain!r}"
        )
    context = target_context or RenderObsTargetContext(domain=expected_domain)
    scan = scan or scan_render_obs_fits_directory(
        directory,
        target_domain=expected_domain,
        exclude_paths=exclude_paths,
        target_context=context,
    )
    excluded_channel = (
        None if exclude_channel is None else normalize_euv_channel(exclude_channel)
    )
    discovered: list[str] = []
    for entry in scan.compatible:
        if entry.channel is None:
            continue
        channel = str(entry.channel)
        if excluded_channel is not None and channel == str(excluded_channel):
            continue
        discovered.append(channel)
    if not discovered:
        exclude_note = ""
        if excluded_channel is not None:
            exclude_note = f" after excluding the target channel ({excluded_channel})"
        raise _empty_auxiliary_render_error(
            scan=scan,
            kind_label="EUV/UV channels",
            exclude_note=exclude_note,
        )
    return tuple(str(value) for value in unique_preserve_order(discovered))
