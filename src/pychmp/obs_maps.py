"""Shared observational-map loading helpers.

This module provides a package-owned observation-ingestion surface so the
example workflows do not need to keep their FITS-loading logic in script-local
helpers. The initial implementation focuses on external single-map FITS inputs
used by the current MW workflows, while shaping the API for future internal
pyAMPP-refmap loading and EUV support.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any

import h5py
import numpy as np
from astropy.io import fits

from .fits_utils import extract_frequency_ghz, load_2d_fits_image
from .map_noise import MapNoiseEstimate, estimate_map_noise


_MODEL_H5_CACHE: dict[str, Path] = {}
DEFAULT_TESTDATA_EOVSA_FITS = "eovsa.synoptic_daily.20201126T200000Z.f2.874GHz.tb.disk.fits"
DEFAULT_TESTDATA_MODEL_H5 = "hmi.M_720s.20201126_195831.E18S19CR.CEA.NAS.GEN.CHR.h5"
DEFAULT_TESTDATA_EBTEL = "ebtel.sav"


@dataclass(slots=True)
class ObservationalMap:
    """Normalized observational map payload used by downstream workflows."""

    data: np.ndarray
    header: fits.Header
    domain: str
    instrument: str | None
    observer: str | None
    spectral_label: str | None
    frequency_ghz: float | None
    wavelength_angstrom: float | None
    date_obs: str | None
    source_mode: str
    source_path: str | None
    source_map_id: str | None
    psf_metadata: dict[str, Any] | None
    wcs_metadata: dict[str, Any] | None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def estimate_obs_map_noise(
    obs_map: ObservationalMap,
    *,
    method: str = "histogram_clip",
) -> MapNoiseEstimate:
    """Estimate observation noise using the current shared MW-policy path.

    At the current development stage, MW and EUV observational maps should use
    the same practical policy:

    1. estimate noise from the observed map with ``estimate_map_noise(...)``
    2. if that fails, fall back to a uniform ``sigma = observed.std()`` map

    This deliberately mirrors the current MW example workflows so the first EUV
    path inherits the same behavior rather than introducing a separate,
    unvalidated noise model.
    """

    data = np.asarray(obs_map.data, dtype=float)
    noise_result = estimate_map_noise(data, method=method)
    if noise_result is not None:
        return noise_result

    sigma = float(np.asarray(data, dtype=float).std())
    sigma_map = np.full_like(data, sigma, dtype=float)
    diagnostics = {
        "method": "fallback_std",
        "fallback_sigma": sigma,
        "source_method": str(method),
        "spectral_domain": str(obs_map.domain),
        "source_mode": str(obs_map.source_mode),
        "source_path": obs_map.source_path,
        "source_map_id": obs_map.source_map_id,
    }
    return MapNoiseEstimate(
        sigma=sigma,
        sigma_map=sigma_map,
        method_used="fallback_std",
        n_pixels_used=int(data.size),
        mask_fraction=1.0,
        diagnostics=diagnostics,
    )


def _normalize_domain(domain: str | None) -> str:
    text = str(domain or "").strip().lower()
    if not text:
        return "generic"
    return text


def _domain_from_wavelength_hint(wavelength_angstrom: float | None) -> str:
    numeric = _optional_float(wavelength_angstrom)
    if numeric is None:
        return "generic"
    if np.isclose(numeric, 1600.0, rtol=0.0, atol=1e-9) or np.isclose(numeric, 1700.0, rtol=0.0, atol=1e-9):
        return "uv"
    return "euv"


def _decode_h5_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_h5_scalar(value.item())
    return str(value)


def _first_header_value(header: fits.Header, keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in header:
            return header[key]
    return None


def _extract_wavelength_angstrom(header: fits.Header) -> float | None:
    value = _first_header_value(header, ("WAVELNTH", "WAVE_LEN", "WAVELENGTH"))
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None

    unit = str(_first_header_value(header, ("WAVEUNIT", "WAVELNTHU", "WAVEUNITA")) or "angstrom").strip().lower()
    if unit in {"angstrom", "angstroms", "a", "aa"}:
        return numeric
    if unit in {"nm", "nanometer", "nanometers"}:
        return numeric * 10.0
    if unit in {"m", "meter", "meters"}:
        return numeric * 1.0e10
    return numeric


def _infer_frequency_ghz(header: fits.Header) -> float | None:
    try:
        return extract_frequency_ghz(header)
    except Exception:
        return None


def _infer_instrument(header: fits.Header, *, instrument: str | None) -> str | None:
    if instrument is not None and str(instrument).strip():
        return str(instrument).strip()
    for key in ("INSTRUME", "TELESCOP", "DETECTOR"):
        value = header.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _infer_observer(header: fits.Header) -> str | None:
    for key in ("OBSERVER", "OBSERVR", "TELESCOP"):
        value = header.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _infer_date_obs(header: fits.Header) -> str | None:
    for key in ("DATE-OBS", "DATE_OBS", "T_OBS"):
        value = header.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _infer_wavelength_from_map_id(map_id: str) -> float | None:
    text = str(map_id).strip()
    if not text:
        return None
    upper = text.upper()
    if upper.startswith("AIA_"):
        suffix = upper.split("_", 1)[1]
        try:
            return float(suffix)
        except Exception:
            return None
    return None


def _infer_instrument_from_map_id(map_id: str, *, instrument: str | None) -> str | None:
    if instrument is not None and str(instrument).strip():
        return str(instrument).strip()
    text = str(map_id).strip()
    if not text or "_" not in text:
        return instrument
    prefix = text.split("_", 1)[0].strip()
    return prefix or instrument


def _infer_domain_from_map_id(
    map_id: str,
    *,
    frequency_ghz: float | None,
    wavelength_angstrom: float | None,
) -> str:
    upper = str(map_id).strip().upper()
    if frequency_ghz is not None:
        return "mw"
    if upper.startswith("AIA_"):
        if wavelength_angstrom is not None and float(wavelength_angstrom) in {1600.0, 1700.0}:
            return "uv"
        return "euv"
    if wavelength_angstrom is not None:
        return "euv"
    return "generic"


def _resolve_model_h5_path(model_path: Path) -> Path:
    resolved_model_path = Path(model_path).expanduser().resolve()
    cache_key = str(resolved_model_path)
    cached = _MODEL_H5_CACHE.get(cache_key)
    if cached is not None and cached.exists():
        return cached

    suffix = resolved_model_path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        if not resolved_model_path.exists():
            raise FileNotFoundError(f"model file not found: {resolved_model_path}")
        _MODEL_H5_CACHE[cache_key] = resolved_model_path
        return resolved_model_path

    if suffix != ".sav":
        raise ValueError(f"unsupported model file format for refmap loading: {resolved_model_path}")
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"model file not found: {resolved_model_path}")

    converters: list[Any] = []
    try:
        from pyampp.util.build_h5_from_sav import build_h5_from_sav as pyampp_build_h5_from_sav

        converters.append(pyampp_build_h5_from_sav)
    except Exception:
        pass
    try:
        from gxrender.io import build_h5_from_sav as gxrender_build_h5_from_sav

        converters.append(gxrender_build_h5_from_sav)
    except Exception:
        pass

    if not converters:
        raise RuntimeError(
            "SAV model-refmap loading requires an available SAV-to-HDF5 converter "
            "(pyampp.util.build_h5_from_sav or gxrender.io.build_h5_from_sav)."
        )

    out_h5 = Path(tempfile.gettempdir()) / f"pychmp_obs_map_{resolved_model_path.stem}.h5"
    if not out_h5.exists():
        last_error: Exception | None = None
        for converter in converters:
            try:
                converter(resolved_model_path, out_h5, template_h5=None)
                break
            except TypeError:
                try:
                    converter(sav_path=resolved_model_path, out_h5=out_h5, template_h5=None)
                    break
                except Exception as exc:
                    last_error = exc
            except Exception as exc:
                last_error = exc
        else:
            raise RuntimeError(f"failed to convert SAV model to HDF5: {resolved_model_path}") from last_error

    _MODEL_H5_CACHE[cache_key] = out_h5
    return out_h5


def _list_model_refmap_ids(model_h5_path: Path) -> list[str]:
    try:
        with h5py.File(model_h5_path, "r") as h5f:
            if "refmaps" not in h5f:
                return []
            return sorted(str(name) for name in h5f["refmaps"].keys())
    except Exception:
        return []


def find_named_testdata_file(parent: str | Path, filename: str) -> Path | None:
    """Find a specific fixture file by filename, independent of folder stamp."""

    root = Path(parent).expanduser().resolve()
    if not root.exists():
        return None
    matches = sorted(path for path in root.rglob(filename) if path.is_file())
    return matches[0] if matches else None


def resolve_default_testdata_fixture_paths(
    *,
    repo_root: str | Path,
    testdata_repo: str | Path | None = None,
) -> tuple[Path | None, Path | None, Path]:
    """Resolve the default 2020-11-26 CHR/EOVSA fixture trio by file identity."""

    resolved_repo_root = Path(repo_root).expanduser().resolve()
    resolved_testdata_repo = (
        Path(testdata_repo).expanduser().resolve()
        if testdata_repo is not None
        else resolved_repo_root.parent / "pyGXrender-test-data"
    )
    eovsa_root = resolved_testdata_repo / "raw" / "eovsa_maps"
    model_root = resolved_testdata_repo / "raw" / "models"
    ebtel_path = resolved_testdata_repo / "raw" / "ebtel" / "ebtel_gxsimulator_euv" / DEFAULT_TESTDATA_EBTEL
    eovsa_fits = find_named_testdata_file(eovsa_root, DEFAULT_TESTDATA_EOVSA_FITS)
    model_h5 = find_named_testdata_file(model_root, DEFAULT_TESTDATA_MODEL_H5)
    return eovsa_fits, model_h5, ebtel_path


def _format_spectral_label(
    *,
    domain: str,
    frequency_ghz: float | None,
    wavelength_angstrom: float | None,
) -> str | None:
    if domain == "mw" and frequency_ghz is not None:
        return f"{float(frequency_ghz):.3f} GHz"
    if domain == "euv" and wavelength_angstrom is not None:
        rounded = round(float(wavelength_angstrom))
        if np.isclose(float(wavelength_angstrom), float(rounded), rtol=0.0, atol=1e-9):
            return f"{int(rounded)} A"
        return f"{float(wavelength_angstrom):.3f} A"
    if frequency_ghz is not None:
        return f"{float(frequency_ghz):.3f} GHz"
    if wavelength_angstrom is not None:
        rounded = round(float(wavelength_angstrom))
        if np.isclose(float(wavelength_angstrom), float(rounded), rtol=0.0, atol=1e-9):
            return f"{int(rounded)} A"
        return f"{float(wavelength_angstrom):.3f} A"
    return None


def validate_obs_map_identity(
    obs_map: ObservationalMap,
    *,
    domain_hint: str | None = None,
    frequency_ghz_hint: float | None = None,
    wavelength_angstrom_hint: float | None = None,
) -> ObservationalMap:
    """Validate and complete normalized observation identity.

    Hints are treated conservatively:
    - if they fill missing metadata, they are accepted
    - if they contradict the selected observation, a ``ValueError`` is raised
    """

    resolved_domain = _normalize_domain(obs_map.domain)
    if not resolved_domain:
        resolved_domain = "generic"

    domain_hint_norm = _normalize_domain(domain_hint)
    frequency_hint = _optional_float(frequency_ghz_hint)
    wavelength_hint = _optional_float(wavelength_angstrom_hint)

    frequency_ghz = _optional_float(obs_map.frequency_ghz)
    wavelength_angstrom = _optional_float(obs_map.wavelength_angstrom)

    if frequency_hint is not None and wavelength_hint is not None:
        raise ValueError("conflicting observation hints: both MW frequency and EUV/UV wavelength were supplied")

    if frequency_hint is not None:
        if resolved_domain in {"euv", "uv"}:
            raise ValueError(
                f"conflicting observation request: selected observation resolves to domain='{resolved_domain}' "
                f"but --obs-frequency-ghz={frequency_hint:.6g} requests MW"
            )
        if wavelength_angstrom is not None:
            raise ValueError(
                f"conflicting observation request: selected observation already resolves to wavelength "
                f"{wavelength_angstrom:.6g} A but --obs-frequency-ghz={frequency_hint:.6g} was also supplied"
            )
        if frequency_ghz is None:
            frequency_ghz = frequency_hint
        elif not np.isclose(float(frequency_ghz), float(frequency_hint), rtol=0.0, atol=1e-9):
            raise ValueError(
                f"conflicting observation request: selected observation resolves to "
                f"{float(frequency_ghz):.6g} GHz but --obs-frequency-ghz={float(frequency_hint):.6g} was supplied"
            )
        if resolved_domain == "generic":
            resolved_domain = "mw"

    if wavelength_hint is not None:
        if resolved_domain == "mw":
            raise ValueError(
                f"conflicting observation request: selected observation resolves to domain='mw' "
                f"but --obs-wavelength-angstrom={wavelength_hint:.6g} requests EUV/UV"
            )
        if frequency_ghz is not None:
            raise ValueError(
                f"conflicting observation request: selected observation already resolves to frequency "
                f"{frequency_ghz:.6g} GHz but --obs-wavelength-angstrom={wavelength_hint:.6g} was also supplied"
            )
        if wavelength_angstrom is None:
            wavelength_angstrom = wavelength_hint
        elif not np.isclose(float(wavelength_angstrom), float(wavelength_hint), rtol=0.0, atol=1e-9):
            raise ValueError(
                f"conflicting observation request: selected observation resolves to "
                f"{float(wavelength_angstrom):.6g} A but --obs-wavelength-angstrom={float(wavelength_hint):.6g} was supplied"
            )
        if resolved_domain == "generic":
            resolved_domain = _domain_from_wavelength_hint(wavelength_angstrom)

    if domain_hint_norm != "generic":
        if resolved_domain != "generic" and resolved_domain != domain_hint_norm:
            raise ValueError(
                f"conflicting observation request: selected observation resolves to domain='{resolved_domain}' "
                f"but --obs-domain={domain_hint_norm} was supplied"
            )
        resolved_domain = domain_hint_norm

    if resolved_domain == "generic":
        if frequency_ghz is not None:
            resolved_domain = "mw"
        elif wavelength_angstrom is not None:
            resolved_domain = _domain_from_wavelength_hint(wavelength_angstrom)

    if resolved_domain == "mw" and frequency_ghz is None:
        raise ValueError("selected MW observation is missing frequency metadata; supply --obs-frequency-ghz")
    if resolved_domain in {"euv", "uv"} and wavelength_angstrom is None:
        raise ValueError(
            f"selected {resolved_domain.upper()} observation is missing wavelength metadata; "
            "supply --obs-wavelength-angstrom"
        )
    if frequency_ghz is not None and wavelength_angstrom is not None:
        raise ValueError(
            "selected observation simultaneously resolves to MW frequency and EUV/UV wavelength; "
            "the observation identity is ambiguous"
        )

    obs_map.domain = resolved_domain
    obs_map.frequency_ghz = frequency_ghz
    obs_map.wavelength_angstrom = wavelength_angstrom
    obs_map.spectral_label = _format_spectral_label(
        domain=resolved_domain,
        frequency_ghz=frequency_ghz,
        wavelength_angstrom=wavelength_angstrom,
    ) or obs_map.spectral_label
    return obs_map


def _load_external_obs_map(
    *,
    obs_path: Path,
    domain: str | None,
    instrument: str | None,
) -> ObservationalMap:
    data_arr, header, hdu_name = load_2d_fits_image(Path(obs_path))
    resolved_domain = _normalize_domain(domain)

    frequency_ghz = extract_frequency_ghz(header) if resolved_domain == "mw" else _infer_frequency_ghz(header)
    wavelength_angstrom = _extract_wavelength_angstrom(header)

    return ObservationalMap(
        data=np.asarray(data_arr, dtype=float),
        header=header.copy(),
        domain=resolved_domain,
        instrument=_infer_instrument(header, instrument=instrument),
        observer=_infer_observer(header),
        spectral_label=_format_spectral_label(
            domain=resolved_domain,
            frequency_ghz=frequency_ghz,
            wavelength_angstrom=wavelength_angstrom,
        ),
        frequency_ghz=frequency_ghz,
        wavelength_angstrom=wavelength_angstrom,
        date_obs=_infer_date_obs(header),
        source_mode="external_fits",
        source_path=str(Path(obs_path).expanduser().resolve()),
        source_map_id=None,
        psf_metadata=None,
        wcs_metadata={"hdu_name": str(hdu_name)},
    )


def _load_internal_obs_map(
    *,
    model_h5: Path,
    map_id: str,
    domain: str | None,
    instrument: str | None,
) -> ObservationalMap:
    resolved_model_h5 = _resolve_model_h5_path(model_h5)
    resolved_map_id = str(map_id).strip()
    if not resolved_map_id:
        raise ValueError("map_id is required when source_mode='model_refmap'")

    available_map_ids = _list_model_refmap_ids(resolved_model_h5)
    group_path = f"refmaps/{resolved_map_id}"
    with h5py.File(resolved_model_h5, "r") as h5f:
        if group_path not in h5f:
            available_text = ", ".join(available_map_ids) if available_map_ids else "(none)"
            raise KeyError(
                f"refmap '{resolved_map_id}' not found in {resolved_model_h5}. "
                f"Available refmaps: {available_text}"
            )
        group = h5f[group_path]
        if "data" not in group or "wcs_header" not in group:
            raise ValueError(
                f"refmap '{resolved_map_id}' in {resolved_model_h5} must contain both 'data' and 'wcs_header'"
            )
        data = np.asarray(group["data"], dtype=float)
        header_text = _decode_h5_scalar(group["wcs_header"][()])

    header = fits.Header.fromstring(header_text, sep="\n")
    frequency_ghz = _infer_frequency_ghz(header)
    wavelength_angstrom = _extract_wavelength_angstrom(header)
    if wavelength_angstrom is None:
        wavelength_angstrom = _infer_wavelength_from_map_id(resolved_map_id)

    resolved_domain = _normalize_domain(domain)
    if resolved_domain == "generic":
        resolved_domain = _infer_domain_from_map_id(
            resolved_map_id,
            frequency_ghz=frequency_ghz,
            wavelength_angstrom=wavelength_angstrom,
        )

    resolved_instrument = _infer_instrument(
        header,
        instrument=_infer_instrument_from_map_id(resolved_map_id, instrument=instrument),
    )

    spectral_label = _format_spectral_label(
        domain=resolved_domain,
        frequency_ghz=frequency_ghz,
        wavelength_angstrom=wavelength_angstrom,
    )
    if spectral_label is None:
        spectral_label = resolved_map_id

    return ObservationalMap(
        data=np.asarray(data, dtype=float),
        header=header.copy(),
        domain=resolved_domain,
        instrument=resolved_instrument,
        observer=_infer_observer(header),
        spectral_label=spectral_label,
        frequency_ghz=frequency_ghz,
        wavelength_angstrom=wavelength_angstrom,
        date_obs=_infer_date_obs(header),
        source_mode="model_refmap",
        source_path=str(resolved_model_h5),
        source_map_id=resolved_map_id,
        psf_metadata=None,
        wcs_metadata={
            "group_path": group_path,
            "available_map_ids": available_map_ids,
            "converted_from_sav": str(model_h5).lower().endswith(".sav"),
        },
    )


def load_obs_map(
    *,
    obs_path: str | Path | None = None,
    model_h5: str | Path | None = None,
    map_id: str | None = None,
    domain: str | None = None,
    instrument: str | None = None,
    source_mode: str | None = None,
) -> ObservationalMap:
    """Load an observational map into a normalized package payload.

    The initial implementation supports external FITS products, which is enough
    to migrate the current MW workflows away from script-local FITS loaders.
    Internal model-refmap loading is intentionally reserved for the next slice.
    """

    explicit_mode = str(source_mode).strip().lower() if source_mode is not None else None
    if explicit_mode is None:
        if obs_path is not None:
            explicit_mode = "external_fits"
        elif model_h5 is not None or map_id is not None:
            explicit_mode = "model_refmap"
        else:
            raise ValueError("load_obs_map requires either obs_path or model_h5/map_id")

    if explicit_mode == "external_fits":
        if obs_path is None:
            raise ValueError("obs_path is required when source_mode='external_fits'")
        return _load_external_obs_map(
            obs_path=Path(obs_path),
            domain=domain,
            instrument=instrument,
        )

    if explicit_mode == "model_refmap":
        if model_h5 is None:
            raise ValueError("model_h5 is required when source_mode='model_refmap'")
        if map_id is None:
            resolved_model_h5 = _resolve_model_h5_path(Path(model_h5))
            available_map_ids = _list_model_refmap_ids(resolved_model_h5)
            available_text = ", ".join(available_map_ids) if available_map_ids else "(none)"
            raise ValueError(
                "map_id is required when source_mode='model_refmap'. "
                f"Available refmaps in {resolved_model_h5}: {available_text}"
            )
        return _load_internal_obs_map(
            model_h5=Path(model_h5),
            map_id=map_id,
            domain=domain,
            instrument=instrument,
        )

    raise ValueError(f"unsupported obs_source/source_mode: {explicit_mode}")
