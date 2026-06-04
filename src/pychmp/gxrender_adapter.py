"""Adapters for using gxrender as a concrete pyCHMP forward model backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
from importlib import import_module
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any, Sequence
import warnings

import numpy as np


_EUV_PROJECTION_FLAGS_WARNING = (
    "Current Python EUV workflow uses projection flags off "
    "(parallel=False, exact=False, nthreads=0) for the DLL simbox path."
)
EUV_RESPONSE_IDENTITY_VERSION = "pychmp.euv_response_identity.v1"
FORWARD_MODEL_IDENTITY_VERSION = "pychmp.forward_model.file_sha256.v0"
_euv_projection_flags_warning_emitted = False
_euv_projection_flags_warning_lock = threading.Lock()


@dataclass(frozen=True, slots=True)
class EUVResponseIdentity:
    sha256: str
    version: str
    payload: dict[str, Any]
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ForwardModelIdentity:
    sha256: str
    version: str
    source_path: str
    summary: dict[str, Any]


@dataclass(slots=True)
class _CachedEUVResponse:
    response: Any
    response_dt: Any
    response_meta: Any
    response_identity: EUVResponseIdentity | None = None


def _canonical_float64_array(values: Any) -> list[Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        return [None if not np.isfinite(arr.item()) else float(arr.item())]
    finite = np.isfinite(arr)
    normalized = arr.astype(np.float64, copy=True)
    if not finite.all():
        normalized[~finite] = np.nan
    return normalized.tolist()


def _normalize_dtype_descriptor(value: Any) -> Any:
    try:
        dtype = np.dtype(value)
    except Exception:
        return str(value)
    if dtype.names:
        return [_normalize_identity_value(item) for item in dtype.descr]
    return str(dtype)


def _normalize_identity_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    if isinstance(value, np.generic):
        return _normalize_identity_value(value.item())
    if isinstance(value, np.dtype):
        return _normalize_dtype_descriptor(value)
    if isinstance(value, np.ndarray):
        if value.dtype.fields:
            return _normalize_identity_value(value.tolist())
        if np.issubdtype(value.dtype, np.number):
            return _canonical_float64_array(value)
        return [_normalize_identity_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_normalize_identity_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_identity_value(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if hasattr(value, "unit") and hasattr(value, "value"):
        return {
            "value": _normalize_identity_value(getattr(value, "value")),
            "unit": str(getattr(value, "unit")),
        }
    if is_dataclass(value):
        return _normalize_identity_value(asdict(value))
    if hasattr(value, "__dict__"):
        return _normalize_identity_value(vars(value))
    return str(value)


def _normalize_euv_response_meta_for_identity(response_meta: Any) -> dict[str, Any]:
    return {
        "instrument": str(getattr(response_meta, "instrument", "") or ""),
        "channels": [str(channel) for channel in (getattr(response_meta, "channels", ()) or ())],
    }


def _build_euv_response_identity_payload(*, response: Any, response_dt: Any, response_meta: Any) -> dict[str, Any]:
    arr = np.asarray(response)
    payload: dict[str, Any] = {
        "schema": EUV_RESPONSE_IDENTITY_VERSION,
        "meta": _normalize_euv_response_meta_for_identity(response_meta),
        "response_dt": _normalize_dtype_descriptor(response_dt if response_dt is not None else arr.dtype),
    }
    if arr.dtype.fields and {"ds", "NT", "Nchannels", "logte", "all"}.issubset(set(arr.dtype.fields)) and arr.size >= 1:
        flat = arr.reshape(-1)
        entry = flat[0]
        ds_value = np.asarray(entry["ds"]).reshape(-1)[0]
        nt_value = np.asarray(entry["NT"]).reshape(-1)[0]
        nchannels_value = np.asarray(entry["Nchannels"]).reshape(-1)[0]
        logte = np.asarray(entry["logte"], dtype=np.float64)
        response_matrix = np.asarray(entry["all"], dtype=np.float64)
        payload["response"] = {
            "layout": "gx_structured_array",
            "ds": None if not np.isfinite(float(ds_value)) else float(ds_value),
            "nt": int(nt_value),
            "nchannels": int(nchannels_value),
            "logte": _canonical_float64_array(logte),
            "all": _canonical_float64_array(response_matrix),
        }
        return payload
    payload["response"] = _normalize_identity_value(response)
    return payload


def _serialize_euv_response_identity_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def compute_euv_response_identity(*, response: Any, response_dt: Any, response_meta: Any) -> EUVResponseIdentity:
    payload = _build_euv_response_identity_payload(
        response=response,
        response_dt=response_dt,
        response_meta=response_meta,
    )
    digest = hashlib.sha256(_serialize_euv_response_identity_bytes(payload)).hexdigest()
    response_payload = dict(payload.get("response", {})) if isinstance(payload.get("response"), dict) else {}
    logte_values = np.asarray(response_payload.get("logte", []), dtype=float)
    summary = {
        "instrument": str(getattr(response_meta, "instrument", "") or ""),
        "channels": [str(channel) for channel in (getattr(response_meta, "channels", ()) or ())],
        "source": str(getattr(response_meta, "source", "") or ""),
        "mode": str(getattr(response_meta, "mode", "") or ""),
        "layout": str(response_payload.get("layout", "generic")),
        "ds": response_payload.get("ds"),
        "nt": response_payload.get("nt"),
        "nchannels": response_payload.get("nchannels"),
        "logte_min": (None if logte_values.size == 0 else float(np.nanmin(logte_values))),
        "logte_max": (None if logte_values.size == 0 else float(np.nanmax(logte_values))),
    }
    return EUVResponseIdentity(
        sha256=digest,
        version=EUV_RESPONSE_IDENTITY_VERSION,
        payload=payload,
        summary=summary,
    )


def resolve_euv_response_identity(**adapter_kwargs: Any) -> EUVResponseIdentity | None:
    return GXRenderEUVAdapter(**adapter_kwargs).response_identity()


def compute_forward_model_identity_placeholder(*, model_path: str | Path) -> ForwardModelIdentity:
    """Interim forward-model identity: SHA256 of the source model H5 file.

    Replace with gxrender-backed identity once ``compute_forward_model_identity``
    is available upstream. Artifacts tagged with this version are not reusable
    across future identity scheme upgrades.
    """

    path = Path(model_path).expanduser()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    sha256 = digest.hexdigest()
    return ForwardModelIdentity(
        sha256=sha256,
        version=FORWARD_MODEL_IDENTITY_VERSION,
        source_path=str(path),
        summary={"source_path": str(path), "method": "file_sha256"},
    )


def build_tr_region_mask_from_blos(
    blos_map: np.ndarray,
    *,
    threshold_gauss: float = 1000.0,
    use_absolute_field: bool = True,
) -> np.ndarray:
    """Build a simple EUV transition-region mask from a projected B_los map."""

    arr = np.asarray(blos_map, dtype=float)
    threshold = float(threshold_gauss)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError(f"threshold_gauss must be a finite non-negative scalar, got {threshold_gauss!r}")
    field = np.abs(arr) if bool(use_absolute_field) else arr
    mask = np.isfinite(field) & (field >= threshold)
    return np.asarray(mask, dtype=bool)


def recombine_euv_components(
    flux_corona: np.ndarray,
    flux_tr: np.ndarray,
    *,
    tr_region_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Combine EUV coronal/TR components under a chosen transition-region mask.

    Current pyCHMP assumes the TR mask is a downstream gate on an already
    computed TR contribution. If upstream voxel typing or TR physics later
    become mask-dependent in pyGXrender, this helper and the artifact contract
    are the main places that must change.
    """

    cor = np.asarray(flux_corona, dtype=float)
    tr = np.asarray(flux_tr, dtype=float)
    if cor.shape != tr.shape:
        raise ValueError(f"EUV component shape mismatch: {cor.shape} vs {tr.shape}")
    if tr_region_mask is None:
        return cor + tr
    mask = np.asarray(tr_region_mask, dtype=bool)
    if mask.shape != cor.shape:
        raise ValueError(f"TR region mask shape mismatch: {mask.shape} vs {cor.shape}")
    return cor + (tr * mask.astype(float))


def _load_gxrender_sdk() -> Any:
    try:
        return import_module("gxrender.sdk")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "gxrender is not installed or not importable. Install gximagecomputing/pyGXrender "
            "into the active environment before using GXRenderMWAdapter."
        ) from exc


def _rename_h5_output_if_needed(output_dir: str | Path | None, h5_path_raw: str | Path | None) -> None:
    if not output_dir or not h5_path_raw:
        return
    h5_path = Path(h5_path_raw)
    if h5_path.exists() and h5_path.suffix != ".h5":
        h5_path.rename(h5_path.with_suffix(".h5"))


def _load_gxrender_module() -> Any:
    try:
        return import_module("gxrender")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "gxrender is not installed or not importable. Install gximagecomputing/pyGXrender "
            "into the active environment before using GXRenderMWAdapter."
        ) from exc


def _load_common_workflow_helpers() -> Any:
    try:
        return import_module("gxrender.workflows._render_common")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "gxrender shared workflow helpers are not importable. Install gximagecomputing/pyGXrender "
            "into the active environment before using GXRenderMWAdapter."
        ) from exc


@dataclass(frozen=True, slots=True)
class ResolvedRenderGeometry:
    """Geometry resolved by the upstream gxrender workflow policy."""

    geometry: Any
    observer_name: str
    observer_source: str
    center_source: str
    fov_x_arcsec: float
    fov_y_arcsec: float


def resolve_render_geometry_via_gxrender(
    *,
    model_path: str | Path,
    model_format: str = "auto",
    ebtel_path: str | None = None,
    pixel_scale_arcsec: float = 2.0,
    geometry: Any | None = None,
    observer_name: str | None = None,
    observer: Any | None = None,
    omp_threads: int = 8,
    prefer_execute_center: bool = True,
    use_saved_fov: bool = False,
) -> ResolvedRenderGeometry:
    """Resolve render geometry by delegating observer/FOV policy to gxrender."""

    sdk = _load_gxrender_sdk()
    common_mod = _load_common_workflow_helpers()
    effective_observer_name = None if bool(use_saved_fov) and geometry is None and observer is None else observer_name
    args = SimpleNamespace(
        omp_threads=int(omp_threads),
        model_path=Path(model_path),
        model_format=str(model_format),
        ebtel_path=ebtel_path,
        observer=effective_observer_name,
        dsun_cm=None if observer is None else getattr(observer, "dsun_cm", None),
        lonc_deg=None if observer is None else getattr(observer, "lonc_deg", None),
        b0sun_deg=None if observer is None else getattr(observer, "b0sun_deg", None),
        xc=None if geometry is None else getattr(geometry, "xc", None),
        yc=None if geometry is None else getattr(geometry, "yc", None),
        dx=None if geometry is None else getattr(geometry, "dx", None),
        dy=None if geometry is None else getattr(geometry, "dy", None),
        pixel_scale_arcsec=float(pixel_scale_arcsec),
        nx=None if geometry is None else getattr(geometry, "nx", None),
        ny=None if geometry is None else getattr(geometry, "ny", None),
        xrange=None if geometry is None else getattr(geometry, "xrange", None),
        yrange=None if geometry is None else getattr(geometry, "yrange", None),
        auto_fov=False,
        use_saved_fov=bool(use_saved_fov),
    )
    common = common_mod.prepare_common_inputs(args, prefer_execute_center=prefer_execute_center)
    resolved_geometry = sdk.MapGeometry(
        xc=float(common.xc),
        yc=float(common.yc),
        dx=float(common.dx),
        dy=float(common.dy),
        nx=int(common.nx),
        ny=int(common.ny),
    )
    observer_geometry = common.observer_geometry
    return ResolvedRenderGeometry(
        geometry=resolved_geometry,
        observer_name=str(getattr(observer_geometry, "observer_name", effective_observer_name or "")),
        observer_source=str(getattr(observer_geometry, "observer_source", "")),
        center_source=str(common.center_source),
        fov_x_arcsec=float(common.fov_x),
        fov_y_arcsec=float(common.fov_y),
    )


def _load_render_mw_workflow() -> Any:
    try:
        return import_module("gxrender.workflows.render_mw")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "gxrender microwave workflow helpers are not importable. Install gximagecomputing/pyGXrender "
            "into the active environment before using GXRenderMWAdapter."
        ) from exc


def _normalize_euv_channel_token(value: str) -> str:
    token = "".join(ch for ch in str(value).strip().upper() if ch not in {" ", "_", "-"})
    while token and token[0].isalpha():
        token = token[1:]
    return token


@dataclass(slots=True)
class GXRenderMWContext:
    """Per-process reusable MW render context.

    The expensive gxrender setup path loads the model, EBTEL tables, observer
    geometry, and output grid only once per process. Individual renders then
    vary only the coronal parameters and frequency.
    """

    model_path: str | Path
    model_format: str = "auto"
    ebtel_path: str | None = None
    omp_threads: int = 8
    pixel_scale_arcsec: float = 2.0
    geometry: Any | None = None
    observer: Any | None = None
    observer_name: str | None = None
    _gxi: Any = field(init=False, repr=False)
    _common: Any = field(init=False, repr=False)
    _workflow_helpers: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        sdk = _load_gxrender_sdk()
        gxrender = _load_gxrender_module()
        workflow_helpers = _load_common_workflow_helpers()

        geometry = self.geometry
        if geometry is None:
            geometry = sdk.MapGeometry(pixel_scale_arcsec=float(self.pixel_scale_arcsec))
        observer = self.observer
        observer_name = self.observer_name or (str(observer) if isinstance(observer, str) else None)

        args = SimpleNamespace(
            omp_threads=int(self.omp_threads),
            model_path=Path(self.model_path),
            model_format=str(self.model_format),
            ebtel_path=self.ebtel_path,
            observer=observer_name,
            dsun_cm=None if isinstance(observer, str) else getattr(observer, "dsun_cm", None),
            lonc_deg=None if isinstance(observer, str) else getattr(observer, "lonc_deg", None),
            b0sun_deg=None if isinstance(observer, str) else getattr(observer, "b0sun_deg", None),
            xc=getattr(geometry, "xc", None),
            yc=getattr(geometry, "yc", None),
            dx=getattr(geometry, "dx", None),
            dy=getattr(geometry, "dy", None),
            pixel_scale_arcsec=getattr(geometry, "pixel_scale_arcsec", float(self.pixel_scale_arcsec)),
            nx=getattr(geometry, "nx", None),
            ny=getattr(geometry, "ny", None),
            xrange=getattr(geometry, "xrange", None),
            yrange=getattr(geometry, "yrange", None),
            auto_fov=False,
            use_saved_fov=False,
        )
        self._workflow_helpers = workflow_helpers
        self._common = workflow_helpers.prepare_common_inputs(args)
        self._gxi = gxrender.GXRadioImageComputing()

    def render_stokes_raw(
        self,
        *,
        frequency_ghz: float,
        tbase: float,
        nbase: float,
        q0: float,
        a: float,
        b: float,
        mode: int = 0,
        selective_heating: bool = False,
        shtable: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        stokes_i, stokes_v = self._render_stokes_cube(
            frequencies_ghz=[float(frequency_ghz)],
            tbase=tbase,
            nbase=nbase,
            q0=q0,
            a=a,
            b=b,
            mode=mode,
            selective_heating=selective_heating,
            shtable=shtable,
        )
        return np.asarray(stokes_i[:, :, 0], dtype=float), np.asarray(stokes_v[:, :, 0], dtype=float)

    def render(
        self,
        *,
        frequency_ghz: float,
        tbase: float,
        nbase: float,
        q0: float,
        a: float,
        b: float,
        mode: int = 0,
        selective_heating: bool = False,
        shtable: Any | None = None,
    ) -> np.ndarray:
        stokes_i, _stokes_v = self.render_stokes_raw(
            frequency_ghz=frequency_ghz,
            tbase=tbase,
            nbase=nbase,
            q0=q0,
            a=a,
            b=b,
            mode=mode,
            selective_heating=selective_heating,
            shtable=shtable,
        )
        return stokes_i

    def _render_stokes_cube(
        self,
        *,
        frequencies_ghz: Sequence[float],
        tbase: float,
        nbase: float,
        q0: float,
        a: float,
        b: float,
        mode: int = 0,
        selective_heating: bool = False,
        shtable: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        freqs = np.asarray([float(v) for v in frequencies_ghz], dtype=np.float64)
        if freqs.size == 0:
            raise ValueError("frequencies_ghz must contain at least one frequency")
        plasma_args = SimpleNamespace(
            tbase=tbase,
            nbase=nbase,
            q0=float(q0),
            a=a,
            b=b,
            corona_mode=mode,
            force_isothermal=False,
            interpol_b=False,
            analytical_nt=False,
            selective_heating=bool(selective_heating),
            shtable=shtable,
            shtable_path=None,
        )
        plasma = self._workflow_helpers.resolve_plasma_parameters(plasma_args)
        result = self._gxi.synth_model(
            self._common.model,
            self._common.model_dt,
            self._common.ebtel_c,
            self._common.ebtel_dt,
            freqs,
            int(self._common.nx),
            int(self._common.ny),
            float(self._common.xc),
            float(self._common.yc),
            float(self._common.dx),
            float(self._common.dy),
            float(plasma.tbase),
            float(plasma.nbase),
            float(plasma.q0),
            float(plasma.a),
            float(plasma.b),
            SHtable=plasma.shtable,
            mode=int(plasma.mode),
            warn_defaults=False,
        )
        ti = np.asarray(result["TI"], dtype=float)
        tv = np.asarray(result["TV"], dtype=float)
        if ti.ndim != 3 or ti.shape[2] != freqs.size:
            raise ValueError(f"expected MW TI cube with shape (ny, nx, {freqs.size}), got {ti.shape}")
        if tv.ndim != 3 or tv.shape[2] != freqs.size:
            raise ValueError(f"expected MW TV cube with shape (ny, nx, {freqs.size}), got {tv.shape}")
        return ti, tv

    def render_cube(
        self,
        *,
        frequencies_ghz: Sequence[float],
        tbase: float,
        nbase: float,
        q0: float,
        a: float,
        b: float,
        mode: int = 0,
        selective_heating: bool = False,
        shtable: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._render_stokes_cube(
            frequencies_ghz=frequencies_ghz,
            tbase=tbase,
            nbase=nbase,
            q0=q0,
            a=a,
            b=b,
            mode=mode,
            selective_heating=selective_heating,
            shtable=shtable,
        )


@dataclass(slots=True)
class GXRenderMWAdapter:
    """Concrete Q0 renderer backed by a persistent gxrender MW context."""

    model_path: str | Path
    frequency_ghz: float
    render_frequencies_ghz: Sequence[float] | None = None
    model_format: str = "auto"
    ebtel_path: str | None = None
    tbase: float | None = None
    nbase: float | None = None
    a: float | None = None
    b: float | None = None
    mode: int = 0
    selective_heating: bool = False
    shtable: Any | None = None
    omp_threads: int = 8
    pixel_scale_arcsec: float = 2.0
    geometry: Any | None = None
    observer: Any | None = None
    observer_name: str | None = None
    output_dir: str | Path | None = None
    output_name: str | None = None
    output_format: str = "h5"
    verbose: bool = False
    render_call_count: int = 0
    _context: GXRenderMWContext = field(init=False, repr=False)
    _cube_cache: dict[float, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._context = GXRenderMWContext(
            model_path=self.model_path,
            model_format=self.model_format,
            ebtel_path=self.ebtel_path,
            omp_threads=int(self.omp_threads),
            pixel_scale_arcsec=float(self.pixel_scale_arcsec),
            geometry=self.geometry,
            observer=self.observer,
            observer_name=self.observer_name,
        )

    def render(self, q0: float) -> np.ndarray:
        self.render_call_count += 1
        if not self.output_dir and self.render_frequencies_ghz:
            cube_payload = self.render_cube(float(q0))
            for freq, rendered in cube_payload["raw_modeled_by_frequency"].items():
                if np.isclose(float(freq), float(self.frequency_ghz), rtol=0.0, atol=1e-12):
                    return np.asarray(rendered, dtype=float)
            raise ValueError(f"target frequency {self.frequency_ghz} was not present in rendered MW cube")
        if self.output_dir:
            workflow = _load_render_mw_workflow()
            geometry = self.geometry
            if geometry is None:
                sdk = _load_gxrender_sdk()
                geometry = sdk.MapGeometry(pixel_scale_arcsec=float(self.pixel_scale_arcsec))
            observer = self.observer
            observer_name = self.observer_name or (str(observer) if isinstance(observer, str) else None)
            args = SimpleNamespace(
                model_path=Path(self.model_path),
                model_format=str(self.model_format),
                ebtel_path=self.ebtel_path,
                output_dir=Path(self.output_dir),
                output_name=self.output_name,
                output_format=str(self.output_format),
                frequencies_ghz=[float(self.frequency_ghz)],
                omp_threads=int(self.omp_threads),
                save_outputs=True,
                write_preview=False,
                observer=observer_name,
                dsun_cm=None if isinstance(observer, str) else getattr(observer, "dsun_cm", None),
                lonc_deg=None if isinstance(observer, str) else getattr(observer, "lonc_deg", None),
                b0sun_deg=None if isinstance(observer, str) else getattr(observer, "b0sun_deg", None),
                xc=getattr(geometry, "xc", None),
                yc=getattr(geometry, "yc", None),
                dx=getattr(geometry, "dx", None),
                dy=getattr(geometry, "dy", None),
                pixel_scale_arcsec=getattr(geometry, "pixel_scale_arcsec", float(self.pixel_scale_arcsec)),
                nx=getattr(geometry, "nx", None),
                ny=getattr(geometry, "ny", None),
                xrange=getattr(geometry, "xrange", None),
                yrange=getattr(geometry, "yrange", None),
                auto_fov=False,
                use_saved_fov=False,
                tbase=self.tbase,
                nbase=self.nbase,
                q0=float(q0),
                a=self.a,
                b=self.b,
                corona_mode=self.mode,
                selective_heating=bool(self.selective_heating),
                shtable=self.shtable,
                shtable_path=None,
                force_isothermal=False,
                interpol_b=False,
                analytical_nt=False,
            )
            run_result = workflow.run(args, verbose=bool(self.verbose))
            _rename_h5_output_if_needed(self.output_dir, run_result["outputs"].get("h5_path"))
            ti = np.asarray(run_result["result"]["TI"], dtype=float)
            if ti.ndim != 3 or ti.shape[2] != 1:
                raise ValueError(f"expected single-frequency TI cube with shape (ny, nx, 1), got {ti.shape}")
            return ti[:, :, 0]
        return self._context.render(
            frequency_ghz=float(self.frequency_ghz),
            tbase=float(self.tbase),
            nbase=float(self.nbase),
            q0=float(q0),
            a=float(self.a),
            b=float(self.b),
            mode=int(self.mode),
            selective_heating=bool(self.selective_heating),
            shtable=self.shtable,
        )

    def render_stokes_raw(self, q0: float) -> tuple[np.ndarray, np.ndarray]:
        return self._context.render_stokes_raw(
            frequency_ghz=float(self.frequency_ghz),
            tbase=float(self.tbase),
            nbase=float(self.nbase),
            q0=float(q0),
            a=float(self.a),
            b=float(self.b),
            mode=int(self.mode),
            selective_heating=bool(self.selective_heating),
            shtable=self.shtable,
        )

    def render_cube(self, q0: float) -> dict[str, Any]:
        q0_key = float(q0)
        cached = self._cube_cache.get(q0_key)
        if cached is not None:
            return cached
        frequencies = [float(self.frequency_ghz)]
        for value in list(self.render_frequencies_ghz or []):
            numeric = float(value)
            if not any(np.isclose(numeric, existing, rtol=0.0, atol=1e-12) for existing in frequencies):
                frequencies.append(numeric)
        cube, stokes_v_cube = self._context.render_cube(
            frequencies_ghz=frequencies,
            tbase=float(self.tbase),
            nbase=float(self.nbase),
            q0=float(q0),
            a=float(self.a),
            b=float(self.b),
            mode=int(self.mode),
            selective_heating=bool(self.selective_heating),
            shtable=self.shtable,
        )
        payload = {
            "frequencies_ghz": frequencies,
            "raw_modeled_cube": cube,
            "raw_stokes_v_cube": stokes_v_cube,
            "raw_modeled_by_frequency": {
                float(freq): np.asarray(cube[:, :, index], dtype=float)
                for index, freq in enumerate(frequencies)
            },
            "stokes_v_by_frequency": {
                float(freq): np.asarray(stokes_v_cube[:, :, index], dtype=float)
                for index, freq in enumerate(frequencies)
            },
        }
        self._cube_cache[q0_key] = payload
        return payload


@dataclass(slots=True)
class GXRenderEUVAdapter:
    """Concrete Q0 renderer backed by gxrender's EUV workflow."""

    model_path: str | Path
    channel: str
    render_channels: Sequence[str] | None = None
    instrument: str = "AIA"
    response_sav: str | Path | None = None
    model_format: str = "auto"
    ebtel_path: str | None = None
    tbase: float | None = None
    nbase: float | None = None
    a: float | None = None
    b: float | None = None
    mode: int = 0
    selective_heating: bool = False
    shtable: Any | None = None
    omp_threads: int = 8
    pixel_scale_arcsec: float = 2.0
    geometry: Any | None = None
    observer: Any | None = None
    observer_name: str | None = None
    tr_region_mask: np.ndarray | None = None
    output_dir: str | Path | None = None
    output_name: str | None = None
    verbose: bool = False
    render_call_count: int = 0
    cache_response: bool = True
    _response_cache: _CachedEUVResponse | None = field(default=None, init=False, repr=False)
    _response_cache_key: tuple[Any, ...] | None = field(default=None, init=False, repr=False)
    _response_cache_attempted: bool = field(default=False, init=False, repr=False)
    _response_cache_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _components_cache: dict[float, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        instrument = str(self.instrument).strip()
        if not instrument:
            raise ValueError("instrument must be a non-empty string")
        self.instrument = instrument

        channel = str(self.channel).strip()
        if not channel:
            raise ValueError("channel must be a non-empty string")
        self.channel = channel

    def _observer_to_kwargs(self) -> dict[str, Any]:
        observer = self.observer
        return {
            "dsun_cm": None if observer is None else getattr(observer, "dsun_cm", None),
            "lonc_deg": None if observer is None else getattr(observer, "lonc_deg", None),
            "b0sun_deg": None if observer is None else getattr(observer, "b0sun_deg", None),
            "observer": self.observer_name,
        }

    def _geometry_to_kwargs(self) -> dict[str, Any]:
        geometry = self.geometry
        return {
            "xc": None if geometry is None else getattr(geometry, "xc", None),
            "yc": None if geometry is None else getattr(geometry, "yc", None),
            "dx": None if geometry is None else getattr(geometry, "dx", None),
            "dy": None if geometry is None else getattr(geometry, "dy", None),
            "pixel_scale_arcsec": (
                float(self.pixel_scale_arcsec)
                if geometry is None or getattr(geometry, "pixel_scale_arcsec", None) is None
                else getattr(geometry, "pixel_scale_arcsec", None)
            ),
            "nx": None if geometry is None else getattr(geometry, "nx", None),
            "ny": None if geometry is None else getattr(geometry, "ny", None),
            "xrange": None if geometry is None else getattr(geometry, "xrange", None),
            "yrange": None if geometry is None else getattr(geometry, "yrange", None),
        }

    def _response_cache_key_for_current_request(self) -> tuple[Any, ...]:
        return (
            str(Path(self.model_path).expanduser()),
            str(self.model_format),
            str(self.instrument),
            str(self.channel),
            tuple(str(channel) for channel in (self.render_channels or ())),
            None if self.response_sav is None else str(Path(self.response_sav).expanduser()),
            tuple(sorted(self._observer_to_kwargs().items())),
        )

    def _resolve_euv_response_cache(self) -> _CachedEUVResponse | None:
        try:
            common_mod = import_module("gxrender.workflows._render_common")
            contracts = import_module("gxrender.policy.contracts")
            response_policy = import_module("gxrender.policy.euv_response_policy")
        except Exception:
            return None

        args = SimpleNamespace(
            model_path=Path(self.model_path),
            model_format=str(self.model_format),
            ebtel_path=self.ebtel_path,
            channels=list(dict.fromkeys([str(self.channel), *[str(channel) for channel in (self.render_channels or ())]])),
            instrument=str(self.instrument),
            response_sav=(None if self.response_sav is None else Path(self.response_sav)),
            response=None,
            response_dt=None,
            response_meta=None,
            omp_threads=int(self.omp_threads),
            output_dir=None,
            output_name=None,
            save_outputs=False,
            write_preview=False,
            tbase=float(self.tbase),
            nbase=float(self.nbase),
            q0=0.0,
            a=float(self.a),
            b=float(self.b),
            corona_mode=int(self.mode),
            selective_heating=bool(self.selective_heating),
            shtable=self.shtable,
            shtable_path=None,
            auto_fov=False,
            use_saved_fov=False,
            **self._geometry_to_kwargs(),
            **self._observer_to_kwargs(),
        )
        try:
            common = common_mod.prepare_common_inputs(args, prefer_execute_center=False)
            response_policy.apply_default_response_selection(args, observer_geometry=common.observer_geometry)
            resolved = response_policy.resolve_euv_response(
                contracts.EUVResponseRequest(
                    args=args,
                    obs_time_iso=common_mod.model_obstime_iso(common.model),
                )
            )
        except Exception:
            return None

        return _CachedEUVResponse(
            response=resolved.response,
            response_dt=resolved.response_dt,
            response_meta=resolved.response_meta,
            response_identity=compute_euv_response_identity(
                response=resolved.response,
                response_dt=resolved.response_dt,
                response_meta=resolved.response_meta,
            ),
        )

    def _ensure_euv_response_cache(self) -> _CachedEUVResponse | None:
        if not bool(self.cache_response):
            return None
        if self.response_sav is None:
            return None
        cache_key = self._response_cache_key_for_current_request()
        cached = self._response_cache
        if self._response_cache_key == cache_key and (cached is not None or self._response_cache_attempted):
            return cached
        with self._response_cache_lock:
            cached = self._response_cache
            if self._response_cache_key != cache_key:
                self._response_cache = None
                self._response_cache_attempted = False
                self._response_cache_key = cache_key
                cached = None
            if cached is None and not self._response_cache_attempted:
                cached = self._resolve_euv_response_cache()
                self._response_cache = cached
                self._response_cache_attempted = True
            return cached

    def response_identity(self) -> EUVResponseIdentity | None:
        cached = self._ensure_euv_response_cache()
        if cached is None:
            return None
        if cached.response_identity is None:
            cached.response_identity = compute_euv_response_identity(
                response=cached.response,
                response_dt=cached.response_dt,
                response_meta=cached.response_meta,
            )
        return cached.response_identity

    def render_components(self, q0: float) -> dict[str, Any]:
        q0_key = float(q0)
        cached = self._components_cache.get(q0_key)
        if cached is not None:
            return cached
        self.render_call_count += 1
        sdk = _load_gxrender_sdk()

        geometry = self.geometry
        if geometry is None:
            geometry = sdk.MapGeometry(pixel_scale_arcsec=float(self.pixel_scale_arcsec))

        plasma = sdk.CoronalPlasmaParameters(
            tbase=float(self.tbase),
            nbase=float(self.nbase),
            q0=float(q0),
            a=float(self.a),
            b=float(self.b),
            mode=int(self.mode),
            selective_heating=bool(self.selective_heating),
            shtable=self.shtable,
        )
        cached_response = self._ensure_euv_response_cache()
        cached_response_identity = None if cached_response is None else self.response_identity()
        options = sdk.EUVRenderOptions(
            model_path=Path(self.model_path),
            model_format=str(self.model_format),
            ebtel_path=self.ebtel_path,
            output_dir=(Path(self.output_dir) if self.output_dir is not None else None),
            output_name=self.output_name,
            channels=list(dict.fromkeys([str(self.channel), *[str(channel) for channel in (self.render_channels or ())]])),
            instrument=str(self.instrument),
            response_sav=(
                None
                if cached_response is not None or self.response_sav is None
                else Path(self.response_sav)
            ),
            response=None if cached_response is None else cached_response.response,
            response_dt=None if cached_response is None else cached_response.response_dt,
            response_meta=None if cached_response is None else cached_response.response_meta,
            plasma=plasma,
            omp_threads=int(self.omp_threads),
            geometry=geometry,
            observer=self.observer,
            observer_name=self.observer_name,
            save_outputs=bool(self.output_dir),
            write_preview=False,
            verbose=bool(self.verbose),
        )
        global _euv_projection_flags_warning_emitted
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.filterwarnings(
                "always",
                message=(
                    r"Current Python EUV workflow uses projection flags off "
                    r"\(parallel=False, exact=False, nthreads=0\) for the DLL simbox path\..*"
                ),
                category=UserWarning,
            )
            result = sdk.render_euv_maps(options)
        for warning in caught_warnings:
            if (
                issubclass(warning.category, UserWarning)
                and str(warning.message).startswith(_EUV_PROJECTION_FLAGS_WARNING)
            ):
                with _euv_projection_flags_warning_lock:
                    if not _euv_projection_flags_warning_emitted:
                        warnings.warn(str(warning.message), category=warning.category, stacklevel=2)
                        _euv_projection_flags_warning_emitted = True
                continue
            warnings.warn_explicit(
                warning.message,
                warning.category,
                warning.filename,
                warning.lineno,
            )
        flux_corona = np.asarray(result.flux_corona, dtype=float)
        flux_tr = np.asarray(result.flux_tr, dtype=float)
        if flux_corona.shape != flux_tr.shape:
            raise ValueError(
                "expected EUV corona and transition-region cubes with identical shapes, got "
                f"{flux_corona.shape} and {flux_tr.shape}"
            )
        if flux_corona.ndim != 3:
            raise ValueError(f"expected EUV cubes with shape (ny, nx, nch), got {flux_corona.shape}")

        response_channels = [str(channel) for channel in getattr(result.response, "channels", [])]
        try:
            channel_index = response_channels.index(str(self.channel))
        except ValueError:
            normalized_channels = [_normalize_euv_channel_token(channel) for channel in response_channels]
            normalized_requested = _normalize_euv_channel_token(str(self.channel))
            try:
                channel_index = normalized_channels.index(normalized_requested)
            except ValueError as exc:
                raise ValueError(
                    f"requested EUV channel {self.channel!r} was not present in the rendered response set {response_channels}"
                ) from exc
        except Exception as exc:
            raise ValueError(
                f"requested EUV channel {self.channel!r} was not present in the rendered response set {response_channels}"
            ) from exc
        selected_corona = np.asarray(flux_corona[:, :, channel_index], dtype=float)
        selected_tr = np.asarray(flux_tr[:, :, channel_index], dtype=float)
        rendered_by_channel: dict[str, np.ndarray] = {}
        corona_by_channel: dict[str, np.ndarray] = {}
        tr_by_channel: dict[str, np.ndarray] = {}
        for index, channel in enumerate(response_channels):
            coronal_slice = np.asarray(flux_corona[:, :, index], dtype=float)
            tr_slice = np.asarray(flux_tr[:, :, index], dtype=float)
            rendered_by_channel[str(channel)] = recombine_euv_components(
                coronal_slice,
                tr_slice,
                tr_region_mask=self.tr_region_mask,
            )
            corona_by_channel[str(channel)] = coronal_slice
            tr_by_channel[str(channel)] = tr_slice
        rendered = recombine_euv_components(
            selected_corona,
            selected_tr,
            tr_region_mask=self.tr_region_mask,
        )
        if np.isfinite(rendered).all():
            payload = {
                "rendered": rendered,
                "flux_corona": selected_corona,
                "flux_tr": selected_tr,
                "render_channels": response_channels,
                "rendered_by_channel": rendered_by_channel,
                "flux_corona_by_channel": corona_by_channel,
                "flux_tr_by_channel": tr_by_channel,
                "tr_region_mask": (
                    None if self.tr_region_mask is None else np.asarray(self.tr_region_mask, dtype=bool)
                ),
                "response_identity": cached_response_identity,
            }
            self._components_cache[q0_key] = payload
            return payload

        finite = rendered[np.isfinite(rendered)]
        if finite.size == 0:
            raise ValueError(
                f"EUV render for channel {self.channel!r} produced no finite pixels; cannot evaluate the fit objective"
            )
        finite_min = float(np.nanmin(finite))
        finite_max = float(np.nanmax(finite))
        rendered = np.nan_to_num(rendered, nan=0.0, posinf=finite_max, neginf=finite_min)
        payload = {
            "rendered": rendered,
            "flux_corona": np.nan_to_num(selected_corona, nan=0.0, posinf=finite_max, neginf=finite_min),
            "flux_tr": np.nan_to_num(selected_tr, nan=0.0, posinf=finite_max, neginf=finite_min),
            "render_channels": response_channels,
            "rendered_by_channel": {
                channel: np.nan_to_num(values, nan=0.0, posinf=finite_max, neginf=finite_min)
                for channel, values in rendered_by_channel.items()
            },
            "flux_corona_by_channel": {
                channel: np.nan_to_num(values, nan=0.0, posinf=finite_max, neginf=finite_min)
                for channel, values in corona_by_channel.items()
            },
            "flux_tr_by_channel": {
                channel: np.nan_to_num(values, nan=0.0, posinf=finite_max, neginf=finite_min)
                for channel, values in tr_by_channel.items()
            },
            "tr_region_mask": (
                None if self.tr_region_mask is None else np.asarray(self.tr_region_mask, dtype=bool)
            ),
            "response_identity": cached_response_identity,
        }
        self._components_cache[q0_key] = payload
        return payload

    def render(self, q0: float) -> np.ndarray:
        return np.asarray(self.render_components(q0)["rendered"], dtype=float)
