"""Adapters for using gxrender as a concrete pyCHMP forward model backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


def _load_gxrender_sdk() -> Any:
    try:
        return import_module("gxrender.sdk")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "gxrender is not installed or not importable. Install gximagecomputing/pyGXrender "
            "into the active environment before using GXRenderMWAdapter."
        ) from exc


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

        args = SimpleNamespace(
            omp_threads=int(self.omp_threads),
            model_path=Path(self.model_path),
            model_format=str(self.model_format),
            ebtel_path=self.ebtel_path,
            observer=None,
            dsun_cm=getattr(observer, "dsun_cm", None),
            lonc_deg=getattr(observer, "lonc_deg", None),
            b0sun_deg=getattr(observer, "b0sun_deg", None),
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
            np.asarray([float(frequency_ghz)], dtype=np.float64),
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
        if ti.ndim != 3 or ti.shape[2] != 1:
            raise ValueError(f"expected single-frequency TI cube with shape (ny, nx, 1), got {ti.shape}")
        return ti[:, :, 0]


@dataclass(slots=True)
class GXRenderMWAdapter:
    """Concrete Q0 renderer backed by a persistent gxrender MW context."""

    model_path: str | Path
    frequency_ghz: float
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
    output_dir: str | Path | None = None
    output_name: str | None = None
    output_format: str = "h5"
    verbose: bool = False
    render_call_count: int = 0
    _context: GXRenderMWContext = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._context = GXRenderMWContext(
            model_path=self.model_path,
            model_format=self.model_format,
            ebtel_path=self.ebtel_path,
            omp_threads=int(self.omp_threads),
            pixel_scale_arcsec=float(self.pixel_scale_arcsec),
            geometry=self.geometry,
            observer=self.observer,
        )

    def render(self, q0: float) -> np.ndarray:
        self.render_call_count += 1
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
