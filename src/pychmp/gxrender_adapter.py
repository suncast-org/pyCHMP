"""Adapters for using gxrender as a concrete pyCHMP forward model backend."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
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


@dataclass(slots=True)
class GXRenderMWAdapter:
    """Concrete Q0 renderer backed by gxrender.render_mw_maps.

    The adapter renders a single Stokes-I microwave map at one frequency for a
    supplied Q0, holding all other plasma/model settings fixed.
    """

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
    verbose: bool = False

    def render(self, q0: float) -> np.ndarray:
        sdk = _load_gxrender_sdk()

        geometry = self.geometry
        if geometry is None:
            geometry = sdk.MapGeometry(pixel_scale_arcsec=float(self.pixel_scale_arcsec))

        plasma = sdk.CoronalPlasmaParameters(
            tbase=self.tbase,
            nbase=self.nbase,
            q0=float(q0),
            a=self.a,
            b=self.b,
            mode=self.mode,
            selective_heating=self.selective_heating,
            shtable=self.shtable,
        )
        options = sdk.MWRenderOptions(
            model_path=Path(self.model_path),
            model_format=self.model_format,
            ebtel_path=self.ebtel_path,
            freqlist_ghz=[float(self.frequency_ghz)],
            plasma=plasma,
            omp_threads=self.omp_threads,
            geometry=geometry,
            observer=self.observer,
            save_outputs=False,
            write_preview=False,
            verbose=self.verbose,
        )
        result = sdk.render_mw_maps(options)
        ti = np.asarray(result.ti, dtype=float)
        if ti.ndim != 3 or ti.shape[2] != 1:
            raise ValueError(f"expected single-frequency TI cube with shape (ny, nx, 1), got {ti.shape}")
        return ti[:, :, 0]
