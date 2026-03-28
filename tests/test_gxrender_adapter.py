from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pytest

import pychmp.gxrender_adapter as gxrender_adapter
from pychmp.gxrender_adapter import GXRenderMWAdapter


class FakeSDK:
    class MapGeometry:
        def __init__(self, **kwargs):
            self.pixel_scale_arcsec = kwargs.get("pixel_scale_arcsec")
            self.xc = kwargs.get("xc")
            self.yc = kwargs.get("yc")
            self.dx = kwargs.get("dx")
            self.dy = kwargs.get("dy")
            self.nx = kwargs.get("nx")
            self.ny = kwargs.get("ny")
            self.xrange = kwargs.get("xrange")
            self.yrange = kwargs.get("yrange")


class FakeGXRender:
    class GXRadioImageComputing:
        def synth_model(
            self,
            model,
            model_dt,
            ebtel_c,
            ebtel_dt,
            frequencies,
            nx,
            ny,
            xc,
            yc,
            dx,
            dy,
            tbase,
            nbase,
            q0,
            a,
            b,
            SHtable=None,
            mode=0,
            warn_defaults=False,
        ):
            cube = np.full((int(ny), int(nx), 1), float(q0), dtype=float)
            return {"TI": cube}


class WrongCubeGXRender(FakeGXRender):
    class GXRadioImageComputing:
        def synth_model(
            self,
            model,
            model_dt,
            ebtel_c,
            ebtel_dt,
            frequencies,
            nx,
            ny,
            xc,
            yc,
            dx,
            dy,
            tbase,
            nbase,
            q0,
            a,
            b,
            SHtable=None,
            mode=0,
            warn_defaults=False,
        ):
            return {"TI": np.ones((int(ny), int(nx)), dtype=float)}


class FakeWorkflowHelpers:
    prepared_args = None

    @staticmethod
    def prepare_common_inputs(args):
        FakeWorkflowHelpers.prepared_args = args
        return SimpleNamespace(
            model="model",
            model_dt="model_dt",
            ebtel_c="ebtel_c",
            ebtel_dt="ebtel_dt",
            nx=3,
            ny=2,
            xc=0.0,
            yc=0.0,
            dx=1.0,
            dy=1.0,
        )

    @staticmethod
    def resolve_plasma_parameters(args):
        return SimpleNamespace(
            tbase=args.tbase,
            nbase=args.nbase,
            q0=args.q0,
            a=args.a,
            b=args.b,
            mode=args.corona_mode,
            shtable=args.shtable,
        )


def test_gxrender_mw_adapter_renders_single_frequency_map(monkeypatch) -> None:
    monkeypatch.setattr(gxrender_adapter, "_load_gxrender_sdk", lambda: FakeSDK)
    monkeypatch.setattr(gxrender_adapter, "_load_gxrender_module", lambda: FakeGXRender)
    monkeypatch.setattr(gxrender_adapter, "_load_common_workflow_helpers", lambda: FakeWorkflowHelpers)

    adapter = GXRenderMWAdapter(
        model_path="model.h5",
        frequency_ghz=5.8,
        tbase=1e6,
        nbase=1e8,
        a=0.3,
        b=2.7,
    )

    image = adapter.render(0.0217)

    assert image.shape == (2, 3)
    assert np.allclose(image, 0.0217)


def test_gxrender_mw_adapter_requires_single_frequency_cube(monkeypatch) -> None:
    monkeypatch.setattr(gxrender_adapter, "_load_gxrender_sdk", lambda: FakeSDK)
    monkeypatch.setattr(gxrender_adapter, "_load_gxrender_module", lambda: WrongCubeGXRender)
    monkeypatch.setattr(gxrender_adapter, "_load_common_workflow_helpers", lambda: FakeWorkflowHelpers)

    adapter = GXRenderMWAdapter(
        model_path="model.h5",
        frequency_ghz=5.8,
        tbase=1e6,
        nbase=1e8,
        a=0.3,
        b=2.7,
    )

    with pytest.raises(ValueError, match="single-frequency TI cube"):
        adapter.render(0.1)


def test_gxrender_mw_adapter_surfaces_missing_dependency(monkeypatch) -> None:
    def raise_missing():
        raise ModuleNotFoundError("gxrender")

    monkeypatch.setattr(gxrender_adapter, "_load_gxrender_sdk", lambda: FakeSDK)
    monkeypatch.setattr(gxrender_adapter, "_load_gxrender_module", raise_missing)
    monkeypatch.setattr(gxrender_adapter, "_load_common_workflow_helpers", lambda: FakeWorkflowHelpers)

    with pytest.raises(ModuleNotFoundError, match="gxrender"):
        GXRenderMWAdapter(
            model_path="model.h5",
            frequency_ghz=5.8,
            tbase=1e6,
            nbase=1e8,
            a=0.3,
            b=2.7,
        )


def test_gxrender_context_forwards_named_observer_to_workflow_args(monkeypatch) -> None:
    FakeWorkflowHelpers.prepared_args = None
    monkeypatch.setattr(gxrender_adapter, "_load_gxrender_sdk", lambda: FakeSDK)
    monkeypatch.setattr(gxrender_adapter, "_load_gxrender_module", lambda: FakeGXRender)
    monkeypatch.setattr(gxrender_adapter, "_load_common_workflow_helpers", lambda: FakeWorkflowHelpers)

    GXRenderMWAdapter(
        model_path="model.h5",
        frequency_ghz=5.8,
        tbase=1e6,
        nbase=1e8,
        a=0.3,
        b=2.7,
        observer="earth",
    )

    assert FakeWorkflowHelpers.prepared_args is not None
    assert FakeWorkflowHelpers.prepared_args.observer == "earth"


def test_gxrender_adapter_uses_sdk_path_when_output_dir_requested(monkeypatch) -> None:
    class FakeRenderMWWorkflow:
        last_args = None

        @staticmethod
        def run(args, *, verbose=True):
            del verbose
            FakeRenderMWWorkflow.last_args = args
            output_dir = Path(args.output_dir)
            output_name = str(args.output_name or "rendered_map")
            raw_h5_path = output_dir / output_name
            output_dir.mkdir(parents=True, exist_ok=True)
            raw_h5_path.write_bytes(b"fake")
            cube = np.full((2, 3, 1), float(args.q0), dtype=float)
            return {
                "result": {"TI": cube},
                "outputs": {"h5_path": str(raw_h5_path)},
            }

    monkeypatch.setattr(gxrender_adapter, "_load_gxrender_sdk", lambda: FakeSDK)
    monkeypatch.setattr(gxrender_adapter, "_load_gxrender_module", lambda: FakeGXRender)
    monkeypatch.setattr(gxrender_adapter, "_load_common_workflow_helpers", lambda: FakeWorkflowHelpers)
    monkeypatch.setattr(gxrender_adapter, "_load_render_mw_workflow", lambda: FakeRenderMWWorkflow)

    with TemporaryDirectory() as tmpdir:
        adapter = GXRenderMWAdapter(
            model_path="model.h5",
            frequency_ghz=5.8,
            tbase=1e6,
            nbase=1e8,
            a=0.3,
            b=2.7,
            observer="earth",
            output_dir=tmpdir,
            output_name="demo_map",
            output_format="h5",
        )

        image = adapter.render(0.0217)

        assert image.shape == (2, 3)
        assert np.allclose(image, 0.0217)
        assert FakeRenderMWWorkflow.last_args is not None
        assert FakeRenderMWWorkflow.last_args.observer == "earth"
        assert Path(tmpdir, "demo_map.h5").exists()
