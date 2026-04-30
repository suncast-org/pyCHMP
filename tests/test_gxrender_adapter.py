from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pytest

import pychmp.gxrender_adapter as gxrender_adapter
from pychmp.gxrender_adapter import (
    GXRenderEUVAdapter,
    GXRenderMWAdapter,
    build_tr_region_mask_from_blos,
    recombine_euv_components,
)


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

    class CoronalPlasmaParameters:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class EUVRenderOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs


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


class FakeEUVResult:
    def __init__(self, cube: np.ndarray, *, channels: list[str]):
        self.flux_corona = cube
        self.flux_tr = cube * 2.0
        self.response = SimpleNamespace(channels=channels)


class FakeSDKWithEUV(FakeSDK):
    last_euv_options = None

    @staticmethod
    def render_euv_maps(options):
        FakeSDKWithEUV.last_euv_options = options
        cube = np.stack(
            [
                np.full((2, 3), 2.0, dtype=float),
                np.full((2, 3), 4.0, dtype=float),
            ],
            axis=-1,
        )
        return FakeEUVResult(cube, channels=["A94", "A171"])


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
    """Render a single-frequency microwave map through the adapter."""
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
    """Reject gxrender outputs that are not single-frequency cubes."""
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
    """Surface missing gxrender dependencies during adapter setup."""
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
    """Forward the named observer into shared gxrender workflow args."""
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
    """Use the SDK workflow path when persistent output is requested."""
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


def test_gxrender_euv_adapter_renders_single_channel_sum_map(monkeypatch) -> None:
    """Render a single-channel EUV map and sum corona+TR flux."""
    monkeypatch.setattr(gxrender_adapter, "_load_gxrender_sdk", lambda: FakeSDKWithEUV)
    monkeypatch.setattr(gxrender_adapter, "_resolve_default_euv_response_sav", lambda *, instrument: Path(f"{instrument}.sav"))

    adapter = GXRenderEUVAdapter(
        model_path="model.h5",
        channel="171",
        instrument="AIA",
        ebtel_path="ebtel.sav",
        tbase=1e6,
        nbase=1e8,
        a=0.3,
        b=2.7,
    )

    image = adapter.render(0.0217)

    assert image.shape == (2, 3)
    assert np.allclose(image, 12.0)
    assert FakeSDKWithEUV.last_euv_options is not None
    assert FakeSDKWithEUV.last_euv_options.kwargs["channels"] == ["171"]
    assert FakeSDKWithEUV.last_euv_options.kwargs["instrument"] == "AIA"


def test_build_tr_region_mask_from_blos_uses_absolute_threshold() -> None:
    blos = np.asarray([[500.0, -1200.0], [np.nan, 1800.0]], dtype=float)

    mask = build_tr_region_mask_from_blos(blos, threshold_gauss=1000.0)

    expected = np.asarray([[False, True], [False, True]], dtype=bool)
    np.testing.assert_array_equal(mask, expected)


def test_recombine_euv_components_applies_tr_mask() -> None:
    coronal = np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=float)
    tr = np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=float)
    mask = np.asarray([[True, False], [False, True]], dtype=bool)

    combined = recombine_euv_components(coronal, tr, tr_region_mask=mask)

    np.testing.assert_allclose(combined, np.asarray([[11.0, 1.0], [1.0, 41.0]], dtype=float))


def test_gxrender_euv_adapter_render_components_returns_masked_components(monkeypatch) -> None:
    monkeypatch.setattr(gxrender_adapter, "_load_gxrender_sdk", lambda: FakeSDKWithEUV)
    monkeypatch.setattr(gxrender_adapter, "_resolve_default_euv_response_sav", lambda *, instrument: Path(f"{instrument}.sav"))

    tr_mask = np.asarray([[True, False, True], [False, True, False]], dtype=bool)
    adapter = GXRenderEUVAdapter(
        model_path="model.h5",
        channel="171",
        instrument="AIA",
        ebtel_path="ebtel.sav",
        tbase=1e6,
        nbase=1e8,
        a=0.3,
        b=2.7,
        tr_region_mask=tr_mask,
    )

    payload = adapter.render_components(0.0217)

    np.testing.assert_allclose(payload["flux_corona"], np.full((2, 3), 4.0, dtype=float))
    np.testing.assert_allclose(payload["flux_tr"], np.full((2, 3), 8.0, dtype=float))
    np.testing.assert_array_equal(payload["tr_region_mask"], tr_mask)
    np.testing.assert_allclose(
        payload["rendered"],
        np.asarray([[12.0, 4.0, 12.0], [4.0, 12.0, 4.0]], dtype=float),
    )


def test_gxrender_euv_adapter_sanitizes_nonfinite_pixels(monkeypatch) -> None:
    """Clamp localized non-finite EUV pixels so the optimizer sees finite maps."""

    class FakeSDKWithNonFinite(FakeSDK):
        @staticmethod
        def render_euv_maps(options):
            del options
            cube = np.stack(
                [
                    np.array([[1.0, np.inf], [np.nan, 4.0]], dtype=float),
                    np.array([[10.0, 20.0], [30.0, 40.0]], dtype=float),
                ],
                axis=-1,
            )
            return FakeEUVResult(cube, channels=["A94", "A171"])

    monkeypatch.setattr(gxrender_adapter, "_load_gxrender_sdk", lambda: FakeSDKWithNonFinite)
    monkeypatch.setattr(gxrender_adapter, "_resolve_default_euv_response_sav", lambda *, instrument: Path(f"{instrument}.sav"))

    adapter = GXRenderEUVAdapter(
        model_path="model.h5",
        channel="94",
        instrument="AIA",
        ebtel_path="ebtel.sav",
        tbase=1e6,
        nbase=1e8,
        a=0.3,
        b=2.7,
    )

    image = adapter.render(0.0217)

    assert np.isfinite(image).all()
    np.testing.assert_allclose(image, np.array([[3.0, 12.0], [0.0, 12.0]], dtype=float))
