from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import pychmp.gxrender_adapter as gxrender_adapter
from pychmp.gxrender_adapter import GXRenderMWAdapter


class FakeSDK:
    class MapGeometry:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class CoronalPlasmaParameters:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class MWRenderOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    @staticmethod
    def render_mw_maps(options):
        q0 = options.kwargs["plasma"].kwargs["q0"]
        cube = np.full((2, 3, 1), q0, dtype=float)
        return SimpleNamespace(ti=cube)


class WrongCubeSDK(FakeSDK):
    @staticmethod
    def render_mw_maps(options):
        return SimpleNamespace(ti=np.ones((2, 3), dtype=float))


def test_gxrender_mw_adapter_renders_single_frequency_map(monkeypatch) -> None:
    monkeypatch.setattr(gxrender_adapter, "_load_gxrender_sdk", lambda: FakeSDK)

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
    monkeypatch.setattr(gxrender_adapter, "_load_gxrender_sdk", lambda: WrongCubeSDK)

    adapter = GXRenderMWAdapter(model_path="model.h5", frequency_ghz=5.8)

    with pytest.raises(ValueError, match="single-frequency TI cube"):
        adapter.render(0.1)


def test_gxrender_mw_adapter_surfaces_missing_dependency(monkeypatch) -> None:
    def raise_missing():
        raise ModuleNotFoundError("gxrender")

    monkeypatch.setattr(gxrender_adapter, "_load_gxrender_sdk", raise_missing)
    adapter = GXRenderMWAdapter(model_path="model.h5", frequency_ghz=5.8)

    with pytest.raises(ModuleNotFoundError, match="gxrender"):
        adapter.render(0.1)
