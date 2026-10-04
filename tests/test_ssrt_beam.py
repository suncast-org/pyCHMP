from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from pychmp.ssrt_beam import fit_ssrt_beam_ellipse, make_ssrt_beam


def _idl_beam_path() -> Path | None:
    env = os.environ.get("PYCHMP_SSRT_IDL_BEAM")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env).expanduser())
    repo = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            repo.parent / "SRH-NORH-4CHMP" / "ssrt2oct2012_idl_beam.fits",
            Path("/Users/gelu/code/SUNCAST-ORG/SRH-NORH-4CHMP/ssrt2oct2012_idl_beam.fits"),
        ]
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


_IDL_BEAM = _idl_beam_path()
# GetSSRTangles output for map time "2-Oct-2012 03:16:23.917"
_IDL_ANGLES = (19.157769, 25.826949, -27.850994, -85.062927)
_IDL_FIT = (14.500631, 7.3190091, 0.79709549)

requires_idl_beam = pytest.mark.skipif(
    _IDL_BEAM is None, reason="SSRT IDL beam FITS fixture not found"
)


@requires_idl_beam
def test_make_ssrt_beam_matches_idl_fixture() -> None:
    d_ew, d_ns, p_ew, p_ns = _IDL_ANGLES
    _x, _y, beam = make_ssrt_beam(
        d_ew, d_ns, p_ew, p_ns, nx=50, ny=50, dx_arcsec=1.0, dy_arcsec=1.0
    )
    idl = np.asarray(fits.getdata(_IDL_BEAM), dtype=float).T
    assert beam.shape == idl.shape
    assert float(np.max(np.abs(beam - idl))) < 1e-7
    assert float(np.corrcoef(beam.ravel(), idl.ravel())[0, 1]) > 1.0 - 1e-12


@requires_idl_beam
def test_fit_ssrt_beam_matches_idl_beamfitssrt() -> None:
    d_ew, d_ns, p_ew, p_ns = _IDL_ANGLES
    x, y, beam = make_ssrt_beam(
        d_ew, d_ns, p_ew, p_ns, nx=50, ny=50, dx_arcsec=1.0, dy_arcsec=1.0
    )
    sx, sy, theta = fit_ssrt_beam_ellipse(beam, x, y)
    py_axes = sorted([sx, sy], reverse=True)
    id_axes = sorted([_IDL_FIT[0], _IDL_FIT[1]], reverse=True)
    assert py_axes[0] == pytest.approx(id_axes[0], rel=1e-4)
    assert py_axes[1] == pytest.approx(id_axes[1], rel=1e-4)
    if sx >= sy:
        pa = theta
    else:
        pa = theta + 0.5 * np.pi
    # wrap to (-pi/2, pi/2]
    pa = float(np.arctan2(np.sin(pa), np.cos(pa)))
    id_pa = float(_IDL_FIT[2]) if _IDL_FIT[0] >= _IDL_FIT[1] else float(_IDL_FIT[2] + 0.5 * np.pi)
    id_pa = float(np.arctan2(np.sin(id_pa), np.cos(id_pa)))
    assert pa == pytest.approx(id_pa, abs=1e-3)
