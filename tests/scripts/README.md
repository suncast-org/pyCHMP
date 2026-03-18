# pyCHMP Demo Scripts

This folder contains demonstration scripts that exercise pyCHMP APIs outside of pytest.

## Scripts

- `demo_synthetic_q0_recovery.py`
  - Uses a synthetic renderer and verifies that fitting recovers a planted Q0.
  - No gxrender dependency required.

- `demo_gxrender_mw_adapter.py`
  - Shows how to instantiate `GXRenderMWAdapter` for a real model path.
  - Renders a single brightness temperature map and optionally saves to HDF5 for visualization.
  - Requires gxrender to be installed and valid model/EBTEL inputs.

- `demo_earth_eovsa_q0_recovery.py`
  - Runs an end-to-end Q0 recovery setup in an Earth-oriented SAV model frame.
  - Uses geometry center=(-257, -233) arcsec, dx=dy=2.5 arcsec, nx=ny=64.
  - Applies an EOVSA-like Gaussian beam (Bmaj=5.77", Bmin=5.77", BPA=-17.5 deg).
  - Adds Gaussian noise to the pseudo-observed map by default (`noise-frac=0.02`, `noise-seed=12345`).

## Usage

Run from repository root:

```bash
python tests/scripts/demo_synthetic_q0_recovery.py
```

```bash
python tests/scripts/demo_gxrender_mw_adapter.py \
  --model-path /path/to/test.chr.h5 \
  --ebtel-path /path/to/ebtel.sav \
  --frequency-ghz 5.8 \
  --pixel-scale-arcsec 2.0 \
  --q0 0.0217
```

To save and visualize the rendered map:

```bash
python tests/scripts/demo_gxrender_mw_adapter.py \
  --model-path /path/to/test.chr.h5 \
  --ebtel-path /path/to/ebtel.sav \
  --frequency-ghz 5.8 \
  --pixel-scale-arcsec 2.0 \
  --q0 0.0217 \
  --output-h5 /tmp/rendered_map.h5

# View the result:
gxrender-map-view /tmp/rendered_map.h5
```

```bash
python tests/scripts/demo_earth_eovsa_q0_recovery.py \
  --model-path /path/to/test.chr.sav \
  --ebtel-path /path/to/ebtel.sav \
  --q0-true 0.0217 \
  --q0-min 0.005 \
  --q0-max 0.05 \
  --noise-frac 0.02 \
  --noise-seed 12345 \
  --artifacts-dir /tmp/pychmp_artifacts \
  --save-raw-h5 /tmp/earth_eovsa_raw.h5
```

Note: for H5 inputs, `gxrender-mw --observer earth ...` is the native way to force Earth observer orientation.

Integration test (disabled by default because it requires local gxrender + test fixtures):

```bash
PYCHMP_RUN_GXRENDER_INTEGRATION=1 \
python -m pytest -q tests/test_integration_earth_eovsa_q0.py
```

To print quantitative diagnostics (truth/recovered q0, error, metrics, tolerances):

```bash
PYCHMP_RUN_GXRENDER_INTEGRATION=1 PYCHMP_VERBOSE_INTEGRATION=1 \
python -m pytest -s -q tests/test_integration_earth_eovsa_q0.py
```

Optional artifact mode for integration test:

```bash
PYCHMP_RUN_GXRENDER_INTEGRATION=1 \
PYCHMP_VERBOSE_INTEGRATION=1 \
PYCHMP_SAVE_INTEGRATION_ARTIFACTS=1 \
PYCHMP_ARTIFACTS_DIR=/tmp/pychmp_artifacts \
PYCHMP_SAVE_INTEGRATION_PNG=1 \
python -m pytest -s -q tests/test_integration_earth_eovsa_q0.py --disable-warnings
```

This writes:
- `earth_eovsa_q0_artifacts.npz` with observed_clean, observed_noisy, modeled_best, residual, diagnostics
- `earth_eovsa_q0_artifacts.png` panel with colorbars (if matplotlib is available)

Verbose diagnostics also report whether PSF and noise were applied, including beam parameters and noise settings.
The integration test runs two noise regimes by default: `noise_frac=0.02` and `noise_frac=0.05`.
