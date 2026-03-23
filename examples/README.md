# pyCHMP Examples

This folder contains user-facing runnable workflows and validation applications.

## Scripts

- `estimate_map_noise_cli.py`
  - Command-line tool to estimate noise from observational FITS maps.
  - Supports multiple estimation methods: histogram_clip, offlimb_mad.
  - Useful for understanding noise characteristics before fitting.
  - No additional dependencies beyond pyCHMP core.

- `validate_synthetic_q0_recovery.py`
  - Synthetic renderer sanity check.
  - No gxrender dependency required.

- `demo_gxrender_mw_adapter.py`
  - Demonstrates direct use of `GXRenderMWAdapter` with real model input.
  - Requires gxrender and valid model/EBTEL inputs.

- `validate_q0_recovery.py`
  - End-to-end validation workflow with configurable observer, PSF, and noise.
  - Produces optional visual artifacts (H5/PNG).

## Usage

Run from repository root:

```bash
# Estimate noise from observational map (all methods)
python examples/estimate_map_noise_cli.py /path/to/eovsa_map.fits

# Using specific method
python examples/estimate_map_noise_cli.py /path/to/eovsa_map.fits --method histogram_clip

# Comparison of all methods
python examples/estimate_map_noise_cli.py /path/to/eovsa_map.fits --all-methods
```

```bash
python examples/validate_synthetic_q0_recovery.py
```

```bash
python examples/demo_gxrender_mw_adapter.py \
  --model-path /path/to/test.chr.h5 \
  --ebtel-path /path/to/ebtel.sav \
  --frequency-ghz 5.8 \
  --pixel-scale-arcsec 2.0 \
  --q0 0.0217
```

```bash
python examples/validate_q0_recovery.py \
  --model-path /path/to/test.chr.sav \
  --ebtel-path /path/to/ebtel.sav \
  --q0-true 0.0217 \
  --q0-min 0.005 \
  --q0-max 0.05 \
  --noise-frac 0.02 \
  --noise-seed 12345 \
  --artifacts-dir /tmp/pychmp_artifacts
```

## Relationship to tests

- `tests/` should contain deterministic automated checks.
- `examples/` contains heavier, user-run validation workflows.
