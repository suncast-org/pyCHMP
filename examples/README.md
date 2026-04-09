# pyCHMP Examples

This folder contains user-facing runnable workflows and validation applications.

## Scripts

- `estimate_map_noise_cli.py`
  - Command-line tool to estimate noise from observational FITS maps.
  - Supports multiple estimation methods: histogram_clip, offlimb_mad.
  - Useful for understanding noise characteristics before fitting.
  - No additional dependencies beyond pyCHMP core.

- `fit_q0_obs_map.py`
  - Fit Q0 to real observational maps (EOVSA).
  - Requires explicit inputs: observational FITS map and model H5.
  - Loads full-disk FITS maps, estimates background noise, crops/regrids the
    observation onto the saved model-aligned FOV, then runs Q0 optimization.
  - Saves fitting results to H5/PNG artifacts and reports fit diagnostics.

- `validate_synthetic_q0_recovery.py`
  - Synthetic renderer sanity check.
  - No gxrender dependency required.

- `scan_synthetic_ab_grid.py`
  - Synthetic rectangular `(a, b)` grid scan using the nested Q0 fitter.
  - No gxrender dependency required.
  - Useful for validating summary-grid behavior before running real model scans.

- `scan_ab_obs_map.py`
  - Real observational rectangular `(a, b)` grid scan against an EOVSA FITS map
    plus matching model H5.
  - Produces one consolidated H5 file containing the full scan summary and
    per-point best-fit products.

- `plot_ab_scan_artifacts.py`
  - On-demand plotting utility for the consolidated `scan_ab_obs_map.py` H5
    output.
  - Can generate a selected-point solution panel and a grid-summary heatmap
    figure.

- `demo_gxrender_mw_adapter.py`
  - Demonstrates direct use of `GXRenderMWAdapter` with real model input.
  - Requires gxrender and valid model/EBTEL inputs.

- `validate_q0_recovery.py`
  - End-to-end validation workflow with configurable observer, PSF, and noise.
  - Produces optional visual artifacts (H5/PNG).

## Usage

Run from repository root:

```bash
python examples/estimate_map_noise_cli.py /path/to/eovsa_map.fits
python examples/estimate_map_noise_cli.py /path/to/eovsa_map.fits --method histogram_clip
python examples/estimate_map_noise_cli.py /path/to/eovsa_map.fits --all-methods
```

```bash
python examples/fit_q0_obs_map.py /path/to/eovsa_map.fits /path/to/model.h5 \
  --ebtel-path /path/to/ebtel.sav
```

```bash
python examples/validate_synthetic_q0_recovery.py
```

```bash
python examples/scan_synthetic_ab_grid.py
```

```bash
python examples/scan_ab_obs_map.py /path/to/eovsa_map.fits /path/to/model.h5 \
  --ebtel-path /path/to/ebtel.sav \
  --a-values 0.0,0.3,0.6 \
  --b-values 2.1,2.4,2.7
```

```bash
python examples/plot_ab_scan_artifacts.py /path/to/ab_scan.h5 --show-plot
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

## Tracked Launcher Scripts

For the heavier manual workflows, tracked launchers live in `scripts/unix/`
and `scripts/windows/`:

- `scripts/unix/fit_q0_obs_map_options_test.sh`
  - Wraps `examples/fit_q0_obs_map.py` with a commented option block for easy
    interactive toggling.
  - Resolves test data from a sibling `pyGXrender-test-data` checkout by
    default.
  - Supports `--dry-run` to print the resolved command without starting a fit.

- `scripts/unix/validate_q0_recovery_options_test.sh`
  - Wraps `examples/validate_q0_recovery.py` with the same style of line-by-line
    option editing.
  - Resolves the matching model and EBTEL input from sibling test data.
  - Supports `--dry-run` to print the resolved command without starting a run.

- `scripts/unix/scan_ab_obs_map_options_test.sh`
  - Wraps `examples/scan_ab_obs_map.py` with one option per line for easy grid
    editing.
  - Resolves the matching EOVSA map, model, and EBTEL input from sibling test
    data.
  - Supports `--dry-run` to print the resolved command without starting the scan.

- `scripts/windows/*.cmd`
  - Windows counterparts intended for `cmd.exe` usage and the viewer Run tab.

## Relationship to tests

- `tests/` should contain deterministic automated checks.
- `examples/` contains heavier, user-run validation workflows.
