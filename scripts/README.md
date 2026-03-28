# pyCHMP Script Launchers

This directory contains tracked shell launchers for the heavier manual pyCHMP
workflows. They wrap the Python entry points in `../examples/` with editable
option blocks and shared test-data path resolution.

## Scripts

- `fit_q0_obs_map_options_test.sh`
  - Wraps `examples/fit_q0_obs_map.py`.
  - Runs Q0 fitting against a real observational EOVSA FITS map plus a matching
    model H5.

- `validate_q0_recovery_options_test.sh`
  - Wraps `examples/validate_q0_recovery.py`.
  - Runs the synthetic recovery workflow against a matching model H5.

- `scan_ab_obs_map_options_test.sh`
  - Wraps `examples/scan_ab_obs_map.py`.
  - Runs a real observational rectangular `(a, b)` scan against the matching
    model H5 and writes a consolidated H5 scan file.

## Expected Layout

By default these launchers expect sibling repositories:

```text
<workspace>/
  pyCHMP/
  pyGXrender-test-data/
```

The launchers resolve shared inputs from `pyGXrender-test-data`, including:

- `raw/models/models_<timestamp>/`
- `raw/eovsa_maps/eovsa_maps_<timestamp>/`
- `raw/ebtel/ebtel_gxsimulator_euv/ebtel.sav`

You can override the default test-data location with:

```bash
PYCHMP_TESTDATA_REPO=/path/to/pyGXrender-test-data
```

## Common Usage

Run from anywhere:

```bash
pyCHMP/scripts/fit_q0_obs_map_options_test.sh
pyCHMP/scripts/validate_q0_recovery_options_test.sh
pyCHMP/scripts/scan_ab_obs_map_options_test.sh
```

All tracked launchers support:

```bash
--dry-run
```

This prints the resolved inputs and the exact Python command, then exits
without starting a render, fit, or scan.

## Editing Model / Map Choices

Each script is intentionally written with one option per line so the common
workflow is simple:

- comment or uncomment the default map/model selection lines
- comment or uncomment optional CLI flags
- rerun the launcher

For more targeted overrides, use environment variables such as:

```bash
MODEL_H5_PATH=/path/to/model.h5
OBS_FITS_PATH=/path/to/map.fits
EBTEL_PATH=/path/to/ebtel.sav
PYTHON_BIN=/path/to/python
```

## Notes

- These are developer-facing/manual run harnesses, not installed CLI commands.
- The actual workflow implementations remain in `../examples/`.
- Detailed per-script behavior is also documented inline at the top of each
  launcher.
