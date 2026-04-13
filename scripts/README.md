# pyCHMP Script Launchers

This directory contains tracked shell launchers for the heavier manual pyCHMP
workflows. They wrap the Python entry points in `../examples/` with editable
option blocks and shared test-data path resolution.

It does not currently contain a `python/` subfolder by design. Standalone
Python workflow scripts live under `../examples/`, and the heavier real-data
Python validation scripts live under `../examples/python/`.

## Layout

- `unix/`
  - Canonical Unix shell launchers (`.sh`).
- `windows/`
  - Windows Command Prompt launchers (`.cmd`).

Python entry point locations:

- `../examples/`
  - Primary user-facing Python workflows.
- `../examples/python/`
  - Heavier standalone Python validation scripts that are not part of the
    default automated test suite.

## Scripts

- `unix/fit_q0_obs_map_options_test.sh`
  - Wraps `examples/fit_q0_obs_map.py`.
  - Runs Q0 fitting against a real observational EOVSA FITS map plus a matching
    model H5.

- `unix/validate_q0_recovery_options_test.sh`
  - Wraps `examples/validate_q0_recovery.py`.
  - Runs the synthetic recovery workflow against a matching model H5.

- `unix/scan_ab_obs_map_options_test.sh`
  - Wraps `examples/scan_ab_obs_map.py`.
  - Runs a real observational rectangular `(a, b)` scan against the matching
    model H5 and writes a consolidated H5 scan file.

- `unix/benchmark_scan_ab_obs_map.sh`
  - Wraps `examples/benchmark_scan_ab_obs_map.py`.
  - Runs a serial-plus-process-pool 3x3 benchmark using the same resolved
    EOVSA/model/EBTEL inputs as the scan options-test launcher.

- `unix/adaptive_ab_search_single_frequency_options_test.sh`
  - Wraps `examples/python/adaptive_ab_search_single_frequency.py`.
  - Runs the adaptive real-data single-frequency `(a, b)` search against the
    matching EOVSA/model/EBTEL inputs and writes a sparse live-update artifact.

- `windows/fit_q0_obs_map_options_test.cmd`
- `windows/validate_q0_recovery_options_test.cmd`
- `windows/scan_ab_obs_map_options_test.cmd`
- `windows/benchmark_scan_ab_obs_map.cmd`
- `windows/adaptive_ab_search_single_frequency_options_test.cmd`
  - Windows counterparts of the Unix launchers, intended for `cmd.exe` and the
    viewer Run tab on Windows.

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
pyCHMP/scripts/unix/fit_q0_obs_map_options_test.sh
pyCHMP/scripts/unix/validate_q0_recovery_options_test.sh
pyCHMP/scripts/unix/scan_ab_obs_map_options_test.sh
pyCHMP/scripts/unix/adaptive_ab_search_single_frequency_options_test.sh
```

From Git Bash on Windows, use the Unix launchers above. They are shell scripts
and now run directly with either:

```bash
./scripts/unix/adaptive_ab_search_single_frequency_options_test.sh --dry-run
```

or:

```bash
bash scripts/unix/adaptive_ab_search_single_frequency_options_test.sh --dry-run
```

On Windows `cmd.exe`:

```bat
pyCHMP\scripts\windows\fit_q0_obs_map_options_test.cmd
pyCHMP\scripts\windows\validate_q0_recovery_options_test.cmd
pyCHMP\scripts\windows\scan_ab_obs_map_options_test.cmd
pyCHMP\scripts\windows\adaptive_ab_search_single_frequency_options_test.cmd
```

The `windows/*.cmd` launchers are for `cmd.exe` or PowerShell, not Git Bash.
If you are already inside Git Bash, prefer the matching `scripts/unix/*.sh`
launcher instead of calling the `.cmd` file directly.

All tracked launchers support:

```bash
--dry-run
```

This prints the resolved inputs and the exact Python command, then exits
without starting a render, fit, or scan.

Benchmark launchers also accept the same style of pass-through arguments, for
example:

```bash
scripts/unix/benchmark_scan_ab_obs_map.sh --repeats 1 --worker-counts 1,2,3,4,5,6,7,8,9
```

```bat
scripts\windows\benchmark_scan_ab_obs_map.cmd --repeats 1 --worker-counts 1,2,3,4,5,6,7,8,9
```

The recorded real-data benchmark bundle produced from those launchers is kept
under `reports/parallel benchmark test/`. That directory is intended to be the
portable home for the raw CSV, generated reports, plot, per-run benchmark
artifacts, and the benchmark-specific report generator:

- `reports/parallel benchmark test/generate_scan_ab_obs_map_benchmark_report.py`

The scan launchers reuse the same artifact by default so reruns resume and
skip points already present in the artifact. For a fresh timestamped artifact,
set `PYCHMP_TIMESTAMP_ARTIFACTS=1`. To force a specific artifact file, set
`ARTIFACT_H5`.

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
ARTIFACT_H5=/path/to/scan_output.h5
ARTIFACTS_STEM=custom_scan_name
ARTIFACTS_DIR=/path/to/output_dir
PYCHMP_TIMESTAMP_ARTIFACTS=1
```

## Notes

- These are developer-facing/manual run harnesses, not installed CLI commands.
- The actual workflow implementations remain in `../examples/` and `../examples/python/`.
- Detailed per-script behavior is also documented inline at the top of each
  launcher.
