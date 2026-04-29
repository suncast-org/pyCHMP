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
  - Wraps `examples/python/adaptive_ab_search_single_observation.py`.
  - Runs the adaptive real-data single-slice `(a, b)` search against the
    matching observation/model/EBTEL inputs and writes a sparse live-update artifact.

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

Choose the launcher family that matches the shell you are currently using:

- POSIX shell launchers: `scripts/unix/*.sh`
  - Use from `bash`, `zsh`, Linux/macOS terminals, Git Bash, or similar Unix-like shells.
- Windows launchers: `scripts/windows/*.cmd`
  - Use from `cmd.exe`, PowerShell, or the viewer Run tab on Windows.

You can invoke the launchers either from the repository root or from any other
working directory by using an absolute or repo-relative path.

From a POSIX shell:

```bash
pyCHMP/scripts/unix/fit_q0_obs_map_options_test.sh
pyCHMP/scripts/unix/validate_q0_recovery_options_test.sh
pyCHMP/scripts/unix/scan_ab_obs_map_options_test.sh
pyCHMP/scripts/unix/adaptive_ab_search_single_frequency_options_test.sh
```

Equivalent invocation from inside the `pyCHMP/` repository itself:

```bash
./scripts/unix/adaptive_ab_search_single_frequency_options_test.sh
```

If the script is not marked executable in your current checkout, invoke it
explicitly through the shell:

```bash
bash scripts/unix/adaptive_ab_search_single_frequency_options_test.sh --dry-run
```

From `cmd.exe` or PowerShell on Windows:

```bat
pyCHMP\scripts\windows\fit_q0_obs_map_options_test.cmd
pyCHMP\scripts\windows\validate_q0_recovery_options_test.cmd
pyCHMP\scripts\windows\scan_ab_obs_map_options_test.cmd
pyCHMP\scripts\windows\adaptive_ab_search_single_frequency_options_test.cmd
```

The `windows/*.cmd` launchers are intended for native Windows shells.
If you are already inside a POSIX-style shell on Windows, prefer the matching
`scripts/unix/*.sh` launcher instead of calling the `.cmd` file directly.

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

## `fit_q0_obs_map_options_test.sh`

The Unix launcher:

- `scripts/unix/fit_q0_obs_map_options_test.sh`

and the Windows counterpart:

- `scripts/windows/fit_q0_obs_map_options_test.cmd`

both support:

- MW fitting from an external observational FITS map
- EUV/UV fitting from an internal model refmap such as `AIA_171`
- `--obs-source external_fits|model_refmap`
- `--obs-map-id AIA_171`
- `--euv-instrument AIA`
- `--euv-response-sav /path/to/resp_aia_*.sav`
- `--tr-mask-bmin-gauss 1000`
- `--metrics-mask-threshold 0.5`
- `--metrics-mask-fits /path/to/mask.fits`

Example MW fit from the default external EOVSA FITS path:

```bash
bash /Users/gelu/code/SUNCAST-ORG/pyCHMP/scripts/unix/fit_q0_obs_map_options_test.sh
```

Equivalent Windows MW fit:

```bat
pyCHMP\scripts\windows\fit_q0_obs_map_options_test.cmd
```

Example EUV fit against the internal `AIA_171` refmap:

```bash
PYTHON_BIN=/Users/gelu/miniforge3/envs/suncast/bin/python \
bash /Users/gelu/code/SUNCAST-ORG/pyCHMP/scripts/unix/fit_q0_obs_map_options_test.sh \
  --obs-source model_refmap \
  --obs-map-id AIA_171 \
  --tr-mask-bmin-gauss 1000 \
  --metrics-mask-threshold 0.5
```

Equivalent Windows EUV fit:

```bat
pyCHMP\scripts\windows\fit_q0_obs_map_options_test.cmd ^
  --obs-source model_refmap ^
  --obs-map-id AIA_171 ^
  --tr-mask-bmin-gauss 1000 ^
  --metrics-mask-threshold 0.5
```

## `validate_q0_recovery_options_test.sh`

The Unix launcher:

- `scripts/unix/validate_q0_recovery_options_test.sh`

and the Windows counterpart:

- `scripts/windows/validate_q0_recovery_options_test.cmd`

both support:

- MW synthetic recovery via `--domain mw`
- EUV/UV synthetic recovery via `--domain euv`
- `--tr-mask-bmin-gauss 1000`
- `--metrics-mask-threshold 0.5`
- `--metrics-mask-fits /path/to/mask.fits`

Example MW recovery run:

```bash
bash /Users/gelu/code/SUNCAST-ORG/pyCHMP/scripts/unix/validate_q0_recovery_options_test.sh \
  --domain mw
```

Equivalent Windows MW recovery run:

```bat
pyCHMP\scripts\windows\validate_q0_recovery_options_test.cmd ^
  --domain mw
```

Example EUV recovery run:

```bash
PYTHON_BIN=/Users/gelu/miniforge3/envs/suncast/bin/python \
bash /Users/gelu/code/SUNCAST-ORG/pyCHMP/scripts/unix/validate_q0_recovery_options_test.sh \
  --domain euv \
  --tr-mask-bmin-gauss 1000 \
  --metrics-mask-threshold 0.5
```

Equivalent Windows EUV recovery run:

```bat
pyCHMP\scripts\windows\validate_q0_recovery_options_test.cmd ^
  --domain euv ^
  --tr-mask-bmin-gauss 1000 ^
  --metrics-mask-threshold 0.5
```

## `benchmark_scan_ab_obs_map.sh`

The Unix launcher:

- `scripts/unix/benchmark_scan_ab_obs_map.sh`

wraps:

- `examples/benchmark_scan_ab_obs_map.py`

and now supports both:

- MW benchmark scans from an external observational FITS map
- EUV/UV benchmark scans from an internal model refmap such as `AIA_171`

The launcher resolves shared defaults from `pyGXrender-test-data`:

- the canonical 2020-11-26 EOVSA fixture file found under `raw/eovsa_maps/`
- the matching 2020-11-26 CHR model file found under `raw/models/`
- fixed EBTEL path under `raw/ebtel/ebtel_gxsimulator_euv/ebtel.sav`
- newest dated response folder under `raw/responses/` for EUV/UV mode

Important benchmark-specific options:

- `--obs-source external_fits|model_refmap`
- `--obs-map-id AIA_171`
- `--euv-instrument AIA`
- `--euv-response-sav /path/to/resp_aia_*.sav`
- `--tr-mask-bmin-gauss 1000`
- `--metrics-mask-threshold 0.5`
- `--metrics-mask-fits /path/to/mask.fits`
- `--repeats 1`
- `--worker-counts 1,2,3,4,5,6,7,8,9`

The benchmark launcher writes a CSV summary plus one per-run artifact per mode,
worker count, and repeat. The generated benchmark artifact filenames encode the
domain (`mw` or `euv`) so mixed-domain benchmark bundles stay distinguishable.

The Windows counterpart:

- `scripts/windows/benchmark_scan_ab_obs_map.cmd`

accepts the same observation-selection and mask options.

Example MW benchmark from the default external EOVSA FITS path:

```bash
bash /Users/gelu/code/SUNCAST-ORG/pyCHMP/scripts/unix/benchmark_scan_ab_obs_map.sh \
  --repeats 1 \
  --worker-counts 1,2,4
```

Equivalent Windows MW benchmark:

```bat
pyCHMP\scripts\windows\benchmark_scan_ab_obs_map.cmd ^
  --repeats 1 ^
  --worker-counts 1,2,4
```

Example MW benchmark with a tighter metrics mask:

```bash
bash /Users/gelu/code/SUNCAST-ORG/pyCHMP/scripts/unix/benchmark_scan_ab_obs_map.sh \
  --repeats 1 \
  --worker-counts 1,2,4 \
  --metrics-mask-threshold 0.5
```

Example EUV benchmark against the internal `AIA_171` refmap:

```bash
PYTHON_BIN=/Users/gelu/miniforge3/envs/suncast/bin/python \
bash /Users/gelu/code/SUNCAST-ORG/pyCHMP/scripts/unix/benchmark_scan_ab_obs_map.sh \
  --repeats 1 \
  --worker-counts 1,2,4 \
  --obs-source model_refmap \
  --obs-map-id AIA_171 \
  --tr-mask-bmin-gauss 1000 \
  --metrics-mask-threshold 0.5
```

Example EUV benchmark using an explicit metrics-mask FITS file:

```bash
PYTHON_BIN=/Users/gelu/miniforge3/envs/suncast/bin/python \
bash /Users/gelu/code/SUNCAST-ORG/pyCHMP/scripts/unix/benchmark_scan_ab_obs_map.sh \
  --repeats 1 \
  --worker-counts 1,2,4 \
  --obs-source model_refmap \
  --obs-map-id AIA_171 \
  --tr-mask-bmin-gauss 1000 \
  --metrics-mask-fits /path/to/metrics_mask.fits
```

Example EUV dry-run to inspect the resolved benchmark and scan commands without
starting the benchmark:

```bash
PYTHON_BIN=/Users/gelu/miniforge3/envs/suncast/bin/python \
bash /Users/gelu/code/SUNCAST-ORG/pyCHMP/scripts/unix/benchmark_scan_ab_obs_map.sh \
  --dry-run \
  --repeats 1 \
  --worker-counts 1,2,4 \
  --obs-source model_refmap \
  --obs-map-id AIA_171 \
  --tr-mask-bmin-gauss 1000 \
  --metrics-mask-threshold 0.5
```

Equivalent Windows EUV benchmark example:

```bat
pyCHMP\scripts\windows\benchmark_scan_ab_obs_map.cmd ^
  --repeats 1 ^
  --worker-counts 1,2,4 ^
  --obs-source model_refmap ^
  --obs-map-id AIA_171 ^
  --tr-mask-bmin-gauss 1000 ^
  --metrics-mask-threshold 0.5
```

The scan launchers reuse the same artifact by default so reruns resume and
skip points already present in the artifact. For a fresh timestamped artifact,
set `PYCHMP_TIMESTAMP_ARTIFACTS=1`. To force a specific artifact file, set
`ARTIFACT_H5`.

## `scan_ab_obs_map_options_test.sh`

The Unix launcher:

- `scripts/unix/scan_ab_obs_map_options_test.sh`

wraps:

- `examples/scan_ab_obs_map.py`

and now supports both:

- MW scans from an external observational FITS map
- EUV/UV scans from an internal model refmap such as `AIA_171`

The launcher resolves shared defaults from `pyGXrender-test-data`:

- the canonical 2020-11-26 EOVSA fixture file found under `raw/eovsa_maps/`
- the matching 2020-11-26 CHR model file found under `raw/models/`
- fixed EBTEL path under `raw/ebtel/ebtel_gxsimulator_euv/ebtel.sav`
- newest dated response folder under `raw/responses/` for EUV/UV mode

Important scan-specific options:

- `--obs-source external_fits|model_refmap`
- `--obs-map-id AIA_171`
- `--euv-instrument AIA`
- `--euv-response-sav /path/to/resp_aia_*.sav`
- `--tr-mask-bmin-gauss 1000`
- `--metrics-mask-threshold 0.5`
- `--metrics-mask-fits /path/to/mask.fits`

The launcher reuses the same consolidated artifact by default so reruns resume.
For a fresh scan artifact:

- set `ARTIFACTS_STEM=custom_name`
- or set `PYCHMP_TIMESTAMP_ARTIFACTS=1`
- or force a specific file with `ARTIFACT_H5=/path/to/scan.h5`

The Windows counterpart:

- `scripts/windows/scan_ab_obs_map_options_test.cmd`

accepts the same observation-selection and mask options.

Example MW scan from the default external EOVSA FITS path:

```bash
bash /Users/gelu/code/SUNCAST-ORG/pyCHMP/scripts/unix/scan_ab_obs_map_options_test.sh
```

Equivalent Windows MW scan:

```bat
pyCHMP\scripts\windows\scan_ab_obs_map_options_test.cmd
```

Example MW scan with a tighter metrics mask:

```bash
bash /Users/gelu/code/SUNCAST-ORG/pyCHMP/scripts/unix/scan_ab_obs_map_options_test.sh \
  --metrics-mask-threshold 0.5
```

Example EUV scan against the internal `AIA_171` refmap:

```bash
PYTHON_BIN=/Users/gelu/miniforge3/envs/suncast/bin/python \
bash /Users/gelu/code/SUNCAST-ORG/pyCHMP/scripts/unix/scan_ab_obs_map_options_test.sh \
  --obs-source model_refmap \
  --obs-map-id AIA_171 \
  --tr-mask-bmin-gauss 1000 \
  --metrics-mask-threshold 0.5
```

Example EUV scan using an explicit metrics-mask FITS file:

```bash
PYTHON_BIN=/Users/gelu/miniforge3/envs/suncast/bin/python \
bash /Users/gelu/code/SUNCAST-ORG/pyCHMP/scripts/unix/scan_ab_obs_map_options_test.sh \
  --obs-source model_refmap \
  --obs-map-id AIA_171 \
  --tr-mask-bmin-gauss 1000 \
  --metrics-mask-fits /path/to/metrics_mask.fits
```

Example EUV dry-run to inspect the resolved Python command without starting the
scan:

```bash
PYTHON_BIN=/Users/gelu/miniforge3/envs/suncast/bin/python \
bash /Users/gelu/code/SUNCAST-ORG/pyCHMP/scripts/unix/scan_ab_obs_map_options_test.sh \
  --dry-run \
  --obs-source model_refmap \
  --obs-map-id AIA_171 \
  --tr-mask-bmin-gauss 1000 \
  --metrics-mask-threshold 0.5
```

Equivalent Windows EUV scan example:

```bat
pyCHMP\scripts\windows\scan_ab_obs_map_options_test.cmd ^
  --obs-source model_refmap ^
  --obs-map-id AIA_171 ^
  --tr-mask-bmin-gauss 1000 ^
  --metrics-mask-threshold 0.5
```

## `adaptive_ab_search_single_frequency_options_test.sh`

The Unix launcher:

- `scripts/unix/adaptive_ab_search_single_frequency_options_test.sh`

wraps:

- `examples/python/adaptive_ab_search_single_observation.py`

and now supports both:

- MW adaptive search from an external observational FITS map
- EUV/UV adaptive search from an internal model refmap such as `AIA_171`

The launcher resolves shared defaults from `pyGXrender-test-data`:

- the canonical 2020-11-26 EOVSA fixture file found under `raw/eovsa_maps/`
- the matching 2020-11-26 CHR model file found under `raw/models/`
- fixed EBTEL path under `raw/ebtel/ebtel_gxsimulator_euv/ebtel.sav`
- newest dated response folder under `raw/responses/` for EUV/UV mode

Important adaptive-search options:

- `--obs-source external_fits|model_refmap`
- `--obs-map-id AIA_171`
- `--euv-instrument AIA`
- `--euv-response-sav /path/to/resp_aia_*.sav`
- `--tr-mask-bmin-gauss 1000`
- `--metrics-mask-threshold 0.5`
- `--metrics-mask-fits /path/to/mask.fits`

The launcher reuses the same sparse artifact path by default so reruns resume.
For a fresh adaptive artifact:

- set `ARTIFACTS_STEM=custom_name`
- or set `PYCHMP_TIMESTAMP_ARTIFACTS=1`
- or force a specific file with `ARTIFACT_H5=/path/to/adaptive.h5`

The Windows counterpart:

- `scripts/windows/adaptive_ab_search_single_frequency_options_test.cmd`

accepts the same observation-selection, adaptive-range, and mask options.

Example MW adaptive search from the default external EOVSA FITS path:

```bash
bash /Users/gelu/code/SUNCAST-ORG/pyCHMP/scripts/unix/adaptive_ab_search_single_frequency_options_test.sh
```

Equivalent Windows MW adaptive search:

```bat
pyCHMP\scripts\windows\adaptive_ab_search_single_frequency_options_test.cmd
```

Example MW adaptive search with a tighter metrics mask:

```bash
bash /Users/gelu/code/SUNCAST-ORG/pyCHMP/scripts/unix/adaptive_ab_search_single_frequency_options_test.sh \
  --metrics-mask-threshold 0.5
```

Example EUV adaptive search against the internal `AIA_171` refmap:

```bash
PYTHON_BIN=/Users/gelu/miniforge3/envs/suncast/bin/python \
bash /Users/gelu/code/SUNCAST-ORG/pyCHMP/scripts/unix/adaptive_ab_search_single_frequency_options_test.sh \
  --obs-source model_refmap \
  --obs-map-id AIA_171 \
  --tr-mask-bmin-gauss 1000 \
  --metrics-mask-threshold 0.5
```

Example EUV adaptive search using an explicit metrics-mask FITS file:

```bash
PYTHON_BIN=/Users/gelu/miniforge3/envs/suncast/bin/python \
bash /Users/gelu/code/SUNCAST-ORG/pyCHMP/scripts/unix/adaptive_ab_search_single_frequency_options_test.sh \
  --obs-source model_refmap \
  --obs-map-id AIA_171 \
  --tr-mask-bmin-gauss 1000 \
  --metrics-mask-fits /path/to/metrics_mask.fits
```

Example EUV dry-run to inspect the resolved Python command without starting the
adaptive search:

```bash
PYTHON_BIN=/Users/gelu/miniforge3/envs/suncast/bin/python \
bash /Users/gelu/code/SUNCAST-ORG/pyCHMP/scripts/unix/adaptive_ab_search_single_frequency_options_test.sh \
  --dry-run \
  --obs-source model_refmap \
  --obs-map-id AIA_171 \
  --tr-mask-bmin-gauss 1000 \
  --metrics-mask-threshold 0.5
```

Example adaptive-search invocation from a POSIX shell:

```bash
./scripts/unix/adaptive_ab_search_single_frequency_options_test.sh \
  --artifact-h5 "/path/to/adaptive_ab_search_single_frequency.h5" \
  --a-min -4.5 \
  --a-max 3.0 \
  --b-min -3.0 \
  --b-max 4.8 \
  --b-start 0.0 \
  --q0-start 0.0001 \
  --max-bracket-steps 30 \
  --threshold-metric 1.3
```

Equivalent Windows launcher:

```bat
scripts\windows\adaptive_ab_search_single_frequency_options_test.cmd
```

Equivalent Windows EUV adaptive-search example:

```bat
pyCHMP\scripts\windows\adaptive_ab_search_single_frequency_options_test.cmd ^
  --obs-source model_refmap ^
  --obs-map-id AIA_171 ^
  --tr-mask-bmin-gauss 1000 ^
  --metrics-mask-threshold 0.5
```

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

On Windows shells, set the same variables with native syntax before launching
the `.cmd` wrapper, for example:

```bat
set MODEL_H5_PATH=C:\path\to\model.h5
set OBS_FITS_PATH=C:\path\to\map.fits
set EBTEL_PATH=C:\path\to\ebtel.sav
set PYTHON_BIN=C:\path\to\python.exe
set ARTIFACT_H5=C:\path\to\scan_output.h5
set ARTIFACTS_STEM=custom_scan_name
set ARTIFACTS_DIR=C:\path\to\output_dir
set PYCHMP_TIMESTAMP_ARTIFACTS=1
```

## Notes

- These are developer-facing/manual run harnesses, not installed CLI commands.
- The actual workflow implementations remain in `../examples/` and `../examples/python/`.
- Detailed per-script behavior is also documented inline at the top of each
  launcher.
