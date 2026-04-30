# pyCHMP Examples

This folder contains user-facing runnable workflows and validation applications.

The `python/` subfolder is reserved for standalone Python validation scripts
that are heavier than the normal automated test suite and may require real
external data or gxrender to be installed.

The `notebooks/` subfolder is reserved for developer and collaborator notebook
examples that may be added later by pull requests.

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
  - The consolidated artifact stores per-point final maps plus per-trial
    `q0`, `chi2`, `rho2`, and `eta2` histories so the viewer can inspect the
    selected metric history directly from the artifact.

- `benchmark_scan_ab_obs_map.py`
  - Benchmarks the real observational 3x3 `scan_ab_obs_map.py` workflow using
    pyGXrender-test-data inputs.
  - Compares a serial baseline against process-pool runs across user-selected
    worker counts.
  - Emits console CSV and optional CSV-file output for local performance notes.

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

- `python/validate_q0_recovery_earth_eovsa_psf.py`
  - Real-data gxrender validation script derived from the former integration smoke test.
  - Requires explicit model and EBTEL paths plus gxrender in the active environment.
  - Runs one or more fixed PSF-plus-noise recovery checks and exits nonzero if a profile fails.

- `python/adaptive_ab_search_single_observation.py`
  - Generic real-data adaptive local `(a, b)` search for a single observational map.
  - Supports both MW external FITS observations and EUV/UV model-refmap selections.
  - Defaults to the current 2.874 GHz EOVSA workflow when no explicit observation is supplied.
  - Persists each evaluated point into a sparse H5 artifact using the same
     point schema as the fixed-grid scan path, including full stored metric
     histories and per-trial map cubes when available.
  - Completed point results are appended as they arrive so the viewer can
    inspect progress while the run is active without waiting for a batch flush.
  - Supports `--dry-run` to resolve inputs and artifact locations without starting the search.

- `python/adaptive_ab_search_single_frequency.py`
  - Compatibility wrapper that preserves the original MW-facing entrypoint name.
  - Delegates to `python/adaptive_ab_search_single_observation.py`.

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

Execution policy notes for `scan_ab_obs_map.py`:

- `--execution-policy serial` is the default and keeps one worker for the full scan.
- `--execution-policy process-pool` is the explicit opt-in path and is honored literally even for a single pending point; if `--max-workers` is omitted, the worker count is capped only by the number of pending points and available CPUs.
- `--execution-policy auto` stays serial for very small scans, but otherwise chooses a conservative process-pool size from the number of pending points, available CPUs, and any `--max-workers` cap.

```bash
python examples/benchmark_scan_ab_obs_map.py /path/to/eovsa_map.fits /path/to/model.h5 \
  --ebtel-path /path/to/ebtel.sav \
  --repeats 1 \
  --worker-counts 1,2,3,4,5,6,7,8,9 \
  --csv-out /tmp/pychmp_scan_ab_obs_map_benchmark.csv
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

```bash
python examples/python/validate_q0_recovery_earth_eovsa_psf.py \
  --model-path /path/to/test.chr.sav \
  --ebtel-path /path/to/ebtel.sav
```

```bash
python examples/python/adaptive_ab_search_single_observation.py --dry-run
python examples/python/adaptive_ab_search_single_observation.py
python examples/python/adaptive_ab_search_single_frequency.py --dry-run
python examples/python/adaptive_ab_search_single_frequency.py
```

Using your own data with the adaptive search:

- Prefer the explicit launcher flags:
  `--obs-fits-path`, `--model-h5-path`, and `--ebtel-path`.
- `OBS_FITS_PATH`, `MODEL_H5_PATH`, and `EBTEL_PATH` remain supported as
  fallback environment overrides when that is more convenient.
- Precedence is: explicit launcher flag, then environment variable, then the
  launcher's built-in default.
- `--artifact-h5`, `--a-min`, `--a-max`, and similar flags are normal command
  line options forwarded by the launcher to the Python workflow.

Git Bash launcher example with your own files:

```bash
bash ./scripts/unix/adaptive_ab_search_single_frequency_options_test.sh \
  --obs-fits-path "/path/to/your_map.fits" \
  --model-h5-path "/path/to/your_model.h5" \
  --ebtel-path "/path/to/your_ebtel.sav" \
  --artifact-h5 "/path/to/output/adaptive_ab_search_single_frequency.h5" \
  --a-min -4.5 \
  --a-max 3.0 \
  --b-min -3.0 \
  --b-max 4.8 \
  --b-start 0.0 \
  --q0-start 0.0001 \
  --max-bracket-steps 30 \
  --threshold-metric 1.3
```

PowerShell launcher example with your own files:

```powershell
bash .\scripts\unix\adaptive_ab_search_single_frequency_options_test.sh `
  --obs-fits-path "C:\path\to\your_map.fits" `
  --model-h5-path "C:\path\to\your_model.h5" `
  --ebtel-path "C:\path\to\your_ebtel.sav" `
  --artifact-h5 "C:\path\to\output\adaptive_ab_search_single_frequency.h5" `
  --a-min -4.5 `
  --a-max 3.0 `
  --b-min -3.0 `
  --b-max 4.8 `
  --b-start 0.0 `
  --q0-start 0.0001 `
  --max-bracket-steps 30 `
  --threshold-metric 1.3
```

`cmd.exe` launcher example with your own files:

```bat
scripts\windows\adaptive_ab_search_single_frequency_options_test.cmd ^
  --obs-fits-path C:\path\to\your_map.fits ^
  --model-h5-path C:\path\to\your_model.h5 ^
  --ebtel-path C:\path\to\your_ebtel.sav ^
  --artifact-h5 C:\path\to\output\adaptive_ab_search_single_frequency.h5 ^
  --a-min -4.5 ^
  --a-max 3.0 ^
  --b-min -3.0 ^
  --b-max 4.8 ^
  --b-start 0.0 ^
  --q0-start 0.0001 ^
  --max-bracket-steps 30 ^
  --threshold-metric 1.3
```

Direct Python example with your own files:

```bash
python examples/python/adaptive_ab_search_single_frequency.py \
  /path/to/your_map.fits \
  /path/to/your_model.h5 \
  --ebtel-path /path/to/your_ebtel.sav \
  --artifact-h5 /path/to/output/adaptive_ab_search_single_frequency.h5 \
  --a-min -4.5 \
  --a-max 3.0 \
  --b-min -3.0 \
  --b-max 4.8 \
  --b-start 0.0 \
  --q0-start 0.0001 \
  --max-bracket-steps 30 \
  --threshold-metric 1.3
```

Tracked 2.874 GHz development-data example:

```bash
bash ./scripts/unix/adaptive_ab_search_single_frequency_options_test.sh \
  --obs-fits-path "/path/to/pyGXrender-test-data/raw/eovsa_maps/eovsa_maps_20260323T195655/eovsa.synoptic_daily.20201126T200000Z.f2.874GHz.tb.disk.fits" \
  --model-h5-path "/path/to/pyGXrender-test-data/raw/models/models_20260323T195655/hmi.M_720s.20201126_195831.E18S19CR.CEA.NAS.GEN.CHR.h5" \
  --ebtel-path "/path/to/pyGXrender-test-data/raw/ebtel/ebtel_gxsimulator_euv/ebtel.sav" \
  --artifact-h5 "C:/Users/gelu_/AppData/Local/Temp/pychmp_adaptive_ab_runs/adaptive_ab_search_single_frequency.h5" \
  --a-min -4.5 \
  --a-max 3.0 \
  --b-min -3.0 \
  --b-max 4.8 \
  --b-start 0.0 \
  --q0-start 0.0001 \
  --max-bracket-steps 30 \
  --threshold-metric 1.3
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

- `scripts/unix/benchmark_scan_ab_obs_map.sh`
  - Wraps `examples/benchmark_scan_ab_obs_map.py` using the same pyGXrender-test-data
    resolution pattern as the scan options-test launcher.
  - Benchmarks serial and process-pool 3x3 scans and writes a CSV report.

- `scripts/unix/adaptive_ab_search_single_frequency_options_test.sh`
  - Wraps `examples/python/adaptive_ab_search_single_observation.py` with the same
    sibling test-data resolution pattern used by the other real-data launchers.
  - Prints both the adaptive-search command and the matching viewer command so
    the sparse artifact can be inspected while the search is running.

- `scripts/windows/*.cmd`
  - Windows counterparts intended for `cmd.exe` usage and the viewer Run tab.
  - If you are working from Git Bash on Windows, use the matching
    `scripts/unix/*.sh` launcher instead.

## Benchmark note

The tracked benchmark scripts are intentionally based on the same test-data
selection pattern as `scan_ab_obs_map_options_test` so users can assess
performance on their own machines without first rewriting paths.

The canonical recorded benchmark bundle now lives under:

- `reports/parallel benchmark test/`

That folder keeps the raw CSV, Markdown report, PDF report, plot image, and the
per-run H5/log/refresh artifact bundle together so the recorded result is
portable and no longer depends on a personal temporary directory. The report
generator now lives there as well:

- `reports/parallel benchmark test/generate_scan_ab_obs_map_benchmark_report.py`

Benchmark launchers:

- Windows: `scripts\windows\benchmark_scan_ab_obs_map.cmd --repeats 1 --worker-counts 1,2,3,4,5,6,7,8,9`
- Unix: `scripts/unix/benchmark_scan_ab_obs_map.sh --repeats 1 --worker-counts 1,2,3,4,5,6,7,8,9`

Reference result from the current Windows development machine using
`pyGXrender-test-data` and the 2.874 GHz EOVSA/model pair:

- Serial 3x3 scan: `447.226 s`
- Process-pool 1 worker: `449.688 s` (`0.995x` vs serial)
- Process-pool 2 workers: `402.418 s` (`1.111x`)
- Process-pool 3 workers: `388.132 s` (`1.152x`)
- Process-pool 4 workers: `392.635 s` (`1.139x`)
- Process-pool 5 workers: `386.832 s` (`1.156x`)
- Process-pool 6 workers: `388.845 s` (`1.150x`)
- Process-pool 7 workers: `390.362 s` (`1.146x`)
- Process-pool 8 workers: `392.125 s` (`1.141x`)
- Process-pool 9 workers: `388.756 s` (`1.150x`)

Interpretation:

- On this machine, the real-data 3x3 scan benefits from process-based
  parallelism, but only moderately.
- The best observed run in this single-repeat sweep was 5 workers at `386.832 s`
  (`1.156x` speedup), with 3, 6, and 9 workers very close behind.
- Because gxrender startup, disk I/O, and CPU topology vary by system, users
  should still run the tracked benchmark launchers locally before choosing a
  default worker count.

## Relationship to tests

- `tests/` should contain deterministic automated checks.
- `examples/` contains heavier, user-run validation workflows.
