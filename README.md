# pyCHMP

Python Coronal Heating Modeling Pipeline for data-constrained fitting of GX Simulator active-region models.

## Overview

pyCHMP is a Python application for parameter-space exploration of EBTEL-based magneto-thermal models in search of best agreement between synthetic and observational maps.

Initial scope:
- Replicate the validated CHMP search strategy used in the IDL GX Simulator ecosystem.
- Use pyAMPP-produced models and pyGXrender synthetic maps.
- Support microwave fitting first, then extend the same workflow to EUV constraints.

## Provenance and Acknowledgement

This project is algorithmically grounded in the model-fitting approach developed and maintained by Alexey Kuznetsov in gxmodelfitting:

- https://github.com/kuznetsov-radio/gxmodelfitting

pyCHMP is an independent Python implementation under SUNCAST-ORG. The intent is scientific reproducibility and extensibility, while preserving explicit provenance and credit.

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
pychmp --help
```

User-facing runnable workflows live under `examples/`, and tracked shell
launchers for the heavier manual validation and observational fitting runs live
under `scripts/`.

If you use the tracked launcher scripts, install `pyGXrender-test-data` as a
sibling checkout next to `pyCHMP` so they can resolve shared models, EOVSA
maps, and EBTEL inputs without machine-specific absolute paths.

## Development

```bash
pytest -q
```

User-facing runnable workflows are available in `examples/`.

Consolidated scan artifacts now use one normalized point schema across
fixed-grid and adaptive search workflows. Each stored point carries the final
best-fit maps plus per-trial `q0`, `chi2`, `rho2`, and `eta2` histories when
available, so `pychmp-view` can plot the selected metric history directly from
the artifact.

Real-data gxrender validation workflows live under `examples/python/` rather
than in the default automated test suite.

The `examples/` area is now structured to allow parallel workflow formats:
standalone Python validation scripts live under `examples/python/`, while
future developer and collaborator notebook examples can live under
`examples/notebooks/`.

The `scripts/` directory remains reserved for tracked shell launchers (`.sh`
and `.cmd`) that wrap those Python workflows; it is not the home of the Python
entry-point files themselves.

## Workflow Call Graph

The package has a layered structure. The most important distinction is between:

- package-level fitting/search primitives under `src/pychmp/`
- real-data runnable workflows under `examples/`
- shell launchers under `scripts/`

The current call hierarchy for the observational workflows is:

```text
shell launchers (.sh / .cmd)
  -> examples/fit_q0_obs_map.py
     -> pychmp.load_obs_map(...)
     -> pychmp.estimate_obs_map_noise(...)
     -> pychmp.fit_q0_to_observation(...)

shell launchers (.sh / .cmd)
  -> examples/scan_ab_obs_map.py
     -> pychmp.load_obs_map(...)
     -> pychmp.estimate_obs_map_noise(...)
     -> per-point fit workflow
        -> examples/fit_q0_obs_map.py
           -> pychmp.fit_q0_to_observation(...)

shell launchers (.sh / .cmd)
  -> examples/python/adaptive_ab_search_single_observation.py
     -> pychmp.load_obs_map(...)
     -> pychmp.validate_obs_map_identity(...)
     -> pychmp.estimate_obs_map_noise(...)
     -> pychmp.search_local_minimum_ab(...)
        -> pychmp.evaluate_ab_point(...)
           -> pychmp.fit_q0_to_observation(...)
```

Contributor notes:

- `fit_q0_to_observation(...)` is the core single-point numerical primitive.
- `fit_q0_obs_map.py` is the single-point real observational workflow wrapper around that primitive.
- `scan_ab_obs_map.py` is the fixed-grid / sparse-grid orchestration layer.
- `search_local_minimum_ab(...)` is the adaptive `(a, b)` search core.
- `adaptive_ab_search_single_observation.py` is the real observational wrapper around the adaptive search core.
- `src/pychmp/obs_maps.py` is the shared observation-ingestion layer used by the real observational workflows.

When adding new observational workflows, prefer reusing:

- `load_obs_map(...)`
- `validate_obs_map_identity(...)`
- `estimate_obs_map_noise(...)`

instead of introducing new script-local FITS/refmap loaders.

### Version Bumping

This repository uses `bumpver` to keep package versions in sync between
`pyproject.toml` and `src/pychmp/__init__.py`.

```bash
pip install -e .[dev]
bumpver show
```

For normal stable-version increments you can use the usual `bumpver update`
subcommands. For explicit pre-release bumps like `0.1.0a0 -> 0.1.0a1`, update
the version fields directly or use an explicit `bumpver` target rather than
assuming `--patch` is the right semantic move.

```bash
python -m bumpver show
```

## Citation

Please use repository citation metadata in `CITATION.cff` and release metadata in `.zenodo.json`.

## License

BSD-3-Clause
