# pyCHMP

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20549272.svg)](https://doi.org/10.5281/zenodo.20549272)

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

### pyCHMP only

```bash
pip install pychmp
```

For development:

```bash
pip install -e ".[dev]"
```

### SUNCAST fitting stack (pyAMPP → pyGXrender → pyCHMP)

Observational fitting uses three installable packages. Pin versions in publications
(for example `pyampp==1.0.2`, your chosen `pyGXrender` release, and `pychmp==0.1.0`).

```bash
pip install -U pip setuptools wheel
pip install "pyampp>=1.0.2" pyGXrender pychmp
```

`pyGXrender` currently requires **Python 3.12+** on PyPI; `pychmp` supports **3.10+**.
Use a 3.12 environment when installing the full stack.

Packages alone do not include model cubes, EBTEL tables, or observation FITS. For
runnable examples, clone [pyGXrender-test-data](https://github.com/suncast-org/pyGXrender-test-data)
next to this repository (or set explicit `--model-h5`, `--ebtel-path`, and
`--obs-fits-path` / `--obs-map-id` on the workflow scripts). See `examples/python/`
and `scripts/README.md`.

Console entry points after install:

| Command | Role |
|---------|------|
| `pychmp` | Package CLI |
| `pychmp-view` | Artifact viewer |
| `pychmp-rescore` | Rescore `map_store` into a parallel search identity |
| `pychmp-repair-grid-trial-maps` | Repair invalid trial HDF5 groups |

Release history: [CHANGELOG.md](CHANGELOG.md).

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

For EUV/UV model-refmap workflows, `--euv-response-sav` is now an explicit
compatibility override rather than the default path. When omitted,
gximagecomputing resolves supported instrument responses through its
pyEUVTools-backed provider and surfaces the chosen `response.source` /
`response.mode` metadata in the downstream render result.

## User Manual

### Artifact Reuse Policy

pyCHMP artifacts are meant to represent meaningful scientific products, not to
act as an audit log of failed or abandoned attempts.

- A new slice should appear only when the scientific target changes in a way
   that changes slice identity, such as a different EUV channel, MW frequency,
   geometry/WCS, observer contract, or other incompatibility that makes the
   stored maps physically non-reusable.
- If the current run is compatible with the existing artifact and no grid-reset
   flag is set, pyCHMP resumes implicitly from the existing search state.
   Changing only search boundaries does not create a new slice or a new search
   lineage.
- `--recompute-search-id SEARCH_ID` (with `--artifact-h5`): **repair** mode —
  restores the stored scoring recipe, preserves valid map-linked trials, resets
  only contract-broken points, and resumes the adaptive walk to complete incomplete
  cells. Allowed flags: `--artifact-h5`, `--recompute-search-id`, optional
  `--no-viewer` / `--dry-run`.
- `--expand-grid-search-id SEARCH_ID` (with `--artifact-h5`): widen `a`/`b` on the
  same search id and stored recipe; hydrates completed cells; explores only new or
  outstanding cells in the enlarged domain (Phase‑1 seeds from the prior footprint
  wall toward the widened bound).
- `pychmp-repair-grid-trial-maps`: purge invalid trial HDF5 groups and rebuild
  headers; does not refit or clear `map_store`.
- `pychmp-rescore build|commit`: rescore existing `map_store` maps into a new
  `{root}_rN` search via a sidecar file (no gxrender, no viewer). Commit when no
  adaptive run is active.
- `--recompute-existing` clears the grid for the matching search identity and
   refits every point, reusing `map_store` maps for warm q0 start (no gxrender
   for compatible stored q0 values).
- `--new-search-identity` does the same refit under a parallel search identity
   for debugging (e.g. algorithm changes with the same target contract). Prior
   searches remain in the artifact; `map_store` warm start still applies.
- Changes that alter the scientific meaning of the stored results, such as
   different masks, thresholds, or beams, must not silently reuse incompatible
   results. They should branch only when the metadata signature requires it.

Practical interpretation:

- same target + same compatible signature + default (no flags): resume
- same target + same compatible signature + `--recompute-existing`: same search
   id, fresh grid, map_store warm start
- same target + same compatible signature + `--new-search-identity`: parallel
   search id, fresh grid, prior searches kept, map_store warm start
- changed slice identity: create a new slice

## Development

```bash
pytest -q
```

User-facing runnable workflows are available in `examples/`.

Consolidated scan artifacts now use one normalized point schema across
fixed-grid and adaptive search workflows. Each artifact keeps common
slice-level metadata plus per-search point records under a unified slice/search
layout, with reusable rendered arrays promoted into the shared `map_store`
when applicable. Each stored point carries the final best-fit maps plus
per-trial `q0`, `chi2`, `rho2`, and `eta2` histories when available, so
`pychmp-view` can plot the selected metric history directly from the artifact.

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
     -> pychmp.resolve_geometry_policy(...)
     -> pychmp.fit_q0_to_observation(...)

shell launchers (.sh / .cmd)
  -> examples/scan_ab_obs_map.py
     -> pychmp.load_obs_map(...)
     -> pychmp.estimate_obs_map_noise(...)
     -> pychmp.resolve_geometry_policy(...)
     -> per-point fit workflow
        -> examples/fit_q0_obs_map.py
           -> pychmp.fit_q0_to_observation(...)

shell launchers (.sh / .cmd)
  -> examples/python/adaptive_ab_search_single_observation.py
     -> pychmp.load_obs_map(...)
     -> pychmp.validate_obs_map_identity(...)
     -> pychmp.resolve_render_geometry_via_gxrender(...)
     -> pychmp.validate_scan_artifact_reuse_preflight(...) [when reusing an existing target slice]
     -> pychmp.estimate_obs_map_noise(...)
     -> pychmp.resolve_geometry_policy(...)
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
- `src/pychmp/geometry_policy.py` resolves observation LOS compatibility before workflows reuse model-saved observer/FOV metadata.
- `src/pychmp/ab_scan_artifacts.py` owns the unified slice/search artifact contract, compatibility validation, and reusable map-store plumbing shared by fixed-grid and adaptive workflows.

When adding new observational workflows, prefer reusing:

- `load_obs_map(...)`
- `validate_obs_map_identity(...)`
- `estimate_obs_map_noise(...)`
- `resolve_geometry_policy(...)`

instead of introducing new script-local FITS/refmap loaders.

The user-facing `fit_q0_obs_map.py`, `scan_ab_obs_map.py`, and
`adaptive_ab_search_single_observation.py` workflows all expose the bounded
Q0 minimizer controls `--xatol` and `--maxiter`. These tune the final bounded
minimization accuracy and iteration budget; adaptive-search
`--max-bracket-steps` only limits additional bracket expansion attempts before
that minimization stage.

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

Please cite using [CITATION.cff](CITATION.cff) (version **0.1.0**) and the Zenodo
record DOI [10.5281/zenodo.20549272](https://doi.org/10.5281/zenodo.20549272).
The all-versions concept DOI is
[10.5281/zenodo.20549271](https://doi.org/10.5281/zenodo.20549271). See also
`.zenodo.json` for archive metadata.

## License

BSD-3-Clause
