# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- EUV/UV observation loading: external FITS and embedded pyAMPP refmaps are converted
  from exposure-integrated `DN` to `DN s^-1 pix^-1` when `EXPTIME` is available (header
  or refmap `source_path` FITS), matching gxrender modeled-map units.

## [0.1.0] - 2026-06-04

First public release of the unified artifact and adaptive-search infrastructure,
intended for observational fitting workflows together with [pyAMPP](https://pypi.org/project/pyampp/)
and [pyGXrender](https://pypi.org/project/pyGXrender/).

### Added

- Unified HDF5 layout: slice-level `common`, per-search `grid_points`, shared `map_store`.
- Adaptive runner: `examples/python/adaptive_ab_search_single_observation.py` with live
  `pychmp-view` refresh, warm Q₀ start from `map_store`, and HDF5 grid-point events.
- CLI utilities: `pychmp-rescore`, `pychmp-repair-grid-trial-maps`.
- Targeted search modes: `--recompute-search-id` (repair incomplete/contract-broken cells),
  `--expand-grid-search-id` (widen `a`/`b` with prior-footprint wall seed on resume).
- Library modules: `grid_points`, `slice_map_index`, `warm_q0`, `search_contract`,
  `refresh_signal`, expand/resume helpers in `ab_search`.
- Viewer: heatmap log scale, improved runner liveness vs terminal search status,
  HDF5 read retries, dedicated heatmap colorbar axes.

### Changed

- Version `0.1.0a2` → `0.1.0` (drops alpha pre-release label for this artifact generation).
- Consolidated sparse/fixed/adaptive point metadata under one schema for `pychmp-view`.
- Documentation: `README.md`, `docs/workflow_architecture.md`, `docs/artifact_data_contract.rst`.

### Removed

- `examples/python/adaptive_ab_search_single_frequency.py` and its dedicated CLI tests
  (use `adaptive_ab_search_single_observation.py`).

### Migration from 0.1.0a2

- Artifacts produced with `0.1.0a2` may not resume cleanly; prefer a new search identity
  or `--recompute-search-id` / `--expand-grid-search-id` after validating compatibility.
- Pin versions in publications: `pychmp==0.1.0`, `pyampp>=1.0.2`, and your chosen `pyGXrender`
  release; record model/observation provenance in the artifact.

### Known limitations

- Cherry-picked refit of individual grid points under an existing `search_id` is planned
  but not shipped in 0.1.0 (see project implementation notes).
- Full observational runs still require external model H5, EBTEL, and observation inputs
  (for example the `pyGXrender-test-data` checkout referenced in `README.md`).

[0.1.0]: https://github.com/suncast-org/pyCHMP/releases/tag/v0.1.0
