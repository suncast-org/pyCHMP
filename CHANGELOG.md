# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Microwave SRH beam header support in `pychmp.psf`: Viktor/srhimages σ axes
  (`BEAM_SA` / `BEAM_SB` / `BEAM_PHI`, `BEAM_P==1` only) and gx_simulator
  `MakeSRHbeam` correlated σ (`BEAM_SX` / `BEAM_SY` / `BEAM_RHO`). σ values are
  converted to FWHM for the existing elliptical-kernel path; frequency scaling
  remains enabled for both radio forms.
- NORH clean-beam PSF path (`pychmp.norh_beam`): in-tree port of SSW
  `norh_prog2pinf` / `norh_beam` and gx `BeamFitNoRH`/`FitBeam`. Nobeyama FITS
  headers with `PMAT*` + `OBS-FREQ` resolve to a frequency-scalable Gaussian
  ellipse without an IDL runtime. Validated against Viktor’s IFZ sample
  (`ifz140202_022005_corrected`) vs IDL reference beam samples and fit params.
- Example `examples/python/compare_norh_ifz_beam_python_vs_idl.py`: side-by-side
  Python vs `sswidl` ``norh_beam`` plot + numeric metrics for any NORH IFZ/FITS
  (`--ifz` / `PYCHMP_NORH_IFZ`); P-angle from live IDL `get_rb0p` or header `SOLP`.
- SSRT restoring-beam path (`pychmp.ssrt_beam`): in-tree port of gx
  `GetSSRTangles` / `MakeSSRTbeam` / `BeamFitSSRT`. Example
  `examples/python/compare_ssrt_beam_python_vs_idl.py` compares Python vs IDL
  for any observation time / classic SSRT map ``.sav``.

### Known limitations

- Super-Gaussian SRH `BEAM_P≠1` is rejected (no kernel) until a later slice.
- NORH beam uses header `SOLP` (degrees) rather than recomputing `get_rb0p`;
  on standard products this matches IDL to ≲0.01°.

## [0.2.0] - 2026-10-02

Observational EUV/UV unit correction, map-store indexing, warm-restart hardening, and
viewer operator tools on top of the 0.1.0 unified-artifact stack.

### Added

- EUV/UV observation loading: external FITS and embedded pyAMPP refmaps are converted
  from exposure-integrated `DN` to `DN s^-1 pix^-1` when `EXPTIME` is available (header
  or refmap `source_path` FITS), matching gxrender modeled-map units.
- EUV/UV pixel-area normalization before rebinned rate maps (frebin-like area factor) so
  reprojected observation maps stay in `DN s^-1 pix^-1` after spatial resampling.
- Persisted `map_store` slice index (`slice_index` v1) so channel-aware `(a,b,q0)` lookups
  survive artifact reopen without rescanning every map group.
- Shared EUV response / calibration store with launcher load-or-save (resolve once, reuse
  from the artifact; avoids repeat provider/network calls on restart).
- Viewer operator tools: lock Active/Best Q₀ pointer selection; Stored-maps overlay for
  unvisited map-store `(a,b)` dots on the heatmap.
- CLI `pychmp-clean-map-store`: report (default) or delete map-store entries whose `(a,b)`
  was never visited by any search grid. Orphan detection always unions **all** searches;
  `--slice-key` / `--search-id` only filter the per-search report section.
- Scan-artifact packaging hooks for area-corrected common observation refs and slice-index
  append on write.

### Changed

- HDF5 artifact opens retry locking modes (False/True/`best-effort`/bare) when process-local
  lock-flag mismatches occur, including nested opens against an already-open bare handle.
- Ordinary restart / warm-start paths preserve finished search work: a matching completed
  search early-exits instead of silently rewriting or rescoring (bypass via recompute,
  expand, new identity, or `--retry-failed`).
- Viewer status badge: dead search PID beats a stale heartbeat for non-empty grids
  (INCOMPLETE rather than a false RUNNING/INTERRUPTED path when the process is gone).
- FINISHED banner requires authoritative successful completion; stale `completed_at` or
  failed-phase markers no longer paint green FINISHED.
- Residual display defaults to **O−M** (observation minus model) in the viewer; scoring
  residuals remain M−O internally.
- `--lonc-deg` help clarifies gxrender renderer-relative model longitude (not Carrington).
- EUV/UV map-store entries without a channel identity are fail-closed (not indexed / not
  reused) to prevent cross-channel contamination (for example 94↔171).

### Known limitations

- Map identity / render-product provenance hashes (`forward_model_sha256`, EBTEL,
  geometry, response, projection, etc.) are **written** onto new map-store products, but
  warm-start and slice-index lookup still key primarily on domain/channel + `(a,b,q0)`.
  Two maps with the same channel/`(a,b,q0)` but different model or response can still
  collide (last register wins). Channel fail-closed for EUV/UV is in place; full
  provenance-checked “compatible render product” reuse from
  `docs/moddir_compatible_map_store_plan.rst` is **not** complete yet — do not claim
  modDir parity.
- `bind_legacy_search_responses` is exported and unit-tested but not wired into the
  launcher (parked).
- Migrating/repairing ambiguous legacy map-store entries (plan Phase 6) is not shipped.

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

[0.2.0]: https://github.com/suncast-org/pyCHMP/releases/tag/v0.2.0
[0.1.0]: https://github.com/suncast-org/pyCHMP/releases/tag/v0.1.0
