# CHMP-Fidelity Observation Shift & Evaluation Plan

Recorded: 2026-05-28

## Goal

Make pyCHMP’s **default** behavior a faithful CHMP migration (safeguards + auto-shift per Q₀ trial), with a later **opt-in** two-stage mask search (`data` → `union`). Artifacts remain the source of truth; the viewer displays the **trial-true** reference map (canvas + shift + crop), not the unshifted canvas alone.

## Architecture (three layers)

### Slice init (once per spectral slice)

1. Raw obs FITS → rotate to model epoch
2. Regrid to a **shared shift canvas**: padded by `slice_canvas_max_shift_arcsec` (slice-level max shift envelope, common to all searches on this slice)
3. Store in slice `common/`: `observation_canvas`, `sigma_canvas`, `canvas_wcs_header`, plus model-FOV `observed`/`sigma_map` at extract shift zero (convenience)
4. All searches on this slice **share** the same canvas; per-search `max_shift_arcsec` only clamps FindShift at runtime

### Per Q₀ trial (metrics)

1. Render model on model FOV
2. `fixed`: extract obs/sigma from shared canvas (optional fixed `xy_shift` at extract)
3. `auto`: FindShift(canvas vs model) → extract obs/sigma to model FOV (clamped by search `max_shift_arcsec`)
4. Smoothed obs max → union mask → CHMP validity gates → metrics
5. Persist `shift_x_arcsec`, `shift_y_arcsec`, metrics (not shifted obs arrays)

### Viewer / plotter

1. Load shared canvas from slice `common/`
2. Load per-trial `shift_x`, `shift_y`
3. `extract_observation_to_model_fov(canvas, model_header, shift)` → display observed + residual

**Single primitive:** `extract_observation_to_model_fov` — used by fitting, artifacts, viewer, tests.

## Phase 1 — CHMP-faithful evaluation core (default)

### 1.1 Observation reference (`obs_preprocessing.py`)

- Schema `pychmp.slice_observation_reference.v2`
- **Slice identity:** source FITS hash, artifact geometry, epoch alignment, `slice_canvas_max_shift_arcsec` (not per-search shift policy)
- **Slice storage:** canvas + sigma canvas (primary); model-FOV extracts derived at shift zero
- **Search runtime:** `shift_policy`, `max_shift_arcsec` clamp, `xy_shift` at extract — do not duplicate canvas per search

### 1.2 Shift search (`obs_alignment.py`)

- Port CHMP FindShift (correlation hill-climb)
- Reject/clamp beyond search `max_shift` (canvas already padded to slice max)

### 1.3 CHMP safeguards

1. GetSmoothedMax-style obs threshold for masks
2. Union mask (default), recomputed per trial
3. Invalidate if `maskMod > 0.99` or `maskMod/maskObs > 4`
4. EMthreshold when renderer exposes EBTEL miss ratio

### 1.4 Fitting (`fitting.py`, `optimize.py`)

- Auto mode: FindShift + extract every metric evaluation
- Per-trial: `shift_x_arcsec`, `shift_y_arcsec`, `find_shift_valid`
- Optimizer: CHMP IDL ``FindBestFitQ`` bracket + golden/Brent when ``adaptive_bracketing``; SciPy bounded for non-adaptive

## Phase 2 — Artifact schema

### Contract amendment (2026-05-28): slice-shared canvas

**Slice** owns spectral identity + shared observation canvas. **Search** owns eval contract only (no duplicate obs maps).

| Layer | Identity | Stores |
|-------|----------|--------|
| **Artifact geometry** | Observer LOS + render FOV/WCS | Shared across all slices/searches; reuse blocked if geometry differs |
| **Slice** (`slices/<key>/common/`) | Spectral channel + `slice_canvas_max_shift_arcsec` | Canvas obs/sigma, model-FOV extracts, geometry block, PSF, refmaps |
| **Search** (`searches/<id>/`) | Eval contract (`search_contract.py`) | Masks, EBTEL/response, shift **policy** + clamp, Q₀ stages, target metric, point records |
| **`map_store`** | Forward-model `(a,b,q0,…)` | Raw renders reused across searches |

Disk/memory: **one** padded canvas pair per spectral slice, not per search. Shifts applied at eval/plot time from trial metadata.

Legacy: per-search `observation_ref/` (intermediate contract) and flat `common/observed` remain readable via loader fallback.

### Slice `common/` (new writes)

- `observation_canvas`, `sigma_canvas`, `canvas_wcs_header`
- `observed`, `sigma_map` (model FOV at shift zero)
- Geometry, descriptors, PSF, refmaps

### Per-trial records

- `shift_x_arcsec`, `shift_y_arcsec`, `find_shift_valid`, `find_shift_version`
- Best/selected point: CHMP `shiftX`/`shiftY` parity

Do **not** store per-trial shifted obs — canvas + shift suffices.

## Phase 3 — Viewer

- Build display obs from slice canvas + trial shift before map panels
- Residual = modeled − extracted obs
- Live heartbeat includes shift for active trial
- Legacy artifacts (no shift fields): fixed-mode fallback + warning

## Phase 4 — CLI defaults (implemented)

- Default: `shift_policy=auto`, `max_shift_arcsec` (e.g. 20)
- `--xy-shift dx,dy` → fixed mode
- Escape hatches: `--plain-obs-max`, `--no-emthreshold-gate`, etc.
- Opt-in two-stage: `--q0-search-stages data,union`

## Phase 5 — Opt-in two-stage mask (implemented)

- `--q0-search-stages data,union`
- Same canvas/shift/safeguard stack; `mask_stage` per trial
- Incompatible with explicit FITS metrics mask

## Testing

- Canvas geometry / extract round-trip
- FindShift on synthetic pair
- Metrics replay from canvas + stored shift
- Validity gates
- Viewer displayed obs == metrics obs
- Legacy artifact fallback

## User-facing summary

- **Slice canvas:** one max-padded obs/sigma canvas per spectral channel, shared by all searches
- **Per trial:** shift + metrics only
- **Displayed observed:** shared canvas + trial shift + crop = metrics reference
