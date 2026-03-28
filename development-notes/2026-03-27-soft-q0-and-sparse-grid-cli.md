# Soft Q0 Search Bounds And Sparse Grid CLI

## Scope

This note defines the agreed CLI grammar and precedence rules for two related upgrades:

1. single-point `q0` search should treat user `q0_min`/`q0_max` as a soft initialization interval by default
2. `(a,b)` scan input should support sparse point specifications with optional per-point `q0` interval overrides

This note is the implementation contract for `examples/fit_q0_obs_map.py`, `examples/scan_ab_obs_map.py`, and the underlying optimization helpers.

## Q0 Search Model

### Terminology

- `q0_min`, `q0_max`
  - soft initialization interval unless explicit hard bounds are supplied
- `hard_q0_min`, `hard_q0_max`
  - optional true clipping boundaries imposed by theory or user policy
- `q0_start`
  - optional explicit start point; otherwise the geometric midpoint of the soft interval

### Single-point adaptive search initialization

Adaptive search initializes from three evaluations:

1. `q0_min`
2. `q0_start` if explicitly supplied, otherwise `sqrt(q0_min * q0_max)`
3. `q0_max`

The three initial evaluations are used to decide whether:

- the minimum is already bracketed inside the soft interval
- left expansion is needed
- right expansion is needed

### Expansion behavior

- expansion uses multiplicative steps based on `q0_step`
- if hard bounds are absent, the search may expand beyond the soft interval
- if hard bounds are present, expansion is clipped to them
- search remains controlled by:
  - `max_bracket_steps`
  - positive-`q0` requirement
  - invalid/non-finite evaluation rejection

### Final refinement

- if an interior bracket is found, bounded scalar refinement is run on that bracket
- if no interior bracket is found:
  - with explicit hard bounds: bounded refinement may fall back to the hard interval
  - without explicit hard bounds: return the best boundary-tested point with a non-success status and explanatory message

## CLI Grammar

### Single-point search

The following options are supported:

- `--q0-min FLOAT`
- `--q0-max FLOAT`
- `--hard-q0-min FLOAT`
- `--hard-q0-max FLOAT`
- `--q0-start FLOAT`
- `--q0-step FLOAT`
- `--max-bracket-steps INT`

Semantics:

- `--q0-min` and `--q0-max` are required initialization hints
- `--hard-q0-min` and `--hard-q0-max` are optional true bounds
- if a hard bound is omitted, that side is unbounded apart from internal safety rules

### Sparse point CLI

Sparse points may be supplied by any mix of:

- repeated `--grid-point`
- `--grid-file`
- legacy `--ab-pairs`

#### `--grid-point`

Grammar:

```text
--grid-point A:B
--grid-point A:B:Q0_MIN:Q0_MAX
--grid-point A:B:Q0_MIN:
--grid-point A:B::Q0_MAX
```

Rules:

- `A`, `B`, `Q0_MIN`, `Q0_MAX` are floats
- empty `Q0_MIN` or `Q0_MAX` fields inherit global defaults
- whitespace around tokens is ignored
- examples:
  - `--grid-point 0.1:2.3`
  - `--grid-point 0.0:2.4:1e-5:2e-3`
  - `--grid-point -0.5:1.75::1e-3`

#### `--grid-file`

Supported formats:

- `.csv`
- `.json`

CSV columns:

```text
a,b,q0_min,q0_max
```

JSON format:

```json
[
  {"a": 0.1, "b": 2.3},
  {"a": 0.0, "b": 2.4, "q0_min": 1e-5, "q0_max": 2e-3}
]
```

Rules:

- `a` and `b` are required
- `q0_min` and `q0_max` are optional
- omitted or empty per-point `q0` fields inherit global defaults

#### Legacy `--ab-pairs`

Legacy grammar remains valid:

```text
--ab-pairs A:B,A:B,...
```

This is equivalent to repeated `--grid-point A:B` with no per-point `q0` overrides.

## Input Mode Precedence

### Sparse mode vs rectangular mode

Sparse mode is enabled if any of the following are supplied:

- `--grid-point`
- `--grid-file`
- `--ab-pairs`
- or an existing sparse artifact is being resumed

If sparse mode is active:

- rectangular grid generators (`--a-values`, `--b-values`, starts/stops/steps) are ignored for target-point generation
- the display grid is still derived later from the sparse point set

Rectangular mode is used only when no sparse inputs are provided.

### Sparse point merge order

Sparse target points are constructed in this order:

1. all points from `--grid-file`, in file order
2. all repeated `--grid-point`, in CLI order
3. all legacy `--ab-pairs`, in listed order

If duplicate `(a,b)` points are provided:

- the last occurrence wins for per-point overrides

## Per-point Q0 Precedence

For each sparse point:

1. per-point `q0_min` / `q0_max` from the point specification, if present
2. otherwise global `--q0-min` / `--q0-max`

Hard bounds are global in this iteration:

1. `--hard-q0-min`, if present
2. `--hard-q0-max`, if present

Per-point hard-bound overrides are intentionally deferred.

## Compatibility

- existing rectangular workflows remain valid
- existing sparse `--ab-pairs` workflows remain valid
- existing users of `--q0-min` / `--q0-max` keep the same syntax
- semantic change:
  - in adaptive mode, `q0_min` / `q0_max` are soft initialization bounds unless matching hard bounds are explicitly set
