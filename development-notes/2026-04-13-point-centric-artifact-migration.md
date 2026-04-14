# 2026-04-13 Point-Centric Artifact Migration

## Problem

The current artifact implementation still carries a conceptual split between
"sparse" and "rectangular" scan outputs. That split is useful for legacy file
compatibility, but it is not the right primary persistence model.

Scientifically, the canonical unit of persisted work is a completed `(a, b)`
point evaluation:

- slice identity
- `(a, b)` coordinates
- fitted `q0`
- metric values and trial histories
- elapsed time
- compatibility/provenance diagnostics

The search strategy that produced the point should not determine the artifact
architecture.

## Target Model

All new artifacts should be point-centric:

- canonical source of truth: `point_records`
- cached convenience views: grid summaries (`best_q0`, `chi2`, `rho2`, `eta2`,
  `success`, `a_values`, `b_values`)
- viewer and resume code should reconstruct their working model from canonical
  point records first

This allows the same artifact to support:

- rectangular scans
- adaptive searches
- sparse/manual point lists
- resumed refinement under a different search strategy

## Migration Rules

1. Backward reading support for old rectangular artifacts must remain.
2. All new writes should emit canonical `point_records`.
3. Rectangular/grid-style runs may still write cached summary arrays for
   convenience and fast viewing.
4. Resume and viewer logic should prefer canonical point records whenever they
   are present.
5. Search-mode-specific code should generate tasks, not own persistence rules.

## Incremental Implementation

Phase 1:

- keep legacy rectangular read support
- make new rectangular writes also emit `point_records`
- load from `point_records` when available, otherwise fall back to legacy
  `points`
- keep existing summary arrays as caches

Phase 2:

- tighten the shared append API so artifact creation is never format-ambiguous
- treat append-only point records as the only required persistence primitive

Phase 3:

- simplify viewer/resume code so it depends only on canonical point records plus
  optional cached summaries

## Expected Outcome

After Phase 1, old artifacts still load, but all newly written artifacts become
compatible with the unified point-centric model. That provides a safe migration
path without breaking existing users or reports.
