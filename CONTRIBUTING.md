# Contributing to pyCHMP

## Workflow

- Create a feature branch from `main`.
- Open a pull request into `main`.
- Keep changes scoped and documented.
- Ensure CI passes before merge.

## Architecture

Keep the package layering clear:

- `src/pychmp/` contains package-level primitives and reusable workflow logic.
- `examples/` contains runnable real-data and validation workflows.
- `scripts/` contains tracked shell launchers that wrap those Python workflows.

For the current observational fitting/search stack:

- `fit_q0_to_observation(...)` is the core single-point numerical primitive.
- `examples/fit_q0_obs_map.py` is the single-point real observational workflow.
- `examples/scan_ab_obs_map.py` is the fixed-grid / sparse-grid orchestration layer.
- `search_local_minimum_ab(...)` is the adaptive `(a, b)` search core.
- `examples/python/adaptive_ab_search_single_observation.py` is the real observational wrapper around the adaptive search core.

For observational-map ingestion, prefer reusing:

- `load_obs_map(...)`
- `validate_obs_map_identity(...)`
- `estimate_obs_map_noise(...)`

Do not introduce new script-local FITS/refmap loaders unless there is a clear reason they cannot fit the shared `obs_maps.py` contract.

For artifact work, align new storage and viewer changes with the tracked
contract in [docs/artifact_data_contract.rst](docs/artifact_data_contract.rst).
Do not introduce new file families or MW-only metadata assumptions if the same
goal can be expressed through the canonical slice / trial / solution model.

## Commit style

Use clear, action-oriented commit messages.

## Scientific provenance

When implementing logic ported from IDL CHMP/gxmodelfitting, include concise provenance notes in code docstrings and PR descriptions.

## Testing

Run locally before opening PR:

```bash
pytest -q
```
