# Contributing to pyCHMP

## Workflow

- Create a feature branch from `main`.
- Open a pull request into `main`.
- Keep changes scoped and documented.
- Ensure CI passes before merge.

## Commit style

Use clear, action-oriented commit messages.

## Scientific provenance

When implementing logic ported from IDL CHMP/gxmodelfitting, include concise provenance notes in code docstrings and PR descriptions.

## Testing

Run locally before opening PR:

```bash
pytest -q
```
