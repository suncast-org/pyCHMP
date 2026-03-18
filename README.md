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

## Development

```bash
pytest -q
```

Demo scripts are available in `tests/scripts/` for quick hands-on validation
outside the pytest suite.

### Version Bumping

This repository uses `bumpver` to keep package versions in sync between
`pyproject.toml` and `src/pychmp/__init__.py`.

```bash
pip install -e .[dev]
bumpver update --patch
```

Preview without writing files:

```bash
bumpver show
```

## Citation

Please use repository citation metadata in `CITATION.cff` and release metadata in `.zenodo.json`.

## License

BSD-3-Clause
