#!/usr/bin/env python
"""Convert a rectangular AB scan artifact into sparse point-record format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from pychmp.ab_scan_artifacts import convert_rectangular_artifact_to_sparse
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from pychmp.ab_scan_artifacts import convert_rectangular_artifact_to_sparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src_h5", type=Path, help="Existing rectangular AB scan artifact")
    parser.add_argument("dst_h5", type=Path, help="Output sparse point-record artifact")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite destination if it already exists")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    convert_rectangular_artifact_to_sparse(args.src_h5, args.dst_h5, overwrite=bool(args.overwrite))
    print(f"✓ Wrote sparse artifact: {args.dst_h5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
