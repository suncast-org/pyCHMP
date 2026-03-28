#!/usr/bin/env python
"""Thin wrapper for the first working `pychmp-view` GUI slice."""

from __future__ import annotations

import sys
from pathlib import Path


try:
    from pychmp.viewer import main
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from pychmp.viewer import main


if __name__ == "__main__":
    raise SystemExit(main())
