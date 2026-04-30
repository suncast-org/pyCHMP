#!/usr/bin/env python
"""Compatibility wrapper for the original MW-named adaptive entrypoint.

The implementation now lives in ``adaptive_ab_search_single_observation.py`` so
the generic observation-selection path has a stable home. This wrapper stays in
place to preserve existing scripts, docs, and user habits while the workflow is
still MW-first operationally.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples"
for candidate in (REPO_ROOT, EXAMPLES_ROOT):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

from examples.python import adaptive_ab_search_single_observation as _impl


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)


def __dir__() -> list[str]:
    combined = set(globals()) | set(dir(_impl))
    return sorted(combined)


main = _impl.main


if __name__ == "__main__":
    raise SystemExit(main())
