from __future__ import annotations

import sys
from pathlib import Path

try:
    from pychmp.q0_artifact_panel import plot_q0_artifact_panel
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from pychmp.q0_artifact_panel import plot_q0_artifact_panel
