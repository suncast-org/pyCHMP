"""pyCHMP package.

Algorithmic provenance:
- The search workflow is inspired by the established IDL implementation in
  gxmodelfitting (Alexey Kuznetsov):
  https://github.com/kuznetsov-radio/gxmodelfitting
"""

from .fitting import Q0MapRenderer, fit_q0_to_observation
from .gxrender_adapter import GXRenderMWAdapter
from .metrics import MetricValues, compute_metrics, threshold_union_mask
from .optimize import Q0OptimizationResult, find_best_q0

__all__ = [
  "__version__",
  "MetricValues",
  "compute_metrics",
  "threshold_union_mask",
  "Q0MapRenderer",
  "fit_q0_to_observation",
  "GXRenderMWAdapter",
  "Q0OptimizationResult",
  "find_best_q0",
]

__version__ = "0.1.0a0"
