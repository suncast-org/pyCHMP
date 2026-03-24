"""pyCHMP package.

Algorithmic provenance:
- The search workflow is inspired by the established IDL implementation in
  gxmodelfitting (Alexey Kuznetsov):
  https://github.com/kuznetsov-radio/gxmodelfitting
"""

from .fitting import Q0MapRenderer, fit_q0_to_observation
from .gxrender_adapter import GXRenderMWAdapter
from .map_noise import MapNoiseEstimate, estimate_map_noise
from .metrics import MetricValues, compute_metrics, threshold_union_mask
from .optimize import Q0MetricEvaluation, Q0OptimizationResult, find_best_q0

__all__ = [
  "__version__",
  "MetricValues",
  "compute_metrics",
  "threshold_union_mask",
  "Q0MapRenderer",
  "fit_q0_to_observation",
  "GXRenderMWAdapter",
  "MapNoiseEstimate",
  "estimate_map_noise",
  "Q0MetricEvaluation",
  "Q0OptimizationResult",
  "find_best_q0",
]

__version__ = "0.1.0a1"
