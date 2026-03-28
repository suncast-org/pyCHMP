"""pyCHMP package.

Algorithmic provenance:
- The search workflow is inspired by the established IDL implementation in
  gxmodelfitting (Alexey Kuznetsov):
  https://github.com/kuznetsov-radio/gxmodelfitting
"""

from .ab_search import ABPointResult, ABScanResult, ABRendererFactory, evaluate_ab_point, idl_q0_start_heuristic, multi_scan_ab
from .fits_utils import extract_frequency_ghz, load_2d_fits_image
from .fitting import Q0MapRenderer, fit_q0_to_observation
from .gxrender_adapter import GXRenderMWAdapter, GXRenderMWContext
from .map_noise import MapNoiseEstimate, estimate_map_noise
from .metrics import MetricValues, compute_metrics, threshold_union_mask
from .optimize import Q0MetricEvaluation, Q0OptimizationResult, find_best_q0

__all__ = [
  "__version__",
  "MetricValues",
  "compute_metrics",
  "threshold_union_mask",
  "load_2d_fits_image",
  "extract_frequency_ghz",
  "Q0MapRenderer",
  "fit_q0_to_observation",
  "ABRendererFactory",
  "ABPointResult",
  "ABScanResult",
  "evaluate_ab_point",
  "idl_q0_start_heuristic",
  "multi_scan_ab",
  "GXRenderMWAdapter",
  "GXRenderMWContext",
  "MapNoiseEstimate",
  "estimate_map_noise",
  "Q0MetricEvaluation",
  "Q0OptimizationResult",
  "find_best_q0",
]

__version__ = "0.1.0a2"
