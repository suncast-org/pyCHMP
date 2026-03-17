"""pyCHMP package.

Algorithmic provenance:
- The search workflow is inspired by the established IDL implementation in
  gxmodelfitting (Alexey Kuznetsov):
  https://github.com/kuznetsov-radio/gxmodelfitting
"""

from .metrics import MetricValues, compute_metrics, threshold_union_mask

__all__ = [
  "__version__",
  "MetricValues",
  "compute_metrics",
  "threshold_union_mask",
]

__version__ = "0.1.0a0"
