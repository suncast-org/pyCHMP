"""pyCHMP package.

Algorithmic provenance:
- The search workflow is inspired by the established IDL implementation in
  gxmodelfitting (Alexey Kuznetsov):
  https://github.com/kuznetsov-radio/gxmodelfitting
"""

from .ab_search import ABLocalSearchResult, ABPointResult, ABScanResult, ABRendererFactory, evaluate_ab_point, idl_q0_start_heuristic, multi_scan_ab, search_local_minimum_ab
from .fits_utils import extract_frequency_ghz, load_2d_fits_image
from .fitting import Q0MapRenderer, fit_q0_to_observation
from .gxrender_adapter import GXRenderEUVAdapter, GXRenderMWAdapter, GXRenderMWContext, ResolvedRenderGeometry, build_tr_region_mask_from_blos, recombine_euv_components, resolve_render_geometry_via_gxrender
from .geometry_policy import GeometryPolicyDecision, infer_observation_observer, resolve_geometry_policy
from .map_noise import MapNoiseEstimate, estimate_map_noise
from .metrics import MetricValues, compute_metrics, threshold_union_mask
from .obs_maps import ObservationalMap, estimate_obs_map_noise, load_obs_map, obs_map_noise_unit_label, validate_obs_map_identity, find_named_testdata_file, resolve_default_testdata_fixture_paths
from .optimize import Q0MetricEvaluation, Q0OptimizationResult, find_best_q0
from .psf import PSFMetadata, KernelConvolvedRenderer, build_psf_kernel, default_psf_metadata, effective_psf_parameters, elliptical_gaussian_kernel, extract_psf_metadata_from_header, format_psf_report, resolve_psf_metadata
from .spectral import RenderSliceRequest, default_euv_channels_for_instrument, parse_csv_floats, parse_csv_tokens

__all__ = [
  "__version__",
  "MetricValues",
  "compute_metrics",
  "threshold_union_mask",
  "load_2d_fits_image",
  "extract_frequency_ghz",
  "ObservationalMap",
  "load_obs_map",
  "estimate_obs_map_noise",
  "obs_map_noise_unit_label",
  "validate_obs_map_identity",
  "find_named_testdata_file",
  "resolve_default_testdata_fixture_paths",
  "Q0MapRenderer",
  "fit_q0_to_observation",
  "ABRendererFactory",
  "ABPointResult",
  "ABScanResult",
  "ABLocalSearchResult",
  "evaluate_ab_point",
  "idl_q0_start_heuristic",
  "multi_scan_ab",
  "search_local_minimum_ab",
  "GXRenderMWAdapter",
  "GXRenderMWContext",
  "GXRenderEUVAdapter",
  "ResolvedRenderGeometry",
  "resolve_render_geometry_via_gxrender",
  "GeometryPolicyDecision",
  "infer_observation_observer",
  "resolve_geometry_policy",
  "build_tr_region_mask_from_blos",
  "recombine_euv_components",
  "MapNoiseEstimate",
  "estimate_map_noise",
  "Q0MetricEvaluation",
  "Q0OptimizationResult",
  "find_best_q0",
  "PSFMetadata",
  "KernelConvolvedRenderer",
  "build_psf_kernel",
  "default_psf_metadata",
  "effective_psf_parameters",
  "elliptical_gaussian_kernel",
  "extract_psf_metadata_from_header",
  "format_psf_report",
  "resolve_psf_metadata",
  "RenderSliceRequest",
  "default_euv_channels_for_instrument",
  "parse_csv_floats",
  "parse_csv_tokens",
]

__version__ = "0.1.0"
