#!/usr/bin/env bash
set -euo pipefail

# Test harness for examples/scan_ab_obs_map.py.
#
# Supported observation modes:
# - external_fits
#   - Default MW path using the canonical 2020-11-26 EOVSA FITS map plus the matching
#     model H5 and EBTEL inputs.
# - model_refmap
#   - Internal model refmap path intended for EUV/UV scan experiments such as
#     AIA_171, forwarding the same observation-selection and mask controls used
#     by the one-point real-data launcher.
#
# Key overrides:
# - --obs-source external_fits|model_refmap
# - --obs-map-id AIA_171
# - --euv-instrument AIA
# - --euv-response-sav /path/to/resp_aia_*.sav
# - --tr-mask-bmin-gauss 1000
# - --metrics-mask-threshold 0.5
# - --metrics-mask-fits /path/to/mask.fits
#
# Artifact policy:
# - Reuses the same consolidated scan artifact by default.
# - Set ARTIFACTS_STEM to create a fresh named run.
# - Set ARTIFACT_H5 to target a specific artifact file.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYCHMP_REPO="$(cd "$SCRIPTS_ROOT/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PYCHMP_REPO/.." && pwd)"

TESTDATA_REPO="${PYCHMP_TESTDATA_REPO:-$WORKSPACE_ROOT/pyGXrender-test-data}"
DRY_RUN=0
EXTRA_ARGS=()
OBS_SOURCE="${OBS_SOURCE:-external_fits}"
OBS_MAP_ID="${OBS_MAP_ID:-}"
OBS_PATH_OVERRIDE="${OBS_PATH_OVERRIDE:-}"
TR_MASK_BMIN_GAUSS="${TR_MASK_BMIN_GAUSS:-1000}"
METRICS_MASK_THRESHOLD="${METRICS_MASK_THRESHOLD:-0.1}"
METRICS_MASK_FITS="${METRICS_MASK_FITS:-}"
EUV_INSTRUMENT="${EUV_INSTRUMENT:-AIA}"
EUV_RESPONSE_SAV="${EUV_RESPONSE_SAV:-}"

latest_dated_dir() {
  local parent="$1"
  local prefix="$2"
  find "$parent" -maxdepth 1 -mindepth 1 -type d -name "${prefix}_*" | sort | tail -n 1
}

named_fixture_dir() {
  local parent="$1"
  local filename="$2"
  local match
  match="$(find "$parent" -type f -name "$filename" | sort | head -n 1)"
  [[ -n "$match" ]] || return 1
  dirname "$match"
}

latest_matching_file() {
  local parent="$1"
  local pattern="$2"
  find "$parent" -maxdepth 1 -type f -name "$pattern" | sort | tail -n 1
}

python_supports_scan_ab_obs_map() {
  local pycmd="$1"
  "$pycmd" -c 'required = ["gxrender.sdk", "h5py", "numpy", "astropy.io.fits", "scipy.ndimage", "matplotlib.pyplot"]; [__import__(name) for name in required]' >/dev/null 2>&1
}

print_cmd() {
  printf 'Command:'
  printf ' %q' "$@"
  printf '\n'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --obs-source=*)
      OBS_SOURCE="${1#*=}"
      shift
      ;;
    --obs-source)
      [[ $# -ge 2 ]] || { echo "ERROR: --obs-source requires a value." >&2; exit 1; }
      OBS_SOURCE="$2"
      shift 2
      ;;
    --obs-map-id=*)
      OBS_MAP_ID="${1#*=}"
      shift
      ;;
    --obs-map-id)
      [[ $# -ge 2 ]] || { echo "ERROR: --obs-map-id requires a value." >&2; exit 1; }
      OBS_MAP_ID="$2"
      shift 2
      ;;
    --obs-path=*)
      OBS_PATH_OVERRIDE="${1#*=}"
      shift
      ;;
    --obs-path)
      [[ $# -ge 2 ]] || { echo "ERROR: --obs-path requires a file path." >&2; exit 1; }
      OBS_PATH_OVERRIDE="$2"
      shift 2
      ;;
    --tr-mask-bmin-gauss=*)
      TR_MASK_BMIN_GAUSS="${1#*=}"
      shift
      ;;
    --tr-mask-bmin-gauss)
      [[ $# -ge 2 ]] || { echo "ERROR: --tr-mask-bmin-gauss requires a numeric value." >&2; exit 1; }
      TR_MASK_BMIN_GAUSS="$2"
      shift 2
      ;;
    --metrics-mask-threshold=*)
      METRICS_MASK_THRESHOLD="${1#*=}"
      shift
      ;;
    --metrics-mask-threshold)
      [[ $# -ge 2 ]] || { echo "ERROR: --metrics-mask-threshold requires a numeric value." >&2; exit 1; }
      METRICS_MASK_THRESHOLD="$2"
      shift 2
      ;;
    --metrics-mask-fits=*)
      METRICS_MASK_FITS="${1#*=}"
      shift
      ;;
    --metrics-mask-fits)
      [[ $# -ge 2 ]] || { echo "ERROR: --metrics-mask-fits requires a file path." >&2; exit 1; }
      METRICS_MASK_FITS="$2"
      shift 2
      ;;
    --euv-instrument=*)
      EUV_INSTRUMENT="${1#*=}"
      shift
      ;;
    --euv-instrument)
      [[ $# -ge 2 ]] || { echo "ERROR: --euv-instrument requires a value." >&2; exit 1; }
      EUV_INSTRUMENT="$2"
      shift 2
      ;;
    --euv-response-sav=*)
      EUV_RESPONSE_SAV="${1#*=}"
      shift
      ;;
    --euv-response-sav)
      [[ $# -ge 2 ]] || { echo "ERROR: --euv-response-sav requires a file path." >&2; exit 1; }
      EUV_RESPONSE_SAV="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

OBS_SOURCE="$(printf '%s' "$OBS_SOURCE" | tr '[:upper:]' '[:lower:]')"
[[ "$OBS_SOURCE" == "external_fits" || "$OBS_SOURCE" == "model_refmap" ]] || {
  echo "ERROR: --obs-source must be one of: external_fits, model_refmap" >&2
  exit 1
}

RUNTIME_CACHE_ROOT="${RUNTIME_CACHE_ROOT:-/tmp/pychmp_runtime_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$RUNTIME_CACHE_ROOT/matplotlib}"
export SUNPY_CONFIGDIR="${SUNPY_CONFIGDIR:-$RUNTIME_CACHE_ROOT/sunpy}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
DEFAULT_ASTROPY_CACHE="$HOME/.astropy/cache"
if [[ -n "${XDG_CACHE_HOME:-}" ]]; then
  export XDG_CACHE_HOME="$XDG_CACHE_HOME"
elif [[ ! -d "$DEFAULT_ASTROPY_CACHE" ]]; then
  export XDG_CACHE_HOME="$RUNTIME_CACHE_ROOT/xdg"
else
  unset XDG_CACHE_HOME 2>/dev/null || true
fi
if [[ -n "${OMP_PREFIX:-}" ]]; then
  echo "Unsetting OMP_PREFIX=$OMP_PREFIX to avoid conflicting OpenMP runtimes"
  unset OMP_PREFIX
fi
mkdir -p "$MPLCONFIGDIR" "$SUNPY_CONFIGDIR"
if [[ -n "${XDG_CACHE_HOME:-}" ]]; then
  mkdir -p "$XDG_CACHE_HOME"
fi

PYTHON_CMD=""
if [[ -n "${PYTHON_BIN:-}" ]]; then
  echo "Using PYTHON_BIN override; skipping interpreter probe."
  PYTHON_CMD="$PYTHON_BIN"
else
  if [[ -x "$HOME/miniforge3/envs/suncast/bin/python" ]]; then
    echo "Using preferred interpreter: $HOME/miniforge3/envs/suncast/bin/python"
    PYTHON_CMD="$HOME/miniforge3/envs/suncast/bin/python"
  elif [[ -x "$HOME/miniforge3/bin/python" ]]; then
    echo "Using preferred interpreter: $HOME/miniforge3/bin/python"
    PYTHON_CMD="$HOME/miniforge3/bin/python"
  else
    CANDIDATES=(
      "$WORKSPACE_ROOT/pyCHMP/.conda/bin/python"
      "$WORKSPACE_ROOT/pyCHMP/.conda/python.exe"
      "$WORKSPACE_ROOT/gximagecomputing/.conda/bin/python"
      "$WORKSPACE_ROOT/gximagecomputing/.conda/python.exe"
      "$HOME/miniforge3/envs/suncast/bin/python"
    )
    if [[ -d "$HOME/.conda/envs" ]]; then
      while IFS= read -r env_python; do
        CANDIDATES+=("$env_python")
      done < <(find "$HOME/.conda/envs" -maxdepth 2 -type f \( -path "*/bin/python" -o -name "python.exe" \) | sort)
    fi
    for CANDIDATE in "${CANDIDATES[@]}"; do
      if [[ -x "$CANDIDATE" ]] && python_supports_scan_ab_obs_map "$CANDIDATE"; then
        PYTHON_CMD="$CANDIDATE"
        break
      fi
    done
    if [[ -z "$PYTHON_CMD" ]]; then
      for command_name in python3 python; do
        command_path="$(command -v "$command_name" 2>/dev/null || true)"
        if [[ -n "$command_path" && "$command_path" != *"/WindowsApps/"* ]] && python_supports_scan_ab_obs_map "$command_path"; then
          PYTHON_CMD="$command_path"
          break
        fi
      done
    fi
  fi
fi

EOVSA_MAPS_ROOT="$TESTDATA_REPO/raw/eovsa_maps"
MODELS_ROOT="$TESTDATA_REPO/raw/models"
EBTEL_PATH="$TESTDATA_REPO/raw/ebtel/ebtel_gxsimulator_euv/ebtel.sav"
RESPONSES_ROOT="$TESTDATA_REPO/raw/responses"
[[ -d "$TESTDATA_REPO" ]] || { echo "ERROR: Test-data repository not found: $TESTDATA_REPO"; exit 1; }

LATEST_EOVSA_DIR="$(named_fixture_dir "$EOVSA_MAPS_ROOT" "eovsa.synoptic_daily.20201126T200000Z.f2.874GHz.tb.disk.fits" || true)"
LATEST_MODEL_DIR="$(named_fixture_dir "$MODELS_ROOT" "hmi.M_720s.20201126_195831.E18S19CR.CEA.NAS.GEN.CHR.h5" || true)"
OBS_FITS_PATH="${OBS_FITS_PATH:-$LATEST_EOVSA_DIR/eovsa.synoptic_daily.20201126T200000Z.f2.874GHz.tb.disk.fits}"
MODEL_H5_PATH="${MODEL_H5_PATH:-$LATEST_MODEL_DIR/hmi.M_720s.20201126_195831.E18S19CR.CEA.NAS.GEN.CHR.h5}"
LATEST_RESPONSE_DIR="$(latest_dated_dir "$RESPONSES_ROOT" "*")"
if [[ -z "${LATEST_RESPONSE_DIR:-}" || ! -d "$LATEST_RESPONSE_DIR" ]]; then
  LATEST_RESPONSE_DIR="$(find "$RESPONSES_ROOT" -maxdepth 1 -mindepth 1 -type d | sort | tail -n 1)"
fi
if [[ -z "$EUV_RESPONSE_SAV" && -n "${LATEST_RESPONSE_DIR:-}" && -d "$LATEST_RESPONSE_DIR" ]]; then
  EUV_RESPONSE_SAV="$(latest_matching_file "$LATEST_RESPONSE_DIR" 'resp_aia*.sav')"
fi
ARTIFACTS_DIR="/tmp/pychmp_ab_scan_runs"
ARTIFACTS_STEM="${ARTIFACTS_STEM:-scan_ab_obs_map_options_test}"
if [[ "${PYCHMP_TIMESTAMP_ARTIFACTS:-0}" == "1" ]]; then
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  ARTIFACTS_STEM="${ARTIFACTS_STEM}_${TIMESTAMP}"
fi
mkdir -p "$ARTIFACTS_DIR"

[[ -n "${LATEST_EOVSA_DIR:-}" && -d "$LATEST_EOVSA_DIR" ]] || { echo "ERROR: No dated EOVSA folder found under: $EOVSA_MAPS_ROOT"; exit 1; }
[[ -n "${LATEST_MODEL_DIR:-}" && -d "$LATEST_MODEL_DIR" ]] || { echo "ERROR: No dated model folder found under: $MODELS_ROOT"; exit 1; }
[[ -f "$OBS_FITS_PATH" ]] || { echo "ERROR: Observational FITS file not found: $OBS_FITS_PATH"; exit 1; }
[[ -f "$MODEL_H5_PATH" ]] || { echo "ERROR: Model H5 file not found: $MODEL_H5_PATH"; exit 1; }
[[ -f "$EBTEL_PATH" ]] || { echo "ERROR: EBTEL .sav file not found: $EBTEL_PATH"; exit 1; }
[[ -n "$PYTHON_CMD" ]] || { echo "ERROR: Could not find a Python interpreter with the full scan dependency set."; exit 1; }

ARGS=(
  --model-h5 "$MODEL_H5_PATH"
  --ebtel-path "$EBTEL_PATH"
  --a-values 0.0,0.3,0.6
  --b-values 2.1,2.4,2.7
  --q0-min 0.00001
  --q0-max 0.001
  --target-metric chi2
  --adaptive-bracketing
  --metrics-mask-threshold "$METRICS_MASK_THRESHOLD"
  --tr-mask-bmin-gauss "$TR_MASK_BMIN_GAUSS"
)

if [[ "$OBS_SOURCE" == "external_fits" ]]; then
  ARGS=(
    "$OBS_FITS_PATH"
    "$MODEL_H5_PATH"
    --ebtel-path "$EBTEL_PATH"
    --a-values 0.0,0.3,0.6
    --b-values 2.1,2.4,2.7
    --q0-min 0.00001
    --q0-max 0.001
    --target-metric chi2
    --adaptive-bracketing
    --metrics-mask-threshold "$METRICS_MASK_THRESHOLD"
    --tr-mask-bmin-gauss "$TR_MASK_BMIN_GAUSS"
    --psf-bmaj-arcsec 5.77
    --psf-bmin-arcsec 5.77
    --psf-bpa-deg -17.5
    --psf-ref-frequency-ghz 17.0
    --psf-scale-inverse-frequency
  )
else
  ARGS+=(--obs-source model_refmap)
  if [[ -n "$OBS_MAP_ID" ]]; then
    ARGS+=(--obs-map-id "$OBS_MAP_ID")
  fi
  if [[ -n "$EUV_INSTRUMENT" ]]; then
    ARGS+=(--euv-instrument "$EUV_INSTRUMENT")
  fi
  if [[ -n "$EUV_RESPONSE_SAV" ]]; then
    ARGS+=(--euv-response-sav "$EUV_RESPONSE_SAV")
  fi
fi

if [[ -n "$METRICS_MASK_FITS" ]]; then
  ARGS+=(--metrics-mask-fits "$METRICS_MASK_FITS")
fi
if [[ -n "$OBS_PATH_OVERRIDE" ]]; then
  ARGS+=(--obs-path "$OBS_PATH_OVERRIDE")
fi

if [[ -n "${ARTIFACT_H5:-}" ]]; then
  ARGS+=(--artifact-h5 "$ARTIFACT_H5")
else
  ARGS+=(--artifacts-dir "$ARTIFACTS_DIR" --artifacts-stem "$ARTIFACTS_STEM")
fi

cd "$PYCHMP_REPO"
echo "Using Python: $PYTHON_CMD"
echo "Using test-data repo: $TESTDATA_REPO"
echo "Using EOVSA folder: $LATEST_EOVSA_DIR"
echo "Using model folder: $LATEST_MODEL_DIR"
echo "Using observation source: $OBS_SOURCE"
if [[ "$OBS_SOURCE" == "model_refmap" && -n "$OBS_MAP_ID" ]]; then
  echo "Using observation map id: $OBS_MAP_ID"
fi
if [[ "$OBS_SOURCE" == "model_refmap" ]]; then
  if [[ -n "${LATEST_RESPONSE_DIR:-}" ]]; then
    echo "Using response folder: $LATEST_RESPONSE_DIR"
  fi
  if [[ -n "$EUV_INSTRUMENT" ]]; then
    echo "Using EUV instrument: $EUV_INSTRUMENT"
  fi
  if [[ -n "$EUV_RESPONSE_SAV" ]]; then
    echo "Using EUV response SAV: $EUV_RESPONSE_SAV"
  fi
fi
echo "Using EUV TR-mask Bmin [G]: $TR_MASK_BMIN_GAUSS"
echo "Using metrics-mask threshold: $METRICS_MASK_THRESHOLD"
if [[ -n "$METRICS_MASK_FITS" ]]; then
  echo "Using metrics-mask FITS: $METRICS_MASK_FITS"
fi
if [[ -n "${ARTIFACT_H5:-}" ]]; then
  echo "Artifact mode: explicit artifact-h5 ($ARTIFACT_H5)"
else
  echo "Artifact mode: reusable artifacts-dir/stem ($ARTIFACTS_DIR/$ARTIFACTS_STEM.h5)"
fi
if ((${#EXTRA_ARGS[@]})); then
  print_cmd "$PYTHON_CMD" examples/scan_ab_obs_map.py "${ARGS[@]}" "${EXTRA_ARGS[@]}"
else
  print_cmd "$PYTHON_CMD" examples/scan_ab_obs_map.py "${ARGS[@]}"
fi
echo "Launching scan_ab_obs_map.py..."
if (( DRY_RUN )); then
  echo "Dry run only; command not executed."
  exit 0
fi
if ((${#EXTRA_ARGS[@]})); then
  "$PYTHON_CMD" examples/scan_ab_obs_map.py "${ARGS[@]}" "${EXTRA_ARGS[@]}"
else
  "$PYTHON_CMD" examples/scan_ab_obs_map.py "${ARGS[@]}"
fi

echo "Artifacts directory: $ARTIFACTS_DIR"
if [[ -n "${ARTIFACT_H5:-}" ]]; then
  echo "Artifact path: $ARTIFACT_H5"
else
  echo "Artifacts stem: $ARTIFACTS_STEM"
fi
