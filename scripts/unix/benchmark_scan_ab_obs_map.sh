#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYCHMP_REPO="$(cd "$SCRIPTS_ROOT/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PYCHMP_REPO/.." && pwd)"

TESTDATA_REPO="${PYCHMP_TESTDATA_REPO:-$WORKSPACE_ROOT/pyGXrender-test-data}"
DRY_RUN=0
EXTRA_ARGS=()
OBS_SOURCE="${OBS_SOURCE:-}"
OBS_MAP_ID=""
EUV_INSTRUMENT="${EUV_INSTRUMENT:-AIA}"
EUV_RESPONSE_SAV="${EUV_RESPONSE_SAV:-}"
TR_MASK_BMIN_GAUSS=""
METRICS_MASK_THRESHOLD=""
METRICS_MASK_FITS=""

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

python_supports_scan_ab_obs_map() {
  local pycmd="$1"
  "$pycmd" -c 'required = ["gxrender.sdk", "h5py", "numpy", "astropy.io.fits", "scipy.ndimage", "matplotlib.pyplot"]; [__import__(name) for name in required]' >/dev/null 2>&1
}

print_cmd() {
  printf 'Command:'
  printf ' %q' "$@"
  printf '\n'
}

for arg in "$@"; do
  :
done

while (($#)); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      EXTRA_ARGS+=("$1")
      shift
      ;;
    --obs-source)
      OBS_SOURCE="$2"
      shift 2
      ;;
    --obs-map-id)
      OBS_MAP_ID="$2"
      shift 2
      ;;
    --obs-fits-path)
      OBS_FITS_PATH="$2"
      shift 2
      ;;
    --obs-fits-path=*)
      OBS_FITS_PATH="${1#*=}"
      shift
      ;;
    --obs-path)
      OBS_FITS_PATH="$2"
      shift 2
      ;;
    --obs-path=*)
      OBS_FITS_PATH="${1#*=}"
      shift
      ;;
    --euv-instrument)
      EUV_INSTRUMENT="$2"
      shift 2
      ;;
    --euv-response-sav)
      EUV_RESPONSE_SAV="$2"
      shift 2
      ;;
    --tr-mask-bmin-gauss)
      TR_MASK_BMIN_GAUSS="$2"
      shift 2
      ;;
    --metrics-mask-threshold)
      METRICS_MASK_THRESHOLD="$2"
      shift 2
      ;;
    --metrics-mask-fits)
      METRICS_MASK_FITS="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

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
  PYTHON_CMD="$PYTHON_BIN"
else
  CANDIDATES=(
    "$HOME/miniforge3/envs/suncast/bin/python"
    "$HOME/miniforge3/bin/python"
    "$WORKSPACE_ROOT/pyCHMP/.conda/bin/python"
    "$WORKSPACE_ROOT/pyCHMP/.conda/python.exe"
    "$WORKSPACE_ROOT/gximagecomputing/.conda/bin/python"
    "$WORKSPACE_ROOT/gximagecomputing/.conda/python.exe"
  )
  if [[ -d "$HOME/.conda/envs" ]]; then
    while IFS= read -r env_python; do
      CANDIDATES+=("$env_python")
    done < <(find "$HOME/.conda/envs" -maxdepth 2 -type f \( -path "*/bin/python" -o -name "python.exe" \) | sort)
  fi
  for candidate in "${CANDIDATES[@]}"; do
    if [[ -x "$candidate" ]] && python_supports_scan_ab_obs_map "$candidate"; then
      PYTHON_CMD="$candidate"
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

EOVSA_MAPS_ROOT="$TESTDATA_REPO/raw/eovsa_maps"
MODELS_ROOT="$TESTDATA_REPO/raw/models"
RESPONSES_ROOT="$TESTDATA_REPO/raw/responses"
EBTEL_PATH="$TESTDATA_REPO/raw/ebtel/ebtel_gxsimulator_euv/ebtel.sav"
[[ -d "$TESTDATA_REPO" ]] || { echo "ERROR: Test-data repository not found: $TESTDATA_REPO"; exit 1; }
LATEST_MODEL_DIR="$(named_fixture_dir "$MODELS_ROOT" "hmi.M_720s.20201126_195831.E18S19CR.CEA.NAS.GEN.CHR.h5" || true)"
LATEST_RESPONSE_DIR="$(find "$RESPONSES_ROOT" -maxdepth 1 -mindepth 1 -type d | sort | tail -n 1)"
OBS_FITS_PATH="${OBS_FITS_PATH:-}"
MODEL_H5_PATH="${MODEL_H5_PATH:-$LATEST_MODEL_DIR/hmi.M_720s.20201126_195831.E18S19CR.CEA.NAS.GEN.CHR.h5}"
BENCHMARK_CSV="${BENCHMARK_CSV:-/tmp/pychmp_scan_ab_obs_map_benchmark.csv}"

[[ -n "$PYTHON_CMD" ]] || { echo "ERROR: Could not find a Python interpreter with the full scan dependency set."; exit 1; }
[[ -f "$MODEL_H5_PATH" ]] || { echo "ERROR: Model H5 file not found: $MODEL_H5_PATH"; exit 1; }
[[ -f "$EBTEL_PATH" ]] || { echo "ERROR: EBTEL .sav file not found: $EBTEL_PATH"; exit 1; }
if [[ -z "$OBS_SOURCE" ]]; then
  if [[ -n "$OBS_FITS_PATH" && -n "$OBS_MAP_ID" ]]; then
    echo "ERROR: Use either an explicit external FITS path or --obs-map-id, not both." >&2
    exit 1
  elif [[ -n "$OBS_FITS_PATH" ]]; then
    OBS_SOURCE="external_fits"
  elif [[ -n "$OBS_MAP_ID" ]]; then
    OBS_SOURCE="model_refmap"
  else
    echo "ERROR: Observation selection is required. Use --obs-fits-path /path/to/obs.fits for an external FITS map, or --obs-map-id MAP_ID for an internal model refmap." >&2
    exit 1
  fi
fi
if [[ "$OBS_SOURCE" == "external_fits" ]]; then
  [[ -n "$OBS_FITS_PATH" ]] || { echo "ERROR: --obs-fits-path is required for --obs-source=external_fits"; exit 1; }
  [[ -z "$OBS_MAP_ID" ]] || { echo "ERROR: --obs-map-id cannot be used with --obs-source=external_fits"; exit 1; }
  [[ -f "$OBS_FITS_PATH" ]] || { echo "ERROR: Observational FITS file not found: $OBS_FITS_PATH"; exit 1; }
elif [[ "$OBS_SOURCE" == "model_refmap" ]]; then
  [[ -z "$OBS_FITS_PATH" ]] || { echo "ERROR: External FITS paths cannot be used with --obs-source=model_refmap"; exit 1; }
  [[ -n "$OBS_MAP_ID" ]] || { echo "ERROR: --obs-map-id is required for --obs-source=model_refmap"; exit 1; }
  if [[ -n "$EUV_RESPONSE_SAV" ]]; then
    [[ -f "$EUV_RESPONSE_SAV" ]] || { echo "ERROR: EUV response SAV file not found: $EUV_RESPONSE_SAV"; exit 1; }
  fi
else
  echo "ERROR: Unsupported observation source: $OBS_SOURCE"
  exit 1
fi
if [[ -n "$METRICS_MASK_FITS" ]]; then
  [[ -f "$METRICS_MASK_FITS" ]] || { echo "ERROR: Metrics-mask FITS file not found: $METRICS_MASK_FITS"; exit 1; }
fi

cd "$PYCHMP_REPO"
echo "Using Python: $PYTHON_CMD"
echo "Using test-data repo: $TESTDATA_REPO"
echo "Using model folder: $LATEST_MODEL_DIR"
echo "Using observation source: $OBS_SOURCE"
if [[ "$OBS_SOURCE" == "external_fits" ]]; then
  echo "Using observation FITS: $OBS_FITS_PATH"
fi
if [[ "$OBS_SOURCE" == "model_refmap" ]]; then
  echo "Using observation map id: $OBS_MAP_ID"
  [[ -n "$LATEST_RESPONSE_DIR" ]] && echo "Using response folder: $LATEST_RESPONSE_DIR"
  echo "Using EUV instrument: $EUV_INSTRUMENT"
  [[ -n "$EUV_RESPONSE_SAV" ]] && echo "Using EUV response SAV: $EUV_RESPONSE_SAV"
  [[ -n "$TR_MASK_BMIN_GAUSS" ]] && echo "Using EUV TR-mask Bmin [G]: $TR_MASK_BMIN_GAUSS"
fi
[[ -n "$METRICS_MASK_THRESHOLD" ]] && echo "Using metrics-mask threshold: $METRICS_MASK_THRESHOLD"
[[ -n "$METRICS_MASK_FITS" ]] && echo "Using metrics-mask FITS: $METRICS_MASK_FITS"
echo "CSV output: $BENCHMARK_CSV"
BENCHMARK_CMD=("$PYTHON_CMD" examples/benchmark_scan_ab_obs_map.py)
if [[ "$OBS_SOURCE" == "external_fits" ]]; then
  BENCHMARK_CMD+=("$OBS_FITS_PATH" "$MODEL_H5_PATH")
else
  BENCHMARK_CMD+=(--model-h5 "$MODEL_H5_PATH" --obs-source model_refmap --obs-map-id "$OBS_MAP_ID" --euv-instrument "$EUV_INSTRUMENT")
  if [[ -n "$EUV_RESPONSE_SAV" ]]; then
    BENCHMARK_CMD+=(--euv-response-sav "$EUV_RESPONSE_SAV")
  fi
fi
BENCHMARK_CMD+=(--ebtel-path "$EBTEL_PATH" --csv-out "$BENCHMARK_CSV")
print_cmd "${BENCHMARK_CMD[@]}" "${EXTRA_ARGS[@]}"
if (( DRY_RUN )); then
  echo "Dry run only; command not executed."
  exit 0
fi
"${BENCHMARK_CMD[@]}" "${EXTRA_ARGS[@]}"
echo "CSV output: $BENCHMARK_CSV"
