#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYCHMP_REPO="$(cd "$SCRIPTS_ROOT/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PYCHMP_REPO/.." && pwd)"
ADAPTIVE_ENTRYPOINT="examples/python/adaptive_ab_search_single_observation.py"

TESTDATA_REPO="${PYCHMP_TESTDATA_REPO:-$WORKSPACE_ROOT/pyGXrender-test-data}"
DRY_RUN=0
EXTRA_ARGS=()
ORIGINAL_ARGS=("$@")
CLI_ARTIFACT_H5=""
CLI_ARTIFACTS_DIR=""
CLI_ARTIFACTS_STEM=""
CLI_TARGET_METRIC=""
CLI_A_START=""
CLI_B_START=""
CLI_DA=""
CLI_DB=""
CLI_A_MIN=""
CLI_A_MAX=""
CLI_B_MIN=""
CLI_B_MAX=""
CLI_Q0_MIN=""
CLI_Q0_MAX=""
CLI_OBS_FITS_PATH=""
CLI_MODEL_H5_PATH=""
CLI_EBTEL_PATH=""
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

python_supports_adaptive_example() {
  local pycmd="$1"
  "$pycmd" -c 'required = ["gxrender.sdk", "h5py", "numpy", "astropy.io.fits", "scipy.ndimage", "matplotlib.pyplot"]; [__import__(name) for name in required]' >/dev/null 2>&1
}

print_cmd() {
  printf 'Command:'
  printf ' %q' "$@"
  printf '\n'
}

command_text() {
  local rendered=""
  printf -v rendered '%q ' "$@"
  printf '%s' "${rendered% }"
}

while (($#)); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --artifact-h5)
      [[ $# -ge 2 ]] || { echo "ERROR: --artifact-h5 requires a path argument"; exit 1; }
      CLI_ARTIFACT_H5="$2"
      shift 2
      ;;
    --obs-fits-path)
      [[ $# -ge 2 ]] || { echo "ERROR: --obs-fits-path requires a path argument"; exit 1; }
      CLI_OBS_FITS_PATH="$2"
      shift 2
      ;;
    --model-h5-path)
      [[ $# -ge 2 ]] || { echo "ERROR: --model-h5-path requires a path argument"; exit 1; }
      CLI_MODEL_H5_PATH="$2"
      shift 2
      ;;
    --ebtel-path)
      [[ $# -ge 2 ]] || { echo "ERROR: --ebtel-path requires a path argument"; exit 1; }
      CLI_EBTEL_PATH="$2"
      shift 2
      ;;
    --artifacts-dir)
      [[ $# -ge 2 ]] || { echo "ERROR: --artifacts-dir requires a path argument"; exit 1; }
      CLI_ARTIFACTS_DIR="$2"
      shift 2
      ;;
    --artifacts-stem)
      [[ $# -ge 2 ]] || { echo "ERROR: --artifacts-stem requires a value argument"; exit 1; }
      CLI_ARTIFACTS_STEM="$2"
      shift 2
      ;;
    --target-metric)
      [[ $# -ge 2 ]] || { echo "ERROR: --target-metric requires a value argument"; exit 1; }
      CLI_TARGET_METRIC="$2"
      shift 2
      ;;
    --a-start)
      [[ $# -ge 2 ]] || { echo "ERROR: --a-start requires a value argument"; exit 1; }
      CLI_A_START="$2"
      shift 2
      ;;
    --b-start)
      [[ $# -ge 2 ]] || { echo "ERROR: --b-start requires a value argument"; exit 1; }
      CLI_B_START="$2"
      shift 2
      ;;
    --da)
      [[ $# -ge 2 ]] || { echo "ERROR: --da requires a value argument"; exit 1; }
      CLI_DA="$2"
      shift 2
      ;;
    --db)
      [[ $# -ge 2 ]] || { echo "ERROR: --db requires a value argument"; exit 1; }
      CLI_DB="$2"
      shift 2
      ;;
    --a-min)
      [[ $# -ge 2 ]] || { echo "ERROR: --a-min requires a value argument"; exit 1; }
      CLI_A_MIN="$2"
      shift 2
      ;;
    --a-max)
      [[ $# -ge 2 ]] || { echo "ERROR: --a-max requires a value argument"; exit 1; }
      CLI_A_MAX="$2"
      shift 2
      ;;
    --b-min)
      [[ $# -ge 2 ]] || { echo "ERROR: --b-min requires a value argument"; exit 1; }
      CLI_B_MIN="$2"
      shift 2
      ;;
    --b-max)
      [[ $# -ge 2 ]] || { echo "ERROR: --b-max requires a value argument"; exit 1; }
      CLI_B_MAX="$2"
      shift 2
      ;;
    --q0-min)
      [[ $# -ge 2 ]] || { echo "ERROR: --q0-min requires a value argument"; exit 1; }
      CLI_Q0_MIN="$2"
      shift 2
      ;;
    --q0-max)
      [[ $# -ge 2 ]] || { echo "ERROR: --q0-max requires a value argument"; exit 1; }
      CLI_Q0_MAX="$2"
      shift 2
      ;;
    --obs-source=*)
      OBS_SOURCE="${1#*=}"
      shift
      ;;
    --obs-source)
      [[ $# -ge 2 ]] || { echo "ERROR: --obs-source requires a value argument"; exit 1; }
      OBS_SOURCE="$2"
      shift 2
      ;;
    --obs-map-id=*)
      OBS_MAP_ID="${1#*=}"
      shift
      ;;
    --obs-map-id)
      [[ $# -ge 2 ]] || { echo "ERROR: --obs-map-id requires a value argument"; exit 1; }
      OBS_MAP_ID="$2"
      shift 2
      ;;
    --obs-path=*)
      OBS_PATH_OVERRIDE="${1#*=}"
      shift
      ;;
    --obs-path)
      [[ $# -ge 2 ]] || { echo "ERROR: --obs-path requires a path argument"; exit 1; }
      OBS_PATH_OVERRIDE="$2"
      shift 2
      ;;
    --tr-mask-bmin-gauss=*)
      TR_MASK_BMIN_GAUSS="${1#*=}"
      shift
      ;;
    --tr-mask-bmin-gauss)
      [[ $# -ge 2 ]] || { echo "ERROR: --tr-mask-bmin-gauss requires a value argument"; exit 1; }
      TR_MASK_BMIN_GAUSS="$2"
      shift 2
      ;;
    --metrics-mask-threshold=*)
      METRICS_MASK_THRESHOLD="${1#*=}"
      shift
      ;;
    --metrics-mask-threshold)
      [[ $# -ge 2 ]] || { echo "ERROR: --metrics-mask-threshold requires a value argument"; exit 1; }
      METRICS_MASK_THRESHOLD="$2"
      shift 2
      ;;
    --metrics-mask-fits=*)
      METRICS_MASK_FITS="${1#*=}"
      shift
      ;;
    --metrics-mask-fits)
      [[ $# -ge 2 ]] || { echo "ERROR: --metrics-mask-fits requires a path argument"; exit 1; }
      METRICS_MASK_FITS="$2"
      shift 2
      ;;
    --euv-instrument=*)
      EUV_INSTRUMENT="${1#*=}"
      shift
      ;;
    --euv-instrument)
      [[ $# -ge 2 ]] || { echo "ERROR: --euv-instrument requires a value argument"; exit 1; }
      EUV_INSTRUMENT="$2"
      shift 2
      ;;
    --euv-response-sav=*)
      EUV_RESPONSE_SAV="${1#*=}"
      shift
      ;;
    --euv-response-sav)
      [[ $# -ge 2 ]] || { echo "ERROR: --euv-response-sav requires a path argument"; exit 1; }
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
  if [[ -x "$HOME/miniforge3/envs/suncast/bin/python" ]] && python_supports_adaptive_example "$HOME/miniforge3/envs/suncast/bin/python"; then
    echo "Using preferred interpreter: $HOME/miniforge3/envs/suncast/bin/python"
    PYTHON_CMD="$HOME/miniforge3/envs/suncast/bin/python"
  elif [[ -x "$HOME/miniforge3/bin/python" ]] && python_supports_adaptive_example "$HOME/miniforge3/bin/python"; then
    echo "Using preferred interpreter: $HOME/miniforge3/bin/python"
    PYTHON_CMD="$HOME/miniforge3/bin/python"
  else
    CANDIDATES=(
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
    for CANDIDATE in "${CANDIDATES[@]}"; do
      if [[ -x "$CANDIDATE" ]] && python_supports_adaptive_example "$CANDIDATE"; then
        PYTHON_CMD="$CANDIDATE"
        break
      fi
    done
    if [[ -z "$PYTHON_CMD" ]]; then
      for command_name in python3 python; do
        command_path="$(command -v "$command_name" 2>/dev/null || true)"
        if [[ -n "$command_path" && "$command_path" != *"/WindowsApps/"* ]] && python_supports_adaptive_example "$command_path"; then
          PYTHON_CMD="$command_path"
          break
        fi
      done
    fi
  fi
fi

EOVSA_MAPS_ROOT="$TESTDATA_REPO/raw/eovsa_maps"
MODELS_ROOT="$TESTDATA_REPO/raw/models"
EBTEL_PATH="${EBTEL_PATH:-$TESTDATA_REPO/raw/ebtel/ebtel_gxsimulator_euv/ebtel.sav}"
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
ARTIFACTS_DIR="${ARTIFACTS_DIR:-/tmp/pychmp_adaptive_ab_runs}"
ARTIFACTS_STEM="${ARTIFACTS_STEM:-adaptive_ab_search_single_frequency}"
TARGET_METRIC="${TARGET_METRIC:-chi2}"
A_START="${A_START:-0.3}"
B_START="${B_START:-2.7}"
DA="${DA:-0.3}"
DB="${DB:-0.3}"
A_MIN="${A_MIN:-0.0}"
A_MAX="${A_MAX:-1.2}"
B_MIN="${B_MIN:-2.1}"
B_MAX="${B_MAX:-3.6}"
Q0_MIN="${Q0_MIN:-0.00001}"
Q0_MAX="${Q0_MAX:-0.001}"
if [[ -n "$CLI_ARTIFACTS_DIR" ]]; then
  ARTIFACTS_DIR="$CLI_ARTIFACTS_DIR"
fi
if [[ -n "$CLI_ARTIFACTS_STEM" ]]; then
  ARTIFACTS_STEM="$CLI_ARTIFACTS_STEM"
fi
if [[ -n "$CLI_TARGET_METRIC" ]]; then
  TARGET_METRIC="$CLI_TARGET_METRIC"
fi
if [[ -n "$CLI_A_START" ]]; then
  A_START="$CLI_A_START"
fi
if [[ -n "$CLI_B_START" ]]; then
  B_START="$CLI_B_START"
fi
if [[ -n "$CLI_DA" ]]; then
  DA="$CLI_DA"
fi
if [[ -n "$CLI_DB" ]]; then
  DB="$CLI_DB"
fi
if [[ -n "$CLI_A_MIN" ]]; then
  A_MIN="$CLI_A_MIN"
fi
if [[ -n "$CLI_A_MAX" ]]; then
  A_MAX="$CLI_A_MAX"
fi
if [[ -n "$CLI_B_MIN" ]]; then
  B_MIN="$CLI_B_MIN"
fi
if [[ -n "$CLI_B_MAX" ]]; then
  B_MAX="$CLI_B_MAX"
fi
if [[ -n "$CLI_Q0_MIN" ]]; then
  Q0_MIN="$CLI_Q0_MIN"
fi
if [[ -n "$CLI_Q0_MAX" ]]; then
  Q0_MAX="$CLI_Q0_MAX"
fi
if [[ -n "$CLI_OBS_FITS_PATH" ]]; then
  OBS_FITS_PATH="$CLI_OBS_FITS_PATH"
fi
if [[ -n "$CLI_MODEL_H5_PATH" ]]; then
  MODEL_H5_PATH="$CLI_MODEL_H5_PATH"
fi
if [[ -n "$CLI_EBTEL_PATH" ]]; then
  EBTEL_PATH="$CLI_EBTEL_PATH"
fi
if [[ -n "$CLI_ARTIFACT_H5" ]]; then
  ARTIFACT_H5="$CLI_ARTIFACT_H5"
fi
if [[ "${PYCHMP_TIMESTAMP_ARTIFACTS:-0}" == "1" ]]; then
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  ARTIFACTS_STEM="${ARTIFACTS_STEM}_${TIMESTAMP}"
fi
mkdir -p "$ARTIFACTS_DIR"

[[ -n "${LATEST_EOVSA_DIR:-}" && -d "$LATEST_EOVSA_DIR" ]] || { echo "ERROR: No dated EOVSA folder found under: $EOVSA_MAPS_ROOT"; exit 1; }
[[ -n "${LATEST_MODEL_DIR:-}" && -d "$LATEST_MODEL_DIR" ]] || { echo "ERROR: No dated model folder found under: $MODELS_ROOT"; exit 1; }
[[ -f "$MODEL_H5_PATH" ]] || { echo "ERROR: Model H5 file not found: $MODEL_H5_PATH"; exit 1; }
[[ -f "$EBTEL_PATH" ]] || { echo "ERROR: EBTEL .sav file not found: $EBTEL_PATH"; exit 1; }
[[ -n "$PYTHON_CMD" ]] || { echo "ERROR: Could not find a Python interpreter with the adaptive-example dependency set."; exit 1; }
if [[ "$OBS_SOURCE" == "external_fits" ]]; then
  [[ -f "$OBS_FITS_PATH" ]] || { echo "ERROR: Observational FITS file not found: $OBS_FITS_PATH"; exit 1; }
fi

ARGS=(
  --model-h5 "$MODEL_H5_PATH"
  --ebtel-path "$EBTEL_PATH"
  --a-start "$A_START"
  --b-start "$B_START"
  --da "$DA"
  --db "$DB"
  --a-min "$A_MIN"
  --a-max "$A_MAX"
  --b-min "$B_MIN"
  --b-max "$B_MAX"
  --q0-min "$Q0_MIN"
  --q0-max "$Q0_MAX"
  --target-metric "$TARGET_METRIC"
  --adaptive-bracketing
  --metrics-mask-threshold "$METRICS_MASK_THRESHOLD"
  --tr-mask-bmin-gauss "$TR_MASK_BMIN_GAUSS"
)

if [[ "$OBS_SOURCE" == "external_fits" ]]; then
  ARGS=(
    "$OBS_FITS_PATH"
    "$MODEL_H5_PATH"
    --ebtel-path "$EBTEL_PATH"
    --a-start "$A_START"
    --b-start "$B_START"
    --da "$DA"
    --db "$DB"
    --a-min "$A_MIN"
    --a-max "$A_MAX"
    --b-min "$B_MIN"
    --b-max "$B_MAX"
    --q0-min "$Q0_MIN"
    --q0-max "$Q0_MAX"
    --target-metric "$TARGET_METRIC"
    --adaptive-bracketing
    --metrics-mask-threshold "$METRICS_MASK_THRESHOLD"
    --tr-mask-bmin-gauss "$TR_MASK_BMIN_GAUSS"
    --fallback-psf-bmaj-arcsec 5.77
    --fallback-psf-bmin-arcsec 5.77
    --fallback-psf-bpa-deg -17.5
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
  VIEWER_ARTIFACT_PATH="$ARTIFACT_H5"
else
  ARGS+=(--artifacts-dir "$ARTIFACTS_DIR" --artifacts-stem "$ARTIFACTS_STEM")
  VIEWER_ARTIFACT_PATH="$ARTIFACTS_DIR/$ARTIFACTS_STEM.h5"
fi

RUN_CMD=(
  "$PYTHON_CMD"
  "$ADAPTIVE_ENTRYPOINT"
  "${ARGS[@]}"
)
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  RUN_CMD+=("${EXTRA_ARGS[@]}")
fi
VIEW_CMD=(
  "$PYTHON_CMD"
  examples/pychmp_view.py
  "$VIEWER_ARTIFACT_PATH"
)

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
  echo "Artifact mode: reusable artifacts-dir/stem ($VIEWER_ARTIFACT_PATH)"
fi
print_cmd "${RUN_CMD[@]}"
print_cmd "${VIEW_CMD[@]}"
echo "Launching $(basename "$ADAPTIVE_ENTRYPOINT")..."
if (( DRY_RUN )); then
  echo "Dry run only; command not executed."
  exit 0
fi
export PYCHMP_WRAPPER_COMMAND
if [[ -n "${MSYS2_ENV_CONV_EXCL:-}" ]]; then
  export MSYS2_ENV_CONV_EXCL="${MSYS2_ENV_CONV_EXCL}:PYCHMP_WRAPPER_COMMAND"
else
  export MSYS2_ENV_CONV_EXCL="PYCHMP_WRAPPER_COMMAND"
fi
WRAPPER_CMD=(bash "$0")
if [[ ${#ORIGINAL_ARGS[@]} -gt 0 ]]; then
  WRAPPER_CMD+=("${ORIGINAL_ARGS[@]}")
fi
PYCHMP_WRAPPER_COMMAND="$(command_text "${WRAPPER_CMD[@]}")"
"${RUN_CMD[@]}"

echo "Artifacts directory: $ARTIFACTS_DIR"
if [[ -n "${ARTIFACT_H5:-}" ]]; then
  echo "Artifact path: $ARTIFACT_H5"
else
  echo "Artifacts stem: $ARTIFACTS_STEM"
fi
