#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYCHMP_REPO="$(cd "$SCRIPTS_ROOT/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PYCHMP_REPO/.." && pwd)"

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

latest_dated_dir() {
  local parent="$1"
  local prefix="$2"
  find "$parent" -maxdepth 1 -mindepth 1 -type d -name "${prefix}_*" | sort | tail -n 1
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
  echo "Using PYTHON_BIN override; skipping interpreter probe."
  PYTHON_CMD="$PYTHON_BIN"
else
  if [[ -x "$HOME/miniforge3/bin/python" ]]; then
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
[[ -d "$TESTDATA_REPO" ]] || { echo "ERROR: Test-data repository not found: $TESTDATA_REPO"; exit 1; }

LATEST_EOVSA_DIR="$(latest_dated_dir "$EOVSA_MAPS_ROOT" "eovsa_maps")"
LATEST_MODEL_DIR="$(latest_dated_dir "$MODELS_ROOT" "models")"
OBS_FITS_PATH="${OBS_FITS_PATH:-$LATEST_EOVSA_DIR/eovsa.synoptic_daily.20201126T200000Z.f2.874GHz.tb.disk.fits}"
MODEL_H5_PATH="${MODEL_H5_PATH:-$LATEST_MODEL_DIR/hmi.M_720s.20201126_195831.E18S19CR.CEA.NAS.GEN.CHR.h5}"
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
[[ -f "$OBS_FITS_PATH" ]] || { echo "ERROR: Observational FITS file not found: $OBS_FITS_PATH"; exit 1; }
[[ -f "$MODEL_H5_PATH" ]] || { echo "ERROR: Model H5 file not found: $MODEL_H5_PATH"; exit 1; }
[[ -f "$EBTEL_PATH" ]] || { echo "ERROR: EBTEL .sav file not found: $EBTEL_PATH"; exit 1; }
[[ -n "$PYTHON_CMD" ]] || { echo "ERROR: Could not find a Python interpreter with the adaptive-example dependency set."; exit 1; }

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
  --fallback-psf-bmaj-arcsec 5.77
  --fallback-psf-bmin-arcsec 5.77
  --fallback-psf-bpa-deg -17.5
  --psf-ref-frequency-ghz 17.0
  --psf-scale-inverse-frequency
)

if [[ -n "${ARTIFACT_H5:-}" ]]; then
  ARGS+=(--artifact-h5 "$ARTIFACT_H5")
  VIEWER_ARTIFACT_PATH="$ARTIFACT_H5"
else
  ARGS+=(--artifacts-dir "$ARTIFACTS_DIR" --artifacts-stem "$ARTIFACTS_STEM")
  VIEWER_ARTIFACT_PATH="$ARTIFACTS_DIR/$ARTIFACTS_STEM.h5"
fi

cd "$PYCHMP_REPO"
echo "Using Python: $PYTHON_CMD"
echo "Using test-data repo: $TESTDATA_REPO"
echo "Using EOVSA folder: $LATEST_EOVSA_DIR"
echo "Using model folder: $LATEST_MODEL_DIR"
if [[ -n "${ARTIFACT_H5:-}" ]]; then
  echo "Artifact mode: explicit artifact-h5 ($ARTIFACT_H5)"
else
  echo "Artifact mode: reusable artifacts-dir/stem ($VIEWER_ARTIFACT_PATH)"
fi
print_cmd "$PYTHON_CMD" examples/python/adaptive_ab_search_single_frequency.py "${ARGS[@]}" "${EXTRA_ARGS[@]}"
print_cmd "$PYTHON_CMD" examples/pychmp_view.py "$VIEWER_ARTIFACT_PATH"
echo "Launching adaptive_ab_search_single_frequency.py..."
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
PYCHMP_WRAPPER_COMMAND="$(command_text bash "$0" "${ORIGINAL_ARGS[@]}")"
"$PYTHON_CMD" examples/python/adaptive_ab_search_single_frequency.py "${ARGS[@]}" "${EXTRA_ARGS[@]}"

echo "Artifacts directory: $ARTIFACTS_DIR"
if [[ -n "${ARTIFACT_H5:-}" ]]; then
  echo "Artifact path: $ARTIFACT_H5"
else
  echo "Artifacts stem: $ARTIFACTS_STEM"
fi
