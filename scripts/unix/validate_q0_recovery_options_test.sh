#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYCHMP_REPO="$(cd "$SCRIPTS_ROOT/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PYCHMP_REPO/.." && pwd)"

TESTDATA_REPO="${PYCHMP_TESTDATA_REPO:-$WORKSPACE_ROOT/pyGXrender-test-data}"
DRY_RUN=0
EXTRA_ARGS=()

latest_dated_dir() {
  local parent="$1"
  local prefix="$2"
  find "$parent" -maxdepth 1 -mindepth 1 -type d -name "${prefix}_*" | sort | tail -n 1
}

python_supports_validate_example() {
  local pycmd="$1"
  "$pycmd" -c 'required = ["gxrender.sdk", "h5py", "numpy", "matplotlib.pyplot"]; [__import__(name) for name in required]' >/dev/null 2>&1
}

print_cmd() {
  printf 'Command:'
  printf ' %q' "$@"
  printf '\n'
}

for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    DRY_RUN=1
  else
    EXTRA_ARGS+=("$arg")
  fi
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
    "$WORKSPACE_ROOT/pyCHMP/.conda/bin/python"
    "$WORKSPACE_ROOT/pyCHMP/.conda/python.exe"
    "$WORKSPACE_ROOT/gximagecomputing/.conda/bin/python"
    "$WORKSPACE_ROOT/gximagecomputing/.conda/python.exe"
    "$HOME/miniforge3/envs/suncast/bin/python"
    "$HOME/miniforge3/bin/python"
  )
  if [[ -d "$HOME/.conda/envs" ]]; then
    while IFS= read -r env_python; do
      CANDIDATES+=("$env_python")
    done < <(find "$HOME/.conda/envs" -maxdepth 2 -type f \( -path "*/bin/python" -o -name "python.exe" \) | sort)
  fi
  for CANDIDATE in "${CANDIDATES[@]}"; do
    if [[ -x "$CANDIDATE" ]] && python_supports_validate_example "$CANDIDATE"; then
      PYTHON_CMD="$CANDIDATE"
      break
    fi
  done
  if [[ -z "$PYTHON_CMD" ]]; then
    for command_name in python3 python; do
      command_path="$(command -v "$command_name" 2>/dev/null || true)"
      if [[ -n "$command_path" && "$command_path" != *"/WindowsApps/"* ]] && python_supports_validate_example "$command_path"; then
        PYTHON_CMD="$command_path"
        break
      fi
    done
  fi
fi

MODELS_ROOT="$TESTDATA_REPO/raw/models"
LATEST_MODEL_DIR="$(latest_dated_dir "$MODELS_ROOT" "models")"
MODEL_PATH="${MODEL_PATH:-$LATEST_MODEL_DIR/hmi.M_720s.20201126_195831.E18S19CR.CEA.NAS.GEN.CHR.h5}"
EBTEL_PATH="${EBTEL_PATH:-$TESTDATA_REPO/raw/ebtel/ebtel_gxsimulator_euv/ebtel.sav}"
FREQUENCY_GHZ="5.8"
ARTIFACTS_DIR="/tmp/pychmp_q0_runs"
MODEL_STEM="$(basename "$MODEL_PATH" .h5)"
FREQ_TAG="$(echo "$FREQUENCY_GHZ" | tr '.' 'p')"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ARTIFACTS_STEM="${MODEL_STEM}_${FREQ_TAG}ghz_${TIMESTAMP}"
mkdir -p "$ARTIFACTS_DIR"

[[ -d "$TESTDATA_REPO" ]] || { echo "ERROR: Test-data repository not found: $TESTDATA_REPO"; exit 1; }
[[ -n "${LATEST_MODEL_DIR:-}" && -d "$LATEST_MODEL_DIR" ]] || { echo "ERROR: No dated model folder found under: $MODELS_ROOT"; exit 1; }
[[ -f "$MODEL_PATH" ]] || { echo "ERROR: Model H5 file not found: $MODEL_PATH"; exit 1; }
[[ -f "$EBTEL_PATH" ]] || { echo "ERROR: EBTEL .sav file not found: $EBTEL_PATH"; exit 1; }
[[ -n "$PYTHON_CMD" ]] || { echo "ERROR: Could not find a Python interpreter with gxrender installed."; exit 1; }

ARGS=(
  --model-path "$MODEL_PATH"
  --ebtel-path "$EBTEL_PATH"
  --adaptive-bracketing
  --noise-frac 0.05
  --target-metric chi2
  --frequency-ghz "$FREQUENCY_GHZ"
  --artifacts-dir "$ARTIFACTS_DIR"
  --artifacts-stem "$ARTIFACTS_STEM"
  --show-plot
)

RUN_CMD=(
  "$PYTHON_CMD"
  examples/validate_q0_recovery.py
  "${ARGS[@]}"
)
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  RUN_CMD+=("${EXTRA_ARGS[@]}")
fi

cd "$PYCHMP_REPO"
echo "Using Python: $PYTHON_CMD"
echo "Using test-data repo: $TESTDATA_REPO"
echo "Using model folder: $LATEST_MODEL_DIR"
print_cmd "${RUN_CMD[@]}"
if (( DRY_RUN )); then
  echo "Dry run only; command not executed."
  exit 0
fi
"${RUN_CMD[@]}"

echo "Artifacts directory: $ARTIFACTS_DIR"
