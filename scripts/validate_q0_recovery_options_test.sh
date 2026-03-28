#!/bin/zsh
set -euo pipefail

# NOTE:
# - examples/validate_synthetic_q0_recovery.py has no CLI options.
# - This test harness targets examples/validate_q0_recovery.py,
#   which is the entry point that accepts --model-path / --ebtel-path and the
#   full option surface you asked to toggle line-by-line.
#
# Test-data layout expected by this launcher:
# - Install `pyGXrender-test-data` as a sibling repo next to `pyCHMP`, e.g.
#     <root>/pyCHMP
#     <root>/pyGXrender-test-data
# - By default this script resolves test data from:
#     $WORKSPACE_ROOT/pyGXrender-test-data
# - You may override that location with:
#     PYCHMP_TESTDATA_REPO=/path/to/pyGXrender-test-data
#
# Data resolution policy:
# - EBTEL is taken from the fixed path:
#     raw/ebtel/ebtel_gxsimulator_euv/ebtel.sav
# - Matching models are taken from the newest dated folder under:
#     raw/models/models_<timestamp>/
# - The active model is selected by MODEL_PATH below.
#
# Override knobs:
# - Set MODEL_PATH to use an explicit model H5 file.
# - Set EBTEL_PATH to use an explicit EBTEL .sav file.
# - Set PYCHMP_TESTDATA_REPO to point at a non-sibling test-data checkout.
# - Pass --dry-run to print the resolved Python command and exit without
#   starting the validation run.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYCHMP_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PYCHMP_REPO/.." && pwd)"

TESTDATA_REPO="${PYCHMP_TESTDATA_REPO:-$WORKSPACE_ROOT/pyGXrender-test-data}"
DRY_RUN=0
EXTRA_ARGS=()

latest_dated_dir() {
  local parent="$1"
  local prefix="$2"
  find "$parent" -maxdepth 1 -mindepth 1 -type d -name "${prefix}_*" | sort | tail -n 1
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

# Python selection:
# - Use PYTHON_BIN if provided.
# - Otherwise probe common workspace envs and pick the first that imports gxrender.sdk.
PYTHON_CMD=""
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD="$PYTHON_BIN"
else
  CANDIDATES=(
    "$WORKSPACE_ROOT/pyCHMP/.conda/bin/python"
    "$WORKSPACE_ROOT/gximagecomputing/.conda/bin/python"
    "$HOME/miniforge3/bin/python"
    "$HOME/miniforge3/envs/suncast/bin/python"
    "python"
  )

  for CANDIDATE in "${CANDIDATES[@]}"; do
    if [[ "$CANDIDATE" == "python" ]]; then
      if command -v python >/dev/null 2>&1 && python -c "import gxrender.sdk" >/dev/null 2>&1; then
        PYTHON_CMD="python"
        break
      fi
    elif [[ -x "$CANDIDATE" ]] && "$CANDIDATE" -c "import gxrender.sdk" >/dev/null 2>&1; then
      PYTHON_CMD="$CANDIDATE"
      break
    fi
  done
fi

MODELS_ROOT="$TESTDATA_REPO/raw/models"
LATEST_MODEL_DIR="$(latest_dated_dir "$MODELS_ROOT" "models")"
MODEL_PATH="${MODEL_PATH:-$LATEST_MODEL_DIR/hmi.M_720s.20201126_195831.E18S19CR.CEA.NAS.GEN.CHR.h5}"
EBTEL_PATH="${EBTEL_PATH:-$TESTDATA_REPO/raw/ebtel/ebtel_gxsimulator_euv/ebtel.sav}"
FREQUENCY_GHZ="5.8"

# All runs land in one shared directory; each run gets a unique file stem so
# nothing is ever overwritten.
ARTIFACTS_DIR="/tmp/pychmp_q0_runs"
MODEL_STEM="$(basename "$MODEL_PATH" .h5)"
FREQ_TAG="$(echo "$FREQUENCY_GHZ" | tr '.' 'p')"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ARTIFACTS_STEM="${MODEL_STEM}_${FREQ_TAG}ghz_${TIMESTAMP}"
# Keep raw render alongside the other artifacts using the same stem.
RAW_H5_PATH="${RAW_H5_PATH:-$ARTIFACTS_DIR/${ARTIFACTS_STEM}_raw.h5}"

mkdir -p "$ARTIFACTS_DIR"

if [[ ! -d "$TESTDATA_REPO" ]]; then
  echo "ERROR: Test-data repository not found: $TESTDATA_REPO"
  echo "Hint: install pyGXrender-test-data next to pyCHMP, or set PYCHMP_TESTDATA_REPO."
  exit 1
fi

if [[ -z "${LATEST_MODEL_DIR:-}" || ! -d "$LATEST_MODEL_DIR" ]]; then
  echo "ERROR: No dated model folder found under: $MODELS_ROOT"
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "ERROR: Model H5 file not found: $MODEL_PATH"
  exit 1
fi

if [[ ! -f "$EBTEL_PATH" ]]; then
  echo "ERROR: EBTEL .sav file not found: $EBTEL_PATH"
  exit 1
fi

if [[ -z "$PYTHON_CMD" ]]; then
  echo "ERROR: Could not find a Python interpreter with gxrender installed."
  echo "Hint: set PYTHON_BIN to an environment where gxrender is installed."
  exit 1
fi

# One option per line for easy comment/uncomment editing.
# Active lines below are the minimum practical run setup.
ARGS=(
  --model-path "$MODEL_PATH"
  --ebtel-path "$EBTEL_PATH"

  # Defaults from examples/validate_q0_recovery.py (commented):
  # --q0-true 0.0217
  # --q0-min 0.005
  # --q0-max 0.05
  --adaptive-bracketing
  # --q0-start 0.02
  # --q0-step 1.61803398875
  # --max-bracket-steps 12
  --noise-frac 0.05
  # --noise-seed 12345
  --target-metric chi2
  --frequency-ghz "$FREQUENCY_GHZ"
  # --pixel-scale-arcsec 2.0
  # --no-progress
  # --no-spinner
  # --no-log-metrics
  # --no-log-q0
  # --zoom2best 3
  # --psf-bmaj-arcsec 5.77
  # --psf-bmin-arcsec 5.77
  # --psf-bpa-deg -17.5
  # --psf-ref-frequency-ghz 17.0
  # --psf-scale-inverse-frequency

  # Commonly useful optional overrides:
  # --observer earth
  # --observer stereo-a
  # --observer stereo-b
  # --dsun-cm 1.495978707e13
  # --lonc-deg 0.0
  # --b0sun-deg 0.0

  # Explicit geometry overrides (otherwise saved FOV / auto behavior applies):
  # --xc -635.0
  # --yc -300.0
  # --dx 2.0
  # --dy 2.0
  # --nx 100
  # --ny 100

  # Artifacts / outputs:
  --artifacts-dir "$ARTIFACTS_DIR"
  --artifacts-stem "$ARTIFACTS_STEM"
  # --save-raw-h5 "$RAW_H5_PATH"
  # --no-artifacts-png
  --show-plot

  # Utility:
  # --defaults
)

cd "$PYCHMP_REPO"
echo "Using Python: $PYTHON_CMD"
echo "Using test-data repo: $TESTDATA_REPO"
echo "Using model folder: $LATEST_MODEL_DIR"
print_cmd "$PYTHON_CMD" examples/validate_q0_recovery.py "${ARGS[@]}" "${EXTRA_ARGS[@]}"
if (( DRY_RUN )); then
  echo "Dry run only; command not executed."
  exit 0
fi
"$PYTHON_CMD" examples/validate_q0_recovery.py "${ARGS[@]}" "${EXTRA_ARGS[@]}"

echo "Artifacts directory: $ARTIFACTS_DIR"
