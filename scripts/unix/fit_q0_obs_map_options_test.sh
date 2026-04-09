#!/bin/zsh
set -euo pipefail

# Test harness for examples/fit_q0_obs_map.py with explicit user-provided inputs.
# - No model discovery is performed by the Python script.
# - This launcher provides sensible defaults and keeps each run's artifacts unique.
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
# - Observational FITS maps are taken from the newest dated folder under:
#     raw/eovsa_maps/eovsa_maps_<timestamp>/
# - Matching models are taken from the newest dated folder under:
#     raw/models/models_<timestamp>/
# - Within those dated folders, the active observational map is selected by the
#   OBS_FITS_PATH assignment block below, and the matching model is selected by
#   MODEL_H5_PATH below.
#
# Override knobs:
# - Set OBS_FITS_PATH to use an explicit observational FITS file.
# - Set MODEL_H5_PATH to use an explicit model H5 file.
# - Set PYTHON_BIN to force a specific Python interpreter.
# - Set PYCHMP_TESTDATA_REPO to point at a non-sibling test-data checkout.
# - Pass --dry-run to print the resolved Python command and exit without
#   starting the fit.

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

EOVSA_MAPS_ROOT="$TESTDATA_REPO/raw/eovsa_maps"
MODELS_ROOT="$TESTDATA_REPO/raw/models"
EBTEL_PATH="$TESTDATA_REPO/raw/ebtel/ebtel_gxsimulator_euv/ebtel.sav"

if [[ ! -d "$TESTDATA_REPO" ]]; then
  echo "ERROR: Test-data repository not found: $TESTDATA_REPO"
  echo "Hint: install pyGXrender-test-data next to pyCHMP, or set PYCHMP_TESTDATA_REPO."
  exit 1
fi

LATEST_EOVSA_DIR="$(latest_dated_dir "$EOVSA_MAPS_ROOT" "eovsa_maps")"
LATEST_MODEL_DIR="$(latest_dated_dir "$MODELS_ROOT" "models")"

# OBS_FITS_PATH="${OBS_FITS_PATH:-$LATEST_EOVSA_DIR/eovsa.synoptic_daily.20201126T200000Z.f1.418GHz.tb.disk.fits}"
OBS_FITS_PATH="${OBS_FITS_PATH:-$LATEST_EOVSA_DIR/eovsa.synoptic_daily.20201126T200000Z.f2.874GHz.tb.disk.fits}"
# OBS_FITS_PATH="${OBS_FITS_PATH:-$LATEST_EOVSA_DIR/eovsa.synoptic_daily.20201126T200000Z.f4.332GHz.tb.disk.fits}"
# OBS_FITS_PATH="${OBS_FITS_PATH:-$LATEST_EOVSA_DIR/eovsa.synoptic_daily.20201126T200000Z.f6.930GHz.tb.disk.fits}"
# OBS_FITS_PATH="${OBS_FITS_PATH:-$LATEST_EOVSA_DIR/eovsa.synoptic_daily.20201126T200000Z.f10.180GHz.tb.disk.fits}"
# OBS_FITS_PATH="${OBS_FITS_PATH:-$LATEST_EOVSA_DIR/eovsa.synoptic_daily.20201126T200000Z.f13.917GHz.tb.disk.fits}"
# OBS_FITS_PATH="${OBS_FITS_PATH:-$LATEST_EOVSA_DIR/eovsa.synoptic_daily.20201126T200000Z.f17.005GHz.tb.disk.fits}"

MODEL_H5_PATH="${MODEL_H5_PATH:-$LATEST_MODEL_DIR/hmi.M_720s.20201126_195831.E18S19CR.CEA.NAS.GEN.CHR.h5}"

ARTIFACTS_DIR="/tmp/pychmp_fit_q0_obs_map_runs"
OBS_STEM="$(basename "$OBS_FITS_PATH" .fits)"
MODEL_STEM="$(basename "$MODEL_H5_PATH" .h5)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ARTIFACTS_STEM="${OBS_STEM}_${MODEL_STEM}_${TIMESTAMP}"

mkdir -p "$ARTIFACTS_DIR"

[[ -n "${LATEST_EOVSA_DIR:-}" && -d "$LATEST_EOVSA_DIR" ]] || { echo "ERROR: No dated EOVSA map folder found under: $EOVSA_MAPS_ROOT"; exit 1; }
[[ -n "${LATEST_MODEL_DIR:-}" && -d "$LATEST_MODEL_DIR" ]] || { echo "ERROR: No dated model folder found under: $MODELS_ROOT"; exit 1; }
[[ -f "$OBS_FITS_PATH" ]] || { echo "ERROR: Observational FITS file not found: $OBS_FITS_PATH"; exit 1; }
[[ -f "$MODEL_H5_PATH" ]] || { echo "ERROR: Model H5 file not found: $MODEL_H5_PATH"; exit 1; }
[[ -f "$EBTEL_PATH" ]] || { echo "ERROR: EBTEL .sav file not found: $EBTEL_PATH"; exit 1; }
[[ -n "$PYTHON_CMD" ]] || { echo "ERROR: Could not find a Python interpreter with gxrender installed."; exit 1; }

ARGS=(
  "$OBS_FITS_PATH"
  "$MODEL_H5_PATH"
  --ebtel-path "$EBTEL_PATH"
  --q0-min 0.01
  --q0-max 2.5
  --target-metric chi2
  --psf-bmaj-arcsec 5.77
  --psf-bmin-arcsec 5.77
  --psf-bpa-deg -17.5
  --psf-ref-frequency-ghz 17.0
  --psf-scale-inverse-frequency
  --artifacts-dir "$ARTIFACTS_DIR"
  --show-plot
)

cd "$PYCHMP_REPO"
echo "Using Python: $PYTHON_CMD"
echo "Using test-data repo: $TESTDATA_REPO"
echo "Using EOVSA folder: $LATEST_EOVSA_DIR"
echo "Using model folder: $LATEST_MODEL_DIR"
print_cmd "$PYTHON_CMD" examples/fit_q0_obs_map.py "${ARGS[@]}" "${EXTRA_ARGS[@]}"
if (( DRY_RUN )); then
  echo "Dry run only; command not executed."
  exit 0
fi
"$PYTHON_CMD" examples/fit_q0_obs_map.py "${ARGS[@]}" "${EXTRA_ARGS[@]}"

echo "Artifacts directory: $ARTIFACTS_DIR"
echo "Artifacts stem hint: $ARTIFACTS_STEM"