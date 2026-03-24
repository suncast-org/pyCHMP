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
#
# Practical usage:
# - Keep one OBS_FITS_PATH line active in the selection block below and comment
#   the others out for easy frequency switching.
# - The launcher prints the resolved test-data repo, dated EOVSA folder, and
#   dated model folder at startup so the user can verify the intended inputs.

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
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RUNTIME_CACHE_ROOT/xdg}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
mkdir -p "$MPLCONFIGDIR" "$SUNPY_CONFIGDIR" "$XDG_CACHE_HOME"

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
    "/Users/gelu/miniforge3/envs/suncast/bin/python"
    "/Users/gelu/miniforge3/bin/python"
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

# Select one observational map below. Keep one active line and comment the rest.
# All paths resolve relative to the newest dated EOVSA folder found under:
#   $TESTDATA_REPO/raw/eovsa_maps
# OBS_FITS_PATH="${OBS_FITS_PATH:-$LATEST_EOVSA_DIR/eovsa.synoptic_daily.20201126T200000Z.f1.418GHz.tb.disk.fits}"
OBS_FITS_PATH="${OBS_FITS_PATH:-$LATEST_EOVSA_DIR/eovsa.synoptic_daily.20201126T200000Z.f2.874GHz.tb.disk.fits}"
# OBS_FITS_PATH="${OBS_FITS_PATH:-$LATEST_EOVSA_DIR/eovsa.synoptic_daily.20201126T200000Z.f4.332GHz.tb.disk.fits}"
# OBS_FITS_PATH="${OBS_FITS_PATH:-$LATEST_EOVSA_DIR/eovsa.synoptic_daily.20201126T200000Z.f6.930GHz.tb.disk.fits}"
# OBS_FITS_PATH="${OBS_FITS_PATH:-$LATEST_EOVSA_DIR/eovsa.synoptic_daily.20201126T200000Z.f10.180GHz.tb.disk.fits}"
# OBS_FITS_PATH="${OBS_FITS_PATH:-$LATEST_EOVSA_DIR/eovsa.synoptic_daily.20201126T200000Z.f13.917GHz.tb.disk.fits}"
# OBS_FITS_PATH="${OBS_FITS_PATH:-$LATEST_EOVSA_DIR/eovsa.synoptic_daily.20201126T200000Z.f17.005GHz.tb.disk.fits}"

# Select the matching model H5 below. By default this uses the newest dated
# model folder found under:
#   $TESTDATA_REPO/raw/models
MODEL_H5_PATH="${MODEL_H5_PATH:-$LATEST_MODEL_DIR/hmi.M_720s.20201126_195831.E18S19CR.CEA.NAS.GEN.CHR.h5}"

ARTIFACTS_DIR="/tmp/pychmp_fit_q0_obs_map_runs"
OBS_STEM="$(basename "$OBS_FITS_PATH" .fits)"
MODEL_STEM="$(basename "$MODEL_H5_PATH" .h5)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ARTIFACTS_STEM="${OBS_STEM}_${MODEL_STEM}_${TIMESTAMP}"

mkdir -p "$ARTIFACTS_DIR"

if [[ -z "${LATEST_EOVSA_DIR:-}" || ! -d "$LATEST_EOVSA_DIR" ]]; then
  echo "ERROR: No dated EOVSA map folder found under: $EOVSA_MAPS_ROOT"
  exit 1
fi

if [[ -z "${LATEST_MODEL_DIR:-}" || ! -d "$LATEST_MODEL_DIR" ]]; then
  echo "ERROR: No dated model folder found under: $MODELS_ROOT"
  exit 1
fi

if [[ ! -f "$OBS_FITS_PATH" ]]; then
  echo "ERROR: Observational FITS file not found: $OBS_FITS_PATH"
  exit 1
fi

if [[ ! -f "$MODEL_H5_PATH" ]]; then
  echo "ERROR: Model H5 file not found: $MODEL_H5_PATH"
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


# One option per line for easy comment/uncomment editing.
ARGS=(
  "$OBS_FITS_PATH"
  "$MODEL_H5_PATH"
  --ebtel-path "$EBTEL_PATH"

  # Q0 fitting controls:
  --q0-min 0.00001
  --q0-max 0.001
  --target-metric chi2

  # Plasma/geometry/observer overrides (comment/uncomment as needed):
  # --tbase 1.5e6
  # --nbase 1e9
  # --a 1.0
  # --b 1.0
  # --observer earth
  # --observer stereo-a
  # --observer stereo-b
  # --dsun-cm 1.495978707e13
  # --lonc-deg 0.0
  # --b0sun-deg 0.0
  # --xc -635.0
  # --yc -300.0
  # --dx 2.0
  # --dy 2.0
  # --nx 100
  # --ny 100
  # --pixel-scale-arcsec 2.0

  # PSF/beam options:
  # --psf-bmaj-arcsec 5.77
  # --psf-bmin-arcsec 5.77
  # --psf-bpa-deg -17.5
  # --psf-ref-frequency-ghz 17.0
  # --psf-scale-inverse-frequency

  # Artifacts/outputs:
  --artifacts-dir "$ARTIFACTS_DIR"
  # --artifacts-stem "$ARTIFACTS_STEM"
  # --save-raw-h5 "$RAW_H5_PATH"
  # --no-artifacts-png
  --show-plot
  # --no-artifacts

  # Utility:
  # --defaults
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
