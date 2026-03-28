#!/bin/zsh
set -euo pipefail

# Test harness for examples/scan_ab_obs_map.py with explicit user-provided inputs.
# - Resolves sibling pyGXrender-test-data by default.
# - Keeps one CLI option per line for easy comment/uncomment editing.
# - Supports --dry-run to print the resolved command without executing it.

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

python_supports_scan_ab_obs_map() {
  local pycmd="$1"
  "$pycmd" - <<'PY' >/dev/null 2>&1
required = [
    "gxrender.sdk",
    "h5py",
    "numpy",
    "astropy.io.fits",
    "scipy.ndimage",
    "matplotlib.pyplot",
]
for name in required:
    __import__(name)
PY
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
  echo "Using PYTHON_BIN override; skipping interpreter probe."
  PYTHON_CMD="$PYTHON_BIN"
else
  if [[ -x "$HOME/miniforge3/bin/python" ]]; then
    echo "Using preferred interpreter: $HOME/miniforge3/bin/python"
    PYTHON_CMD="$HOME/miniforge3/bin/python"
  else
    echo "Preferred interpreter not found; probing fallbacks for gxrender/h5py/numpy/astropy/scipy/matplotlib support..."
    CANDIDATES=(
      "$WORKSPACE_ROOT/pyCHMP/.conda/bin/python"
      "$WORKSPACE_ROOT/gximagecomputing/.conda/bin/python"
      "$HOME/miniforge3/envs/suncast/bin/python"
      "python"
    )
    for CANDIDATE in "${CANDIDATES[@]}"; do
      echo "  checking: $CANDIDATE"
      if [[ "$CANDIDATE" == "python" ]]; then
        if command -v python >/dev/null 2>&1 && python_supports_scan_ab_obs_map python; then
          PYTHON_CMD="python"
          break
        fi
      elif [[ -x "$CANDIDATE" ]] && python_supports_scan_ab_obs_map "$CANDIDATE"; then
        PYTHON_CMD="$CANDIDATE"
        break
      fi
    done
  fi
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
# OBS_FITS_PATH="${OBS_FITS_PATH:-$LATEST_EOVSA_DIR/eovsa.synoptic_daily.20201126T200000Z.f1.418GHz.tb.disk.fits}"
OBS_FITS_PATH="${OBS_FITS_PATH:-$LATEST_EOVSA_DIR/eovsa.synoptic_daily.20201126T200000Z.f2.874GHz.tb.disk.fits}"
# OBS_FITS_PATH="${OBS_FITS_PATH:-$LATEST_EOVSA_DIR/eovsa.synoptic_daily.20201126T200000Z.f4.332GHz.tb.disk.fits}"
# OBS_FITS_PATH="${OBS_FITS_PATH:-$LATEST_EOVSA_DIR/eovsa.synoptic_daily.20201126T200000Z.f6.930GHz.tb.disk.fits}"
# OBS_FITS_PATH="${OBS_FITS_PATH:-$LATEST_EOVSA_DIR/eovsa.synoptic_daily.20201126T200000Z.f10.180GHz.tb.disk.fits}"
# OBS_FITS_PATH="${OBS_FITS_PATH:-$LATEST_EOVSA_DIR/eovsa.synoptic_daily.20201126T200000Z.f13.917GHz.tb.disk.fits}"
# OBS_FITS_PATH="${OBS_FITS_PATH:-$LATEST_EOVSA_DIR/eovsa.synoptic_daily.20201126T200000Z.f17.005GHz.tb.disk.fits}"

MODEL_H5_PATH="${MODEL_H5_PATH:-$LATEST_MODEL_DIR/hmi.M_720s.20201126_195831.E18S19CR.CEA.NAS.GEN.CHR.h5}"
ARTIFACTS_DIR="/tmp/pychmp_ab_scan_runs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ARTIFACTS_STEM="ab_scan_${TIMESTAMP}"
mkdir -p "$ARTIFACTS_DIR"

[[ -n "${LATEST_EOVSA_DIR:-}" && -d "$LATEST_EOVSA_DIR" ]] || { echo "ERROR: No dated EOVSA folder found under: $EOVSA_MAPS_ROOT"; exit 1; }
[[ -n "${LATEST_MODEL_DIR:-}" && -d "$LATEST_MODEL_DIR" ]] || { echo "ERROR: No dated model folder found under: $MODELS_ROOT"; exit 1; }
[[ -f "$OBS_FITS_PATH" ]] || { echo "ERROR: Observational FITS file not found: $OBS_FITS_PATH"; exit 1; }
[[ -f "$MODEL_H5_PATH" ]] || { echo "ERROR: Model H5 file not found: $MODEL_H5_PATH"; exit 1; }
[[ -f "$EBTEL_PATH" ]] || { echo "ERROR: EBTEL .sav file not found: $EBTEL_PATH"; exit 1; }
[[ -n "$PYTHON_CMD" ]] || { echo "ERROR: Could not find a Python interpreter with the full scan dependency set (gxrender, h5py, numpy, astropy, scipy, matplotlib)."; exit 1; }

ARGS=(
  "$OBS_FITS_PATH"
  "$MODEL_H5_PATH"
  --ebtel-path "$EBTEL_PATH"

  # (a,b) grid controls:
  --a-values 0.0,0.3,0.6
  --b-values 2.1,2.4,2.7
  # --a-start 0.0
  # --a-stop 0.6
  # --a-step 0.3
  # --b-start 2.1
  # --b-stop 2.7
  # --b-step 0.3

  # Q0 fitting controls:
  --q0-min 0.00001
  --q0-max 0.001
  --target-metric chi2
  --adaptive-bracketing
  # --q0-start-scalar 0.0001
  # --use-idl-q0-start-heuristic
  # --q0-step 1.61803398875
  # --max-bracket-steps 12

  # Plasma/geometry/observer overrides:
  # --tbase 1.0e6
  # --nbase 1.0e8
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
  # Default EOVSA fallback for these tracked test maps: use the 17 GHz beam
  # and scale it by inverse frequency when no FITS beam cards are present.
  --psf-bmaj-arcsec 5.77
  --psf-bmin-arcsec 5.77
  --psf-bpa-deg -17.5
  --psf-ref-frequency-ghz 17.0
  --psf-scale-inverse-frequency

  # Output / plotting:
  --artifacts-dir "$ARTIFACTS_DIR"
  --artifacts-stem "$ARTIFACTS_STEM"
  # --selected-a-index 1
  # --selected-b-index 1
  # --no-grid-png
  # --no-point-png
  # --show-plot

  # Utility:
  # --defaults
)

cd "$PYCHMP_REPO"
echo "Using Python: $PYTHON_CMD"
echo "Using test-data repo: $TESTDATA_REPO"
echo "Using EOVSA folder: $LATEST_EOVSA_DIR"
echo "Using model folder: $LATEST_MODEL_DIR"
print_cmd "$PYTHON_CMD" examples/scan_ab_obs_map.py "${ARGS[@]}" "${EXTRA_ARGS[@]}"
echo "Launching scan_ab_obs_map.py..."
echo "Note: first startup can pause while Python imports gxrender/scientific packages and initializes caches."
if (( DRY_RUN )); then
  echo "Dry run only; command not executed."
  exit 0
fi
"$PYTHON_CMD" examples/scan_ab_obs_map.py "${ARGS[@]}" "${EXTRA_ARGS[@]}"

echo "Artifacts directory: $ARTIFACTS_DIR"
echo "Artifacts stem: $ARTIFACTS_STEM"
