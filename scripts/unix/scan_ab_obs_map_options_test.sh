#!/bin/zsh
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

python_supports_scan_ab_obs_map() {
  local pycmd="$1"
  "$pycmd" - <<'PY' >/dev/null 2>&1
required = ["gxrender.sdk", "h5py", "numpy", "astropy.io.fits", "scipy.ndimage", "matplotlib.pyplot"]
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
    CANDIDATES=(
      "$WORKSPACE_ROOT/pyCHMP/.conda/bin/python"
      "$WORKSPACE_ROOT/gximagecomputing/.conda/bin/python"
      "$HOME/miniforge3/envs/suncast/bin/python"
      "python"
    )
    for CANDIDATE in "${CANDIDATES[@]}"; do
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
[[ -d "$TESTDATA_REPO" ]] || { echo "ERROR: Test-data repository not found: $TESTDATA_REPO"; exit 1; }

LATEST_EOVSA_DIR="$(latest_dated_dir "$EOVSA_MAPS_ROOT" "eovsa_maps")"
LATEST_MODEL_DIR="$(latest_dated_dir "$MODELS_ROOT" "models")"
OBS_FITS_PATH="${OBS_FITS_PATH:-$LATEST_EOVSA_DIR/eovsa.synoptic_daily.20201126T200000Z.f2.874GHz.tb.disk.fits}"
MODEL_H5_PATH="${MODEL_H5_PATH:-$LATEST_MODEL_DIR/hmi.M_720s.20201126_195831.E18S19CR.CEA.NAS.GEN.CHR.h5}"
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
  "$OBS_FITS_PATH"
  "$MODEL_H5_PATH"
  --ebtel-path "$EBTEL_PATH"
  --a-values 0.0,0.3,0.6
  --b-values 2.1,2.4,2.7
  --q0-min 0.00001
  --q0-max 0.001
  --target-metric chi2
  --adaptive-bracketing
  --psf-bmaj-arcsec 5.77
  --psf-bmin-arcsec 5.77
  --psf-bpa-deg -17.5
  --psf-ref-frequency-ghz 17.0
  --psf-scale-inverse-frequency
)

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
if [[ -n "${ARTIFACT_H5:-}" ]]; then
  echo "Artifact mode: explicit artifact-h5 ($ARTIFACT_H5)"
else
  echo "Artifact mode: reusable artifacts-dir/stem ($ARTIFACTS_DIR/$ARTIFACTS_STEM.h5)"
fi
print_cmd "$PYTHON_CMD" examples/scan_ab_obs_map.py "${ARGS[@]}" "${EXTRA_ARGS[@]}"
echo "Launching scan_ab_obs_map.py..."
if (( DRY_RUN )); then
  echo "Dry run only; command not executed."
  exit 0
fi
"$PYTHON_CMD" examples/scan_ab_obs_map.py "${ARGS[@]}" "${EXTRA_ARGS[@]}"

echo "Artifacts directory: $ARTIFACTS_DIR"
if [[ -n "${ARTIFACT_H5:-}" ]]; then
  echo "Artifact path: $ARTIFACT_H5"
else
  echo "Artifacts stem: $ARTIFACTS_STEM"
fi