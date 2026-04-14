@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%.") do set "SCRIPT_DIR=%%~fI"
for %%I in ("%SCRIPT_DIR%\..") do set "SCRIPTS_ROOT=%%~fI"
for %%I in ("%SCRIPTS_ROOT%\..") do set "PYCHMP_REPO=%%~fI"
for %%I in ("%PYCHMP_REPO%\..") do set "WORKSPACE_ROOT=%%~fI"

if defined PYCHMP_TESTDATA_REPO (
  set "TESTDATA_REPO=%PYCHMP_TESTDATA_REPO%"
) else (
  set "TESTDATA_REPO=%WORKSPACE_ROOT%\pyGXrender-test-data"
)

set "DRY_RUN=0"
set "EXTRA_ARGS="
:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--dry-run" (
  set "DRY_RUN=1"
) else (
  set "EXTRA_ARGS=!EXTRA_ARGS! ^"%~1^""
)
shift
goto parse_args
:args_done

set "RUNTIME_CACHE_ROOT=%TEMP%\pychmp_runtime_cache"
if not defined MPLCONFIGDIR set "MPLCONFIGDIR=%RUNTIME_CACHE_ROOT%\matplotlib"
if not defined SUNPY_CONFIGDIR set "SUNPY_CONFIGDIR=%RUNTIME_CACHE_ROOT%\sunpy"
if not defined KMP_DUPLICATE_LIB_OK set "KMP_DUPLICATE_LIB_OK=TRUE"
if not defined XDG_CACHE_HOME set "XDG_CACHE_HOME=%RUNTIME_CACHE_ROOT%\xdg"
if defined OMP_PREFIX set "OMP_PREFIX="
if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%"
if not exist "%SUNPY_CONFIGDIR%" mkdir "%SUNPY_CONFIGDIR%"
if not exist "%XDG_CACHE_HOME%" mkdir "%XDG_CACHE_HOME%"

if defined PYTHON_BIN (
  set "PYTHON_CMD=%PYTHON_BIN%"
) else if exist "C:\Users\gelu_\.conda\envs\pyampp-dev\python.exe" (
  set "PYTHON_CMD=C:\Users\gelu_\.conda\envs\pyampp-dev\python.exe"
) else (
  set "PYTHON_CMD=python"
)

set "EOVSA_MAPS_ROOT=%TESTDATA_REPO%\raw\eovsa_maps"
set "MODELS_ROOT=%TESTDATA_REPO%\raw\models"
set "EBTEL_PATH=%TESTDATA_REPO%\raw\ebtel\ebtel_gxsimulator_euv\ebtel.sav"
if not exist "%TESTDATA_REPO%" (
  echo ERROR: Test-data repository not found: %TESTDATA_REPO%
  exit /b 1
)

call :latest_matching_dir "%EOVSA_MAPS_ROOT%" "eovsa_maps_*" LATEST_EOVSA_DIR
call :latest_matching_dir "%MODELS_ROOT%" "models_*" LATEST_MODEL_DIR

if defined OBS_FITS_PATH (
  set "OBS_FITS_PATH=%OBS_FITS_PATH%"
) else (
  set "OBS_FITS_PATH=%LATEST_EOVSA_DIR%\eovsa.synoptic_daily.20201126T200000Z.f2.874GHz.tb.disk.fits"
)
if defined MODEL_H5_PATH (
  set "MODEL_H5_PATH=%MODEL_H5_PATH%"
) else (
  set "MODEL_H5_PATH=%LATEST_MODEL_DIR%\hmi.M_720s.20201126_195831.E18S19CR.CEA.NAS.GEN.CHR.h5"
)

set "BENCHMARK_CSV=%TEMP%\pychmp_scan_ab_obs_map_benchmark.csv"

echo Using Python: %PYTHON_CMD%
echo Using test-data repo: %TESTDATA_REPO%
echo Using EOVSA folder: %LATEST_EOVSA_DIR%
echo Using model folder: %LATEST_MODEL_DIR%
echo CSV output: %BENCHMARK_CSV%
echo Command: "%PYTHON_CMD%" examples\benchmark_scan_ab_obs_map.py "%OBS_FITS_PATH%" "%MODEL_H5_PATH%" --ebtel-path "%EBTEL_PATH%" --csv-out "%BENCHMARK_CSV%" %EXTRA_ARGS%
if "%DRY_RUN%"=="1" exit /b 0

pushd "%PYCHMP_REPO%"
"%PYTHON_CMD%" examples\benchmark_scan_ab_obs_map.py "%OBS_FITS_PATH%" "%MODEL_H5_PATH%" --ebtel-path "%EBTEL_PATH%" --csv-out "%BENCHMARK_CSV%" %EXTRA_ARGS%
set "EXIT_CODE=%ERRORLEVEL%"
popd
echo CSV output: %BENCHMARK_CSV%
exit /b %EXIT_CODE%

:latest_matching_dir
setlocal
set "PARENT=%~1"
set "MASK=%~2"
set "LATEST_NAME="
for /f "delims=" %%I in ('dir /b /ad "%PARENT%\%MASK%" 2^>nul ^| sort') do set "LATEST_NAME=%%I"
if defined LATEST_NAME (
  endlocal & set "%~3=%PARENT%\%LATEST_NAME%" & exit /b 0
)
endlocal & set "%~3=" & exit /b 0
