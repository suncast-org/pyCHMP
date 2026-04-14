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
set "EXTRA_ARGS_PRINT="
set "CLI_ARTIFACT_H5="
set "CLI_ARTIFACTS_DIR="
set "CLI_ARTIFACTS_STEM="
set "CLI_TARGET_METRIC="
set "CLI_A_START="
set "CLI_B_START="
set "CLI_DA="
set "CLI_DB="
set "CLI_A_MIN="
set "CLI_A_MAX="
set "CLI_B_MIN="
set "CLI_B_MAX="
set "CLI_Q0_MIN="
set "CLI_Q0_MAX="
:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--dry-run" (
  set "DRY_RUN=1"
) else if /I "%~1"=="--artifact-h5" (
  if "%~2"=="" (
    echo ERROR: --artifact-h5 requires a path argument
    exit /b 1
  )
  set "CLI_ARTIFACT_H5=%~2"
  shift
) else if /I "%~1"=="--artifacts-dir" (
  if "%~2"=="" (
    echo ERROR: --artifacts-dir requires a path argument
    exit /b 1
  )
  set "CLI_ARTIFACTS_DIR=%~2"
  shift
) else if /I "%~1"=="--artifacts-stem" (
  if "%~2"=="" (
    echo ERROR: --artifacts-stem requires a value argument
    exit /b 1
  )
  set "CLI_ARTIFACTS_STEM=%~2"
  shift
) else if /I "%~1"=="--target-metric" (
  if "%~2"=="" (
    echo ERROR: --target-metric requires a value argument
    exit /b 1
  )
  set "CLI_TARGET_METRIC=%~2"
  shift
) else if /I "%~1"=="--a-start" (
  if "%~2"=="" (
    echo ERROR: --a-start requires a value argument
    exit /b 1
  )
  set "CLI_A_START=%~2"
  shift
) else if /I "%~1"=="--b-start" (
  if "%~2"=="" (
    echo ERROR: --b-start requires a value argument
    exit /b 1
  )
  set "CLI_B_START=%~2"
  shift
) else if /I "%~1"=="--da" (
  if "%~2"=="" (
    echo ERROR: --da requires a value argument
    exit /b 1
  )
  set "CLI_DA=%~2"
  shift
) else if /I "%~1"=="--db" (
  if "%~2"=="" (
    echo ERROR: --db requires a value argument
    exit /b 1
  )
  set "CLI_DB=%~2"
  shift
) else if /I "%~1"=="--a-min" (
  if "%~2"=="" (
    echo ERROR: --a-min requires a value argument
    exit /b 1
  )
  set "CLI_A_MIN=%~2"
  shift
) else if /I "%~1"=="--a-max" (
  if "%~2"=="" (
    echo ERROR: --a-max requires a value argument
    exit /b 1
  )
  set "CLI_A_MAX=%~2"
  shift
) else if /I "%~1"=="--b-min" (
  if "%~2"=="" (
    echo ERROR: --b-min requires a value argument
    exit /b 1
  )
  set "CLI_B_MIN=%~2"
  shift
) else if /I "%~1"=="--b-max" (
  if "%~2"=="" (
    echo ERROR: --b-max requires a value argument
    exit /b 1
  )
  set "CLI_B_MAX=%~2"
  shift
) else if /I "%~1"=="--q0-min" (
  if "%~2"=="" (
    echo ERROR: --q0-min requires a value argument
    exit /b 1
  )
  set "CLI_Q0_MIN=%~2"
  shift
) else if /I "%~1"=="--q0-max" (
  if "%~2"=="" (
    echo ERROR: --q0-max requires a value argument
    exit /b 1
  )
  set "CLI_Q0_MAX=%~2"
  shift
) else (
  set "EXTRA_ARGS=!EXTRA_ARGS! ^"%~1^""
  set "EXTRA_ARGS_PRINT=!EXTRA_ARGS_PRINT! %~1"
)
shift
goto parse_args
:args_done

if defined RUNTIME_CACHE_ROOT (
  set "RUNTIME_CACHE_ROOT=%RUNTIME_CACHE_ROOT%"
) else (
  set "RUNTIME_CACHE_ROOT=%TEMP%\pychmp_runtime_cache"
)
if not defined MPLCONFIGDIR set "MPLCONFIGDIR=%RUNTIME_CACHE_ROOT%\matplotlib"
if not defined SUNPY_CONFIGDIR set "SUNPY_CONFIGDIR=%RUNTIME_CACHE_ROOT%\sunpy"
if not defined KMP_DUPLICATE_LIB_OK set "KMP_DUPLICATE_LIB_OK=TRUE"
if defined OMP_PREFIX (
  echo Unsetting OMP_PREFIX=%OMP_PREFIX% to avoid conflicting OpenMP runtimes
  set "OMP_PREFIX="
)
if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%"
if not exist "%SUNPY_CONFIGDIR%" mkdir "%SUNPY_CONFIGDIR%"

set "PYTHON_CMD="
if defined PYTHON_BIN (
  echo Using PYTHON_BIN override; skipping interpreter probe.
  set "PYTHON_CMD=%PYTHON_BIN%"
) else (
  call :try_python "C:\Users\gelu_\.conda\envs\pyampp-dev\python.exe"
  if not defined PYTHON_CMD call :try_python "%WORKSPACE_ROOT%\pyCHMP\.conda\python.exe"
  if not defined PYTHON_CMD call :try_python "%WORKSPACE_ROOT%\gximagecomputing\.conda\python.exe"
  if not defined PYTHON_CMD call :try_python "%USERPROFILE%\miniforge3\envs\suncast\python.exe"
  if not defined PYTHON_CMD call :try_python "%USERPROFILE%\miniforge3\python.exe"
  if not defined PYTHON_CMD call :find_conda_python
  if not defined PYTHON_CMD call :try_python "python"
)

set "EOVSA_MAPS_ROOT=%TESTDATA_REPO%\raw\eovsa_maps"
set "MODELS_ROOT=%TESTDATA_REPO%\raw\models"
if defined EBTEL_PATH (
  set "EBTEL_PATH=%EBTEL_PATH%"
) else (
  set "EBTEL_PATH=%TESTDATA_REPO%\raw\ebtel\ebtel_gxsimulator_euv\ebtel.sav"
)
if not exist "%TESTDATA_REPO%" (
  echo ERROR: Test-data repository not found: %TESTDATA_REPO%
  exit /b 1
)

call :latest_matching_dir "%EOVSA_MAPS_ROOT%" "eovsa_maps_*" LATEST_EOVSA_DIR
call :latest_matching_dir "%MODELS_ROOT%" "models_*" LATEST_MODEL_DIR

if not defined LATEST_EOVSA_DIR (
  echo ERROR: No dated EOVSA folder found under: %EOVSA_MAPS_ROOT%
  exit /b 1
)
if not defined LATEST_MODEL_DIR (
  echo ERROR: No dated model folder found under: %MODELS_ROOT%
  exit /b 1
)

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

if not defined ARTIFACTS_DIR set "ARTIFACTS_DIR=%TEMP%\pychmp_adaptive_ab_runs"
if not defined ARTIFACTS_STEM set "ARTIFACTS_STEM=adaptive_ab_search_single_frequency"
if defined CLI_ARTIFACTS_DIR set "ARTIFACTS_DIR=%CLI_ARTIFACTS_DIR%"
if defined CLI_ARTIFACTS_STEM set "ARTIFACTS_STEM=%CLI_ARTIFACTS_STEM%"
if /I "%PYCHMP_TIMESTAMP_ARTIFACTS%"=="1" (
  call :set_timestamp TIMESTAMP
  set "ARTIFACTS_STEM=%ARTIFACTS_STEM%_%TIMESTAMP%"
)
if not exist "%ARTIFACTS_DIR%" mkdir "%ARTIFACTS_DIR%"

if not exist "%OBS_FITS_PATH%" (
  echo ERROR: Observational FITS file not found: %OBS_FITS_PATH%
  exit /b 1
)
if not exist "%MODEL_H5_PATH%" (
  echo ERROR: Model H5 file not found: %MODEL_H5_PATH%
  exit /b 1
)
if not exist "%EBTEL_PATH%" (
  echo ERROR: EBTEL .sav file not found: %EBTEL_PATH%
  exit /b 1
)
if not defined PYTHON_CMD (
  echo ERROR: Could not find a Python interpreter with the adaptive-example dependency set.
  exit /b 1
)

set "TARGET_METRIC=chi2"
set "A_START=0.3"
set "B_START=2.7"
set "DA=0.3"
set "DB=0.3"
set "A_MIN=0.0"
set "A_MAX=1.2"
set "B_MIN=2.1"
set "B_MAX=3.6"
set "Q0_MIN=0.00001"
set "Q0_MAX=0.001"

if defined CLI_TARGET_METRIC set "TARGET_METRIC=%CLI_TARGET_METRIC%"
if defined CLI_A_START set "A_START=%CLI_A_START%"
if defined CLI_B_START set "B_START=%CLI_B_START%"
if defined CLI_DA set "DA=%CLI_DA%"
if defined CLI_DB set "DB=%CLI_DB%"
if defined CLI_A_MIN set "A_MIN=%CLI_A_MIN%"
if defined CLI_A_MAX set "A_MAX=%CLI_A_MAX%"
if defined CLI_B_MIN set "B_MIN=%CLI_B_MIN%"
if defined CLI_B_MAX set "B_MAX=%CLI_B_MAX%"
if defined CLI_Q0_MIN set "Q0_MIN=%CLI_Q0_MIN%"
if defined CLI_Q0_MAX set "Q0_MAX=%CLI_Q0_MAX%"

if defined CLI_ARTIFACT_H5 (
  set "ARTIFACT_H5=%CLI_ARTIFACT_H5%"
) 

if defined ARTIFACT_H5 (
  set "VIEWER_ARTIFACT_PATH=%ARTIFACT_H5%"
  set "ARTIFACT_ARGS=--artifact-h5 "%ARTIFACT_H5%""
) else (
  set "VIEWER_ARTIFACT_PATH=%ARTIFACTS_DIR%\%ARTIFACTS_STEM%.h5"
  set "ARTIFACT_ARGS=--artifacts-dir "%ARTIFACTS_DIR%" --artifacts-stem "%ARTIFACTS_STEM%""
)

set "BASE_ARGS="%OBS_FITS_PATH%" "%MODEL_H5_PATH%" --ebtel-path "%EBTEL_PATH%" --a-start %A_START% --b-start %B_START% --da %DA% --db %DB% --a-min %A_MIN% --a-max %A_MAX% --b-min %B_MIN% --b-max %B_MAX% --q0-min %Q0_MIN% --q0-max %Q0_MAX% --target-metric %TARGET_METRIC% --adaptive-bracketing --fallback-psf-bmaj-arcsec 5.77 --fallback-psf-bmin-arcsec 5.77 --fallback-psf-bpa-deg -17.5 --psf-ref-frequency-ghz 17.0 --psf-scale-inverse-frequency"

echo Using Python: %PYTHON_CMD%
echo Using test-data repo: %TESTDATA_REPO%
echo Using EOVSA folder: %LATEST_EOVSA_DIR%
echo Using model folder: %LATEST_MODEL_DIR%
if defined ARTIFACT_H5 (
  echo Artifact mode: explicit artifact-h5 ^(%ARTIFACT_H5%^)
) else (
  echo Artifact mode: reusable artifacts-dir/stem ^(%VIEWER_ARTIFACT_PATH%^)
)
echo Command: "%PYTHON_CMD%" examples\python\adaptive_ab_search_single_frequency.py %BASE_ARGS% %ARTIFACT_ARGS% %EXTRA_ARGS_PRINT%
echo Command: "%PYTHON_CMD%" examples\pychmp_view.py "%VIEWER_ARTIFACT_PATH%"
echo Launching adaptive_ab_search_single_frequency.py...
if "%DRY_RUN%"=="1" (
  echo Dry run only; command not executed.
  exit /b 0
)

pushd "%PYCHMP_REPO%"
set "PYCHMP_WRAPPER_COMMAND=%~f0 %*"
"%PYTHON_CMD%" examples\python\adaptive_ab_search_single_frequency.py %BASE_ARGS% %ARTIFACT_ARGS% %EXTRA_ARGS%
set "EXIT_CODE=%ERRORLEVEL%"
popd

echo Artifacts directory: %ARTIFACTS_DIR%
if defined ARTIFACT_H5 (
  echo Artifact path: %ARTIFACT_H5%
) else (
  echo Artifacts stem: %ARTIFACTS_STEM%
)
exit /b %EXIT_CODE%

:try_python
setlocal
set "CANDIDATE=%~1"
if not defined CANDIDATE endlocal & exit /b 0
if /I "%CANDIDATE%"=="python" goto probe_python
if not exist "%CANDIDATE%" endlocal & exit /b 0
:probe_python
"%CANDIDATE%" -c "required = ['gxrender.sdk','h5py','numpy','astropy.io.fits','scipy.ndimage','matplotlib.pyplot']; [__import__(name) for name in required]" >nul 2>&1
if errorlevel 1 endlocal & exit /b 0
endlocal & set "PYTHON_CMD=%~1" & exit /b 0

:find_conda_python
if not exist "%USERPROFILE%\.conda\envs" exit /b 0
for /f "delims=" %%I in ('dir /b /ad "%USERPROFILE%\.conda\envs" 2^>nul ^| sort') do (
  call :try_python "%USERPROFILE%\.conda\envs\%%~I\python.exe"
  if defined PYTHON_CMD exit /b 0
)
exit /b 0

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

:set_timestamp
setlocal
set "STAMP="
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value 2^>nul') do if not defined STAMP set "STAMP=%%I"
if defined STAMP (
  set "STAMP=%STAMP:~0,8%_%STAMP:~8,6%"
) else (
  set "STAMP=%RANDOM%%RANDOM%"
)
endlocal & set "%~1=%STAMP%" & exit /b 0
