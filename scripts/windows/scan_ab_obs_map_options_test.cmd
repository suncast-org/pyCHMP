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
set "OBS_SOURCE=external_fits"
set "OBS_MAP_ID="
set "OBS_PATH_OVERRIDE="
set "TR_MASK_BMIN_GAUSS=1000"
set "METRICS_MASK_THRESHOLD=0.1"
set "METRICS_MASK_FITS="
set "EUV_INSTRUMENT=AIA"
set "EUV_RESPONSE_SAV="

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--dry-run" (
  set "DRY_RUN=1"
) else if /I "%~1"=="--obs-source" (
  set "OBS_SOURCE=%~2"
  shift
) else if /I "%~1"=="--obs-map-id" (
  set "OBS_MAP_ID=%~2"
  shift
) else if /I "%~1"=="--obs-path" (
  set "OBS_PATH_OVERRIDE=%~2"
  shift
) else if /I "%~1"=="--tr-mask-bmin-gauss" (
  set "TR_MASK_BMIN_GAUSS=%~2"
  shift
) else if /I "%~1"=="--metrics-mask-threshold" (
  set "METRICS_MASK_THRESHOLD=%~2"
  shift
) else if /I "%~1"=="--metrics-mask-fits" (
  set "METRICS_MASK_FITS=%~2"
  shift
) else if /I "%~1"=="--euv-instrument" (
  set "EUV_INSTRUMENT=%~2"
  shift
) else if /I "%~1"=="--euv-response-sav" (
  set "EUV_RESPONSE_SAV=%~2"
  shift
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
) else if exist "%USERPROFILE%\miniforge3\envs\suncast\python.exe" (
  set "PYTHON_CMD=%USERPROFILE%\miniforge3\envs\suncast\python.exe"
) else if exist "%USERPROFILE%\miniforge3\python.exe" (
  set "PYTHON_CMD=%USERPROFILE%\miniforge3\python.exe"
) else (
  set "PYTHON_CMD=python"
)

set "EOVSA_MAPS_ROOT=%TESTDATA_REPO%\raw\eovsa_maps"
set "MODELS_ROOT=%TESTDATA_REPO%\raw\models"
set "RESPONSES_ROOT=%TESTDATA_REPO%\raw\responses"
set "EBTEL_PATH=%TESTDATA_REPO%\raw\ebtel\ebtel_gxsimulator_euv\ebtel.sav"
if not exist "%TESTDATA_REPO%" (
  echo ERROR: Test-data repository not found: %TESTDATA_REPO%
  exit /b 1
)
call :named_fixture_dir "%EOVSA_MAPS_ROOT%" "eovsa.synoptic_daily.20201126T200000Z.f2.874GHz.tb.disk.fits" LATEST_EOVSA_DIR
call :named_fixture_dir "%MODELS_ROOT%" "hmi.M_720s.20201126_195831.E18S19CR.CEA.NAS.GEN.CHR.h5" LATEST_MODEL_DIR
call :latest_any_dir "%RESPONSES_ROOT%" LATEST_RESPONSE_DIR

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
if not defined EUV_RESPONSE_SAV if defined LATEST_RESPONSE_DIR call :latest_matching_file "%LATEST_RESPONSE_DIR%" "resp_aia*.sav" EUV_RESPONSE_SAV

set "ARTIFACTS_DIR=%TEMP%\pychmp_ab_scan_runs"
if not defined ARTIFACTS_STEM set "ARTIFACTS_STEM=scan_ab_obs_map_options_test"
if /I "%PYCHMP_TIMESTAMP_ARTIFACTS%"=="1" call :set_timestamp TIMESTAMP & set "ARTIFACTS_STEM=%ARTIFACTS_STEM%_%TIMESTAMP%"
if not exist "%ARTIFACTS_DIR%" mkdir "%ARTIFACTS_DIR%"

if /I not "%OBS_SOURCE%"=="external_fits" if /I not "%OBS_SOURCE%"=="model_refmap" (
  echo ERROR: --obs-source must be external_fits or model_refmap
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
if /I "%OBS_SOURCE%"=="external_fits" (
  if not exist "%OBS_FITS_PATH%" (
    echo ERROR: Observational FITS file not found: %OBS_FITS_PATH%
    exit /b 1
  )
) else (
  if not defined OBS_MAP_ID (
    echo ERROR: --obs-map-id is required for --obs-source=model_refmap
    exit /b 1
  )
  if not exist "%EUV_RESPONSE_SAV%" (
    echo ERROR: EUV response SAV file not found: %EUV_RESPONSE_SAV%
    exit /b 1
  )
)
if defined METRICS_MASK_FITS if not exist "%METRICS_MASK_FITS%" (
  echo ERROR: Metrics-mask FITS file not found: %METRICS_MASK_FITS%
  exit /b 1
)

set "BASE_ARGS=--ebtel-path "%EBTEL_PATH%" --a-values 0.0,0.3,0.6 --b-values 2.1,2.4,2.7 --q0-min 0.00001 --q0-max 0.001 --target-metric chi2 --adaptive-bracketing --metrics-mask-threshold %METRICS_MASK_THRESHOLD% --tr-mask-bmin-gauss %TR_MASK_BMIN_GAUSS%"
if defined METRICS_MASK_FITS set "BASE_ARGS=%BASE_ARGS% --metrics-mask-fits "%METRICS_MASK_FITS%""
if defined OBS_PATH_OVERRIDE set "BASE_ARGS=%BASE_ARGS% --obs-path "%OBS_PATH_OVERRIDE%""
if defined ARTIFACT_H5 (
  set "ARTIFACT_ARGS=--artifact-h5 "%ARTIFACT_H5%""
) else (
  set "ARTIFACT_ARGS=--artifacts-dir "%ARTIFACTS_DIR%" --artifacts-stem "%ARTIFACTS_STEM%""
)

if /I "%OBS_SOURCE%"=="external_fits" (
  set "RUN_ARGS="%OBS_FITS_PATH%" "%MODEL_H5_PATH%" %BASE_ARGS% --psf-bmaj-arcsec 5.77 --psf-bmin-arcsec 5.77 --psf-bpa-deg -17.5 --psf-ref-frequency-ghz 17.0 --psf-scale-inverse-frequency %ARTIFACT_ARGS%"
) else (
  set "RUN_ARGS=--model-h5 "%MODEL_H5_PATH%" %BASE_ARGS% --obs-source model_refmap --obs-map-id "%OBS_MAP_ID%" --euv-instrument "%EUV_INSTRUMENT%" --euv-response-sav "%EUV_RESPONSE_SAV%" %ARTIFACT_ARGS%"
)

echo Using Python: %PYTHON_CMD%
echo Using test-data repo: %TESTDATA_REPO%
echo Using EOVSA folder: %LATEST_EOVSA_DIR%
echo Using model folder: %LATEST_MODEL_DIR%
echo Using observation source: %OBS_SOURCE%
if /I "%OBS_SOURCE%"=="model_refmap" (
  echo Using observation map id: %OBS_MAP_ID%
  if defined LATEST_RESPONSE_DIR echo Using response folder: %LATEST_RESPONSE_DIR%
  echo Using EUV instrument: %EUV_INSTRUMENT%
  echo Using EUV response SAV: %EUV_RESPONSE_SAV%
)
echo Using EUV TR-mask Bmin [G]: %TR_MASK_BMIN_GAUSS%
echo Using metrics-mask threshold: %METRICS_MASK_THRESHOLD%
if defined METRICS_MASK_FITS echo Using metrics-mask FITS: %METRICS_MASK_FITS%
if defined ARTIFACT_H5 (
  echo Artifact mode: explicit artifact-h5 ^(%ARTIFACT_H5%^)
) else (
  echo Artifact mode: reusable artifacts-dir/stem ^(%ARTIFACTS_DIR%\%ARTIFACTS_STEM%.h5^)
)
echo Command: "%PYTHON_CMD%" examples\scan_ab_obs_map.py %RUN_ARGS% %EXTRA_ARGS%
if "%DRY_RUN%"=="1" (
  echo Dry run only; command not executed.
  exit /b 0
)

pushd "%PYCHMP_REPO%"
"%PYTHON_CMD%" examples\scan_ab_obs_map.py %RUN_ARGS% %EXTRA_ARGS%
set "EXIT_CODE=%ERRORLEVEL%"
popd
echo Artifacts directory: %ARTIFACTS_DIR%
if defined ARTIFACT_H5 (
  echo Artifact path: %ARTIFACT_H5%
) else (
  echo Artifacts stem: %ARTIFACTS_STEM%
)
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

:latest_any_dir
setlocal
set "PARENT=%~1"
set "LATEST_NAME="
for /f "delims=" %%I in ('dir /b /ad "%PARENT%\*" 2^>nul ^| sort') do set "LATEST_NAME=%%I"
if defined LATEST_NAME (
  endlocal & set "%~2=%PARENT%\%LATEST_NAME%" & exit /b 0
)
endlocal & set "%~2=" & exit /b 0

:latest_matching_file
setlocal
set "PARENT=%~1"
set "MASK=%~2"
set "LATEST_NAME="
for /f "delims=" %%I in ('dir /b /a-d "%PARENT%\%MASK%" 2^>nul ^| sort') do set "LATEST_NAME=%%I"
if defined LATEST_NAME (
  endlocal & set "%~3=%PARENT%\%LATEST_NAME%" & exit /b 0
)
endlocal & set "%~3=" & exit /b 0

:named_fixture_dir
setlocal
set "PARENT=%~1"
set "FILENAME=%~2"
set "MATCH_PATH="
for /f "delims=" %%I in ('dir /b /s /a-d "%PARENT%\%FILENAME%" 2^>nul ^| sort') do set "MATCH_PATH=%%~dpI"
if defined MATCH_PATH (
  if "!MATCH_PATH:~-1!"=="\" set "MATCH_PATH=!MATCH_PATH:~0,-1!"
  endlocal & set "%~3=%MATCH_PATH%" & exit /b 0
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
