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
set "VALIDATION_DOMAIN=mw"
set "TR_MASK_BMIN_GAUSS=1000"
set "METRICS_MASK_THRESHOLD=0.1"
set "METRICS_MASK_FITS="

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--dry-run" (
  set "DRY_RUN=1"
) else if /I "%~1"=="--domain" (
  set "VALIDATION_DOMAIN=%~2"
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

set "MODELS_ROOT=%TESTDATA_REPO%\raw\models"
set "RESPONSES_ROOT=%TESTDATA_REPO%\raw\responses"
set "EBTEL_PATH=%TESTDATA_REPO%\raw\ebtel\ebtel_gxsimulator_euv\ebtel.sav"
if not exist "%TESTDATA_REPO%" (
  echo ERROR: Test-data repository not found: %TESTDATA_REPO%
  exit /b 1
)
call :named_fixture_dir "%MODELS_ROOT%" "hmi.M_720s.20201126_195831.E18S19CR.CEA.NAS.GEN.CHR.h5" LATEST_MODEL_DIR
call :latest_any_dir "%RESPONSES_ROOT%" LATEST_RESPONSE_DIR

if defined MODEL_PATH (
  set "MODEL_PATH=%MODEL_PATH%"
) else (
  set "MODEL_PATH=%LATEST_MODEL_DIR%\hmi.M_720s.20201126_195831.E18S19CR.CEA.NAS.GEN.CHR.h5"
)
if not defined FREQUENCY_GHZ set "FREQUENCY_GHZ=5.8"
if not defined OBS_MAP_ID set "OBS_MAP_ID=AIA_171"
if not defined EUV_INSTRUMENT set "EUV_INSTRUMENT=AIA"
if not defined EUV_RESPONSE_SAV if defined LATEST_RESPONSE_DIR call :latest_matching_file "%LATEST_RESPONSE_DIR%" "resp_aia*.sav" EUV_RESPONSE_SAV

set "ARTIFACTS_DIR=%TEMP%\pychmp_q0_runs"
call :set_timestamp TIMESTAMP
if /I "%VALIDATION_DOMAIN%"=="mw" (
  set "ARTIFACTS_STEM=validate_q0_%TIMESTAMP%"
) else (
  set "ARTIFACTS_STEM=validate_q0_euv_%TIMESTAMP%"
)
if not exist "%ARTIFACTS_DIR%" mkdir "%ARTIFACTS_DIR%"

if /I not "%VALIDATION_DOMAIN%"=="mw" if /I not "%VALIDATION_DOMAIN%"=="euv" if /I not "%VALIDATION_DOMAIN%"=="uv" (
  echo ERROR: --domain must be one of: mw, euv, uv
  exit /b 1
)
if not exist "%MODEL_PATH%" (
  echo ERROR: Model H5 file not found: %MODEL_PATH%
  exit /b 1
)
if not exist "%EBTEL_PATH%" (
  echo ERROR: EBTEL .sav file not found: %EBTEL_PATH%
  exit /b 1
)
if /I not "%VALIDATION_DOMAIN%"=="mw" (
  if not exist "%EUV_RESPONSE_SAV%" (
    echo ERROR: EUV response SAV file not found: %EUV_RESPONSE_SAV%
    exit /b 1
  )
)
if defined METRICS_MASK_FITS if not exist "%METRICS_MASK_FITS%" (
  echo ERROR: Metrics-mask FITS file not found: %METRICS_MASK_FITS%
  exit /b 1
)

set "RUN_ARGS=--model-path "%MODEL_PATH%" --ebtel-path "%EBTEL_PATH%" --adaptive-bracketing --noise-frac 0.05 --target-metric chi2 --domain %VALIDATION_DOMAIN% --artifacts-dir "%ARTIFACTS_DIR%" --artifacts-stem "%ARTIFACTS_STEM%" --show-plot --metrics-mask-threshold %METRICS_MASK_THRESHOLD%"
if /I "%VALIDATION_DOMAIN%"=="mw" (
  set "RUN_ARGS=%RUN_ARGS% --frequency-ghz %FREQUENCY_GHZ%"
) else (
  set "RUN_ARGS=%RUN_ARGS% --obs-map-id "%OBS_MAP_ID%" --euv-instrument "%EUV_INSTRUMENT%" --euv-response-sav "%EUV_RESPONSE_SAV%" --tr-mask-bmin-gauss %TR_MASK_BMIN_GAUSS%"
)
if defined METRICS_MASK_FITS set "RUN_ARGS=%RUN_ARGS% --metrics-mask-fits "%METRICS_MASK_FITS%""

echo Using Python: %PYTHON_CMD%
echo Using test-data repo: %TESTDATA_REPO%
echo Using model folder: %LATEST_MODEL_DIR%
echo Using validation domain: %VALIDATION_DOMAIN%
if /I "%VALIDATION_DOMAIN%"=="mw" (
  echo Using MW frequency [GHz]: %FREQUENCY_GHZ%
) else (
  if defined LATEST_RESPONSE_DIR echo Using response folder: %LATEST_RESPONSE_DIR%
  echo Using EUV map id: %OBS_MAP_ID%
  echo Using EUV instrument: %EUV_INSTRUMENT%
  echo Using EUV response SAV: %EUV_RESPONSE_SAV%
  echo Using EUV TR-mask Bmin [G]: %TR_MASK_BMIN_GAUSS%
)
echo Using metrics-mask threshold: %METRICS_MASK_THRESHOLD%
if defined METRICS_MASK_FITS echo Using metrics-mask FITS: %METRICS_MASK_FITS%
echo Command: "%PYTHON_CMD%" examples\validate_q0_recovery.py %RUN_ARGS% %EXTRA_ARGS%
if "%DRY_RUN%"=="1" (
  echo Dry run only; command not executed.
  exit /b 0
)

pushd "%PYCHMP_REPO%"
"%PYTHON_CMD%" examples\validate_q0_recovery.py %RUN_ARGS% %EXTRA_ARGS%
set "EXIT_CODE=%ERRORLEVEL%"
popd
echo Artifacts directory: %ARTIFACTS_DIR%
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
