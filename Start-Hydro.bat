@echo off
rem Plug-and-play launcher for Windows. Double-click, or run from a terminal:
rem   Start-Hydro.bat                preflight + calibrate + self-check, then start the service
rem   Start-Hydro.bat --check-only   gate only, start nothing
rem   Start-Hydro.bat --repair       rebuild the Python environment first
rem Extra arguments go to "python -m hydro deploy".
setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"
set "HYDRO_ROOT=%~dp0"
if "%HYDRO_ROOT:~-1%"=="\" set "HYDRO_ROOT=%HYDRO_ROOT:~0,-1%"
set "PYTHONPATH=%HYDRO_ROOT%\src;%PYTHONPATH%"
set "PYTHONDONTWRITEBYTECODE=1"

set CHECK_ONLY=0
set REPAIR=0
set ARGS=
:parse
if "%~1"=="" goto parsed
if /i "%~1"=="--check-only" ( set CHECK_ONLY=1 ) else if /i "%~1"=="--repair" ( set REPAIR=1 ) else ( set ARGS=!ARGS! %1 )
shift
goto parse
:parsed

set "PY="
for %%C in ("py -3.13" "py -3.12" "python") do (
  if not defined PY (
    %%~C -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1 && set "PY=%%~C"
  )
)
if not defined PY (
  echo ERROR: Python 3.12 or newer was not found. Install it from https://www.python.org/downloads/
  exit /b 3
)

rem The environment lives under LOCALAPPDATA, not on the drive (USB sticks are slow and often FAT).
for /f %%H in ('%PY% -c "import hashlib,os;print(hashlib.sha256(os.environ['HYDRO_ROOT'].encode()).hexdigest()[:10])"') do set "KEY=%%H"
if not defined HYDRO_VENV set "HYDRO_VENV=%LOCALAPPDATA%\hydro\venv-%KEY%"
if "%REPAIR%"=="1" if exist "%HYDRO_VENV%" rmdir /s /q "%HYDRO_VENV%"

if not exist "%HYDRO_VENV%\Scripts\python.exe" (
  echo Creating Python environment in %HYDRO_VENV% ...
  %PY% -m venv "%HYDRO_VENV%" || exit /b 4
)
"%HYDRO_VENV%\Scripts\python.exe" -c "import numpy, scipy, pandas, yaml, onnxruntime, asyncua" >nul 2>&1
if errorlevel 1 (
  if exist "%HYDRO_ROOT%\wheelhouse" (
    echo Installing from the offline wheelhouse ...
    "%HYDRO_VENV%\Scripts\python.exe" -m pip install --quiet --no-index --find-links "%HYDRO_ROOT%\wheelhouse" -r "%HYDRO_ROOT%\wheelhouse\requirements.txt" || exit /b 5
  ) else (
    echo No wheelhouse on the drive; installing from the package index ^(needs internet^) ...
    "%HYDRO_VENV%\Scripts\python.exe" -m pip install --quiet numpy scipy pandas pyyaml onnxruntime asyncua || exit /b 5
  )
)

if "%CHECK_ONLY%"=="1" (
  "%HYDRO_VENV%\Scripts\python.exe" -m hydro deploy %ARGS%
) else (
  "%HYDRO_VENV%\Scripts\python.exe" -m hydro deploy --start %ARGS%
)
exit /b %ERRORLEVEL%
