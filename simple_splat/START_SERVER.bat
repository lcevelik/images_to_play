@echo off
title Gaussian Splatting Server - Standalone Edition
color 0A

echo ========================================================================
echo   Gaussian Splatting Server - STANDALONE EDITION
echo   Everything Included - Zero Dependencies!
echo ========================================================================
echo.

REM Set up paths
set "ROOT=%~dp0"
set "PYTHON=%ROOT%Python"
set "APP=%ROOT%App"
set "COLMAP=%ROOT%COLMAP\bin"
set "COLMAP_LIB=%ROOT%COLMAP\lib"
set "BRUSH=%ROOT%Brush"

REM Add bundled tools to PATH
set "PATH=%PYTHON%;%PYTHON%\Scripts;%COLMAP%;%COLMAP_LIB%;%BRUSH%;%PATH%"

REM -------------------------------------------------------------------------
echo [1/3] Checking Python...
"%PYTHON%\python.exe" --version
if errorlevel 1 (
    echo ERROR: Python not found at %PYTHON%
    pause
    exit /b 1
)
echo       Python OK

REM -------------------------------------------------------------------------
echo [2/3] Installing dependencies (first run only)...
cd /d "%APP%"

if exist ".deps_installed" goto :deps_done

echo       Installing Python packages from bundled wheels...

if exist "%APP%\wheels" goto :install_offline

echo       WARNING: wheels folder missing, trying online install...
"%PYTHON%\python.exe" -m pip install --no-warn-script-location -r requirements.txt
if errorlevel 1 goto :deps_error
goto :deps_mark_done

:install_offline
"%PYTHON%\python.exe" -m pip install --no-warn-script-location --no-index --find-links="%APP%\wheels" -r requirements.txt
if errorlevel 1 goto :deps_error

:deps_mark_done
echo.  > .deps_installed
echo       Dependencies installed successfully!
goto :start_server

:deps_error
echo.
echo ERROR: Failed to install dependencies.
echo Check that App\wheels\ folder exists and contains the .whl files.
pause
exit /b 1

:deps_done
echo       Dependencies already installed (skipping)

REM -------------------------------------------------------------------------
:start_server
echo [3/3] Starting server...
echo.
echo ========================================================================
echo   Server running at: http://localhost:5000
echo   Press Ctrl+C to stop
echo ========================================================================
echo.

"%PYTHON%\python.exe" app.py

REM Only reaches here if the server exits
if errorlevel 1 (
    echo.
    echo ========================================================================
    echo   ERROR: Server stopped unexpectedly.
    echo ========================================================================
)
pause
