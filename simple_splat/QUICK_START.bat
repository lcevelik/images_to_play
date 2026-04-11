@echo off
title Gaussian Splatting - Quick Start

echo ========================================================================
echo   Gaussian Splatting - Quick Start
echo ========================================================================
echo.

REM Start the server in a separate minimized window
start "Gaussian Splatting Server" /min "%~dp0START_SERVER.bat"

echo   Server starting in background...
echo   (First run installs packages and may take ~30 seconds)
echo.

REM Wait for the server to actually respond on port 5000
REM curl.exe is built into Windows 10/11
set MAX_WAIT=120
set WAITED=0

:wait_loop
timeout /t 1 /nobreak >nul
set /a WAITED+=1

curl -s --max-time 1 http://localhost:5000 >nul 2>&1
if %errorlevel% == 0 goto server_ready

if %WAITED% GEQ %MAX_WAIT% goto timeout_error

echo   Waiting for server... (%WAITED%s / %MAX_WAIT%s max)
goto wait_loop

:server_ready
echo.
echo   Server is ready! Opening browser...
echo.
start http://localhost:5000
goto end

:timeout_error
echo.
echo   ERROR: Server did not respond within %MAX_WAIT% seconds.
echo   Please run START_SERVER.bat directly to see error messages.
echo.
pause

:end
