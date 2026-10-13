@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-local.ps1"
if errorlevel 1 (
    echo.
    echo GMEE could not start. Read the error above, fix it, and run this launcher again.
) else (
    echo.
    echo GMEE is running at http://localhost:3000. Keep Docker Desktop running.
)
pause
