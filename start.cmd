@echo off
rem Cabo one-click server launcher (double-clickable).
rem Passes all arguments to start.ps1, e.g.  start.cmd -Port 9000
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
if errorlevel 1 (
  echo.
  echo Failed to start. See the error message above.
  pause
)
endlocal
