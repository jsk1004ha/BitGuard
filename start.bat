@echo off
setlocal EnableExtensions DisableDelayedExpansion

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start.ps1" -PauseOnError %*
set "BITGUARD_EXIT_CODE=%ERRORLEVEL%"

exit /b %BITGUARD_EXIT_CODE%
