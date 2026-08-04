@echo off
setlocal EnableExtensions DisableDelayedExpansion

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start.ps1" %*
set "BITGUARD_EXIT_CODE=%ERRORLEVEL%"

if not "%BITGUARD_EXIT_CODE%"=="0" (
    echo.
    echo BitGuard did not finish successfully. Review the error above, then press any key.
    pause >nul
)

exit /b %BITGUARD_EXIT_CODE%
