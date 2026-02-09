@echo off
chcp 65001 >nul
echo ============================================================
echo   update_daily_page v2.0
echo ============================================================
echo.

set "SCRIPT_DIR=%~dp0"
set "PY_SCRIPT=%SCRIPT_DIR%update_daily_page.py"

if not exist "%PY_SCRIPT%" (
    echo [ERROR] update_daily_page.py not found
    pause
    exit /b 1
)

echo   [1] Auto detect latest master file
echo   [2] Manual input master file name
echo.
set /p "MODE=Select (1 or 2, default=1): "

if "%MODE%"=="" set MODE=1
if "%MODE%"=="1" goto AUTO
if "%MODE%"=="2" goto MANUAL
echo [ERROR] Invalid selection.
pause
exit /b 1

:AUTO
echo.
echo [INFO] Searching for latest master file...
echo.
python "%PY_SCRIPT%"
goto CHECK_RESULT

:MANUAL
set "DATA_DIR=%SCRIPT_DIR%data\krx_ohlcv\daily"
echo.
echo Data folder: %DATA_DIR%
echo.
echo Available master files:
echo -----------------------------------------------------------
dir /b "%DATA_DIR%\krx_stock_master_updated*.csv" 2>nul
echo -----------------------------------------------------------
echo.
set /p "MASTER_FILE=Enter master file name: "

if "%MASTER_FILE%"=="" (
    echo [ERROR] No file name entered.
    pause
    exit /b 1
)

set "MASTER_PATH=%DATA_DIR%\%MASTER_FILE%"

if not exist "%MASTER_PATH%" (
    echo [ERROR] File not found: %MASTER_PATH%
    pause
    exit /b 1
)

echo.
python "%PY_SCRIPT%" "%MASTER_PATH%"
goto CHECK_RESULT

:CHECK_RESULT
echo.
if %ERRORLEVEL% equ 0 (
    echo ============================================================
    echo   Done! krx-daily.html updated successfully.
    echo ============================================================
) else (
    echo ============================================================
    echo   ERROR - Check messages above.
    echo ============================================================
)
echo.
pause
