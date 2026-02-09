@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo.
echo ============================================================
echo   update_daily_page v2.0
echo ============================================================
echo.
echo   [1] Auto detect latest master file
echo   [2] Manual input master file name
echo.
set /p "MODE=Select (1 or 2, default=1): "
if "%MODE%"=="" set MODE=1
if "%MODE%"=="2" goto MANUAL
echo.
python update_daily_page.py
goto END
:MANUAL
echo.
dir /b "data\krx_ohlcv\daily\krx_stock_master_updated*.csv" 2>nul
echo.
set /p "MF=Enter master file name: "
python update_daily_page.py "data\krx_ohlcv\daily\%MF%"
:END
echo.
pause
