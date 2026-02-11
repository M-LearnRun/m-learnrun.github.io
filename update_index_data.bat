@echo off
chcp 65001 >nul
echo ============================================================
echo   주가지수 데이터 업데이트 (KOSPI / KOSDAQ / S^&P500 / QQQ)
echo ============================================================
echo.

cd /d "%~dp0"

"C:\ProgramData\anaconda3\python.exe" update_index_data.py

echo.
echo 완료! 아무 키나 누르면 종료합니다.
pause >nul
