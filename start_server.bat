@echo off
chcp 65001 >nul
echo ================================
echo 머신러너 웹사이트 로컬 서버 시작
echo ================================
echo.
echo 🌐 서버 시작 중...
echo 📍 주소: http://localhost:8000
echo.
echo ⚠️  서버를 종료하려면 Ctrl+C를 누르세요
echo.
echo ================================
echo.
python -m http.server 8000
