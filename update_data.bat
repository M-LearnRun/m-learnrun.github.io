@echo off
chcp 65001 >nul
echo ================================
echo 머신러너 데이터 업데이트 스크립트
echo ================================
echo.

echo [1/3] HTML 목록 자동 생성 중...
python generate_price_list.py
echo.

echo [2/3] ZIP 파일 생성 중...
cd data
powershell Compress-Archive -Path krx_ohlcv\daily\*.csv -DestinationPath krx_ohlcv_daily.zip -Force
cd ..
echo ✅ krx_ohlcv_daily.zip 생성 완료
echo.

echo [3/3] 완료!
echo.
echo 다음 파일들이 업데이트되었습니다:
echo   - krx-daily.html (자동 생성)
echo   - data\krx_ohlcv_daily.zip (전체 다운로드용)
echo.
pause
