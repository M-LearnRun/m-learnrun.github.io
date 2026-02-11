@echo off
chcp 65001 >nul
echo ============================================================
echo   GitHub Push - 주가지수 자동업데이트 설정
echo ============================================================
echo.

cd /d "%~dp0"

echo [1/4] 워크플로우 파일 추가...
git add .github/workflows/update-index-data.yml

echo [2/4] 주가지수 CSV 파일 추가...
git add data/chapter8/fdr_INDEX_p1d_KS11.csv
git add data/chapter8/fdr_INDEX_p1d_KQ11.csv
git add data/chapter8/fdr_INDEX_p1d_US-SnP500.csv
git add data/chapter8/fdr_INDEX_p1d_QQQ.csv

echo [3/4] 기타 변경파일 추가...
git add index.html
git add update_index_data.py
git add update_index_data.bat

echo [4/4] Commit & Push...
git commit -m "Add auto-update workflow and index data"
git push

echo.
echo ============================================================
echo   완료! GitHub에 푸시되었습니다.
echo   이제 GitHub에서 Actions 권한을 설정하세요.
echo ============================================================
echo.
pause
