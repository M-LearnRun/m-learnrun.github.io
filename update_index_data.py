# -*- coding: utf-8 -*-
# update_index_data.py
# FinanceDataReader를 이용해 주요 주가지수 데이터를 다운로드하고
# data/chapter8/ 폴더에 CSV 파일로 저장합니다.
#
# 대상 지수:
#   - KOSPI   (KS11)  -> fdr_INDEX_p1d_KS11.csv
#   - KOSDAQ  (KQ11)  -> fdr_INDEX_p1d_KQ11.csv
#   - S&P 500 (US500) -> fdr_INDEX_p1d_US-SnP500.csv
#   - QQQ     (QQQ)   -> fdr_INDEX_p1d_QQQ.csv
#
# 사용법:
#   python update_index_data.py
#   C:\ProgramData\anaconda3\python.exe update_index_data.py

import os
import sys
from datetime import datetime, timedelta

# FinanceDataReader 임포트
try:
    import FinanceDataReader as fdr
except ImportError:
    print("[오류] FinanceDataReader가 설치되지 않았습니다.")
    print("       다음 명령어로 설치하세요:")
    print("       pip install finance-datareader")
    sys.exit(1)

import pandas as pd

# ─────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────

# 저장 폴더 (이 스크립트 기준 상대경로)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "chapter8")

# 다운로드 기간: 최근 3년 (넉넉하게)
END_DATE   = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")

# 지수 설정: (FDR 심볼, 출력 파일명, 표시명)
# KOSPI/KOSDAQ: Yahoo Finance 심볼(^KS11, ^KQ11) 사용
INDICES = [
    ("^KS11",  "fdr_INDEX_p1d_KS11.csv",       "KOSPI"),
    ("^KQ11",  "fdr_INDEX_p1d_KQ11.csv",        "KOSDAQ"),
    ("US500",  "fdr_INDEX_p1d_US-SnP500.csv",   "S&P 500"),
    ("QQQ",    "fdr_INDEX_p1d_QQQ.csv",         "QQQ"),
]

# ─────────────────────────────────────────────────────────────
# 함수
# ─────────────────────────────────────────────────────────────

def download_index(symbol: str, start: str, end: str, display_name: str) -> pd.DataFrame | None:
    """지수 데이터를 다운로드하여 DataFrame으로 반환"""
    print(f"  [{display_name}] 다운로드 중... (심볼: {symbol}, {start} ~ {end})")
    try:
        df = fdr.DataReader(symbol, start, end)
        if df is None or df.empty:
            print(f"  [{display_name}] 경고: 데이터가 비어있습니다.")
            return None
        print(f"  [{display_name}] 완료: {len(df)}행 수신")
        return df
    except Exception as e:
        print(f"  [{display_name}] 오류: {e}")
        return None


def save_to_csv(df: pd.DataFrame, output_path: str, display_name: str) -> bool:
    """DataFrame을 CSV로 저장 (기존 파일이 있으면 최신 데이터로 덮어쓰기)"""
    try:
        # 인덱스(날짜)를 'Time' 컬럼으로 저장 (기존 CSV 포맷과 동일하게)
        df_out = df.copy()
        df_out.index.name = "Time"

        # 컬럼명 소문자 통일
        df_out.columns = [c.lower() for c in df_out.columns]

        # close 컬럼이 없으면 'Close' → 'close' 재시도
        required = {"open", "high", "low", "close"}
        missing = required - set(df_out.columns)
        if missing:
            print(f"  [{display_name}] 경고: 컬럼 누락 {missing}, 저장 스킵")
            return False

        df_out.to_csv(output_path, encoding="utf-8-sig")
        print(f"  [{display_name}] 저장 완료: {output_path}")
        return True
    except Exception as e:
        print(f"  [{display_name}] 저장 오류: {e}")
        return False


def main():
    print("=" * 60)
    print("  주가지수 데이터 업데이트")
    print("=" * 60)
    print(f"  기간: {START_DATE} ~ {END_DATE}")
    print(f"  저장위치: {OUTPUT_DIR}")
    print()

    # 출력 폴더 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    success_count = 0
    fail_count = 0

    for symbol, filename, display_name in INDICES:
        output_path = os.path.join(OUTPUT_DIR, filename)
        print(f"[{display_name}]")

        df = download_index(symbol, START_DATE, END_DATE, display_name)
        if df is not None:
            ok = save_to_csv(df, output_path, display_name)
            if ok:
                success_count += 1
            else:
                fail_count += 1
        else:
            fail_count += 1
        print()

    print("=" * 60)
    print(f"  완료: 성공 {success_count}개 / 실패 {fail_count}개")
    print("=" * 60)

    if success_count > 0:
        print("\n[OK] index.html에서 차트 데이터를 바로 사용할 수 있습니다.")
    if fail_count > 0:
        print("\n[경고] 일부 지수 다운로드에 실패했습니다. 인터넷 연결을 확인하세요.")


if __name__ == "__main__":
    main()
