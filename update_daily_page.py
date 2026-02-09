#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
머신러너 - 일간 데이터 페이지 자동 업데이트 스크립트

마스터 CSV 파일을 읽어 krx-daily.html의 종목 목록을 자동으로 생성합니다.

사용법:
    python update_daily_page.py                          # 최신 마스터 파일 자동 탐색
    python update_daily_page.py <마스터_파일_경로>        # 마스터 파일 직접 지정

변경 이력:
    2026-02-09  v2.0  자동 마스터 파일 탐색, 종가/시총 표시, 타임스탬프 추가
"""

import csv
import glob
import json
import os
import re
import sys
import shutil
from datetime import datetime
import html as html_lib

# ─── 설정 ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_FILE = os.path.join(BASE_DIR, "krx-daily.html")
DATA_DIR = os.path.join(BASE_DIR, "data", "krx_ohlcv", "daily")
MASTER_GLOB_PATTERN = "krx_stock_master_updated*.csv"
MAPPING_FILE = os.path.join(BASE_DIR, "file_id_mapping.json")


# ─── 마스터 파일 탐색 ────────────────────────────────────────────
def find_latest_master_file(data_dir):
    """
    데이터 디렉토리에서 가장 최신 마스터 CSV 파일을 자동으로 찾습니다.
    파일명 패턴: krx_stock_master_updated*.csv

    Args:
        data_dir (str): 데이터 디렉토리 경로

    Returns:
        str: 최신 마스터 파일 경로 또는 None
    """
    pattern = os.path.join(data_dir, MASTER_GLOB_PATTERN)
    candidates = glob.glob(pattern)

    if not candidates:
        return None

    # 파일명 기준 정렬 (날짜가 파일명에 포함되므로 사전순 = 시간순)
    candidates.sort()
    latest = candidates[-1]
    print(f"[정보] 최신 마스터 파일 자동 탐색: {os.path.basename(latest)}")

    if len(candidates) > 1:
        print(f"       (후보 {len(candidates)}개 중 가장 최신 파일 선택)")

    return latest


# ─── 구글 드라이브 매핑 로드 ──────────────────────────────────────
def load_drive_mapping(mapping_file=None):
    """
    구글 드라이브 FILE_ID 매핑을 로드합니다.

    Args:
        mapping_file (str): 매핑 파일 경로 (기본값: file_id_mapping.json)

    Returns:
        dict: {ticker: file_id} 매핑 딕셔너리
    """
    if mapping_file is None:
        mapping_file = MAPPING_FILE

    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        print(f"[정보] 구글 드라이브 매핑 로드: {len(mapping):,}개 종목")
        return mapping
    except FileNotFoundError:
        print(f"[경고] {mapping_file} 파일을 찾을 수 없습니다.")
        print("       generate_drive_mapping.py를 먼저 실행하세요.")
        print("       로컬 다운로드 링크를 사용합니다.")
        return {}
    except json.JSONDecodeError as e:
        print(f"[경고] 매핑 파일 형식 오류: {e}")
        print("       로컬 다운로드 링크를 사용합니다.")
        return {}
    except Exception as e:
        print(f"[경고] 매핑 파일 로드 실패: {e}")
        print("       로컬 다운로드 링크를 사용합니다.")
        return {}


# ─── 마스터 파일 파싱 ────────────────────────────────────────────
def parse_master_file(master_path):
    """
    마스터 CSV 파일을 파싱하여 종목 정보를 추출합니다.

    Args:
        master_path (str): 마스터 CSV 파일 경로

    Returns:
        list: (ticker_code, company_name, last_update, status) 튜플 리스트
    """
    stocks = []

    try:
        with open(master_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticker = row.get('ticker_code', '').strip()
                company = row.get('company_name', '').strip()
                last_update = row.get('last_update', '').strip()
                status = row.get('status', '').strip()

                if ticker and company:
                    stocks.append((ticker, company, last_update, status))

        print(f"[정보] 마스터 파일에서 {len(stocks):,}개 종목을 읽었습니다.")
        return stocks

    except Exception as e:
        print(f"[오류] 마스터 파일 읽기 실패: {e}")
        sys.exit(1)


# ─── CSV 통계 추출 ───────────────────────────────────────────────
def get_csv_statistics(csv_path):
    """
    CSV 파일의 통계 정보를 추출합니다.

    Args:
        csv_path (str): CSV 파일 경로

    Returns:
        dict: 통계 정보 딕셔너리 또는 None (파일 없음)
              keys: first_date, last_date, count, last_close,
                    week52_high, week52_low
    """
    if not os.path.exists(csv_path):
        return None

    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            if not rows:
                return None

            first_row = rows[0]
            last_row = rows[-1]

            # 기본 정보
            stats = {
                'first_date': first_row.get('date', ''),
                'last_date': last_row.get('date', ''),
                'count': len(rows),
            }

            # 최신 종가 (빈 값 처리)
            last_close = last_row.get('close', '')
            if last_close and last_close.strip():
                try:
                    stats['last_close'] = float(last_close)
                except ValueError:
                    stats['last_close'] = None
            else:
                stats['last_close'] = None

            # 52주 신고가/신저가 계산 (최근 252 거래일)
            # 252 거래일 = 약 52주 (1년)
            recent_252 = rows[-252:] if len(rows) >= 252 else rows

            highs = []
            lows = []
            for row in recent_252:
                high = row.get('high', '')
                low = row.get('low', '')

                if high and high.strip():
                    try:
                        highs.append(float(high))
                    except ValueError:
                        pass

                if low and low.strip():
                    try:
                        lows.append(float(low))
                    except ValueError:
                        pass

            stats['week52_high'] = max(highs) if highs else None
            stats['week52_low'] = min(lows) if lows else None

            return stats

    except Exception as e:
        print(f"[경고] CSV 읽기 실패 ({os.path.basename(csv_path)}): {e}")
        return None


# ─── 포맷 유틸리티 ───────────────────────────────────────────────
def format_date(date_str):
    """날짜 형식 변환: YYYY-MM-DD → YYYY.MM"""
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return date_obj.strftime('%Y.%m')
    except:
        return date_str


def format_price(value):
    """가격 포맷: 12345 → 12,345원"""
    if value is None:
        return "-"
    return f"{int(value):,}원"


# ─── HTML 생성 ───────────────────────────────────────────────────
def generate_data_item_html(ticker, company, stats, csv_filename, drive_mapping=None):
    """
    개별 종목의 HTML div 블록을 생성합니다.
    krx-daily.html의 들여쓰기(공백 20칸 = data-item 기준)에 맞춥니다.

    Args:
        ticker (str): 티커 코드
        company (str): 회사명
        stats (dict): CSV 통계 정보
        csv_filename (str): CSV 파일명
        drive_mapping (dict): 구글 드라이브 FILE_ID 매핑 (선택)
    """
    safe_company = html_lib.escape(company)
    formatted_first = format_date(stats['first_date'])
    formatted_last = format_date(stats['last_date'])
    price_str = format_price(stats.get('last_close'))
    high_52w_str = format_price(stats.get('week52_high'))
    low_52w_str = format_price(stats.get('week52_low'))

    # 다운로드 URL 결정: 구글 드라이브 우선, 없으면 로컬 경로
    file_id = drive_mapping.get(ticker) if drive_mapping else None

    if file_id:
        # 구글 드라이브 직접 다운로드 URL
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        download_attr = ""
    else:
        # 로컬 경로로 fallback
        download_url = f"data/krx_ohlcv/daily/{csv_filename}"
        download_attr = " download"

    # 기존 HTML의 들여쓰기(공백 20칸)에 맞춤
    html = f'''                    <div class="data-item">
                        <h4>{ticker} - {safe_company}</h4>
                        <div class="data-info">
                            <div class="data-info-item">
                                <span>기간:</span>
                                <span>{formatted_first} ~ {formatted_last}</span>
                            </div>
                            <div class="data-info-item">
                                <span>데이터 수:</span>
                                <span>{stats["count"]:,}개</span>
                            </div>
                            <div class="data-info-item">
                                <span>최신 주가:</span>
                                <span>{price_str}</span>
                            </div>
                            <div class="data-info-item">
                                <span>52주 신고가:</span>
                                <span>{high_52w_str}</span>
                            </div>
                            <div class="data-info-item">
                                <span>52주 신저가:</span>
                                <span>{low_52w_str}</span>
                            </div>
                        </div>
                        <a href="{download_url}" class="download-link"{download_attr}>다운로드</a>
                    </div>'''

    return html


def generate_all_data_items(stocks, data_dir, drive_mapping=None):
    """
    모든 종목의 HTML div 블록을 생성합니다.

    Args:
        stocks (list): 종목 정보 리스트
        data_dir (str): 데이터 디렉토리 경로
        drive_mapping (dict): 구글 드라이브 FILE_ID 매핑 (선택)

    Returns:
        tuple: (html_string, processed_count, latest_data_date)
    """
    html_items = []
    processed = 0
    skipped = 0
    total = len(stocks)
    latest_data_date = ""

    print(f"\n[정보] {total:,}개 종목 처리 시작...")
    print("-" * 60)

    for idx, (ticker, company, last_update, status) in enumerate(stocks, 1):
        if idx % 500 == 0 or idx == total:
            print(f"  진행: {idx:,}/{total:,} ({idx*100//total}%) "
                  f"- 처리: {processed:,}, 건너뜀: {skipped:,}")

        csv_filename = f"fdr_KRX_p1d_{ticker}.csv"
        csv_path = os.path.join(data_dir, csv_filename)

        stats = get_csv_statistics(csv_path)

        if stats is None:
            skipped += 1
            continue

        if stats['last_date'] > latest_data_date:
            latest_data_date = stats['last_date']

        html_item = generate_data_item_html(ticker, company, stats, csv_filename, drive_mapping)
        html_items.append(html_item)
        processed += 1

    print("-" * 60)
    print(f"[완료] 처리: {processed:,}개, 건너뜀: {skipped:,}개")
    if latest_data_date:
        print(f"[정보] 최신 데이터 날짜: {latest_data_date}")

    return '\n\n'.join(html_items), processed, latest_data_date


# ─── HTML 파일 처리 ──────────────────────────────────────────────
def backup_html_file(html_path):
    """HTML 파일을 타임스탬프 포함하여 백업합니다."""
    if not os.path.exists(html_path):
        print("[경고] HTML 파일이 없습니다. 백업을 건너뜁니다.")
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"{html_path}.{timestamp}.bak"
    try:
        shutil.copy2(html_path, backup_path)
        print(f"[정보] 백업 파일 생성: {os.path.basename(backup_path)}")
    except Exception as e:
        print(f"[경고] 백업 실패: {e}")


def find_datagrid_range(html_content):
    """
    HTML에서 <div id="dataGrid" ...> 내부 콘텐츠 범위를 찾습니다.
    중첩된 div를 올바르게 카운팅하여 대응하는 닫는 태그를 찾습니다.

    Returns:
        tuple: (content_start, content_end)
               content_start: 여는 태그 '>' 다음 위치
               content_end: 대응하는 '</div>' 시작 위치
    """
    match = re.search(r'<div\s+id="dataGrid"[^>]*>', html_content)
    if not match:
        return None, None

    open_tag_end = match.end()

    # 중첩 div 카운팅
    depth = 1
    pos = open_tag_end

    while depth > 0 and pos < len(html_content):
        next_open = html_content.find('<div', pos)
        next_close = html_content.find('</div>', pos)

        if next_close == -1:
            break

        if next_open != -1 and next_open < next_close:
            depth += 1
            pos = next_open + 4
        else:
            depth -= 1
            if depth == 0:
                return open_tag_end, next_close
            pos = next_close + 6

    return None, None


def update_krx_daily_html(html_path, data_items_html, total_count, latest_data_date):
    """
    krx-daily.html 파일을 업데이트합니다.

    수정 대상:
      1. dataGrid 내부 콘텐츠 → 새 종목 목록으로 교체
      2. totalCount → 실제 종목 수로 업데이트
      3. stats 섹션 → 업데이트 날짜 추가/교체
      4. info-box 종목 수 → 실제 수로 업데이트
    """
    if not os.path.exists(html_path):
        print(f"[오류] HTML 파일을 찾을 수 없습니다: {html_path}")
        sys.exit(1)

    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except Exception as e:
        print(f"[오류] HTML 파일 읽기 실패: {e}")
        sys.exit(1)

    # ── 1) dataGrid 내부 콘텐츠 교체 ──
    content_start, content_end = find_datagrid_range(html_content)

    if content_start is None or content_end is None:
        print("[오류] dataGrid 섹션을 찾을 수 없습니다.")
        sys.exit(1)

    print(f"[정보] dataGrid 범위 탐지 완료 (위치: {content_start} ~ {content_end})")

    new_html = (
        html_content[:content_start] +
        '\n' +
        data_items_html + '\n' +
        '                '      # dataGrid 닫는 </div>와 같은 들여쓰기 (16칸)
        +
        html_content[content_end:]
    )

    # ── 2) totalCount 업데이트 ──
    new_html = re.sub(
        r'(<h3 id="totalCount">)[^<]*(</h3>)',
        rf'\g<1>{total_count:,}\g<2>',
        new_html
    )

    # ── 3) stats 섹션에 업데이트 날짜 추가/교체 ──
    update_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    update_info_html = (
        f'<p class="update-info" style="font-size: 0.85rem; color: #999; margin-top: 8px;">'
        f'최종 업데이트: {update_timestamp}'
    )
    if latest_data_date:
        update_info_html += f' | 최신 데이터: {latest_data_date}'
    update_info_html += '</p>'

    if 'class="update-info"' in new_html:
        # 기존 update-info 교체
        new_html = re.sub(
            r'<p class="update-info"[^>]*>.*?</p>',
            update_info_html,
            new_html
        )
    else:
        # stats 닫는 </div> 직전에 삽입
        stats_close_pattern = r'(                    <p>종목의 일간 OHLCV 데이터</p>\n)(                </div>)'
        stats_match = re.search(stats_close_pattern, new_html)
        if stats_match:
            insert_pos = stats_match.start(2)
            new_html = (
                new_html[:insert_pos] +
                f'                    {update_info_html}\n' +
                new_html[insert_pos:]
            )

    # ── 4) info-box 종목 수 업데이트 ──
    new_html = re.sub(
        r'(<li><strong>종목 수:</strong>)[^<]*(</li>)',
        rf'\g<1> {total_count:,}개 (KOSPI + KOSDAQ)\g<2>',
        new_html
    )

    # ── 파일 쓰기 ──
    try:
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(new_html)
        print(f"\n[성공] HTML 파일 업데이트 완료: {os.path.basename(html_path)}")
        print(f"[정보] 총 {total_count:,}개 종목 반영")
        print(f"[정보] 업데이트 시각: {update_timestamp}")
    except Exception as e:
        print(f"[오류] HTML 파일 쓰기 실패: {e}")
        sys.exit(1)


# ─── 메인 ────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  머신러너 - 일간 데이터 페이지 자동 업데이트 v2.0")
    print("=" * 60)
    print()

    # 마스터 파일 결정
    if len(sys.argv) >= 2:
        master_path = sys.argv[1]
        if not os.path.exists(master_path):
            print(f"[오류] 마스터 파일을 찾을 수 없습니다: {master_path}")
            sys.exit(1)
        print(f"[정보] 마스터 파일 (직접 지정): {master_path}")
    else:
        master_path = find_latest_master_file(DATA_DIR)
        if master_path is None:
            print(f"[오류] 데이터 폴더에서 마스터 파일을 찾을 수 없습니다.")
            print(f"       탐색 경로: {DATA_DIR}")
            print(f"       패턴: {MASTER_GLOB_PATTERN}")
            print(f"\n사용법: python update_daily_page.py [마스터_파일_경로]")
            sys.exit(1)

    print(f"[정보] 데이터 폴더: {DATA_DIR}")
    print(f"[정보] HTML 파일:  {HTML_FILE}")
    print()

    # 1. 마스터 파일 파싱
    stocks = parse_master_file(master_path)

    # 2. 구글 드라이브 매핑 로드
    print()
    drive_mapping = load_drive_mapping()
    if drive_mapping:
        print("[정보] 구글 드라이브 다운로드 링크를 사용합니다.")
    else:
        print("[정보] 로컬 다운로드 링크를 사용합니다.")
    print()

    # 3. HTML 백업
    backup_html_file(HTML_FILE)

    # 4. 모든 종목의 HTML 생성
    data_items_html, total_count, latest_data_date = generate_all_data_items(stocks, DATA_DIR, drive_mapping)

    if total_count == 0:
        print("[경고] 처리된 종목이 없습니다. HTML을 업데이트하지 않습니다.")
        sys.exit(1)

    # 4. HTML 파일 업데이트
    update_krx_daily_html(HTML_FILE, data_items_html, total_count, latest_data_date)

    print()
    print("=" * 60)
    print("  업데이트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[중단] 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[오류] 예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
