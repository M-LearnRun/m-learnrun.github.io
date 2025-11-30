"""
주식 가격 데이터 목록을 자동으로 생성하는 스크립트

매월 CSV 파일을 업데이트한 후 이 스크립트를 실행하면
price-data.html 파일이 자동으로 업데이트됩니다.

사용법:
    python generate_price_list.py
"""

import os
import pandas as pd
from datetime import datetime
from pathlib import Path

# 경로 설정
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "krx_ohlcv" / "daily"
OUTPUT_FILE = BASE_DIR / "krx-daily.html"

def get_csv_info(csv_path):
    """CSV 파일의 정보를 추출"""
    try:
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])

        start_date = df['Date'].min()
        end_date = df['Date'].max()
        num_months = len(df)

        return {
            'start': start_date.strftime('%Y.%m'),
            'end': end_date.strftime('%Y.%m'),
            'months': num_months
        }
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

def parse_filename(filename):
    """파일명에서 종목코드와 종목명을 추출"""
    # 파일명 형식: 005930_samsung.csv
    name = filename.replace('.csv', '')
    parts = name.split('_')

    if len(parts) >= 2:
        code = parts[0]
        name_en = '_'.join(parts[1:])
        return code, name_en
    return None, None

# 종목명 매핑 (종목코드 -> 한글명)
STOCK_NAMES = {
    '005930': '삼성전자',
    '000660': 'SK하이닉스',
    '035420': 'NAVER',
    '035720': '카카오',
    '005380': '현대차',
    '000270': '기아',
    '051910': 'LG화학',
    '006400': '삼성SDI',
    '207940': '삼성바이오로직스',
    '005490': 'POSCO홀딩스',
    # 여기에 나머지 종목들을 추가하세요
    # 또는 별도의 mapping.csv 파일을 만들어 사용할 수 있습니다
}

def get_stock_name(code):
    """종목코드로 한글명 조회"""
    return STOCK_NAMES.get(code, f'종목{code}')

def generate_html_items():
    """모든 CSV 파일을 읽어서 HTML 아이템 생성"""
    csv_files = sorted(DATA_DIR.glob('*.csv'))
    csv_files = [f for f in csv_files if f.name not in ['sample_template.csv', 'README.md']]

    html_items = []

    for csv_file in csv_files:
        code, name_en = parse_filename(csv_file.name)
        if not code:
            continue

        info = get_csv_info(csv_file)
        if not info:
            continue

        stock_name = get_stock_name(code)

        item_html = f'''                    <div class="data-item">
                        <h4>{code} - {stock_name}</h4>
                        <div class="data-info">
                            <div class="data-info-item">
                                <span>기간:</span>
                                <span>{info['start']} ~ {info['end']}</span>
                            </div>
                            <div class="data-info-item">
                                <span>데이터 수:</span>
                                <span>{info['months']}개월</span>
                            </div>
                        </div>
                        <a href="data/krx_ohlcv/daily/{csv_file.name}" class="download-link" download>다운로드</a>
                    </div>
'''
        html_items.append(item_html)

    return html_items, len(html_items)

def update_html():
    """HTML 파일 업데이트"""
    print("CSV 파일 스캔 중...")
    html_items, total_count = generate_html_items()

    if not html_items:
        print("CSV 파일을 찾을 수 없습니다!")
        return

    print(f"총 {total_count}개 종목 발견")

    # HTML 템플릿 읽기
    template_path = BASE_DIR / "price-data-template.html"

    if not template_path.exists():
        print("템플릿 파일이 없습니다. price-data.html을 템플릿으로 사용합니다.")
        template_path = OUTPUT_FILE

    with open(template_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # 데이터 항목 부분 교체
    items_html = '\n'.join(html_items)

    # {{DATA_ITEMS}} 마커를 찾아서 교체
    if '{{DATA_ITEMS}}' in html_content:
        html_content = html_content.replace('{{DATA_ITEMS}}', items_html)
    else:
        # 마커가 없으면 기존 방식으로 교체
        import re
        pattern = r'<div id="dataGrid" class="data-grid">.*?</div>\s*</div>\s*</section>'
        replacement = f'<div id="dataGrid" class="data-grid">\n{items_html}\n                </div>\n            </div>\n        </section>'
        html_content = re.sub(pattern, replacement, html_content, flags=re.DOTALL)

    # 총 개수 업데이트
    html_content = html_content.replace('id="totalCount">2,500+', f'id="totalCount">{total_count:,}')
    html_content = html_content.replace('약 2,500개', f'약 {total_count:,}개')

    # 파일 저장
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ {OUTPUT_FILE} 업데이트 완료!")
    print(f"   총 {total_count}개 종목 추가됨")

if __name__ == '__main__':
    update_html()
