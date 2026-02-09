#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
구글 드라이브 파일 ID 매핑 생성 스크립트

구글 드라이브 API를 사용하여 지정된 폴더의 모든 CSV 파일에 대한
ticker -> FILE_ID 매핑을 생성하고 JSON 파일로 저장합니다.

사용법:
    1. credentials.json 파일을 프로젝트 루트에 배치
    2. python generate_drive_mapping.py
    3. 생성된 file_id_mapping.json 파일 확인

필요한 패키지:
    pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
"""

import json
import os
import re
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ─── 설정 ───────────────────────────────────────────────────────
CREDENTIALS_FILE = 'credentials.json'
FOLDER_ID = '1F0gadKCMyw50TaSCAWcSTq_wcpigc-Ix'  # 구글 드라이브 폴더 ID
OUTPUT_FILE = 'file_id_mapping.json'
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']


# ─── 구글 드라이브 인증 ──────────────────────────────────────────
def authenticate():
    """
    Service Account를 사용하여 구글 드라이브 API에 인증합니다.

    Returns:
        Resource: 구글 드라이브 API 서비스 객체
    """
    try:
        credentials = service_account.Credentials.from_service_account_file(
            CREDENTIALS_FILE, scopes=SCOPES)
        service = build('drive', 'v3', credentials=credentials)
        print("[정보] 구글 드라이브 API 인증 성공")
        return service
    except FileNotFoundError:
        print(f"[오류] {CREDENTIALS_FILE} 파일을 찾을 수 없습니다.")
        print("       구글 클라우드 콘솔에서 서비스 계정을 생성하고")
        print("       credentials.json 파일을 프로젝트 루트에 저장하세요.")
        return None
    except Exception as e:
        print(f"[오류] 인증 실패: {e}")
        return None


# ─── 폴더 내 파일 목록 조회 ──────────────────────────────────────
def list_files_in_folder(service, folder_id):
    """
    지정된 구글 드라이브 폴더의 모든 파일을 조회합니다.

    Args:
        service: 구글 드라이브 API 서비스 객체
        folder_id (str): 폴더 ID

    Returns:
        list: 파일 정보 딕셔너리 리스트 (id, name)
    """
    try:
        files = []
        page_token = None

        while True:
            query = f"'{folder_id}' in parents and trashed=false"
            response = service.files().list(
                q=query,
                spaces='drive',
                fields='nextPageToken, files(id, name)',
                pageToken=page_token
            ).execute()

            files.extend(response.get('files', []))
            page_token = response.get('nextPageToken', None)

            if page_token is None:
                break

        print(f"[정보] 구글 드라이브에서 {len(files):,}개 파일을 찾았습니다.")
        return files

    except HttpError as e:
        print(f"[오류] 파일 목록 조회 실패: {e}")
        print("       폴더 ID가 올바른지, 서비스 계정에 폴더 접근 권한이 있는지 확인하세요.")
        return []
    except Exception as e:
        print(f"[오류] 예기치 않은 오류: {e}")
        return []


# ─── 티커 코드 추출 ──────────────────────────────────────────────
def extract_ticker(filename):
    """
    파일명에서 티커 코드를 추출합니다.

    예시:
        fdr_KRX_p1d_005930.csv -> 005930
        fdr_KRX_p1d_035720.csv -> 035720

    Args:
        filename (str): CSV 파일명

    Returns:
        str: 티커 코드 또는 None
    """
    # 패턴: fdr_KRX_p1d_{ticker}.csv
    match = re.search(r'fdr_KRX_p1d_(\d{6})\.csv', filename)
    if match:
        return match.group(1)
    return None


# ─── 매핑 생성 ───────────────────────────────────────────────────
def generate_mapping(service, folder_id):
    """
    폴더 내 모든 CSV 파일의 ticker -> FILE_ID 매핑을 생성합니다.

    Args:
        service: 구글 드라이브 API 서비스 객체
        folder_id (str): 폴더 ID

    Returns:
        dict: {ticker: file_id} 매핑
    """
    files = list_files_in_folder(service, folder_id)

    if not files:
        print("[경고] 폴더에 파일이 없습니다.")
        return {}

    mapping = {}
    skipped = []

    for file in files:
        file_id = file['id']
        filename = file['name']

        # CSV 파일만 처리
        if not filename.endswith('.csv'):
            continue

        ticker = extract_ticker(filename)
        if ticker:
            mapping[ticker] = file_id
        else:
            skipped.append(filename)

    print(f"[정보] {len(mapping):,}개 종목에 대한 FILE_ID 매핑 생성 완료")

    if skipped:
        print(f"[경고] {len(skipped)}개 파일은 티커 코드를 추출할 수 없어 건너뛰었습니다:")
        for name in skipped[:10]:  # 최대 10개만 표시
            print(f"       - {name}")
        if len(skipped) > 10:
            print(f"       ... 외 {len(skipped) - 10}개")

    return mapping


# ─── 매핑 저장 ───────────────────────────────────────────────────
def save_mapping(mapping, output_file):
    """
    매핑을 JSON 파일로 저장합니다.

    Args:
        mapping (dict): {ticker: file_id} 매핑
        output_file (str): 출력 파일 경로
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)

        print(f"[정보] 매핑 파일 저장 완료: {output_file}")
        print(f"       파일 크기: {os.path.getsize(output_file):,} bytes")
    except Exception as e:
        print(f"[오류] 매핑 파일 저장 실패: {e}")


# ─── 메인 함수 ───────────────────────────────────────────────────
def main():
    """메인 실행 함수"""
    print("============================================================")
    print("  구글 드라이브 파일 ID 매핑 생성기")
    print("============================================================")
    print()

    # 1. 인증
    service = authenticate()
    if not service:
        return

    print()

    # 2. 매핑 생성
    print(f"[정보] 폴더 ID: {FOLDER_ID}")
    print("[정보] 파일 목록 조회 중...")
    print()

    mapping = generate_mapping(service, FOLDER_ID)

    if not mapping:
        print("[오류] 매핑을 생성할 수 없습니다.")
        return

    print()

    # 3. 매핑 저장
    save_mapping(mapping, OUTPUT_FILE)

    print()

    # 4. 샘플 표시
    print("샘플 매핑 (처음 5개):")
    print("-----------------------------------------------------------")
    for i, (ticker, file_id) in enumerate(list(mapping.items())[:5]):
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        print(f"{ticker}: {file_id[:20]}...")
        print(f"       → {download_url}")
    print("-----------------------------------------------------------")

    print()
    print("============================================================")
    print("  완료! 이제 update_daily_data.bat를 실행하세요.")
    print("============================================================")


if __name__ == '__main__':
    main()
