# 구글 드라이브 연동 설정 가이드

이 가이드는 CSV 파일을 구글 드라이브에서 직접 다운로드할 수 있도록 설정하는 방법을 설명합니다.

## 1단계: 구글 클라우드 프로젝트 설정

### 1-1. 구글 클라우드 콘솔 접속
1. https://console.cloud.google.com/ 접속
2. 구글 계정으로 로그인

### 1-2. 프로젝트 생성
1. 상단의 프로젝트 선택 드롭다운 클릭
2. "새 프로젝트" 클릭
3. 프로젝트 이름 입력 (예: "machine-learner-drive")
4. "만들기" 클릭

### 1-3. Google Drive API 활성화
1. 왼쪽 메뉴에서 "API 및 서비스" → "라이브러리" 클릭
2. 검색창에 "Google Drive API" 입력
3. "Google Drive API" 선택
4. "사용" 버튼 클릭

## 2단계: 서비스 계정 생성

### 2-1. 서비스 계정 만들기
1. 왼쪽 메뉴에서 "API 및 서비스" → "사용자 인증 정보" 클릭
2. 상단의 "+ 사용자 인증 정보 만들기" 클릭
3. "서비스 계정" 선택
4. 서비스 계정 세부정보 입력:
   - 서비스 계정 이름: `drive-reader` (원하는 이름)
   - 서비스 계정 ID: 자동 생성됨
   - 설명: "Read files from Google Drive" (선택사항)
5. "만들기 및 계속하기" 클릭
6. 역할 선택: 건너뛰기 (선택사항 단계)
7. "완료" 클릭

### 2-2. 인증 키(JSON) 다운로드
1. 생성된 서비스 계정 목록에서 방금 만든 계정 클릭
2. 상단 탭에서 "키" 클릭
3. "키 추가" → "새 키 만들기" 클릭
4. 키 유형: **JSON** 선택
5. "만들기" 클릭
6. **자동으로 다운로드된 JSON 파일을 안전하게 보관**

### 2-3. 서비스 계정 이메일 복사
1. 서비스 계정 목록에서 이메일 주소 복사
   - 형식: `drive-reader@machine-learner-drive.iam.gserviceaccount.com`
   - 이 이메일은 다음 단계에서 사용됩니다

## 3단계: 구글 드라이브 폴더 공유

### 3-1. 구글 드라이브에서 폴더 공유
1. https://drive.google.com/ 접속
2. CSV 파일이 있는 폴더로 이동
   - 현재 폴더: https://drive.google.com/drive/folders/1F0gadKCMyw50TaSCAWcSTq_wcpigc-Ix
3. 폴더에서 마우스 오른쪽 클릭 → "공유" 선택
4. "사용자 및 그룹 추가" 입력란에 **서비스 계정 이메일 주소** 붙여넣기
5. 권한: **뷰어** (보기 전용) 선택
6. "전송" 클릭

**중요**: 서비스 계정에게 폴더 공유를 하지 않으면 파일 목록을 조회할 수 없습니다!

## 4단계: 프로젝트에 인증 파일 배치

### 4-1. JSON 파일 이름 변경 및 이동
1. 다운로드된 JSON 파일 (예: `machine-learner-drive-abc123.json`)을 찾기
2. 파일 이름을 **`credentials.json`**으로 변경
3. 이 파일을 프로젝트 루트 폴더에 복사:
   ```
   C:\Users\KBLEE\OneDrive\바탕 화면\home_page_ml\credentials.json
   ```

**보안 주의사항**:
- `credentials.json` 파일은 절대 GitHub에 업로드하지 마세요!
- `.gitignore` 파일에 이미 추가되어 있어 Git이 무시합니다
- 이 파일을 분실하면 다시 서비스 계정 키를 생성해야 합니다

## 5단계: Python 패키지 설치

### 5-1. 필요한 패키지 설치
명령 프롬프트(CMD)를 열고 다음 명령어 실행:

```bash
cd C:\Users\KBLEE\OneDrive\바탕 화면\home_page_ml
pip install -r requirements.txt
```

설치되는 패키지:
- `google-auth`: 구글 인증
- `google-auth-oauthlib`: OAuth 라이브러리
- `google-auth-httplib2`: HTTP 라이브러리
- `google-api-python-client`: 구글 드라이브 API 클라이언트

## 6단계: 파일 ID 매핑 생성

### 6-1. 매핑 스크립트 실행
명령 프롬프트에서 다음 실행:

```bash
python generate_drive_mapping.py
```

**예상 출력**:
```
============================================================
  구글 드라이브 파일 ID 매핑 생성기
============================================================

[정보] 구글 드라이브 API 인증 성공
[정보] 폴더 ID: 1F0gadKCMyw50TaSCAWcSTq_wcpigc-Ix
[정보] 파일 목록 조회 중...

[정보] 구글 드라이브에서 2,814개 파일을 찾았습니다.
[정보] 2,814개 종목에 대한 FILE_ID 매핑 생성 완료
[정보] 매핑 파일 저장 완료: file_id_mapping.json
       파일 크기: 234,567 bytes

샘플 매핑 (처음 5개):
-----------------------------------------------------------
005930: 1abc123xyz...
       → https://drive.google.com/uc?export=download&id=1abc123xyz...
005380: 1def456uvw...
       → https://drive.google.com/uc?export=download&id=1def456uvw...
...
-----------------------------------------------------------

============================================================
  완료! 이제 update_daily_data.bat를 실행하세요.
============================================================
```

### 6-2. 생성된 파일 확인
프로젝트 폴더에 `file_id_mapping.json` 파일이 생성되었는지 확인:
```
C:\Users\KBLEE\OneDrive\바탕 화면\home_page_ml\file_id_mapping.json
```

이 파일은 다음과 같은 형식입니다:
```json
{
  "005930": "1abc123xyz...",
  "005380": "1def456uvw...",
  ...
}
```

## 7단계: HTML 페이지 업데이트

### 7-1. 기존 BAT 파일 실행
명령 프롬프트에서:

```bash
update_daily_data.bat
```

또는 파일 탐색기에서 `update_daily_data.bat` 더블클릭

### 7-2. 출력 확인
이제 다음과 같은 메시지가 표시됩니다:

```
[정보] 구글 드라이브 매핑 로드: 2,814개 종목
[정보] 구글 드라이브 다운로드 링크를 사용합니다.
```

### 7-3. HTML 파일 확인
생성된 `krx-daily.html` 파일을 열어서 다운로드 링크가 다음 형식인지 확인:

```html
<a href="https://drive.google.com/uc?export=download&id=1abc123xyz..." class="download-link">다운로드</a>
```

## 8단계: 테스트

### 8-1. 브라우저에서 테스트
1. `krx-daily.html` 파일을 브라우저에서 열기
2. 아무 종목의 "다운로드" 버튼 클릭
3. **예상 동작**: 구글 드라이브 미리보기 없이 **바로 CSV 파일 다운로드**
4. 여러 종목으로 테스트해보기

### 8-2. 문제 해결
**다운로드가 안 되는 경우**:
- 구글 드라이브 폴더가 서비스 계정과 공유되었는지 확인
- `file_id_mapping.json` 파일이 존재하는지 확인
- 해당 티커의 FILE_ID가 매핑 파일에 있는지 확인

**미리보기가 뜨는 경우**:
- URL 형식이 `https://drive.google.com/uc?export=download&id=...` 인지 확인
- 다른 형식(`/file/d/.../view`)이면 설정이 잘못된 것

## 9단계: GitHub Pages에 배포

### 9-1. 파일 준비
다음 파일들만 GitHub에 업로드:
- `krx-daily.html` (구글 드라이브 링크가 포함된 업데이트된 HTML)
- `index.html`
- `kr-data.html`
- `about.html`
- 기타 HTML, CSS, JS 파일

**업로드하지 말 것**:
- `credentials.json` (보안 위험!)
- `file_id_mapping.json` (선택사항, 로컬에만 필요)
- `generate_drive_mapping.py` (선택사항)
- `update_daily_page.py` (선택사항)
- `data/` 폴더 (이제 구글 드라이브 사용)

### 9-2. Git 커밋 및 푸시
```bash
git add krx-daily.html index.html kr-data.html about.html
git commit -m "Update download links to use Google Drive"
git push origin main
```

### 9-3. GitHub Pages 확인
- GitHub Pages URL에서 웹사이트 접속
- 다운로드 버튼이 정상 작동하는지 확인

## 10단계: 유지보수

### 파일이 추가/변경된 경우
1. 구글 드라이브에 새 CSV 파일 업로드
2. `generate_drive_mapping.py` 다시 실행:
   ```bash
   python generate_drive_mapping.py
   ```
3. `update_daily_data.bat` 실행하여 HTML 업데이트
4. GitHub에 푸시

### 정기적인 업데이트 워크플로우
```
[로컬 PC]
1. 새 CSV 파일 생성 → 구글 드라이브에 업로드
2. python generate_drive_mapping.py
3. update_daily_data.bat
4. git add krx-daily.html
5. git commit -m "Update stock data"
6. git push

[GitHub Pages]
7. 자동 배포
8. 사용자가 웹사이트에서 구글 드라이브로부터 다운로드
```

## 요약

✅ **로컬에서 실행되는 것**:
- `generate_drive_mapping.py`: 구글 드라이브 API로 FILE_ID 매핑 생성
- `update_daily_page.py`: 매핑 읽고 HTML 생성
- Python 스크립트는 로컬에서만 실행

✅ **GitHub Pages에 배포되는 것**:
- 정적 HTML 파일 (`krx-daily.html` 등)
- 구글 드라이브 다운로드 URL이 포함된 HTML
- Python 코드는 배포되지 않음

✅ **사용자가 경험하는 것**:
- 웹사이트 방문 → "다운로드" 클릭
- 구글 드라이브에서 직접 CSV 파일 다운로드 (미리보기 없음)
- 빠른 다운로드 속도

## 문제 해결

### 오류: `[오류] credentials.json 파일을 찾을 수 없습니다.`
→ 3단계로 돌아가서 JSON 파일을 올바른 위치에 배치했는지 확인

### 오류: `[오류] 파일 목록 조회 실패: 403 Forbidden`
→ 구글 드라이브 폴더가 서비스 계정과 공유되지 않음. 2단계의 서비스 계정 이메일로 폴더 공유 필요

### 경고: `[경고] file_id_mapping.json 파일을 찾을 수 없습니다.`
→ 5단계의 `generate_drive_mapping.py`를 아직 실행하지 않음

### 다운로드 시 미리보기가 뜸
→ URL이 `https://drive.google.com/uc?export=download&id=...` 형식이 아닌 경우. HTML 파일에서 URL 형식 확인

## 추가 참고자료

- [Google Drive API 문서](https://developers.google.com/drive/api/guides/about-sdk)
- [서비스 계정 가이드](https://cloud.google.com/iam/docs/service-accounts)
- [GitHub Pages 문서](https://docs.github.com/pages)
