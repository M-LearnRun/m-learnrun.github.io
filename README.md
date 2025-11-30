# 머신러너 (Machine Learner) 웹사이트

## 소개
『AI 주식투자』 책의 공식 데이터 다운로드 및 브랜딩 웹사이트입니다.

## 파일 구조
```
home_page_ml/
├── index.html          # 메인 홈페이지
├── style.css           # 스타일시트
├── logo.png            # 머신러너 CI 로고
└── README.md           # 프로젝트 설명
```

## 웹사이트 실행 방법
1. `index.html` 파일을 웹 브라우저에서 열기
2. 또는 간단한 로컬 서버 실행:
   ```bash
   python -m http.server 8000
   ```
   브라우저에서 `http://localhost:8000` 접속

## 데이터 파일 추가 방법
1. `home_page_ml` 폴더에 `data` 폴더 생성
2. 데이터 파일들을 `data` 폴더에 추가
3. `index.html`의 다운로드 링크를 실제 파일 경로로 수정:
   ```html
   <a href="data/dataset.zip" class="download-btn" download>다운로드</a>
   ```

## 웹 호스팅 방법
### GitHub Pages (무료)
1. GitHub 저장소 생성
2. 파일들 업로드
3. Settings → Pages에서 활성화
4. `https://username.github.io/repository-name` 형태로 접속 가능

### Netlify (무료)
1. [Netlify](https://netlify.com)에 가입
2. 폴더를 드래그 앤 드롭으로 업로드
3. 자동으로 URL 생성

## 커스터마이징
- `style.css`: 색상, 폰트, 레이아웃 수정
- `index.html`: 콘텐츠 및 구조 수정
- 로고: `logo.png` 파일 교체

## 특징
- 반응형 디자인 (모바일, 태블릿, 데스크톱)
- 심플하고 전문적인 디자인
- 다운로드 섹션 포함
- 업데이트 로그 섹션
