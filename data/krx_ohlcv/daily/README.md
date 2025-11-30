# 주식 가격 데이터 (OHLCV)

## 데이터 형식

각 CSV 파일은 다음과 같은 컬럼으로 구성되어 있습니다:

- **Date**: 거래일 (YYYY-MM-DD 형식)
- **Open**: 시가
- **High**: 고가
- **Low**: 저가
- **Close**: 종가
- **Volume**: 거래량

## 파일 명명 규칙

파일명은 다음과 같은 형식을 따릅니다:
```
{종목코드}_{종목명영문}.csv
```

예시:
- `005930_samsung.csv` - 삼성전자
- `000660_skhynix.csv` - SK하이닉스
- `035420_naver.csv` - NAVER
- `035720_kakao.csv` - 카카오

## 데이터 추가 방법

1. **CSV 파일 준비**: 위의 형식에 맞춰 CSV 파일을 준비합니다
2. **파일 업로드**: 이 폴더에 CSV 파일들을 복사합니다
3. **HTML 업데이트** (선택사항): `price-data.html` 파일을 편집하여 새로운 종목을 목록에 추가합니다

## 샘플 데이터

`sample_template.csv` 파일을 참조하여 데이터 형식을 확인할 수 있습니다.

## Python으로 데이터 읽기

```python
import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('005930_samsung.csv')

# Date 컬럼을 datetime으로 변환
df['Date'] = pd.to_datetime(df['Date'])

# 데이터 확인
print(df.head())
print(df.info())
```

## 데이터 사용 예시

```python
import pandas as pd
import matplotlib.pyplot as plt

# 삼성전자 데이터 로드
df = pd.read_csv('005930_samsung.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)

# 종가 차트 그리기
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'])
plt.title('Samsung Electronics Stock Price')
plt.xlabel('Date')
plt.ylabel('Close Price (KRW)')
plt.grid(True)
plt.show()

# 월간 수익률 계산
df['Returns'] = df['Close'].pct_change()
print(df[['Close', 'Returns']].tail())
```

## 주의사항

- 모든 가격은 원화(KRW) 기준입니다
- 데이터는 월간(Monthly) 기준입니다
- 거래량은 주식 수량 기준입니다
- 데이터는 교육 및 연구 목적으로만 사용하시기 바랍니다
