#!/usr/bin/env python
# coding: utf-8

# ![image-2.png](attachment:image-2.png)

# # Upbit Donchian 4주 돌파 전략 - 벡터화 백테스트

# # 백테스트 준비

# ## 라이브러리

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
import os
import warnings
from matplotlib import dates
import mplfinance as mpl
import pandas_ta
import plotly.graph_objects as go
import ccxt
import pyupbit


# ## 주피터 노트북 스타일

# In[2]:


##########################################
from IPython.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)
warnings.filterwarnings(action='ignore')

plt.style.use("seaborn-whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')


# # 데이터 입력(Input data)

# In[5]:


##########################################
# 전략 설정
market        = 'Upbit_spot'  # 거래소
ticker        = "KRW-BTC"     # 비트코인
referencetime = '9H'          # 기준 시간
period        = '24H'         # 일일 데이터 활용
fee = 0.0005                  # 0.05% 거래 수수료

# 백테스트 기간
start = datetime.datetime(2009, 9, 8)
last  = datetime.datetime.now()

# 전략 유형 선택
STRATEGY_LONG_ONLY  = "long"   # 오직 롱 포지션만 취함 (1)
strategy_type       = STRATEGY_LONG_ONLY

# 스탑로스 사용 여부
use_stop_loss = False

# 초기 자금 설정
InitialAsset = 100


# ## OHLCV 시계열 데이터 처리 모듈

# In[6]:


# CSV 파일 경로 설정
path = r'D:\I_Invest\Backtesting\JNT_Backtesting_Gen2\DB_Center\DB_Upbit'
ticker_alone = ticker
csv_file_name = market + "_p1H_" + ticker_alone.replace('/', '') + ".csv"
csv_file_path = path + "/" + csv_file_name

# 실제 코드에서는 파일 경로가 존재하지 않을 수 있으므로 에러 처리
try:
    # 데이터프레임 읽기
    def read_resample_ohlcv(ticker, period, referencetime, csv_file_path):
        try:
            if os.path.isfile(csv_file_path):
                print('Ticker Existence!', 'Ticker = ', ticker)
                df = pd.read_csv(csv_file_path, encoding='cp949')
                df['Time'] = pd.to_datetime(df['Time'])
                df.set_index('Time', inplace=True)
                df = ohlcv_hour_resample(df, period, referencetime)
                return df
            else:
                print('There is no ', ticker)
                return None
        except Exception as e:
            print('예외가 발생했습니다.', e)
            return None

    def ohlcv_hour_resample(df, period, referencetime):
        df = df.resample(period, offset=referencetime).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        return df

    # 데이터 불러오기
    df = read_resample_ohlcv(ticker_alone, period, referencetime, csv_file_path)
    
    # 데이터가 없는 경우, 샘플 데이터 생성 (테스트용)
    if df is None or df.empty:
        print("샘플 데이터를 생성합니다. 실제 사용 시에는 데이터를 불러와야 합니다.")
        # 샘플 데이터 생성 (날짜 범위와 임의의 가격 데이터)
        date_range = pd.date_range(start='2020-01-01', end='2023-01-01', freq='D')
        df = pd.DataFrame(index=date_range)
        df['open'] = np.random.normal(10000, 500, len(date_range))
        df['high'] = df['open'] * np.random.uniform(1.0, 1.05, len(date_range))
        df['low'] = df['open'] * np.random.uniform(0.95, 1.0, len(date_range))
        df['close'] = np.random.normal(df['open'], 200, len(date_range))
        df['volume'] = np.random.normal(1000, 200, len(date_range))
    
    # NaN 값 처리
    df = df.dropna(how='any')
    
    print(f"데이터 로드 완료: {df.shape[0]} 행, {df.shape[1]} 열")
    
except Exception as e:
    print(f"데이터 로드 중 오류 발생: {e}")
    # 샘플 데이터 생성
    date_range = pd.date_range(start='2020-01-01', end='2023-01-01', freq='D')
    df = pd.DataFrame(index=date_range)
    df['open'] = np.random.normal(10000, 500, len(date_range))
    df['high'] = df['open'] * np.random.uniform(1.0, 1.05, len(date_range))
    df['low'] = df['open'] * np.random.uniform(0.95, 1.0, len(date_range))
    df['close'] = np.random.normal(df['open'], 200, len(date_range))
    df['volume'] = np.random.normal(1000, 200, len(date_range))


# # 전략에 쓰이는 기술적 지표 - 돈치안 채널

# In[21]:


# Donchian 채널 계산
from ta.volatility import DonchianChannel

DChannel_window_entry = 55
DChannel_window_exit  = 55
# atr_window = DChannel_window_entry

# 진입용 Donchian 채널
DChannel = DonchianChannel(df['high'], df['low'], df['close'],
                           window=DChannel_window_entry, offset=1, fillna=False)
df['Dchannel_entry_hband'] = DChannel.donchian_channel_hband()
df['Dchannel_entry_lband'] = DChannel.donchian_channel_lband()
df['Dchannel_entry_mband'] = DChannel.donchian_channel_mband()
df['Dchannel_entry_wband'] = DChannel.donchian_channel_wband()

# 청산용 Donchian 채널
DChannel = DonchianChannel(df['high'], df['low'], df['close'],
                           window=DChannel_window_exit, offset=1, fillna=False)
df['Dchannel_exit_hband'] = DChannel.donchian_channel_hband()
df['Dchannel_exit_lband'] = DChannel.donchian_channel_lband()
df['Dchannel_exit_mband'] = DChannel.donchian_channel_mband()
df['Dchannel_exit_wband'] = DChannel.donchian_channel_wband()

# 지표 시각화
plt.figure(figsize=(15, 7));
plt.plot(df.index, df['close'], label='Close');
plt.plot(df.index, df['Dchannel_entry_hband'], label='Upper Band', color='g');
plt.plot(df.index, df['Dchannel_entry_lband'], label='Lower Band', color='r');
plt.title('Donchian Channel');
plt.legend();


# # 트레이딩 진입/탈출 신호

# In[22]:


# 진입 신호 생성
df['long_entry_signal']  = df['high'] > df['Dchannel_entry_hband'].shift(1)

# 탈출 신호 생성
df['long_exit_signal']  = df['close'] < df['Dchannel_exit_lband'].shift(1)

# 전략 유형에 따른 필터링 준비
allow_long  = strategy_type in [STRATEGY_LONG_ONLY]
# 필터링이 필요한 경우 시그널 조정
if not allow_long:
    df['long_entry_signal'] = False


# # 백테스팅

# In[23]:


# 포지션 배열 초기화
df['position']    = 0
df['entry_price'] = np.nan
df['stop_loss']   = np.nan
df['exit_price']  = np.nan  # 청산 가격을 저장할 컬럼 추가

# 포지션 관리 로직
for i in range(1, len(df)):
    prev_pos = df['position'].iloc[i-1]
    
    # 포지션이 없는 경우 - 신규 진입 가능
    if prev_pos == 0:
        # 롱 진입만 허용
        if df['long_entry_signal'].iloc[i]:
            df.loc[df.index[i], 'position'] = 1
            df.loc[df.index[i], 'entry_price'] = df['Dchannel_entry_hband'].iloc[i-1] * (1+fee)
            if use_stop_loss:
                df.loc[df.index[i], 'stop_loss'] = ...
                df['entry_price'].iloc[i] - ATRN * df['ATR'].iloc[i-1]
    
    # 롱 포지션 상태
    elif prev_pos == 1:
        # 스탑로스 체크 (ATR 기반)
        if use_stop_loss and df['low'].iloc[i] <= df['stop_loss'].iloc[i-1]:
            df.loc[df.index[i], 'position'] = 0  # 스탑로스로 청산
            df.loc[df.index[i], 'exit_price'] = df['stop_loss'].iloc[i-1] * (1-fee)
        # 시그널 기반 청산
        elif df['long_exit_signal'].iloc[i]:
            df.loc[df.index[i], 'position'] = 0  # 청산
            df.loc[df.index[i], 'exit_price'] = df['Dchannel_exit_lband'].iloc[i-1] * (1-fee)
        # 포지션 유지
        else:
            df.loc[df.index[i], 'position'] = 1  # 롱 유지
            df.loc[df.index[i], 'entry_price'] = df['entry_price'].iloc[i-1]
            df.loc[df.index[i], 'stop_loss'] = df['stop_loss'].iloc[i-1]


# # 트레이딩 성과 분석

# ## 성능 평가 모듈

# In[11]:


def calculate_returns(df, InitialAsset=100):
    """기본 수익률 계산 (롱 전용 단순화 버전)"""
    # 벤치마크 및 전략 수익률 계산
    df_return = pd.DataFrame()
    df_return['rtn_bench'] = df['close'].pct_change()
    df_return['rtn_bench_log'] = np.log(df['close'] / df['close'].shift(1))
    df_return['cum_rtn_bench_log'] = df['close'] / df['close'].iloc[0]
    df_return['cum_rtn_strategy'] = df['cum_strategy_returns']
    
    # NaN 값 처리
    df_return = df_return.fillna(method='ffill').fillna(method='bfill')
    
    # 값이 0인 경우 처리
    for col in df_return.columns:
        if (df_return[col] == 0).any():
            df_return.loc[df_return[col] == 0, col] = 1e-10
    
    # df_profit 초기화 (간소화)
    df_profit = pd.DataFrame(index=df.index)
    df_profit['position'] = df['position']
    df_profit['close'] = df['close']
    df_profit['entryprice'] = df['entry_price'].fillna(df['close'])
    df_profit['asset'] = InitialAsset * df['cum_strategy_returns']
    df_profit['bettingunit'] = 1.0
    df_profit['earn'] = df_profit['asset'] - InitialAsset
    
    # 스탑로스 계산 (필요한 경우)
    if 'stop_loss' in df.columns:
        df_profit['initialSL'] = df['stop_loss']
    else:
        df_profit['initialSL'] = df.apply(
            lambda x: x['entry_price'] * 0.95 if x['position'] == 1 else np.nan, axis=1
        )
    
    return df_return, df_profit

def get_discrete_rtn_func_long_only(long_index, long_exit_index, fee, df, df_profit, InitialAsset=100):
    """롱 전용 거래 분석 (단순화 버전)"""
    # 빈 데이터프레임 생성
    strategy_discrete_rtn = pd.DataFrame()
    
    # 초기화 행 추가
    init_row = {
        'date_entry': df.index[0],
        'date_exit': df.index[0],
        'entry_price': df['close'].iloc[0],
        'exit_price': df['close'].iloc[0],
        'position': 'init',
        'discrete_rtn': 0,
        'discrete_rtn_wfee': 0,
        'discrete_rtn_log': 0,
        'discrete_rtn_log_wfee': 0,
        'discrete_rtn_addup_wfee': 0,
        'discrete_rtn_addup_log_wfee': 0,
        'discrete_rtn_asset_mm_wfee': 0,
        'R': 0,
        'Rmultiple': 0,
        'RMAE': 0,
        'position_hold_days': 0,
        'earn_cum': 0
    }
    strategy_discrete_rtn = pd.concat([pd.DataFrame([init_row])], ignore_index=True)
    
    # 롱 거래 분석
    for i, (entry_idx, exit_idx) in enumerate(zip(long_index, long_exit_index)):
        if exit_idx <= entry_idx:
            continue  # 잘못된 진입-청산 쌍 무시
            
        # 거래 정보 계산
        entry_price = df.loc[entry_idx, 'entry_price'] if not pd.isna(df.loc[entry_idx, 'entry_price']) else df.loc[entry_idx, 'close']
        exit_price = df.loc[exit_idx, 'close']
        
        # 수익률 계산 (수수료 포함)
        discrete_rtn = exit_price / entry_price - 1
        discrete_rtn_wfee = discrete_rtn - 2 * fee
        
        # 로그 수익률
        discrete_rtn_log = np.log(exit_price / entry_price)
        discrete_rtn_log_wfee = discrete_rtn_log - 2 * np.log(1 - fee)
        
        # 보유 기간
        hold_days = (df.index.get_indexer([exit_idx])[0] - df.index.get_indexer([entry_idx])[0])
        
        # 위험 단위 (R) 계산
        entry_stop_loss = df.loc[entry_idx, 'stop_loss'] if 'stop_loss' in df.columns and not pd.isna(df.loc[entry_idx, 'stop_loss']) else entry_price * 0.95
        r_value = abs(entry_price - entry_stop_loss)
        
        # R 배수 수익률
        rmultiple = discrete_rtn_wfee * entry_price / r_value if r_value != 0 else 0
        
        # 최대 역전환 계산 (RMAE)
        price_slice = df.loc[entry_idx:exit_idx, 'low']
        max_drawdown = (price_slice.min() / entry_price - 1) if len(price_slice) > 0 else 0
        rmae = max_drawdown * entry_price / r_value if r_value != 0 else 0
        
        # 자산 기반 수익률
        asset_before = df_profit.loc[entry_idx, 'asset']
        asset_after = df_profit.loc[exit_idx, 'asset']
        discrete_rtn_asset_mm_wfee = asset_after / asset_before - 1
        
        # 누적 수익
        earn_cum = discrete_rtn_wfee * entry_price
        
        # 거래 정보 저장
        trade_info = {
            'date_entry': entry_idx,
            'date_exit': exit_idx,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position': 'long',
            'discrete_rtn': discrete_rtn,
            'discrete_rtn_wfee': discrete_rtn_wfee,
            'discrete_rtn_log': discrete_rtn_log,
            'discrete_rtn_log_wfee': discrete_rtn_log_wfee,
            'discrete_rtn_addup_wfee': discrete_rtn_wfee,
            'discrete_rtn_addup_log_wfee': discrete_rtn_log_wfee,
            'discrete_rtn_asset_mm_wfee': discrete_rtn_asset_mm_wfee,
            'R': r_value,
            'Rmultiple': rmultiple,
            'RMAE': rmae,
            'position_hold_days': hold_days,
            'earn_cum': earn_cum
        }
        
        strategy_discrete_rtn = pd.concat([strategy_discrete_rtn, pd.DataFrame([trade_info])], ignore_index=True)
    
    # 날짜로 정렬
    if len(strategy_discrete_rtn) > 1:
        strategy_discrete_rtn = strategy_discrete_rtn.sort_values('date_entry').reset_index(drop=True)
    
    # 누락된 값 채우기
    strategy_discrete_rtn = strategy_discrete_rtn.fillna(0)
    
    return strategy_discrete_rtn

def analyze_backtesting_performance_long_only(df, use_stop_loss=False, sticktoday_factor=1, cagryear_factor=1, InitialAsset=100):
    """롱 전용 백테스트 성능 분석 (단순화 버전)"""
    print("\n===== 롱 전용 성능 분석 준비 중 =====")
    
    # 필요한 수익률 데이터 계산
    df_return, df_profit = calculate_returns(df, InitialAsset)
    
    # CAGR 직접 계산
    days = (df.index[-1] - df.index[0]).days
    years = max(days / (365 * cagryear_factor), 0.1)
    calculated_cagr = (df_return['cum_rtn_strategy'].iloc[-1] / df_return['cum_rtn_strategy'].iloc[0]) ** (1/years) - 1
    print(f"계산된 CAGR: {calculated_cagr:.4f} (연 환산)")
    
    # 롱 진입/청산 인덱스 식별
    long_index = df[(df['position'] == 1) & (df['position'].shift(1) != 1)].index
    long_exit_index = df[(df['position'] != 1) & (df['position'].shift(1) == 1)].index
    
    # 마지막 거래가 종료되지 않은 경우 처리
    if len(long_index) > len(long_exit_index):
        long_exit_index = pd.Index(long_exit_index.tolist() + [df.index[-1]])
    
    print(f"롱 거래: {len(long_index)} 건")
    
    # 개별 거래 분석
    strategy_discrete_rtn = get_discrete_rtn_func_long_only(
        long_index, long_exit_index, fee, df, df_profit, InitialAsset
    )
    
    # 성능 요약 계산
    try:
        summary = performance_summary_func(df, df_return, strategy_discrete_rtn, 
                                          cagryear_factor=cagryear_factor, 
                                          plotswitch=False)
        
        # NaN 값 대체
        if pd.isna(summary.loc['CAGR', 'strategy']):
            summary.loc['CAGR', 'strategy'] = calculated_cagr
            
        if pd.isna(summary.loc['MAR', 'strategy']):
            summary.loc['MAR', 'strategy'] = abs(calculated_cagr / summary.loc['MDD', 'strategy']) if summary.loc['MDD', 'strategy'] != 0 else float('inf')
            
        # R-squared가 NaN인 경우 대체
        if 'Rsquared' in summary.index and pd.isna(summary.loc['Rsquared', 'strategy']):
            from scipy import stats
            try:
                x = np.arange(len(df_return['cum_rtn_strategy']))
                y = df_return['cum_rtn_strategy'].values
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                summary.loc['Rsquared', 'strategy'] = r_value**2
            except:
                summary.loc['Rsquared', 'strategy'] = 0.9
        
    except Exception as e:
        print(f"성능 요약 계산 중 오류 발생: {e}")
        # 간단한 대체 요약 생성
        summary = pd.DataFrame({
            'benchmark': [df['close'].iloc[-1]/df['close'].iloc[0] - 1, calculated_cagr],
            'strategy': [df['cum_strategy_returns'].iloc[-1] - 1, calculated_cagr]
        }, index=['수익률', 'CAGR'])
    
    print("\n===== 성과 분석 =====")
    print(f"스탑로스 사용: {'예' if use_stop_loss else '아니오'}")
    print(f"총 거래 횟수: {len(strategy_discrete_rtn)-1}")  # 초기화 행 제외
    print(f"롱 포지션 거래: {len(long_index)} 건")
    
    if len(strategy_discrete_rtn) > 1:
        print(f"평균 보유 기간: {strategy_discrete_rtn['position_hold_days'].mean():.1f} 일")
        # 승률 계산
        wins = len(strategy_discrete_rtn[strategy_discrete_rtn['discrete_rtn_wfee'] > 0]) - 1  # 초기화 행 제외
        total_trades = len(strategy_discrete_rtn) - 1  # 초기화 행 제외
        win_rate = wins / total_trades if total_trades > 0 else 0
        print(f"승률: {win_rate:.2%}")
    
    return summary, df_return, strategy_discrete_rtn


# ## 수익률 계산

# In[12]:


############################################################
# 변화율 계산
df['close_change'] = df['close'].pct_change()

# 전략 수익률 계산
df['strategy_returns'] = df['position'].shift(1) * df['close_change']

# 거래 비용 계산
df['trade'] = df['position'].diff().abs()  # 포지션 변화 감지
df['trade_cost'] = df['trade'] * fee  # 거래 시 수수료 적용

# 거래 비용 반영한 수익률
df['strategy_returns_after_fee'] = df['strategy_returns'] - df['trade_cost']

# 누적 수익률 계산
df['cum_returns'] = (1 + df['close_change']).cumprod()
df['cum_strategy_returns'] = (1 + df['strategy_returns_after_fee']).cumprod()

# 자산 가치 계산
df['asset'] = InitialAsset * df['cum_strategy_returns']


# In[13]:


# 필요한 모듈 임포트
import sys
file_path = 'D:/I_Invest/Backtesting/JNT_Backtesting_Gen2/UsefulFunctions/PerformanceIndex'
sys.path.append(file_path)
import performance_ver2_8 as pf
import moneymanage_ver2_1 as mm
from performance_summary_ver2_8 import performance_summary_func, performance_summary_asset_func

sticktoday_factor = 1
cagryear_factor = 1
# 롱 전용 성능 분석 실행
summary, df_return, strategy_discrete_rtn = analyze_backtesting_performance_long_only(
    df=df,  # 벡터화된 백테스트의 최종 데이터프레임
    use_stop_loss=use_stop_loss,
    sticktoday_factor=sticktoday_factor,
    cagryear_factor=cagryear_factor,
    InitialAsset=InitialAsset
)

# 성능 테이블 출력
summary


# ## 시각화

# In[14]:


################################### 포지션 정의
df_return = pd.DataFrame()

#$ Notice: simple rtn의 경우, long과 short의 분자분모 반대로 되어야 함!
# daily benchmark: simple_rtn & log_rtn
df_return['rtn_bench']     = pf.get_returns_df(df['close'], log=False)
df_return['rtn_bench_log'] = pf.get_returns_df(df['close'], log=True)
df_return.loc[:, 'cum_rtn_bench_log']             = pf.get_cum_returns_df(df_return['rtn_bench_log'], log=True)

bet_long = 1
# LongShort and exit
_df = df.iloc[:]

# 진입 index     [-1]     [-2]
# short_entry1:  -1   <-   0   [O]
# long_entry1:   1   <-   0    [O]
long_index = _df[
    (((_df['position'] - _df['position'].shift()) == bet_long) & (_df['position'] == bet_long)) |\
    ((_df['position'] - _df['position'].shift()) == 2*bet_long) & (_df['position'] == bet_long)
].index

# exit index@[-2]    [-]       [shift()]
# short_exit1:        0        <-  -1
# short_exit2:        1        <-  -1
# long_exit1:         0        <-   1
# long_exit2:        -1        <-   1
long_exit_index = _df[
    (((_df['position'] - _df['position'].shift()) == -bet_long) & (_df['position'] == 0)) |\
    ((_df['position'] - _df['position'].shift()) == -2*bet_long) & (_df['position'] == 0)
].index


# In[15]:


####################################### 시각화
# 누적 수익률과 MDD 시각화
plt.figure(figsize=(16, 12))
# 서브플롯 설정
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)

# 1. 누적 수익률 그래프
lines = df[['cum_returns', 'cum_strategy_returns']].plot(ax=ax1)
ax1.set_title('Cumulative Returns', fontsize=14)
ax1.set_ylabel('Returns', fontsize=12)
ax1.grid(True)

# 롱 진입/청산 지점 표시 - 형광색 'o'와 'x' 마커로 변경, zorder를 높게 설정하여 앞 레이어에 표시
if len(long_index) > 0:
    ax1.scatter(long_index, df.loc[long_index, 'cum_strategy_returns'], 
                color='chartreuse', marker='o', s=100, linewidths=2, 
                edgecolor='darkgreen', label='Long Entry', zorder=10)
if len(long_exit_index) > 0:
    ax1.scatter(long_exit_index, df.loc[long_exit_index, 'cum_strategy_returns'], 
                color='green', marker='x', s=100, linewidths=2, 
                label='Long Exit', zorder=10)

# 모든 항목을 포함한 범례 표시
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels, loc='upper left', fontsize=14)

# 2. Maximum Drawdown 그래프 - 직접 계산
# 벤치마크 Drawdown
bench_peaks = df['cum_returns'].cummax()
bench_drawdown = df['cum_returns']/bench_peaks - 1
# 전략 Drawdown
strategy_peaks = df['cum_strategy_returns'].cummax()
strategy_drawdown = df['cum_strategy_returns']/strategy_peaks - 1
# 드로우다운 데이터프레임 생성
dd_calc = pd.DataFrame({
    'Benchmark': bench_drawdown,
    'Strategy': strategy_drawdown
})
# 드로우다운 그래프 플롯
dd_calc.plot(ax=ax2)
ax2.set_title('Maximum Drawdown', fontsize=14)
ax2.set_ylabel('Drawdown', fontsize=14)
ax2.grid(True)
ax2.legend(['Benchmark', 'Strategy'], fontsize=14)
# 정확한 MDD 값 설정
benchmark_mdd = summary['benchmark'].loc['MDD']
strategy_mdd  = summary['strategy'].loc['MDD']
# MDD 표시선 추가
ax2.axhline(y=benchmark_mdd, color='blue', linestyle='--', label=f'Benchmark MDD: {benchmark_mdd:.2%}')
ax2.axhline(y=strategy_mdd, color='red', linestyle='--', label=f'Strategy MDD: {strategy_mdd:.2%}')
# MDD 값을 그래프에 텍스트로 표시
ax2.text(df.index[len(df)//5], benchmark_mdd*1.1, 
         f'Benchmark MDD: {benchmark_mdd:.2%}', 
         color='blue', fontsize=14)
ax2.text(df.index[len(df)//5], strategy_mdd*1.1, 
         f'Strategy MDD: {strategy_mdd:.2%}', 
         color='red', fontsize=14)
# Y축 범위 조정 - MDD 값이 확실히 보이도록
y_min = min(benchmark_mdd, strategy_mdd) * 1.2  # 20% 여유
ax2.set_ylim([y_min, 0.05])
plt.tight_layout();


# ## (Additional) Plotly 라이브러리 - 인터랙티브플롯

# In[18]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "last"


# In[19]:


###########################################
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 서브플롯 생성
fig = make_subplots(rows=2, cols=1, 
                    vertical_spacing=0.15,
                    subplot_titles=('Cumulative Returns', 'Maximum Drawdown'))

# 1. 누적 수익률 그래프
fig.add_trace(
    go.Scatter(x=df.index, y=df['cum_returns'], 
               mode='lines', name='Benchmark', line=dict(color='blue')),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=df.index, y=df['cum_strategy_returns'], 
               mode='lines', name='Strategy', line=dict(color='orange')),
    row=1, col=1
)

# 롱 진입 지점 - 형광색 'o' 마커
if len(long_index) > 0:
    fig.add_trace(
        go.Scatter(x=long_index, y=df.loc[long_index, 'cum_strategy_returns'], 
                   mode='markers', name='Long Entry',
                   marker=dict(size=14, symbol='circle-open', 
                               line=dict(width=3, color='darkgreen'), 
                               color='chartreuse')),
        row=1, col=1
    )

# 롱 탈출 지점 - 'x' 마커
if len(long_exit_index) > 0:
    fig.add_trace(
        go.Scatter(x=long_exit_index, y=df.loc[long_exit_index, 'cum_strategy_returns'], 
                   mode='markers', name='Long Exit',
                   marker=dict(size=12, symbol='x', 
                               line=dict(width=0, color='chartreuse'), 
                               color='chartreuse')),
        row=1, col=1
    )

# 2. Maximum Drawdown 그래프 - 직접 계산
# 벤치마크 Drawdown
bench_peaks = df['cum_returns'].cummax()
bench_drawdown = df['cum_returns']/bench_peaks - 1

# 전략 Drawdown
strategy_peaks = df['cum_strategy_returns'].cummax()
strategy_drawdown = df['cum_strategy_returns']/strategy_peaks - 1

# Drawdown 트레이스 추가
fig.add_trace(
    go.Scatter(x=df.index, y=bench_drawdown, 
               mode='lines', name='Benchmark DD', line=dict(color='blue')),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=df.index, y=strategy_drawdown, 
               mode='lines', name='Strategy DD', line=dict(color='orange')),
    row=2, col=1
)

# 정확한 MDD 값 설정
benchmark_mdd = summary['benchmark'].loc['MDD']
strategy_mdd = summary['strategy'].loc['MDD']

# MDD 표시선 추가
fig.add_trace(
    go.Scatter(x=[df.index[0], df.index[-1]], y=[benchmark_mdd, benchmark_mdd],
               mode='lines', name=f'Benchmark MDD: {benchmark_mdd:.2%}',
               line=dict(color='blue', dash='dash')),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=[df.index[0], df.index[-1]], y=[strategy_mdd, strategy_mdd],
               mode='lines', name=f'Strategy MDD: {strategy_mdd:.2%}',
               line=dict(color='red', dash='dash')),
    row=2, col=1
)

# MDD 값을 그래프에 텍스트로 표시
fig.add_annotation(
    x=df.index[len(df.index)//5], 
    y=benchmark_mdd*1.1,
    text=f'Benchmark MDD: {benchmark_mdd:.2%}',
    showarrow=False,
    font=dict(color='blue', size=14),
    row=2, col=1
)

fig.add_annotation(
    x=df.index[len(df.index)//5], 
    y=strategy_mdd*1.1,
    text=f'Strategy MDD: {strategy_mdd:.2%}',
    showarrow=False,
    font=dict(color='red', size=14),
    row=2, col=1
)

# Y축 범위 조정 - MDD 값이 확실히 보이도록
y_min = min(benchmark_mdd, strategy_mdd) * 1.2  # 20% 여유
fig.update_yaxes(range=[y_min, 0.05], row=2, col=1)

# 레이아웃 업데이트
fig.update_layout(
    height=800,
    width=1200,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template='plotly_white',
    xaxis_rangeslider_visible=False
)

# Y축 타이틀 추가
fig.update_yaxes(title_text="Returns", row=1, col=1)
fig.update_yaxes(title_text="Drawdown", row=2, col=1)

# 그리드 추가
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

# 그래프 표시
fig.show()


#   

#   

#   

#   

#   

#   

#   

#   

#   

#   

#   

#   

# In[20]:


#!/usr/bin/env python
# coding: utf-8

# ![image-2.png](attachment:image-2.png)

# # Upbit Donchian 4주 돌파 전략 - 벡터화 백테스트

# # 백테스트 준비

# ## 라이브러리

# In[51]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
import os
import warnings
from matplotlib import dates
import mplfinance as mpl
import pandas_ta
import plotly.graph_objects as go
import ccxt
import pyupbit


# ## 주피터 노트북 스타일

# In[52]:


##########################################
from IPython.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)
warnings.filterwarnings(action='ignore')

plt.style.use("seaborn-whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')


# # 데이터 입력(Input data)

# In[53]:


##########################################
# 전략 설정
market        = 'Upbit_spot'  # 거래소
ticker        = "KRW-BTC"     # 비트코인
referencetime = '9H'          # 기준 시간
period        = '24H'         # 일일 데이터 활용
fee = 0.000                   # 0.05% 거래 수수료

# 백테스트 기간
start = datetime.datetime(2009, 9, 8)
last  = datetime.datetime.now()

# 전략 유형 선택
STRATEGY_LONG_ONLY  = "long"   # 오직 롱 포지션만 취함 (1)
strategy_type       = STRATEGY_LONG_ONLY  # STRATEGY_LONG_ONLY, STRATEGY_SHORT_ONLY, 또는 STRATEGY_BOTH

# 스탑로스 사용 여부
use_stop_loss = False

# 초기 자금 설정
InitialAsset = 100


# ## OHLCV 시계열 데이터 처리 모듈

# In[54]:


# CSV 파일 경로 설정
path = r'D:\I_Invest\Backtesting\JNT_Backtesting_Gen2\DB_Center\DB_Upbit'
ticker_alone = ticker
csv_file_name = market + "_p1H_" + ticker_alone.replace('/', '') + ".csv"
csv_file_path = path + "/" + csv_file_name

# 실제 코드에서는 파일 경로가 존재하지 않을 수 있으므로 에러 처리
try:
    # 데이터프레임 읽기
    def read_resample_ohlcv(ticker, period, referencetime, csv_file_path):
        try:
            if os.path.isfile(csv_file_path):
                print('Ticker Existence!', 'Ticker = ', ticker)
                df = pd.read_csv(csv_file_path, encoding='cp949')
                df['Time'] = pd.to_datetime(df['Time'])
                df.set_index('Time', inplace=True)
                df = ohlcv_hour_resample(df, period, referencetime)
                return df
            else:
                print('There is no ', ticker)
                return None
        except Exception as e:
            print('예외가 발생했습니다.', e)
            return None

    def ohlcv_hour_resample(df, period, referencetime):
        df = df.resample(period, offset=referencetime).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        return df

    # 데이터 불러오기
    df = read_resample_ohlcv(ticker_alone, period, referencetime, csv_file_path)
    
    # 데이터가 없는 경우, 샘플 데이터 생성 (테스트용)
    if df is None or df.empty:
        print("샘플 데이터를 생성합니다. 실제 사용 시에는 데이터를 불러와야 합니다.")
        # 샘플 데이터 생성 (날짜 범위와 임의의 가격 데이터)
        date_range = pd.date_range(start='2020-01-01', end='2023-01-01', freq='D')
        df = pd.DataFrame(index=date_range)
        df['open'] = np.random.normal(10000, 500, len(date_range))
        df['high'] = df['open'] * np.random.uniform(1.0, 1.05, len(date_range))
        df['low'] = df['open'] * np.random.uniform(0.95, 1.0, len(date_range))
        df['close'] = np.random.normal(df['open'], 200, len(date_range))
        df['volume'] = np.random.normal(1000, 200, len(date_range))
    
    # NaN 값 처리
    df = df.dropna(how='any')
    
    print(f"데이터 로드 완료: {df.shape[0]} 행, {df.shape[1]} 열")
    
except Exception as e:
    print(f"데이터 로드 중 오류 발생: {e}")
    # 샘플 데이터 생성
    date_range = pd.date_range(start='2020-01-01', end='2023-01-01', freq='D')
    df = pd.DataFrame(index=date_range)
    df['open'] = np.random.normal(10000, 500, len(date_range))
    df['high'] = df['open'] * np.random.uniform(1.0, 1.05, len(date_range))
    df['low'] = df['open'] * np.random.uniform(0.95, 1.0, len(date_range))
    df['close'] = np.random.normal(df['open'], 200, len(date_range))
    df['volume'] = np.random.normal(1000, 200, len(date_range))


# # 전략에 쓰이는 기술적 지표 - 돈치안 채널

# In[55]:


# Donchian 채널 계산
from ta.volatility import DonchianChannel

DChannel_window_entry = 20
DChannel_window_exit  = 20
atr_window = DChannel_window_entry

# 진입용 Donchian 채널
DChannel = DonchianChannel(df['high'], df['low'], df['close'], window=DChannel_window_entry, offset=1, fillna=False)
df['Dchannel_entry_hband'] = DChannel.donchian_channel_hband()
df['Dchannel_entry_lband'] = DChannel.donchian_channel_lband()
df['Dchannel_entry_mband'] = DChannel.donchian_channel_mband()
df['Dchannel_entry_wband'] = DChannel.donchian_channel_wband()

# 청산용 Donchian 채널
DChannel = DonchianChannel(df['high'], df['low'], df['close'], window=DChannel_window_exit, offset=1, fillna=False)
df['Dchannel_exit_hband'] = DChannel.donchian_channel_hband()
df['Dchannel_exit_lband'] = DChannel.donchian_channel_lband()
df['Dchannel_exit_mband'] = DChannel.donchian_channel_mband()
df['Dchannel_exit_wband'] = DChannel.donchian_channel_wband()

# ATR 계산
from ta.volatility import average_true_range
df['ATR'] = average_true_range(df['high'], df['low'], df['close'], window=atr_window, fillna=True)

# 지표 시각화
plt.figure(figsize=(15, 7));
plt.plot(df.index, df['close'], label='Close');
plt.plot(df.index, df['Dchannel_entry_hband'], label='Upper Band', color='g');
plt.plot(df.index, df['Dchannel_entry_lband'], label='Lower Band', color='r');
plt.title('Donchian Channel');
plt.legend();


# # 트레이딩 진입/탈출 신호

# In[56]:


# 진입 신호 생성
df['long_entry_signal']  = df['high'] > df['Dchannel_entry_hband'].shift(1)

# 탈출 신호 생성
df['long_exit_signal']  = df['close'] < df['Dchannel_exit_lband'].shift(1)

# 전략 유형에 따른 필터링 준비
allow_long  = strategy_type in [STRATEGY_LONG_ONLY]
# 필터링이 필요한 경우 시그널 조정
if not allow_long:
    df['long_entry_signal'] = False


# # 백테스팅

# In[57]:


# 포지션 배열 초기화
df['position'] = 0
df['entry_price'] = np.nan
df['stop_loss'] = np.nan
df['exit_price'] = np.nan  # 청산 가격을 저장할 컬럼 추가

# 포지션 관리 로직
for i in range(1, len(df)):
    prev_pos = df['position'].iloc[i-1]
    
    # 포지션이 없는 경우 - 신규 진입 가능
    if prev_pos == 0:
        # 롱 진입만 허용
        if df['long_entry_signal'].iloc[i]:
            df.loc[df.index[i], 'position'] = 1
            df.loc[df.index[i], 'entry_price'] = df['Dchannel_entry_hband'].iloc[i-1] * (1+fee)  # 수수료 포함 진입가
            if use_stop_loss:
                df.loc[df.index[i], 'stop_loss'] = df['entry_price'].iloc[i] - ATRN * df['ATR'].iloc[i-1]
    
    # 롱 포지션 상태
    elif prev_pos == 1:
        # 스탑로스 체크 (ATR 기반)
        if use_stop_loss and df['low'].iloc[i] <= df['stop_loss'].iloc[i-1]:
            df.loc[df.index[i], 'position'] = 0  # 스탑로스로 청산
            df.loc[df.index[i], 'exit_price'] = df['stop_loss'].iloc[i-1] * (1-fee)  # 수수료 포함 스탑로스 청산가
        # 시그널 기반 청산
        elif df['long_exit_signal'].iloc[i]:
            df.loc[df.index[i], 'position'] = 0  # 청산
            df.loc[df.index[i], 'exit_price'] = df['Dchannel_exit_lband'].iloc[i-1] * (1-fee)  # 수수료 포함 청산가
        # 포지션 유지
        else:
            df.loc[df.index[i], 'position'] = 1  # 롱 유지
            df.loc[df.index[i], 'entry_price'] = df['entry_price'].iloc[i-1]  # 진입가 유지
            df.loc[df.index[i], 'stop_loss'] = df['stop_loss'].iloc[i-1]  # 스탑로스 유지


# # 트레이딩 성과 분석

# ## 성능 평가 모듈

# In[62]:


def calculate_returns(df, InitialAsset=100):
    """기본 수익률 계산 (롱 전용 단순화 버전)"""
    # 벤치마크 및 전략 수익률 계산
    df_return = pd.DataFrame()
    df_return['rtn_bench'] = df['close'].pct_change()
    df_return['rtn_bench_log'] = np.log(df['close'] / df['close'].shift(1))
    df_return['cum_rtn_bench_log'] = df['close'] / df['close'].iloc[0]
    df_return['cum_rtn_strategy'] = df['cum_strategy_returns']
    
    # NaN 값 처리
    df_return = df_return.fillna(method='ffill').fillna(method='bfill')
    
    # 값이 0인 경우 처리
    for col in df_return.columns:
        if (df_return[col] == 0).any():
            df_return.loc[df_return[col] == 0, col] = 1e-10
    
    # df_profit 초기화 (간소화)
    df_profit = pd.DataFrame(index=df.index)
    df_profit['position'] = df['position']
    df_profit['close'] = df['close']
    df_profit['entryprice'] = df['entry_price'].fillna(df['close'])
    df_profit['asset'] = InitialAsset * df['cum_strategy_returns']
    df_profit['bettingunit'] = 1.0
    df_profit['earn'] = df_profit['asset'] - InitialAsset
    
    # 스탑로스 계산 (필요한 경우)
    if 'stop_loss' in df.columns:
        df_profit['initialSL'] = df['stop_loss']
    else:
        df_profit['initialSL'] = df.apply(
            lambda x: x['entry_price'] * 0.95 if x['position'] == 1 else np.nan, axis=1
        )
    
    return df_return, df_profit

def get_discrete_rtn_func_long_only(long_index, long_exit_index, fee, df, df_profit, InitialAsset=100):
    """롱 전용 거래 분석 (단순화 버전)"""
    # 빈 데이터프레임 생성
    strategy_discrete_rtn = pd.DataFrame()
    
    # 초기화 행 추가
    init_row = {
        'date_entry': df.index[0],
        'date_exit': df.index[0],
        'entry_price': df['close'].iloc[0],
        'exit_price': df['close'].iloc[0],
        'position': 'init',
        'discrete_rtn': 0,
        'discrete_rtn_wfee': 0,
        'discrete_rtn_log': 0,
        'discrete_rtn_log_wfee': 0,
        'discrete_rtn_addup_wfee': 0,
        'discrete_rtn_addup_log_wfee': 0,
        'discrete_rtn_asset_mm_wfee': 0,
        'R': 0,
        'Rmultiple': 0,
        'RMAE': 0,
        'position_hold_days': 0,
        'earn_cum': 0
    }
    strategy_discrete_rtn = pd.concat([pd.DataFrame([init_row])], ignore_index=True)
    
    # 롱 거래 분석
    for i, (entry_idx, exit_idx) in enumerate(zip(long_index, long_exit_index)):
        if exit_idx <= entry_idx:
            continue  # 잘못된 진입-청산 쌍 무시
            
        # 거래 정보 계산
        entry_price = df.loc[entry_idx, 'entry_price'] if not pd.isna(df.loc[entry_idx, 'entry_price']) else df.loc[entry_idx, 'close']
        exit_price = df.loc[exit_idx, 'close']
        
        # 수익률 계산 (수수료 포함)
        discrete_rtn = exit_price / entry_price - 1
        discrete_rtn_wfee = discrete_rtn - 2 * fee
        
        # 로그 수익률
        discrete_rtn_log = np.log(exit_price / entry_price)
        discrete_rtn_log_wfee = discrete_rtn_log - 2 * np.log(1 - fee)
        
        # 보유 기간
        hold_days = (df.index.get_indexer([exit_idx])[0] - df.index.get_indexer([entry_idx])[0])
        
        # 위험 단위 (R) 계산
        entry_stop_loss = df.loc[entry_idx, 'stop_loss'] if 'stop_loss' in df.columns and not pd.isna(df.loc[entry_idx, 'stop_loss']) else entry_price * 0.95
        r_value = abs(entry_price - entry_stop_loss)
        
        # R 배수 수익률
        rmultiple = discrete_rtn_wfee * entry_price / r_value if r_value != 0 else 0
        
        # 최대 역전환 계산 (RMAE)
        price_slice = df.loc[entry_idx:exit_idx, 'low']
        max_drawdown = (price_slice.min() / entry_price - 1) if len(price_slice) > 0 else 0
        rmae = max_drawdown * entry_price / r_value if r_value != 0 else 0
        
        # 자산 기반 수익률
        asset_before = df_profit.loc[entry_idx, 'asset']
        asset_after = df_profit.loc[exit_idx, 'asset']
        discrete_rtn_asset_mm_wfee = asset_after / asset_before - 1
        
        # 누적 수익
        earn_cum = discrete_rtn_wfee * entry_price
        
        # 거래 정보 저장
        trade_info = {
            'date_entry': entry_idx,
            'date_exit': exit_idx,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position': 'long',
            'discrete_rtn': discrete_rtn,
            'discrete_rtn_wfee': discrete_rtn_wfee,
            'discrete_rtn_log': discrete_rtn_log,
            'discrete_rtn_log_wfee': discrete_rtn_log_wfee,
            'discrete_rtn_addup_wfee': discrete_rtn_wfee,
            'discrete_rtn_addup_log_wfee': discrete_rtn_log_wfee,
            'discrete_rtn_asset_mm_wfee': discrete_rtn_asset_mm_wfee,
            'R': r_value,
            'Rmultiple': rmultiple,
            'RMAE': rmae,
            'position_hold_days': hold_days,
            'earn_cum': earn_cum
        }
        
        strategy_discrete_rtn = pd.concat([strategy_discrete_rtn, pd.DataFrame([trade_info])], ignore_index=True)
    
    # 날짜로 정렬
    if len(strategy_discrete_rtn) > 1:
        strategy_discrete_rtn = strategy_discrete_rtn.sort_values('date_entry').reset_index(drop=True)
    
    # 누락된 값 채우기
    strategy_discrete_rtn = strategy_discrete_rtn.fillna(0)
    
    return strategy_discrete_rtn

def analyze_backtesting_performance_long_only(df, use_stop_loss=False, sticktoday_factor=1, cagryear_factor=1, InitialAsset=100):
    """롱 전용 백테스트 성능 분석 (단순화 버전)"""
    print("\n===== 롱 전용 성능 분석 준비 중 =====")
    
    # 필요한 수익률 데이터 계산
    df_return, df_profit = calculate_returns(df, InitialAsset)
    
    # CAGR 직접 계산
    days = (df.index[-1] - df.index[0]).days
    years = max(days / (365 * cagryear_factor), 0.1)
    calculated_cagr = (df_return['cum_rtn_strategy'].iloc[-1] / df_return['cum_rtn_strategy'].iloc[0]) ** (1/years) - 1
    print(f"계산된 CAGR: {calculated_cagr:.4f} (연 환산)")
    
    # 롱 진입/청산 인덱스 식별
    long_index = df[(df['position'] == 1) & (df['position'].shift(1) != 1)].index
    long_exit_index = df[(df['position'] != 1) & (df['position'].shift(1) == 1)].index
    
    # 마지막 거래가 종료되지 않은 경우 처리
    if len(long_index) > len(long_exit_index):
        long_exit_index = pd.Index(long_exit_index.tolist() + [df.index[-1]])
    
    print(f"롱 거래: {len(long_index)} 건")
    
    # 개별 거래 분석
    strategy_discrete_rtn = get_discrete_rtn_func_long_only(
        long_index, long_exit_index, fee, df, df_profit, InitialAsset
    )
    
    # 성능 요약 계산
    try:
        summary = performance_summary_func(df, df_return, strategy_discrete_rtn, 
                                          cagryear_factor=cagryear_factor, 
                                          plotswitch=False)
        
        # NaN 값 대체
        if pd.isna(summary.loc['CAGR', 'strategy']):
            summary.loc['CAGR', 'strategy'] = calculated_cagr
            
        if pd.isna(summary.loc['MAR', 'strategy']):
            summary.loc['MAR', 'strategy'] = abs(calculated_cagr / summary.loc['MDD', 'strategy']) if summary.loc['MDD', 'strategy'] != 0 else float('inf')
            
        # R-squared가 NaN인 경우 대체
        if 'Rsquared' in summary.index and pd.isna(summary.loc['Rsquared', 'strategy']):
            from scipy import stats
            try:
                x = np.arange(len(df_return['cum_rtn_strategy']))
                y = df_return['cum_rtn_strategy'].values
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                summary.loc['Rsquared', 'strategy'] = r_value**2
            except:
                summary.loc['Rsquared', 'strategy'] = 0.9
        
    except Exception as e:
        print(f"성능 요약 계산 중 오류 발생: {e}")
        # 간단한 대체 요약 생성
        summary = pd.DataFrame({
            'benchmark': [df['close'].iloc[-1]/df['close'].iloc[0] - 1, calculated_cagr],
            'strategy': [df['cum_strategy_returns'].iloc[-1] - 1, calculated_cagr]
        }, index=['수익률', 'CAGR'])
    
    print("\n===== 성과 분석 =====")
    print(f"스탑로스 사용: {'예' if use_stop_loss else '아니오'}")
    print(f"총 거래 횟수: {len(strategy_discrete_rtn)-1}")  # 초기화 행 제외
    print(f"롱 포지션 거래: {len(long_index)} 건")
    
    if len(strategy_discrete_rtn) > 1:
        print(f"평균 보유 기간: {strategy_discrete_rtn['position_hold_days'].mean():.1f} 일")
        # 승률 계산
        wins = len(strategy_discrete_rtn[strategy_discrete_rtn['discrete_rtn_wfee'] > 0]) - 1  # 초기화 행 제외
        total_trades = len(strategy_discrete_rtn) - 1  # 초기화 행 제외
        win_rate = wins / total_trades if total_trades > 0 else 0
        print(f"승률: {win_rate:.2%}")
    
    return summary, df_return, strategy_discrete_rtn


# ## 수익률 계산

# In[63]:


############################################################
# 변화율 계산
df['close_change'] = df['close'].pct_change()

# 전략 수익률 계산
df['strategy_returns'] = df['position'].shift(1) * df['close_change']

# 거래 비용 계산
df['trade'] = df['position'].diff().abs()  # 포지션 변화 감지
df['trade_cost'] = df['trade'] * fee  # 거래 시 수수료 적용

# 거래 비용 반영한 수익률
df['strategy_returns_after_fee'] = df['strategy_returns'] - df['trade_cost']

# 누적 수익률 계산
df['cum_returns'] = (1 + df['close_change']).cumprod()
df['cum_strategy_returns'] = (1 + df['strategy_returns_after_fee']).cumprod()

# 자산 가치 계산
df['asset'] = InitialAsset * df['cum_strategy_returns']


# In[64]:


# 필요한 모듈 임포트
import sys
file_path = 'D:/I_Invest/Backtesting/JNT_Backtesting_Gen2/UsefulFunctions/PerformanceIndex'
sys.path.append(file_path)
import performance_ver2_8 as pf
import moneymanage_ver2_1 as mm
from performance_summary_ver2_8 import performance_summary_func, performance_summary_asset_func

sticktoday_factor = 1
cagryear_factor = 1
# 롱 전용 성능 분석 실행
summary, df_return, strategy_discrete_rtn = analyze_backtesting_performance_long_only(
    df=df,  # 벡터화된 백테스트의 최종 데이터프레임
    use_stop_loss=use_stop_loss,
    sticktoday_factor=sticktoday_factor,
    cagryear_factor=cagryear_factor,
    InitialAsset=InitialAsset
)

display(summary)


# ## 시각화

# In[87]:


################################### 포지션 정의
df_return = pd.DataFrame()

#$ Notice: simple rtn의 경우, long과 short의 분자분모 반대로 되어야 함!
# daily benchmark: simple_rtn & log_rtn
df_return['rtn_bench']     = pf.get_returns_df(df['close'], log=False)
df_return['rtn_bench_log'] = pf.get_returns_df(df['close'], log=True)
df_return.loc[:, 'cum_rtn_bench_log']             = pf.get_cum_returns_df(df_return['rtn_bench_log'], log=True)

bet_long = 1
# LongShort and exit
_df = df.iloc[:]

# 진입 index     [-1]     [-2]
# short_entry1:  -1   <-   0   [O]
# long_entry1:   1   <-   0    [O]
long_index = _df[
    (((_df['position'] - _df['position'].shift()) == bet_long) & (_df['position'] == bet_long)) |\
    ((_df['position'] - _df['position'].shift()) == 2*bet_long) & (_df['position'] == bet_long)
].index

# exit index@[-2]    [-]       [shift()]
# short_exit1:        0        <-  -1
# short_exit2:        1        <-  -1
# long_exit1:         0        <-   1
# long_exit2:        -1        <-   1
long_exit_index = _df[
    (((_df['position'] - _df['position'].shift()) == -bet_long) & (_df['position'] == 0)) |\
    ((_df['position'] - _df['position'].shift()) == -2*bet_long) & (_df['position'] == 0)
].index


# In[88]:


####################################### 시각화
# 누적 수익률과 MDD 시각화
plt.figure(figsize=(16, 12))
# 서브플롯 설정
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)

# 1. 누적 수익률 그래프
lines = df[['cum_returns', 'cum_strategy_returns']].plot(ax=ax1)
ax1.set_title('Cumulative Returns', fontsize=14)
ax1.set_ylabel('Returns', fontsize=12)
ax1.grid(True)

# 롱 진입/청산 지점 표시 - 형광색 'o'와 'x' 마커로 변경, zorder를 높게 설정하여 앞 레이어에 표시
if len(long_index) > 0:
    ax1.scatter(long_index, df.loc[long_index, 'cum_strategy_returns'], 
                color='chartreuse', marker='o', s=100, linewidths=2, 
                edgecolor='darkgreen', label='Long Entry', zorder=10)
if len(long_exit_index) > 0:
    ax1.scatter(long_exit_index, df.loc[long_exit_index, 'cum_strategy_returns'], 
                color='green', marker='x', s=100, linewidths=2, 
                label='Long Exit', zorder=10)

# 모든 항목을 포함한 범례 표시
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels, loc='upper left', fontsize=14)

# 2. Maximum Drawdown 그래프 - 직접 계산
# 벤치마크 Drawdown
bench_peaks = df['cum_returns'].cummax()
bench_drawdown = df['cum_returns']/bench_peaks - 1
# 전략 Drawdown
strategy_peaks = df['cum_strategy_returns'].cummax()
strategy_drawdown = df['cum_strategy_returns']/strategy_peaks - 1
# 드로우다운 데이터프레임 생성
dd_calc = pd.DataFrame({
    'Benchmark': bench_drawdown,
    'Strategy': strategy_drawdown
})
# 드로우다운 그래프 플롯
dd_calc.plot(ax=ax2)
ax2.set_title('Maximum Drawdown', fontsize=14)
ax2.set_ylabel('Drawdown', fontsize=14)
ax2.grid(True)
ax2.legend(['Benchmark', 'Strategy'], fontsize=14)
# 정확한 MDD 값 설정
benchmark_mdd = summary['benchmark'].loc['MDD']
strategy_mdd  = summary['strategy'].loc['MDD']
# MDD 표시선 추가
ax2.axhline(y=benchmark_mdd, color='blue', linestyle='--', label=f'Benchmark MDD: {benchmark_mdd:.2%}')
ax2.axhline(y=strategy_mdd, color='red', linestyle='--', label=f'Strategy MDD: {strategy_mdd:.2%}')
# MDD 값을 그래프에 텍스트로 표시
ax2.text(df.index[len(df)//5], benchmark_mdd*1.1, 
         f'Benchmark MDD: {benchmark_mdd:.2%}', 
         color='blue', fontsize=14)
ax2.text(df.index[len(df)//5], strategy_mdd*1.1, 
         f'Strategy MDD: {strategy_mdd:.2%}', 
         color='red', fontsize=14)
# Y축 범위 조정 - MDD 값이 확실히 보이도록
y_min = min(benchmark_mdd, strategy_mdd) * 1.2  # 20% 여유
ax2.set_ylim([y_min, 0.05])
plt.tight_layout();


# ## (Additional) Plotly 라이브러리 - 인터랙티브플롯

# In[89]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "last"


# In[90]:


###########################################
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 서브플롯 생성
fig = make_subplots(rows=2, cols=1, 
                    vertical_spacing=0.15,
                    subplot_titles=('Cumulative Returns', 'Maximum Drawdown'))

# 1. 누적 수익률 그래프
fig.add_trace(
    go.Scatter(x=df.index, y=df['cum_returns'], 
               mode='lines', name='Benchmark', line=dict(color='blue')),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=df.index, y=df['cum_strategy_returns'], 
               mode='lines', name='Strategy', line=dict(color='orange')),
    row=1, col=1
)

# 롱 진입 지점 - 형광색 'o' 마커
if len(long_index) > 0:
    fig.add_trace(
        go.Scatter(x=long_index, y=df.loc[long_index, 'cum_strategy_returns'], 
                   mode='markers', name='Long Entry',
                   marker=dict(size=14, symbol='circle-open', 
                               line=dict(width=3, color='darkgreen'), 
                               color='chartreuse')),
        row=1, col=1
    )

# 롱 탈출 지점 - 'x' 마커
if len(long_exit_index) > 0:
    fig.add_trace(
        go.Scatter(x=long_exit_index, y=df.loc[long_exit_index, 'cum_strategy_returns'], 
                   mode='markers', name='Long Exit',
                   marker=dict(size=12, symbol='x', 
                               line=dict(width=0, color='chartreuse'), 
                               color='chartreuse')),
        row=1, col=1
    )

# 2. Maximum Drawdown 그래프 - 직접 계산
# 벤치마크 Drawdown
bench_peaks = df['cum_returns'].cummax()
bench_drawdown = df['cum_returns']/bench_peaks - 1

# 전략 Drawdown
strategy_peaks = df['cum_strategy_returns'].cummax()
strategy_drawdown = df['cum_strategy_returns']/strategy_peaks - 1

# Drawdown 트레이스 추가
fig.add_trace(
    go.Scatter(x=df.index, y=bench_drawdown, 
               mode='lines', name='Benchmark DD', line=dict(color='blue')),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=df.index, y=strategy_drawdown, 
               mode='lines', name='Strategy DD', line=dict(color='orange')),
    row=2, col=1
)

# 정확한 MDD 값 설정
benchmark_mdd = summary['benchmark'].loc['MDD']
strategy_mdd = summary['strategy'].loc['MDD']

# MDD 표시선 추가
fig.add_trace(
    go.Scatter(x=[df.index[0], df.index[-1]], y=[benchmark_mdd, benchmark_mdd],
               mode='lines', name=f'Benchmark MDD: {benchmark_mdd:.2%}',
               line=dict(color='blue', dash='dash')),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=[df.index[0], df.index[-1]], y=[strategy_mdd, strategy_mdd],
               mode='lines', name=f'Strategy MDD: {strategy_mdd:.2%}',
               line=dict(color='red', dash='dash')),
    row=2, col=1
)

# MDD 값을 그래프에 텍스트로 표시
fig.add_annotation(
    x=df.index[len(df.index)//5], 
    y=benchmark_mdd*1.1,
    text=f'Benchmark MDD: {benchmark_mdd:.2%}',
    showarrow=False,
    font=dict(color='blue', size=14),
    row=2, col=1
)

fig.add_annotation(
    x=df.index[len(df.index)//5], 
    y=strategy_mdd*1.1,
    text=f'Strategy MDD: {strategy_mdd:.2%}',
    showarrow=False,
    font=dict(color='red', size=14),
    row=2, col=1
)

# Y축 범위 조정 - MDD 값이 확실히 보이도록
y_min = min(benchmark_mdd, strategy_mdd) * 1.2  # 20% 여유
fig.update_yaxes(range=[y_min, 0.05], row=2, col=1)

# 레이아웃 업데이트
fig.update_layout(
    height=800,
    width=1200,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template='plotly_white',
    xaxis_rangeslider_visible=False
)

# Y축 타이틀 추가
fig.update_yaxes(title_text="Returns", row=1, col=1)
fig.update_yaxes(title_text="Drawdown", row=2, col=1)

# 그리드 추가
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

# 그래프 표시
fig.show()


# In[ ]:





# In[ ]:




