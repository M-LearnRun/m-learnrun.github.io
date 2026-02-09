###
import pandas as pd
import numpy as np
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')


class MonthlyTrackingBacktest:
    def __init__(self, data_folder):
        """
        연말 리밸런싱 + 월간 데이터 추적 백테스트
        (리밸런싱: 연말에만, 추적: 월간 데이터로 세밀하게)
        
        Parameters:
        data_folder (str): p1m 월간 데이터 폴더 경로
        """
        self.data_folder = Path(data_folder)
        self.stock_data = {}
        
    def load_data(self):
        """월간 데이터 로드"""
        print("📊 데이터 로딩...")
        
        # 월간 데이터 파일들 찾기
        pattern1 = str(self.data_folder / "*p1m*.csv")
        pattern2 = str(self.data_folder / "*p1m*.csv")
        
        csv_files = glob.glob(pattern1)
        if not csv_files:
            csv_files = glob.glob(pattern2)
            
        if not csv_files:
            raise FileNotFoundError(f"❌ {self.data_folder}에서 p1m 파일을 찾을 수 없습니다.")
        
        print(f"📁 파일 수: {len(csv_files):,}개")
        
        loaded_count = 0
        for file_path in csv_files:
            try:
                # 종목코드 추출
                file_name = Path(file_path).name
                parts = file_name.replace('.csv', '').split('_')
                stock_code = parts[-1]  # 마지막 부분이 종목코드
                
                # 순수 숫자 종목코드만 허용 (6자리)
                if not (stock_code.isdigit() and len(stock_code) == 6):
                    continue
                
                # 데이터 로드
                df = pd.read_csv(file_path)
                
                # 필수 컬럼 확인
                if not {'close', 'marcap'}.issubset(df.columns):
                    continue
                
                # 날짜 처리
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df['year'] = df['date'].dt.year
                    df['month'] = df['date'].dt.month
                    df['year_month'] = df['date'].dt.to_period('M')
                else:
                    # date 컬럼이 없으면 연월 추정 (2000년 1월부터 시작)
                    start_date = pd.Period('2000-01')
                    df['year_month'] = [start_date + i for i in range(len(df))]
                    df['year'] = df['year_month'].apply(lambda x: x.year)
                    df['month'] = df['year_month'].apply(lambda x: x.month)
                
                # 유효 데이터 필터링
                df = df[(df['close'] > 0) & (df['marcap'] > 0)]
                df = df.sort_values('year_month' if 'year_month' in df.columns else 'date')
                
                if len(df) >= 12:  # 최소 12개월 데이터
                    self.stock_data[stock_code] = df[['year', 'month', 'year_month', 'close', 'marcap']].copy()
                    loaded_count += 1
                    
            except Exception as e:
                continue
        
        print(f"✅ 로드 완료: {loaded_count:,}개 종목\n")
        
    def get_quantile_stocks_by_year(self, year, quantile_num):
        """
        특정 연도 12월 말 시총 기준 분위수 종목 선택
        
        Parameters:
        year (int): 기준 연도 (12월 말 시총)
        quantile_num (int): 분위수 (1~10, 1=최하위 10%, 10=최상위 10%)
        
        Returns:
        list: 선택된 종목 리스트
        """
        year_data = []
        
        # 해당 연도 12월 말 시총 데이터 수집
        for stock_code, df in self.stock_data.items():
            # 12월 데이터 찾기 (연말)
            dec_data = df[(df['year'] == year) & (df['month'] == 12)]
            if len(dec_data) > 0:
                year_data.append({
                    'stock_code': stock_code,
                    'marcap': dec_data.iloc[-1]['marcap']  # 12월 마지막 데이터
                })
        
        if len(year_data) < 10:  # 최소 10개 종목 필요
            return []
        
        # DataFrame 생성 및 순위 계산
        df_year = pd.DataFrame(year_data)
        df_year = df_year.sort_values('marcap')
        
        # 10분위로 나누기
        total_stocks = len(df_year)
        stocks_per_decile = total_stocks // 10
        
        start_idx = (quantile_num - 1) * stocks_per_decile
        end_idx = quantile_num * stocks_per_decile
        
        # 마지막 분위는 나머지 모두 포함
        if quantile_num == 10:
            end_idx = total_stocks
        
        selected_stocks = df_year.iloc[start_idx:end_idx]['stock_code'].tolist()
        
        return selected_stocks
    
    def calculate_portfolio_return(self, stock_list, year):
        """
        포트폴리오 수익률 계산 (전년 말 → 해당 연도 말)
        1번 코드와 동일한 로직
        
        Parameters:
        stock_list (list): 종목 리스트
        year (int): 투자 연도
        
        Returns:
        float: 포트폴리오 수익률
        """
        if len(stock_list) == 0:
            return 0.0
        
        returns = []
        start_year = year - 1
        end_year = year
        
        for stock_code in stock_list:
            if stock_code not in self.stock_data:
                continue
                
            df = self.stock_data[stock_code]
            
            # 시작 연도 12월 말 데이터
            start_data = df[(df['year'] == start_year) & (df['month'] == 12)]
            # 종료 연도 12월 말 데이터
            end_data = df[(df['year'] == end_year) & (df['month'] == 12)]
            
            if len(start_data) > 0 and len(end_data) > 0:
                start_price = start_data.iloc[-1]['close']  # 12월 마지막
                end_price = end_data.iloc[-1]['close']      # 12월 마지막
                
                if start_price > 0:
                    stock_return = (end_price - start_price) / start_price
                    
                    # 이상치 제거 (-99% ~ +999%)
                    if -0.99 <= stock_return <= 9.99:
                        returns.append(stock_return)
        
        # 동일가중 평균
        if len(returns) > 0:
            return np.mean(returns)
        else:
            return 0.0
    
    def calculate_portfolio_value_monthly(self, stock_list, base_year, target_year_month):
        """
        포트폴리오의 특정 월 가치 계산 (기준년도 12월 말 = 1.0)
        
        Parameters:
        stock_list (list): 종목 리스트 (기준년도에 선택된 종목들)
        base_year (int): 기준 연도 (포트폴리오 선택 기준 연도)
        target_year_month (Period): 목표 연월
        
        Returns:
        float: 포트폴리오 가치 (기준년도 12월 말 대비)
        """
        if len(stock_list) == 0:
            return 1.0
        
        valid_returns = []
        
        for stock_code in stock_list:
            if stock_code not in self.stock_data:
                continue
                
            df = self.stock_data[stock_code]
            
            # 기준점: 기준년도 12월 말 가격
            base_data = df[(df['year'] == base_year) & (df['month'] == 12)]
            
            # 목표점: 타겟 연월 가격
            target_data = df[df['year_month'] == target_year_month]
            
            if len(base_data) > 0 and len(target_data) > 0:
                base_price = base_data.iloc[-1]['close']  # 12월 마지막
                target_price = target_data.iloc[0]['close']
                
                if base_price > 0:
                    stock_value = target_price / base_price
                    valid_returns.append(stock_value)
        
        # 동일가중 평균으로 포트폴리오 가치 계산
        if len(valid_returns) > 0:
            return np.mean(valid_returns)
        else:
            return 1.0
    
    def run_backtest(self, quantile_num, start_year=2005, end_year=None, debug=False):
        """
        연말 리밸런싱 + 월간 추적 백테스트 실행 (1번 코드와 동일한 로직)
        
        Parameters:
        quantile_num (int): 분위수 (1~10)
        start_year (int): 시작 연도
        end_year (int): 종료 연도
        debug (bool): 디버그 모드
        """
        if not self.stock_data:
            self.load_data()
        
        # 사용 가능한 연도 범위
        all_years = set()
        for df in self.stock_data.values():
            all_years.update(df['year'].unique())
        all_years = sorted(list(all_years))
        
        if end_year is None:
            end_year = max(all_years)
        
        print(f"🚀 백테스트 시작")
        print(f"📊 분위수: {quantile_num}분위 ({'하위' if quantile_num <= 5 else '상위'} {quantile_num*10}%)")
        print(f"📅 기간: {start_year}~{end_year}")
        print("="*60)
        
        # 초기값
        portfolio_value = 1.0
        annual_returns = []
        annual_results = []
        monthly_results = []
        
        # 현재 포트폴리오 (연도별로 관리)
        portfolio_by_year = {}
        
        # 1. 먼저 모든 연도의 포트폴리오를 미리 선택 (1번 코드와 동일한 방식)
        for year in range(start_year, end_year + 1):
            selection_year = year - 1  # 전년도 기준으로 종목 선택
            
            if selection_year < min(all_years):
                continue
            
            # 전년도 12월 말 시총으로 해당 연도 포트폴리오 선택
            selected_stocks = self.get_quantile_stocks_by_year(selection_year, quantile_num)
            portfolio_by_year[year] = selected_stocks
        
        # 2. 연도별 수익률 계산 및 누적 (1번 코드와 동일한 방식)
        for year in range(start_year, end_year + 1):
            if year not in portfolio_by_year or len(portfolio_by_year[year]) == 0:
                print(f"⚠️ {year}년: 선택된 종목 없음")
                continue
            
            # 포트폴리오 수익률 계산 (전년 말 → 해당 연도 말)
            annual_return = self.calculate_portfolio_return(portfolio_by_year[year], year)
            
            # 포트폴리오 가치 업데이트
            portfolio_value *= (1 + annual_return)
            annual_returns.append(annual_return)
            
            annual_results.append({
                'year': year,
                'num_stocks': len(portfolio_by_year[year]),
                'annual_return': annual_return,
                'portfolio_value': portfolio_value
            })
            
            print(f"📈 {year}년: {annual_return:>7.2%} | 누적: {portfolio_value:>8.3f} | 종목: {len(portfolio_by_year[year])}개")
            
            # 디버그 모드 - 첫 해 상세 분석
            if debug and year == start_year:
                self.debug_first_year(portfolio_by_year[year], year)
        
        # 3. 월간 추적 데이터 생성 (MDD 계산용)
        # 사용 가능한 모든 연월 수집
        all_months = set()
        for df in self.stock_data.values():
            all_months.update(df['year_month'].unique())
        all_months = sorted(list(all_months))
        
        # 누적 수익률을 월별로 계산
        cumulative_value = 1.0
        current_portfolio = []
        current_base_year = start_year - 1
        
        for year_month in all_months:
            year = year_month.year
            month = year_month.month
            
            # 백테스트 기간 체크
            if year < start_year or year > end_year:
                continue
            
            # 해당 연도의 포트폴리오 사용
            if year in portfolio_by_year:
                current_portfolio = portfolio_by_year[year]
                current_base_year = year - 1
            
            # 월간 포트폴리오 가치 계산
            if len(current_portfolio) > 0:
                # 해당 연도의 연초(전년 12월 말) 대비 현재 월의 가치
                monthly_portfolio_value = self.calculate_portfolio_value_monthly(
                    current_portfolio, current_base_year, year_month
                )
                
                # 전체 누적 수익률에 반영
                # 전년도까지의 누적 가치 * 올해 현재까지의 성과
                if year == start_year:
                    total_value = monthly_portfolio_value
                else:
                    # 전년도 말까지의 누적 가치 찾기
                    prev_year_value = 1.0
                    for result in annual_results:
                        if result['year'] < year:
                            prev_year_value = result['portfolio_value']
                    
                    total_value = prev_year_value * monthly_portfolio_value
                
                monthly_results.append({
                    'year_month': str(year_month),
                    'year': year,
                    'month': month,
                    'portfolio_value': total_value,
                    'num_stocks': len(current_portfolio),
                    'is_rebalancing': month == 12
                })
        
        # 성과 지표 계산
        performance = self.calculate_performance(annual_returns, annual_results, monthly_results)
        
        # 시각화 및 결과 저장
        if len(monthly_results) > 0:
            self.plot_cumulative_returns(monthly_results, quantile_num, start_year)
            self.save_results_to_csv(monthly_results, performance, quantile_num, start_year)
        
        return {
            'annual_returns': annual_returns,
            'annual_results': annual_results,
            'monthly_results': monthly_results,
            'performance': performance
        }
    
    def debug_first_year(self, stock_list, year):
        """첫 해 상세 분석"""
        print(f"\n🔍 {year}년 상세 분석 (처음 10개 종목)")
        print("-" * 80)
        
        returns = []
        for i, stock_code in enumerate(stock_list[:10]):
            if stock_code not in self.stock_data:
                continue
                
            df = self.stock_data[stock_code]
            start_data = df[(df['year'] == year-1) & (df['month'] == 12)]
            end_data = df[(df['year'] == year) & (df['month'] == 12)]
            
            if len(start_data) > 0 and len(end_data) > 0:
                start_price = start_data.iloc[-1]['close']
                end_price = end_data.iloc[-1]['close']
                stock_return = (end_price - start_price) / start_price
                
                returns.append(stock_return)
                print(f"{stock_code}: {start_price:>8,.0f}원 → {end_price:>8,.0f}원 ({stock_return:>7.2%})")
        
        if returns:
            avg_return = np.mean(returns)
            print(f"\n📊 평균 수익률: {avg_return:.2%}")
            print(f"📊 수익률 범위: {min(returns):.2%} ~ {max(returns):.2%}")
        print("-" * 80)
    
    def calculate_performance(self, annual_returns, annual_results, monthly_results):
        """성과 지표 계산 (1번 코드와 동일한 방식)"""
        if len(annual_returns) == 0:
            return {}
        
        total_years = len(annual_returns)
        final_value = annual_results[-1]['portfolio_value']
        
        # CAGR
        cagr = (final_value ** (1/total_years)) - 1
        
        # 변동성 (연간 수익률 기준)
        volatility = np.std(annual_returns)
        
        # 샤프 비율
        sharpe = np.mean(annual_returns) / volatility if volatility > 0 else 0
        
        # MDD (월간 추적 데이터 기준)
        if monthly_results:
            values = [r['portfolio_value'] for r in monthly_results]
            peaks = np.maximum.accumulate(values)
            drawdowns = (np.array(values) - peaks) / peaks
            mdd = np.min(drawdowns)
        else:
            # 월간 데이터가 없으면 연간 데이터로 MDD 계산
            values = [r['portfolio_value'] for r in annual_results]
            peaks = np.maximum.accumulate(values)
            drawdowns = (np.array(values) - peaks) / peaks
            mdd = np.min(drawdowns)
        
        # 승률
        win_rate = np.mean(np.array(annual_returns) > 0)
        
        return {
            'total_years': total_years,
            'final_value': final_value,
            'total_return': final_value - 1,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe': sharpe,
            'mdd': mdd,
            'win_rate': win_rate,
            'best_year': max(annual_returns) if annual_returns else 0,
            'worst_year': min(annual_returns) if annual_returns else 0
        }
    
    def plot_cumulative_returns(self, results, quantile_num, start_year):
        """누적수익률 그래프 (월간 추적)"""
        
        # 데이터 준비 (시작 연도 전년도에 1.0 추가)
        months = [f"{start_year-1}-12"] + [r['year_month'] for r in results]
        values = [1.0] + [r['portfolio_value'] for r in results]
        
        # 그래프 생성
        plt.figure(figsize=(15, 8))
        
        # 주 그래프 (월간 추적)
        plt.plot(range(len(values)), values, linewidth=2, alpha=0.8, label='월간 추적')
        
        # 그래프 꾸미기
        plt.title(f'{quantile_num}분위 포트폴리오 누적수익률 (연말 리밸런싱: {months[0]}~{months[-1]})', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('월', fontsize=12)
        plt.ylabel('누적수익률 (초기자금 1.0)', fontsize=12)
        
        # X축 라벨 설정 (연말만 표시)
        x_ticks = [0] + [i+1 for i, r in enumerate(results) if r['month'] == 12]
        x_labels = [months[i][:4] for i in x_ticks]  # 연도만
        plt.xticks(x_ticks, x_labels, rotation=45)
        
        # 격자 및 범례
        plt.grid(True, alpha=0.3)
        
        # Y축 로그 스케일 (큰 수익률 시각화를 위해)
        if max(values) > 10:
            plt.yscale('log')
            plt.ylabel('누적수익률 (초기자금 1.0, 로그스케일)', fontsize=12)
        
        # 최종 수익률 텍스트 표시
        final_return = (values[-1] - 1) * 100
        total_years = len(set([r['year'] for r in results]))
        plt.text(0.02, 0.98, f'최종 수익률: {final_return:,.1f}%\n총 {total_years}년', 
                transform=plt.gca().transAxes, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                verticalalignment='top')
        
        plt.tight_layout()
        
        # result 폴더에 저장
        result_folder = Path("result")
        result_folder.mkdir(exist_ok=True)
        
        plot_filename = result_folder / f"cumulative_returns_{quantile_num}분위_{start_year-1}_{results[-1]['year']}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📈 그래프 저장: {plot_filename}")
        
        plt.show()
    
    def save_results_to_csv(self, results, performance, quantile_num, start_year):
        """결과를 CSV 파일로 저장"""
        
        result_folder = Path("result")
        result_folder.mkdir(exist_ok=True)
        
        # 1. 월간 추적 데이터
        tracking_df = pd.DataFrame(results)
        tracking_filename = result_folder / f"monthly_tracking_{quantile_num}분위_{start_year-1}_{results[-1]['year']}.csv"
        tracking_df.to_csv(tracking_filename, index=False, encoding='utf-8-sig')
        print(f"📊 월간 추적 데이터 저장: {tracking_filename}")
        
        # 2. 성과 요약 테이블
        summary_data = {
            '분위수': f"{quantile_num}분위 ({'하위' if quantile_num <= 5 else '상위'} {quantile_num*10}%)",
            '투자기간': f"{start_year}~{results[-1]['year']} ({performance['total_years']}년)",
            '리밸런싱': "연말 (12월 말)",
            '추적주기': "월간",
            '최종가치': f"{performance['final_value']:.3f}",
            '총수익률(%)': f"{performance['total_return']*100:.1f}",
            'CAGR(%)': f"{performance['cagr']*100:.2f}",
            '변동성(%)': f"{performance['volatility']*100:.2f}",
            '샤프비율': f"{performance['sharpe']:.3f}",
            'MDD(%)': f"{performance['mdd']*100:.2f}",
            '승률(%)': f"{performance['win_rate']*100:.1f}",
            '최고연간수익률(%)': f"{performance['best_year']*100:.2f}",
            '최악연간수익률(%)': f"{performance['worst_year']*100:.2f}"
        }
        
        summary_df = pd.DataFrame([summary_data])
        summary_filename = result_folder / f"performance_summary_{quantile_num}분위_{start_year-1}_{results[-1]['year']}.csv"
        summary_df.to_csv(summary_filename, index=False, encoding='utf-8-sig')
        print(f"📋 성과 요약 저장: {summary_filename}")
        
        return tracking_filename, summary_filename
    
    def print_performance(self, performance):
        """결과 출력"""
        print(f"\n📊 투자 성과 요약 (월간 추적)")
        print("="*60)
        print(f"📅 투자기간: {performance['total_years']}년")
        print(f"💰 최종가치: {performance['final_value']:.3f} (초기자금 1.0)")
        print(f"📈 총 수익률: {performance['total_return']:.1%}")
        print(f"🚀 연평균 수익률(CAGR): {performance['cagr']:.2%}")
        print(f"📊 변동성: {performance['volatility']:.2%}")
        print(f"⚡ 샤프 비율: {performance['sharpe']:.3f}")
        print(f"📉 최대 낙폭(MDD): {performance['mdd']:.2%}")
        print(f"🎯 승률: {performance['win_rate']:.1%}")
        print(f"🔥 최고 연간 수익률: {performance['best_year']:.2%}")
        print(f"❄️ 최악 연간 수익률: {performance['worst_year']:.2%}")
    
    def compare_quantiles(self, quantile_list, start_year=2005, end_year=None):
        """여러 분위수 비교 백테스트"""
        print(f"\n🔍 여러 분위수 성과 비교 (연말 리밸런싱)")
        print("="*80)
        
        comparison_results = []
        
        for quantile in quantile_list:
            print(f"\n📊 {quantile}분위 백테스트 진행 중...")
            result = self.run_backtest(quantile, start_year, end_year, debug=False)
            perf = result['performance']
            
            comparison_results.append({
                '분위수': quantile,
                '최종가치': perf['final_value'],
                'CAGR(%)': perf['cagr'] * 100,
                '변동성(%)': perf['volatility'] * 100,
                '샤프비율': perf['sharpe'],
                'MDD(%)': perf['mdd'] * 100,
                '승률(%)': perf['win_rate'] * 100
            })
        
        # 비교 결과 출력
        comparison_df = pd.DataFrame(comparison_results)
        print(f"\n📋 분위수별 성과 비교표 (연말 리밸런싱)")
        print("="*80)
        print(comparison_df.round(2).to_string(index=False))
        
        # 비교 결과 CSV 저장
        result_folder = Path("result")
        result_folder.mkdir(exist_ok=True)
        comparison_filename = result_folder / f"quantile_comparison_{start_year-1}_{end_year if end_year else 'latest'}.csv"
        comparison_df.to_csv(comparison_filename, index=False, encoding='utf-8-sig')
        print(f"\n📊 비교 결과 저장: {comparison_filename}")
        
        return comparison_df


# 월간 추적 백테스트 (수정된 버전)
data_folder = r"D:\I_Invest\Backtesting\JNT_Backtesting_Gen2\DataAcquisition_toCSV\FinanceDataReader_KRX\data\p1m_data"
backtest = MonthlyTrackingBacktest(data_folder)

# 단일 분위수 테스트
result = backtest.run_backtest(quantile_num=6, start_year=2000, debug=True)
backtest.print_performance(result['performance'])

# 여러 분위수 비교
# comparison = backtest.compare_quantiles([1, 5, 10], start_year=2005)