"""
지하철 혼잡도 예측 - 데이터 전처리 모듈
subway-congestion-prediction/src/preprocessor.py

주요 기능:
1. 와이드 포맷 → 롱 포맷 변환
2. 날짜/시간 데이터 처리
3. 결측치 처리
4. 이상치 탐지 및 처리
5. 데이터 정규화
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')


class SubwayDataPreprocessor:
    """지하철 승하차 데이터 전처리 클래스"""
    
    def __init__(self, raw_data_path, processed_data_path):
        """
        Args:
            raw_data_path: 원본 데이터 경로 (data/raw/)
            processed_data_path: 전처리된 데이터 저장 경로 (data/processed/)
        """
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        
        # 폴더가 없으면 생성
        os.makedirs(processed_data_path, exist_ok=True)
        
        self.df = None
        self.df_long = None
        
    def load_data(self, filename='subway_20250101_20251101.csv'):
        """원본 데이터 로드"""
        filepath = os.path.join(self.raw_data_path, filename)
        
        print(f"📂 데이터 로딩 중: {filepath}")
        self.df = pd.read_csv(filepath, encoding='utf-8-sig')
        
        print(f"✅ 로딩 완료!")
        print(f"   - 행 수: {len(self.df):,}")
        print(f"   - 열 수: {len(self.df.columns)}")
        print(f"   - 메모리: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return self.df
    
    def check_data_quality(self):
        """데이터 품질 체크"""
        print("\n🔍 데이터 품질 체크")
        print("=" * 60)
        
        # 실제 컬럼명 출력
        print("\n📋 실제 컬럼명:")
        print(self.df.columns.tolist())
        
        # 1. 결측치 확인
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print("\n⚠️  결측치 발견:")
            print(missing[missing > 0])
        else:
            print("✅ 결측치 없음")
        
        # 컬럼명이 실제로 존재하는지 확인
        required_cols = []
        for col in self.df.columns:
            if '일자' in col or '날짜' in col or '사용일' in col:
                date_col = col
                required_cols.append(col)
            elif '노선' in col or '호선' in col:
                line_col = col
                required_cols.append(col)
            elif '역명' in col or '역' in col:
                station_col = col
                required_cols.append(col)
        
        # 2. 중복 데이터 확인 (컬럼이 존재하는 경우만)
        if len(required_cols) >= 3:
            duplicates = self.df.duplicated(subset=required_cols).sum()
            print(f"\n중복 행: {duplicates}개")
        
        # 3. 날짜 컬럼 찾기 및 변환
        date_column = None
        for col in self.df.columns:
            if '일자' in col or '날짜' in col or '사용일' in col:
                date_column = col
                break
        
        if date_column:
            # 날짜 형식 자동 감지 및 변환
            try:
                self.df[date_column] = pd.to_datetime(self.df[date_column], format='%Y%m%d')
            except:
                try:
                    self.df[date_column] = pd.to_datetime(self.df[date_column])
                except:
                    print(f"⚠️  날짜 변환 실패: {date_column}")
            
            print(f"\n날짜 범위: {self.df[date_column].min()} ~ {self.df[date_column].max()}")
        
        # 4. 노선 및 역 개수 (컬럼이 존재하는 경우만)
        for col in self.df.columns:
            if '노선' in col or '호선' in col:
                print(f"\n노선 수: {self.df[col].nunique()}개")
            if '역명' in col or '역' in col:
                print(f"역 수: {self.df[col].nunique()}개")
        
        return self.df
    
    def wide_to_long(self):
        """와이드 포맷 → 롱 포맷 변환"""
        print("\n🔄 데이터 형태 변환 중 (Wide → Long)")
        
        # 시간대 컬럼 추출
        time_columns = [col for col in self.df.columns if '시-' in col]
        
        print(f"   - 시간대 컬럼 수: {len(time_columns)}개")
        
        # 롱 포맷으로 변환
        self.df_long = pd.melt(
            self.df,
            id_vars=['USE_YMD', 'USE_YMD', 'SBWY_STNS_NM'],
            value_vars=time_columns,
            var_name='시간대',
            value_name='승하차인원'
        )
        
        # 시간대 정리 (05시-06시 → 05)
        self.df_long['시간'] = self.df_long['시간대'].str.extract(r'(\d+)시').astype(int)
        
        # 불필요한 컬럼 제거
        self.df_long = self.df_long.drop('시간대', axis=1)
        
        print(f"✅ 변환 완료!")
        print(f"   - 변환 후 행 수: {len(self.df_long):,}")
        
        return self.df_long
    
    def add_time_features(self):
        """날짜/시간 관련 피처 추가"""
        print("\n📅 시간 피처 생성 중...")
        
        # 날짜 정보 추출
        self.df_long['연도'] = self.df_long['사용일자'].dt.year
        self.df_long['월'] = self.df_long['사용일자'].dt.month
        self.df_long['일'] = self.df_long['사용일자'].dt.day
        self.df_long['요일'] = self.df_long['사용일자'].dt.dayofweek
        self.df_long['요일명'] = self.df_long['사용일자'].dt.day_name()
        
        # 주중/주말 구분
        self.df_long['주말여부'] = self.df_long['요일'].apply(lambda x: 1 if x >= 5 else 0)
        
        # 출퇴근 시간대 구분
        def classify_time_period(hour):
            if 7 <= hour <= 9:
                return '출근시간'
            elif 18 <= hour <= 20:
                return '퇴근시간'
            elif 11 <= hour <= 13:
                return '점심시간'
            elif 22 <= hour or hour <= 5:
                return '심야시간'
            else:
                return '일반시간'
        
        self.df_long['시간대구분'] = self.df_long['시간'].apply(classify_time_period)
        
        print("✅ 시간 피처 추가 완료!")
        
        return self.df_long
    
    def handle_outliers(self, method='iqr'):
        """이상치 처리"""
        print(f"\n🔧 이상치 처리 중...")
        
        Q1 = self.df_long['승하차인원'].quantile(0.25)
        Q3 = self.df_long['승하차인원'].quantile(0.75)
        IQR = Q3 - Q1
        
        upper_bound = Q3 + 1.5 * IQR
        
        # 음수 값은 0으로 처리
        self.df_long['승하차인원'] = self.df_long['승하차인원'].clip(lower=0)
        
        # 이상치는 상한값으로 대체
        outliers = (self.df_long['승하차인원'] > upper_bound * 2)
        outlier_count = outliers.sum()
        self.df_long.loc[outliers, '승하차인원'] = upper_bound
        
        print(f"✅ 이상치 처리 완료! 처리된 이상치: {outlier_count:,}개")
        
        return self.df_long
    
    def add_station_statistics(self):
        """역별 통계 정보 추가"""
        print("\n📊 역별 통계 생성 중...")
        
        # 역별 평균 승하차 인원
        station_avg = self.df_long.groupby('역명')['승하차인원'].mean().reset_index()
        station_avg.columns = ['역명', '역_평균승하차']
        
        # 시간대별 평균 승하차 인원
        time_avg = self.df_long.groupby('시간')['승하차인원'].mean().reset_index()
        time_avg.columns = ['시간', '시간_평균승하차']
        
        # 원본 데이터에 병합
        self.df_long = self.df_long.merge(station_avg, on='역명', how='left')
        self.df_long = self.df_long.merge(time_avg, on='시간', how='left')
        
        print("✅ 역별 통계 추가 완료!")
        
        return self.df_long
    
    def normalize_station_names(self):
        """역명 표준화"""
        print("\n🔤 역명 표준화 중...")
        
        self.df_long['역명'] = self.df_long['역명'].str.strip()
        self.df_long['노선명'] = self.df_long['노선명'].str.strip()
        
        print(f"✅ 표준화 완료!")
        
        return self.df_long
    
    def save_processed_data(self, filename='subway_processed.csv'):
        """전처리된 데이터 저장"""
        filepath = os.path.join(self.processed_data_path, filename)
        
        print(f"\n💾 전처리 데이터 저장 중: {filepath}")
        
        self.df_long.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        print(f"✅ 저장 완료!")
        print(f"   - 최종 행 수: {len(self.df_long):,}")
        print(f"   - 파일 크기: {os.path.getsize(filepath) / 1024**2:.2f} MB")
        
        return filepath
    
    def get_summary(self):
        """전처리 결과 요약"""
        print("\n" + "="*60)
        print("📋 전처리 결과 요약")
        print("="*60)
        
        print(f"\n총 데이터 수: {len(self.df_long):,}개")
        print(f"기간: {self.df_long['사용일자'].min()} ~ {self.df_long['사용일자'].max()}")
        print(f"노선 수: {self.df_long['노선명'].nunique()}개")
        print(f"역 수: {self.df_long['역명'].nunique()}개")
        
        print("\n승하차 인원 통계:")
        print(self.df_long['승하차인원'].describe())
        
        print("\n시간대구분별 평균 승하차:")
        print(self.df_long.groupby('시간대구분')['승하차인원'].mean().sort_values(ascending=False))


def main():
    """메인 실행 함수"""
    
    print("🚇 지하철 혼잡도 예측 - 데이터 전처리 시작")
    print("="*60)
    
    # 경로 설정
    RAW_DATA_PATH = 'data/raw/subway'
    PROCESSED_DATA_PATH = 'data/processed'
    
    # 전처리 객체 생성
    preprocessor = SubwayDataPreprocessor(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    
    # 1. 데이터 로드
    preprocessor.load_data()
    
    # 2. 데이터 품질 체크
    preprocessor.check_data_quality()
    
    # 3. 와이드 → 롱 포맷 변환
    preprocessor.wide_to_long()
    
    # 4. 시간 피처 추가
    preprocessor.add_time_features()
    
    # 5. 역명 표준화
    preprocessor.normalize_station_names()
    
    # 6. 이상치 처리
    preprocessor.handle_outliers()
    
    # 7. 역별 통계 추가
    preprocessor.add_station_statistics()
    
    # 8. 저장
    preprocessor.save_processed_data()
    
    # 9. 요약
    preprocessor.get_summary()
    
    print("\n✅ 전처리 완료!")


if __name__ == '__main__':
    main()