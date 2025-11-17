"""
지하철 혼잡도 예측 - 데이터 전처리 모듈 (시간대별 버전)
subway-congestion-prediction/src/preprocessor.py

주요 기능:
1. 와이드 포맷 → 롱 포맷 변환 (시간대별)
2. 날짜/시간 데이터 처리
3. 결측치 처리
4. 이상치 탐지 및 처리
5. 혼잡도 레벨 분류
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')


class SubwayDataPreprocessor:
    """지하철 승하차 데이터 전처리 클래스 (시간대별)"""
    
    def __init__(self, raw_data_path, processed_data_path):
        """
        Args:
            raw_data_path: 원본 데이터 경로 (data/raw/subway)
            processed_data_path: 전처리된 데이터 저장 경로 (data/processed/)
        """
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        
        # 폴더가 없으면 생성
        os.makedirs(processed_data_path, exist_ok=True)
        
        self.df = None
        self.df_long = None
        
    def load_data(self, filename='서울시 지하철 호선별 역별 시간대별 승하차 인원 정보 (1).csv'):
        """원본 데이터 로드"""
        filepath = os.path.join(self.raw_data_path, filename)
        
        print(f"📂 데이터 로딩 중: {filepath}")
        
        # cp949 인코딩으로 읽기
        self.df = pd.read_csv(filepath, encoding='cp949')
        
        print(f"✅ 로딩 완료!")
        print(f"   - 행 수: {len(self.df):,}")
        print(f"   - 열 수: {len(self.df.columns)}")
        print(f"   - 메모리: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return self.df
    
    def check_data_quality(self):
        """데이터 품질 체크"""
        print("\n🔍 데이터 품질 체크")
        print("=" * 60)
        
        # 1. 결측치 확인
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print("\n⚠️  결측치 발견:")
            print(missing[missing > 0].head(10))
        else:
            print("✅ 결측치 없음")
        
        # 2. 중복 데이터 확인
        duplicates = self.df.duplicated(subset=['사용월', '호선명', '지하철역']).sum()
        print(f"\n중복 행: {duplicates}개")
        
        # 3. 사용월 범위 확인
        print(f"\n사용월 범위: {self.df['사용월'].min()} ~ {self.df['사용월'].max()}")
        
        # 4. 노선 및 역 개수
        print(f"\n호선 수: {self.df['호선명'].nunique()}개")
        print(f"역 수: {self.df['지하철역'].nunique()}개")
        
        return self.df
    
    def wide_to_long(self):
        """와이드 포맷 → 롱 포맷 변환 (시간대별)"""
        print("\n🔄 데이터 형태 변환 중 (Wide → Long)")
        
        # 시간대 승차 컬럼 추출
        boarding_cols = [col for col in self.df.columns if '승차인원' in col]
        alighting_cols = [col for col in self.df.columns if '하차인원' in col]
        
        print(f"   - 시간대 수: {len(boarding_cols)}개")
        
        # 승차 데이터 롱 포맷 변환
        df_boarding = pd.melt(
            self.df,
            id_vars=['사용월', '호선명', '지하철역'],
            value_vars=boarding_cols,
            var_name='시간대',
            value_name='승차인원'
        )
        
        # 하차 데이터 롱 포맷 변환
        df_alighting = pd.melt(
            self.df,
            id_vars=['사용월', '호선명', '지하철역'],
            value_vars=alighting_cols,
            var_name='시간대',
            value_name='하차인원'
        )
        
        # 시간대에서 시간 추출 (04시-05시 → 04)
        df_boarding['시간'] = df_boarding['시간대'].str.extract(r'(\d+)시-').astype(int)
        df_alighting['시간'] = df_alighting['시간대'].str.extract(r'(\d+)시-').astype(int)
        
        # 승차/하차 데이터 병합
        self.df_long = df_boarding.merge(
            df_alighting[['사용월', '호선명', '지하철역', '시간', '하차인원']],
            on=['사용월', '호선명', '지하철역', '시간'],
            how='left'
        )
        
        # 불필요한 컬럼 제거
        self.df_long = self.df_long.drop('시간대', axis=1)
        
        # 총 승하차 인원 계산
        self.df_long['총승하차인원'] = self.df_long['승차인원'] + self.df_long['하차인원']
        
        print(f"✅ 변환 완료!")
        print(f"   - 변환 후 행 수: {len(self.df_long):,}")
        
        return self.df_long
    
    def add_time_features(self):
        """날짜/시간 관련 피처 추가"""
        print("\n📅 시간 피처 생성 중...")
        
        # 사용월을 날짜로 변환 (202510 → 2025-10-01)
        self.df_long['사용일자'] = pd.to_datetime(self.df_long['사용월'].astype(str) + '01', format='%Y%m%d')
        
        # 날짜 정보 추출
        self.df_long['연도'] = self.df_long['사용일자'].dt.year
        self.df_long['월'] = self.df_long['사용일자'].dt.month
        
        # 시간대 구분 (출퇴근 시간, 심야 등)
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
        
        # 혼잡도 레벨 분류 (4단계)
        def classify_congestion(passengers):
            if passengers < 500:
                return '여유'
            elif passengers < 1500:
                return '보통'
            elif passengers < 3000:
                return '혼잡'
            else:
                return '매우혼잡'
        
        self.df_long['혼잡도'] = self.df_long['총승하차인원'].apply(classify_congestion)
        
        # 혼잡도 레벨 (숫자)
        congestion_map = {'여유': 0, '보통': 1, '혼잡': 2, '매우혼잡': 3}
        self.df_long['혼잡도레벨'] = self.df_long['혼잡도'].map(congestion_map)
        
        print("✅ 시간 피처 추가 완료!")
        print(f"   - 추가된 피처: 사용일자, 연도, 월, 시간대구분, 혼잡도, 혼잡도레벨")
        
        return self.df_long
    
    def handle_missing_values(self):
        """결측치 처리"""
        print("\n🔧 결측치 처리 중...")
        
        before_count = len(self.df_long)
        
        # NaN을 0으로 대체 (승하차 인원이 없는 경우)
        self.df_long['승차인원'] = self.df_long['승차인원'].fillna(0)
        self.df_long['하차인원'] = self.df_long['하차인원'].fillna(0)
        self.df_long['총승하차인원'] = self.df_long['총승하차인원'].fillna(0)
        
        after_count = len(self.df_long)
        
        print(f"✅ 결측치 처리 완료!")
        print(f"   - 처리 전: {before_count:,}개")
        print(f"   - 처리 후: {after_count:,}개")
        
        return self.df_long
    
    def handle_outliers(self, method='iqr'):
        """이상치 처리"""
        print(f"\n🔧 이상치 처리 중 (방법: {method})...")
        
        for col in ['승차인원', '하차인원', '총승하차인원']:
            Q1 = self.df_long[col].quantile(0.25)
            Q3 = self.df_long[col].quantile(0.75)
            IQR = Q3 - Q1
            
            upper_bound = Q3 + 1.5 * IQR
            
            # 음수 값은 0으로 처리
            self.df_long[col] = self.df_long[col].clip(lower=0)
            
            # 극단적인 이상치만 처리 (상한값의 2배 이상)
            outliers = (self.df_long[col] > upper_bound * 2)
            outlier_count = outliers.sum()
            
            # 이상치는 상한값으로 대체
            self.df_long.loc[outliers, col] = upper_bound
            
            if outlier_count > 0:
                print(f"   - {col}: {outlier_count:,}개 처리")
        
        print(f"✅ 이상치 처리 완료!")
        
        return self.df_long
    
    def add_station_statistics(self):
        """역별 통계 정보 추가 (평균 혼잡도 등)"""
        print("\n📊 역별 통계 생성 중...")
        
        # 역별 평균 승하차 인원
        station_avg = self.df_long.groupby('지하철역')['총승하차인원'].mean().reset_index()
        station_avg.columns = ['지하철역', '역_평균승하차']
        
        # 시간대별 평균 승하차 인원
        time_avg = self.df_long.groupby('시간')['총승하차인원'].mean().reset_index()
        time_avg.columns = ['시간', '시간_평균승하차']
        
        # 호선별 평균 승하차 인원
        line_avg = self.df_long.groupby('호선명')['총승하차인원'].mean().reset_index()
        line_avg.columns = ['호선명', '호선_평균승하차']
        
        # 원본 데이터에 병합
        self.df_long = self.df_long.merge(station_avg, on='지하철역', how='left')
        self.df_long = self.df_long.merge(time_avg, on='시간', how='left')
        self.df_long = self.df_long.merge(line_avg, on='호선명', how='left')
        
        print("✅ 역별 통계 추가 완료!")
        
        return self.df_long
    
    def normalize_names(self):
        """역명/호선명 표준화 (공백, 특수문자 제거)"""
        print("\n🔤 역명/호선명 표준화 중...")
        
        # 공백 제거
        self.df_long['지하철역'] = self.df_long['지하철역'].str.strip()
        self.df_long['호선명'] = self.df_long['호선명'].str.strip()
        
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
        print(f"호선 수: {self.df_long['호선명'].nunique()}개")
        print(f"역 수: {self.df_long['지하철역'].nunique()}개")
        print(f"시간대 수: {self.df_long['시간'].nunique()}개")
        
        print("\n총승하차 인원 통계:")
        print(self.df_long['총승하차인원'].describe())
        
        print("\n시간대구분별 평균 승하차:")
        print(self.df_long.groupby('시간대구분')['총승하차인원'].mean().sort_values(ascending=False))
        
        print("\n혼잡도 분포:")
        print(self.df_long['혼잡도'].value_counts().sort_index())
        
        print("\n상위 10개 혼잡 역 (평균):")
        top_stations = self.df_long.groupby('지하철역')['총승하차인원'].mean().sort_values(ascending=False).head(10)
        print(top_stations)
        
        print("\n컬럼 목록:")
        print(self.df_long.columns.tolist())


def main():
    """메인 실행 함수"""
    
    print("🚇 지하철 혼잡도 예측 - 데이터 전처리 시작 (시간대별)")
    print("="*60)
    
    # 경로 설정
    RAW_DATA_PATH = 'data/raw/subway'
    PROCESSED_DATA_PATH = 'data/processed'
    
    # 전처리 객체 생성
    preprocessor = SubwayDataPreprocessor(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    
    try:
        # 1. 데이터 로드
        preprocessor.load_data()
        
        # 2. 데이터 품질 체크
        preprocessor.check_data_quality()
        
        # 3. 와이드 → 롱 포맷 변환
        preprocessor.wide_to_long()
        
        # 4. 결측치 처리
        preprocessor.handle_missing_values()
        
        # 5. 시간 피처 추가
        preprocessor.add_time_features()
        
        # 6. 역명/호선명 표준화
        preprocessor.normalize_names()
        
        # 7. 이상치 처리
        preprocessor.handle_outliers()
        
        # 8. 역별 통계 추가
        preprocessor.add_station_statistics()
        
        # 9. 저장
        preprocessor.save_processed_data()
        
        # 10. 요약
        preprocessor.get_summary()
        
        print("\n✅ 전처리 완료!")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()