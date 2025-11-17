"""
데이터 수집 모듈 (수정 버전)
서울시 일반 키 + 실시간 키 분리
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import time

load_dotenv()

class SubwayDataCollector:
    """서울시 지하철 승하차 데이터 수집 (일반 인증키 사용)"""
    
    def __init__(self, api_key=None):
        # 일반 인증키 사용
        self.api_key = api_key or os.getenv('SEOUL_GENERAL_API_KEY')
        self.base_url = "http://openapi.seoul.go.kr:8088"
    
    def get_ridership_data(self, start_date, end_date, save_dir='data/raw/subway'):
        """
        승하차 인원 데이터 수집
        
        Parameters:
        start_date: YYYYMMDD
        end_date: YYYYMMDD
        save_dir: 저장 디렉토리
        """
        if not self.api_key:
            print("❌ SEOUL_GENERAL_API_KEY가 .env 파일에 설정되지 않았습니다.")
            print("서울 열린데이터광장에서 '일반 인증키'를 발급받으세요.")
            return None
        
        print(f"\n📡 지하철 승하차 데이터 수집 중...")
        print(f"기간: {start_date} ~ {end_date}")
        print(f"사용 키: 일반 인증키")
        
        # API 엔드포인트
        # 형식: /인증키/json/서비스명/시작위치/종료위치/날짜
        url = f"{self.base_url}/{self.api_key}/json/CardSubwayStatsNew/1/1000/{start_date}"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # 응답 확인
            if 'CardSubwayStatsNew' in data:
                result = data['CardSubwayStatsNew']
                
                # 에러 체크
                if 'RESULT' in result:
                    code = result['RESULT'].get('CODE')
                    message = result['RESULT'].get('MESSAGE')
                    
                    if code != 'INFO-000':
                        print(f"❌ API 오류: {code} - {message}")
                        return None
                
                # 데이터 추출
                if 'row' in result:
                    records = result['row']
                    df = pd.DataFrame(records)
                    
                    # 저장
                    os.makedirs(save_dir, exist_ok=True)
                    filename = f"{save_dir}/subway_{start_date}_{end_date}.csv"
                    df.to_csv(filename, index=False, encoding='utf-8-sig')
                    
                    print(f"✓ 수집 완료: {len(df)}건")
                    print(f"✓ 저장: {filename}")
                    return df
                else:
                    print("❌ 데이터가 없습니다.")
                    return None
            else:
                print(f"❌ 예상과 다른 응답: {data}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ API 요청 실패: {e}")
            return None
        except Exception as e:
            print(f"❌ 데이터 수집 실패: {e}")
            return None

class RealtimeArrivalCollector:
    """서울시 실시간 도착정보 수집 (실시간 지하철 인증키 사용)"""
    
    def __init__(self, api_key=None):
        # 실시간 지하철 인증키 사용
        self.api_key = api_key or os.getenv('SEOUL_REALTIME_API_KEY')
        self.base_url = "http://swopenapi.seoul.go.kr/api/subway"
    
    def get_arrival_info(self, station_name):
        """
        실시간 열차 도착정보 조회
        
        Parameters:
        station_name: 역명 (예: "강남")
        
        Returns:
        list: 도착정보 리스트
        """
        if not self.api_key:
            print("❌ SEOUL_REALTIME_API_KEY가 .env 파일에 설정되지 않았습니다.")
            print("서울 열린데이터광장에서 '실시간 지하철 인증키'를 발급받으세요.")
            return None
        
        # API 엔드포인트
        # 형식: /인증키/json/서비스명/시작위치/종료위치/역명
        url = f"{self.base_url}/{self.api_key}/json/realtimeStationArrival/0/10/{station_name}"
        
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            
            # 에러 체크
            if 'errorMessage' in data:
                error = data['errorMessage']
                print(f"❌ API 오류: {error.get('status')} - {error.get('message')}")
                return None
            
            # 데이터 추출
            if 'realtimeArrivalList' in data:
                arrivals = []
                for item in data['realtimeArrivalList']:
                    arrivals.append({
                        'line': item.get('subwayId', ''),           # 호선 ID
                        'station': item.get('statnNm', ''),         # 역명
                        'updnLine': item.get('updnLine', ''),       # 상행/하행
                        'trainLineNm': item.get('trainLineNm', ''), # 행선지
                        'arvlMsg2': item.get('arvlMsg2', ''),       # 도착 메시지
                        'arvlMsg3': item.get('arvlMsg3', ''),       # 현재 위치
                        'btrainSttus': item.get('btrainSttus', ''), # 급행/일반
                        'arvlCd': item.get('arvlCd', '')            # 도착코드
                    })
                
                print(f"✓ 도착정보 조회 완료: {len(arrivals)}건")
                return arrivals
            else:
                print("❌ 도착정보가 없습니다.")
                return None
            
        except requests.exceptions.RequestException as e:
            print(f"❌ API 요청 실패: {e}")
            return None
        except Exception as e:
            print(f"❌ 도착정보 조회 실패: {e}")
            return None

class WeatherDataCollector:
    """기상청 날씨 데이터 수집"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('WEATHER_API_KEY')
        self.base_url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
    
    def get_weather_data(self, date, nx=60, ny=127, save_dir='data/raw/weather'):
        """
        기상청 단기예보 조회
        
        Parameters:
        date: YYYYMMDD
        nx, ny: 격자 좌표 (서울: 60, 127)
        save_dir: 저장 디렉토리
        """
        if not self.api_key:
            print("❌ WEATHER_API_KEY가 .env 파일에 설정되지 않았습니다.")
            print("공공데이터포털에서 '기상청 단기예보' API를 신청하세요.")
            return None
        
        print(f"\n🌤️  날씨 데이터 수집 중...")
        print(f"날짜: {date}, 좌표: ({nx}, {ny})")
        
        params = {
            'serviceKey': self.api_key,
            'pageNo': '1',
            'numOfRows': '100',
            'dataType': 'JSON',
            'base_date': date,
            'base_time': '0500',
            'nx': nx,
            'ny': ny
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # 응답 확인
            if 'response' in data:
                header = data['response']['header']
                result_code = header.get('resultCode')
                result_msg = header.get('resultMsg')
                
                if result_code == '00':
                    body = data['response']['body']
                    
                    if 'items' in body and 'item' in body['items']:
                        items = body['items']['item']
                        df = pd.DataFrame(items)
                        
                        # 저장
                        os.makedirs(save_dir, exist_ok=True)
                        filename = f"{save_dir}/weather_{date}.csv"
                        df.to_csv(filename, index=False, encoding='utf-8-sig')
                        
                        print(f"✓ 수집 완료: {len(df)}건")
                        print(f"✓ 저장: {filename}")
                        return df
                    else:
                        print("❌ 날씨 데이터가 없습니다.")
                        return None
                else:
                    print(f"❌ API 오류: {result_code} - {result_msg}")
                    return None
            else:
                print(f"❌ 예상과 다른 응답: {data}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ API 요청 실패: {e}")
            return None
        except Exception as e:
            print(f"❌ 날씨 데이터 수집 실패: {e}")
            return None

class HolidayCollector:
    """공휴일 데이터 관리"""
    
    def __init__(self):
        self.holidays_2025 = [
            '20250101',  # 신정
            '20250128', '20250129', '20250130',  # 설날 연휴
            '20250301',  # 삼일절
            '20250505',  # 어린이날
            '20250506',  # 대체공휴일
            '20250815',  # 광복절
            '20251003',  # 개천절
            '20251009',  # 한글날
            '20251225',  # 크리스마스
        ]
    
    def get_holidays(self, year=2025, save_dir='data/external'):
        """
        공휴일 목록 저장
        
        Parameters:
        year: 연도
        save_dir: 저장 디렉토리
        """
        print(f"\n📅 공휴일 데이터 생성 중...")
        
        df = pd.DataFrame({
            'date': self.holidays_2025,
            'is_holiday': 1,
            'holiday_name': [
                '신정',
                '설날 연휴', '설날', '설날 연휴',
                '삼일절',
                '어린이날', '대체공휴일',
                '광복절',
                '개천절',
                '한글날',
                '크리스마스'
            ]
        })
        
        # 저장
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{save_dir}/holidays_{year}.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        
        print(f"✓ 생성 완료: {len(df)}건")
        print(f"✓ 저장: {filename}")
        return df
    
    def is_holiday(self, date_str):
        """특정 날짜가 공휴일인지 확인"""
        return date_str in self.holidays_2025

# ========================================
# 테스트 코드
# ========================================

if __name__ == "__main__":
    print("="*70)
    print("🔍 데이터 수집 모듈 테스트")
    print("="*70)
    print()
    
    # 날짜 설정
    today = datetime.now().strftime("%Y%m%d")
    
    # 1. 지하철 승하차 데이터 (일반 인증키)
    print("1️⃣  지하철 승하차 데이터 테스트")
    print("-"*70)
    subway = SubwayDataCollector()
    subway_df = subway.get_ridership_data(today, today)
    
    # 2. 실시간 도착정보 (실시간 지하철 인증키)
    print("\n2️⃣  실시간 도착정보 테스트")
    print("-"*70)
    arrival = RealtimeArrivalCollector()
    arrival_info = arrival.get_arrival_info("강남")
    
    if arrival_info:
        print(f"\n📊 강남역 도착정보:")
        for i, train in enumerate(arrival_info[:3], 1):
            print(f"  {i}. {train['updnLine']} - {train['arvlMsg2']}")
    
    # 3. 날씨 데이터 (기상청 인증키)
    print("\n3️⃣  날씨 데이터 테스트")
    print("-"*70)
    weather = WeatherDataCollector()
    weather_df = weather.get_weather_data(today)
    
    # 4. 공휴일 데이터
    print("\n4️⃣  공휴일 데이터 테스트")
    print("-"*70)
    holiday = HolidayCollector()
    holiday_df = holiday.get_holidays()
    
    # 결과 요약
    print("\n" + "="*70)
    print("📊 테스트 결과 요약")
    print("="*70)
    print(f"지하철 데이터: {'✅ 성공' if subway_df is not None else '❌ 실패'}")
    print(f"도착정보: {'✅ 성공' if arrival_info is not None else '❌ 실패'}")
    print(f"날씨 데이터: {'✅ 성공' if weather_df is not None else '❌ 실패'}")
    print(f"공휴일 데이터: {'✅ 성공' if holiday_df is not None else '❌ 실패'}")
    print("="*70)
    
    # API 키 상태 확인
    print("\n🔑 API 키 설정 상태:")
    print(f"  SEOUL_GENERAL_API_KEY: {'✅ 설정됨' if os.getenv('SEOUL_GENERAL_API_KEY') else '❌ 없음'}")
    print(f"  SEOUL_REALTIME_API_KEY: {'✅ 설정됨' if os.getenv('SEOUL_REALTIME_API_KEY') else '❌ 없음'}")
    print(f"  WEATHER_API_KEY: {'✅ 설정됨' if os.getenv('WEATHER_API_KEY') else '❌ 없음'}")
    print()
