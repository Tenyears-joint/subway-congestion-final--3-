"""
데이터 수집 모듈 (수정 버전 - URL 인코딩 문제 해결)
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import time
from urllib.parse import unquote

from subway_congestion_analyzer import SubwayCongestionAnalyzer

load_dotenv()
class SubwayDataCollector:
    """서울시 지하철 승하차 데이터 수집 (일반 인증키 사용)"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('SEOUL_GENERAL_API_KEY')
        self.base_url = "http://openapi.seoul.go.kr:8088"
    
    def get_ridership_data(self, start_date, end_date, save_dir='data/raw/subway'):
        """
        승하차 인원 데이터 수집
        """
        if not self.api_key:
            print("❌ SEOUL_GENERAL_API_KEY가 설정되지 않았습니다.")
            return None
        
        print(f"\n📡 지하철 승하차 데이터 수집 중...")
        print(f"기간: {start_date} ~ {end_date}")
        
        # API 엔드포인트
        url = f"{self.base_url}/{self.api_key}/json/CardSubwayStatsNew/1/1000/{start_date}"
        
        try:
            response = requests.get(url, timeout=10)
            response.encoding = 'utf-8'
            response.raise_for_status()
            data = response.json()
            
            if 'CardSubwayStatsNew' in data:
                result = data['CardSubwayStatsNew']
                
                if 'RESULT' in result:
                    code = result['RESULT'].get('CODE')
                    message = result['RESULT'].get('MESSAGE')
                    
                    if code == 'INFO-200':
                        print(f"⚠️  {message}")
                        print(f"💡 팁: 지하철 데이터는 보통 전날 또는 전월 데이터만 제공됩니다.")
                        print(f"      이전 날짜로 다시 시도해보세요.")
                        return None
                    elif code != 'INFO-000':
                        print(f"❌ API 오류: {code} - {message}")
                        return None
                
                if 'row' in result:
                    records = result['row']
                    df = pd.DataFrame(records)
                    
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
                
        except Exception as e:
            print(f"❌ 데이터 수집 실패: {e}")
            return None

class RealtimeArrivalCollector:
    """서울시 실시간 도착정보 수집"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('SEOUL_REALTIME_API_KEY')
        self.base_url = "http://swopenapi.seoul.go.kr/api/subway"
    
    def get_arrival_info(self, station_name):
        """실시간 열차 도착정보 조회"""
        if not self.api_key:
            print("❌ SEOUL_REALTIME_API_KEY가 설정되지 않았습니다.")
            return None
        
        url = f"{self.base_url}/{self.api_key}/json/realtimeStationArrival/0/10/{station_name}"
        
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            message = data['errorMessage']
            status_code = message['status'] 

            if status_code != 200:
                error = data['errorMessage']
                print(f"❌ API 오류: {error.get('status')} - {error.get('message')}")
                return None
            
            if 'realtimeArrivalList' in data:
                arrivals = []
                for item in data['realtimeArrivalList']:
                    arrivals.append({
                        'line': item.get('subwayId', ''),
                        'station': item.get('statnNm', ''),
                        'updnLine': item.get('updnLine', ''),
                        'trainLineNm': item.get('trainLineNm', ''),
                        'arvlMsg2': item.get('arvlMsg2', ''),
                        'arvlMsg3': item.get('arvlMsg3', ''),
                        'btrainSttus': item.get('btrainSttus', ''),
                        'arvlCd': item.get('arvlCd', '')
                    })
                
                if len(arrivals) == 0:
                    print(f"⚠️  {station_name}역에 도착 예정인 열차가 없습니다.")
                    print(f"💡 팁: 운행 시간이 아니거나 심야 시간일 수 있습니다.")
                else:
                    print(f"✓ 도착정보 조회 완료: {len(arrivals)}건")
                return arrivals
            else:
                print("❌ 도착정보가 없습니다.")
                return None
            
        except Exception as e:
            print(f"❌ 도착정보 조회 실패: {e}")
            return None

class WeatherDataCollector:
    """기상청 날씨 데이터 수집"""
    
    def __init__(self, api_key=None):
        # API 키 URL 디코딩 (중요!)
        raw_key = api_key or os.getenv('WEATHER_API_KEY')
        if raw_key:
            # URL 인코딩이 되어있다면 디코딩
            self.api_key = unquote(raw_key)
        else:
            self.api_key = None
        
        self.base_url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
    
    def get_weather_data(self, date, nx=60, ny=127, save_dir='data/raw/weather'):
        """기상청 단기예보 조회"""
        if not self.api_key:
            print("❌ WEATHER_API_KEY가 설정되지 않았습니다.")
            return None
        
        print(f"\n🌤️  날씨 데이터 수집 중...")
        print(f"날짜: {date}, 좌표: ({nx}, {ny})")
        
        params = {
            'serviceKey': self.api_key,  # 디코딩된 키 사용
            'pageNo': '1',
            'numOfRows': '100',
            'dataType': 'JSON',
            'base_date': date,
            'base_time': '0500',
            'nx': nx,
            'ny': ny
        }
        
        try:
            # requests가 자동으로 URL 인코딩하므로 디코딩된 키를 넘겨야 함
            response = requests.get(self.base_url, params=params, timeout=10)
            
            # 401 오류인 경우 상세 정보 출력
            if response.status_code == 401:
                print(f"❌ 인증 오류 (401)")
                print(f"💡 API 키를 확인하세요:")
                print(f"   - 공공데이터포털에서 승인되었는지 확인")
                print(f"   - .env 파일의 키가 정확한지 확인")
                print(f"   - URL 인코딩(%2F, %3D 등)이 있다면 제거")
                return None
            
            response.raise_for_status()
            data = response.json()
            
            if 'response' in data:
                header = data['response']['header']
                result_code = header.get('resultCode')
                result_msg = header.get('resultMsg')
                
                if result_code == '00':
                    body = data['response']['body']
                    
                    if 'items' in body and 'item' in body['items']:
                        items = body['items']['item']
                        df = pd.DataFrame(items)
                        
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
                print(f"❌ 예상과 다른 응답")
                return None
                
        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTP 오류: {e}")
            return None
        except Exception as e:
            print(f"❌ 날씨 데이터 수집 실패: {e}")
            return None

class HolidayCollector:
    """공휴일 데이터 관리"""
    
    def __init__(self):
        self.holidays_2025 = [
            '20250101', '20250128', '20250129', '20250130',
            '20250301', '20250505', '20250506', '20250815',
            '20251003', '20251009', '20251225',
        ]
    
    def get_holidays(self, year=2025, save_dir='data/external'):
        """공휴일 목록 저장"""
        print(f"\n📅 공휴일 데이터 생성 중...")
        
        df = pd.DataFrame({
            'date': self.holidays_2025,
            'is_holiday': 1,
            'holiday_name': [
                '신정', '설날 연휴', '설날', '설날 연휴',
                '삼일절', '어린이날', '대체공휴일', '광복절',
                '개천절', '한글날', '크리스마스'
            ]
        })
        
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{save_dir}/holidays_{year}.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        
        print(f"✓ 생성 완료: {len(df)}건")
        print(f"✓ 저장: {filename}")
        return df
    
    def is_holiday(self, date_str):
        return date_str in self.holidays_2025

if __name__ == "__main__":
    print("="*70)
    print("🔍 데이터 수집 모듈 테스트")
    print("="*70)
    

    analyzer = SubwayCongestionAnalyzer(os.getenv('SEOUL_HOUR_API_KEY'))

    # 예시 1: 특정 역의 전체 시간대 혼잡도 분석
    print("\n[예시 1] 강남역 전체 시간대 혼잡도")
    analyzer.print_congestion_report(
        date="202510",
        line="2호선",
        station="당산"
    )

    # 1. 지하철 승하차 데이터
    print("\n1️⃣  지하철 승하차 데이터 테스트")
    print("-"*70)
    subway = SubwayDataCollector()
    
    # 이번년도 데이터 (오늘 데이터는 없을 가능성 높음)
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    print(f"💡 2025년 데이터를 불러옵니다.")
    subway_df = subway.get_ridership_data('20250101', '20251101')
    
    # 2. 실시간 도착정보
    print("\n2️⃣  실시간 도착정보 테스트")
    print("-"*70)
    arrival = RealtimeArrivalCollector()
    arrival_info = arrival.get_arrival_info("강남")
    
    if arrival_info and len(arrival_info) > 0:
        print(f"\n📊 강남역 도착정보:")
        for i, train in enumerate(arrival_info[:3], 1):
            print(f"  {i}. {train['updnLine']} - {train['arvlMsg2']}")
    
    # 3. 날씨 데이터
    print("\n3️⃣  날씨 데이터 테스트")
    print("-"*70)
    weather = WeatherDataCollector()
    today = datetime.now().strftime("%Y%m%d")
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
    print(f"도착정보: {'✅ 성공' if arrival_info and len(arrival_info) > 0 else '❌ 실패'}")
    print(f"날씨 데이터: {'✅ 성공' if weather_df is not None else '❌ 실패'}")
    print(f"공휴일 데이터: {'✅ 성공' if holiday_df is not None else '❌ 실패'}")
    print("="*70)
    
    # API 키 상태
    print("\n🔑 API 키 설정 상태:")
    print(f"  SEOUL_GENERAL_API_KEY: {'✅ 설정됨' if os.getenv('SEOUL_GENERAL_API_KEY') else '❌ 없음'}")
    print(f"  SEOUL_REALTIME_API_KEY: {'✅ 설정됨' if os.getenv('SEOUL_REALTIME_API_KEY') else '❌ 없음'}")
    
    # 기상청 API 키 디코딩 상태 확인
    raw_weather_key = os.getenv('WEATHER_API_KEY')
    if raw_weather_key:
        decoded_key = unquote(raw_weather_key)
        is_encoded = raw_weather_key != decoded_key
        print(f"  WEATHER_API_KEY: ✅ 설정됨 {'(URL 인코딩됨 - 자동 디코딩 처리됨)' if is_encoded else ''}")
    else:
        print(f"  WEATHER_API_KEY: ❌ 없음")
    print()
