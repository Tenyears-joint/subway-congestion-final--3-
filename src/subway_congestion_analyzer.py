# -*- coding: utf-8 -*-
import sys
import io
import requests
import json
from datetime import datetime

# Windows 환경 인코딩 설정
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

class SubwayCongestionAnalyzer:
    """지하철 역 혼잡도 분석기 (최대 이용객 대비 %)"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://openapi.seoul.go.kr:8088"
    
    def get_station_data(self, date, line, station):
        """
        API에서 특정 역의 데이터 가져오기
        
        Parameters:
        - date: 날짜 (YYYYMM 형식)
        - line: 호선명 (예: "2호선")
        - station: 역명 (예: "동대문")
        
        Returns:
        - API 응답 데이터
        """
        # URL 인코딩 처리
        from urllib.parse import quote
        
        encoded_line = quote(line)
        encoded_station = quote(station)
        
        url = f"{self.base_url}/{self.api_key}/json/CardSubwayTime/1/5/{date}/{encoded_line}/{encoded_station}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            # API 에러 체크
            if 'RESULT' in data:
                code = data['RESULT'].get('CODE', '')
                if code != 'INFO-200':
                    message = data['RESULT'].get('MESSAGE', '알 수 없는 오류')
                    print(f"API 에러 [{code}]: {message}")
                    return None
            
            return data
        
        except requests.exceptions.RequestException as e:
            print(f"API 요청 실패: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 실패: {e}")
            return None
    
    def calculate_hourly_congestion(self, year_month, line, station):
        """
        시간대별 혼잡도를 % 계산
        
        Returns:
        - dict: 시간대별 혼잡도 정보
        """
        # 1. 데이터 가져오기
        data = self.get_station_data(year_month, line, station)
        
        if not data or 'CardSubwayTime' not in data:
            print("데이터를 가져올 수 없습니다.")
            return None
        
        records = data['CardSubwayTime']['row']
        
        if not records:
            print("데이터가 비어있습니다.")
            return None
        
        record = records[0]  # 첫 번째 레코드 사용
        
        # 2. 시간대별 승하차 인원 추출
        hourly_passengers = {}
        
        for hour in range(24):  # 0시부터 23시까지
            hour_str = str(hour).zfill(2)
            
            # 승차 인원
            get_on_key = f'HR_{hour}_GET_ON_NOPE'
            # 하차 인원
            get_off_key = f'HR_{hour}_GET_OFF_NOPE'
            
            ride_on = float(record.get(get_on_key, 0))
            ride_off = float(record.get(get_off_key, 0))
            total_passengers = ride_on + ride_off
            
            if total_passengers > 0:  # 이용객이 있는 시간대만 저장
                hourly_passengers[hour_str] = {
                    'total': total_passengers,
                    'ride_on': ride_on,
                    'ride_off': ride_off
                }
        
        if not hourly_passengers:
            print("시간대별 데이터가 없습니다.")
            return None
        
        # 3. 최대 이용객 수 찾기 (피크 시간대)
        max_passengers = max(data['total'] for data in hourly_passengers.values())
        
        if max_passengers == 0:
            print("이용객 데이터가 0입니다.")
            return None
        
        # 4. 각 시간대별 혼잡도 % 계산
        congestion_result = {}
        
        for hour, data in hourly_passengers.items():
            passengers = data['total']
            congestion_percent = (passengers / max_passengers) * 100
            
            congestion_result[hour] = {
                'passengers': int(passengers),
                'ride_on': int(data['ride_on']),
                'ride_off': int(data['ride_off']),
                'congestion_percent': round(congestion_percent, 1),
                'level': self._get_congestion_level(congestion_percent),
                'is_peak': passengers == max_passengers
            }
        
        return {
            'station': station,
            'line': line,
            'year_month': year_month,
            'max_passengers': int(max_passengers),
            'hourly_data': congestion_result
        }
    
    def get_specific_hour_congestion(self, year_month, line, station, target_hour):
        """
        특정 시간대의 혼잡도만 조회
        
        Parameters:
        - target_hour: 조회할 시간 (예: "08", "18", 8, 18)
        
        Returns:
        - dict: 해당 시간대 혼잡도 정보
        """
        result = self.calculate_hourly_congestion(year_month, line, station)
        if not result:
            return None
        
        hourly_data = result['hourly_data']
        
        # 시간 형식 통일 (2자리)
        target_hour = str(target_hour).zfill(2)
        
        if target_hour not in hourly_data:
            print(f"{target_hour}시 데이터가 없습니다.")
            print(f"사용 가능한 시간대: {sorted(hourly_data.keys())}")
            return None
        
        return {
            'station': station,
            'line': line,
            'year_month': year_month,
            'hour': target_hour,
            'max_passengers': result['max_passengers'],
            **hourly_data[target_hour]
        }

    def _get_congestion_level(self, percent):
        """혼잡도 %에 따른 등급 반환"""
        if percent >= 90:
            return "매우혼잡"
        elif percent >= 70:
            return "혼잡"
        elif percent >= 40:
            return "보통"
        else:
            return "여유"
    
    def print_congestion_report(self, date, line, station):
        """혼잡도 리포트 출력 (전체 시간대)"""
        result = self.calculate_hourly_congestion(date, line, station)
        if not result:
            return
        
        print(f"\n{'='*60}")
        print(f"📊 {result['station']} ({result['line']}) 혼잡도 분석")
        print(f"📅 날짜: {result['year_month']}")
        print(f"👥 최대 이용객: {result['max_passengers']:,}명")
        print(f"{'='*60}\n")
        
        # 시간대별 정렬
        sorted_hours = sorted(result['hourly_data'].items(), 
                            key=lambda x: int(x[0]) if x[0].isdigit() else 0)
        
        for hour, data in sorted_hours:
            peak_mark = "🔥" if data['is_peak'] else "  "
            
            # 혼잡도 바 그래프
            bar_length = int(data['congestion_percent'] / 5)
            bar = "█" * bar_length
            
            print(f"{peak_mark} {hour}시 | {bar:<20} {data['congestion_percent']:>5.1f}% | "
                  f"{data['level']:<6} | {data['passengers']:>7,}명 "
                  f"(승차: {data['ride_on']:>6,} / 하차: {data['ride_off']:>6,})")
        
        print(f"\n{'='*60}")
    
    def print_specific_hour_report(self, date, line, station, target_hour):
        """특정 시간대 혼잡도 리포트 출력"""
        result = self.get_specific_hour_congestion(date, line, station, target_hour)
        
        if not result:
            return
        
        print(f"\n{'='*60}")
        print(f"📊 {result['station']} ({result['line']}) - {result['hour']}시 혼잡도")
        print(f"📅 날짜: {result['date']}")
        print(f"{'='*60}\n")
        
        # 혼잡도 바 그래프
        bar_length = int(result['congestion_percent'] / 5)
        bar = "█" * bar_length
        
        print(f"혼잡도: {bar} {result['congestion_percent']}%")
        print(f"등급: {result['level']}")
        print(f"\n이용객 수: {result['passengers']:,}명")
        print(f"  - 승차: {result['ride_on']:,}명")
        print(f"  - 하차: {result['ride_off']:,}명")
        print(f"\n최대 이용객 대비: {result['passengers']:,}명 / {result['max_passengers']:,}명")
        
        if result['is_peak']:
            print(f"\n🔥 이 시간대가 가장 혼잡합니다!")
        
        print(f"{'='*60}\n")