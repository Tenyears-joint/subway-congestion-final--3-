"""
서울시 API 테스트 스크립트
"""

import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

def test_subway_api():
    """지하철 승하차 데이터 여러 날짜로 테스트"""
    api_key = os.getenv('SEOUL_GENERAL_API_KEY')
    base_url = "http://openapi.seoul.go.kr:8088"
    
    print("="*70)
    print("🚇 지하철 승하차 데이터 테스트")
    print("="*70)
    
    # 테스트할 날짜들
    test_dates = [
        (datetime.now() - timedelta(days=1)).strftime("%Y%m%d"),   # 어제
        (datetime.now() - timedelta(days=7)).strftime("%Y%m%d"),   # 일주일 전
        (datetime.now() - timedelta(days=30)).strftime("%Y%m%d"),  # 한달 전
        "20241001",  # 10월 1일
        "20240901",  # 9월 1일
    ]
    
    for date in test_dates:
        print(f"\n📅 날짜: {date} 테스트 중...")
        url = f"{base_url}/{api_key}/json/CardSubwayStatsNew/1/10/{date}"
        
        try:
            response = requests.get(url, timeout=5)
            data = response.json()
            
            if 'CardSubwayStatsNew' in data:
                result = data['CardSubwayStatsNew']
                
                if 'row' in result:
                    print(f"   ✅ 성공! {len(result['row'])}건의 데이터")
                    return date, result['row'][:3]  # 처음 3개 반환
                elif 'RESULT' in result:
                    code = result['RESULT'].get('CODE')
                    msg = result['RESULT'].get('MESSAGE')
                    print(f"   ⚠️  {code}: {msg}")
        except Exception as e:
            print(f"   ❌ 오류: {e}")
    
    print("\n💡 모든 날짜에서 데이터를 찾을 수 없습니다.")
    print("   이 API는 월별 통계 데이터일 수 있습니다.")
    return None, None

def test_realtime_api():
    """실시간 도착정보 여러 방법으로 테스트"""
    api_key = os.getenv('SEOUL_REALTIME_API_KEY')
    
    print("\n" + "="*70)
    print("🚊 실시간 도착정보 테스트")
    print("="*70)
    
    # 테스트할 엔드포인트들
    endpoints = [
        ("http://swopenapi.seoul.go.kr/api/subway", "기본 엔드포인트"),
        ("http://openapi.seoul.go.kr:8088", "통합 엔드포인트"),
    ]
    
    stations = ["강남", "서울역", "홍대입구"]
    
    for base_url, desc in endpoints:
        print(f"\n🔗 {desc}: {base_url}")
        
        for station in stations:
            print(f"\n   역: {station}")
            url = f"{base_url}/{api_key}/json/realtimeStationArrival/0/5/{station}"
            
            try:
                response = requests.get(url, timeout=5)
                print(f"   상태 코드: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # 다양한 응답 형식 체크
                    if 'realtimeArrivalList' in data:
                        arrivals = data['realtimeArrivalList']
                        if len(arrivals) > 0:
                            print(f"   ✅ 성공! {len(arrivals)}건의 도착정보")
                            print(f"      첫 번째: {arrivals[0].get('arvlMsg2', 'N/A')}")
                            return True
                        else:
                            print(f"   ⚠️  응답은 왔지만 도착정보가 비어있음")
                    elif 'errorMessage' in data:
                        error = data['errorMessage']
                        print(f"   ❌ API 오류: {error}")
                    else:
                        print(f"   ⚠️  예상과 다른 응답 구조")
                        print(f"   응답 키: {list(data.keys())}")
                else:
                    print(f"   ❌ HTTP 오류: {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ 요청 실패: {e}")
    
    return False

def test_api_key_format():
    """API 키 형식 확인"""
    print("\n" + "="*70)
    print("🔑 API 키 형식 확인")
    print("="*70)
    
    general_key = os.getenv('SEOUL_GENERAL_API_KEY')
    realtime_key = os.getenv('SEOUL_REALTIME_API_KEY')
    
    print(f"\n일반 인증키:")
    print(f"  길이: {len(general_key) if general_key else 0}")
    print(f"  형식: {general_key[:10]}...{general_key[-10:] if general_key and len(general_key) > 20 else ''}")
    
    print(f"\n실시간 인증키:")
    print(f"  길이: {len(realtime_key) if realtime_key else 0}")
    print(f"  형식: {realtime_key[:10]}...{realtime_key[-10:] if realtime_key and len(realtime_key) > 20 else ''}")
    
    # 16진수 여부 확인
    if general_key:
        is_hex = all(c in '0123456789abcdefABCDEF' for c in general_key)
        print(f"  16진수 형식: {'✅ 예' if is_hex else '❌ 아니오'}")
    
    if realtime_key:
        is_hex = all(c in '0123456789abcdefABCDEF' for c in realtime_key)
        print(f"  16진수 형식: {'✅ 예' if is_hex else '❌ 아니오'}")

if __name__ == "__main__":
    print("\n🔍 서울시 API 상세 진단")
    print("="*70)
    
    # 1. API 키 형식 확인
    test_api_key_format()
    
    # 2. 지하철 승하차 데이터 테스트
    success_date, sample_data = test_subway_api()
    
    if success_date and sample_data:
        print(f"\n📊 샘플 데이터 (날짜: {success_date}):")
        for i, record in enumerate(sample_data, 1):
            print(f"  {i}. {record.get('SUB_STA_NM', 'N/A')} - 승차: {record.get('RIDE_PASGR_NUM', 'N/A')}명")
    
    # 3. 실시간 도착정보 테스트
    realtime_success = test_realtime_api()
    
    # 최종 결과
    print("\n" + "="*70)
    print("📊 최종 진단 결과")
    print("="*70)
    print(f"지하철 승하차 데이터: {'✅ 작동' if success_date else '❌ 문제 있음'}")
    print(f"실시간 도착정보: {'✅ 작동' if realtime_success else '❌ 문제 있음'}")
    print("="*70)
    
    if not success_date:
        print("\n💡 지하철 데이터 해결책:")
        print("   1. 서울 열린데이터광장에서 API 문서 확인")
        print("   2. 다른 API 서비스 사용 (예: 지하철 혼잡도 API)")
        print("   3. 공공데이터포털의 다른 지하철 데이터 API 찾기")
    
    if not realtime_success:
        print("\n💡 실시간 도착정보 해결책:")
        print("   1. API 키를 재발급 받기")
        print("   2. 서울교통공사 API 사용 (대안)")
        print("   3. 지하철 앱의 공개 API 사용")
