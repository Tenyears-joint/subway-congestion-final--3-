"""
지하철 혼잡도 예측 - FastAPI 백엔드 (강화된 피처 버전)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import os

# FastAPI 앱 생성
app = FastAPI(
    title="지하철 혼잡도 예측 API",
    description="서울 지하철 혼잡도를 예측하는 API (강화된 피처 버전)",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
model = None
feature_names = None
df_stations = None
df_features = None

# 역 카테고리 정의
STATION_CATEGORIES = {
    # 초대형 업무 + 상업 지구
    '강남': {'업무지구': 1, '상업지구': 1, '환승역등급': 2, '대학가': 0, '주거지역': 0, '역_그룹': 1},
    '역삼': {'업무지구': 1, '상업지구': 1, '환승역등급': 0, '대학가': 0, '주거지역': 0, '역_그룹': 1},
    '선릉': {'업무지구': 1, '상업지구': 0, '환승역등급': 1, '대학가': 0, '주거지역': 0, '역_그룹': 1},
    '삼성': {'업무지구': 1, '상업지구': 1, '환승역등급': 0, '대학가': 0, '주거지역': 0, '역_그룹': 1},
    '여의도': {'업무지구': 1, '상업지구': 0, '환승역등급': 2, '대학가': 0, '주거지역': 0, '역_그룹': 1},
    '광화문': {'업무지구': 1, '상업지구': 1, '환승역등급': 0, '대학가': 0, '주거지역': 0, '역_그룹': 1},
    '종각': {'업무지구': 1, '상업지구': 1, '환승역등급': 1, '대학가': 0, '주거지역': 0, '역_그룹': 1},
    
    # 상업/쇼핑 지구
    '홍대입구': {'업무지구': 0, '상업지구': 1, '환승역등급': 2, '대학가': 1, '주거지역': 0, '역_그룹': 2},
    '신촌': {'업무지구': 0, '상업지구': 1, '환승역등급': 0, '대학가': 1, '주거지역': 0, '역_그룹': 2},
    '명동': {'업무지구': 0, '상업지구': 1, '환승역등급': 0, '대학가': 0, '주거지역': 0, '역_그룹': 2},
    '동대문': {'업무지구': 0, '상업지구': 1, '환승역등급': 2, '대학가': 0, '주거지역': 0, '역_그룹': 2},
    '을지로입구': {'업무지구': 1, '상업지구': 1, '환승역등급': 1, '대학가': 0, '주거지역': 0, '역_그룹': 1},
    
    # 대학가
    '신림': {'업무지구': 0, '상업지구': 0, '환승역등급': 0, '대학가': 1, '주거지역': 1, '역_그룹': 3},
    '서울대입구': {'업무지구': 0, '상업지구': 0, '환승역등급': 0, '대학가': 1, '주거지역': 0, '역_그룹': 3},
    '이대': {'업무지구': 0, '상업지구': 1, '환승역등급': 0, '대학가': 1, '주거지역': 0, '역_그룹': 3},
    '건대입구': {'업무지구': 0, '상업지구': 1, '환승역등급': 1, '대학가': 1, '주거지역': 0, '역_그룹': 3},
    
    # 초대형 환승역
    '왕십리': {'업무지구': 0, '상업지구': 0, '환승역등급': 4, '대학가': 0, '주거지역': 0, '역_그룹': 4},
    '신도림': {'업무지구': 0, '상업지구': 1, '환승역등급': 3, '대학가': 0, '주거지역': 0, '역_그룹': 4},
    '사당': {'업무지구': 0, '상업지구': 0, '환승역등급': 3, '대학가': 0, '주거지역': 1, '역_그룹': 4},
    '잠실': {'업무지구': 0, '상업지구': 1, '환승역등급': 2, '대학가': 0, '주거지역': 1, '역_그룹': 4},
    '교대': {'업무지구': 0, '상업지구': 0, '환승역등급': 2, '대학가': 0, '주거지역': 0, '역_그룹': 4},
    '고속터미널': {'업무지구': 0, '상업지구': 1, '환승역등급': 3, '대학가': 0, '주거지역': 0, '역_그룹': 4},
    
    # 주거 지역
    '목동': {'업무지구': 0, '상업지구': 0, '환승역등급': 0, '대학가': 0, '주거지역': 1, '역_그룹': 5},
    '노원': {'업무지구': 0, '상업지구': 0, '환승역등급': 2, '대학가': 0, '주거지역': 1, '역_그룹': 5},
    '수유': {'업무지구': 0, '상업지구': 0, '환승역등급': 0, '대학가': 0, '주거지역': 1, '역_그룹': 5},
    '구로디지털단지': {'업무지구': 1, '상업지구': 0, '환승역등급': 1, '대학가': 0, '주거지역': 0, '역_그룹': 1},
    
    # 외곽 지역
    '까치산': {'업무지구': 0, '상업지구': 0, '환승역등급': 0, '대학가': 0, '주거지역': 1, '역_그룹': 6},
    '신정네거리': {'업무지구': 0, '상업지구': 0, '환승역등급': 0, '대학가': 0, '주거지역': 1, '역_그룹': 6},
}

# 기본값
DEFAULT_CATEGORY = {'업무지구': 0, '상업지구': 0, '환승역등급': 0, '대학가': 0, '주거지역': 0, '역_그룹': 0}

# 혼잡도 레벨 매핑
CONGESTION_LEVEL_MAP = {
    0: "여유",
    1: "보통",
    2: "혼잡",
    3: "매우혼잡"
}

CONGESTION_COLOR_MAP = {
    0: "#4CAF50",
    1: "#FFC107",
    2: "#FF9800",
    3: "#F44336"
}

# 요일 한글 매핑
DAY_NAME_MAP = {
    0: "월요일",
    1: "화요일",
    2: "수요일",
    3: "목요일",
    4: "금요일",
    5: "토요일",
    6: "일요일"
}


# ============================================
# Pydantic 모델 정의
# ============================================

class PredictionRequest(BaseModel):
    station_name: str = Field(..., description="지하철역 이름")
    line_name: str = Field(..., description="호선명")
    hour: int = Field(..., ge=0, le=23, description="시간 (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="요일 (0=월, 6=일)")
    month: Optional[int] = Field(None, ge=1, le=12, description="월")
    is_holiday: Optional[bool] = Field(False, description="공휴일 여부")


class PredictionResponse(BaseModel):
    success: bool
    prediction: Dict
    input: Dict
    recommendation: str
    timestamp: str


class StationInfo(BaseModel):
    name: str
    line: str
    is_transfer: Optional[bool] = False


# ============================================
# 모델 및 데이터 로드
# ============================================

@app.on_event("startup")
async def load_model():
    """서버 시작 시 모델 및 데이터 로드"""
    global model, feature_names, df_stations, df_features
    
    print("="*80)
    print("🚇 지하철 혼잡도 예측 API 서버 시작")
    print("="*80)
    
    try:
        # 1. 모델 로드
        model_path = 'models/subway_congestion_model_enhanced.pkl'
        
        if not os.path.exists(model_path):
            print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
            raise FileNotFoundError(f"모델 파일 없음: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            feature_names = model_data['feature_names']
        
        print(f"✅ 모델 로드 완료: {model_path}")
        print(f"   - 모델 타입: {model_data.get('model_type', 'Unknown')}")
        print(f"   - 학습 일자: {model_data.get('train_date', 'Unknown')}")
        print(f"   - 필요한 피처 수: {len(feature_names)}")
        
        # 2. 피처 데이터 로드
        feature_path = 'data/processed/subway_features_balanced.csv'
        
        if os.path.exists(feature_path):
            print(f"\n📊 피처 데이터 로딩 중...")
            df_features = pd.read_csv(feature_path, encoding='utf-8-sig')
            print(f"✅ 피처 데이터 로드 완료: {len(df_features):,}개 레코드")
            
            # 3. 역 목록 생성
            if '지하철역' in df_features.columns and '호선명' in df_features.columns:
                unique_stations = df_features[['지하철역', '호선명']].drop_duplicates()
                station_counts = df_features.groupby('지하철역')['호선명'].nunique()
                is_transfer = station_counts > 1
                
                df_stations = unique_stations.copy()
                df_stations['환승역여부'] = df_stations['지하철역'].map(is_transfer).fillna(False)
                
                print(f"✅ 역 정보 생성 완료: {len(df_stations)}개 역")
        else:
            print(f"⚠️  피처 데이터 없음: {feature_path}")
            df_features = pd.DataFrame()
            df_stations = pd.DataFrame()
        
        print(f"\n" + "="*80)
        print("✅ 서버 준비 완료!")
        print("="*80)
        print(f"📍 API 문서: http://localhost:8000/docs")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ 서버 시작 실패: {e}")
        import traceback
        traceback.print_exc()
        raise


# ============================================
# 피처 준비 함수
# ============================================

def prepare_features(
    station_name: str,
    line_name: str,
    hour: int,
    day_of_week: int,
    month: int,
    is_holiday: bool
) -> Dict[str, float]:
    """예측을 위한 피처 준비 (강화된 피처 포함)"""
    
    current_year = datetime.now().year
    features = {}
    
    # 1. 시간 관련 피처
    features['시간'] = float(hour)
    features['시간_sin'] = np.sin(2 * np.pi * hour / 24)
    features['시간_cos'] = np.cos(2 * np.pi * hour / 24)
    
    # 시간대 구분
    if 0 <= hour < 6:
        시간대 = 0
    elif 6 <= hour < 9:
        시간대 = 1
    elif 9 <= hour < 12:
        시간대 = 2
    elif 12 <= hour < 14:
        시간대 = 3
    elif 14 <= hour < 18:
        시간대 = 4
    elif 18 <= hour < 21:
        시간대 = 5
    else:
        시간대 = 6
    
    features['시간대구분_encoded'] = float(시간대)
    
    # 2. 강화된 시간 피처
    features['출근시간대'] = float(1 if 7 <= hour <= 9 else 0)
    features['퇴근시간대'] = float(1 if 18 <= hour <= 20 else 0)
    features['점심시간대'] = float(1 if 12 <= hour <= 13 else 0)
    features['심야시간대'] = float(1 if 0 <= hour <= 5 else 0)
    features['저녁시간대'] = float(1 if 21 <= hour <= 23 else 0)
    
    # 러시아워 강도
    rush_intensity = 0.0
    if hour == 8:
        rush_intensity = 1.0
    elif hour in [7, 9]:
        rush_intensity = 0.7
    elif hour in [18, 19]:
        rush_intensity = 0.9
    elif hour == 20:
        rush_intensity = 0.6
    
    features['러시아워강도'] = rush_intensity
    
    # 3. 날짜 관련 피처
    features['요일'] = float(day_of_week)
    features['요일_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    features['요일_cos'] = np.cos(2 * np.pi * day_of_week / 7)
    
    features['월'] = float(month)
    features['월_sin'] = np.sin(2 * np.pi * month / 12)
    features['월_cos'] = np.cos(2 * np.pi * month / 12)
    
    features['연도'] = float(current_year)
    features['분기'] = float((month - 1) // 3 + 1)
    
    # 4. 주말/공휴일 정보
    is_weekend = day_of_week >= 5
    features['주말여부'] = float(1 if is_weekend else 0)
    features['공휴일여부'] = float(1 if is_holiday else 0)
    features['공휴일전날'] = 0.0
    features['공휴일다음날'] = 0.0
    features['연휴여부'] = float(1 if (is_weekend or is_holiday) else 0)
    
    # 5. 역/노선 정보
    station_encoded = 0.0
    line_encoded = 0.0
    
    if df_features is not None and not df_features.empty:
        try:
            station_data = df_features[
                (df_features['지하철역'] == station_name) &
                (df_features['호선명'] == line_name)
            ]
            
            if not station_data.empty:
                if '지하철역_encoded' in station_data.columns:
                    station_encoded = float(station_data['지하철역_encoded'].iloc[0])
                
                if '호선명_encoded' in station_data.columns:
                    line_encoded = float(station_data['호선명_encoded'].iloc[0])
                else:
                    line_encoding = {
                        '1호선': 0, '2호선': 1, '3호선': 2, '4호선': 3,
                        '5호선': 4, '6호선': 5, '7호선': 6, '8호선': 7, '9호선': 8,
                        '경의중앙선': 9, '공항철도': 10, '경춘선': 11, '수인분당선': 12,
                        '신분당선': 13, '경강선': 14, '서해선': 15, '인천1호선': 16,
                        '인천2호선': 17, '우이신설선': 18, '신림선': 19
                    }
                    line_encoded = float(line_encoding.get(line_name, 0))
                
                if '환승역여부' in station_data.columns:
                    features['환승역여부'] = float(station_data['환승역여부'].iloc[0])
                else:
                    station_lines = df_features[df_features['지하철역'] == station_name]['호선명'].nunique()
                    features['환승역여부'] = float(1 if station_lines > 1 else 0)
        except Exception as e:
            print(f"❌ 인코딩 오류: {e}")
    
    features['지하철역_encoded'] = station_encoded
    features['호선명_encoded'] = line_encoded
    
    # 6. 역 카테고리 피처
    station_category = STATION_CATEGORIES.get(station_name, DEFAULT_CATEGORY)
    
    features['역_업무지구'] = float(station_category['업무지구'])
    features['역_상업지구'] = float(station_category['상업지구'])
    features['역_환승역등급'] = float(station_category['환승역등급'])
    features['역_대학가'] = float(station_category['대학가'])
    features['역_주거지역'] = float(station_category['주거지역'])
    features['역_그룹'] = float(station_category['역_그룹'])
    
    # 7. 기본 상호작용 피처
    features['time_dow_interaction'] = 시간대 * 10 + day_of_week
    features['station_time_interaction'] = station_encoded * 10 + 시간대
    
    # 8. 강화된 상호작용 피처
    features['business_rush_interaction'] = features['역_업무지구'] * rush_intensity * 10
    features['transfer_rush_interaction'] = features['역_환승역등급'] * rush_intensity * 10
    features['university_morning_interaction'] = features['역_대학가'] * features['출근시간대']
    
    # 평일/주말 × 역 카테고리
    features['business_weekday'] = features['역_업무지구'] * (1 - features['주말여부'])
    features['commercial_weekend'] = features['역_상업지구'] * features['주말여부']
    
    # 9. 3-way 상호작용
    features['group_time_dow'] = (features['역_그룹'] * 100 + 
                                   시간대 * 10 + 
                                   day_of_week)
    
    features['transfer_rush_weekday'] = (features['역_환승역등급'] * 
                                          features['출근시간대'] * 
                                          (1 - features['주말여부']))
    
    # 10. 누락된 피처 0으로 채우기
    for feature_name in feature_names:
        if feature_name not in features:
            features[feature_name] = 0.0
    
    return features


# ============================================
# API 엔드포인트
# ============================================

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "지하철 혼잡도 예측 API",
        "version": "1.0.0",
        "model": "enhanced (강화된 피처)",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """헬스체크"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "features_count": len(feature_names) if feature_names else 0,
        "data_loaded": df_features is not None and not df_features.empty,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/stations")
async def get_stations(
    line_name: Optional[str] = None,
    search: Optional[str] = None
):
    """역 목록 조회"""
    if df_stations is None or df_stations.empty:
        raise HTTPException(status_code=503, detail="역 정보를 사용할 수 없습니다")
    
    result = df_stations.copy()
    
    if line_name:
        result = result[result['호선명'] == line_name]
    
    if search:
        result = result[result['지하철역'].str.contains(search, case=False, na=False)]
    
    stations = []
    for _, row in result.head(100).iterrows():
        stations.append({
            "name": row['지하철역'],
            "line": row['호선명'],
            "is_transfer": bool(row.get('환승역여부', False))
        })
    
    return {
        "success": True,
        "total": len(result),
        "stations": stations
    }


@app.get("/lines")
async def get_lines():
    """호선 목록 조회"""
    if df_stations is None or df_stations.empty:
        lines = ["1호선", "2호선", "3호선", "4호선", "5호선", "6호선", "7호선", "8호선", "9호선"]
    else:
        lines = sorted(df_stations['호선명'].unique().tolist())
    
    return {
        "success": True,
        "total": len(lines),
        "lines": lines
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_congestion(request: PredictionRequest):
    """혼잡도 예측"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")
    
    try:
        month = request.month or datetime.now().month
        hour = request.hour
        
        # 피처 준비
        features = prepare_features(
            station_name=request.station_name,
            line_name=request.line_name,
            hour=hour,
            day_of_week=request.day_of_week,
            month=month,
            is_holiday=request.is_holiday
        )
        
        # 모델 입력 준비
        feature_values = [features[name] for name in feature_names]
        X = np.array([feature_values])
        
        # 예측
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # 🔥 매우 공격적인 분류 로직: 러시아워는 무조건 혼잡 이상
        max_prob_class = int(np.argmax(probabilities))
        
        # 기본 예측
        adjusted_prediction = max_prob_class
        
        # 🔥🔥 러시아워 강제 조정 (7-9시, 18-20시)
        if 7 <= hour <= 9 or 18 <= hour <= 20:
            # 러시아워에는 최소 "혼잡" 보장
            if adjusted_prediction < 2:
                adjusted_prediction = 2  # 무조건 혼잡 이상
            
            # 추가 상향 조건: 확률 합산
            혼잡_이상_확률 = probabilities[2] + probabilities[3]
            
            if 혼잡_이상_확률 > 0.05:  # 5%만 넘어도 혼잡
                adjusted_prediction = 2
            
            if probabilities[3] > 0.02:  # 매우혼잡 확률 2% 이상
                adjusted_prediction = 3
            
            # 🔥 역 특성 추가 상향
            station_category = STATION_CATEGORIES.get(request.station_name, DEFAULT_CATEGORY)
            
            # 환승역은 더 공격적으로
            if station_category['환승역등급'] >= 2:
                if probabilities[3] > 0.01:  # 1%만 넘어도 매우혼잡
                    adjusted_prediction = 3
                else:
                    adjusted_prediction = max(adjusted_prediction, 2)  # 최소 혼잡
            
            # 업무지구 + 출근시간 (7-9시)
            if station_category['업무지구'] == 1 and 7 <= hour <= 9:
                if probabilities[3] > 0.015:
                    adjusted_prediction = 3
                else:
                    adjusted_prediction = max(adjusted_prediction, 2)
            
            # 상업지구 + 퇴근시간 (18-20시)
            if station_category['상업지구'] == 1 and 18 <= hour <= 20:
                adjusted_prediction = max(adjusted_prediction, 2)
        
        # 점심 시간대 (12-13시): 최소 보통
        elif 12 <= hour <= 13:
            if adjusted_prediction == 0:
                adjusted_prediction = 1
        
        # 저녁 시간대 (21-22시): 최소 보통
        elif 21 <= hour <= 22:
            if adjusted_prediction == 0 and probabilities[1] > 0.03:
                adjusted_prediction = 1
        
        # 오전/오후 (10-11시, 14-17시): 가벼운 상향
        elif (10 <= hour <= 11) or (14 <= hour <= 17):
            if adjusted_prediction == 0 and probabilities[1] + probabilities[2] > 0.08:
                adjusted_prediction = 1
        
        # 특수 케이스 하향 조정 (최소한만)
        # 심야 시간(0-5시)은 최대 "보통"
        if hour <= 5:
            adjusted_prediction = min(adjusted_prediction, 1)
        
        # 주말 심야(23시)는 최대 "혼잡"
        if request.day_of_week >= 5 and hour >= 23:
            adjusted_prediction = min(adjusted_prediction, 2)
        
        final_prediction = adjusted_prediction
        
        # 결과 포맷
        prediction_result = {
            "congestion_level": int(final_prediction),
            "congestion_label": CONGESTION_LEVEL_MAP[int(final_prediction)],
            "probability": {
                "여유": float(probabilities[0]),
                "보통": float(probabilities[1]),
                "혼잡": float(probabilities[2]),
                "매우혼잡": float(probabilities[3])
            }
        }
        
        # 입력 정보
        input_info = {
            "station_name": request.station_name,
            "line_name": request.line_name,
            "hour": request.hour,
            "day_of_week": request.day_of_week,
            "day_name": DAY_NAME_MAP[request.day_of_week],
            "month": month,
            "is_holiday": request.is_holiday,
            "is_weekend": request.day_of_week >= 5
        }
        
        # 추천 메시지
        level = int(final_prediction)
        is_rush_hour = hour in [7, 8, 9, 18, 19, 20]
        
        if level == 0:
            recommendation = "여유로운 시간대입니다. 편하게 이용하세요."
        elif level == 1:
            recommendation = "보통 수준의 혼잡도입니다."
        elif level == 2:
            if is_rush_hour:
                recommendation = "혼잡한 시간대입니다. 가능하면 30분 전후 이용을 권장합니다."
            else:
                recommendation = "다소 혼잡할 수 있습니다."
        else:
            if is_rush_hour:
                recommendation = "매우 혼잡한 시간대입니다. 다음 시간대 이용을 적극 권장합니다."
            else:
                recommendation = "매우 혼잡합니다. 시간 조정을 권장합니다."
        
        return PredictionResponse(
            success=True,
            prediction=prediction_result,
            input=input_info,
            recommendation=recommendation,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"예측 중 오류: {str(e)}")


@app.post("/predict/hourly")
async def predict_hourly(
    station_name: str,
    line_name: str,
    day_of_week: int,
    month: Optional[int] = None,
    is_holiday: Optional[bool] = False
):
    """시간대별 혼잡도 예측"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")
    
    month = month or datetime.now().month
    predictions = []
    
    for hour in range(24):
        try:
            features = prepare_features(
                station_name=station_name,
                line_name=line_name,
                hour=hour,
                day_of_week=day_of_week,
                month=month,
                is_holiday=is_holiday
            )
            
            feature_values = [features[name] for name in feature_names]
            X = np.array([feature_values])
            
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            
            # 🔥 동일한 공격적 로직 적용
            max_prob_class = int(np.argmax(probabilities))
            adjusted_prediction = max_prob_class
            
            # 🔥🔥 러시아워 강제 조정 (7-9시, 18-20시)
            if 7 <= hour <= 9 or 18 <= hour <= 20:
                # 러시아워에는 최소 "혼잡" 보장
                if adjusted_prediction < 2:
                    adjusted_prediction = 2
                
                혼잡_이상_확률 = probabilities[2] + probabilities[3]
                
                if 혼잡_이상_확률 > 0.05:
                    adjusted_prediction = 2
                
                if probabilities[3] > 0.02:
                    adjusted_prediction = 3
                
                # 역 특성 추가 상향
                station_category = STATION_CATEGORIES.get(station_name, DEFAULT_CATEGORY)
                
                if station_category['환승역등급'] >= 2:
                    if probabilities[3] > 0.01:
                        adjusted_prediction = 3
                    else:
                        adjusted_prediction = max(adjusted_prediction, 2)
                
                if station_category['업무지구'] == 1 and 7 <= hour <= 9:
                    if probabilities[3] > 0.015:
                        adjusted_prediction = 3
                    else:
                        adjusted_prediction = max(adjusted_prediction, 2)
                
                if station_category['상업지구'] == 1 and 18 <= hour <= 20:
                    adjusted_prediction = max(adjusted_prediction, 2)
            
            # 점심 시간대 (12-13시): 최소 보통
            elif 12 <= hour <= 13:
                if adjusted_prediction == 0:
                    adjusted_prediction = 1
            
            # 저녁 시간대 (21-22시): 최소 보통
            elif 21 <= hour <= 22:
                if adjusted_prediction == 0 and probabilities[1] > 0.03:
                    adjusted_prediction = 1
            
            # 오전/오후 (10-11시, 14-17시): 가벼운 상향
            elif (10 <= hour <= 11) or (14 <= hour <= 17):
                if adjusted_prediction == 0 and probabilities[1] + probabilities[2] > 0.08:
                    adjusted_prediction = 1
            
            # 특수 케이스 하향 조정
            if hour <= 5:
                adjusted_prediction = min(adjusted_prediction, 1)
            
            if day_of_week >= 5 and hour >= 23:
                adjusted_prediction = min(adjusted_prediction, 2)
            
            final_prediction = adjusted_prediction
            
            predictions.append({
                "hour": hour,
                "time": f"{hour:02d}:00",
                "congestion_level": int(final_prediction),
                "congestion_label": CONGESTION_LEVEL_MAP[int(final_prediction)],
                "probability": float(probabilities[int(final_prediction)])
            })
            
        except Exception as e:
            print(f"시간 {hour} 예측 실패: {e}")
            continue
    
    # 요약 통계
    if predictions:
        levels = [p['congestion_level'] for p in predictions]
        most_congested = [p['hour'] for p in predictions if p['congestion_level'] >= 2]
        least_congested = [p['hour'] for p in predictions if p['congestion_level'] == 0]
        
        summary = {
            "most_congested_hours": most_congested[:5],
            "least_congested_hours": least_congested[:5],
            "average_congestion": round(sum(levels) / len(levels), 2)
        }
    else:
        summary = {}
    
    return {
        "success": True,
        "station_name": station_name,
        "line_name": line_name,
        "day_of_week": day_of_week,
        "day_name": DAY_NAME_MAP[day_of_week],
        "predictions": predictions,
        "summary": summary,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model/info")
async def get_model_info():
    """모델 정보 조회"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")
    
    try:
        with open('models/subway_congestion_model_enhanced.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        return {
            "success": True,
            "model_info": {
                "model_type": model_data.get('model_type', 'Unknown'),
                "train_date": model_data.get('train_date', 'Unknown'),
                "feature_count": len(feature_names),
                "features": feature_names[:10],
                "note": model_data.get('note', '')
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 정보 조회 실패: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*80)
    print("🚇 지하철 혼잡도 예측 API 서버")
    print("="*80)
    print("📍 서버 주소: http://localhost:8000")
    print("📚 API 문서: http://localhost:8000/docs")
    print("="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)