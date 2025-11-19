"""
지하철 혼잡도 예측 - FastAPI 백엔드 (수정 버전)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import os

# FastAPI 앱 생성
app = FastAPI(
    title="지하철 혼잡도 예측 API",
    description="서울 지하철 혼잡도를 예측하고 실시간 정보를 제공하는 API",
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
df_features = None  # ✅ 추가!

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


# 요청/응답 모델 정의
class PredictionRequest(BaseModel):
    station_name: str
    line_name: str
    hour: int
    date: Optional[str] = None


class PredictionResponse(BaseModel):
    station_name: str
    line_name: str
    hour: int
    date: str
    congestion_level: int
    congestion_label: str
    congestion_color: str
    predicted_passengers: int
    confidence: float
    recommendation: str


class StationInfo(BaseModel):
    station_name: str
    line_name: str
    avg_passengers: int



@app.post("/debug/features")
async def debug_features(request: PredictionRequest):
    """디버깅: 실제 입력되는 피처 확인"""
    date_str = request.date or datetime.now().strftime('%Y-%m-%d')
    
    try:
        features = prepare_features(
            request.station_name,
            request.line_name,
            request.hour,
            date_str
        )
        
        # 피처를 모델 순서로 정렬
        feature_values = [features[name] for name in feature_names]
        
        # 통계
        import statistics
        
        return {
            "station": request.station_name,
            "line": request.line_name,
            "hour": request.hour,
            "feature_count": len(feature_values),
            "non_zero_features": sum(1 for v in feature_values if v != 0),
            "feature_stats": {
                "min": min(feature_values),
                "max": max(feature_values),
                "mean": statistics.mean(feature_values),
                "zeros": sum(1 for v in feature_values if v == 0)
            },
            "key_features": {
                "시간": features.get("시간"),
                "승차인원": features.get("승차인원"),
                "하차인원": features.get("하차인원"),
                "총승하차인원": features.get("총승하차인원"),
                "역_평균승하차": features.get("역_평균승하차"),
                "시간_평균승하차": features.get("시간_평균승하차"),
                "혼잡도레벨": features.get("혼잡도레벨")
            },
            "first_10_features": dict(list(features.items())[:10]),
            "model_expects": feature_names[:10]
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.on_event("startup")
async def load_model():
    """서버 시작 시 모델 및 데이터 로드"""
    global model, feature_names, df_stations, df_features
    
    print("🚇 모델 및 데이터 로딩 중...")
    
    try:
        # 1. 모델 로드
        model_path = 'models/subway_congestion_model_improved.pkl'  # ✅ 수정!
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            feature_names = model_data['feature_names']
        
        print(f"✅ 모델 로드 완료: {model_path}")
        print(f"   필요한 피처 수: {len(feature_names)}")
        
        # 2. 피처 데이터 로드
        try:
            print("📊 피처 데이터 로딩 중...")
            df_features = pd.read_csv('data/processed/subway_features_balanced.csv', encoding='utf-8-sig')  # ✅ 수정!
            print(f"✅ 피처 데이터 로드 완료: {len(df_features):,}개 레코드")
            print(f"   컬럼 수: {len(df_features.columns)}")
            
            # 3. 역 목록 생성 (간단하게)
            unique_stations = df_features[['지하철역', '호선명']].drop_duplicates()
            unique_stations = unique_stations.rename(columns={
                '지하철역': 'station_name',
                '호선명': 'line_name'
            })
            df_stations = unique_stations
            
            print(f"✅ 역 정보 생성 완료: {len(df_stations)}개 역")
            
        except FileNotFoundError:
            print("⚠️  subway_features_balanced.csv를 찾을 수 없습니다!")  # ✅ 메시지도 수정
            df_features = pd.DataFrame()
            df_stations = pd.DataFrame()
        except Exception as e:
            print(f"⚠️  데이터 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            df_features = pd.DataFrame()
            df_stations = pd.DataFrame()
        
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        raise


def prepare_features(station_name: str, line_name: str, hour: int, date_str: str):
    """예측을 위한 피처 준비 - 시간대별 실제 데이터 사용"""
    global df_features, feature_names
    
    date = datetime.strptime(date_str, '%Y-%m-%d')
    
    if df_features is None or df_features.empty:
        raise HTTPException(status_code=500, detail="피처 데이터가 로드되지 않았습니다")
    
    # 해당 역-시간 찾기
    mask = (
        (df_features['지하철역'] == station_name) & 
        (df_features['호선명'] == line_name) &
        (df_features['시간'] == hour)
    )
    
    matching_data = df_features[mask]
    
    # 유연한 검색
    if matching_data.empty:
        mask = (
            (df_features['지하철역'].str.contains(station_name, na=False)) & 
            (df_features['호선명'].str.contains(line_name.replace('호선', ''), na=False))
        )
        all_station_data = df_features[mask]
        
        if all_station_data.empty:
            raise HTTPException(status_code=404, detail=f"{station_name} {line_name} 데이터 없음")
        
        matching_data = all_station_data[all_station_data['시간'] == hour]
        
        if matching_data.empty:
            available_hours = all_station_data['시간'].unique()
            closest_hour = min(available_hours, key=lambda x: abs(x - hour))
            matching_data = all_station_data[all_station_data['시간'] == closest_hour]
    
    # 🔥 핵심: 해당 시간대의 평균값 사용 (여러 날짜의 평균)
    # 한 행만 쓰지 말고, 해당 시간대 전체 평균!
    time_avg_data = matching_data.mean(numeric_only=True)
    
    # 타겟 변수 제외
    EXCLUDE_FEATURES = [
        '혼잡도레벨', '혼잡도', '총승하차인원',
        '사용일자', '지하철역', '호선명', 'Unnamed: 0'
    ]
    
    features = {}
    
    for feature_name in feature_names:
        if feature_name in EXCLUDE_FEATURES:
            continue
            
        if feature_name in time_avg_data.index:
            value = time_avg_data[feature_name]
            features[feature_name] = 0.0 if pd.isna(value) else float(value)
        else:
            features[feature_name] = 0.0
    
    # 날짜 관련 피처 업데이트 (현재 날짜)
    features['요일'] = float(date.weekday())
    features['월'] = float(date.month)
    features['연도'] = float(date.year)
    features['분기'] = float((date.month - 1) // 3 + 1)
    features['주말여부'] = float(1 if date.weekday() >= 5 else 0)
    features['연휴여부'] = features['주말여부']
    features['시간'] = float(hour)
    
    # sin/cos 업데이트
    features['시간_sin'] = np.sin(2 * np.pi * hour / 24)
    features['시간_cos'] = np.cos(2 * np.pi * hour / 24)
    features['요일_sin'] = np.sin(2 * np.pi * date.weekday() / 7)
    features['요일_cos'] = np.cos(2 * np.pi * date.weekday() / 7)
    features['월_sin'] = np.sin(2 * np.pi * date.month / 12)
    features['월_cos'] = np.cos(2 * np.pi * date.month / 12)
    
    return features

@app.get("/")
async def root():
    return {
        "message": "지하철 혼잡도 예측 API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "data_loaded": df_features is not None and not df_features.empty
    }


@app.get("/stations", response_model=List[StationInfo])
async def get_stations():
    if df_stations is None or df_stations.empty:
        raise HTTPException(status_code=500, detail="역 정보를 로드할 수 없습니다")
    
    return df_stations.head(100).to_dict('records')


@app.post("/predict", response_model=PredictionResponse)
async def predict_congestion(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다")
    
    date_str = request.date or datetime.now().strftime('%Y-%m-%d')
    
    if not 0 <= request.hour <= 23:
        raise HTTPException(status_code=400, detail="시간은 0~23 사이여야 합니다")
    
    try:
        features = prepare_features(
            request.station_name,
            request.line_name,
            request.hour,
            date_str
        )
        
        feature_values = [features[name] for name in feature_names]
        
        # 모델 예측
        X = np.array([feature_values])
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        confidence = float(probability[prediction])
        
        # 🔥 핵심: 역 평균 대비 비율 계산
        actual_passengers = features.get('승차인원', 0) + features.get('하차인원', 0)
        station_avg = features.get('역_평균승하차', 30000)
        
        # 상대적 혼잡도 = 현재 승하차 / 역 평균
        relative_congestion = actual_passengers / station_avg if station_avg > 0 else 1.0
        
        date = datetime.strptime(date_str, '%Y-%m-%d')
        is_weekend = date.weekday() >= 5
        
        # 🎯 상대적 비율 기반 혼잡도 판단
        if relative_congestion < 0.4:
            adjusted_prediction = 0  # 여유 (평균의 40% 미만)
        elif relative_congestion < 0.8:
            adjusted_prediction = 1  # 보통 (평균의 40~80%)
        elif relative_congestion < 1.3:
            adjusted_prediction = 2  # 혼잡 (평균의 80~130%)
        else:
            adjusted_prediction = 3  # 매우혼잡 (평균의 130% 이상)
        
        # 시간대별 추가 보정
        if request.hour <= 5 or request.hour >= 23:
            adjusted_prediction = min(adjusted_prediction, 1)  # 심야는 최대 "보통"
        
        # 주말 보정
        if is_weekend and adjusted_prediction >= 3:
            adjusted_prediction = 2  # 주말은 최대 "혼잡"
        
        # 모델 예측과 규칙 기반 예측 중 더 보수적인 것 선택
        # (안전을 위해 더 혼잡한 쪽 선택)
        final_prediction = max(int(prediction), adjusted_prediction)
        
        # 예상 승하차 인원
        predicted_passengers = int(actual_passengers)
        
        # 추천 메시지 (상대적 혼잡도 정보 포함)
        congestion_percent = int(relative_congestion * 100)
        
        base_recommendations = {
            0: f"여유로워요! (평균의 {congestion_percent}%) 지금 바로 이용하세요 😊",
            1: f"보통 수준이에요. (평균의 {congestion_percent}%) 편하게 이용 가능합니다 👍",
            2: f"다소 혼잡해요. (평균의 {congestion_percent}%) 시간 여유가 있다면 다른 시간을 고려하세요 ⚠️",
            3: f"매우 혼잡해요! (평균의 {congestion_percent}%) 가능하면 다른 시간대를 이용하세요 🚫"
        }
        
        return PredictionResponse(
            station_name=request.station_name,
            line_name=request.line_name,
            hour=request.hour,
            date=date_str,
            congestion_level=final_prediction,
            congestion_label=CONGESTION_LEVEL_MAP[final_prediction],
            congestion_color=CONGESTION_COLOR_MAP[final_prediction],
            predicted_passengers=predicted_passengers,
            confidence=round(confidence, 2),
            recommendation=base_recommendations[final_prediction]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 중 오류: {str(e)}")


@app.get("/predict/hourly/{station_name}/{line_name}")
async def predict_hourly(station_name: str, line_name: str, date: Optional[str] = None):
    if model is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다")
    
    date = date or datetime.now().strftime('%Y-%m-%d')
    results = []
    
    for hour in range(5, 24):
        try:
            features = prepare_features(station_name, line_name, hour, date)
            feature_values = [features[name] for name in feature_names]
            
            X = np.array([feature_values])
            prediction = model.predict(X)[0]
            
            results.append({
                "hour": hour,
                "congestion_level": int(prediction),
                "congestion_label": CONGESTION_LEVEL_MAP[prediction],
                "congestion_color": CONGESTION_COLOR_MAP[prediction]
            })
        except:
            continue
    
    return {
        "station_name": station_name,
        "line_name": line_name,
        "date": date,
        "hourly_predictions": results
    }


@app.get("/stations/search/{query}")
async def search_stations(query: str):
    if df_stations is None or df_stations.empty:
        raise HTTPException(status_code=500, detail="역 정보를 로드할 수 없습니다")
    
    results = df_stations[
        df_stations['station_name'].str.contains(query, case=False, na=False)
    ].head(20)
    
    return results.to_dict('records')


if __name__ == "__main__":
    import uvicorn
    
    print("🚇 지하철 혼잡도 예측 API 서버 시작")
    print("=" * 60)
    print("📍 서버 주소: http://localhost:8000")
    print("📚 API 문서: http://localhost:8000/docs")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)