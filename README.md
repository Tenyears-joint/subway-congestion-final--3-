# 🚇 지하철 혼잡도 예측 + 실시간 도착정보 서비스
 RandomForest 모델을 사용하여 저장된 지하철 승하차 데이터를 참고, 시간대 별 혼잡도 예측과 실시간 열차 도착정보를 제공하고, 출력해낸 혼잡도를 시각 자료로 나타내는 스마트 정보 앱.

## 📋 프로젝트 개요

이 프로젝트는 서울시 지하철 이용객들이 혼잡도를 미리 예측하여 효율적인 이동 계획을 세울 수 있도록 돕는 서비스입니다.

### 주요 기능

1. **혼잡도 예측** (AI 기반)
   - 시간대별, 역별 승하차 인원 예측
   - 날씨, 공휴일, 요일 등 다양한 변수 고려
   - Random Forest 모델 사용

2. **실시간 도착정보**
   - 현재 열차 도착 시간
   - 상행/하행선 구분
   - 행선지 정보

3. **통합 대시보드**
   - 혼잡도 + 도착정보 한눈에 확인
   - 시간대별 혼잡도 그래프
   - 추천 시간대 제시

## 🎯 데이터 소스

| 데이터 | 출처 | API 키 | 용도 |
|--------|------|--------|------|
| 지하철 승하차 인원 | 서울 열린데이터광장 | 일반 인증키 | 모델 학습 |
| 실시간 도착정보 | 서울 열린데이터광장 | 실시간 지하철 인증키 | 실시간 정보 |
| 혼잡도 통계 | 서울교통공사 | - | 검증 및 보정 |
| 날씨 데이터 | 기상청 | 기상청 인증키 | 외부 요인 |
| 공휴일 정보 | 한국천문연구원 | - | 특일 반영 |

## 🛠️ 기술 스택

- **언어**: Python 3.9+
- **머신러닝**: scikit-learn
- **웹 프레임워크**: Flask
- **데이터 처리**: pandas, numpy
- **시각화**: matplotlib, seaborn

## 📦 설치 방법

### 1. 프로젝트 클론

```bash
git clone https://github.com/yourusername/subway-congestion-prediction.git
cd subway-congestion-prediction
```

### 2. 가상환경 생성 및 활성화

```bash
python -m venv venv   # 개인 환경 차이가 있으므로 재설치 필요

# Windows

venv\Scripts\activate


```

### 3. 필요한 라이브러리 설치

```bash
pip install -r requirements.txt
```
## 3.5 pip 업데이트 필요시 업데이트.

```bash
python.exe -m pip install --upgrade pip

```

### 4. API 키 설정 ⭐ 중요!

#### 필요한 API 키 (총 3개)

| API 키 | 발급처 | 소요시간 |
|--------|--------|---------|
| 서울시 일반 인증키 | https://data.seoul.go.kr | 즉시 |
| 서울시 실시간 지하철 인증키 | https://data.seoul.go.kr | 즉시 |
| 기상청 단기예보 인증키 | https://www.data.go.kr | 1-2일 |

#### API 키 발급 방법

자세한 발급 방법은 **[API_KEY_GUIDE.md](API_KEY_GUIDE.md)** 참고

#### .env 파일 설정

```bash
# .env.template을 .env로 복사
copy .env.template .env  # Windows
cp .env.template .env    # Mac/Linux
```

`.env` 파일을 열어서 발급받은 API 키 입력:

```env
# 서울시 API 키 (2개)
SEOUL_GENERAL_API_KEY=발급받은_일반_인증키
SEOUL_REALTIME_API_KEY=발급받은_실시간_인증키

# 기상청 API 키
WEATHER_API_KEY=발급받은_기상청_인증키
```

## 🚀 실행 방법

### 1. 프로젝트 구조 생성 (최초 1회)

```bash
python setup_project.py
```

### 2. 데이터 수집

```bash
# data_collector.py를 src 폴더로 이동
move data_collector.py src/  # Windows
mv data_collector.py src/    # Mac/Linux

# 데이터 수집 테스트
python src/data_collector.py
```

### 3. 모델 학습 (추후 진행)

```bash
python scripts/train_model.py
```

### 4. 웹 애플리케이션 실행 (추후 진행)

```bash
python app/app.py
```

브라우저에서 `http://localhost:5000` 접속

## 📊 프로젝트 구조

```
subway-congestion-prediction/
├── data/              # 데이터 저장
│   ├── raw/          # 원본 데이터
│   ├── processed/    # 전처리 데이터
│   └── external/     # 외부 데이터
├── src/               # 소스 코드
│   ├── data_collector.py      # 데이터 수집 ✅
│   ├── preprocessor.py        # 전처리
│   ├── feature_engineering.py # 피처 엔지니어링
│   ├── model_trainer.py       # 모델 학습
│   └── predictor.py           # 예측
├── app/               # 웹 애플리케이션
├── models/            # 학습된 모델
├── notebooks/         # Jupyter 노트북
└── scripts/           # 실행 스크립트
```

## 🔑 API 키 관련 주의사항

### 보안
- ✅ `.env` 파일은 Git에 절대 업로드하지 마세요
- ✅ API 키를 공개된 곳에 노출하지 마세요
- ✅ 키가 노출되었다면 즉시 재발급하세요

### 사용 제한
- 서울시 실시간 API: **하루 1,000건 제한**
- 일반 API: 한 번에 최대 1,000건 조회 가능

### API 키 구분
- **일반 인증키**: 승하차 데이터 수집용
- **실시간 지하철 인증키**: 도착정보 조회용
- **두 개는 서로 다른 키입니다!**

## 📈 개발 로드맵

- [x] 프로젝트 구조 설정
- [x] 데이터 수집 모듈 개발
- [ ] 데이터 전처리
- [ ] 피처 엔지니어링
- [ ] 모델 학습 및 평가
- [ ] 웹 인터페이스 개발
- [ ] 실시간 도착정보 통합
- [ ] 배포

## 🤝 기여 방법

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

This project is licensed under the MIT License

## 👥 개발자

- **이름**: [Your Name]
- **이메일**: [your.email@example.com]
- **GitHub**: [@yourusername](https://github.com/yourusername)

## 🙏 감사의 말

- 서울 열린데이터광장
- 공공데이터포털
- 서울교통공사
- 기상청

## 📞 문의

프로젝트에 대한 문의사항은 Issues 탭을 이용해주세요.

---

## 🚨 문제 해결

### API 키가 작동하지 않아요
1. `.env` 파일이 프로젝트 루트에 있는지 확인
2. API 키를 정확히 복사했는지 확인
3. `python-dotenv`가 설치되어 있는지 확인
4. 서울시 API는 **두 개의 다른 키**가 필요합니다

### 데이터 수집이 실패해요
1. API 키가 올바른지 확인
2. 인터넷 연결 상태 확인
3. API 호출 제한(1,000건/일)을 초과했는지 확인
4. `python src/data_collector.py`로 테스트 실행

### 더 많은 도움이 필요하면
- [API 키 발급 가이드](API_KEY_GUIDE.md) 참고
- Issues 탭에 질문 남기기
