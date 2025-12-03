"""
model_trainer_enhanced.py
강화된 피처를 포함한 모델 학습 + 시각화
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

print("=" * 80)
print("🚂 모델 학습 시작 (강화된 피처 포함)")
print("=" * 80)

# 1. 데이터 로드
print("\n1. 데이터 로드 중...")
df = pd.read_csv('data/processed/subway_features_enhanced.csv', encoding='utf-8-sig')
print(f"   로드 완료: {len(df):,}개 레코드, {len(df.columns)}개 컬럼")

# 2. 피처와 타겟 분리
print("\n2. 피처와 타겟 분리 중...")
TARGET = '혼잡도레벨'

EXCLUDE_COLUMNS = [
    TARGET,
    '사용일자', 'Unnamed: 0',
    '지하철역', '호선명'
]

feature_columns = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
print(f"   피처 수: {len(feature_columns)}개")

# 새로운 피처 확인
category_features = [col for col in feature_columns if '역_' in col]
interaction_features = [col for col in feature_columns if 'interaction' in col]

if category_features:
    print(f"\n   🎯 역 카테고리 피처: {len(category_features)}개")
    for feat in category_features:
        print(f"      - {feat}")

if interaction_features:
    print(f"\n   ✨ 상호작용 피처: {len(interaction_features)}개")
    for feat in interaction_features:
        print(f"      - {feat}")

X = df[feature_columns]
y = df[TARGET]

print(f"\n   X shape: {X.shape}")
print(f"   y shape: {y.shape}")

# 3. 데이터 품질 확인
print("\n3. 데이터 품질 확인...")
print(f"   결측치: {X.isnull().sum().sum()}개")
print(f"   무한대: {np.isinf(X.select_dtypes(include=[np.number])).sum().sum()}개")

# 4. 학습/테스트 데이터 분리
print("\n4. 학습/테스트 데이터 분리 중...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print(f"   학습 데이터: {X_train.shape[0]:,}개")
print(f"   테스트 데이터: {X_test.shape[0]:,}개")

# 5. 모델 학습
print("\n5. Random Forest 모델 학습 중...")
print("   (이 과정은 몇 분 소요될 수 있습니다...)")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=50,
    min_samples_leaf=20,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    verbose=0
)

model.fit(X_train, y_train)
print("   ✅ 학습 완료!")

# 6. 예측 및 평가
print("\n6. 모델 평가 중...")

# 학습 데이터 평가
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred, average='weighted')

# 테스트 데이터 평가
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

print(f"\n📊 모델 성능:")
print(f"   학습 정확도: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"   학습 F1-Score: {train_f1:.4f}")
print(f"   테스트 정확도: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"   테스트 F1-Score: {test_f1:.4f}")

# 과적합 확인
overfit = train_accuracy - test_accuracy
print(f"\n   과적합 정도: {overfit:.4f} ({overfit*100:.2f}%p)")
if overfit < 0.05:
    print("   ✅ 과적합 없음 (5%p 미만)")
elif overfit < 0.10:
    print("   ⚠️ 약간 과적합 (5-10%p)")
else:
    print("   ❌ 과적합 심각 (10%p 이상)")

# 7. 상세 리포트
print(f"\n7. 클래스별 성능:")
print("\n" + classification_report(y_test, y_test_pred, 
                                   target_names=['여유', '보통', '혼잡', '매우혼잡']))

# 8. Confusion Matrix
print(f"\n8. Confusion Matrix:")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

# models 폴더 생성
import os
os.makedirs('models', exist_ok=True)

# 📊 혼동 행렬 시각화
print(f"\n   시각화 생성 중...")
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['여유', '보통', '혼잡', '매우혼잡'],
            yticklabels=['여유', '보통', '혼잡', '매우혼잡'],
            cbar_kws={'label': '샘플 수'})
plt.title('혼동 행렬 (Confusion Matrix)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('예측값', fontsize=12)
plt.ylabel('실제값', fontsize=12)

# 정확도를 각 셀에 추가
for i in range(4):
    for j in range(4):
        total = cm[i].sum()
        percentage = cm[i, j] / total * 100 if total > 0 else 0
        plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                ha='center', va='center', fontsize=9, color='gray')

plt.tight_layout()
plt.savefig('models/confusion_matrix_enhanced.png', dpi=300, bbox_inches='tight')
print(f"   ✅ 저장: models/confusion_matrix_enhanced.png")
plt.close()

# 9. Feature Importance
print(f"\n9. Feature Importance (상위 20개):")
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(20).to_string(index=False))

# 역 카테고리 피처 중요도 확인
if category_features:
    print(f"\n   🎯 역 카테고리 피처 중요도:")
    for feat in category_features:
        imp = feature_importance[feature_importance['feature'] == feat]['importance'].values
        if len(imp) > 0:
            print(f"      {feat}: {imp[0]:.4f}")

# 상호작용 피처 중요도 확인
if interaction_features:
    print(f"\n   ✨ 상호작용 피처 중요도:")
    for feat in interaction_features:
        imp = feature_importance[feature_importance['feature'] == feat]['importance'].values
        if len(imp) > 0:
            print(f"      {feat}: {imp[0]:.4f}")

# 📊 특성 중요도 시각화 (상위 25개)
print(f"\n   시각화 생성 중...")
top_n = 25
top_features = feature_importance.head(top_n)

plt.figure(figsize=(12, 10))

# 색상 구분: 역 카테고리(빨강), 상호작용(주황), 기본(청록)
colors = []
for feat in top_features['feature']:
    if '역_' in feat:
        colors.append('#FF6B6B')  # 빨강 - 역 카테고리
    elif 'interaction' in feat:
        colors.append('#FFA500')  # 주황 - 상호작용
    else:
        colors.append('#4ECDC4')  # 청록 - 기본

bars = plt.barh(range(top_n), top_features['importance'], color=colors)
plt.yticks(range(top_n), top_features['feature'])
plt.xlabel('중요도 (Importance)', fontsize=12)
plt.title(f'특성 중요도 Top {top_n}', fontsize=16, fontweight='bold', pad=20)
plt.gca().invert_yaxis()

# 범례 추가
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#FF6B6B', label='역 카테고리 피처'),
    Patch(facecolor='#FFA500', label='상호작용 피처'),
    Patch(facecolor='#4ECDC4', label='기본 피처')
]
plt.legend(handles=legend_elements, loc='lower right')

# 값 표시
for i, (idx, row) in enumerate(top_features.iterrows()):
    plt.text(row['importance'], i, f" {row['importance']:.4f}", 
             va='center', fontsize=9)

plt.tight_layout()
plt.savefig('models/feature_importance_enhanced.png', dpi=300, bbox_inches='tight')
print(f"   ✅ 저장: models/feature_importance_enhanced.png")
plt.close()

# 10. 학습 곡선
print(f"\n10. 학습 곡선 생성 중...")
print("   (이 과정은 몇 분 소요될 수 있습니다...)")

sample_size = min(100000, len(X_train))
if len(X_train) > sample_size:
    print(f"   샘플링: {len(X_train):,} → {sample_size:,}개")
    indices = np.random.choice(len(X_train), sample_size, replace=False)
    X_train_sample = X_train.iloc[indices]
    y_train_sample = y_train.iloc[indices]
else:
    X_train_sample = X_train
    y_train_sample = y_train

train_sizes = np.linspace(0.1, 1.0, 10)
train_sizes_abs, train_scores, val_scores = learning_curve(
    model, X_train_sample, y_train_sample,
    train_sizes=train_sizes,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes_abs, train_mean, 'o-', color='#4ECDC4', 
         label='학습 정확도', linewidth=2, markersize=8)
plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, 
                 alpha=0.2, color='#4ECDC4')

plt.plot(train_sizes_abs, val_mean, 'o-', color='#FF6B6B', 
         label='검증 정확도', linewidth=2, markersize=8)
plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, 
                 alpha=0.2, color='#FF6B6B')

plt.xlabel('학습 샘플 수', fontsize=12)
plt.ylabel('정확도', fontsize=12)
plt.title('학습 곡선 (Learning Curve)', fontsize=16, fontweight='bold', pad=20)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('models/learning_curve_enhanced.png', dpi=300, bbox_inches='tight')
print(f"   ✅ 저장: models/learning_curve_enhanced.png")
plt.close()

# 11. 모델 저장
print(f"\n11. 모델 저장 중...")

model_data = {
    'model': model,
    'feature_names': feature_columns,
    'model_type': 'RandomForestClassifier',
    'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'train_accuracy': train_accuracy,
    'test_accuracy': test_accuracy,
    'train_f1': train_f1,
    'test_f1': test_f1,
    'note': '역 카테고리 피처 + 상호작용 피처 포함 (업무지구, 상업지구, 환승역등급, 대학가, 주거지역)'
}

model_path = 'models/subway_congestion_model_enhanced.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"   ✅ 모델 저장 완료: {model_path}")

# 12. 최종 요약
print(f"\n" + "=" * 80)
print(f"✅ 학습 완료!")
print(f"=" * 80)
print(f"📊 최종 결과:")
print(f"   모델: Random Forest (100 trees)")
print(f"   피처 수: {len(feature_columns)}개")
print(f"   - 역 카테고리: {len(category_features)}개")
print(f"   - 상호작용: {len(interaction_features)}개")
print(f"   테스트 정확도: {test_accuracy*100:.2f}%")
print(f"   테스트 F1-Score: {test_f1:.4f}")
print(f"   과적합: {overfit*100:.2f}%p")
print(f"\n저장된 파일:")
print(f"   📁 모델: {model_path}")
print(f"   📊 혼동행렬: models/confusion_matrix_enhanced.png")
print(f"   📊 특성 중요도: models/feature_importance_enhanced.png")
print(f"   📊 학습 곡선: models/learning_curve_enhanced.png")
print(f"\n다음 단계: main.py에서 모델 경로를 '{model_path}'로 변경")
print("=" * 80)
