import pickle
import pandas as pd
import numpy as np

# 모델 로드
with open('models/subway_congestion_model_final.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
feature_names = model_data['feature_names']
importances = model.feature_importances_

# 상위 10개
df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\n상위 10개 중요 피처:")
print(df.head(10).to_string(index=False))

# 시간 관련 피처 중요도
time_features = ['시간', '시간_sin', '시간_cos', '요일', '주말여부']
print(f"\n시간 관련 피처 중요도:")
for feat in time_features:
    if feat in df['feature'].values:
        imp = df[df['feature'] == feat]['importance'].values[0]
        print(f"  {feat}: {imp:.4f}")
