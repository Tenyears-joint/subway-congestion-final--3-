import pandas as pd

# 데이터 로드
df = pd.read_csv('data/processed/subway_features_balanced.csv')

# 방법 1: 모든 컬럼 이름 출력
print("전체 컬럼 목록:")
print(df.columns.tolist())

# 방법 2: 컬럼 개수와 함께 출력
print(f"\n총 컬럼 수: {len(df.columns)}")
print("\n컬럼 목록:")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

# 방법 3: 데이터 타입과 함께 확인
print("\n컬럼 정보 (이름, 타입, 결측치):")
print(df.info())

# 방법 4: 각 컬럼의 샘플 데이터 확인
print("\n상위 5개 행:")
print(df.head())

# 방법 5: 컬럼별 기본 통계
print("\n기본 통계:")
print(df.describe())