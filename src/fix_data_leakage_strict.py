"""
데이터 누출 완전 제거 - 의심스러운 모든 피처 제거
subway-congestion-prediction/src/fix_data_leakage_strict.py
"""

import pandas as pd
import os

def strict_feature_cleaning(input_path, output_path):
    """
    타겟과 직간접적으로 관련된 모든 피처 제거
    """
    print("="*70)
    print("🔧 엄격한 데이터 누출 제거")
    print("="*70)
    
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    print(f"\n📂 원본 데이터: {len(df.columns)}개 컬럼")
    
    # 🔥 엄격한 제거: 타겟과 관련된 모든 통계 피처
    cols_to_remove = [
        # 타겟 직접 관련
        '총승하차인원',
        '혼잡도',
        '역내_상대혼잡도',
        '혼잡도레벨_new',
        '혼잡도레벨_old',
        '혼잡도레벨_3class',
        
        # 🔥 타겟과 너무 직접적으로 연관된 통계 피처
        '승하차_1일전',       # 어제 데이터 = 거의 동일
        '승하차_7일전',       # 1주일 전 = 매우 유사
        '승하차_3일평균',     # 최근 평균 = 타겟과 직결
        '승하차_7일평균',     # 주간 평균 = 타겟과 직결
        '승하차_7일표준편차', # 변동성 = 타겟 정보 포함
        
        # 🔥 역/시간 평균도 제거 (타겟 계산에 사용됨)
        '역_평균승하차',      # 역 전체 평균
        '시간_평균승하차',    # 시간대 평균
        '호선_평균승하차',    # 호선 평균
        '역_최대승하차',      # 역 최대값
        '역_최소승하차',      # 역 최소값
        
        # 기타
        'Unnamed: 0',
    ]
    
    # 실제 존재하는 컬럼만
    existing = [col for col in cols_to_remove if col in df.columns]
    
    print(f"\n🗑️  제거할 컬럼 ({len(existing)}개):")
    for col in existing:
        print(f"   - {col}")
    
    # 제거
    df_clean = df.drop(columns=existing, errors='ignore')
    
    print(f"\n✅ 정리 완료!")
    print(f"   - 원본: {len(df.columns)}개")
    print(f"   - 최종: {len(df_clean.columns)}개")
    print(f"   - 제거: {len(existing)}개")
    
    # 남은 피처 확인
    print(f"\n📋 남은 피처 ({len(df_clean.columns)}개):")
    for i, col in enumerate(df_clean.columns, 1):
        if col == '혼잡도레벨':
            print(f"   {i:2d}. {col}  ← 타겟 변수")
        elif col in ['사용일자', '지하철역', '호선명']:
            print(f"   {i:2d}. {col}  ← 식별자 (제외됨)")
        else:
            print(f"   {i:2d}. {col}")
    
    # 승하차/혼잡도 관련 컬럼 확인
    suspicious = [col for col in df_clean.columns 
                  if any(keyword in col.lower() for keyword in ['승하차', '혼잡', '평균'])]
    suspicious = [col for col in suspicious if col != '혼잡도레벨']
    
    if suspicious:
        print(f"\n⚠️  경고: 여전히 의심스러운 컬럼:")
        for col in suspicious:
            print(f"   - {col}")
    else:
        print(f"\n✅ 의심스러운 통계 피처 모두 제거!")
    
    # 저장
    df_clean.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 저장 완료: {output_path}")
    
    return df_clean


def main():
    print("\n🚇 지하철 혼잡도 예측 - 엄격한 데이터 누출 제거")
    print("="*70)
    
    INPUT = 'data/processed/subway_features_balanced.csv'
    OUTPUT = 'data/processed/subway_features_strict.csv'
    
    df_clean = strict_feature_cleaning(INPUT, OUTPUT)
    
    print("\n" + "="*70)
    print("✅ 작업 완료!")
    print("="*70)
    
    print("\n📊 예상 결과:")
    print("   이제 정확도가 70~80% 정도로 낮아질 것입니다.")
    print("   이것이 정상입니다! 타겟 정보 없이 순수하게 예측하는 것입니다.")
    
    print("\n📋 다음 단계:")
    print("   1. 모델 재학습:")
    print("      python src/model_trainer_improved.py")
    print()
    print("   2. model_trainer_improved.py 수정:")
    print("      trainer.load_feature_data('subway_features_strict.csv')")
    print()
    print("   3. 예상 성능:")
    print("      - 정확도: 70~80% (정상!)")
    print("      - 특성 중요도 1위: 시간, 승차인원, 하차인원 등")


if __name__ == '__main__':
    main()
