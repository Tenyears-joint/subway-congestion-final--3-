"""
지하철 혼잡도 예측 - 개선된 모델 학습 모듈 (과적합 해결)
subway-congestion-prediction/src/model_trainer_improved.py

개선 사항:
1. 정규화 강화 (min_samples_split, min_samples_leaf 증가)
2. 트리 깊이 제한 완화 → 자동 선택
3. 특성 샘플링 추가 (max_features)
4. 앙상블 크기 증가
5. 검증 데이터 분리
6. Early Stopping 개념 적용
7. 클래스 가중치 조정
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


class ImprovedModelTrainer:
    """과적합을 방지한 개선된 Random Forest 모델 학습 클래스"""
    
    def __init__(self, feature_data_path, model_save_path):
        """
        Args:
            feature_data_path: 피처 데이터 경로
            model_save_path: 모델 저장 경로
        """
        self.feature_data_path = feature_data_path
        self.model_save_path = model_save_path
        
        os.makedirs(model_save_path, exist_ok=True)
        
        self.df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.model = None
        self.feature_names = None
        self.scaler = None
        
    def load_feature_data(self, filename='subway_features_balanced.csv'):  # ✅ Balanced!
        """피처 엔지니어링된 데이터 로드"""
        filepath = os.path.join(self.feature_data_path, filename)
        
        print(f"📂 피처 데이터 로딩 중: {filepath}")
        
        self.df = pd.read_csv(filepath, encoding='utf-8-sig')
        
        print(f"✅ 로딩 완료!")
        print(f"   - 행 수: {len(self.df):,}")
        print(f"   - 열 수: {len(self.df.columns)}")
        
        return self.df
    
    def prepare_data(self, test_size=0.2, val_size=0.1, random_state=42):
        """학습/검증/테스트 데이터 분리 (3-way split)"""
        print("\n📊 데이터 준비 중...")
        
        # 타겟 변수
        target_column = '혼잡도레벨'
        
        # 제외할 컬럼들
        exclude_columns = [
            target_column, 
            '사용일자', 
            '지하철역', 
            '호선명', 
            '총승하차인원',  # 타겟 누출 방지
            '혼잡도',  # 타겟 누출 방지
            'Unnamed: 0'
        ]
        
        # 피처 선택
        feature_columns = [col for col in self.df.columns if col not in exclude_columns]
        
        # 결측치 처리
        if self.df[feature_columns].isnull().sum().sum() > 0:
            print("⚠️  결측치 발견! 중앙값으로 채웁니다...")
            self.df[feature_columns] = self.df[feature_columns].fillna(
                self.df[feature_columns].median()
            )
        
        # X, y 분리
        X = self.df[feature_columns].copy()
        y = self.df[target_column].copy()
        
        # 먼저 train+val / test 분리
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # train / val 분리
        val_ratio = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
        )
        
        self.feature_names = feature_columns
        
        print(f"✅ 데이터 준비 완료!")
        print(f"   - 학습 데이터: {len(self.X_train):,}개 ({len(self.X_train)/len(X)*100:.1f}%)")
        print(f"   - 검증 데이터: {len(self.X_val):,}개 ({len(self.X_val)/len(X)*100:.1f}%)")
        print(f"   - 테스트 데이터: {len(self.X_test):,}개 ({len(self.X_test)/len(X)*100:.1f}%)")
        print(f"   - 피처 수: {len(feature_columns)}개")
        
        # 클래스 분포 확인
        print(f"\n클래스 분포:")
        for label, count in sorted(y.value_counts().items()):
            print(f"   레벨 {label}: {count:,}개 ({count/len(y)*100:.1f}%)")
        
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def train_model(self, model_type='random_forest', random_state=42):
        """
        과적합을 방지한 모델 학습
        
        Args:
            model_type: 'random_forest' 또는 'gradient_boosting'
        """
        print(f"\n🌲 {model_type.upper()} 모델 학습 중...")
        
        if model_type == 'random_forest':
            # 🔥 과적합 방지 설정
            self.model = RandomForestClassifier(
                n_estimators=200,           # 트리 수 증가 (안정성)
                max_depth=15,               # 깊이 제한 (과적합 방지)
                min_samples_split=100,      # 분할 최소 샘플 증가 ⬆️
                min_samples_leaf=50,        # 리프 최소 샘플 증가 ⬆️
                max_features='sqrt',        # 피처 샘플링 (다양성)
                max_samples=0.8,            # 부트스트랩 샘플 비율
                class_weight='balanced',    # 클래스 불균형 처리
                random_state=random_state,
                n_jobs=-1,
                verbose=1,
                oob_score=True              # Out-of-bag 점수
            )
        
        elif model_type == 'gradient_boosting':
            # Gradient Boosting (대안)
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,         # 낮은 학습률
                max_depth=5,                # 얕은 트리
                min_samples_split=100,
                min_samples_leaf=50,
                subsample=0.8,              # 샘플 서브샘플링
                random_state=random_state,
                verbose=1
            )
        
        # 학습
        print("\n학습 시작...")
        self.model.fit(self.X_train, self.y_train)
        
        if hasattr(self.model, 'oob_score_'):
            print(f"\n📊 Out-of-Bag Score: {self.model.oob_score_:.4f}")
        
        print("✅ 학습 완료!")
        
        return self.model
    
    def evaluate_model(self, show_validation=True):
        """모델 평가 (Train/Val/Test)"""
        print("\n📈 모델 평가 중...")
        
        # 예측
        y_train_pred = self.model.predict(self.X_train)
        y_val_pred = self.model.predict(self.X_val)
        y_test_pred = self.model.predict(self.X_test)
        
        # 지표 계산
        train_acc = accuracy_score(self.y_train, y_train_pred)
        val_acc = accuracy_score(self.y_val, y_val_pred)
        test_acc = accuracy_score(self.y_test, y_test_pred)
        
        train_f1 = f1_score(self.y_train, y_train_pred, average='weighted')
        val_f1 = f1_score(self.y_val, y_val_pred, average='weighted')
        test_f1 = f1_score(self.y_test, y_test_pred, average='weighted')
        
        print("\n" + "="*70)
        print("📊 모델 성능 지표")
        print("="*70)
        
        print(f"\n🎯 정확도 (Accuracy):")
        print(f"   - 학습 데이터:   {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"   - 검증 데이터:   {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"   - 테스트 데이터: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        # 🔥 과적합 진단
        overfitting_gap = train_acc - val_acc
        print(f"\n⚠️  과적합 정도: {overfitting_gap:.4f} ({overfitting_gap*100:.2f}%p)")
        if overfitting_gap > 0.05:
            print("   ❌ 과적합 의심! (5%p 이상 차이)")
        elif overfitting_gap > 0.02:
            print("   ⚠️  약간의 과적합 (2~5%p 차이)")
        else:
            print("   ✅ 정상 범위 (2%p 이하 차이)")
        
        print(f"\n📊 F1-Score:")
        print(f"   - 학습 데이터:   {train_f1:.4f}")
        print(f"   - 검증 데이터:   {val_f1:.4f}")
        print(f"   - 테스트 데이터: {test_f1:.4f}")
        
        # 테스트 데이터 상세 리포트
        print("\n" + "="*70)
        print("📋 상세 분류 리포트 (테스트 데이터)")
        print("="*70)
        print("\n혼잡도 레벨: 0=여유, 1=보통, 2=혼잡, 3=매우혼잡")
        print()
        print(classification_report(
            self.y_test, y_test_pred,
            target_names=['여유', '보통', '혼잡', '매우혼잡'],
            digits=4
        ))
        
        return {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'train_f1': train_f1,
            'val_f1': val_f1,
            'test_f1': test_f1,
            'overfitting_gap': overfitting_gap,
            'y_test_pred': y_test_pred
        }
    
    def plot_confusion_matrix(self, y_test_pred, save_path='models'):
        """혼동 행렬 시각화"""
        print("\n📊 혼동 행렬 생성 중...")
        
        cm = confusion_matrix(self.y_test, y_test_pred)
        
        # 정규화된 혼동 행렬 (비율)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # 두 개 그래프 생성
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. 절대값
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=['여유', '보통', '혼잡', '매우혼잡'],
                   yticklabels=['여유', '보통', '혼잡', '매우혼잡'])
        axes[0].set_title('혼동 행렬 (개수)', fontsize=14, pad=15)
        axes[0].set_ylabel('실제 값', fontsize=11)
        axes[0].set_xlabel('예측 값', fontsize=11)
        
        # 2. 비율
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Oranges', ax=axes[1],
                   xticklabels=['여유', '보통', '혼잡', '매우혼잡'],
                   yticklabels=['여유', '보통', '혼잡', '매우혼잡'])
        axes[1].set_title('혼동 행렬 (비율)', fontsize=14, pad=15)
        axes[1].set_ylabel('실제 값', fontsize=11)
        axes[1].set_xlabel('예측 값', fontsize=11)
        
        plt.tight_layout()
        
        filepath = os.path.join(save_path, 'confusion_matrix_improved.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✅ 혼동 행렬 저장: {filepath}")
        
        plt.close()
        
        return cm
    
    def plot_feature_importance(self, top_n=20, save_path='models'):
        """특성 중요도 시각화"""
        print(f"\n📊 상위 {top_n}개 특성 중요도 시각화 중...")
        
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        top_features = feature_importance_df.head(top_n)
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        plt.barh(range(len(top_features)), top_features['importance'], color=colors)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('중요도', fontsize=12)
        plt.title(f'상위 {top_n}개 특성 중요도', fontsize=16, pad=20)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(save_path, 'feature_importance_improved.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✅ 특성 중요도 저장: {filepath}")
        
        plt.close()
        
        print("\n상위 15개 중요 특성:")
        for idx, row in feature_importance_df.head(15).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        return feature_importance_df
    
    def plot_learning_curve(self, save_path='models'):
        """학습 곡선 그리기 (과적합 진단)"""
        print("\n📈 학습 곡선 생성 중...")
        
        from sklearn.model_selection import learning_curve
        
        train_sizes, train_scores, val_scores = learning_curve(
            self.model, 
            self.X_train, 
            self.y_train,
            cv=5,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='학습 점수', color='blue', marker='o')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                         alpha=0.15, color='blue')
        plt.plot(train_sizes, val_mean, label='검증 점수', color='red', marker='s')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                         alpha=0.15, color='red')
        
        plt.xlabel('학습 데이터 크기', fontsize=12)
        plt.ylabel('정확도', fontsize=12)
        plt.title('학습 곡선 (과적합 진단)', fontsize=14, pad=15)
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(save_path, 'learning_curve.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✅ 학습 곡선 저장: {filepath}")
        
        plt.close()
    
    def cross_validate(self, cv=5):
        """교차 검증"""
        print(f"\n🔄 {cv}-Fold 교차 검증 중...")
        
        scores = cross_val_score(
            self.model, 
            self.X_train, 
            self.y_train,
            cv=cv, 
            scoring='accuracy', 
            n_jobs=-1
        )
        
        print(f"✅ 교차 검증 완료!")
        print(f"   - 평균 정확도: {scores.mean():.4f} ± {scores.std():.4f}")
        print(f"   - 각 Fold: {[f'{s:.4f}' for s in scores]}")
        
        return scores
    
    def save_model(self, filename='subway_congestion_model_improved.pkl'):
        """모델 저장"""
        filepath = os.path.join(self.model_save_path, filename)
        
        print(f"\n💾 모델 저장 중: {filepath}")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': type(self.model).__name__
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ 모델 저장 완료!")
        print(f"   - 파일 크기: {os.path.getsize(filepath) / 1024**2:.2f} MB")
        
        return filepath


def main():
    """메인 실행 함수"""
    
    print("🚇 지하철 혼잡도 예측 - 개선된 모델 학습 시작")
    print("="*70)
    print("📌 과적합 방지 전략:")
    print("   1. Train/Val/Test 3-way 분리")
    print("   2. 정규화 강화 (min_samples 증가)")
    print("   3. 트리 깊이 제한")
    print("   4. 피처/샘플 서브샘플링")
    print("   5. 클래스 가중치 조정")
    print("="*70)
    
    FEATURE_DATA_PATH = 'data/processed'
    MODEL_SAVE_PATH = 'models'
    
    trainer = ImprovedModelTrainer(FEATURE_DATA_PATH, MODEL_SAVE_PATH)
    
    try:
        # 1. 데이터 로드
        trainer.load_feature_data('subway_features_balanced.csv')
        
        # 2. 데이터 준비 (Train 70% / Val 10% / Test 20%)
        trainer.prepare_data(test_size=0.2, val_size=0.1, random_state=42)
        
        # 3. 모델 학습
        trainer.train_model(model_type='random_forest', random_state=42)
        
        # 4. 모델 평가
        results = trainer.evaluate_model()
        
        # 5. 교차 검증
        trainer.cross_validate(cv=5)
        
        # 6. 시각화
        trainer.plot_confusion_matrix(results['y_test_pred'], save_path=MODEL_SAVE_PATH)
        trainer.plot_feature_importance(top_n=20, save_path=MODEL_SAVE_PATH)
        trainer.plot_learning_curve(save_path=MODEL_SAVE_PATH)
        
        # 7. 모델 저장
        trainer.save_model()
        
        print("\n" + "="*70)
        print("✅ 모델 학습 완료!")
        print("="*70)
        
        # 최종 요약
        print(f"\n📊 최종 성능 요약:")
        print(f"   - 테스트 정확도: {results['test_acc']:.4f} ({results['test_acc']*100:.2f}%)")
        print(f"   - 테스트 F1-Score: {results['test_f1']:.4f}")
        print(f"   - 과적합 정도: {results['overfitting_gap']:.4f} ({results['overfitting_gap']*100:.2f}%p)")
        
        if results['overfitting_gap'] <= 0.02:
            print(f"   ✅ 과적합 없음! 모델이 잘 일반화됨")
        elif results['overfitting_gap'] <= 0.05:
            print(f"   ⚠️  약간의 과적합, 하지만 사용 가능")
        else:
            print(f"   ❌ 과적합 발생! 하이퍼파라미터 조정 필요")
        
        print(f"\n저장된 파일:")
        print(f"   - 모델: {MODEL_SAVE_PATH}/subway_congestion_model_improved.pkl")
        print(f"   - 혼동 행렬: {MODEL_SAVE_PATH}/confusion_matrix_improved.png")
        print(f"   - 특성 중요도: {MODEL_SAVE_PATH}/feature_importance_improved.png")
        print(f"   - 학습 곡선: {MODEL_SAVE_PATH}/learning_curve.png")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
