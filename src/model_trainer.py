"""
지하철 혼잡도 예측 - 모델 학습 및 평가 모듈
subway-congestion-prediction/src/model_trainer.py

주요 기능:
1. Random Forest 모델 학습
2. 하이퍼파라미터 튜닝
3. 모델 평가 (정확도, F1-score 등)
4. 특성 중요도 분석
5. 학습된 모델 저장
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


class ModelTrainer:
    """Random Forest 모델 학습 클래스"""
    
    def __init__(self, feature_data_path, model_save_path):
        """
        Args:
            feature_data_path: 피처 데이터 경로
            model_save_path: 모델 저장 경로
        """
        self.feature_data_path = feature_data_path
        self.model_save_path = model_save_path
        
        # 폴더가 없으면 생성
        os.makedirs(model_save_path, exist_ok=True)
        
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.feature_names = None
        
    def load_feature_data(self, filename='subway_features.csv'):
        """피처 엔지니어링된 데이터 로드"""
        filepath = os.path.join(self.feature_data_path, filename)
        
        print(f"📂 피처 데이터 로딩 중: {filepath}")
        
        self.df = pd.read_csv(filepath, encoding='utf-8-sig')
        
        print(f"✅ 로딩 완료!")
        print(f"   - 행 수: {len(self.df):,}")
        print(f"   - 열 수: {len(self.df.columns)}")
        
        return self.df
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """학습/테스트 데이터 분리"""
        print("\n📊 데이터 준비 중...")
        
        # 타겟 변수
        target_column = '혼잡도레벨'
        
        # 학습에 사용하지 않을 컬럼들
        exclude_columns = [
            target_column, '사용일자', '지하철역', '호선명', 
            '총승하차인원', '혼잡도'  # 타겟과 직접 관련된 컬럼 제외
        ]
        
        # 피처 컬럼 선택
        feature_columns = [col for col in self.df.columns if col not in exclude_columns]
        
        # 결측치 확인 및 처리
        if self.df[feature_columns].isnull().sum().sum() > 0:
            print("⚠️  결측치 발견! 0으로 채웁니다...")
            self.df[feature_columns] = self.df[feature_columns].fillna(0)
        
        # X, y 분리
        X = self.df[feature_columns]
        y = self.df[target_column]
        
        # 학습/테스트 데이터 분리
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.feature_names = feature_columns
        
        print(f"✅ 데이터 준비 완료!")
        print(f"   - 학습 데이터: {len(self.X_train):,}개")
        print(f"   - 테스트 데이터: {len(self.X_test):,}개")
        print(f"   - 피처 수: {len(feature_columns)}개")
        print(f"\n   피처 목록:")
        for i, col in enumerate(feature_columns, 1):
            print(f"   {i}. {col}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self, n_estimators=100, max_depth=20, random_state=42):
        """Random Forest 모델 학습"""
        print("\n🌲 Random Forest 모델 학습 중...")
        print(f"   - n_estimators: {n_estimators}")
        print(f"   - max_depth: {max_depth}")
        
        # Random Forest 분류 모델 생성
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,  # 모든 CPU 코어 사용
            verbose=1
        )
        
        # 학습
        print("\n학습 시작...")
        self.model.fit(self.X_train, self.y_train)
        
        print("✅ 학습 완료!")
        
        return self.model
    
    def evaluate_model(self):
        """모델 평가"""
        print("\n📈 모델 평가 중...")
        
        # 예측
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        # 정확도
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        
        # F1-score
        train_f1 = f1_score(self.y_train, y_train_pred, average='weighted')
        test_f1 = f1_score(self.y_test, y_test_pred, average='weighted')
        
        # 정밀도 (Precision)
        train_precision = precision_score(self.y_train, y_train_pred, average='weighted')
        test_precision = precision_score(self.y_test, y_test_pred, average='weighted')
        
        # 재현율 (Recall)
        train_recall = recall_score(self.y_train, y_train_pred, average='weighted')
        test_recall = recall_score(self.y_test, y_test_pred, average='weighted')
        
        print("\n" + "="*60)
        print("📊 모델 성능 지표")
        print("="*60)
        
        print(f"\n🎯 정확도 (Accuracy):")
        print(f"   - 학습 데이터: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"   - 테스트 데이터: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        print(f"\n📊 F1-Score:")
        print(f"   - 학습 데이터: {train_f1:.4f}")
        print(f"   - 테스트 데이터: {test_f1:.4f}")
        
        print(f"\n🎯 정밀도 (Precision):")
        print(f"   - 학습 데이터: {train_precision:.4f}")
        print(f"   - 테스트 데이터: {test_precision:.4f}")
        
        print(f"\n🎯 재현율 (Recall):")
        print(f"   - 학습 데이터: {train_recall:.4f}")
        print(f"   - 테스트 데이터: {test_recall:.4f}")
        
        # 분류 리포트
        print("\n" + "="*60)
        print("📋 상세 분류 리포트 (테스트 데이터)")
        print("="*60)
        print("\n혼잡도 레벨:")
        print("0: 여유, 1: 보통, 2: 혼잡, 3: 매우혼잡")
        print()
        print(classification_report(self.y_test, y_test_pred, 
                                   target_names=['여유', '보통', '혼잡', '매우혼잡']))
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'y_test_pred': y_test_pred
        }
    
    def plot_confusion_matrix(self, y_test_pred, save_path='models'):
        """혼동 행렬(Confusion Matrix) 시각화"""
        print("\n📊 혼동 행렬 생성 중...")
        
        # 혼동 행렬 계산
        cm = confusion_matrix(self.y_test, y_test_pred)
        
        # 시각화
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['여유', '보통', '혼잡', '매우혼잡'],
                   yticklabels=['여유', '보통', '혼잡', '매우혼잡'])
        plt.title('혼동 행렬 (Confusion Matrix)', fontsize=16, pad=20)
        plt.ylabel('실제 값', fontsize=12)
        plt.xlabel('예측 값', fontsize=12)
        plt.tight_layout()
        
        # 저장
        filepath = os.path.join(save_path, 'confusion_matrix.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✅ 혼동 행렬 저장: {filepath}")
        
        plt.close()
        
        return cm
    
    def plot_feature_importance(self, top_n=20, save_path='models'):
        """특성 중요도(Feature Importance) 시각화"""
        print(f"\n📊 상위 {top_n}개 특성 중요도 시각화 중...")
        
        # 특성 중요도 추출
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # 상위 N개만 선택
        top_features = feature_importance_df.head(top_n)
        
        # 시각화
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('중요도', fontsize=12)
        plt.title(f'상위 {top_n}개 특성 중요도', fontsize=16, pad=20)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # 저장
        filepath = os.path.join(save_path, 'feature_importance.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✅ 특성 중요도 저장: {filepath}")
        
        plt.close()
        
        # 상위 10개 출력
        print("\n상위 10개 중요 특성:")
        for i, row in feature_importance_df.head(10).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        return feature_importance_df
    
    def cross_validate(self, cv=5):
        """교차 검증"""
        print(f"\n🔄 {cv}-Fold 교차 검증 중...")
        
        scores = cross_val_score(self.model, self.X_train, self.y_train, 
                                cv=cv, scoring='accuracy', n_jobs=-1)
        
        print(f"✅ 교차 검증 완료!")
        print(f"   - 평균 정확도: {scores.mean():.4f} ({scores.mean()*100:.2f}%)")
        print(f"   - 표준편차: {scores.std():.4f}")
        print(f"   - 각 Fold 점수: {scores}")
        
        return scores
    
    def save_model(self, filename='subway_congestion_model.pkl'):
        """학습된 모델 저장"""
        filepath = os.path.join(self.model_save_path, filename)
        
        print(f"\n💾 모델 저장 중: {filepath}")
        
        # 모델과 피처 정보를 함께 저장
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ 모델 저장 완료!")
        print(f"   - 파일 크기: {os.path.getsize(filepath) / 1024**2:.2f} MB")
        
        return filepath
    
    def hyperparameter_tuning(self, param_grid=None):
        """하이퍼파라미터 튜닝 (선택사항)"""
        print("\n🔧 하이퍼파라미터 튜닝 중...")
        print("⚠️  이 작업은 시간이 오래 걸릴 수 있습니다.")
        
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [10, 20, 30],
                'min_samples_split': [5, 10, 15],
                'min_samples_leaf': [2, 5, 10]
            }
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            param_grid,
            cv=3,
            scoring='accuracy',
            verbose=2,
            n_jobs=-1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print("\n✅ 튜닝 완료!")
        print(f"   - 최적 파라미터: {grid_search.best_params_}")
        print(f"   - 최적 점수: {grid_search.best_score_:.4f}")
        
        self.model = grid_search.best_estimator_
        
        return grid_search.best_params_


def main():
    """메인 실행 함수"""
    
    print("🚇 지하철 혼잡도 예측 - 모델 학습 시작")
    print("="*60)
    
    # 경로 설정
    FEATURE_DATA_PATH = 'data/processed'
    MODEL_SAVE_PATH = 'models'
    
    # 모델 트레이너 객체 생성
    trainer = ModelTrainer(FEATURE_DATA_PATH, MODEL_SAVE_PATH)
    
    try:
        # 1. 데이터 로드
        trainer.load_feature_data()
        
        # 2. 데이터 준비
        trainer.prepare_data(test_size=0.2, random_state=42)
        
        # 3. 모델 학습
        trainer.train_model(n_estimators=100, max_depth=20, random_state=42)
        
        # 4. 모델 평가
        results = trainer.evaluate_model()
        
        # 5. 교차 검증
        trainer.cross_validate(cv=5)
        
        # 6. 혼동 행렬 시각화
        trainer.plot_confusion_matrix(results['y_test_pred'], save_path=MODEL_SAVE_PATH)
        
        # 7. 특성 중요도 시각화
        trainer.plot_feature_importance(top_n=20, save_path=MODEL_SAVE_PATH)
        
        # 8. 모델 저장
        trainer.save_model()
        
        print("\n" + "="*60)
        print("✅ 모델 학습 완료!")
        print("="*60)
        print(f"\n저장된 파일:")
        print(f"   - 모델: {MODEL_SAVE_PATH}/subway_congestion_model.pkl")
        print(f"   - 혼동 행렬: {MODEL_SAVE_PATH}/confusion_matrix.png")
        print(f"   - 특성 중요도: {MODEL_SAVE_PATH}/feature_importance.png")
        
        # 하이퍼파라미터 튜닝 (선택사항 - 주석 처리)
        # print("\n🔧 하이퍼파라미터 튜닝을 시작하시겠습니까? (시간이 오래 걸립니다)")
        # trainer.hyperparameter_tuning()
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
