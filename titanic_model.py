import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import pandas as pd


class TitanicModel:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.trained_models = []
        self.feature_importance = None
        
    def create_model(self, learning_rate=0.001):
        """
        신경망 모델 생성
        
        모델 구조:
        1. 입력층 → Feature Importance 계산을 위한 특별 레이어
        2. 듀얼 패스 아키텍처:
           - 깊은 경로: 복잡한 패턴 학습
           - 얕은 경로: 기본적인 특성 학습
        3. 앙상블 결과를 위한 출력 결합
        
        개선사항:
        - 듀얼 패스 아키텍처 도입
        - Residual connections 추가
        - Advanced regularization 기법 적용
        - Feature importance 분석 기능
        - 학습률 스케줄링 최적화
        """
        inputs = layers.Input(shape=(self.input_dim,), name='input')
        
        # Feature Importance Layer
        feature_weights = layers.Dense(
            self.input_dim,
            activation='relu',
            kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),
            name='feature_importance'
        )(inputs)
        
        # Deep Path
        deep = self._create_deep_path(feature_weights)
        
        # Shallow Path
        shallow = self._create_shallow_path(feature_weights)
        
        # Combine Paths
        combined = layers.Concatenate(name='path_combine')([deep, shallow])
        
        # Final Dense Layers with Residual Connection
        x = layers.Dense(32, activation='relu', name='final_dense_1')(combined)
        x = layers.BatchNormalization(name='final_bn_1')(x)
        x = layers.Dropout(0.2, name='final_dropout_1')(x)
        
        #residual = layers.Dense(32, activation='relu')(combined)
        #x = layers.Add()([x, residual])
        
        # Output Layer with Attention
        #attention = layers.Dense(32, activation='tanh', name='attention')(x)
        #attention_weights = layers.Dense(1, activation='sigmoid', name='attention_weights')(attention)
        #weighted_features = layers.Multiply(name='weighted_features')([x, attention_weights])
        #
        #outputs = layers.Dense(1, activation='sigmoid', name='output')(weighted_features)
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
        # Model Creation
        model = models.Model(inputs=inputs, outputs=outputs, name='enhanced_titanic_model')
        
        # Advanced Optimizer Configuration
        optimizer = self._create_optimizer(learning_rate)
        
        # Compile with Additional Metrics
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(
            from_logits=False,
            label_smoothing=0.1  # label smoothing 추가
        ),
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                self._f1_score
            ]
        )
        
        return model

  
    def _create_deep_path(self, inputs):
        """심층 패스 생성 - 복잡한 패턴 학습용"""
        x = inputs
        
        for i, units in enumerate([128, 64, 32]):
            x = layers.Dense(
                units,
                kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),
                name=f'deep_dense_{i+1}'
            )(x)
            x = layers.BatchNormalization(name=f'deep_bn_{i+1}')(x)
            x = layers.Activation('relu', name=f'deep_relu_{i+1}')(x)
            x = layers.Dropout(0.3, name=f'deep_dropout_{i+1}')(x)
            
            # Add Residual Connection if shapes match
            if units == x.shape[-1]:
                residual = x
                x = layers.Add(name=f'deep_residual_{i+1}')([x, residual])
        
        return x
    
    def _create_shallow_path(self, inputs):
        """얕은 패스 생성 - 기본 특성 학습용"""
        x = layers.Dense(
            64,
            kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),
            name='shallow_dense_1'
        )(inputs)
        x = layers.BatchNormalization(name='shallow_bn_1')(x)
        x = layers.Activation('relu', name='shallow_relu_1')(x)
        return x
    
    def _create_optimizer(self, learning_rate):
        """Advanced Optimizer Configuration"""
        return tf.keras.optimizers.Adam(
            learning_rate=learning_rate,  # 직접 learning_rate 값을 사용
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=True
        )

    def _custom_loss(self, y_true, y_pred):
        """Custom Loss Function with Focal Loss"""
        gamma = 2.0  # focusing parameter
        alpha = 0.25  # class balance parameter
        
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        
        pt_1 = tf.clip_by_value(pt_1, 1e-9, 1.0)
        pt_0 = tf.clip_by_value(pt_0, 1e-9, 1.0)
        
        return -tf.reduce_mean(
            alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1) +
            (1-alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0)
        )
    
    def _f1_score(self, y_true, y_pred):
        """F1 Score Metric"""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # 예측값을 이진화 (필요한 경우)
        y_pred = tf.where(y_pred > 0.5, 1.0, 0.0)
        
        # True Positives, False Positives, False Negatives 계산
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))
        
        # epsilon을 사용하여 0으로 나누는 것을 방지
        epsilon = tf.keras.backend.epsilon()
        
        # Precision과 Recall 계산
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        
        # F1 Score 계산
        f1 = 2 * ((precision * recall) / (precision + recall + epsilon))
        
        return f1
    def train_with_kfold(self, X, y, n_splits=5, epochs=100, batch_size=32):
        """Enhanced K-fold Cross Validation Training"""
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_scores = []
        histories = []
        detailed_metrics = []
        self.trained_models = []
    
        
   
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
            print(f'\nFold {fold}/{n_splits}')
            print('-' * 50)

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # 모델 생성
            model = self.create_model()

            # Early Stopping 콜백
            callbacks = self._create_callbacks()

            # 모델 학습
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )

            # 예측 및 NaN 처리
            val_pred = model.predict(X_val)
            val_pred = np.nan_to_num(val_pred, nan=0.5)  # NaN을 0.5로 대체
            val_pred_binary = (val_pred > 0.5).astype(int)

            # 메트릭스 계산
            metrics = self._calculate_detailed_metrics(y_val, val_pred, val_pred_binary)
            detailed_metrics.append(metrics)

            # 결과 저장
            fold_scores.append(metrics['accuracy'])
            histories.append(history.history)
            self.trained_models.append(model)

            # 결과 출력
            self._print_fold_results(fold, metrics)

        return fold_scores, histories, detailed_metrics
    
    def _create_callbacks(self):
        """Enhanced Training Callbacks"""
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
    def _calculate_detailed_metrics(self, y_true, y_pred_prob, y_pred_binary):
        """Calculate Comprehensive Metrics"""
        y_pred_prob = np.nan_to_num(y_pred_prob, nan=0.5)  # NaN을 0.5로 대체
        y_pred_binary = np.nan_to_num(y_pred_binary, nan=0)  # NaN을 0으로 대체
    
        return {
            'accuracy': accuracy_score(y_true, y_pred_binary),
            'roc_auc': roc_auc_score(y_true, y_pred_prob),
            'confusion_matrix': confusion_matrix(y_true, y_pred_binary),
            'classification_report': classification_report(y_true, y_pred_binary, output_dict=True),
            'pr_auc': self._calculate_pr_auc(y_true, y_pred_prob)
        }
    
    def _calculate_pr_auc(self, y_true, y_pred_prob):
        """Calculate Precision-Recall AUC"""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
        return auc(recall, precision)
    
    def _print_fold_results(self, fold, metrics):
        """Print Detailed Fold Results"""
        print(f'\nFold {fold} Results:')
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"PR AUC: {metrics['pr_auc']:.4f}")
        print('\nClassification Report:')
        print(metrics['classification_report'])  # 수정된 부분

    
    def _print_overall_results(self, fold_scores, detailed_metrics):
        """Print Comprehensive Overall Results"""
        print('\nOverall K-fold Cross Validation Results:')
        print(f'Mean Accuracy: {np.mean(fold_scores):.4f} (±{np.std(fold_scores):.4f})')
        
        # Calculate and print average metrics across all folds
        avg_roc_auc = np.mean([m['roc_auc'] for m in detailed_metrics])
        avg_pr_auc = np.mean([m['pr_auc'] for m in detailed_metrics])
        print(f'Mean ROC AUC: {avg_roc_auc:.4f}')
        print(f'Mean PR AUC: {avg_pr_auc:.4f}')
    
    def _update_feature_importance(self, model, feature_names):
        """Calculate and Update Feature Importance"""
        feature_importance_layer = model.get_layer('feature_importance')
        weights = feature_importance_layer.get_weights()[0]
        importance_scores = np.mean(np.abs(weights), axis=1)
        
        if self.feature_importance is None:
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_scores
            })
        else:
            self.feature_importance['importance'] += importance_scores
            
    def get_feature_importance(self):
        """Get Normalized Feature Importance"""
        if self.feature_importance is not None:
            self.feature_importance['importance'] /= len(self.trained_models)
            return self.feature_importance.sort_values('importance', ascending=False)
        return None
    
    #def predict(self, X):
    #    """Ensemble Prediction with Uncertainty Estimation"""
    #    predictions = []
    #    for model in self.trained_models:
    #        pred = model.predict(X)
    #        predictions.append(pred)
    #    
    #    # Calculate mean and standard deviation of predictions
    #    mean_pred = np.mean(predictions, axis=0)
    #    std_pred = np.std(predictions, axis=0)
    #    
    #    return {
    #        'predictions': (mean_pred > 0.6).astype(int),
    #        'probabilities': mean_pred,
    #        'uncertainty': std_pred
    #    }
        
    def plot_training_history(self, histories):
        metrics = ['loss', 'accuracy']
        metric_names = ['Loss', 'Accuracy', 'AUC']  # 표시될 이름
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for metric, ax in zip(metrics, axes[:2]):  # loss, accuracy 처리
            for i, history in enumerate(histories):
                #print(f"Fold {i+1} metrics:", history.keys())
                ax.plot(history[metric], label=f'Train {metric} Fold {i+1}')
                ax.plot(history[f'val_{metric}'], label=f'Val {metric} Fold {i+1}')
            ax.set_title(f'Model {metric.capitalize()}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
        
        # AUC 처리 (동적 키 검색)
        auc_ax = axes[2]
        for i, history in enumerate(histories):
            auc_key = next((key for key in history.keys() if key.startswith('auc')), None)
            val_auc_key = next((key for key in history.keys() if key.startswith('val_auc')), None)
            
            if auc_key and val_auc_key:
                auc_ax.plot(history[auc_key], label=f'Train AUC Fold {i+1}')
                auc_ax.plot(history[val_auc_key], label=f'Val AUC Fold {i+1}')
        
        auc_ax.set_title('Model AUC')
        auc_ax.set_xlabel('Epoch')
        auc_ax.set_ylabel('AUC')
        auc_ax.legend()
        
        plt.tight_layout()
        plt.show()
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """단일 모델 학습"""
        model = self.create_model()
        
        # 클래스 가중치 계산
        class_weights = dict(enumerate(compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )))
        
        # 콜백 설정
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
        
       # 모델 학습
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            callbacks=callbacks,
            class_weight=class_weights,
            **kwargs
        )
        
        return model, history
    def predict(self, X_test):
        """앙상블 예측 수행"""
        if not self.trained_models:
            raise ValueError("No trained models available. Please train the models first.")
        
        predictions = []
        
        for i, model in enumerate(self.trained_models, 1):
            print(f'\nGenerating predictions from model {i}')
            pred = model.predict(X_test)
            predictions.append(pred)
        
        # 앙상블 예측 (평균)
        ensemble_predictions = np.mean(predictions, axis=0)
        
        # 임계값 0.5 적용
        final_predictions = (ensemble_predictions > 0.5).astype(int)
        
        return final_predictions
    
    
#앙상블 예측 정교화 작업 필요할 시.
#def predict(self, X_test, threshold=0.3, method='weighted', uncertainty_threshold=0.15):
#    """
#    앙상블 예측 수행
    
#    Parameters:
#    -----------
#    X_test : array-like
#        예측할 데이터
#    threshold : float, default=0.3
#        예측 임계값
#    method : str, default='weighted'
#        앙상블 방법 ('simple', 'weighted', 'dynamic')
#    uncertainty_threshold : float, default=0.15
#        불확실성 임계값
    
#    Returns:
#    --------
#    dict: 예측 결과와 관련 통계
#    """
#    if not self.trained_models:
#        raise ValueError("No trained models available. Please train the models first.")
    
#    print(f"\nGenerating predictions with {method} ensemble method...")
#    print(f"Using threshold: {threshold}")
    
#    predictions = []
#    model_weights = []
    
#    # 각 모델별 예측 수행
#    for i, model in enumerate(self.trained_models, 1):
#        print(f'Processing model {i}/{len(self.trained_models)}...')
#        pred = model.predict(X_test, verbose=0)
#        predictions.append(pred)
        
#        # 모델 가중치 계산 (validation 성능 기반)
#        if hasattr(model, 'history') and model.history is not None:
#            val_accuracy = max(model.history.history.get('val_accuracy', [0]))
#            model_weights.append(val_accuracy)
#        else:
#            model_weights.append(1.0)
    
#    predictions = np.array(predictions)
    
#    # 앙상블 방법 선택
#    if method == 'simple':
#        # 단순 평균
#        ensemble_predictions = np.mean(predictions, axis=0)
#    elif method == 'weighted':
#        # 가중치 기반 평균
#        model_weights = np.array(model_weights) / sum(model_weights)
#        ensemble_predictions = np.average(predictions, axis=0, weights=model_weights)
#    elif method == 'dynamic':
#        # 예측 확실성 기반 동적 가중치
#        prediction_std = np.std(predictions, axis=0)
#        certainty_weights = 1 / (1 + prediction_std)
#        ensemble_predictions = np.average(predictions, axis=0, weights=certainty_weights)
#    else:
#        raise ValueError(f"Unknown ensemble method: {method}")
    
#    # 불확실성 추정
#    prediction_std = np.std(predictions, axis=0)
#    uncertainty_mask = prediction_std > uncertainty_threshold
    
#    # 최종 예측
#    final_predictions = (ensemble_predictions > threshold).astype(int)
    
#    # 통계 수집
#    results = {
#        'predictions': final_predictions,
#        'probabilities': ensemble_predictions,
#        'uncertainty': prediction_std,
#        'statistics': {
#            'total_predictions': len(final_predictions),
#            'predicted_survivors': np.sum(final_predictions),
#            'survival_rate': np.mean(final_predictions),
#            'high_uncertainty_count': np.sum(uncertainty_mask),
#            'high_uncertainty_rate': np.mean(uncertainty_mask),
#            'probability_distribution': {
#                'mean': np.mean(ensemble_predictions),
#                'std': np.std(ensemble_predictions),
#                'min': np.min(ensemble_predictions),
#                'max': np.max(ensemble_predictions)
#            }
#        }
#    }
    
#    # 결과 출력
#    print("\nPrediction Summary:")
#    print(f"Total predictions: {results['statistics']['total_predictions']}")
#    print(f"Predicted survivors: {results['statistics']['predicted_survivors']}")
#    print(f"Survival rate: {results['statistics']['survival_rate']:.2%}")
#    print(f"High uncertainty predictions: {results['statistics']['high_uncertainty_count']} "
#          f"({results['statistics']['high_uncertainty_rate']:.2%})")
    
#    # 신뢰도 구간별 분포
#    probability_ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
#    print("\nProbability Distribution:")
#    for start, end in probability_ranges:
#        count = np.sum((ensemble_predictions >= start) & (ensemble_predictions < end))
#        print(f"{start:.1f} - {end:.1f}: {count} predictions ({count/len(ensemble_predictions):.2%})")
    
#    return results


def train_and_evaluate_model(X_train_processed, y_train):
    # 모델 인스턴스 생성
    model = TitanicModel(input_dim=X_train_processed.shape[1])
    
    # K-fold 교차 검증으로 학습
    fold_scores, histories, confusion_matrices = model.train_with_kfold(
        X_train_processed, 
        y_train,
        n_splits=5,
        epochs=100,
        batch_size=32
    )
    
    # 학습 과정 시각화
    #model.plot_training_history(histories)
    
    return fold_scores, histories, confusion_matrices

    
if __name__ == "__main__":
    # 이전 전처리 코드에서 생성한 데이터 사용
    from titanic_preprocessing import preprocess_data
    X_train_processed, X_test_processed, y_train, preprocessor = preprocess_data()
    
    # 모델 학습 및 평가
    fold_scores, histories, confusion_matrices = train_and_evaluate_model(X_train_processed, y_train)