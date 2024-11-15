import pandas as pd
from titanic_preprocessing import preprocess_data
from titanic_model import TitanicModel

def create_submission(predictions, test_data_path='test.csv'):
    """제출 파일 생성"""
    test_df = pd.read_csv(test_data_path)
    
    submission_df = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': predictions.flatten()
    })
    
    submission_file = 'titanic_submission.csv'
    submission_df.to_csv(submission_file, index=False)
    
    print('\nSubmission File Statistics:')
    print(f'Total predictions: {len(predictions)}')
    print(f'Predicted survivors: {submission_df["Survived"].sum()}')
    print(f'Survival rate: {submission_df["Survived"].mean():.2%}')
    
    print('\nSubmission File Preview:')
    print(submission_df.head())
    
    print(f'\nSubmission file saved as: {submission_file}')
    
    return submission_df

def main():
    # 1. 데이터 전처리
    print("Loading and preprocessing data...")
    X_train_processed, X_test_processed, y_train, preprocessor = preprocess_data()
    
    # 2. 예측기 초기화 및 학습
    print("\nTraining models...")
    print(X_train_processed.shape)
    print(y_train.shape)
    
    predictor = TitanicModel(input_dim=X_train_processed.shape[1])
    fold_scores = predictor.train_with_kfold(X_train_processed, y_train)
    
    # 3. 예측 수행
    print("\nGenerating predictions...")
    predictions = predictor.predict(X_test_processed)
    
    # 4. 제출 파일 생성
    print("\nCreating submission file...")
    submission_df = create_submission(predictions)

if __name__ == "__main__":
    main()