# Titanic Survival Prediction Project

## Project Overview
This project participates in Kaggle's Titanic competition, using machine learning to predict passenger survival. We implement a deep learning model that predicts survival probabilities based on various passenger characteristics.

## Project Structure
```
project_folder/
├── titanic_preprocessing.py  # Data preprocessing module
├── titanic_model.py         # Deep learning model definition
├── main.py                  # Execution script
├── train.csv               # Training data
└── test.csv                # Test data
```

## Key Features

### 1. Data Preprocessing (`titanic_preprocessing.py`)
- Missing value handling
- Feature engineering:
  - Family size calculation
  - Title extraction and grouping
  - Cabin number processing
  - Fare binning
  - Age grouping
  - Family survival rate calculation

### 2. Model Implementation (`titanic_model.py`)
- Deep learning architecture:
  - Input layer: Matches feature dimensions
  - Hidden layer 1: 64 units + BatchNorm + Dropout(0.3)
  - Hidden layer 2: 32 units + BatchNorm + Dropout(0.2)
  - Output layer: Sigmoid activation
- Class imbalance handling
- K-fold cross-validation
- Ensemble predictions

### 3. Execution and Analysis (`main.py`)
- Data loading and preprocessing
- Model training and prediction
- Detailed analysis report generation
- Kaggle submission file creation

## Feature Importance
Correlation analysis reveals the most significant features for survival:
1. FamilySurvivalRate (0.91)
2. Title_Mr (-0.55)
3. Sex_male (-0.54)
4. Pclass_Sex_3_male (-0.41)
5. Title_Mrs (0.34)

## Performance Metrics
- Cross-validation average accuracy: 96.52% (±1.52%)
- Predicted survival rate: ~32% (Actual Titanic survival rate: 31.7%)

## Installation and Execution

1. Install Required Libraries
```bash
pip install tensorflow pandas numpy scikit-learn
```

2. Data Preparation
- Place train.csv and test.csv files in the project folder

3. Execution
```bash
python main.py
```

## Output Files
The program generates:
1. `titanic_submission.csv`: File for Kaggle submission
2. `titanic_predictions_detailed.csv`: Detailed analysis results

## Analysis Features
- Survival analysis by gender
- Survival analysis by passenger class
- Survival analysis by age group
- Survival analysis by port of embarkation
- Survival analysis by fare category

## Future Improvements
1. Enhanced Feature Engineering
   - More detailed family relationship analysis
   - Additional cabin location-based features
2. Model Architecture Optimization
   - Hyperparameter tuning
   - Diverse model ensemble
3. Prediction Threshold Optimization

## Reference Information
- Actual Titanic Statistics:
  - Total passengers: 2,224
  - Survivors: 706
  - Actual survival rate: 31.7%
- Kaggle Dataset:
  - Training data: 891 passengers
  - Test data: 418 passengers

## Technical Details

### Preprocessing Pipeline
```python
def preprocess_data():
    # Load data
    # Handle missing values
    # Feature engineering
    # Scale numerical features
    # Encode categorical features
    return X_train_processed, X_test_processed, y_train
```

### Model Architecture
```python
def create_model():
    # Input layer
    # Hidden layers with BatchNormalization and Dropout
    # Output layer with sigmoid activation
    # Compile with binary crossentropy loss
    return model
```

### Training Process
```python
def train_with_kfold():
    # 5-fold cross validation
    # Class weight balancing
    # Early stopping
    # Model ensemble
    return trained_models
```

## Performance Analysis
- Training accuracy: >95%
- Validation accuracy: ~96%
- Predictions align with historical survival patterns
- Balanced predictions across different passenger classes

## Dependencies
- Python 3.8+
- TensorFlow 2.x
- Pandas
- NumPy
- Scikit-learn

## License
This project is licensed under the MIT License.

## Acknowledgments
- Based on Kaggle's Titanic competition dataset
- Implements deep learning techniques for binary classification
- Utilizes ensemble methods for improved prediction accuracy

## Contact
For any queries regarding this project, please open an issue in the repository.