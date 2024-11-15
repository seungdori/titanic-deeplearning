import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
current_dir = os.getcwd()
save_path = os.path.join(current_dir, 'submission')
if not os.path.exists(save_path):
    os.makedirs(save_path)
# 한글 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False   # 마이너스 기호 깨짐 방지

#plt.style.use('seaborn-v0_8-darkgrid')  # 유효한 seaborn 스타일 중 하나 사용

# Windows의 경우 다음과 같이 폰트 설정을 변경하세요:
"""
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows용 한글 폰트
"""

# Linux의 경우:
"""
plt.rcParams['font.family'] = 'NanumGothic'  # Linux용 한글 폰트
"""

NUMERIC_FEATURES = ['Age', 'Fare', 'FamilySize', 'FarePerPerson', 
                   'TicketGroupSize']

CATEGORICAL_FEATURES_ONEHOT = ['Sex', 'Embarked', 'Deck', 'FamilyType',
                              'FareRange', 'AgeGroup', 'Age_Pclass']

CATEGORICAL_FEATURES_TARGET = ['Title', 'Pclass_Sex']

BINARY_FEATURES = ['IsAlone']

def get_readable_feature_names(feature_names):
    """특성 이름을 읽기 쉬운 형태로 변환"""
    mapping = {
        # 성별 관련
        'cat_onehot__Sex_male': '남성',
        'cat_onehot__Sex_female': '여성',
        
        # 승선 항구 관련
        'cat_onehot__Embarked_Q': '퀸즈타운 승선',
        'cat_onehot__Embarked_S': '사우샘프턴 승선',
        'cat_onehot__Embarked_C': '셰르부르 승선',
        
        # 객실 데크 관련
        'cat_onehot__Deck_DE': '중층 데크(D,E)',
        'cat_onehot__Deck_FG': '하층 데크(F,G)',
        'cat_onehot__Deck_missing': '데크 정보 없음',
        
        # 가족 유형 관련
        'cat_onehot__FamilyType_Large_Family': '대가족(5인 이상)',
        'cat_onehot__FamilyType_Single': '1인 승객',
        'cat_onehot__FamilyType_Small_Family': '소규모 가족(2-4인)',
        
        # 요금 범위 관련
        'cat_onehot__FareRange_Very_Low': '최저가 티켓',
        'cat_onehot__FareRange_Low': '저가 티켓',
        'cat_onehot__FareRange_Medium': '중가 티켓',
        'cat_onehot__FareRange_High': '고가 티켓',
        'cat_onehot__FareRange_Very_High': '최고가 티켓',
        
        # 나이 그룹 관련
        'cat_onehot__AgeGroup_Child': '어린이(0-12세)',
        'cat_onehot__AgeGroup_Teen': '청소년(13-20세)',
        'cat_onehot__AgeGroup_Young_Adult': '청년(21-30세)',
        'cat_onehot__AgeGroup_Adult': '중년(31-50세)',
        'cat_onehot__AgeGroup_Senior': '장년/노년(51세 이상)',
        
        # 나이와 객실 등급 조합
        'cat_onehot__Age_Pclass_Child_1': '어린이-1등실',
        'cat_onehot__Age_Pclass_Child_2': '어린이-2등실',
        'cat_onehot__Age_Pclass_Child_3': '어린이-3등실',
        'cat_onehot__Age_Pclass_Teen_1': '청소년-1등실',
        'cat_onehot__Age_Pclass_Teen_2': '청소년-2등실',
        'cat_onehot__Age_Pclass_Teen_3': '청소년-3등실',
        'cat_onehot__Age_Pclass_Young_Adult_1': '청년-1등실',
        'cat_onehot__Age_Pclass_Young_Adult_2': '청년-2등실',
        'cat_onehot__Age_Pclass_Young_Adult_3': '청년-3등실',
        'cat_onehot__Age_Pclass_Adult_1': '중년-1등실',
        'cat_onehot__Age_Pclass_Adult_2': '중년-2등실',
        'cat_onehot__Age_Pclass_Adult_3': '중년-3등실',
        'cat_onehot__Age_Pclass_Senior_1': '장년/노년-1등실',
        'cat_onehot__Age_Pclass_Senior_2': '장년/노년-2등실',
        'cat_onehot__Age_Pclass_Senior_3': '장년/노년-3등실',
        
        # 수치형 특성
        'num__Age': '나이',
        'num__Fare': '티켓 요금',
        'num__FamilySize': '가족 규모',
        'num__FarePerPerson': '1인당 요금',
        'num__TicketGroupSize': '티켓 그룹 크기',
        'num__FamilySurvivalRate': '가족 생존율',
        
        # 이진 특성
        'bin__IsAlone': '혼자 탑승',
        
        'cat_onehot__Deck_ABC': 'ABC 데크',
        # 타겟 인코딩 특성
        'cat_target__0': '직위 기반 생존율',
        'cat_target__1': '객실등급+성별 기반 생존율',
        'cat_target__2': '티켓번호 기반 생존율'
    }
    
    #XGB 
    
    readable_names = []
    for feature in feature_names:
        if feature in mapping:
            readable_names.append(mapping[feature])
        else:
            readable_names.append(feature)
    
    return readable_names

def load_data():
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    return train_data, test_data

def process_ticket_prefix(df):
    """티켓 접두사를 처리하는 함수"""
    df['TicketPrefix'] = df['Ticket'].str.extract(r'([A-Za-z]+)', expand=False)
    df['TicketPrefix'] = df['TicketPrefix'].fillna('NUM')
    
    # 빈도가 낮은 접두사(20개 미만)를 'RARE'로 변경
    prefix_counts = df['TicketPrefix'].value_counts()
    rare_prefixes = prefix_counts[prefix_counts < 20].index
    df['TicketPrefix'] = df['TicketPrefix'].apply(lambda x: 'RARE' if x in rare_prefixes else x)
    
    return df


def handle_missing_values(df):
    """개선된 결측치 처리 함수"""
    # Age 결측치: KNN Imputer 사용
    age_imputer = KNNImputer(n_neighbors=5)
    age_features = ['Pclass', 'SibSp', 'Parch', 'Fare']
    df['Age'] = age_imputer.fit_transform(df[['Age'] + age_features])[:, 0]
    
    # Fare 결측치: 같은 Pclass의 중앙값으로 대체
    df['Fare'] = df.groupby('Pclass')['Fare'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # Cabin 결측치: 더 세분화된 카테고리로 대체
    df['Cabin'] = df['Cabin'].fillna('Unknown')
    df['Deck'] = df['Cabin'].str[0]
    deck_mapping = {
        'A': 'ABC', 'B': 'ABC', 'C': 'ABC',
        'D': 'DE', 'E': 'DE',
        'F': 'FG', 'G': 'FG',
        'Unknown': 'Unknown'
    }
    df['Deck'] = df['Deck'].map(deck_mapping)
    
    # Embarked 결측치: 최빈값으로 대체
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    return df


def feature_engineering(df):
    # === 기존 특성 ===
    # 가족 크기
    df = handle_missing_values(df)
    
    # 가족 관련 특성
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['FamilyType'] = pd.cut(
        df['FamilySize'],
        bins=[-np.inf, 1, 4, np.inf],
        labels=['Single', 'Small_Family', 'Large_Family']
    )
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # 요금 관련 특성
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    df['FareRange'] = pd.qcut(
        df['Fare'],
        q=5,
        labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High']
    )
    
    # 나이 관련 특성
    df['AgeGroup'] = pd.cut(
        df['Age'],
        bins=[0, 12, 20, 30, 50, np.inf],
        labels=['Child', 'Teen', 'Young_Adult', 'Adult', 'Senior']
    )
    
    # === 새로운 특성 추가 ===
    # 이름 관련 특성
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']
    df.loc[df['Title'].isin(rare_titles), 'Title'] = 'Rare'
    df['Ticket'] = df['Ticket'].str.extract(r'(\d+)', expand=False).astype(float)
    # 티켓 관련 특성
    ticket_counts = df['Ticket'].value_counts()
    df['TicketGroupSize'] = df['Ticket'].map(ticket_counts)
    
    # 복합 특성
    df['Pclass_Sex'] = df['Pclass'].astype(str) + '_' + df['Sex']
    df['Age_Pclass'] = df['AgeGroup'].astype(str) + '_' + df['Pclass'].astype(str)
    
    return df


def create_preprocessing_pipeline(X_train=None, y_train=None):
    # 수치형 변수 전처리
    # 수치형 변수 전처리
    numeric_features = ['Age', 'Fare', 'FamilySize', 'FarePerPerson', 
                       'TicketGroupSize']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())  # StandardScaler 대신 RobustScaler 사용
    ])
    
    # 일반 범주형 변수 전처리
    categorical_features_onehot = ['Sex', 'Embarked', 'Deck', 'FamilyType',
                                 'FareRange', 'AgeGroup', 'Age_Pclass']
    categorical_transformer_onehot = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))  # drop='first' 제거
])
    
        # 높은 카디널리티 특성을 위한 타겟 인코딩
    categorical_features_target = ['Title', 'Pclass_Sex']
    categorical_transformer_target = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('target', TargetEncoder(handle_unknown='value'))
    ])
    
    # 이진 특성
    binary_features = ['IsAlone']
    binary_transformer = 'passthrough'
   # 전체 전처리 파이프라인
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat_onehot', categorical_transformer_onehot, CATEGORICAL_FEATURES_ONEHOT),
            ('cat_target', categorical_transformer_target, CATEGORICAL_FEATURES_TARGET),
            ('bin', binary_transformer, BINARY_FEATURES)
        ],
        remainder='drop'
    )
    # 특성 선택 추가
    if X_train is not None and y_train is not None:
        selector = SelectKBest(score_func=mutual_info_classif, k='all')
        preprocessor = Pipeline([
            ('preprocessor', preprocessor),
            ('selector', selector)
        ])
    
    return preprocessor

def preprocess_data():
    """개선된 전처리 함수"""
    # 데이터 로드
    train_data, test_data = load_data()
    
    # 특성 엔지니어링
    train_data = feature_engineering(train_data)
    test_data = feature_engineering(test_data)
    
    # 타겟 변수 분리
    y_train = train_data['Survived']
    
    # 사용하지 않을 컬럼 제거
    columns_to_drop = ['Survived', 'PassengerId', 'Name', 'Ticket', 'SibSp', 
                      'Parch', 'Cabin']
    
    X_train = train_data.drop(columns_to_drop, axis=1, errors='ignore')
    X_test = test_data.drop([col for col in columns_to_drop if col not in ['Survived']], 
                           axis=1, errors='ignore')
     # 전처리 파이프라인 생성 및 적용
    preprocessor = create_preprocessing_pipeline(X_train, y_train)
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 특성 이름 가져오기
    if hasattr(preprocessor, 'named_steps'):
        feature_names = preprocessor.named_steps['preprocessor'].get_feature_names_out()
    else:
        feature_names = preprocessor.get_feature_names_out()
    
    # 읽기 쉬운 특성 이름으로 변환
    readable_names = get_readable_feature_names(feature_names)
    
    # numpy array를 DataFrame으로 변환
    X_train_processed = pd.DataFrame(X_train_processed, columns=readable_names)
    X_test_processed = pd.DataFrame(X_test_processed, columns=readable_names)
    
    # 특성 중요도 계산 및 시각화 (선택적)
    if hasattr(preprocessor, 'named_steps') and hasattr(preprocessor.named_steps.get('selector', None), 'scores_'):
        feature_scores = pd.DataFrame({
            'Feature': readable_names,
            'Score': preprocessor.named_steps['selector'].scores_
        })
        feature_scores = feature_scores.sort_values('Score', ascending=False)
        print("\nTop 10 most important features and their scores:")
        print(feature_scores.head(10))
        
        print("\nTop 10 features statistics:")
        print(X_train_processed[feature_scores.head(10)['Feature']].describe())
    
    return X_train_processed, X_test_processed, y_train, preprocessor


def visualize_feature_importance(X_train_processed, y_train, preprocessor):
    """특성 중요도 시각화"""
    plt.figure(figsize=(12, 6))
    
    # 특성 중요도 계산
    feature_names = X_train_processed.columns
    if hasattr(preprocessor, 'named_steps') and hasattr(preprocessor.named_steps.get('selector', None), 'scores_'):
        scores = preprocessor.named_steps['selector'].scores_
    else:
        # mutual_info_classif를 직접 사용
        scores = mutual_info_classif(X_train_processed, y_train)
    
    # 특성 중요도를 데이터프레임으로 변환
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': scores
    }).sort_values('Importance', ascending=True)
    
    # 상위 15개 특성만 선택
    importance_df = importance_df.tail(30)
    
    # 수평 막대 그래프 생성
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.xlabel('Mutual Information Score')
    plt.ylabel('Features')
    plt.title('중요도 높은 특징')
    plt.tight_layout()
    
    # 그리드 추가
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    filename = os.path.join(save_path, 'Important Features.png')
    plt.savefig(filename, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()

def visualize_correlation_matrix(X_train_processed, y_train):
    """개선된 상관관계 행렬 시각화"""
    # y_train을 포함한 데이터프레임 생성
    data_with_target = X_train_processed.copy()
    data_with_target['생존여부'] = y_train
    
    # 상위 10개 특성 선택
    correlations = data_with_target.corrwith(data_with_target['생존여부']).abs().sort_values(ascending=False)
    top_features = correlations.head(11).index.tolist()  # '생존여부' 포함
    
    # 상관관계 행렬 계산
    correlation_matrix = data_with_target[top_features].corr()
    
    # 시각화
    plt.figure(figsize=(15, 12))
    
    # 히트맵 생성
    ax = sns.heatmap(correlation_matrix,
                     annot=True,  # 상관계수 표시
                     cmap='coolwarm',  # 색상 맵
                     center=0,  # 0을 중심으로 색상 대칭
                     fmt='.2f',  # 소수점 2자리까지 표시
                     square=True,  # 정사각형 형태로 표시
                     annot_kws={'size': 10},  # 글자 크기 조정
                     cbar_kws={'label': '상관계수'})  # 컬러바 레이블 추가
    
    # 제목과 부제목 추가
    plt.title('생존과 주요 특성들 간의 상관관계', pad=20, size=15)
    plt.figtext(0.02, 0.95, 
                '* 빨간색(1.0)에 가까울수록 강한 양의 상관관계 (생존 정보와 관계성 높음)\n'
                '* 파란색(-1.0)에 가까울수록 강한 음의 상관관계 (생존 정보와 관계성 낮음)\n'
                '* 흰색(0)에 가까울수록 상관관계가 약함', 
                fontsize=12)
    
    # 눈금 레이블 회전
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 레이아웃 조정
    plt.tight_layout()
    
    
    # 설명 텍스트를 위한 새로운 figure 생성

    filename = os.path.join(save_path, 'correlation_matrix.png')
    plt.savefig(filename, 
                bbox_inches='tight', 
                dpi=300,
                facecolor='white')
    
    print(f"Correlation matrix saved to {filename}")
    plt.close()




def visualize_feature_distributions(X_train_processed, y_train):
    """주요 특성들의 분포 시각화"""
    # 상위 6개 특성 선택
    correlations = X_train_processed.apply(lambda x: abs(x.corr(y_train)))
    top_features = correlations.sort_values(ascending=False).head(6).index
    
    # 서브플롯 생성
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, feature in enumerate(top_features):
        # 생존/사망에 따른 분포 비교
        sns.kdeplot(data=pd.DataFrame({
            'Feature': X_train_processed[feature],
            '생존': y_train
        }), x='Feature', hue='생존', ax=axes[idx])
        
        axes[idx].set_title(f'분포 : {feature}')
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('밀도')
    
    plt.tight_layout()
    # 파일로 저장
    filename = os.path.join(save_path, 'feature_distributions.png')
    plt.savefig(filename, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
def visualize_top_and_bottom_correlations(X_train_processed, y_train, save_path='.', top_n=10):
    """상관관계가 높은/낮은 특성을 막대그래프로 시각화"""
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # y_train을 포함한 데이터프레임 생성
    data_with_target = X_train_processed.copy()
    data_with_target['생존여부'] = y_train
    
    # 상관계수 계산
    correlations = data_with_target.corrwith(data_with_target['생존여부']).sort_values(ascending=False)
    
    # 상위 top_n 특성과 하위 top_n 특성 선택
    top_correlations = correlations.head(top_n + 1).drop('생존여부')
    bottom_correlations = correlations.tail(top_n)
  
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle('상관관계가 높은/낮은 특성들', fontsize=16, y=1.02)
    
    # 상위 특성 막대그래프
    axes[0].barh(top_correlations.index, top_correlations.values, color='skyblue')
    axes[0].set_title(f'생존 가장 높은 특성 (Top {top_n})', fontsize=14)
    axes[0].invert_yaxis()  # 높은 값을 위로
    axes[0].set_xlabel('상관계수')
    axes[0].set_ylabel('특성')
    
    # 하위 특성 막대그래프
    axes[1].barh(bottom_correlations.index, bottom_correlations.values, color='salmon')
    axes[1].set_title(f'상관관계가 가장 낮은 특성 (Bottom {top_n})', fontsize=14)
    axes[1].invert_yaxis()  # 높은 값을 위로
    axes[1].set_xlabel('상관계수')
    
    # 레이아웃 조정
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # 저장
    filename = os.path.join(save_path, 'correlation_barcharts.png')
    plt.savefig(filename, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"Correlation barcharts saved to {filename}")
    
    return top_correlations, bottom_correlations


    
def visualize_survival_rates(X_train_processed, y_train):
    """범주형 특성들의 생존율 시각화"""
    # 범주형으로 보이는 특성들 선택
    categorical_features = [col for col in X_train_processed.columns 
                          if len(X_train_processed[col].unique()) < 10][:6]
    
    # 서브플롯 생성
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # 색상 설정 - 부드러운 파스텔톤으로 변경
    colors = ['#AED6F1', '#F5B7B1']  # 연한 파랑, 연한 빨강
    
    for idx, feature in enumerate(categorical_features):
        # 각 카테고리별 생존율 계산
        survival_data = pd.DataFrame({
            'Feature': X_train_processed[feature],
            'Status': y_train.map({0: '사망', 1: '생존'})
        })
        
        # 각 카테고리의 생존/사망 비율 계산
        survival_counts = pd.crosstab(survival_data['Feature'], 
                                    survival_data['Status'], 
                                    normalize='index') * 100
        
        # 막대 그래프 생성
        ax = survival_counts.plot(kind='bar', 
                                ax=axes[idx], 
                                color=colors,
                                width=0.8)
        
        # 그래프 스타일링
        axes[idx].set_title(f'{feature} 생존/사망 비율', 
                          fontsize=12, 
                          pad=20)
        axes[idx].set_ylabel('비율 (%)', fontsize=10)
        axes[idx].set_xlabel('')
        
        # 범례 스타일링
        axes[idx].legend(title='상태', 
                        bbox_to_anchor=(1.0, 1.0),
                        fontsize=9)
        
        # 격자 추가
        axes[idx].grid(axis='y', linestyle='--', alpha=0.3)
        
        # y축 범위 설정 (0-100%)
        axes[idx].set_ylim(0, 100)
        
        # 막대 위에 비율 표시
        for container in axes[idx].containers:
            axes[idx].bar_label(container, 
                              fmt='%.1f%%', 
                              padding=3, 
                              fontsize=9)
        
        # x축 레이블 회전 - 수정된 부분
        plt.setp(axes[idx].get_xticklabels(), 
                rotation=45, 
                ha='right',
                fontsize=9)
    
    # 전체 타이틀 설정
    plt.suptitle('특성별 생존/사망 비율 분석', 
                fontsize=16, 
                y=1.05,
                fontweight='bold')
    
    # 설명 텍스트 추가
    fig.text(0.02, 0.98, 
             '* 각 막대는 해당 특성 카테고리의 전체 승객 중 생존/사망 비율을 나타냅니다.\n'
             '* 연한 파란색: 생존한 승객의 비율\n'
             '* 연한 붉은색: 사망한 승객의 비율', 
             fontsize=10, 
             va='top',
             fontfamily='NanumGothic')
    
    # 레이아웃 조정
    plt.tight_layout()
    filename = os.path.join(save_path, 'survival_rate.png')
    plt.savefig(filename, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"Visualization saved to {filename}")
    
    
if __name__ == "__main__":
    X_train_processed, X_test_processed, y_train, preprocessor = preprocess_data()
    # 경로가 없으면 생성

    print("\nTraining set shape:", X_train_processed.shape)
    print("Test set shape:", X_test_processed.shape)
    
    print("\nProcessed features:")
    for col in X_train_processed.columns:
        print(f"- {col}")
        
    # 상관관계 분석
    correlation_with_target = pd.DataFrame({
        'Feature': X_train_processed.columns,
        'Correlation': [X_train_processed[col].corr(y_train) for col in X_train_processed.columns]
    }).sort_values('Correlation', key=abs, ascending=False)
    
    print("\nTop 10 features by correlation with target:")
    print(correlation_with_target.head(10))
    # 시각화
    #plt.style.use('seaborn')  # 시각화 스타일 설정
    
    print("Generating visualizations...")
    
    # 1. 특성 중요도 시각화
    print("\n1. Feature Importance Plot")
    visualize_feature_importance(X_train_processed, y_train, preprocessor)
    
    # 2. 상관관계 행렬 시각화
    print("\n2. Correlation Matrix")
    visualize_correlation_matrix(X_train_processed, y_train)
    
    print("\3. Correlation Analysis")
    visualize_top_and_bottom_correlations(X_train_processed, y_train, save_path=save_path, top_n=10)    
    # 4. 주요 특성 분포 시각화
    print("\n4. Feature Distributions")
    visualize_feature_distributions(X_train_processed, y_train)
    
    # 5. 범주형 특성의 생존율 시각화
    print("\n5. Survival Rates by Categories")
    visualize_survival_rates(X_train_processed, y_train)
    
    print("\nVisualization complete!")
    