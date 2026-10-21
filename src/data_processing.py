import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

TARGET = 'Exam_Score'

def get_feature_info(df, target=TARGET):
    if target not in df.columns:
        raise ValueError(f'找不到目標欄位：{target}')
    X = df.drop(columns=[target]).copy()
    y = pd.to_numeric(df[target], errors='coerce')
    num = X.select_dtypes(include='number').columns.tolist()
    cat = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    unsupported = X.columns.difference(num + cat).tolist()
    if unsupported:
        raise ValueError(f'不支援的欄位型態: {unsupported}')
    return X, y, num, cat

def split_data(df, target=TARGET):
    X, y, num, cat = get_feature_info(df, target)
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]
    if len(X) < 40:
        raise ValueError('有效資料筆數至少需要 40 筆')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, num, cat

def build_preprocessor(numeric_features, categorical_features):
    transformers = []
    if numeric_features:
        transformers.append(('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), numeric_features))
    if categorical_features:
        transformers.append(('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features))
    if not transformers:
        raise ValueError('資料集沒有可用的特徵欄位')
    return ColumnTransformer(transformers=transformers, remainder='drop')
