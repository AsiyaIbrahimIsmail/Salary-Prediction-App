"""Train a simple regression model on `salary_data_cleaned.csv` and save `model.joblib`.

Usage: run this from the project folder where `salary_data_cleaned.csv` lives.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib


DATA_PATH = os.path.join(os.path.dirname(__file__), "salary_data_cleaned.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df


def prepare_xy(df):
    # Determine target
    if 'avg_salary' in df.columns:
        y = df['avg_salary']
    elif 'min_salary' in df.columns and 'max_salary' in df.columns:
        y = (df['min_salary'] + df['max_salary']) / 2
    else:
        raise ValueError('No salary target column found (avg_salary or min/max)')

    # Select a small, robust set of features available in the dataset
    candidate_features = ['Rating', 'age', 'python_yn', 'R_yn', 'spark', 'aws', 'excel', 'job_state']
    features = [c for c in candidate_features if c in df.columns]
    X = df[features].copy()

    # Convert flag columns to numeric if present
    for col in ['python_yn', 'R_yn', 'spark', 'aws', 'excel']:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(int)

    return X, y


def build_pipeline(X):
    numeric_features = [c for c in X.columns if X[c].dtype.kind in 'biufc' and c != 'job_state']
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Use sparse_output for newer scikit-learn versions; fall back to dense if needed
    try:
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    pipeline = Pipeline([
        ('pre', preprocessor),
        ('model', Ridge())
    ])
    return pipeline


def train_and_save(X, y, model_path=MODEL_PATH):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = build_pipeline(X_train)

    param_grid = {'model__alpha': [0.1, 1.0, 10.0, 50.0, 100.0]}
    search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    search.fit(X_train, y_train)

    best = search.best_estimator_
    preds = best.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"Best params: {search.best_params_}")
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test MAE: {mae:.2f}")
    print(f"Test R2: {r2:.4f}")

    joblib.dump(best, model_path)
    print(f"Saved model to: {model_path}")


def main():
    print("Loading data...")
    df = load_data()
    print(f"Rows: {len(df)} Columns: {len(df.columns)}")
    X, y = prepare_xy(df)
    print(f"Using features: {list(X.columns)}")
    train_and_save(X, y)


if __name__ == '__main__':
    main()
