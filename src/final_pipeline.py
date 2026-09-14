"""Final pipeline creation utilities for the bank-marketing project."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import load_bank_data
from src.preprocessing import clean_dataset

TARGET_COLUMN = 'y'


def build_final_pipeline(df: pd.DataFrame, random_state: int = 42) -> Pipeline:
    """Train a production-ready logistic-regression pipeline excluding the leakage feature."""
    cleaned = clean_dataset(df)
    X = cleaned.drop(columns=[TARGET_COLUMN])
    y = cleaned[TARGET_COLUMN].str.lower().map({'yes': 1, 'no': 0}).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    numeric_cols = [col for col in X_train.columns if pd.api.types.is_numeric_dtype(X_train[col])]
    categorical_cols = [col for col in X_train.columns if col not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                'num',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                ]),
                numeric_cols,
            ),
            (
                'cat',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore')),
                ]),
                categorical_cols,
            ),
        ],
        remainder='drop',
    )

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=2000, class_weight='balanced')),
    ])
    model.fit(X_train, y_train)
    return model


def save_final_pipeline(model: Pipeline, output_path: str | Path = 'models/final_model.joblib') -> Path:
    """Persist the trained model artifact to disk."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    return destination


def train_and_save_final_pipeline(data_path: str | Path = 'data/bank_marketing.csv') -> Path:
    """Load the real dataset, train the final model, and save the artifact."""
    df = load_bank_data(data_path)
    model = build_final_pipeline(df)
    return save_final_pipeline(model)
