"""Modeling utilities for the bank marketing project."""

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


def map_target_to_binary(y: pd.Series) -> pd.Series:
    """Map the original yes/no labels to binary 1/0 values."""
    mapped = y.astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
    if mapped.isna().any():
        unknown = mapped.isna().sum()
        raise ValueError(f'Unexpected target values found: {unknown} non-binary labels were present.')
    return mapped.astype(int)


def prepare_model_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix and binary target while excluding the leakage feature."""
    cleaned = clean_dataset(df)
    X = cleaned.drop(columns=[TARGET_COLUMN])
    y = map_target_to_binary(cleaned[TARGET_COLUMN])
    return X, y


def build_model_pipeline(X: pd.DataFrame) -> Pipeline:
    """Create a production-style preprocessing + logistic regression pipeline."""
    numeric_cols = [
        col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])
    ]
    categorical_cols = [
        col for col in X.columns if col not in numeric_cols
    ]

    transformers: list[tuple[str, object, list[str]]] = []

    if numeric_cols:
        numeric_pipeline = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
            ]
        )
        transformers.append(('num', numeric_pipeline, numeric_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore')),
            ]
        )
        transformers.append(('cat', categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

    model = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=2000, class_weight='balanced')),
        ]
    )
    return model


def train_baseline_model(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train a baseline logistic regression model and optionally persist it."""
    X, y = prepare_model_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = build_model_pipeline(X_train)
    model.fit(X_train, y_train)

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, destination)

    return model, X_train, X_test, y_train, y_test


def load_and_train_default_model(data_path: str | Path | None = None) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the project dataset and train the baseline model with default settings."""
    path = Path(data_path) if data_path is not None else Path('data/bank_marketing.csv')
    df = load_bank_data(path)
    return train_baseline_model(df, output_path='models/final_model.joblib')
