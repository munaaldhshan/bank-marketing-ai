"""Preprocessing utilities for the bank marketing project."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = 'y'


def identify_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Separate numerical and categorical features while excluding the target."""
    target_cols = {TARGET_COLUMN}
    feature_cols = [col for col in df.columns if col not in target_cols]
    numeric_cols = [
        col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col]) and col != 'duration'
    ]
    categorical_cols = [
        col for col in feature_cols if col not in numeric_cols and col != 'duration'
    ]
    return numeric_cols, categorical_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create a preprocessing pipeline for numeric and categorical variables."""
    numeric_cols, categorical_cols = identify_feature_types(X)

    transformers = []
    if numeric_cols:
        transformers.append(
            ('num', Pipeline([('scaler', StandardScaler())]), numeric_cols)
        )
    if categorical_cols:
        transformers.append(
            ('cat', OneHotEncoder(handle_unknown='ignore', drop=None), categorical_cols)
        )

    return ColumnTransformer(transformers=transformers, remainder='drop')


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic cleaning and return a dataset ready for train/test splitting."""
    cleaned = df.copy()

    if 'duration' in cleaned.columns:
        cleaned = cleaned.drop(columns=['duration'])

    if cleaned.duplicated().any():
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    for col in cleaned.columns:
        if cleaned[col].dtype == 'object':
            cleaned[col] = cleaned[col].replace({'unknown': np.nan, 'Unknown': np.nan, 'nan': np.nan})

    return cleaned


def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the cleaned dataset into features and target."""
    cleaned = clean_dataset(df)
    X = cleaned.drop(columns=[TARGET_COLUMN])
    y = cleaned[TARGET_COLUMN].copy()
    return X, y
