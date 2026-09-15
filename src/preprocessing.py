"""Cleaning, target encoding and the one preprocessing definition every model uses."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import LEAKAGE_COLUMNS, TARGET


def clean_dataset(df: pd.DataFrame, drop_leakage: bool = True) -> pd.DataFrame:
    """Remove exact duplicate records, then (by default) the leakage columns.

    Duplicates are checked on the full record *before* `duration` is dropped;
    doing it afterwards merges ~1,800 distinct customers who only differ in
    call length. 'unknown' stays a category of its own on purpose: it carries
    signal (default='unknown' converts at 5% vs 13% for 'no'), and keeping it
    means cleaning behaves identically on pandas 2 and pandas 3.
    """
    cleaned = df.drop_duplicates().reset_index(drop=True)
    if drop_leakage:
        cleaned = cleaned.drop(columns=[col for col in LEAKAGE_COLUMNS if col in cleaned.columns])
    return cleaned


def encode_target(y: pd.Series) -> pd.Series:
    """Map yes/no labels to 1/0, failing loudly on anything else."""
    mapped = y.astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
    if mapped.isna().any():
        unexpected = sorted(y[mapped.isna()].astype(str).unique())
        raise ValueError(f'Unexpected target values: {unexpected}')
    return mapped.astype(int).rename(TARGET)


def split_features_target(df: pd.DataFrame, drop_leakage: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Clean the raw frame and return (X, y) with y encoded as 0/1."""
    cleaned = clean_dataset(df, drop_leakage=drop_leakage)
    return cleaned.drop(columns=[TARGET]), encode_target(cleaned[TARGET])


def identify_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split the feature columns into numeric and categorical lists."""
    features = [col for col in X.columns if col != TARGET]
    numeric = [col for col in features if pd.api.types.is_numeric_dtype(X[col])]
    categorical = [col for col in features if col not in numeric]
    return numeric, categorical


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Median-impute and scale numbers; mode-impute and one-hot encode categories.

    Living inside the saved pipeline, these steps run identically at training
    and prediction time. Unseen categories are encoded as all zeros, which is
    why the app validates inputs against the training schema first.
    """
    numeric, categorical = identify_feature_types(X)
    numeric_steps = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_steps = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # Dense output so that HistGradientBoosting can consume it.
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer(
        [('num', numeric_steps, numeric), ('cat', categorical_steps, categorical)],
        remainder='drop',
    )
