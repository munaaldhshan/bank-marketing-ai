"""Training utilities for bank marketing models."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import load_bank_data
from src.preprocessing import build_preprocessor, clean_dataset


def create_train_test_split(
    df: pd.DataFrame,
    target_col: str = 'y',
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible train/test split with optional stratification."""
    cleaned = clean_dataset(df)
    X = cleaned.drop(columns=[target_col])
    y = cleaned[target_col].copy()

    stratify_param = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param,
    )
    return X_train, X_test, y_train, y_test


def save_training_data(df: pd.DataFrame, output_dir: str | Path = 'data') -> None:
    """Save a cleaned copy of the raw dataset for analysis notebooks."""
    cleaned = clean_dataset(df)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(target / 'bank_marketing_cleaned.csv', index=False)
