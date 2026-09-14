"""Utilities for loading and validating the bank marketing dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_DATA_PATHS: Iterable[str] = (
    'data/bank_marketing.csv',
    'data/bank-marketing.csv',
    'data/bank_marketing_campaign.csv',
    'data/bank-marketing-campaign.csv',
)


def find_dataset_path(base_dir: str | Path | None = None) -> Path:
    """Locate the dataset file in the project data folder."""
    base = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parents[1]
    for candidate in DEFAULT_DATA_PATHS:
        path = base / candidate
        if path.exists():
            return path

    # If no exact-named file is found, search the data folder for CSVs.
    data_dir = base / 'data'
    if data_dir.exists():
        csv_files = sorted(data_dir.glob('*.csv'))
        if csv_files:
            return csv_files[0]

    raise FileNotFoundError(
        'No dataset file was found in data/. Download the Kaggle CSV and place it in data/.'
    )


def load_bank_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the bank marketing dataset and return a DataFrame."""
    file_path = Path(path) if path is not None else find_dataset_path()

    # The Kaggle Bank Marketing CSV uses a semicolon delimiter, so we must
    # read it with the correct separator instead of assuming a default CSV.
    try:
        df = pd.read_csv(file_path, sep=';')
    except pd.errors.ParserError:
        df = pd.read_csv(file_path)

    if 'y' not in df.columns:
        raise ValueError("The dataset does not contain the target column 'y'.")
    return df


def target_summary(df: pd.DataFrame) -> pd.Series:
    """Return the distribution of the target variable."""
    return df['y'].value_counts(dropna=False)
