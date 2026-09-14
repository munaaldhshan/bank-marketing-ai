from pathlib import Path

import pandas as pd

from src.data_loader import load_bank_data
from src.preprocessing import clean_dataset, identify_feature_types, prepare_features_and_target


def test_data_loading_requires_target_column(tmp_path):
    bad_file = tmp_path / 'bad_bank_marketing.csv'
    pd.DataFrame({'age': [30, 40], 'job': ['admin', 'teacher']}).to_csv(
        bad_file,
        index=False,
        sep=';',
    )

    try:
        load_bank_data(bad_file)
        raise AssertionError('Expected a ValueError when the target column is missing.')
    except ValueError as exc:
        assert "target column 'y'" in str(exc)


def test_clean_dataset_removes_duration_and_duplicates():
    df = pd.DataFrame(
        {
            'age': [30, 30],
            'job': ['admin', 'admin'],
            'duration': [100, 100],
            'y': ['yes', 'yes'],
        }
    )
    cleaned = clean_dataset(df)
    assert 'duration' not in cleaned.columns
    assert cleaned.shape[0] == 1


def test_prepare_features_and_target_separates_target():
    df = pd.DataFrame(
        {
            'age': [30, 40],
            'job': ['admin', 'teacher'],
            'duration': [120, 200],
            'y': ['yes', 'no'],
        }
    )
    X, y = prepare_features_and_target(df)
    assert list(X.columns) == ['age', 'job']
    assert y.name == 'y'
    assert set(y.unique()) == {'yes', 'no'}


def test_identify_feature_types_excludes_target():
    df = pd.DataFrame({
        'age': [30, 40],
        'job': ['admin', 'teacher'],
        'y': ['yes', 'no'],
    })
    numeric_cols, categorical_cols = identify_feature_types(df)
    assert numeric_cols == ['age']
    assert categorical_cols == ['job']
