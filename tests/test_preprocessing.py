import pandas as pd
import pytest

from src.data_loader import load_bank_data
from src.preprocessing import clean_dataset, identify_feature_types, split_features_target


def test_data_loading_requires_target_column(tmp_path):
    bad_file = tmp_path / 'bad_bank_marketing.csv'
    pd.DataFrame({'age': [30, 40], 'job': ['admin', 'teacher']}).to_csv(bad_file, index=False, sep=';')

    with pytest.raises(ValueError, match="target column 'y'"):
        load_bank_data(bad_file)


def test_clean_dataset_removes_duration_and_duplicates():
    df = pd.DataFrame({'age': [30, 30], 'job': ['admin', 'admin'], 'duration': [100, 100], 'y': ['yes', 'yes']})
    cleaned = clean_dataset(df)
    assert 'duration' not in cleaned.columns
    assert cleaned.shape[0] == 1


def test_clean_dataset_can_keep_duration_for_the_leakage_benchmark():
    df = pd.DataFrame({'age': [30, 40], 'duration': [100, 200], 'y': ['yes', 'no']})
    cleaned = clean_dataset(df, drop_leakage=False)
    assert 'duration' in cleaned.columns


def test_clean_dataset_drops_duplicates_before_removing_duration():
    # Same profile, different call length: these are two distinct customer contacts, not
    # duplicate rows, and must not collapse into one just because `duration` is dropped later.
    df = pd.DataFrame({'age': [30, 30], 'duration': [100, 999], 'y': ['yes', 'yes']})
    cleaned = clean_dataset(df)
    assert cleaned.shape[0] == 2


def test_unknown_category_is_preserved_not_treated_as_missing():
    # 'unknown' carries real signal (e.g. default='unknown' converts very differently from
    # default='no'), and preserving it as its own category is also pandas-version independent.
    df = pd.DataFrame({'age': [30, 40], 'job': ['admin', 'unknown'], 'y': ['yes', 'no']})
    cleaned = clean_dataset(df)
    assert cleaned['job'].tolist() == ['admin', 'unknown']
    assert not cleaned['job'].isna().any()


def test_split_features_target_separates_and_encodes_target():
    df = pd.DataFrame({'age': [30, 40], 'job': ['admin', 'teacher'], 'duration': [120, 200], 'y': ['yes', 'no']})
    X, y = split_features_target(df)
    assert list(X.columns) == ['age', 'job']
    assert y.name == 'y'
    assert set(y.unique()) == {0, 1}
    assert y.tolist() == [1, 0]


def test_split_features_target_rejects_unexpected_labels():
    df = pd.DataFrame({'age': [30, 40], 'y': ['yes', 'maybe']})
    with pytest.raises(ValueError, match='Unexpected target values'):
        split_features_target(df)


def test_identify_feature_types_splits_numeric_and_categorical():
    df = pd.DataFrame({'age': [30, 40], 'job': ['admin', 'teacher'], 'y': ['yes', 'no']})
    numeric_cols, categorical_cols = identify_feature_types(df.drop(columns=['y']))
    assert numeric_cols == ['age']
    assert categorical_cols == ['job']
