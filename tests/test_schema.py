import pandas as pd

from src.schema import build_economic_snapshots, build_feature_schema, validate_record
from src.preprocessing import split_features_target


def test_build_feature_schema_captures_ranges_and_categories(synthetic_df):
    X, _ = split_features_target(synthetic_df)
    schema = build_feature_schema(X)

    assert schema['numeric']['age']['min'] == X['age'].min()
    assert schema['numeric']['age']['max'] == X['age'].max()
    assert set(schema['categorical']['job']) == set(X['job'].unique())
    # The 999 "never contacted" sentinel must not swamp the real pdays range.
    assert schema['pdays_if_contacted_before']['max'] < 999


def test_build_economic_snapshots_is_one_row_per_month(synthetic_df):
    X, _ = split_features_target(synthetic_df)
    snapshots = build_economic_snapshots(X)
    assert set(snapshots) == set(X['month'].unique())
    for month, values in snapshots.items():
        assert set(values) == {'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'}


def test_validate_record_flags_out_of_range_numeric_value():
    schema = {'numeric': {'age': {'min': 18, 'max': 90}}, 'categorical': {}}
    warnings = validate_record({'age': 150}, schema)
    assert len(warnings) == 1
    assert 'age' in warnings[0] and 'training range' in warnings[0]


def test_validate_record_flags_unseen_category():
    schema = {'numeric': {}, 'categorical': {'job': ['admin.', 'student']}}
    warnings = validate_record({'job': 'astronaut'}, schema)
    assert len(warnings) == 1
    assert 'astronaut' in warnings[0]


def test_validate_record_accepts_in_range_values_silently():
    schema = {'numeric': {'age': {'min': 18, 'max': 90}}, 'categorical': {'job': ['admin.']}}
    assert validate_record({'age': 40, 'job': 'admin.'}, schema) == []


def test_validate_record_ignores_missing_fields():
    schema = {'numeric': {'age': {'min': 18, 'max': 90}}, 'categorical': {}}
    assert validate_record({}, schema) == []
