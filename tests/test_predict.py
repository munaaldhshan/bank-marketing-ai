import pytest
from sklearn.linear_model import LogisticRegression

from src.modeling import build_pipeline
from src.predict import load_model, predict_customer
from src.preprocessing import split_features_target
from src.schema import build_feature_schema


def test_load_model_raises_a_helpful_error_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match='python -m src.train'):
        load_model(tmp_path / 'no_such_model.joblib')


def test_predict_customer_returns_probability_label_and_warnings(synthetic_df):
    X, y = split_features_target(synthetic_df)
    model = build_pipeline(LogisticRegression(max_iter=1000), X)
    model.fit(X, y)
    schema = build_feature_schema(X)

    record = X.iloc[0].to_dict()
    probability, label, warnings = predict_customer(model, record, schema, threshold=0.5)

    assert 0.0 <= probability <= 1.0
    assert label in {'yes', 'no'}
    assert label == ('yes' if probability >= 0.5 else 'no')
    assert warnings == []  # a real training row must not trip its own schema


def test_predict_customer_threshold_changes_the_label(synthetic_df):
    X, y = split_features_target(synthetic_df)
    model = build_pipeline(LogisticRegression(max_iter=1000), X)
    model.fit(X, y)
    record = X.iloc[0].to_dict()
    assert predict_customer(model, record, threshold=0.0)[1] == 'yes'  # every probability clears 0.0
    assert predict_customer(model, record, threshold=1.0)[1] == 'no'   # nothing clears 1.0


def test_predict_customer_tolerates_missing_fields(synthetic_df):
    X, y = split_features_target(synthetic_df)
    model = build_pipeline(LogisticRegression(max_iter=1000), X)
    model.fit(X, y)

    probability, label, _ = predict_customer(model, {'age': 40})  # everything else missing
    assert 0.0 <= probability <= 1.0


def test_predict_customer_without_schema_returns_no_warnings(synthetic_df):
    X, y = split_features_target(synthetic_df)
    model = build_pipeline(LogisticRegression(max_iter=1000), X)
    model.fit(X, y)
    _, _, warnings = predict_customer(model, {'age': 9999})  # would warn if a schema were passed
    assert warnings == []
