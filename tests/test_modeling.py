import pandas as pd

from src.modeling import build_pipeline, candidate_models
from src.preprocessing import split_features_target


def _toy_df():
    return pd.DataFrame({
        'age': [25, 40, 60, 35, 50, 70, 22, 65],
        'job': ['student', 'admin', 'retired', 'admin', 'teacher', 'retired', 'student', 'retired'],
        'housing': ['yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no'],
        'duration': [100, 200, 150, 300, 90, 250, 130, 180],
        'y': ['yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no'],
    })


def test_split_features_target_removes_duration_and_maps_target():
    X, y = split_features_target(_toy_df())
    assert 'duration' not in X.columns
    assert set(y.unique()) == {0, 1}


def test_candidate_models_returns_fresh_unfitted_estimators():
    models = candidate_models()
    assert set(models) == {'logistic_regression', 'decision_tree', 'random_forest', 'gradient_boosting'}
    for estimator in models.values():
        assert not hasattr(estimator, 'classes_')  # unfitted


def test_build_pipeline_fits_and_predicts_probabilities():
    X, y = split_features_target(_toy_df())
    pipeline = build_pipeline(candidate_models()['logistic_regression'], X)
    pipeline.fit(X, y)
    probs = pipeline.predict_proba(X)[:, 1]
    assert probs.shape == (len(X),)
    assert ((probs >= 0) & (probs <= 1)).all()
