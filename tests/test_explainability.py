from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.explainability import top_feature_importance
from src.modeling import build_pipeline
from src.preprocessing import split_features_target


def test_top_feature_importance_aggregates_one_hot_categories_for_linear_models(synthetic_df):
    X, y = split_features_target(synthetic_df)
    model = build_pipeline(LogisticRegression(max_iter=1000), X)
    model.fit(X, y)

    importance = top_feature_importance(model, top_n=20)
    # One row per original column (e.g. 'job'), never per one-hot category ('job_admin.').
    assert 'job' in importance['feature'].values
    assert not any(f.startswith('job_') for f in importance['feature'])
    assert (importance['abs_importance'] >= 0).all()


def test_top_feature_importance_works_for_native_tree_importances(synthetic_df):
    X, y = split_features_target(synthetic_df)
    model = build_pipeline(RandomForestClassifier(n_estimators=20, random_state=0), X)
    model.fit(X, y)

    importance = top_feature_importance(model, top_n=10)
    assert len(importance) > 0
    assert 'job' in importance['feature'].values


def test_top_feature_importance_falls_back_to_permutation_for_histgbm(synthetic_df):
    X, y = split_features_target(synthetic_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    model = build_pipeline(HistGradientBoostingClassifier(max_iter=50, random_state=0), X_train)
    model.fit(X_train, y_train)

    assert not hasattr(model.named_steps['classifier'], 'feature_importances_')
    importance = top_feature_importance(model, X_test, y_test, top_n=10, n_repeats=3)
    assert set(importance['feature']) <= set(X.columns)


def test_top_feature_importance_raises_a_clear_error_without_a_fallback_split(synthetic_df):
    X, y = split_features_target(synthetic_df)
    model = build_pipeline(HistGradientBoostingClassifier(max_iter=50, random_state=0), X)
    model.fit(X, y)

    try:
        top_feature_importance(model)
        assert False, 'expected a ValueError'
    except ValueError as exc:
        assert 'held-out' in str(exc)
