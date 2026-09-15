from src.final_pipeline import build_final_pipeline


def test_build_final_pipeline_trains_and_reports_a_tuned_threshold(synthetic_df):
    model, metrics = build_final_pipeline(synthetic_df)

    assert hasattr(model, 'predict_proba')
    assert model.named_steps['classifier'] is not None

    assert 0.0 < metrics['decision_threshold'] < 1.0
    assert metrics['priority_medium_cutoff'] <= metrics['priority_high_cutoff']
    for key in ('test_metrics', 'test_metrics_at_0.5'):
        assert {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'} <= set(metrics[key])


def test_build_final_pipeline_never_sees_duration(synthetic_df, monkeypatch):
    # A pipeline that was handed `duration` would trivially separate the classes at
    # prediction time on this synthetic set too, since duration is drawn independently
    # of y here — so instead assert directly that the fitted preprocessor was never
    # given the column at all.
    from sklearn.compose import ColumnTransformer

    # Pipeline.fit() calls each intermediate transformer's fit_transform, not fit, so that's
    # the method to intercept.
    captured = {}
    original_fit_transform = ColumnTransformer.fit_transform

    def spy_fit_transform(self, X, y=None, **kwargs):
        captured['columns'] = list(X.columns)
        return original_fit_transform(self, X, y, **kwargs)

    monkeypatch.setattr(ColumnTransformer, 'fit_transform', spy_fit_transform)
    build_final_pipeline(synthetic_df)
    assert 'duration' not in captured['columns']
