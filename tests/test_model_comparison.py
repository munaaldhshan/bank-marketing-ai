import numpy as np
import pandas as pd

from src.model_comparison import compare_models, compute_metrics, optimize_threshold
from src.preprocessing import split_features_target


def _toy_df(n=40):
    rng = np.random.RandomState(0)
    ages = rng.randint(20, 70, size=n)
    jobs = rng.choice(['student', 'admin', 'retired', 'blue-collar'], size=n)
    duration = rng.randint(50, 400, size=n)
    # Make the target loosely dependent on age/job so metrics aren't degenerate.
    score = (ages < 30).astype(int) + (jobs == 'retired').astype(int) + rng.rand(n)
    y = np.where(score > np.median(score), 'yes', 'no')
    return pd.DataFrame({'age': ages, 'job': jobs, 'duration': duration, 'y': y})


def test_split_features_target_maps_target_and_drops_duration():
    X, y = split_features_target(_toy_df())
    assert 'duration' not in X.columns
    assert set(y.unique()) == {0, 1}


def test_compare_models_returns_cv_and_test_metrics_for_every_candidate():
    results = compare_models(_toy_df(60))
    assert set(results) == {'logistic_regression', 'decision_tree', 'random_forest', 'gradient_boosting'}
    expected_keys = {
        'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision',
        'cv_roc_auc_mean', 'cv_roc_auc_std', 'cv_average_precision_mean',
    }
    for metrics in results.values():
        assert expected_keys <= set(metrics)


def test_compute_metrics_perfect_predictions():
    y_true = pd.Series([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.9, 0.8])
    metrics = compute_metrics(y_true, y_prob)
    assert metrics['accuracy'] == 1.0
    assert metrics['f1'] == 1.0
    assert metrics['roc_auc'] == 1.0


def test_optimize_threshold_finds_the_f1_maximizing_cutoff():
    # A gap between the negative and positive score clusters at 0.5: the F1-maximizing
    # threshold should land inside that gap, and a coarser or finer sweep should agree.
    y_true = pd.Series([0] * 10 + [1] * 10)
    y_prob = np.array([0.1] * 10 + [0.9] * 10)
    threshold, metrics = optimize_threshold(y_true, y_prob)
    assert 0.1 < threshold < 0.9
    assert metrics['f1'] == 1.0


def test_optimize_threshold_never_looks_at_a_threshold_outside_the_given_range():
    y_true = pd.Series([0, 1, 0, 1])
    y_prob = np.array([0.3, 0.6, 0.4, 0.7])
    thresholds = np.array([0.5])
    threshold, metrics = optimize_threshold(y_true, y_prob, thresholds=thresholds)
    assert threshold == 0.5
