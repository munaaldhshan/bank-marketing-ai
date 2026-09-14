import pandas as pd

from src.model_comparison import compare_models, prepare_comparison_data


def test_prepare_comparison_data_maps_target_and_drops_duration():
    df = pd.DataFrame(
        {
            'age': [30, 40, 25],
            'job': ['admin', 'teacher', 'student'],
            'duration': [100, 200, 300],
            'y': ['yes', 'no', 'yes'],
        }
    )

    X, y = prepare_comparison_data(df)

    assert 'duration' not in X.columns
    assert set(y.unique()) == {0, 1}


def test_compare_models_returns_supported_metrics():
    df = pd.DataFrame(
        {
            'age': [25, 35, 40, 50, 60, 70, 30, 28, 42, 55,
                    20, 32, 48, 62, 65, 72, 27, 38, 52, 58],
            'job': ['student', 'admin', 'teacher', 'retired', 'admin', 'retired', 'student', 'admin', 'teacher', 'admin',
                    'student', 'admin', 'teacher', 'retired', 'admin', 'retired', 'student', 'admin', 'teacher', 'admin'],
            'duration': [100, 200, 300, 250, 150, 220, 180, 280, 110, 260,
                         90, 190, 330, 240, 160, 200, 120, 300, 170, 210],
            'y': ['yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no',
                  'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no'],
        }
    )

    results = compare_models(df, models={})

    assert set(results) == {'logistic_regression', 'decision_tree', 'random_forest'}
    for metrics in results.values():
        assert set(metrics) == {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}
