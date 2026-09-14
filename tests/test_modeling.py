import pandas as pd

from src.modeling import prepare_model_data


def test_prepare_model_data_removes_duration_and_maps_target():
    df = pd.DataFrame(
        {
            'age': [30, 40, 55],
            'job': ['admin', 'blue-collar', 'retired'],
            'duration': [100, 200, 300],
            'y': ['yes', 'no', 'yes'],
        }
    )

    X, y = prepare_model_data(df)

    assert 'duration' not in X.columns
    assert set(y.unique()) == {0, 1}
    assert list(X.columns) == ['age', 'job']
