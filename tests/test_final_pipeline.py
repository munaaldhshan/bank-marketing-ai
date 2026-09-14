import pandas as pd

from src.final_pipeline import build_final_pipeline


def test_build_final_pipeline_works_on_clean_data():
    df = pd.DataFrame(
        {
            'age': [25, 40, 60, 35, 50, 70],
            'job': ['student', 'admin', 'retired', 'admin', 'teacher', 'retired'],
            'housing': ['yes', 'no', 'yes', 'no', 'yes', 'no'],
            'loan': ['no', 'yes', 'no', 'no', 'yes', 'no'],
            'contact': ['cellular', 'cellular', 'telephone', 'cellular', 'telephone', 'cellular'],
            'month': ['mar', 'apr', 'nov', 'may', 'sep', 'dec'],
            'campaign': [1, 2, 3, 1, 2, 4],
            'y': ['yes', 'no', 'yes', 'no', 'yes', 'no'],
        }
    )

    model = build_final_pipeline(df)

    assert hasattr(model, 'predict_proba')
    assert model.named_steps['classifier'] is not None
