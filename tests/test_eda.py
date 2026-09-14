import pandas as pd

from src.eda_analysis import summarize_segment_response


def test_summarize_segment_response_returns_expected_structure():
    df = pd.DataFrame(
        {
            'job': ['student', 'student', 'retired', 'admin.'],
            'month': ['mar', 'mar', 'dec', 'apr'],
            'contact': ['cellular', 'telephone', 'cellular', 'cellular'],
            'y': ['yes', 'no', 'yes', 'no'],
        }
    )

    summary = summarize_segment_response(df)

    assert set(summary) == {'job', 'month', 'contact'}
    assert 'student' in summary['job']
    assert 'mar' in summary['month']
    assert 'cellular' in summary['contact']
    assert summary['job']['student'] >= 0
    assert summary['job']['student'] <= 1
