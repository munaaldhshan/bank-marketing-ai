from pathlib import Path

import pandas as pd

from src.eda_analysis import generate_eda_figures


def test_generate_eda_figures_creates_png_outputs(tmp_path):
    df = pd.DataFrame(
        {
            'y': ['no', 'yes', 'no', 'yes', 'no', 'yes'],
            'job': ['admin', 'admin', 'student', 'student', 'retired', 'retired'],
            'month': ['may', 'may', 'jun', 'jun', 'nov', 'nov'],
            'contact': ['cellular', 'telephone', 'cellular', 'telephone', 'cellular', 'telephone'],
        }
    )

    output_dir = tmp_path / 'figures'
    paths = generate_eda_figures(df, output_dir=output_dir)

    assert set(paths.keys()) == {'target_distribution', 'job_response', 'month_response'}
    for path in paths.values():
        assert Path(path).exists()
        assert Path(path).suffix == '.png'
