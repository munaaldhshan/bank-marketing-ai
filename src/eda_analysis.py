"""EDA helpers: segment summaries and the report figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import DATA_PATH, FIGURES_DIR, PDAYS_NEVER_CONTACTED, TARGET

SEGMENT_COLUMNS = ('job', 'month', 'contact', 'poutcome')


def _response_rate(df: pd.DataFrame, column: str, target_col: str = TARGET) -> pd.Series:
    return df.groupby(column)[target_col].apply(lambda s: (s == 'yes').mean()).sort_values(ascending=False)


def summarize_segment_response(df: pd.DataFrame, target_col: str = TARGET) -> dict[str, dict[str, float]]:
    """Conversion rate per category, for every segment column present in the data."""
    return {col: _response_rate(df, col, target_col).to_dict() for col in SEGMENT_COLUMNS if col in df.columns}


def build_segment_summary(df: pd.DataFrame, target_col: str = TARGET) -> dict[str, Any]:
    """A compact business-oriented summary: overall rate, per-segment rates, and the
    prior-contact effect, which is the single strongest signal in the dataset
    (customers contacted in an earlier campaign convert at roughly 6x the rate
    of those who weren't, regardless of that prior outcome)."""
    summary: dict[str, Any] = {
        'target_share': df[target_col].value_counts(normalize=True).to_dict(),
        'segment_response': summarize_segment_response(df, target_col=target_col),
    }
    if 'pdays' in df.columns:
        previously_contacted = df['pdays'] != PDAYS_NEVER_CONTACTED
        summary['response_by_prior_contact'] = (
            df.groupby(previously_contacted)[target_col]
            .apply(lambda s: (s == 'yes').mean())
            .rename(index={True: 'contacted_before', False: 'first_contact'})
            .to_dict()
        )
    return summary


def _bar_chart(rates: pd.Series, title: str, xlabel: str, ylabel: str, color: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    rates.plot(kind='bar', ax=ax, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(rates.max() * 1.4, 0.2))
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def generate_eda_figures(df: pd.DataFrame, output_dir: str | Path = FIGURES_DIR) -> dict[str, str]:
    """Save the core EDA PNGs used in the report and return their paths."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x=TARGET, order=['no', 'yes'], ax=ax)
    ax.set_title('Target distribution')
    ax.set_xlabel('Subscription outcome')
    ax.set_ylabel('Count')
    fig.tight_layout()
    target_path = output_path / 'target_distribution.png'
    fig.savefig(target_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    paths['target_distribution'] = str(target_path)

    charts = [
        ('job', 'job_response.png', 'Response rate by job type', 'Job', 'steelblue'),
        ('month', 'month_response.png', 'Response rate by contact month', 'Month', 'darkgreen'),
    ]
    for column, filename, title, xlabel, color in charts:
        if column not in df.columns:
            continue
        chart_path = output_path / filename
        _bar_chart(_response_rate(df, column), title, xlabel, 'Subscription rate', color, chart_path)
        paths[filename.removesuffix('.png')] = str(chart_path)

    if 'pdays' in df.columns:
        previously_contacted = (df['pdays'] != PDAYS_NEVER_CONTACTED).map(
            {True: 'Contacted before', False: 'First contact'}
        )
        rates = df.groupby(previously_contacted)[TARGET].apply(lambda s: (s == 'yes').mean())
        prior_path = output_path / 'prior_contact_response.png'
        _bar_chart(rates, 'Response rate: contacted in a prior campaign or not', '', 'Subscription rate', 'darkorange', prior_path)
        paths['prior_contact_response'] = str(prior_path)

    return paths


def export_eda_report_figures(data_path: str | Path = DATA_PATH, output_dir: str | Path = FIGURES_DIR) -> dict[str, str]:
    """Load the dataset and regenerate every figure used in reports/final_report.md."""
    from src.data_loader import load_bank_data

    return generate_eda_figures(load_bank_data(data_path), output_dir=output_dir)
