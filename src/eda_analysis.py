"""Lightweight EDA helpers for the bank-marketing project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def summarize_segment_response(df: pd.DataFrame, target_col: str = 'y') -> dict[str, dict[str, float]]:
    """Return conversion-rate summaries for categorical segments in the dataset."""
    summary: dict[str, dict[str, float]] = {}

    for column in ['job', 'month', 'contact']:
        if column not in df.columns:
            continue

        rates = (
            df.groupby(column)[target_col]
            .apply(lambda s: (s == 'yes').mean())
            .sort_values(ascending=False)
            .to_dict()
        )
        summary[column] = rates

    return summary


def build_segment_summary(df: pd.DataFrame, target_col: str = 'y') -> dict[str, Any]:
    """Assemble a compact business-oriented summary for EDA reporting."""
    return {
        'target_share': df[target_col].value_counts(normalize=True).to_dict(),
        'segment_response': summarize_segment_response(df, target_col=target_col),
    }


def generate_eda_figures(df: pd.DataFrame, output_dir: str | Path = 'reports/figures') -> dict[str, str]:
    """Create core EDA plots and save them as PNG files in the target directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    target_counts = df['y'].value_counts().reindex(['no', 'yes'], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='y', order=['no', 'yes'], ax=ax)
    ax.set_title('Target distribution')
    ax.set_xlabel('Subscription outcome')
    ax.set_ylabel('Count')
    target_path = output_path / 'target_distribution.png'
    fig.tight_layout()
    fig.savefig(target_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    job_rates = (
        df.groupby('job')['y']
        .apply(lambda s: (s == 'yes').mean())
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    job_rates.plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title('Response rate by job type')
    ax.set_xlabel('Job')
    ax.set_ylabel('Subscription rate')
    ax.set_ylim(0, max(job_rates.max() * 1.4, 0.2))
    job_path = output_path / 'job_response.png'
    fig.tight_layout()
    fig.savefig(job_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    month_rates = (
        df.groupby('month')['y']
        .apply(lambda s: (s == 'yes').mean())
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    month_rates.plot(kind='bar', ax=ax, color='darkgreen')
    ax.set_title('Response rate by contact month')
    ax.set_xlabel('Month')
    ax.set_ylabel('Subscription rate')
    ax.set_ylim(0, max(month_rates.max() * 1.4, 0.2))
    month_path = output_path / 'month_response.png'
    fig.tight_layout()
    fig.savefig(month_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    return {
        'target_distribution': str(target_path),
        'job_response': str(job_path),
        'month_response': str(month_path),
    }


def export_eda_report_figures(data_path: str | Path = 'data/bank_marketing.csv', output_dir: str | Path = 'reports/figures') -> dict[str, str]:
    """Load the dataset and generate the core EDA PNG figures for the project report."""
    from src.data_loader import load_bank_data

    df = load_bank_data(data_path)
    return generate_eda_figures(df, output_dir=output_dir)
