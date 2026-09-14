"""Lightweight EDA helpers for the bank-marketing project."""

from __future__ import annotations

from typing import Any

import pandas as pd


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
