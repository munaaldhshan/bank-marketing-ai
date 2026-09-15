"""Shared fixtures: a small synthetic dataset shaped like the real bank-marketing schema."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import ECONOMIC_COLUMNS, PDAYS_NEVER_CONTACTED


def make_synthetic_bank_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """A dataset with the real column names/categories but fabricated values.

    Large and varied enough to exercise the full train/validation/test split
    in `build_final_pipeline`, the per-month economic snapshot grouping, and
    the categorical/numeric schema builder, without touching the real
    (ungitted) dataset or the network.
    """
    rng = np.random.RandomState(seed)
    months = ['mar', 'may', 'jul', 'oct']
    # Distinct per-month economic regime, mirroring the real data's month/economy coupling.
    econ_by_month = {
        'mar': {'emp.var.rate': -1.8, 'cons.price.idx': 92.8, 'cons.conf.idx': -50.0, 'euribor3m': 1.5, 'nr.employed': 5099.1},
        'may': {'emp.var.rate': 1.1, 'cons.price.idx': 94.0, 'cons.conf.idx': -36.4, 'euribor3m': 4.86, 'nr.employed': 5191.0},
        'jul': {'emp.var.rate': 1.4, 'cons.price.idx': 93.9, 'cons.conf.idx': -42.7, 'euribor3m': 4.96, 'nr.employed': 5228.1},
        'oct': {'emp.var.rate': -3.4, 'cons.price.idx': 92.4, 'cons.conf.idx': -26.9, 'euribor3m': 0.74, 'nr.employed': 5017.5},
    }

    month = rng.choice(months, size=n)
    contacted_before = rng.rand(n) < 0.3
    poutcome = np.where(contacted_before, rng.choice(['success', 'failure'], size=n), 'nonexistent')
    pdays = np.where(contacted_before, rng.randint(0, 27, size=n), PDAYS_NEVER_CONTACTED)
    previous = np.where(contacted_before, rng.randint(1, 4, size=n), 0)
    age = rng.randint(18, 90, size=n)

    # A synthetic but non-trivial relationship so metrics/importances aren't degenerate.
    score = (
        0.5 * contacted_before.astype(float)
        + 0.3 * (poutcome == 'success').astype(float)
        + 0.2 * (age < 30).astype(float)
        + rng.rand(n)
    )
    y = np.where(score > np.quantile(score, 0.7), 'yes', 'no')

    df = pd.DataFrame({
        'age': age,
        'job': rng.choice(['admin.', 'student', 'retired', 'blue-collar', 'unknown'], size=n),
        'marital': rng.choice(['single', 'married', 'divorced'], size=n),
        'education': rng.choice(['university.degree', 'high.school', 'basic.9y', 'unknown'], size=n),
        'default': rng.choice(['no', 'unknown', 'yes'], size=n, p=[0.7, 0.29, 0.01]),
        'housing': rng.choice(['yes', 'no', 'unknown'], size=n),
        'loan': rng.choice(['yes', 'no', 'unknown'], size=n),
        'contact': rng.choice(['cellular', 'telephone'], size=n),
        'month': month,
        'day_of_week': rng.choice(['mon', 'tue', 'wed', 'thu', 'fri'], size=n),
        'duration': rng.randint(0, 1000, size=n),
        'campaign': rng.randint(1, 10, size=n),
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome,
        'y': y,
    })
    for col in ECONOMIC_COLUMNS:
        df[col] = [econ_by_month[m][col] for m in month]
    return df


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    return make_synthetic_bank_df()
