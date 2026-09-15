"""A record of what the training data looked like, used to validate inputs at prediction time.

The trained pipeline can't tell you when it's guessing: OneHotEncoder(handle_unknown='ignore')
silently encodes an unseen category as all-zeros, and a numeric value far outside the training
range is silently standard-scaled to a large z-score. Both produce a confident-looking
probability with no warning. This module captures the valid categories and numeric ranges
seen during training so callers (the app, batch scoring) can flag inputs the model has never
seen anything like, before trusting its output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import ECONOMIC_COLUMNS, PDAYS_NEVER_CONTACTED
from src.preprocessing import identify_feature_types


def build_feature_schema(X: pd.DataFrame) -> dict[str, Any]:
    """Capture each column's observed categories or numeric range.

    `pdays` gets a second, narrower range covering only rows where the
    customer actually was contacted before (excluding the `999` sentinel for
    "never contacted"), since that sentinel would otherwise dominate the
    min/max and make the real range of "days since last contact" invisible.
    """
    numeric, categorical = identify_feature_types(X)
    schema: dict[str, Any] = {'numeric': {}, 'categorical': {}}
    for col in numeric:
        schema['numeric'][col] = {'min': float(X[col].min()), 'max': float(X[col].max())}
    for col in categorical:
        schema['categorical'][col] = sorted(str(v) for v in X[col].dropna().unique())
    if 'pdays' in X.columns:
        contacted_before = X.loc[X['pdays'] != PDAYS_NEVER_CONTACTED, 'pdays']
        if len(contacted_before):
            schema['pdays_if_contacted_before'] = {
                'min': float(contacted_before.min()), 'max': float(contacted_before.max()),
            }
    return schema


def build_economic_snapshots(X: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Median economic-indicator readings for each contact month in the training data.

    These five columns are national indicators for the time of contact, not
    facts about an individual customer, and they move together (e.g.
    `euribor3m` near 5 only ever co-occurs with `emp.var.rate` >= 1.1). Asking
    an app user to set them independently invites combinations that never
    occurred historically, so instead they're derived from the chosen month.
    """
    if 'month' not in X.columns or not all(col in X.columns for col in ECONOMIC_COLUMNS):
        return {}
    return {
        str(month): {col: float(group[col].median()) for col in ECONOMIC_COLUMNS}
        for month, group in X.groupby('month')
    }


def save_feature_schema(schema: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(schema, indent=2))
    return destination


def load_feature_schema(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def validate_record(record: dict, schema: dict[str, Any]) -> list[str]:
    """Return a plain-language warning for every field the model hasn't seen the like of.

    Never raises: an out-of-schema record is still valid input to
    `predict_customer`, just one the caller should treat with more caution.
    """
    warnings: list[str] = []
    for col, bounds in schema.get('numeric', {}).items():
        if col not in record or record[col] is None:
            continue
        value = record[col]
        if value < bounds['min'] or value > bounds['max']:
            warnings.append(
                f"'{col}' = {value} is outside the training range "
                f"[{bounds['min']:g}, {bounds['max']:g}]; the prediction is an extrapolation."
            )
    for col, categories in schema.get('categorical', {}).items():
        if col not in record or record[col] is None:
            continue
        value = str(record[col])
        if value not in categories:
            warnings.append(
                f"'{col}' = '{value}' was never seen during training; the model will silently "
                f"treat it as a blank category instead of raising an error."
            )
    return warnings
