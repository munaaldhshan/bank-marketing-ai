"""Load the trained pipeline and score a single customer record."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.config import DEFAULT_THRESHOLD, FEATURE_COLUMNS, MODEL_PATH, SCHEMA_PATH


def load_model(path: str | Path = MODEL_PATH):
    """Load the saved pipeline, with a message that says how to create one if it's missing."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(
            f'No trained model at {model_path}. Run `python -m src.train` to create one.'
        )
    return joblib.load(model_path)


def load_schema(path: str | Path = SCHEMA_PATH) -> dict | None:
    """Load the feature schema saved alongside the model, or None if it isn't there.

    Optional by design: an older model artifact without a schema file should
    still be usable for prediction, just without the extrapolation warnings.
    """
    from src.schema import load_feature_schema

    schema_path = Path(path)
    return load_feature_schema(schema_path) if schema_path.exists() else None


def predict_customer(
    model, record: dict, schema: dict | None = None, threshold: float = DEFAULT_THRESHOLD,
) -> tuple[float, str, list[str]]:
    """Return (probability, predicted label, validation warnings) for one customer.

    Missing fields are left for the pipeline's own imputers to fill in, so a
    partially-filled record still gets a prediction rather than a crash. Pass
    the threshold saved in reports/metrics.json (tuned on a validation split,
    not the default 0.5) so the label reflects the same cutoff the reported
    metrics used.
    """
    from src.schema import validate_record

    row = {col: record.get(col) for col in FEATURE_COLUMNS}
    probability = float(model.predict_proba(pd.DataFrame([row]))[0, 1])
    label = 'yes' if probability >= threshold else 'no'
    warnings = validate_record(record, schema) if schema else []
    return probability, label, warnings
