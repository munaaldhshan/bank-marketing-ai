"""Load the trained pipeline and score a single customer record."""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import pandas as pd

from src.config import DEFAULT_THRESHOLD, FEATURE_COLUMNS, MODEL_PATH, SCHEMA_PATH


def _describe_model_load_failure(path: str | Path, exc: Exception | str | None = None) -> str:
    """Return a human-readable explanation for stale or broken model artifacts."""
    model_path = Path(path)
    message = str(exc).lower()
    if 'sklearn' in message or 'pickle' in message or '_remaindercolslist' in message:
        return (
            f'Stale or incompatible model artifact detected at {model_path}. The model was '
            'trained with a different scikit-learn version than the one currently installed. '
            'Rebuild it with `python -m src.train`.'
        )
    return (
        f'The saved model at {model_path} is unreadable or incompatible with the current '
        'Python/scikit-learn environment. Re-train it with `python -m src.train`.'
    )


def _validate_loaded_model(model) -> None:
    """Reject obviously invalid artifacts before the app ever uses them."""
    if model is None or not hasattr(model, 'predict_proba'):
        raise TypeError('The saved model does not look like a valid trained sklearn pipeline.')


def check_model_compatibility(path: str | Path = MODEL_PATH) -> tuple[bool, str]:
    """Return whether a saved model is compatible with the current environment."""
    model_path = Path(path)
    if not model_path.exists():
        return False, f'No trained model at {model_path}. Run `python -m src.train` to create one.'

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model = joblib.load(model_path)
    except Exception as exc:
        return False, _describe_model_load_failure(model_path, exc)

    warnings_text = ' '.join(str(w.message).lower() for w in caught)
    if (
        'inconsistentversionwarning' in warnings_text
        or 'pickle' in warnings_text
        or '_remaindercolslist' in warnings_text
        or 'sklearn' in warnings_text
    ):
        return False, _describe_model_load_failure(model_path, 'incompatible sklearn pickle')

    try:
        _validate_loaded_model(model)
    except TypeError:
        return False, (
            f'The saved model at {model_path} is stale or not a valid trained pipeline. '
            'Rebuild it with `python -m src.train`.'
        )

    return True, f'Model artifact at {model_path} is compatible with the current environment.'


def load_model(path: str | Path = MODEL_PATH):
    """Load the saved pipeline, with a message that says how to create one if it's missing."""
    model_path = Path(path)
    ok, message = check_model_compatibility(model_path)
    if not ok:
        raise ValueError(message)
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
