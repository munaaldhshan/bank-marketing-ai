"""Prediction utilities for a trained bank-marketing pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import joblib


def load_model(path: str | Path = 'models/final_model.joblib') -> object:
    """Load the saved model pipeline from disk."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f'Model artifact was not found at {model_path}.')
    return joblib.load(model_path)


def predict_customer(model: object, row: dict) -> tuple[float, str]:
    """Return predicted probability and class for a single customer record."""
    df = pd.DataFrame([row])
    probability = float(model.predict_proba(df)[0, 1])
    label = 'yes' if probability >= 0.5 else 'no'
    return probability, label
