"""Train the production model (best candidate from model_comparison) and save it."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import numpy as np

from src.config import (
    ECONOMIC_SNAPSHOTS_PATH,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    SCHEMA_PATH,
    TEST_SIZE,
    VALIDATION_SIZE,
)
from src.data_loader import load_bank_data
from src.model_comparison import compute_metrics, optimize_threshold
from src.modeling import build_pipeline
from src.preprocessing import split_features_target
from src.schema import build_economic_snapshots, build_feature_schema, save_feature_schema

# HistGradientBoosting was the best candidate in model_comparison's CV + held-out
# test sweep (test ROC-AUC 0.814 vs 0.800 for logistic regression, 0.806 for a
# random forest; see reports/final_report.md). Native missing-value handling
# also means no imputation step is needed for its numeric inputs.
PRODUCTION_ESTIMATOR = HistGradientBoostingClassifier(
    learning_rate=0.05, max_iter=400, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=30, random_state=RANDOM_STATE,
)


def build_final_pipeline(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    validation_size: float = VALIDATION_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[Pipeline, dict]:
    """Fit the production pipeline and evaluate it on a held-out test split.

    The decision threshold is chosen on a validation split carved out of the
    training data, then the model is refit on train+validation combined
    before the one-time test evaluation. This keeps the test set from
    influencing either the threshold or the model, while still letting the
    final artifact train on every row except the held-out test set.
    """
    X, y = split_features_target(df)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=validation_size, random_state=random_state, stratify=y_trainval,
    )

    threshold_model = build_pipeline(PRODUCTION_ESTIMATOR, X_train)
    threshold_model.fit(X_train, y_train)
    val_probs = threshold_model.predict_proba(X_val)[:, 1]
    threshold, _ = optimize_threshold(y_val, val_probs)
    # With an 11% base rate, fixed cutoffs like 0.7/0.4 almost never fire —
    # the mean predicted probability itself is close to 0.11. Percentiles of
    # the validation scores instead say "the top 10% of prospects" and
    # "the next 20%", which maps directly onto limited campaign capacity.
    priority_high_cutoff, priority_medium_cutoff = np.quantile(val_probs, [0.9, 0.7])

    model = build_pipeline(PRODUCTION_ESTIMATOR, X_trainval)
    model.fit(X_trainval, y_trainval)

    test_probs = model.predict_proba(X_test)[:, 1]
    metrics = {
        'decision_threshold': float(threshold),
        'priority_high_cutoff': float(priority_high_cutoff),
        'priority_medium_cutoff': float(priority_medium_cutoff),
        'test_metrics': compute_metrics(y_test, test_probs, threshold=threshold),
        'test_metrics_at_0.5': compute_metrics(y_test, test_probs, threshold=0.5),
    }
    return model, metrics


def save_final_pipeline(model: Pipeline, output_path: str | Path = MODEL_PATH) -> Path:
    """Persist the trained pipeline artifact to disk."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    return destination


def train_and_save_final_pipeline(
    data_path: str | Path | None = None,
    model_path: str | Path = MODEL_PATH,
    metrics_path: str | Path = METRICS_PATH,
    schema_path: str | Path = SCHEMA_PATH,
    economic_snapshots_path: str | Path = ECONOMIC_SNAPSHOTS_PATH,
) -> Path:
    """Load the dataset, train the final model, and save everything prediction needs
    alongside it: the metrics, the feature schema, and the monthly economic snapshots."""
    df = load_bank_data(data_path)
    model, metrics = build_final_pipeline(df)
    destination = save_final_pipeline(model, model_path)

    X, _ = split_features_target(df)
    save_feature_schema(build_feature_schema(X), schema_path)
    Path(economic_snapshots_path).parent.mkdir(parents=True, exist_ok=True)
    Path(economic_snapshots_path).write_text(json.dumps(build_economic_snapshots(X), indent=2))

    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).write_text(json.dumps(metrics, indent=2))
    return destination
