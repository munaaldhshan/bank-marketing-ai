"""Compare candidate models with cross-validation and pick a decision threshold."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src.config import CV_FOLDS, RANDOM_STATE, TEST_SIZE
from src.modeling import build_pipeline, candidate_models
from src.preprocessing import split_features_target


def compute_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Threshold-based and threshold-free metrics for one set of predictions."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'average_precision': average_precision_score(y_true, y_prob),
    }


def compare_models(
    df: pd.DataFrame,
    models: dict[str, Any] | None = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    cv_folds: int = CV_FOLDS,
) -> dict[str, dict[str, float]]:
    """Cross-validate each candidate on the training split, then score once on the held-out test split.

    Reports both so overfitting is visible: a model whose CV score is much
    higher than its test score is fitting noise in the training folds.
    """
    models = models if models else candidate_models(random_state)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    results: dict[str, dict[str, float]] = {}
    for name, estimator in models.items():
        pipeline = build_pipeline(estimator, X_train)
        cv_scores = cross_validate(
            pipeline, X_train, y_train, cv=cv,
            scoring={'roc_auc': 'roc_auc', 'average_precision': 'average_precision'},
        )
        pipeline.fit(X_train, y_train)
        probs = pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, probs)
        metrics['cv_roc_auc_mean'] = cv_scores['test_roc_auc'].mean()
        metrics['cv_roc_auc_std'] = cv_scores['test_roc_auc'].std()
        metrics['cv_average_precision_mean'] = cv_scores['test_average_precision'].mean()
        results[name] = metrics
    return results


def optimize_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    """Sweep thresholds and return the one maximizing F1, with its full metrics.

    Callers must pass validation predictions, never the final test set —
    picking a threshold on the test set and then reporting test metrics at
    that threshold overstates performance (notebook 04's original bug).
    """
    thresholds = np.linspace(0.05, 0.95, 91) if thresholds is None else thresholds
    best_threshold, best_metrics = thresholds[0], {'f1': -1.0}
    for threshold in thresholds:
        metrics = compute_metrics(y_true, y_prob, threshold=threshold)
        if metrics['f1'] > best_metrics['f1']:
            best_threshold, best_metrics = threshold, {**metrics, 'threshold': threshold}
    return best_threshold, best_metrics
