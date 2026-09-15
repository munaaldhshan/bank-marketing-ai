"""Summarize a fitted pipeline's performance on a held-out set."""

from __future__ import annotations

from sklearn.metrics import classification_report, confusion_matrix

from src.model_comparison import compute_metrics


def summarize_model(model, X_test, y_test, threshold: float = 0.5) -> dict:
    """Metrics plus a confusion matrix and text report, for printing or logging."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    metrics = compute_metrics(y_test, y_prob, threshold=threshold)
    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
    metrics['classification_report'] = classification_report(y_test, y_pred, zero_division=0)
    return metrics
