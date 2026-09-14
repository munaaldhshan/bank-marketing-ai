"""Comparison utilities for candidate bank-marketing models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import load_bank_data
from src.preprocessing import clean_dataset


def _make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create the default preprocessing transformer for numeric and categorical columns."""
    numeric_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    transformers: list[tuple[str, object, list[str]]] = []

    if numeric_cols:
        transformers.append(
            (
                'num',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                ]),
                numeric_cols,
            )
        )

    if categorical_cols:
        transformers.append(
            (
                'cat',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore')),
                ]),
                categorical_cols,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder='drop')


def _binary_target(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
    if mapped.isna().any():
        raise ValueError('Target column contains values other than yes/no.')
    return mapped.astype(int)


def prepare_comparison_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return cleaned feature matrix and numeric target for model comparison."""
    cleaned = clean_dataset(df)
    X = cleaned.drop(columns=['y'])
    y = _binary_target(cleaned['y'])
    return X, y


def _compute_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
    }


def compare_models(
    df: pd.DataFrame,
    models: dict[str, Any] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    """Train and compare a set of candidate models on the cleaned bank marketing dataset."""
    if models is None or len(models) == 0:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier

        models = {
            'logistic_regression': LogisticRegression(max_iter=2000, class_weight='balanced'),
            'decision_tree': DecisionTreeClassifier(max_depth=6, random_state=random_state),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                random_state=random_state,
                class_weight='balanced',
                min_samples_leaf=5,
            ),
        }

    X, y = prepare_comparison_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    results: dict[str, dict[str, float]] = {}
    for name, estimator in models.items():
        model = Pipeline([
            ('preprocessor', _make_preprocessor(X_train)),
            ('classifier', estimator),
        ])
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        results[name] = _compute_metrics(y_test, probs)

    return results


def optimize_threshold(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    thresholds: list[float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Select the threshold maximizing F1-score over the candidate thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 81)

    best_threshold = thresholds[0]
    best_metrics = {'f1': -1.0}

    probs = model.predict_proba(X_test)[:, 1]
    for threshold in thresholds:
        metrics = _compute_metrics(y_test, probs, threshold=threshold)
        if metrics['f1'] > best_metrics['f1']:
            best_threshold = threshold
            best_metrics = metrics
            best_metrics['threshold'] = threshold

    return best_threshold, best_metrics
