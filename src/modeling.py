"""Candidate models and the pipeline factory shared by training, notebooks and tests."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE
from src.preprocessing import build_preprocessor


def candidate_models(random_state: int = RANDOM_STATE) -> dict[str, BaseEstimator]:
    """Fresh, unfitted instances of every model compared during training.

    None of them use class weighting. With an 11% positive rate, re-weighting
    barely changes ranking quality but inflates predicted probabilities
    (balanced logistic regression averages 0.39 against a 0.11 base rate),
    and the app shows those probabilities to people.
    """
    return {
        'logistic_regression': LogisticRegression(max_iter=3000),
        'decision_tree': DecisionTreeClassifier(max_depth=6, random_state=random_state),
        'random_forest': RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=random_state,
        ),
        'gradient_boosting': HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=400, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=30, random_state=random_state,
        ),
    }


def build_pipeline(estimator: BaseEstimator, X: pd.DataFrame) -> Pipeline:
    """Wrap an estimator with the shared preprocessing so it accepts raw rows."""
    return Pipeline([('preprocessor', build_preprocessor(X)), ('classifier', estimator)])
