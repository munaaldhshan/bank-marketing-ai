"""Explainability utilities for the bank-marketing model."""

from __future__ import annotations

from typing import Any

import pandas as pd


def top_feature_importance(model: Any, feature_names: list[str], top_n: int = 10) -> pd.DataFrame:
    """Return the top coefficients or feature importances for a fitted model."""
    if hasattr(model, 'named_steps'):
        estimator = model.named_steps['classifier']
        preprocessor = model.named_steps['preprocessor']
        transformed_names = list(preprocessor.get_feature_names_out())
        if hasattr(estimator, 'coef_'):
            coef = estimator.coef_[0]
            importance = pd.DataFrame({
                'feature': transformed_names,
                'importance': coef,
            })
            importance['abs_importance'] = importance['importance'].abs()
            return importance.sort_values('abs_importance', ascending=False).head(top_n)

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
        })
        importance['abs_importance'] = importance['importance'].abs()
        return importance.sort_values('abs_importance', ascending=False).head(top_n)

    return pd.DataFrame(columns=['feature', 'importance'])
