"""Feature-importance extraction for the trained pipeline.

Two paths, chosen automatically:

- Linear or tree models expose `coef_` / `feature_importances_` per *transformed* column,
  which for a one-hot-encoded categorical means one row per category (`job_admin.`,
  `job_student`, ...). Those are aggregated back to one score per original column, since
  "job matters" is what a business reader wants, not fifteen separate category rows.
- `HistGradientBoostingClassifier` — the production model — exposes neither attribute at all,
  so importance is instead measured directly by permutation: shuffle one original column at a
  time and see how much held-out ROC-AUC drops. This is model-agnostic (works for any fitted
  pipeline) and, unlike impurity-based importance, isn't biased toward high-cardinality columns.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


def _native_importance(model: Pipeline) -> pd.DataFrame | None:
    classifier = model.named_steps['classifier']
    if hasattr(classifier, 'coef_'):
        values = np.abs(classifier.coef_[0])
    elif hasattr(classifier, 'feature_importances_'):
        values = classifier.feature_importances_
    else:
        return None

    # Transformed names look like 'num__age' or 'cat__job_admin.'; recover the original
    # column by dropping the 'num__'/'cat__' prefix, then (for one-hot columns) the
    # category suffix, by matching against the source columns the transformer was given.
    preprocessor = model.named_steps['preprocessor']
    transformed_names = list(preprocessor.get_feature_names_out())
    source_columns = [col for _, _, cols in preprocessor.transformers_ for col in cols]
    source_columns = sorted(source_columns, key=len, reverse=True)

    def original_column(name: str) -> str:
        stripped = name.split('__', 1)[-1]
        for col in source_columns:
            if stripped == col or stripped.startswith(col + '_'):
                return col
        return stripped

    grouped = pd.DataFrame({'feature': [original_column(n) for n in transformed_names], 'value': values})
    return grouped.groupby('feature')['value'].sum().rename('importance').reset_index()


def top_feature_importance(
    model: Pipeline,
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    top_n: int = 15,
    scoring: str = 'roc_auc',
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Top features by importance, one row per original column.

    `X`/`y` are only required as a fallback for models with no native importance
    (currently just HistGradientBoosting); pass a held-out split, never the
    training data, since permuting a memorized training row understates how
    much the model actually depends on that column.
    """
    importance = _native_importance(model)
    if importance is None:
        if X is None or y is None:
            raise ValueError(
                f"{type(model.named_steps['classifier']).__name__} has no built-in importances; "
                'pass a held-out X and y so permutation importance can be computed instead.'
            )
        result = permutation_importance(
            model, X, y, scoring=scoring, n_repeats=n_repeats, random_state=random_state, n_jobs=-1,
        )
        importance = pd.DataFrame({'feature': X.columns, 'importance': result.importances_mean})

    importance['abs_importance'] = importance['importance'].abs()
    return importance.sort_values('abs_importance', ascending=False).head(top_n).reset_index(drop=True)
