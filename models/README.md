# Trained model artifacts

This folder is populated by `python -m src.train` (see the [README](../README.md)) and is not
committed to GitHub — everything in it is derived from the local `data/bank_marketing.csv`, which
also isn't committed.

Running the training command creates:

- `final_model.joblib` — the fitted scikit-learn pipeline used by `src/predict.py` and the
  Streamlit app.
- `feature_schema.json` — the categories and numeric ranges observed during training, used to
  warn when a prediction request falls outside what the model has ever seen.
- `economic_snapshots.json` — median economic-indicator readings per contact month, used by the
  app to auto-fill those columns from the selected month instead of asking a user to pick values
  that were never seen in combination during training.

`reports/metrics.json` (which *is* committed) records the held-out test metrics and decision
threshold from the training run referenced in the README and final report.
