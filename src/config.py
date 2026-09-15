"""Project-wide constants and paths.

Every path is anchored at the project root, so the code behaves the same when it
is run from the repo root, from inside notebooks/, or by Streamlit.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
DATA_PATH = DATA_DIR / 'bank_marketing.csv'
MODELS_DIR = PROJECT_ROOT / 'models'
MODEL_PATH = MODELS_DIR / 'final_model.joblib'
SCHEMA_PATH = MODELS_DIR / 'feature_schema.json'
ECONOMIC_SNAPSHOTS_PATH = MODELS_DIR / 'economic_snapshots.json'
REPORTS_DIR = PROJECT_ROOT / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'
METRICS_PATH = REPORTS_DIR / 'metrics.json'

# Official UCI archive (same data as the Kaggle mirror, no account needed).
DATASET_URL = 'https://archive.ics.uci.edu/static/public/222/bank+marketing.zip'

TARGET = 'y'
# Known only after the call has happened, so it must never be a model input.
LEAKAGE_COLUMNS = ('duration',)
# In this version of the dataset, pdays == 999 means "never contacted before".
PDAYS_NEVER_CONTACTED = 999

CUSTOMER_COLUMNS = ('age', 'job', 'marital', 'education', 'default', 'housing', 'loan')
CAMPAIGN_COLUMNS = ('contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome')
ECONOMIC_COLUMNS = ('emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed')
FEATURE_COLUMNS = CUSTOMER_COLUMNS + CAMPAIGN_COLUMNS + ECONOMIC_COLUMNS

RANDOM_STATE = 42
TEST_SIZE = 0.2
# Fraction of the remaining (non-test) data held out to pick the decision
# threshold, so the threshold is never chosen on the same rows used to report
# test metrics. Net split is roughly 64% train / 16% validation / 20% test.
VALIDATION_SIZE = 0.2
CV_FOLDS = 5
DEFAULT_THRESHOLD = 0.5
