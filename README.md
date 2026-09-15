# AI-Powered Bank Marketing Campaign Response Prediction

A machine learning project for predicting whether a customer is likely to subscribe to a term
deposit after a direct-marketing call, built around a business-critical data-leakage lesson: the
most predictive column in the raw data (`duration`, the length of the call) is only known *after*
the call happens, so a usable model has to be evaluated without it.

## Overview

Banks must decide which customers to contact and how to allocate limited campaign capacity. This
project builds a binary classifier that estimates the probability a customer will subscribe,
using only information available *before* a call is made, so it can support real targeting
decisions rather than describe outcomes after the fact.

## Dataset

The public [Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) from
the UCI Machine Learning Repository (also mirrored on
[Kaggle](https://www.kaggle.com/datasets/volodymyrgavrysh/bank-marketing-campaigns-dataset)).

- File: `bank-additional-full.csv`, semicolon-delimited
- Shape: 41,188 rows x 21 columns
- Target: `y` (`yes` = subscribed, `no` = did not) — `no` = 36,548, `yes` = 4,640 (**11.27%** positive rate)
- 12 exact duplicate rows; 1,784 additional rows become duplicates once `duration` is excluded
  (repeat contacts of the same customer profile) — see [notebook 01](notebooks/01_data_understanding.ipynb)
- No `NaN`s, but six columns use the string `'unknown'` in place of missing data (`default` is
  21% unknown); `'unknown'` is kept as its own category rather than imputed away, since it
  measurably correlates with the outcome

The raw file isn't committed to the repo. Get it with:

```bash
python -m src.download_data
```

which fetches it directly from the official UCI archive (no account needed) into
`data/bank_marketing.csv`. See [data/README.md](data/README.md) for the manual alternative.

## The leakage lesson

`duration` — how long the call lasted — is extremely predictive of `y`, for an obvious reason:
long, engaged calls end in a sale far more often. But nobody knows how long a call will last
*before making it*, so a production model can't use it as an input without silently assuming
information it will never actually have at decision time. [Notebook 03](notebooks/03_preprocessing_and_baseline.ipynb)
quantifies the effect directly — same split, same preprocessing, `duration` included or excluded:

| | Realistic (no `duration`) | Leakage benchmark (+ `duration`) |
|---|---|---|
| Accuracy | 0.898 | 0.910 |
| Precision | 0.650 | 0.652 |
| Recall | 0.210 | 0.422 |
| F1 | 0.318 | 0.513 |
| ROC-AUC | 0.800 | 0.939 |

Every model and figure elsewhere in this project excludes `duration`; `src/config.LEAKAGE_COLUMNS`
is the single place that decision is encoded, and `src/preprocessing.clean_dataset` enforces it.

## What the EDA found

Full analysis in [notebook 02](notebooks/02_eda.ipynb):

- **Prior contact history is the strongest legitimate signal in the data.** Customers contacted
  in an earlier campaign convert at **64%**, versus **9%** for first-time contacts — a ~6x lift,
  and one the bank can act on since it's fully known before the call.
- **Contact month correlates strongly with the economic climate** (r = -0.91 between median
  `euribor3m` and monthly conversion rate): low-rate months (Sep/Oct/Dec, under 1%) convert at
  44-51%, while the high-rate, high-volume months (May-Aug, ~4.9%) convert at only 6-11%. Because
  `month` and the five economic indicator columns trace the same underlying timeline, notebook 04
  includes a temporal train/test split as a sanity check alongside the standard random split.
- Students and retirees convert well above average (31%, 25%); blue-collar and services workers
  convert below it (~7-8%) — plausibly a life-stage effect (more flexible time, different savings
  priorities) rather than job title itself.

## Models compared

[Notebook 04](notebooks/04_model_comparison.ipynb) compares four candidates with 5-fold
cross-validation plus a held-out test set: logistic regression, a decision tree, a random forest,
and gradient boosting (`HistGradientBoostingClassifier`). Gradient boosting had the best test
ROC-AUC (0.813) and average precision (0.492), and is the model `src/final_pipeline.py` trains for
production.

### Choosing a decision threshold

At the default 0.5 cutoff, the production model's recall is only **25%** — it misses 3 of every 4
actual subscribers, because the 0.5 threshold implicitly assumes a false positive (an unwanted
call) costs as much as a false negative (a missed sale), which isn't the right assumption for
campaign targeting. The threshold is instead tuned to maximize F1 on a **validation split carved
out of the training data** (never the test set the final numbers are reported on):

| Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| 0.50 (default) | 0.901 | 0.652 | 0.252 | 0.364 | 0.813 |
| **0.25 (tuned)** | **0.883** | **0.484** | **0.572** | **0.525** | 0.813 |

(exact numbers from the last training run are in [`reports/metrics.json`](reports/metrics.json),
regenerated by `python -m src.train`)

## Explainability

[Notebook 05](notebooks/05_model_explainability.ipynb) ranks feature importance by permutation on
a held-out split (`HistGradientBoostingClassifier` has no built-in `feature_importances_`, unlike
the random forest or decision tree candidates). The economic indicators and contact type dominate;
prior-contact features rank lower here than their large *raw* effect in the EDA would suggest —
the notebook explains why that isn't a contradiction (redundancy between correlated features, not
the signal disappearing). This measures statistical association in historical data, not a causal
effect — see the notebook's "Responsible use" section.

## Deployment

A Streamlit app (`app/app.py`) scores one customer at a time:

- Every dropdown is populated from the categories the model actually saw during training
  (`models/feature_schema.json`), and out-of-range inputs are flagged with a warning rather than
  silently mishandled.
- The five economic-indicator columns are national figures for the time of contact, not something
  an individual customer has — the app derives them from the selected month
  (`models/economic_snapshots.json`) instead of asking the user to guess values that may never
  have co-occurred historically.
- The predicted label uses the tuned decision threshold above, not a hardcoded 0.5, and outreach
  priority (High/Medium/Low) is based on percentiles of the model's own validation-set
  predictions — with an 11% base rate, fixed cutoffs like 0.7/0.4 are almost never reached.

```bash
streamlit run app/app.py
```

## Project structure

```
data/        dataset instructions; the CSV itself is downloaded, not committed
notebooks/   01 data understanding · 02 EDA · 03 baseline vs. leakage · 04 model comparison · 05 explainability
src/         data loading, preprocessing, modeling, training, prediction, EDA and explainability code
app/         the Streamlit app
models/      trained pipeline + schema + economic snapshots (generated locally, not committed)
reports/     final report, figures, and reports/metrics.json (the reported metrics, committed)
tests/       pytest suite: preprocessing, modeling, comparison, the full pipeline, and the app itself
```

## Installation and usage

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.download_data   # fetch the dataset into data/
python -m src.train           # train the production model, save it + its schema + metrics
streamlit run app/app.py      # try it
```

Run the notebooks with `jupyter notebook` from either the repo root or `notebooks/` — each one
locates the project root itself. Run the test suite with `pytest`.

## Business recommendations

- Prioritize outreach to customers with the highest predicted probability of subscribing; the
  app's High/Medium/Low bands correspond to roughly the top 10% / next 20% / remaining 70% of
  prospects.
- Use the model as decision support, not a fully automatic targeting or exclusion tool.
- Prior-contact history is the strongest actionable signal available before a call — a customer
  who converted before is worth recontacting.
- Re-validate periodically: notebook 04's temporal-split check found ROC-AUC drops from 0.81 to
  0.63 when trained on an earlier period and tested on a later one with a very different base
  rate, so a static model shouldn't be assumed to hold indefinitely as economic conditions shift.

## Limitations

- This is historical observational data, not an experiment — the model finds associations, not
  guaranteed causal effects.
- `poutcome`/`pdays`/`previous` only carry information for customers contacted in a prior
  campaign; a genuinely new customer is scored using the `'nonexistent'`/`999` sentinel values,
  which resemble the largest, lowest-converting segment in training.
- Performance is sensitive to the economic regime at prediction time (see above); the model
  should be monitored and retrained as conditions drift from `models/economic_snapshots.json`.

## Dataset citation

Public UCI Bank Marketing dataset (also mirrored on Kaggle by Volodymyr Gavrysh). Check the
original dataset license and terms before redistribution or commercial reuse.
