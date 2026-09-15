# Final Project Report

## Executive Summary

This project builds a decision-support model for a bank's term-deposit marketing campaign:
given a customer's profile and campaign history, estimate the probability they'll subscribe, so
outreach can be prioritized toward the customers most likely to convert. The central
methodological finding is a data-leakage lesson: the single most predictive column in the raw
data, `duration` (the length of the sales call), is only known *after* the call happens, and a
model built on it looks dramatically better while being unusable for its actual purpose. The
production model excludes it; a leakage benchmark is kept only to measure and illustrate the
effect.

## Business Context

Direct-marketing campaigns have finite staff capacity. Contacting too many low-probability
prospects wastes it; identifying likely responders early lets the bank allocate that capacity
more effectively. The model's job is to rank prospects by likelihood, not to make an autonomous
accept/reject decision or to explain *why* a customer would subscribe in any causal sense.

## Dataset and Data Quality

The public Bank Marketing dataset from the UCI Machine Learning Repository (also mirrored on
Kaggle): a semicolon-delimited CSV, 41,188 rows, 21 columns, target `y` (`yes`/`no`).

- Target distribution: `no` = 36,548, `yes` = 4,640 — an 11.27% positive rate
- 12 exact duplicate rows in the raw data; a further 1,784 rows become duplicates once `duration`
  is dropped (i.e. the same customer profile contacted more than once) — cleaning removes
  duplicates on the full row *before* dropping `duration`, so these are not conflated
- No missing values in the technical sense, but six columns use the literal string `'unknown'`
  in place of missing data — most notably `default` (20.9% unknown). `'unknown'` is kept as its
  own category rather than imputed away: it measurably correlates with the outcome (`default` =
  `'unknown'` converts at roughly half the rate of `default` = `'no'`), so collapsing it into
  the mode would discard real signal.

With an 11% positive rate, accuracy alone is a poor measure of model quality — a model that
always predicts `no` is already ~89% "accurate" while being useless for targeting.

## Data Understanding and EDA

Full analysis in `notebooks/02_eda.ipynb`. The clearest findings:

**Prior contact history is the strongest signal in the data.** Customers contacted in an earlier
campaign convert at 64%, against 9% for first-time contacts — roughly a 6x lift. It's also fully
known before the call, making it the highest-value legitimate feature available.

**Contact month tracks the underlying economic climate, not seasonal mood.** Median `euribor3m`
by month correlates with monthly conversion rate at r = -0.91. The four lowest-rate months
(Sep/Oct/Dec under 1%, Mar at 1.5%) convert at 44-51%; the four highest-rate, highest-volume
months (May-Aug, ~4.9%) convert at only 6-11%. Because `month` and the five economic-indicator
columns trace the same shifting timeline, `notebooks/04_model_comparison.ipynb` includes a
purely time-ordered train/test split alongside the standard random split as a sanity check —
more on this below.

**Job type carries a life-stage signal.** Students and retirees convert well above average (31%,
25%); blue-collar and services workers convert below it (7-8%) — plausibly reflecting age and
available time rather than occupation itself being causal.

## Methodology

1. Confirm the raw schema, target structure, and data-quality issues (`notebooks/01`).
2. EDA around legitimate, pre-contact signal (`notebooks/02`).
3. Shared cleaning/preprocessing (`src/preprocessing.py`): drop exact duplicates on the full row,
   drop `duration`, keep `'unknown'` as a category, median-impute and scale numeric columns,
   mode-impute and one-hot encode categorical ones — one definition, reused by every model so
   they're compared on identical footing (`notebooks/03`).
4. Compare four candidate models with 5-fold cross-validation plus a held-out test set
   (`notebooks/04`).
5. Train the best candidate as the production pipeline, tune its decision threshold on a
   validation split, and evaluate once on the test set (`src/final_pipeline.py`).
6. Explain the production model with permutation importance (`notebooks/05`).
7. Serve it through a Streamlit app with input validation (`app/app.py`).

## Modeling Results

### The leakage comparison

Same split, same preprocessing shape, the only difference is whether `duration` is a feature
(`notebooks/03`, logistic regression in both cases):

| Metric | Realistic (no `duration`) | Leakage benchmark (+ `duration`) |
|---|---|---|
| Accuracy | 0.8983 | 0.9095 |
| Precision | 0.6500 | 0.6522 |
| Recall | 0.2101 | 0.4224 |
| F1 | 0.3176 | 0.5128 |
| ROC-AUC | 0.8004 | 0.9390 |
| Average precision | 0.4541 | 0.6014 |

The leakage model's recall and ROC-AUC are dramatically higher because it has effectively
learned "long calls convert" — true, but useless, since nobody knows a call's length before
making it.

### Model comparison

Four candidates, 5-fold CV on the training split plus one held-out test evaluation
(`notebooks/04`): logistic regression, a decision tree, a random forest, and gradient boosting
(scikit-learn's `HistGradientBoostingClassifier`). Gradient boosting had the best test ROC-AUC
(0.813) and average precision (0.492), with a cross-validation standard deviation comparable to
the other candidates — the improvement isn't one lucky split. It's the model
`src/final_pipeline.py` trains for production.

### Threshold selection

At the default 0.5 threshold, the production model's recall is only 25%, missing 3 of every 4
actual subscribers — an artifact of the 0.5 cutoff implicitly weighing a false positive (an
unwanted call) as costly as a false negative (a missed sale), which doesn't match the business
reality. The threshold is instead chosen by maximizing F1 on a validation split carved out of the
training data, distinct from the test set the final numbers below are reported on:

| Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| 0.50 (default) | 0.9006 | 0.6518 | 0.2522 | 0.3636 | 0.8133 |
| 0.25 (tuned on validation) | 0.8832 | 0.4845 | 0.5722 | 0.5247 | 0.8133 |

(from the training run recorded in `reports/metrics.json`; re-running `python -m src.train`
reproduces these up to floating-point/seed differences)

### A caution about the random split

Because `month` and the economic indicators trace one shifting timeline, `notebooks/04` also
trains on the first 80% of rows in file order and tests on the last 20%. ROC-AUC drops from 0.81
(random split) to 0.63, and the base rate itself shifts from 6.4% to 30.8% between the two
periods. This doesn't invalidate the random-split numbers reported above — they're an honest
estimate for data drawn from the same mixture of regimes seen in training — but it does mean the
model shouldn't be assumed to generalize indefinitely to a genuinely new economic period without
monitoring and retraining.

## Interpretation and Business Implications

The model's output is a prioritization score, not a certainty. It supports questions like: which
customers should be called first given limited capacity, and which segments are unusually
responsive right now. It should not be used to exclude customers from being offered products, and
its outreach-priority bands (in the Streamlit app) are calibrated to the actual predicted-score
distribution — top ~10% = High, next ~20% = Medium — rather than fixed cutoffs like 0.7/0.4, which
an 11%-base-rate model would almost never reach.

## Explainability and Responsible Use

`HistGradientBoostingClassifier` (the production model) has no built-in `feature_importances_`,
so `notebooks/05` measures importance by permutation on a held-out split instead: shuffle one
original column at a time and measure the drop in held-out ROC-AUC. By this measure, the economic
indicators and contact type dominate, while prior-contact features (`pdays`, `poutcome`) rank
lower than their large *raw* effect in the EDA would suggest. This is not a contradiction —
permutation importance measures each feature's *unique* contribution once every other feature is
already in the model, and the prior-contact features are highly redundant with each other and
correlated with timing information the economic columns also carry. The takeaway for a business
reader: prior-contact history is still real, strong signal; a lower permutation-importance rank
reflects redundancy with other features, not the signal disappearing.

None of this is causal. The model identifies statistical association in historical campaign data,
not a mechanism the bank can act on beyond prioritization.

## Limitations

- Historical observational data, not an experiment — associations, not proven causal effects.
- `poutcome`, `pdays`, and `previous` only carry information for customers contacted in a prior
  campaign. A genuinely new customer is scored using the `'nonexistent'`/`999` sentinel values,
  which resemble the largest and lowest-converting segment in the training data — the model has
  comparatively little to distinguish among brand-new prospects.
- Sensitive to economic regime, as shown by the temporal-split check above; should be
  re-validated and retrained as conditions drift from what's in
  `models/economic_snapshots.json`.
- The positive class is relatively rare, which is why threshold and priority-band decisions use
  business-informed cutoffs rather than the default 0.5 or fixed percentages.

## Conclusion

The project's central lesson holds up under a full, reproducible pipeline: a decision-support
model must be built and evaluated using only information available at the moment the business
decision is made. Excluding `duration` costs apparent performance (ROC-AUC 0.80 vs. 0.94) but is
the only version of the model that could actually be deployed to decide who to call next. Beyond
that lesson, the project also demonstrates the supporting practices a deployable model needs and
the original version of this project lacked: a properly separated validation split for threshold
selection, input validation against the training distribution, business-calibrated priority
bands, and an explicit check of how performance holds up outside the training period.

## Future Directions

- Recalibrate the threshold and priority bands periodically as the economic regime shifts, rather
  than treating the current training run as permanent.
- Extend the temporal-split check into a proper rolling-window backtest across more than one
  split point.
- Layer campaign-capacity constraints directly into the priority bands (e.g. "top N customers
  this week") instead of fixed percentiles.
- Explore calibrated probability outputs (e.g. `CalibratedClassifierCV`) if the raw predicted
  probabilities are used for anything beyond ranking.
