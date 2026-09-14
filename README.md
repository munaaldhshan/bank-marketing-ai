# AI-Powered Bank Marketing Campaign Response Prediction

This repository contains a professional, business-focused machine learning project for predicting whether a customer is likely to subscribe to a term deposit after a direct-marketing campaign. The project is designed for both coursework and portfolio use, with emphasis on data quality, realistic business framing, and deployment-oriented modeling.

## Overview

Banks must decide which customers to contact and how to allocate campaign resources efficiently. This project builds a binary classification model to estimate the probability that a customer will subscribe to a term deposit. The objective is to support better targeting and resource allocation, not to guarantee a revenue uplift.

## Business problem

Direct marketing costs money and time. If a bank contacts too many customers who are unlikely to subscribe, it wastes staff capacity and reduces campaign efficiency. A predictive model can help prioritize customers with a higher likelihood of conversion while avoiding over-contacting low-probability prospects.

## Objective

The machine learning objective is to predict the target variable `y`, where:

- `yes` = the customer subscribed to a term deposit
- `no` = the customer did not subscribe

This is a binary classification problem with a strong business constraint: the model should use only information available before or during campaign prioritization, not information that becomes known only after the call has already occurred.

## Dataset

The project uses the public Bank Marketing dataset from Kaggle, based on the UCI banking-marketing campaign data.

### Verified dataset facts

- File: `data/bank_marketing.csv`
- Format: semicolon-delimited CSV
- Shape: 41,188 rows x 21 columns
- Target variable: `y`
- Target distribution: `no = 36,548`, `yes = 4,640`
- Positive rate: 11.27%
- Duplicate rows: 12
- Missing values: 0

A full raw-data copy is intentionally not committed to GitHub. The dataset should be downloaded from Kaggle and placed in the `data/` folder before running notebooks or training scripts.

## Features

The dataset contains a mix of:

- Customer profile variables: age, job, marital status, education, default, housing, loan
- Campaign variables: contact, month, day_of_week, duration, campaign, pdays, previous, poutcome
- Economic indicators: emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed
- Outcome: y

## Methodology

Data Collection
→ Data Cleaning
→ EDA
→ Feature Engineering
→ Model Training
→ Model Evaluation
→ Explainability
→ Business Recommendations
→ Deployment

## Models evaluated

- Logistic regression
- Decision tree
- Random forest
- Gradient boosting / XGBoost-style approaches when appropriate
- Benchmark leakage model including `duration` for educational comparison

## Results

The realistic production model excludes `duration` because it represents the duration of the last phone call and is only known after the call occurs. This is a major data leakage risk.

### Realistic model without `duration`

- Accuracy: 0.835
- Precision: 0.368
- Recall: 0.647
- F1-score: 0.469
- ROC-AUC: 0.801

### Leakage benchmark including `duration`

- Accuracy: 0.865
- Precision: 0.451
- Recall: 0.912
- F1-score: 0.604
- ROC-AUC: 0.943

This comparison strongly demonstrates the lesson that a leakage model can look much better while being unsuitable for real-world decision making.

## Key insights

- The dataset is heavily imbalanced, with only about 11% positive responses.
- Contact month and customer type are relevant patterns for campaign performance.
- `duration` is highly predictive but is not valid for pre-contact targeting because it leaks post-call information.
- The bank should prioritize high-probability customers while avoiding unnecessary calls to low-probability prospects.

## Business recommendations

- Prioritize outreach to customers with the highest predicted probability of subscribing.
- Use the model as decision support rather than a fully automatic targeting tool.
- Consider campaign timing, contact method, and previous campaign outcomes in marketing strategy.
- Define operational thresholds using business cost trade-offs rather than defaulting to a 0.5 decision threshold.

## Limitations

- This is a historical marketing dataset, not a direct causal study.
- The model identifies association, not guaranteed causal effects.
- Performance depends on class imbalance and the practical choice of evaluation metric.
- The realistic model excludes post-call variables, which is correct for business use but may lower apparent performance.

## Deployment

The project includes a Streamlit app that can accept customer and campaign attributes and return a prediction probability, predicted class, and priority label. The app should clearly mention that predictions are decision-support estimates rather than guaranteed customer outcomes.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
jupyter notebook
```

Or run the app:

```bash
streamlit run app/app.py
```

## Project structure

- `data/` — dataset instructions and local raw data
- `notebooks/` — exploratory and modeling notebooks
- `src/` — reusable data-loading, preprocessing, training, and prediction logic
- `app/` — Streamlit application
- `models/` — trained pipeline artifacts
- `reports/` — project report and figures
- `tests/` — validation tests

## Dataset citation

This project uses the public Kaggle Bank Marketing dataset by Volodymyr Gavrysh and the associated UCI banking marketing campaign resource. Please check the original dataset license and Kaggle terms before redistribution or commercial reuse.

## Important leakage note

The `duration` feature must not be used in the production model. It reflects the length of the last phone call and becomes known only after the call has happened. Including it would create data leakage and would make the model look stronger than it is in a real pre-contact setting.
