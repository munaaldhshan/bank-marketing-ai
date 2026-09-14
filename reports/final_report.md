# Final Project Report

## Introduction

This project addresses a common business problem in direct marketing: identifying which customers are most likely to respond to a bank campaign and open a term deposit. The objective is to build an explainable and realistic classification model that supports efficient outreach without using information that only becomes known after the call has happened.

## Business problem

Banks spend significant time and money on marketing outreach. A major challenge is deciding which customers to contact and which to deprioritize. A predictive model can help the bank focus on customers with a higher expected probability of subscribing while reducing wasted calls to lower-probability leads.

This project does not claim guaranteed revenue gains. Instead, it supports better decision-making and more efficient use of campaign resources.

## Dataset

The data source is the public Bank Marketing dataset from Kaggle, based on the UCI banking campaign dataset. The raw file is a semicolon-delimited CSV with 41,188 rows and 21 columns. The target variable is `y`, with labels `yes` and `no`.

Important observed facts:

- Target distribution: `no = 36,548`, `yes = 4,640`
- Positive rate: 11.27%
- Duplicate rows: 12
- Missing values: 0
- Key leakage variable: `duration`

## Methodology

The workflow follows a professional ML lifecycle:

1. Data inspection and schema validation
2. Business and data understanding
3. EDA and campaign-pattern analysis
4. Data cleaning and preprocessing
5. Stratified train/test split
6. Baseline and model comparison
7. Leakage demonstration and explainability
8. Deployment-ready pipeline and documentation

## EDA

The dataset contains customer demographics, campaign attributes, and macroeconomic indicators. Customer factors such as age, job, marital status, education, housing, and personal loan status are relevant business signals. Campaign behaviors such as contact method, month, number of contacts, and previous campaign outcome also matter.

The class imbalance is significant: only ~11% of customers subscribed. That means accuracy alone is not a sufficient evaluation metric, and precision/recall must be interpreted in business context.

The strongest observed leakage pattern is with `duration`, because it is the length of the call and is only known after the call has happened. This variable is therefore excluded from the realistic model.

## Preprocessing

The preprocessing logic is designed to avoid leakage and to respect the train/test split. Numeric variables are scaled, categorical variables are one-hot encoded, and unknown values are handled carefully. Importantly, preprocessing is fit only on the training data to prevent information leakage from the test set.

The realistic feature set excludes `duration`, because predicting before contact means that call length is not available at decision time.

## Models

The project explicitly distinguishes between:

- Realistic production model: excludes `duration`
- Leakage benchmark model: includes `duration` for educational comparison only

This contrast is a central learning objective in the project.

## Evaluation

The realistic model excluding `duration` achieved the following held-out test performance:

- Accuracy: 0.835
- Precision: 0.368
- Recall: 0.647
- F1-score: 0.469
- ROC-AUC: 0.801

The leakage benchmark including `duration` achieved:

- Accuracy: 0.865
- Precision: 0.451
- Recall: 0.912
- F1-score: 0.604
- ROC-AUC: 0.943

This demonstrates the danger of leakage: the benchmark looks much stronger, but it is not valid for a real pre-contact targeting scenario because it uses post-call information.

## Model selection

The realistic model is the correct deployment choice because it uses information available before or during the campaign prioritization workflow. It is also more aligned with business needs, safer from a ML-engineering perspective, and more defensible in an academic or portfolio presentation.

## Explainability

The model should explain which variables are associated with higher subscription probability. In practice, customer attributes, prior campaign behavior, and contact timing are among the most useful predictors. These are statistical associations, not causal effects. The model is not proving that a feature causes a customer to subscribe; it is identifying patterns associated with subscription likelihood.

## Business implications

The model can help the bank focus outreach and staff effort on higher-probability customers. This reduces waste, improves campaign efficiency, and supports more informed decision making. It should be used as a decision-support tool, not as a fully automated system that decides customer contact without human oversight.

## Ethical considerations

Responsible AI practices are important in marketing. The project must consider fairness, privacy, explainability, and the risk of unfair targeting. Sensitive attributes and proxy variables should be reviewed before deployment. Human judgment remains essential, especially when the model is used to define customer outreach priorities.

## Limitations

- This is historical campaign data and not a causal experiment.
- The model identifies association rather than causality.
- Class imbalance affects evaluation and threshold selection.
- The production model excludes `duration`, which is correct for business realism but reduces the apparent predictive power compared with a leakage model.

## Conclusion

This project demonstrates a realistic AI-for-business workflow: understanding the business problem, validating the data, documenting leakage, building a sensible preprocessing pipeline, and evaluating models with business context in mind. The central lesson is that high model performance does not automatically mean a model is useful in the real world if it uses information that would not be available at the decision point.

## Future work

- Add additional models such as random forest and gradient boosting.
- Tune hyperparameters using cross-validation.
- Optimize thresholds using a business cost framework.
- Deploy the final pipeline through a Streamlit app.
- Generate stronger explainability outputs with SHAP or permutation importance.
