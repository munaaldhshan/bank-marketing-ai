# Final Project Report

## Executive Summary

This project addresses a real business problem in direct marketing: identifying which customers are most likely to respond to a bank campaign and subscribe to a term deposit. The goal is not simply to maximize a model score in a notebook, but to build a decision-support system that a bank could actually use when deciding who to contact, which customers to prioritize, and how to allocate campaign effort more efficiently.

The central methodological lesson of the project is that a model must be evaluated under realistic business conditions. In this case, the variable `duration` is highly predictive, but it is not valid for pre-contact decision-making because it represents the length of the call and is only known after the call has already happened. This makes `duration` a classic data-leakage feature. The production model therefore excludes it, while a leakage benchmark is used only to illustrate why the feature is misleading in a real deployment scenario.

## Business Context

Banks spend substantial time, money, and staff capacity on direct-marketing campaigns. A major operational challenge is deciding which customers should be prioritized for outreach and which should be deprioritized. If a campaign contacts too many low-probability prospects, the bank wastes both time and budget. If it identifies the most likely responders early, the institution can allocate resources more efficiently and improve campaign effectiveness.

The problem is therefore not only statistical; it is operational and strategic. The model should support real marketing decisions and help protect staff time, but it should not be treated as an automatic approval engine or a causal explanation of customer behavior.

## Dataset and Data Quality

The project uses the public Bank Marketing dataset from Kaggle, based on the widely referenced UCI campaign dataset. The raw data file is a semicolon-delimited CSV and contains 41,188 rows and 21 columns. The target variable is `y`, which contains the binary outcomes `yes` and `no`.

Key dataset observations:

- Target distribution: `no = 36,548`, `yes = 4,640`
- Positive rate: 11.27%
- Duplicate rows: 12
- Missing values: 0
- Major leakage variable: `duration`

The class imbalance is an important business reality. Only about 11% of customers subscribed, so accuracy is not a sufficient model quality measure. In a campaign setting, the cost of false positives and false negatives matters, which is why precision, recall, F1-score, and ROC-AUC must be interpreted together.

## Data Understanding and EDA

The exploratory analysis revealed several meaningful customer and campaign patterns. Customer characteristics such as job type, age, education, housing, and loan status were relevant, but they were not by themselves enough to define a reliable targeting system. Campaign-specific variables such as month, contact type, previous campaign outcome, and contact count were also informative.

From a business standpoint, the most useful insight was not only which variables were predictive, but which variables were predictive at the correct decision point. This distinction is crucial. The dataset contains variables that are strongly associated with subscription behavior, but not all of them are valid during pre-contact targeting. In particular, the variable `duration` is strongly predictive because it measures how long the call lasted, and it becomes known only after the call has finished.

This is exactly why the deployment model excludes it.

## Methodology

The project follows a professional machine-learning workflow:

1. Confirm the raw file schema and target structure.
2. Perform data quality checks and leakage review.
3. Build an exploratory analysis around real customer and campaign patterns.
4. Apply cleaning and preprocessing without leaking information from the test set.
5. Train a realistic baseline model that excludes `duration`.
6. Compare it with a leakage benchmark that includes `duration` for teaching purposes.
7. Evaluate using business-aware metrics.
8. Document trade-offs and prepare a deployment-ready pipeline.

The preprocessing pipeline uses a standard `ColumnTransformer` pattern with a median imputer and scaling for numeric columns, and a most-frequent imputer plus one-hot encoding for categorical columns. Model training uses stratified train/test splitting, and all preprocessing is fit only on the training data to preserve the integrity of the evaluation.

## Modeling Results

The realistic production model excludes `duration`, reflecting the fact that call duration is not available before a customer is contacted. The held-out test results are as follows:

- Accuracy: 0.8255
- Precision: 0.3615
- Recall: 0.6457
- F1-score: 0.4635
- ROC-AUC: 0.7988

The leakage benchmark includes `duration` and is not intended for business deployment. Its results are:

- Accuracy: 0.8651
- Precision: 0.4512
- Recall: 0.9116
- F1-score: 0.6036
- ROC-AUC: 0.9438

This comparison illustrates the central lesson of the project: a leakage model can appear dramatically better, but it is not a valid decision-support model because it uses information that would not exist at the point when the decision is made. The realistic model is therefore the correct model to use for actual campaign targeting.

## Interpretation and Business Implications

From a business perspective, the output of the final model should be interpreted as a prioritization score rather than a guaranteed customer outcome. The bank is not predicting certainty; it is estimating relative likelihood. This is an appropriate framing for a direct-marketing setting in which the goal is to allocate limited outreach resources to the highest-value opportunities.

The model helps answer practical questions such as:

- Which customers should be contacted first?
- Which customer segments appear unusually responsive?
- How should the bank balance campaign coverage against resource constraints?

These are operational decisions, and the model is best used as a decision-support tool in that context.

## Explainability and Responsible Use

Explainability matters because business stakeholders need to understand which signals are driving the prediction. In general, the model identifies patterns associated with customer type, campaign timing, and prior contact behavior. These features help explain why some prospects appear more likely to subscribe than others.

However, this is not causal inference. The model does not prove that a given feature causes subscription behavior. It identifies statistical relationships in historical campaign data. That is useful for prioritization, but it should not be treated as a universal causal law.

Responsible use also requires human oversight. The model should not be used as a fully automated decision engine for customer exclusion or unfair targeting. Marketing outreach must remain transparent, ethically grounded, and aligned with business and regulatory standards.

## Limitations

This project has several important limitations:

- It uses historical campaign data rather than an experimental design.
- It identifies associations rather than causal effects.
- The positive class is relatively rare, which affects threshold decisions.
- The realistic model excludes `duration`, which is correct for deployment but reduces apparent predictive power compared with leakage-based benchmarks.

These limitations are not weaknesses of the project; they are part of the realistic ML narrative. They reflect how real business problems are framed in practice.

## Conclusion

This project demonstrates the difference between a model that looks impressive in the lab and a model that is genuinely useful in a real business context. The major lesson is straightforward: a decision-support model must use only information that exists at the time the business decision is made.

By excluding post-call leakage features such as `duration`, the final system is more faithful to the real bank marketing problem. It supports better customer prioritization, reduces wasted outreach, and provides a robust foundation for a practical AI-for-business use case.

The project therefore succeeds not only as a technical modeling exercise, but as a realistic demonstration of how machine learning should be designed, evaluated, and explained in a commercial setting.

## Future Directions

Possible next steps include:

- Comparing a wider range of models such as gradient boosting and random forests.
- Tuning model thresholds using business cost assumptions rather than the default 0.5 cutoff.
- Adding SHAP or permutation-based explainability for stakeholder communication.
- Deploying the final trained pipeline in Streamlit for end-user interaction.
- Presenting the final model in a business-friendly story aligned with campaign performance metrics and operational decision-making.
