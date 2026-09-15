"""Streamlit app: score one customer's likelihood of subscribing to a term deposit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ECONOMIC_SNAPSHOTS_PATH, METRICS_PATH, MODEL_PATH, PDAYS_NEVER_CONTACTED, SCHEMA_PATH
from src.predict import load_model, load_schema, predict_customer

st.set_page_config(page_title='Bank Marketing Response Prediction')
st.title('AI-Powered Bank Marketing Campaign Response Prediction')
st.write(
    'Estimates the probability that a customer will subscribe to a term deposit, to help '
    'prioritize call-campaign outreach. This is decision support, not a guarantee: the model '
    'finds patterns in past campaigns, and correlation with subscribing is not the same as '
    'causing it.'
)


@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    schema = load_schema(SCHEMA_PATH)
    snapshots = json.loads(ECONOMIC_SNAPSHOTS_PATH.read_text()) if ECONOMIC_SNAPSHOTS_PATH.exists() else {}
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    return model, schema, snapshots, metrics


try:
    model, schema, economic_snapshots, metrics = load_artifacts()
except FileNotFoundError as exc:
    st.warning(f'{exc}')
    st.stop()

if schema is None:
    st.error(
        'The saved model has no feature_schema.json next to it, so this app cannot show valid '
        'options or catch out-of-range inputs. Re-run `python -m src.train` to regenerate both.'
    )
    st.stop()

cats = schema['categorical']
nums = schema['numeric']
threshold = metrics.get('decision_threshold', 0.5)
priority_high = metrics.get('priority_high_cutoff', 0.7)
priority_medium = metrics.get('priority_medium_cutoff', 0.4)

with st.form('customer_form'):
    st.subheader('Customer profile')
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider('Age', int(nums['age']['min']), int(nums['age']['max']), 35)
        job = st.selectbox('Job', cats['job'])
        marital = st.selectbox('Marital status', cats['marital'])
        education = st.selectbox('Education', cats['education'])
    with col2:
        default = st.selectbox('Has credit in default?', cats['default'])
        housing = st.selectbox('Has a housing loan?', cats['housing'])
        loan = st.selectbox('Has a personal loan?', cats['loan'])

    st.subheader('This campaign')
    col3, col4 = st.columns(2)
    with col3:
        contact = st.selectbox('Contact type', cats['contact'])
        month = st.selectbox('Contact month', cats['month'])
        day_of_week = st.selectbox('Contact day of week', cats['day_of_week'])
    with col4:
        campaign = st.slider(
            'Number of contacts so far this campaign',
            1, max(int(nums['campaign']['max']), 5), 2,
        )

    st.subheader('Previous campaigns')
    contacted_before = st.checkbox('Contacted in a previous campaign?')
    if contacted_before:
        pdays_range = schema.get('pdays_if_contacted_before', {'min': 0, 'max': 30})
        pdays = st.slider(
            'Days since that previous contact',
            int(pdays_range['min']), int(pdays_range['max']), int(pdays_range['min']),
        )
        previous = st.slider('Number of previous contacts', 1, max(int(nums['previous']['max']), 3), 1)
        poutcome = st.selectbox(
            'Outcome of the previous campaign',
            [c for c in cats['poutcome'] if c != 'nonexistent'] or cats['poutcome'],
        )
    else:
        pdays, previous, poutcome = PDAYS_NEVER_CONTACTED, 0, 'nonexistent'

    submitted = st.form_submit_button('Predict')

if submitted:
    record = {
        'age': age, 'job': job, 'marital': marital, 'education': education,
        'default': default, 'housing': housing, 'loan': loan,
        'contact': contact, 'month': month, 'day_of_week': day_of_week,
        'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome,
        # Derived from the month, not user input: see src/schema.py for why these five
        # national indicators can't be set independently without producing combinations
        # that never occurred historically.
        **economic_snapshots.get(month, {}),
    }

    probability, label, warnings = predict_customer(model, record, schema, threshold=threshold)

    for warning in warnings:
        st.warning(warning)

    st.metric('Predicted subscription probability', f'{probability * 100:.1f}%')
    st.write(f'Predicted class (at the {threshold:.0%} decision threshold tuned for this model): **{label}**')

    if probability >= priority_high:
        priority = 'High — top ~10% of prospects in this dataset'
    elif probability >= priority_medium:
        priority = 'Medium — next ~20%'
    else:
        priority = 'Low — remaining ~70%'
    st.write(f'Outreach priority: **{priority}**')

    with st.expander('Economic conditions used for this prediction'):
        st.write(
            f"Auto-filled from typical `{month}` conditions in the training data "
            '(these move together nationally and are not something an individual customer chooses):'
        )
        st.json(economic_snapshots.get(month, {}))

    st.caption(
        'This estimate supports prioritization; it does not guarantee the customer will or '
        "will not subscribe. It also does not use the call's duration, which is unknown before "
        'the call happens and would leak the outcome if included.'
    )
