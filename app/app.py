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

st.set_page_config(page_title='Bank Marketing Response Prediction', page_icon='🏦')

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f5f9ff 0%, #eef4fb 100%);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    h1 {
        color: #0f172a;
        letter-spacing: -0.03em;
    }
    .stAlert {
        border-radius: 0.75rem;
    }
    div[data-testid="stForm"] {
        background: rgba(255,255,255,0.7);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 1rem;
        padding: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
except (FileNotFoundError, ValueError, TypeError, AttributeError) as exc:
    st.warning(f'{exc}')
    st.info('Rebuild the model artifact with `python -m src.train` to restore scoring.')
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

st.info(
    'Model status: the app loaded a compatible model artifact. If this warning appears after a '
    'recent environment or package update, retrain the model with `python -m src.train`.'
)

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

    if probability >= priority_high:
        priority = 'High'
        priority_detail = 'This customer is in the top ~10% of likely responders in this dataset.'
    elif probability >= priority_medium:
        priority = 'Medium'
        priority_detail = 'This customer is in the next ~20% of likely responders in this dataset.'
    else:
        priority = 'Low'
        priority_detail = 'This customer is in the lower-risk group for this campaign based on current patterns.'

    st.markdown(f'### Predicted subscription probability: {probability * 100:.1f}%')
    st.metric('Predicted subscription probability', f'{probability * 100:.1f}%')
    st.write(f'Predicted class (at the {threshold:.0%} decision threshold tuned for this model): **{label.upper()}**')
    st.write(f'Priority: **{priority}**')
    st.write(priority_detail)

    if label == 'yes':
        explanation = (
            'This profile shows characteristics that historically correspond to a higher likelihood '
            'of subscribing, especially given the selected campaign and customer history.'
        )
    else:
        explanation = (
            'This profile shows characteristics that historically correspond to a lower likelihood '
            'of subscribing, even though the final outcome is not guaranteed.'
        )
    st.info(explanation)

    st.caption(
        'Prediction: this estimate is not guaranteed. It is intended to support outreach '
        'prioritization and should be used as decision support rather than a certainty.'
    )

    with st.expander('Economic conditions used for this prediction'):
        st.write(
            f"Auto-filled from typical `{month}` conditions in the training data "
            '(these move together nationally and are not something an individual customer chooses):'
        )
        st.json(economic_snapshots.get(month, {}))
