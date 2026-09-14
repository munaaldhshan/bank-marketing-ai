"""Minimal Streamlit app for bank-marketing prediction."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.predict import load_model, predict_customer


st.title('AI-Powered Bank Marketing Campaign Response Prediction')

st.write(
    'This tool estimates the probability that a customer will subscribe to a bank term deposit. '
    'It is a decision-support tool and does not guarantee future customer behavior.'
)

# Minimal app shell: users can provide values for key fields.
# In a full version, this would be expanded to the real Kaggle feature set.
if 'model' not in st.session_state:
    try:
        st.session_state.model = load_model('models/final_model.joblib')
    except FileNotFoundError:
        st.warning('The trained model artifact has not been created yet. Train the model first and save it to models/final_model.joblib.')
        st.stop()

with st.form('customer_form'):
    age = st.slider('Age', 18, 90, 35)
    job = st.selectbox('Job', ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed'])
    marital = st.selectbox('Marital status', ['single', 'married', 'divorced'])
    education = st.selectbox('Education', ['primary', 'secondary', 'tertiary', 'unknown'])
    housing = st.selectbox('Housing loan', ['yes', 'no', 'unknown'])
    loan = st.selectbox('Personal loan', ['yes', 'no', 'unknown'])
    contact = st.selectbox('Contact type', ['cellular', 'telephone', 'unknown'])
    month = st.selectbox('Month', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    campaign = st.slider('Number of contacts during this campaign', 1, 20, 2)
    pdays = st.slider('Days since last contact', -1, 1000, -1)
    previous = st.slider('Previous contacts', 0, 20, 0)
    poutcome = st.selectbox('Previous campaign outcome', ['failure', 'nonexistent', 'success', 'unknown'])

    submitted = st.form_submit_button('Predict')

if submitted:
    record = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': 'no',
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'month': month,
        'day_of_week': 'mon',
        'duration': 0,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome,
        'emp.var.rate': 0.0,
        'cons.price.idx': 0.0,
        'cons.conf.idx': 0.0,
        'euribor3m': 0.0,
        'nr.employed': 0.0,
    }

    probability, label = predict_customer(st.session_state.model, record)
    st.metric('Predicted subscription probability', f'{probability * 100:.1f}%')
    st.write(f'Predicted class: {label}')
    if probability >= 0.7:
        st.write('Priority: High')
    elif probability >= 0.4:
        st.write('Priority: Medium')
    else:
        st.write('Priority: Low')
    st.caption('This prediction is designed to support marketing prioritization, not to guarantee a future subscription outcome.')
