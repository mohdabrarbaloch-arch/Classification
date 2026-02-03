import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.title("🏦 Loan Approval System")

# Load models
models = {
    "Logistic Regression": joblib.load("logistic.pkl"),
    "Decision Tree": joblib.load("decision_tree.pkl"),
    "Random Forest": joblib.load("random_forest.pkl"),
    "Naive Bayes": joblib.load("naive_bayes.pkl")
}

scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# Model selection
model_name = st.selectbox("Select Model", list(models.keys()))
model = models[model_name]

# Inputs
income = st.number_input("Annual Income", 1000, 500000, step=1000)
credit_score = st.number_input("Credit Score", 300, 850)
loan_amount = st.number_input("Loan Amount", 1000, 500000, step=1000)
interest_rate = st.number_input("Interest Rate (%)", 0.0, 30.0)
employment_years = st.number_input("Employment Years", 0, 40)

occupation = st.selectbox("Occupation Status", ["Employed", "Self-employed", "Unemployed"])
product_type = st.selectbox("Product Type", ["Personal", "Home", "Auto"])
loan_intent = st.selectbox("Loan Intent", ["Education", "Medical", "Business", "Personal"])

# Prepare input
input_dict = {
    "annual_income": income,
    "credit_score": credit_score,
    "loan_amount": loan_amount,
    "interest_rate": interest_rate,
    "employment_years": employment_years,
    "occupation_status": occupation,
    "product_type": product_type,
    "loan_intent": loan_intent
}

input_df = pd.DataFrame([input_dict])

# Encode and align features
input_encoded = pd.get_dummies(input_df)
for col in feature_names:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[feature_names]

# Scale and predict
input_scaled = scaler.transform(input_encoded)

if st.button("🔍 Predict"):
    result = model.predict(input_scaled)
    if result[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
