import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Loan Approval System",
    page_icon="🏦",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
/* Page background */
body {
    background-color: #f4f6f9;
}

/* Main Title */
.main-title {
    font-size: 38px;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(90deg, #6a11cb, #2575fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Sub title */
.sub-title {
    text-align: center;
    color: #34495e;
    font-size: 18px;
}

/* Card */
.card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0px 6px 16px rgba(0,0,0,0.12);
    margin-top: 20px;
}

/* Approved result */
.approved {
    background: linear-gradient(135deg, #d4efdf, #abebc6);
    color: #145a32;   /* ✅ visible text */
    padding: 15px;
    border-radius: 12px;
    font-size: 22px;
    text-align: center;
    font-weight: bold;
}

/* Rejected result */
.rejected {
    background: linear-gradient(135deg, #fadbd8, #f5b7b1);
    color: #922b21;
    padding: 15px;
    border-radius: 12px;
    font-size: 22px;
    text-align: center;
    font-weight: bold;
}

/* Model info box (accuracy waghera) */
.model-box {
    background: #f0f3f4;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
    color: #2c3e50;   /* ✅ accuracy ab clear dikhegi */
    font-size: 16px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<div class="main-title">🏦 Loan Approval System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-powered smart loan decision platform</div>', unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------
models = {
    "Logistic Regression": joblib.load("logistic.pkl"),
    "Decision Tree": joblib.load("decision_tree.pkl"),
    "Random Forest": joblib.load("random_forest.pkl"),
    "Naive Bayes": joblib.load("naive_bayes.pkl")
}

# Model accuracies (replace with your real scores)
model_accuracy = {
    "Logistic Regression": 0.81,
    "Decision Tree": 0.78,
    "Random Forest": 0.86,
    "Naive Bayes": 0.75
}

scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# ---------------- MODEL SELECTION ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)

model_name = st.selectbox("🧠 Select ML Model", list(models.keys()))
model = models[model_name]

st.markdown(f"""
<div class="model-box">
<b>Selected Model:</b> {model_name}<br>
<b>Accuracy:</b> {int(model_accuracy[model_name]*100)}%
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- INPUT SECTION ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📥 Applicant Information")

income = st.number_input("Annual Income", 1000, 500000, step=1000)
credit_score = st.number_input("Credit Score", 300, 850)
loan_amount = st.number_input("Loan Amount", 1000, 500000, step=1000)
interest_rate = st.number_input("Interest Rate (%)", 0.0, 30.0)
employment_years = st.number_input("Employment Years", 0, 40)

occupation = st.selectbox("Occupation Status", ["Employed", "Self-employed", "Unemployed"])
product_type = st.selectbox("Product Type", ["Personal", "Home", "Auto"])
loan_intent = st.selectbox("Loan Intent", ["Education", "Medical", "Business", "Personal"])

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREPARE INPUT ----------------
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

# Encode
input_encoded = pd.get_dummies(input_df)
for col in feature_names:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[feature_names]

# Scale
input_scaled = scaler.transform(input_encoded)

# ---------------- PREDICTION ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)

if st.button("🔍 Predict Loan Status"):
    with st.spinner("Analyzing application..."):
        time.sleep(1.5)

    result = model.predict(input_scaled)

    if result[0] == 1:
        st.markdown('<div class="approved">✅ Loan Approved</div>', unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown('<div class="rejected">❌ Loan Rejected</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("<br><center>🚀 Created by Abrar & Machine Learning</center>", unsafe_allow_html=True)
