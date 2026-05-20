import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

st.set_page_config(page_title="Loan Approval System", page_icon="🏦", layout="centered")

st.markdown("""
<style>
.main-title { font-size: 38px; font-weight: bold; text-align: center; background: linear-gradient(90deg, #6a11cb, #2575fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.sub-title { text-align: center; color: #34495e; font-size: 18px; }
.card { background: white; padding: 20px; border-radius: 14px; box-shadow: 0px 6px 16px rgba(0,0,0,0.12); margin-top: 20px; }
.approved { background: linear-gradient(135deg, #d4efdf, #abebc6); color: #145a32; padding: 15px; border-radius: 12px; font-size: 22px; text-align: center; font-weight: bold; }
.rejected { background: linear-gradient(135deg, #fadbd8, #f5b7b1); color: #922b21; padding: 15px; border-radius: 12px; font-size: 22px; text-align: center; font-weight: bold; }
.model-box { background: #f0f3f4; padding: 12px; border-radius: 10px; margin-bottom: 10px; color: #2c3e50; font-size: 16px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def train_models():
    data = pd.read_csv("Loan_approval_data_2025.csv")
    data = data.dropna()
    y = data['loan_status']
    X = data.drop(['loan_status', 'customer_id'], axis=1)
    X = pd.get_dummies(X, columns=['occupation_status', 'product_type', 'loan_intent'], drop_first=True)
    X = X.select_dtypes(include=['number'])
    X = X.fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
        'Naive Bayes': GaussianNB(),
    }
    scores = {}
    for name, m in models.items():
        m.fit(X_train_s, y_train)
        scores[name] = round(m.score(X_test_s, y_test), 4)
    return models, scaler, X.columns.tolist(), scores

st.markdown('<div class="main-title">🏦 Loan Approval System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-powered smart loan decision platform</div>', unsafe_allow_html=True)

with st.spinner("Training ML models..."):
    models, scaler, feature_names, scores = train_models()

st.markdown('<div class="card">', unsafe_allow_html=True)
model_name = st.selectbox("🧠 Select ML Model", list(models.keys()))
model = models[model_name]
st.markdown(f"""
<div class="model-box">
<b>Selected Model:</b> {model_name}<br>
<b>Accuracy:</b> {int(scores[model_name]*100)}%
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

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

if st.button("🔍 Predict Loan Status", use_container_width=True, type="primary"):
    input_df = pd.DataFrame([{
        "annual_income": income, "credit_score": credit_score, "loan_amount": loan_amount,
        "interest_rate": interest_rate, "employment_years": employment_years,
        "occupation_status": occupation, "product_type": product_type, "loan_intent": loan_intent
    }])
    input_encoded = pd.get_dummies(input_df, columns=['occupation_status', 'product_type', 'loan_intent'], drop_first=True)
    input_encoded = input_encoded.select_dtypes(include=['number'])
    for col in feature_names:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[feature_names].fillna(0)
    input_scaled = scaler.transform(input_encoded)
    result = model.predict(input_scaled)[0]
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if result == 1:
        st.markdown('<div class="approved">✅ Loan Approved</div>', unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown('<div class="rejected">❌ Loan Rejected</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br><center>🚀 Created by Abrar & Machine Learning</center>", unsafe_allow_html=True)
