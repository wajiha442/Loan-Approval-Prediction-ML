import streamlit as st
import numpy as np
import pickle

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Loan Approval Prediction App")
st.write("Enter applicant details to predict loan approval status.")

# ===============================
# Load Models
# ===============================
logistic_model = pickle.load(open("models/logistic_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# ===============================
# User Inputs
# ===============================
no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1)

education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

income_annum = st.number_input("Annual Income", min_value=0, value=500000)
loan_amount = st.number_input("Loan Amount", min_value=0, value=200000)
loan_term = st.number_input("Loan Term (Years)", min_value=1, value=10)

cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=750)

residential_assets = st.number_input("Residential Assets Value", min_value=0, value=100000)
commercial_assets = st.number_input("Commercial Assets Value", min_value=0, value=50000)
luxury_assets = st.number_input("Luxury Assets Value", min_value=0, value=20000)
bank_assets = st.number_input("Bank Asset Value", min_value=0, value=100000)

# ===============================
# Encode Inputs (Same as Training)
# ===============================
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

# ===============================
# Prediction
# ===============================
if st.button("Predict Loan Status"):
    
    input_data = np.array([[  
        no_of_dependents,
        education,
        self_employed,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets,
        commercial_assets,
        luxury_assets,
        bank_assets
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = logistic_model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
