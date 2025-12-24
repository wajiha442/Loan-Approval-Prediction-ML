import streamlit as st
import pandas as pd
import os
from utils.helper import load_model, preprocess_input

st.set_page_config(page_title="Loan Approval Predictor", page_icon="💰", layout="wide")
st.title("Loan Approval Prediction 💰")
st.write("Enter the applicant's details to predict loan approval.")

# Dataset (optional)
DATA_PATH = "data/loan_approval_dataset.csv"
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    st.write("Sample Dataset:")
    st.dataframe(df.head())
else:
    st.warning("Dataset not found in `data/` folder. Predictions will still work.")

# Sidebar Inputs
st.sidebar.header("Applicant Details")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.sidebar.number_input("Applicant Income", min_value=0)
coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
loan_term = st.sidebar.number_input("Loan Term (in months)", min_value=0)
credit_history = st.sidebar.selectbox("Credit History", [1.0, 0.0])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

input_data = {
    "Gender": gender,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": credit_history,
    "Property_Area": property_area
}

# Models
model_paths = {
    "Logistic Regression": "models/logistic_model.pkl",
    "Random Forest": "models/random_forest_model.pkl",
    "SVC": "models/svc_model.pkl"
}

loaded_models = {}
model_columns = {}  # To store expected columns for each model

# Load models safely
for name, path in model_paths.items():
    model = load_model(path)
    if model:
        loaded_models[name] = model
        # Capture columns if model has it (for dummy preprocessing)
        if hasattr(model, 'columns'):
            model_columns[name] = model.columns
    else:
        st.warning(f"{name} model not found at `{path}`. This model will be skipped.")

# Prediction
if st.button("Predict Loan Approval"):
    if not loaded_models:
        st.error("No models available. Please upload pickle files in `models/` folder.")
    else:
        for name, model in loaded_models.items():
            cols = model_columns.get(name)
            processed_input = preprocess_input(input_data, model_columns=cols)
            prediction = model.predict(processed_input)[0]
            result = "Approved ✅" if prediction == 1 else "Rejected ❌"
            st.write(f"{name}: {result}")

