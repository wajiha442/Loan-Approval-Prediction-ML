 import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

st.set_page_config(page_title="Loan AI", page_icon="💰")
st.title("💰 Loan Approval Prediction AI")

# --- Load Data ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "loan_approval_dataset.csv")

@st.cache_resource
def train_model():
    df = pd.read_csv(dataset_path)
    df.columns = df.columns.str.strip()
    
    # Drop ID column if exists
    for col in df.columns:
        if 'id' in col.lower():
            df.drop(columns=col, inplace=True)
    
    # The actual columns from your Kaggle dataset:
    # no_of_dependents, education, self_employed, income_annum, loan_amount, 
    # loan_term, cibil_score, commercial_assets_value, luxury_assets_value, 
    # bank_asset_value, loan_status
    
    # Target encoding: Approved = 1, Rejected = 0
    target = 'loan_status'
    le_target = LabelEncoder()
    df[target] = le_target.fit_transform(df[target])
    
    # Encode categorical features
    le_education = LabelEncoder()
    le_self_employed = LabelEncoder()
    
    df['education'] = le_education.fit_transform(df['education'])
    df['self_employed'] = le_self_employed.fit_transform(df['self_employed'])
    
    # Features and target
    feature_cols = ['no_of_dependents', 'education', 'self_employed', 'income_annum', 
                    'loan_amount', 'loan_term', 'cibil_score', 'commercial_assets_value', 
                    'luxury_assets_value', 'bank_asset_value']
    
    X = df[feature_cols]
    y = df[target]
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, feature_cols, df, le_education, le_self_employed, le_target

try:
    model, feature_cols, full_df, le_education, le_self_employed, le_target = train_model()
    st.success(f"✅ Model Trained on {len(full_df)} rows!")
    
    # Show what 0 and 1 mean
    st.info(f"**Encoding:** {le_target.classes_[0]} = 0, {le_target.classes_[1]} = 1")
    
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- User Input UI ---
st.subheader("📝 Applicant Information")

col1, col2 = st.columns(2)

with col1:
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=2)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    income_annum = st.number_input("Annual Income (₹)", min_value=200000, max_value=10000000, value=5000000, step=100000)
    loan_amount = st.number_input("Loan Amount (₹)", min_value=300000, max_value=40000000, value=15000000, step=100000)

with col2:
    loan_term = st.number_input("Loan Term (years)", min_value=2, max_value=20, value=10)
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=650)
    commercial_assets = st.number_input("Commercial Assets Value (₹)", min_value=0, max_value=20000000, value=5000000, step=100000)
    luxury_assets = st.number_input("Luxury Assets Value (₹)", min_value=300000, max_value=40000000, value=15000000, step=100000)
    bank_assets = st.number_input("Bank Asset Value (₹)", min_value=0, max_value=15000000, value=5000000, step=100000)

if st.button("🔍 Check Loan Eligibility", type="primary"):
    # Encode user inputs
    education_encoded = 1 if education == "Graduate" else 0
    self_employed_encoded = 1 if self_employed == "Yes" else 0
    
    # Create input dataframe
    user_data = pd.DataFrame([{
        'no_of_dependents': no_of_dependents,
        'education': education_encoded,
        'self_employed': self_employed_encoded,
        'income_annum': income_annum,
        'loan_amount': loan_amount,
        'loan_term': loan_term,
        'cibil_score': cibil_score,
        'commercial_assets_value': commercial_assets,
        'luxury_assets_value': luxury_assets,
        'bank_asset_value': bank_assets
    }])
    
    # Ensure correct column order
    user_data = user_data[feature_cols]
    
    # Make prediction
    prediction = model.predict(user_data)[0]
    prediction_proba = model.predict_proba(user_data)[0]
    
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    # Determine approval status based on encoding
    # Check which class is "Approved" in the original data
    approved_label = le_target.classes_[1] if "approved" in str(le_target.classes_[1]).lower() else le_target.classes_[0]
    approved_value = 1 if "approved" in str(le_target.classes_[1]).lower() else 0
    
    if prediction == approved_value:
        st.balloons()
        st.success("### ✅ Loan Status: **APPROVED**")
        st.metric("Approval Confidence", f"{prediction_proba[approved_value]*100:.1f}%")
    else:
        st.error("### ❌ Loan Status: **REJECTED**")
        st.metric("Rejection Confidence", f"{prediction_proba[1-approved_value]*100:.1f}%")
    
    # Show probability breakdown
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(f"Probability: {le_target.classes_[0]}", f"{prediction_proba[0]*100:.1f}%")
    with col_b:
        st.metric(f"Probability: {le_target.classes_[1]}", f"{prediction_proba[1]*100:.1f}%")
    
    # Feature importance feedback
    st.markdown("### 💡 Key Factors")
    if cibil_score < 500:
        st.warning("⚠️ Low CIBIL Score may affect approval")
    if income_annum < loan_amount * 0.3:
        st.warning("⚠️ Income to Loan ratio is low")
    if cibil_score >= 700:
        st.success("✅ Good CIBIL Score")

# Data Preview
with st.expander("📂 View Training Dataset"):
    st.dataframe(full_df.head(10))
    st.write(f"**Total Records:** {len(full_df)}")
    st.write(f"**Features:** {', '.join(feature_cols)}")

# Model Performance
with st.expander("🎯 Model Information"):
    st.write("**Model Type:** Random Forest Classifier")
    st.write("**Number of Trees:** 100")
    st.write(f"**Training Samples:** {len(full_df)}")
    st.write(f"**Features Used:** {len(feature_cols)}")
