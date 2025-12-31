import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

st.set_page_config(page_title="Loan AI", page_icon="💰")
st.title("💰 Loan Approval Prediction AI")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "loan_approval_dataset.csv")

@st.cache_resource
def load_data_and_train():
    # Load dataset
    df = pd.read_csv(dataset_path)
    df.columns = df.columns.str.strip()
    
    # Drop ID column
    for col in df.columns:
        if 'id' in col.lower():
            df.drop(columns=col, inplace=True)
    
    # Check what values exist in loan_status
    st.write("**Original loan_status values:**", df['loan_status'].unique())
    
    # Prepare data
    target = 'loan_status'
    
    # Encode categorical features
    le_education = LabelEncoder()
    le_self_employed = LabelEncoder()
    le_target = LabelEncoder()
    
    df['education_encoded'] = le_education.fit_transform(df['education'])
    df['self_employed_encoded'] = le_self_employed.fit_transform(df['self_employed'])
    
    # IMPORTANT: Manually set the encoding for loan_status
    # Based on common sense: Approved = 1, Rejected = 0
    df['loan_status_encoded'] = df['loan_status'].apply(
        lambda x: 1 if 'approve' in str(x).lower() else 0
    )
    
    feature_cols = ['no_of_dependents', 'education_encoded', 'self_employed_encoded', 
                    'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 
                    'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']
    
    X = df[feature_cols]
    y = df['loan_status_encoded']
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Decision Tree (best model from your Kaggle results - 98% accuracy)
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(
        max_depth=10,  # Prevent overfitting
        min_samples_split=20,  # Require at least 20 samples to split
        min_samples_leaf=10,  # Require at least 10 samples in leaf
        random_state=42
    )
    model.fit(X_scaled, y)
    
    # Show class distribution
    st.write("**Class distribution in training data:**")
    st.write(f"- Rejected (0): {(y == 0).sum()} samples")
    st.write(f"- Approved (1): {(y == 1).sum()} samples")
    
    return model, scaler, feature_cols, df, le_education, le_self_employed

try:
    model, scaler, feature_cols, full_df, le_education, le_self_employed = load_data_and_train()
    st.success(f"✅ Model Trained on {len(full_df)} rows!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.markdown("---")
st.subheader("📝 Applicant Information")

col1, col2 = st.columns(2)

with col1:
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=2)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    income_annum = st.number_input("Annual Income (₹)", min_value=200000, max_value=10000000, value=3000000, step=100000)
    loan_amount = st.number_input("Loan Amount (₹)", min_value=300000, max_value=40000000, value=10000000, step=100000)

with col2:
    loan_term = st.number_input("Loan Term (years)", min_value=2, max_value=20, value=10)
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=550)
    commercial_assets = st.number_input("Commercial Assets Value (₹)", min_value=0, max_value=20000000, value=2000000, step=100000)
    luxury_assets = st.number_input("Luxury Assets Value (₹)", min_value=300000, max_value=40000000, value=8000000, step=100000)
    bank_assets = st.number_input("Bank Asset Value (₹)", min_value=0, max_value=15000000, value=3000000, step=100000)

if st.button("🔍 Check Loan Eligibility", type="primary"):
    # Encode inputs
    education_encoded = 1 if education == "Graduate" else 0
    self_employed_encoded = 1 if self_employed == "Yes" else 0
    
    # Create input array in exact same order as training
    user_input = np.array([[
        no_of_dependents,
        education_encoded,
        self_employed_encoded,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        commercial_assets,
        luxury_assets,
        bank_assets
    ]])
    
    # Scale the input
    user_input_scaled = scaler.transform(user_input)
    
    # Make prediction
    prediction = model.predict(user_input_scaled)[0]
    prediction_proba = model.predict_proba(user_input_scaled)[0]
    
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    # Show probabilities
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.metric("Rejection Probability", f"{prediction_proba[0]*100:.1f}%")
    with col_p2:
        st.metric("Approval Probability", f"{prediction_proba[1]*100:.1f}%")
    
    st.markdown("---")
    
    # Decision based on prediction
    if prediction == 1:  # Approved
        st.balloons()
        st.success("## ✅ Loan Status: **APPROVED**")
        st.write(f"**Confidence:** {prediction_proba[1]*100:.1f}%")
        
        st.markdown("### 🎉 Congratulations!")
        st.write("Your loan application has been approved!")
        
        # Show positive factors
        st.markdown("#### Strong Points:")
        if cibil_score >= 700:
            st.write("✅ Excellent CIBIL Score")
        elif cibil_score >= 600:
            st.write("✅ Good CIBIL Score")
        
        if income_annum >= loan_amount * 0.35:
            st.write("✅ Strong Income-to-Loan Ratio")
        
        total_assets = commercial_assets + luxury_assets + bank_assets
        if total_assets >= loan_amount * 0.5:
            st.write("✅ Sufficient Asset Coverage")
    
    else:  # Rejected (prediction == 0)
        st.error("## ❌ Loan Status: **REJECTED**")
        st.write(f"**Confidence:** {prediction_proba[0]*100:.1f}%")
        
        st.markdown("### 😔 Application Not Approved")
        
        # Show reasons
        st.markdown("#### Reasons for Rejection:")
        rejection_reasons = []
        
        if cibil_score < 500:
            rejection_reasons.append("❌ CIBIL Score is critically low (below 500)")
        elif cibil_score < 600:
            rejection_reasons.append("⚠️ CIBIL Score needs improvement (below 600)")
        
        loan_to_income = (loan_amount / income_annum) * 100
        if loan_to_income > 300:
            rejection_reasons.append(f"❌ Loan amount is too high compared to income ({loan_to_income:.0f}% ratio)")
        
        total_assets = commercial_assets + luxury_assets + bank_assets
        asset_coverage = (total_assets / loan_amount) * 100
        if asset_coverage < 50:
            rejection_reasons.append(f"❌ Insufficient asset coverage ({asset_coverage:.0f}%)")
        
        if loan_term > 15 and cibil_score < 650:
            rejection_reasons.append("⚠️ Long loan term combined with moderate credit score")
        
        if not rejection_reasons:
            rejection_reasons.append("⚠️ Overall risk assessment indicates high default probability")
        
        for reason in rejection_reasons:
            st.write(reason)
        
        st.markdown("#### 💡 How to Improve Your Application:")
        suggestions = []
        
        if cibil_score < 650:
            suggestions.append("📈 **Improve CIBIL Score:** Pay existing debts on time, reduce credit utilization")
        
        if loan_to_income > 250:
            suggestions.append("💰 **Adjust Loan Amount:** Consider applying for a lower amount or increase income")
        
        if asset_coverage < 60:
            suggestions.append("🏦 **Build Assets:** Increase savings and asset holdings before reapplying")
        
        if loan_term > 15:
            suggestions.append("⏰ **Shorter Term:** Consider reducing the loan term")
        
        for suggestion in suggestions:
            st.write(suggestion)
    
    # Financial metrics
    st.markdown("---")
    st.markdown("### 💡 Financial Analysis")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        loan_to_income = (loan_amount / income_annum) * 100
        st.metric("Loan-to-Income", f"{loan_to_income:.0f}%")
        if loan_to_income < 200:
            st.write("🟢 Healthy")
        elif loan_to_income < 300:
            st.write("🟡 Moderate")
        else:
            st.write("🔴 High Risk")
    
    with col_m2:
        monthly_emi = (loan_amount * 0.08) / (12 * loan_term)  # Rough estimate
        st.metric("Est. Monthly EMI", f"₹{monthly_emi:,.0f}")
    
    with col_m3:
        total_assets = commercial_assets + luxury_assets + bank_assets
        st.metric("Total Assets", f"₹{total_assets/1000000:.1f}M")
    
    with col_m4:
        asset_coverage = (total_assets / loan_amount) * 100
        st.metric("Asset Coverage", f"{asset_coverage:.0f}%")
        if asset_coverage >= 70:
            st.write("🟢 Strong")
        elif asset_coverage >= 40:
            st.write("🟡 Moderate")
        else:
            st.write("🔴 Weak")

# Test scenarios
with st.expander("🧪 Test Scenarios"):
    st.markdown("### Try these test cases:")
    
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.markdown("#### ✅ Should be APPROVED:")
        st.code("""
Income: ₹7,000,000
Loan: ₹15,000,000
CIBIL: 750
Term: 10 years
Assets: High
        """)
    
    with col_t2:
        st.markdown("#### ❌ Should be REJECTED:")
        st.code("""
Income: ₹2,000,000
Loan: ₹20,000,000
CIBIL: 400
Term: 18 years
Assets: Low
        """)

# Dataset preview
with st.expander("📂 View Training Dataset"):
    st.dataframe(full_df.head(20))
    st.write(f"**Total Records:** {len(full_df)}")

# Model info
with st.expander("🎯 Model Information"):
    st.write("**Model Type:** Decision Tree Classifier")
    st.write("**Prevents Overfitting:** Yes (max_depth=10, min_samples constraints)")
    st.write(f"**Training Samples:** {len(full_df)}")
    st.write(f"**Features:** {len(feature_cols)}")
    st.write("**Feature List:**")
    for feat in feature_cols:
        st.write(f"- {feat}")
