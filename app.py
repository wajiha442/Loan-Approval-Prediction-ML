import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

st.set_page_config(page_title="Loan AI", page_icon="💰")
st.title("💰 Loan Approval Prediction AI")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "loan_approval_dataset.csv")

@st.cache_resource
def train_model():
    df = pd.read_csv(dataset_path)
    df.columns = df.columns.str.strip()
    
    for col in df.columns:
        if 'id' in col.lower():
            df.drop(columns=col, inplace=True)
    
    target = 'loan_status'
    
    # Store original values before encoding
    original_values = df[target].unique()
    
    # Encode: This will automatically assign 0 and 1
    le_target = LabelEncoder()
    df[target] = le_target.fit_transform(df[target])
    
    # Find which value is "Approved" and which is "Rejected"
    # The classes_ array stores the original labels in order [0, 1]
    class_0_original = le_target.classes_[0]  # What 0 represents
    class_1_original = le_target.classes_[1]  # What 1 represents
    
    le_education = LabelEncoder()
    le_self_employed = LabelEncoder()
    
    df['education'] = le_education.fit_transform(df['education'])
    df['self_employed'] = le_self_employed.fit_transform(df['self_employed'])
    
    feature_cols = ['no_of_dependents', 'education', 'self_employed', 'income_annum', 
                    'loan_amount', 'loan_term', 'cibil_score', 'commercial_assets_value', 
                    'luxury_assets_value', 'bank_asset_value']
    
    X = df[feature_cols]
    y = df[target]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, feature_cols, df, le_education, le_self_employed, le_target

try:
    model, feature_cols, full_df, le_education, le_self_employed, le_target = train_model()
    
    # Display what the encoding means
    st.success(f"✅ Model Trained on {len(full_df)} rows!")
    
    # Show the encoding mapping clearly
    encoding_map = f"""
    **Label Encoding:**
    - `0` = **{le_target.classes_[0]}**
    - `1` = **{le_target.classes_[1]}**
    """
    st.info(encoding_map)
    
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

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
    # Encode user inputs to match training data
    education_encoded = 1 if education == "Graduate" else 0
    self_employed_encoded = 1 if self_employed == "Yes" else 0
    
    # Create user input dataframe
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
    
    user_data = user_data[feature_cols]
    
    # Get prediction (0 or 1)
    prediction = model.predict(user_data)[0]
    
    # Get probability for each class
    prediction_proba = model.predict_proba(user_data)[0]
    
    # Decode the prediction back to original label
    predicted_status = le_target.inverse_transform([prediction])[0]
    
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    # Debug info (you can remove this later)
    st.write(f"**Debug:** Model predicted: {prediction} which means '{predicted_status}'")
    
    # Check if the prediction is "Approved" or "Rejected"
    # Common patterns in datasets: "Approved", "Rejected", "approve", "reject", " Approved", " Rejected"
    if "approve" in str(predicted_status).lower().strip():
        # APPROVED
        st.balloons()
        st.success("### ✅ Loan Status: **APPROVED**")
        st.metric("Approval Confidence", f"{prediction_proba[prediction]*100:.1f}%")
        
        st.markdown("#### 🎉 Congratulations!")
        st.write("Your loan application has been approved based on the following factors:")
        
        # Show positive factors
        positive_factors = []
        if cibil_score >= 700:
            positive_factors.append("✅ Excellent CIBIL Score")
        elif cibil_score >= 600:
            positive_factors.append("✅ Good CIBIL Score")
        
        if income_annum >= loan_amount * 0.4:
            positive_factors.append("✅ Strong Income-to-Loan Ratio")
        
        total_assets = commercial_assets + luxury_assets + bank_assets
        if total_assets >= loan_amount * 0.5:
            positive_factors.append("✅ Sufficient Asset Coverage")
        
        if positive_factors:
            for factor in positive_factors:
                st.write(factor)
    
    elif "reject" in str(predicted_status).lower().strip():
        # REJECTED
        st.error("### ❌ Loan Status: **REJECTED**")
        st.metric("Rejection Confidence", f"{prediction_proba[prediction]*100:.1f}%")
        
        st.markdown("#### 😔 Reasons for Rejection")
        
        # Show reasons for rejection
        rejection_reasons = []
        if cibil_score < 500:
            rejection_reasons.append("❌ CIBIL Score too low (below 500)")
        elif cibil_score < 600:
            rejection_reasons.append("⚠️ CIBIL Score needs improvement")
        
        if income_annum < loan_amount * 0.3:
            rejection_reasons.append("❌ Income-to-Loan ratio is too low")
        
        total_assets = commercial_assets + luxury_assets + bank_assets
        if total_assets < loan_amount * 0.3:
            rejection_reasons.append("❌ Insufficient asset coverage")
        
        if loan_term > 15 and cibil_score < 650:
            rejection_reasons.append("⚠️ Long loan term with moderate credit score")
        
        if rejection_reasons:
            for reason in rejection_reasons:
                st.write(reason)
        
        st.markdown("#### 💡 Suggestions to Improve")
        st.write("- Improve your CIBIL score by paying bills on time")
        st.write("- Increase your annual income or reduce loan amount")
        st.write("- Build more assets before reapplying")
        st.write("- Consider a shorter loan term")
    
    else:
        # Fallback if status is neither
        st.warning(f"### ⚠️ Prediction: {predicted_status}")
        st.write(f"Confidence: {prediction_proba[prediction]*100:.1f}%")
    
    # Show probability breakdown
    st.markdown("---")
    st.markdown("#### 📊 Detailed Probability Breakdown")
    col_a, col_b = st.columns(2)
    with col_a:
        status_0 = le_target.classes_[0]
        st.metric(f"{status_0}", f"{prediction_proba[0]*100:.1f}%")
    with col_b:
        status_1 = le_target.classes_[1]
        st.metric(f"{status_1}", f"{prediction_proba[1]*100:.1f}%")
    
    # Additional insights
    st.markdown("### 💡 Key Financial Metrics")
    col_m1, col_m2, col_m3 = st.columns(3)
    
    with col_m1:
        loan_to_income = (loan_amount / income_annum) * 100
        st.metric("Loan-to-Income Ratio", f"{loan_to_income:.1f}%")
    
    with col_m2:
        total_assets = commercial_assets + luxury_assets + bank_assets
        st.metric("Total Assets", f"₹{total_assets:,.0f}")
    
    with col_m3:
        asset_coverage = (total_assets / loan_amount) * 100 if loan_amount > 0 else 0
        st.metric("Asset Coverage", f"{asset_coverage:.1f}%")

# Dataset Preview
with st.expander("📂 View Training Dataset"):
    st.dataframe(full_df.head(10))
    st.write(f"**Total Records:** {len(full_df)}")
    st.write(f"**Features:** {', '.join(feature_cols)}")
    
    # Show distribution of loan status in training data
    st.markdown("#### Target Distribution")
    status_counts = full_df['loan_status'].value_counts()
    for idx, count in status_counts.items():
        original_label = le_target.inverse_transform([idx])[0]
        st.write(f"- {original_label}: {count} samples")

# Model Information
with st.expander("🎯 Model Information"):
    st.write("**Model Type:** Random Forest Classifier")
    st.write("**Number of Trees:** 100")
    st.write(f"**Training Samples:** {len(full_df)}")
    st.write(f"**Features Used:** {len(feature_cols)}")
    st.write("**Features:**")
    for i, feat in enumerate(feature_cols, 1):
        st.write(f"{i}. {feat}")
