  import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
    
    # Handling Categorical Data (Education, Gender etc)
    # Hum 'Loan_Status' ko target banayenge
    target = 'Loan_Status'
    df[target] = df[target].apply(lambda x: 1 if 'Y' in str(x).upper() or 'APP' in str(x).upper() else 0)
    
    # Sirf kaam ki columns select karein
    features = ['Gender', 'Married', 'Education', 'Self_Employed', 'ApplicantIncome', 'LoanAmount', 'Credit_History']
    X = df[features].copy()
    y = df[target]
    
    # Categorical text ko numbers mein badalna
    X = pd.get_dummies(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns, df

try:
    model, model_cols, full_df = train_model()
    st.success(f"✅ Model Trained on {len(full_df)} rows!")
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# --- User UI ---
st.subheader("📝 Applicant Information")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_emp = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    income = st.number_input("Applicant Income", min_value=0, value=5000)
    loan_amt = st.number_input("Loan Amount", min_value=0, value=150)
    # Credit History sabse important hai: 1.0 matlab record acha hai, 0.0 matlab bura
    credit = st.selectbox("Credit History Score", [1.0, 0.0], help="1.0 is Good, 0.0 is Bad")

if st.button("Check Eligibility"):
    # Input data prepare karna
    user_data = pd.DataFrame([{
        'ApplicantIncome': income,
        'LoanAmount': loan_amt,
        'Credit_History': credit,
        'Gender_Male': 1 if gender == "Male" else 0,
        'Gender_Female': 1 if gender == "Female" else 0,
        'Married_Yes': 1 if married == "Yes" else 0,
        'Married_No': 1 if married == "No" else 0,
        'Education_Graduate': 1 if education == "Graduate" else 0,
        'Education_Not Graduate': 1 if education == "Not Graduate" else 0,
        'Self_Employed_Yes': 1 if self_emp == "Yes" else 0,
        'Self_Employed_No': 1 if self_emp == "No" else 0
    }])
    
    # Reindex takay columns model ke mutabiq hon
    user_data = user_data.reindex(columns=model_cols, fill_value=0)
    
    prediction = model.predict(user_data)
    
    if prediction[0] == 1:
        st.balloons()
        st.success("### Status: Approved ✅")
    else:
        st.error("### Status: Rejected ❌")

with st.expander("Data Preview"):
    st.write(full_df.head())
