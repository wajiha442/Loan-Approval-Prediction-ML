import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

st.set_page_config(page_title="Loan Approval Prediction", page_icon="💰")
st.title("💰 Loan Approval Prediction App")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "loan_approval_dataset.csv")

if not os.path.exists(dataset_path):
    st.error("❌ Dataset file 'loan_approval_dataset.csv' nahi mili!")
    st.stop()

try:
    df = pd.read_csv(dataset_path, sep=None, engine='python')
    df.columns = df.columns.str.strip() # Remove spaces from names
    df = df.dropna()
except Exception as e:
    st.error(f"Data Load Error: {e}")
    st.stop()

# Target column find karna
target_col = None
for c in df.columns:
    if 'status' in c.lower():
        target_col = c
        break

if not target_col:
    st.error("❌ Target column (Loan Status) nahi mila!")
    st.stop()

# CLEANING: Values ko 0 aur 1 mein badalna
def clean_target(val):
    val = str(val).strip().lower()
    if 'app' in val or '1' in val or 'y' in val: # Approved, Y, 1
        return 1
    return 0

df[target_col] = df[target_col].apply(clean_target)

# Check if we have both 0 and 1
classes = df[target_col].unique()
if len(classes) < 2:
    st.error(f"❌ Error: Dataset mein sirf aik hi class hai: {classes}. Model ko train karne ke liye Approved aur Rejected dono tarah ka data chahiye.")
    st.write("Aapke dataset ki values ye hain:", df[target_col].value_counts())
    st.stop()

# Features handling
y = df[target_col]
X = df.drop(columns=[target_col])
X = pd.get_dummies(X)

try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    st.success(f"✅ Model Trained! (Rows: {len(X_train)})")
except Exception as e:
    st.error(f"Training Error: {e}")
    st.stop()

st.write("---")
st.subheader("📝 Enter Application Details:")
user_input = {}
input_cols = X.columns[:10]

# User input form
for col in input_cols:
    user_input[col] = st.number_input(f"Enter {col}", value=0.0)

if st.button("Predict Loan Status"):
    input_df = pd.DataFrame([user_input])
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0.0
    
    input_df = input_df[X.columns]
    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        st.balloons()
        st.success("### Result: Loan Approved ✅")
    else:
        st.error("### Result: Loan Rejected ❌")

with st.expander("View Data Preview"):
    st.dataframe(df.head())
