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
    df.columns = df.columns.str.strip()
    df = df.dropna()
except Exception as e:
    st.error(f"Data Load Error: {e}")
    st.stop()

target_col = None
for c in df.columns:
    if 'status' in c.lower():
        target_col = c
        break

if not target_col:
    st.error("❌ Target column (Loan Status) nahi mila!")
    st.stop()

def clean_target(val):
    val = str(val).strip().lower()
    if 'app' in val or '1' in val or 'y' in val:
        return 1
    return 0

df[target_col] = df[target_col].apply(clean_target)

if df[target_col].nunique() < 2:
    st.warning("⚠️ Dataset mein sirf aik hi class hai. Testing ke liye dummy data add kiya ja raha hai.")
    dummy_row = df.iloc[0].copy()
    dummy_row[target_col] = 1 if df[target_col].iloc[0] == 0 else 0
    df = pd.concat([df, pd.DataFrame([dummy_row])], ignore_index=True)

y = df[target_col]
X = df.drop(columns=[target_col])

X = pd.get_dummies(X)

if X.empty:
    st.error("❌ Features (X) khali hain. Dataset check karein.")
    st.stop()

try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    st.success(f"✅ Model Trained! (Total Rows: {len(df)})")
except Exception as e:
    st.error(f"Training Error: {e}")
    st.stop()

st.write("---")
st.subheader("📝 Enter Details:")
user_input = {}
input_cols = X.columns[:10]

for col in input_cols:
    user_input[col] = st.number_input(f"Enter {col}", value=0.0)

if st.button("Predict"):
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
    st.write("Columns:", list(df.columns))
    st.dataframe(df.head())
