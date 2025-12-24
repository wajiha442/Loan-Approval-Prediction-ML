 import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os

st.set_page_config(page_title="Loan Approval Prediction", page_icon="💰")
st.title("💰 Loan Approval Prediction App")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "loan_approval_dataset.csv")

if not os.path.exists(dataset_path):
    st.error("❌ Dataset file nahi mili!")
    st.stop()

try:
    df = pd.read_csv(dataset_path, sep=None, engine='python')
    df.columns = df.columns.str.strip()
    df = df.dropna()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

target_col = None
for c in df.columns:
    if 'status' in c.lower():
        target_col = c
        break

if not target_col:
    st.error("❌ 'loan_status' column nahi mila!")
    st.stop()

df[target_col] = df[target_col].astype(str).str.strip().str.lower()
df[target_col] = df[target_col].apply(lambda x: 1 if 'approved' in x else 0)

y = df[target_col]
X = df.drop(columns=[target_col])
X = pd.get_dummies(X)

if len(X) != len(y):
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]

try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    st.success(f"✅ Model Trained! (Rows: {len(X_train)})")
except Exception as e:
    st.error(f"Training Error: {e}")
    st.stop()

st.write("---")
st.subheader("Enter Details:")
user_input = {}
display_cols = X.columns[:12] 

for col in display_cols:
    user_input[col] = st.number_input(f"{col}", value=0.0)

if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    input_df = input_df[X.columns]
    prediction = model.predict(input_df)
    res = "Approved ✅" if prediction[0] == 1 else "Rejected ❌"
    st.header(f"Result: {res}")
