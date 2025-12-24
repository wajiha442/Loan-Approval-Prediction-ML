import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os

st.set_page_config(page_title="Loan Approval Prediction", page_icon="💰")
st.title("💰 Loan Approval Prediction App")

# 1. Path Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "loan_approval_dataset.csv")

if not os.path.exists(dataset_path):
    st.error(f"❌ Dataset file nahi mili!")
    st.stop()

# 2. Load Data with Auto-Separator Detection
try:
    # sep=None aur engine='python' se pandas khud dhoond lega ke separator comma hai ya semicolon
    df = pd.read_csv(dataset_path, sep=None, engine='python')
    
    # Column names se spaces khatam karein
    df.columns = df.columns.str.strip()
    
    st.write("### Data Preview:")
    st.dataframe(df.head())
    
    # Agar ab bhi 1 hi column dikh raha ho
    if len(df.columns) <= 1:
        st.error(f"⚠️ Dataset mein sirf {len(df.columns)} column mila. Shayad file format sahi nahi hai.")
        st.write("Columns found:", list(df.columns))
        st.stop()

except Exception as e:
    st.error(f"File load karne mein error: {e}")
    st.stop()

# 3. Target Column dhoondein
target_col = None
for c in df.columns:
    if 'status' in c.lower():
        target_col = c
        break

if not target_col:
    st.error("❌ 'loan_status' column nahi mila!")
    st.stop()

# 4. Data Preprocessing
# Target labels clean karein
df[target_col] = df[target_col].astype(str).str.strip().str.lower()
df[target_col] = df[target_col].apply(lambda x: 1 if 'approved' in x else 0)

# Features (X) - Sirf numeric columns lein taake error na aaye
X = df.drop(columns=[target_col])

# Important: Convert categorical columns to numbers
X = pd.get_dummies(X, drop_first=True)

# 5. Model Training
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y_train_dummy := df[target_col], test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train_dummy)
    st.success("✅ Model ready for predictions!")
except Exception as e:
    st.error(f"Training Error: {e}")
    st.stop()

# 6. Prediction Input
st.write("---")
st.subheader("Enter Details:")
user_input = {}
# Sirf pehle 8 features dikhayein taake asani ho
for col in X.columns[:8]:
    user_input[col] = st.number_input(f"{col}", value=0.0)

if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    # Ensure all columns exist
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    # Reorder columns to match X_train
    input_df = input_df[X.columns]
    
    prediction = model.predict(input_df)
    res = "Approved ✅" if prediction[0] == 1 else "Rejected ❌"
    st.header(f"Result: {res}")
