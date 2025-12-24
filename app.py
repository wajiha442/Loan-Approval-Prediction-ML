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
    st.error(f"❌ Dataset file ('loan_approval_dataset.csv') nahi mili!")
    st.stop()

# 2. Load Data
try:
    df = pd.read_csv(dataset_path)
    # Sab se pehle column names se extra spaces khatam karein
    df.columns = df.columns.str.strip()
    
    # Preview for debugging
    with st.expander("Dataset Preview"):
        st.write(df.head())
        st.write("Total Columns:", list(df.columns))

except Exception as e:
    st.error(f"File load karne mein masla: {e}")
    st.stop()

# 3. Target Column dhoondein (Loan Status)
# Hum check kar rahe hain ke 'loan_status' (choti ya bari abc mein) kahan hai
target_col = None
for c in df.columns:
    if 'status' in c.lower():
        target_col = c
        break

if not target_col:
    st.error("❌ 'Loan Status' wala column nahi mila. Apni CSV file check karein.")
    st.stop()

# 4. Data Preprocessing
# Target ko binary (0/1) mein convert karein
df[target_col] = df[target_col].astype(str).str.strip().str.lower()
df[target_col] = df[target_col].apply(lambda x: 1 if 'approved' in x else 0)

# Features (X) aur Target (y)
X = df.drop(columns=[target_col])
y = df[target_col]

# Error Fix: Ensure X is not empty
if X.empty:
    st.error("X (Features) khali hain! Dataset mein sirf ek hi column hai shayad.")
    st.stop()

# Text columns ko numbers mein badalna
X = pd.get_dummies(X)

# 5. Model Training
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    st.success("✅ Model Train ho gaya!")
except Exception as e:
    st.error(f"Training mein error: {e}")
    st.stop()

# 6. User Prediction Interface
st.write("---")
st.subheader("📝 Enter Details for Prediction")

user_input = {}
# Columns ko loops mein dikhayenge input ke liye
# Pehle 5 columns dikhate hain taake screen bhar na jaye
cols_to_show = X.columns[:10] 

for col in cols_to_show:
    user_input[col] = st.number_input(f"Enter {col}", value=0.0)

if st.button("Predict Loan Status"):
    # Baqi missing columns ko 0 se bhar dena agar koi reh gaya ho
    full_input = pd.DataFrame(columns=X.columns)
    full_input.loc[0] = 0 # Default 0
    for k, v in user_input.items():
        full_input[k] = v
        
    prediction = model.predict(full_input)
    res = "Approved ✅" if prediction[0] == 1 else "Rejected ❌"
    
    if prediction[0] == 1:
        st.balloons()
        st.success(f"Loan Status: **{res}**")
    else:
        st.error(f"Loan Status: **{res}**")
