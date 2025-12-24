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
    st.error(f"❌ Dataset file nahi mili! File ka naam check karein.")
    st.stop()

# Load Data
df = pd.read_csv(dataset_path)

# --- DEBUGGING SECTION ---
# Column names ko clean karna (Spaces khatam karna)
df.columns = df.columns.str.strip()

# Screen par columns dikhana taake aap check kar sakein
st.write("### Aapke Dataset ke Columns:", list(df.columns))

# Automatic Target Column Finder
# Hum dhoond rahe hain aisa column jis mein 'status' ya 'loan_status' likha ho
target_col = None
for col in df.columns:
    if 'status' in col.lower():
        target_col = col
        break

if target_col:
    st.success(f"✅ Target column mil gaya: '{target_col}'")
else:
    st.error("❌ Error: 'loan_status' jaisa koi column nahi mila. Upar list mein se sahi naam dekh kar code mein likhein.")
    st.stop()

# --- PREPROCESSING ---
# Status ko numbers mein badalna (Approved = 1, Rejected = 0)
df[target_col] = df[target_col].astype(str).str.strip().str.lower()
df[target_col] = df[target_col].apply(lambda x: 1 if 'approved' in x else 0)

# Baki columns (Features) - Sirf numerical columns le rahe hain training ke liye (Asaan rakhne ke liye)
X = df.drop(target_col, axis=1)
X = pd.get_dummies(X, drop_first=True) # Text columns ko numbers mein badal dega
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

st.write("---")
st.subheader("Make a Prediction")
user_input = {}
for col in X.columns:
    user_input[col] = st.number_input(f"Enter {col}", value=0)

if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)
    res = "Approved ✅" if prediction[0] == 1 else "Rejected ❌"
    st.header(f"Result: {res}")
