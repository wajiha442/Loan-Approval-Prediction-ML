import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import os

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(page_title="Loan Approval Prediction", page_icon="💰", layout="wide")
st.title("Loan Approval Prediction App")

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder where app.py is
DATA_FOLDER = os.path.join(BASE_DIR, "data")
dataset_path = os.path.join(DATA_FOLDER, "loan_approval_dataset.csv")

# -----------------------------
# Check dataset existence
# -----------------------------
if not os.path.exists(dataset_path):
    st.error("Dataset not found! Please upload 'loan_approval_dataset.csv' inside the 'data/' folder.")
    st.stop()

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(dataset_path)

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Debug: show columns
st.write("Columns in dataset:", df.columns)

# Ensure 'Loan_Status' exists
if "Loan_Status" not in df.columns:
    st.error("Error: 'Loan_Status' column not found in dataset!")
    st.stop()

# -----------------------------
# Prepare features and target
# -----------------------------
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Train models
# -----------------------------
with st.spinner("Training models..."):
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train, y_train)

    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    svc_model = SVC(probability=True)
    svc_model.fit(X_train, y_train)

st.success("✅ Models trained successfully!")

# -----------------------------
# Dataset preview
# -----------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Prediction form
# -----------------------------
st.subheader("Make a Prediction")
user_input = {}
for col in X.columns:
    user_input[col] = st.text_input(f"Enter {col}:")

if st.button("Predict"):
    try:
        input_df = pd.DataFrame([user_input])
        prediction = logistic_model.predict(input_df)
        st.write(f"Predicted Loan Status (Logistic Regression): {prediction[0]}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")

