import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import os
import urllib.request

# Page configuration
st.set_page_config(page_title="Loan Approval Prediction", page_icon="💰", layout="wide")
st.title("Loan Approval Prediction App")

# BASE_DIR is the folder where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset path
dataset_folder = os.path.join(BASE_DIR, "data")
dataset_path = os.path.join(dataset_folder, "loan_approval_dataset.csv")

# Create data folder if it doesn't exist
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

# URL of your dataset (replace this with your dataset's raw GitHub URL or Kaggle URL)
dataset_url = "https://raw.githubusercontent.com/yourusername/yourrepo/main/data/loan_approval_dataset.csv"

# Download dataset if it doesn't exist
if not os.path.exists(dataset_path):
    st.warning("Dataset not found locally. Downloading dataset...")
    try:
        urllib.request.urlretrieve(dataset_url, dataset_path)
        st.success("Dataset downloaded successfully!")
    except Exception as e:
        st.error(f"Failed to download dataset: {e}")
        st.stop()

# Load dataset
df = pd.read_csv(dataset_path)

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Debug: show columns
st.write("Columns in dataset:", df.columns)

# Check if 'Loan_Status' exists
if "Loan_Status" not in df.columns:
    st.error("Error: 'Loan_Status' column not found in dataset!")
    st.stop()

# Separate features and target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
with st.spinner("Training models..."):
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train, y_train)

    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    svc_model = SVC(probability=True)
    svc_model.fit(X_train, y_train)

st.success("✅ Models trained successfully!")

# Optional: show first 5 rows of dataset
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Example: prediction form
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

