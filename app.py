import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# CSV load karne ke baad column names ko strip karo
df = pd.read_csv("data/loan_approval_dataset.csv")
df.columns = df.columns.str.strip()  # extra spaces remove karne ke liye

# Debug: columns check karo
st.write("Columns in dataset:", df.columns)

def get_trained_models(df):
    # Ensure 'Loan_Status' column exists
    if "Loan_Status" not in df.columns:
        st.error("Error: 'Loan_Status' column not found in dataset!")
        return None

    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train, y_train)

    # Random Forest
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    # Support Vector Classifier
    svc_model = SVC(probability=True)
    svc_model.fit(X_train, y_train)

    return {
        "logistic": logistic_model,
        "random_forest": rf_model,
        "svc": svc_model
    }

# Call the function safely
trained_models = get_trained_models(df)
if trained_models:
    st.success("Models trained successfully!")

