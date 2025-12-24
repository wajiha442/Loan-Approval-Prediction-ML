import streamlit as st
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# --- Page config ---
st.set_page_config(page_title="Loan Approval Predictor", page_icon="💰", layout="wide")

# --- Paths ---
data_path = "data/loan_approval_dataset.csv"  # Tumhara dataset path
model_folder = "models"
os.makedirs(model_folder, exist_ok=True)

# --- Load dataset safely ---
if not os.path.exists(data_path):
    st.warning(f"Dataset not found at {data_path}. Please upload the CSV in the data/ folder.")
    df = None
else:
    df = pd.read_csv(data_path)
    st.success("Dataset loaded successfully!")
    st.dataframe(df.head())

# --- Function to train/load models ---
def get_trained_models(df):
    models_info = {
        "Logistic Regression": {"model": LogisticRegression(max_iter=1000), "file": os.path.join(model_folder, "logistic_model.pkl")},
        "Random Forest": {"model": RandomForestClassifier(), "file": os.path.join(model_folder, "random_forest_model.pkl")},
        "SVC": {"model": SVC(probability=True), "file": os.path.join(model_folder, "svc_model.pkl")}
    }

    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]
    
    # Encode categorical columns if any
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype('category').cat.codes

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    trained_models = {}
    for name, info in models_info.items():
        model_file = info["file"]
        if os.path.exists(model_file):
            with open(model_file, "rb") as f:
                trained_models[name] = pickle.load(f)
            st.info(f"{name} model loaded from file.")
        else:
            info["model"].fit(X_train, y_train)
            with open(model_file, "wb") as f:
                pickle.dump(info["model"], f)
            trained_models[name] = info["model"]
            st.success(f"{name} trained and saved to {model_file}.")
    return trained_models

# --- Only train/load models if dataset exists ---
if df is not None:
    trained_models = get_trained_models(df)

    # --- Prediction UI ---
    st.header("Make a Prediction")
    input_data = {}
    for col in df.drop("Loan_Status", axis=1).columns:
        value = st.text_input(f"Enter value for {col}", "")
        # Handle numeric or categorical inputs
        if df[col].dtype in ['int64','float64']:
            input_data[col] = [float(value) if value else 0]
        else:
            # Encode categorical input same as training
            if value:
                input_data[col] = [pd.Series(df[col]).astype('category').cat.categories.get_loc(value) if value in pd.Series(df[col]).astype('category').cat.categories else 0]
            else:
                input_data[col] = [0]

    input_df = pd.DataFrame(input_data)

    # Choose model for prediction
    model_choice = st.selectbox("Select Model", list(trained_models.keys()))

    if st.button("Predict"):
        prediction = trained_models[model_choice].predict(input_df)[0]
        st.subheader("Prediction Result")
        result = "Approved ✅" if prediction==1 else "Rejected ❌"
        st.write(result)
else:
    st.info("Upload the dataset first to enable predictions.")

