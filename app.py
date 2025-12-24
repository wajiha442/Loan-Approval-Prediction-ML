import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import os

# -----------------------------
# 1. Page Configuration
# -----------------------------
st.set_page_config(page_title="Loan Approval Prediction", page_icon="💰", layout="wide")
st.title("💰 Loan Approval Prediction App")
st.markdown("This app predicts whether a loan will be approved based on the provided dataset.")

# -----------------------------
# 2. Paths (Directly using GitHub root folder)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Hum 'data' folder hata rahe hain kyunke aapne file direct upload ki hai
dataset_path = os.path.join(BASE_DIR, "loan_approval_dataset.csv")

# -----------------------------
# 3. Load and Clean Dataset
# -----------------------------
if not os.path.exists(dataset_path):
    st.error(f"❌ Dataset not found! Make sure 'loan_approval_dataset.csv' is in your GitHub repository.")
    st.info(f"Looking at path: {dataset_path}")
    st.stop()

@st.cache_data # Cache taake baar baar load na ho
def load_data():
    data = pd.read_csv(dataset_path)
    # Column names se spaces khatam karein aur lowercase karein
    data.columns = data.columns.str.strip().str.lower()
    return data

df = load_data()

# -----------------------------
# 4. Handle Target Column
# -----------------------------
# Kaggle ke dataset mein aksar 'loan_status' hota hai
target_col = 'loan_status'

if target_col not in df.columns:
    st.error(f"❌ Error: Column '{target_col}' not found!")
    st.write("Available columns are:", list(df.columns))
    st.stop()

# -----------------------------
# 5. Data Preprocessing
# -----------------------------
# Text columns (Categorical) ko numbers mein convert karna zaroori hai
# 'loan_status' ko 1 aur 0 mein convert karein
df[target_col] = df[target_col].apply(lambda x: 1 if str(x).strip().lower() == 'approved' else 0)

# Features (X) aur Target (y)
X = df.drop(target_col, axis=1)
y = df[target_col]

# Agar koi aur text columns hain (jaise Education), unhein numbers mein badlein
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 6. Train Models
# -----------------------------
@st.cache_resource # Model training ko cache karein
def train_models(X_train, y_train):
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    
    return lr, rf

with st.spinner("Training models, please wait..."):
    logistic_model, rf_model = train_models(X_train, y_train)

st.success("✅ Models trained successfully!")

# -----------------------------
# 7. Sidebar for User Input
# -----------------------------
st.sidebar.header("User Input Features")
def get_user_input():
    user_data = {}
    for col in X.columns:
        # Har column ke liye input box
        val = st.sidebar.number_input(f"Enter {col}", value=0.0)
        user_data[col] = val
    return pd.DataFrame([user_data])

input_df = get_user_input()

# -----------------------------
# 8. Prediction Logic
# -----------------------------
st.subheader("Prediction Result")
if st.button("Predict Loan Status"):
    prediction = logistic_model.predict(input_df)
    result = "Approved ✅" if prediction[0] == 1 else "Rejected ❌"
    
    if prediction[0] == 1:
        st.balloons()
        st.success(f"The predicted status is: **{result}**")
    else:
        st.error(f"The predicted status is: **{result}**")

# Show Data Preview
with st.expander("View Dataset Preview"):
    st.dataframe(df.head())
