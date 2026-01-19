import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

plt.switch_backend('Agg')
st.set_page_config(page_title="ML Model Trainer", page_icon="🤖", layout="wide")

st.title("🤖 Machine Learning Model Trainer - OPTIMIZED")
st.markdown("---")

@st.cache_data
def create_loan_dataset():
    np.random.seed(42)
    n = 1000
    credit = np.clip(np.random.normal(680, 80, n), 300, 850)
    income = np.clip(np.random.normal(65000, 30000, n), 20000, 200000)
    
    df = pd.DataFrame({
        'Age': np.random.randint(21, 65, n),
        'Annual_Income': income,
        'Credit_Score': credit,
        'Employment_Years': np.clip(np.random.exponential(5, n), 0, 40),
        'Debt_to_Income_Ratio': np.random.beta(2, 5, n) * 0.7,
        'Loan_Amount': np.random.randint(5000, 100000, n),
        'Previous_Defaults': np.random.choice(['Yes', 'No'], n, p=[0.15, 0.85])
    })
    
    # VERY STRONG PATTERNS
    score = ((credit > 650)*1.0) * 0.45 + ((income > 50000)*1.0) * 0.35 + ((df['Debt_to_Income_Ratio'] < 0.35)*1.0) * 0.20
    df['Loan_Status'] = np.where(score + np.random.normal(0, 0.03, n) > 0.50, 'Approved', 'Rejected')
    return df

@st.cache_data  
def create_churn_dataset():
    np.random.seed(42)
    n = 1000
    tenure = np.clip(np.random.exponential(12, n), 1, 72)
    
    df = pd.DataFrame({
        'Tenure_Months': tenure,
        'Monthly_Charges': np.clip(np.random.normal(70, 30, n), 20, 150),
        'Contract_Type': np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], n, p=[0.5, 0.3, 0.2]),
        'Customer_Service_Calls': np.clip(np.random.poisson(2, n), 0, 10),
        'Tech_Support': np.random.choice(['Yes', 'No'], n, p=[0.4, 0.6])
    })
    
    score = ((df['Contract_Type']=='Month-to-Month')*1.0)*0.45 + ((tenure<12)*1.0)*0.35 + ((df['Customer_Service_Calls']>4)*1.0)*0.20
    df['Churn'] = np.where(score + np.random.normal(0, 0.04, n) > 0.50, 'Yes', 'No')
    return df

@st.cache_data
def create_fraud_dataset():
    np.random.seed(42)
    n_normal, n_fraud = 1050, 150
    
    normal = pd.DataFrame({
        'Transaction_Amount': np.clip(np.random.gamma(2, 50, n_normal), 5, 500),
        'Transaction_Hour': np.random.choice(range(6, 23), n_normal),
        'Transactions_Today': np.random.choice([1,2,3,4], n_normal, p=[0.5,0.3,0.15,0.05]),
        'Card_Age_Days': np.random.uniform(365, 2000, n_normal),
        'Is_Fraud': 'No'
    })
    
    fraud = pd.DataFrame({
        'Transaction_Amount': np.random.uniform(800, 2500, n_fraud),
        'Transaction_Hour': np.random.choice(list(range(0,6)) + list(range(22,24)), n_fraud),
        'Transactions_Today': np.random.choice(range(7, 20), n_fraud),
        'Card_Age_Days': np.random.uniform(10, 400, n_fraud),
        'Is_Fraud': 'Yes'
    })
    
    df = pd.concat([normal, fraud], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    return df

st.header("📁 Step 1: Choose Dataset")

dataset_choice = st.selectbox("Choose a sample dataset:", 
                              ["Loan Approval Dataset", 
                               "Customer Churn Dataset", 
                               "Credit Card Fraud Detection"])

if st.button("📊 Load Dataset", type="primary"):
    if dataset_choice == "Loan Approval Dataset":
        df = create_loan_dataset()
    elif dataset_choice == "Customer Churn Dataset":
        df = create_churn_dataset()
    else:
        df = create_fraud_dataset()
    
    st.session_state.df = df
    st.session_state.loaded = True
    st.success(f"✅ Loaded! Shape: {df.shape}")
    st.rerun()

if 'loaded' not in st.session_state:
    st.warning("⬆️ Please load a dataset to continue")
    st.stop()

df = st.session_state.df

st.subheader("📊 Dataset Preview")
st.dataframe(df.head(10))
st.info(f"💡 **Target Column (Last Column)**: {df.columns[-1]}")

st.markdown("---")
st.header("🎯 Step 2: Model Training")

col1, col2 = st.columns(2)

with col1:
    target = st.selectbox("Target Column:", df.columns.tolist(), index=len(df.columns)-1)

with col2:
    model_choice = st.selectbox("Model:", 
                               ["Gradient Boosting ⭐", 
                                "Random Forest", 
                                "Logistic Regression"])

if st.button("🚀 Train Model", type="primary"):
    try:
        with st.spinner('Training...'):
            X = df.drop(columns=[target]).copy()
            y = df[target].copy()
            
            # Encode categoricals
            for col in X.select_dtypes(include=['object']).columns:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            
            if y.dtype == 'object':
                y = LabelEncoder().fit_transform(y)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Select model
            if "Gradient" in model_choice:
                model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
            elif "Random" in model_choice:
                model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
            else:
                model = LogisticRegression(max_iter=2000, random_state=42)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        st.success("✅ Model trained!")
        
        st.markdown("---")
        st.header("📊 Results")
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='binary' if len(np.unique(y_test))==2 else 'weighted')
        rec = recall_score(y_test, y_pred, average='binary' if len(np.unique(y_test))==2 else 'weighted')
        f1 = f1_score(y_test, y_pred, average='binary' if len(np.unique(y_test))==2 else 'weighted')
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{acc:.1%}")
        c2.metric("Precision", f"{prec:.1%}")
        c3.metric("Recall", f"{rec:.1%}")
        c4.metric("F1-Score", f"{f1:.1%}")
        c5.metric("CV Score", f"{cv_scores.mean():.1%}")
        
        if acc > 0.85:
            st.success("🎉 EXCELLENT Performance!")
        elif acc > 0.75:
            st.info("👍 GOOD Performance!")
        else:
            st.warning("⚠️ Moderate Performance")
        
        st.subheader("🔍 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        st.pyplot(fig)
        
        if hasattr(model, 'feature_importances_'):
            st.subheader("🎯 Top Features")
            importance_df = pd.DataFrame({
                'Feature': df.drop(columns=[target]).columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(8)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(importance_df['Feature'], importance_df['Importance'], color='purple', alpha=0.7)
            ax.set_title('Feature Importance')
            ax.invert_yaxis()
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())

st.markdown("---")
st.markdown("<center>🤖 ML Model Trainer | Streamlit</center>", unsafe_allow_html=True)
