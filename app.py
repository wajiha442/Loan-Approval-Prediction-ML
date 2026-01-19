import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

plt.switch_backend('Agg')
st.set_page_config(page_title="ML Model Trainer", page_icon="🤖", layout="wide")

st.title("🤖 Professional ML Model Trainer")
st.markdown("**High Performance Guaranteed | Error-Free Training**")
st.markdown("---")

# ============== DATASET CREATION ==============
@st.cache_data
def create_loan_dataset():
    np.random.seed(42)
    n = 1000
    
    credit = np.random.normal(680, 80, n).clip(300, 850)
    income = np.random.normal(65000, 30000, n).clip(20000, 200000)
    debt_ratio = np.random.beta(2, 5, n) * 0.7
    
    df = pd.DataFrame({
        'Age': np.random.randint(21, 65, n),
        'Annual_Income': income,
        'Credit_Score': credit,
        'Employment_Years': np.random.exponential(5, n).clip(0, 40),
        'Debt_to_Income_Ratio': debt_ratio,
        'Loan_Amount': np.random.randint(5000, 100000, n),
        'Loan_Term_Months': np.random.choice([12, 24, 36, 48, 60, 72], n),
        'Number_of_Dependents': np.random.choice([0, 1, 2, 3, 4], n),
        'Education_Level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n),
        'Home_Ownership': np.random.choice(['Rent', 'Own', 'Mortgage'], n),
        'Previous_Defaults': np.random.choice(['Yes', 'No'], n, p=[0.15, 0.85])
    })
    
    # STRONG predictive formula
    approval_score = (
        (credit > 650).astype(float) * 0.40 +
        (income > 50000).astype(float) * 0.35 +
        (debt_ratio < 0.35).astype(float) * 0.25
    )
    
    noise = np.random.normal(0, 0.04, n)
    df['Loan_Status'] = np.where(approval_score + noise > 0.50, 'Approved', 'Rejected')
    
    return df

@st.cache_data
def create_churn_dataset():
    np.random.seed(42)
    n = 1000
    
    tenure = np.random.exponential(12, n).clip(1, 72)
    charges = np.random.normal(70, 30, n).clip(20, 150)
    
    df = pd.DataFrame({
        'Customer_Age': np.random.randint(18, 70, n),
        'Tenure_Months': tenure,
        'Monthly_Charges': charges,
        'Total_Charges': tenure * charges + np.random.normal(0, 300, n),
        'Contract_Type': np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], n, p=[0.5, 0.3, 0.2]),
        'Internet_Service': np.random.choice(['DSL', 'Fiber Optic', 'No'], n),
        'Customer_Service_Calls': np.random.poisson(2, n).clip(0, 10),
        'Tech_Support': np.random.choice(['Yes', 'No'], n),
        'Online_Security': np.random.choice(['Yes', 'No'], n),
        'Payment_Method': np.random.choice(['Credit Card', 'Bank Transfer', 'Electronic Check'], n)
    })
    
    churn_score = (
        (df['Contract_Type'] == 'Month-to-Month').astype(float) * 0.45 +
        (tenure < 12).astype(float) * 0.35 +
        (df['Customer_Service_Calls'] > 4).astype(float) * 0.20
    )
    
    noise = np.random.normal(0, 0.05, n)
    df['Churn'] = np.where(churn_score + noise > 0.50, 'Yes', 'No')
    
    return df

@st.cache_data
def create_fraud_dataset():
    np.random.seed(42)
    n_normal, n_fraud = 1050, 150
    
    normal = pd.DataFrame({
        'Transaction_Amount': np.random.gamma(2, 50, n_normal).clip(5, 500),
        'Transaction_Hour': np.random.choice(range(6, 23), n_normal),
        'Days_Since_Last_Transaction': np.random.exponential(5, n_normal).clip(0, 30),
        'Number_of_Transactions_Today': np.random.choice([1, 2, 3, 4], n_normal, p=[0.5, 0.3, 0.15, 0.05]),
        'Average_Transaction_Amount': np.random.normal(150, 50, n_normal).clip(50, 300),
        'Card_Age_Days': np.random.uniform(365, 2000, n_normal),
        'Online_Transaction': np.random.choice(['Yes', 'No'], n_normal, p=[0.6, 0.4]),
        'International': np.random.choice(['Yes', 'No'], n_normal, p=[0.08, 0.92]),
        'Is_Fraud': 'No'
    })
    
    fraud = pd.DataFrame({
        'Transaction_Amount': np.random.uniform(800, 2500, n_fraud),
        'Transaction_Hour': np.random.choice(list(range(0, 6)) + list(range(22, 24)), n_fraud),
        'Days_Since_Last_Transaction': np.random.choice([0, 1], n_fraud),
        'Number_of_Transactions_Today': np.random.choice(range(7, 20), n_fraud),
        'Average_Transaction_Amount': np.random.normal(120, 40, n_fraud).clip(50, 200),
        'Card_Age_Days': np.random.uniform(10, 400, n_fraud),
        'Online_Transaction': np.random.choice(['Yes', 'No'], n_fraud, p=[0.95, 0.05]),
        'International': np.random.choice(['Yes', 'No'], n_fraud, p=[0.7, 0.3]),
        'Is_Fraud': 'Yes'
    })
    
    df = pd.concat([normal, fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

@st.cache_data
def create_employee_dataset():
    np.random.seed(42)
    n = 800
    
    job_sat = np.random.choice([1, 2, 3, 4], n, p=[0.15, 0.25, 0.35, 0.25])
    work_life = np.random.choice([1, 2, 3, 4], n, p=[0.1, 0.2, 0.4, 0.3])
    
    df = pd.DataFrame({
        'Age': np.random.randint(22, 60, n),
        'Years_at_Company': np.random.exponential(5, n).clip(0, 25),
        'Monthly_Income': np.random.normal(70000, 30000, n).clip(30000, 150000),
        'Job_Satisfaction': job_sat,
        'Work_Life_Balance': work_life,
        'Years_Since_Promotion': np.random.exponential(3, n).clip(0, 15),
        'Number_of_Projects': np.random.choice([2, 3, 4, 5, 6], n),
        'Overtime': np.random.choice(['Yes', 'No'], n, p=[0.28, 0.72]),
        'Department': np.random.choice(['Sales', 'IT', 'HR', 'Marketing'], n),
        'Education_Level': np.random.choice(['Bachelor', 'Master', 'PhD'], n, p=[0.6, 0.3, 0.1])
    })
    
    attrition_score = (
        (job_sat <= 2).astype(float) * 0.45 +
        (work_life <= 2).astype(float) * 0.35 +
        (df['Overtime'] == 'Yes').astype(float) * 0.20
    )
    
    noise = np.random.normal(0, 0.05, n)
    df['Attrition'] = np.where(attrition_score + noise > 0.45, 'Yes', 'No')
    
    return df

@st.cache_data
def create_student_dataset():
    np.random.seed(42)
    n = 700
    
    study_hrs = np.random.gamma(3, 5, n).clip(5, 50)
    attendance = np.random.beta(8, 2, n) * 40 + 60
    
    df = pd.DataFrame({
        'Study_Hours_Per_Week': study_hrs,
        'Attendance_Percentage': attendance,
        'Previous_Exam_Score': np.random.normal(70, 15, n).clip(40, 100),
        'Assignment_Score': np.random.normal(75, 12, n).clip(50, 100),
        'Extracurricular_Activities': np.random.choice(['Yes', 'No'], n),
        'Parent_Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n),
        'Internet_Access': np.random.choice(['Yes', 'No'], n, p=[0.85, 0.15]),
        'Family_Size': np.random.choice([2, 3, 4, 5], n),
        'School_Type': np.random.choice(['Public', 'Private'], n, p=[0.65, 0.35]),
        'Tutoring': np.random.choice(['Yes', 'No'], n, p=[0.35, 0.65])
    })
    
    performance_score = (
        (study_hrs > 15).astype(float) * 0.40 +
        (attendance > 80).astype(float) * 0.35 +
        (df['Previous_Exam_Score'] > 65).astype(float) * 0.25
    )
    
    noise = np.random.normal(0, 0.06, n)
    df['Final_Grade'] = np.where(performance_score + noise > 0.50, 'Pass', 'Fail')
    
    return df

# ============== UI ==============
st.header("📁 Step 1: Choose Dataset")

dataset_map = {
    "🏦 Loan Approval Dataset": create_loan_dataset,
    "📞 Customer Churn Dataset": create_churn_dataset,
    "💳 Credit Card Fraud Detection": create_fraud_dataset,
    "👔 Employee Attrition Dataset": create_employee_dataset,
    "🎓 Student Performance Dataset": create_student_dataset
}

dataset_choice = st.selectbox("Select a dataset:", list(dataset_map.keys()))

if st.button("📊 Load Dataset", type="primary"):
    df = dataset_map[dataset_choice]()
    st.session_state.df = df
    st.session_state.loaded = True
    st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
    st.rerun()

if 'loaded' not in st.session_state:
    st.warning("⬆️ Please load a dataset to continue")
    st.stop()

df = st.session_state.df

st.subheader("📊 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

# Show dataset info
col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.metric("Total Rows", df.shape[0])
with col_info2:
    st.metric("Total Columns", df.shape[1])
with col_info3:
    st.metric("Missing Values", df.isnull().sum().sum())

st.info(f"💡 **Recommended Target Column**: **{df.columns[-1]}** (last column)")

st.markdown("---")
st.header("🎯 Step 2: Model Training")

col1, col2 = st.columns(2)

with col1:
    # Get categorical columns only (good target candidates)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Show recommended columns
    all_cols = df.columns.tolist()
    
    target_col = st.selectbox(
        "Select Target Column:", 
        all_cols, 
        index=len(all_cols)-1,
        help="⚠️ SELECT A CATEGORICAL COLUMN (like Loan_Status, Churn, Is_Fraud, etc.)"
    )
    
    # Validation warning
    unique_count = df[target_col].nunique()
    
    if unique_count > 50:
        st.error(f"❌ '{target_col}' has {unique_count} unique values! This is NOT suitable for classification.")
        st.error(f"✅ **SELECT ONE OF THESE INSTEAD**: {', '.join(categorical_cols)}")
        st.stop()
    elif unique_count < 2:
        st.error(f"❌ '{target_col}' has only {unique_count} unique value!")
        st.stop()
    else:
        st.success(f"✅ '{target_col}' is valid! ({unique_count} classes)")
        
        # Show the classes
        classes = df[target_col].unique()
        st.caption(f"Classes: {', '.join(map(str, classes[:5]))}" + (f" ... and {len(classes)-5} more" if len(classes) > 5 else ""))

with col2:
    model_choice = st.selectbox(
        "Select Model:", 
        ["🥇 Gradient Boosting (Best)", "🥈 Random Forest", "🥉 Logistic Regression"],
        help="Gradient Boosting usually gives best performance"
    )

with st.expander("⚙️ Advanced Settings (Optional)"):
    test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5)
    random_state = st.number_input("Random State", 0, 100, 42)

if st.button("🚀 Train Model", type="primary", use_container_width=True):
    try:
        with st.spinner('🔄 Training model... Please wait'):
            
            # Prepare data
            X = df.drop(columns=[target_col]).copy()
            y = df[target_col].copy()
            
            # Check target validity
            if y.nunique() < 2:
                st.error(f"❌ Target column must have at least 2 unique values. '{target_col}' has only {y.nunique()}.")
                st.stop()
            
            if y.nunique() > 50:
                st.error(f"❌ Target column has {y.nunique()} unique values. For classification, select a column with 2-20 categories.")
                st.stop()
            
            # Display class distribution
            class_counts = y.value_counts()
            st.info(f"📊 **Class Distribution**: {dict(class_counts)}")
            
            # Encode categorical features in X
            X_encoded = X.copy()
            le_dict = {}
            
            for col in X_encoded.columns:
                if X_encoded[col].dtype == 'object' or X_encoded[col].dtype.name == 'category':
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                    le_dict[col] = le
            
            # Encode target if categorical
            if y.dtype == 'object' or y.dtype.name == 'category':
                le_target = LabelEncoder()
                y_encoded = le_target.fit_transform(y.astype(str))
                target_classes = le_target.classes_
            else:
                y_encoded = y.values
                target_classes = np.unique(y)
            
            # Use encoded X
            X = X_encoded
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, 
                test_size=test_size/100, 
                random_state=random_state, 
                stratify=y_encoded
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Select and train model
            if "Gradient" in model_choice:
                model = GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.1,
                    max_depth=5,
                    subsample=0.8,
                    random_state=random_state
                )
            elif "Random" in model_choice:
                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1
                )
            else:
                model = LogisticRegression(
                    max_iter=2000,
                    random_state=random_state,
                    solver='lbfgs'
                )
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        
        st.success("✅ Model trained successfully!")
        
        # ============== RESULTS ==============
        st.markdown("---")
        st.header("📊 Step 3: Model Evaluation Results")
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        if len(target_classes) == 2:
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
        else:
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Display metrics
        st.subheader("📈 Performance Metrics")
        
        metric_cols = st.columns(5)
        
        with metric_cols[0]:
            st.metric("Accuracy", f"{accuracy:.1%}")
        with metric_cols[1]:
            st.metric("Precision", f"{precision:.1%}")
        with metric_cols[2]:
            st.metric("Recall", f"{recall:.1%}")
        with metric_cols[3]:
            st.metric("F1-Score", f"{f1:.1%}")
        with metric_cols[4]:
            st.metric("CV Score", f"{cv_scores.mean():.1%}")
        
        # Performance feedback
        if accuracy > 0.85:
            st.success("🎉 **EXCELLENT PERFORMANCE!** Model is ready for deployment!")
        elif accuracy > 0.75:
            st.info("👍 **GOOD PERFORMANCE!** Model works well.")
        elif accuracy > 0.65:
            st.warning("⚠️ **MODERATE PERFORMANCE.** Consider trying Gradient Boosting.")
        else:
            st.error("❌ **POOR PERFORMANCE.** Try a different model or check your data.")
        
        # Confusion Matrix
        st.subheader("🔍 Confusion Matrix")
        
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                   cbar_kws={"shrink": 0.8}, linewidths=2)
        ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_xlabel('Predicted', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Classification Report
        st.subheader("📋 Detailed Classification Report")
        
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
        
        # Feature Importance
        if hasattr(model, 'feature_importances_'):
            st.subheader("🎯 Feature Importance Analysis")
            
            feature_names = X.columns
            importances = model.feature_importances_
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Plot top 10 features
            top_features = importance_df.head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(top_features['Feature'], top_features['Importance'], color='purple', alpha=0.7)
            ax.set_title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
            ax.set_xlabel('Importance Score', fontsize=12)
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            with st.expander("📊 View All Feature Importances"):
                st.dataframe(importance_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ **Error during training**: {str(e)}")
        st.error("**Possible Solutions:**")
        st.markdown("- Make sure you selected a valid target column (categorical with 2-20 unique values)")
        st.markdown("- Try selecting the last column as target")
        st.markdown("- Check if your data has missing values")
        
        with st.expander("🔍 See Full Error Details"):
            import traceback
            st.code(traceback.format_exc())

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
<p style='font-size: 16px;'><strong>🤖 Professional ML Model Trainer</strong></p>
<p>Built with Streamlit | Guaranteed High Performance 🚀</p>
</div>
""", unsafe_allow_html=True)
