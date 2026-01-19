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

st.title("🤖 Machine Learning Model Trainer")
st.markdown("""
Welcome to the **ML Model Trainer**! This application allows you to:
- Use a sample dataset or upload your own CSV
- Perform exploratory data analysis (EDA)
- Visualize data patterns
- Train and evaluate machine learning models with optimized performance
""")
st.markdown("---")

@st.cache_data
def create_sample_dataset(dataset_name):
    if dataset_name == "Loan Approval Dataset":
        np.random.seed(42)
        n_samples = 1000
        
        # Create more realistic features with stronger correlations
        credit_scores = np.random.normal(680, 80, n_samples).clip(300, 850)
        annual_income = np.random.normal(65000, 30000, n_samples).clip(20000, 200000)
        employment_years = np.random.exponential(5, n_samples).clip(0, 40)
        
        df = pd.DataFrame({
            'Age': np.random.randint(21, 65, n_samples),
            'Annual_Income': annual_income,
            'Credit_Score': credit_scores,
            'Employment_Years': employment_years,
            'Debt_to_Income_Ratio': np.random.beta(2, 5, n_samples) * 0.7,
            'Loan_Amount': np.random.randint(5000, 100000, n_samples),
            'Loan_Term_Months': np.random.choice([12, 24, 36, 48, 60, 72], n_samples),
            'Number_of_Dependents': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.3, 0.3, 0.2, 0.15, 0.05]),
            'Education_Level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
            'Home_Ownership': np.random.choice(['Rent', 'Own', 'Mortgage'], n_samples, p=[0.3, 0.3, 0.4]),
            'Previous_Defaults': np.random.choice(['Yes', 'No'], n_samples, p=[0.15, 0.85])
        })
        
        # Create strong predictive relationship
        approval_score = (
            (df['Credit_Score'] > 650).astype(int) * 0.30 +
            (df['Annual_Income'] > 50000).astype(int) * 0.25 +
            (df['Debt_to_Income_Ratio'] < 0.35).astype(int) * 0.20 +
            (df['Employment_Years'] > 2).astype(int) * 0.15 +
            (df['Previous_Defaults'] == 'No').astype(int) * 0.10
        )
        
        # Add some noise but keep strong signal
        noise = np.random.normal(0, 0.08, n_samples)
        df['Loan_Status'] = np.where(approval_score + noise > 0.55, 'Approved', 'Rejected')
        
        return df
    
    elif dataset_name == "Customer Churn Dataset":
        np.random.seed(42)
        n_samples = 1000
        
        tenure = np.random.exponential(12, n_samples).clip(1, 72)
        monthly_charges = np.random.normal(70, 30, n_samples).clip(20, 150)
        
        df = pd.DataFrame({
            'Customer_Age': np.random.randint(18, 70, n_samples),
            'Tenure_Months': tenure,
            'Monthly_Charges': monthly_charges,
            'Total_Charges': tenure * monthly_charges + np.random.normal(0, 500, n_samples),
            'Contract_Type': np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], n_samples, p=[0.5, 0.3, 0.2]),
            'Internet_Service': np.random.choice(['DSL', 'Fiber Optic', 'No'], n_samples, p=[0.3, 0.5, 0.2]),
            'Customer_Service_Calls': np.random.poisson(2, n_samples).clip(0, 10),
            'Tech_Support': np.random.choice(['Yes', 'No'], n_samples, p=[0.4, 0.6]),
            'Online_Security': np.random.choice(['Yes', 'No'], n_samples, p=[0.45, 0.55]),
            'Payment_Method': np.random.choice(['Credit Card', 'Bank Transfer', 'Electronic Check', 'Mailed Check'], n_samples)
        })
        
        churn_prob = (
            (df['Contract_Type'] == 'Month-to-Month').astype(int) * 0.35 +
            (df['Tenure_Months'] < 12).astype(int) * 0.25 +
            (df['Customer_Service_Calls'] > 4).astype(int) * 0.20 +
            (df['Tech_Support'] == 'No').astype(int) * 0.12 +
            (df['Monthly_Charges'] > 80).astype(int) * 0.08
        )
        
        noise = np.random.normal(0, 0.1, n_samples)
        df['Churn'] = np.where(churn_prob + noise > 0.5, 'Yes', 'No')
        
        return df
    
    elif dataset_name == "Credit Card Fraud Detection":
        np.random.seed(42)
        n_samples = 1200
        n_fraud = int(n_samples * 0.12)
        n_normal = n_samples - n_fraud
        
        normal_transactions = pd.DataFrame({
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
        
        fraud_transactions = pd.DataFrame({
            'Transaction_Amount': np.random.uniform(500, 2500, n_fraud),
            'Transaction_Hour': np.random.choice(list(range(0, 6)) + list(range(22, 24)), n_fraud),
            'Days_Since_Last_Transaction': np.random.choice([0, 1], n_fraud),
            'Number_of_Transactions_Today': np.random.choice(range(6, 20), n_fraud),
            'Average_Transaction_Amount': np.random.normal(120, 40, n_fraud).clip(50, 200),
            'Card_Age_Days': np.random.uniform(10, 400, n_fraud),
            'Online_Transaction': np.random.choice(['Yes', 'No'], n_fraud, p=[0.95, 0.05]),
            'International': np.random.choice(['Yes', 'No'], n_fraud, p=[0.6, 0.4]),
            'Is_Fraud': 'Yes'
        })
        
        df = pd.concat([normal_transactions, fraud_transactions], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df
    
    elif dataset_name == "Employee Attrition Dataset":
        np.random.seed(42)
        n_samples = 800
        
        job_satisfaction = np.random.choice([1, 2, 3, 4], n_samples, p=[0.15, 0.25, 0.35, 0.25])
        work_life_balance = np.random.choice([1, 2, 3, 4], n_samples, p=[0.1, 0.2, 0.4, 0.3])
        
        df = pd.DataFrame({
            'Age': np.random.randint(22, 60, n_samples),
            'Years_at_Company': np.random.exponential(5, n_samples).clip(0, 25),
            'Monthly_Income': np.random.normal(70000, 30000, n_samples).clip(30000, 150000),
            'Job_Satisfaction': job_satisfaction,
            'Work_Life_Balance': work_life_balance,
            'Years_Since_Promotion': np.random.exponential(3, n_samples).clip(0, 15),
            'Number_of_Projects': np.random.choice([2, 3, 4, 5, 6, 7], n_samples, p=[0.1, 0.2, 0.3, 0.2, 0.15, 0.05]),
            'Overtime': np.random.choice(['Yes', 'No'], n_samples, p=[0.28, 0.72]),
            'Department': np.random.choice(['Sales', 'IT', 'HR', 'Marketing', 'Finance'], n_samples),
            'Education_Level': np.random.choice(['Bachelor', 'Master', 'PhD'], n_samples, p=[0.6, 0.3, 0.1]),
            'Business_Travel': np.random.choice(['Rarely', 'Frequently', 'No Travel'], n_samples, p=[0.5, 0.3, 0.2])
        })
        
        attrition_score = (
            (df['Job_Satisfaction'] <= 2).astype(int) * 0.35 +
            (df['Work_Life_Balance'] <= 2).astype(int) * 0.25 +
            (df['Overtime'] == 'Yes').astype(int) * 0.20 +
            (df['Years_Since_Promotion'] > 6).astype(int) * 0.15 +
            (df['Years_at_Company'] < 2).astype(int) * 0.05
        )
        
        noise = np.random.normal(0, 0.08, n_samples)
        df['Attrition'] = np.where(attrition_score + noise > 0.45, 'Yes', 'No')
        
        return df
    
    elif dataset_name == "Student Performance Dataset":
        np.random.seed(42)
        n_samples = 700
        
        study_hours = np.random.gamma(3, 5, n_samples).clip(5, 50)
        attendance = np.random.beta(8, 2, n_samples) * 40 + 60
        
        df = pd.DataFrame({
            'Study_Hours_Per_Week': study_hours,
            'Attendance_Percentage': attendance,
            'Previous_Exam_Score': np.random.normal(70, 15, n_samples).clip(40, 100),
            'Assignment_Score': np.random.normal(75, 12, n_samples).clip(50, 100),
            'Extracurricular_Activities': np.random.choice(['Yes', 'No'], n_samples, p=[0.45, 0.55]),
            'Parent_Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
            'Internet_Access': np.random.choice(['Yes', 'No'], n_samples, p=[0.85, 0.15]),
            'Family_Size': np.random.choice([2, 3, 4, 5, 6], n_samples, p=[0.15, 0.25, 0.35, 0.2, 0.05]),
            'School_Type': np.random.choice(['Public', 'Private'], n_samples, p=[0.65, 0.35]),
            'Tutoring': np.random.choice(['Yes', 'No'], n_samples, p=[0.35, 0.65])
        })
        
        performance_score = (
            (df['Study_Hours_Per_Week'] > 15).astype(int) * 0.28 +
            (df['Attendance_Percentage'] > 80).astype(int) * 0.22 +
            (df['Previous_Exam_Score'] > 65).astype(int) * 0.25 +
            (df['Assignment_Score'] > 70).astype(int) * 0.18 +
            (df['Tutoring'] == 'Yes').astype(int) * 0.07
        )
        
        noise = np.random.normal(0, 0.1, n_samples)
        df['Final_Grade'] = np.where(performance_score + noise > 0.5, 'Pass', 'Fail')
        
        return df

st.header("📁 Step 1: Choose Dataset")

tab1, tab2 = st.tabs(["🎯 Use Sample Dataset", "📤 Upload Your Own CSV"])

df = None

with tab1:
    st.markdown("### Select a Sample Dataset to Try the App")
    st.info("👉 Perfect for testing! Choose one of our optimized sample datasets below.")
    sample_dataset = st.selectbox("Choose a sample dataset:", 
                                  ["Loan Approval Dataset", 
                                   "Customer Churn Dataset", 
                                   "Credit Card Fraud Detection", 
                                   "Employee Attrition Dataset", 
                                   "Student Performance Dataset"])
    if st.button("📊 Load Sample Dataset", type="primary"):
        df = create_sample_dataset(sample_dataset)
        st.session_state.df = df
        st.session_state.dataset_loaded = True
        st.success(f"✅ {sample_dataset} loaded! Shape: {df.shape}")
        st.rerun()

with tab2:
    st.markdown("### Upload Your Own CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.dataset_loaded = True
            st.success(f"✅ Dataset uploaded successfully! Shape: {df.shape}")
            st.rerun()
        except Exception as e:
            st.error(f"Error loading file: {e}")

if 'dataset_loaded' not in st.session_state or not st.session_state.dataset_loaded:
    st.warning("⬆️ Please select a sample dataset or upload your own CSV file to continue")
    st.stop()

df = st.session_state.df

st.subheader("📊 Dataset Preview")
st.dataframe(df.head(10))

st.markdown("---")
st.header("🔍 Step 2: Exploratory Data Analysis (EDA)")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Rows", df.shape[0])
with col2:
    st.metric("Total Columns", df.shape[1])
with col3:
    st.metric("Missing Values", df.isnull().sum().sum())

st.subheader("📋 Dataset Information")

col_info1, col_info2 = st.columns(2)

with col_info1:
    st.write("**Column Names and Data Types:**")
    info_df = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.values,
        'Non-Null Count': df.count().values,
        'Null Count': df.isnull().sum().values
    })
    st.dataframe(info_df, use_container_width=True)

with col_info2:
    st.write("**Summary Statistics:**")
    st.dataframe(df.describe(), use_container_width=True)

st.subheader("🔍 Missing Values Analysis")
missing_data = df.isnull().sum()
missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

if len(missing_data) > 0:
    fig, ax = plt.subplots(figsize=(10, 4))
    missing_data.plot(kind='bar', color='coral', ax=ax)
    ax.set_title('Missing Values by Column', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count')
    ax.set_xlabel('Columns')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()
else:
    st.success("✅ No missing values found in the dataset!")

st.markdown("---")
st.header("📈 Step 3: Data Visualizations")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

if numeric_cols:
    st.subheader("📊 Histograms (Numerical Features)")
    default_hist_cols = numeric_cols[:min(4, len(numeric_cols))]
    selected_cols_hist = st.multiselect("Select columns for histograms:", numeric_cols, default=default_hist_cols)
    
    if selected_cols_hist:
        n_cols = 2
        n_rows = (len(selected_cols_hist) + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 4))
        
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for idx, col in enumerate(selected_cols_hist):
            if idx < len(axes):
                axes[idx].hist(df[col].dropna(), bins=30, color='skyblue', edgecolor='black', alpha=0.7)
                axes[idx].set_title(f'Distribution of {col}', fontweight='bold', fontsize=12)
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(alpha=0.3)
        
        for idx in range(len(selected_cols_hist), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.clf()

if len(numeric_cols) > 1:
    st.subheader("🔥 Correlation Heatmap")
    corr_matrix = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                square=True, linewidths=1, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title('Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()

if categorical_cols:
    st.subheader("📊 Count Plots (Categorical Features)")
    selected_cat_col = st.selectbox("Select categorical column:", categorical_cols)
    
    if selected_cat_col:
        fig, ax = plt.subplots(figsize=(10, 5))
        value_counts = df[selected_cat_col].value_counts()
        
        if len(value_counts) > 20:
            value_counts = value_counts.head(20)
            st.warning(f"Showing top 20 categories only (out of {len(df[selected_cat_col].unique())})")
        
        value_counts.plot(kind='bar', color='teal', ax=ax, alpha=0.7)
        ax.set_title(f'Count Plot of {selected_cat_col}', fontsize=14, fontweight='bold')
        ax.set_xlabel(selected_cat_col)
        ax.set_ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        ax.grid(alpha=0.3, axis='y')
        
        for i, v in enumerate(value_counts.values):
            ax.text(i, v + 0.5, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.clf()

st.markdown("---")
st.header("🎯 Step 4: Model Selection & Training")

col_model1, col_model2 = st.columns(2)

with col_model1:
    target_column = st.selectbox("Select Target Column:", df.columns.tolist(), 
                                help="Choose the column you want to predict")

with col_model2:
    model_option = st.selectbox("Select Machine Learning Model:", 
                               ["Random Forest (Optimized)", 
                                "Gradient Boosting (Best Performance)",
                                "Logistic Regression", 
                                "Support Vector Machine (SVM)", 
                                "K-Nearest Neighbors (KNN)"])

with st.expander("⚙️ Advanced Settings"):
    test_size = st.slider("Test Set Size (%)", 10, 50, 20, 5) / 100
    random_state = st.number_input("Random State", 0, 100, 42)
    
    if "Random Forest" in model_option:
        n_estimators = st.slider("Number of Trees", 50, 300, 150, 50)
        max_depth = st.slider("Max Depth", 5, 30, 15, 5)
    elif "Gradient Boosting" in model_option:
        n_estimators_gb = st.slider("Number of Estimators", 50, 300, 100, 50)
        learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
    elif model_option == "K-Nearest Neighbors (KNN)":
        n_neighbors = st.slider("Number of Neighbors", 3, 15, 7, 2)
    elif model_option == "Support Vector Machine (SVM)":
        kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"])
        C_param = st.slider("Regularization (C)", 0.1, 10.0, 1.0, 0.1)

if st.button("🚀 Train Model", type="primary"):
    try:
        with st.spinner('🔄 Preparing data and training model...'):
            X = df.drop(columns=[target_column]).copy()
            y = df[target_column].copy()
            
            categorical_features = X.select_dtypes(include=['object']).columns
            
            if len(categorical_features) > 0:
                st.info(f"🔄 Encoding categorical features: {', '.join(categorical_features)}")
                label_encoders = {}
                for col in categorical_features:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    label_encoders[col] = le
            
            if y.dtype == 'object':
                le_target = LabelEncoder()
                y_encoded = le_target.fit_transform(y)
                st.info(f"🎯 Target classes: {', '.join(map(str, le_target.classes_))}")
            else:
                y_encoded = y
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
            )
            
            st.info(f"📊 Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            if "Random Forest" in model_option:
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1
                )
            elif "Gradient Boosting" in model_option:
                model = GradientBoostingClassifier(
                    n_estimators=n_estimators_gb,
                    learning_rate=learning_rate,
                    max_depth=5,
                    min_samples_split=5,
                    random_state=random_state
                )
            elif model_option == "Logistic Regression":
                model = LogisticRegression(
                    max_iter=2000,
                    random_state=random_state,
                    C=1.0,
                    solver='lbfgs'
                )
            elif model_option == "Support Vector Machine (SVM)":
                model = SVC(
                    kernel=kernel,
                    C=C_param,
                    random_state=random_state,
                    probability=True,
                    gamma='scale'
                )
            elif model_option == "K-Nearest Neighbors (KNN)":
                model = KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    weights='distance',
                    metric='minkowski'
                )
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        
        st.success("✅ Model trained successfully!")
        
        st.markdown("---")
        st.header("📊 Step 5: Model Evaluation Results")
        
        accuracy = accuracy_score(y_test, y_pred)
        unique_classes = len(np.unique(y_test))
        
        if unique_classes == 2:
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
        else:
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        st.subheader("📈 Performance Metrics")
        
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        
        with col_m1:
            st.metric("Accuracy", f"{accuracy:.2%}", help="Overall correctness of predictions")
        with col_m2:
            st.metric("Precision", f"{precision:.2%}", help="Accuracy of positive predictions")
        with col_m3:
            st.metric("Recall", f"{recall:.2%}", help="Coverage of actual positives")
        with col_m4:
            st.metric("F1-Score", f"{f1:.2%}", help="Harmonic mean of precision and recall")
        with col_m5:
            st.metric("CV Score", f"{cv_scores.mean():.2%}", 
                     help=f"5-Fold Cross-Validation (±{cv_scores.std():.2%})")
        
        if accuracy > 0.85:
            st.success("🎉 Excellent model performance!")
        elif accuracy > 0.75:
            st.info("👍 Good model performance!")
        elif accuracy > 0.65:
            st.warning("⚠️ Moderate model performance. Consider trying Gradient Boosting.")
        else:
            st.error("❌ Poor model performance. Try feature engineering or a different model.")
        
        st.subheader("📋 Detailed Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
        
        st.subheader("🔍 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                   cbar_kws={"shrink": 0.8}, linewidths=2, linecolor='white')
        ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_xlabel('Predicted', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.clf()
        
        if hasattr(model, 'feature_importances_'):
            st.subheader("🎯 Feature Importance")
            feature_names = X.columns
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(fig
