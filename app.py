 import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend
plt.switch_backend('Agg')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="ML Model Trainer",
    page_icon="🤖",
    layout="wide"
)

# ============================================================================
# TITLE AND INTRODUCTION
# ============================================================================
st.title("🤖 Machine Learning Model Trainer")
st.markdown("""
Welcome to the **ML Model Trainer**! This application allows you to:
- Use a sample dataset or upload your own CSV
- Perform exploratory data analysis (EDA)
- Visualize data patterns
- Train and evaluate machine learning models
""")
st.markdown("---")

# ============================================================================
# FUNCTION TO CREATE SAMPLE DATASETS
# ============================================================================
@st.cache_data
def create_sample_dataset(dataset_name):
    """Create sample datasets for demo purposes"""
    if dataset_name == "Iris Flowers":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        return df
    
    elif dataset_name == "Wine Quality":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['wine_class'] = data.target
        return df
    
    elif dataset_name == "Breast Cancer":
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['diagnosis'] = data.target
        df['diagnosis_label'] = df['diagnosis'].map({0: 'malignant', 1: 'benign'})
        return df
    
    elif dataset_name == "Customer Churn (Synthetic)":
        np.random.seed(42)
        n_samples = 500
        df = pd.DataFrame({
            'age': np.random.randint(18, 70, n_samples),
            'tenure': np.random.randint(1, 72, n_samples),
            'monthly_charges': np.random.uniform(20, 120, n_samples),
            'total_charges': np.random.uniform(100, 8000, n_samples),
            'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'customer_service_calls': np.random.randint(0, 10, n_samples),
        })
        # Create target based on some logic
        churn_prob = (
            (df['contract_type'] == 'Month-to-month') * 0.3 +
            (df['monthly_charges'] > 80) * 0.2 +
            (df['customer_service_calls'] > 5) * 0.3 +
            np.random.random(n_samples) * 0.2
        )
        df['churn'] = (churn_prob > 0.5).astype(int)
        df['churn_label'] = df['churn'].map({0: 'No', 1: 'Yes'})
        return df
    
    elif dataset_name == "Loan Approval (Synthetic)":
        np.random.seed(42)
        n_samples = 600
        df = pd.DataFrame({
            'age': np.random.randint(21, 65, n_samples),
            'income': np.random.randint(20000, 150000, n_samples),
            'loan_amount': np.random.randint(5000, 50000, n_samples),
            'credit_score': np.random.randint(300, 850, n_samples),
            'employment_years': np.random.randint(0, 40, n_samples),
            'debt_to_income': np.random.uniform(0, 0.6, n_samples),
            'loan_term': np.random.choice([12, 24, 36, 48, 60], n_samples),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
            'home_ownership': np.random.choice(['Rent', 'Own', 'Mortgage'], n_samples),
        })
        # Create target based on some logic
        approval_score = (
            (df['credit_score'] > 650) * 0.3 +
            (df['income'] > 50000) * 0.2 +
            (df['debt_to_income'] < 0.4) * 0.2 +
            (df['employment_years'] > 2) * 0.15 +
            np.random.random(n_samples) * 0.15
        )
        df['loan_approved'] = (approval_score > 0.5).astype(int)
        df['approval_status'] = df['loan_approved'].map({0: 'Rejected', 1: 'Approved'})
        return df

# ============================================================================
# SECTION 1: DATASET SELECTION/UPLOAD
# ============================================================================
st.header("📁 Step 1: Choose Dataset")

# Create tabs for sample vs upload
tab1, tab2 = st.tabs(["🎯 Use Sample Dataset", "📤 Upload Your Own CSV"])

df = None

with tab1:
    st.markdown("### Select a Sample Dataset to Try the App")
    st.info("👉 Perfect for testing! Choose one of our sample datasets below.")
    
    sample_dataset = st.selectbox(
        "Choose a sample dataset:",
        [
            "Iris Flowers",
            "Wine Quality", 
            "Breast Cancer",
            "Customer Churn (Synthetic)",
            "Loan Approval (Synthetic)"
        ]
    )
    
    if st.button("📊 Load Sample Dataset", type="primary"):
        df = create_sample_dataset(sample_dataset)
        st.session_state.df = df
        st.session_state.dataset_loaded = True
        st.success(f"✅ {sample_dataset} dataset loaded! Shape: {df.shape}")
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

# Check if dataset is loaded
if 'dataset_loaded' not in st.session_state or not st.session_state.dataset_loaded:
    st.warning("⬆️ Please select a sample dataset or upload your own CSV file to continue")
    st.stop()

# Get dataframe from session state
df = st.session_state.df

# Display dataset preview
st.subheader("📊 Dataset Preview")
st.dataframe(df.head(10))

# ============================================================================
# SECTION 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
st.markdown("---")
st.header("🔍 Step 2: Exploratory Data Analysis (EDA)")

st.success("✅ **EDA PERFORMED**")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Rows", df.shape[0])
with col2:
    st.metric("Total Columns", df.shape[1])
with col3:
    st.metric("Missing Values", df.isnull().sum().sum())

# Dataset Information
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

# Missing Values Visualization
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

# Data types distribution
st.subheader("📊 Data Types Distribution")
dtype_counts = df.dtypes.value_counts()
col_dtype1, col_dtype2 = st.columns(2)

with col_dtype1:
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    dtype_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=colors[:len(dtype_counts)])
    ax.set_ylabel('')
    ax.set_title('Data Types Distribution')
    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()

with col_dtype2:
    st.write("**Numerical Columns:**")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        for col in numeric_cols:
            st.write(f"• {col}")
    else:
        st.write("None")
    
    st.write("**Categorical Columns:**")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        for col in categorical_cols:
            st.write(f"• {col}")
    else:
        st.write("None")

# ============================================================================
# SECTION 3: DATA VISUALIZATIONS
# ============================================================================
st.markdown("---")
st.header("📈 Step 3: Data Visualizations")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Histograms
if numeric_cols:
    st.subheader("📊 Histograms (Numerical Features)")
    
    default_hist_cols = numeric_cols[:min(4, len(numeric_cols))]
    selected_cols_hist = st.multiselect(
        "Select columns for histograms:",
        numeric_cols,
        default=default_hist_cols
    )
    
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
        
        # Hide empty subplots
        for idx in range(len(selected_cols_hist), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.clf()
    else:
        st.info("Select at least one column to display histograms")

# Box Plots
if numeric_cols:
    st.subheader("📦 Box Plots (Outlier Detection)")
    
    default_box_cols = numeric_cols[:min(3, len(numeric_cols))]
    selected_cols_box = st.multiselect(
        "Select columns for box plots:",
        numeric_cols,
        default=default_box_cols,
        key='boxplot_select'
    )
    
    if selected_cols_box:
        n_cols = 2
        n_rows = (len(selected_cols_box) + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 4))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for idx, col in enumerate(selected_cols_box):
            if idx < len(axes):
                bp = axes[idx].boxplot(df[col].dropna(), vert=True, patch_artist=True,
                                 boxprops=dict(facecolor='lightgreen', alpha=0.7),
                                 medianprops=dict(color='red', linewidth=2))
                axes[idx].set_title(f'Box Plot of {col}', fontweight='bold', fontsize=12)
                axes[idx].set_ylabel(col)
                axes[idx].grid(alpha=0.3, axis='y')
        
        # Hide empty subplots
        for idx in range(len(selected_cols_box), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.clf()
    else:
        st.info("Select at least one column to display box plots")

# Correlation Heatmap
if len(numeric_cols) > 1:
    st.subheader("🔥 Correlation Heatmap")
    
    corr_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
               center=0, square=True, linewidths=1, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title('Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()

# Count Plots for Categorical Features
if categorical_cols:
    st.subheader("📊 Count Plots (Categorical Features)")
    
    selected_cat_col = st.selectbox("Select categorical column:", categorical_cols)
    
    if selected_cat_col:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        value_counts = df[selected_cat_col].value_counts()
        
        # Limit to top 20 categories if too many
        if len(value_counts) > 20:
            value_counts = value_counts.head(20)
            st.warning(f"Showing top 20 categories only (out of {len(df[selected_cat_col].unique())})")
        
        value_counts.plot(kind='bar', color='teal', ax=ax, alpha=0.7)
        ax.set_title(f'Count Plot of {selected_cat_col}', fontsize=14, fontweight='bold')
        ax.set_xlabel(selected_cat_col)
        ax.set_ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        ax.grid(alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(value_counts.values):
            ax.text(i, v + 0.5, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.clf()

# ============================================================================
# SECTION 4: MODEL SELECTION AND TRAINING
# ============================================================================
st.markdown("---")
st.header("🎯 Step 4: Model Selection & Training")

col_model1, col_model2 = st.columns(2)

with col_model1:
    # Select target column
    target_column = st.selectbox(
        "Select Target Column:",
        df.columns.tolist(),
        help="Choose the column you want to predict"
    )

with col_model2:
    # Select model
    model_option = st.selectbox(
        "Select Machine Learning Model:",
        ["Logistic Regression", "Support Vector Machine (SVM)", 
         "Random Forest", "K-Nearest Neighbors (KNN)"]
    )

# Advanced options
with st.expander("⚙️ Advanced Settings"):
    test_size = st.slider("Test Set Size (%)", 10, 50, 20, 5) / 100
    random_state = st.number_input("Random State", 0, 100, 42)
    
    # Model-specific parameters
    if model_option == "Random Forest":
        n_estimators = st.slider("Number of Trees", 10, 200, 100, 10)
        max_depth = st.slider("Max Depth", 1, 20, 10)
    elif model_option == "K-Nearest Neighbors (KNN)":
        n_neighbors = st.slider("Number of Neighbors", 1, 20, 5)
    elif model_option == "Support Vector Machine (SVM)":
        kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"])

# Train Model Button
if st.button("🚀 Train Model", type="primary"):
    try:
        with st.spinner('🔄 Preparing data and training model...'):
            # Prepare data
            X = df.drop(columns=[target_column]).copy()
            y = df[target_column].copy()
            
            # Handle categorical features in X
            categorical_features = X.select_dtypes(include=['object']).columns
            
            if len(categorical_features) > 0:
                st.info(f"🔄 Encoding categorical features: {', '.join(categorical_features)}")
                for col in categorical_features:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
            
            # Encode target if categorical
            if y.dtype == 'object':
                le_target = LabelEncoder()
                y_encoded = le_target.fit_transform(y)
                st.info(f"🎯 Target classes: {', '.join(map(str, le_target.classes_))}")
            else:
                y_encoded = y
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
            )
            
            st.info(f"📊 Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Initialize model
            if model_option == "Logistic Regression":
                model = LogisticRegression(max_iter=1000, random_state=random_state)
            elif model_option == "Support Vector Machine (SVM)":
                model = SVC(kernel=kernel, random_state=random_state, probability=True)
            elif model_option == "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=n_estimators, 
                    max_depth=max_depth, 
                    random_state=random_state
                )
            elif model_option == "K-Nearest Neighbors (KNN)":
                model = KNeighborsClassifier(n_neighbors=n_neighbors)
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
        
        st.success("✅ Model trained successfully!")
        
        # ============================================================================
        # MODEL EVALUATION
        # ============================================================================
        st.markdown("---")
        st.header("📊 Step 5: Model Evaluation Results")
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Handle binary vs multiclass
        unique_classes = len(np.unique(y_test))
        if unique_classes == 2:
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
        else:
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Display metrics
        st.subheader("📈 Performance Metrics")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.metric("Accuracy", f"{accuracy:.2%}", help="Overall correctness of predictions")
        with col_m2:
            st.metric("Precision", f"{precision:.2%}", help="Accuracy of positive predictions")
        with col_m3:
            st.metric("Recall", f"{recall:.2%}", help="Coverage of actual positives")
        with col_m4:
            st.metric("F1-Score", f"{f1:.2%}", help="Harmonic mean of precision and recall")
        
        # Interpretation
        if accuracy > 0.9:
            st.success("🎉 Excellent model performance!")
        elif accuracy > 0.8:
            st.info("👍 Good model performance!")
        elif accuracy > 0.7:
            st.warning("⚠️ Moderate model performance. Consider tuning parameters.")
        else:
            st.error("❌ Poor model performance. Try a different model or feature engineering.")
        
        # Classification Report
        st.subheader("📋 Detailed Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
        
        # Confusion Matrix
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
        
        # Feature Importance (if applicable)
        if hasattr(model, 'feature_importances_'):
            st.subheader("🎯 Feature Importance")
            
            feature_names = X.columns
            importances = model.feature_importances_
            
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            top_features = feature_importance_df.head(10)
            ax.barh(top_features['Feature'], top_features['Importance'], color='purple', alpha=0.7)
            ax.set_title('Top 10 Feature Importances', fontsize=14, fontweight='bold')
            ax.set_xlabel('Importance Score')
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
            plt.clf()
            
            with st.expander("📊 View All Feature Importances"):
                st.dataframe(feature_importance_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error during training: {e}")
        import traceback
        st.code(traceback.format_exc())

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p style='font-size: 16px;'><strong>🤖 ML Model Trainer | Built with Streamlit</strong></p>
    <p>Try different datasets and models to see how they perform!</p>
    <p style='font-size: 12px;'>Ready to deploy on Streamlit Cloud 🚀</p>
</div>
""", unsafe_allow_html=True)
