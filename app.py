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
import warnings
warnings.filterwarnings('ignore')

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
- Upload your dataset
- Perform exploratory data analysis (EDA)
- Visualize data patterns
- Train and evaluate machine learning models
""")
st.markdown("---")

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'df' not in st.session_state:
    st.session_state.df = None
if 'eda_performed' not in st.session_state:
    st.session_state.eda_performed = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# ============================================================================
# SECTION 1: DATASET UPLOAD
# ============================================================================
st.header("📁 Step 1: Upload Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        # Load the dataset
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success(f"✅ Dataset uploaded successfully! Shape: {df.shape}")
        
        # Display dataset preview
        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head(10))
        
        # Download sample dataset option
        st.info("💡 First time? The app works with any CSV file containing features and a target column.")
        
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
else:
    st.info("👆 Please upload a CSV file to get started")
    st.stop()

# ============================================================================
# SECTION 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
st.markdown("---")
st.header("🔍 Step 2: Exploratory Data Analysis (EDA)")

if st.button("🚀 Perform EDA", type="primary"):
    st.session_state.eda_performed = True

if st.session_state.eda_performed:
    df = st.session_state.df
    
    # EDA Status Indicator
    st.success("✅ **EDA PERFORMED SUCCESSFULLY**")
    
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
        st.dataframe(info_df)
    
    with col_info2:
        st.write("**Summary Statistics:**")
        st.dataframe(df.describe())
    
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
        st.pyplot(fig)
        plt.close()
    else:
        st.success("✅ No missing values found in the dataset!")
    
    # Data types distribution
    st.subheader("📊 Data Types Distribution")
    dtype_counts = df.dtypes.value_counts()
    col_dtype1, col_dtype2 = st.columns(2)
    
    with col_dtype1:
        fig, ax = plt.subplots(figsize=(6, 4))
        dtype_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=['#ff9999','#66b3ff','#99ff99'])
        ax.set_ylabel('')
        ax.set_title('Data Types Distribution')
        st.pyplot(fig)
        plt.close()
    
    with col_dtype2:
        st.write("**Numerical Columns:**")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        st.write(numeric_cols if numeric_cols else "None")
        
        st.write("**Categorical Columns:**")
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        st.write(categorical_cols if categorical_cols else "None")

# ============================================================================
# SECTION 3: DATA VISUALIZATIONS
# ============================================================================
if st.session_state.eda_performed:
    st.markdown("---")
    st.header("📈 Step 3: Data Visualizations")
    
    df = st.session_state.df
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Histograms
    if numeric_cols:
        st.subheader("📊 Histograms (Numerical Features)")
        
        selected_cols_hist = st.multiselect(
            "Select columns for histograms:",
            numeric_cols,
            default=numeric_cols[:min(4, len(numeric_cols))]
        )
        
        if selected_cols_hist:
            n_cols = 2
            n_rows = (len(selected_cols_hist) + 1) // 2
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 4))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
            
            for idx, col in enumerate(selected_cols_hist):
                if idx < len(axes):
                    axes[idx].hist(df[col].dropna(), bins=30, color='skyblue', edgecolor='black')
                    axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
                    axes[idx].set_xlabel(col)
                    axes[idx].set_ylabel('Frequency')
                    axes[idx].grid(alpha=0.3)
            
            # Hide empty subplots
            for idx in range(len(selected_cols_hist), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    # Box Plots
    if numeric_cols:
        st.subheader("📦 Box Plots (Outlier Detection)")
        
        selected_cols_box = st.multiselect(
            "Select columns for box plots:",
            numeric_cols,
            default=numeric_cols[:min(4, len(numeric_cols))],
            key='boxplot_select'
        )
        
        if selected_cols_box:
            n_cols = 2
            n_rows = (len(selected_cols_box) + 1) // 2
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 4))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
            
            for idx, col in enumerate(selected_cols_box):
                if idx < len(axes):
                    axes[idx].boxplot(df[col].dropna(), vert=True, patch_artist=True,
                                     boxprops=dict(facecolor='lightgreen', alpha=0.7))
                    axes[idx].set_title(f'Box Plot of {col}', fontweight='bold')
                    axes[idx].set_ylabel(col)
                    axes[idx].grid(alpha=0.3)
            
            # Hide empty subplots
            for idx in range(len(selected_cols_box), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    # Correlation Heatmap
    if len(numeric_cols) > 1:
        st.subheader("🔥 Correlation Heatmap")
        
        corr_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title('Correlation Matrix', fontsize=16, fontweight='bold')
        st.pyplot(fig)
        plt.close()
    
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
            
            value_counts.plot(kind='bar', color='teal', ax=ax)
            ax.set_title(f'Count Plot of {selected_cat_col}', fontsize=14, fontweight='bold')
            ax.set_xlabel(selected_cat_col)
            ax.set_ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            ax.grid(alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ============================================================================
# SECTION 4: MODEL SELECTION AND TRAINING
# ============================================================================
if st.session_state.eda_performed:
    st.markdown("---")
    st.header("🎯 Step 4: Model Selection & Training")
    
    df = st.session_state.df
    
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
            # Prepare data
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Handle categorical features in X
            categorical_features = X.select_dtypes(include=['object']).columns
            
            if len(categorical_features) > 0:
                st.info(f"Encoding categorical features: {', '.join(categorical_features)}")
                le_dict = {}
                for col in categorical_features:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    le_dict[col] = le
            
            # Encode target if categorical
            if y.dtype == 'object':
                le_target = LabelEncoder()
                y = le_target.fit_transform(y)
                st.info(f"Target classes: {', '.join(le_target.classes_)}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
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
            with st.spinner('Training model...'):
                model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Store in session state
            st.session_state.model_trained = True
            st.session_state.model = model
            st.session_state.X_test_scaled = X_test_scaled
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred
            
            st.success("✅ Model trained successfully!")
            
        except Exception as e:
            st.error(f"Error during training: {e}")
            st.stop()

# ============================================================================
# SECTION 5: MODEL EVALUATION
# ============================================================================
if st.session_state.get('model_trained', False):
    st.markdown("---")
    st.header("📊 Step 5: Model Evaluation")
    
    y_test = st.session_state.y_test
    y_pred = st.session_state.y_pred
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Handle binary vs multiclass
    if len(np.unique(y_test)) == 2:
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
    else:
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Display metrics
    st.subheader("📈 Performance Metrics")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.metric("Accuracy", f"{accuracy:.2%}")
    with col_m2:
        st.metric("Precision", f"{precision:.2%}")
    with col_m3:
        st.metric("Recall", f"{recall:.2%}")
    with col_m4:
        st.metric("F1-Score", f"{f1:.2%}")
    
    # Classification Report
    st.subheader("📋 Detailed Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.highlight_max(axis=0, color='lightgreen'))
    
    # Confusion Matrix
    st.subheader("🔍 Confusion Matrix")
    
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                cbar_kws={"shrink": 0.8}, linewidths=1, linecolor='gray')
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12)
    
    st.pyplot(fig)
    plt.close()
    
    # Feature Importance (if applicable)
    if hasattr(st.session_state.model, 'feature_importances_'):
        st.subheader("🎯 Feature Importance")
        
        feature_names = df.drop(columns=[target_column]).columns
        importances = st.session_state.model.feature_importances_
        
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        feature_importance_df.head(10).plot(
            x='Feature', y='Importance', kind='barh', ax=ax, color='purple', alpha=0.7
        )
        ax.set_title('Top 10 Feature Importances', fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance')
        plt.tight_layout()
        
        st.pyplot(fig)
        plt.close()
        
        with st.expander("View All Feature Importances"):
            st.dataframe(feature_importance_df)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🤖 ML Model Trainer | Built with Streamlit</p>
    <p>Ready to deploy on Streamlit Cloud</p>
</div>
""", unsafe_allow_html=True)
