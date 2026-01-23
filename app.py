import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

plt.switch_backend('Agg')

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦", layout="wide")
st.title("🏦 Loan Approval Prediction System")
st.markdown("**Wajiha Haleem | BS AI**")
st.markdown("---")

# ============== LOAD YOUR LOAN DATASET ==============
@st.cache_data
def load_loan_data():
    """Load the loan approval dataset"""
    try:
        # Try to load from file (put your CSV file path here)
        df = pd.read_csv('loan_approval_dataset.csv')
        
        # Drop ID column if exists
        for col in df.columns:
            if 'id' in col.lower():
                df.drop(columns=col, inplace=True)
        
        df.drop_duplicates(inplace=True)
        
        return df
    except:
        st.error("⚠️ Dataset file not found! Please upload your loan_approval_dataset.csv file")
        return None

# ============== FILE UPLOADER ==============
st.header("📁 Step 1: Load Loan Dataset")

uploaded_file = st.file_uploader("Upload your loan_approval_dataset.csv file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Drop ID column if exists
    for col in df.columns:
        if 'id' in col.lower():
            df = df.drop(columns=col)
    
    df.drop_duplicates(inplace=True)
    
    st.session_state.df = df
    st.session_state.loaded = True
    st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
else:
    # Try to load from local file
    df = load_loan_data()
    if df is not None:
        st.session_state.df = df
        st.session_state.loaded = True
        st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")

if 'loaded' not in st.session_state:
    st.warning("⬆️ Please upload your loan approval dataset to continue")
    st.info("💡 **File should be**: loan_approval_dataset.csv")
    st.stop()

df = st.session_state.df

# ============== DATASET INFO ==============
st.subheader("📊 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.metric("Total Rows", df.shape[0])
with col_info2:
    st.metric("Total Columns", df.shape[1])
with col_info3:
    st.metric("Missing Values", df.isnull().sum().sum())

# Show dataset info
with st.expander("📋 Dataset Information"):
    st.write("**Column Names:**")
    st.write(df.columns.tolist())
    st.write("\n**Data Types:**")
    st.write(df.dtypes)
    st.write("\n**Statistical Summary:**")
    st.dataframe(df.describe())

st.markdown("---")

# ============== EDA SECTION ==============
st.header("📈 Step 2: Exploratory Data Analysis (EDA)")

# Auto-detect target column
possible_targets = ['loan_status', 'loan status', 'status']
target_col = None
for col in df.columns:
    if col.lower() in possible_targets:
        target_col = col
        break
if target_col is None:
    target_col = df.columns[-1]

st.info(f"🎯 **Detected Target Column**: **{target_col}**")

# Target Distribution
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Loan Status Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    df[target_col].value_counts().plot(kind='bar', color=['#2ecc71', '#e74c3c'], ax=ax)
    ax.set_title("Loan Approval Distribution", fontsize=14, fontweight='bold')
    ax.set_xlabel("Loan Status", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("📊 Class Distribution")
    class_counts = df[target_col].value_counts()
    for cls, count in class_counts.items():
        st.metric(f"{cls}", count, f"{count/len(df)*100:.1f}%")

# Numerical Features Distribution
st.subheader("📊 Numerical Features Distribution")
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

if len(num_cols) > 0:
    selected_features = st.multiselect(
        "Select features to visualize:",
        num_cols,
        default=num_cols[:3] if len(num_cols) >= 3 else num_cols
    )
    
    if selected_features:
        cols_per_row = 3
        num_rows = (len(selected_features) + cols_per_row - 1) // cols_per_row
        
        for row in range(num_rows):
            cols = st.columns(cols_per_row)
            for i in range(cols_per_row):
                idx = row * cols_per_row + i
                if idx < len(selected_features):
                    col_name = selected_features[idx]
                    with cols[i]:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        df[col_name].hist(bins=20, color='skyblue', edgecolor='black', ax=ax)
                        ax.set_title(f"{col_name}", fontsize=12, fontweight='bold')
                        ax.set_xlabel(col_name, fontsize=10)
                        ax.set_ylabel("Frequency", fontsize=10)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

# Correlation Heatmap
if len(num_cols) > 1:
    st.subheader("🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', center=0, 
                linewidths=1, ax=ax, fmt='.2f', cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ============== MODEL TRAINING SECTION ==============
st.header("🤖 Step 3: Model Training")

col1, col2 = st.columns(2)

with col1:
    all_cols = df.columns.tolist()
    
    target_col_select = st.selectbox(
        "Select Target Column:",
        all_cols,
        index=all_cols.index(target_col) if target_col in all_cols else len(all_cols)-1,
        help="Select the column you want to predict"
    )
    
    # Validation
    unique_count = df[target_col_select].nunique()
    if unique_count > 50:
        st.error(f"❌ '{target_col_select}' has {unique_count} unique values! Select a categorical column.")
        st.stop()
    elif unique_count < 2:
        st.error(f"❌ '{target_col_select}' has only {unique_count} unique value!")
        st.stop()
    else:
        st.success(f"✅ '{target_col_select}' is valid! ({unique_count} classes)")
    
    classes = df[target_col_select].unique()
    st.caption(f"Classes: {', '.join(map(str, classes))}")

with col2:
    model_choice = st.selectbox(
        "Select Model:",
        ["🥇 Logistic Regression", "🥈 Decision Tree", "🥉 Support Vector Machine (SVM)"],
        help="Choose the machine learning model"
    )

with st.expander("⚙️ Advanced Settings"):
    test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5)
    random_state = st.number_input("Random State", 0, 100, 42)

if st.button("🚀 Train Model", type="primary", use_container_width=True):
    try:
        with st.spinner('🔄 Training model... Please wait'):
            # Prepare data
            X = df.drop(columns=[target_col_select]).copy()
            y = df[target_col_select].copy()
            
            class_counts = y.value_counts()
            st.info(f"📊 **Class Distribution**: {dict(class_counts)}")
            
            # Handle missing values
            num_cols_X = X.select_dtypes(include=['int64', 'float64']).columns
            cat_cols_X = X.select_dtypes(include=['object']).columns
            
            # Fill missing values
            if len(num_cols_X) > 0:
                X[num_cols_X] = X[num_cols_X].fillna(X[num_cols_X].mean())
            if len(cat_cols_X) > 0:
                for col in cat_cols_X:
                    X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown')
            
            # Encode categorical features
            le_dict = {}
            for col in cat_cols_X:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                le_dict[col] = le
            
            # Encode target
            if y.dtype == 'object':
                le_target = LabelEncoder()
                y_encoded = le_target.fit_transform(y.astype(str))
                target_classes = le_target.classes_
            else:
                y_encoded = y.values
                target_classes = np.unique(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size/100, random_state=random_state, stratify=y_encoded
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Select and train model
            if "Logistic" in model_choice:
                model = LogisticRegression(max_iter=1000, random_state=random_state)
            elif "Decision" in model_choice:
                model = DecisionTreeClassifier(random_state=random_state)
            else:  # SVM
                model = SVC(random_state=random_state)
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            
            st.success("✅ Model trained successfully!")
            
            # ============== RESULTS ==============
            st.markdown("---")
            st.header("📊 Step 4: Model Evaluation Results")
            
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
                st.metric("Accuracy", f"{accuracy:.2%}")
            with metric_cols[1]:
                st.metric("Precision", f"{precision:.2%}")
            with metric_cols[2]:
                st.metric("Recall", f"{recall:.2%}")
            with metric_cols[3]:
                st.metric("F1-Score", f"{f1:.2%}")
            with metric_cols[4]:
                st.metric("CV Score", f"{cv_scores.mean():.2%}")
            
            # Performance feedback
            if accuracy > 0.85:
                st.success("🎉 **EXCELLENT PERFORMANCE!** Model is ready for deployment!")
            elif accuracy > 0.75:
                st.info("👍 **GOOD PERFORMANCE!** Model works well.")
            elif accuracy > 0.65:
                st.warning("⚠️ **MODERATE PERFORMANCE.** Consider trying different model.")
            else:
                st.error("❌ **POOR PERFORMANCE.** Try a different model or check your data.")
            
            # Print metrics (like Kaggle code)
            st.subheader("📋 Model Performance Summary")
            st.code(f"""
{model_choice.split()[1]} Model Results:
{'='*50}
Accuracy  : {accuracy:.4f}
Precision : {precision:.4f}
Recall    : {recall:.4f}
F1 Score  : {f1:.4f}
CV Score  : {cv_scores.mean():.4f} (±{cv_scores.std():.4f})
{'='*50}
""")
            
            # Confusion Matrix
            st.subheader("🔍 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                           cbar_kws={"shrink": 0.8}, linewidths=2)
                ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
                ax.set_ylabel('Actual', fontsize=12)
                ax.set_xlabel('Predicted', fontsize=12)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.write("**Confusion Matrix Values:**")
                st.code(f"Confusion Matrix:\n{cm}")
                
                if len(target_classes) == 2:
                    tn, fp, fn, tp = cm.ravel()
                    st.write(f"- True Negatives: {tn}")
                    st.write(f"- False Positives: {fp}")
                    st.write(f"- False Negatives: {fn}")
                    st.write(f"- True Positives: {tp}")
            
            # Classification Report
            st.subheader("📋 Detailed Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.highlight_max(axis=0, color='lightgreen'), 
                        use_container_width=True)
            
            # Feature Importance (for Decision Tree only)
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
                ax.barh(top_features['Feature'], top_features['Importance'], 
                       color='purple', alpha=0.7)
                ax.set_title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
                ax.set_xlabel('Importance Score', fontsize=12)
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                with st.expander("📊 View All Feature Importances"):
                    st.dataframe(importance_df, use_container_width=True)
            
            st.success("✅ EDA + ML completed successfully!")
    
    except Exception as e:
        st.error(f"❌ **Error during training**: {str(e)}")
        st.error("**Possible Solutions:**")
        st.markdown("- Make sure you selected a valid target column")
        st.markdown("- Check if your data has the correct format")
        st.markdown("- Verify that the target column has 2-20 unique values")
        
        with st.expander("🔍 See Full Error Details"):
            import traceback
            st.code(traceback.format_exc())

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    🏦 <b>Loan Approval Prediction System</b><br>
    Wajiha Haleem | BS AI | Built with Streamlit 🚀
</div>
""", unsafe_allow_html=True)
