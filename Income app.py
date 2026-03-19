import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Income Classification", layout="wide")
st.title("🎯 Income Classification - End-to-End ML Platform")

@st.cache_data
def load_data():
    return pd.read_csv('income_evaluation.csv')

@st.cache_resource
def load_trained_model():
    """Load the pre-trained model from pickle file"""
    try:
        with open('Income.pkl', 'rb') as f:
            model_package = pickle.load(f)
        return model_package
    except FileNotFoundError:
        st.error("❌ Income.pkl not found! Please run train_and_save_model.py first.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

@st.cache_data
def preprocess_for_display(df):
    """Preprocess data for EDA visualization only"""
    df = df.copy()
    df = df.fillna(df.median(numeric_only=True))
    
    # Encode categorical variables for correlation
    df_encoded = df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
    
    return df, df_encoded

# Load data and model
df = load_data()
df_display, df_encoded = preprocess_for_display(df)

# Load pre-trained model
model_package = load_trained_model()

if model_package is None:
    st.stop()

# Extract model components
model = model_package['model']
scaler = model_package['scaler']
label_encoders = model_package['label_encoders']
features = model_package['features']
best_model_name = model_package['model_name']
best_accuracy = model_package['accuracy']

st.sidebar.success(f"✓ Model Loaded: {best_model_name}")
st.sidebar.info(f"📊 Accuracy: {best_accuracy:.2%}")

# Main navigation
tabs = st.tabs(["📈 Dataset Overview", "🤖 Model Performance", "🎯 Make Prediction", "📊 Visualizations", "📋 Model Details"])

with tabs[0]:
    st.header("Dataset Overview")
    st.write(f"**Total samples:** {df.shape[0]}")
    st.write(f"**Total features:** {df.shape[1]}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**First 5 rows:**")
        st.dataframe(df.head(), use_container_width=True)
    
    with col2:
        st.write("**Data Info:**")
        st.write(f"- Numerical features: {df.select_dtypes(include=[np.number]).shape[1]}")
        st.write(f"- Categorical features: {df.select_dtypes(include=['object']).shape[1]}")
        st.write(f"- Missing values: {df.isnull().sum().sum()}")

with tabs[1]:
    st.header("🤖 Model Performance")
    st.write(f"**Best Model:** {best_model_name}")
    st.write(f"**Test Accuracy:** {best_accuracy:.4f}")
    
    if 'all_trained_models' in model_package and 'results' in model_package:
        st.write("**Model Comparison:**")
        results_df = pd.DataFrame(model_package['results']).T
        st.dataframe(results_df.round(4), use_container_width=True)
    
    st.write("**Features Used for Prediction:**")
    st.text(", ".join(features))

with tabs[2]:
    st.header("🎯 Make a Prediction")
    st.write("Enter feature values to predict income level:")
    
    prediction_data = {}
    
    # Create columns for better layout
    cols = st.columns(2)
    col_idx = 0
    
    for feature in features:
        with cols[col_idx % 2]:
            if feature in label_encoders:
                # Categorical feature
                choices = sorted(list(label_encoders[feature].classes_))
                selected = st.selectbox(
                    f"📌 {feature}",
                    choices,
                    key=f"select_{feature}"
                )
                prediction_data[feature] = label_encoders[feature].transform([selected])[0]
            else:
                # Numerical feature
                mean_val = float(df[feature].mean())
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                
                val = st.slider(
                    f"📌 {feature}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    key=f"slider_{feature}"
                )
                prediction_data[feature] = val
            
            col_idx += 1
    
    if st.button("🚀 Predict Income", use_container_width=True):
        try:
            # Prepare input data
            input_df = pd.DataFrame([prediction_data])
            input_scaled = scaler.transform(input_df[features])
            
            # Make prediction
            pred = model.predict(input_scaled)[0]
            pred_proba = None
            
            # Try to get probability if available
            try:
                pred_proba = model.predict_proba(input_scaled)[0]
            except:
                pass
            
            # Decode prediction
            if ' income' in label_encoders:
                pred_label = label_encoders[' income'].inverse_transform([int(pred)])[0]
            else:
                pred_label = str(pred)
            
            # Display result
            st.success(f"✅ Predicted Income Class: **{pred_label}**")
            
            if pred_proba is not None and len(pred_proba) > 0:
                st.write("**Prediction Confidence:**")
                classes = label_encoders[' income'].classes_
                for class_name, prob in zip(classes, pred_proba):
                    st.progress(float(prob))
                    st.text(f"{class_name}: {prob:.2%}")
        
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")

with tabs[3]:
    st.header("📊 Visualizations")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.subheader("Income Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df_display, x=' income', ax=ax, palette='Set2')
        ax.set_title("Income Class Distribution", fontsize=12, fontweight='bold')
        ax.set_xlabel('Income')
        ax.set_ylabel('Count')
        st.pyplot(fig)
    
    with viz_col2:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation = df_encoded.corr()
        sns.heatmap(correlation, annot=False, cmap='coolwarm', ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title("Feature Correlation Heatmap", fontsize=12, fontweight='bold')
        st.pyplot(fig)
    
    st.subheader("Feature Distribution")
    selected_feature = st.selectbox("Select a feature to visualize:", features)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    if selected_feature in label_encoders:
        # Categorical feature
        sns.countplot(data=df_display, x=selected_feature, ax=ax, palette='Set2')
        ax.set_title(f"Distribution of {selected_feature}", fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
    else:
        # Numerical feature
        sns.histplot(data=df_display, x=selected_feature, kde=True, ax=ax, color='skyblue')
        ax.set_title(f"Distribution of {selected_feature}", fontsize=12, fontweight='bold')
    
    st.pyplot(fig)

with tabs[4]:
    st.header("📋 Model Details")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best Model", best_model_name)
    with col2:
        st.metric("Accuracy", f"{best_accuracy:.2%}")
    with col3:
        st.metric("Features", len(features))
    
    st.subheader("Model Components")
    st.write(f"✓ **Scaler:** StandardScaler")
    st.write(f"✓ **Categorical Encoders:** {len(label_encoders)} LabelEncoders")
    st.write(f"✓ **Features Used:** {len(features)} features")
    
    st.subheader("Dataset Information")
    st.write(f"✓ **Total Samples:** {df.shape[0]:,}")
    st.write(f"✓ **Total Features:** {df.shape[1]}")
    st.write(f"✓ **Numerical Features:** {df.select_dtypes(include=[np.number]).shape[1]}")
    st.write(f"✓ **Categorical Features:** {df.select_dtypes(include=['object']).shape[1]}")
    
    st.subheader("Feature List")
    st.text("\n".join(features))
