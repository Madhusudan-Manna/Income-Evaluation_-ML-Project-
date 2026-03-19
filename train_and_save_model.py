"""
Train and save the Income Classification model to pickle file
Based on Income_Classification.ipynb notebook workflow
"""

import pandas as pd
import numpy as np
import pickle
import warnings
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

print("=" * 80)
print("INCOME CLASSIFICATION: MODEL TRAINING & SAVING")
print("=" * 80)

# 1. Load the dataset
print("\n[1/7] Loading dataset...")
df = pd.read_csv('income_evaluation.csv')
print(f"Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# 2. Basic EDA
print("\n[2/7] Exploratory Data Analysis...")
print("Data Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())

# Get numerical and categorical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=['object']).columns
print(f"\nNumerical columns: {list(numerical_cols)}")
print(f"Categorical columns: {list(categorical_cols)}")

# 3. Check skewness and apply transformations
print("\n[3/7] Checking Skewness & Applying Transformations...")
skewness = df[numerical_cols].apply(lambda x: skew(x.dropna()))
print("Skewness of numerical features:")
print(skewness)

skewed_features = skewness[abs(skewness) > 0.5].index
df_transformed = df.copy()
for col in skewed_features:
    if (df[col] > 0).all():
        df_transformed[col] = np.log1p(df[col])
        print(f"Applied log transformation to {col}")

# 4. Handle missing values, encode categorical variables, detect and treat outliers
print("\n[4/7] Data Cleaning, Encoding & Outlier Treatment...")
df_clean = df_transformed.fillna(df_transformed.median(numeric_only=True))

# Encode categorical variables
df_encoded = df_clean.copy()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_clean[col].astype(str))
    label_encoders[col] = le
print(f"Encoded {len(categorical_cols)} categorical columns")

# Detect and cap outliers using IQR
outlier_count = 0
for col in numerical_cols:
    Q1 = df_encoded[col].quantile(0.25)
    Q3 = df_encoded[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((df_encoded[col] < lower_bound) | (df_encoded[col] > upper_bound)).sum()
    outlier_count += outliers
    df_encoded[col] = np.clip(df_encoded[col], lower_bound, upper_bound)
print(f"Detected and capped {outlier_count} outliers")

# 5. Feature scaling
print("\n[5/7] Feature Scaling...")
scaler = StandardScaler()
features = [col for col in df_encoded.columns if col != ' income']
df_scaled = df_encoded.copy()
if len(features) > 0:
    df_scaled[features] = scaler.fit_transform(df_encoded[features])
print(f"Scaled {len(features)} features")

# 6. Prepare data and train models
print("\n[6/7] Training Multiple Models...")
X = df_scaled[features]
y = df_scaled[' income']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Define and train models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'SVM': SVC(random_state=42),
}

if XGBClassifier is not None:
    models['XGBoost'] = XGBClassifier(random_state=42, verbosity=0)

results = {}
trained_models = {}

for name, model in models.items():
    print(f"\n  Training {name}...")
    model.fit(X_train, y_train)
    trained_models[name] = model
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_acc = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    results[name] = {'Accuracy': acc, 'CV_Accuracy': cv_acc}
    print(f"    Test Accuracy: {acc:.4f}")
    print(f"    CV Accuracy: {cv_acc:.4f}")

# 7. Select best model and save
print("\n[7/7] Selecting Best Model & Saving...")
results_df = pd.DataFrame(results).T
best_model_name = results_df['Accuracy'].idxmax()
best_model = trained_models[best_model_name]
best_acc = results_df.loc[best_model_name, 'Accuracy']

print(f"\nBest Model: {best_model_name}")
print(f"Best Accuracy: {best_acc:.4f}")

# Prepare model package for saving
model_package = {
    'model': best_model,
    'model_name': best_model_name,
    'accuracy': best_acc,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'features': features,
    'numerical_cols': list(numerical_cols),
    'categorical_cols': list(categorical_cols),
    'all_trained_models': trained_models,
    'results': results
}

# Save to pickle file
with open('Income.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print(f"\n✓ Model successfully saved to Income.pkl")
print(f"\nModel Package Contents:")
print(f"  - Best Model: {best_model_name}")
print(f"  - Model Accuracy: {best_acc:.4f}")
print(f"  - Scaler: StandardScaler")
print(f"  - Label Encoders: {len(label_encoders)} encoders")
print(f"  - Features: {len(features)} features")
print(f"  - All Models: {len(trained_models)} models trained")

print("\n" + "=" * 80)
print("MODEL TRAINING COMPLETE!")
print("=" * 80)
