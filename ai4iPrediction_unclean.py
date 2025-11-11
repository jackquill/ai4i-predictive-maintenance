# ==============================================================
#  ai4iPrediction_unclean.py
#  Predictive Maintenance Model - AI4I PMDI Dataset (Unclean)
#
#  Description:
#       This script processes the unclean AI4I-PMDI dataset.
#       It performs feature engineering, encodes categorical
#       variables, handles missing data, and trains ML models
#       (Random Forest, Gradient Boosting, XGBoost) to predict
#       if there will be an failure.
#
#   Simular to ai4iPrediction.py but there are missing variables 
#   and 2 new columns "Control" and "System".
#
#  Course: CPTS 437 - Machine Learning
# ==============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# ==============================================================
#  Helper Function
# ==============================================================

def evaluate_model(model, X_train, y_train, X_test, y_test, name="Model"):
    """Train a model, evaluate metrics, and print results."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test) # predict returns class labels (0 or 1). used for confusion matrix and classification report
    probs = model.predict_proba(X_test)[:, 1] # predict_proba returns probabilities for each class. used for ROC-AUC 

    auc_score = roc_auc_score(y_test, probs)
    print(f"\n=== {name} ===")
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))
    print(f"ROC-AUC: {auc_score:.4f}")

    return auc_score


# ==============================================================
#  Load & Preprocess Data
# ==============================================================

df = pd.read_csv("data/AI4I-PMDI.csv")

# Handle missing values by filling numeric columns with their mean
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].mean(), inplace=True)

# Drop metadata columns not relevant for prediction
df.drop(columns=["UDI", "Date"], inplace=True, errors="ignore")

# Encode 'Control' (categorical: A, B, C)
if "Control" in df.columns:
    df = pd.get_dummies(df, columns=["Control"], prefix="control")

if "Type" in df.columns:
    df = pd.get_dummies(df, columns=["Type"], prefix="quality")

# ==============================================================
#  Target Variable
# ==============================================================

# Diagnostic: "No Failure" = 0, any failure type = 1
df["machine_failure"] = np.where(df["Diagnostic"].str.lower().str.contains("no failure"), 0, 1)
df.drop(columns=["Diagnostic"], inplace=True)

# ==============================================================
#  Feature Engineering
# ==============================================================

# 1) Temperature difference
df["temp_diff"] = df["Process temperature (K)"] - df["Air temperature (K)"]

# 2) Convert rotational speed (rpm) to radians per second
df["omega_rad_s"] = (df["Rotational speed (rpm)"] * 2 * np.pi) / 60

# 3) Mechanical power = torque × angular speed
df["power"] = df["Torque (Nm)"] * df["omega_rad_s"]

# 4) Overstrain: wear × torque
df["wear_and_torque"] = df["Tool wear (min)"] * df["Torque (Nm)"]

# Clean column names
df.columns = (
    df.columns
    .str.replace('[', '', regex=False)
    .str.replace(']', '', regex=False)
    .str.replace('<', '', regex=False)
    .str.replace(' ', '_')
)

# print(f"Final dataset shape: {df.shape}")
# print(df.head())

# ==============================================================
#  Train/Test Split
# ==============================================================

X = df.drop(columns=["machine_failure", "Product_ID"], errors='ignore')
y = df["machine_failure"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTarget distribution:\n", y.value_counts(normalize=True))

# ==============================================================
#  Model Definitions
# ==============================================================

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    max_features="sqrt",
    min_samples_leaf=4,
    class_weight="balanced",
    n_jobs=-1,
)

gb = GradientBoostingClassifier(
    n_estimators=234,
    learning_rate=0.0715,
    max_depth=2,
    min_samples_split=8,
    min_samples_leaf=4,
    subsample=0.9728,
)

xgb = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=10,
    eval_metric="auc",
    n_jobs=-1,
)

# ==============================================================
#  Model Evaluation
# ==============================================================

auc_rf = evaluate_model(rf, X_train, y_train, X_test, y_test, "Random Forest")
auc_gb = evaluate_model(gb, X_train, y_train, X_test, y_test, "Gradient Boosting")
auc_xgb = evaluate_model(xgb, X_train, y_train, X_test, y_test, "XGBoost")

# ==============================================================
#  Feature Importance
# ==============================================================

feature_importances = pd.DataFrame({
    "Feature": X.columns,
    "RandomForest": rf.feature_importances_,
    "GradientBoosting": gb.feature_importances_,
    "XGBoost": xgb.feature_importances_
}).set_index("Feature")

feature_importances.plot(kind="bar", figsize=(12, 6))
plt.title("Feature Importance Comparison (Unclean AI4I-PMDI)")
plt.tight_layout()
plt.show()
