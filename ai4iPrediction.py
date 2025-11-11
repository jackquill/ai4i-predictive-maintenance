# ==============================================================
#  ai4iPrediction.py
#  Predictive Maintenance Model - AI4I 2020 Dataset
#
#  Description:
#       Feature enginnering has been performed based on the datasets documentation. 
#       Machine learning models such as Random Forest, Gradient Boosting, and XGBoost are trained to predict machine failures.
#       The feature importance of each model is visualized for comparison. 
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
    # train a model and print evaluation metrics 
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    auc_score = roc_auc_score(y_test, probs)
    print(f"\n=== {name} ===")
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))
    print(f"ROC-AUC: {auc_score:.4f}")

    return auc_score
# ==============================================================
# Load and Clean Data
# ==============================================================
ai4i_data = pd.read_csv("data/ai4i2020.csv")

# print(ai4i_data.columns)

# Drop columns that give away the cause of the failure
ai4i_data = ai4i_data.drop(
    columns=['UDI', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'],
    errors='ignore'
)
# confirm the data shape
# print(ai4i_data.shape)

# copy to a dataframe to work with the data
df = ai4i_data.copy()

# ==============================================================
#  Feature Engineering
# =============================================================

# Feature engineering variables based on documentation provided with datset
# 1) Temperature difference: how much hotter the process is than the air
df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]

# 2) Convert rotational speed (rpm) to angular speed (radians per second)
df["omega_rad_s"] = (df["Rotational speed [rpm]"] * 2 * np.pi) / 60

# 3) Compute mechanical power = torque × angular speed
df["power"] = df["Torque [Nm]"] * df["omega_rad_s"]

# 4) Overstrain Failure may happen when wear and torque are both high
df["wear_and_torque"] = df["Tool wear [min]"] * df["Torque [Nm]"]

# 5) looks at column type and crates true or false columns for each type
df = pd.get_dummies(df, columns=["Type"], prefix="quality")

# clean column names so xgboost does not have issues reading varible names
df.columns = (
    df.columns
    .str.replace('[', '', regex=False)
    .str.replace(']', '', regex=False)
    .str.replace('<', '', regex=False)
    .str.replace(' ', '_')
)

#  check to modifued dataframe
print(df.head())

# ==============================================================
#  Train/Test Split
# ==============================================================

# separate features and target variable
X = df.drop(columns=["Machine_failure", "Product_ID"], errors='ignore')
y = df["Machine_failure"]

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# see the class imbalance
# print(y.value_counts(normalize=True))

# ==============================================================
#  Model Definitions
# ==============================================================
# used a grid search to find optimal hyperparameters for the random forest model

rf = RandomForestClassifier(
    n_estimators=200,         # optimal number of trees
    max_depth=15,             # limits tree depth for better generalization
    max_features='sqrt',      # optimal number of features per split
    min_samples_leaf=4,       # prevents overfitting on very small leaf nodes
    min_samples_split=2,      # default split threshold, kept optimal
    class_weight='balanced',  # handle class imbalance
    n_jobs=-1,                # use all CPU cores
)

# gradient boost
gb = GradientBoostingClassifier(
    n_estimators=234,
    learning_rate=0.0715,
    max_depth=2,
    min_samples_split=8,
    min_samples_leaf=4,
    subsample=0.9728,
)

# xgboost
# experinented with tuning hyperparameters using grid search
xgb = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=10, 
    eval_metric='auc',
    n_jobs=-1
)

# ==============================================================
#  Model Evaluation
# ==============================================================
auc_rf = evaluate_model(rf, X_train, y_train, X_test, y_test, "Random Forest")
auc_gb = evaluate_model(gb, X_train, y_train, X_test, y_test, "Gradient Boosting")
auc_xgb = evaluate_model(xgb, X_train, y_train, X_test, y_test, "XGBoost")


# === Feature Importance Visualization ===
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'RandomForest': rf.feature_importances_,
    'GradientBoosting': gb.feature_importances_,
    'XGBoost': xgb.feature_importances_
}).set_index('Feature')

feature_importances.plot(kind='bar', figsize=(12,6))
plt.title("Feature Importance Comparison")
plt.tight_layout()
plt.show()