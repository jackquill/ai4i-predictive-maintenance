from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from src.utils import evaluate_model, plot_feature_importance, save_metrics

    # Train and evaluate models on one dataset.
def train_on_dataset(df, dataset_name):

    # removes id from feature list and separates target label
    X = df.drop(columns=["machine_failure", "product_id", "product id"], errors='ignore')
    y = df["machine_failure"]

    # this print helped to find that the unclean datset has slightly more faulures than clean
    # print(f"\n[{dataset_name}] Target distribution:\n{y.value_counts()}\n")

    #  Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define models
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=15, class_weight="balanced", n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=234, learning_rate=0.0715, max_depth=2,
            min_samples_split=8, min_samples_leaf=4, subsample=0.9728
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=10,
            eval_metric="auc", n_jobs=-1
        ),
    }

    results = []
    for name, model in models.items():
        #auc = evaluate_model(model, X_train, y_train, X_test, y_test, name)
        auc = evaluate_model(model, X_train, y_train, X_test, y_test, name, dataset_name)

        results.append({"Dataset": dataset_name, "Model": name, "ROC_AUC": auc})

    save_metrics(results, f"metrics_{dataset_name}.csv")
    plot_feature_importance(models, X, dataset_name)