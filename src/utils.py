import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

    # Train, evaluate, and print metrics.
def evaluate_model(model, X_train, y_train, X_test, y_test, name="Model"):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probs)
    print(f"\n=== {name} ===")
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))
    print(f"ROC-AUC: {auc:.4f}")
    return auc

    # Compare feature importances across models.
def plot_feature_importance(models, X, dataset_name=""):
    feature_df = pd.DataFrame(
        {name: model.feature_importances_ for name, model in models.items()},
        index=X.columns
    )
    feature_df.plot(kind="bar", figsize=(14, 6))
    plt.title(f"Feature Importance Comparison — {dataset_name.capitalize()} Dataset")
    plt.tight_layout()
    save_path = f"results/feature_importance_{dataset_name.lower()}.png"
    plt.savefig(save_path)
    plt.show()

    # Save results to CSV.
def save_metrics(results, filename):
    df = pd.DataFrame(results)
    df.to_csv(f"results/{filename}", index=False)
