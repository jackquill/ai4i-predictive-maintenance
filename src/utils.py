import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
import os

    # Train, evaluate, and print metrics.
def evaluate_model(model, X_train, y_train, X_test, y_test, name="Model", dataset_name=""):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probs)
    print(f"\n=== {name} ===")

    cm = confusion_matrix(y_test, preds)
    print(cm)

    # Save confusion matrix PDF
    save_confusion_matrix(cm, name, dataset_name)

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


def save_confusion_matrix(cm, model_name, dataset_name):
    """Save confusion matrix as PDF inside results/confusion_matrices/"""
    os.makedirs("results/confusion_matrices", exist_ok=True)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix — {model_name} ({dataset_name})")

    filename = f"results/confusion_matrices/{dataset_name}_{model_name}_cm.pdf"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
