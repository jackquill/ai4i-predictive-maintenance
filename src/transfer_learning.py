# ==============================================================
#  transfer_learning.py
#  Sequential Transfer Learning for Synthetic ↔ Realistic AI4I Data
#
#  Uses your existing data loaders + models and performs:
#     Experiment A: Train Clean → Finetune Irregular
#     Experiment B: Train Irregular → Finetune Clean
#
#  Evaluates on both domains to measure transfer improvements.
# ==============================================================

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd

from src.comparison import align_columns

import matplotlib.pyplot as plt
import seaborn as sns

def plot_transfer_learning_results(df, save_path="results/transfer_learning_plot.png"):
    """
    Create a grouped bar chart where:
        - X-axis: TL Experiment (Clean→Irregular eval source, etc.)
        - Bar color: Model (RF, GB, XGB)
        - Y-axis: ROC-AUC (auto-scaled for visibility)
    """

    # Prepare DataFrame for seaborn
    df_plot = df.copy()

    # Give clean readable labels for x-axis
    df_plot["Experiment Label"] = (
        df_plot["TL Experiment"]
        .str.replace("eval source", "Eval Source", regex=False)
        .str.replace("eval target", "Eval Target", regex=False)
    )

    plt.figure(figsize=(14, 8))

    # Create grouped bar chart
    sns.barplot(
        data=df_plot,
        x="Experiment Label",
        y="ROC_AUC",
        hue="Model",
        palette="Set2",
        edgecolor="black"
    )

    # Rotate labels for readability
    plt.xticks(rotation=45, ha="right")

    # Expand y-limits slightly so differences are easier to see
    ymin = df_plot["ROC_AUC"].min()
    ymax = df_plot["ROC_AUC"].max()
    padding = (ymax - ymin) * 0.15
    plt.ylim([max(0.0, ymin - padding), min(1.0, ymax + padding)])

    plt.ylabel("ROC-AUC")
    plt.xlabel("Transfer Learning Experiment")
    plt.title("Transfer Learning Performance Across Models & Directions")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


# --------------------------------------------------------------
#  HELPER — train, then finetune
# --------------------------------------------------------------
def sequential_train(model, X_pre, y_pre, X_finetune, y_finetune):
    """
    Train on X_pre first, then continue fitting (fine-tuning)
    on X_finetune.
    """

    # Ensure both datasets share column alignment
    X_pre, X_finetune = align_columns(X_pre, X_finetune)

    # STEP 1 — pretrain
    model.fit(X_pre, y_pre)

    # STEP 2 — fine-tune
    model.fit(X_finetune, y_finetune)

    return model


# --------------------------------------------------------------
#  Run a full TL experiment (one direction)
# --------------------------------------------------------------
def run_tl_direction(model, name, src_df, tgt_df):
    """
    src_df → pretraining dataset
    tgt_df → fine-tuning dataset

    Returns a DataFrame with evaluation results.
    """

    # Remove product_id from features
    X_src = src_df.drop(columns=["machine_failure", "product_id"], errors="ignore")
    y_src = src_df["machine_failure"]

    X_tgt = tgt_df.drop(columns=["machine_failure", "product_id"], errors="ignore")
    y_tgt = tgt_df["machine_failure"]

    # Same split strategy as your other files
    X_src_train, X_src_test, y_src_train, y_src_test = train_test_split(
        X_src, y_src, test_size=0.2, random_state=42
    )
    X_tgt_train, X_tgt_test, y_tgt_train, y_tgt_test = train_test_split(
        X_tgt, y_tgt, test_size=0.2, random_state=42
    )

    # Sequential TL
    model = sequential_train(model, X_src_train, y_src_train, X_tgt_train, y_tgt_train)

    # Evaluate on both domains
    # (always probability scores because ROC-AUC)
    X_src_test_aligned, X_tgt_test_aligned = align_columns(X_src_test, X_tgt_test)

    auc_src = roc_auc_score(
        y_src_test,
        model.predict_proba(X_src_test_aligned)[:, 1]
    )

    auc_tgt = roc_auc_score(
        y_tgt_test,
        model.predict_proba(X_tgt_test_aligned)[:, 1]
    )

    print(f"\n=== Transfer Learning ({name}) ===")
    print(f"Evaluation on Source Dataset:  ROC-AUC = {auc_src:.4f}")
    print(f"Evaluation on Target Dataset:  ROC-AUC = {auc_tgt:.4f}")

    return pd.DataFrame({
        "TL Experiment": [f"{name} — eval source", f"{name} — eval target"],
        "ROC_AUC": [auc_src, auc_tgt]
    })


# --------------------------------------------------------------
#  Top-level function called from main.py
# --------------------------------------------------------------
def run_transfer_learning(models, clean_df, irregular_df):
    """
    Runs both TL directions for each model:

        1. Clean → Irregular
        2. Irregular → Clean

    Returns combined results.
    """
    all_results = []

    for name, model in models.items():
        print(f"\n============================")
        print(f" Transfer Learning: {name}")
        print(f"============================\n")

        # A: Synthetic → Realistic
        tl_clean_to_irregular = run_tl_direction(
            model, f"{name} Clean→Irregular",
            clean_df, irregular_df
        )
        tl_clean_to_irregular["Model"] = name

        # B: Realistic → Synthetic
        tl_irregular_to_clean = run_tl_direction(
            model, f"{name} Irregular→Clean",
            irregular_df, clean_df
        )
        tl_irregular_to_clean["Model"] = name

        all_results.append(tl_clean_to_irregular)
        all_results.append(tl_irregular_to_clean)

    final = pd.concat(all_results, ignore_index=True)
    final.to_csv("results/transfer_learning_metrics.csv", index=False)
    print("\n=== Transfer Learning Summary ===")
    print(final)

    # Create TL summary plot
    plot_transfer_learning_results(final)

    return final
