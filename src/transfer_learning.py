# ==============================================================
#  transfer_learning.py
#  Paired Transfer Learning for Clean <-> Unclean AI4I Data
# ==============================================================

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

from src.comparison import (
    align_columns,
    merge_clean_unclean,
    split_clean_unclean_pairs
)


# --------------------------------------------------------------
#  Train on one dataset, then fine-tune on another (simple form)
# --------------------------------------------------------------
def sequential_train(model, X_pre, y_pre, X_ft, y_ft):
    # ensure both datasets share identical columns
    X_pre, X_ft = align_columns(X_pre, X_ft)

    # stage 1 — pretrain
    model.fit(X_pre, y_pre)

    # stage 2 — fine-tune
    model.fit(X_ft, y_ft)

    return model


# --------------------------------------------------------------
#  Build leakage-free paired splits
# --------------------------------------------------------------
def build_paired_splits(clean_df, unclean_df):
    merged = merge_clean_unclean(clean_df, unclean_df)
    X_clean, y_clean, X_unclean, y_unclean = split_clean_unclean_pairs(merged)

    ids = merged["product_id"].unique()

    pretrain_ids, other_ids = train_test_split(
        ids, test_size=0.4, random_state=42
    )
    finetune_ids, test_ids = train_test_split(
        other_ids, test_size=0.5, random_state=42
    )

    def pick(ids_subset, X, y):
        mask = merged["product_id"].isin(ids_subset)
        return X[mask], y[mask]

    # pretraining
    clean_pre, clean_y_pre = pick(pretrain_ids, X_clean, y_clean)
    unclean_pre, unclean_y_pre = pick(pretrain_ids, X_unclean, y_unclean)

    # fine-tuning
    clean_ft, clean_y_ft = pick(finetune_ids, X_clean, y_clean)
    unclean_ft, unclean_y_ft = pick(finetune_ids, X_unclean, y_unclean)

    # testing
    clean_test, clean_y_test = pick(test_ids, X_clean, y_clean)
    unclean_test, unclean_y_test = pick(test_ids, X_unclean, y_unclean)

    return (
        clean_pre, clean_y_pre, clean_ft, clean_y_ft, clean_test, clean_y_test,
        unclean_pre, unclean_y_pre, unclean_ft, unclean_y_ft, unclean_test, unclean_y_test
    )


# --------------------------------------------------------------
#  Run one TL direction (Clean -> Unclean or Unclean -> Clean)
# --------------------------------------------------------------
def run_tl_direction(model, direction, clean_df, unclean_df):
    (
        clean_pre, clean_y_pre, 
        clean_ft, clean_y_ft, 
        clean_test, clean_y_test,
        unclean_pre, unclean_y_pre, 
        unclean_ft, unclean_y_ft,
        unclean_test, unclean_y_test
    ) = build_paired_splits(clean_df, unclean_df)

    # Set the correct ordering depending on direction
    if direction == "Clean→Unclean":
        X_pre, y_pre = clean_pre, clean_y_pre
        X_ft, y_ft = unclean_ft, unclean_y_ft

        X_test_source, y_test_source = clean_test, clean_y_test
        X_test_target, y_test_target = unclean_test, unclean_y_test

    else:  # "Unclean→Clean"
        X_pre, y_pre = unclean_pre, unclean_y_pre
        X_ft, y_ft = clean_ft, clean_y_ft

        X_test_source, y_test_source = unclean_test, unclean_y_test
        X_test_target, y_test_target = clean_test, clean_y_test

    # ---- TRAINING PHASES ----
    model = sequential_train(model, X_pre, y_pre, X_ft, y_ft)

    # ---- ALIGN COLUMNS ----
    X_test_source, X_test_target = align_columns(X_test_source, X_test_target)

    # ---- EVALUATION ----
    auc_source = roc_auc_score(
        y_test_source, model.predict_proba(X_test_source)[:, 1]
    )
    auc_target = roc_auc_score(
        y_test_target, model.predict_proba(X_test_target)[:, 1]
    )

    print(f"\n=== Transfer Learning ({direction}) ===")
    print(f"Eval Source Dataset: ROC-AUC = {auc_source:.4f}")
    print(f"Eval Target Dataset: ROC-AUC = {auc_target:.4f}")

    return pd.DataFrame({
        "Experiment": [
            f"{direction} — eval source",
            f"{direction} — eval target"
        ],
        "ROC_AUC": [auc_source, auc_target]
    })


def plot_tl_results(df, save_path="results/transfer_learning_plot.png"):
    plt.figure(figsize=(12, 6))

    experiments = df["Experiment"].unique()
    models = df["Model"].unique()

    x = range(len(experiments))
    width = 0.12  # narrow bars so multiple models fit

    for i, model in enumerate(models):
        subset = df[df["Model"] == model]["ROC_AUC"].tolist()
        offsets = [p + i * width for p in x]
        plt.bar(offsets, subset, width, label=model)

    ymin = df["ROC_AUC"].min()
    ymax = df["ROC_AUC"].max()

    padding = (ymax - ymin) * 0.05 if ymax > ymin else 0.02

    plt.ylim([max(0, ymin - padding), min(1, ymax + padding)])

    plt.xticks(
        [p + (len(models) - 1) * width / 2 for p in x],
        experiments,
        rotation=45,
        ha="right"
    )

    plt.ylabel("ROC-AUC")
    plt.title("Transfer Learning Summary (Clean ↔ Unclean)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


# --------------------------------------------------------------
#  Top-level TL call from main.py
# --------------------------------------------------------------
def run_transfer_learning(models, clean_df, unclean_df):
    all_results = []

    for name, model in models.items():
        print("\n============================")
        print(f" Transfer Learning: {name}")
        print("============================")

        results_clean_to_unclean = run_tl_direction(
            model, "Clean→Unclean", clean_df, unclean_df
        )
        results_clean_to_unclean["Model"] = name

        results_unclean_to_clean = run_tl_direction(
            model, "Unclean→Clean", clean_df, unclean_df
        )
        results_unclean_to_clean["Model"] = name

        all_results.append(results_clean_to_unclean)
        all_results.append(results_unclean_to_clean)

    final = pd.concat(all_results, ignore_index=True)
    final.to_csv("results/transfer_learning_metrics.csv", index=False)

    print("\n=== Transfer Learning Summary ===")
    print(final)

    plot_tl_results(final)

    return final
