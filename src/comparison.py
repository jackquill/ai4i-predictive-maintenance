from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd


    # checks that both datasets share identical columns. which they do now.
def align_columns(train_df, test_df):
    common_cols = list(set(train_df.columns) & set(test_df.columns))
    return train_df[common_cols], test_df[common_cols]

    # join the datasets on Product_ID so when we split, we avoid leakage.
def merge_clean_unclean(clean_df, unclean_df):
    for df in [clean_df, unclean_df]:
        df.columns = df.columns.str.lower()
        if "product id" in df.columns:
            df.rename(columns={"product id": "product_id"}, inplace=True)

    merged = pd.merge(
        clean_df,
        unclean_df,
        on="product_id",
        suffixes=("_clean", "_unclean")
    )
    return merged


    # Extract aligned clean and unclean subsets from merged dataframe.
def split_clean_unclean_pairs(merged):
    X_clean = merged.filter(like="_clean").copy()
    X_clean.columns = [c.replace("_clean", "") for c in X_clean.columns]
    y_clean = X_clean.pop("machine_failure")

    X_unclean = merged.filter(like="_unclean").copy()
    X_unclean.columns = [c.replace("_unclean", "") for c in X_unclean.columns]
    y_unclean = X_unclean.pop("machine_failure")

    return X_clean, y_clean, X_unclean, y_unclean

    # Train a model on one dataset and test it on another.
def cross_train_test(model, X_train, y_train, X_test, y_test):
    X_train, X_test = align_columns(X_train, X_test)
    model.fit(X_train, y_train)

    # use a predict_proba for ROC-AUC basically a probability score instead of hard class labels for each sample
    preds = model.predict_proba(X_test)[:, 1]

    return roc_auc_score(y_test, preds)


def cross_dataset_pair_test(model, clean_df, unclean_df):
    """
    Perform paired clean vs unclean cross-dataset testing:
    - Train Clean → Test Unclean
    - Train Unclean → Test Clean
    Uses the same product IDs for train/test splits.
    """
    merged = merge_clean_unclean(clean_df, unclean_df)
    X_clean, y_clean, X_unclean, y_unclean = split_clean_unclean_pairs(merged)

    # Split based on product IDs not rows to avoid overlap and data leakage
    ids = merged["product_id"].unique()
    train_ids, test_ids = train_test_split(ids, test_size=0.2, random_state=42)

    X_clean_train = X_clean[merged["product_id"].isin(train_ids)]
    y_clean_train = y_clean[merged["product_id"].isin(train_ids)]
    X_unclean_train = X_unclean[merged["product_id"].isin(train_ids)]
    y_unclean_train = y_unclean[merged["product_id"].isin(train_ids)]

    X_clean_test = X_clean[merged["product_id"].isin(test_ids)]
    y_clean_test = y_clean[merged["product_id"].isin(test_ids)]
    X_unclean_test = X_unclean[merged["product_id"].isin(test_ids)]
    y_unclean_test = y_unclean[merged["product_id"].isin(test_ids)]

    # Cross-dataset AUCs 
    auc_clean_to_unclean = cross_train_test(
        model, X_clean_train, y_clean_train, X_unclean_test, y_unclean_test
    )

    auc_unclean_to_clean = cross_train_test(
        model, X_unclean_train, y_unclean_train, X_clean_test, y_clean_test
    )

    # Print cross-dataset results 
    print(f"\nTrain Clean -> Test Unclean: ROC-AUC = {auc_clean_to_unclean:.4f}")
    print(f"Train Unclean -> Test Clean: ROC-AUC = {auc_unclean_to_clean:.4f}")

    return pd.DataFrame({
        "Experiment": [
            "Train Clean -> Test Unclean (Paired Split)",
            "Train Unclean -> Test Clean (Paired Split)"
        ],
        "ROC_AUC": [auc_clean_to_unclean, auc_unclean_to_clean]
    })
