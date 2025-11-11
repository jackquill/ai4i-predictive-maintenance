from src.data_prep import load_and_engineer_data
from src.train_models import train_on_dataset
from src.comparison import cross_dataset_pair_test 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import pandas as pd

# run using command: python main.py
def main():
    # Load datasets 
    clean_df = load_and_engineer_data("data/ai4i2020.csv")
    irregular_df = load_and_engineer_data("data/AI4I-PMDI.csv")

    # Train on each dataset separately 
    train_on_dataset(clean_df, "clean")
    train_on_dataset(irregular_df, "irregular")

    # Define models to test cross-dataset 
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            class_weight="balanced",
            n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=234,
            learning_rate=0.0715,
            max_depth=2,
            min_samples_split=8,
            min_samples_leaf=4,
            subsample=0.9728
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=10,
            eval_metric="auc",
            n_jobs=-1
        ),
    }

    all_results = []

    #  Run cross-dataset comparison for each model 
    for name, model in models.items():
        print(f"\n=== Cross-Dataset Evaluation: {name} ===")
        results = cross_dataset_pair_test(model, clean_df, irregular_df)
        results["Model"] = name
        all_results.append(results)

    # Combine and save results 
    final_results = pd.concat(all_results, ignore_index=True)
    print("\n=== Cross-Dataset Results (All Models) ===")
    print(final_results)
    final_results.to_csv("results/cross_dataset_metrics.csv", index=False)


if __name__ == "__main__":
    main()