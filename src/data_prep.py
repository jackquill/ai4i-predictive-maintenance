# ==============================================================
#  data_prep.py
#  Data loading and feature engineering for AI4I datasets
#
#  Handles both clean (AI4I 2020) and irregular (AI4I-PMDI) datasets.
#  Automatically standardizes column names, encodes categorical data,
#  adds engineered features, and ensures all columns are numeric.
# ==============================================================

import pandas as pd
import numpy as np


    # Loads the dataset, cleans it, and performs feature engineering.
def load_and_engineer_data(path: str) -> pd.DataFrame:

    # Load data 
    df = pd.read_csv(path)

    # Normalize column names by making everying lowcase and removing whitespace
    df.columns = [col.strip().lower() for col in df.columns]

    # Drop ID. (product id is kept for merging later)
    for col in ["udi"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Drop known cause-of-failure columns presnest in clean dataset 
    leak_cols = ["twf", "hdf", "pwf", "osf", "rnf"]
    df.drop(columns=[c for c in leak_cols if c in df.columns], inplace=True, errors="ignore")

    # rename columns in uncleaned dataset to match clean dataset
    rename_map = {
        'air temperature (k)': 'air temperature [k]',
        'process temperature (k)': 'process temperature [k]',
        'rotational speed (rpm)': 'rotational speed [rpm]',
        'torque (nm)': 'torque [nm]',
        'tool wear (min)': 'tool wear [min]',
        'product id': 'product_id'
    }
    df.rename(columns=rename_map, inplace=True)


    # Fill missing data in the numeric columns in unclean datset with their mean
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())

    # ==============================================================
    #  Feature Engineering
    # ==============================================================

    df["temp_diff"] = df["process temperature [k]"] - df["air temperature [k]"]

    df["omega_rad_s"] = (df["rotational speed [rpm]"] * 2 * np.pi) / 60

    df["power"] = df["torque [nm]"] * df["omega_rad_s"]

    df["wear_and_torque"] = df["tool wear [min]"] * df["torque [nm]"]

    # ==============================================================
    #  Handle categorical columns
    # ==============================================================

    # Encode 'type' (L, M, H) 
    df = pd.get_dummies(df, columns=["type"], prefix="quality", drop_first=False)

    # --- Encode 'control' column (A/B/C), case-insensitive ---
    if "control" in df.columns:
        df = pd.get_dummies(df, columns=["control"], prefix="control", drop_first=False)
    elif "control type" in df.columns:
        df = pd.get_dummies(df, columns=["control type"], prefix="control", drop_first=False)
    else:
        # Ensure feature alignment for datasets without control
        for val in ["A", "B", "C"]:
            df[f"control_{val}"] = 0

    # ==============================================================
    #  Handle target column
    # ==============================================================

    if "diagnostic" in df.columns:
        df["diagnostic"] = df["diagnostic"].str.lower()

        # 'no failure' is 0, everything else (failures) is 1
        df["machine_failure"] = df["diagnostic"].apply(
            lambda x: 0 if x == "no failure" else 1
        )

        #once we have created the target column we can drop the original diagnostic column
        df.drop(columns=["diagnostic"], inplace=True)
    # ==============================================================
    #  Drop non-numeric columns like timestamps
    # ==============================================================

    if "date" in df.columns:
        df.drop(columns=["date"], inplace=True)

    # ==============================================================
    #  Final cleaning
    # ==============================================================

    # Convert all numeric-looking columns to numeric
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass

    # Clean column names for compatibility with models (XGBoost)
    df.columns = (
        df.columns
        .str.replace('[', '', regex=False)
        .str.replace(']', '', regex=False)
        .str.replace('<', '', regex=False)
        .str.replace(' ', '_')
    )

    # print(f"Final shape: {df.shape}")
    return df
