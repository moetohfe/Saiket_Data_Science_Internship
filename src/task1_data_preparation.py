"""
Task 1 - Data Preparation

* Load the dataset and explore its structure
* Find and handle missing values
* Encode categorical variables as numbers
* Split into stratified training and testing sets
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from common import (PROCESSED_DIR, RANDOM_STATE, REPORTS_DIR, TARGET,
                    ensure_dirs, header, load_raw, clean)

# Columns that are Yes/No and become 1/0
BINARY_COLS = ["gender", "SeniorCitizen", "Partner", "Dependents",
               "PhoneService", "PaperlessBilling"]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def explore(raw: pd.DataFrame):
    header("TASK 1: DATA PREPARATION")
    print(f"Rows: {raw.shape[0]:,}   Columns: {raw.shape[1]}")
    print(f"Duplicate customer IDs: {raw['customerID'].duplicated().sum()}")
    print("\nColumn types:")
    print(raw.dtypes.value_counts().to_string())

    blank_total = (raw["TotalCharges"].astype(str).str.strip() == "").sum()
    print(f"\nNull values in any column: {int(raw.isna().sum().sum())}")
    print(f"Blank TotalCharges values (hidden missing data): {blank_total}")
    print("  -> all of them have tenure = 0:",
          bool((raw.loc[raw['TotalCharges'].astype(str).str.strip() == '', 'tenure'] == 0).all()))

    print("\nTarget distribution:")
    counts = raw[TARGET].value_counts()
    for label, n in counts.items():
        print(f"  {label:<4} {n:>5}  ({n / len(raw):.1%})")


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Turn every column into a number so models can use it."""
    df = df.drop(columns=["customerID"]).copy()

    df["gender"] = (df["gender"] == "Male").astype(int)
    for col in BINARY_COLS[1:]:
        df[col] = (df[col] == "Yes").astype(int)
    df[TARGET] = (df[TARGET] == "Yes").astype(int)

    # Remaining text columns have 3+ categories -> one-hot encoding
    multi_cols = df.select_dtypes(exclude="number").columns.tolist()
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True, dtype=int)

    # Clean, model-friendly column names
    df.columns = (df.columns.str.replace(r"[ ()]+", "_", regex=True)
                  .str.replace("-", "_").str.strip("_"))
    return df


def main():
    ensure_dirs()
    raw = load_raw()
    explore(raw)

    df = clean(raw)
    print(f"\nAfter cleaning, missing values: {int(df.isna().sum().sum())}")

    encoded = encode(df)
    print(f"Encoded feature matrix: {encoded.shape[0]:,} rows x {encoded.shape[1] - 1} features")

    X = encoded.drop(columns=[TARGET])
    y = encoded[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    pd.concat([X_train, y_train], axis=1).to_csv(PROCESSED_DIR / "train.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(PROCESSED_DIR / "test.csv", index=False)

    print(f"\nTrain set: {len(X_train):,} rows  (churn rate {y_train.mean():.1%})")
    print(f"Test set:  {len(X_test):,} rows  (churn rate {y_test.mean():.1%})")
    print("Saved data/processed/train.csv and data/processed/test.csv")

    # Data dictionary for reference
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "unique_values": df.nunique(),
        "example": df.iloc[0],
    })
    summary.to_csv(REPORTS_DIR / "task1_data_summary.csv")
    print("Saved outputs/reports/task1_data_summary.csv")


if __name__ == "__main__":
    main()
