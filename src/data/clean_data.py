import pandas as pd
import os

RAW_DATA_PATH = os.path.join("data", "raw", "loan_default.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "clean_data.csv")


def load_data(path):
    return pd.read_csv(path)


def clean_data(df):
    # standardize column names
    df.columns = df.columns.str.lower()

    # drop duplicates
    df = df.drop_duplicates()

    # identify target column
    target_col = "default"  # CHANGE if needed

    # map categorical target
    if df[target_col].dtype == "object":
        df[target_col] = df[target_col].map({"Y": 1, "N": 0})

    # handle missing values
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("UNKNOWN")

    return df


def save_data(df, path):
    df.to_csv(path, index=False)


if __name__ == "__main__":
    df = load_data(RAW_DATA_PATH)
    df = clean_data(df)
    save_data(df, PROCESSED_DATA_PATH)

    print(f"Cleaned data saved to {PROCESSED_DATA_PATH}")
