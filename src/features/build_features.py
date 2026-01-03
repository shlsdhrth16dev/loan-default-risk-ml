import pandas as pd
import numpy as np
import os
from feature_config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COL

INPUT_PATH = os.path.join("data", "processed", "clean_data.csv")
OUTPUT_PATH = os.path.join("data", "processed", "features.csv")


def load_data(path):
    """Load cleaned data from CSV."""
    return pd.read_csv(path)


def create_features(df):
    """
    Create engineered features from the base dataset.
    
    Features include:
    - Loan-to-income ratio
    - Credit utilization proxy
    - Employment stability score
    - Log transforms for skewed distributions
    - Interaction features
    """
    # ---------- Ratio Features ----------
    # Loan to income ratio (handle division by zero)
    df["loan_to_income_ratio"] = df["loanamount"] / (df["income"] + 1)
    
    # DTI ratio is already provided, create a binary high/low indicator
    df["high_dti"] = (df["dtiratio"] > df["dtiratio"].median()).astype(int)
    
    # ---------- Credit Features ----------
    # Credit utilization proxy: average debt per credit line
    df["avg_debt_per_line"] = df["loanamount"] / (df["numcreditlines"] + 1)
    
    # Credit score bins
    df["credit_score_bin"] = pd.cut(
        df["creditscore"],
        bins=[0, 580, 670, 740, 850],
        labels=["poor", "fair", "good", "excellent"]
    )
    
    # ---------- Employment Stability ----------
    # Use months employed directly as a stability metric
    df["employment_stability"] = df["monthsemployed"].fillna(0)
    
    # Employment quality: high income + long employment
    df["stable_high_earner"] = (
        (df["income"] > df["income"].median()) & 
        (df["monthsemployed"] > df["monthsemployed"].median())
    ).astype(int)
    
    # ---------- Age-based Features ----------
    # Age bins
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 50, 100],
        labels=["young", "mid", "mature", "senior"]
    )
    
    # Credit history per year of age (proxy)
    df["credit_score_per_age"] = df["creditscore"] / (df["age"] + 1)
    
    # ---------- Log Transforms for Skewed Variables ----------
    # Use numpy log instead of deprecated pd.np
    df["log_income"] = np.log1p(df["income"])  # log1p handles log(1+x) safely
    df["log_loan_amount"] = np.log1p(df["loanamount"])
    
    # ---------- Interaction Features ----------
    # Interest rate impact: higher rate on larger loans is riskier
    df["rate_times_amount"] = df["interestrate"] * df["loanamount"] / 10000
    
    # Risky loan indicator: high interest + high DTI
    df["risky_loan"] = (
        (df["interestrate"] > df["interestrate"].median()) &
        (df["dtiratio"] > df["dtiratio"].median())
    ).astype(int)
    
    return df


def select_features(df):
    """Select base features and engineered features for modeling."""
    engineered_features = [
        "loan_to_income_ratio",
        "high_dti",
        "avg_debt_per_line",
        "credit_score_bin",
        "employment_stability",
        "stable_high_earner",
        "age_group",
        "credit_score_per_age",
        "log_income",
        "log_loan_amount",
        "rate_times_amount",
        "risky_loan",
    ]
    
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES + engineered_features
    
    return df[all_features + [TARGET_COL]]


def save_features(df, path):
    """Save feature dataset to CSV."""
    df.to_csv(path, index=False)
    print(f"✓ Saved {len(df)} rows and {len(df.columns)} columns")


if __name__ == "__main__":
    print("Loading cleaned data...")
    df = load_data(INPUT_PATH)
    print(f"✓ Loaded {len(df)} rows")
    
    print("\nCreating features...")
    df = create_features(df)
    print(f"✓ Created {len(df.columns)} total columns")
    
    print("\nSelecting features for modeling...")
    df = select_features(df)
    
    print("\nSaving feature dataset...")
    save_features(df, OUTPUT_PATH)
    
    print(f"\n✓ Feature dataset saved to {OUTPUT_PATH}")
    print(f"Features ready for model training!")
