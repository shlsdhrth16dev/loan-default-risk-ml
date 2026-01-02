import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering pipeline for loan default risk.
    
    Args:
        df (pd.DataFrame): Raw dataframe containing loan data.
        
    Returns:
        pd.DataFrame: Processed dataframe with new features and encoded categoricals.
    """
    # Work on a copy to avoid SettingWithCopy warnings
    df = df.copy()
    
    # Handle missing values
    df.fillna(0, inplace=True)

    # create interaction features
    df["income_to_loan_ratio"] = df["person_income"] / (df["loan_amnt"] + 1)
    df["employment_stability"] = df["person_emp_length"] / (df["person_age"] + 1)
    df["interest_burden"] = df["loan_int_rate"] * df["loan_percent_income"]

    # One-hot encode categorical variables
    categorical_cols = [
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file"
    ]

    # Check if columns exist before encoding to be robust
    existing_cat_cols = [col for col in categorical_cols if col in df.columns]
    
    df = pd.get_dummies(df, columns=existing_cat_cols, drop_first=True)
    
    return df
