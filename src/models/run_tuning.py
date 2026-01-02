import pandas as pd
import os
import sys
import pickle

# Ensure project root is in path
sys.path.append(os.getcwd())

from src.features.feature_pipeline import build_features
from src.models.train import perform_grid_search
from sklearn.model_selection import train_test_split

def main():
    # Load data
    data_path = "data/raw/loan_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")
        
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Feature engineering
    print("Applying feature pipeline...")
    df = build_features(df)

    # Split features and target
    target = "loan_status"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")
        
    X = df.drop(columns=[target])
    y = df[target]

    # Split data to ensure we tune on training set only
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Calculate scale_pos_weight
    num_neg = (y_train_full == 0).sum()
    num_pos = (y_train_full == 1).sum()
    scale_pos_weight = num_neg / num_pos
    print(f"Using scale_pos_weight: {scale_pos_weight:.2f}")

    # Perform Grid Search
    best_model, best_params = perform_grid_search(X_train_full, y_train_full, scale_pos_weight)

    # Save best parameters
    params_path = "reports/best_params.pkl"
    with open(params_path, "wb") as f:
        pickle.dump(best_params, f)
    
    print(f"✅ Best parameters saved to {params_path}")
    print("Best Parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
