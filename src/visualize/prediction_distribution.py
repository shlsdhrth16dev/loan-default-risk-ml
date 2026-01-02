
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import os
import sys

# Ensure project root is in path
sys.path.append(os.getcwd())

from src.features.feature_pipeline import build_features

def plot_prediction_distribution():
    """
    Plots the distribution of predicted default probabilities for the entire dataset.
    """
    model_path = "reports/trained_model.pkl"
    data_path = "data/raw/loan_data.csv"
    columns_path = "reports/training_columns.pkl"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    if not os.path.exists(data_path):
        print(f"Error: Data not found at {data_path}")
        return

    print("Loading and processing data...")
    df = pd.read_csv(data_path)
    
    # Apply feature engineering
    df_processed = build_features(df)
    
    # Align with training columns (critical for XGBoost)
    if os.path.exists(columns_path):
        with open(columns_path, "rb") as f:
            training_columns = pickle.load(f)
        # Realign columns: this ensures correct order and handles missing/extra columns (e.g. dropping target)
        df_processed = df_processed.reindex(columns=training_columns, fill_value=0)
    else:
        print("Warning: Training columns file not found. Prediction might fail if features mismatch.")
        # Attempt to drop target if it exists and we don't have columns file
        if "loan_status" in df_processed.columns:
            df_processed = df_processed.drop(columns=["loan_status"])

    print("Generating predictions...")
    probs = model.predict_proba(df_processed)[:, 1] 

    plt.figure(figsize=(10, 6))
    plt.hist(probs, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel("Default Probability")
    plt.ylabel("Number of Applicants")
    plt.title("Distribution of Loan Default Risk")
    plt.axvline(x=0.5, color='red', linestyle='--', label='Default Threshold (0.5)')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.show()

if __name__ == "__main__":
    plot_prediction_distribution()
