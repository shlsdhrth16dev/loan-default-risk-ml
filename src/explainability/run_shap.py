import pickle
import pandas as pd
import shap
import os
import sys

# Add project root to path to allow importing src
sys.path.append(os.getcwd())

from src.features.feature_pipeline import build_features

def main():
    # Load model
    model_path = "reports/trained_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please run training first.")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load and preprocess data
    data_path = "data/raw/loan_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}")

    print("Loading and processing data for explanation...")
    df = pd.read_csv(data_path)
    
    # We must apply the same feature engineering as during training
    df_processed = build_features(df)
    
    # Drop target if present
    target = "loan_status"
    if target in df_processed.columns:
        X = df_processed.drop(columns=[target])
    else:
        X = df_processed

    # Create explainer
    print("Generating SHAP values (this may take a while)...")
    # For tree based models, TreeExplainer is much faster and exact
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Save summary plot (optional but useful)
    import matplotlib.pyplot as plt
    os.makedirs("reports/figures", exist_ok=True)
    
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig("reports/figures/shap_summary.png")
    print("Saved SHAP summary plot to reports/figures/shap_summary.png")

    print("✅ SHAP explanations generated")

if __name__ == "__main__":
    main()
