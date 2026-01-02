import pickle
import pandas as pd
import os
import sys

# Add project root to path to allow importing src
sys.path.append(os.getcwd())

from src.features.feature_pipeline import build_features
from src.models.train import train_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

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

    # STEP 1.2: Strict Train/Test Split (20% held out)
    # This test set will NOT be seen by the training process at all
    print("Splitting data into Training and Held-Out Test sets...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # STEP 3: Handle Class Imbalance
    # Calculate ratio of negative to positive samples
    num_neg = (y_train_full == 0).sum()
    num_pos = (y_train_full == 1).sum()
    scale_pos_weight = num_neg / num_pos
    print(f"Class Imbalance Detected: {num_neg} Negatives, {num_pos} Positives")
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    # Train model
    print("Training XGBoost model...")
    # Passing the calculated weight to the training function
    model = train_model(X_train_full, y_train_full, scale_pos_weight=scale_pos_weight)

    # STEP 1.1: Real Evaluation on Held-Out Test Set
    print("\n" + "="*30)
    print("FINAL MODEL EVALUATION (HELD-OUT TEST SET)")
    print("="*30)
    
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:, 1]
    
    roc_test = roc_auc_score(y_test, y_prob_test)
    print(f"Test ROC-AUC: {roc_test:.4f}")
    print("\nTest Classification Report:")
    report_str = classification_report(y_test, y_pred_test)
    print(report_str)

    # Save metrics to file
    metrics_path = os.path.join("reports", "evaluation_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Test ROC-AUC: {roc_test:.4f}\n\n")
        f.write("Test Classification Report:\n")
        f.write(report_str)
    print(f"✅ Metrics saved to {metrics_path}")

    # Save model and columns
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    output_path = os.path.join(output_dir, "trained_model.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
        
    # Save feature names/columns
    columns_path = os.path.join(output_dir, "training_columns.pkl")
    with open(columns_path, "wb") as f:
        pickle.dump(X.columns.tolist(), f)

    print(f"✅ Model trained and saved to {output_path}")
    print(f"✅ Training columns saved to {columns_path}")

if __name__ == "__main__":
    main()
