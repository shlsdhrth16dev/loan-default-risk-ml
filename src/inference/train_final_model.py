"""
Train final production model using ALL data with best hyperparameters.

This script:
- Loads optimized hyperparameters from tuning results
- Trains on FULL dataset (no train/test split for max performance)
- Applies proper class weight handling
- Saves model with comprehensive metadata
"""
import pandas as pd
import numpy as np
import os
import sys
import json
import pickle
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'features'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from feature_config import TARGET_COL
from model_utils import load_data, calculate_class_weight

# Paths
TUNING_RESULTS_PATH = os.path.join("models", "tuning", "tuning_results.json")
MODEL_DIR = os.path.join("models", "production")
MODEL_PATH = os.path.join(MODEL_DIR, "loan_default_model_final.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")


def load_best_hyperparameters():
    """
    Load best hyperparameters from tuning results.
    
    Returns:
        Dictionary of best parameters
    """
    if not os.path.exists(TUNING_RESULTS_PATH):
        print(f"⚠ Warning: Tuning results not found at {TUNING_RESULTS_PATH}")
        print("Using default optimized parameters")
        return {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 7,
            'gamma': 0.3,
            'reg_alpha': 7.3,
            'reg_lambda': 8.7
        }
    
    with open(TUNING_RESULTS_PATH, 'r') as f:
        results = json.load(f)
    
    params = results['best_params']
    print(f"✓ Loaded tuned hyperparameters from {TUNING_RESULTS_PATH}")
    return params


def build_production_pipeline(X, scale_pos_weight, best_params):
    """
    Build final production pipeline with best hyperparameters.
    
    Args:
        X: Feature dataframe
        scale_pos_weight: Class weight for imbalance
        best_params: Best hyperparameters from tuning
    
    Returns:
        Sklearn Pipeline
    """
    # Identify column types
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    print(f"\nPipeline configuration:")
    print(f"  Numeric features: {len(num_cols)}")
    print(f"  Categorical features: {len(cat_cols)}")
    
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    
    # XGBoost with tuned parameters + class weight
    model_params = {
        **best_params,
        'scale_pos_weight': scale_pos_weight,
        'objective': "binary:logistic",
        'eval_metric': "auc",
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    model = XGBClassifier(**model_params)
    
    # Create pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    
    return pipeline


def save_model_with_metadata(pipeline, metadata, model_path, metadata_path):
    """
    Save model and metadata for production deployment.
    
    Args:
        pipeline: Trained pipeline
        metadata: Model metadata dictionary
        model_path: Path to save model
        metadata_path: Path to save metadata
    """
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"✓ Model saved to {model_path}")
    
    # Save metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to {metadata_path}")


if __name__ == "__main__":
    print("="*70)
    print("TRAINING FINAL PRODUCTION MODEL")
    print("="*70)
    
    # Load ALL data (no split for final model)
    print("\n[1/5] Loading full dataset...")
    df = load_data()
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    print(f"✓ Loaded {len(df)} samples for training")
    print(f"  Features: {len(X.columns)}")
    print(f"  Target: {TARGET_COL}")
    
    # Calculate class weight
    print("\n[2/5] Analyzing class distribution...")
    scale_pos_weight = calculate_class_weight(y)
    
    # Load best hyperparameters
    print("\n[3/5] Loading optimized hyperparameters...")
    best_params = load_best_hyperparameters()
    print(f"\nHyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Build pipeline
    print("\n[4/5] Building production pipeline...")
    pipeline = build_production_pipeline(X, scale_pos_weight, best_params)
    
    # Train on FULL dataset
    print("\n[5/5] Training on full dataset...")
    print("This may take a few minutes...")
    pipeline.fit(X, y)
    print("✓ Training complete!")
    
    # Create metadata
    metadata = {
        'model_type': 'XGBoost',
        'training_date': datetime.now().isoformat(),
        'n_samples': len(df),
        'n_features': len(X.columns),
        'feature_names': X.columns.tolist(),
        'target_col': TARGET_COL,
        'class_distribution': {
            'negative': int((y == 0).sum()),
            'positive': int((y == 1).sum())
        },
        'scale_pos_weight': float(scale_pos_weight),
        'hyperparameters': best_params,
        'sklearn_version': __import__('sklearn').__version__,
        'xgboost_version': __import__('xgboost').__version__,
        'python_version': sys.version,
        'model_path': MODEL_PATH,
        'recommended_threshold': 0.614,  # F1-optimized from tuning
        'notes': 'Production model trained on full dataset with tuned hyperparameters'
    }
    
    # Save model and metadata
    print("\nSaving model and metadata...")
    save_model_with_metadata(pipeline, metadata, MODEL_PATH, METADATA_PATH)
    
    # Summary
    print("\n" + "="*70)
    print("✓ FINAL MODEL TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel Details:")
    print(f"  Trained on: {len(df):,} samples (FULL dataset)")
    print(f"  Features: {len(X.columns)}")
    print(f"  Class weight: {scale_pos_weight:.2f}")
    print(f"  Recommended threshold: {metadata['recommended_threshold']}")
    print(f"\nFiles saved:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Metadata: {METADATA_PATH}")
    print(f"\nReady for production deployment! 🚀")
