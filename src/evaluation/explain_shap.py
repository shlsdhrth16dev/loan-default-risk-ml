"""
SHAP (SHapley Additive exPlanations) for model interpretability.

Features:
- Load pre-trained tuned XGBoost model
- Generate comprehensive SHAP visualizations
- Feature importance analysis
- Individual prediction explanations
"""
import pandas as pd
import numpy as np
import os
import sys
import pickle
import shap
import matplotlib.pyplot as plt

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'features'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from feature_config import TARGET_COL
from model_utils import load_data, split_data

# Paths
TUNED_MODEL_PATH = os.path.join("models", "tuning", "best_xgboost_tuned.pkl")
SHAP_DIR = os.path.join("reports", "shap")


def load_trained_model(model_path):
    """
    Load pre-trained model from file.
    
    Args:
        model_path: Path to saved model
    
    Returns:
        Loaded pipeline
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please run tuning/tuning_xgboost.py first."
        )
    
    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)
    
    print(f"✓ Loaded model from {model_path}")
    return pipeline


def get_feature_names(pipeline, X):
    """
    Get feature names after preprocessing.
    
    Args:
        pipeline: Trained sklearn pipeline
        X: Original feature dataframe
    
    Returns:
        List of feature names
    """
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Get numeric and categorical column names
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    if cat_cols:
        # Get encoded categorical feature names
        cat_encoder = preprocessor.named_transformers_['cat']
        cat_feature_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
        feature_names = num_cols + cat_feature_names
    else:
        feature_names = num_cols
    
    return feature_names


def plot_shap_summary(shap_values, X_transformed, feature_names, save_path=None):
    """
    Generate SHAP summary plot (feature importance).
    
    Args:
        shap_values: SHAP values
        X_transformed: Transformed features
        feature_names: Feature names
        save_path: Path to save plot
    """
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, 
        X_transformed, 
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ SHAP summary plot saved to {save_path}")
    
    plt.close()


def plot_shap_bar(shap_values, feature_names, save_path=None):
    """
    Generate SHAP bar plot (mean absolute SHAP values).
    
    Args:
        shap_values: SHAP values
        feature_names: Feature names
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 8))
    
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Sort by importance
    indices = np.argsort(mean_abs_shap)[::-1][:20]  # Top 20
    
    plt.barh(range(len(indices)), mean_abs_shap[indices][::-1])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices][::-1])
    plt.xlabel('Mean |SHAP value|')
    plt.title('Feature Importance (Top 20)')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ SHAP bar plot saved to {save_path}")
    
    plt.close()


def plot_shap_waterfall(explainer, X_transformed, feature_names, idx=0, save_path=None):
    """
    Generate SHAP waterfall plot for individual prediction.
    
    Args:
        explainer: SHAP explainer
        X_transformed: Transformed features
        feature_names: Feature names
        idx: Index of sample to explain
        save_path: Path to save plot
    """
    # Create explanation object
    explanation = explainer(X_transformed[idx:idx+1])
    explanation.feature_names = feature_names
    
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(explanation[0], show=False)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ SHAP waterfall plot saved to {save_path}")
    
    plt.close()


def plot_shap_dependence(shap_values, X_transformed, feature_names, feature_idx, save_path=None):
    """
    Generate SHAP dependence plot for a feature.
    
    Args:
        shap_values: SHAP values
        X_transformed: Transformed features
        feature_names: Feature names
        feature_idx: Index of feature to plot
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feature_idx,
        shap_values,
        X_transformed,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ SHAP dependence plot saved to {save_path}")
    
    plt.close()


if __name__ == "__main__":
    print("="*70)
    print("SHAP EXPLANATIONS FOR XGBOOST MODEL")
    print("="*70)
    
    # Load data
    print("\n[1/5] Loading data...")
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Load trained model
    print("\n[2/5] Loading pre-trained model...")
    pipeline = load_trained_model(TUNED_MODEL_PATH)
    
    # Extract model and transform data
    print("\n[3/5] Preparing data for SHAP analysis...")
    model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Use a sample of test set for faster SHAP computation
    sample_size = min(1000, len(X_test))
    X_test_sample = X_test.iloc[:sample_size]
    
    X_transformed = preprocessor.transform(X_test_sample)
    feature_names = get_feature_names(pipeline, X_test_sample)
    
    print(f"✓ Analyzing {sample_size} samples with {len(feature_names)} features")
    
    # Compute SHAP values
    print("\n[4/5] Computing SHAP values...")
    print("This may take a few minutes...")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)
    
    print("✓ SHAP values computed")
    
    # Generate visualizations
    print("\n[5/5] Generating SHAP visualizations...")
    os.makedirs(SHAP_DIR, exist_ok=True)
    
    # 1. Summary plot (beeswarm)
    plot_shap_summary(
        shap_values,
        X_transformed,
        feature_names,
        save_path=os.path.join(SHAP_DIR, "shap_summary.png")
    )
    
    # 2. Bar plot (feature importance)
    plot_shap_bar(
        shap_values,
        feature_names,
        save_path=os.path.join(SHAP_DIR, "shap_importance.png")
    )
    
    # 3. Waterfall plot for first prediction (high-risk loan)
    plot_shap_waterfall(
        explainer,
        X_transformed,
        feature_names,
        idx=0,
        save_path=os.path.join(SHAP_DIR, "shap_waterfall_sample1.png")
    )
    
    # 4. Dependence plot for top feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_feature_idx = np.argmax(mean_abs_shap)
    
    plot_shap_dependence(
        shap_values,
        X_transformed,
        feature_names,
        top_feature_idx,
        save_path=os.path.join(SHAP_DIR, f"shap_dependence_{feature_names[top_feature_idx]}.png")
    )
    
    # Print feature importance
    print("\nTop 10 Most Important Features (by mean |SHAP value|):")
    indices = np.argsort(mean_abs_shap)[::-1][:10]
    for rank, idx in enumerate(indices, 1):
        print(f"  {rank}. {feature_names[idx]}: {mean_abs_shap[idx]:.4f}")
    
    print("\n" + "="*70)
    print("✓ SHAP ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nAll visualizations saved to: {SHAP_DIR}/")
    print("\nInterpretation Guide:")
    print("  - Summary plot: Shows feature importance and impact direction")
    print("  - Bar plot: Mean absolute SHAP values (overall importance)")
    print("  - Waterfall plot: Explains individual prediction")
    print("  - Dependence plot: Shows how top feature affects predictions")
