"""
Permutation Importance Analysis - SHAP Alternative.

Provides model interpretability using sklearn's permutation_importance:
- Calculate feature importance by permuting feature values
- Works with any model (no dependency issues)
- Compare with model's built-in importance
- Generate comprehensive visualizations
"""
import pandas as pd
import numpy as np
import pickle
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
import sys

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'features'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from feature_config import TARGET_COL
from model_utils import load_data, split_data

# Paths
MODEL_PATH = os.path.join("models", "production", "loan_default_model_final.pkl")
IMPORTANCE_DIR = os.path.join("reports", "importance")
RESULTS_PATH = os.path.join(IMPORTANCE_DIR, "permutation_importance.json")


def load_model(model_path):
    """Load trained model."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def get_feature_names(pipeline, X):
    """Get feature names after preprocessing."""
    preprocessor = pipeline.named_steps['preprocessor']
    
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    if cat_cols:
        cat_encoder = preprocessor.named_transformers_['cat']
        cat_feature_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
        feature_names = num_cols + cat_feature_names
    else:
        feature_names = num_cols
    
    return feature_names


def calculate_permutation_importance(model, X, y, n_repeats=10, random_state=42):
    """
    Calculate permutation importance.
    
    Args:
        model: Trained model pipeline
        X: Features
        y: Target
        n_repeats: Number of times to permute each feature
        random_state: Random seed
    
    Returns:
        Permutation importance result
    """
    print(f"\nCalculating permutation importance ({n_repeats} repeats)...")
    print("This may take a few minutes...")
    
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    print("✓ Permutation importance calculated")
    return result


def plot_permutation_importance(
    feature_names, 
    importances_mean, 
    importances_std,
    top_n=20,
    save_path=None
):
    """
    Plot permutation importance as bar chart.
    
    Args:
        feature_names: List of feature names
        importances_mean: Mean importance values
        importances_std: Std of importance values
        top_n: Number of top features to show
        save_path: Path to save plot
    """
    # Sort by importance
    indices = np.argsort(importances_mean)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.barh(
        range(len(indices)),
        importances_mean[indices][::-1],
        xerr=importances_std[indices][::-1],
        alpha=0.7
    )
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices][::-1])
    plt.xlabel('Permutation Importance (ROC-AUC decrease)')
    plt.title(f'Top {top_n} Features by Permutation Importance')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Bar plot saved to {save_path}")
    
    plt.close()


def plot_importance_boxplot(
    feature_names,
    importances,
    top_n=20,
    save_path=None
):
    """
    Plot importance distribution as box plot.
    
    Args:
        feature_names: List of feature names
        importances: Array of importance values (n_repeats x n_features)
        top_n: Number of top features
        save_path: Path to save plot
    """
    # Get top features by mean
    importances_mean = importances.mean(axis=0)
    indices = np.argsort(importances_mean)[::-1][:top_n]
    
    # Prepare data
    plot_data = []
    for idx in indices:
        for val in importances[:, idx]:
            plot_data.append({
                'feature': feature_names[idx],
                'importance': val
            })
    
    df_plot = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(
        data=df_plot,
        y='feature',
        x='importance',
        order=[feature_names[i] for i in indices]
    )
    plt.xlabel('Permutation Importance')
    plt.title(f'Importance Distribution (Top {top_n} Features)')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Box plot saved to {save_path}")
    
    plt.close()


def compare_with_model_importance(
    feature_names,
    perm_importance_mean,
    model,
    save_path=None
):
    """
    Compare permutation importance with model's built-in importance.
    
    Args:
        feature_names: List of feature names
        perm_importance_mean: Permutation importance values
        model: Trained model pipeline
        save_path: Path to save plot
    """
    try:
        # Get XGBoost feature importance
        xgb_model = model.named_steps['model']
        model_importance = xgb_model.feature_importances_
        
        # Get top 20 features by permutation importance
        top_indices = np.argsort(perm_importance_mean)[::-1][:20]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Permutation importance
        ax1.barh(
            range(len(top_indices)),
            perm_importance_mean[top_indices][::-1],
            alpha=0.7,
            color='steelblue'
        )
        ax1.set_yticks(range(len(top_indices)))
        ax1.set_yticklabels([feature_names[i] for i in top_indices][::-1])
        ax1.set_xlabel('Permutation Importance')
        ax1.set_title('Permutation Importance (Top 20)')
        ax1.grid(axis='x', alpha=0.3)
        
        # Model importance
        ax2.barh(
            range(len(top_indices)),
            model_importance[top_indices][::-1],
            alpha=0.7,
            color='coral'
        )
        ax2.set_yticks(range(len(top_indices)))
        ax2.set_yticklabels([feature_names[i] for i in top_indices][::-1])
        ax2.set_xlabel('XGBoost Feature Importance')
        ax2.set_title('XGBoost Built-in Importance (Top 20)')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Comparison plot saved to {save_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"⚠ Could not create comparison plot: {e}")


if __name__ == "__main__":
    print("="*70)
    print("PERMUTATION IMPORTANCE ANALYSIS")
    print("="*70)
    
    # Load data
    print("\n[1/5] Loading data...")
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Sample for faster computation (use 5000 samples)
    sample_size = min(5000, len(X_test))
    X_sample = X_test.iloc[:sample_size]
    y_sample = y_test.iloc[:sample_size]
    print(f"✓ Using {sample_size} samples for analysis")
    
    # Load model
    print("\n[2/5] Loading model...")
    model = load_model(MODEL_PATH)
    
    # Get feature names
    feature_names = get_feature_names(model, X_sample)
    print(f"✓ Model loaded with {len(feature_names)} features")
    
    # Calculate permutation importance
    print("\n[3/5] Calculating permutation importance...")
    perm_result = calculate_permutation_importance(
        model, X_sample, y_sample,
        n_repeats=10,
        random_state=42
    )
    
    # Extract results
    importances_mean = perm_result.importances_mean
    importances_std = perm_result.importances_std
    importances = perm_result.importances
    
    # Print top 10 features
    print("\nTop 10 Most Important Features:")
    indices = np.argsort(importances_mean)[::-1][:10]
    for rank, idx in enumerate(indices, 1):
        print(f"  {rank}. {feature_names[idx]}: {importances_mean[idx]:.4f} (+/- {importances_std[idx]:.4f})")
    
    # Generate visualizations
    print("\n[4/5] Generating visualizations...")
    os.makedirs(IMPORTANCE_DIR, exist_ok=True)
    
    plot_permutation_importance(
        feature_names,
        importances_mean,
        importances_std,
        top_n=20,
        save_path=os.path.join(IMPORTANCE_DIR, "permutation_importance_bar.png")
    )
    
    plot_importance_boxplot(
        feature_names,
        importances,
        top_n=20,
        save_path=os.path.join(IMPORTANCE_DIR, "permutation_importance_boxplot.png")
    )
    
    compare_with_model_importance(
        feature_names,
        importances_mean,
        model,
        save_path=os.path.join(IMPORTANCE_DIR, "importance_comparison.png")
    )
    
    # Save results
    print("\n[5/5] Saving results...")
    results = {
        'feature_importance': {
            feature_names[i]: {
                'mean': float(importances_mean[i]),
                'std': float(importances_std[i])
            }
            for i in range(len(feature_names))
        },
        'top_20_features': [
            {
                'rank': rank,
                'feature': feature_names[idx],
                'importance_mean': float(importances_mean[idx]),
                'importance_std': float(importances_std[idx])
            }
            for rank, idx in enumerate(np.argsort(importances_mean)[::-1][:20], 1)
        ],
        'n_samples': sample_size,
        'n_repeats': 10
    }
    
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {RESULTS_PATH}")
    
    print("\n" + "="*70)
    print("✓ PERMUTATION IMPORTANCE ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nAll visualizations saved to: {IMPORTANCE_DIR}/")
    print("\nKey Findings:")
    print(f"  - Top feature: {feature_names[indices[0]]} (importance: {importances_mean[indices[0]]:.4f})")
    print(f"  - Analysis based on {sample_size} samples with {10} repeats")
