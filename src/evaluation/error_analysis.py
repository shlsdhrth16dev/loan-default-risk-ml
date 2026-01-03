"""
Error Analysis for loan default prediction model.

Features:
- Load pre-trained tuned model
- Use optimized decision threshold
- Analyze false positives and false negatives
- Profile error patterns by feature distributions
"""
import pandas as pd
import numpy as np
import os
import sys
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'features'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from feature_config import TARGET_COL
from model_utils import load_data, split_data

# Paths
TUNED_MODEL_PATH = os.path.join("models", "tuning", "best_xgboost_tuned.pkl")
TUNING_RESULTS_PATH = os.path.join("models", "tuning", "tuning_results.json")
ERROR_DIR = os.path.join("reports", "error_analysis")


def load_trained_model(model_path):
    """Load pre-trained model from file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please run tuning/tuning_xgboost.py first."
        )
    
    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)
    
    print(f"✓ Loaded model from {model_path}")
    return pipeline


def load_optimal_threshold(results_path):
    """Load optimal threshold from tuning results."""
    if not os.path.exists(results_path):
        print(f"⚠ Warning: Tuning results not found at {results_path}")
        print("Using default F1-optimized threshold: 0.614")
        return 0.614
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Get F1-optimized threshold
    threshold = results['threshold_optimizations']['f1']['threshold']
    print(f"✓ Using F1-optimized threshold: {threshold:.3f}")
    
    return threshold


def analyze_error_distribution(errors_df, save_path=None):
    """
    Analyze distribution of errors vs correct predictions.
    
    Args:
        errors_df: DataFrame with predictions and actuals
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Error Analysis: Feature Distributions', fontsize=16)
    
    # Define error types
    errors_df['error_type'] = 'Correct'
    errors_df.loc[(errors_df['actual'] == 1) & (errors_df['predicted'] == 0), 'error_type'] = 'False Negative'
    errors_df.loc[(errors_df['actual'] == 0) & (errors_df['predicted'] == 1), 'error_type'] = 'False Positive'
    
    # Select numeric features to analyze
    numeric_cols = errors_df.select_dtypes(include=['int64', 'float64']).columns
    # Exclude prediction columns
    numeric_cols = [col for col in numeric_cols if col not in ['actual', 'predicted', 'probability']]
    
    # Plot top 4 most important features
    for idx, col in enumerate(numeric_cols[:4]):
        ax = axes[idx // 2, idx % 2]
        
        for error_type in ['Correct', 'False Negative', 'False Positive']:
            subset = errors_df[errors_df['error_type'] == error_type]
            if len(subset) > 0:
                ax.hist(
                    subset[col], 
                    alpha=0.5, 
                    label=error_type, 
                    bins=30
                )
        
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution: {col}')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Error distribution plot saved to {save_path}")
    
    plt.close()


def plot_probability_distribution(errors_df, save_path=None):
    """
    Plot probability distribution by prediction correctness.
    
    Args:
        errors_df: DataFrame with predictions and probabilities
        save_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Probability distribution by error type
    errors_df['error_type'] = 'Correct'
    errors_df.loc[(errors_df['actual'] == 1) & (errors_df['predicted'] == 0), 'error_type'] = 'False Negative'
    errors_df.loc[(errors_df['actual'] == 0) & (errors_df['predicted'] == 1), 'error_type'] = 'False Positive'
    
    for error_type in ['Correct', 'False Negative', 'False Positive']:
        subset = errors_df[errors_df['error_type'] == error_type]
        if len(subset) > 0:
            ax1.hist(
                subset['probability'], 
                alpha=0.6, 
                label=error_type, 
                bins=50
            )
    
    ax1.set_xlabel('Predicted Probability')
    ax1.set_ylabel('Count')
    ax1.set_title('Probability Distribution by Error Type')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Box plot
    sns.boxplot(
        data=errors_df,
        x='error_type',
        y='probability',
        ax=ax2
    )
    ax2.set_title('Probability Distribution by Error Type')
    ax2.set_xlabel('Error Type')
    ax2.set_ylabel('Predicted Probability')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Probability distribution plot saved to {save_path}")
    
    plt.close()


def profile_errors(errors_df):
    """
    Generate statistical profile of errors.
    
    Args:
        errors_df: DataFrame with predictions and actuals
    
    Returns:
        Dictionary with error statistics
    """
    fn_mask = (errors_df['actual'] == 1) & (errors_df['predicted'] == 0)
    fp_mask = (errors_df['actual'] == 0) & (errors_df['predicted'] == 1)
    
    profile = {
        'total_samples': len(errors_df),
        'false_negatives': {
            'count': fn_mask.sum(),
            'percentage': fn_mask.sum() / len(errors_df) * 100,
            'avg_probability': errors_df[fn_mask]['probability'].mean() if fn_mask.any() else 0
        },
        'false_positives': {
            'count': fp_mask.sum(),
            'percentage': fp_mask.sum() / len(errors_df) * 100,
            'avg_probability': errors_df[fp_mask]['probability'].mean() if fp_mask.any() else 0
        }
    }
    
    return profile


if __name__ == "__main__":
    print("="*70)
    print("ERROR ANALYSIS FOR XGBOOST MODEL")
    print("="*70)
    
    # Load data
    print("\n[1/5] Loading data...")
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Load trained model
    print("\n[2/5] Loading pre-trained model...")
    pipeline = load_trained_model(TUNED_MODEL_PATH)
    
    # Load optimal threshold
    print("\n[3/5] Loading optimal threshold...")
    threshold = load_optimal_threshold(TUNING_RESULTS_PATH)
    
    # Make predictions
    print("\n[4/5] Generating predictions...")
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))
    
    # Create error analysis DataFrame
    errors_df = X_test.copy()
    errors_df['actual'] = y_test.values
    errors_df['predicted'] = y_pred
    errors_df['probability'] = y_proba
    
    # Error statistics
    print("\n[5/5] Analyzing error patterns...")
    profile = profile_errors(errors_df)
    
    print(f"\nError Profile:")
    print(f"  Total test samples: {profile['total_samples']}")
    print(f"\n  False Negatives (Missed Defaults):")
    print(f"    Count: {profile['false_negatives']['count']}")
    print(f"    Percentage: {profile['false_negatives']['percentage']:.2f}%")
    print(f"    Avg Probability: {profile['false_negatives']['avg_probability']:.3f}")
    print(f"\n  False Positives (False Alarms):")
    print(f"    Count: {profile['false_positives']['count']}")
    print(f"    Percentage: {profile['false_positives']['percentage']:.2f}%")
    print(f"    Avg Probability: {profile['false_positives']['avg_probability']:.3f}")
    
    # Sample errors
    fn_mask = (errors_df['actual'] == 1) & (errors_df['predicted'] == 0)
    fp_mask = (errors_df['actual'] == 0) & (errors_df['predicted'] == 1)
    
    print("\n" + "-"*70)
    print("Sample False Negatives (Top 5 by probability):")
    print("-"*70)
    if fn_mask.any():
        fn_samples = errors_df[fn_mask].nlargest(5, 'probability')
        print(fn_samples[['probability'] + list(X_test.columns[:5])].to_string())
    else:
        print("No false negatives found!")
    
    print("\n" + "-"*70)
    print("Sample False Positives (Top 5 by probability):")
    print("-"*70)
    if fp_mask.any():
        fp_samples = errors_df[fp_mask].nlargest(5, 'probability')
        print(fp_samples[['probability'] + list(X_test.columns[:5])].to_string())
    else:
        print("No false positives found!")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    os.makedirs(ERROR_DIR, exist_ok=True)
    
    analyze_error_distribution(
        errors_df,
        save_path=os.path.join(ERROR_DIR, "error_feature_distributions.png")
    )
    
    plot_probability_distribution(
        errors_df,
        save_path=os.path.join(ERROR_DIR, "error_probability_distribution.png")
    )
    
    # Save error samples to CSV
    errors_df.to_csv(
        os.path.join(ERROR_DIR, "error_analysis_full.csv"),
        index=False
    )
    print(f"✓ Full error analysis saved to {ERROR_DIR}/error_analysis_full.csv")
    
    print("\n" + "="*70)
    print("✓ ERROR ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nAll visualizations saved to: {ERROR_DIR}/")
    print("\nKey Insights:")
    print(f"  - FN Rate: {profile['false_negatives']['percentage']:.2f}% (missed defaults)")
    print(f"  - FP Rate: {profile['false_positives']['percentage']:.2f}% (false alarms)")
    print(f"  - FN avg prob: {profile['false_negatives']['avg_probability']:.3f} (close to threshold)")
    print(f"  - FP avg prob: {profile['false_positives']['avg_probability']:.3f} (model confidence)")
