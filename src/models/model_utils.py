"""
Shared utilities for model training and evaluation.
"""
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import (
    roc_auc_score, 
    classification_report, 
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# Import feature configuration
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'features'))
from feature_config import TARGET_COL

DATA_PATH = os.path.join("data", "processed", "features.csv")
MODEL_DIR = "models"


def load_data(path=DATA_PATH):
    """Load feature dataset from CSV."""
    df = pd.read_csv(path)
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def split_data(df, test_size=0.2, random_state=42):
    """
    Split data into train and test sets with stratification.
    
    Args:
        df: DataFrame with features and target
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    print(f"✓ Train set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    print(f"✓ Class distribution (train): {y_train.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test


def cross_validate_model(pipeline, X, y, cv=5):
    """
    Perform cross-validation and return metrics.
    
    Args:
        pipeline: Sklearn pipeline with model
        X: Features
        y: Target
        cv: Number of folds
    
    Returns:
        Dictionary of CV scores
    """
    print(f"\nPerforming {cv}-fold cross-validation...")
    
    scoring = {
        'roc_auc': 'roc_auc',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1'
    }
    
    cv_results = cross_validate(
        pipeline, X, y, 
        cv=cv, 
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )
    
    print("\nCross-Validation Results:")
    for metric, scores in cv_results.items():
        if metric.startswith('test_'):
            metric_name = metric.replace('test_', '')
            print(f"  {metric_name.upper()}: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    return cv_results


def evaluate_model(pipeline, X_test, y_test, model_name="Model"):
    """
    Evaluate model on test set and print metrics.
    
    Args:
        pipeline: Trained sklearn pipeline
        X_test: Test features
        y_test: Test target
        model_name: Name for display
    
    Returns:
        Dictionary with predictions and probabilities
    """
    print(f"\n{'='*60}")
    print(f"{model_name} - Test Set Evaluation")
    print(f"{'='*60}")
    
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    return {
        'y_pred': y_pred,
        'y_proba': y_proba,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(cm, model_name, save_path=None):
    """Plot confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_roc_curve(y_test, y_proba, model_name, save_path=None):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve saved to {save_path}")
    
    plt.close()


def save_model(pipeline, model_name):
    """
    Save trained pipeline to disk.
    
    Args:
        pipeline: Trained sklearn pipeline
        model_name: Name for the saved file (without extension)
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    print(f"✓ Model saved to {model_path}")
    return model_path


def calculate_class_weight(y_train):
    """
    Calculate scale_pos_weight for imbalanced datasets.
    
    Args:
        y_train: Training target variable
    
    Returns:
        Scale weight for positive class
    """
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    
    print(f"\nClass imbalance:")
    print(f"  Negative class: {neg_count} ({neg_count/(neg_count+pos_count)*100:.1f}%)")
    print(f"  Positive class: {pos_count} ({pos_count/(neg_count+pos_count)*100:.1f}%)")
    print(f"  Scale_pos_weight: {scale_pos_weight:.2f}")
    
    return scale_pos_weight
