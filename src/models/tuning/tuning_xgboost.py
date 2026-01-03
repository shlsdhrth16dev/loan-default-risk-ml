"""
XGBoost Hyperparameter Tuning with Optuna.

Features:
- Systematic hyperparameter search with cross-validation
- Decision threshold optimization for business objectives
- Comprehensive visualization and model persistence
"""
import optuna
import pandas as pd
import numpy as np
import os
import sys
import json
import pickle
from datetime import datetime

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'features'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    fbeta_score,
    confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Import from feature config and model utils
from feature_config import TARGET_COL
from model_utils import (
    load_data,
    split_data,
    calculate_class_weight,
    plot_confusion_matrix,
    plot_roc_curve
)

# Paths
TUNING_DIR = os.path.join("models", "tuning")
BEST_MODEL_PATH = os.path.join(TUNING_DIR, "best_xgboost_tuned.pkl")
RESULTS_PATH = os.path.join(TUNING_DIR, "tuning_results.json")


def build_pipeline(trial, X, scale_pos_weight):
    """
    Build XGBoost pipeline with Optuna trial suggestions.
    
    Args:
        trial: Optuna trial object
        X: Feature dataframe
        scale_pos_weight: Weight for positive class
    
    Returns:
        Sklearn Pipeline
    """
    # Identify column types
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    
    # Suggest hyperparameters
    params = {
        'n_estimators': trial.suggest_int("n_estimators", 200, 800),
        'max_depth': trial.suggest_int("max_depth", 3, 10),
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        'subsample': trial.suggest_float("subsample", 0.6, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 1.0),
        'min_child_weight': trial.suggest_int("min_child_weight", 1, 10),
        'gamma': trial.suggest_float("gamma", 0.0, 5.0),
        'reg_alpha': trial.suggest_float("reg_alpha", 0.0, 10.0),
        'reg_lambda': trial.suggest_float("reg_lambda", 0.0, 10.0),
        'scale_pos_weight': scale_pos_weight,
        'objective': "binary:logistic",
        'eval_metric': "auc",
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    model = XGBClassifier(**params)
    
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def objective(trial, X_train, y_train, scale_pos_weight):
    """
    Optuna objective function with cross-validation.
    
    Args:
        trial: Optuna trial
        X_train: Training features
        y_train: Training target
        scale_pos_weight: Class weight
    
    Returns:
        Mean CV ROC-AUC score
    """
    pipeline = build_pipeline(trial, X_train, scale_pos_weight)
    
    # 5-fold cross-validation
    scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    return scores.mean()


def optimize_threshold(y_true, y_proba, objective='f1'):
    """
    Find optimal classification threshold for different business objectives.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        objective: Optimization objective ('f1', 'precision', 'recall', 'balanced')
    
    Returns:
        Optimal threshold and metrics
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    
    if objective == 'f1':
        # Maximize F1 score
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        metric_name = "F1"
        
    elif objective == 'precision':
        # Maximize precision while keeping recall > 0.5
        valid_idx = recalls[:-1] > 0.5
        if valid_idx.any():
            valid_precisions = precisions[:-1][valid_idx]
            valid_thresholds = thresholds[valid_idx]
            best_idx = np.argmax(valid_precisions)
            best_threshold = valid_thresholds[best_idx]
        else:
            best_threshold = 0.5
        metric_name = "Precision (Recall>0.5)"
        
    elif objective == 'recall':
        # Maximize recall while keeping precision > 0.3
        valid_idx = precisions[:-1] > 0.3
        if valid_idx.any():
            valid_recalls = recalls[:-1][valid_idx]
            valid_thresholds = thresholds[valid_idx]
            best_idx = np.argmax(valid_recalls)
            best_threshold = valid_thresholds[best_idx]
        else:
            best_threshold = 0.5
        metric_name = "Recall (Precision>0.3)"
        
    else:  # balanced
        # Maximize F2 score (favors recall slightly)
        f2_scores = 5 * (precisions[:-1] * recalls[:-1]) / (4 * precisions[:-1] + recalls[:-1] + 1e-10)
        best_idx = np.argmax(f2_scores)
        best_threshold = thresholds[best_idx]
        metric_name = "F2"
    
    # Calculate metrics at best threshold
    y_pred = (y_proba >= best_threshold).astype(int)
    
    metrics = {
        'threshold': float(best_threshold),
        'optimized_for': metric_name,
        'precision': float(precision_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred)),
        'f2': float(fbeta_score(y_true, y_pred, beta=2)),
        'roc_auc': float(roc_auc_score(y_true, y_proba))
    }
    
    return best_threshold, metrics


def plot_threshold_analysis(y_true, y_proba, save_path=None):
    """
    Visualize precision-recall tradeoff at different thresholds.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        save_path: Path to save plot
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Precision-Recall vs Threshold
    ax1.plot(thresholds, precisions[:-1], label='Precision', linewidth=2)
    ax1.plot(thresholds, recalls[:-1], label='Recall', linewidth=2)
    ax1.plot(thresholds, f1_scores, label='F1 Score', linewidth=2, linestyle='--')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Metrics vs Classification Threshold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Precision-Recall Curve
    ax2.plot(recalls, precisions, linewidth=2)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Threshold analysis saved to {save_path}")
    
    plt.close()


def plot_optimization_history(study, save_path=None):
    """
    Plot Optuna optimization history.
    
    Args:
        study: Optuna study object
        save_path: Path to save plot
    """
    fig = optuna.visualization.matplotlib.plot_optimization_history(study)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Optimization history saved to {save_path}")
    
    plt.close()


def plot_param_importances(study, save_path=None):
    """
    Plot parameter importances from Optuna study.
    
    Args:
        study: Optuna study object
        save_path: Path to save plot
    """
    fig = optuna.visualization.matplotlib.plot_param_importances(study)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Parameter importances saved to {save_path}")
    
    plt.close()


if __name__ == "__main__":
    print("="*70)
    print("XGBOOST HYPERPARAMETER TUNING WITH OPTUNA")
    print("="*70)
    
    # Load and split data
    print("\n[1/6] Loading data...")
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Calculate class weight
    print("\n[2/6] Analyzing class distribution...")
    scale_pos_weight = calculate_class_weight(y_train)
    
    # Run Optuna optimization
    print("\n[3/6] Running hyperparameter optimization...")
    print("Trials: 10 (testing mode)")
    print("Cross-validation: 5-fold")
    print("Metric: ROC-AUC")
    
    study = optuna.create_study(
        direction="maximize",
        study_name="xgboost_tuning",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, scale_pos_weight),
        n_trials=10,  # Use 50+ for production
        show_progress_bar=True
    )
    
    print(f"\n✓ Optimization complete!")
    print(f"Best ROC-AUC (CV): {study.best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    
    # Train final model with best params
    print("\n[4/6] Training final model with best parameters...")
    best_pipeline = build_pipeline(study.best_trial, X_train, scale_pos_weight)
    best_pipeline.fit(X_train, y_train)
    
    # Evaluate on test set
    print("\n[5/6] Evaluating on test set...")
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred_default = best_pipeline.predict(X_test)
    
    print(f"\nTest ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"\nWith default threshold (0.5):")
    print(f"  Precision: {precision_score(y_test, y_pred_default):.4f}")
    print(f"  Recall: {recall_score(y_test, y_pred_default):.4f}")
    print(f"  F1: {f1_score(y_test, y_pred_default):.4f}")
    
    # Optimize decision threshold
    print("\n[6/6] Optimizing decision thresholds...")
    
    objectives = ['f1', 'precision', 'recall', 'balanced']
    threshold_results = {}
    
    for obj in objectives:
        threshold, metrics = optimize_threshold(y_test, y_proba, objective=obj)
        threshold_results[obj] = metrics
        print(f"\n{obj.upper()} Optimization:")
        print(f"  Threshold: {threshold:.3f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    os.makedirs(TUNING_DIR, exist_ok=True)
    
    plot_optimization_history(
        study,
        save_path=os.path.join(TUNING_DIR, "optimization_history.png")
    )
    
    plot_param_importances(
        study,
        save_path=os.path.join(TUNING_DIR, "param_importances.png")
    )
    
    plot_threshold_analysis(
        y_test, y_proba,
        save_path=os.path.join(TUNING_DIR, "threshold_analysis.png")
    )
    
    plot_confusion_matrix(
        confusion_matrix(y_test, y_pred_default),
        "XGBoost Tuned (Default Threshold)",
        save_path=os.path.join(TUNING_DIR, "confusion_matrix.png")
    )
    
    plot_roc_curve(
        y_test, y_proba,
        "XGBoost Tuned",
        save_path=os.path.join(TUNING_DIR, "roc_curve.png")
    )
    
    # Save results
    print("\nSaving results...")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'best_cv_roc_auc': study.best_value,
        'test_roc_auc': float(roc_auc_score(y_test, y_proba)),
        'best_params': study.best_params,
        'threshold_optimizations': threshold_results,
        'n_trials': len(study.trials)
    }
    
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {RESULTS_PATH}")
    
    # Save best model
    with open(BEST_MODEL_PATH, 'wb') as f:
        pickle.dump(best_pipeline, f)
    print(f"✓ Best model saved to {BEST_MODEL_PATH}")
    
    print("\n" + "="*70)
    print("✓ HYPERPARAMETER TUNING COMPLETE!")
    print("="*70)
    print(f"\nKey Results:")
    print(f"  Best CV ROC-AUC: {study.best_value:.4f}")
    print(f"  Test ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"\nRecommended threshold for balanced performance: {threshold_results['f1']['threshold']:.3f}")
    print(f"  (Precision: {threshold_results['f1']['precision']:.4f}, Recall: {threshold_results['f1']['recall']:.4f})")
