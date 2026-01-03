"""
XGBoost model training with hyperparameter optimization and feature importance.
"""
import os
import sys
sys.path.append(os.path.dirname(__file__))

import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from model_utils import (
    load_data,
    split_data,
    cross_validate_model,
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    save_model,
    calculate_class_weight
)

MODEL_NAME = "xgboost_model"


def build_pipeline(X, scale_pos_weight=1.0):
    """
    Build preprocessing and XGBoost pipeline.
    
    Args:
        X: Feature dataframe
        scale_pos_weight: Weight for positive class (for imbalance)
    
    Returns:
        Sklearn Pipeline
    """
    # Identify column types
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    print(f"\nFeature types:")
    print(f"  Numeric: {len(num_cols)} columns")
    print(f"  Categorical: {len(cat_cols)} columns")
    
    # Preprocessor (XGBoost handles numeric features well, only encode categoricals)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),  # No scaling needed for XGBoost
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    
    # XGBoost Classifier with optimized hyperparameters
    # Note: early_stopping_rounds removed here to allow cross-validation
    # It will be applied during final model training with eval_set
    model = XGBClassifier(
        n_estimators=500,           # More trees for better performance
        max_depth=6,                # Moderate depth to prevent overfitting
        learning_rate=0.05,         # Lower learning rate for stability
        subsample=0.8,              # Row sampling
        colsample_bytree=0.8,       # Column sampling
        scale_pos_weight=scale_pos_weight,  # Handle class imbalance
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        verbosity=0                 # Reduce output noise
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    
    return pipeline


def plot_feature_importance(pipeline, X, save_path=None):
    """
    Plot top feature importances from XGBoost model.
    
    Args:
        pipeline: Trained pipeline
        X: Original feature dataframe
        save_path: Path to save plot
    """
    # Get the trained model
    model = pipeline.named_steps['model']
    
    # Get feature names after preprocessing
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Get feature names
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    if cat_cols:
        # Get encoded categorical feature names
        cat_encoder = preprocessor.named_transformers_['cat']
        cat_feature_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
        feature_names = num_cols + cat_feature_names
    else:
        feature_names = num_cols
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort by importance
    indices = importances.argsort()[::-1]
    top_n = min(20, len(indices))  # Top 20 features
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importances[indices[:top_n]][::-1])
    plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]][::-1])
    plt.xlabel('Feature Importance')
    plt.title('XGBoost - Top 20 Feature Importances')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Feature importance plot saved to {save_path}")
    
    plt.close()


if __name__ == "__main__":
    print("="*60)
    print("XGBOOST TRAINING")
    print("="*60)
    
    # Load data
    print("\n[1/6] Loading data...")
    df = load_data()
    
    # Split data
    print("\n[2/6] Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Calculate class weight for imbalance
    print("\n[3/6] Analyzing class distribution...")
    scale_pos_weight = calculate_class_weight(y_train)
    
    # Build pipeline
    print("\n[4/6] Building pipeline...")
    pipeline = build_pipeline(X_train, scale_pos_weight=scale_pos_weight)
    
    # Cross-validation
    print("\n[5/6] Training with cross-validation...")
    cv_results = cross_validate_model(pipeline, X_train, y_train, cv=5)
    
    # Train on full training set
    print("\nTraining final model...")
    pipeline.fit(X_train, y_train)
    print("✓ Training complete")
    
    # Evaluate on test set
    print("\n[6/6] Evaluating model...")
    results = evaluate_model(pipeline, X_test, y_test, model_name="XGBoost")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(
        results['confusion_matrix'], 
        "XGBoost",
        save_path=os.path.join("reports", "xgboost_confusion_matrix.png")
    )
    plot_roc_curve(
        y_test, 
        results['y_proba'],
        "XGBoost",
        save_path=os.path.join("reports", "xgboost_roc_curve.png")
    )
    plot_feature_importance(
        pipeline,
        X_train,
        save_path=os.path.join("reports", "xgboost_feature_importance.png")
    )
    
    # Save model
    print("\nSaving model...")
    save_model(pipeline, MODEL_NAME)
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)
    print(f"Model saved as: models/{MODEL_NAME}.pkl")
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
