"""
Logistic Regression model training with comprehensive evaluation.
"""
import os
import sys
sys.path.append(os.path.dirname(__file__))

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from model_utils import (
    load_data,
    split_data,
    cross_validate_model,
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    save_model
)

MODEL_NAME = "logistic_regression"


def build_pipeline(X):
    """
    Build preprocessing and model pipeline.
    
    Args:
        X: Feature dataframe
    
    Returns:
        Sklearn Pipeline
    """
    # Identify column types
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    print(f"\nFeature types:")
    print(f"  Numeric: {len(num_cols)} columns")
    print(f"  Categorical: {len(cat_cols)} columns")
    
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    
    # Logistic Regression with balanced class weights
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # Handle class imbalance
        solver='lbfgs',
        n_jobs=-1,
        random_state=42
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    
    return pipeline


if __name__ == "__main__":
    print("="*60)
    print("LOGISTIC REGRESSION TRAINING")
    print("="*60)
    
    # Load data
    print("\n[1/5] Loading data...")
    df = load_data()
    
    # Split data
    print("\n[2/5] Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Build pipeline
    print("\n[3/5] Building pipeline...")
    pipeline = build_pipeline(X_train)
    
    # Cross-validation
    print("\n[4/5] Training with cross-validation...")
    cv_results = cross_validate_model(pipeline, X_train, y_train, cv=5)
    
    # Train on full training set
    print("\nTraining final model on full training set...")
    pipeline.fit(X_train, y_train)
    print("✓ Training complete")
    
    # Evaluate on test set
    print("\n[5/5] Evaluating model...")
    results = evaluate_model(pipeline, X_test, y_test, model_name="Logistic Regression")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(
        results['confusion_matrix'], 
        "Logistic Regression",
        save_path=os.path.join("reports", "logistic_confusion_matrix.png")
    )
    plot_roc_curve(
        y_test, 
        results['y_proba'],
        "Logistic Regression",
        save_path=os.path.join("reports", "logistic_roc_curve.png")
    )
    
    # Save model
    print("\nSaving model...")
    save_model(pipeline, MODEL_NAME)
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)
    print(f"Model saved as: models/{MODEL_NAME}.pkl")
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
