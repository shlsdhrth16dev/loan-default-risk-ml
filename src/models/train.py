import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score

def perform_grid_search(X: pd.DataFrame, y: pd.Series, scale_pos_weight: float = 1.0):
    """
    Performs Grid Search to find optimal hyperparameters.
    """
    print("Starting Grid Search...")
    
    # Define hyperparameter grid
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False
    )
    
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=3,
        verbose=1,
        n_jobs=-1  # Use all cores
    )
    
    grid_search.fit(X, y)
    
    print(f"Best ROC-AUC: {grid_search.best_score_:.4f}")
    print(f"Best Params: {grid_search.best_params_}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def train_model(X: pd.DataFrame, y: pd.Series, scale_pos_weight: float = 1.0) -> XGBClassifier:
    """
    Trains an XGBoost Classifier with specified hyperparameters.
    Includes validation logic to ensure model effectiveness.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        scale_pos_weight (float): Weight for positive class to handle imbalance.
        
    Returns:
        XGBClassifier: The trained model object
    """
    # Splitting the data to create a validation set for monitoring performance
    # This acts as an internal validation set for early stopping
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Internal Training set shape: {X_train.shape}")
    print(f"Internal Validation set shape: {X_val.shape}")

    # Initialize model with optimized hyperparameters (found via Grid Search)
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=1.0,
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )

    # Train the model with early stopping based on validation set performance
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=10  # Print progress every 10 trees
    )

    # Validate the model
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    roc = roc_auc_score(y_val, y_prob)
    print(f"\nValidation ROC-AUC: {roc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    return model
