"""
MLflow Configuration and Utilities.

Centralized MLflow setup for:
- Experiment tracking
- Model registry
- Artifact logging
- Metric tracking
"""
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import os
from pathlib import Path


# MLflow tracking URI (local by default)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")

# Experiment names
EXPERIMENT_TRAINING = "loan-default-training"
EXPERIMENT_TUNING = "loan-default-tuning"
EXPERIMENT_EVALUATION = "loan-default-evaluation"

# Model registry
REGISTRY_MODEL_NAME = "loan-default-predictor"


def setup_mlflow(experiment_name: str = EXPERIMENT_TRAINING):
    """
    Setup MLflow tracking.
    
    Args:
        experiment_name: Name of experiment
    """
    # Set tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Set experiment
    mlflow.set_experiment(experiment_name)
    
    print(f"✓ MLflow tracking configured")
    print(f"  Tracking URI: {ML FLOW_TRACKING_URI}")
    print(f"  Experiment: {experiment_name}")


def log_model_info(model_type: str, params: dict, metrics: dict, artifacts: dict = None):
    """
    Log model information to MLflow.
    
    Args:
        model_type: Type of model (e.g., "XGBoost", "Logistic")
        params: Model hyperparameters
        metrics: Model metrics
        artifacts: Optional dict of artifact paths
    """
    # Log params
    mlflow.log_params(params)
    
    # Log metrics
    mlflow.log_metrics(metrics)
    
    # Log model type as tag
    mlflow.set_tag("model_type", model_type)
    
    # Log artifacts if provided
    if artifacts:
        for name, path in artifacts.items():
            if os.path.exists(path):
                mlflow.log_artifact(path, artifact_path=name)
    
    print(f"✓ Logged to MLflow: {model_type}")
    print(f"  Params: {len(params)} parameters")
    print(f"  Metrics: {len(metrics)} metrics")
    if artifacts:
        print(f"  Artifacts: {len(artifacts)} files")


def register_model(model, model_name: str = REGISTRY_MODEL_NAME, stage: str = "None"):
    """
    Register model in MLflow Model Registry.
    
    Args:
        model: Trained model
        model_name: Name in registry
        stage: Stage to assign ("None", "Staging", "Production")
    
    Returns:
        Model version
    """
    # Log model
    model_info = mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name=model_name
    )
    
    # Get version
    model_version = model_info.registered_model_version
    
    # Transition to stage if requested
    if stage != "None":
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage=stage
        )
        print(f"✓ Model registered: {model_name} v{model_version} → {stage}")
    else:
        print(f"✓ Model registered: {model_name} v{model_version}")
    
    return model_version


def load_production_model(model_name: str = REGISTRY_MODEL_NAME):
    """
    Load production model from registry.
    
    Args:
        model_name: Name in registry
    
    Returns:
        Loaded model
    """
    model_uri = f"models:/{model_name}/Production"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"✓ Loaded production model: {model_name}")
    return model


def get_mlflow_ui_command():
    """Get command to start MLflow UI."""
    return f"mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}"
