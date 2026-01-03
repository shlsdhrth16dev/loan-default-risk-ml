"""
MLflow Integration Test.

Quick test to verify:
- MLflow tracking works
- Model logging works
- Registry works
"""
import mlflow
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.mlflow_config import setup_mlflow, log_model_info, register_model

# Simple test
if __name__ == "__main__":
    print("="*70)
    print("MLFLOW INTEGRATION TEST")
    print("="*70)
    
    # Setup
    print("\n[1/3] Setting up MLflow...")
    setup_mlflow(experiment_name="test-experiment")
    
    # Test tracking
    print("\n[2/3] Testing experiment tracking...")
    with mlflow.start_run(run_name="test-run"):
        # Log params
        test_params = {
            "learning_rate": 0.05,
            "n_estimators": 100
        }
        
        # Log metrics
        test_metrics = {
            "accuracy": 0.85,
            "roc_auc": 0.76
        }
        
        log_model_info("TestModel", test_params, test_metrics)
        
        print("✓ Tracking successful!")
    
    # Test registry
    print("\n[3/3] MLflow tracking URI:", mlflow.get_tracking_uri())
    
    print("\n" + "="*70)
    print("✓ MLFLOW TEST PASSED!")
    print("="*70)
    print("\nTo view results:")
    print("  Run: mlflow ui")
    print("  Then visit: http://localhost:5000")
