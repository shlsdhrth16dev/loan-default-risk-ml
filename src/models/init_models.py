"""
Model Initialization Utility.

Handles model retraining if pickles are missing (common on first cloud deploy).
"""
import os
import sys
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.train_xgboost import train_model as train_xgb
from models.train_logistic import train_model as train_log
from inference.train_final_model import train_final_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_and_init_models(force=False):
    """Check if required models exist and train them if not."""
    required_files = [
        "models/production/loan_default_model_final.pkl",
        "models/logistic_model.pkl"
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing or force:
        logger.info(f"Missing models: {missing}. Starting initialization...")
        
        # Ensure directories exist
        os.makedirs("models/production", exist_ok=True)
        os.makedirs("reports/figures", exist_ok=True)
        
        try:
            # 1. Train Logistic Regression (Baseline)
            logger.info("Training Logistic Regression...")
            train_log()
            
            # 2. Train XGBoost
            logger.info("Training XGBoost...")
            train_xgb()
            
            # 3. Train Final Model
            logger.info("Training Final Model...")
            train_final_model()
            
            logger.info("✓ Model initialization complete!")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            return False
    else:
        logger.info("✓ All models present. No initialization needed.")
        return True

if __name__ == "__main__":
    check_and_init_models()
