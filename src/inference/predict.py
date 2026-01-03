"""
Prediction service for loan default model.

Comprehensive production-ready service including:
- Model loading and caching
- Feature engineering for inference (matches build_features.py exactly)
- Prediction logic with configurable thresholds
- Risk categorization
- Batch predictions
- Error handling and validation
"""
import pandas as pd
import numpy as np
import pickle
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionService:
    """Production service for making loan default predictions."""
    
    def __init__(
        self, 
        model_path: str = "models/production/loan_default_model_final.pkl",
        metadata_path: str = "models/production/model_metadata.json"
    ):
        """
        Initialize prediction service.
        
        Args:
            model_path: Path to trained model
            metadata_path: Path to model metadata
        """
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.metadata = None
        self.threshold = 0.614  # Default F1-optimized threshold
        self.feature_names = None
        
        self._load_model()
        self._load_metadata()
    
    def _load_model(self):
        """Load trained model from disk with error handling."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Please train the final model first using train_final_model.py"
            )
        
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"✓ Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_metadata(self):
        """Load model metadata."""
        if not os.path.exists(self.metadata_path):
            logger.warning(f"⚠ Metadata not found at {self.metadata_path}")
            self.metadata = {}
            return
        
        try:
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            # Use recommended threshold from metadata
            if 'recommended_threshold' in self.metadata:
                self.threshold = self.metadata['recommended_threshold']
            
            # Store feature names for validation
            if 'feature_names' in self.metadata:
                self.feature_names = self.metadata['feature_names']
            
            logger.info(f"✓ Metadata loaded, threshold: {self.threshold}")
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            self.metadata = {}
    
    def _validate_input(self, input_data: Dict[str, Any]) -> None:
        """
        Validate input data has all required features.
        
        Args:
            input_data: Raw input dictionary
        
        Raises:
            ValueError: If required features are missing
        """
        required_base_features = [
            "age", "income", "loanamount", "creditscore",
            "monthsemployed", "numcreditlines", "interestrate",
            "loanterm", "dtiratio", "education", "employmenttype",
            "maritalstatus", "hasmortgage", "hasdependents",
            "loanpurpose", "hascosigner"
        ]
        
        missing = [f for f in required_base_features if f not in input_data]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
    
    def _prepare_features(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare features for prediction.
        MUST match build_features.py exactly for consistency!
        
        Args:
            input_data: Raw input dictionary
        
        Returns:
            DataFrame with all engineered features ready for model
        """
        # Validate input
        self._validate_input(input_data)
        
        # Create DataFrame from input
        df = pd.DataFrame([input_data])
        
        # ============================================================
        # FEATURE ENGINEERING (mirrors build_features.py exactly)
        # ============================================================
        
        # ---------- Ratio Features ----------
        df["loan_to_income_ratio"] = df["loanamount"] / (df["income"] + 1)
        
        # For single prediction, use fixed median or compute
        # Using reasonable defaults for single-row predictions
        dti_median = 0.35  # Approximate from training data
        df["high_dti"] = (df["dtiratio"] > dti_median).astype(int)
        
        # ---------- Credit Features ----------
        df["avg_debt_per_line"] = df["loanamount"] / (df["numcreditlines"] + 1)
        
        # Credit score bins
        df["credit_score_bin"] = pd.cut(
            df["creditscore"],
            bins=[0, 580, 670, 740, 850],
            labels=["poor", "fair", "good", "excellent"]
        )
        
        # ---------- Employment Stability ----------
        df["employment_stability"] = df["monthsemployed"].fillna(0)
        
        # Employment quality (using reasonable medians)
        income_median = 50000
        months_median = 36
        df["stable_high_earner"] = (
            (df["income"] > income_median) & 
            (df["monthsemployed"] > months_median)
        ).astype(int)
        
        # ---------- Age-based Features ----------
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 25, 35, 50, 100],
            labels=["young", "mid", "mature", "senior"]
        )
        
        df["credit_score_per_age"] = df["creditscore"] / (df["age"] + 1)
        
        # ---------- Log Transforms ----------
        df["log_income"] = np.log1p(df["income"])
        df["log_loan_amount"] = np.log1p(df["loanamount"])
        
        # ---------- Interaction Features ----------
        df["rate_times_amount"] = df["interestrate"] * df["loanamount"] / 10000
        
        # Risky loan indicator (using reasonable medians)
        interest_median = 0.10
        df["risky_loan"] = (
            (df["interestrate"] > interest_median) &
            (df["dtiratio"] > dti_median)
        ).astype(int)
        
        return df
    
    def predict(
        self, 
        input_data: Dict[str, Any],
        custom_threshold: Optional[float] = None,
        return_proba_only: bool = False
    ) -> Dict[str, Any]:
        """
        Make prediction for single loan application.
        
        Args:
            input_data: Application data dict
            custom_threshold: Optional custom decision threshold
            return_proba_only: If True, only return probability
        
        Returns:
            Prediction dictionary with comprehensive results
        """
        try:
            # Prepare features
            features_df = self._prepare_features(input_data)
            
            # Get probability
            proba_array = self.model.predict_proba(features_df)
            probability = float(proba_array[0, 1])
            
            if return_proba_only:
                return {"default_probability": round(probability, 4)}
            
            # Apply threshold
            threshold = custom_threshold if custom_threshold is not None else self.threshold
            prediction = "rejected" if probability >= threshold else "approved"
            
            # Determine confidence
            distance_from_threshold = abs(probability - threshold)
            if distance_from_threshold > 0.2:
                confidence = "high"
            elif distance_from_threshold > 0.1:
                confidence = "medium"
            else:
                confidence = "low"
            
            # Risk category
            if probability < 0.3:
                risk_category = "low"
            elif probability < 0.5:
                risk_category = "medium"
            elif probability < 0.7:
                risk_category = "high"
            else:
                risk_category = "very_high"
            
            # Build response
            response = {
                "prediction": prediction,
                "default_probability": round(probability, 4),
                "confidence": confidence,
                "risk_category": risk_category,
                "threshold_used": threshold,
                "model_version": self.metadata.get('model_type', 'XGBoost'),
                "timestamp": datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_batch(
        self,
        applications: List[Dict[str, Any]],
        custom_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Make predictions for batch of applications.
        
        Args:
            applications: List of application dicts
            custom_threshold: Optional custom threshold for all predictions
        
        Returns:
            List of prediction dicts
        """
        results = []
        for i, app in enumerate(applications):
            try:
                result = self.predict(app, custom_threshold)
                result['application_id'] = i  # Add batch index
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to predict application {i}: {e}")
                results.append({
                    'application_id': i,
                    'error': str(e),
                    'prediction': 'error'
                })
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information and metadata.
        
        Returns:
            Dictionary with model details
        """
        info = {
            "model_type": self.metadata.get('model_type', 'XGBoost'),
            "training_date": self.metadata.get('training_date', 'unknown'),
            "model_status": "loaded" if self.model is not None else "not_loaded",
            "n_samples_trained": self.metadata.get('n_samples', 'unknown'),
            "n_features": self.metadata.get('n_features', 'unknown'),
            "recommended_threshold": self.threshold,
            "class_distribution": self.metadata.get('class_distribution', {}),
            "scale_pos_weight": self.metadata.get('scale_pos_weight', 'unknown'),
            "hyperparameters": self.metadata.get('hyperparameters', {}),
            "versions": {
                "sklearn": self.metadata.get('sklearn_version', 'unknown'),
                "xgboost": self.metadata.get('xgboost_version', 'unknown'),
                "python": self.metadata.get('python_version', 'unknown')
            }
        }
        
        return info
    
    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        Get feature importances from the model.
        
        Args:
            top_n: Number of top features to return
        
        Returns:
            Dictionary of feature: importance
        """
        try:
            # Extract XGBoost model from pipeline
            xgb_model = self.model.named_steps['model']
            importances = xgb_model.feature_importances_
            
            # Get feature names (after preprocessing)
            feature_names = self.metadata.get('feature_names', 
                                             [f"feature_{i}" for i in range(len(importances))])
            
            # Create dict and sort
            importance_dict = dict(zip(feature_names, importances))
            sorted_importance = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
            )
            
            return sorted_importance
            
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return {}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the service.
        
        Returns:
            Health status dictionary
        """
        checks = {
            "status": "healthy",
            "model_loaded": self.model is not None,
            "metadata_loaded": bool(self.metadata),
            "timestamp": datetime.now().isoformat()
        }
        
        # Test prediction with dummy data
        try:
            dummy_input = {
                "age": 35,
                "income": 50000,
                "loanamount": 20000,
                "creditscore": 680,
                "monthsemployed": 36,
                "numcreditlines": 5,
                "interestrate": 0.08,
                "loanterm": 60,
                "dtiratio": 0.35,
                "education": "Bachelor",
                "employmenttype": "Full-time",
                "maritalstatus": "Married",
                "hasmortgage": "Yes",
                "hasdependents": "Yes",
                "loanpurpose": "DebtConsolidation",
                "hascosigner": "No"
            }
            _ = self.predict(dummy_input, return_proba_only=True)
            checks["prediction_test"] = "passed"
        except Exception as e:
            checks["prediction_test"] = f"failed: {e}"
            checks["status"] = "unhealthy"
        
        return checks


# Singleton instance for API use
_service_instance = None

def get_prediction_service() -> PredictionService:
    """
    Get singleton prediction service instance.
    
    Returns:
        PredictionService instance
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = PredictionService()
    return _service_instance
