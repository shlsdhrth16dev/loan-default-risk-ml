"""
Robust Prediction Service with Model Fallback.

Multi-tier fallback system:
1. XGBoost (primary) - Best performance
2. Logistic Regression (fallback) - Stable baseline
3. Conservative rules (emergency) - Always works
"""
import pickle
import os
import sys
import logging
from typing import Dict, Any, Optional
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from inference.circuit_breaker import CircuitBreaker
from inference.predict import PredictionService

logger = logging.getLogger(__name__)


class RobustPredictionService:
    """
    Production prediction service with automatic fallback.
    
    Features:
    - Multi-model fallback (XGBoost → Logistic → Rules)
    - Circuit breaker protection
    - Metrics tracking
    - Automatic recovery
    """
    
    def __init__(
        self,
        xgboost_model_path: str = "models/production/loan_default_model_final.pkl",
        logistic_model_path: str = "models/logistic_model.pkl"
    ):
        """
        Initialize robust prediction service.
        
        Args:
            xgboost_model_path: Path to XGBoost model
            logistic_model_path: Path to Logistic Regression model
        """
        # Load models
        self.models = {}
        self.models['xgboost'] = self._load_model(xgboost_model_path, 'xgboost')
        self.models['logistic'] = self._load_model(logistic_model_path, 'logistic')
        
        # Circuit breakers
        self.circuit_breakers = {
            'xgboost': CircuitBreaker('xgboost', failure_threshold=5, timeout_seconds=60),
            'logistic': CircuitBreaker('logistic', failure_threshold=10, timeout_seconds=120)
        }
        
        # Metrics
        self.usage_count = defaultdict(int)
        self.fallback_count = defaultdict(int)
        
        # Prediction service for feature engineering
        self.prediction_service = PredictionService()
    
    def _load_model(self, path: str, model_type: str) -> Optional[Any]:
        """Load model with error handling."""
        if not os.path.exists(path):
            logger.warning(f"{model_type} model not found at {path}")
            return None
        
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"✓ Loaded {model_type} model from {path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {e}")
            return None
    
    def predict(
        self,
        input_data: Dict[str, Any],
        custom_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Make prediction with automatic fallback.
        
        Args:
            input_data: Application data
            custom_threshold: Optional custom threshold
        
        Returns:
            Prediction dict with model_used indicator
        """
        # Try XGBoost first
        if self._can_use_model('xgboost'):
            try:
                result = self.prediction_service.predict(input_data, custom_threshold)
                self.circuit_breakers['xgboost'].record_success()
                self.usage_count['xgboost'] += 1
                result['model_used'] = 'xgboost'
                return result
            except Exception as e:
                logger.error(f"XGBoost prediction failed: {e}")
                self.circuit_breakers['xgboost'].record_failure()
                self.fallback_count['xgboost_to_logistic'] += 1
        
        # Fallback to Logistic Regression
        if self._can_use_model('logistic'):
            try:
                logger.warning("Using fallback model: Logistic Regression")
                result = self._predict_with_logistic(input_data, custom_threshold)
                self.circuit_breakers['logistic'].record_success()
                self.usage_count['logistic'] += 1
                result['model_used'] = 'logistic_fallback'
                result['fallback_reason'] = 'xgboost_unavailable'
                return result
            except Exception as e:
                logger.error(f"Logistic prediction failed: {e}")
                self.circuit_breakers['logistic'].record_failure()
                self.fallback_count['logistic_to_rules'] += 1
        
        # Emergency fallback: Conservative rules
        logger.critical("All ML models failed, using conservative rule-based system")
        self.usage_count['emergency_rules'] += 1
        result = self._rule_based_prediction(input_data)
        result['model_used'] = 'emergency_rules'
        result['fallback_reason'] = 'all_models_failed'
        
        # Alert ops team
        self._alert_ops_team("CRITICAL: Using emergency rule-based fallback!")
        
        return result
    
    def _can_use_model(self, model_name: str) -> bool:
        """Check if model can be used."""
        if self.models.get(model_name) is None:
            return False
        
        if model_name in self.circuit_breakers:
            return self.circuit_breakers[model_name].is_available()
        
        return True
    
    def _predict_with_logistic(
        self,
        input_data: Dict[str, Any],
        custom_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Predict using Logistic Regression fallback."""
        # Use same feature engineering as XGBoost
        features_df = self.prediction_service._prepare_features(input_data)
        
        # Get probability from logistic model
        proba = self.models['logistic'].predict_proba(features_df)[0, 1]
        
        # Apply threshold
        threshold = custom_threshold if custom_threshold else 0.614
        prediction = "rejected" if proba >= threshold else "approved"
        
        # Risk categorization (same logic as main service)
        if proba < 0.3:
            risk_category = "low"
            confidence = "high"
        elif proba < 0.5:
            risk_category = "medium"
            confidence = "medium"
        elif proba < 0.7:
            risk_category = "high"
            confidence = "medium"
        else:
            risk_category = "very_high"
            confidence = "high"
        
        return {
            "prediction": prediction,
            "default_probability": round(float(proba), 4),
            "confidence": confidence,
            "risk_category": risk_category,
            "threshold_used": threshold,
            "model_version": "LogisticRegression_v1.0"
        }
    
    def _rule_based_prediction(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Emergency rule-based prediction (no ML).
        
        Conservative rules for when all models fail.
        """
        # Extract key features
        income = input_data.get('income', 0)
        credit_score = input_data.get('creditscore', 300)
        dti = input_data.get('dtiratio', 1.0)
        loan_amount = input_data.get('loanamount', 0)
        
        # Conservative decision logic
        if income < 20000 or credit_score < 550 or dti > 0.55:
            decision = "rejected"
            prob = 0.85  # High risk
        elif income > 80000 and credit_score > 720 and dti < 0.35:
            decision = "approved"
            prob = 0.15  # Low risk
        else:
            decision = "rejected"  # Conservative: reject if uncertain
            prob = 0.60
        
        return {
            "prediction": decision,
            "default_probability": prob,
            "confidence": "low",  # Low confidence for rules
            "risk_category": "high" if prob > 0.5 else "low",
            "threshold_used": 0.5,
            "model_version": "RuleBased_Emergency"
        }
    
    def _alert_ops_team(self, message: str):
        """Alert operations team (placeholder for real alerting)."""
        logger.critical(f"OPS ALERT: {message}")
        # TODO: Integrate with PagerDuty, Slack, etc.
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get health metrics for monitoring."""
        return {
            "model_usage": dict(self.usage_count),
            "fallback_usage": dict(self.fallback_count),
            "circuit_breakers": {
                name: cb.get_metrics()
                for name, cb in self.circuit_breakers.items()
            },
            "xgboost_available": self._can_use_model('xgboost'),
            "logistic_available": self._can_use_model('logistic')
        }


# Singleton instance
_robust_service_instance = None

def get_robust_prediction_service() -> RobustPredictionService:
    """Get singleton robust prediction service."""
    global _robust_service_instance
    if _robust_service_instance is None:
        _robust_service_instance = RobustPredictionService()
    return _robust_service_instance
