"""
Risk Analyst Agent - Model-based risk assessment.

Interprets model probability and categorizes risk level.
"""
from typing import Dict, Any
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.base_agent import BaseAgent, AgentOutput
from inference.robust_predict import get_robust_prediction_service


class RiskAnalystAgent(BaseAgent):
    """
    Analyzes loan risk using trained ML model.
    
    Categorizes applications into:
    - CLEAR_ACCEPT (low risk, auto-approve)
    - BORDERLINE (needs policy review)
    - CLEAR_REJECT (high risk, auto-reject)
    """
    
    def __init__(self, threshold: float = 0.614):
        super().__init__(name="RiskAnalystAgent", version="1.0.0")
        self.threshold = threshold
        self.prediction_service = get_robust_prediction_service()
        
        # Risk thresholds
        self.clear_accept_threshold = 0.3  # prob < 0.3 → clear accept
        self.clear_reject_threshold = 0.8  # prob > 0.8 → clear reject
    
    def run(self, input_data: Dict[str, Any]) -> AgentOutput:
        """
        Perform risk analysis on loan application.
        
        Args:
            input_data: Validated loan application
        
        Returns:
            AgentOutput with risk assessment
        """
        start_time = time.time()
        
        # Get model prediction
        prediction_result = self.prediction_service.predict(input_data)
        
        probability = prediction_result["default_probability"]
        
        # Categorize risk
        if probability < self.clear_accept_threshold:
            risk_level = "LOW_RISK"
            category = "CLEAR_ACCEPT"
            recommended_action = "AUTO_APPROVE"
        elif probability > self.clear_reject_threshold:
            risk_level = "VERY_HIGH_RISK"
            category = "CLEAR_REJECT"
            recommended_action = "AUTO_REJECT"
        elif probability >= self.threshold:
            risk_level = "HIGH_RISK"
            category = "BORDERLINE"
            recommended_action = "MANUAL_REVIEW"
        else:
            risk_level = "MEDIUM_RISK"
            category = "BORDERLINE"
            recommended_action = "POLICY_CHECK"
        
        # Calculate confidence based on distance from threshold
        distance_from_threshold = abs(probability - self.threshold)
        confidence = min(1.0, distance_from_threshold / 0.3)  # Max at 0.3 distance
        
        # Identify flags
        flags = []
        if input_data.get("dtiratio", 0) > 0.45:
            flags.append("high_dti")
        if input_data.get("monthsemployed", 0) < 12:
            flags.append("short_employment")
        if input_data.get("creditscore", 850) < 620:
            flags.append("low_credit_score")
        if abs(probability - self.threshold) < 0.05:
            flags.append("borderline_probability")
        
        execution_time = (time.time() - start_time) * 1000
        
        output = AgentOutput(
            success=True,
            data={
                "default_probability": probability,
                "risk_level": risk_level,
                "category": category,
                "recommended_action": recommended_action,
                "threshold_used": self.threshold,
                "is_clear_case": category in ["CLEAR_ACCEPT", "CLEAR_REJECT"]
            },
            confidence=confidence,
            flags=flags,
            reasoning=f"Probability {probability:.3f} vs threshold {self.threshold:.3f} → {category}",
            execution_time_ms=execution_time
        )
        
        self.log_execution(input_data, output)
        return output
