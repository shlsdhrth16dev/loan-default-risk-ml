"""
Loan Decision Crew - Multi-agent orchestration.

Coordinates agents with conditional routing for optimal performance.
"""
from typing import Dict, Any
import sys
import os
import time
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.validation_agent import ValidationAgent
from agents.risk_analyst_agent import RiskAnalystAgent
from agents.policy_agent import PolicyAgent
from agents.explanation_agent import ExplanationAgent


class LoanDecisionCrew:
    """
    Orchestrates multi-agent loan decision pipeline.
    
    Flow:
    1. Validation (always)
    2. Risk Analysis (if valid)
    3. Policy Check (if borderline)
   4. Explanation (always for valid inputs)
    
    Uses conditional routing for performance optimization.
    """
    
    def __init__(self):
        self.validation_agent = ValidationAgent()
        self.risk_analyst_agent = RiskAnalystAgent()
        self.policy_agent = PolicyAgent()
        self.explanation_agent = ExplanationAgent()
    
    def process(self, application: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process loan application through agent pipeline.
        
        Args:
            application: Loan application data
        
        Returns:
            Comprehensive decision with explanations
        """
        start_time = time.time()
        decision_id = str(uuid.uuid4())
        
        # PHASE 1: Validation (always runs)
        validation_output = self.validation_agent.run(application)
        
        if not validation_output.data["is_valid"]:
            # Early rejection - invalid input
            return {
                "decision_id": decision_id,
                "decision": "REJECTED",
                "reason": "Invalid application data",
                "validation_errors": validation_output.data["errors"],
                "total_execution_time_ms": (time.time() - start_time) * 1000,
                "agents_executed": ["ValidationAgent"]
            }
        
        # PHASE 2: Risk Analysis (always runs for valid input)
        risk_output = self.risk_analyst_agent.run(application)
        risk_data = risk_output.data
        
        # PHASE 3: Conditional routing based on risk category
        policy_output = None
        policy_data = {}
        
        if risk_data["is_clear_case"]:
            # Clear case - skip policy agent
            if risk_data["category"] == "CLEAR_ACCEPT":
                final_decision = "APPROVED"
            else:  # CLEAR_REJECT
                final_decision = "REJECTED"
            
            agents_executed = ["ValidationAgent", "RiskAnalystAgent"]
        
        else:
            # Borderline case - run policy agent
            policy_output = self.policy_agent.run(application, risk_data)
            policy_data = policy_output.data
            
            # Determine final decision
            if policy_data.get("override_applied"):
                final_decision = policy_data["final_decision"]
            elif policy_data.get("requires_manual_review"):
                final_decision = "MANUAL_REVIEW"
            elif risk_data["default_probability"] >= risk_data["threshold_used"]:
                final_decision = "REJECTED"
            else:
                final_decision = "APPROVED"
            
            agents_executed = ["ValidationAgent", "RiskAnalystAgent", "PolicyAgent"]
        
        # PHASE 4: Generate explanations (always for valid input)
        explanation_output = self.explanation_agent.run(
            decision=final_decision,
            risk_data=risk_data,
            policy_data=policy_data,
            input_data=application,
            decision_id=decision_id
        )
        
        agents_executed.append("ExplanationAgent")
        
        # Calculate aggregate confidence
        confidences = [risk_output.confidence]
        if policy_output:
            confidences.append(policy_output.confidence)
        aggregate_confidence = min(confidences)
        
        # Build comprehensive response
        total_time = (time.time() - start_time) * 1000
        
        return {
            "decision_id": decision_id,
            "decision": final_decision,
            "confidence": aggregate_confidence,
            
            # Risk assessment
            "risk_assessment": {
                "default_probability": risk_data["default_probability"],
                "risk_level": risk_data["risk_level"],
                "model_confidence": risk_output.confidence,
                "flags": risk_output.flags
            },
            
            # Policy evaluation
            "policy_evaluation": {
                "override_applied": policy_data.get("override_applied", False),
                "override_type": policy_data.get("override_type"),
                "soft_flags": policy_data.get("soft_flags", []),
                "preferences_applied": policy_data.get("preferences_applied", []),
                "requires_manual_review": policy_data.get("requires_manual_review", False)
            } if policy_data else None,
            
            # Explanations
            "explanations": explanation_output.data,
            
            # Metadata
            "metadata": {
                "decision_id": decision_id,
                "agents_executed": agents_executed,
                "total_execution_time_ms": total_time,
                "agent_execution_times": {
                    "validation": validation_output.execution_time_ms,
                    "risk_analysis": risk_output.execution_time_ms,
                    "policy": policy_output.execution_time_ms if policy_output else 0,
                    "explanation": explanation_output.execution_time_ms
                }
            }
        }
    
    def get_crew_info(self) -> Dict[str, Any]:
        """Get information about the crew."""
        return {
            "name": "LoanDecisionCrew",
            "version": "1.0.0",
            "agents": [
                {"name": "ValidationAgent", "version": "1.0.0"},
                {"name": "RiskAnalystAgent", "version": "1.0.0"},
                {"name": "PolicyAgent", "version": "1.0.0"},
                {"name": "ExplanationAgent", "version": "1.0.0"}
            ],
            "flow": "Conditional pipeline with early exit for clear cases"
        }


# Singleton instance
_crew_instance = None

def get_loan_decision_crew() -> LoanDecisionCrew:
    """Get singleton crew instance."""
    global _crew_instance
    if _crew_instance is None:
        _crew_instance = LoanDecisionCrew()
    return _crew_instance
