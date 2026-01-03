"""
Explanation Agent - Generate human-readable explanations.

Creates audience-specific narratives for loan decisions.
"""
from typing import Dict, Any
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.base_agent import BaseAgent, AgentOutput


class ExplanationAgent(BaseAgent):
    """
    Generates human-readable explanations for loan decisions.
    
    Creates three types of explanations:
    - Borrower: Simple, non-technical
    - Loan Officer: Detailed with model insights
    - Auditor: Regulatory-compliant format
    """
    
    def __init__(self):
        super().__init__(name="ExplanationAgent", version="1.0.0")
    
    def generate_borrower_explanation(
        self, 
        decision: str,
        risk_data: Dict,
        policy_data: Dict,
        input_data: Dict
    ) -> str:
        """Generate simple explanation for borrower."""
        prob = risk_data.get("default_probability", 0)
        
        if decision == "APPROVED":
          return (
                f"Your loan application has been approved. "
                f"Based on your financial profile, we assess this as a {risk_data['risk_level'].replace('_', ' ').lower()} situation."
            )
        elif decision == "REJECTED":
            reasons = []
            
            # Check hard violations
            violations = policy_data.get("hard_violations", [])
            if violations:
                return f"Your application was declined due to: {violations[0]['description']}."
            
            # Check risk factors
            if prob > 0.7:
                reasons.append("high predicted default risk")
            
            soft_flags = policy_data.get("soft_flags", [])
            for flag in soft_flags[:2]:  # Top 2
                reasons.append(flag["description"].lower())
            
            if reasons:
                return f"Your application was declined due to: {', '.join(reasons)}."
            else:
                return "Your application was declined based on our credit assessment."
        
        else:  # MANUAL_REVIEW
            return (
                "Your application requires additional review. "
                "Our team will contact you within 2 business days."
            )
    
    def generate_officer_explanation(
        self,
        decision: str,
        risk_data: Dict,
        policy_data: Dict,
        input_data: Dict
    ) -> str:
        """Generate detailed explanation for loan officer."""
        prob = risk_data.get("default_probability", 0)
        threshold = risk_data.get("threshold_used", 0.614)
        
        explanation = f"Decision: {decision}\n\n"
        explanation += f"Model Assessment:\n"
        explanation += f"  - Default Probability: {prob:.3f}\n"
        explanation += f"  - Decision Threshold: {threshold:.3f}\n"
        explanation += f"  - Risk Category: {risk_data.get('risk_level', 'UNKNOWN')}\n"
        explanation += f"  - Confidence: {risk_data.get('confidence', 0):.2f}\n"
        
        # Risk flags
        if risk_data.get("flags"):
            explanation += f"\nRisk Flags:\n"
            for flag in risk_data["flags"]:
                explanation += f"  - {flag}\n"
        
        # Policy details
        soft_flags = policy_data.get("soft_flags", [])
        if soft_flags:
            explanation += f"\nPolicy Flags:\n"
            for flag in soft_flags:
                explanation += f"  - {flag['description']}\n"
        
        # Preferences
        prefs = policy_data.get("preferences_applied", [])
        if prefs:
            explanation += f"\nPositive Factors:\n"
            for pref in prefs:
                explanation += f"  - {pref['description']}\n"
        
        return explanation
    
    def generate_auditor_explanation(
        self,
        decision: str,
        risk_data: Dict,
        policy_data: Dict,
        input_data: Dict,
        decision_id: str = "N/A"
    ) -> str:
        """Generate regulatory-compliant explanation for auditors."""
        explanation = f"LOAN DECISION AUDIT RECORD\n"
        explanation += f"Decision ID: {decision_id}\n"
        explanation += f"Final Decision: {decision}\n\n"
        
        explanation += f"RISK ASSESSMENT:\n"
        explanation += f"  Model: XGBoost v1.0.0\n"
        explanation += f"  Default Probability: {risk_data.get('default_probability', 0):.4f}\n"
        explanation += f"  Threshold Applied: {risk_data.get('threshold_used', 0):.4f}\n"
        explanation += f"  Risk Level: {risk_data.get('risk_level', 'UNKNOWN')}\n\n"
        
        explanation += f"POLICY COMPLIANCE:\n"
        
        # Hard rules
        violations = policy_data.get("hard_violations", [])
        if violations:
            explanation += f"  Regulatory Violations:\n"
            for v in violations:
                explanation += f"    - {v['rule']}: {v['description']}\n"
        else:
            explanation += f"  No regulatory violations.\n"
        
        # Overrides
        if policy_data.get("override_applied"):
            explanation += f"  Override Applied: {policy_data.get('override_type', 'UNKNOWN')}\n"
        
        return explanation
    
    def run(
        self,
        decision: str,
        risk_data: Dict,
        policy_data: Dict,
        input_data: Dict,
        decision_id: str = "N/A"
    ) -> AgentOutput:
        """
        Generate explanations for loan decision.
        
        Args:
            decision: Final decision (APPROVED/REJECTED/MANUAL_REVIEW)
            risk_data: Output from RiskAnalystAgent
            policy_data: Output from PolicyAgent
            input_data: Original application
            decision_id: Unique decision identifier
        
        Returns:
            AgentOutput with multi-audience explanations
        """
        start_time = time.time()
        
        # Generate explanations
        borrower_exp = self.generate_borrower_explanation(
            decision, risk_data, policy_data, input_data
        )
        
        officer_exp = self.generate_officer_explanation(
            decision, risk_data, policy_data, input_data
        )
        
        auditor_exp = self.generate_auditor_explanation(
            decision, risk_data, policy_data, input_data, decision_id
        )
        
        # Identify key factors
        key_factors = {
            "negative": list(risk_data.get("flags", [])) + 
                       [f["rule"] for f in policy_data.get("soft_flags", [])],
            "positive": [p["rule"] for p in policy_data.get("preferences_applied", [])]
        }
        
        execution_time = (time.time() - start_time) * 1000
        
        output = AgentOutput(
            success=True,
            data={
                "borrower_explanation": borrower_exp,
                "loan_officer_explanation": officer_exp,
                "auditor_explanation": auditor_exp,
                "key_factors": key_factors
            },
            confidence=1.0,
            flags=[],
            reasoning="Explanations generated for all audiences",
            execution_time_ms=execution_time
        )
        
        self.log_execution({}, output)
        return output
