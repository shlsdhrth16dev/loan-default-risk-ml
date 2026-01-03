"""
Policy Agent - Business rules and regulatory compliance.

Applies hard rules, soft rules, and preference rules.
"""
from typing import Dict, Any, List,Tuple
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.base_agent import BaseAgent, AgentOutput


class PolicyAgent(BaseAgent):
    """
    Applies business policies and regulatory rules.
    
    Three rule types:
    - HARD: Cannot be overridden (regulatory)
    - SOFT: Flag for review (risk management)
    - PREFERENCE: Business strategy (can be adjusted)
    """
    
    def __init__(self):
        super().__init__(name="PolicyAgent", version="1.0.0")
        
        # HARD RULES - Regulatory compliance (auto-reject)
        self.hard_rules = [
            ("income_too_low", lambda d: d.get("income", 0) < 15000, "Income below minimum threshold"),
            ("credit_too_low", lambda d: d.get("creditscore", 850) < 500, "Credit score below regulatory minimum"),
            ("dti_too_high", lambda d: d.get("dtiratio", 0) > 0.60, "DTI ratio exceeds regulatory limit")
        ]
        
        # SOFT RULES - Flag for manual review
        self.soft_rules = [
            ("high_dti", lambda d: d.get("dtiratio", 0) > 0.50, "DTI ratio > 50%"),
            ("short_employment", lambda d: d.get("monthsemployed", 999) < 6, "Less than 6 months employment"),
            ("large_loan", lambda d: d.get("loanamount", 0) > 40000, "Loan amount > $40k"),
            ("high_rate", lambda d: d.get("interestrate", 0) > 0.20, "Interest rate > 20%")
        ]
        
        # PREFERENCE RULES - Business strategy (bonuses/penalties)
        self.preference_rules = [
            ("stable_high_earner", 
             lambda d: d.get("income", 0) > 75000 and d.get("monthsemployed", 0) > 36,
             "High income + stable employment", 0.1),  # Reduce risk by 10%
            
            ("excellent_credit",
             lambda d: d.get("creditscore", 0) > 760,
             "Excellent credit score", 0.05),
            
            ("has_cosigner",
             lambda d: d.get("hascosigner") == "Yes",
             "Has cosigner", 0.08)
        ]
    
    def check_rules(self, input_data: Dict[str, Any], rules: List[Tuple]) -> List[Dict]:
        """Check a set of rules against input."""
        triggered = []
        for rule_name, rule_func, description, *bonus in rules:
            try:
                if rule_func(input_data):
                    triggered.append({
                        "rule": rule_name,
                        "description": description,
                        "bonus": bonus[0] if bonus else None
                    })
            except Exception as e:
                self.logger.error(f"Error checking rule {rule_name}: {e}")
        return triggered
    
    def run(self, input_data: Dict[str, Any], risk_data: Dict[str, Any]) -> AgentOutput:
        """
        Apply policy rules to loan application.
        
        Args:
            input_data: Loan application
            risk_data: Risk assessment from RiskAnalystAgent
        
        Returns:
            AgentOutput with policy decision
        """
        start_time = time.time()
        
        # Check hard rules
        hard_violations = self.check_rules(input_data, self.hard_rules)
        
        # If hard rules violated, auto-reject
        if hard_violations:
            execution_time = (time.time() - start_time) * 1000
            return AgentOutput(
                success=True,
                data={
                    "override_applied": True,
                    "override_type": "HARD_RULE_REJECTION",
                    "final_decision": "REJECT",
                    "violations": hard_violations
                },
                confidence=1.0,
                flags=[v["rule"] for v in hard_violations],
                reasoning=f"Hard rule violation: {hard_violations[0]['description']}",
                execution_time_ms=execution_time
            )
        
        # Check soft rules
        soft_flags = self.check_rules(input_data, self.soft_rules)
        
        # Check preference rules
        preferences = self.check_rules(input_data, self.preference_rules)
        
        # Calculate risk adjustment
        total_adjustment = sum(p.get("bonus", 0) for p in preferences)
        
        # Determine if override needed
        override_applied = False
        override_type = None
        final_decision = None
        
        # Apply preferences
        if total_adjustment > 0.1 and risk_data.get("risk_level") == "MEDIUM_RISK":
            override_applied = True
            override_type = "PREFERENCE_UPGRADE"
            final_decision = "APPROVE"
        
        # Flag for manual review if multiple soft rules triggered
        requires_manual_review = len(soft_flags) >= 2
        
        execution_time = (time.time() - start_time) * 1000
        
        output = AgentOutput(
            success=True,
            data={
                "override_applied": override_applied,
                "override_type": override_type,
                "final_decision": final_decision,
                "hard_violations": hard_violations,
                "soft_flags": soft_flags,
                "preferences_applied": preferences,
                "risk_adjustment": total_adjustment,
                "requires_manual_review": requires_manual_review
            },
            confidence=1.0 if not soft_flags else 0.8,
            flags=[f["rule"] for f in soft_flags],
            reasoning=f"Policy check: {len(soft_flags)} flags, {len(preferences)} preferences",
            execution_time_ms=execution_time
        )
        
        self.log_execution(input_data, output)
        return output
