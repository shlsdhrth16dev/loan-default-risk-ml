"""
Validation Agent - First line of defense.

Validates input data quality before expensive model inference.
"""
from typing import Dict, Any
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.base_agent import BaseAgent, AgentOutput
from inference.schemas import CATEGORICAL_VALUES
import time


class ValidationAgent(BaseAgent):
    """
    Validates loan application data.
    
    Checks:
    - Required fields present
    - Data types correct
    - Value ranges valid
    - Categorical values allowed
    """
    
    def __init__(self):
        super().__init__(name="ValidationAgent", version="1.0.0")
        
        self.required_fields = [
            "age", "income", "loanamount", "creditscore",
            "monthsemployed", "numcreditlines", "interestrate",
            "loanterm", "dtiratio", "education", "employmenttype",
            "maritalstatus", "hasmortgage", "hasdependents",
            "loanpurpose", "hascosigner"
        ]
        
        self.numeric_ranges = {
            "age": (18, 100),
            "income": (0, 1000000),
            "loanamount": (1000, 100000),
            "creditscore": (300, 850),
            "monthsemployed": (0, 600),
            "numcreditlines": (0, 50),
            "interestrate": (0.01, 0.35),
            "loanterm": (12, 360),
            "dtiratio": (0.0, 1.0)
        }
    
    def run(self, input_data: Dict[str, Any]) -> AgentOutput:
        """
        Validate loan application input.
        
        Args:
            input_data: Loan application data
        
        Returns:
            AgentOutput with validation results
        """
        start_time = time.time()
        errors = []
        warnings = []
        
        # Check required fields
        missing = [f for f in self.required_fields if f not in input_data]
        if missing:
            errors.append(f"Missing required fields: {', '.join(missing)}")
        
        # Validate numeric ranges
        for field, (min_val, max_val) in self.numeric_ranges.items():
            if field in input_data:
                value = input_data[field]
                if not isinstance(value, (int, float)):
                    errors.append(f"{field} must be numeric, got {type(value)}")
                elif value < min_val or value > max_val:
                    errors.append(f"{field}={value} out of range [{min_val}, {max_val}]")
        
        # Validate categorical values
        for field, allowed in CATEGORICAL_VALUES.items():
            if field in input_data:
                value = input_data[field]
                if value not in allowed:
                    errors.append(f"{field}='{value}' not in allowed values: {allowed}")
        
        # Business logic warnings
        if "dtiratio" in input_data and input_data["dtiratio"] > 0.5:
            warnings.append("DTI ratio > 50% is very high")
        
        if "creditscore" in input_data and input_data["creditscore"] < 600:
            warnings.append("Credit score < 600 indicates high risk")
        
        # Determine success
        is_valid = len(errors) == 0
        
        execution_time = (time.time() - start_time) * 1000
        
        output = AgentOutput(
            success=is_valid,
            data={
                "is_valid": is_valid,
                "errors": errors,
                "warnings": warnings
            },
            confidence=1.0 if is_valid else 0.0,
            flags=warnings,
            reasoning=f"Validation {'passed' if is_valid else 'failed'} with {len(errors)} errors, {len(warnings)} warnings",
            execution_time_ms=execution_time
        )
        
        self.log_execution(input_data, output)
        return output
