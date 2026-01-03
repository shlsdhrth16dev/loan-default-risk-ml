"""
Base Agent Class for Loan Decision System.

Provides common interface and utilities for all agents.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentOutput:
    """Standard output format for all agents."""
    
    def __init__(
        self,
        success: bool,
        data: Dict[str, Any],
        confidence: float = 1.0,
        flags: list = None,
        reasoning: str = "",
        execution_time_ms: float = 0
    ):
        self.success = success
        self.data = data
        self.confidence = confidence
        self.flags = flags or []
        self.reasoning = reasoning
        self.execution_time_ms = execution_time_ms
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "confidence": self.confidence,
            "flags": self.flags,
            "reasoning": self.reasoning,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp
        }


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    All agents must implement the run() method.
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.logger = logging.getLogger(f"agents.{name}")
    
    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> AgentOutput:
        """
        Execute agent logic.
        
        Args:
            input_data: Input dictionary
        
        Returns:
            AgentOutput with results
        """
        pass
    
    def log_execution(self, input_data: Dict[str, Any], output: AgentOutput):
        """Log agent execution for audit trail."""
        self.logger.info(f"{self.name} executed", extra={
            "agent": self.name,
            "version": self.version,
            "success": output.success,
            "confidence": output.confidence,
            "flags": output.flags,
            "execution_time_ms": output.execution_time_ms
        })
    
    def validate_input(self, input_data: Dict[str, Any], required_fields: list) -> bool:
        """
        Validate input has required fields.
        
        Args:
            input_data: Input dictionary
            required_fields: List of required field names
        
        Returns:
            True if valid, False otherwise
        """
        missing = [f for f in required_fields if f not in input_data]
        if missing:
            self.logger.error(f"Missing required fields: {missing}")
            return False
        return True
