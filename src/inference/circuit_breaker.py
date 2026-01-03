"""
Circuit Breaker Pattern for ML Model Protection.

Prevents cascade failures by temporarily disabling failing models.
"""
import time
import logging
from typing import Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "CLOSED"      # Healthy, accepting requests
    OPEN = "OPEN"          # Unhealthy, rejecting requests  
    HALF_OPEN = "HALF_OPEN"  # Testing if recovered


class CircuitBreaker:
    """
    Circuit breaker for protecting against failing models.
    
    States:
    - CLOSED: Model is healthy, all requests go through
    - OPEN: Model failed too many times, block requests
    - HALF_OPEN: Testing if model recovered
    
    After threshold failures, circuit opens for timeout period.
    Then enters half-open to test recovery.
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        success_threshold: int = 2
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Name of protected component
            failure_threshold: Failures before opening circuit
            timeout_seconds: How long to wait before retry
            success_threshold: Successes needed in half-open to close
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.success_threshold = success_threshold
        
        # State
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_state_change = time.time()
        
        # Metrics
        self.total_requests = 0
        self.total_failures = 0
        self.total_successes = 0
    
    def is_available(self) -> bool:
        """
        Check if requests can go through.
        
        Returns:
            True if circuit allows requests
        """
        self.total_requests += 1
        
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if timeout elapsed
            if time.time() - self.last_failure_time >= self.timeout_seconds:
                logger.info(f"Circuit {self.name}: OPEN → HALF_OPEN (timeout elapsed)")
                self._transition_to(CircuitState.HALF_OPEN)
                return True
            return False
        
        # HALF_OPEN state - allow limited requests
        return True
    
    def record_success(self):
        """Record successful request."""
        self.total_successes += 1
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            
            if self.success_count >= self.success_threshold:
                logger.info(f"Circuit {self.name}: HALF_OPEN → CLOSED (recovered)")
                self._transition_to(CircuitState.CLOSED)
                self.failure_count = 0
                self.success_count = 0
        
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def record_failure(self):
        """Record failed request."""
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            # Failed during recovery test - reopen
            logger.warning(f"Circuit {self.name}: HALF_OPEN → OPEN (recovery failed)")
            self._transition_to(CircuitState.OPEN)
            self.success_count = 0
        
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                logger.error(f"Circuit {self.name}: CLOSED → OPEN ({self.failure_count} failures)")
                self._transition_to(CircuitState.OPEN)
    
    def _transition_to(self, new_state: CircuitState):
        """Transition to new state."""
        old_state = self.state
        self.state = new_state
        self.last_state_change = time.time()
        
        logger.info(f"Circuit {self.name}: State transition {old_state.value} → {new_state.value}")
    
    def get_metrics(self) -> dict:
        """Get circuit breaker metrics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "failure_rate": (
                self.total_failures / self.total_requests 
                if self.total_requests > 0 else 0
            ),
            "last_state_change": datetime.fromtimestamp(self.last_state_change).isoformat()
        }
    
    def reset(self):
        """Manually reset circuit (for testing or ops intervention)."""
        logger.info(f"Circuit {self.name}: Manual reset to CLOSED")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
