"""
Error Handler for sub-agent isolation and recovery.

Implements error boundaries, cascade prevention, and recovery mechanisms
to ensure system resilience.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import traceback


logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"          # Can be ignored or logged
    MEDIUM = "medium"    # Should be handled but not critical
    HIGH = "high"        # Requires immediate attention
    CRITICAL = "critical"  # System-threatening, requires intervention


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"              # Retry the operation
    FALLBACK = "fallback"        # Use fallback agent/method
    DEGRADE = "degrade"          # Continue with reduced functionality
    ISOLATE = "isolate"          # Isolate the failing component
    RESTART = "restart"          # Restart the agent
    ESCALATE = "escalate"        # Escalate to human/higher level


@dataclass
class ErrorContext:
    """Context information for an error."""
    agent_name: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    timestamp: datetime = field(default_factory=datetime.now)
    stack_trace: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def should_retry(self) -> bool:
        """Check if operation should be retried."""
        return (self.recovery_strategy == RecoveryStrategy.RETRY and
                self.retry_count < self.max_retries)


@dataclass 
class CircuitBreaker:
    """
    Circuit breaker pattern for preventing cascade failures.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failures exceeded threshold, requests blocked
    - HALF_OPEN: Testing if service recovered
    """
    
    agent_name: str
    failure_threshold: int = 5
    recovery_timeout: timedelta = timedelta(minutes=5)
    half_open_max_calls: int = 3
    
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    success_count: int = 0
    
    def record_success(self) -> None:
        """Record a successful call."""
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= self.half_open_max_calls:
                self.reset()
        elif self.state == "CLOSED":
            self.failure_count = 0
    
    def record_failure(self) -> None:
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == "HALF_OPEN":
            self.trip()
        elif self.state == "CLOSED" and self.failure_count >= self.failure_threshold:
            self.trip()
    
    def trip(self) -> None:
        """Trip the circuit breaker to OPEN state."""
        self.state = "OPEN"
        logger.warning(f"Circuit breaker tripped for {self.agent_name}")
    
    def reset(self) -> None:
        """Reset the circuit breaker to CLOSED state."""
        self.state = "CLOSED"
        self.failure_count = 0
        self.success_count = 0
        logger.info(f"Circuit breaker reset for {self.agent_name}")
    
    def can_proceed(self) -> bool:
        """Check if request can proceed through circuit."""
        if self.state == "CLOSED":
            return True
        
        if self.state == "OPEN":
            if self.last_failure_time:
                time_since_failure = datetime.now() - self.last_failure_time
                if time_since_failure > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    self.success_count = 0
                    logger.info(f"Circuit breaker half-open for {self.agent_name}")
                    return True
            return False
        
        # HALF_OPEN state
        return self.success_count < self.half_open_max_calls


class ErrorHandler:
    """
    Centralized error handling for sub-agents.
    
    Provides error boundaries, recovery mechanisms, and cascade prevention.
    """
    
    def __init__(self):
        """Initialize the error handler."""
        self.error_history: List[ErrorContext] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.recovery_handlers: Dict[RecoveryStrategy, Callable] = {}
        self.fallback_agents: Dict[str, str] = {}  # agent -> fallback mapping
        
        # Register default recovery handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self) -> None:
        """Register default recovery strategy handlers."""
        self.recovery_handlers[RecoveryStrategy.RETRY] = self._retry_handler
        self.recovery_handlers[RecoveryStrategy.FALLBACK] = self._fallback_handler
        self.recovery_handlers[RecoveryStrategy.DEGRADE] = self._degrade_handler
        self.recovery_handlers[RecoveryStrategy.ISOLATE] = self._isolate_handler
        self.recovery_handlers[RecoveryStrategy.RESTART] = self._restart_handler
        self.recovery_handlers[RecoveryStrategy.ESCALATE] = self._escalate_handler
    
    async def handle_error(
        self,
        agent_name: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorContext:
        """
        Handle an error from a sub-agent.
        
        Args:
            agent_name: Name of the agent that encountered the error
            error: The exception that occurred
            context: Additional context information
            
        Returns:
            ErrorContext with recovery information
        """
        # Create error context
        error_ctx = ErrorContext(
            agent_name=agent_name,
            error_type=type(error).__name__,
            error_message=str(error),
            severity=self._determine_severity(error),
            stack_trace=traceback.format_exc(),
            recovery_strategy=self._determine_recovery_strategy(error),
            metadata=context or {}
        )
        
        # Store in history
        self.error_history.append(error_ctx)
        
        # Update circuit breaker
        circuit_breaker = self.get_or_create_circuit_breaker(agent_name)
        circuit_breaker.record_failure()
        
        # Log the error
        logger.error(
            f"Error in {agent_name}: {error_ctx.error_type} - {error_ctx.error_message}"
        )
        
        # Apply recovery strategy
        await self.apply_recovery(error_ctx)
        
        return error_ctx
    
    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine the severity of an error."""
        error_type = type(error).__name__
        
        # Critical errors
        if error_type in ["SystemError", "MemoryError", "RecursionError"]:
            return ErrorSeverity.CRITICAL
        
        # High severity
        if error_type in ["ValueError", "TypeError", "AttributeError"]:
            return ErrorSeverity.HIGH
        
        # Medium severity
        if error_type in ["TimeoutError", "ConnectionError"]:
            return ErrorSeverity.MEDIUM
        
        # Default to low
        return ErrorSeverity.LOW
    
    def _determine_recovery_strategy(self, error: Exception) -> RecoveryStrategy:
        """Determine the appropriate recovery strategy for an error."""
        error_type = type(error).__name__
        
        # Timeout errors - retry
        if error_type in ["TimeoutError", "asyncio.TimeoutError"]:
            return RecoveryStrategy.RETRY
        
        # Connection errors - fallback
        if error_type in ["ConnectionError", "HTTPError"]:
            return RecoveryStrategy.FALLBACK
        
        # Resource errors - degrade
        if error_type in ["MemoryError", "ResourceWarning"]:
            return RecoveryStrategy.DEGRADE
        
        # System errors - isolate
        if error_type in ["SystemError", "RecursionError"]:
            return RecoveryStrategy.ISOLATE
        
        # Default to retry
        return RecoveryStrategy.RETRY
    
    async def apply_recovery(self, error_ctx: ErrorContext) -> bool:
        """
        Apply recovery strategy for an error.
        
        Args:
            error_ctx: Error context with recovery information
            
        Returns:
            True if recovery successful, False otherwise
        """
        strategy = error_ctx.recovery_strategy
        
        if strategy in self.recovery_handlers:
            handler = self.recovery_handlers[strategy]
            return await handler(error_ctx)
        
        logger.warning(f"No handler for recovery strategy: {strategy}")
        return False
    
    async def _retry_handler(self, error_ctx: ErrorContext) -> bool:
        """Handle retry recovery strategy."""
        if not error_ctx.should_retry():
            logger.warning(
                f"Max retries ({error_ctx.max_retries}) reached for {error_ctx.agent_name}"
            )
            # Switch to fallback strategy
            error_ctx.recovery_strategy = RecoveryStrategy.FALLBACK
            return await self._fallback_handler(error_ctx)
        
        error_ctx.retry_count += 1
        wait_time = 2 ** error_ctx.retry_count  # Exponential backoff
        
        logger.info(
            f"Retrying {error_ctx.agent_name} (attempt {error_ctx.retry_count}/"
            f"{error_ctx.max_retries}) after {wait_time}s"
        )
        
        await asyncio.sleep(wait_time)
        return True
    
    async def _fallback_handler(self, error_ctx: ErrorContext) -> bool:
        """Handle fallback recovery strategy."""
        fallback_agent = self.fallback_agents.get(error_ctx.agent_name)
        
        if fallback_agent:
            logger.info(f"Falling back from {error_ctx.agent_name} to {fallback_agent}")
            error_ctx.metadata["fallback_agent"] = fallback_agent
            return True
        
        logger.warning(f"No fallback agent configured for {error_ctx.agent_name}")
        return False
    
    async def _degrade_handler(self, error_ctx: ErrorContext) -> bool:
        """Handle degrade recovery strategy."""
        logger.info(f"Degrading functionality for {error_ctx.agent_name}")
        error_ctx.metadata["degraded"] = True
        return True
    
    async def _isolate_handler(self, error_ctx: ErrorContext) -> bool:
        """Handle isolate recovery strategy."""
        logger.warning(f"Isolating {error_ctx.agent_name} due to errors")
        
        # Trip circuit breaker to prevent further calls
        circuit_breaker = self.get_or_create_circuit_breaker(error_ctx.agent_name)
        circuit_breaker.trip()
        
        error_ctx.metadata["isolated"] = True
        return True
    
    async def _restart_handler(self, error_ctx: ErrorContext) -> bool:
        """Handle restart recovery strategy."""
        logger.info(f"Restarting {error_ctx.agent_name}")
        
        # Reset circuit breaker
        circuit_breaker = self.get_or_create_circuit_breaker(error_ctx.agent_name)
        circuit_breaker.reset()
        
        error_ctx.metadata["restarted"] = True
        return True
    
    async def _escalate_handler(self, error_ctx: ErrorContext) -> bool:
        """Handle escalate recovery strategy."""
        logger.critical(
            f"Escalating error from {error_ctx.agent_name}: {error_ctx.error_message}"
        )
        
        error_ctx.metadata["escalated"] = True
        
        # Could trigger alerts, notifications, etc.
        return False
    
    def get_or_create_circuit_breaker(self, agent_name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for an agent."""
        if agent_name not in self.circuit_breakers:
            self.circuit_breakers[agent_name] = CircuitBreaker(agent_name)
        return self.circuit_breakers[agent_name]
    
    def can_proceed(self, agent_name: str) -> bool:
        """Check if an agent can proceed (circuit breaker check)."""
        circuit_breaker = self.get_or_create_circuit_breaker(agent_name)
        return circuit_breaker.can_proceed()
    
    def record_success(self, agent_name: str) -> None:
        """Record a successful operation for an agent."""
        circuit_breaker = self.get_or_create_circuit_breaker(agent_name)
        circuit_breaker.record_success()
    
    def set_fallback(self, agent_name: str, fallback_agent: str) -> None:
        """Set a fallback agent for error recovery."""
        self.fallback_agents[agent_name] = fallback_agent
        logger.info(f"Set fallback for {agent_name} to {fallback_agent}")
    
    def get_error_stats(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Get error statistics for an agent or all agents."""
        if agent_name:
            errors = [e for e in self.error_history if e.agent_name == agent_name]
        else:
            errors = self.error_history
        
        if not errors:
            return {"error_count": 0}
        
        severity_counts = {}
        strategy_counts = {}
        
        for error in errors:
            # Count by severity
            severity = error.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Count by recovery strategy
            strategy = error.recovery_strategy.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return {
            "error_count": len(errors),
            "severity_distribution": severity_counts,
            "recovery_strategies": strategy_counts,
            "circuit_breaker_states": {
                name: cb.state for name, cb in self.circuit_breakers.items()
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on error handling system."""
        return {
            "total_errors": len(self.error_history),
            "recent_errors": len([
                e for e in self.error_history
                if (datetime.now() - e.timestamp) < timedelta(minutes=5)
            ]),
            "circuit_breakers": {
                name: {
                    "state": cb.state,
                    "failure_count": cb.failure_count,
                    "can_proceed": cb.can_proceed()
                }
                for name, cb in self.circuit_breakers.items()
            },
            "critical_errors": len([
                e for e in self.error_history
                if e.severity == ErrorSeverity.CRITICAL
            ])
        }