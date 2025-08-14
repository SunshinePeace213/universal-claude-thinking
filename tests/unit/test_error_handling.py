"""
Unit tests for error handling and recovery mechanisms.

Tests error boundaries, circuit breakers, and recovery strategies.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from src.agents.error_handler import (
    ErrorSeverity,
    RecoveryStrategy,
    ErrorContext,
    CircuitBreaker,
    ErrorHandler
)
from src.agents.base import BaseSubAgent, AgentMessage, MessageType


class TestErrorContext:
    """Test error context functionality."""
    
    def test_error_context_creation(self):
        """Test creating error context."""
        context = ErrorContext(
            agent_name="test-agent",
            error_type="ValueError",
            error_message="Test error",
            severity=ErrorSeverity.MEDIUM
        )
        
        assert context.agent_name == "test-agent"
        assert context.error_type == "ValueError"
        assert context.error_message == "Test error"
        assert context.severity == ErrorSeverity.MEDIUM
        assert context.retry_count == 0
        assert context.max_retries == 3
        assert context.recovery_strategy == RecoveryStrategy.RETRY
    
    def test_should_retry(self):
        """Test retry logic."""
        context = ErrorContext(
            agent_name="test-agent",
            error_type="NetworkError",
            error_message="Connection failed",
            severity=ErrorSeverity.MEDIUM,
            recovery_strategy=RecoveryStrategy.RETRY
        )
        
        # Should retry initially
        assert context.should_retry() is True
        
        # Increment retries
        context.retry_count = 1
        assert context.should_retry() is True
        
        context.retry_count = 2
        assert context.should_retry() is True
        
        # Max retries reached
        context.retry_count = 3
        assert context.should_retry() is False
        
        # Different strategy - no retry
        context.recovery_strategy = RecoveryStrategy.FALLBACK
        context.retry_count = 0
        assert context.should_retry() is False


class TestCircuitBreaker:
    """Test circuit breaker pattern."""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        breaker = CircuitBreaker(
            agent_name="test-agent",
            failure_threshold=3,
            recovery_timeout=timedelta(minutes=5)
        )
        
        assert breaker.agent_name == "test-agent"
        assert breaker.state == "CLOSED"
        assert breaker.failure_count == 0
        assert breaker.success_count == 0
    
    def test_circuit_breaker_trip(self):
        """Test circuit breaker tripping on failures."""
        breaker = CircuitBreaker(
            agent_name="test-agent",
            failure_threshold=3
        )
        
        # Initial state is CLOSED
        assert breaker.state == "CLOSED"
        
        # Record failures
        breaker.record_failure()
        assert breaker.failure_count == 1
        assert breaker.state == "CLOSED"
        
        breaker.record_failure()
        assert breaker.failure_count == 2
        assert breaker.state == "CLOSED"
        
        # Trip on threshold
        breaker.record_failure()
        assert breaker.failure_count == 3
        assert breaker.state == "OPEN"
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery."""
        breaker = CircuitBreaker(
            agent_name="test-agent",
            failure_threshold=2,
            recovery_timeout=timedelta(seconds=1),
            half_open_max_calls=2
        )
        
        # Trip the breaker
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == "OPEN"
        
        # Attempt recovery after timeout
        breaker.last_failure_time = datetime.now() - timedelta(seconds=2)
        assert breaker.can_attempt()
        breaker.state = "HALF_OPEN"
        
        # Success in HALF_OPEN
        breaker.record_success()
        assert breaker.success_count == 1
        assert breaker.state == "HALF_OPEN"
        
        # Enough successes to reset
        breaker.record_success()
        assert breaker.state == "CLOSED"
        assert breaker.failure_count == 0
    
    def test_circuit_breaker_half_open_failure(self):
        """Test circuit breaker failing again in HALF_OPEN state."""
        breaker = CircuitBreaker(
            agent_name="test-agent",
            failure_threshold=2
        )
        
        # Trip the breaker
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == "OPEN"
        
        # Move to HALF_OPEN
        breaker.state = "HALF_OPEN"
        
        # Failure in HALF_OPEN immediately trips again
        breaker.record_failure()
        assert breaker.state == "OPEN"


@pytest.mark.asyncio
class TestErrorHandler:
    """Test the main error handler."""
    
    async def test_error_handler_initialization(self):
        """Test error handler initialization."""
        handler = ErrorHandler()
        
        assert isinstance(handler.error_contexts, dict)
        assert isinstance(handler.circuit_breakers, dict)
        assert handler.global_error_count == 0
    
    async def test_handle_agent_error(self):
        """Test handling agent errors."""
        handler = ErrorHandler()
        
        # Handle an error
        context = await handler.handle_error(
            agent_name="test-agent",
            error=ValueError("Test error"),
            severity=ErrorSeverity.MEDIUM
        )
        
        assert context.agent_name == "test-agent"
        assert context.error_type == "ValueError"
        assert context.error_message == "Test error"
        assert handler.global_error_count == 1
        
        # Error should be stored
        assert "test-agent" in handler.error_contexts
    
    async def test_recovery_strategies(self):
        """Test different recovery strategies."""
        handler = ErrorHandler()
        
        # Test RETRY strategy
        context = await handler.handle_error(
            agent_name="agent1",
            error=ConnectionError("Network error"),
            severity=ErrorSeverity.MEDIUM,
            recovery_strategy=RecoveryStrategy.RETRY
        )
        assert context.recovery_strategy == RecoveryStrategy.RETRY
        assert context.should_retry()
        
        # Test FALLBACK strategy
        context = await handler.handle_error(
            agent_name="agent2",
            error=ValueError("Invalid data"),
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.FALLBACK
        )
        assert context.recovery_strategy == RecoveryStrategy.FALLBACK
        
        # Test ISOLATE strategy
        context = await handler.handle_error(
            agent_name="agent3",
            error=RuntimeError("Critical failure"),
            severity=ErrorSeverity.CRITICAL,
            recovery_strategy=RecoveryStrategy.ISOLATE
        )
        assert context.recovery_strategy == RecoveryStrategy.ISOLATE
    
    async def test_cascade_prevention(self):
        """Test that errors in one agent don't cascade."""
        handler = ErrorHandler()
        
        # Create multiple agent errors
        await handler.handle_error(
            agent_name="agent1",
            error=ValueError("Error 1"),
            severity=ErrorSeverity.HIGH
        )
        
        await handler.handle_error(
            agent_name="agent2",
            error=TypeError("Error 2"),
            severity=ErrorSeverity.MEDIUM
        )
        
        # Check isolation
        assert "agent1" in handler.error_contexts
        assert "agent2" in handler.error_contexts
        
        # Check circuit breakers are independent
        breaker1 = handler.get_circuit_breaker("agent1")
        breaker2 = handler.get_circuit_breaker("agent2")
        
        assert breaker1.agent_name == "agent1"
        assert breaker2.agent_name == "agent2"
        assert breaker1.failure_count != breaker2.failure_count
    
    async def test_error_recovery_with_fallback(self):
        """Test error recovery with fallback agents."""
        handler = ErrorHandler()
        
        # Primary agent fails
        context = await handler.handle_error(
            agent_name="primary-agent",
            error=RuntimeError("Primary failed"),
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.FALLBACK
        )
        
        # Suggest fallback
        fallback = await handler.get_fallback_agent("primary-agent")
        assert fallback is not None  # Would be determined by agent capabilities
    
    async def test_health_monitoring(self):
        """Test health monitoring of agents."""
        handler = ErrorHandler()
        
        # Simulate multiple errors for one agent
        for i in range(5):
            await handler.handle_error(
                agent_name="unhealthy-agent",
                error=RuntimeError(f"Error {i}"),
                severity=ErrorSeverity.MEDIUM
            )
        
        # Check circuit breaker tripped
        breaker = handler.get_circuit_breaker("unhealthy-agent")
        assert breaker.state == "OPEN"
        
        # Health check should identify unhealthy agent
        unhealthy = await handler.get_unhealthy_agents()
        assert "unhealthy-agent" in unhealthy
    
    async def test_error_metrics(self):
        """Test error metrics collection."""
        handler = ErrorHandler()
        
        # Generate various errors
        await handler.handle_error(
            agent_name="agent1",
            error=ValueError("Error 1"),
            severity=ErrorSeverity.LOW
        )
        
        await handler.handle_error(
            agent_name="agent2",
            error=TypeError("Error 2"),
            severity=ErrorSeverity.HIGH
        )
        
        await handler.handle_error(
            agent_name="agent1",
            error=RuntimeError("Error 3"),
            severity=ErrorSeverity.CRITICAL
        )
        
        # Get metrics
        metrics = await handler.get_error_metrics()
        
        assert metrics["total_errors"] == 3
        assert metrics["by_agent"]["agent1"] == 2
        assert metrics["by_agent"]["agent2"] == 1
        assert metrics["by_severity"][ErrorSeverity.LOW] == 1
        assert metrics["by_severity"][ErrorSeverity.HIGH] == 1
        assert metrics["by_severity"][ErrorSeverity.CRITICAL] == 1


class TestBaseSubAgentErrorHandling:
    """Test error handling in BaseSubAgent."""
    
    @pytest.mark.asyncio
    async def test_message_error_boundary(self):
        """Test that message handling has error boundaries."""
        # This would test the actual BaseSubAgent.handle_message method
        # with various error conditions
        pass  # Placeholder for integration with actual BaseSubAgent
    
    @pytest.mark.asyncio
    async def test_validation_error_handling(self):
        """Test handling of validation errors."""
        # This would test validation error handling
        pass  # Placeholder for integration with actual validation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])