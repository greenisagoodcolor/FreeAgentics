"""Isolated tests for orchestration modules."""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock
from uuid import uuid4
import sys
import os

# Add the project root to the path to import modules directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Test the error module directly
def test_error_imports():
    """Test that we can import error classes."""
    from orchestration.errors import (
        OrchestrationError,
        ComponentTimeoutError,
        ValidationError,
        PipelineExecutionError,
        FallbackError,
        create_error_context,
        categorize_error,
        is_retryable_error,
        get_retry_delay,
    )
    
    # Test creating basic errors
    basic_error = OrchestrationError("Test error")
    assert str(basic_error) == "Test error"
    
    timeout_error = ComponentTimeoutError("test_component", 5000)
    assert "test_component" in str(timeout_error)
    
    validation_error = ValidationError("field", "value", "rule")
    assert "field" in str(validation_error)
    
    pipeline_error = PipelineExecutionError("step", 1, 3)
    assert "step" in str(pipeline_error)
    
    fallback_error = FallbackError(Exception("primary"), [Exception("fallback")])
    assert "fallback" in str(fallback_error)
    
    print("✓ All error classes imported and instantiated successfully")


def test_pipeline_imports():
    """Test that we can import pipeline classes."""
    from orchestration.pipeline import (
        PipelineContext,
        StepResult,
        StepStatus,
        PipelineStep,
        ConversationPipeline,
    )
    
    # Test creating basic objects
    context = PipelineContext("trace", "conv", "user")
    assert context.trace_id == "trace"
    
    result = StepResult("test_step", StepStatus.COMPLETED)
    assert result.step_name == "test_step"
    assert result.success is True
    
    print("✓ All pipeline classes imported and instantiated successfully")


def test_monitoring_imports():
    """Test that we can import monitoring classes."""
    from orchestration.monitoring import (
        HealthStatus,
        ComponentHealth,
        OrchestrationMetrics,
        HealthChecker,
        MetricsCollector,
    )
    
    # Test creating basic objects
    from datetime import datetime, timezone
    health = ComponentHealth("test", HealthStatus.HEALTHY, datetime.now(timezone.utc))
    assert health.name == "test"
    assert health.status == HealthStatus.HEALTHY
    
    metrics = OrchestrationMetrics()
    assert metrics.total_executions == 0
    
    health_checker = HealthChecker()
    assert health_checker is not None
    
    metrics_collector = MetricsCollector()
    assert metrics_collector is not None
    
    print("✓ All monitoring classes imported and instantiated successfully")


class TestErrorCategories:
    """Test error categorization and handling."""
    
    def test_component_timeout_error(self):
        """Test component timeout error."""
        from orchestration.errors import ComponentTimeoutError, categorize_error
        
        error = ComponentTimeoutError("llm_provider", 15000)
        
        assert "llm_provider" in str(error)
        assert "15000.0ms" in str(error)
        assert error.recoverable is True
        assert categorize_error(error) == "timeout"
    
    def test_validation_error(self):
        """Test validation error.""" 
        from orchestration.errors import ValidationError, categorize_error
        
        error = ValidationError("prompt", "", "cannot be empty")
        
        assert "prompt" in str(error)
        assert "cannot be empty" in str(error)
        assert error.recoverable is False
        assert categorize_error(error) == "validation"
    
    def test_retry_logic(self):
        """Test retry logic for different errors."""
        from orchestration.errors import (
            ComponentTimeoutError,
            ValidationError,
            is_retryable_error,
            get_retry_delay,
        )
        
        timeout_error = ComponentTimeoutError("test", 1000)
        validation_error = ValidationError("test", "value", "rule")
        
        assert is_retryable_error(timeout_error) is True
        assert is_retryable_error(validation_error) is False
        
        # Test delay calculation
        delay1 = get_retry_delay(timeout_error, 0)
        delay2 = get_retry_delay(timeout_error, 1)
        
        assert delay1 > 0
        assert delay2 > delay1
        assert delay2 <= 60.0  # Max delay for timeout errors


class TestPipelineComponents:
    """Test pipeline components in isolation."""
    
    def test_pipeline_context(self):
        """Test pipeline context data flow."""
        from orchestration.pipeline import PipelineContext
        
        context = PipelineContext("trace_123", "conv_456", "user_789")
        
        # Test data operations
        context.set_data("step1", {"result": "success"})
        context.set_data("step2", {"input": context.get_data("step1")})
        
        assert context.get_data("step1") == {"result": "success"}
        assert context.get_data("step2")["input"] == {"result": "success"}
        assert context.get_data("missing", "default") == "default"
        
        # Test serialization
        context_dict = context.to_dict()
        assert context_dict["trace_id"] == "trace_123"
        assert context_dict["conversation_id"] == "conv_456"
        assert context_dict["data"]["step1"] == {"result": "success"}
    
    def test_step_result(self):
        """Test step result model."""
        from orchestration.pipeline import StepResult, StepStatus
        
        # Success result
        success_result = StepResult(
            step_name="test_step",
            status=StepStatus.COMPLETED,
            output="success_output",
            execution_time_ms=125.5,
        )
        
        assert success_result.success is True
        assert success_result.failed is False
        assert success_result.output == "success_output"
        
        result_dict = success_result.to_dict()
        assert result_dict["success"] is True
        assert result_dict["execution_time_ms"] == 125.5
        
        # Failure result
        error = Exception("Test failure")
        failure_result = StepResult(
            step_name="failed_step",
            status=StepStatus.FAILED,
            error=error,
            attempt_count=3,
        )
        
        assert failure_result.success is False
        assert failure_result.failed is True
        assert failure_result.error == error
        assert failure_result.attempt_count == 3


class TestMonitoringComponents:
    """Test monitoring components in isolation."""
    
    def test_health_status(self):
        """Test health status enumeration."""
        from orchestration.monitoring import HealthStatus
        
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"
    
    def test_component_health(self):
        """Test component health model."""
        from orchestration.monitoring import ComponentHealth, HealthStatus
        from datetime import datetime, timezone
        
        health = ComponentHealth(
            name="test_component",
            status=HealthStatus.HEALTHY,
            last_check=datetime.now(timezone.utc),
            response_time_ms=42.5,
            error_rate=0.01,
            message="All systems operational",
        )
        
        assert health.name == "test_component"
        assert health.status == HealthStatus.HEALTHY
        assert health.response_time_ms == 42.5
        assert health.error_rate == 0.01
        
        health_dict = health.to_dict()
        assert health_dict["name"] == "test_component"
        assert health_dict["status"] == "healthy"
        assert health_dict["response_time_ms"] == 42.5
    
    def test_orchestration_metrics(self):
        """Test orchestration metrics model."""
        from orchestration.monitoring import OrchestrationMetrics
        
        metrics = OrchestrationMetrics()
        
        # Initial state
        assert metrics.total_executions == 0
        assert metrics.successful_executions == 0
        assert metrics.failed_executions == 0
        
        # Simulate updates
        metrics.total_executions = 10
        metrics.successful_executions = 8
        metrics.failed_executions = 2
        metrics.avg_execution_time_ms = 150.0
        
        metrics_dict = metrics.to_dict()
        assert metrics_dict["execution_metrics"]["total_executions"] == 10
        assert metrics_dict["execution_metrics"]["success_rate"] == 0.8
        assert metrics_dict["execution_metrics"]["failure_rate"] == 0.2
        assert metrics_dict["timing_metrics"]["avg_execution_time_ms"] == 150.0


class MockStep:
    """Simple mock step for testing."""
    
    def __init__(self, name: str, should_fail: bool = False):
        self.name = name
        self.should_fail = should_fail
        self.call_count = 0
    
    async def execute(self, context):
        """Mock execution."""
        self.call_count += 1
        if self.should_fail:
            raise Exception(f"Mock failure in {self.name}")
        return f"output_{self.name}"


@pytest.mark.asyncio
class TestAsyncComponents:
    """Test async components."""
    
    async def test_health_checker_lifecycle(self):
        """Test health checker start/stop."""
        from orchestration.monitoring import HealthChecker
        
        checker = HealthChecker(check_interval_seconds=0.1)
        
        assert checker._is_running is False
        
        await checker.start()
        assert checker._is_running is True
        
        # Register a component
        checker.register_component("test_component")
        health = checker.get_component_health("test_component")
        assert health is not None
        assert health.name == "test_component"
        
        await checker.stop()
        assert checker._is_running is False
    
    async def test_metrics_collector_execution_tracking(self):
        """Test metrics collector execution tracking."""
        from orchestration.monitoring import MetricsCollector
        
        collector = MetricsCollector(window_size=5)
        
        conv_id = "test_conversation_123"
        
        # Start execution
        collector.record_execution_start(conv_id)
        metrics = collector.get_metrics()
        assert metrics.active_conversations == 1
        
        # End execution
        collector.record_execution_end(
            conversation_id=conv_id,
            success=True,
            execution_time_ms=250.0,
        )
        
        metrics = collector.get_metrics()
        assert metrics.active_conversations == 0
        assert metrics.total_executions == 1
        assert metrics.successful_executions == 1
        assert metrics.avg_execution_time_ms == 250.0
    
    async def test_error_context_with_timing(self):
        """Test error context creation with timing."""
        from orchestration.errors import create_error_context
        import time
        
        start_time = time.time()
        await asyncio.sleep(0.01)  # Small delay
        
        context = create_error_context(
            trace_id="test_trace",
            conversation_id="test_conv",
            step_name="test_step",
            component="test_component",
            start_time=start_time,
            custom_data="test_value",
        )
        
        assert context.trace_id == "test_trace"
        assert context.conversation_id == "test_conv"
        assert context.step_name == "test_step"
        assert context.component == "test_component"
        assert context.execution_time_ms > 0
        assert context.metadata["custom_data"] == "test_value"
        
        context_dict = context.to_dict()
        assert "execution_time_ms" in context_dict
        assert context_dict["execution_time_ms"] > 0


def test_full_error_flow():
    """Test complete error handling flow."""
    from orchestration.errors import (
        OrchestrationError,
        ComponentTimeoutError,
        ValidationError,
        PipelineExecutionError,
        create_error_context,
        categorize_error,
        is_retryable_error,
        get_retry_delay,
    )
    
    # Create various error types
    errors = [
        ComponentTimeoutError("llm_provider", 15000),
        ValidationError("prompt", "", "cannot be empty"),
        PipelineExecutionError("gmn_parsing", 1, 5),
        OrchestrationError("Generic orchestration error"),
    ]
    
    for error in errors:
        # Test categorization
        category = categorize_error(error)
        assert category in ["timeout", "validation", "pipeline_execution", "orchestration"]
        
        # Test retryability
        retryable = is_retryable_error(error)
        assert isinstance(retryable, bool)
        
        # Test delay calculation if retryable
        if retryable:
            delay = get_retry_delay(error, 0)
            assert delay >= 0
        
        # Test serialization
        error_dict = error.to_dict()
        assert "error_type" in error_dict
        assert "message" in error_dict
        assert "recoverable" in error_dict
        
        print(f"✓ {error.__class__.__name__}: {category} (retryable: {retryable})")


if __name__ == "__main__":
    # Run basic import tests
    test_error_imports()
    test_pipeline_imports()
    test_monitoring_imports()
    
    # Run full error flow test
    test_full_error_flow()
    
    print("\n✅ All orchestration components imported and tested successfully!")
    
    # Run pytest for the class-based tests
    pytest.main([__file__, "-v"])