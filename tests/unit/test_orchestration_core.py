"""Core tests for orchestration system without complex dependencies."""

import asyncio
import pytest

# Import only the core modules without complex dependencies
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
from orchestration.pipeline import (
    PipelineContext,
    StepResult,
    StepStatus,
    PipelineStep,
    ConversationPipeline,
)
from orchestration.monitoring import (
    HealthStatus,
    HealthChecker,
    MetricsCollector,
)


class TestOrchestrationErrors:
    """Test orchestration error types."""

    def test_orchestration_error_creation(self):
        """Test creating orchestration error with context."""
        error = OrchestrationError(
            message="Test error",
            recoverable=True,
            suggested_action="Retry the operation",
        )

        assert str(error) == "Test error"
        assert error.recoverable is True
        assert error.suggested_action == "Retry the operation"

        error_dict = error.to_dict()
        assert error_dict["error_type"] == "OrchestrationError"
        assert error_dict["message"] == "Test error"
        assert error_dict["recoverable"] is True

    def test_component_timeout_error(self):
        """Test component timeout error."""
        error = ComponentTimeoutError(
            component="test_component",
            timeout_ms=5000,
        )

        assert "test_component" in str(error)
        assert "5000.0ms" in str(error)
        assert error.recoverable is True
        assert "timeout" in error.suggested_action.lower()

    def test_validation_error(self):
        """Test validation error."""
        error = ValidationError(
            field="test_field",
            value="invalid_value",
            validation_rule="must be positive",
        )

        assert "test_field" in str(error)
        assert "invalid_value" in str(error)
        assert "must be positive" in str(error)
        assert error.recoverable is False

    def test_pipeline_execution_error(self):
        """Test pipeline execution error."""
        error = PipelineExecutionError(
            step_name="test_step",
            step_index=2,
            total_steps=5,
        )

        assert "test_step" in str(error)
        assert "3/5" in str(error)  # step_index + 1
        assert error.recoverable is True

    def test_fallback_error(self):
        """Test fallback error."""
        primary_error = Exception("Primary failure")
        fallback_errors = [Exception("Fallback 1"), Exception("Fallback 2")]

        error = FallbackError(
            primary_error=primary_error,
            fallback_errors=fallback_errors,
        )

        assert "2 fallback options exhausted" in str(error)
        assert error.recoverable is False
        assert error.primary_error == primary_error
        assert len(error.fallback_errors) == 2

    def test_error_categorization(self):
        """Test error categorization."""
        timeout_error = ComponentTimeoutError("test", 1000)
        validation_error = ValidationError("test", "value", "rule")
        generic_error = Exception("Generic error")

        assert categorize_error(timeout_error) == "timeout"
        assert categorize_error(validation_error) == "validation"
        assert categorize_error(generic_error) == "unknown"

    def test_retryable_error_detection(self):
        """Test retryable error detection."""
        timeout_error = ComponentTimeoutError("test", 1000)
        validation_error = ValidationError("test", "value", "rule")

        assert is_retryable_error(timeout_error) is True
        assert is_retryable_error(validation_error) is False

        # Test string-based detection
        rate_limit_error = Exception("Rate limit exceeded")
        server_error = Exception("500 Internal Server Error")
        auth_error = Exception("401 Unauthorized")

        assert is_retryable_error(rate_limit_error) is True
        assert is_retryable_error(server_error) is True
        assert is_retryable_error(auth_error) is False

    def test_retry_delay_calculation(self):
        """Test retry delay calculation."""
        timeout_error = ComponentTimeoutError("test", 1000)
        generic_error = Exception("Generic error")

        # Should increase with attempt number
        delay1 = get_retry_delay(timeout_error, 0)
        delay2 = get_retry_delay(timeout_error, 1)
        delay3 = get_retry_delay(timeout_error, 2)

        assert delay1 < delay2 < delay3
        assert delay1 >= 1.0  # Base delay
        assert delay3 <= 60.0  # Max delay for timeout errors

        # Generic errors should have different max
        generic_delay3 = get_retry_delay(generic_error, 2)
        assert generic_delay3 <= 45.0

    def test_error_context_creation(self):
        """Test error context creation."""
        import time

        start_time = time.time()

        context = create_error_context(
            trace_id="test_trace",
            conversation_id="test_conv",
            step_name="test_step",
            component="test_component",
            start_time=start_time,
            custom_field="custom_value",
        )

        assert context.trace_id == "test_trace"
        assert context.conversation_id == "test_conv"
        assert context.step_name == "test_step"
        assert context.component == "test_component"
        assert context.execution_time_ms >= 0
        assert context.metadata["custom_field"] == "custom_value"


class TestPipelineContext:
    """Test pipeline context."""

    def test_context_creation(self):
        """Test creating pipeline context."""
        context = PipelineContext(
            trace_id="test_trace",
            conversation_id="test_conversation",
            user_id="test_user",
        )

        assert context.trace_id == "test_trace"
        assert context.conversation_id == "test_conversation"
        assert context.user_id == "test_user"
        assert context.data == {}

    def test_context_data_operations(self):
        """Test context data operations."""
        context = PipelineContext(
            trace_id="test",
            conversation_id="test",
            user_id="test",
        )

        context.set_data("key1", "value1")
        context.set_data("key2", {"nested": "value"})

        assert context.get_data("key1") == "value1"
        assert context.get_data("key2") == {"nested": "value"}
        assert context.get_data("missing", "default") == "default"

    def test_context_to_dict(self):
        """Test context serialization."""
        context = PipelineContext(
            trace_id="test_trace",
            conversation_id="test_conv",
            user_id="test_user",
            metadata={"custom": "value"},
        )
        context.set_data("step1", {"output": "data"})

        context_dict = context.to_dict()

        assert context_dict["trace_id"] == "test_trace"
        assert context_dict["conversation_id"] == "test_conv"
        assert context_dict["user_id"] == "test_user"
        assert context_dict["metadata"]["custom"] == "value"
        assert context_dict["data"]["step1"] == {"output": "data"}


class TestStepResult:
    """Test step result model."""

    def test_step_result_success(self):
        """Test successful step result."""
        result = StepResult(
            step_name="test_step",
            status=StepStatus.COMPLETED,
            output={"data": "test"},
            execution_time_ms=100.0,
        )

        assert result.success is True
        assert result.failed is False
        assert result.step_name == "test_step"
        assert result.output == {"data": "test"}

        result_dict = result.to_dict()
        assert result_dict["success"] is True
        assert result_dict["status"] == "completed"
        assert result_dict["execution_time_ms"] == 100.0

    def test_step_result_failure(self):
        """Test failed step result."""
        error = Exception("Test error")
        result = StepResult(
            step_name="test_step",
            status=StepStatus.FAILED,
            error=error,
            attempt_count=3,
        )

        assert result.success is False
        assert result.failed is True
        assert result.error == error
        assert result.attempt_count == 3

        result_dict = result.to_dict()
        assert result_dict["success"] is False
        assert result_dict["error"] == "Test error"


class MockPipelineStep(PipelineStep):
    """Mock pipeline step for testing."""

    def __init__(self, name: str, should_fail: bool = False, delay: float = 0.0):
        super().__init__(name=name, timeout_ms=1000, max_retries=2)
        self.should_fail = should_fail
        self.delay = delay
        self.call_count = 0

    async def execute(self, context: PipelineContext) -> str:
        """Mock execution."""
        self.call_count += 1

        if self.delay > 0:
            await asyncio.sleep(self.delay)

        if self.should_fail:
            raise Exception(f"Mock failure in {self.name}")

        return f"output_from_{self.name}"


@pytest.mark.asyncio
class TestConversationPipeline:
    """Test conversation pipeline."""

    async def test_pipeline_success(self):
        """Test successful pipeline execution."""
        steps = [
            MockPipelineStep("step1"),
            MockPipelineStep("step2"),
            MockPipelineStep("step3"),
        ]

        pipeline = ConversationPipeline(steps)

        context = PipelineContext(
            trace_id="test_trace",
            conversation_id="test_conv",
            user_id="test_user",
        )

        result = await pipeline.execute(context)

        assert result["success"] is True
        assert result["total_retries"] == 0
        assert len(result["steps"]) == 3

        # Check that all steps completed
        for step_result in result["steps"]:
            assert step_result["success"] is True
            assert step_result["status"] == "completed"

        # Check context data
        assert context.get_data("step1") == "output_from_step1"
        assert context.get_data("step2") == "output_from_step2"
        assert context.get_data("step3") == "output_from_step3"

    async def test_pipeline_step_failure(self):
        """Test pipeline with step failure."""
        steps = [
            MockPipelineStep("step1"),
            MockPipelineStep("step2", should_fail=True),  # This will fail
            MockPipelineStep("step3"),  # Should not execute
        ]

        pipeline = ConversationPipeline(steps)

        context = PipelineContext(
            trace_id="test_trace",
            conversation_id="test_conv",
            user_id="test_user",
        )

        with pytest.raises(PipelineExecutionError):
            await pipeline.execute(context)

        # Step 1 should have completed
        assert context.get_data("step1") == "output_from_step1"
        # Step 2 should have failed, no data stored
        assert context.get_data("step2") is None
        # Step 3 should not have executed
        assert context.get_data("step3") is None

    async def test_pipeline_optional_step_failure(self):
        """Test pipeline with optional step failure."""
        steps = [
            MockPipelineStep("step1"),
            MockPipelineStep("step2", should_fail=True),  # Optional failure
            MockPipelineStep("step3"),
        ]
        steps[1].required = False  # Make step2 optional

        pipeline = ConversationPipeline(steps)

        context = PipelineContext(
            trace_id="test_trace",
            conversation_id="test_conv",
            user_id="test_user",
        )

        result = await pipeline.execute(context)

        # Pipeline should succeed despite step2 failure
        assert result["success"] is True

        # Step 1 and 3 should complete, step 2 should fail
        assert context.get_data("step1") == "output_from_step1"
        assert context.get_data("step2") is None  # Failed
        assert context.get_data("step3") == "output_from_step3"

    async def test_pipeline_timeout(self):
        """Test pipeline step timeout."""
        steps = [
            MockPipelineStep("step1", delay=2.0),  # 2 second delay with 1 second timeout
        ]

        pipeline = ConversationPipeline(steps)

        context = PipelineContext(
            trace_id="test_trace",
            conversation_id="test_conv",
            user_id="test_user",
        )

        with pytest.raises(PipelineExecutionError):
            await pipeline.execute(context)

    async def test_pipeline_retry_logic(self):
        """Test pipeline retry logic."""

        # Step that fails first time but succeeds on retry
        class RetryableStep(PipelineStep):
            def __init__(self):
                super().__init__("retryable_step", max_retries=3)
                self.attempts = 0

            async def execute(self, context):
                self.attempts += 1
                if self.attempts < 2:  # Fail first attempt
                    raise Exception("Temporary failure")
                return "success_after_retry"

        steps = [RetryableStep()]
        pipeline = ConversationPipeline(steps)

        context = PipelineContext(
            trace_id="test_trace",
            conversation_id="test_conv",
            user_id="test_user",
        )

        result = await pipeline.execute(context)

        assert result["success"] is True
        assert result["total_retries"] == 1  # One retry
        assert context.get_data("retryable_step") == "success_after_retry"


class TestHealthChecker:
    """Test health checker."""

    @pytest.fixture
    def health_checker(self):
        """Create health checker for testing."""
        return HealthChecker(check_interval_seconds=0.1)

    def test_component_registration(self, health_checker):
        """Test component registration."""
        health_checker.register_component("test_component")

        health = health_checker.get_component_health("test_component")
        assert health is not None
        assert health.name == "test_component"
        assert health.status == HealthStatus.UNKNOWN

    def test_health_update(self, health_checker):
        """Test updating component health."""
        health_checker.register_component("test_component")

        health_checker.update_component_health(
            name="test_component",
            status=HealthStatus.HEALTHY,
            response_time_ms=50.0,
            error_rate=0.01,
            message="All good",
            custom_field="custom_value",
        )

        health = health_checker.get_component_health("test_component")
        assert health.status == HealthStatus.HEALTHY
        assert health.response_time_ms == 50.0
        assert health.error_rate == 0.01
        assert health.message == "All good"
        assert health.metadata["custom_field"] == "custom_value"

    def test_overall_health_status(self, health_checker):
        """Test overall health status calculation."""
        health_checker.register_component("comp1")
        health_checker.register_component("comp2")
        health_checker.register_component("comp3")

        # All healthy
        health_checker.update_component_health("comp1", HealthStatus.HEALTHY)
        health_checker.update_component_health("comp2", HealthStatus.HEALTHY)
        health_checker.update_component_health("comp3", HealthStatus.HEALTHY)

        overall = health_checker.get_overall_health()
        assert overall["overall_status"] == "healthy"
        assert overall["summary"]["healthy"] == 3

        # One degraded
        health_checker.update_component_health("comp2", HealthStatus.DEGRADED)
        overall = health_checker.get_overall_health()
        assert overall["overall_status"] == "degraded"
        assert overall["summary"]["degraded"] == 1

        # One unhealthy
        health_checker.update_component_health("comp3", HealthStatus.UNHEALTHY)
        overall = health_checker.get_overall_health()
        assert overall["overall_status"] == "unhealthy"
        assert overall["summary"]["unhealthy"] == 1


class TestMetricsCollector:
    """Test metrics collector."""

    @pytest.fixture
    def metrics_collector(self):
        """Create metrics collector for testing."""
        return MetricsCollector(window_size=10)

    def test_execution_tracking(self, metrics_collector):
        """Test execution tracking."""
        conv_id = "test_conversation"

        # Start execution
        metrics_collector.record_execution_start(conv_id)
        metrics = metrics_collector.get_metrics()
        assert metrics.active_conversations == 1

        # End execution successfully
        metrics_collector.record_execution_end(
            conversation_id=conv_id,
            success=True,
            execution_time_ms=100.0,
        )

        metrics = metrics_collector.get_metrics()
        assert metrics.active_conversations == 0
        assert metrics.total_executions == 1
        assert metrics.successful_executions == 1
        assert metrics.failed_executions == 0
        assert metrics.avg_execution_time_ms == 100.0

    def test_error_tracking(self, metrics_collector):
        """Test error tracking."""
        conv_id = "test_conversation"
        error = ComponentTimeoutError("test_component", 1000)

        metrics_collector.record_execution_start(conv_id)
        metrics_collector.record_execution_end(
            conversation_id=conv_id,
            success=False,
            execution_time_ms=200.0,
            error=error,
        )

        metrics = metrics_collector.get_metrics()
        assert metrics.failed_executions == 1
        assert metrics.error_count_by_type["timeout"] == 1

        error_history = metrics_collector.get_error_history()
        assert len(error_history) == 1
        assert error_history[0]["type"] == "timeout"

    def test_component_error_tracking(self, metrics_collector):
        """Test component-specific error tracking."""
        error = ValidationError("field", "value", "rule")

        metrics_collector.record_component_error("test_component", error)

        metrics = metrics_collector.get_metrics()
        assert metrics.error_count_by_component["test_component"] == 1
        assert metrics.error_count_by_type["validation"] == 1

    def test_metrics_reset(self, metrics_collector):
        """Test metrics reset."""
        conv_id = "test_conversation"

        metrics_collector.record_execution_start(conv_id)
        metrics_collector.record_execution_end(conv_id, True, 100.0)

        metrics = metrics_collector.get_metrics()
        assert metrics.total_executions == 1

        metrics_collector.reset_metrics()

        metrics = metrics_collector.get_metrics()
        assert metrics.total_executions == 0
        assert metrics.active_conversations == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
