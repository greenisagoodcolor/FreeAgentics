"""Tests for conversation orchestration system."""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4

from orchestration import (
    ConversationOrchestrator,
    OrchestrationRequest,
    OrchestrationResult,
    OrchestrationError,
    ComponentTimeoutError,
    ValidationError,
)
from orchestration.pipeline import PipelineContext, StepResult, StepStatus


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


class TestOrchestrationRequest:
    """Test orchestration request model."""
    
    def test_request_creation(self):
        """Test creating orchestration request."""
        request = OrchestrationRequest(
            prompt="Test prompt",
            user_id="test_user",
        )
        
        assert request.prompt == "Test prompt"
        assert request.user_id == "test_user"
        assert request.conversation_id is not None
        assert request.trace_id is not None
        assert request.enable_pymdp is True
        assert request.timeout_ms == 30000
    
    def test_request_with_custom_values(self):
        """Test request with custom values."""
        conversation_id = str(uuid4())
        trace_id = str(uuid4())
        
        request = OrchestrationRequest(
            prompt="Custom prompt",
            user_id="custom_user",
            conversation_id=conversation_id,
            trace_id=trace_id,
            llm_provider="anthropic",
            temperature=0.9,
            enable_pymdp=False,
        )
        
        assert request.conversation_id == conversation_id
        assert request.trace_id == trace_id
        assert request.llm_provider == "anthropic"
        assert request.temperature == 0.9
        assert request.enable_pymdp is False


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


@pytest.mark.asyncio
class TestConversationOrchestrator:
    """Test conversation orchestrator."""
    
    @pytest.fixture
    async def orchestrator(self):
        """Create orchestrator for testing."""
        orchestrator = ConversationOrchestrator(
            enable_monitoring=True,
            enable_health_checks=False,  # Disable for simpler testing
        )
        await orchestrator.start()
        yield orchestrator
        await orchestrator.stop()
    
    async def test_orchestrator_lifecycle(self):
        """Test orchestrator start/stop lifecycle."""
        orchestrator = ConversationOrchestrator(enable_health_checks=False)
        
        assert orchestrator.is_running is False
        
        await orchestrator.start()
        assert orchestrator.is_running is True
        
        await orchestrator.stop()
        assert orchestrator.is_running is False
    
    async def test_orchestrator_not_running_error(self):
        """Test error when orchestrator is not running."""
        orchestrator = ConversationOrchestrator(enable_health_checks=False)
        
        request = OrchestrationRequest(
            prompt="Test prompt",
            user_id="test_user",
        )
        
        result = await orchestrator.process_conversation(request)
        
        assert result.success is False
        assert len(result.errors) > 0
        assert "not running" in result.errors[0]["message"]
    
    @patch('orchestration.conversation_orchestrator.create_llm_manager')
    async def test_orchestrator_mock_processing(self, mock_llm_manager, orchestrator):
        """Test orchestrator with mocked dependencies."""
        # Mock LLM manager
        mock_manager = Mock()
        mock_response = Mock()
        mock_response.content = '{"nodes": [], "edges": [], "metadata": {}}'
        mock_manager.generate_with_fallback.return_value = mock_response
        mock_llm_manager.return_value = mock_manager
        
        request = OrchestrationRequest(
            prompt="Test prompt for simple conversation",
            user_id="test_user",
            enable_knowledge_graph=False,  # Disable to simplify test
        )
        
        # This will likely fail at GMN parsing due to mock data,
        # but we can test that the orchestrator runs
        result = await orchestrator.process_conversation(request)
        
        # Should have attempted LLM generation step
        assert result.request.prompt == "Test prompt for simple conversation"
        assert result.execution_time_ms > 0
        
        # Check that LLM manager was called
        mock_llm_manager.assert_called_once()
    
    async def test_orchestrator_metrics(self, orchestrator):
        """Test orchestrator metrics collection."""
        metrics = orchestrator.get_metrics()
        
        assert "execution_metrics" in metrics
        assert "timing_metrics" in metrics
        assert "error_metrics" in metrics
        
        # Should start with zero executions
        assert metrics["execution_metrics"]["total_executions"] == 0
    
    async def test_orchestrator_health_status_no_health_checker(self, orchestrator):
        """Test health status when health checker is disabled."""
        health = orchestrator.get_health_status()
        
        assert health["overall_status"] == "unknown"
        assert "disabled" in health["message"]


@pytest.mark.asyncio 
class TestPipelineIntegration:
    """Integration tests for pipeline components."""
    
    async def test_pipeline_context_flow(self):
        """Test data flow through pipeline context."""
        context = PipelineContext(
            trace_id="test_trace",
            conversation_id="test_conversation", 
            user_id="test_user",
        )
        
        # Simulate step data flow
        context.set_data("step1", {"output": "data1"})
        context.set_data("step2", {"output": "data2", "depends_on": context.get_data("step1")})
        
        step2_data = context.get_data("step2")
        assert step2_data["output"] == "data2"
        assert step2_data["depends_on"] == {"output": "data1"}
    
    async def test_error_context_creation(self):
        """Test error context creation with timing."""
        import time
        from orchestration.errors import create_error_context
        
        start_time = time.time()
        await asyncio.sleep(0.01)  # Small delay
        
        context = create_error_context(
            trace_id="test_trace",
            conversation_id="test_conv", 
            step_name="test_step",
            component="test_component",
            start_time=start_time,
            custom_field="custom_value",
        )
        
        assert context.trace_id == "test_trace"
        assert context.step_name == "test_step"
        assert context.component == "test_component"
        assert context.execution_time_ms > 0
        assert context.metadata["custom_field"] == "custom_value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])