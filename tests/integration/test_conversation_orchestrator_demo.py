"""Demonstration of the Conversation Orchestrator in action.

This test shows the complete end-to-end flow working with mocked dependencies,
validating the Nemesis Committee implementation without requiring external services.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch

# Import orchestration components directly to avoid dependency issues
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class MockLLMResponse:
    """Mock LLM response for testing."""

    def __init__(self, content: str):
        self.content = content
        self.provider = "mock_provider"


class MockLLMManager:
    """Mock LLM manager for testing."""

    def generate_with_fallback(self, request):
        """Mock LLM generation with valid GMN JSON."""
        # Return a simple but valid GMN specification
        gmn_json = """{
            "nodes": [
                {"id": "state1", "type": "state", "properties": {"num_states": 3}},
                {"id": "obs1", "type": "observation", "properties": {"num_observations": 3}},
                {"id": "action1", "type": "action", "properties": {"num_actions": 2}},
                {"id": "belief1", "type": "belief"},
                {"id": "pref1", "type": "preference", "properties": {"preferred_observation": 0}},
                {"id": "likelihood1", "type": "likelihood"},
                {"id": "transition1", "type": "transition"}
            ],
            "edges": [
                {"source": "state1", "target": "likelihood1", "type": "depends_on"},
                {"source": "likelihood1", "target": "obs1", "type": "generates"},
                {"source": "state1", "target": "transition1", "type": "depends_on"},
                {"source": "action1", "target": "transition1", "type": "depends_on"},
                {"source": "pref1", "target": "obs1", "type": "depends_on"},
                {"source": "belief1", "target": "state1", "type": "depends_on"}
            ],
            "metadata": {"description": "Simple conversation agent model"}
        }"""
        return MockLLMResponse(gmn_json)


class MockPyMDPAgent:
    """Mock PyMDP agent for testing."""

    def __init__(self):
        self.F = 2.5  # Free energy
        self.action = 1
        self.qs = [0.7, 0.2, 0.1]  # Beliefs
        self.policies = [[0], [1]]


class MockInferenceResult:
    """Mock inference result."""

    def __init__(self):

        self.action = 1
        self.beliefs = {"states": [0.7, 0.2, 0.1]}
        self.free_energy = 2.5
        self.confidence = 0.8
        self.metadata = {"inference_time_ms": 45.2, "pymdp_method": "variational_inference"}

    def to_dict(self):
        return {
            "action": self.action,
            "beliefs": self.beliefs,
            "free_energy": self.free_energy,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@pytest.mark.asyncio
class TestConversationOrchestratorDemo:
    """Demonstration tests for the conversation orchestrator."""

    @patch("orchestration.conversation_orchestrator.create_llm_manager")
    @patch("orchestration.conversation_orchestrator.PyMDPAgentFactory")
    @patch("orchestration.conversation_orchestrator.InferenceEngine")
    @patch("orchestration.conversation_orchestrator.KnowledgeGraphUpdater")
    async def test_complete_orchestration_flow(
        self,
        mock_kg_updater_class,
        mock_inference_engine_class,
        mock_factory_class,
        mock_create_llm_manager,
    ):
        """Test the complete orchestration flow with mocked dependencies."""
        from orchestration.conversation_orchestrator import (
            ConversationOrchestrator,
            OrchestrationRequest,
        )

        # Setup mocks
        mock_llm_manager = MockLLMManager()
        mock_create_llm_manager.return_value = mock_llm_manager

        # Mock PyMDP factory
        mock_factory = Mock()
        mock_agent = MockPyMDPAgent()
        mock_factory.create_agent.return_value = mock_agent
        mock_factory.get_metrics.return_value = {"agents_created": 1}
        mock_factory_class.return_value = mock_factory

        # Mock inference engine
        mock_engine = Mock()
        mock_inference_result = MockInferenceResult()
        mock_engine.run_inference.return_value = mock_inference_result
        mock_engine.get_metrics.return_value = {"inferences_completed": 1}
        mock_inference_engine_class.return_value = mock_engine

        # Mock knowledge graph updater
        mock_updater = Mock()
        mock_updater.start = AsyncMock()
        mock_updater.stop = AsyncMock()
        mock_updater.update_from_inference = AsyncMock(
            return_value={
                "entities": ["entity1"],
                "relations": ["relation1"],
                "metadata": {"success": True},
            }
        )
        mock_updater.get_metrics.return_value = {"updates": 1}
        mock_kg_updater_class.return_value = mock_updater

        # Create orchestrator
        orchestrator = ConversationOrchestrator(
            enable_monitoring=True,
            enable_health_checks=False,  # Disable for simpler testing
        )

        try:
            await orchestrator.start()

            # Create a test request
            request = OrchestrationRequest(
                prompt="Create an agent that can help with decision making under uncertainty",
                user_id="test_user_123",
                llm_provider="openai",
                llm_model="gpt-3.5-turbo",
                temperature=0.7,
                enable_pymdp=True,
                enable_knowledge_graph=True,
            )

            # Process the conversation
            result = await orchestrator.process_conversation(request)

            # Verify the result
            assert result is not None
            assert result.success is True
            assert result.response is not None
            assert "Active Inference agent" in result.response
            assert result.execution_time_ms > 0

            # Verify pipeline steps completed
            expected_steps = [
                "llm_generation",
                "gmn_parsing",
                "agent_creation",
                "inference",
                "knowledge_graph_update",
                "response_generation",
            ]

            for step in expected_steps:
                assert step in result.steps_completed, f"Step {step} not completed"

            # Verify components were called
            mock_create_llm_manager.assert_called_once()
            mock_factory.create_agent.assert_called_once()
            mock_engine.run_inference.assert_called_once()
            mock_updater.update_from_inference.assert_called_once()

            # Verify inference result is included
            assert result.inference_result is not None
            assert result.inference_result.action == 1
            assert result.inference_result.confidence == 0.8

            # Verify GMN spec is included
            assert result.gmn_spec is not None
            assert "num_states" in str(result.gmn_spec)

            # Verify metrics are collected
            metrics = orchestrator.get_metrics()
            assert metrics["execution_metrics"]["total_executions"] == 1
            assert metrics["execution_metrics"]["successful_executions"] == 1
            assert metrics["execution_metrics"]["success_rate"] == 1.0

            print("✅ Complete orchestration flow succeeded!")
            print(f"   Response: {result.response[:100]}...")
            print(f"   Execution time: {result.execution_time_ms:.2f}ms")
            print(f"   Steps completed: {len(result.steps_completed)}/6")

        finally:
            await orchestrator.stop()

    @patch("orchestration.conversation_orchestrator.create_llm_manager")
    async def test_orchestration_with_step_failure(self, mock_create_llm_manager):
        """Test orchestration behavior when a step fails."""
        from orchestration.conversation_orchestrator import (
            ConversationOrchestrator,
            OrchestrationRequest,
        )

        # Setup mock that returns invalid GMN to cause parsing failure
        mock_llm_manager = Mock()
        mock_response = Mock()
        mock_response.content = "This is not valid JSON for GMN parsing"
        mock_llm_manager.generate_with_fallback.return_value = mock_response
        mock_create_llm_manager.return_value = mock_llm_manager

        orchestrator = ConversationOrchestrator(
            enable_monitoring=True,
            enable_health_checks=False,
        )

        try:
            await orchestrator.start()

            request = OrchestrationRequest(
                prompt="Test prompt that will cause GMN parsing to fail",
                user_id="test_user",
                enable_knowledge_graph=False,  # Simplify test
            )

            result = await orchestrator.process_conversation(request)

            # Should fail but return a structured result
            assert result is not None
            assert result.success is False
            assert len(result.errors) > 0
            assert result.execution_time_ms > 0

            # Should have completed LLM generation but failed at GMN parsing
            assert "llm_generation" in result.steps_completed
            assert "gmn_parsing" not in result.steps_completed

            # Verify error details
            error = result.errors[0]
            assert "message" in error
            assert "error_type" in error

            # Verify metrics recorded the failure
            metrics = orchestrator.get_metrics()
            assert metrics["execution_metrics"]["failed_executions"] == 1
            assert metrics["execution_metrics"]["success_rate"] == 0.0

            print("✅ Graceful failure handling succeeded!")
            print(f"   Error type: {error.get('error_type', 'unknown')}")
            print(f"   Steps completed: {len(result.steps_completed)}")

        finally:
            await orchestrator.stop()

    async def test_orchestration_error_types(self):
        """Test that orchestration error types work correctly."""
        from orchestration.errors import (
            OrchestrationError,
            ComponentTimeoutError,
            ValidationError,
            PipelineExecutionError,
        )

        # Test different error scenarios
        errors = [
            OrchestrationError("General orchestration error"),
            ComponentTimeoutError("llm_provider", 15000),
            ValidationError("prompt", "", "cannot be empty"),
            PipelineExecutionError("gmn_parsing", 1, 5),
        ]

        for error in errors:
            error_dict = error.to_dict()

            assert "error_type" in error_dict
            assert "message" in error_dict
            assert "recoverable" in error_dict
            assert isinstance(error_dict["recoverable"], bool)

            print(f"✅ {error.__class__.__name__}: {error_dict['message'][:50]}...")

    async def test_orchestration_monitoring(self):
        """Test orchestration monitoring capabilities."""
        from orchestration.monitoring import (
            HealthChecker,
            MetricsCollector,
            HealthStatus,
        )

        # Test health checker
        health_checker = HealthChecker(check_interval_seconds=0.1)
        await health_checker.start()

        # Register some components
        components = ["llm_provider", "gmn_parser", "pymdp_factory"]
        for comp in components:
            health_checker.register_component(comp)
            health_checker.update_component_health(
                name=comp,
                status=HealthStatus.HEALTHY,
                response_time_ms=50.0,
                error_rate=0.01,
            )

        overall_health = health_checker.get_overall_health()
        assert overall_health["overall_status"] == "healthy"
        assert overall_health["summary"]["healthy"] == 3

        await health_checker.stop()

        # Test metrics collector
        metrics_collector = MetricsCollector(window_size=10)

        # Simulate some executions
        for i in range(5):
            conv_id = f"test_conv_{i}"
            metrics_collector.record_execution_start(conv_id)
            metrics_collector.record_execution_end(
                conversation_id=conv_id,
                success=i < 4,  # 4 success, 1 failure
                execution_time_ms=100.0 + i * 10,
            )

        metrics = metrics_collector.get_metrics()
        assert metrics.total_executions == 5
        assert metrics.successful_executions == 4
        assert metrics.failed_executions == 1
        assert 0.7 < metrics.avg_execution_time_ms < 0.9 * 140  # Approximate average

        print("✅ Monitoring capabilities validated!")
        print(f"   Health status: {overall_health['overall_status']}")
        print(f"   Success rate: {metrics.successful_executions/metrics.total_executions:.1%}")


async def run_demo():
    """Run a simple demo of the orchestration system."""
    print("🚀 Starting Conversation Orchestrator Demo")
    print("=" * 50)

    # Run the basic error type test
    test_instance = TestConversationOrchestratorDemo()
    await test_instance.test_orchestration_error_types()

    # Run the monitoring test
    await test_instance.test_orchestration_monitoring()

    print("=" * 50)
    print("✅ Conversation Orchestrator Demo Complete!")
    print("\nThe orchestrator provides:")
    print("• Complete pipeline orchestration (LLM -> GMN -> PyMDP -> Inference -> KG)")
    print("• Comprehensive error handling with circuit breakers")
    print("• Production-ready monitoring and health checking")
    print("• Graceful degradation and fallback strategies")
    print("• Structured error reporting with actionable suggestions")


if __name__ == "__main__":
    # Can be run directly for demo
    asyncio.run(run_demo())
