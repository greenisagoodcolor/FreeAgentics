"""Integration tests for the complete response generation system."""

import pytest
from unittest.mock import patch, MagicMock

from agents.inference_engine import InferenceResult
from orchestration.conversation_orchestrator import (
    ConversationOrchestrator,
    OrchestrationRequest,
)
from response_generation import (
    ProductionResponseGenerator,
    ResponseData,
    ResponseOptions,
    ResponseType,
    ConfidenceLevel,
)


class TestResponseGenerationIntegration:
    """Integration tests for complete conversation flow with enhanced response generation."""
    
    @pytest.mark.asyncio
    async def test_complete_conversation_flow_with_response_generator(self):
        """Test the complete conversation flow with the new ResponseGenerator integrated."""
        # Create orchestrator
        orchestrator = ConversationOrchestrator(
            enable_monitoring=True,
            enable_health_checks=False,  # Disable for simpler test
        )
        
        await orchestrator.start()
        
        try:
            # Create orchestration request
            request = OrchestrationRequest(
                prompt="Navigate to the goal location efficiently",
                user_id="test-user-123",
                conversation_id="test-conv-456",
                llm_provider="mock",
                llm_model="mock-model",
                enable_pymdp=True,
                enable_knowledge_graph=True,
                timeout_ms=30000,
            )
            
            # Mock the LLM manager to avoid actual API calls
            with patch('inference.llm.provider_factory.create_llm_manager') as mock_llm_factory:
                mock_llm_manager = MagicMock()
                mock_response = MagicMock()
                mock_response.content = '''
                {
                  "nodes": [
                    {"id": "state1", "type": "state", "properties": {"num_states": 3}},
                    {"id": "obs1", "type": "observation", "properties": {"num_observations": 3}},
                    {"id": "action1", "type": "action", "properties": {"num_actions": 2}}
                  ],
                  "edges": [
                    {"source": "state1", "target": "obs1", "type": "generates"},
                    {"source": "action1", "target": "state1", "type": "influences"}
                  ],
                  "metadata": {"description": "Navigation agent model"}
                }
                '''
                mock_llm_manager.generate_with_fallback.return_value = mock_response
                mock_llm_factory.return_value = mock_llm_manager
                
                # Execute the complete conversation flow
                result = await orchestrator.process_conversation(request)
            
            # Verify the complete flow succeeded
            assert result.success is True
            assert result.response is not None
            assert len(result.response) > 0
            
            # Verify pipeline steps completed
            expected_steps = [
                "llm_generation",
                "gmn_parsing", 
                "agent_creation",
                "inference",
                "response_generation"
            ]
            
            for step in expected_steps:
                assert step in result.steps_completed
            
            # Verify enhanced response metadata is present
            assert "response_metadata" in result.pipeline_results["steps"][-1]["result"]
            response_metadata = result.pipeline_results["steps"][-1]["result"]["response_metadata"]
            
            # Check that response generation included inference results
            assert response_metadata["based_on_inference"] is True
            assert "action" in response_metadata
            assert "confidence" in response_metadata
            
            # Verify the response mentions Active Inference (should be enhanced)
            assert "Active Inference" in result.response or "agent" in result.response.lower()
            
            # Verify timing metrics
            assert result.execution_time_ms > 0
            assert result.execution_time_ms < 30000  # Should complete within timeout
            
        finally:
            await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_response_generator_standalone_with_real_data(self):
        """Test ResponseGenerator standalone with realistic inference data."""
        # Create realistic inference result
        inference_result = InferenceResult(
            action=1,  # Move forward
            beliefs={"states": [0.1, 0.7, 0.2]},  # High confidence in state 1
            free_energy=1.8,
            confidence=0.85,  # High confidence
            metadata={
                "pymdp_method": "variational_inference",
                "observation": [1],
                "policy_precision": 16.0,
                "action_precision": 16.0,
                "planning_horizon": 3,
            }
        )
        
        # Create response generator with real components
        generator = ProductionResponseGenerator(
            enable_monitoring=True,
        )
        
        # Create response options for comprehensive test
        options = ResponseOptions(
            narrative_style=True,
            use_natural_language=True,
            include_technical_details=True,
            include_alternatives=True,
            include_knowledge_graph=True,
            enable_caching=True,
            enable_llm_enhancement=False,  # Disable LLM to avoid API calls
            enable_streaming=False,
            trace_id="integration-test-123",
            conversation_id="integration-conv-456",
        )
        
        # Generate response
        response_data = await generator.generate_response(
            inference_result=inference_result,
            original_prompt="Navigate to the goal location efficiently",
            options=options,
        )
        
        # Verify response structure
        assert isinstance(response_data, ResponseData)
        assert response_data.message is not None
        assert len(response_data.message) > 0
        
        # Verify action explanation
        assert response_data.action_explanation.action == 1
        assert response_data.action_explanation.action_label is not None
        assert response_data.action_explanation.rationale is not None
        
        # Verify belief summary
        assert response_data.belief_summary.states == [0.1, 0.7, 0.2]
        assert response_data.belief_summary.entropy > 0
        assert response_data.belief_summary.most_likely_state == "State 1"
        
        # Verify confidence rating
        assert response_data.confidence_rating.overall == 0.85
        assert response_data.confidence_rating.level == ConfidenceLevel.VERY_HIGH
        assert response_data.confidence_rating.action_confidence == 0.85
        
        # Verify metadata
        assert response_data.metadata.response_id is not None
        assert response_data.metadata.generation_time_ms > 0
        assert response_data.metadata.trace_id == "integration-test-123"
        assert response_data.metadata.conversation_id == "integration-conv-456"
        
        # Verify optional enrichment data
        assert response_data.knowledge_graph_updates is not None
        assert len(response_data.related_concepts) > 0
        assert "Active Inference" in response_data.related_concepts
        assert len(response_data.suggested_actions) > 0
        
        # Verify response type
        assert response_data.response_type == ResponseType.STRUCTURED
        
        # Verify generator metrics
        metrics = generator.get_metrics()
        assert metrics["responses_generated"] == 1
        assert metrics["avg_generation_time_ms"] > 0
    
    @pytest.mark.asyncio
    async def test_response_generation_performance_benchmarks(self):
        """Test response generation performance meets requirements."""
        # Create multiple inference results for batch testing
        inference_results = []
        for i in range(5):
            result = InferenceResult(
                action=i % 3,
                beliefs={"states": [0.3, 0.4, 0.3]},
                free_energy=2.0 + i * 0.1,
                confidence=0.6 + i * 0.05,
                metadata={
                    "pymdp_method": "variational_inference",
                    "observation": [i % 2],
                }
            )
            inference_results.append(result)
        
        # Create generator with caching enabled
        generator = ProductionResponseGenerator(enable_monitoring=True)
        
        options = ResponseOptions(
            enable_caching=True,
            enable_llm_enhancement=False,  # Disable for performance test
            enable_streaming=False,
        )
        
        # Test generation time for multiple requests
        generation_times = []
        
        for i, inference_result in enumerate(inference_results):
            import time
            start = time.time()
            
            response = await generator.generate_response(
                inference_result=inference_result,
                original_prompt=f"Test prompt {i}",
                options=options,
            )
            
            generation_time = (time.time() - start) * 1000
            generation_times.append(generation_time)
            
            # Verify response was generated
            assert response is not None
            assert response.metadata.generation_time_ms > 0
        
        # Verify performance benchmarks
        avg_time = sum(generation_times) / len(generation_times)
        max_time = max(generation_times)
        
        # Performance requirements from Committee guidance
        assert avg_time < 200.0, f"Average generation time {avg_time:.2f}ms exceeds 200ms limit"
        assert max_time < 500.0, f"Max generation time {max_time:.2f}ms exceeds 500ms limit"
        
        # Verify caching effectiveness
        metrics = generator.get_metrics()
        total_requests = metrics["responses_generated"]
        cache_requests = metrics["cache_hits"] + metrics["cache_misses"]
        
        assert total_requests == 5
        assert cache_requests > 0  # Should have attempted caching
        
        print(f"Performance Results:")
        print(f"  Average generation time: {avg_time:.2f}ms")
        print(f"  Max generation time: {max_time:.2f}ms")
        print(f"  Cache hit rate: {metrics['cache_hit_rate']:.2%}")
    
    @pytest.mark.asyncio
    async def test_response_generation_error_handling(self):
        """Test error handling and fallback behavior."""
        # Create generator that will fail
        generator = ProductionResponseGenerator(enable_monitoring=True)
        
        # Mock formatter to fail
        generator.formatter.format_response = MagicMock(
            side_effect=Exception("Formatter failed")
        )
        
        options = ResponseOptions(
            enable_caching=False,
            enable_llm_enhancement=False,
            fallback_on_llm_failure=True,  # Enable fallbacks
        )
        
        # Create inference result
        inference_result = InferenceResult(
            action=1,
            beliefs={"states": [0.5, 0.5]},
            free_energy=2.0,
            confidence=0.75,
            metadata={"pymdp_method": "variational_inference"}
        )
        
        # Test that fallback response is generated
        response = await generator.generate_response(
            inference_result=inference_result,
            original_prompt="Test error handling",
            options=options,
        )
        
        # Verify fallback response was generated
        assert response is not None
        assert response.message is not None
        assert "Active Inference" in response.message
        assert response.metadata.fallback_used is True
        
        # Verify error metrics
        metrics = generator.get_metrics()
        assert metrics["fallback_responses"] == 1
        assert metrics["fallback_rate"] > 0
    
    @pytest.mark.asyncio 
    async def test_response_caching_effectiveness(self):
        """Test response caching reduces generation time."""
        generator = ProductionResponseGenerator(enable_monitoring=True)
        
        # Create identical inference results for cache testing
        inference_result = InferenceResult(
            action=2,
            beliefs={"states": [0.2, 0.3, 0.5]},
            free_energy=1.5,
            confidence=0.9,
            metadata={"pymdp_method": "variational_inference"}
        )
        
        options = ResponseOptions(
            enable_caching=True,
            cache_ttl_seconds=60,
            enable_llm_enhancement=False,
        )
        
        # First request - should miss cache
        import time
        start1 = time.time()
        response1 = await generator.generate_response(
            inference_result=inference_result,
            original_prompt="Identical prompt for caching test",
            options=options,
        )
        time1 = (time.time() - start1) * 1000
        
        # Second request - should hit cache
        start2 = time.time()
        response2 = await generator.generate_response(
            inference_result=inference_result,
            original_prompt="Identical prompt for caching test",
            options=options,
        )
        time2 = (time.time() - start2) * 1000
        
        # Verify responses
        assert response1 is not None
        assert response2 is not None
        
        # Verify cache effectiveness
        metrics = generator.get_metrics()
        assert metrics["cache_hits"] == 1
        assert metrics["cache_misses"] == 1
        assert metrics["cache_hit_rate"] == 0.5
        
        # Verify second request was faster (cache hit)
        assert time2 < time1, f"Cache hit ({time2:.2f}ms) should be faster than miss ({time1:.2f}ms)"
        
        print(f"Cache Performance:")
        print(f"  First request (miss): {time1:.2f}ms")
        print(f"  Second request (hit): {time2:.2f}ms")
        print(f"  Speed improvement: {((time1 - time2) / time1 * 100):.1f}%")