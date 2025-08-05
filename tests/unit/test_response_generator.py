"""Tests for response generation with performance optimization."""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from agents.inference_engine import InferenceResult
from response_generation import (
    ProductionResponseGenerator,
    ResponseData,
    ResponseOptions,
    ResponseType,
    ConfidenceLevel,
    InMemoryResponseCache,
    StructuredResponseFormatter,
    LLMEnhancedGenerator,
    WebSocketResponseStreamer,
)


class TestProductionResponseGenerator:
    """Test suite for ProductionResponseGenerator following TDD principles."""

    @pytest.fixture
    def mock_inference_result(self):
        """Create mock inference result for testing."""
        return InferenceResult(
            action=1,
            beliefs={"states": [0.7, 0.3]},
            free_energy=2.5,
            confidence=0.85,
            metadata={
                "pymdp_method": "variational_inference",
                "observation": [0],
                "policy_precision": 16.0,
                "action_precision": 16.0,
            },
        )

    @pytest.fixture
    def response_options(self):
        """Create response options for testing."""
        return ResponseOptions(
            narrative_style=True,
            use_natural_language=True,
            enable_caching=True,
            enable_llm_enhancement=True,
            enable_streaming=True,
            trace_id="test-trace-123",
            conversation_id="test-conv-456",
        )

    @pytest.fixture
    def generator(self):
        """Create production response generator with mocked dependencies."""
        mock_formatter = MagicMock(spec=StructuredResponseFormatter)
        mock_cache = MagicMock(spec=InMemoryResponseCache)
        mock_nlg = MagicMock(spec=LLMEnhancedGenerator)
        mock_streamer = MagicMock(spec=WebSocketResponseStreamer)

        return ProductionResponseGenerator(
            formatter=mock_formatter,
            cache=mock_cache,
            nlg_generator=mock_nlg,
            streamer=mock_streamer,
            enable_monitoring=True,
        )

    @pytest.mark.asyncio
    async def test_generate_response_success_path(
        self, generator, mock_inference_result, response_options
    ):
        """Test successful response generation with all features enabled."""
        # Arrange
        original_prompt = "Test prompt for agent"

        # Mock cache miss
        generator.cache.get = AsyncMock(return_value=None)
        generator.cache.set = AsyncMock()

        # Mock formatter response
        from response_generation.models import ResponseMetadata

        mock_metadata = MagicMock(spec=ResponseMetadata)
        mock_metadata.cached = False
        mock_metadata.nlg_enhanced = False
        mock_metadata.errors = []

        mock_response_data = MagicMock(spec=ResponseData)
        mock_response_data.message = "Structured response"
        mock_response_data.response_type = ResponseType.STRUCTURED
        mock_response_data.metadata = mock_metadata
        generator.formatter.format_response = AsyncMock(return_value=mock_response_data)

        # Mock NLG enhancement
        enhanced_message = "Enhanced natural language response"
        generator.nlg_generator.enhance_message = AsyncMock(return_value=enhanced_message)

        # Mock streaming
        generator.streamer.stream_response = AsyncMock()

        # Act
        result = await generator.generate_response(
            inference_result=mock_inference_result,
            original_prompt=original_prompt,
            options=response_options,
        )

        # Assert
        assert result is not None
        assert isinstance(result, ResponseData)

        # Verify formatter was called
        generator.formatter.format_response.assert_called_once()

        # Verify NLG enhancement was called and applied
        generator.nlg_generator.enhance_message.assert_called_once()
        assert result.message == enhanced_message
        # When streaming is enabled, response type becomes STREAMING even if enhanced
        assert result.response_type == ResponseType.STREAMING

        # Verify streaming was attempted
        generator.streamer.stream_response.assert_called_once()

        # Verify caching was attempted
        generator.cache.set.assert_called_once()

        # Verify metrics were updated
        metrics = generator.get_metrics()
        assert metrics["responses_generated"] == 1
        assert metrics["nlg_enhancements"] == 1

    @pytest.mark.asyncio
    async def test_generate_response_cache_hit(
        self, generator, mock_inference_result, response_options
    ):
        """Test response generation with cache hit."""
        # Arrange
        original_prompt = "Test prompt"
        from response_generation.models import ResponseMetadata

        cached_metadata = MagicMock(spec=ResponseMetadata)
        cached_metadata.cached = False  # Will be updated

        cached_response = MagicMock(spec=ResponseData)
        cached_response.metadata = cached_metadata
        cached_response.response_type = ResponseType.STRUCTURED

        generator.cache.get = AsyncMock(return_value=cached_response)

        # Act
        result = await generator.generate_response(
            inference_result=mock_inference_result,
            original_prompt=original_prompt,
            options=response_options,
        )

        # Assert
        assert result is cached_response
        assert result.metadata.cached is True
        assert result.response_type == ResponseType.CACHED

        # Verify formatter was NOT called (cache hit)
        generator.formatter.format_response.assert_not_called()

        # Verify metrics
        metrics = generator.get_metrics()
        assert metrics["cache_hits"] == 1
        assert metrics["cache_misses"] == 0

    @pytest.mark.asyncio
    async def test_generate_response_nlg_failure_with_fallback(
        self, generator, mock_inference_result, response_options
    ):
        """Test response generation when NLG enhancement fails but fallback is enabled."""
        # Arrange
        original_prompt = "Test prompt"

        # Mock cache miss
        generator.cache.get = AsyncMock(return_value=None)

        # Mock formatter success
        from response_generation.models import ResponseMetadata

        mock_metadata = MagicMock(spec=ResponseMetadata)
        mock_metadata.errors = []
        mock_metadata.fallback_used = False

        mock_response_data = MagicMock(spec=ResponseData)
        mock_response_data.message = "Original message"
        mock_response_data.response_type = ResponseType.STRUCTURED
        mock_response_data.metadata = mock_metadata
        generator.formatter.format_response = AsyncMock(return_value=mock_response_data)

        # Mock NLG failure
        generator.nlg_generator.enhance_message = AsyncMock(
            side_effect=Exception("LLM provider unavailable")
        )

        # Enable fallback
        response_options.fallback_on_llm_failure = True

        # Act
        result = await generator.generate_response(
            inference_result=mock_inference_result,
            original_prompt=original_prompt,
            options=response_options,
        )

        # Assert
        assert result is not None
        assert result.message == "Original message"  # Original message preserved
        assert "NLG enhancement failed" in result.metadata.errors[0]
        assert result.metadata.fallback_used is True

        # Verify metrics
        metrics = generator.get_metrics()
        assert metrics["nlg_failures"] == 1

    @pytest.mark.asyncio
    async def test_generate_response_complete_failure_with_fallback(
        self, generator, mock_inference_result, response_options
    ):
        """Test fallback response generation when main pipeline fails."""
        # Arrange
        original_prompt = "Test prompt"

        # Mock cache miss
        generator.cache.get = AsyncMock(return_value=None)

        # Mock formatter failure
        generator.formatter.format_response = AsyncMock(side_effect=Exception("Formatting failed"))

        # Mock fallback response generation
        from response_generation.models import ResponseMetadata

        fallback_metadata = MagicMock(spec=ResponseMetadata)
        fallback_metadata.fallback_used = True

        fallback_response = MagicMock(spec=ResponseData)
        fallback_response.message = "Fallback response"
        fallback_response.metadata = fallback_metadata

        with patch.object(
            generator, "_generate_fallback_response", return_value=fallback_response
        ) as mock_fallback:
            # Act
            result = await generator.generate_response(
                inference_result=mock_inference_result,
                original_prompt=original_prompt,
                options=response_options,
            )

            # Assert
            assert result is fallback_response
            mock_fallback.assert_called_once()

            # Verify metrics
            metrics = generator.get_metrics()
            assert metrics["fallback_responses"] == 1

    @pytest.mark.asyncio
    async def test_generate_response_disabled_features(self, generator, mock_inference_result):
        """Test response generation with all optional features disabled."""
        # Arrange
        options = ResponseOptions(
            enable_caching=False,
            enable_llm_enhancement=False,
            enable_streaming=False,
            use_natural_language=False,
        )

        from response_generation.models import ResponseMetadata

        mock_metadata = MagicMock(spec=ResponseMetadata)
        mock_metadata.cached = False
        mock_metadata.nlg_enhanced = False

        mock_response_data = MagicMock(spec=ResponseData)
        mock_response_data.message = "Basic response"
        mock_response_data.response_type = ResponseType.STRUCTURED
        mock_response_data.metadata = mock_metadata
        generator.formatter.format_response = AsyncMock(return_value=mock_response_data)

        # Act
        result = await generator.generate_response(
            inference_result=mock_inference_result,
            original_prompt="Test prompt",
            options=options,
        )

        # Assert
        assert result is not None

        # Verify optional features were skipped
        generator.cache.get.assert_not_called()
        generator.nlg_generator.enhance_message.assert_not_called()
        generator.streamer.stream_response.assert_not_called()

    def test_cache_key_generation(self, generator, mock_inference_result):
        """Test cache key generation for deterministic caching."""
        # Arrange
        options = ResponseOptions(narrative_style=True, use_natural_language=True)

        # Act
        key1 = generator._generate_cache_key(mock_inference_result, "prompt", options)
        key2 = generator._generate_cache_key(mock_inference_result, "prompt", options)
        key3 = generator._generate_cache_key(mock_inference_result, "different", options)

        # Assert
        assert key1 == key2  # Same inputs should generate same key
        assert key1 != key3  # Different inputs should generate different keys
        assert key1.startswith("response_cache:")

    def test_metrics_tracking(self, generator):
        """Test comprehensive metrics tracking."""
        # Arrange - initial state
        initial_metrics = generator.get_metrics()
        assert initial_metrics["responses_generated"] == 0

        # Act - simulate metric updates
        generator._update_metrics("responses_generated", 1)
        generator._update_metrics("cache_hits", 1)
        generator._update_avg_time(150.0)
        generator._update_avg_time(100.0)

        # Assert
        metrics = generator.get_metrics()
        assert metrics["responses_generated"] == 1
        assert metrics["cache_hits"] == 1
        assert 100.0 < metrics["avg_generation_time_ms"] < 150.0  # Moving average
        assert "cache_hit_rate" in metrics
        assert "nlg_success_rate" in metrics


class TestResponseDataModels:
    """Test response data models and value objects."""

    def test_confidence_level_from_score(self):
        """Test confidence level enumeration from numeric scores."""
        assert ConfidenceLevel.from_score(0.1) == ConfidenceLevel.VERY_LOW
        assert ConfidenceLevel.from_score(0.3) == ConfidenceLevel.LOW
        assert ConfidenceLevel.from_score(0.5) == ConfidenceLevel.MODERATE
        assert ConfidenceLevel.from_score(0.7) == ConfidenceLevel.HIGH
        assert ConfidenceLevel.from_score(0.9) == ConfidenceLevel.VERY_HIGH

    def test_response_data_serialization(self):
        """Test ResponseData serialization for JSON APIs."""
        from response_generation.models import (
            ResponseData,
            ActionExplanation,
            BeliefSummary,
            ConfidenceRating,
            ResponseMetadata,
        )

        # Create test data
        action_explanation = ActionExplanation(
            action=1,
            action_label="Move Forward",
            rationale="Best option based on beliefs",
        )

        belief_summary = BeliefSummary(
            states=[0.7, 0.3],
            entropy=0.61,
            most_likely_state="State 0",
        )

        confidence_rating = ConfidenceRating(
            overall=0.85,
            level=ConfidenceLevel.VERY_HIGH,
            action_confidence=0.85,
            belief_confidence=0.80,
        )

        metadata = ResponseMetadata(
            response_id="test-123",
            generation_time_ms=150.0,
        )

        response_data = ResponseData(
            message="Test response message",
            action_explanation=action_explanation,
            belief_summary=belief_summary,
            confidence_rating=confidence_rating,
            metadata=metadata,
        )

        # Test serialization
        serialized = response_data.to_dict()
        assert isinstance(serialized, dict)
        assert serialized["message"] == "Test response message"
        assert serialized["action_explanation"]["action"] == 1
        assert serialized["confidence_rating"]["overall"] == 0.85
        assert serialized["metadata"]["response_id"] == "test-123"

        # Test JSON serialization
        json_data = response_data.to_json_serializable()
        assert json_data == serialized


class TestPerformanceOptimizations:
    """Test performance optimization features."""

    @pytest.mark.asyncio
    async def test_response_caching_performance(self):
        """Test response caching reduces generation time."""
        # Arrange
        cache = InMemoryResponseCache(max_size=10)

        from response_generation.models import (
            ResponseData,
            ActionExplanation,
            BeliefSummary,
            ConfidenceRating,
            ResponseMetadata,
        )

        test_response = ResponseData(
            message="Cached response",
            action_explanation=ActionExplanation(action=1),
            belief_summary=BeliefSummary(states=[0.5, 0.5], entropy=0.69),
            confidence_rating=ConfidenceRating(
                overall=0.8,
                level=ConfidenceLevel.HIGH,
                action_confidence=0.8,
                belief_confidence=0.75,
            ),
            metadata=ResponseMetadata(response_id="cache-test", generation_time_ms=50.0),
        )

        # Act - Set and get from cache
        await cache.set("test-key", test_response, ttl_seconds=60)

        start_time = time.time()
        cached_result = await cache.get("test-key")
        cache_time = (time.time() - start_time) * 1000

        # Assert
        assert cached_result is not None
        assert cached_result.message == "Cached response"
        assert cache_time < 10.0  # Cache lookup should be very fast (< 10ms)

        # Verify cache stats
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["sets"] == 1

    @pytest.mark.asyncio
    async def test_lazy_loading_simulation(self):
        """Test lazy loading behavior for knowledge graph integration."""
        # This test simulates lazy loading of expensive KG operations

        expensive_operation_called = False

        async def expensive_kg_operation():
            nonlocal expensive_operation_called
            expensive_operation_called = True
            await asyncio.sleep(0.1)  # Simulate expensive operation
            return {"nodes": [], "edges": []}

        # Simulate lazy loading in response formatter
        async def format_with_lazy_kg(include_kg: bool):
            if include_kg:
                kg_data = await expensive_kg_operation()
                return {"message": "Response with KG", "kg_data": kg_data}
            else:
                return {"message": "Response without KG"}

        # Test without KG (lazy loading should not trigger)
        result1 = await format_with_lazy_kg(include_kg=False)
        assert not expensive_operation_called
        assert "kg_data" not in result1

        # Test with KG (lazy loading should trigger)
        result2 = await format_with_lazy_kg(include_kg=True)
        assert expensive_operation_called
        assert "kg_data" in result2

    def test_memory_usage_bounds(self):
        """Test that caching respects memory bounds."""
        # Arrange
        cache = InMemoryResponseCache(max_size=3)  # Small cache for testing

        from response_generation.models import (
            ResponseData,
            ActionExplanation,
            BeliefSummary,
            ConfidenceRating,
            ResponseMetadata,
        )

        # Create test responses
        responses = []
        for i in range(5):  # More than cache size
            response = ResponseData(
                message=f"Response {i}",
                action_explanation=ActionExplanation(action=i),
                belief_summary=BeliefSummary(states=[0.5, 0.5], entropy=0.69),
                confidence_rating=ConfidenceRating(
                    overall=0.8,
                    level=ConfidenceLevel.HIGH,
                    action_confidence=0.8,
                    belief_confidence=0.75,
                ),
                metadata=ResponseMetadata(response_id=f"test-{i}", generation_time_ms=50.0),
            )
            responses.append(response)

        # Act - Fill cache beyond capacity
        async def fill_cache():
            for i, response in enumerate(responses):
                await cache.set(f"key-{i}", response)

        # Run the async function
        asyncio.run(fill_cache())

        # Assert - Cache should respect size bounds
        stats = cache.get_stats()
        assert stats["size"] <= 3  # Should not exceed max_size
        assert stats["evictions"] >= 2  # Should have evicted at least 2 items
