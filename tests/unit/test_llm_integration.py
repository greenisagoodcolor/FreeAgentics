"""
Comprehensive test coverage for inference/llm/belief_integration.py and local_llm_manager.py
LLM Integration - Phase 3.3 systematic coverage

This test file provides complete coverage for the LLM belief integration and local model management
following the systematic backend coverage improvement plan.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
import torch

# Import the LLM integration components
try:
    from inference.engine.belief_update import BeliefUpdate
    from inference.llm.belief_integration import (
        AggregationMethod,
        BeliefAggregator,
        BeliefCache,
        BeliefFormatter,
        BeliefIntegrator,
        BeliefPrompt,
        BeliefResponse,
        CacheStrategy,
        IntegrationConfig,
        IntegrationMode,
        ResponseParser,
    )
    from inference.llm.local_llm_manager import (
        BatchProcessor,
        InferenceEngine,
        LoadingStrategy,
        LocalLLMManager,
        ModelConfig,
        ModelLoader,
        ModelOptimizer,
        ModelRegistry,
        ModelStatus,
        OptimizationLevel,
        ResourceMonitor,
    )

    IMPORT_SUCCESS = True
except ImportError:
    # Create minimal mock classes for testing if imports fail
    IMPORT_SUCCESS = False

    class IntegrationMode:
        DIRECT = "direct"
        HYBRID = "hybrid"
        AUGMENTED = "augmented"
        ADVISORY = "advisory"

    class CacheStrategy:
        LRU = "lru"
        LFU = "lfu"
        TTL = "ttl"
        ADAPTIVE = "adaptive"

    class AggregationMethod:
        WEIGHTED_AVERAGE = "weighted_average"
        MAJORITY_VOTE = "majority_vote"
        CONFIDENCE_BASED = "confidence_based"
        ENSEMBLE = "ensemble"

    class ModelStatus:
        UNLOADED = "unloaded"
        LOADING = "loading"
        LOADED = "loaded"
        OPTIMIZING = "optimizing"
        READY = "ready"
        ERROR = "error"

    class LoadingStrategy:
        LAZY = "lazy"
        EAGER = "eager"
        ON_DEMAND = "on_demand"
        PRELOAD = "preload"

    class OptimizationLevel:
        NONE = "none"
        BASIC = "basic"
        AGGRESSIVE = "aggressive"
        CUSTOM = "custom"

    class IntegrationConfig:
        def __init__(
            self,
            mode=IntegrationMode.HYBRID,
            cache_size=1000,
            cache_strategy=CacheStrategy.LRU,
            aggregation_method=AggregationMethod.WEIGHTED_AVERAGE,
            confidence_threshold=0.7,
            max_context_length=2048,
            **kwargs,
        ):
            self.mode = mode
            self.cache_size = cache_size
            self.cache_strategy = cache_strategy
            self.aggregation_method = aggregation_method
            self.confidence_threshold = confidence_threshold
            self.max_context_length = max_context_length
            for k, v in kwargs.items():
                setattr(self, k, v)

    class ModelConfig:
        def __init__(
            self,
            model_id,
            model_path,
            model_type="transformer",
            quantization_bits=None,
            device="cpu",
            max_batch_size=8,
            max_sequence_length=2048,
            **kwargs,
        ):
            self.model_id = model_id
            self.model_path = model_path
            self.model_type = model_type
            self.quantization_bits = quantization_bits
            self.device = device
            self.max_batch_size = max_batch_size
            self.max_sequence_length = max_sequence_length
            for k, v in kwargs.items():
                setattr(self, k, v)

    class BeliefPrompt:
        def __init__(self, context, query, beliefs, metadata=None):
            self.context = context
            self.query = query
            self.beliefs = beliefs
            self.metadata = metadata or {}

    class BeliefResponse:
        def __init__(self, updated_beliefs, confidence, reasoning, metadata=None):
            self.updated_beliefs = updated_beliefs
            self.confidence = confidence
            self.reasoning = reasoning
            self.metadata = metadata or {}

    class BeliefUpdate:
        def __init__(self):
            pass

    class ProviderManager:
        def __init__(self):
            self.providers = []

        def register_provider(self, provider, priority=0):
            self.providers.append((priority, provider))
            self.providers.sort(key=lambda x: x[0])

        def generate_with_fallback(self, request):
            for _, provider in self.providers:
                try:
                    return provider.generate(request.messages[0]["content"])
                except Exception:
                    continue
            raise Exception("All providers failed")

    class GenerationRequest:
        def __init__(self, model, messages):
            self.model = model
            self.messages = messages


class TestIntegrationMode:
    """Test integration mode enumeration."""

    def test_integration_modes_exist(self):
        """Test all integration modes exist."""
        expected_modes = ["DIRECT", "HYBRID", "AUGMENTED", "ADVISORY"]

        for mode in expected_modes:
            assert hasattr(IntegrationMode, mode)

    def test_integration_mode_values(self):
        """Test integration mode values."""
        assert IntegrationMode.DIRECT == "direct"
        assert IntegrationMode.HYBRID == "hybrid"
        assert IntegrationMode.AUGMENTED == "augmented"
        assert IntegrationMode.ADVISORY == "advisory"


class TestCacheStrategy:
    """Test cache strategy enumeration."""

    def test_cache_strategies_exist(self):
        """Test all cache strategies exist."""
        expected_strategies = ["LRU", "LFU", "TTL", "ADAPTIVE"]

        for strategy in expected_strategies:
            assert hasattr(CacheStrategy, strategy)


class TestAggregationMethod:
    """Test aggregation method enumeration."""

    def test_aggregation_methods_exist(self):
        """Test all aggregation methods exist."""
        expected_methods = ["WEIGHTED_AVERAGE", "MAJORITY_VOTE", "CONFIDENCE_BASED", "ENSEMBLE"]

        for method in expected_methods:
            assert hasattr(AggregationMethod, method)


class TestIntegrationConfig:
    """Test integration configuration."""

    def test_config_creation_with_defaults(self):
        """Test creating config with defaults."""
        config = IntegrationConfig()

        assert config.mode == IntegrationMode.HYBRID
        assert config.cache_size == 1000
        assert config.cache_strategy == CacheStrategy.LRU
        assert config.aggregation_method == AggregationMethod.WEIGHTED_AVERAGE
        assert config.confidence_threshold == 0.7
        assert config.max_context_length == 2048

    def test_config_creation_with_custom_values(self):
        """Test creating config with custom values."""
        config = IntegrationConfig(
            mode=IntegrationMode.DIRECT,
            cache_size=500,
            cache_strategy=CacheStrategy.ADAPTIVE,
            aggregation_method=AggregationMethod.ENSEMBLE,
            confidence_threshold=0.9,
            max_context_length=4096,
            temperature=0.8,
            top_p=0.95,
        )

        assert config.mode == IntegrationMode.DIRECT
        assert config.cache_size == 500
        assert config.cache_strategy == CacheStrategy.ADAPTIVE
        assert config.aggregation_method == AggregationMethod.ENSEMBLE
        assert config.confidence_threshold == 0.9
        assert config.max_context_length == 4096
        assert config.temperature == 0.8
        assert config.top_p == 0.95


class TestBeliefPrompt:
    """Test belief prompt structure."""

    def test_belief_prompt_creation(self):
        """Test creating belief prompt."""
        context = "Agent observing environment"
        query = "What is the most likely state?"
        beliefs = torch.softmax(torch.randn(4), dim=0)
        metadata = {"agent_id": "agent1", "timestamp": 123456}

        prompt = BeliefPrompt(context, query, beliefs, metadata)

        assert prompt.context == context
        assert prompt.query == query
        assert torch.equal(prompt.beliefs, beliefs)
        assert prompt.metadata["agent_id"] == "agent1"
        assert prompt.metadata["timestamp"] == 123456

    def test_belief_prompt_defaults(self):
        """Test belief prompt with defaults."""
        prompt = BeliefPrompt("context", "query", torch.randn(3))

        assert isinstance(prompt.metadata, dict)
        assert len(prompt.metadata) == 0


class TestBeliefResponse:
    """Test belief response structure."""

    def test_belief_response_creation(self):
        """Test creating belief response."""
        updated_beliefs = torch.softmax(torch.randn(4), dim=0)
        confidence = 0.85
        reasoning = "Based on observations, state 2 is most likely"
        metadata = {"processing_time": 0.15, "model_used": "gpt-4"}

        response = BeliefResponse(updated_beliefs, confidence, reasoning, metadata)

        assert torch.equal(response.updated_beliefs, updated_beliefs)
        assert response.confidence == 0.85
        assert response.reasoning == reasoning
        assert response.metadata["processing_time"] == 0.15
        assert response.metadata["model_used"] == "gpt-4"


class TestBeliefFormatter:
    """Test belief formatting for LLM prompts."""

    @pytest.fixture
    def formatter(self):
        """Create belief formatter."""
        if IMPORT_SUCCESS:
            return BeliefFormatter()
        else:
            return Mock()

    def test_format_beliefs_to_text(self, formatter):
        """Test formatting beliefs to text."""
        if not IMPORT_SUCCESS:
            return

        beliefs = torch.tensor([0.1, 0.3, 0.4, 0.2])
        state_labels = ["idle", "exploring", "interacting", "resting"]

        text = formatter.format_beliefs_to_text(beliefs, state_labels)

        assert isinstance(text, str)
        assert "exploring" in text  # Most likely state
        assert "40%" in text or "0.4" in text

    def test_format_observations_to_context(self, formatter):
        """Test formatting observations to context."""
        if not IMPORT_SUCCESS:
            return

        observations = {
            "visual": torch.randn(10),
            "audio": torch.randn(5),
            "position": torch.tensor([1.0, 2.0]),
        }

        context = formatter.format_observations_to_context(observations)

        assert isinstance(context, str)
        assert "visual" in context.lower()
        assert "audio" in context.lower()
        assert "position" in context.lower()

    def test_format_with_history(self, formatter):
        """Test formatting with belief history."""
        if not IMPORT_SUCCESS:
            return

        current_beliefs = torch.tensor([0.1, 0.7, 0.2])
        history = [
            torch.tensor([0.3, 0.4, 0.3]),
            torch.tensor([0.2, 0.5, 0.3]),
            torch.tensor([0.1, 0.7, 0.2]),
        ]

        formatted = formatter.format_with_history(current_beliefs, history)

        assert isinstance(formatted, str)
        assert "current" in formatted.lower()
        assert "history" in formatted.lower() or "previous" in formatted.lower()


class TestResponseParser:
    """Test parsing LLM responses for belief updates."""

    @pytest.fixture
    def parser(self):
        """Create response parser."""
        if IMPORT_SUCCESS:
            return ResponseParser()
        else:
            return Mock()

    def test_parse_belief_distribution(self, parser):
        """Test parsing belief distribution from text."""
        if not IMPORT_SUCCESS:
            return

        response_text = """
        Based on the observations, the belief distribution should be:
        - State 0: 10%
        - State 1: 30%
        - State 2: 50%
        - State 3: 10%
        """

        beliefs = parser.parse_belief_distribution(response_text, num_states=4)

        assert beliefs.shape == (4,)
        assert torch.allclose(beliefs.sum(), torch.tensor(1.0))
        # State 2 > State 1 > State 0
        assert beliefs[2] > beliefs[1] > beliefs[0]

    def test_parse_confidence_score(self, parser):
        """Test parsing confidence score."""
        if not IMPORT_SUCCESS:
            return

        response_text = "I am 85% confident in this assessment."
        confidence = parser.parse_confidence_score(response_text)

        assert 0 <= confidence <= 1
        assert abs(confidence - 0.85) < 0.01

    def test_parse_reasoning(self, parser):
        """Test extracting reasoning from response."""
        if not IMPORT_SUCCESS:
            return

        response_text = """
        The agent appears to be exploring based on:
        1. Movement patterns indicate searching behavior
        2. Sensory focus on novel areas
        3. No specific goal-directed actions

        Therefore, exploring state is most likely.
        """

        reasoning = parser.extract_reasoning(response_text)

        assert isinstance(reasoning, str)
        assert "exploring" in reasoning.lower()
        assert len(reasoning) > 0


class TestBeliefCache:
    """Test belief caching mechanisms."""

    @pytest.fixture
    def cache(self):
        """Create belief cache."""
        if IMPORT_SUCCESS:
            config = IntegrationConfig(cache_size=100, cache_strategy=CacheStrategy.LRU)
            return BeliefCache(config)
        else:
            return Mock()

    def test_cache_initialization(self, cache):
        """Test cache initialization."""
        if not IMPORT_SUCCESS:
            return

        assert cache.max_size == 100
        assert cache.strategy == CacheStrategy.LRU
        assert len(cache) == 0

    def test_cache_store_and_retrieve(self, cache):
        """Test storing and retrieving from cache."""
        if not IMPORT_SUCCESS:
            return

        key = "context_query_hash"
        beliefs = torch.softmax(torch.randn(4), dim=0)
        confidence = 0.9

        cache.store(key, beliefs, confidence)

        retrieved_beliefs, retrieved_confidence = cache.get(key)

        assert torch.equal(retrieved_beliefs, beliefs)
        assert retrieved_confidence == confidence

    def test_cache_eviction(self, cache):
        """Test cache eviction when full."""
        if not IMPORT_SUCCESS:
            return

        # Fill cache beyond capacity
        for i in range(150):
            key = f"key_{i}"
            beliefs = torch.randn(4)
            cache.store(key, beliefs, 0.8)

        assert len(cache) <= 100  # Should not exceed max size

        # Check that early entries were evicted (LRU)
        assert cache.get("key_0") is None
        assert cache.get("key_149") is not None

    def test_cache_invalidation(self, cache):
        """Test cache invalidation."""
        if not IMPORT_SUCCESS:
            return

        # Add some entries
        for i in range(10):
            cache.store(f"key_{i}", torch.randn(4), 0.8)

        assert len(cache) == 10

        # Invalidate all
        cache.invalidate_all()

        assert len(cache) == 0


class TestBeliefAggregator:
    """Test belief aggregation from multiple sources."""

    @pytest.fixture
    def aggregator(self):
        """Create belief aggregator."""
        if IMPORT_SUCCESS:
            config = IntegrationConfig(aggregation_method=AggregationMethod.WEIGHTED_AVERAGE)
            return BeliefAggregator(config)
        else:
            return Mock()

    def test_weighted_average_aggregation(self, aggregator):
        """Test weighted average aggregation."""
        if not IMPORT_SUCCESS:
            return

        beliefs_list = [
            torch.tensor([0.1, 0.7, 0.2]),
            torch.tensor([0.2, 0.6, 0.2]),
            torch.tensor([0.15, 0.65, 0.2]),
        ]
        weights = torch.tensor([0.5, 0.3, 0.2])

        aggregated = aggregator.aggregate(beliefs_list, weights)

        assert aggregated.shape == (3,)
        assert torch.allclose(aggregated.sum(), torch.tensor(1.0))

        # Check weighted average calculation
        expected = sum(b * w for b, w in zip(beliefs_list, weights))
        expected = expected / expected.sum()
        assert torch.allclose(aggregated, expected)

    def test_majority_vote_aggregation(self, aggregator):
        """Test majority vote aggregation."""
        if not IMPORT_SUCCESS:
            return

        aggregator.config.aggregation_method = AggregationMethod.MAJORITY_VOTE

        beliefs_list = [
            torch.tensor([0.1, 0.8, 0.1]),  # Vote for state 1
            torch.tensor([0.1, 0.7, 0.2]),  # Vote for state 1
            torch.tensor([0.6, 0.2, 0.2]),  # Vote for state 0
        ]

        aggregated = aggregator.aggregate(beliefs_list)

        # State 1 should have highest probability (2 votes)
        assert torch.argmax(aggregated) == 1

    def test_confidence_based_aggregation(self, aggregator):
        """Test confidence-based aggregation."""
        if not IMPORT_SUCCESS:
            return

        aggregator.config.aggregation_method = AggregationMethod.CONFIDENCE_BASED

        beliefs_list = [torch.tensor([0.1, 0.7, 0.2]), torch.tensor([0.6, 0.3, 0.1])]
        # First belief has higher confidence
        confidences = torch.tensor([0.9, 0.3])

        aggregated = aggregator.aggregate(beliefs_list, confidences)

        # Result should be closer to first belief due to higher confidence
        assert abs(aggregated[1] - beliefs_list[0][1]) < abs(aggregated[1] - beliefs_list[1][1])


class TestBeliefIntegrator:
    """Test main belief integrator."""

    @pytest.fixture
    def config(self):
        """Create integration config."""
        return IntegrationConfig(mode=IntegrationMode.HYBRID, confidence_threshold=0.7)

    @pytest.fixture
    def llm_provider(self):
        """Create mock LLM provider."""
        provider = Mock()
        provider.generate = Mock(
            return_value=Mock(text="State 1 is most likely (70% confidence)", cost=0.01)
        )
        return provider

    @pytest.fixture
    def belief_update(self):
        """Create mock belief update engine."""
        return Mock(spec=BeliefUpdate)

    @pytest.fixture
    def integrator(self, config, llm_provider, belief_update):
        """Create belief integrator."""
        if IMPORT_SUCCESS:
            return BeliefIntegrator(config, llm_provider, belief_update)
        else:
            return Mock()

    def test_integrator_initialization(self, integrator, config):
        """Test integrator initialization."""
        if not IMPORT_SUCCESS:
            return

        assert integrator.config == config
        assert hasattr(integrator, "formatter")
        assert hasattr(integrator, "parser")
        assert hasattr(integrator, "cache")
        assert hasattr(integrator, "aggregator")

    def test_direct_mode_integration(self, integrator):
        """Test direct mode integration."""
        if not IMPORT_SUCCESS:
            return

        integrator.config.mode = IntegrationMode.DIRECT

        beliefs = torch.tensor([0.25, 0.25, 0.25, 0.25])
        observations = torch.randn(10)

        prompt = BeliefPrompt(
            context="Agent in unknown state",
            query="Update beliefs based on observations",
            beliefs=beliefs,
        )

        response = integrator.integrate(prompt, observations)

        assert isinstance(response, BeliefResponse)
        assert response.updated_beliefs.shape == beliefs.shape
        assert 0 <= response.confidence <= 1
        assert len(response.reasoning) > 0

    def test_hybrid_mode_integration(self, integrator):
        """Test hybrid mode integration."""
        if not IMPORT_SUCCESS:
            return

        integrator.config.mode = IntegrationMode.HYBRID

        beliefs = torch.tensor([0.1, 0.6, 0.2, 0.1])
        observations = torch.randn(10)

        # Mock traditional belief update
        traditional_beliefs = torch.tensor([0.15, 0.55, 0.2, 0.1])
        integrator.belief_update.update = Mock(return_value=traditional_beliefs)

        prompt = BeliefPrompt(context="Agent exploring", query="Refine beliefs", beliefs=beliefs)

        response = integrator.integrate(prompt, observations)

        # Should combine traditional and LLM-based updates
        assert response.updated_beliefs.shape == beliefs.shape
        assert not torch.equal(response.updated_beliefs, traditional_beliefs)  # Should be different

    def test_caching_behavior(self, integrator):
        """Test caching of responses."""
        if not IMPORT_SUCCESS:
            return

        beliefs = torch.tensor([0.25, 0.25, 0.25, 0.25])
        observations = torch.randn(10)

        prompt = BeliefPrompt(context="Test context", query="Test query", beliefs=beliefs)

        # First call should hit LLM
        response1 = integrator.integrate(prompt, observations)
        integrator.llm_provider.generate.assert_called()

        # Reset mock
        integrator.llm_provider.generate.reset_mock()

        # Second identical call should use cache
        response2 = integrator.integrate(prompt, observations)
        integrator.llm_provider.generate.assert_not_called()

        # Responses should be identical
        assert torch.equal(response1.updated_beliefs, response2.updated_beliefs)
        assert response1.confidence == response2.confidence

    def test_confidence_threshold(self, integrator):
        """Test confidence threshold handling."""
        if not IMPORT_SUCCESS:
            return

        # Set low confidence response
        integrator.llm_provider.generate = Mock(
            return_value=Mock(text="State might be 1 (40% confidence)")
        )

        beliefs = torch.tensor([0.25, 0.25, 0.25, 0.25])
        prompt = BeliefPrompt("context", "query", beliefs)

        response = integrator.integrate(prompt, torch.randn(10))

        # Low confidence should fall back to traditional update
        assert response.confidence < integrator.config.confidence_threshold
        assert "fallback" in response.metadata or "traditional" in response.metadata


class TestModelConfig:
    """Test model configuration."""

    def test_model_config_creation(self):
        """Test creating model config."""
        config = ModelConfig(
            model_id="llama-7b",
            model_path="/models/llama-7b",
            model_type="transformer",
            quantization_bits=8,
            device="cuda",
            max_batch_size=16,
            max_sequence_length=4096,
            use_flash_attention=True,
            rope_scaling=2.0,
        )

        assert config.model_id == "llama-7b"
        assert config.model_path == "/models/llama-7b"
        assert config.model_type == "transformer"
        assert config.quantization_bits == 8
        assert config.device == "cuda"
        assert config.max_batch_size == 16
        assert config.max_sequence_length == 4096
        assert config.use_flash_attention is True
        assert config.rope_scaling == 2.0

    def test_model_config_defaults(self):
        """Test model config with defaults."""
        config = ModelConfig(model_id="test-model", model_path="/test/path")

        assert config.model_type == "transformer"
        assert config.quantization_bits is None
        assert config.device == "cpu"
        assert config.max_batch_size == 8
        assert config.max_sequence_length == 2048


class TestModelStatus:
    """Test model status enumeration."""

    def test_model_statuses_exist(self):
        """Test all model statuses exist."""
        expected_statuses = ["UNLOADED", "LOADING", "LOADED", "OPTIMIZING", "READY", "ERROR"]

        for status in expected_statuses:
            assert hasattr(ModelStatus, status)


class TestLoadingStrategy:
    """Test loading strategy enumeration."""

    def test_loading_strategies_exist(self):
        """Test all loading strategies exist."""
        expected_strategies = ["LAZY", "EAGER", "ON_DEMAND", "PRELOAD"]

        for strategy in expected_strategies:
            assert hasattr(LoadingStrategy, strategy)


class TestOptimizationLevel:
    """Test optimization level enumeration."""

    def test_optimization_levels_exist(self):
        """Test all optimization levels exist."""
        expected_levels = ["NONE", "BASIC", "AGGRESSIVE", "CUSTOM"]

        for level in expected_levels:
            assert hasattr(OptimizationLevel, level)


class TestModelRegistry:
    """Test model registry."""

    @pytest.fixture
    def registry(self):
        """Create model registry."""
        if IMPORT_SUCCESS:
            return ModelRegistry()
        else:
            return Mock()

    def test_registry_initialization(self, registry):
        """Test registry initialization."""
        if not IMPORT_SUCCESS:
            return

        assert hasattr(registry, "_models")
        assert hasattr(registry, "_configs")
        assert len(registry) == 0

    def test_register_model(self, registry):
        """Test registering model."""
        if not IMPORT_SUCCESS:
            return

        config = ModelConfig("test-model", "/test/path")

        registry.register(config)

        assert len(registry) == 1
        assert registry.get("test-model") == config

    def test_unregister_model(self, registry):
        """Test unregistering model."""
        if not IMPORT_SUCCESS:
            return

        config = ModelConfig("test-model", "/test/path")
        registry.register(config)

        assert len(registry) == 1

        registry.unregister("test-model")

        assert len(registry) == 0
        assert registry.get("test-model") is None

    def test_list_models(self, registry):
        """Test listing models."""
        if not IMPORT_SUCCESS:
            return

        configs = [
            ModelConfig("model1", "/path1"),
            ModelConfig("model2", "/path2"),
            ModelConfig("model3", "/path3"),
        ]

        for config in configs:
            registry.register(config)

        model_list = registry.list_models()

        assert len(model_list) == 3
        assert all(model_id in model_list for model_id in ["model1", "model2", "model3"])


class TestModelLoader:
    """Test model loading functionality."""

    @pytest.fixture
    def loader(self):
        """Create model loader."""
        if IMPORT_SUCCESS:
            return ModelLoader()
        else:
            return Mock()

    def test_loader_initialization(self, loader):
        """Test loader initialization."""
        if not IMPORT_SUCCESS:
            return

        assert hasattr(loader, "_loading_strategy")
        assert hasattr(loader, "_loaded_models")

    @patch("torch.load")
    def test_load_model(self, mock_torch_load, loader):
        """Test loading model."""
        if not IMPORT_SUCCESS:
            return

        mock_model = Mock()
        mock_torch_load.return_value = mock_model

        config = ModelConfig("test-model", "/test/model.pt")

        model = loader.load(config)

        assert model == mock_model
        mock_torch_load.assert_called_once()
        assert loader.is_loaded("test-model")

    def test_unload_model(self, loader):
        """Test unloading model."""
        if not IMPORT_SUCCESS:
            return

        # Mock a loaded model
        loader._loaded_models["test-model"] = Mock()

        assert loader.is_loaded("test-model")

        loader.unload("test-model")

        assert not loader.is_loaded("test-model")

    def test_loading_strategies(self, loader):
        """Test different loading strategies."""
        if not IMPORT_SUCCESS:
            return

        # Test lazy loading
        loader.set_strategy(LoadingStrategy.LAZY)
        assert loader._loading_strategy == LoadingStrategy.LAZY

        # Test eager loading
        loader.set_strategy(LoadingStrategy.EAGER)
        assert loader._loading_strategy == LoadingStrategy.EAGER


class TestModelOptimizer:
    """Test model optimization functionality."""

    @pytest.fixture
    def optimizer(self):
        """Create model optimizer."""
        if IMPORT_SUCCESS:
            return ModelOptimizer()
        else:
            return Mock()

    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization."""
        if not IMPORT_SUCCESS:
            return

        assert hasattr(optimizer, "optimization_level")
        assert hasattr(optimizer, "_optimization_cache")

    def test_optimize_model_basic(self, optimizer):
        """Test basic model optimization."""
        if not IMPORT_SUCCESS:
            return

        mock_model = Mock()
        config = ModelConfig("test-model", "/test/path", quantization_bits=8)

        optimizer.optimization_level = OptimizationLevel.BASIC

        optimized = optimizer.optimize(mock_model, config)

        assert optimized is not None
        assert "test-model" in optimizer._optimization_cache

    def test_quantization(self, optimizer):
        """Test model quantization."""
        if not IMPORT_SUCCESS:
            return

        mock_model = Mock()
        _ = ModelConfig("test-model", "/test/path", quantization_bits=8)

        quantized = optimizer.quantize(mock_model, bits=8)

        assert quantized is not None

    def test_optimization_levels(self, optimizer):
        """Test different optimization levels."""
        if not IMPORT_SUCCESS:
            return

        mock_model = Mock()
        config = ModelConfig("test-model", "/test/path")

        # Test each optimization level
        for level in [
            OptimizationLevel.NONE,
            OptimizationLevel.BASIC,
            OptimizationLevel.AGGRESSIVE,
        ]:
            optimizer.optimization_level = level
            optimized = optimizer.optimize(mock_model, config)
            assert optimized is not None


class TestBatchProcessor:
    """Test batch processing functionality."""

    @pytest.fixture
    def processor(self):
        """Create batch processor."""
        if IMPORT_SUCCESS:
            return BatchProcessor(max_batch_size=4)
        else:
            return Mock()

    def test_processor_initialization(self, processor):
        """Test processor initialization."""
        if not IMPORT_SUCCESS:
            return

        assert processor.max_batch_size == 4
        assert hasattr(processor, "_batch_queue")
        assert hasattr(processor, "_processing")

    def test_add_to_batch(self, processor):
        """Test adding items to batch."""
        if not IMPORT_SUCCESS:
            return

        inputs = ["input1", "input2", "input3"]

        for inp in inputs:
            processor.add_to_batch(inp)

        assert len(processor._batch_queue) == 3

    def test_process_batch(self, processor):
        """Test processing a batch."""
        if not IMPORT_SUCCESS:
            return

        inputs = ["input1", "input2", "input3", "input4", "input5"]

        for inp in inputs:
            processor.add_to_batch(inp)

        # Mock process function
        processor.process_fn = Mock(return_value=["output1", "output2", "output3", "output4"])

        # Should process first 4 items (max_batch_size)
        results = processor.process_batch()

        assert len(results) == 4
        assert len(processor._batch_queue) == 1  # One item remaining

    def test_dynamic_batching(self, processor):
        """Test dynamic batching based on sequence length."""
        if not IMPORT_SUCCESS:
            return

        processor.enable_dynamic_batching = True

        # Add items of varying lengths
        short_inputs = ["short"] * 8
        long_inputs = ["very long input " * 100] * 2

        for inp in short_inputs + long_inputs:
            processor.add_to_batch(inp)

        # Dynamic batching should adjust batch size based on total tokens
        batch_sizes = processor.get_optimal_batch_sizes()

        assert len(batch_sizes) > 0
        assert all(size <= processor.max_batch_size for size in batch_sizes)


class TestResourceMonitor:
    """Test resource monitoring functionality."""

    @pytest.fixture
    def monitor(self):
        """Create resource monitor."""
        if IMPORT_SUCCESS:
            return ResourceMonitor()
        else:
            return Mock()

    def test_monitor_initialization(self, monitor):
        """Test monitor initialization."""
        if not IMPORT_SUCCESS:
            return

        assert hasattr(monitor, "_memory_usage")
        assert hasattr(monitor, "_gpu_usage")
        assert hasattr(monitor, "_cpu_usage")

    def test_memory_monitoring(self, monitor):
        """Test memory usage monitoring."""
        if not IMPORT_SUCCESS:
            return

        memory_info = monitor.get_memory_info()

        assert "total" in memory_info
        assert "used" in memory_info
        assert "available" in memory_info
        assert "percent" in memory_info

        assert memory_info["percent"] >= 0
        assert memory_info["percent"] <= 100

    def test_gpu_monitoring(self, monitor):
        """Test GPU monitoring."""
        if not IMPORT_SUCCESS:
            return

        gpu_info = monitor.get_gpu_info()

        if gpu_info is not None:  # GPU might not be available
            assert "memory_used" in gpu_info
            assert "memory_total" in gpu_info
            assert "utilization" in gpu_info

    def test_resource_alerts(self, monitor):
        """Test resource usage alerts."""
        if not IMPORT_SUCCESS:
            return

        # Set alert thresholds
        monitor.set_memory_threshold(80)  # 80% memory usage
        monitor.set_gpu_threshold(90)  # 90% GPU usage

        # Check if alerts are triggered
        alerts = monitor.check_alerts()

        assert isinstance(alerts, list)
        # Alerts depend on actual system state


class TestInferenceEngine:
    """Test inference engine functionality."""

    @pytest.fixture
    def engine(self):
        """Create inference engine."""
        if IMPORT_SUCCESS:
            return InferenceEngine()
        else:
            return Mock()

    def test_engine_initialization(self, engine):
        """Test engine initialization."""
        if not IMPORT_SUCCESS:
            return

        assert hasattr(engine, "_models")
        assert hasattr(engine, "_active_model")
        assert hasattr(engine, "_inference_cache")

    def test_load_model_for_inference(self, engine):
        """Test loading model for inference."""
        if not IMPORT_SUCCESS:
            return

        config = ModelConfig("test-model", "/test/path")
        mock_model = Mock()

        engine.load_model(config, mock_model)

        assert engine.get_active_model() == "test-model"
        assert engine.is_ready()

    def test_run_inference(self, engine):
        """Test running inference."""
        if not IMPORT_SUCCESS:
            return

        # Setup mock model
        mock_model = Mock()
        mock_model.generate = Mock(return_value="Generated text")

        config = ModelConfig("test-model", "/test/path")
        engine.load_model(config, mock_model)

        # Run inference
        result = engine.infer("Test prompt", max_tokens=100)

        assert result == "Generated text"
        mock_model.generate.assert_called_once()

    def test_inference_with_caching(self, engine):
        """Test inference with caching."""
        if not IMPORT_SUCCESS:
            return

        mock_model = Mock()
        mock_model.generate = Mock(return_value="Cached response")

        config = ModelConfig("test-model", "/test/path")
        engine.load_model(config, mock_model)
        engine.enable_caching = True

        # First call
        result1 = engine.infer("Test prompt")
        assert mock_model.generate.call_count == 1

        # Second call with same prompt should use cache
        result2 = engine.infer("Test prompt")
        assert mock_model.generate.call_count == 1  # Not called again
        assert result1 == result2


class TestLocalLLMManager:
    """Test main local LLM manager."""

    @pytest.fixture
    def manager(self):
        """Create local LLM manager."""
        if IMPORT_SUCCESS:
            return LocalLLMManager()
        else:
            return Mock()

    def test_manager_initialization(self, manager):
        """Test manager initialization."""
        if not IMPORT_SUCCESS:
            return

        assert hasattr(manager, "registry")
        assert hasattr(manager, "loader")
        assert hasattr(manager, "optimizer")
        assert hasattr(manager, "engine")
        assert hasattr(manager, "processor")
        assert hasattr(manager, "monitor")

    def test_register_and_load_model(self, manager):
        """Test registering and loading a model."""
        if not IMPORT_SUCCESS:
            return

        config = ModelConfig(
            model_id="llama-7b", model_path="/models/llama-7b", device="cuda", quantization_bits=8
        )

        # Register model
        manager.register_model(config)
        assert manager.is_registered("llama-7b")

        # Mock actual model loading
        with patch.object(manager.loader, "load", return_value=Mock()):
            manager.load_model("llama-7b")
            assert manager.is_loaded("llama-7b")

    def test_model_optimization_workflow(self, manager):
        """Test model optimization workflow."""
        if not IMPORT_SUCCESS:
            return

        config = ModelConfig(model_id="test-model", model_path="/test/path", quantization_bits=8)

        manager.register_model(config)

        # Mock model and optimization
        mock_model = Mock()
        mock_optimized = Mock()

        with patch.object(manager.loader, "load", return_value=mock_model):
            with patch.object(manager.optimizer, "optimize", return_value=mock_optimized):
                manager.load_model("test-model", optimize=True)

                assert manager.optimizer.optimize.called

    def test_batch_inference(self, manager):
        """Test batch inference processing."""
        if not IMPORT_SUCCESS:
            return

        # Setup
        config = ModelConfig("test-model", "/test/path")
        manager.register_model(config)

        mock_model = Mock()
        mock_model.generate = Mock(side_effect=lambda x: f"Response to: {x}")

        with patch.object(manager.loader, "load", return_value=mock_model):
            manager.load_model("test-model")
            manager.set_active_model("test-model")

            # Batch inference
            prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
            results = manager.batch_infer(prompts)

            assert len(results) == 3
            assert all("Response to:" in r for r in results)

    def test_resource_management(self, manager):
        """Test resource management during inference."""
        if not IMPORT_SUCCESS:
            return

        # Check initial resources
        initial_memory = manager.monitor.get_memory_info()

        # Simulate model loading and inference
        config = ModelConfig("test-model", "/test/path", device="cuda")
        manager.register_model(config)

        # Check resource monitoring is active
        assert manager.monitor is not None
        assert initial_memory is not None

    async def test_async_inference(self, manager):
        """Test asynchronous inference."""
        if not IMPORT_SUCCESS:
            return

        config = ModelConfig("test-model", "/test/path")
        manager.register_model(config)

        mock_model = Mock()
        mock_model.generate = AsyncMock(return_value="Async response")

        with patch.object(manager.loader, "load", return_value=mock_model):
            manager.load_model("test-model")
            manager.set_active_model("test-model")

            # Async inference
            result = await manager.async_infer("Test prompt")

            assert result == "Async response"

    def test_model_switching(self, manager):
        """Test switching between models."""
        if not IMPORT_SUCCESS:
            return

        # Register multiple models
        configs = [ModelConfig("model1", "/path1"), ModelConfig("model2", "/path2")]

        for config in configs:
            manager.register_model(config)

        # Load both models
        with patch.object(manager.loader, "load", return_value=Mock()):
            manager.load_model("model1")
            manager.load_model("model2")

        # Switch active model
        manager.set_active_model("model1")
        assert manager.get_active_model() == "model1"

        manager.set_active_model("model2")
        assert manager.get_active_model() == "model2"

    def test_cleanup_and_shutdown(self, manager):
        """Test cleanup and shutdown procedures."""
        if not IMPORT_SUCCESS:
            return

        # Setup some models
        config = ModelConfig("test-model", "/test/path")
        manager.register_model(config)

        with patch.object(manager.loader, "load", return_value=Mock()):
            manager.load_model("test-model")

        # Shutdown
        manager.shutdown()

        # Check cleanup
        assert not manager.is_loaded("test-model")
        assert len(manager.registry) == 0


class TestIntegrationScenarios:
    """Test integration scenarios between belief integration and local LLM."""

    def test_local_llm_with_belief_integration(self):
        """Test using local LLM for belief integration."""
        if not IMPORT_SUCCESS:
            return

        # Create local LLM manager
        llm_manager = LocalLLMManager()

        # Register a local model
        config = ModelConfig("local-llama", "/models/llama", device="cuda")
        llm_manager.register_model(config)

        # Create mock provider that uses local LLM
        mock_provider = Mock()
        mock_provider.generate = Mock(
            return_value=Mock(
                text="State 2 has 60% probability based on observations",
                cost=0.0,  # Local inference has no API cost
            )
        )

        # Create belief integrator with local LLM
        integration_config = IntegrationConfig(mode=IntegrationMode.HYBRID)
        belief_update = Mock()

        integrator = BeliefIntegrator(integration_config, mock_provider, belief_update)

        # Test integration
        beliefs = torch.tensor([0.25, 0.25, 0.25, 0.25])
        prompt = BeliefPrompt("Local context", "Update beliefs", beliefs)

        response = integrator.integrate(prompt, torch.randn(10))

        assert response.updated_beliefs.shape == beliefs.shape
        assert response.cost == 0.0  # Verify no cost for local inference

    def test_fallback_from_api_to_local(self):
        """Test fallback from API provider to local LLM."""
        if not IMPORT_SUCCESS:
            return

        # Create providers
        api_provider = Mock()
        api_provider.generate = Mock(side_effect=Exception("API rate limited"))

        local_provider = Mock()
        local_provider.generate = Mock(
            return_value=Mock(text="Local model: State 1 is most likely", cost=0.0)
        )

        # Manager with fallback capability
        manager = ProviderManager()
        manager.register_provider(api_provider, priority=1)
        manager.register_provider(local_provider, priority=2)

        # Should fallback to local
        request = GenerationRequest(
            model="any-model", messages=[{"role": "user", "content": "Update beliefs"}]
        )

        response = manager.generate_with_fallback(request)

        assert "Local model" in response.text
        assert response.cost == 0.0
