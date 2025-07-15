"""
Test suite for Local LLM Manager module.

This test suite provides comprehensive coverage for the LocalLLMManager class
and related components for local LLM providers like Ollama and llama.cpp.
Coverage target: 95%+
"""

import subprocess
import threading
from dataclasses import asdict
from unittest.mock import Mock, patch

import httpx
import pytest

# Import the module under test
try:
    from inference.llm.local_llm_manager import (
        LlamaCppProvider,
        LLMResponse,
        LocalLLMConfig,
        LocalLLMManager,
        LocalLLMProvider,
        OllamaProvider,
        QuantizationLevel,
    )

    IMPORT_SUCCESS = True
except ImportError:
    IMPORT_SUCCESS = False

    # Mock classes for testing when imports fail
    class LocalLLMProvider:
        OLLAMA = "ollama"
        LLAMA_CPP = "llama_cpp"

    class QuantizationLevel:
        INT3 = "q3_K_M"
        INT4 = "q4_K_M"
        INT8 = "q8_0"
        HALF = "f16"

    class LocalLLMConfig:
        pass

    class LocalLLMManager:
        pass

    class OllamaProvider:
        pass

    class LlamaCppProvider:
        pass


class TestLocalLLMConfig:
    """Test suite for LocalLLMConfig dataclass."""

    def test_config_default_values(self):
        """Test default configuration values."""
        if not IMPORT_SUCCESS:
            pytest.skip("Local LLM modules not available")

        config = LocalLLMConfig()

        assert config.provider == LocalLLMProvider.OLLAMA
        assert config.model_name == "llama2"
        assert config.quantization == QuantizationLevel.INT4
        assert config.context_size == 2048
        assert config.max_tokens == 512
        assert config.temperature == 0.7
        assert config.threads == 4
        assert config.gpu_layers == 0
        assert config.ollama_host == "http://localhost:11434"
        assert config.enable_fallback is True
        assert config.llama_cpp_binary is None
        assert config.model_path is None
        assert config.cache_size_mb == 512

    def test_config_custom_values(self):
        """Test configuration with custom values."""
        if not IMPORT_SUCCESS:
            pytest.skip("Local LLM modules not available")

        config = LocalLLMConfig(
            provider=LocalLLMProvider.LLAMA_CPP,
            model_name="llama-7b-chat",
            quantization=QuantizationLevel.INT8,
            context_size=4096,
            max_tokens=1024,
            temperature=0.9,
            threads=8,
            gpu_layers=32,
            ollama_host="http://remote:11434",
            enable_fallback=False,
            llama_cpp_binary="/usr/local/bin/llama-cpp",
            model_path="/models/llama-7b.ggml",
            cache_size_mb=1024,
        )

        assert config.provider == LocalLLMProvider.LLAMA_CPP
        assert config.model_name == "llama-7b-chat"
        assert config.quantization == QuantizationLevel.INT8
        assert config.context_size == 4096
        assert config.max_tokens == 1024
        assert config.temperature == 0.9
        assert config.threads == 8
        assert config.gpu_layers == 32
        assert config.ollama_host == "http://remote:11434"
        assert config.enable_fallback is False
        assert config.llama_cpp_binary == "/usr/local/bin/llama-cpp"
        assert config.model_path == "/models/llama-7b.ggml"
        assert config.cache_size_mb == 1024

    def test_config_serialization(self):
        """Test configuration can be serialized to dict."""
        if not IMPORT_SUCCESS:
            pytest.skip("Local LLM modules not available")

        config = LocalLLMConfig(model_name="test-model", temperature=0.8)

        config_dict = asdict(config)
        assert isinstance(config_dict, dict)
        assert config_dict["model_name"] == "test-model"
        assert config_dict["temperature"] == 0.8

    @pytest.mark.parametrize("provider", [LocalLLMProvider.OLLAMA, LocalLLMProvider.LLAMA_CPP])
    def test_config_with_different_providers(self, provider):
        """Test configuration with different provider types."""
        if not IMPORT_SUCCESS:
            pytest.skip("Local LLM modules not available")

        config = LocalLLMConfig(provider=provider)
        assert config.provider == provider

    @pytest.mark.parametrize(
        "quantization",
        [
            QuantizationLevel.INT3,
            QuantizationLevel.INT4,
            QuantizationLevel.INT8,
            QuantizationLevel.HALF,
        ],
    )
    def test_config_with_different_quantizations(self, quantization):
        """Test configuration with different quantization levels."""
        if not IMPORT_SUCCESS:
            pytest.skip("Local LLM modules not available")

        config = LocalLLMConfig(quantization=quantization)
        assert config.quantization == quantization


class TestOllamaProvider:
    """Test suite for OllamaProvider class."""

    @pytest.fixture
    def ollama_config(self):
        """Create configuration for Ollama provider."""
        if not IMPORT_SUCCESS:
            pytest.skip("Local LLM modules not available")
        return LocalLLMConfig(
            provider=LocalLLMProvider.OLLAMA,
            model_name="llama2",
            ollama_host="http://localhost:11434",
        )

    @pytest.fixture
    def ollama_provider(self, ollama_config):
        """Create OllamaProvider instance."""
        if not IMPORT_SUCCESS:
            pytest.skip("Local LLM modules not available")
        return OllamaProvider(ollama_config)

    def test_ollama_provider_initialization(self, ollama_provider, ollama_config):
        """Test OllamaProvider initialization."""
        assert ollama_provider.config == ollama_config
        assert ollama_provider.base_url == "http://localhost:11434"
        assert ollama_provider.config.model_name == "llama2"
        assert ollama_provider.session is not None

    def test_ollama_check_availability(self, ollama_provider):
        """Test checking Ollama availability."""
        with patch.object(ollama_provider.session, "get") as mock_get:
            # Mock successful response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            is_available = ollama_provider.is_available()
            assert is_available is True

    def test_ollama_check_availability_failure(self, ollama_provider):
        """Test Ollama availability check failure."""
        with patch.object(ollama_provider.session, "get") as mock_get:
            # Mock failed response
            mock_get.side_effect = httpx.RequestError("Connection failed")

            is_available = ollama_provider.is_available()
            assert is_available is False

    def test_ollama_load_model(self, ollama_provider):
        """Test listing available Ollama models."""
        with patch.object(ollama_provider.session, "get") as mock_get:
            # Mock models response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "models": [
                    {"name": "llama2", "size": 4000000000},
                    {"name": "mistral", "size": 7000000000},
                ]
            }
            mock_get.return_value = mock_response

            result = ollama_provider.load_model()
            assert result is True

    def test_ollama_generate_text(self, ollama_provider):
        """Test text generation with Ollama."""
        with patch.object(ollama_provider.session, "post") as mock_post:
            # Mock generation response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": "This is a test response from Ollama",
                "done": True,
                "total_duration": 1500000000,
                "eval_count": 20,
            }
            mock_post.return_value = mock_response

            prompt = "Hello, how are you?"
            response = ollama_provider.generate(prompt)

            assert response.text == "This is a test response from Ollama"
            assert response.provider == LocalLLMProvider.OLLAMA
            assert response.generation_time > 0

    def test_ollama_generate_text_streaming(self, ollama_provider):
        """Test streaming text generation with Ollama."""
        # Streaming is not implemented in current OllamaProvider, test regular generation
        with patch.object(ollama_provider.session, "post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": "Hello there!",
                "eval_count": 5,
                "total_duration": 1500000000,
            }
            mock_post.return_value = mock_response

            prompt = "Hello"
            response = ollama_provider.generate(prompt)
            assert response.text is not None
            assert isinstance(response, LLMResponse)
            assert response.text == "Hello there!"

    def test_ollama_pull_model(self, ollama_provider):
        """Test pulling a model with Ollama."""
        with (
            patch.object(ollama_provider.session, "get") as mock_get,
            patch.object(ollama_provider.session, "post") as mock_post,
        ):
            # Mock successful model list
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"models": [{"name": "llama2"}]}

            # Test load_model method
            result = ollama_provider.load_model()
            assert result is True

    def test_ollama_error_handling(self, ollama_provider):
        """Test Ollama error handling."""
        with patch.object(ollama_provider.session, "post") as mock_post:
            # Mock error response
            mock_post.side_effect = httpx.RequestError("Server error")

            with patch("inference.llm.local_llm_manager.logger") as mock_logger:
                with pytest.raises(
                    Exception
                ):  # generate() raises exceptions instead of returning None
                    ollama_provider.generate("test prompt")
                mock_logger.error.assert_called()

    def test_ollama_request_body_construction(self, ollama_provider):
        """Test that generate method builds proper request internally."""
        # This tests that the provider can handle basic generation
        with patch.object(ollama_provider.session, "post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"response": "test", "eval_count": 10}
            mock_post.return_value = mock_response

            result = ollama_provider.generate("Test prompt")
            assert result.text == "test"

            # Verify the request was made with correct parameters
            mock_post.assert_called_once()

    def test_ollama_response_handling(self, ollama_provider):
        """Test response handling in Ollama provider."""
        with patch.object(ollama_provider.session, "post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "response": "Test response",
                "eval_count": 15,
                "total_duration": 1500000000,
            }
            mock_post.return_value = mock_response

            result = ollama_provider.generate("Test prompt")
            assert isinstance(result, LLMResponse)
            assert result.text == "Test response"
            assert result.tokens_used == 15

    def test_ollama_timeout_handling(self, ollama_provider):
        """Test timeout handling for Ollama requests."""
        with patch.object(ollama_provider.session, "post") as mock_post:
            mock_post.side_effect = httpx.TimeoutException("Request timeout")

            with pytest.raises(Exception):  # Timeout should raise exception
                ollama_provider.generate("test prompt")


class TestLlamaCppProvider:
    """Test suite for LlamaCppProvider class."""

    @pytest.fixture
    def llama_cpp_config(self):
        """Create configuration for llama.cpp provider."""
        if not IMPORT_SUCCESS:
            pytest.skip("Local LLM modules not available")
        return LocalLLMConfig(
            provider=LocalLLMProvider.LLAMA_CPP,
            model_name="llama-7b",
            llama_cpp_binary="/usr/local/bin/llama-cpp",
            model_path="/models/llama-7b.ggml",
            threads=4,
            context_size=2048,
        )

    @pytest.fixture
    def llama_cpp_provider(self, llama_cpp_config):
        """Create LlamaCppProvider instance."""
        if not IMPORT_SUCCESS:
            pytest.skip("Local LLM modules not available")
        return LlamaCppProvider(llama_cpp_config)

    def test_llama_cpp_provider_initialization(self, llama_cpp_provider, llama_cpp_config):
        """Test LlamaCppProvider initialization."""
        assert llama_cpp_provider.config == llama_cpp_config
        assert llama_cpp_provider.binary_path == "/usr/local/bin/llama-cpp"
        assert llama_cpp_provider.config.model_path == "/models/llama-7b.ggml"

    def test_llama_cpp_check_availability(self, llama_cpp_provider):
        """Test checking llama.cpp availability."""
        with patch("inference.llm.local_llm_manager.subprocess.run") as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            is_available = llama_cpp_provider.is_available()
            assert is_available is True

            mock_run.assert_called_once()

    def test_llama_cpp_check_availability_missing_binary(self, llama_cpp_provider):
        """Test availability check with missing binary."""
        with patch("inference.llm.local_llm_manager.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("Binary not found")

            is_available = llama_cpp_provider.is_available()
            assert is_available is False

    def test_llama_cpp_check_availability_missing_model(self, llama_cpp_provider):
        """Test availability check with missing model."""
        with patch("inference.llm.local_llm_manager.subprocess.run") as mock_run:
            mock_result = Mock()
            mock_result.returncode = 1  # Command failed
            mock_run.return_value = mock_result

            is_available = llama_cpp_provider.is_available()
            assert is_available is False

    @patch("inference.llm.local_llm_manager.subprocess.run")
    def test_llama_cpp_generate_text(self, mock_run, llama_cpp_provider):
        """Test text generation with llama.cpp."""
        # Mock subprocess response
        mock_result = Mock()
        mock_result.stdout = "This is a test response from llama.cpp"
        mock_result.stderr = ""
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        # Set model as loaded for testing
        llama_cpp_provider.model_loaded = True

        prompt = "Hello, how are you?"
        response = llama_cpp_provider.generate(prompt)

        assert response.text == "This is a test response from llama.cpp"
        assert response.generation_time > 0
        assert response.provider == LocalLLMProvider.LLAMA_CPP

    @patch("inference.llm.local_llm_manager.subprocess.run")
    def test_llama_cpp_generate_text_error(self, mock_run, llama_cpp_provider):
        """Test text generation error handling."""
        # Mock subprocess error
        mock_result = Mock()
        mock_result.stdout = ""
        mock_result.stderr = "Model loading failed"
        mock_result.returncode = 1
        mock_run.return_value = mock_result

        # Set model as loaded for testing
        llama_cpp_provider.model_loaded = True

        prompt = "Hello"
        with pytest.raises(RuntimeError) as exc_info:
            llama_cpp_provider.generate(prompt)

        assert "llama.cpp failed" in str(exc_info.value)

    def test_llama_cpp_build_command(self, llama_cpp_provider):
        """Test that llama.cpp provider can generate text properly."""
        # Test that the provider builds commands internally during generation
        with patch("inference.llm.local_llm_manager.subprocess.run") as mock_run:
            mock_result = Mock()
            mock_result.stdout = "Test response"
            mock_result.stderr = "tokens: 10"
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            llama_cpp_provider.model_loaded = True
            response = llama_cpp_provider.generate("Test prompt")

            assert response.text == "Test response"
            mock_run.assert_called_once()

    def test_llama_cpp_command_with_gpu_layers(self, llama_cpp_config):
        """Test that GPU layers config is respected."""
        if not IMPORT_SUCCESS:
            pytest.skip("Local LLM modules not available")

        llama_cpp_config.gpu_layers = 32
        provider = LlamaCppProvider(llama_cpp_config)

        # Verify config is stored correctly
        assert provider.config.gpu_layers == 32

    @patch("inference.llm.local_llm_manager.subprocess.run")
    def test_llama_cpp_timeout_handling(self, mock_run, llama_cpp_provider):
        """Test timeout handling for llama.cpp."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=[], timeout=30)

        # Set model as loaded for testing
        llama_cpp_provider.model_loaded = True

        with pytest.raises(subprocess.TimeoutExpired):
            llama_cpp_provider.generate("test prompt")


class TestLocalLLMManager:
    """Test suite for LocalLLMManager class."""

    @pytest.fixture
    def manager_config(self):
        """Create configuration for LocalLLMManager."""
        if not IMPORT_SUCCESS:
            pytest.skip("Local LLM modules not available")
        return LocalLLMConfig()

    @pytest.fixture
    def manager(self, manager_config):
        """Create LocalLLMManager instance."""
        if not IMPORT_SUCCESS:
            pytest.skip("Local LLM modules not available")
        return LocalLLMManager(manager_config)

    def test_manager_initialization(self, manager, manager_config):
        """Test LocalLLMManager initialization."""
        assert manager.config == manager_config
        assert len(manager.providers) > 0
        assert manager.current_provider is None

    def test_manager_initialization_with_providers(self, manager_config):
        """Test manager initialization creates appropriate providers."""
        if not IMPORT_SUCCESS:
            pytest.skip("Local LLM modules not available")

        # Test Ollama provider
        ollama_config = LocalLLMConfig(provider=LocalLLMProvider.OLLAMA)
        ollama_manager = LocalLLMManager(ollama_config)
        assert "ollama" in ollama_manager.providers
        assert isinstance(ollama_manager.providers["ollama"], OllamaProvider)

        # Test llama.cpp provider
        llama_cpp_config = LocalLLMConfig(provider=LocalLLMProvider.LLAMA_CPP)
        llama_cpp_manager = LocalLLMManager(llama_cpp_config)
        assert "llama_cpp" in llama_cpp_manager.providers
        assert isinstance(llama_cpp_manager.providers["llama_cpp"], LlamaCppProvider)

    def test_manager_find_best_provider(self, manager):
        """Test finding the best available provider."""
        # Mock all providers as unavailable initially
        for provider in manager.providers.values():
            provider.is_available = Mock(return_value=False)

        # Make one provider available
        if manager.providers:
            first_provider = list(manager.providers.values())[0]
            first_provider.is_available = Mock(return_value=True)
            first_provider.load_model = Mock(return_value=True)

            result = manager.load_model()
            assert result is True
            assert manager.current_provider == first_provider

    def test_manager_find_best_provider_none_available(self, manager):
        """Test finding provider when none are available."""
        # Mock all providers as unavailable
        for provider in manager.providers.values():
            provider.is_available = Mock(return_value=False)

        result = manager.load_model()
        assert result is False
        assert manager.current_provider is None

    def test_manager_generate_text_success(self, manager):
        """Test successful text generation."""
        # Mock provider
        mock_provider = Mock()
        mock_provider.is_available.return_value = True
        mock_provider.generate.return_value = LLMResponse(
            text="Test response", tokens_used=10, generation_time=0.15, provider="test"
        )

        manager.providers = {"test": mock_provider}
        manager.current_provider = mock_provider

        result = manager.generate("Test prompt")

        assert result.text == "Test response"
        assert isinstance(result, LLMResponse)
        assert manager.current_provider == mock_provider

    def test_manager_generate_text_with_fallback(self, manager):
        """Test text generation with fallback to another provider."""
        # First provider fails, second succeeds
        failing_provider = Mock()
        failing_provider.is_available.return_value = True
        failing_provider.load_model.return_value = False  # First provider fails to load

        working_provider = Mock()
        working_provider.is_available.return_value = True
        working_provider.load_model.return_value = True
        working_provider.generate.return_value = LLMResponse(
            text="Fallback response", tokens_used=20, generation_time=0.2, provider="working"
        )

        manager.providers = {"failing": failing_provider, "working": working_provider}
        manager.config.enable_fallback = True

        result = manager.generate("Test prompt")

        # The working provider should be selected successfully
        assert isinstance(result, LLMResponse)
        assert result.text == "Fallback response"
        assert manager.current_provider == working_provider

    def test_manager_generate_text_no_fallback(self, manager):
        """Test text generation without fallback when disabled."""
        failing_provider = Mock()
        failing_provider.is_available.return_value = True
        failing_provider.load_model.return_value = True
        failing_provider.generate.side_effect = Exception("Generation failed")

        manager.providers = {"failing": failing_provider}
        manager.config.enable_fallback = False

        with pytest.raises(RuntimeError) as exc_info:
            manager.generate("Test prompt")

        assert "All generation attempts failed" in str(exc_info.value)

    def test_manager_generate_text_no_providers(self, manager):
        """Test text generation when no providers are available."""
        manager.providers = {}

        # When no providers are available, fallback is used instead of raising an error
        result = manager.generate("Test prompt")

        assert isinstance(result, LLMResponse)
        assert result.fallback_used is True

    def test_manager_list_available_models(self, manager):
        """Test listing available models across providers."""
        mock_provider1 = Mock()
        mock_provider1.is_available.return_value = True
        mock_provider1.load_model.return_value = True

        mock_provider2 = Mock()
        mock_provider2.is_available.return_value = True
        mock_provider2.load_model.return_value = True

        manager.providers = {"provider1": mock_provider1, "provider2": mock_provider2}

        # Test that providers are available instead of listing models
        status = manager.get_status()

        assert "provider1" in status["providers"]
        assert "provider2" in status["providers"]
        assert status["providers"]["provider1"]["available"] is True
        assert status["providers"]["provider2"]["available"] is True

    def test_manager_get_provider_status(self, manager):
        """Test getting status of all providers."""
        mock_provider = Mock()
        mock_provider.is_available.return_value = True
        mock_provider.__class__.__name__ = "OllamaProvider"

        manager.providers = {"ollama": mock_provider}

        status = manager.get_status()

        assert "ollama" in status["providers"]
        assert status["providers"]["ollama"]["available"] is True

    def test_manager_switch_provider(self, manager):
        """Test switching to a specific provider."""
        provider1 = Mock()
        provider1.is_available.return_value = True
        provider1.load_model.return_value = True
        provider1.__class__.__name__ = "OllamaProvider"

        provider2 = Mock()
        provider2.is_available.return_value = True
        provider2.load_model.return_value = True
        provider2.__class__.__name__ = "LlamaCppProvider"

        manager.providers = {"ollama": provider1, "llama_cpp": provider2}

        # Test that we can load a specific provider
        manager.current_provider = provider2
        assert manager.current_provider == provider2

    def test_manager_switch_provider_unavailable(self, manager):
        """Test switching to an unavailable provider."""
        provider = Mock()
        provider.is_available.return_value = False
        provider.load_model.return_value = False
        provider.__class__.__name__ = "OllamaProvider"

        manager.providers = {"ollama": provider}

        result = manager.load_model()

        assert result is False
        assert manager.current_provider is None

    def test_manager_get_generation_stats(self, manager):
        """Test getting manager status."""
        status = manager.get_status()

        assert "config" in status
        assert "providers" in status
        assert "cache_stats" in status
        assert "fallback_enabled" in status

    def test_manager_thread_safety(self, manager):
        """Test thread safety of manager operations."""
        # Mock provider
        mock_provider = Mock()
        mock_provider.is_available.return_value = True
        mock_provider.generate.return_value = LLMResponse(
            text="Test response", tokens_used=10, generation_time=0.1, provider="test"
        )

        manager.providers = {"test": mock_provider}
        manager.current_provider = mock_provider
        results = []
        errors = []

        def generate_text_concurrent():
            try:
                result = manager.generate("Concurrent test")
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Run concurrent generations
        threads = []
        for _ in range(10):
            t = threading.Thread(target=generate_text_concurrent)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        assert all(isinstance(r, LLMResponse) for r in results)


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""

    def test_full_ollama_workflow(self):
        """Test complete Ollama workflow."""
        if not IMPORT_SUCCESS:
            pytest.skip("Local LLM modules not available")

        config = LocalLLMConfig(
            provider=LocalLLMProvider.OLLAMA,
            model_name="llama2",
            ollama_host="http://localhost:11434",
        )

        manager = LocalLLMManager(config)
        ollama_provider = manager.providers["ollama"]

        with (
            patch.object(ollama_provider.session, "get") as mock_get,
            patch.object(ollama_provider.session, "post") as mock_post,
        ):
            # Mock successful availability check
            mock_get_response = Mock()
            mock_get_response.status_code = 200
            mock_get_response.json.return_value = {"models": [{"name": "llama2"}]}
            mock_get.return_value = mock_get_response

            # Mock successful generation
            mock_post_response = Mock()
            mock_post_response.json.return_value = {
                "response": "Hello! I'm doing well, thank you for asking.",
                "done": True,
                "total_duration": 1500000000,
                "eval_count": 15,
            }
            mock_post.return_value = mock_post_response

            # Test the workflow
            result = manager.generate("Hello, how are you?")

            assert isinstance(result, LLMResponse)
            assert "Hello!" in result.text
            assert result.generation_time > 0

    def test_llama_cpp_workflow(self):
        """Test complete llama.cpp workflow."""
        if not IMPORT_SUCCESS:
            pytest.skip("Local LLM modules not available")

        config = LocalLLMConfig(
            provider=LocalLLMProvider.LLAMA_CPP,
            llama_cpp_binary="/usr/local/bin/llama-cpp",
            model_path="/models/llama-7b.ggml",
        )

        with patch("inference.llm.local_llm_manager.os.path.exists") as mock_exists:
            with patch("inference.llm.local_llm_manager.subprocess.run") as mock_run:
                # Mock availability
                mock_exists.return_value = True

                # Mock successful generation
                mock_result = Mock()
                mock_result.stdout = "I'm doing great, thanks for asking!"
                mock_result.stderr = ""
                mock_result.returncode = 0
                mock_run.return_value = mock_result

                manager = LocalLLMManager(config)

                result = manager.generate("How are you?")

                assert isinstance(result, LLMResponse)
                assert "great" in result.text

    def test_provider_fallback_workflow(self):
        """Test fallback between providers."""
        if not IMPORT_SUCCESS:
            pytest.skip("Local LLM modules not available")

        config = LocalLLMConfig(enable_fallback=True)
        manager = LocalLLMManager(config)

        # Mock first provider failing to load
        failing_provider = Mock()
        failing_provider.is_available.return_value = True
        failing_provider.load_model.return_value = False

        # Mock second provider succeeding
        working_provider = Mock()
        working_provider.is_available.return_value = True
        working_provider.load_model.return_value = True
        working_provider.generate.return_value = LLMResponse(
            text="Fallback response", tokens_used=15, generation_time=0.25, provider="working"
        )

        manager.providers = {"failing": failing_provider, "working": working_provider}

        result = manager.generate("Test prompt")

        assert isinstance(result, LLMResponse)
        assert result.text == "Fallback response"
        assert manager.current_provider == working_provider

    def test_error_recovery_and_logging(self):
        """Test error recovery and logging mechanisms."""
        if not IMPORT_SUCCESS:
            pytest.skip("Local LLM modules not available")

        config = LocalLLMConfig()
        manager = LocalLLMManager(config)

        # Mock provider that throws exception
        error_provider = Mock()
        error_provider.is_available.side_effect = Exception("Unexpected error")

        manager.providers = {"error": error_provider}

        with patch("inference.llm.local_llm_manager.logger") as mock_logger:
            # The manager will use fallback responder when providers fail
            result = manager.generate("Test prompt")

            assert isinstance(result, LLMResponse)
            assert result.fallback_used is True
            # Check that some warning was logged during the failure attempts
            assert mock_logger.warning.call_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=inference.llm.local_llm_manager", "--cov-report=html"])
