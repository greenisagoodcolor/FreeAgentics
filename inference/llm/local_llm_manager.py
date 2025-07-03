"""
Module for FreeAgentics Active Inference implementation.
"""

import hashlib
import json
import logging
import random
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import requests

"""
Local LLM Manager for Edge Deployment
Manages local LLM integration for autonomous agent operation without cloud dependencies.
Supports Ollama, llama.cpp, and other local inference engines.
"""
logger = logging.getLogger(__name__)


class LocalLLMProvider(Enum):
    """Supported local LLM providers"""

    OLLAMA = "ollama"
    LLAMA_CPP = "llama_cpp"
    GGML = "ggml"
    CANDLE = "candle"


class QuantizationLevel(Enum):
    """Model quantization levels"""

    FULL = "f32"  # Full precision
    HALF = "f16"  # Half precision
    INT8 = "q8_0"  # 8-bit quantization
    INT4 = "q4_K_M"  # 4-bit quantization (recommended for edge)
    INT3 = "q3_K_M"  # 3-bit quantization (extreme compression)


@dataclass
class LocalLLMConfig:
    """Configuration for local LLM"""

    provider: LocalLLMProvider
    model_name: str
    model_path: Optional[Path] = None
    quantization: QuantizationLevel = QuantizationLevel.INT4
    context_size: int = 2048
    max_tokens: int = 512
    temperature: float = 0.7
    threads: int = 4
    gpu_layers: int = 0  # Number of layers to offload to GPU
    use_mmap: bool = True
    use_mlock: bool = False
    seed: int = -1
    # Provider-specific settings
    ollama_host: str = "http://localhost:11434"
    llama_cpp_binary: str = "llama.cpp/main"
    # Performance settings
    batch_size: int = 8
    cache_prompt: bool = True
    cache_size_mb: int = 100
    # Fallback settings
    enable_fallback: bool = True
    fallback_timeout: float = 30.0
    retry_attempts: int = 3


@dataclass
class LLMResponse:
    """Response from local LLM"""

    text: str
    tokens_used: int
    generation_time: float
    provider: LocalLLMProvider
    cached: bool = False
    fallback_used: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class LocalLLMProvider(ABC):
    """Abstract base class for local LLM providers"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response from prompt"""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available"""

    @abstractmethod
    def load_model(self, model_path: Path, config: LocalLLMConfig) -> bool:
        """Load model into memory"""


class OllamaProvider(LocalLLMProvider):
    """Ollama integration for local LLM inference"""

    def __init__(self, config: LocalLLMConfig) -> None:
        """Initialize Ollama provider"""
        self.config = config
        self.base_url = config.ollama_host
        self.session = requests.Session()

    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def load_model(self, model_path: Path, config: LocalLLMConfig) -> bool:
        """Load model in Ollama"""
        try:
            # Check if model exists
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                if config.model_name in model_names:
                    logger.info(
                        f"Model {
                            config.model_name} already loaded in Ollama"
                    )
                    return True
            # Pull model if not exists
            logger.info(f"Pulling model {config.model_name} in Ollama...")
            pull_data = {"name": config.model_name}
            response = self.session.post(f"{self.base_url}/api/pull", json=pull_data, stream=True)
            # Monitor pull progress
            for line in response.iter_lines():
                if line:
                    progress = json.loads(line)
                    if "status" in progress:
                        logger.debug(f"Ollama: {progress['status']}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model in Ollama: {e}")
            return False

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Ollama"""
        start_time = time.time()
        try:
            data = {
                "model": self.config.model_name,
                "prompt": prompt,
                "options": {
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                    "num_ctx": self.config.context_size,
                    "num_thread": self.config.threads,
                    "seed": self.config.seed,
                },
                "stream": False,
            }
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=self.config.fallback_timeout,
            )
            if response.status_code == 200:
                result = response.json()
                return LLMResponse(
                    text=result["response"],
                    tokens_used=result.get("eval_count", 0),
                    generation_time=time.time() - start_time,
                    provider=LocalLLMProvider.OLLAMA,
                    metadata={
                        "model": result.get("model"),
                        "total_duration": result.get("total_duration"),
                        "load_duration": result.get("load_duration"),
                        "eval_duration": result.get("eval_duration"),
                    },
                )
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise


class LlamaCppProvider(LocalLLMProvider):
    """llama.cpp integration for local LLM inference"""

    def __init__(self, config: LocalLLMConfig) -> None:
        """Initialize llama.cpp provider"""
        self.config = config
        self.process = None
        self.model_loaded = False

    def is_available(self) -> bool:
        """Check if llama.cpp binary exists"""
        binary_path = Path(self.config.llama_cpp_binary)
        return binary_path.exists() and binary_path.is_file()

    def load_model(self, model_path: Path, config: LocalLLMConfig) -> bool:
        """Verify model file exists"""
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False
        self.config.model_path = model_path
        self.model_loaded = True
        return True

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using llama.cpp"""
        if not self.model_loaded or not self.config.model_path:
            raise Exception("Model not loaded")
        start_time = time.time()
        # Build command
        cmd = [
            self.config.llama_cpp_binary,
            "-m",
            str(self.config.model_path),
            "-p",
            prompt,
            "-n",
            str(kwargs.get("max_tokens", self.config.max_tokens)),
            "-c",
            str(self.config.context_size),
            "-t",
            str(self.config.threads),
            "--temp",
            str(kwargs.get("temperature", self.config.temperature)),
            "-b",
            str(self.config.batch_size),
        ]
        # Add GPU layers if available
        if self.config.gpu_layers > 0:
            cmd.extend(["-ngl", str(self.config.gpu_layers)])
        # Add memory options
        if self.config.use_mmap:
            cmd.append("--mmap")
        if self.config.use_mlock:
            cmd.append("--mlock")
        try:
            # Run llama.cpp
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.fallback_timeout,
            )
            if result.returncode == 0:
                # Parse output (simplified - real implementation would parse
                # properly)
                output = result.stdout
                # Extract just the generated text (after the prompt)
                if prompt in output:
                    generated = output.split(prompt)[-1].strip()
                else:
                    generated = output.strip()
                return LLMResponse(
                    text=generated,
                    tokens_used=len(generated.split()),  # Approximate
                    generation_time=time.time() - start_time,
                    provider=LocalLLMProvider.LLAMA_CPP,
                    metadata={"command": " ".join(cmd)},
                )
            else:
                raise Exception(f"llama.cpp error: {result.stderr}")
        except subprocess.TimeoutExpired:
            raise Exception("llama.cpp generation timeout")
        except Exception as e:
            logger.error(f"llama.cpp generation failed: {e}")
            raise


class ResponseCache:
    """Cache for LLM responses"""

    def __init__(self, max_size_mb: int = 100) -> None:
        """Initialize response cache"""
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache: Dict[str, tuple[LLMResponse, float]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.lock = threading.Lock()

    def _get_cache_key(self, prompt: str, **kwargs) -> str:
        """Generate cache key from prompt and parameters"""
        key_data = {
            "prompt": prompt,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 512),
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(self, prompt: str, **kwargs) -> Optional[LLMResponse]:
        """Get cached response if available"""
        key = self._get_cache_key(prompt, **kwargs)
        with self.lock:
            if key in self.cache:
                response, timestamp = self.cache[key]
                # Check if cache entry is still valid (1 hour)
                if time.time() - timestamp < 3600:
                    self.cache_hits += 1
                    response.cached = True
                    return response
                else:
                    del self.cache[key]
            self.cache_misses += 1
            return None

    def put(self, prompt: str, response: LLMResponse, **kwargs):
        """Cache a response"""
        key = self._get_cache_key(prompt, **kwargs)
        with self.lock:
            # Simple size management - remove oldest entries if too large
            if len(self.cache) > 1000:  # Arbitrary limit
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
            self.cache[key] = (response, time.time())

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
            return {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_rate": hit_rate,
                "entries": len(self.cache),
            }


class FallbackResponder:
    """Provides fallback responses when LLM is unavailable"""

    def __init__(self) -> None:
        """Initialize fallback responder"""
        self.templates = {
            "greeting": [
                "Hello! I'm operating in offline mode.",
                "Greetings! Local processing active.",
                "Hi there! Running autonomously.",
            ],
            "acknowledgment": [
                "Understood. Processing your request.",
                "Acknowledged. Working on it.",
                "Got it. Let me handle that.",
            ],
            "exploration": [
                "Exploring the area for resources.",
                "Scanning environment for opportunities.",
                "Investigating nearby locations.",
            ],
            "trading": [
                "Evaluating trade opportunities.",
                "Analyzing market conditions.",
                "Considering available exchanges.",
            ],
            "research": [
                "Analyzing available data.",
                "Processing information patterns.",
                "Examining knowledge connections.",
            ],
            "security": [
                "Monitoring perimeter security.",
                "Checking for potential threats.",
                "Maintaining defensive positions.",
            ],
            "error": [
                "I'm having trouble processing that request.",
                "Unable to generate a detailed response.",
                "Operating with limited capabilities.",
            ],
        }

    def get_fallback_response(self, prompt: str) -> str:
        """Generate a fallback response based on prompt content"""
        prompt_lower = prompt.lower()
        # Detect intent from prompt
        if any(word in prompt_lower for word in ["hello", "hi", "greet"]):
            category = "greeting"
        elif any(word in prompt_lower for word in ["explore", "find", "search"]):
            category = "exploration"
        elif any(word in prompt_lower for word in ["trade", "exchange", "buy", "sell"]):
            category = "trading"
        elif any(word in prompt_lower for word in ["research", "analyze", "study"]):
            category = "research"
        elif any(word in prompt_lower for word in ["guard", "protect", "secure"]):
            category = "security"
        elif any(word in prompt_lower for word in ["ok", "yes", "understand"]):
            category = "acknowledgment"
        else:
            category = "error"
        # Select random response from category
        responses = self.templates.get(category, self.templates["error"])
        return random.choice(responses)


class LocalLLMManager:
    """
    Manages local LLM providers for edge deployment.
    Features:
    - Multiple provider support (Ollama, llama.cpp)
    - Automatic fallback between providers
    - Response caching
    - Graceful degradation
    """

    def __init__(self, config: LocalLLMConfig) -> None:
        """Initialize Local LLM Manager"""
        self.config = config
        self.providers: Dict[LocalLLMProvider, LocalLLMProvider] = {}
        self.cache = ResponseCache(config.cache_size_mb)
        self.fallback_responder = FallbackResponder()
        self.current_provider = None
        # Initialize providers
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize available providers"""
        # Try Ollama first
        if self.config.provider == LocalLLMProvider.OLLAMA:
            ollama = OllamaProvider(self.config)
            if ollama.is_available():
                self.providers[LocalLLMProvider.OLLAMA] = ollama
                self.current_provider = ollama
                logger.info("Ollama provider initialized")
        # Try llama.cpp as fallback
        if self.config.provider == LocalLLMProvider.LLAMA_CPP or not self.current_provider:
            llama_cpp = LlamaCppProvider(self.config)
            if llama_cpp.is_available():
                self.providers[LocalLLMProvider.LLAMA_CPP] = llama_cpp
                if not self.current_provider:
                    self.current_provider = llama_cpp
                logger.info("llama.cpp provider initialized")
        if not self.current_provider:
            logger.warning("No local LLM providers available")

    def load_model(self, model_path: Optional[Path] = None) -> bool:
        """Load model for inference"""
        if not self.current_provider:
            logger.error("No provider available to load model")
            return False
        try:
            if model_path:
                return self.current_provider.load_model(model_path, self.config)
            else:
                # For Ollama, model_path is optional
                return self.current_provider.load_model(Path(""), self.config)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate response with automatic fallback.
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
        Returns:
            LLMResponse with generated text
        """
        # Check cache first
        cached_response = self.cache.get(prompt, **kwargs)
        if cached_response:
            logger.debug("Returning cached response")
            return cached_response
        # Try primary provider
        if self.current_provider:
            try:
                response = self._try_generate(self.current_provider, prompt, **kwargs)
                self.cache.put(prompt, response, **kwargs)
                return response
            except Exception as e:
                logger.warning(f"Primary provider failed: {e}")
        # Try fallback providers
        for provider_type, provider in self.providers.items():
            if provider != self.current_provider:
                try:
                    logger.info(f"Trying fallback provider: {provider_type}")
                    response = self._try_generate(provider, prompt, **kwargs)
                    self.cache.put(prompt, response, **kwargs)
                    return response
                except Exception as e:
                    logger.warning(f"Fallback provider {provider_type} failed: {e}")
        # Use fallback responder as last resort
        if self.config.enable_fallback:
            logger.warning("Using fallback responder")
            fallback_text = self.fallback_responder.get_fallback_response(prompt)
            return LLMResponse(
                text=fallback_text,
                tokens_used=len(fallback_text.split()),
                generation_time=0.01,
                provider=LocalLLMProvider.OLLAMA,  # Dummy provider
                fallback_used=True,
            )
        raise Exception("All LLM providers failed and fallback disabled")

    def _try_generate(self, provider: LocalLLMProvider, prompt: str, **kwargs) -> LLMResponse:
        """Try to generate with retries"""
        last_error = None
        for attempt in range(self.config.retry_attempts):
            try:
                return provider.generate(prompt, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(1 * (attempt + 1))  # Exponential backoff
        raise last_error

    def get_status(self) -> Dict[str, Any]:
        """Get status of LLM manager"""
        return {
            "current_provider": (
                type(self.current_provider).__name__ if self.current_provider else None
            ),
            "available_providers": list(self.providers.keys()),
            "model_loaded": bool(self.current_provider),
            "cache_stats": self.cache.get_stats(),
            "config": {
                "model": self.config.model_name,
                "quantization": self.config.quantization.value,
                "context_size": self.config.context_size,
                "threads": self.config.threads,
            },
        }

    def optimize_for_hardware(
        self, ram_gb: float, cpu_cores: int, has_gpu: bool = False
    ) -> LocalLLMConfig:
        """
        Optimize configuration for specific hardware.
        Args:
            ram_gb: Available RAM in GB
            cpu_cores: Number of CPU cores
            has_gpu: Whether GPU is available
        Returns:
            Optimized configuration
        """
        config = LocalLLMConfig(provider=self.config.provider, model_name=self.config.model_name)
        # Adjust quantization based on RAM
        if ram_gb < 4:
            config.quantization = QuantizationLevel.INT3
            config.context_size = 512
            config.cache_size_mb = 50
        elif ram_gb < 8:
            config.quantization = QuantizationLevel.INT4
            config.context_size = 1024
            config.cache_size_mb = 100
        elif ram_gb < 16:
            config.quantization = QuantizationLevel.INT8
            config.context_size = 2048
            config.cache_size_mb = 200
        else:
            config.quantization = QuantizationLevel.HALF
            config.context_size = 4096
            config.cache_size_mb = 500
        # Adjust threads
        config.threads = min(cpu_cores - 1, 8)  # Leave one core for system
        # GPU settings
        if has_gpu:
            # Offload layers to GPU based on quantization
            if config.quantization == QuantizationLevel.INT3:
                config.gpu_layers = 20
            elif config.quantization == QuantizationLevel.INT4:
                config.gpu_layers = 16
            else:
                config.gpu_layers = 12
        # Memory settings
        config.use_mmap = True  # Always use memory mapping
        config.use_mlock = ram_gb >= 16  # Lock in RAM only if plenty available
        logger.info(
            f"Optimized config for {ram_gb}GB RAM, {cpu_cores} cores: "
            f"quantization={config.quantization.value}, "
            f"context={config.context_size}, "
            f"threads={config.threads}"
        )
        return config
