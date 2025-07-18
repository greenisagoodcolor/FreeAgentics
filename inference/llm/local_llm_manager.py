"""Local LLM manager for FreeAgentics with Ollama and llama.cpp support."""

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx

logger = logging.getLogger(__name__)


class LocalLLMProvider(Enum):
    """Local LLM provider types."""

    OLLAMA = "ollama"
    LLAMA_CPP = "llama_cpp"


class QuantizationLevel(Enum):
    """Model quantization levels."""

    INT3 = "q3_K_M"
    INT4 = "q4_K_M"
    INT8 = "q8_0"
    HALF = "f16"


@dataclass
class LocalLLMConfig:
    """Configuration for local LLM providers."""

    provider: LocalLLMProvider = LocalLLMProvider.OLLAMA
    model_name: str = "llama2"
    quantization: QuantizationLevel = QuantizationLevel.INT4
    context_size: int = 2048
    max_tokens: int = 512
    temperature: float = 0.7
    threads: int = 4
    gpu_layers: int = 0
    ollama_host: str = "http://localhost:11434"
    enable_fallback: bool = True
    llama_cpp_binary: Optional[str] = None
    model_path: Optional[str] = None
    cache_size_mb: int = 512
    use_mmap: bool = True
    use_mlock: bool = False
    retry_attempts: int = 3


@dataclass
class LLMResponse:
    """Response from LLM generation."""

    text: str
    tokens_used: int
    generation_time: float
    provider: str
    cached: bool = False
    fallback_used: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class OllamaProvider:
    """Provider for Ollama local LLM service."""

    def __init__(self, config: LocalLLMConfig):
        """Initialize Ollama provider."""
        self.config = config
        self.base_url = config.ollama_host
        self.session = httpx.Client(timeout=30.0)

    def is_available(self) -> bool:
        """Check if Ollama service is available."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def load_model(self) -> bool:
        """Ensure model is loaded in Ollama."""
        try:
            # Check if model exists
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                return False

            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]

            if self.config.model_name not in model_names:
                logger.warning(
                    f"Model {self.config.model_name} not found in Ollama"
                )
                # Optionally pull the model
                pull_response = self.session.post(
                    f"{self.base_url}/api/pull",
                    json={"name": self.config.model_name},
                )
                if pull_response.status_code != 200:
                    return False

                # Process the pull response
                for line in pull_response.iter_lines():
                    if line:
                        try:
                            status = json.loads(line)
                            if status.get("status") == "error":
                                logger.error(
                                    f"Failed to pull model: {status.get('error', 'Unknown error')}"
                                )
                                return False
                        except json.JSONDecodeError:
                            pass

                logger.info(
                    f"Successfully pulled model {self.config.model_name}"
                )

            return True

        except Exception as e:
            logger.error(f"Failed to load model in Ollama: {e}")
            return False

    def generate(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Generate text using Ollama."""
        start_time = time.time()

        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
                "num_ctx": self.config.context_size,
                "num_thread": self.config.threads,
                "num_gpu": self.config.gpu_layers,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = self.session.post(
                f"{self.base_url}/api/generate", json=payload
            )
            response.raise_for_status()

            result = response.json()
            generation_time = time.time() - start_time

            return LLMResponse(
                text=result.get("response", ""),
                tokens_used=result.get(
                    "eval_count", 0
                ),  # Use eval_count for tokens
                generation_time=generation_time,
                provider=LocalLLMProvider.OLLAMA,
                metadata={
                    "model": self.config.model_name,
                    "eval_count": result.get("eval_count", 0),
                    "prompt_eval_count": result.get("prompt_eval_count", 0),
                    "total_duration": result.get("total_duration", 0),
                },
            )

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise


class LlamaCppProvider:
    """Provider for llama.cpp direct execution."""

    def __init__(self, config: LocalLLMConfig):
        """Initialize llama.cpp provider."""
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.model_loaded = False

        # Find llama.cpp binary
        if config.llama_cpp_binary:
            self.binary_path = config.llama_cpp_binary
        else:
            # Try common locations
            possible_paths = [
                "./llama.cpp/main",
                "/usr/local/bin/llama",
                "~/llama.cpp/main",
                "./main",
            ]
            for path in possible_paths:
                expanded_path = os.path.expanduser(path)
                if os.path.exists(expanded_path):
                    self.binary_path = expanded_path
                    break
            else:
                self.binary_path = "llama"  # Hope it's in PATH

    def is_available(self) -> bool:
        """Check if llama.cpp is available."""
        try:
            result = subprocess.run(
                [self.binary_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def load_model(self) -> bool:
        """Load model for llama.cpp."""
        if not self.config.model_path:
            logger.error("No model path specified for llama.cpp")
            return False

        if not os.path.exists(self.config.model_path):
            logger.error(f"Model file not found: {self.config.model_path}")
            return False

        self.model_loaded = True
        return True

    def generate(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Generate text using llama.cpp."""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")

        start_time = time.time()

        # Build command
        cmd = [
            self.binary_path,
            "-m",
            self.config.model_path,
            "-p",
            prompt,
            "-n",
            str(self.config.max_tokens),
            "-c",
            str(self.config.context_size),
            "-t",
            str(self.config.threads),
            "--temp",
            str(self.config.temperature),
        ]

        if self.config.gpu_layers > 0:
            cmd.extend(["-ngl", str(self.config.gpu_layers)])

        if self.config.use_mmap:
            cmd.append("--mmap")

        if self.config.use_mlock:
            cmd.append("--mlock")

        if system_prompt:
            cmd.extend(["--system", system_prompt])

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60
            )

            if result.returncode != 0:
                raise RuntimeError(f"llama.cpp failed: {result.stderr}")

            generation_time = time.time() - start_time

            # Parse output (llama.cpp outputs the generated text directly)
            generated_text = result.stdout.strip()

            # Try to extract token count from stderr
            tokens_used = 0
            if "tokens" in result.stderr:
                # Simple parsing - actual format may vary
                import re

                match = re.search(r"(\d+) tokens", result.stderr)
                if match:
                    tokens_used = int(match.group(1))

            return LLMResponse(
                text=generated_text,
                tokens_used=tokens_used,
                generation_time=generation_time,
                provider=LocalLLMProvider.LLAMA_CPP,
                metadata={
                    "model": self.config.model_path,
                    "command": " ".join(cmd),
                },
            )

        except subprocess.TimeoutExpired:
            logger.error("llama.cpp generation timed out")
            raise
        except Exception as e:
            logger.error(f"llama.cpp generation failed: {e}")
            raise


class ResponseCache:
    """Cache for LLM responses."""

    def __init__(
        self, max_size_bytes: int = 100 * 1024 * 1024
    ):  # 100MB default
        """Initialize response cache."""
        self.max_size_bytes = max_size_bytes
        self.cache: Dict[str, LLMResponse] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.lock = threading.Lock()
        self.max_memory_entries = (
            1000  # Limit entries to prevent unbounded growth
        )

    def _get_cache_key(
        self,
        prompt: str,
        system_prompt: Optional[str],
        provider: str,
        model: str,
    ) -> str:
        """Generate cache key from request parameters."""
        content = f"{provider}:{model}:{system_prompt or ''}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(
        self,
        prompt: str,
        system_prompt: Optional[str],
        provider: str,
        model: str,
    ) -> Optional[LLMResponse]:
        """Get cached response if available."""
        key = self._get_cache_key(prompt, system_prompt, provider, model)

        with self.lock:
            if key in self.cache:
                self.cache_hits += 1
                response = self.cache[key]
                # Mark as cached
                return LLMResponse(
                    text=response.text,
                    tokens_used=response.tokens_used,
                    generation_time=0.0,  # Cached responses are instant
                    provider=response.provider,
                    cached=True,
                    metadata=response.metadata,
                )
            else:
                self.cache_misses += 1
                return None

    def put(
        self,
        prompt: str,
        system_prompt: Optional[str],
        provider: str,
        model: str,
        response: LLMResponse,
    ) -> None:
        """Cache a response."""
        key = self._get_cache_key(prompt, system_prompt, provider, model)

        with self.lock:
            # Simple eviction - remove oldest entries if too many
            if len(self.cache) >= self.max_memory_entries:
                # Remove first (oldest) entry
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]

            self.cache[key] = response

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = (
                self.cache_hits / total_requests if total_requests > 0 else 0
            )

            return {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_rate": hit_rate,
                "entries": len(self.cache),
                "max_entries": self.max_memory_entries,
            }


class FallbackResponder:
    """Provides fallback responses when LLM is unavailable."""

    def __init__(self):
        """Initialize fallback responder."""
        self.templates = {
            "greeting": [
                "Hello! I'm currently operating in fallback mode.",
                "Greetings! The main LLM service is temporarily unavailable.",
                "Hi there! I'm providing basic responses while the main service is offline.",
            ],
            "exploration": [
                "I'm exploring the environment to gather more information.",
                "Currently analyzing the surroundings for relevant data.",
                "Investigating the area for useful insights.",
            ],
            "trading": [
                "Processing trade request. Please wait for confirmation.",
                "Trade operation initiated. Awaiting system response.",
                "Evaluating trade parameters for optimal execution.",
            ],
            "error": [
                "I encountered an issue processing your request.",
                "Unable to complete the requested operation at this time.",
                "An error occurred. Please try again later.",
            ],
        }

    def get_fallback_response(self, prompt: str) -> LLMResponse:
        """Generate a fallback response based on prompt content."""
        prompt_lower = prompt.lower()

        # Determine response type
        if any(word in prompt_lower for word in ["hello", "hi", "greet"]):
            category = "greeting"
        elif any(
            word in prompt_lower for word in ["explore", "search", "find"]
        ):
            category = "exploration"
        elif any(word in prompt_lower for word in ["trade", "buy", "sell"]):
            category = "trading"
        else:
            category = "error"

        # Select response
        import random

        response_text = random.choice(self.templates[category])

        return LLMResponse(
            text=response_text,
            tokens_used=len(response_text.split()),
            generation_time=0.01,
            provider="fallback",
            fallback_used=True,
            metadata={"category": category},
        )


class LocalLLMManager:
    """Manages local LLM providers with caching and fallback."""

    def __init__(self, config: LocalLLMConfig):
        """Initialize the local LLM manager."""
        self.config = config
        self.providers: Dict[str, Union[OllamaProvider, LlamaCppProvider]] = {}
        self.current_provider: Optional[
            Union[OllamaProvider, LlamaCppProvider]
        ] = None
        self.cache = ResponseCache()
        self.fallback_responder = FallbackResponder()

        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Initialize configured providers."""
        # Initialize primary provider
        if self.config.provider == LocalLLMProvider.OLLAMA:
            self.providers["ollama"] = OllamaProvider(self.config)
        else:
            self.providers["llama_cpp"] = LlamaCppProvider(self.config)

        # Initialize fallback provider if different
        if self.config.enable_fallback:
            if self.config.provider == LocalLLMProvider.OLLAMA:
                # Use llama.cpp as fallback
                fallback_config = LocalLLMConfig(
                    provider=LocalLLMProvider.LLAMA_CPP,
                    model_path=self.config.model_path,
                    model_name=self.config.model_name,
                    quantization=self.config.quantization,
                    context_size=self.config.context_size,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    threads=self.config.threads,
                    gpu_layers=self.config.gpu_layers,
                )
                self.providers["llama_cpp"] = LlamaCppProvider(fallback_config)

    def load_model(self) -> bool:
        """Load model in the configured provider."""
        for name, provider in self.providers.items():
            if provider.is_available():
                if provider.load_model():
                    self.current_provider = provider
                    logger.info(f"Loaded model with {name} provider")
                    return True
                else:
                    logger.warning(
                        f"Failed to load model with {name} provider"
                    )

        logger.error("Failed to load model with any provider")
        return False

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        use_cache: bool = True,
    ) -> LLMResponse:
        """Generate text with caching and fallback support."""
        # Check cache first
        if use_cache:
            cached_response = self.cache.get(
                prompt,
                system_prompt,
                self.config.provider.value,
                self.config.model_name,
            )
            if cached_response:
                return cached_response

        # Try generation with retries
        for attempt in range(self.config.retry_attempts):
            try:
                response = self._try_generate(prompt, system_prompt)

                # Cache successful response
                if use_cache and not response.fallback_used:
                    self.cache.put(
                        prompt,
                        system_prompt,
                        self.config.provider.value,
                        self.config.model_name,
                        response,
                    )

                return response

            except Exception as e:
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")

                if attempt < self.config.retry_attempts - 1:
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff

        # All retries failed, use fallback
        if self.config.enable_fallback:
            logger.warning("All generation attempts failed, using fallback")
            return self.fallback_responder.get_fallback_response(prompt)
        else:
            raise RuntimeError(
                "All generation attempts failed and fallback is disabled"
            )

    def _try_generate(
        self, prompt: str, system_prompt: Optional[str]
    ) -> LLMResponse:
        """Try to generate with available providers."""
        if not self.current_provider:
            self.load_model()

        if self.current_provider:
            return self.current_provider.generate(prompt, system_prompt)

        # Try other providers
        for name, provider in self.providers.items():
            if provider != self.current_provider and provider.is_available():
                try:
                    if provider.load_model():
                        response = provider.generate(prompt, system_prompt)
                        self.current_provider = provider
                        return response
                except Exception as e:
                    logger.warning(f"Provider {name} failed: {e}")

        raise RuntimeError("No providers available for generation")

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the LLM manager."""
        provider_status = {}
        for name, provider in self.providers.items():
            provider_status[name] = {
                "available": provider.is_available(),
                "is_current": provider == self.current_provider,
            }

        return {
            "config": {
                "provider": self.config.provider.value,
                "model": self.config.model_name,
                "context_size": self.config.context_size,
                "max_tokens": self.config.max_tokens,
            },
            "providers": provider_status,
            "cache_stats": self.cache.get_stats(),
            "fallback_enabled": self.config.enable_fallback,
        }

    def optimize_for_hardware(self) -> None:
        """Optimize configuration based on available hardware."""
        # Detect available CPU cores
        cpu_count = os.cpu_count() or 4
        self.config.threads = min(cpu_count - 1, 8)  # Leave one core free

        # Check for GPU (simplified - actual implementation would use proper GPU detection)
        try:
            import torch

            if torch.cuda.is_available():
                self.config.gpu_layers = 20  # Use GPU layers
                logger.info(
                    f"GPU detected, using {self.config.gpu_layers} GPU layers"
                )
        except ImportError:
            pass

        # Adjust context size based on available memory
        try:
            import psutil

            available_memory_gb = psutil.virtual_memory().available / (
                1024**3
            )
            if available_memory_gb < 4:
                self.config.context_size = 1024
                self.config.cache_size_mb = 256
            elif available_memory_gb < 8:
                self.config.context_size = 2048
                self.config.cache_size_mb = 512
            else:
                self.config.context_size = 4096
                self.config.cache_size_mb = 1024

            logger.info(
                f"Optimized for {available_memory_gb:.1f}GB available memory"
            )
        except ImportError:
            pass
