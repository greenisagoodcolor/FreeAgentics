"""Ollama LLM provider for local model integration.

This provider enables FreeAgentics to use locally hosted models via Ollama
for GMN generation and prompt processing without requiring API keys.
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

from llm.base import LLMError, LLMMessage, LLMProvider, LLMResponse, LLMRole


class OllamaProvider(LLMProvider):
    """Ollama provider for local models."""

    # Popular models with their approximate context windows
    MODEL_CONFIGS = {
        # Llama models
        "llama3.2": {"tokens": 128000, "size": "3B"},
        "llama3.1": {"tokens": 128000, "size": "8B/70B/405B"},
        "llama3": {"tokens": 8192, "size": "8B/70B"},
        "llama2": {"tokens": 4096, "size": "7B/13B/70B"},
        "llama2-uncensored": {"tokens": 4096, "size": "7B/13B/70B"},
        # Mistral models
        "mistral": {"tokens": 8192, "size": "7B"},
        "mistral-large": {"tokens": 128000, "size": "123B"},
        "mixtral": {"tokens": 32768, "size": "8x7B"},
        "mixtral:8x22b": {"tokens": 65536, "size": "8x22B"},
        # Code models
        "codellama": {"tokens": 16384, "size": "7B/13B/34B/70B"},
        "deepseek-coder-v2": {"tokens": 128000, "size": "236B"},
        "codegemma": {"tokens": 8192, "size": "7B"},
        "starcoder2": {"tokens": 16384, "size": "15B"},
        # Other models
        "phi3": {"tokens": 128000, "size": "3.8B/14B"},
        "gemma2": {"tokens": 8192, "size": "9B/27B"},
        "qwen2.5": {"tokens": 128000, "size": "0.5B-72B"},
        "dolphin-mixtral": {"tokens": 32768, "size": "8x7B"},
        "neural-chat": {"tokens": 8192, "size": "7B"},
        "starling-lm": {"tokens": 8192, "size": "7B"},
        "yi": {"tokens": 32768, "size": "34B"},
        "solar": {"tokens": 8192, "size": "10.7B"},
        # Default fallback
        "default": {"tokens": 4096, "size": "unknown"},
    }

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        keep_alive: str = "5m",
        num_ctx: Optional[int] = None,
        num_predict: Optional[int] = None,
        num_gpu: Optional[int] = None,
    ):
        """Initialize Ollama provider.

        Args:
            model: Model to use (e.g., llama3.2, mistral, mixtral)
            base_url: Ollama API URL (default: http://localhost:11434)
            timeout: Request timeout in seconds (longer for local models)
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries
            keep_alive: How long to keep the model loaded in memory
            num_ctx: Context window size (overrides model default)
            num_predict: Maximum tokens to generate
            num_gpu: Number of layers to offload to GPU
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.num_gpu = num_gpu

        # Session for connection pooling
        self._session: Optional[aiohttp.ClientSession] = None

        # Cache for available models
        self._available_models: Optional[List[str]] = None
        self._models_last_check: Optional[datetime] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            # Ollama doesn't require auth headers
            self._session = aiohttp.ClientSession()

        return self._session

    async def _check_ollama_running(self) -> bool:
        """Check if Ollama service is running."""
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.base_url}/api/tags",
                timeout=aiohttp.ClientTimeout(total=5.0),
            ) as response:
                return response.status == 200
        except:
            return False

    async def _ensure_model_available(self, model: str) -> None:
        """Ensure the model is available locally, pull if needed."""
        # First check if model exists
        available = await self._get_available_models()

        # Check if model is already available
        model_base = model.split(":")[0]  # Handle tags like model:latest
        if any(model_base in m for m in available):
            return

        # Model not available, try to pull it
        try:
            session = await self._get_session()

            pull_data = {"name": model, "stream": True}

            async with session.post(
                f"{self.base_url}/api/pull",
                json=pull_data,
                timeout=aiohttp.ClientTimeout(
                    total=3600
                ),  # 1 hour for large models
            ) as response:
                if response.status != 200:
                    raise LLMError(f"Failed to pull model {model}")

                # Stream the pull progress
                async for line in response.content:
                    if line:
                        try:
                            progress = json.loads(line)
                            # Could log progress here if needed
                            if progress.get("status") == "success":
                                break
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            raise LLMError(f"Failed to pull model {model}: {str(e)}")

    async def _get_available_models(self) -> List[str]:
        """Get list of available models."""
        # Cache for 5 minutes
        if self._available_models and self._models_last_check:
            if (datetime.now() - self._models_last_check).seconds < 300:
                return self._available_models

        try:
            session = await self._get_session()
            async with session.get(
                f"{self.base_url}/api/tags",
                timeout=aiohttp.ClientTimeout(total=10.0),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self._available_models = [
                        m["name"] for m in data.get("models", [])
                    ]
                    self._models_last_check = datetime.now()
                    return self._available_models
        except:
            pass

        return []

    def _convert_messages(self, messages: List[LLMMessage]) -> str:
        """Convert messages to Ollama format (single prompt)."""
        # Ollama uses a template-based approach
        prompt_parts = []

        # Add system messages first
        system_messages = [
            msg for msg in messages if msg.role == LLMRole.SYSTEM
        ]
        if system_messages:
            system_content = "\n\n".join(
                msg.content for msg in system_messages
            )
            prompt_parts.append(f"System: {system_content}")

        # Add conversation history
        conversation_messages = [
            msg for msg in messages if msg.role != LLMRole.SYSTEM
        ]
        for msg in conversation_messages:
            role = "Human" if msg.role == LLMRole.USER else "Assistant"
            prompt_parts.append(f"{role}: {msg.content}")

        # Add final assistant prompt if last message was from user
        if messages and messages[-1].role == LLMRole.USER:
            prompt_parts.append("Assistant:")

        return "\n\n".join(prompt_parts)

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response using Ollama's API."""
        # Check if Ollama is running
        if not await self._check_ollama_running():
            raise LLMError(
                "Ollama service is not running. Start it with 'ollama serve'"
            )

        # Ensure model is available
        await self._ensure_model_available(self.model)

        # Convert messages to prompt
        prompt = self._convert_messages(messages)

        # Build request payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens or self.num_predict or -1,
            },
        }

        # Add optional parameters
        if self.num_ctx:
            payload["options"]["num_ctx"] = self.num_ctx

        if self.num_gpu is not None:
            payload["options"]["num_gpu"] = self.num_gpu

        if stop_sequences:
            payload["options"]["stop"] = stop_sequences

        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive

        # Add any additional options
        for key, value in kwargs.items():
            if key not in payload:
                payload[key] = value

        # Retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                session = await self._get_session()

                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    response_text = await response.text()

                    try:
                        response_json = json.loads(response_text)
                    except json.JSONDecodeError:
                        raise LLMError(
                            f"Invalid JSON response: {response_text[:200]}"
                        )

                    if response.status != 200:
                        error_msg = response_json.get("error", "Unknown error")
                        raise LLMError(
                            f"Ollama API error (status {response.status}): {error_msg}"
                        )

                    # Extract response data
                    content = response_json["response"]

                    # Build usage information (Ollama provides different metrics)
                    usage = {
                        "prompt_tokens": response_json.get(
                            "prompt_eval_count", 0
                        ),
                        "completion_tokens": response_json.get(
                            "eval_count", 0
                        ),
                        "total_tokens": response_json.get(
                            "prompt_eval_count", 0
                        )
                        + response_json.get("eval_count", 0),
                    }

                    # Performance metrics
                    metadata = {
                        "temperature": temperature,
                        "model": response_json.get("model"),
                        "created_at": response_json.get("created_at"),
                        "total_duration": response_json.get("total_duration"),
                        "load_duration": response_json.get("load_duration"),
                        "prompt_eval_duration": response_json.get(
                            "prompt_eval_duration"
                        ),
                        "eval_duration": response_json.get("eval_duration"),
                        "done": response_json.get("done", True),
                    }

                    # Calculate tokens per second if timing info available
                    if (
                        metadata.get("eval_duration")
                        and usage["completion_tokens"] > 0
                    ):
                        eval_duration_s = (
                            metadata["eval_duration"] / 1e9
                        )  # Convert nanoseconds to seconds
                        metadata["tokens_per_second"] = (
                            usage["completion_tokens"] / eval_duration_s
                        )

                    return LLMResponse(
                        content=content,
                        model=response_json.get("model", self.model),
                        usage=usage,
                        metadata=metadata,
                        finish_reason="stop"
                        if response_json.get("done")
                        else "length",
                    )

            except aiohttp.ClientError as e:
                last_error = LLMError(f"Network error: {str(e)}")
            except json.JSONDecodeError as e:
                last_error = LLMError(f"Invalid response format: {str(e)}")
            except KeyError as e:
                last_error = LLMError(
                    f"Unexpected response structure: {str(e)}"
                )
            except LLMError as e:
                last_error = e
            except Exception as e:
                last_error = LLMError(f"Unexpected error: {str(e)}")

            # If this isn't the last attempt, wait before retrying
            if attempt < self.max_retries - 1:
                wait_time = self.retry_delay * (2**attempt)
                await asyncio.sleep(wait_time)

        # All retries failed
        raise last_error or LLMError(
            "Failed to generate response after all retries"
        )

    async def validate_model(self, model_name: str) -> bool:
        """Check if a model is available."""
        # Check if Ollama is running
        if not await self._check_ollama_running():
            return False

        # Get available models
        available = await self._get_available_models()

        # Check exact match or base model match
        model_base = model_name.split(":")[0]
        return any(model_name == m or model_base in m for m in available)

    def get_token_limit(self, model_name: str) -> int:
        """Get the token limit for a specific model."""
        # Check exact match first
        if model_name in self.MODEL_CONFIGS:
            return self.MODEL_CONFIGS[model_name]["tokens"]

        # Check base model name
        model_base = model_name.split(":")[0]
        for key, config in self.MODEL_CONFIGS.items():
            if key == model_base or model_base in key:
                return config["tokens"]

        # Return default
        return self.MODEL_CONFIGS["default"]["tokens"]

    async def list_models(self) -> List[Dict[str, Any]]:
        """List all available models with their details."""
        models = await self._get_available_models()

        result = []
        for model in models:
            model_base = model.split(":")[0]
            config = self.MODEL_CONFIGS.get(
                model_base, self.MODEL_CONFIGS["default"]
            )

            result.append(
                {
                    "name": model,
                    "base": model_base,
                    "context_length": config["tokens"],
                    "size": config.get("size", "unknown"),
                }
            )

        return result

    async def close(self):
        """Close the session when done."""
        if self._session and not self._session.closed:
            await self._session.close()

    def __del__(self):
        """Cleanup on deletion."""
        if self._session and not self._session.closed:
            asyncio.create_task(self._session.close())
