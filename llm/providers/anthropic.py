"""Anthropic LLM provider for Claude integration.

This provider enables FreeAgentics to use Anthropic's Claude models for
GMN generation and prompt processing.
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp

from llm.base import LLMError, LLMMessage, LLMProvider, LLMResponse, LLMRole


class AnthropicProvider(LLMProvider):
    """Anthropic provider for Claude models."""

    # Model configurations with context windows and rate limits
    MODEL_CONFIGS = {
        "claude-3-opus-20240229": {
            "tokens": 200000,
            "rpm": 1000,
            "tpm": 40000,
        },
        "claude-3-sonnet-20240229": {
            "tokens": 200000,
            "rpm": 1000,
            "tpm": 40000,
        },
        "claude-3-haiku-20240307": {
            "tokens": 200000,
            "rpm": 1000,
            "tpm": 50000,
        },
        "claude-3-5-sonnet-20240620": {
            "tokens": 200000,
            "rpm": 1000,
            "tpm": 80000,
        },
        "claude-3-5-sonnet-20241022": {
            "tokens": 200000,
            "rpm": 1000,
            "tpm": 80000,
        },
        "claude-3-5-haiku-20241022": {
            "tokens": 200000,
            "rpm": 1000,
            "tpm": 50000,
        },
        # Aliases for convenience
        "claude-3-opus": {"tokens": 200000, "rpm": 1000, "tpm": 40000},
        "claude-3-sonnet": {"tokens": 200000, "rpm": 1000, "tpm": 40000},
        "claude-3-haiku": {"tokens": 200000, "rpm": 1000, "tpm": 50000},
        "claude-3.5-sonnet": {"tokens": 200000, "rpm": 1000, "tpm": 80000},
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        base_url: str = "https://api.anthropic.com",
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        anthropic_version: str = "2023-06-01",
    ):
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model to use (claude-3-opus, claude-3-sonnet, claude-3-haiku, etc.)
            base_url: API base URL
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries (exponential backoff)
            anthropic_version: API version to use
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise LLMError(
                "Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable."
            )

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.anthropic_version = anthropic_version

        # Rate limiting tracking
        self.request_times: List[datetime] = []
        self.token_usage: List[tuple[datetime, int]] = []

        # Session for connection pooling
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            headers = {
                "anthropic-version": self.anthropic_version,
                "x-api-key": self.api_key,
                "content-type": "application/json",
            }

            self._session = aiohttp.ClientSession(headers=headers)

        return self._session

    async def _check_rate_limits(self, model: str) -> None:
        """Check and enforce rate limits."""
        # Map model to base model for config lookup
        base_model = model
        for key in self.MODEL_CONFIGS:
            if key in model:
                base_model = key
                break

        config = self.MODEL_CONFIGS.get(
            base_model, self.MODEL_CONFIGS["claude-3-sonnet"]
        )
        now = datetime.now()

        # Clean old request times (older than 1 minute)
        self.request_times = [
            t for t in self.request_times if now - t < timedelta(minutes=1)
        ]
        self.token_usage = [
            (t, tokens)
            for t, tokens in self.token_usage
            if now - t < timedelta(minutes=1)
        ]

        # Check requests per minute
        if len(self.request_times) >= config["rpm"]:
            oldest_request = self.request_times[0]
            wait_time = (
                oldest_request + timedelta(minutes=1) - now
            ).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        # Check tokens per minute
        total_tokens = sum(tokens for _, tokens in self.token_usage)
        if total_tokens >= config["tpm"]:
            oldest_token_time = self.token_usage[0][0]
            wait_time = (
                oldest_token_time + timedelta(minutes=1) - now
            ).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)

    def _convert_messages(
        self, messages: List[LLMMessage]
    ) -> tuple[Optional[str], List[Dict[str, str]]]:
        """Convert messages to Anthropic format.

        Returns:
            Tuple of (system_prompt, messages)
        """
        system_prompt = None
        anthropic_messages = []

        for msg in messages:
            if msg.role == LLMRole.SYSTEM:
                # Anthropic uses a separate system parameter
                if system_prompt is None:
                    system_prompt = msg.content
                else:
                    # Append additional system messages
                    system_prompt += "\n\n" + msg.content
            else:
                anthropic_messages.append(
                    {
                        "role": "user"
                        if msg.role == LLMRole.USER
                        else "assistant",
                        "content": msg.content,
                    }
                )

        return system_prompt, anthropic_messages

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response using Anthropic's API."""
        # Check rate limits
        await self._check_rate_limits(self.model)

        # Convert messages to Anthropic format
        system_prompt, anthropic_messages = self._convert_messages(messages)

        # Build request payload
        payload = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,  # Anthropic requires max_tokens
        }

        if system_prompt:
            payload["system"] = system_prompt

        if stop_sequences:
            payload["stop_sequences"] = stop_sequences

        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in payload:
                payload[key] = value

        # Retry logic with exponential backoff
        last_error = None
        for attempt in range(self.max_retries):
            try:
                session = await self._get_session()

                # Record request time
                self.request_times.append(datetime.now())

                async with session.post(
                    f"{self.base_url}/v1/messages",
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
                        error_msg = response_json.get("error", {}).get(
                            "message", "Unknown error"
                        )
                        error_type = response_json.get("error", {}).get(
                            "type", "unknown"
                        )

                        # Handle specific error types
                        if error_type == "rate_limit_error":
                            # Wait longer for rate limit errors
                            await asyncio.sleep(10)

                        raise LLMError(
                            f"Anthropic API error (status {response.status}): {error_msg}"
                        )

                    # Extract response data
                    content = response_json["content"][0]["text"]

                    # Extract usage information
                    usage = response_json.get("usage", {})
                    if usage:
                        # Convert to OpenAI-style usage format
                        usage = {
                            "prompt_tokens": usage.get("input_tokens", 0),
                            "completion_tokens": usage.get("output_tokens", 0),
                            "total_tokens": usage.get("input_tokens", 0)
                            + usage.get("output_tokens", 0),
                        }
                        self.token_usage.append(
                            (datetime.now(), usage["total_tokens"])
                        )

                    return LLMResponse(
                        content=content,
                        model=response_json["model"],
                        usage=usage,
                        metadata={
                            "temperature": temperature,
                            "stop_reason": response_json.get("stop_reason"),
                            "id": response_json.get("id"),
                            "type": response_json.get("type"),
                        },
                        finish_reason=response_json.get("stop_reason"),
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
        # Check both full names and aliases
        if model_name in self.MODEL_CONFIGS:
            return True

        # Check if it's a base model name
        for key in self.MODEL_CONFIGS:
            if model_name in key or key in model_name:
                return True

        return False

    def get_token_limit(self, model_name: str) -> int:
        """Get the token limit for a specific model."""
        # Direct lookup
        if model_name in self.MODEL_CONFIGS:
            return self.MODEL_CONFIGS[model_name]["tokens"]

        # Find by partial match
        for key, config in self.MODEL_CONFIGS.items():
            if key in model_name or model_name in key:
                return config["tokens"]

        # Default to Claude 3 Sonnet limits
        return 200000

    async def close(self):
        """Close the session when done."""
        if self._session and not self._session.closed:
            await self._session.close()

    def __del__(self):
        """Cleanup on deletion."""
        if self._session and not self._session.closed:
            asyncio.create_task(self._session.close())
