"""OpenAI LLM provider for GPT-4 integration.

This provider enables FreeAgentics to use OpenAI's GPT-4 models for
GMN generation and prompt processing.
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import List, Optional

import aiohttp

from llm.base import LLMError, LLMMessage, LLMProvider, LLMResponse


class OpenAIProvider(LLMProvider):
    """OpenAI provider for GPT-4 models."""

    # Model configurations
    MODEL_CONFIGS = {
        "gpt-4": {"tokens": 8192, "rpm": 10000, "tpm": 300000},
        "gpt-4-turbo": {"tokens": 128000, "rpm": 10000, "tpm": 300000},
        "gpt-4o": {"tokens": 128000, "rpm": 10000, "tpm": 300000},
        "gpt-4o-mini": {"tokens": 128000, "rpm": 30000, "tpm": 150000000},
        "gpt-3.5-turbo": {"tokens": 16385, "rpm": 10000, "tpm": 2000000},
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        organization: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini, gpt-3.5-turbo)
            organization: Optional organization ID
            base_url: API base URL
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries (exponential backoff)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise LLMError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")

        self.model = model
        self.organization = organization
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Rate limiting tracking
        self.request_times: List[datetime] = []
        self.token_usage: List[tuple[datetime, int]] = []

        # Session for connection pooling
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            if self.organization:
                headers["OpenAI-Organization"] = self.organization

            self._session = aiohttp.ClientSession(headers=headers)

        return self._session

    async def _check_rate_limits(self, model: str) -> None:
        """Check and enforce rate limits."""
        config = self.MODEL_CONFIGS.get(model, self.MODEL_CONFIGS["gpt-4"])
        now = datetime.now()

        # Clean old request times (older than 1 minute)
        self.request_times = [t for t in self.request_times if now - t < timedelta(minutes=1)]
        self.token_usage = [
            (t, tokens) for t, tokens in self.token_usage if now - t < timedelta(minutes=1)
        ]

        # Check requests per minute
        if len(self.request_times) >= config["rpm"]:
            # Calculate wait time
            oldest_request = self.request_times[0]
            wait_time = (oldest_request + timedelta(minutes=1) - now).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        # Check tokens per minute
        total_tokens = sum(tokens for _, tokens in self.token_usage)
        if total_tokens >= config["tpm"]:
            # Wait for token window to clear
            oldest_token_time = self.token_usage[0][0]
            wait_time = (oldest_token_time + timedelta(minutes=1) - now).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response using OpenAI's API."""
        # Check rate limits
        await self._check_rate_limits(self.model)

        # Convert messages to OpenAI format
        openai_messages = [{"role": msg.role.value, "content": msg.content} for msg in messages]

        # Build request payload
        payload = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": temperature,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        if stop_sequences:
            payload["stop"] = stop_sequences

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
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    response_json = await response.json()

                    if response.status != 200:
                        error_msg = response_json.get("error", {}).get("message", "Unknown error")
                        raise LLMError(f"OpenAI API error (status {response.status}): {error_msg}")

                    # Extract response data
                    choice = response_json["choices"][0]
                    content = choice["message"]["content"]
                    finish_reason = choice.get("finish_reason")

                    # Extract usage information
                    usage = response_json.get("usage", {})
                    if usage:
                        total_tokens = usage.get("total_tokens", 0)
                        self.token_usage.append((datetime.now(), total_tokens))

                    return LLMResponse(
                        content=content,
                        model=response_json["model"],
                        usage=usage,
                        metadata={
                            "temperature": temperature,
                            "finish_reason": finish_reason,
                            "id": response_json.get("id"),
                            "created": response_json.get("created"),
                        },
                        finish_reason=finish_reason,
                    )

            except aiohttp.ClientError as e:
                last_error = LLMError(f"Network error: {str(e)}")
            except json.JSONDecodeError as e:
                last_error = LLMError(f"Invalid response format: {str(e)}")
            except KeyError as e:
                last_error = LLMError(f"Unexpected response structure: {str(e)}")
            except LLMError as e:
                last_error = e
            except Exception as e:
                last_error = LLMError(f"Unexpected error: {str(e)}")

            # If this isn't the last attempt, wait before retrying
            if attempt < self.max_retries - 1:
                wait_time = self.retry_delay * (2**attempt)  # Exponential backoff
                await asyncio.sleep(wait_time)

        # All retries failed
        raise last_error or LLMError("Failed to generate response after all retries")

    async def validate_model(self, model_name: str) -> bool:
        """Check if a model is available."""
        return model_name in self.MODEL_CONFIGS

    def get_token_limit(self, model_name: str) -> int:
        """Get the token limit for a specific model."""
        config = self.MODEL_CONFIGS.get(model_name)
        return config["tokens"] if config else 0

    async def close(self):
        """Close the session when done."""
        if self._session and not self._session.closed:
            await self._session.close()

    def __del__(self):
        """Cleanup on deletion."""
        if self._session and not self._session.closed:
            asyncio.create_task(self._session.close())
