"""
OpenAI Provider Implementation.

Implements the ILLMProvider interface for OpenAI GPT models.
Follows Clean Architecture principles with proper error handling and monitoring.
"""

import logging
import time
from typing import Any, Dict, Optional

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .provider_interface import (
    BaseProvider,
    GenerationRequest,
    GenerationResponse,
    HealthCheckResult,
    ProviderCredentials,
    ProviderStatus,
    ProviderType,
)

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """OpenAI provider implementation with real API integration."""

    def __init__(self):
        """Initialize OpenAI provider."""
        super().__init__(ProviderType.OPENAI)
        self.client: Optional[OpenAI] = None
        self._model_pricing: Dict[str, Dict[str, float]] = {
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-32k": {"input": 0.06, "output": 0.12},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        }

        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI library not available. Install with: pip install openai")

    def configure(self, credentials: ProviderCredentials, **kwargs: Any) -> bool:
        """Configure the OpenAI provider."""
        if not OPENAI_AVAILABLE:
            logger.error("OpenAI library not installed")
            return False

        if not credentials.is_complete():
            logger.error("Invalid OpenAI credentials")
            return False

        try:
            # Create OpenAI client
            client_kwargs = {"api_key": credentials.api_key}

            if credentials.organization_id:
                client_kwargs["organization"] = credentials.organization_id

            if credentials.endpoint_url:
                client_kwargs["base_url"] = credentials.endpoint_url

            # Add additional configuration
            if "timeout" in kwargs:
                client_kwargs["timeout"] = kwargs["timeout"]

            self.client = OpenAI(**client_kwargs)
            self.credentials = credentials
            self._configuration.update(kwargs)

            # Test connection
            health_result = self.test_connection()
            if health_result.status in [ProviderStatus.HEALTHY,
                ProviderStatus.DEGRADED]:
                logger.info("OpenAI provider configured successfully")
                return True
            else:
                logger.error(f"OpenAI provider unhealthy: {health_result.error_message}")
                return False

        except Exception as e:
            logger.error(f"Failed to configure OpenAI provider: {e}")
            return False

    def test_connection(self) -> HealthCheckResult:
        """Test connection to OpenAI API."""
        start_time = time.time()

        if not OPENAI_AVAILABLE or not self.client:
            return HealthCheckResult(
                status=ProviderStatus.OFFLINE,
                latency_ms=0.0,
                error_message="OpenAI client not configured"
            )

        try:
            # Try to list models as a health check
            models_response = self.client.models.list()
            latency_ms = (time.time() - start_time) * 1000

            # Check if we have access to models
            available_models = {model.id: True for model in models_response.data}

            if not available_models:
                return HealthCheckResult(
                    status=ProviderStatus.DEGRADED,
                    latency_ms=latency_ms,
                    error_message="No models available",
                    model_availability=available_models
                )

            # Successful health check
            return HealthCheckResult(
                status=ProviderStatus.HEALTHY,
                latency_ms=latency_ms,
                model_availability=available_models
            )

        except openai.AuthenticationError as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=ProviderStatus.OFFLINE,
                latency_ms=latency_ms,
                error_message=f"Authentication failed: {str(e)}"
            )

        except openai.RateLimitError as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=ProviderStatus.DEGRADED,
                latency_ms=latency_ms,
                error_message=f"Rate limited: {str(e)}",
                rate_limit_info={"status": "rate_limited"}
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=ProviderStatus.UNHEALTHY,
                latency_ms=latency_ms,
                error_message=f"Connection test failed: {str(e)}"
            )
        finally:
            self._last_health_check = time.time()

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using OpenAI API."""
        if not OPENAI_AVAILABLE or not self.client:
            raise RuntimeError("OpenAI provider not properly configured")

        start_time = time.time()

        try:
            # Prepare API request
            api_request = {
                "model": request.model,
                "messages": request.messages,
                "temperature": request.temperature,
                "stream": request.stream,
            }

            if request.max_tokens:
                api_request["max_tokens"] = request.max_tokens

            # Make API call
            response = self.client.chat.completions.create(**api_request)

            latency_ms = (time.time() - start_time) * 1000

            # Extract response data
            if not response.choices:
                raise RuntimeError("No response choices returned")

            choice = response.choices[0]
            text = choice.message.content or ""
            finish_reason = choice.finish_reason

            # Extract usage information
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0

            # Calculate cost
            cost = self.estimate_cost(input_tokens, output_tokens, request.model)

            # Update metrics
            self._update_usage_metrics(
                success=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                latency_ms=latency_ms
            )

            return GenerationResponse(
                text=text,
                model=response.model,
                provider=self.provider_type,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                latency_ms=latency_ms,
                finish_reason=finish_reason
            )

        except openai.AuthenticationError as e:
            self._update_usage_metrics(success=False, error_type="authentication")
            raise RuntimeError(f"OpenAI authentication failed: {str(e)}")

        except openai.RateLimitError as e:
            self._update_usage_metrics(success=False, error_type="rate_limit")
            raise RuntimeError(f"OpenAI rate limit exceeded: {str(e)}")

        except openai.BadRequestError as e:
            self._update_usage_metrics(success=False, error_type="bad_request")
            raise RuntimeError(f"OpenAI bad request: {str(e)}")

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._update_usage_metrics(
                success=False,
                latency_ms=latency_ms,
                error_type="unknown"
            )
            raise RuntimeError(f"OpenAI generation failed: {str(e)}")

    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Estimate cost for given token usage."""
        if model not in self._model_pricing:
            # Use GPT-3.5-turbo pricing as default
            pricing = self._model_pricing.get("gpt-3.5-turbo", {"input": 0.0005,
                "output": 0.0015})
        else:
            pricing = self._model_pricing[model]

        input_cost = (input_tokens / 1000.0) * pricing["input"]
        output_cost = (output_tokens / 1000.0) * pricing["output"]

        return input_cost + output_cost

    def get_supported_models(self) -> Dict[str, Dict[str, Any]]:
        """Get list of supported models and their properties."""
        return {
            "gpt-3.5-turbo": {
                "context_window": 16385,
                "max_output_tokens": 4096,
                "supports_function_calling": True,
                "supports_streaming": True,
                "pricing": self._model_pricing["gpt-3.5-turbo"]
            },
            "gpt-4": {
                "context_window": 8192,
                "max_output_tokens": 4096,
                "supports_function_calling": True,
                "supports_streaming": True,
                "pricing": self._model_pricing["gpt-4"]
            },
            "gpt-4-turbo": {
                "context_window": 128000,
                "max_output_tokens": 4096,
                "supports_function_calling": True,
                "supports_streaming": True,
                "pricing": self._model_pricing["gpt-4-turbo"]
            },
            "gpt-4o": {
                "context_window": 128000,
                "max_output_tokens": 4096,
                "supports_function_calling": True,
                "supports_streaming": True,
                "pricing": self._model_pricing["gpt-4o"]
            },
            "gpt-4o-mini": {
                "context_window": 128000,
                "max_output_tokens": 16384,
                "supports_function_calling": True,
                "supports_streaming": True,
                "pricing": self._model_pricing["gpt-4o-mini"]
            }
        }
