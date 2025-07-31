"""
Anthropic Provider Implementation.

Implements the ILLMProvider interface for Claude models.
Follows Clean Architecture principles with proper error handling and monitoring.
"""

import logging
import time
from typing import Any, Dict, Optional

try:
    import anthropic
    from anthropic import Anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

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


class AnthropicProvider(BaseProvider):
    """Anthropic provider implementation with real API integration."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Anthropic provider.

        Args:
            api_key: Optional API key for test compatibility. If provided, will configure the provider.
        """
        super().__init__(ProviderType.ANTHROPIC)
        self.client: Optional[Anthropic] = None
        self.api_key = api_key  # Store for test compatibility

        # Auto-configure if api_key is provided (for test compatibility)
        if api_key:
            credentials = ProviderCredentials(api_key=api_key)
            self.configure(credentials)
        self._model_pricing: Dict[str, Dict[str, float]] = {
            "claude-3-sonnet-20240229": {"input": 0.015, "output": 0.075},
            "claude-3-opus-20240229": {"input": 0.075, "output": 0.225},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
            "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
        }

        if not ANTHROPIC_AVAILABLE:
            logger.warning("Anthropic library not available. Install with: pip install anthropic")

    def configure(self, credentials: ProviderCredentials, **kwargs: Any) -> bool:
        """Configure the Anthropic provider."""
        if not ANTHROPIC_AVAILABLE:
            logger.error("Anthropic library not installed")
            return False

        if not credentials.is_complete():
            logger.error("Invalid Anthropic credentials")
            return False

        try:
            # Create Anthropic client
            client_kwargs = {"api_key": credentials.api_key}

            if credentials.endpoint_url:
                client_kwargs["base_url"] = credentials.endpoint_url

            # Add additional configuration
            if "timeout" in kwargs:
                client_kwargs["timeout"] = kwargs["timeout"]

            self.client = Anthropic(**client_kwargs)
            self.credentials = credentials
            self._configuration.update(kwargs)

            # Test connection
            health_result = self.test_connection()
            if health_result.status in [
                ProviderStatus.HEALTHY,
                ProviderStatus.DEGRADED,
            ]:
                logger.info("Anthropic provider configured successfully")
                return True
            else:
                logger.error(f"Anthropic provider unhealthy: {health_result.error_message}")
                return False

        except Exception as e:
            logger.error(f"Failed to configure Anthropic provider: {e}")
            return False

    def test_connection(self) -> HealthCheckResult:
        """Test connection to Anthropic API."""
        start_time = time.time()

        if not ANTHROPIC_AVAILABLE or not self.client:
            return HealthCheckResult(
                status=ProviderStatus.OFFLINE,
                latency_ms=0.0,
                error_message="Anthropic client not configured",
            )

        try:
            # Try a minimal request to test connection
            # Anthropic doesn't have a dedicated health check endpoint,
            # so we'll make a very small completion request
            # Make a minimal request to test connection
            self.client.messages.create(
                model="claude-3-haiku-20240307",  # Cheapest model
                max_tokens=1,
                messages=[{"role": "user", "content": "Hi"}],
            )

            latency_ms = (time.time() - start_time) * 1000

            # If we got here, the connection is working
            available_models = {model: True for model in self._model_pricing.keys()}

            return HealthCheckResult(
                status=ProviderStatus.HEALTHY,
                latency_ms=latency_ms,
                model_availability=available_models,
            )

        except anthropic.AuthenticationError as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=ProviderStatus.OFFLINE,
                latency_ms=latency_ms,
                error_message=f"Authentication failed: {str(e)}",
            )

        except anthropic.RateLimitError as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=ProviderStatus.DEGRADED,
                latency_ms=latency_ms,
                error_message=f"Rate limited: {str(e)}",
                rate_limit_info={"status": "rate_limited"},
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=ProviderStatus.UNHEALTHY,
                latency_ms=latency_ms,
                error_message=f"Connection test failed: {str(e)}",
            )
        finally:
            self._last_health_check = time.time()

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using Anthropic API."""
        if not ANTHROPIC_AVAILABLE or not self.client:
            raise RuntimeError("Anthropic provider not properly configured")

        start_time = time.time()

        try:
            # Prepare API request
            # Anthropic API expects messages in a specific format
            api_request = {
                "model": request.model,
                "messages": request.messages,
                "temperature": request.temperature,
            }

            if request.max_tokens:
                api_request["max_tokens"] = request.max_tokens
            else:
                # Anthropic requires max_tokens
                api_request["max_tokens"] = 1000

            # Note: Anthropic doesn't support streaming in the same way as OpenAI
            # We'll implement basic completion first

            # Make API call
            response = self.client.messages.create(**api_request)

            latency_ms = (time.time() - start_time) * 1000

            # Extract response data
            if not response.content:
                raise RuntimeError("No content returned in response")

            # Anthropic returns content as a list of content blocks
            text = ""
            for content_block in response.content:
                if hasattr(content_block, "text"):
                    text += content_block.text

            # Extract usage information
            input_tokens = response.usage.input_tokens if response.usage else 0
            output_tokens = response.usage.output_tokens if response.usage else 0

            # Calculate cost
            cost = self.estimate_cost(input_tokens, output_tokens, request.model)

            # Update metrics
            self._update_usage_metrics(
                success=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                latency_ms=latency_ms,
            )

            return GenerationResponse(
                text=text,
                model=response.model,
                provider=self.provider_type,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                latency_ms=latency_ms,
                finish_reason=response.stop_reason,
            )

        except anthropic.AuthenticationError as e:
            self._update_usage_metrics(success=False, error_type="authentication")
            raise RuntimeError(f"Anthropic authentication failed: {str(e)}")

        except anthropic.RateLimitError as e:
            self._update_usage_metrics(success=False, error_type="rate_limit")
            raise RuntimeError(f"Anthropic rate limit exceeded: {str(e)}")

        except anthropic.BadRequestError as e:
            self._update_usage_metrics(success=False, error_type="bad_request")
            raise RuntimeError(f"Anthropic bad request: {str(e)}")

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._update_usage_metrics(success=False, latency_ms=latency_ms, error_type="unknown")
            raise RuntimeError(f"Anthropic generation failed: {str(e)}")

    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Estimate cost for given token usage."""
        if model not in self._model_pricing:
            # Use Claude-3-haiku pricing as default (cheapest)
            pricing = self._model_pricing.get(
                "claude-3-haiku-20240307", {"input": 0.00025, "output": 0.00125}
            )
        else:
            pricing = self._model_pricing[model]

        input_cost = (input_tokens / 1000.0) * pricing["input"]
        output_cost = (output_tokens / 1000.0) * pricing["output"]

        return input_cost + output_cost

    def get_supported_models(self) -> Dict[str, Dict[str, Any]]:
        """Get list of supported models and their properties."""
        return {
            "claude-3-haiku-20240307": {
                "context_window": 200000,
                "max_output_tokens": 4096,
                "supports_function_calling": False,
                "supports_streaming": True,
                "pricing": self._model_pricing["claude-3-haiku-20240307"],
            },
            "claude-3-sonnet-20240229": {
                "context_window": 200000,
                "max_output_tokens": 4096,
                "supports_function_calling": False,
                "supports_streaming": True,
                "pricing": self._model_pricing["claude-3-sonnet-20240229"],
            },
            "claude-3-opus-20240229": {
                "context_window": 200000,
                "max_output_tokens": 4096,
                "supports_function_calling": False,
                "supports_streaming": True,
                "pricing": self._model_pricing["claude-3-opus-20240229"],
            },
            "claude-3-5-sonnet-20241022": {
                "context_window": 200000,
                "max_output_tokens": 8192,
                "supports_function_calling": True,
                "supports_streaming": True,
                "pricing": self._model_pricing["claude-3-5-sonnet-20241022"],
            },
        }
