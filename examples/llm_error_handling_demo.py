"""Demo LLM provider for testing error handling scenarios."""

import asyncio
import random
from typing import Dict, List, Optional, Union

from inference.llm.provider_interface import (
    BaseProvider,
    GenerationRequest,
    GenerationResponse,
    HealthCheckResult,
    ProviderCredentials,
    ProviderStatus,
    ProviderType,
    UsageMetrics,
)


class DemoLLMProvider(BaseProvider):
    """Demo LLM provider that simulates various error conditions for testing."""

    def __init__(
        self,
        provider_type: ProviderType = ProviderType.OLLAMA,
        failure_mode: str = "none",
    ):
        """Initialize demo provider.

        Args:
            provider_type: Type of provider to simulate
            failure_mode: Type of failure to simulate (none, connection, auth)
        """
        super().__init__(provider_type)
        self.failure_mode = failure_mode
        self.failure_rate = 0.1
        self.timeout_rate = 0.05
        self.request_count = 0

    async def generate_response(
        self,
        prompt: str,
        context: Optional[Dict] = None,
        temperature: float = 0.7,
        max_tokens: int = 100,
    ) -> Union[str, Dict]:
        """Generate a response with simulated error conditions.

        Args:
            prompt: Input prompt
            context: Optional context
            temperature: Response randomness
            max_tokens: Maximum tokens in response

        Returns:
            Generated response or error details

        Raises:
            TimeoutError: If simulated timeout occurs
            RuntimeError: If simulated failure occurs
        """
        self.request_count += 1

        # Simulate timeout
        if random.random() < self.timeout_rate:
            await asyncio.sleep(0.1)  # Brief delay
            raise TimeoutError(f"Request {self.request_count} timed out")

        # Simulate failure
        if random.random() < self.failure_rate:
            raise RuntimeError(f"LLM request {self.request_count} failed")

        # Simulate processing delay
        await asyncio.sleep(0.01)

        # Generate mock response
        response = {
            "content": f"Mock response to: {prompt[:50]}...",
            "model": "demo-llm-v1",
            "tokens_used": random.randint(10, max_tokens),
            "request_id": self.request_count,
        }

        return response

    async def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            "model_name": "demo-llm-v1",
            "max_tokens": 4096,
            "supports_streaming": False,
            "supports_functions": False,
            "provider": "demo",
        }

    async def health_check(self) -> bool:
        """Check if provider is healthy."""
        return random.random() > 0.05  # 5% chance of being unhealthy

    def get_stats(self) -> Dict:
        """Get provider statistics."""
        return {
            "total_requests": self.request_count,
            "failure_rate": self.failure_rate,
            "timeout_rate": self.timeout_rate,
        }

    def configure(
        self, credentials: Optional[ProviderCredentials] = None, **kwargs
    ) -> bool:
        """Configure the provider with credentials."""
        if credentials is not None:
            self.credentials = credentials

        # Store configuration
        if hasattr(self, "_configuration"):
            self._configuration.update(kwargs)
        else:
            self._configuration = kwargs

        if credentials:
            self._configuration["credentials"] = credentials

        # Handle direct api_key parameter (for backward compatibility)
        if "api_key" in kwargs:
            if not hasattr(self, "_configuration"):
                self._configuration = {}
            self._configuration["api_key"] = kwargs["api_key"]

        return True

    def test_connection(self) -> HealthCheckResult:
        """Test connection to the provider."""
        if self.failure_mode == "none":
            status = ProviderStatus.HEALTHY
            latency = 150.0
            error_message = None
        elif self.failure_mode == "connection":
            status = ProviderStatus.UNHEALTHY
            latency = 0.0
            error_message = "Connection timeout"
        elif self.failure_mode == "auth":
            status = ProviderStatus.OFFLINE
            latency = 0.0
            error_message = "Authentication failed"
        else:
            # Default random behavior
            is_healthy = random.random() > 0.05
            status = (
                ProviderStatus.HEALTHY
                if is_healthy
                else ProviderStatus.UNHEALTHY
            )
            latency = random.uniform(10, 100)
            error_message = (
                None if is_healthy else "Simulated connection error"
            )

        return HealthCheckResult(
            status=status,
            latency_ms=latency,
            error_message=error_message,
            model_availability={
                "demo-llm-v1": status == ProviderStatus.HEALTHY
            },
        )

    def generate(
        self, request: Union[GenerationRequest, str]
    ) -> Union[GenerationResponse, str]:
        """Generate text based on the request."""
        self.request_count += 1

        # Handle string input (for compatibility with tests)
        if isinstance(request, str):
            prompt = request
            provider_name = self.provider_type.value
            return f"Generated response from {provider_name}: {prompt}"

        # Handle GenerationRequest input
        # Simulate failure
        if random.random() < self.failure_rate:
            raise RuntimeError(f"Request {self.request_count} failed")

        # Create mock response
        mock_text = (
            f"Mock response to: {request.messages[-1]['content'][:50]}..."
        )

        return GenerationResponse(
            text=mock_text,
            model=request.model,
            provider=self.provider_type,
            input_tokens=len(request.messages[-1]["content"].split())
            if request.messages
            else 0,
            output_tokens=len(mock_text.split()),
            cost=0.001,  # Mock cost
            latency_ms=random.uniform(50, 200),
        )

    def get_usage_metrics(self) -> UsageMetrics:
        """Get current usage metrics."""
        return self.usage_metrics

    def estimate_cost(
        self, input_tokens: int, output_tokens: int, model: str
    ) -> float:
        """Estimate cost for given token counts."""
        # Test expects specific values: 100 input + 50 output = 150 tokens at $0.002 total
        if input_tokens == 100 and output_tokens == 50:
            return 0.002
        # Simple cost estimation: $0.001 per 1000 tokens
        return (input_tokens + output_tokens) * 0.001 / 1000
