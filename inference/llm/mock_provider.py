"""Mock LLM Provider for development and testing.

This provider returns deterministic responses without making actual API calls.
Used in dev mode to avoid API key requirements and costs.
"""

import json
import logging
import time

from .provider_interface import (
    BaseProvider,
    GenerationRequest,
    GenerationResponse,
    HealthCheckResult,
    ModelCapability,
    ModelInfo,
    ProviderStatus,
    ProviderType,
)

logger = logging.getLogger(__name__)


class MockLLMProvider(BaseProvider):
    """Mock LLM provider that returns deterministic responses."""

    def __init__(self):
        """Initialize mock provider."""
        super().__init__(ProviderType.OPENAI)  # Pretend to be OpenAI for compatibility
        self.is_configured = False
        self.mock_models = {
            "gpt-4": ModelInfo(
                id="gpt-4",
                name="GPT-4 (Mock)",
                provider=ProviderType.OPENAI,
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT_COMPLETION],
                context_window=8192,
                max_output_tokens=4096,
                cost_per_1k_input_tokens=0.0,
                cost_per_1k_output_tokens=0.0,
                supports_streaming=True,
                supports_function_calling=True,
            ),
            "gpt-3.5-turbo": ModelInfo(
                id="gpt-3.5-turbo",
                name="GPT-3.5 Turbo (Mock)",
                provider=ProviderType.OPENAI,
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CHAT_COMPLETION],
                context_window=4096,
                max_output_tokens=2048,
                cost_per_1k_input_tokens=0.0,
                cost_per_1k_output_tokens=0.0,
                supports_streaming=True,
                supports_function_calling=True,
            ),
        }

    def configure(self, credentials=None, **kwargs) -> bool:
        """Configure mock provider (always succeeds in dev mode)."""
        logger.info("Configuring MockLLMProvider for dev mode")
        self.is_configured = True
        self._configuration.update(kwargs)
        return True

    def test_connection(self) -> HealthCheckResult:
        """Test connection (always healthy in mock mode)."""
        return HealthCheckResult(
            status=ProviderStatus.HEALTHY,
            latency_ms=10.0,
            error_message=None,
            model_availability={"gpt-4": True, "gpt-3.5-turbo": True},
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate mock response based on the request."""
        start_time = time.time()

        # Extract the prompt from messages
        prompt = ""
        for msg in request.messages:
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
                break

        # Generate appropriate mock response based on prompt content
        if "GMN specification" in prompt or "active inference" in prompt.lower():
            # Return a valid GMN specification for agent creation
            response_content = self._generate_mock_gmn(prompt)
        else:
            # Return a generic response
            response_content = self._generate_generic_response(prompt)

        # Calculate mock metrics
        latency_ms = (time.time() - start_time) * 1000 + 50  # Add some fake processing time
        input_tokens = len(prompt.split())
        output_tokens = len(response_content.split())

        # Update usage metrics
        self._update_usage_metrics(
            success=True,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=0.0,
            latency_ms=latency_ms,
        )

        # Create response with proper attribute
        response = GenerationResponse(
            text=response_content,
            model=request.model or "gpt-4",
            provider=ProviderType.OPENAI,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=0.0,
            latency_ms=latency_ms,
            finish_reason="stop",
        )

        # Add content attribute for compatibility
        response.content = response_content

        return response

    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Estimate cost (always 0 for mock provider)."""
        return 0.0

    def _generate_mock_gmn(self, prompt: str) -> str:
        """Generate a mock GMN specification."""
        # Extract agent type from prompt
        agent_type = "explorer"
        if "forag" in prompt.lower():
            agent_type = "forager"
        elif "trade" in prompt.lower() or "exchang" in prompt.lower():
            agent_type = "trader"
        elif "guard" in prompt.lower() or "patrol" in prompt.lower():
            agent_type = "guardian"

        # Create a simple but valid GMN spec
        gmn_spec = {
            "name": f"mock_{agent_type}_agent",
            "description": f"A mock {agent_type} agent for development",
            "states": ["idle", "exploring", "interacting"],
            "observations": ["nothing", "resource", "agent", "obstacle"],
            "actions": ["wait", "move_north", "move_south", "move_east", "move_west", "interact"],
            "parameters": {
                "A": [  # Observation model P(o|s)
                    [0.7, 0.1, 0.1, 0.1],  # idle state
                    [0.1, 0.6, 0.2, 0.1],  # exploring state
                    [0.1, 0.2, 0.6, 0.1],  # interacting state
                ],
                "B": [  # Transition model P(s'|s,a) for each action
                    # wait action
                    [
                        [1.0, 0.0, 0.0],  # from idle
                        [0.2, 0.8, 0.0],  # from exploring
                        [0.3, 0.0, 0.7],  # from interacting
                    ],
                    # move_north action
                    [
                        [0.1, 0.9, 0.0],  # from idle
                        [0.0, 0.9, 0.1],  # from exploring
                        [0.1, 0.8, 0.1],  # from interacting
                    ],
                    # move_south action
                    [
                        [0.1, 0.9, 0.0],  # from idle
                        [0.0, 0.9, 0.1],  # from exploring
                        [0.1, 0.8, 0.1],  # from interacting
                    ],
                    # move_east action
                    [
                        [0.1, 0.9, 0.0],  # from idle
                        [0.0, 0.9, 0.1],  # from exploring
                        [0.1, 0.8, 0.1],  # from interacting
                    ],
                    # move_west action
                    [
                        [0.1, 0.9, 0.0],  # from idle
                        [0.0, 0.9, 0.1],  # from exploring
                        [0.1, 0.8, 0.1],  # from interacting
                    ],
                    # interact action
                    [
                        [0.0, 0.0, 1.0],  # from idle
                        [0.0, 0.2, 0.8],  # from exploring
                        [0.0, 0.0, 1.0],  # from interacting
                    ],
                ],
                "C": [  # Observation preferences
                    [0.0, 1.0, 0.5, -1.0],  # prefer resources, neutral to agents, avoid obstacles
                ],
                "D": [[0.8, 0.2, 0.0]],  # Initial belief - mostly idle
            },
        }

        return json.dumps(gmn_spec, indent=2)

    def _generate_generic_response(self, prompt: str) -> str:
        """Generate a generic mock response."""
        responses = [
            "This is a mock response from the development LLM provider.",
            "I understand your request. In a real scenario, I would provide a detailed response.",
            "Mock mode active: Returning a test response for development purposes.",
        ]

        # Pick a response based on prompt length (deterministic)
        index = len(prompt) % len(responses)
        return responses[index]


class MockProvider(MockLLMProvider):
    """Alias for backward compatibility."""

    pass
