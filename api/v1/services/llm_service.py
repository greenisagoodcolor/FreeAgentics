"""
LLM Service for Agent Conversation API

Provides dependency injection wrapper for LLM providers with async GMN generation.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from fastapi import HTTPException

from inference.llm.provider_factory import LLMProviderFactory
from inference.llm.provider_interface import GenerationRequest

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Circuit breaker pattern implementation for LLM service calls."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker moving to HALF_OPEN state")
                return False
            return True
        return False

    def record_success(self):
        """Record a successful operation."""
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            logger.info("Circuit breaker restored to CLOSED state")

    def record_failure(self):
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


async def retry_with_exponential_backoff(
    func, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0, *args, **kwargs
):
    """Retry function with exponential backoff."""

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries:
                raise e

            delay = min(base_delay * (2**attempt), max_delay)
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
            await asyncio.sleep(delay)

    raise Exception("Max retries exceeded")


class LLMService:
    """Service wrapper for LLM providers with async GMN generation."""

    def __init__(self, provider_factory: Optional[LLMProviderFactory] = None):
        """Initialize LLM service with provider factory."""
        self.provider_factory = provider_factory or LLMProviderFactory()
        self._cached_providers: Dict[str, Any] = {}
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        self._fallback_gmn_templates = self._create_fallback_templates()
        self._metrics = {
            "requests_total": 0,
            "requests_successful": 0,
            "requests_failed": 0,
            "fallback_used": 0,
            "circuit_breaker_open": 0,
            "avg_response_time": 0.0,
            "last_success_time": None,
            "last_failure_time": None,
        }

    def _create_fallback_templates(self) -> Dict[str, Any]:
        """Create fallback GMN templates for when LLM service fails."""

        return {
            "default": {
                "name": "fallback_agent",
                "description": "A basic conversation agent created as fallback",
                "states": ["listening", "thinking", "responding"],
                "observations": ["message", "silence", "question"],
                "actions": ["respond", "listen", "acknowledge"],
                "parameters": {
                    "A": [[0.8, 0.15, 0.05], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]],
                    "B": [[[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]]],
                    "C": [[1.0, 0.5, 0.8]],
                    "D": [[0.33, 0.33, 0.34]],
                },
            },
            "explorer": {
                "name": "explorer_agent",
                "description": "An exploration-focused agent",
                "states": ["exploring", "investigating", "reporting"],
                "observations": ["discovery", "obstacle", "goal"],
                "actions": ["explore", "investigate", "report", "move"],
                "parameters": {
                    "A": [[0.9, 0.08, 0.02], [0.1, 0.7, 0.2], [0.05, 0.25, 0.7]],
                    "B": [[[0.6, 0.3, 0.1], [0.3, 0.5, 0.2], [0.1, 0.2, 0.7]]],
                    "C": [[1.0, 0.3, 0.8]],
                    "D": [[0.5, 0.3, 0.2]],
                },
            },
        }

    async def get_provider_manager(self, user_id: str):
        """Get or create LLM provider manager for user."""

        if user_id in self._cached_providers:
            return self._cached_providers[user_id]

        try:
            logger.info(f"Creating LLM provider manager for user {user_id}")
            provider_manager = self.provider_factory.create_from_config(user_id=user_id)

            # Check if any providers are available
            healthy_providers = provider_manager.registry.get_healthy_providers()
            logger.info(
                f"Available providers: {[p.provider_type.value for p in healthy_providers]}"
            )

            if not healthy_providers:
                raise HTTPException(
                    status_code=503,
                    detail="No LLM providers available. Please configure API keys in settings.",
                )

            # Cache the provider for this user
            self._cached_providers[user_id] = provider_manager
            return provider_manager

        except Exception as e:
            logger.error(f"Failed to create LLM provider for user {user_id}: {e}")
            raise HTTPException(
                status_code=503,
                detail="Failed to initialize LLM provider. Please check your API key configuration.",
            )

    async def generate_gmn(
        self,
        prompt: str,
        user_id: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> Dict[str, Any]:
        """Generate GMN specification from natural language prompt with circuit breaker and fallback."""

        start_time = time.time()
        self._metrics["requests_total"] += 1

        # Check circuit breaker first
        if self.circuit_breaker.is_open():
            logger.warning("Circuit breaker is open, using fallback GMN template")
            self._metrics["circuit_breaker_open"] += 1
            self._metrics["fallback_used"] += 1
            return self._get_fallback_gmn(prompt)

        try:
            # Use retry logic with exponential backoff
            llm_response = await retry_with_exponential_backoff(
                self._generate_gmn_with_llm,
                max_retries=3,
                base_delay=1.0,
                max_delay=10.0,
                prompt=prompt,
                user_id=user_id,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Parse the LLM response to GMN
            try:
                from api.v1.services.gmn_parser_service import GMNParserService

                parser = GMNParserService()
                gmn_spec = parser.parse_gmn_from_llm_response(llm_response)
            except Exception as e:
                logger.error(f"Failed to parse LLM response to GMN: {e}")
                raise e

            # Record success metrics
            self.circuit_breaker.record_success()
            self._metrics["requests_successful"] += 1
            self._metrics["last_success_time"] = time.time()

            # Update average response time
            response_time = time.time() - start_time
            self._update_avg_response_time(response_time)

            logger.info(f"Successfully generated GMN with LLM in {response_time:.2f}s")
            return gmn_spec

        except Exception as e:
            # Record failure metrics
            self.circuit_breaker.record_failure()
            self._metrics["requests_failed"] += 1
            self._metrics["fallback_used"] += 1
            self._metrics["last_failure_time"] = time.time()

            logger.error(f"LLM generation failed after retries: {e}")

            # Graceful degradation - return fallback template
            logger.info("Falling back to template-based GMN generation")
            return self._get_fallback_gmn(prompt)

    def _update_avg_response_time(self, new_time: float):
        """Update rolling average response time."""

        current_avg = self._metrics["avg_response_time"]
        total_requests = self._metrics["requests_total"]

        if total_requests == 1:
            self._metrics["avg_response_time"] = new_time
        else:
            # Simple exponential moving average
            alpha = 0.1  # Weight for new observation
            self._metrics["avg_response_time"] = (alpha * new_time) + ((1 - alpha) * current_avg)

    async def _generate_gmn_with_llm(
        self, prompt: str, user_id: str, model: str, temperature: float, max_tokens: int
    ) -> Dict[str, Any]:
        """Internal method to generate GMN with LLM (used by retry logic)."""

        provider_manager = await self.get_provider_manager(user_id)

        # Construct GMN generation prompt
        system_prompt = """You are an expert in Active Inference and GMN (Generalized Notation) format.
Convert the user's goal description into a valid GMN specification.

GMN format structure:
{
  "name": "agent_name",
  "description": "what the agent does",
  "states": ["state1", "state2", ...],
  "observations": ["obs1", "obs2", ...],
  "actions": ["action1", "action2", ...],
  "parameters": {
    "A": [[probability_matrix]],  // P(o|s) observation model
    "B": [[[transition_tensor]]],  // P(s'|s,a) transition model
    "C": [[preferences]],  // Observation preferences
    "D": [[initial_beliefs]]  // Initial state distribution
  }
}

Ensure all probability distributions sum to 1.0. Respond with valid JSON only."""

        user_prompt = f"Create a GMN specification for an agent that: {prompt}"

        generation_request = GenerationRequest(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        logger.info(f"Generating GMN from prompt - model: {model}")
        response = provider_manager.generate_with_fallback(generation_request)
        return response

    def _get_fallback_gmn(self, prompt: str) -> Dict[str, Any]:
        """Get a fallback GMN template based on the prompt."""

        # Simple keyword matching to select appropriate template
        prompt_lower = prompt.lower()

        if any(word in prompt_lower for word in ["explore", "discover", "find", "search"]):
            template = self._fallback_gmn_templates["explorer"].copy()
            template["name"] = "explorer_agent"
            template["description"] = f"Explorer agent for: {prompt[:100]}"
        else:
            template = self._fallback_gmn_templates["default"].copy()
            template["name"] = "conversation_agent"
            template["description"] = f"Conversation agent for: {prompt[:100]}"

        # Add fallback indicator
        template["_fallback"] = True
        template["_fallback_reason"] = "LLM service unavailable"

        logger.info(f"Using fallback GMN template: {template['name']}")
        return template

    async def generate_conversation_response(
        self,
        system_prompt: str,
        context_messages: list,
        user_id: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.8,
        max_tokens: int = 150,
    ) -> str:
        """Generate a conversation response from an agent."""

        provider_manager = await self.get_provider_manager(user_id)

        # Build message list
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(context_messages)

        generation_request = GenerationRequest(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        try:
            response = provider_manager.generate_with_fallback(generation_request)

            # Extract content from response
            if hasattr(response, "content"):
                content = response.content
            elif hasattr(response, "text"):
                content = response.text
            else:
                content = str(response)

            return content.strip()

        except Exception as e:
            logger.error(f"Conversation response generation failed: {e}")
            raise HTTPException(status_code=503, detail=f"Failed to generate response: {str(e)}")

    def clear_cache(self):
        """Clear cached providers (useful for testing)."""
        self._cached_providers.clear()

    def get_metrics(self) -> Dict[str, Any]:
        """Get current service metrics."""

        metrics = self._metrics.copy()

        # Add computed metrics
        if metrics["requests_total"] > 0:
            metrics["success_rate"] = metrics["requests_successful"] / metrics["requests_total"]
            metrics["failure_rate"] = metrics["requests_failed"] / metrics["requests_total"]
            metrics["fallback_rate"] = metrics["fallback_used"] / metrics["requests_total"]
        else:
            metrics["success_rate"] = 0.0
            metrics["failure_rate"] = 0.0
            metrics["fallback_rate"] = 0.0

        # Add circuit breaker status
        metrics["circuit_breaker_state"] = self.circuit_breaker.state
        metrics["circuit_breaker_failure_count"] = self.circuit_breaker.failure_count

        return metrics


# Dependency injection factory function
def get_llm_service() -> LLMService:
    """Factory function for FastAPI dependency injection."""
    return LLMService()
