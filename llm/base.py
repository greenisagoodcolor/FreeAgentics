"""Base interfaces for LLM providers.

This module defines the abstract interfaces that all LLM providers must implement,
ensuring a consistent API for prompt processing and GMN generation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class LLMRole(Enum):
    """Message roles in conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class LLMMessage:
    """Single message in a conversation."""

    role: LLMRole
    content: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    model: str
    usage: Optional[Dict[str, int]] = None  # tokens, etc.
    metadata: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None


class LLMError(Exception):
    """Base exception for LLM-related errors."""

    pass


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    All LLM providers must implement this interface to ensure compatibility
    with the FreeAgentics prompt processing pipeline.
    """

    @abstractmethod
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            messages: Conversation history as list of messages
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stop_sequences: List of sequences that stop generation
            **kwargs: Provider-specific parameters

        Returns:
            LLMResponse containing the generated text and metadata

        Raises:
            LLMError: If generation fails
        """
        pass

    @abstractmethod
    async def validate_model(self, model_name: str) -> bool:
        """Check if a model is available and valid.

        Args:
            model_name: Name of the model to validate

        Returns:
            True if model is available, False otherwise
        """
        pass

    @abstractmethod
    def get_token_limit(self, model_name: str) -> int:
        """Get the token limit for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Maximum token limit for the model
        """
        pass

    async def generate_gmn(
        self,
        prompt: str,
        agent_type: str = "general",
        constraints: Optional[Dict[str, Any]] = None,
        examples: Optional[List[str]] = None,
    ) -> str:
        """Generate GMN specification from natural language prompt.

        This is a specialized method for GMN generation that uses the base
        generate method with appropriate system prompts.

        Args:
            prompt: Natural language description of desired agent behavior
            agent_type: Type of agent (explorer, trader, coordinator, etc.)
            constraints: Optional constraints on the GMN structure
            examples: Optional example GMN specifications for few-shot learning

        Returns:
            Generated GMN specification as a string

        Raises:
            LLMError: If GMN generation fails
        """
        system_prompt = self._build_gmn_system_prompt(
            agent_type, constraints, examples
        )

        messages = [
            LLMMessage(role=LLMRole.SYSTEM, content=system_prompt),
            LLMMessage(role=LLMRole.USER, content=prompt),
        ]

        response = await self.generate(
            messages=messages,
            temperature=0.3,  # Lower temperature for more consistent GMN structure
            stop_sequences=["```", "---"],  # Common delimiters
        )

        return self._extract_gmn_from_response(response.content)

    def _build_gmn_system_prompt(
        self,
        agent_type: str,
        constraints: Optional[Dict[str, Any]],
        examples: Optional[List[str]],
    ) -> str:
        """Build the system prompt for GMN generation."""
        base_prompt = """You are an expert in Generative Model Notation (GMN) for active inference agents.
Your task is to convert natural language descriptions into valid GMN specifications.

GMN Syntax Rules:
1. Nodes are defined as: node <type> <name> { ... }
2. Node types: state, observation, action, transition, emission, preference
3. Properties include: type (discrete/continuous), size, matrix, dependencies
4. Transitions link states and actions to next states
5. Emissions link states to observations
6. Preferences define the agent's goals

Example GMN structure:
```
node state s1 {
    type: discrete
    size: 4
}
node observation o1 {
    type: discrete
    size: 5
}
node action a1 {
    type: discrete
    size: 3
}
node transition T1 {
    from: [s1, a1]
    to: s1
    matrix: [[...]]
}
node emission E1 {
    from: s1
    to: o1
    matrix: [[...]]
}
node preference C1 {
    state: s1
    values: [0, 0, 1, 0]  # Prefer state 2
}
```
"""

        if agent_type != "general":
            base_prompt += f"\n\nAgent Type: {agent_type}"
            base_prompt += self._get_agent_type_hints(agent_type)

        if constraints:
            base_prompt += "\n\nConstraints:\n"
            for key, value in constraints.items():
                base_prompt += f"- {key}: {value}\n"

        if examples:
            base_prompt += "\n\nReference Examples:\n"
            for i, example in enumerate(examples[:3]):  # Limit to 3 examples
                base_prompt += f"\nExample {i+1}:\n{example}\n"

        base_prompt += "\n\nGenerate a valid GMN specification based on the user's description."

        return base_prompt

    def _get_agent_type_hints(self, agent_type: str) -> str:
        """Get specific hints for different agent types."""
        hints = {
            "explorer": """
Explorer agents typically need:
- State space representing positions or locations
- Observations of nearby environment
- Actions for movement (up, down, left, right)
- Curiosity-driven preferences or uncertainty reduction goals
""",
            "trader": """
Trader agents typically need:
- State space for resources, prices, market conditions
- Observations of market signals
- Actions for buy, sell, hold decisions
- Preferences for profit maximization or risk management
""",
            "coordinator": """
Coordinator agents typically need:
- State space for team states and task progress
- Observations of other agents' states
- Actions for task assignment and communication
- Preferences for team efficiency and goal completion
""",
        }
        return hints.get(agent_type, "")

    def _extract_gmn_from_response(self, response: str) -> str:
        """Extract GMN specification from LLM response."""
        # Look for code blocks
        if "```" in response:
            # Extract content between triple backticks
            parts = response.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1:  # Odd indices are code blocks
                    # Remove language identifier if present
                    lines = part.strip().split('\n')
                    if lines and lines[0].lower() in ['gmn', 'yaml', 'json']:
                        return '\n'.join(lines[1:])
                    return part.strip()

        # If no code blocks, assume the entire response is GMN
        return response.strip()

    async def validate_gmn(self, gmn_spec: str) -> tuple[bool, List[str]]:
        """Validate a GMN specification using the LLM.

        Args:
            gmn_spec: GMN specification to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        validation_prompt = f"""Validate the following GMN specification and identify any errors:

```
{gmn_spec}
```

Respond with a JSON object containing:
- "valid": true/false
- "errors": list of error messages (empty if valid)
- "warnings": list of warning messages (optional)

Focus on:
1. Syntax errors
2. Missing required properties
3. Invalid matrix dimensions
4. Undefined node references
5. Logical inconsistencies"""

        messages = [
            LLMMessage(
                role=LLMRole.SYSTEM,
                content="You are a GMN validation expert. Respond only with valid JSON.",
            ),
            LLMMessage(role=LLMRole.USER, content=validation_prompt),
        ]

        try:
            response = await self.generate(
                messages=messages,
                temperature=0.1,  # Very low temperature for consistent validation
                max_tokens=500,
            )

            # Parse JSON response
            import json

            result = json.loads(response.content)

            is_valid = result.get("valid", False)
            errors = result.get("errors", [])

            return is_valid, errors

        except Exception as e:
            # If parsing fails, assume invalid
            return False, [f"Validation failed: {str(e)}"]

    async def refine_gmn(self, gmn_spec: str, feedback: str) -> str:
        """Refine a GMN specification based on feedback.

        Args:
            gmn_spec: Current GMN specification
            feedback: Feedback or errors to address

        Returns:
            Refined GMN specification
        """
        refinement_prompt = f"""Refine the following GMN specification based on the feedback:

Current GMN:
```
{gmn_spec}
```

Feedback:
{feedback}

Generate an improved GMN specification that addresses the feedback while maintaining the original intent."""

        messages = [
            LLMMessage(
                role=LLMRole.SYSTEM,
                content=self._build_gmn_system_prompt("general", None, None),
            ),
            LLMMessage(role=LLMRole.USER, content=refinement_prompt),
        ]

        response = await self.generate(
            messages=messages, temperature=0.3, stop_sequences=["```", "---"]
        )

        return self._extract_gmn_from_response(response.content)
