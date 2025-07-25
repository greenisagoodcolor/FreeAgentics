"""Mock LLM provider for testing and development.

This provider generates deterministic GMN specifications based on
patterns in the input prompt, allowing for testing without real LLM calls.
"""

import asyncio
import json
import random
import re
from typing import Any, Dict, List, Optional

from llm.base import LLMError, LLMMessage, LLMProvider, LLMResponse, LLMRole


class MockLLMProvider(LLMProvider):
    """Mock LLM provider that generates realistic GMN specifications."""

    def __init__(self, delay: float = 0.1, error_rate: float = 0.0):
        """Initialize mock provider.

        Args:
            delay: Simulated response delay in seconds
            error_rate: Probability of generating an error (0.0 to 1.0)
        """
        self.delay = delay
        self.error_rate = error_rate
        self.model_name = "mock-gpt-4"
        self.token_limit = 4096

        # Predefined GMN templates for different agent types
        self.gmn_templates = {
            "explorer": self._get_explorer_template(),
            "trader": self._get_trader_template(),
            "coordinator": self._get_coordinator_template(),
            "general": self._get_general_template(),
        }

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a mock response."""
        # Simulate processing delay
        await asyncio.sleep(self.delay)

        # Simulate errors
        if random.random() < self.error_rate:
            raise LLMError("Mock provider simulated error")

        # Extract the last user message
        user_message = None
        for msg in reversed(messages):
            if msg.role == LLMRole.USER:
                user_message = msg.content
                break

        if not user_message:
            raise LLMError("No user message found")

        # Check if this is a validation request
        if "validate" in user_message.lower() and ("json" in user_message.lower() or "gmn" in user_message.lower()):
            content = self._generate_validation_response(user_message)
        # Check if this is a refinement request
        elif "refine" in user_message.lower() and "feedback" in user_message.lower():
            content = self._generate_refinement_response(user_message)
        # Otherwise, assume it's a GMN generation request
        else:
            content = self._generate_gmn_response(user_message)

        # Simulate token usage
        usage = {
            "prompt_tokens": sum(len(msg.content.split()) for msg in messages),
            "completion_tokens": len(content.split()),
            "total_tokens": sum(len(msg.content.split()) for msg in messages)
            + len(content.split()),
        }

        return LLMResponse(
            content=content,
            model=self.model_name,
            usage=usage,
            metadata={"temperature": temperature},
            finish_reason="stop",
        )

    async def validate_model(self, model_name: str) -> bool:
        """Check if model is available."""
        return model_name == self.model_name

    def get_token_limit(self, model_name: str) -> int:
        """Get token limit for model."""
        return self.token_limit if model_name == self.model_name else 0

    async def generate_gmn(
        self,
        prompt: str,
        agent_type: str = "general",
        constraints: Optional[Dict[str, Any]] = None,
        examples: Optional[List[str]] = None,
    ) -> str:
        """Generate GMN specification using the specified agent type.
        
        Override the base class method to use the agent_type parameter directly
        instead of trying to detect it from the prompt.
        """
        # Simulate processing delay
        await asyncio.sleep(self.delay)

        # Simulate errors
        if random.random() < self.error_rate:
            raise LLMError("Mock provider simulated error")

        # Use the specified agent_type directly instead of detecting from prompt
        if agent_type in self.gmn_templates:
            template = self.gmn_templates[agent_type]
        else:
            template = self.gmn_templates["general"]

        # Apply customizations based on prompt keywords
        if "grid" in prompt.lower():
            template = self._customize_for_grid(template, prompt)
        elif "market" in prompt.lower() or "trade" in prompt.lower():
            template = self._customize_for_market(template, prompt)

        # Add preferences based on prompt
        if "cautious" in prompt.lower():
            template = self._add_cautious_preferences(template)
        elif "curious" in prompt.lower() or "explore" in prompt.lower():
            template = self._add_exploration_preferences(template)

        return template

    def _generate_gmn_response(self, prompt: str) -> str:
        """Generate a GMN specification based on the prompt."""
        # Detect agent type from prompt
        agent_type = self._detect_agent_type(prompt)

        # Get base template
        template = self.gmn_templates[agent_type]

        # Customize based on prompt keywords
        if "grid" in prompt.lower():
            template = self._customize_for_grid(template, prompt)
        elif "market" in prompt.lower() or "trade" in prompt.lower():
            template = self._customize_for_market(template, prompt)

        # Add preferences based on prompt
        if "cautious" in prompt.lower():
            template = self._add_cautious_preferences(template)
        elif "curious" in prompt.lower() or "explore" in prompt.lower():
            template = self._add_exploration_preferences(template)

        return template

    def _generate_validation_response(self, prompt: str) -> str:
        """Generate a validation response."""
        # Extract GMN from prompt
        gmn_match = re.search(r"```\n?(.*?)\n?```", prompt, re.DOTALL)

        if gmn_match:
            gmn_content = gmn_match.group(1)

            # Simple validation checks
            errors = []

            if "node" not in gmn_content:
                errors.append("No node definitions found")

            if "state" not in gmn_content:
                errors.append("No state nodes defined")

            if "transition" not in gmn_content:
                errors.append("No transition nodes defined")

            # Check for basic syntax
            if "{" in gmn_content and gmn_content.count("{") != gmn_content.count("}"):
                errors.append("Mismatched braces")

            is_valid = len(errors) == 0

            return json.dumps(
                {
                    "valid": is_valid,
                    "errors": errors,
                    "warnings": (
                        ["Consider adding preference nodes for goal-directed behavior"]
                        if is_valid
                        else []
                    ),
                }
            )

        return json.dumps(
            {
                "valid": False,
                "errors": ["Could not parse GMN specification"],
                "warnings": [],
            }
        )

    def _generate_refinement_response(self, prompt: str) -> str:
        """Generate a refined GMN based on feedback."""
        # Extract current GMN
        gmn_match = re.search(r"Current GMN:\n```\n?(.*?)\n?```", prompt, re.DOTALL)

        if gmn_match:
            current_gmn = gmn_match.group(1)

            # Apply common refinements based on feedback keywords
            if "dimension" in prompt.lower():
                # Fix dimension issues
                current_gmn = re.sub(r"size: \d+", "size: 4", current_gmn)

            if "preference" in prompt.lower() and "preference" not in current_gmn:
                # Add preference node
                current_gmn += """
node preference C1 {
    state: s1
    values: [0.0, 0.0, 1.0, 0.0]
}
"""

            if "observation" in prompt.lower() and "observation" not in current_gmn:
                # Add observation node
                current_gmn += """
node observation o1 {
    type: discrete
    size: 5
}
node emission E1 {
    from: s1
    to: o1
    matrix: [[0.9, 0.05, 0.05, 0.0, 0.0],
             [0.05, 0.9, 0.05, 0.0, 0.0],
             [0.05, 0.05, 0.9, 0.0, 0.0],
             [0.0, 0.0, 0.05, 0.95, 0.0]]
}
"""

            return current_gmn.strip()

        # Fallback to a complete template
        return self.gmn_templates["general"]

    def _detect_agent_type(self, prompt: str) -> str:
        """Detect agent type from prompt keywords."""
        prompt_lower = prompt.lower()

        if any(word in prompt_lower for word in ["explore", "discover", "search", "grid"]):
            return "explorer"
        elif any(word in prompt_lower for word in ["trade", "market", "buy", "sell"]):
            return "trader"
        elif any(word in prompt_lower for word in ["coordinate", "team", "manage"]):
            return "coordinator"
        else:
            return "general"

    def _get_explorer_template(self) -> str:
        """Get GMN template for explorer agents."""
        return """node state s1 {
    type: discrete
    size: 25
    description: "Grid position (5x5)"
}

node observation o1 {
    type: discrete
    size: 5
    description: "Local observations (empty, wall, goal, visited, unknown)"
}

node action a1 {
    type: discrete
    size: 5
    description: "Movement actions (up, down, left, right, stay)"
}

node transition T1 {
    from: [s1, a1]
    to: s1
    description: "State transitions based on actions"
    matrix: "deterministic_grid_transitions"
}

node emission E1 {
    from: s1
    to: o1
    description: "Observations based on current position"
    matrix: [[0.9, 0.05, 0.03, 0.01, 0.01],
             [0.05, 0.9, 0.03, 0.01, 0.01],
             [0.03, 0.03, 0.9, 0.02, 0.02],
             [0.02, 0.02, 0.02, 0.9, 0.04],
             [0.1, 0.1, 0.1, 0.1, 0.6]]
}

node preference C1 {
    state: s1
    description: "Preference for unexplored areas"
    values: "exploration_bonus"
}"""

    def _get_trader_template(self) -> str:
        """Get GMN template for trader agents."""
        return """node state s1 {
    type: discrete
    size: 10
    description: "Market state (price levels)"
}

node state s2 {
    type: discrete
    size: 5
    description: "Portfolio state (holdings)"
}

node observation o1 {
    type: discrete
    size: 8
    description: "Market signals (trends, volume, volatility)"
}

node action a1 {
    type: discrete
    size: 4
    description: "Trading actions (buy, sell, hold, analyze)"
}

node transition T1 {
    from: [s1, a1]
    to: s1
    description: "Market evolution"
    stochastic: true
}

node transition T2 {
    from: [s2, a1]
    to: s2
    description: "Portfolio changes"
    deterministic: true
}

node emission E1 {
    from: [s1, s2]
    to: o1
    description: "Market observations"
}

node preference C1 {
    state: s2
    description: "Profit maximization"
    values: [0.0, 0.2, 0.5, 0.8, 1.0]
}"""

    def _get_coordinator_template(self) -> str:
        """Get GMN template for coordinator agents."""
        return """node state s1 {
    type: discrete
    size: 8
    description: "Team configuration state"
}

node state s2 {
    type: discrete
    size: 6
    description: "Task progress state"
}

node observation o1 {
    type: discrete
    size: 10
    description: "Team status observations"
}

node action a1 {
    type: discrete
    size: 6
    description: "Coordination actions (assign, communicate, wait, reorganize)"
}

node transition T1 {
    from: [s1, s2, a1]
    to: [s1, s2]
    description: "Team and task state evolution"
}

node emission E1 {
    from: [s1, s2]
    to: o1
    description: "Team observations"
}

node preference C1 {
    state: s2
    description: "Task completion preference"
    values: [0.0, 0.1, 0.3, 0.6, 0.9, 1.0]
}

node preference C2 {
    state: s1
    description: "Team efficiency preference"
    values: "balanced_workload"
}"""

    def _get_general_template(self) -> str:
        """Get general GMN template."""
        return """node state s1 {
    type: discrete
    size: 4
}

node observation o1 {
    type: discrete
    size: 5
}

node action a1 {
    type: discrete
    size: 4
}

node transition T1 {
    from: [s1, a1]
    to: s1
    matrix: [[0.9, 0.1, 0.0, 0.0],
             [0.1, 0.8, 0.1, 0.0],
             [0.0, 0.1, 0.8, 0.1],
             [0.0, 0.0, 0.1, 0.9]]
}

node emission E1 {
    from: s1
    to: o1
    matrix: [[0.8, 0.1, 0.05, 0.03, 0.02],
             [0.1, 0.8, 0.05, 0.03, 0.02],
             [0.05, 0.05, 0.8, 0.05, 0.05],
             [0.02, 0.02, 0.06, 0.8, 0.1]]
}

node preference C1 {
    state: s1
    values: [0.0, 0.0, 0.8, 0.2]
}"""

    def _customize_for_grid(self, template: str, prompt: str) -> str:
        """Customize template for grid world."""
        # Extract grid size if mentioned
        size_match = re.search(r"(\d+)x(\d+)", prompt)
        if size_match:
            rows, cols = int(size_match.group(1)), int(size_match.group(2))
            grid_size = rows * cols
            template = re.sub(r"size: \d+", f"size: {grid_size}", template, count=1)

        return template

    def _customize_for_market(self, template: str, prompt: str) -> str:
        """Customize template for market/trading."""
        # Add volatility considerations if mentioned
        if "volatile" in prompt.lower():
            template = template.replace(
                'description: "Market signals"',
                'description: "Market signals with volatility indicators"',
            )

        return template

    def _add_cautious_preferences(self, template: str) -> str:
        """Add cautious behavior preferences."""
        if "preference" in template:
            # Modify existing preferences to be more conservative
            template = re.sub(
                r"values: \[.*?\]",
                "values: [0.9, 0.05, 0.03, 0.02]",
                template,
                count=1,
            )
        else:
            # Add new preference node
            template += """

node preference C_cautious {
    state: s1
    description: "Safety preference"
    values: [0.9, 0.05, 0.03, 0.02]
}"""

        return template

    def _add_exploration_preferences(self, template: str) -> str:
        """Add exploration preferences."""
        if "preference" not in template:
            template += """

node preference C_explore {
    state: s1
    description: "Exploration bonus"
    values: "information_gain"
}"""

        return template
