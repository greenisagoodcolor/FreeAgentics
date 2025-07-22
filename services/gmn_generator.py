"""GMN Generator service using LLM providers.

This service handles the conversion of natural language prompts into
Generative Model Notation (GMN) specifications for active inference agents.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from llm.base import LLMError, LLMProvider
from llm.providers.mock import MockLLMProvider

logger = logging.getLogger(__name__)


class GMNGenerator:
    """Service for generating GMN specifications from natural language."""

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        """Initialize GMN generator with an LLM provider.

        Args:
            llm_provider: LLM provider instance. Defaults to MockLLMProvider.
        """
        self.llm_provider = llm_provider or MockLLMProvider()
        logger.info(f"Initialized GMN generator with {type(self.llm_provider).__name__}")

    async def prompt_to_gmn(
        self,
        prompt: str,
        agent_type: str = "general",
        constraints: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Convert natural language prompt to GMN specification.

        Args:
            prompt: Natural language description of desired agent behavior
            agent_type: Type of agent (explorer, trader, coordinator, general)
            constraints: Optional constraints on GMN structure

        Returns:
            Generated GMN specification as a string

        Raises:
            LLMError: If GMN generation fails
            ValueError: If prompt is invalid
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        logger.info(f"Generating GMN for agent type '{agent_type}' from prompt: {prompt[:100]}...")

        try:
            # Use the provider's GMN generation method
            gmn_spec = await self.llm_provider.generate_gmn(
                prompt=prompt, agent_type=agent_type, constraints=constraints
            )

            # Basic validation of generated GMN
            if not gmn_spec or not gmn_spec.strip():
                raise ValueError("Generated empty GMN specification")

            if "node" not in gmn_spec:
                raise ValueError("Generated GMN contains no node definitions")

            logger.info("Successfully generated GMN specification")
            return gmn_spec

        except LLMError as e:
            logger.error(f"LLM error during GMN generation: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during GMN generation: {str(e)}")
            raise LLMError(f"GMN generation failed: {str(e)}")

    async def validate_gmn(self, gmn_spec: str) -> Tuple[bool, List[str]]:
        """Validate a GMN specification.

        Args:
            gmn_spec: GMN specification to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        if not gmn_spec or not gmn_spec.strip():
            return False, ["GMN specification is empty"]

        logger.info("Validating GMN specification...")

        try:
            # Use LLM provider for validation
            is_valid, errors = await self.llm_provider.validate_gmn(gmn_spec)

            # Additional structural validation
            structural_errors = self._validate_gmn_structure(gmn_spec)
            if structural_errors:
                is_valid = False
                errors.extend(structural_errors)

            if is_valid:
                logger.info("GMN validation successful")
            else:
                logger.warning(f"GMN validation failed with {len(errors)} errors")

            return is_valid, errors

        except Exception as e:
            logger.error(f"Error during GMN validation: {str(e)}")
            return False, [f"Validation error: {str(e)}"]

    async def refine_gmn(self, gmn_spec: str, feedback: str) -> str:
        """Refine a GMN specification based on feedback.

        Args:
            gmn_spec: Current GMN specification
            feedback: Feedback or errors to address

        Returns:
            Refined GMN specification

        Raises:
            LLMError: If refinement fails
        """
        if not gmn_spec or not gmn_spec.strip():
            raise ValueError("GMN specification cannot be empty")

        if not feedback or not feedback.strip():
            raise ValueError("Feedback cannot be empty")

        logger.info(f"Refining GMN based on feedback: {feedback[:100]}...")

        try:
            refined_gmn = await self.llm_provider.refine_gmn(gmn_spec, feedback)

            # Validate the refined GMN
            is_valid, errors = await self.validate_gmn(refined_gmn)
            if not is_valid:
                logger.warning(f"Refined GMN still has issues: {errors}")

            return refined_gmn

        except LLMError as e:
            logger.error(f"LLM error during GMN refinement: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during GMN refinement: {str(e)}")
            raise LLMError(f"GMN refinement failed: {str(e)}")

    def _validate_gmn_structure(self, gmn_spec: str) -> List[str]:
        """Perform basic structural validation of GMN.

        Args:
            gmn_spec: GMN specification to validate

        Returns:
            List of structural errors (empty if valid)
        """
        errors = []
        lines = gmn_spec.strip().split("\n")

        # Track node definitions
        defined_nodes = set()
        referenced_nodes = set()
        node_types = {}

        # Parse GMN line by line
        current_node = None
        brace_count = 0

        for i, line in enumerate(lines):
            line = line.strip()

            # Track braces
            brace_count += line.count("{") - line.count("}")

            # Check for node definition
            if line.startswith("node"):
                parts = line.split()
                if len(parts) >= 3:
                    node_type = parts[1]
                    node_name = parts[2].rstrip("{")
                    defined_nodes.add(node_name)
                    node_types[node_name] = node_type
                    current_node = node_name
                else:
                    errors.append(f"Line {i + 1}: Invalid node definition")

            # Check for node references in properties
            if current_node and ":" in line:
                if "from:" in line or "to:" in line or "state:" in line:
                    # Extract referenced nodes
                    value_part = line.split(":", 1)[1].strip()
                    # Handle both single references and lists
                    if value_part.startswith("["):
                        # List of nodes
                        node_refs = value_part.strip("[]").split(",")
                        for ref in node_refs:
                            ref = ref.strip()
                            if ref and not ref.isdigit():
                                referenced_nodes.add(ref)
                    else:
                        # Single node reference
                        ref = value_part.strip()
                        if ref and not ref.isdigit() and ref != "true" and ref != "false":
                            referenced_nodes.add(ref)

        # Check for unbalanced braces
        if brace_count != 0:
            errors.append(f"Unbalanced braces: {brace_count} unclosed")

        # Check for undefined references
        undefined_refs = referenced_nodes - defined_nodes
        if undefined_refs:
            errors.append(f"Undefined node references: {', '.join(sorted(undefined_refs))}")

        # Check for required node types
        required_types = {"state", "action"}
        found_types = set(node_types.values())
        missing_types = required_types - found_types
        if missing_types:
            errors.append(f"Missing required node types: {', '.join(sorted(missing_types))}")

        # Check for transitions or emissions
        if "transition" not in found_types and "emission" not in found_types:
            errors.append("No transition or emission nodes defined")

        return errors

    async def suggest_improvements(self, gmn_spec: str) -> List[str]:
        """Suggest improvements for a GMN specification.

        Args:
            gmn_spec: GMN specification to analyze

        Returns:
            List of improvement suggestions
        """
        suggestions = []

        # Analyze GMN structure
        lines = gmn_spec.lower()

        # Check for preferences
        if "preference" not in lines:
            suggestions.append("Add preference nodes to define agent goals")

        # Check for observations
        if "observation" not in lines:
            suggestions.append("Add observation nodes for agent perception")

        # Check for emissions
        if "emission" not in lines and "observation" in lines:
            suggestions.append("Add emission nodes to link states to observations")

        # Check for descriptions
        if "description:" not in lines:
            suggestions.append("Add descriptions to nodes for better documentation")

        # Check for deterministic vs stochastic
        if "stochastic:" not in lines and "deterministic:" not in lines:
            suggestions.append("Specify whether transitions are deterministic or stochastic")

        # Check for initial state distribution
        if "initial:" not in lines:
            suggestions.append("Consider adding initial state distribution")

        return suggestions
