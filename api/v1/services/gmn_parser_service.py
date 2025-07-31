"""
GMN Parser Service for Agent Conversation API

Provides dependency injection service for parsing GMN structures from LLM output.
"""

import json
import logging
from typing import Any, Dict

from fastapi import HTTPException

logger = logging.getLogger(__name__)


class GMNParserService:
    """Service for parsing and validating GMN structures from LLM output."""

    def __init__(self):
        """Initialize GMN parser service."""
        pass

    def parse_gmn_from_llm_response(self, llm_response: Any) -> Dict[str, Any]:
        """Parse GMN specification from LLM response."""

        try:
            # Extract text content from response
            if hasattr(llm_response, "content"):
                response_text = llm_response.content
            elif hasattr(llm_response, "text"):
                response_text = llm_response.text
            else:
                response_text = str(llm_response)

            logger.info(f"Parsing GMN from LLM response: {response_text[:200]}...")

            # Parse JSON
            gmn_spec = json.loads(response_text)

            # Validate basic structure
            self._validate_gmn_structure(gmn_spec)

            logger.info(f"Successfully parsed GMN: {gmn_spec.get('name', 'unnamed')}")
            return gmn_spec

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            raise HTTPException(
                status_code=422, detail="LLM generated invalid JSON for GMN specification"
            )
        except Exception as e:
            logger.error(f"GMN parsing failed: {e}")
            raise HTTPException(status_code=422, detail=f"Invalid GMN specification: {str(e)}")

    def _validate_gmn_structure(self, gmn_spec: Dict[str, Any]) -> None:
        """Validate that GMN specification has required structure."""

        required_fields = ["name", "states", "observations", "actions"]

        for field in required_fields:
            if field not in gmn_spec:
                raise ValueError(f"Missing required field: {field}")

        # Validate lists are not empty
        for field in ["states", "observations", "actions"]:
            if not isinstance(gmn_spec[field], list) or len(gmn_spec[field]) == 0:
                raise ValueError(f"Field '{field}' must be a non-empty list")

        # Validate parameters if present
        if "parameters" in gmn_spec:
            self._validate_gmn_parameters(gmn_spec["parameters"])

    def _validate_gmn_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate GMN parameters structure."""

        # Check for required parameter matrices
        expected_params = ["A", "B", "C", "D"]

        for param in expected_params:
            if param in parameters:
                if not isinstance(parameters[param], list):
                    raise ValueError(f"Parameter '{param}' must be a list/matrix")

        # Additional validation could be added here for matrix dimensions
        logger.info("GMN parameters validation passed")

    def create_simple_gmn_template(
        self, name: str, role: str, personality: str, system_prompt: str
    ) -> Dict[str, Any]:
        """Create a simplified GMN template for conversation agents."""

        return {
            "name": name,
            "description": f"{role} agent with {personality} personality",
            "role": role,
            "personality": personality,
            "system_prompt": system_prompt,
            "states": ["listening", "thinking", "responding"],
            "observations": ["message", "silence", "question", "agreement", "disagreement"],
            "actions": ["listen", "respond", "question", "agree", "disagree", "clarify"],
            "conversation_mode": True,
            "parameters": {
                "simplified": True,
                # Simplified parameters for conversation mode
                "response_style": personality,
                "engagement_level": 0.8,
            },
        }

    def enhance_gmn_for_conversation(self, base_gmn: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance existing GMN for conversation capabilities."""

        enhanced_gmn = base_gmn.copy()

        # Add conversation-specific states if not present
        conversation_states = ["listening", "thinking", "responding"]
        for state in conversation_states:
            if state not in enhanced_gmn.get("states", []):
                enhanced_gmn.setdefault("states", []).append(state)

        # Add conversation actions
        conversation_actions = ["respond", "question", "agree", "disagree"]
        for action in conversation_actions:
            if action not in enhanced_gmn.get("actions", []):
                enhanced_gmn.setdefault("actions", []).append(action)

        # Mark as conversation-enabled
        enhanced_gmn["conversation_mode"] = True

        logger.info(f"Enhanced GMN {enhanced_gmn.get('name')} for conversation")
        return enhanced_gmn


# Dependency injection factory function
def get_gmn_parser_service() -> GMNParserService:
    """Factory function for FastAPI dependency injection."""
    return GMNParserService()
