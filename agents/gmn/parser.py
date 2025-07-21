"""
GMN (Generalized Notation) Parser Implementation.

Following Clean Architecture principles:
- Domain layer (no external dependencies)
- Clear interfaces and boundaries
- SOLID principles
"""

import re
from typing import Dict, Any, Optional


class GMNParser:
    """Parser for GMN (Generalized Notation) specifications."""

    def __init__(self):
        """Initialize the GMN parser."""
        self._agent_pattern = re.compile(
            r'agent\s+(\w+)\s*\{([^}]*)\}',
            re.DOTALL | re.MULTILINE
        )

    def parse(self, gmn_spec: str) -> Dict[str, Any]:
        """
        Parse GMN specification into a structured dictionary.

        Args:
            gmn_spec: GMN specification string

        Returns:
            Parsed agent configuration dictionary
        """
        # Clean input
        gmn_spec = gmn_spec.strip()

        # Find agent definition
        match = self._agent_pattern.search(gmn_spec)
        if not match:
            raise ValueError("No agent definition found in GMN specification")

        agent_name = match.group(1)
        agent_body = match.group(2).strip()

        # Build result
        result = {
            'name': agent_name,
            'type': None
        }

        # Parse agent body
        if agent_body:
            # Parse type field
            type_match = re.search(r'type:\s*"([^"]+)"', agent_body)
            if type_match:
                result['type'] = type_match.group(1)

        return result
