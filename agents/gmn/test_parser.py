"""
Test suite for GMN (Generalized Notation) Parser.

Following strict TDD principles as mandated by Kent Beck.
Every test drives the implementation - RED-GREEN-REFACTOR cycle.
"""

import pytest
from typing import Dict, Any


class TestGMNParserTDD:
    """TDD test suite for GMN parser implementation."""

    def test_parse_empty_agent_spec(self):
        """Test 1: Simplest case - parse empty agent specification."""
        # RED: This will fail because GMNParser doesn't exist yet
        from agents.gmn.parser import GMNParser

        gmn_spec = """
        agent Empty {
        }
        """

        parser = GMNParser()
        result = parser.parse(gmn_spec)

        assert result is not None
        assert result['name'] == 'Empty'
        assert result['type'] is None  # No type specified

    def test_parse_agent_with_type(self):
        """Test 2: Parse agent with type specification."""
        from agents.gmn.parser import GMNParser

        gmn_spec = """
        agent Explorer {
            type: "active_inference"
        }
        """

        parser = GMNParser()
        result = parser.parse(gmn_spec)

        assert result is not None
        assert result['name'] == 'Explorer'
        assert result['type'] == 'active_inference'
