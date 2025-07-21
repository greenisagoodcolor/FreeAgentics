"""Tests for agents.type_helpers module."""

import uuid
from unittest.mock import MagicMock, patch

import pytest


class TestAgentsTypeHelpers:
    """Test the agents type helpers module."""

    def test_import_error_handling(self):
        """Test that import error handling works for PyMDPErrorHandler."""
        # Just test that the module can be imported
        try:
            from agents.type_helpers import (
                ensure_string_id,
                get_agent_attribute,
                get_coalition_attribute,
                match_agent_id,
                match_coalition_id,
                safe_get_agent_id,
                safe_get_coalition_id,
            )

            # Test that functions exist
            assert safe_get_agent_id is not None
            assert safe_get_coalition_id is not None
            assert ensure_string_id is not None
            assert match_agent_id is not None
            assert match_coalition_id is not None
            assert get_agent_attribute is not None
            assert get_coalition_attribute is not None
        except ImportError as e:
            assert False, "Test bypass removed - must fix underlying issue"

    @patch("agents.type_helpers.AgentTypeAdapter")
    def test_safe_get_agent_id_success(self, mock_adapter):
        """Test safe_get_agent_id with successful retrieval."""
        from agents.type_helpers import safe_get_agent_id

        mock_adapter.get_id.return_value = "test_agent_id"
        mock_agent = MagicMock()

        result = safe_get_agent_id(mock_agent)
        assert result == "test_agent_id"
        mock_adapter.get_id.assert_called_once_with(mock_agent)

    @patch("agents.type_helpers.AgentTypeAdapter")
    def test_safe_get_agent_id_attribute_error(self, mock_adapter):
        """Test safe_get_agent_id with AttributeError."""
        from agents.type_helpers import safe_get_agent_id

        mock_adapter.get_id.side_effect = AttributeError("No attribute")
        mock_agent = MagicMock()

        result = safe_get_agent_id(mock_agent)
        assert result is None

    @patch("agents.type_helpers.CoalitionTypeAdapter")
    def test_safe_get_coalition_id_success(self, mock_adapter):
        """Test safe_get_coalition_id with successful retrieval."""
        from agents.type_helpers import safe_get_coalition_id

        mock_adapter.get_id.return_value = "test_coalition_id"
        mock_coalition = MagicMock()

        result = safe_get_coalition_id(mock_coalition)
        assert result == "test_coalition_id"
        mock_adapter.get_id.assert_called_once_with(mock_coalition)

    @patch("agents.type_helpers.CoalitionTypeAdapter")
    def test_safe_get_coalition_id_attribute_error(self, mock_adapter):
        """Test safe_get_coalition_id with AttributeError."""
        from agents.type_helpers import safe_get_coalition_id

        mock_adapter.get_id.side_effect = AttributeError("No attribute")
        mock_coalition = MagicMock()

        result = safe_get_coalition_id(mock_coalition)
        assert result is None

    def test_ensure_string_id_with_uuid(self):
        """Test ensure_string_id with UUID input."""
        from agents.type_helpers import ensure_string_id

        test_uuid = uuid.uuid4()
        result = ensure_string_id(test_uuid)
        assert result == str(test_uuid)
        assert isinstance(result, str)

    def test_ensure_string_id_with_string(self):
        """Test ensure_string_id with string input."""
        from agents.type_helpers import ensure_string_id

        test_string = "test_id_string"
        result = ensure_string_id(test_string)
        assert result == test_string
        assert isinstance(result, str)

    def test_ensure_string_id_with_other_type(self):
        """Test ensure_string_id with other type input."""
        from agents.type_helpers import ensure_string_id

        test_int = 12345
        result = ensure_string_id(test_int)
        assert result == "12345"
        assert isinstance(result, str)

    @patch("agents.type_helpers.safe_get_agent_id")
    def test_match_agent_id_success(self, mock_safe_get):
        """Test match_agent_id with successful match."""
        from agents.type_helpers import match_agent_id

        mock_safe_get.return_value = "test_id"
        mock_agent = MagicMock()

        result = match_agent_id(mock_agent, "test_id")
        assert result is True

    @patch("agents.type_helpers.safe_get_agent_id")
    def test_match_agent_id_no_match(self, mock_safe_get):
        """Test match_agent_id with no match."""
        from agents.type_helpers import match_agent_id

        mock_safe_get.return_value = "test_id"
        mock_agent = MagicMock()

        result = match_agent_id(mock_agent, "different_id")
        assert result is False

    @patch("agents.type_helpers.safe_get_agent_id")
    def test_match_agent_id_none_agent_id(self, mock_safe_get):
        """Test match_agent_id with None agent ID."""
        from agents.type_helpers import match_agent_id

        mock_safe_get.return_value = None
        mock_agent = MagicMock()

        result = match_agent_id(mock_agent, "test_id")
        assert result is False

    @patch("agents.type_helpers.safe_get_agent_id")
    def test_match_agent_id_with_uuid(self, mock_safe_get):
        """Test match_agent_id with UUID target."""
        from agents.type_helpers import match_agent_id

        test_uuid = uuid.uuid4()
        mock_safe_get.return_value = str(test_uuid)
        mock_agent = MagicMock()

        result = match_agent_id(mock_agent, test_uuid)
        assert result is True

    @patch("agents.type_helpers.safe_get_coalition_id")
    def test_match_coalition_id_success(self, mock_safe_get):
        """Test match_coalition_id with successful match."""
        from agents.type_helpers import match_coalition_id

        mock_safe_get.return_value = "test_id"
        mock_coalition = MagicMock()

        result = match_coalition_id(mock_coalition, "test_id")
        assert result is True

    @patch("agents.type_helpers.safe_get_coalition_id")
    def test_match_coalition_id_no_match(self, mock_safe_get):
        """Test match_coalition_id with no match."""
        from agents.type_helpers import match_coalition_id

        mock_safe_get.return_value = "test_id"
        mock_coalition = MagicMock()

        result = match_coalition_id(mock_coalition, "different_id")
        assert result is False

    @patch("agents.type_helpers.safe_get_coalition_id")
    def test_match_coalition_id_none_coalition_id(self, mock_safe_get):
        """Test match_coalition_id with None coalition ID."""
        from agents.type_helpers import match_coalition_id

        mock_safe_get.return_value = None
        mock_coalition = MagicMock()

        result = match_coalition_id(mock_coalition, "test_id")
        assert result is False

    @patch("agents.type_helpers.AgentTypeAdapter")
    def test_get_agent_attribute_id(self, mock_adapter):
        """Test get_agent_attribute with id attribute."""
        from agents.type_helpers import get_agent_attribute

        mock_adapter.get_id.return_value = "test_id"
        mock_agent = MagicMock()

        result = get_agent_attribute(mock_agent, "id")
        assert result == "test_id"

    @patch("agents.type_helpers.AgentTypeAdapter")
    def test_get_agent_attribute_agent_id(self, mock_adapter):
        """Test get_agent_attribute with agent_id attribute."""
        from agents.type_helpers import get_agent_attribute

        mock_adapter.get_id.return_value = "test_id"
        mock_agent = MagicMock()

        result = get_agent_attribute(mock_agent, "agent_id")
        assert result == "test_id"

    @patch("agents.type_helpers.AgentTypeAdapter")
    def test_get_agent_attribute_name(self, mock_adapter):
        """Test get_agent_attribute with name attribute."""
        from agents.type_helpers import get_agent_attribute

        mock_adapter.get_name.return_value = "test_name"
        mock_agent = MagicMock()

        result = get_agent_attribute(mock_agent, "name")
        assert result == "test_name"

    @patch("agents.type_helpers.AgentTypeAdapter")
    def test_get_agent_attribute_status(self, mock_adapter):
        """Test get_agent_attribute with status attribute."""
        from agents.type_helpers import get_agent_attribute

        mock_adapter.get_status.return_value = "active"
        mock_agent = MagicMock()

        result = get_agent_attribute(mock_agent, "status")
        assert result == "active"

    @patch("agents.type_helpers.AgentTypeAdapter")
    def test_get_agent_attribute_position(self, mock_adapter):
        """Test get_agent_attribute with position attribute."""
        from agents.type_helpers import get_agent_attribute

        mock_adapter.get_position.return_value = {"x": 10, "y": 20}
        mock_agent = MagicMock()

        result = get_agent_attribute(mock_agent, "position")
        assert result == {"x": 10, "y": 20}

    def test_get_agent_attribute_generic_hasattr(self):
        """Test get_agent_attribute with generic attribute via hasattr."""
        from agents.type_helpers import get_agent_attribute

        mock_agent = MagicMock()
        mock_agent.custom_attr = "custom_value"

        result = get_agent_attribute(mock_agent, "custom_attr")
        assert result == "custom_value"

    def test_get_agent_attribute_dict_access(self):
        """Test get_agent_attribute with dict access."""
        from agents.type_helpers import get_agent_attribute

        agent_dict = {"custom_attr": "custom_value"}

        result = get_agent_attribute(agent_dict, "custom_attr")
        assert result == "custom_value"

    def test_get_agent_attribute_default(self):
        """Test get_agent_attribute with default value."""
        from agents.type_helpers import get_agent_attribute

        mock_agent = MagicMock(spec=[])  # Empty spec, no attributes

        result = get_agent_attribute(mock_agent, "nonexistent", "default_value")
        assert result == "default_value"

    @patch("agents.type_helpers.AgentTypeAdapter")
    def test_get_agent_attribute_id_attribute_error(self, mock_adapter):
        """Test get_agent_attribute with id attribute raising AttributeError."""
        from agents.type_helpers import get_agent_attribute

        mock_adapter.get_id.side_effect = AttributeError("No id")
        mock_agent = MagicMock()

        result = get_agent_attribute(mock_agent, "id", "default")
        assert result == "default"

    @patch("agents.type_helpers.AgentTypeAdapter")
    def test_get_agent_attribute_name_attribute_error(self, mock_adapter):
        """Test get_agent_attribute with name attribute raising AttributeError."""
        from agents.type_helpers import get_agent_attribute

        mock_adapter.get_name.side_effect = AttributeError("No name")
        mock_agent = MagicMock()

        result = get_agent_attribute(mock_agent, "name", "default")
        assert result == "default"

    @patch("agents.type_helpers.CoalitionTypeAdapter")
    def test_get_coalition_attribute_id(self, mock_adapter):
        """Test get_coalition_attribute with id attribute."""
        from agents.type_helpers import get_coalition_attribute

        mock_adapter.get_id.return_value = "test_id"
        mock_coalition = MagicMock()

        result = get_coalition_attribute(mock_coalition, "id")
        assert result == "test_id"

    @patch("agents.type_helpers.CoalitionTypeAdapter")
    def test_get_coalition_attribute_coalition_id(self, mock_adapter):
        """Test get_coalition_attribute with coalition_id attribute."""
        from agents.type_helpers import get_coalition_attribute

        mock_adapter.get_id.return_value = "test_id"
        mock_coalition = MagicMock()

        result = get_coalition_attribute(mock_coalition, "coalition_id")
        assert result == "test_id"

    @patch("agents.type_helpers.CoalitionTypeAdapter")
    def test_get_coalition_attribute_name(self, mock_adapter):
        """Test get_coalition_attribute with name attribute."""
        from agents.type_helpers import get_coalition_attribute

        mock_adapter.get_name.return_value = "test_name"
        mock_coalition = MagicMock()

        result = get_coalition_attribute(mock_coalition, "name")
        assert result == "test_name"

    @patch("agents.type_helpers.CoalitionTypeAdapter")
    def test_get_coalition_attribute_status(self, mock_adapter):
        """Test get_coalition_attribute with status attribute."""
        from agents.type_helpers import get_coalition_attribute

        mock_adapter.get_status.return_value = "active"
        mock_coalition = MagicMock()

        result = get_coalition_attribute(mock_coalition, "status")
        assert result == "active"

    @patch("agents.type_helpers.CoalitionTypeAdapter")
    def test_get_coalition_attribute_members(self, mock_adapter):
        """Test get_coalition_attribute with members attribute."""
        from agents.type_helpers import get_coalition_attribute

        mock_adapter.get_members.return_value = ["agent1", "agent2"]
        mock_coalition = MagicMock()

        result = get_coalition_attribute(mock_coalition, "members")
        assert result == ["agent1", "agent2"]

    @patch("agents.type_helpers.CoalitionTypeAdapter")
    def test_get_coalition_attribute_agents(self, mock_adapter):
        """Test get_coalition_attribute with agents attribute."""
        from agents.type_helpers import get_coalition_attribute

        mock_adapter.get_members.return_value = ["agent1", "agent2"]
        mock_coalition = MagicMock()

        result = get_coalition_attribute(mock_coalition, "agents")
        assert result == ["agent1", "agent2"]

    @patch("agents.type_helpers.CoalitionTypeAdapter")
    def test_get_coalition_attribute_leader_id(self, mock_adapter):
        """Test get_coalition_attribute with leader_id attribute."""
        from agents.type_helpers import get_coalition_attribute

        mock_adapter.get_leader_id.return_value = "leader_id"
        mock_coalition = MagicMock()

        result = get_coalition_attribute(mock_coalition, "leader_id")
        assert result == "leader_id"

    def test_get_coalition_attribute_generic_hasattr(self):
        """Test get_coalition_attribute with generic attribute via hasattr."""
        from agents.type_helpers import get_coalition_attribute

        mock_coalition = MagicMock()
        mock_coalition.custom_attr = "custom_value"

        result = get_coalition_attribute(mock_coalition, "custom_attr")
        assert result == "custom_value"

    def test_get_coalition_attribute_dict_access(self):
        """Test get_coalition_attribute with dict access."""
        from agents.type_helpers import get_coalition_attribute

        coalition_dict = {"custom_attr": "custom_value"}

        result = get_coalition_attribute(coalition_dict, "custom_attr")
        assert result == "custom_value"

    def test_get_coalition_attribute_default(self):
        """Test get_coalition_attribute with default value."""
        from agents.type_helpers import get_coalition_attribute

        mock_coalition = MagicMock(spec=[])  # Empty spec, no attributes

        result = get_coalition_attribute(mock_coalition, "nonexistent", "default_value")
        assert result == "default_value"

    @patch("agents.type_helpers.CoalitionTypeAdapter")
    def test_get_coalition_attribute_id_attribute_error(self, mock_adapter):
        """Test get_coalition_attribute with id attribute raising AttributeError."""
        from agents.type_helpers import get_coalition_attribute

        mock_adapter.get_id.side_effect = AttributeError("No id")
        mock_coalition = MagicMock()

        result = get_coalition_attribute(mock_coalition, "id", "default")
        assert result == "default"

    @patch("agents.type_helpers.CoalitionTypeAdapter")
    def test_get_coalition_attribute_name_attribute_error(self, mock_adapter):
        """Test get_coalition_attribute with name attribute raising AttributeError."""
        from agents.type_helpers import get_coalition_attribute

        mock_adapter.get_name.side_effect = AttributeError("No name")
        mock_coalition = MagicMock()

        result = get_coalition_attribute(mock_coalition, "name", "default")
        assert result == "default"
