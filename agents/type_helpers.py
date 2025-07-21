"""Helper functions for safe type access in agent and coalition systems.

This module provides helper functions to safely access attributes from mixed
types of agent and coalition objects (database models vs in-memory objects).
"""

import uuid
from typing import TYPE_CHECKING, Any, Optional, Union

from agents.type_adapter import AgentTypeAdapter, CoalitionTypeAdapter

# Import PyMDPErrorHandler from the correct module
if TYPE_CHECKING:
    pass
else:
    try:
        pass
    except ImportError:
        # Fallback to hard failure handler if pymdp_error_handling is not available
        pass


def safe_get_agent_id(agent: Any) -> Optional[str]:
    """Safely get agent ID from any agent-like object.

    Args:
        agent: Agent object (database model, in-memory, or dict)

    Returns:
        Agent ID as string or None if not found
    """
    try:
        return AgentTypeAdapter.get_id(agent)
    except AttributeError:
        return None


def safe_get_coalition_id(coalition: Any) -> Optional[str]:
    """Safely get coalition ID from any coalition-like object.

    Args:
        coalition: Coalition object (database model, in-memory, or dict)

    Returns:
        Coalition ID as string or None if not found
    """
    try:
        return CoalitionTypeAdapter.get_id(coalition)
    except AttributeError:
        return None


def ensure_string_id(id_value: Union[str, uuid.UUID, Any]) -> str:
    """Ensure ID value is converted to string.

    Args:
        id_value: ID value that could be string, UUID, or other type

    Returns:
        ID as string
    """
    if isinstance(id_value, uuid.UUID):
        return str(id_value)
    return str(id_value)


def match_agent_id(agent: Any, target_id: Union[str, uuid.UUID]) -> bool:
    """Check if agent matches target ID, handling type conversions.

    Args:
        agent: Agent object to check
        target_id: Target ID to match against

    Returns:
        True if agent ID matches target ID
    """
    agent_id = safe_get_agent_id(agent)
    if agent_id is None:
        return False

    target_str = ensure_string_id(target_id)
    return agent_id == target_str


def match_coalition_id(coalition: Any, target_id: Union[str, uuid.UUID]) -> bool:
    """Check if coalition matches target ID, handling type conversions.

    Args:
        coalition: Coalition object to check
        target_id: Target ID to match against

    Returns:
        True if coalition ID matches target ID
    """
    coalition_id = safe_get_coalition_id(coalition)
    if coalition_id is None:
        return False

    target_str = ensure_string_id(target_id)
    return coalition_id == target_str


def get_agent_attribute(
    agent: Any, attribute: str, default: Optional[Any] = None
) -> Any:
    """Safely get an attribute from an agent object.

    Args:
        agent: Agent object
        attribute: Attribute name to get
        default: Default value if attribute not found

    Returns:
        Attribute value or default
    """
    # Handle special cases with type adapter
    if attribute in ["id", "agent_id"]:
        try:
            return AgentTypeAdapter.get_id(agent)
        except AttributeError:
            return default

    if attribute == "name":
        try:
            return AgentTypeAdapter.get_name(agent)
        except AttributeError:
            return default

    if attribute == "status":
        return AgentTypeAdapter.get_status(agent)

    if attribute == "position":
        return AgentTypeAdapter.get_position(agent)

    # Generic attribute access
    if hasattr(agent, attribute):
        return getattr(agent, attribute)

    if isinstance(agent, dict) and attribute in agent:
        return agent[attribute]

    return default


def get_coalition_attribute(
    coalition: Any, attribute: str, default: Optional[Any] = None
) -> Any:
    """Safely get an attribute from a coalition object.

    Args:
        coalition: Coalition object
        attribute: Attribute name to get
        default: Default value if attribute not found

    Returns:
        Attribute value or default
    """
    # Handle special cases with type adapter
    if attribute in ["id", "coalition_id"]:
        try:
            return CoalitionTypeAdapter.get_id(coalition)
        except AttributeError:
            return default

    if attribute == "name":
        try:
            return CoalitionTypeAdapter.get_name(coalition)
        except AttributeError:
            return default

    if attribute == "status":
        return CoalitionTypeAdapter.get_status(coalition)

    if attribute in ["members", "agents"]:
        return CoalitionTypeAdapter.get_members(coalition)

    if attribute == "leader_id":
        return CoalitionTypeAdapter.get_leader_id(coalition)

    # Generic attribute access
    if hasattr(coalition, attribute):
        return getattr(coalition, attribute)

    if isinstance(coalition, dict) and attribute in coalition:
        return coalition[attribute]

    return default
