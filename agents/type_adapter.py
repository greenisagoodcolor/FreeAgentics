"""Type adapter for consistent agent and coalition attribute access.

This module provides adapters to handle type mismatches between database models
and in-memory objects for agents and coalitions.
"""

import uuid
from typing import Any, Dict, Optional, Union


class AgentTypeAdapter:
    """Adapter for consistent agent attribute access across database and in-memory objects."""

    @staticmethod
    def get_id(agent: Any) -> str:
        """Get agent ID as string regardless of source.

        Args:
            agent: Agent object (database model or in-memory)

        Returns:
            Agent ID as string

        Raises:
            AttributeError: If agent has no valid ID attribute
        """
        # Database model has 'id' attribute
        if hasattr(agent, "id"):
            agent_id = agent.id
            # Convert UUID to string if needed
            if isinstance(agent_id, uuid.UUID):
                return str(agent_id)
            return str(agent_id)

        # In-memory agent has 'agent_id' attribute
        if hasattr(agent, "agent_id"):
            return str(agent.agent_id)

        # Fallback - check if it's a dict
        if isinstance(agent, dict):
            if "id" in agent:
                return str(agent["id"])
            if "agent_id" in agent:
                return str(agent["agent_id"])

        raise AttributeError(f"Agent object {type(agent)} has no 'id' or 'agent_id' attribute")

    @staticmethod
    def get_name(agent: Any) -> str:
        """Get agent name regardless of source.

        Args:
            agent: Agent object (database model or in-memory)

        Returns:
            Agent name as string

        Raises:
            AttributeError: If agent has no name attribute
        """
        if hasattr(agent, "name"):
            return str(agent.name)

        if isinstance(agent, dict) and "name" in agent:
            return str(agent["name"])

        raise AttributeError(f"Agent object {type(agent)} has no 'name' attribute")

    @staticmethod
    def get_status(agent: Any) -> str:
        """Get agent status as string regardless of source.

        Args:
            agent: Agent object (database model or in-memory)

        Returns:
            Agent status as string
        """
        # Database model with enum
        if hasattr(agent, "status"):
            status = agent.status
            if hasattr(status, "value"):  # Enum
                return str(status.value)
            return str(status)

        # In-memory agent
        if hasattr(agent, "is_active"):
            return "active" if agent.is_active else "inactive"

        # Dict representation
        if isinstance(agent, dict):
            if "status" in agent:
                status = agent["status"]
                return str(status.value) if hasattr(status, "value") else str(status)
            if "is_active" in agent:
                return "active" if agent["is_active"] else "inactive"

        return "unknown"

    @staticmethod
    def get_position(agent: Any) -> Optional[Union[tuple, list, dict]]:
        """Get agent position regardless of source.

        Args:
            agent: Agent object (database model or in-memory)

        Returns:
            Agent position as tuple, list, or dict (or None)
        """
        if hasattr(agent, "position"):
            return agent.position

        if isinstance(agent, dict) and "position" in agent:
            return agent["position"]

        return None

    @staticmethod
    def to_dict(agent: Any) -> Dict[str, Any]:
        """Convert agent to dictionary with consistent keys.

        Args:
            agent: Agent object (database model or in-memory)

        Returns:
            Dictionary representation with consistent keys
        """
        result = {}

        # ID - always use 'id' as key for consistency
        try:
            result["id"] = AgentTypeAdapter.get_id(agent)
        except AttributeError:
            result["id"] = "unknown"

        # Name
        try:
            result["name"] = AgentTypeAdapter.get_name(agent)
        except AttributeError:
            result["name"] = "Unknown"

        # Status
        result["status"] = AgentTypeAdapter.get_status(agent)

        # Position
        position = AgentTypeAdapter.get_position(agent)
        if position is not None:
            result["position"] = str(position)

        # Additional attributes from database models
        if hasattr(agent, "template"):
            result["template"] = agent.template
        if hasattr(agent, "created_at"):
            result["created_at"] = (
                agent.created_at.isoformat()
                if hasattr(agent.created_at, "isoformat")
                else str(agent.created_at)
            )
        if hasattr(agent, "metrics"):
            result["metrics"] = agent.metrics

        # Additional attributes from in-memory agents
        if hasattr(agent, "total_steps"):
            result["total_steps"] = agent.total_steps
        if hasattr(agent, "beliefs"):
            result["has_beliefs"] = str(bool(agent.beliefs))

        return result


class CoalitionTypeAdapter:
    """Adapter for consistent coalition attribute access across database and in-memory objects."""

    @staticmethod
    def get_id(coalition: Any) -> str:
        """Get coalition ID as string regardless of source.

        Args:
            coalition: Coalition object (database model or in-memory)

        Returns:
            Coalition ID as string

        Raises:
            AttributeError: If coalition has no valid ID attribute
        """
        # Database model has 'id' attribute
        if hasattr(coalition, "id"):
            coalition_id = coalition.id
            # Convert UUID to string if needed
            if isinstance(coalition_id, uuid.UUID):
                return str(coalition_id)
            return str(coalition_id)

        # In-memory coalition has 'coalition_id' attribute
        if hasattr(coalition, "coalition_id"):
            return str(coalition.coalition_id)

        # Fallback - check if it's a dict
        if isinstance(coalition, dict):
            if "id" in coalition:
                return str(coalition["id"])
            if "coalition_id" in coalition:
                return str(coalition["coalition_id"])

        raise AttributeError(
            f"Coalition object {type(coalition)} has no 'id' or 'coalition_id' attribute"
        )

    @staticmethod
    def get_name(coalition: Any) -> str:
        """Get coalition name regardless of source.

        Args:
            coalition: Coalition object (database model or in-memory)

        Returns:
            Coalition name as string

        Raises:
            AttributeError: If coalition has no name attribute
        """
        if hasattr(coalition, "name"):
            return str(coalition.name)

        if isinstance(coalition, dict) and "name" in coalition:
            return str(coalition["name"])

        raise AttributeError(f"Coalition object {type(coalition)} has no 'name' attribute")

    @staticmethod
    def get_members(coalition: Any) -> Union[list, dict]:
        """Get coalition members regardless of source.

        Args:
            coalition: Coalition object (database model or in-memory)

        Returns:
            Coalition members as list or dict
        """
        # In-memory coalition has members dict
        if hasattr(coalition, "members"):
            return coalition.members

        # Database model has agents relationship
        if hasattr(coalition, "agents"):
            # Convert to dict format similar to in-memory
            members = {}
            for agent in coalition.agents:
                agent_id = AgentTypeAdapter.get_id(agent)
                members[agent_id] = {
                    "agent_id": agent_id,
                    "name": (
                        AgentTypeAdapter.get_name(agent) if hasattr(agent, "name") else "Unknown"
                    ),
                }
            return members

        # Dict representation
        if isinstance(coalition, dict):
            if "members" in coalition:
                return coalition["members"]
            if "agents" in coalition:
                return coalition["agents"]

        return {}

    @staticmethod
    def get_leader_id(coalition: Any) -> Optional[str]:
        """Get coalition leader ID regardless of source.

        Args:
            coalition: Coalition object (database model or in-memory)

        Returns:
            Leader ID as string or None
        """
        if hasattr(coalition, "leader_id"):
            return str(coalition.leader_id) if coalition.leader_id else None

        if isinstance(coalition, dict) and "leader_id" in coalition:
            leader_id = coalition["leader_id"]
            return str(leader_id) if leader_id else None

        # For database models, might need to check agents with leader role
        if hasattr(coalition, "agents"):
            # This would require checking the association table for roles
            # For now, return None
            pass

        return None

    @staticmethod
    def get_status(coalition: Any) -> str:
        """Get coalition status as string regardless of source.

        Args:
            coalition: Coalition object (database model or in-memory)

        Returns:
            Coalition status as string
        """
        if hasattr(coalition, "status"):
            status = coalition.status
            if hasattr(status, "value"):  # Enum
                return str(status.value)
            return str(status)

        if isinstance(coalition, dict) and "status" in coalition:
            status = coalition["status"]
            return str(status.value) if hasattr(status, "value") else str(status)

        return "unknown"

    @staticmethod
    def to_dict(coalition: Any) -> Dict[str, Any]:
        """Convert coalition to dictionary with consistent keys.

        Args:
            coalition: Coalition object (database model or in-memory)

        Returns:
            Dictionary representation with consistent keys
        """
        result = {}

        # ID - always use 'id' as key for consistency
        try:
            result["id"] = CoalitionTypeAdapter.get_id(coalition)
        except AttributeError:
            result["id"] = "unknown"

        # Name
        try:
            result["name"] = CoalitionTypeAdapter.get_name(coalition)
        except AttributeError:
            result["name"] = "Unknown"

        # Status
        result["status"] = CoalitionTypeAdapter.get_status(coalition)

        # Members
        members = CoalitionTypeAdapter.get_members(coalition)
        if isinstance(members, dict):
            result["member_count"] = str(len(members))
            result["member_ids"] = str(list(members.keys()))
        else:
            result["member_count"] = str(len(members) if members else 0)
            result["member_ids"] = str(
                [AgentTypeAdapter.get_id(m) for m in members] if members else []
            )

        # Leader
        leader_id = CoalitionTypeAdapter.get_leader_id(coalition)
        if leader_id:
            result["leader_id"] = leader_id

        # Additional attributes
        if hasattr(coalition, "objective"):
            result["objective"] = str(coalition.objective)
        if hasattr(coalition, "created_at"):
            result["created_at"] = (
                coalition.created_at.isoformat()
                if hasattr(coalition.created_at, "isoformat")
                else str(coalition.created_at)
            )
        if hasattr(coalition, "performance_score"):
            result["performance_score"] = coalition.performance_score

        return result
