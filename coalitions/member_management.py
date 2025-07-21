"""Type-safe member management for coalitions."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

from coalitions.coalition import Coalition, CoalitionMember, CoalitionRole

logger = logging.getLogger(__name__)


class MemberTypeError(Exception):
    """Exception raised for member type errors."""


class CoalitionMemberManager:
    """Type-safe member management for coalitions."""

    @staticmethod
    def ensure_member_collection(
        members: Union[List[Any], Dict[str, Any], Set[Any]],
    ) -> Dict[str, CoalitionMember]:
        """Ensure members is a proper dictionary of CoalitionMember objects.

        Args:
            members: Collection of members in various formats

        Returns:
            Dictionary mapping agent_id to CoalitionMember

        Raises:
            MemberTypeError: If members cannot be converted to proper format
        """
        if isinstance(members, dict):
            # Already a dictionary, validate types
            result = {}
            for agent_id, member in members.items():
                if isinstance(member, CoalitionMember):
                    result[str(agent_id)] = member
                elif isinstance(member, dict):
                    # Convert dict to CoalitionMember
                    result[str(agent_id)] = CoalitionMemberManager._dict_to_member(
                        agent_id, member
                    )
                else:
                    raise MemberTypeError(
                        f"Invalid member type for {agent_id}: {type(member)}"
                    )
            return result

        elif isinstance(members, (list, set)):
            # Convert list/set to dictionary
            result = {}
            for item in members:
                if isinstance(item, CoalitionMember):
                    result[item.agent_id] = item
                elif isinstance(item, dict):
                    agent_id_raw = item.get("agent_id") or item.get("id")
                    if not agent_id_raw:
                        raise MemberTypeError("Member dict missing agent_id")
                    agent_id = str(agent_id_raw)
                    result[str(agent_id)] = CoalitionMemberManager._dict_to_member(
                        agent_id, item
                    )
                elif hasattr(item, "agent_id"):
                    # Agent-like object
                    agent_id = str(item.agent_id)
                    result[agent_id] = CoalitionMember(
                        agent_id=agent_id,
                        role=CoalitionRole.MEMBER,
                        capabilities=getattr(item, "capabilities", []),
                    )
                else:
                    raise MemberTypeError(
                        f"Cannot convert item to member: {type(item)}"
                    )
            return result

        else:
            raise MemberTypeError(f"Invalid members type: {type(members)}")

    @staticmethod
    def _dict_to_member(agent_id: str, member_dict: Dict[str, Any]) -> CoalitionMember:
        """Convert dictionary to CoalitionMember."""
        role_value = member_dict.get("role", "member")
        if isinstance(role_value, str):
            try:
                role = CoalitionRole(role_value)
            except ValueError:
                role = CoalitionRole.MEMBER
        else:
            role = (
                role_value
                if isinstance(role_value, CoalitionRole)
                else CoalitionRole.MEMBER
            )

        return CoalitionMember(
            agent_id=str(agent_id),
            role=role,
            capabilities=member_dict.get("capabilities", []),
            join_time=member_dict.get("joined_at", datetime.now()),
            last_activity=member_dict.get("last_activity", datetime.now()),
            contribution_score=member_dict.get("contribution_score", 0.0),
            trust_score=member_dict.get("trust_score", 1.0),
        )

    @staticmethod
    def validate_member_capabilities(
        capabilities: Union[List[str], Set[str], str, None],
    ) -> List[str]:
        """Validate and normalize member capabilities.

        Args:
            capabilities: Capabilities in various formats

        Returns:
            List of capability strings
        """
        if capabilities is None:
            return []

        if isinstance(capabilities, str):
            return [capabilities]

        if isinstance(capabilities, (list, set, tuple)):
            return [str(cap) for cap in capabilities]

        raise MemberTypeError(f"Invalid capabilities type: {type(capabilities)}")

    @staticmethod
    def get_member_by_id(
        coalition: Coalition, agent_id: str
    ) -> Optional[CoalitionMember]:
        """Safely get a member by ID with type checking.

        Args:
            coalition: Coalition to search
            agent_id: ID of agent to find

        Returns:
            CoalitionMember if found, None otherwise
        """
        if not isinstance(coalition, Coalition):
            raise MemberTypeError(f"Expected Coalition, got {type(coalition)}")

        agent_id = str(agent_id)
        return coalition.members.get(agent_id)

    @staticmethod
    def get_members_by_role(
        coalition: Coalition, role: CoalitionRole
    ) -> List[CoalitionMember]:
        """Get all members with a specific role.

        Args:
            coalition: Coalition to search
            role: Role to filter by

        Returns:
            List of members with the specified role
        """
        if not isinstance(coalition, Coalition):
            raise MemberTypeError(f"Expected Coalition, got {type(coalition)}")

        return [member for member in coalition.members.values() if member.role == role]

    @staticmethod
    def get_members_with_capabilities(
        coalition: Coalition, required_capabilities: Union[List[str], Set[str]]
    ) -> List[CoalitionMember]:
        """Get members who have specific capabilities.

        Args:
            coalition: Coalition to search
            required_capabilities: Capabilities to match

        Returns:
            List of members with the required capabilities
        """
        if not isinstance(coalition, Coalition):
            raise MemberTypeError(f"Expected Coalition, got {type(coalition)}")

        required_caps = set(required_capabilities)
        matching_members = []

        for member in coalition.members.values():
            member_caps = set(member.capabilities)
            if required_caps.issubset(member_caps):
                matching_members.append(member)

        return matching_members

    @staticmethod
    def update_member_status(
        coalition: Coalition, agent_id: str, **updates: Any
    ) -> bool:
        """Update member status with type validation.

        Args:
            coalition: Coalition containing the member
            agent_id: ID of member to update
            **updates: Fields to update

        Returns:
            True if member was updated, False if not found
        """
        member = CoalitionMemberManager.get_member_by_id(coalition, agent_id)
        if not member:
            return False

        # Validate and apply updates
        for field, value in updates.items():
            if field == "role" and not isinstance(value, CoalitionRole):
                try:
                    value = CoalitionRole(value)
                except ValueError:
                    logger.warning(f"Invalid role value: {value}")
                    continue

            if field == "capabilities":
                value = CoalitionMemberManager.validate_member_capabilities(value)

            if hasattr(member, field):
                setattr(member, field, value)
            else:
                logger.warning(f"Unknown member field: {field}")

        member.last_activity = datetime.now()
        return True

    @staticmethod
    def calculate_coalition_capabilities(coalition: Coalition) -> Set[str]:
        """Calculate all capabilities available in a coalition.

        Args:
            coalition: Coalition to analyze

        Returns:
            Set of all unique capabilities
        """
        if not isinstance(coalition, Coalition):
            raise MemberTypeError(f"Expected Coalition, got {type(coalition)}")

        all_capabilities = set()
        for member in coalition.members.values():
            all_capabilities.update(member.capabilities)

        return all_capabilities

    @staticmethod
    def validate_coalition_structure(coalition: Coalition) -> List[str]:
        """Validate coalition structure and return any issues found.

        Args:
            coalition: Coalition to validate

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        if not isinstance(coalition, Coalition):
            issues.append(f"Expected Coalition, got {type(coalition)}")
            return issues

        # Check for leader
        leaders = [
            m for m in coalition.members.values() if m.role == CoalitionRole.LEADER
        ]
        if not leaders:
            issues.append("No leader assigned")
        elif len(leaders) > 1:
            issues.append(f"Multiple leaders assigned: {len(leaders)}")

        # Check for valid member types
        for agent_id, member in coalition.members.items():
            if not isinstance(member, CoalitionMember):
                issues.append(f"Invalid member type for {agent_id}: {type(member)}")

            if member.agent_id != agent_id:
                issues.append(f"Member ID mismatch: {member.agent_id} != {agent_id}")

        # Check leader consistency
        if coalition.leader_id:
            if coalition.leader_id not in coalition.members:
                issues.append(f"Leader ID {coalition.leader_id} not in members")
            else:
                leader_member = coalition.members[coalition.leader_id]
                if leader_member.role != CoalitionRole.LEADER:
                    issues.append(
                        f"Leader {coalition.leader_id} does not have LEADER role"
                    )

        return issues
