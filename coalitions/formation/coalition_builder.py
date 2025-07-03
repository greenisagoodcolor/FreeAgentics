"""
Coalition Builder Module for FreeAgentics.

This module provides the CoalitionBuilder class for orchestrating
coalition formation between agents.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Coalition:
    """Represents a coalition of agents."""

    id: str
    members: List[str]
    type: str
    status: str = "forming"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class CoalitionBuilder:
    """Orchestrates coalition formation between agents."""

    def __init__(self):
        """Initialize the coalition builder."""
        self.coalitions: Dict[str, Coalition] = {}
        self.pending_proposals: List[Dict[str, Any]] = []

    def propose_coalition(
        self, proposer_id: str, member_ids: List[str], coalition_type: str = "resource_sharing"
    ) -> Coalition:
        """Propose a new coalition."""
        coalition_id = f"coalition_{len(self.coalitions)}"
        coalition = Coalition(
            id=coalition_id,
            members=[proposer_id] + member_ids,
            type=coalition_type,
            status="proposed",
        )
        self.coalitions[coalition_id] = coalition
        return coalition

    def accept_proposal(self, coalition_id: str, agent_id: str) -> bool:
        """Accept a coalition proposal."""
        if coalition_id in self.coalitions:
            coalition = self.coalitions[coalition_id]
            if agent_id in coalition.members:
                coalition.status = "active"
                return True
        return False

    async def form_coalition(
        self, coalition_id: str, agent_ids: List[str], business_type: str = "ResourceOptimization"
    ) -> Coalition:
        """Form a coalition with the given agents.
        
        This is the main method for coalition formation that tests expect.
        """
        try:
            # Validate inputs
            if not coalition_id or not agent_ids:
                raise ValueError("Coalition ID and agent IDs are required")
            
            if len(agent_ids) < 2:
                raise ValueError("Coalition requires at least 2 agents")
            
            # Create coalition
            coalition = Coalition(
                id=coalition_id,
                members=agent_ids.copy(),
                type=business_type,
                status="active",
                metadata={
                    "formation_time": "now",
                    "business_type": business_type,
                    "member_count": len(agent_ids)
                }
            )
            
            # Store coalition
            self.coalitions[coalition_id] = coalition
            
            return coalition
            
        except Exception as e:
            # Log error and re-raise for test visibility
            print(f"Coalition formation failed: {e}")
            raise

    def evaluate_coalition_value(self, coalition: Coalition) -> float:
        """Evaluate the potential value of a coalition."""
        # Simple value calculation based on member count
        return len(coalition.members) * 10.0
