"""
Coalition Builder Module for FreeAgentics.

This module provides the CoalitionBuilder class for orchestrating
coalition formation between agents.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Coalition:
    """Represents a coalition of agents."""
    id: str
    members: List[str]
    type: str
    status: str = "forming"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CoalitionBuilder:
    """Orchestrates coalition formation between agents."""
    
    def __init__(self):
        """Initialize the coalition builder."""
        self.coalitions: Dict[str, Coalition] = {}
        self.pending_proposals: List[Dict[str, Any]] = []
    
    def propose_coalition(self, proposer_id: str, member_ids: List[str], 
                         coalition_type: str = "resource_sharing") -> Coalition:
        """Propose a new coalition."""
        coalition_id = f"coalition_{len(self.coalitions)}"
        coalition = Coalition(
            id=coalition_id,
            members=[proposer_id] + member_ids,
            type=coalition_type,
            status="proposed"
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
    
    def evaluate_coalition_value(self, coalition: Coalition) -> float:
        """Evaluate the potential value of a coalition."""
        # Simple value calculation based on member count
        return len(coalition.members) * 10.0