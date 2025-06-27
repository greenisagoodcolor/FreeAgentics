"""
Module for FreeAgentics Active Inference implementation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


class GMNModel:
    """
    A generative model notation model
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.description: str = ""
        self.state_space: Dict[str, Any] = {}
        self.observations: Dict[str, Any] = {}
        self.connections: List[Dict[str, Any]] = []
        self.update_equations: Dict[str, str] = {}
        self.preferences: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}

    def execute(self) -> None:
        """Execute the model"""
        pass


@dataclass
class GNNModel:
    """Represents a parsed GNN model"""

    name: str = ""
    state_space: Dict[str, Any] = field(default_factory=dict)
    observations: Dict[str, Any] = field(default_factory=dict)
    connections: List[Dict[str, Any]] = field(default_factory=list)
    update_equations: Dict[str, str] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
