"""Shared types for coalition system."""

from dataclasses import dataclass
from typing import List

from coalitions.coalition import Coalition, CoalitionObjective


@dataclass
class FormationResult:
    """Result of coalition formation process."""

    coalitions: List[Coalition]
    unassigned_objectives: List[CoalitionObjective]
    formation_quality: float
    objective_coverage: float
    agent_utilization: float
