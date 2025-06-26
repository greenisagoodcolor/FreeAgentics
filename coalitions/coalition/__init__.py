from .business_opportunities import (
    BusinessOpportunity,
    OpportunityDetector,
    OpportunityMetrics,
    OpportunityValidator,
)
from .coalition_criteria import (
    CoalitionFormationCriteria,
    CompatibilityMetric,
    DissolutionCondition,
    FormationTrigger,
)
from .coalition_models import (
    Coalition,
    CoalitionGoal,
    CoalitionGoalStatus,
    CoalitionMember,
    CoalitionRole,
    CoalitionStatus,
)

"""
Coalition Formation and Management System
This module provides functionality for multi-agent coalition formation,
shared goal alignment, and value distribution mechanisms.
"""
__all__ = [
    "CoalitionFormationCriteria",
    "CompatibilityMetric",
    "FormationTrigger",
    "DissolutionCondition",
    "Coalition",
    "CoalitionMember",
    "CoalitionRole",
    "CoalitionStatus",
    "CoalitionGoal",
    "CoalitionGoalStatus",
    "BusinessOpportunity",
    "OpportunityDetector",
    "OpportunityMetrics",
    "OpportunityValidator",
]
