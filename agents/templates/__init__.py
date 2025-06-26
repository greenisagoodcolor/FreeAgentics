"""
Active Inference Agent Templates for FreeAgentics.

This module provides mathematically rigorous agent template interfaces
following ADR-002 canonical structure and ADR-005 Active Inference
architecture requirements.

Mathematical Foundation:
    - Discrete-state Active Inference using pymdp
    - Bayesian belief updates: P(s|o) ∝ P(o|s)P(s)
    - Free energy minimization: F = E_q[ln q(s) - ln p(o,s)]
    - Precision parameters: γ (sensory), β (policy), α (state transitions)

Templates Available:
    - ExplorerTemplate: Epistemic value maximization
    - MerchantTemplate: Resource optimization with uncertainty
    - ScholarTemplate: Knowledge accumulation and sharing
    - GuardianTemplate: Coalition protection and monitoring
"""

from .base_template import (
    ActiveInferenceTemplate,
    BeliefState,
    GenerativeModelParams,
    TemplateConfig,
    TemplateInterface,
)
from .explorer_template import ExplorerTemplate
from .pymdp_integration import PyMDPAgentWrapper, create_pymdp_agent

# Template implementations to be added:
# from .guardian_template import GuardianTemplate
# from .merchant_template import MerchantTemplate
# from .scholar_template import ScholarTemplate

__all__ = [
    # Base template interfaces
    "TemplateInterface",
    "ActiveInferenceTemplate",
    "BeliefState",
    "GenerativeModelParams",
    "TemplateConfig",
    # PyMDP integration
    "PyMDPAgentWrapper",
    "create_pymdp_agent",
    # Specific templates
    "ExplorerTemplate",
    # Future templates:
    # "MerchantTemplate",
    # "ScholarTemplate",
    # "GuardianTemplate",
]

__version__ = "1.0.0"
__author__ = "FreeAgentics Active Inference Team"
