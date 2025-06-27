"""
Module for FreeAgentics Active Inference implementation.
"""

from typing import Optional, Tuple

import torch

from ..engine.belief_update import GraphNNBeliefUpdater as BeliefUpdater
from ..engine.generative_model import GenerativeModel
from ..engine.gnn_integration import GNNIntegrationConfig
from ..engine.policy_selection import Policy, PolicySelector


class GNNActiveInferenceAdapter:
    """
    Adapter for GNN-based Active Inference.
    """

    def __init__(
        self,
        config: GNNIntegrationConfig,
        generative_model: GenerativeModel,
        belief_updater: BeliefUpdater,
        policy_selector: PolicySelector,
    ) -> None:
        self.config = config
        self.generative_model = generative_model
        self.belief_updater = belief_updater
        self.policy_selector = policy_selector

    def update_beliefs(
        self,
        current_beliefs: torch.Tensor,
        observation: torch.Tensor,
        preferences: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Policy, torch.Tensor]:
        """
        Update beliefs and select a policy.
        """
        # Update beliefs
        updated_beliefs = self.belief_updater.update_beliefs(
            current_beliefs, observation, self.generative_model
        )

        # Select policy
        selected_policy, G_values = self.policy_selector.select_policy(
            updated_beliefs, self.generative_model, preferences
        )

        return updated_beliefs, selected_policy, G_values
