"""
PyMDP Integration Module for Active Inference.

This module provides integration between the FreeAgentics active inference
implementation and pymdp conventions, enabling seamless interoperability
with the pymdp library and GeneralizedNotationNotation (GNN/GMN) models.

Key features:
- pymdp-compatible tensor conventions
- GMN/GNN model generation for pymdp structures
- Belief updates following pymdp algorithms
- Free energy computation matching pymdp formulations
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from ..gnn.generator import GMNGenerator
from ..gnn.parser import GMNParser
from .active_inference import InferenceConfig, VariationalMessagePassing
from .generative_model import (
    DiscreteGenerativeModel,
    GenerativeModel,
    ModelDimensions,
    ModelParameters,
)

logger = logging.getLogger(__name__)


class PyMDPActiveInference:
    """
    PyMDP-compatible Active Inference implementation.

    This class provides a bridge between FreeAgentics and pymdp conventions,
    ensuring tensor shapes, belief updates, and free energy calculations
    align with pymdp library standards.
    """

    def __init__(self, generative_model: GenerativeModel,
                 inference_config: Optional[InferenceConfig] = None):
        """Initialize pymdp-compatible active inference."""
        self.generative_model = generative_model
        self.config = inference_config or InferenceConfig()
        self.vmp = VariationalMessagePassing(self.config)

        # PyMDP convention tracking
        self.pymdp_compatible = True
        self._validate_pymdp_compatibility()

    def _validate_pymdp_compatibility(self) -> None:
        """Validate that the generative model follows pymdp conventions."""
        if not hasattr(self.generative_model, "A"):
            logger.warning("No A matrix found - may not be pymdp compatible")
            return

        # Check A matrix shape: should be (num_obs, num_states)
        A = self.generative_model.A
        if hasattr(self.generative_model, "dims"):
            dims = self.generative_model.dims
            expected_shape = (dims.num_observations, dims.num_states)
            if A.shape != expected_shape:
                logger.warning(
                    f"A matrix shape {
                        A.shape} doesn't match pymdp convention {expected_shape}")

        # Check B matrix shape: should be (num_states, num_states, num_actions)
        if hasattr(self.generative_model, "B"):
            B = self.generative_model.B
            expected_B_shape = (
                dims.num_states,
                dims.num_states,
                dims.num_actions)
            if B.shape != expected_B_shape:
                logger.warning(
                    f"B matrix shape {
                        B.shape} doesn't match pymdp convention {expected_B_shape}")

    def update_beliefs(self,
                       observations: Union[int,
                                           torch.Tensor],
                       prior_beliefs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Update beliefs using pymdp-compatible variational message passing.

        Args:
            observations: Observation indices or distributions
            prior_beliefs: Prior belief distribution over states

        Returns:
            Posterior beliefs following pymdp categorical distribution conventions
        """
        # Convert observations to tensor if needed
        if isinstance(observations, int):
            observations = torch.tensor(observations, dtype=torch.long)
        elif isinstance(observations, (list, np.ndarray)):
            observations = torch.tensor(observations, dtype=torch.long)

        # Use pymdp-aligned VMP
        posterior = self.vmp.infer_states(
            observations=observations,
            generative_model=self.generative_model,
            prior_beliefs=prior_beliefs,
        )

        # Ensure categorical distribution properties (pymdp requirement)
        posterior = posterior / (posterior.sum(dim=-1, keepdim=True) + 1e-16)

        return posterior

    def compute_expected_free_energy(self,
                                     beliefs: torch.Tensor,
                                     actions: Union[int,
                                                    torch.Tensor],
                                     time_horizon: int = 1) -> torch.Tensor:
        """
        Compute expected free energy for action selection (pymdp style).

        This implements the pymdp formulation:
        G(π) = E_q[ln q(s) - ln p(s,o|π)]

        Args:
            beliefs: Current belief distribution
            actions: Action or action sequence
            time_horizon: Planning horizon

        Returns:
            Expected free energy for the action(s)
        """
        if isinstance(actions, int):
            actions = torch.tensor([actions], dtype=torch.long)
        elif isinstance(actions, (list, np.ndarray)):
            actions = torch.tensor(actions, dtype=torch.long)

        # Get model components (pymdp convention)
        A = self.generative_model.A  # (num_obs, num_states)
        B = self.generative_model.B  # (num_states, num_states, num_actions)

        # Get preferences
        if hasattr(self.generative_model, "get_preferences"):
            C = self.generative_model.get_preferences()
            if C.dim() > 1:
                C = C[:, 0]  # Use first timestep preferences
        else:
            # Default neutral preferences
            C = torch.zeros(A.shape[0])

        total_efe = 0.0
        current_beliefs = beliefs.clone()

        for t in range(time_horizon):
            action_idx = actions[min(t, len(actions) - 1)]

            # Predict next state: P(s_{t+1}|s_t, a_t) (pymdp transition)
            if current_beliefs.dim() == 1:
                predicted_states = B[:, :, action_idx] @ current_beliefs
            else:
                # Handle batch
                predicted_states = torch.zeros_like(current_beliefs)
                for i in range(current_beliefs.shape[0]):
                    predicted_states[i] = B[:, :,
                                            action_idx] @ current_beliefs[i]

            # Predict observations: P(o|s) (pymdp observation model)
            predicted_obs = (A @ predicted_states if predicted_states.dim()
                             == 1 else (A @ predicted_states.T).T)

            # Instrumental value: E_q[ln P(o|C)] (preference satisfaction)
            if predicted_obs.dim() == 1:
                instrumental = torch.sum(predicted_obs * C)
            else:
                instrumental = torch.sum(
                    predicted_obs * C.unsqueeze(0), dim=1).sum()

            # Epistemic value: E_q[H[P(o|s)]] (information gain)
            if predicted_obs.dim() == 1:
                entropy_obs = -torch.sum(predicted_obs *
                                         torch.log(predicted_obs + 1e-16))
            else:
                entropy_obs = -torch.sum(
                    predicted_obs * torch.log(predicted_obs + 1e-16), dim=1
                ).sum()

            # Expected free energy: G = -Instrumental - Epistemic (pymdp sign
            # convention)
            efe_t = -instrumental - entropy_obs
            total_efe += efe_t

            # Update beliefs for next timestep
            current_beliefs = predicted_states

        return total_efe

    def select_action(
        self, beliefs: torch.Tensor, num_actions: Optional[int] = None
    ) -> Tuple[int, torch.Tensor]:
        """
        Select action by minimizing expected free energy (pymdp policy selection).

        Args:
            beliefs: Current belief distribution
            num_actions: Number of available actions

        Returns:
            Tuple of (selected_action, efe_values_for_all_actions)
        """
        if num_actions is None:
            if hasattr(self.generative_model, "dims"):
                num_actions = self.generative_model.dims.num_actions
            else:
                num_actions = 4  # Default

        # Compute expected free energy for each action
        efe_values = torch.zeros(num_actions)
        for action in range(num_actions):
            efe_values[action] = self.compute_expected_free_energy(
                beliefs, action)

        # Select action with minimum expected free energy (pymdp convention)
        selected_action = torch.argmin(efe_values).item()

        return selected_action, efe_values


class GMNToPyMDPConverter:
    """
    Converter between GeneralizedNotationNotation (GMN) and pymdp structures.

    This class enables agents to use LLMs to generate GMN models that are
    then converted to pymdp-compatible structures for active inference.
    """

    def __init__(self):
        """Initialize the GMN to pymdp converter."""
        self.gmn_generator = GMNGenerator()
        self.gmn_parser = GMNParser()

    def generate_pymdp_model_from_gmn(
        self, agent_config: Dict[str, Any]
    ) -> DiscreteGenerativeModel:
        """
        Generate a pymdp-compatible model from GMN specification.

        Args:
            agent_config: Configuration including agent_name, agent_class, personality

        Returns:
            DiscreteGenerativeModel following pymdp conventions
        """
        # For now, create a simplified GMN-inspired model
        # TODO: Integrate with full GMN generation when available
        agent_class = agent_config.get("agent_class", "Explorer")
        agent_config.get("personality", {})

        # Determine model dimensions based on agent class
        if agent_class == "Explorer":
            num_states = 4  # {unexplored, explored, resource, danger}
            num_observations = 3  # {novel, familiar, threat}
            num_actions = 2  # {explore, exploit}
        elif agent_class == "Merchant":
            num_states = 4  # {searching, trading, stocked, depleted}
            num_observations = 3  # {resource, customer, competitor}
            num_actions = 3  # {search, trade, wait}
        elif agent_class == "Guardian":
            num_states = 3  # {safe, alert, defending}
            num_observations = 3  # {peaceful, suspicious, threat}
            num_actions = 3  # {patrol, watch, defend}
        else:
            num_states = 4
            num_observations = 3
            num_actions = 2

        # Create model dimensions (pymdp style)
        dimensions = ModelDimensions(
            num_states=num_states,
            num_observations=num_observations,
            num_actions=num_actions,
            time_horizon=1,
        )

        # Create model parameters
        parameters = ModelParameters(
            learning_rate=0.01, use_gpu=False, dtype=torch.float32  # CPU for compatibility
        )

        # Create discrete generative model
        model = DiscreteGenerativeModel(dimensions, parameters)

        # Customize model based on agent specification
        self._customize_model_from_config(model, agent_config)

        return model

    def _customize_model_from_config(
        self, model: DiscreteGenerativeModel, agent_config: Dict[str, Any]
    ) -> None:
        """Customize the pymdp model based on agent configuration and personality."""
        personality = agent_config.get("personality", {})
        agent_class = agent_config.get("agent_class", "Explorer")

        # Customize A matrix based on agent class and personality
        if agent_class == "Explorer":
            # Explorers have higher sensory acuity for novel observations
            curiosity = personality.get("curiosity", 0.5)
            model.A.data *= 1.0 + curiosity * 0.5

        elif agent_class == "Merchant":
            # Merchants are better at detecting resource-related observations
            efficiency = personality.get("efficiency", 0.5)
            model.A.data[1, :] *= 1.0 + efficiency * \
                0.3  # Enhance resource detection

        elif agent_class == "Scholar":
            # Scholars have balanced observation capabilities
            model.A.data += torch.randn_like(model.A.data) * 0.1

        elif agent_class == "Guardian":
            # Guardians are better at detecting threats
            risk_awareness = 1.0 - personality.get("risk_tolerance", 0.5)
            model.A.data[2, :] *= 1.0 + risk_awareness * \
                0.4  # Enhance threat detection

        # Normalize A matrix (pymdp requirement)
        model.A.data = model.A.data / \
            (model.A.data.sum(dim=0, keepdim=True) + 1e-16)

        # Customize B matrix based on personality
        exploration = personality.get("exploration", 0.5)
        personality.get("cooperation", 0.5)

        # Adjust transition probabilities
        for a in range(model.dims.num_actions):
            # More exploratory agents have more random transitions
            randomness = exploration * 0.3
            model.B.data[:,
                         :,
                         a] += torch.randn_like(model.B.data[:,
                                                             :,
                                                             a]) * randomness

            # Normalize (pymdp requirement)
            model.B.data[:, :, a] = model.B.data[:, :, a] / (
                model.B.data[:, :, a].sum(dim=0, keepdim=True) + 1e-16
            )

        # Set preferences based on agent class (pymdp C matrix)
        if agent_class == "Explorer":
            # Explorers prefer novel/uncertain states
            # Avoid known, seek novel
            model.C.data = torch.tensor([-0.5, 0.0, -1.0]).unsqueeze(1)
        elif agent_class == "Merchant":
            # Merchants prefer resource-rich states
            model.C.data = torch.tensor(
                [0.0, 1.0, -0.5]).unsqueeze(1)  # Seek resources
        elif agent_class == "Guardian":
            # Guardians prefer safe states
            # Seek safety, avoid danger
            model.C.data = torch.tensor([0.5, 0.0, -2.0]).unsqueeze(1)
        else:
            # Default neutral preferences
            model.C.data = torch.zeros(
                model.dims.num_observations,
                model.dims.time_horizon)


def create_pymdp_agent(agent_config: Dict[str, Any]) -> PyMDPActiveInference:
    """
    Factory function to create a pymdp-compatible active inference agent.

    This function integrates GMN generation with pymdp conventions to create
    agents that can use LLM-generated models for active inference.

    Args:
        agent_config: Dict containing agent_name, agent_class, personality

    Returns:
        PyMDPActiveInference agent ready for deployment
    """
    # Create GMN to pymdp converter
    converter = GMNToPyMDPConverter()

    # Generate pymdp-compatible model from GMN
    generative_model = converter.generate_pymdp_model_from_gmn(agent_config)

    # Create inference configuration
    inference_config = InferenceConfig(
        algorithm="variational_message_passing",
        num_iterations=16,
        convergence_threshold=1e-4,
        use_gpu=False,  # CPU for compatibility
    )

    # Create pymdp active inference agent
    agent = PyMDPActiveInference(generative_model, inference_config)

    logger.info(
        f"Created pymdp-compatible agent: {agent_config['agent_name']} "
        f"({agent_config['agent_class']})"
    )

    return agent


# Example usage
if __name__ == "__main__":
    # Example agent configuration
    agent_config = {
        "agent_name": "CuriousExplorer",
        "agent_class": "Explorer",
        "personality": {
            "exploration": 0.8,
            "cooperation": 0.6,
            "efficiency": 0.4,
            "curiosity": 0.9,
            "risk_tolerance": 0.7,
        },
    }

    # Create pymdp-compatible agent
    agent = create_pymdp_agent(agent_config)

    # Example usage
    # Observed something interesting
    observations = torch.tensor(1, dtype=torch.long)
    beliefs = agent.update_beliefs(observations)
    action, efe_values = agent.select_action(beliefs)

    print(f"Agent beliefs: {beliefs}")
    print(f"Selected action: {action}")
    print(f"Expected free energies: {efe_values}")
