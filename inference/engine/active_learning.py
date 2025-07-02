"""
Module for FreeAgentics Active Inference implementation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

from .active_inference import InferenceAlgorithm, InferenceConfig
from .generative_model import (
    DiscreteGenerativeModel,
    GenerativeModel,
    ModelDimensions,
    ModelParameters,
)
from .policy_selection import (
    DiscreteExpectedFreeEnergy,
    Policy,
    PolicyConfig,
    PolicySelector,
    VariationalMessagePassing,
)


class InformationMetric(Enum):
    """Types of information metrics for active learning"""

    ENTROPY = "entropy"
    MUTUAL_INFORMATION = "mutual_information"
    EXPECTED_INFORMATION_GAIN = "expected_information_gain"
    BAYESIAN_SURPRISE = "bayesian_surprise"
    PREDICTION_ERROR = "prediction_error"


@dataclass
class ActiveLearningConfig:
    """Configuration for active learning"""

    exploration_weight: float = 0.3
    information_metric: InformationMetric = InformationMetric.EXPECTED_INFORMATION_GAIN
    min_uncertainty_threshold: float = 0.1
    max_uncertainty_threshold: float = 0.9
    curiosity_decay: float = 0.99
    novelty_weight: float = 0.2
    diversity_weight: float = 0.1
    planning_horizon: int = 5
    num_samples: int = 100
    use_gpu: bool = True
    dtype: torch.dtype = torch.float32
    eps: float = 1e-16


class InformationSeeker(ABC):
    """Abstract base class for information seeking strategies"""

    def __init__(self, config: ActiveLearningConfig) -> None:
        self.config = config
        self.device = torch.device(
            "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        )

    @abstractmethod
    def compute_information_value(
        self, beliefs: torch.Tensor, possible_observations: torch.Tensor
    ) -> torch.Tensor:
        """Compute the value of potential observations for reducing uncertainty"""

    @abstractmethod
    def select_informative_action(
        self, beliefs: torch.Tensor, available_actions: torch.Tensor
    ) -> torch.Tensor:
        """Select action that maximizes information gain"""


class EntropyBasedSeeker(InformationSeeker):
    """
    Information seeker based on entropy reduction.
    Seeks observations that would maximally reduce belief entropy.
    """

    def __init__(self, config: ActiveLearningConfig,
                 generative_model: GenerativeModel) -> None:
        super().__init__(config)
        self.generative_model = generative_model

    def compute_entropy(self, beliefs: torch.Tensor) -> torch.Tensor:
        """Compute Shannon entropy of beliefs"""

        safe_beliefs = beliefs + self.config.eps
        entropy = -torch.sum(safe_beliefs * torch.log(safe_beliefs), dim=-1)
        return entropy

    def compute_information_value(
        self, beliefs: torch.Tensor, possible_observations: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute expected entropy reduction for each possible observation.
        Args:
            beliefs: Current belief states [batch_size x num_states]
            possible_observations: Potential observations [num_obs x obs_dim]
        Returns:
            Information value for each observation [num_obs]
        """
        num_obs = possible_observations.shape[0]
        current_entropy = self.compute_entropy(beliefs)
        expected_entropies = torch.zeros(num_obs, device=self.device)
        if isinstance(self.generative_model, DiscreteGenerativeModel):
            A_matrix = self.generative_model.A
            for i in range(num_obs):
                likelihood = A_matrix[i]
                posterior = likelihood * beliefs
                posterior = posterior / \
                    (posterior.sum(dim=-1, keepdim=True) + self.config.eps)
                posterior_entropy = self.compute_entropy(posterior)
                expected_entropies[i] = posterior_entropy.mean()
        else:
            for i in range(num_obs):
                posterior_entropy = current_entropy * 0.8
                expected_entropies[i] = posterior_entropy.mean()
        info_values = current_entropy.mean() - expected_entropies
        return info_values

    def select_informative_action(
        self, beliefs: torch.Tensor, available_actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Select action that leads to most informative observations.
        Args:
            beliefs: Current beliefs [batch_size x num_states]
            available_actions: Available actions [num_actions x action_dim]
        Returns:
            Selected action index
        """
        if isinstance(self.generative_model, DiscreteGenerativeModel):
            B_matrix = self.generative_model.B
            num_actions = available_actions.shape[0]
            expected_info_gains = torch.zeros(num_actions, device=self.device)
            for a in range(num_actions):
                next_beliefs = torch.matmul(beliefs, B_matrix[:, :, a].T)
                expected_entropy = self.compute_entropy(next_beliefs)
                expected_info_gains[a] = beliefs.shape[0] - \
                    expected_entropy.mean()
            best_action = torch.argmax(expected_info_gains)
            return best_action
        else:
            return torch.randint(0, available_actions.shape[0], (1,))[0]


class MutualInformationSeeker(InformationSeeker):
    """
    Information seeker based on mutual information.
    Maximizes mutual information between beliefs and observations.
    """

    def __init__(
        self,
        config: ActiveLearningConfig,
        generative_model: GenerativeModel,
        inference_algorithm: InferenceAlgorithm,
    ) -> None:
        super().__init__(config)
        self.generative_model = generative_model
        self.inference = inference_algorithm

    def compute_mutual_information(
        self, beliefs: torch.Tensor, observation_dist: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute mutual information I(S;O) = H(S) - H(S|O)
        Args:
            beliefs: Belief distribution over states
            observation_dist: Distribution over observations
        Returns:
            Mutual information value
        """
        belief_entropy = - \
            torch.sum(beliefs * torch.log(beliefs + self.config.eps), dim=-1)
        if isinstance(self.generative_model, DiscreteGenerativeModel):
            A_matrix = self.generative_model.A
            joint = A_matrix.unsqueeze(0) * beliefs.unsqueeze(1).unsqueeze(2)
            marginal_obs = joint.sum(dim=2)
            conditional_entropy = -torch.sum(joint * torch.log(joint / (
                marginal_obs.unsqueeze(2) + self.config.eps) + self.config.eps)).mean()
        else:
            conditional_entropy = belief_entropy * 0.7
        mutual_info = belief_entropy.mean() - conditional_entropy
        return mutual_info

    def compute_information_value(
        self, beliefs: torch.Tensor, possible_observations: torch.Tensor
    ) -> torch.Tensor:
        """Compute information value based on mutual information"""
        num_obs = possible_observations.shape[0]
        info_values = torch.zeros(num_obs, device=self.device)
        for i in range(num_obs):
            obs_dist = torch.zeros(num_obs, device=self.device)
            obs_dist[i] = 1.0
            info_values[i] = self.compute_mutual_information(beliefs, obs_dist)
        return info_values

    def select_informative_action(
        self, beliefs: torch.Tensor, available_actions: torch.Tensor
    ) -> torch.Tensor:
        """Select action that maximizes mutual information"""
        num_actions = available_actions.shape[0]
        expected_mi = torch.zeros(num_actions, device=self.device)
        if isinstance(self.generative_model, DiscreteGenerativeModel):
            B_matrix = self.generative_model.B
            A_matrix = self.generative_model.A
            for a in range(num_actions):
                next_beliefs = torch.matmul(beliefs, B_matrix[:, :, a].T)
                expected_obs_dist = torch.matmul(
                    next_beliefs, A_matrix.sum(dim=0).T)
                expected_obs_dist = expected_obs_dist / expected_obs_dist.sum()
                expected_mi[a] = self.compute_mutual_information(
                    next_beliefs, expected_obs_dist)
        return torch.argmax(expected_mi)


class ActiveLearningAgent:
    """
    Main active learning agent that combines information seeking with action selection.
    Balances exploration for information gain with exploitation for reward.
    """

    def __init__(
        self,
        config: ActiveLearningConfig,
        generative_model: GenerativeModel,
        inference_algorithm: InferenceAlgorithm,
        policy_selector: PolicySelector,
        information_seeker: Optional[InformationSeeker] = None,
    ) -> None:
        self.config = config
        self.generative_model = generative_model
        self.inference = inference_algorithm
        self.policy_selector = policy_selector
        self.device = torch.device(
            "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        )
        if information_seeker is None:
            if config.information_metric == InformationMetric.ENTROPY:
                self.info_seeker = EntropyBasedSeeker(config, generative_model)
            else:
                self.info_seeker = MutualInformationSeeker(
                    config, generative_model, inference_algorithm
                )
        else:
            self.info_seeker = information_seeker
        self.exploration_rate = config.exploration_weight
        self.novelty_memory = []
        self.visit_counts = {}

    def compute_epistemic_value(
        self, beliefs: torch.Tensor, policies: List[Policy]
    ) -> torch.Tensor:
        """
        Compute epistemic value (information gain) for each policy.
        Args:
            beliefs: Current beliefs
            policies: List of possible policies
        Returns:
            Epistemic values for each policy
        """
        # Handle batch dimension if present
        if beliefs.dim() > 1 and beliefs.shape[0] == 1:
            beliefs = beliefs.squeeze(0)

        num_policies = len(policies)
        epistemic_values = torch.zeros(num_policies, device=self.device)
        for i, policy in enumerate(policies):
            _, epistemic, _ = self.policy_selector.compute_expected_free_energy(
                policy, beliefs, self.generative_model)
            epistemic_values[i] = epistemic
        return epistemic_values

    def compute_pragmatic_value(
        self,
        beliefs: torch.Tensor,
        policies: List[Policy],
        preferences: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute pragmatic value (expected utility) for each policy.
        Args:
            beliefs: Current beliefs
            policies: List of possible policies
            preferences: Optional preferences over outcomes
        Returns:
            Pragmatic values for each policy
        """
        # Handle batch dimension if present
        if beliefs.dim() > 1 and beliefs.shape[0] == 1:
            beliefs = beliefs.squeeze(0)

        pragmatic_values = []
        for policy in policies:
            _, _, pragmatic = self.policy_selector.compute_expected_free_energy(
                policy, beliefs, self.generative_model, preferences)
            pragmatic_values.append(pragmatic)
        return torch.tensor(pragmatic_values, device=self.device)

    def select_exploratory_action(
        self,
        beliefs: torch.Tensor,
        available_actions: torch.Tensor,
        preferences: Optional[torch.Tensor] = None,
    ) -> tuple[int, dict[str, float]]:
        """
        Select action balancing exploration and exploitation.
        Args:
            beliefs: Current beliefs
            available_actions: Available actions
            preferences: Optional preferences
        Returns:
            Selected action index and info dict
        """
        policies = self._generate_policies(available_actions)
        pragmatic_values = self.compute_pragmatic_value(
            beliefs, policies, preferences)
        epistemic_values = self.compute_epistemic_value(beliefs, policies)
        novelty_bonus = self._compute_novelty_bonus(beliefs, policies)
        combined_values = (
            -pragmatic_values
            + self.exploration_rate * epistemic_values
            + self.config.novelty_weight * novelty_bonus
        )
        action_probabilities = F.softmax(combined_values, dim=0)
        selected_policy_index = torch.multinomial(
            action_probabilities, 1).item()
        selected_action_index = policies[selected_policy_index].actions[0].item(
        )
        info = {
            "pragmatic_value": pragmatic_values[selected_policy_index].item(),
            "epistemic_value": epistemic_values[selected_policy_index].item(),
            "novelty_bonus": novelty_bonus[selected_policy_index].item(),
            "combined_value": combined_values[selected_policy_index].item(),
            "action_probabilities": action_probabilities.cpu().numpy().tolist(),
        }
        return selected_action_index, info

    def _simulate_policy_observations(
            self,
            beliefs: torch.Tensor,
            policy: Policy) -> torch.Tensor:
        """Simulate expected observations from executing a policy"""
        current_beliefs = beliefs.clone()
        observations = []
        if isinstance(self.generative_model, DiscreteGenerativeModel):
            B_matrix = self.generative_model.B
            A_matrix = self.generative_model.A
            for action in policy.actions[: self.config.planning_horizon]:
                next_beliefs = torch.matmul(
                    current_beliefs, B_matrix[:, :, action].T)
                expected_obs = torch.matmul(
                    next_beliefs, A_matrix.sum(dim=0).T)
                observations.append(expected_obs)
                current_beliefs = next_beliefs
        if observations:
            return torch.stack(observations)
        else:
            return torch.zeros(
                1,
                self.generative_model.dims.num_observations,
                device=self.device)

    def _generate_policies(
            self,
            available_actions: torch.Tensor) -> List[Policy]:
        """Generate a list of single-step policies for each available action"""
        action_indices = torch.argmax(available_actions, dim=1)
        policies = []
        for action_index in action_indices:
            policies.append(
                Policy(actions=torch.tensor([action_index.item()])))
        return policies

    def _compute_novelty_bonus(
            self,
            beliefs: torch.Tensor,
            policies: List[Policy]) -> torch.Tensor:
        """Compute novelty bonus for each policy based on state visitation"""
        novelty_values = torch.zeros(len(policies), device=self.device)
        for i, policy in enumerate(policies):
            state_hash = self._hash_belief_state(beliefs)
            visit_count = self.visit_counts.get(state_hash, 0)
            novelty_values[i] = 1.0 / (1.0 + visit_count)
        return novelty_values

    def _hash_belief_state(self, beliefs: torch.Tensor) -> str:
        """Create hash of belief state for novelty tracking"""
        discretized = (beliefs * 100).round().int()
        return str(discretized.tolist())

    def update_novelty_memory(
            self,
            beliefs: torch.Tensor,
            observation: torch.Tensor) -> None:
        """Update novelty memory with new experience"""

        state_hash = self._hash_belief_state(beliefs)
        self.visit_counts[state_hash] = self.visit_counts.get(
            state_hash, 0) + 1
        self.novelty_memory.append((beliefs.clone(), observation.clone()))
        if len(self.novelty_memory) > 1000:
            self.novelty_memory.pop(0)


class InformationGainPlanner:
    """
    Planner that explicitly plans information-gathering trajectories.
    Plans sequences of actions to reduce uncertainty about specific aspects.
    """

    def __init__(
        self,
        config: ActiveLearningConfig,
        generative_model: GenerativeModel,
        information_seeker: InformationSeeker,
    ) -> None:
        self.config = config
        self.generative_model = generative_model
        self.info_seeker = information_seeker
        self.device = torch.device(
            "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        )

    def plan_information_gathering(
        self,
        current_beliefs: torch.Tensor,
        target_uncertainty: float = 0.1,
        max_steps: int = 10,
    ) -> List[int]:
        """
        Plan sequence of actions to reduce uncertainty below target.
        Args:
            current_beliefs: Current belief state
            target_uncertainty: Target uncertainty level
            max_steps: Maximum planning steps
        Returns:
            Sequence of actions
        """
        planned_actions = []
        beliefs = current_beliefs.clone()
        for step in range(max_steps):
            entropy = -torch.sum(beliefs * torch.log(beliefs +
                                 self.config.eps), dim=-1).mean()
            if entropy < target_uncertainty:
                break
            if isinstance(self.generative_model, DiscreteGenerativeModel):
                num_actions = self.generative_model.dims.num_actions
                action_values = torch.zeros(num_actions, device=self.device)
                B_matrix = self.generative_model.B
                for a in range(num_actions):
                    next_beliefs = torch.matmul(beliefs, B_matrix[:, :, a].T)
                    next_entropy = -torch.sum(
                        next_beliefs * torch.log(next_beliefs + self.config.eps), dim=-1
                    ).mean()
                    action_values[a] = entropy - next_entropy
                best_action = torch.argmax(action_values)
                planned_actions.append(best_action.item())
                beliefs = torch.matmul(beliefs, B_matrix[:, :, best_action].T)
        return planned_actions


def create_active_learner(
    learner_type: str, config: Optional[ActiveLearningConfig] = None, **kwargs
) -> Union[ActiveLearningAgent, InformationGainPlanner]:
    """
    Factory function to create active learners.
    Args:
        learner_type: Type of learner ('agent', 'planner')
        config: Configuration
        **kwargs: Additional parameters
    Returns:
        Active learner instance
    """
    if config is None:
        config = ActiveLearningConfig()
    if learner_type == "agent":
        generative_model = kwargs.get("generative_model")
        inference_algorithm = kwargs.get("inference_algorithm")
        policy_selector = kwargs.get("policy_selector")
        if None in [generative_model, inference_algorithm, policy_selector]:
            raise ValueError(
                "Agent requires generative_model, inference_algorithm, and policy_selector"
            )
        return ActiveLearningAgent(
            config,
            generative_model,
            inference_algorithm,
            policy_selector)
    elif learner_type == "planner":
        generative_model = kwargs.get("generative_model")
        if generative_model is None:
            raise ValueError("Planner requires generative_model")
        if config.information_metric == InformationMetric.ENTROPY:
            info_seeker = EntropyBasedSeeker(config, generative_model)
        else:
            inference_algorithm = kwargs.get("inference_algorithm")
            if inference_algorithm is None:
                raise ValueError(
                    "Mutual information seeker requires inference_algorithm")
            info_seeker = MutualInformationSeeker(
                config, generative_model, inference_algorithm)
        return InformationGainPlanner(config, generative_model, info_seeker)
    else:
        raise ValueError(f"Unknown learner type: {learner_type}")


if __name__ == "__main__":
    # Example usage
    # DiscreteGenerativeModel, ModelDimensions, ModelParameters already
    # imported above

    config = ActiveLearningConfig(
        exploration_weight=0.3,
        information_metric=InformationMetric.ENTROPY,
        use_gpu=False,
    )
    dims = ModelDimensions(num_states=4, num_observations=3, num_actions=2)
    params = ModelParameters(use_gpu=False)
    gen_model = DiscreteGenerativeModel(dims, params)

    inf_config = InferenceConfig(use_gpu=False)
    inference = VariationalMessagePassing(inf_config)

    pol_config = PolicyConfig(use_gpu=False)
    policy_selector = DiscreteExpectedFreeEnergy(pol_config)

    learner = ActiveLearningAgent(
        config, gen_model, inference, policy_selector)

    beliefs = torch.softmax(torch.randn(1, 4), dim=-1)
    available_actions = torch.eye(2)

    action, info = learner.select_exploratory_action(
        beliefs, available_actions)
    print(f"Selected action: {action}")
    print(f"Info: {info}")
