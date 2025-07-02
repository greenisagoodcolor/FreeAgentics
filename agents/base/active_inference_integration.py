"""
Module for FreeAgentics Active Inference implementation.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

# Graceful degradation for PyTorch-dependent imports
try:
    from inference.engine import (
        create_generative_model,
        create_inference_algorithm,
        create_policy_selector,
        create_precision_optimizer,
        create_temporal_planner,
    )
    from inference.engine.active_inference import InferenceConfig
    from inference.engine.belief_update import (
        BeliefUpdateConfig,
        DirectGraphObservationModel,
        create_belief_updater,
    )
    from inference.engine.generative_model import ModelDimensions
    from inference.engine.policy_selection import PolicyConfig
    from inference.engine.precision import PrecisionConfig
    from inference.engine.temporal_planning import PlanningConfig

    ACTIVE_INFERENCE_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    # Handle PyTorch import errors or runtime errors
    ACTIVE_INFERENCE_AVAILABLE = False
    create_generative_model = None
    create_inference_algorithm = None
    create_policy_selector = None
    create_precision_optimizer = None
    create_temporal_planner = None
    InferenceConfig = None
    BeliefUpdateConfig = None
    DirectGraphObservationModel = None
    create_belief_updater = None
    ModelDimensions = None
    PolicyConfig = None
    PrecisionConfig = None
    PlanningConfig = None
    print(f"Warning: Active Inference engine not available: {e}")

# Import base agent components (these should not depend on PyTorch)
from agents.base.data_model import Agent, AgentGoal, AgentStatus
from agents.base.decision_making import Action, ActionType, DecisionSystem
from agents.base.memory import Memory, MemorySystem, MemoryType
from agents.base.movement import MovementController
from agents.base.perception import Percept, PerceptionSystem, StimulusType
from agents.base.state_manager import AgentStateManager

"""
Active Inference Integration for Basic Agent System
This module provides the integration layer between the Basic Agent system
and the Active Inference Engine, enabling agents to use active inference
for decision-making, planning, and learning.
"""

# from inference.engine.pymdp_policy_selector import PyMDPPolicyAdapter
# Using policy selector from base engine for now

logger = logging.getLogger(__name__)


class IntegrationMode(Enum):
    """Different modes of integration between Basic Agent and
    Active Inference"""

    FULL = "full"
    HYBRID = "hybrid"
    ADVISORY = "advisory"
    LEARNING = "learning"


@dataclass
class ActiveInferenceConfig:
    """Configuration for Active Inference integration"""

    mode: IntegrationMode = IntegrationMode.HYBRID
    num_states: int = 100
    num_observations: int = 50
    num_actions: int = 10
    model_type: str = "discrete"
    inference_type: str = "vmp"
    max_iterations: int = 100
    convergence_threshold: float = 0.0001
    planning_horizon: int = 5
    planning_type: str = "mcts"
    num_simulations: int = 100
    initial_precision: float = 1.0
    adapt_precision: bool = True
    belief_update_rate: float = 1.0
    action_selection_threshold: float = 0.7
    learning_rate: float = 0.01
    enable_visualization: bool = True
    visualization_update_rate: float = 10.0


class StateToBeliefMapper:
    """Maps agent state to active inference belief state"""

    def __init__(self, config: ActiveInferenceConfig) -> None:
        self.config = config
        self.state_dimensions = self._calculate_state_dimensions()

    def _calculate_state_dimensions(self) -> Dict[str, int]:
        """Calculate dimensions for different state components"""
        return {
            "position": 3,
            "status": len(AgentStatus),
            "resources": 3,
            "goals": 5,
            "social": 10,
        }

    def map_to_belief(
            self,
            agent: Agent,
            state_manager: AgentStateManager) -> np.ndarray:
        """Convert agent state to belief vector"""
        belief_vector = []
        pos = agent.position
        belief_vector.extend([pos.x, pos.y, pos.z])
        status_vector = np.zeros(len(AgentStatus))
        status_vector[list(AgentStatus).index(agent.status)] = 1.0
        belief_vector.extend(status_vector)
        belief_vector.extend(
            [
                agent.resources.energy / 100.0,
                agent.resources.health / 100.0,
                agent.resources.memory_used / agent.resources.memory_capacity,
            ]
        )
        goal_features = self._encode_goals(agent.goals)
        belief_vector.extend(goal_features)
        social_features = self._encode_social_state(agent)
        belief_vector.extend(social_features)
        return np.array(belief_vector)

    def _encode_goals(self, goals: List[AgentGoal]) -> List[float]:
        """Encode agent goals into fixed-size feature vector"""
        features = []
        max_goals = self.state_dimensions["goals"]
        for i in range(max_goals):
            if i < len(goals):
                goal = goals[i]
                priority_map: Dict[str, float] = {
                    "high": 1.0, "medium": 0.5, "low": 0.2}
                priority_str = str(
                    goal.priority) if goal.priority is not None else "medium"
                priority_val = priority_map.get(priority_str, 0.5)
                features.append(priority_val)
            else:
                features.append(0.0)
        return features

    def _encode_social_state(self, agent: Agent) -> List[float]:
        """Encode social relationships into feature vector"""
        features = []
        # Use the relationships field instead of social_relationships
        if hasattr(agent, "relationships") and agent.relationships:
            # Assume relationships is a dict with trust-like values
            trust_values = [
                float(v) for v in agent.relationships.values() if isinstance(
                    v, (int, float))]
            avg_trust = float(np.mean(trust_values)) if trust_values else 0.5
            features.append(avg_trust)
            relationship_density = min(len(agent.relationships) / 10.0, 1.0)
            features.append(relationship_density)
        else:
            features.append(0.5)  # Default trust level
            features.append(0.0)  # No relationships
        # Pad to required dimensions
        while len(features) < self.state_dimensions["social"]:
            features.append(0.0)
        return features[: self.state_dimensions["social"]]


class PerceptionToObservationMapper:
    """Maps perception system output to active inference observations"""

    def __init__(self, config: ActiveInferenceConfig) -> None:
        self.config = config
        self.observation_dimensions = self._calculate_observation_dimensions()

    def _calculate_observation_dimensions(self) -> Dict[str, int]:
        """Calculate dimensions for observation components"""
        # Ensure dimensions match the configured total
        total_obs = self.config.num_observations
        return {
            "visual": min(20, total_obs // 2),  # Half for visual
            "auditory": min(10, total_obs // 4),  # Quarter for auditory
            "proximity": min(10, total_obs // 4),  # Quarter for proximity
            "internal": max(
                0,
                total_obs
                - min(20, total_obs // 2)
                - min(10, total_obs // 4)
                - min(10, total_obs // 4),
            ),
        }

    def map_to_observation(self, percepts: List[Percept]) -> np.ndarray:
        """Convert percepts to observation vector"""
        observation = np.zeros(self.config.num_observations)
        visual_percepts = [
            p for p in percepts if p.stimulus.stimulus_type == StimulusType.OBJECT]
        auditory_percepts = [
            p for p in percepts if p.stimulus.stimulus_type == StimulusType.SOUND]
        proximity_percepts = [
            p for p in percepts if p.stimulus.stimulus_type == StimulusType.AGENT]
        idx = 0
        # Visual features
        visual_dim = self.observation_dimensions["visual"]
        if visual_dim > 0:
            visual_features = self._encode_visual_percepts(visual_percepts)
            end_idx = min(idx + visual_dim, len(observation))
            observation[idx:end_idx] = visual_features[: end_idx - idx]
            idx = end_idx
        # Auditory features
        auditory_dim = self.observation_dimensions["auditory"]
        if auditory_dim > 0 and idx < len(observation):
            auditory_features = self._encode_auditory_percepts(
                auditory_percepts)
            end_idx = min(idx + auditory_dim, len(observation))
            observation[idx:end_idx] = auditory_features[: end_idx - idx]
            idx = end_idx
        # Proximity features
        proximity_dim = self.observation_dimensions["proximity"]
        if proximity_dim > 0 and idx < len(observation):
            proximity_features = self._encode_proximity_percepts(
                proximity_percepts)
            end_idx = min(idx + proximity_dim, len(observation))
            observation[idx:end_idx] = proximity_features[: end_idx - idx]
        return observation

    def _encode_visual_percepts(self, percepts: List[Percept]) -> np.ndarray:
        """Encode visual percepts into feature vector"""
        features = np.zeros(self.observation_dimensions["visual"])
        for i, percept in enumerate(percepts[:5]):
            if i * 4 < len(features):
                features[i * 4] = percept.confidence
                features[i * 4 + 1] = percept.confidence
                if hasattr(percept.stimulus, "position"):
                    features[i * 4 + 2] = percept.stimulus.position.x / 100.0
                    features[i * 4 + 3] = percept.stimulus.position.y / 100.0
        return features

    def _encode_auditory_percepts(self, percepts: List[Percept]) -> np.ndarray:
        """Encode auditory percepts into feature vector"""
        features = np.zeros(self.observation_dimensions["auditory"])
        for i, percept in enumerate(percepts[:3]):
            if i * 3 < len(features):
                features[i * 3] = percept.confidence
                features[i * 3 + 1] = percept.confidence
                features[i * 3 +
                         2] = getattr(percept.stimulus, "intensity", 0.5)
        return features

    def _encode_proximity_percepts(
            self, percepts: List[Percept]) -> np.ndarray:
        """Encode proximity percepts into feature vector"""
        features = np.zeros(self.observation_dimensions["proximity"])
        if percepts:
            features[0] = min(len(percepts) / 10.0, 1.0)
            features[1] = max(p.confidence for p in percepts)
        return features


class ActionMapper:
    """Maps between active inference actions and agent actions"""

    def __init__(self, config: ActiveInferenceConfig) -> None:
        self.config = config
        self.action_map = self._create_action_mapping()

    def _create_action_mapping(self) -> Dict[int, ActionType]:
        """Create mapping from action indices to action types"""
        action_types = list(ActionType)
        mapping = {}
        for i in range(min(self.config.num_actions, len(action_types))):
            mapping[i] = action_types[i]
        return mapping

    def map_to_agent_action(self, action_idx: int, agent: Agent) -> Action:
        """Convert active inference action index to agent action"""
        action_type: ActionType = self.action_map.get(
            action_idx, ActionType.WAIT)
        action = Action(
            action_type=action_type,
            parameters=self._get_action_parameters(action_type, agent),
            utility=1.0,
            cost=0.1,
        )
        return action

    def _get_action_parameters(self, action_type: ActionType,
                               agent: Agent) -> Dict[str, Any]:
        """Generate appropriate parameters for action type"""
        params: Dict[str, Any] = {}
        if action_type == ActionType.MOVE:
            params["direction"] = self._calculate_movement_direction(agent)
            params["speed"] = 1.0
        elif action_type == ActionType.INTERACT:
            params["target"] = self._find_interaction_target(agent)
        elif action_type == ActionType.EXPLORE:
            params["focus"] = "environment"
        return params

    def _calculate_movement_direction(self, agent: Agent) -> np.ndarray:
        """Calculate movement direction based on goals"""
        if agent.goals and hasattr(agent.goals[0], "target_position"):
            target = agent.goals[0].target_position
            current = agent.position

            # Ensure we have valid positions before accessing attributes
            if target is None or current is None:
                return np.array([0.0, 0.0, 0.0])

            # Handle different position formats safely
            if (
                hasattr(current, "x")
                and hasattr(current, "y")
                and hasattr(target, "x")
                and hasattr(target, "y")
            ):
                direction = np.array(
                    [float(target.x) - float(current.x), float(target.y) - float(current.y), 0.0]
                )
            elif (
                hasattr(current, "__getitem__")
                and hasattr(current, "__len__")
                and len(current) >= 2
                and target is not None
                and hasattr(target, "x")
                and hasattr(target, "y")
            ):
                direction = np.array([float(target.x) -
                                      float(current[0]), float(target.y) -
                                      float(current[1]), 0.0])
            else:
                direction = np.array([0.0, 0.0, 0.0])
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            return direction
        return np.array([0.0, 0.0, 0.0])

    def _find_interaction_target(self, agent: Agent) -> Optional[str]:
        """Find suitable interaction target"""
        return None


class ActiveInferenceIntegration:
    """Main integration class connecting Basic Agent with Active Inference"""

    def __init__(
        self,
        agent: Agent,
        state_manager: AgentStateManager,
        perception_system: PerceptionSystem,
        decision_system: DecisionSystem,
        movement_controller: MovementController,
        memory_system: MemorySystem,
        config: Optional[ActiveInferenceConfig] = None,
    ) -> None:
        self.agent = agent
        self.state_manager = state_manager
        self.perception_system = perception_system
        self.decision_system = decision_system
        self.movement_controller = movement_controller
        self.memory_system = memory_system
        self.config = config or ActiveInferenceConfig()
        self.state_mapper = StateToBeliefMapper(self.config)
        self.perception_mapper = PerceptionToObservationMapper(self.config)
        self.action_mapper = ActionMapper(self.config)
        self._initialize_active_inference()
        self.current_belief = None
        self.last_observation = None
        self.last_action = None
        logger.info(
            f"Active Inference Integration initialized for agent {
                agent.agent_id}")

    def _initialize_active_inference(self):
        """Initialize all active inference components"""
        dimensions = ModelDimensions(
            num_states=self.config.num_states,
            num_observations=self.config.num_observations,
            num_actions=self.config.num_actions,
        )
        self.generative_model = create_generative_model(
            self.config.model_type, dimensions=dimensions
        )
        self.inference = create_inference_algorithm(
            self.config.inference_type,
            InferenceConfig(
                num_iterations=self.config.max_iterations,
                convergence_threshold=self.config.convergence_threshold,
            ),
        )
        self.policy_selector = create_policy_selector(
            "discrete", config=PolicyConfig(), inference_algorithm=self.inference)
        self.planner = create_temporal_planner(
            self.config.planning_type,
            PlanningConfig(
                planning_horizon=self.config.planning_horizon,
                num_simulations=self.config.num_simulations,
            ),
            policy_selector=self.policy_selector,
            inference_algorithm=self.inference,
        )
        self.precision_optimizer = create_precision_optimizer(
            "adaptive",
            PrecisionConfig(init_precision=self.config.initial_precision),
            num_modalities=self.config.num_observations,
        )
        # TODO: Create a proper observation model instead of using generative model
        # observation_model = DirectGraphObservationModel(config=BeliefUpdateConfig())
        # Create belief updater with correct signature
        self.belief_updater = create_belief_updater(
            "standard", BeliefUpdateConfig())

    def update(self, dt: float) -> None:
        """Main update method called each timestep"""
        current_state = self.state_mapper.map_to_belief(
            self.agent, self.state_manager)
        percepts = self.perception_system.perceive(self.agent.agent_id)
        current_observation = self.perception_mapper.map_to_observation(
            percepts)
        if self.current_belief is None:
            self.current_belief = self._initialize_belief(current_state)
        else:
            self.current_belief = self.belief_updater.update_beliefs(
                self.current_belief, current_observation, self.last_action
            )
        # Calculate prediction errors for precision optimization
        if self.last_observation is not None:
            prediction_errors = current_observation - self.last_observation
        else:
            prediction_errors = current_observation  # First timestep
        self.precision_optimizer.optimize(
            prediction_errors, current_observation)
        if self.config.mode == IntegrationMode.FULL:
            action = self._select_action_full()
        elif self.config.mode == IntegrationMode.HYBRID:
            action = self._select_action_hybrid()
        elif self.config.mode == IntegrationMode.ADVISORY:
            action = self._select_action_advisory()
        else:
            action = self._select_action_learning()
        if action and self._should_execute_action(action):
            self._execute_action(action)
        self._update_memory(current_state, current_observation, action)
        self.last_observation = current_observation
        self.last_action = action

    def _initialize_belief(self, state: np.ndarray) -> np.ndarray:
        """Initialize belief from current state"""
        belief: np.ndarray = np.zeros(self.config.num_states)
        state_idx: int = int(np.sum(state[:3]) * 10) % self.config.num_states
        belief[state_idx] = 1.0
        return belief

    def _select_action_full(self) -> Optional[Action]:
        """Select action using full active inference"""
        result = self.planner.plan(self.current_belief, self.generative_model)
        # Extract policy from tuple (Policy, float) returned by planner
        if isinstance(result, tuple) and len(result) >= 2:
            trajectory, _ = result  # Extract policy and ignore expected value
        else:
            trajectory = result  # Fallback for direct Policy return
        if (
            trajectory
            and hasattr(trajectory, "actions")
            and hasattr(trajectory.actions, "__len__")
            and len(trajectory.actions) > 0
        ):
            action_idx = int(trajectory.actions[0])  # Ensure int type
            return self.action_mapper.map_to_agent_action(
                action_idx, self.agent)
        return None

    def _select_action_hybrid(self) -> Optional[Action]:
        """Select action using hybrid approach"""
        basic_action = self.decision_system.make_decision(self.agent.agent_id)
        ai_action = self._select_action_full()
        if ai_action and basic_action:
            ai_confidence = self._calculate_action_confidence(ai_action)
            if ai_confidence > self.config.action_selection_threshold:
                return ai_action
            else:
                return basic_action
        return basic_action or ai_action

    def _select_action_advisory(self) -> Optional[Action]:
        """Select action with active inference as advisor"""
        basic_action = self.decision_system.make_decision(self.agent.agent_id)
        ai_action = self._select_action_full()
        if ai_action:
            logger.debug(f"Active Inference suggests: {ai_action.action_type}")
        return basic_action

    def _select_action_learning(self) -> Optional[Action]:
        """Select action focused on learning"""
        info_gain = self._calculate_expected_info_gain()
        if info_gain > 0.5:
            return Action(
                action_type=ActionType.EXPLORE,
                parameters={"focus": "novel"},
                utility=info_gain,
                cost=0.1,
            )
        return self._select_action_full()

    def _calculate_action_confidence(self, action: Action) -> float:
        """Calculate confidence in selected action"""
        # Convert PyTorch tensor to NumPy safely if needed
        if hasattr(self.current_belief, "detach"):
            belief_array = self.current_belief.detach().numpy()
        else:
            belief_array = np.array(self.current_belief)
        belief_entropy = -np.sum(belief_array * np.log(belief_array + 1e-10))
        normalized_entropy = belief_entropy / np.log(self.config.num_states)
        confidence = 1.0 - normalized_entropy
        return confidence

    def _calculate_expected_info_gain(self) -> float:
        """Calculate expected information gain from exploration"""
        # Convert PyTorch tensor to NumPy safely if needed
        if hasattr(self.current_belief, "detach"):
            belief_array = self.current_belief.detach().numpy()
        else:
            belief_array = np.array(self.current_belief)
        belief_entropy = -np.sum(belief_array * np.log(belief_array + 1e-10))
        max_entropy = np.log(self.config.num_states)
        return belief_entropy / max_entropy

    def _should_execute_action(self, action: Action) -> bool:
        """Determine if action should be executed"""
        confidence = self._calculate_action_confidence(action)
        if action.action_type == ActionType.MOVE:
            if self.agent.resources.energy < 10:
                return False
        return confidence > self.config.action_selection_threshold

    def _execute_action(self, action: Action):
        """Execute the selected action"""
        logger.debug(f"Executing action: {action.action_type}")
        self.decision_system.execute_action(self.agent.agent_id, action)

    def _update_memory(
            self,
            state: np.ndarray,
            observation: np.ndarray,
            action: Optional[Action]):
        """Update memory with experience"""
        # Convert PyTorch tensor to NumPy safely if needed
        if hasattr(self.current_belief, "detach"):
            belief_array = self.current_belief.detach().numpy()
        else:
            belief_array = np.array(self.current_belief)

        memory = Memory(
            memory_id=f"{self.agent.agent_id}_{uuid.uuid4().hex[:8]}",
            memory_type=MemoryType.EPISODIC,
            content={
                "state": state.tolist(),
                "observation": observation.tolist(),
                "action": action.action_type.value if action else "none",  # Handle None action
                "belief_entropy": -np.sum(belief_array * np.log(belief_array + 1e-10)),
            },
            timestamp=datetime.now(),
            importance=0.5,
        )
        self.memory_system.store_memory(
            content=memory.content,
            memory_type=memory.memory_type,
            importance=memory.importance,
            context={
                "agent_id": self.agent.agent_id,
                "integration_mode": self.config.mode.value,
            },
        )

    def get_visualization_data(self) -> Dict[str, Any]:
        """Get data for visualization"""
        # Convert PyTorch tensor to NumPy safely if needed for belief state
        belief_state_list = None
        belief_entropy_val = None
        if self.current_belief is not None:
            if hasattr(self.current_belief, "detach"):
                belief_array = self.current_belief.detach().numpy()
            else:
                belief_array = np.array(self.current_belief)
            belief_state_list = belief_array.tolist()
            belief_entropy_val = - \
                np.sum(belief_array * np.log(belief_array + 1e-10))
        return {
            "belief_state": belief_state_list,
            "last_observation": (
                self.last_observation.tolist() if self.last_observation is not None else None),
            "last_action": (
                self.last_action.action_type.value if self.last_action else None),
            "mode": self.config.mode.value,
            "belief_entropy": belief_entropy_val,
        }


def create_active_inference_agent(
    agent: Agent,
    state_manager: AgentStateManager,
    perception_system: PerceptionSystem,
    decision_system: DecisionSystem,
    movement_controller: MovementController,
    memory_system: MemorySystem,
    config: Optional[ActiveInferenceConfig] = None,
) -> ActiveInferenceIntegration:
    """Factory function to create an active inference integrated agent"""
    return ActiveInferenceIntegration(
        agent=agent,
        state_manager=state_manager,
        perception_system=perception_system,
        decision_system=decision_system,
        movement_controller=movement_controller,
        memory_system=memory_system,
        config=config,
    )
