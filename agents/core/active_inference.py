"""
Active Inference Agent Implementation

This module implements the core Active Inference agent that uses GNN models
for cognitive processing and follows Active Inference principles of minimizing
free energy through perception and action.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from inference.gnn.executor import GMNExecutor
from knowledge.knowledge_graph import KnowledgeGraph as AgentKnowledgeGraph

from ..base.data_model import AgentStatus, Position
from ..base.interaction import MessageType

logger = logging.getLogger(__name__)


@dataclass
class Belief:
    """Represents an agent's belief about the world state"""

    state: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Observation:
    """Represents a sensory observation"""

    type: str
    data: Dict[str, Any]
    position: Optional[Position] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Action:
    """Represents an action the agent can take"""

    type: str
    parameters: Dict[str, Any]
    expected_outcome: Optional[str] = None
    energy_cost: float = 0.0


class ActiveInferenceAgent:
    """
    Active Inference Agent that uses GNN models for cognitive processing.

    Implements the Active Inference framework where agents:
    1. Maintain generative models of their environment
    2. Minimize free energy through perception and action
    3. Update beliefs based on sensory evidence
    4. Select actions that reduce expected surprise
    """

    def __init__(
        self,
        agent_id: str,
        model_name: str,
        gnn_executor: GMNExecutor,
        initial_position: Position,
        initial_energy: float = 100.0,
    ) -> None:
        """Initialize Active Inference agent"""
        self.id = agent_id
        self.model_name = model_name
        self.executor = gnn_executor
        self.position = initial_position
        self.energy = initial_energy
        self.status = AgentStatus.IDLE

        # Knowledge and beliefs
        self.knowledge_graph = AgentKnowledgeGraph(agent_id=agent_id)
        self.beliefs: Dict[str, Belief] = {}
        self.observations: List[Observation] = []

        # Active Inference components
        self.generative_model = self._initialize_generative_model()
        self.free_energy_history: List[float] = []

        # Action selection
        self.available_actions = self._initialize_actions()
        self.action_history: List[Action] = []

        logger.info(f"Initialized Active Inference agent {agent_id} with model {model_name}")

    def _initialize_generative_model(self) -> Dict[str, Any]:
        """Initialize the agent's generative model of the world"""
        return {
            "world_model": {
                "terrain_types": ["plains", "forest", "mountain", "water"],
                "resource_types": ["food", "water", "shelter", "tools"],
                "agent_types": ["explorer", "merchant", "scholar", "guardian"],
            },
            "transition_model": {
                # Probabilistic transitions between states
                "movement": {"success_rate": 0.9, "energy_cost": 1.0},
                "resource_gathering": {"success_rate": 0.7, "energy_gain": 5.0},
                "communication": {"success_rate": 0.8, "trust_gain": 0.1},
            },
            "preference_model": {
                # Agent's preferences (from GNN model)
                "exploration": self.executor.model.metadata.get("exploration_drive", 0.5),
                "social": self.executor.model.metadata.get("social_drive", 0.5),
                "resource": self.executor.model.metadata.get("resource_drive", 0.5),
            },
        }

    def _initialize_actions(self) -> List[str]:
        """Initialize available actions based on agent capabilities"""
        return ["move", "gather", "communicate", "observe", "rest", "share_knowledge"]

    def perceive(self, observation: Observation) -> None:
        """
        Process a new observation and update beliefs.

        This implements the perception part of Active Inference,
        updating beliefs to minimize prediction error.
        """
        self.observations.append(observation)

        # Update beliefs based on observation
        belief_update = self._process_observation(observation)
        for belief_key, belief in belief_update.items():
            self.beliefs[belief_key] = belief

        # Add to knowledge graph
        self.knowledge_graph.add_experience(
            state={"observation": observation.type, "data": observation.data},
            action="perceive",
            outcome={"belief_update": len(belief_update)},
            reward=0.0,  # Perception itself has no reward
        )

        logger.debug(
            f"Agent {
                self.id} perceived {
                observation.type}, updated {
                len(belief_update)} beliefs"
        )

    def _process_observation(self, observation: Observation) -> Dict[str, Belief]:
        """Process observation through GNN to update beliefs"""
        # Prepare observation for GNN processing
        gnn_input = {
            "observation": {
                "type": observation.type,
                "data": observation.data,
                "position": (observation.position.__dict__ if observation.position else None),
            },
            "current_beliefs": {k: v.confidence for k, v in self.beliefs.items()},
            "energy": self.energy,
        }

        # Process through GNN
        gnn_output = self.executor.execute(gnn_input)

        # Extract belief updates
        belief_updates = {}
        if "beliefs" in gnn_output:
            for belief_key, confidence in gnn_output["beliefs"].items():
                belief_updates[belief_key] = Belief(
                    state=belief_key,
                    confidence=confidence,
                    evidence=[observation.type],
                    timestamp=datetime.utcnow(),
                )

        return belief_updates

    def act(self, world_state: Dict[str, Any]) -> Optional[Action]:
        """
        Select and execute an action based on Active Inference principles.

        Actions are selected to minimize expected free energy, balancing:
        1. Epistemic value (information gain)
        2. Pragmatic value (goal achievement)
        """
        if self.energy <= 0:
            logger.warning(f"Agent {self.id} has no energy, cannot act")
            return None

        # Calculate expected free energy for each action
        action_evaluations = []
        for action_type in self.available_actions:
            if self._can_perform_action(action_type, world_state):
                expected_fe = self._calculate_expected_free_energy(action_type, world_state)
                action_evaluations.append((action_type, expected_fe))

        if not action_evaluations:
            logger.debug(f"Agent {self.id} has no available actions")
            return None

        # Select action with minimum expected free energy
        best_action_type = min(action_evaluations, key=lambda x: x[1])[0]

        # Create action with parameters
        action = self._create_action(best_action_type, world_state)

        # Execute action
        self._execute_action(action)

        return action

    def _can_perform_action(self, action_type: str, world_state: Dict[str, Any]) -> bool:
        """Check if an action can be performed given current state"""
        if action_type == "move":
            return self.energy > 1.0
        elif action_type == "gather":
            # Check if resources are nearby
            return "nearby_resources" in world_state and len(world_state["nearby_resources"]) > 0
        elif action_type == "communicate":
            # Check if other agents are nearby
            return "nearby_agents" in world_state and len(world_state["nearby_agents"]) > 0
        elif action_type == "rest":
            return self.energy < 50.0
        else:
            return True

    def _calculate_expected_free_energy(
        self, action_type: str, world_state: Dict[str, Any]
    ) -> float:
        """
        Calculate expected free energy for an action.

        Free energy = Expected surprise + Expected complexity
        Lower values indicate better actions.
        """
        # Prepare input for GNN
        gnn_input = {
            "action": action_type,
            "world_state": world_state,
            "beliefs": {k: v.confidence for k, v in self.beliefs.items()},
            "energy": self.energy,
            "preferences": self.generative_model["preference_model"],
        }

        # Get free energy estimate from GNN
        gnn_output = self.executor.execute(gnn_input)

        # Extract free energy components
        epistemic_value = gnn_output.get("epistemic_value", 0.0)  # Information gain
        pragmatic_value = gnn_output.get("pragmatic_value", 0.0)  # Goal achievement

        # Free energy = -epistemic_value - pragmatic_value
        # (We want to maximize both values, so minimize negative)
        free_energy = -(epistemic_value + pragmatic_value)

        # Add energy cost consideration
        energy_cost = (
            self.generative_model["transition_model"].get(action_type, {}).get("energy_cost", 0.0)
        )

        if self.energy < energy_cost * 2:
            # Penalize actions that would leave us with too little energy
            free_energy += 10.0

        return free_energy

    def _create_action(self, action_type: str, world_state: Dict[str, Any]) -> Action:
        """Create an action with appropriate parameters"""
        parameters = {}

        if action_type == "move":
            # Select movement direction based on exploration drive
            possible_moves = world_state.get("possible_moves", [])
            if possible_moves:
                # Prefer unexplored directions
                parameters["direction"] = possible_moves[0]

        elif action_type == "gather":
            # Select resource to gather
            resources = world_state.get("nearby_resources", [])
            if resources:
                parameters["resource"] = resources[0]

        elif action_type == "communicate":
            # Select agent to communicate with
            agents = world_state.get("nearby_agents", [])
            if agents:
                parameters["target_agent"] = agents[0]
                parameters["message_type"] = MessageType.CASUAL_GREETING

        return Action(
            type=action_type,
            parameters=parameters,
            energy_cost=self.generative_model["transition_model"]
            .get(action_type, {})
            .get("energy_cost", 0.0),
        )

    def _execute_action(self, action: Action) -> None:
        """Execute an action and update internal state"""
        # Deduct energy cost
        self.energy -= action.energy_cost

        # Record action
        self.action_history.append(action)

        # Update status
        if action.type == "move":
            self.status = AgentStatus.MOVING
        elif action.type == "gather":
            self.status = AgentStatus.INTERACTING
        elif action.type == "communicate":
            self.status = AgentStatus.INTERACTING
        elif action.type == "rest":
            self.status = AgentStatus.IDLE
            self.energy = min(100.0, self.energy + 10.0)  # Restore some energy

        logger.debug(
            f"Agent {
                self.id} executed action {
                action.type} with cost {
                action.energy_cost}"
        )

    def update_position(self, new_position: Position) -> None:
        """Update agent's position"""
        self.position = new_position

    def calculate_free_energy(self) -> float:
        """
        Calculate current free energy based on prediction error.

        This is the core Active Inference calculation.
        """
        if not self.observations:
            return 0.0

        # Get recent observations
        recent_obs = self.observations[-10:]

        # Calculate prediction error
        prediction_error = 0.0
        for obs in recent_obs:
            # Compare observation with beliefs
            for belief_key, belief in self.beliefs.items():
                if belief_key in str(obs.data):
                    # Simple error: difference between belief confidence and observation
                    # presence
                    error = abs(belief.confidence - 1.0)
                    prediction_error += error

        # Calculate complexity (KL divergence proxy)
        complexity = len(self.beliefs) * 0.1

        # Free energy = prediction error + complexity
        free_energy = prediction_error + complexity

        self.free_energy_history.append(free_energy)

        return free_energy

    def share_knowledge(self, other_agent: "ActiveInferenceAgent") -> Dict[str, Any]:
        """Share knowledge with another agent"""
        # Extract high-confidence beliefs
        shared_beliefs = {k: v for k, v in self.beliefs.items() if v.confidence > 0.8}

        # Extract recent positive experiences
        recent_experiences = self.knowledge_graph.get_similar_experiences(
            {"outcome": "positive"}, k=5
        )

        return {
            "agent_id": self.id,
            "model_name": self.model_name,
            "beliefs": shared_beliefs,
            "experiences": recent_experiences,
            "free_energy": self.calculate_free_energy(),
        }

    def integrate_knowledge(self, knowledge_package: Dict[str, Any]) -> None:
        """Integrate knowledge received from another agent"""
        source_agent = knowledge_package.get("agent_id", "unknown")

        # Integrate beliefs with trust-based weighting
        trust_factor = 0.5  # Could be dynamic based on past interactions

        for belief_key, other_belief in knowledge_package.get("beliefs", {}).items():
            if belief_key in self.beliefs:
                # Weighted average of confidences
                my_conf = self.beliefs[belief_key].confidence
                other_conf = (
                    other_belief.confidence if hasattr(other_belief, "confidence") else other_belief
                )
                new_conf = (my_conf + trust_factor * other_conf) / (1 + trust_factor)
                self.beliefs[belief_key].confidence = new_conf
            else:
                # Adopt new belief with reduced confidence
                self.beliefs[belief_key] = Belief(
                    state=belief_key,
                    confidence=trust_factor
                    * (
                        other_belief.confidence
                        if hasattr(other_belief, "confidence")
                        else other_belief
                    ),
                    evidence=[f"shared_by_{source_agent}"],
                    timestamp=datetime.utcnow(),
                )

        logger.info(
            f"Agent {
                self.id} integrated knowledge from {source_agent}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent state to dictionary for serialization"""
        return {
            "id": self.id,
            "model_name": self.model_name,
            "position": self.position.__dict__ if self.position else None,
            "energy": self.energy,
            "status": (self.status.value if hasattr(self.status, "value") else str(self.status)),
            "beliefs": {k: v.confidence for k, v in self.beliefs.items()},
            "free_energy": self.calculate_free_energy(),
            "action_count": len(self.action_history),
            "observation_count": len(self.observations),
        }
