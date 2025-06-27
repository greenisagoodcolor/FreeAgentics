"""
Module for FreeAgentics Active Inference implementation.
"""

import math
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import numpy as np

from .data_model import Agent, Position
from .state_manager import AgentStateManager

"""
Agent Perception System
This module provides sensory perception capabilities for agents including:
- Visual perception (field of view, line of sight)
- Auditory perception (sound detection and localization)
- Proximity sensing
- Environmental awareness
- Perception filtering and memory
"""


class PerceptionType(Enum):
    """Types of perception available to agents"""

    VISUAL = "visual"
    AUDITORY = "auditory"
    TACTILE = "tactile"
    OLFACTORY = "olfactory"
    PROXIMITY = "proximity"
    ENVIRONMENTAL = "environmental"


class StimulusType(Enum):
    """Types of stimuli that can be perceived"""

    AGENT = "agent"
    OBJECT = "object"
    SOUND = "sound"
    SMELL = "smell"
    TEMPERATURE = "temperature"
    LIGHT = "light"
    MOVEMENT = "movement"
    DANGER = "danger"


@dataclass
class Stimulus:
    """Represents a perceivable stimulus in the environment"""

    stimulus_id: str
    stimulus_type: StimulusType
    position: Position
    intensity: float = 1.0
    radius: float = 0.0
    source: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def decay_over_distance(self, distance: float) -> float:
        """Calculate intensity decay over distance"""
        if self.radius <= 0:
            return self.intensity
        if distance <= self.radius:
            return self.intensity * (1.0 - distance / self.radius)
        return 0.0


@dataclass
class Percept:
    """Represents a perceived stimulus with additional context"""

    stimulus: Stimulus
    perception_type: PerceptionType
    confidence: float = 1.0
    distance: float = 0.0
    direction: Optional[np.ndarray] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerceptionCapabilities:
    """Defines an agent's perception capabilities"""

    visual_range: float = 50.0
    field_of_view: float = math.pi * 2 / 3
    visual_acuity: float = 1.0
    hearing_range: float = 100.0
    hearing_sensitivity: float = 1.0
    proximity_range: float = 5.0
    smell_range: float = 20.0
    reaction_time: float = 0.2
    attention_capacity: int = 10
    enabled_types: Set[PerceptionType] = field(
        default_factory=lambda: {
            PerceptionType.VISUAL,
            PerceptionType.AUDITORY,
            PerceptionType.PROXIMITY,
        }
    )


class PerceptionFilter:
    """Base class for perception filters"""

    def filter(self, percepts: List[Percept], agent: Agent) -> List[Percept]:
        """Filter percepts based on criteria"""
        raise NotImplementedError


class ImportanceFilter(PerceptionFilter):
    """Filter percepts by importance/relevance"""

    def __init__(self, importance_threshold: float = 0.3) -> None:
        self.importance_threshold = importance_threshold

    def filter(self, percepts: List[Percept], agent: Agent) -> List[Percept]:
        """Keep only important percepts"""
        filtered = []
        for percept in percepts:
            importance = self._calculate_importance(percept, agent)
            if importance >= self.importance_threshold:
                percept.metadata["importance"] = importance
                filtered.append(percept)
        return filtered

    def _calculate_importance(self, percept: Percept, agent: Agent) -> float:
        """Calculate importance of a percept"""
        importance = percept.confidence
        if percept.distance < 10.0:
            importance *= 1.5
        if percept.stimulus.stimulus_type == StimulusType.DANGER:
            importance = 1.0
        if percept.stimulus.stimulus_type == StimulusType.AGENT:
            importance *= 1.2
        return min(importance, 1.0)


class AttentionFilter(PerceptionFilter):
    """Filter percepts based on attention capacity"""

    def filter(self, percepts: List[Percept], agent: Agent) -> List[Percept]:
        """Keep only the most salient percepts within attention capacity"""
        if not hasattr(agent, "perception_capabilities"):
            return percepts[:10]
        capacity = agent.perception_capabilities.attention_capacity
        sorted_percepts = sorted(
            percepts,
            key=lambda p: p.confidence * p.metadata.get("importance", 1.0) / (p.distance + 1),
            reverse=True,
        )
        return sorted_percepts[:capacity]


class PerceptionMemory:
    """Manages short-term perception memory"""

    def __init__(self, memory_duration: float = 5.0, max_memories: int = 100) -> None:
        self.memory_duration = memory_duration
        self.max_memories = max_memories
        self.memories: deque = deque(maxlen=max_memories)
        self.stimulus_history: Dict[str, List[Percept]] = defaultdict(list)

    def add_percept(self, percept: Percept) -> None:
        """Add a percept to memory"""
        self.memories.append(percept)
        self.stimulus_history[percept.stimulus.stimulus_id].append(percept)

    def get_recent_percepts(self, time_window: Optional[float] = None) -> List[Percept]:
        """Get percepts within time window"""
        if time_window is None:
            time_window = self.memory_duration
        cutoff_time = datetime.now() - timedelta(seconds=time_window)
        recent = [p for p in self.memories if p.timestamp >= cutoff_time]
        return recent

    def get_stimulus_history(self, stimulus_id: str) -> List[Percept]:
        """Get perception history for a specific stimulus"""
        return self.stimulus_history.get(stimulus_id, [])

    def forget_old_memories(self) -> None:
        """Remove memories older than memory_duration"""
        cutoff_time = datetime.now() - timedelta(seconds=self.memory_duration)
        while self.memories and self.memories[0].timestamp < cutoff_time:
            old_percept = self.memories.popleft()
            stimulus_id = old_percept.stimulus.stimulus_id
            if stimulus_id in self.stimulus_history:
                self.stimulus_history[stimulus_id] = [
                    p for p in self.stimulus_history[stimulus_id] if p.timestamp >= cutoff_time
                ]
                if not self.stimulus_history[stimulus_id]:
                    del self.stimulus_history[stimulus_id]


class SensorSystem:
    """Base class for sensor systems"""

    def __init__(self, perception_type: PerceptionType) -> None:
        self.perception_type = perception_type

    def sense(
        self, agent: Agent, stimuli: List[Stimulus], environment: Optional[Any] = None
    ) -> List[Percept]:
        """Sense stimuli from the environment"""
        raise NotImplementedError


class VisualSensor(SensorSystem):
    """Visual perception system"""

    def __init__(self) -> None:
        super().__init__(PerceptionType.VISUAL)

    def sense(
        self, agent: Agent, stimuli: List[Stimulus], environment: Optional[Any] = None
    ) -> List[Percept]:
        """Detect visible stimuli"""
        percepts = []
        capabilities = getattr(agent, "perception_capabilities", PerceptionCapabilities())
        agent_pos = agent.position.to_array()
        roll, pitch, yaw = agent.orientation.to_euler()
        forward = np.array([np.cos(yaw), np.sin(yaw), 0])
        for stimulus in stimuli:
            if (
                stimulus.source is not None
                and hasattr(stimulus.source, "agent_id")
                and stimulus.source.agent_id == agent.agent_id
            ):
                continue
            stimulus_pos = stimulus.position.to_array()
            direction = stimulus_pos - agent_pos
            distance = float(np.linalg.norm(direction))
            if distance > capabilities.visual_range:
                continue
            if distance > 0:
                direction = direction / distance
            angle = np.arccos(np.clip(np.dot(forward, direction), -1.0, 1.0))
            if angle > capabilities.field_of_view / 2:
                continue
            if environment and hasattr(environment, "check_line_of_sight"):
                if not environment.check_line_of_sight(agent.position, stimulus.position):
                    continue
            confidence = capabilities.visual_acuity
            if distance > 0:
                confidence *= 1.0 - distance / capabilities.visual_range
            percept = Percept(
                stimulus=stimulus,
                perception_type=self.perception_type,
                confidence=confidence,
                distance=float(distance),
                direction=direction,
                metadata={
                    "angle_from_forward": angle,
                    "in_peripheral": angle > capabilities.field_of_view / 4,
                },
            )
            percepts.append(percept)
        return percepts


class AuditorySensor(SensorSystem):
    """Auditory perception system"""

    def __init__(self) -> None:
        super().__init__(PerceptionType.AUDITORY)

    def sense(
        self, agent: Agent, stimuli: List[Stimulus], environment: Optional[Any] = None
    ) -> List[Percept]:
        """Detect audible stimuli"""
        percepts = []
        capabilities = getattr(agent, "perception_capabilities", PerceptionCapabilities())
        agent_pos = agent.position.to_array()
        for stimulus in stimuli:
            if stimulus.stimulus_type != StimulusType.SOUND:
                continue
            stimulus_pos = stimulus.position.to_array()
            direction = stimulus_pos - agent_pos
            distance = float(np.linalg.norm(direction))
            perceived_intensity = stimulus.decay_over_distance(distance)
            if perceived_intensity < 0.1 / capabilities.hearing_sensitivity:
                continue
            if distance > capabilities.hearing_range:
                continue
            if distance > 0:
                direction = direction / distance
            confidence = perceived_intensity * capabilities.hearing_sensitivity
            percept = Percept(
                stimulus=stimulus,
                perception_type=self.perception_type,
                confidence=confidence,
                distance=float(distance),
                direction=direction,
                metadata={
                    "perceived_intensity": perceived_intensity,
                    "can_localize": confidence > 0.5,
                },
            )
            percepts.append(percept)
        return percepts


class ProximitySensor(SensorSystem):
    """Proximity/touch perception system"""

    def __init__(self) -> None:
        super().__init__(PerceptionType.PROXIMITY)

    def sense(
        self, agent: Agent, stimuli: List[Stimulus], environment: Optional[Any] = None
    ) -> List[Percept]:
        """Detect nearby stimuli"""
        percepts = []
        capabilities = getattr(agent, "perception_capabilities", PerceptionCapabilities())
        agent_pos = agent.position.to_array()
        for stimulus in stimuli:
            if (
                stimulus.source is not None
                and hasattr(stimulus.source, "agent_id")
                and stimulus.source.agent_id == agent.agent_id
            ):
                continue
            stimulus_pos = stimulus.position.to_array()
            direction = stimulus_pos - agent_pos
            distance = float(np.linalg.norm(direction))
            if distance > capabilities.proximity_range:
                continue
            if distance > 0:
                direction = direction / distance
            confidence = 1.0 - distance / capabilities.proximity_range
            percept = Percept(
                stimulus=stimulus,
                perception_type=self.perception_type,
                confidence=float(confidence),
                distance=float(distance),
                direction=direction,
                metadata={"is_touching": distance < 0.5},
            )
            percepts.append(percept)
        return percepts


class PerceptionSystem:
    """Main perception system managing all sensors"""

    def __init__(self, state_manager: AgentStateManager) -> None:
        self.state_manager = state_manager
        self.sensors: Dict[PerceptionType, SensorSystem] = {
            PerceptionType.VISUAL: VisualSensor(),
            PerceptionType.AUDITORY: AuditorySensor(),
            PerceptionType.PROXIMITY: ProximitySensor(),
        }
        self.filters: List[PerceptionFilter] = [ImportanceFilter(), AttentionFilter()]
        self.perception_memories: Dict[str, PerceptionMemory] = {}
        self.perception_capabilities: Dict[str, PerceptionCapabilities] = {}
        self.stimuli: List[Stimulus] = []
        self.stimulus_sources: Dict[str, Any] = {}

    def register_agent(
        self, agent: Agent, capabilities: Optional[PerceptionCapabilities] = None
    ) -> None:
        """Register an agent with the perception system"""
        if capabilities is None:
            capabilities = PerceptionCapabilities()
        self.perception_capabilities[agent.agent_id] = capabilities
        self.perception_memories[agent.agent_id] = PerceptionMemory()
        # Note: Agent capabilities stored in system, not on agent instance

    def add_stimulus(self, stimulus: Stimulus) -> None:
        """Add a stimulus to the environment"""
        self.stimuli.append(stimulus)
        if stimulus.source:
            self.stimulus_sources[stimulus.stimulus_id] = stimulus.source

    def remove_stimulus(self, stimulus_id: str) -> None:
        """Remove a stimulus from the environment"""
        self.stimuli = [s for s in self.stimuli if s.stimulus_id != stimulus_id]
        self.stimulus_sources.pop(stimulus_id, None)

    def update_stimulus(self, stimulus_id: str, **kwargs) -> None:
        """Update stimulus properties"""
        for stimulus in self.stimuli:
            if stimulus.stimulus_id == stimulus_id:
                for key, value in kwargs.items():
                    if hasattr(stimulus, key):
                        setattr(stimulus, key, value)
                break

    def perceive(self, agent_id: str, environment: Optional[Any] = None) -> List[Percept]:
        """Process perception for an agent"""
        agent = self.state_manager.get_agent(agent_id)
        if not agent:
            return []
        capabilities = self.perception_capabilities.get(agent_id)
        if not capabilities:
            return []
        all_percepts = []
        for perception_type in capabilities.enabled_types:
            if perception_type in self.sensors:
                sensor = self.sensors[perception_type]
                percepts = sensor.sense(agent, self.stimuli, environment)
                all_percepts.extend(percepts)
        filtered_percepts = all_percepts
        for filter in self.filters:
            filtered_percepts = filter.filter(filtered_percepts, agent)
        memory = self.perception_memories.get(agent_id)
        if memory:
            for percept in filtered_percepts:
                memory.add_percept(percept)
            memory.forget_old_memories()
        return filtered_percepts

    def get_perception_memory(self, agent_id: str) -> Optional[PerceptionMemory]:
        """Get perception memory for an agent"""
        return self.perception_memories.get(agent_id)

    def create_agent_stimulus(self, agent: Agent) -> Stimulus:
        """Create a stimulus representing an agent"""
        return Stimulus(
            stimulus_id=f"agent_{agent.agent_id}",
            stimulus_type=StimulusType.AGENT,
            position=agent.position,
            intensity=1.0,
            source=agent,
            metadata={"agent_name": agent.name, "agent_status": agent.status.value},
        )

    def create_sound_stimulus(
        self,
        position: Position,
        intensity: float,
        radius: float,
        source: Optional[Any] = None,
    ) -> Stimulus:
        """Create a sound stimulus"""
        return Stimulus(
            stimulus_id=f"sound_{uuid.uuid4().hex[:8]}",
            stimulus_type=StimulusType.SOUND,
            position=position,
            intensity=intensity,
            radius=radius,
            source=source,
            metadata={"sound_type": "generic"},
        )

    def update_agent_positions(self) -> None:
        """Update positions of agent stimuli"""
        for agent_id in self.perception_capabilities:
            agent = self.state_manager.get_agent(agent_id)
            if agent:
                stimulus_id = f"agent_{agent_id}"
                self.update_stimulus(stimulus_id, position=agent.position)
                if not any((s.stimulus_id == stimulus_id for s in self.stimuli)):
                    self.add_stimulus(self.create_agent_stimulus(agent))

    def get_perceived_agents(self, agent_id: str) -> List[Agent]:
        """Get all agents currently perceived by an agent"""
        percepts = self.perceive(agent_id)
        perceived_agents = []
        for percept in percepts:
            if percept.stimulus.stimulus_type == StimulusType.AGENT:
                source_agent = percept.stimulus.source
                if isinstance(source_agent, Agent):
                    perceived_agents.append(source_agent)
        return perceived_agents

    def can_perceive(
        self,
        observer_id: str,
        target_id: str,
        perception_type: PerceptionType = PerceptionType.VISUAL,
    ) -> bool:
        """Check if one agent can perceive another"""
        observer = self.state_manager.get_agent(observer_id)
        target = self.state_manager.get_agent(target_id)
        if not observer or not target:
            return False
        target_stimulus = self.create_agent_stimulus(target)
        if perception_type in self.sensors:
            sensor = self.sensors[perception_type]
            percepts = sensor.sense(observer, [target_stimulus])
            return len(percepts) > 0
        return False
