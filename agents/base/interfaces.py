"""
Agent Interfaces for FreeAgentics.

This module defines the core interfaces and abstract base classes for agent
components, following the clean architecture principles and ADR-003
dependency rules.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Protocol, Tuple, TypeVar

import numpy as np
from numpy.typing import NDArray

from .data_model import Agent, AgentStatus, Position

# Forward declaration to avoid circular imports
if TYPE_CHECKING:
    from .decision_making import Action
    from .markov_blanket import (
        BoundaryMetrics,
        BoundaryState,
        BoundaryViolationEvent,
        MarkovBlanketDimensions,
    )
else:
    # For runtime, import Action to avoid NameError
    from .decision_making import Action

# Generic type for agent implementations
TAgent = TypeVar("TAgent", bound=Agent)


class IAgentComponent(Protocol):
    """Protocol defining the interface for all agent components"""

    def initialize(self, agent: Agent) -> None:
        """Initialize the component with an agent instance"""
        ...

    def cleanup(self) -> None:
        """Clean up resources when component is destroyed"""
        ...


class IAgentLifecycle(ABC):
    """Abstract interface for agent lifecycle management"""

    @abstractmethod
    def start(self) -> None:
        """Start the agent and initialize all components"""

    @abstractmethod
    def stop(self) -> None:
        """Stop the agent and cleanup resources"""

    @abstractmethod
    def pause(self) -> None:
        """Pause agent execution"""

    @abstractmethod
    def resume(self) -> None:
        """Resume agent execution"""

    @abstractmethod
    def restart(self) -> None:
        """Restart the agent (stop and start)"""


class IAgentBehavior(ABC):
    """Abstract interface for agent behaviors"""

    @abstractmethod
    def can_execute(self, agent: Agent, context: Dict[str, Any]) -> bool:
        """Check if this behavior can be executed in the current context"""

    @abstractmethod
    def execute(self, agent: Agent, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the behavior and return results"""

    @abstractmethod
    def get_priority(self, agent: Agent, context: Dict[str, Any]) -> float:
        """Get the priority of this behavior in the current context"""


class IBehaviorTree(ABC):
    """Abstract interface for behavior tree systems"""

    @abstractmethod
    def add_behavior(self, behavior: IAgentBehavior) -> None:
        """Add a behavior to the tree"""

    @abstractmethod
    def remove_behavior(self, behavior: IAgentBehavior) -> None:
        """Remove a behavior from the tree"""

    @abstractmethod
    def evaluate(self, agent: Agent, context: Dict[str, Any]) -> Optional[IAgentBehavior]:
        """Evaluate the tree and return the best behavior to execute"""


class IAgentFactory(ABC):
    """Abstract interface for agent factories"""

    @abstractmethod
    def create_agent(self, agent_type: str, **kwargs: Any) -> Agent:
        """Create an agent of the specified type"""

    @abstractmethod
    def get_supported_types(self) -> List[str]:
        """Get list of supported agent types"""

    @abstractmethod
    def register_type(self, agent_type: str, factory_func: Any) -> None:
        """Register a new agent type with its factory function"""


class IAgentRegistry(ABC):
    """Abstract interface for agent registries"""

    @abstractmethod
    def register_agent(self, agent: Agent) -> None:
        """Register an agent in the registry"""

    @abstractmethod
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the registry"""

    @abstractmethod
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID"""

    @abstractmethod
    def get_all_agents(self) -> List[Agent]:
        """Get all registered agents"""

    @abstractmethod
    def find_agents_by_type(self, agent_type: str) -> List[Agent]:
        """Find agents by type"""

    @abstractmethod
    def find_agents_in_range(self, position: Position, radius: float) -> List[Agent]:
        """Find agents within a specified range of a position"""


class ICommunicationProtocol(ABC):
    """Abstract interface for agent communication protocols"""

    @abstractmethod
    def send_message(self, from_agent: str, to_agent: str, message: Dict[str, Any]) -> bool:
        """Send a message from one agent to another"""

    @abstractmethod
    def broadcast_message(
        self, from_agent: str, message: Dict[str, Any], radius: Optional[float] = None
    ) -> List[str]:
        """Broadcast a message to nearby agents"""

    @abstractmethod
    def receive_messages(self, agent_id: str) -> List[Dict[str, Any]]:
        """Retrieve pending messages for an agent"""


class IWorldInterface(ABC):
    """Abstract interface for world interaction"""

    @abstractmethod
    def get_world_state(self) -> Dict[str, Any]:
        """Get current world state"""

    @abstractmethod
    def can_move_to(self, agent: Agent, position: Position) -> bool:
        """Check if agent can move to specified position"""

    @abstractmethod
    def move_agent(self, agent: Agent, new_position: Position) -> bool:
        """Move agent to new position"""

    @abstractmethod
    def get_nearby_objects(self, position: Position, radius: float) -> List[Dict[str, Any]]:
        """Get objects near a position"""

    @abstractmethod
    def perform_action(self, agent: Agent, action: Action) -> Dict[str, Any]:
        """Perform an action in the world"""


class IActiveInferenceInterface(ABC):
    """Abstract interface for Active Inference integration"""

    @abstractmethod
    def update_beliefs(self, agent: Agent, observations: NDArray[np.float64]) -> None:
        """Update agent beliefs based on observations"""

    @abstractmethod
    def select_action(self, agent: Agent) -> Action:
        """Select best action using Active Inference"""

    @abstractmethod
    def compute_free_energy(self, agent: Agent) -> float:
        """Compute variational free energy for agent"""

    @abstractmethod
    def get_belief_state(self, agent: Agent) -> NDArray[np.float64]:
        """Get current belief state of agent"""


class IAgentEventHandler(ABC):
    """Abstract interface for handling agent events"""

    @abstractmethod
    def on_agent_created(self, agent: Agent) -> None:
        """Handle agent creation event"""

    @abstractmethod
    def on_agent_destroyed(self, agent: Agent) -> None:
        """Handle agent destruction event"""

    @abstractmethod
    def on_agent_moved(self, agent: Agent, old_position: Position, new_position: Position) -> None:
        """Called when an agent moves"""

    @abstractmethod
    def on_agent_status_changed(
        self, agent: Agent, old_status: AgentStatus, new_status: AgentStatus
    ) -> None:
        """Called when an agent's status changes."""


class IAgentPlugin(ABC):
    """Abstract interface for agent plugins"""

    @abstractmethod
    def get_name(self) -> str:
        """Get plugin name"""

    @abstractmethod
    def get_version(self) -> str:
        """Get plugin version"""

    @abstractmethod
    def initialize(self, agent: Agent) -> None:
        """Initialize plugin for agent"""

    @abstractmethod
    def update(self, agent: Agent, delta_time: float) -> None:
        """Update plugin logic."""

    @abstractmethod
    def cleanup(self, agent: Agent) -> None:
        """Cleanup plugin resources"""


class IConfigurationProvider(ABC):
    """Abstract interface for configuration providers"""

    @abstractmethod
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""

    @abstractmethod
    def set_config(self, key: str, value: Any) -> None:
        """Set configuration value"""

    @abstractmethod
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration values."""

    @abstractmethod
    def reload_config(self) -> None:
        """Reload configuration from source"""


# Protocol for extensible agent types
class IAgentExtension(Protocol):
    """Protocol for agent type extensions"""

    def get_extension_name(self) -> str:
        """Get the name of this extension"""
        ...

    def extend_agent(self, agent: Agent) -> None:
        """Extend an agent with additional capabilities"""
        ...

    def validate_agent(self, agent: Agent) -> bool:
        """Validate that an agent meets extension requirements"""
        ...


# Logging interface
class IAgentLogger(ABC):
    """Abstract interface for agent logging."""

    @abstractmethod
    def log_debug(self, agent_id: str, message: str, **kwargs: Any) -> None:
        """Log debug message"""

    @abstractmethod
    def log_info(self, agent_id: str, message: str, **kwargs: Any) -> None:
        """Log info message"""

    @abstractmethod
    def log_warning(self, agent_id: str, message: str, **kwargs: Any) -> None:
        """Log warning message."""

    @abstractmethod
    def log_error(self, agent_id: str, message: str, **kwargs: Any) -> None:
        """Log error message"""


class IMarkovBlanketInterface(ABC):
    """Abstract interface for Markov blanket boundary management"""

    @abstractmethod
    def get_dimensions(self) -> "MarkovBlanketDimensions":
        """Get the current dimensions of the Markov blanket"""

    @abstractmethod
    def update_states(self, agent_state: Any, environment_state: np.ndarray) -> None:
        """Update the internal and boundary states of the Markov blanket"""

    @abstractmethod
    def verify_independence(self) -> Tuple[float, Dict[str, Any]]:
        """Verify statistical independence of the Markov blanket boundary"""

    @abstractmethod
    def detect_violations(self) -> List["BoundaryViolationEvent"]:
        """Detect any boundary violations"""

    @abstractmethod
    def get_metrics(self) -> "BoundaryMetrics":
        """Get current boundary integrity metrics"""

    @abstractmethod
    def get_boundary_state(self) -> "BoundaryState":
        """Get the current state of the boundary"""

    @abstractmethod
    def set_violation_handler(self, handler: Callable[["BoundaryViolationEvent"], None]) -> None:
        """Set a handler for boundary violation events"""
