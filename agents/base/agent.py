"""
Main Agent Class for FreeAgentics
This module provides the primary Agent class that orchestrates all agent
components and implements the agent lifecycle, following ADR-002, ADR-003,
and ADR-004.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from asyncio import Task
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .active_inference_integration import ActiveInferenceIntegration
from .behaviors import BehaviorTreeManager
from .data_model import Agent as AgentData
from .data_model import AgentStatus, Position
from .decision_making import Action, DecisionSystem
from .interaction import InteractionSystem
from .interfaces import (
    IActiveInferenceInterface,
    IAgentBehavior,
    IAgentEventHandler,
    IAgentLifecycle,
    IAgentLogger,
    IAgentPlugin,
    IConfigurationProvider,
    IMarkovBlanketInterface,
    IWorldInterface,
)
from .markov_blanket import AgentState, BoundaryViolationEvent, MarkovBlanketFactory
from .memory import MemorySystem
from .movement import CollisionSystem, MovementController, PathfindingGrid
from .perception import PerceptionSystem
from .state_manager import AgentStateManager


class AgentLogger(IAgentLogger):
    """Default logger implementation for agents"""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"agent.{agent_id}")

    def log_debug(self, agent_id: str, message: str, **kwargs) -> None:
        self.logger.debug(f"[{agent_id}] {message}", extra=kwargs)

    def log_info(self, agent_id: str, message: str, **kwargs) -> None:
        self.logger.info(f"[{agent_id}] {message}", extra=kwargs)

    def log_warning(self, agent_id: str, message: str, **kwargs) -> None:
        self.logger.warning(f"[{agent_id}] {message}", extra=kwargs)

    def log_error(self, agent_id: str, message: str, **kwargs) -> None:
        self.logger.error(f"[{agent_id}] {message}", extra=kwargs)


class BaseAgent(IAgentLifecycle):
    """
    Main Agent class that orchestrates all agent components and provides
    a unified interface for agent behavior and lifecycle management.
    This class follows the composition pattern to integrate various agent
    subsystems while maintaining separation of concerns.
    """

    def __init__(
        self,
        agent_data: AgentData | None = None,
        world_interface: IWorldInterface | None = None,
        active_inference_interface: IActiveInferenceInterface | None = None,
        markov_blanket_interface: IMarkovBlanketInterface | None = None,
        config_provider: IConfigurationProvider | None = None,
        logger: IAgentLogger | None = None,
        # Backward compatibility parameters
        agent_id: str | None = None,
        name: str | None = None,
        agent_class: Any | None = None,
        agent_type: str | None = None,
        initial_position: tuple[float, float] | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the agent with its data model and optional interfaces.
        Supports both new interface (AgentData) and old interface
        (individual parameters).
        Args:
            agent_data: The agent's data model (new interface)
            world_interface: Interface for world interaction
            active_inference_interface: Interface for Active Inference
                integration
            markov_blanket_interface: Interface for Markov blanket boundary
                management
            config_provider: Configuration provider
            logger: Logger instance
            # Backward compatibility parameters:
            agent_id: Unique agent identifier (old interface)
            name: Agent name (old interface)
            agent_class: Agent class enum (old interface)
            agent_type: Agent type string (old interface)
            initial_position: Tuple of (x, y) coordinates (old interface)
            **kwargs: Additional parameters
        """
        # Handle backward compatibility - create AgentData if individual
        # params provided
        if agent_data is None and (
            agent_id is not None or name is not None or agent_class is not None
        ):
            # Convert agent_class to agent_type if provided
            if agent_class is not None and hasattr(agent_class, "value"):
                agent_type = agent_class.value
            elif agent_type is None:
                agent_type = "basic"
            # Convert initial_position to Position object
            if initial_position is not None:
                position = Position(initial_position[0], initial_position[1], 0.0)
            else:
                position = Position(0.0, 0.0, 0.0)
            # Create AgentData object - filter out non-AgentData kwargs
            # Extract only valid AgentData fields from kwargs
            agent_data_fields = {
                "status",
                "capabilities",
                "personality",
                "resources",
                "relationships",
                "goals",
                "current_goal",
                "short_term_memory",
                "long_term_memory",
                "experience_count",
                "metadata",
                "belief_state",
                "generative_model_params",
            }
            agent_kwargs = {k: v for k, v in kwargs.items() if k in agent_data_fields}

            agent_data = AgentData(
                agent_id=agent_id or str(uuid.uuid4()),
                name=name or "Agent",
                agent_type=agent_type or "basic",
                position=position,
                **agent_kwargs,
            )
        elif agent_data is None:
            # No parameters provided, create default AgentData
            agent_data = AgentData(
                name="Agent", agent_type="basic", position=Position(0.0, 0.0, 0.0)
            )
        # Core data
        self.data = agent_data
        # External interfaces
        self.world_interface = world_interface
        self.active_inference_interface = active_inference_interface
        self.markov_blanket_interface = markov_blanket_interface
        self.config_provider = config_provider
        self.logger = logger or AgentLogger(self.data.agent_id)
        # Internal state
        self._is_running = False
        self._is_paused = False
        self._last_update_time = datetime.now()
        self._update_interval = timedelta(milliseconds=100)  # 10 FPS default
        # Component systems
        self._components: Dict[str, Any] = {}
        self._plugins: List[IAgentPlugin] = []
        self._event_handlers: List[IAgentEventHandler] = []
        # Markov blanket state tracking
        self._boundary_violations: List[BoundaryViolationEvent] = []
        self._last_boundary_check = datetime.now()
        # Thread management
        self._executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix=f"agent-{self.data.agent_id}"
        )
        self._main_loop_task: Optional[Task[Any]] = None
        # Initialize core components
        self._initialize_core_components()
        self.logger.log_info(
            self.data.agent_id,
            f"Agent {
                self.data.name} initialized",
        )

    @classmethod
    def create_from_params(
        cls,
        agent_id: str | None = None,
        name: str = "Agent",
        agent_class: Any | None = None,
        agent_type: str | None = None,
        initial_position: tuple[float, float] | None = None,
        **kwargs,
    ) -> BaseAgent:
        """
        Create a BaseAgent from individual parameters for backward compatibility.
        Args:
            agent_id: Unique agent identifier
            name: Agent name
            agent_class: Agent class (from AgentClass enum)
            agent_type: Agent type string
            initial_position: Tuple of (x, y) coordinates
            **kwargs: Additional parameters
        Returns:
            BaseAgent instance
        """
        # Convert agent_class to agent_type if provided
        if agent_class is not None and hasattr(agent_class, "value"):
            agent_type = agent_class.value
        elif agent_type is None:
            agent_type = "basic"
        # Convert initial_position to Position object
        if initial_position is not None:
            position = Position(initial_position[0], initial_position[1], 0.0)
        else:
            position = Position(0.0, 0.0, 0.0)
        # Create AgentData object - filter out non-AgentData kwargs
        agent_data_fields = {
            "status",
            "capabilities",
            "personality",
            "resources",
            "relationships",
            "goals",
            "current_goal",
            "short_term_memory",
            "long_term_memory",
            "experience_count",
            "metadata",
            "belief_state",
            "generative_model_params",
        }
        agent_kwargs = {k: v for k, v in kwargs.items() if k in agent_data_fields}
        other_kwargs = {k: v for k, v in kwargs.items() if k not in agent_data_fields}

        agent_data = AgentData(
            agent_id=agent_id or str(uuid.uuid4()),
            name=name,
            agent_type=agent_type or "basic",
            position=position,
            **agent_kwargs,
        )
        return cls(agent_data, **other_kwargs)

    def __new__(cls, *args, **kwargs):
        """
        Override __new__ to handle both old and new constructor interfaces.
        """
        # If first argument is AgentData, use normal constructor
        if args and hasattr(args[0], "agent_id"):
            return super().__new__(cls)
        # If we get individual parameters, use create_from_params
        if "agent_id" in kwargs or "agent_class" in kwargs or "initial_position" in kwargs:
            instance = super().__new__(cls)
            return instance
        # Default to normal constructor
        return super().__new__(cls)

    def _initialize_core_components(self) -> None:
        """Initialize core agent components"""
        try:
            # State manager
            self._components["state_manager"] = AgentStateManager()
            self._components["state_manager"].register_agent(self.data)
            # Core systems (using existing implementations)
            state_manager = self._components["state_manager"]
            self._components["perception"] = PerceptionSystem(state_manager)

            # Create movement system components
            collision_system = CollisionSystem()
            pathfinding_grid = PathfindingGrid(width=100, height=100, cell_size=1.0)
            movement_controller = MovementController(
                state_manager, collision_system, pathfinding_grid
            )

            self._components["decision"] = DecisionSystem(
                state_manager, self._components["perception"], movement_controller
            )
            self._components["memory"] = MemorySystem(self.data.agent_id)
            self._components["movement"] = movement_controller
            self._components["interaction"] = InteractionSystem()
            # Behavior tree
            self._components["behavior_tree"] = BehaviorTreeManager()
            # Active Inference integration
            if self.active_inference_interface:
                self._components["active_inference"] = ActiveInferenceIntegration(
                    self.data,
                    state_manager,
                    self._components["perception"],
                    self._components["decision"],
                    self._components["movement"],
                    self._components["memory"],
                )
            # Markov Blanket integration
            if self.markov_blanket_interface:
                self._components["markov_blanket"] = self.markov_blanket_interface
            else:
                # Create default MarkovBlanket implementation
                self._components["markov_blanket"] = MarkovBlanketFactory.create_pymdp_blanket(
                    agent_id=self.data.agent_id, num_states=4, num_observations=3, num_actions=2
                )
            # Set up boundary violation handler
            if hasattr(self._components["markov_blanket"], "set_violation_handler"):
                self._components["markov_blanket"].set_violation_handler(
                    self._handle_boundary_violation
                )
            # Initialize all components
            for name, component in self._components.items():
                if hasattr(component, "initialize"):
                    component.initialize(self.data)
        except Exception as e:
            self.logger.log_error(self.data.agent_id, f"Failed to initialize components: {e}")
            raise

    @property
    def agent_id(self) -> str:
        """Get agent ID"""
        return self.data.agent_id

    @property
    def is_running(self) -> bool:
        """Check if agent is running"""
        return self._is_running

    @property
    def is_paused(self) -> bool:
        """Check if agent is paused"""
        return self._is_paused

    def start(self) -> None:
        """Start the agent and initialize all components"""
        if self._is_running:
            self.logger.log_warning(self.data.agent_id, "Agent is already running")
            return
        try:
            self.logger.log_info(self.data.agent_id, "Starting agent")
            # Initialize plugins
            for plugin in self._plugins:
                plugin.initialize(self.data)
            # Update status
            self.data.update_status(AgentStatus.IDLE)
            self._is_running = True
            self._is_paused = False
            # Start main loop
            self._start_main_loop()
            # Notify event handlers
            for handler in self._event_handlers:
                handler.on_agent_created(self.data)
            self.logger.log_info(self.data.agent_id, "Agent started successfully")
        except Exception as e:
            self.logger.log_error(self.data.agent_id, f"Failed to start agent: {e}")
            self._is_running = False
            raise

    def stop(self) -> None:
        """Stop the agent and cleanup resources"""
        if not self._is_running:
            self.logger.log_warning(self.data.agent_id, "Agent is not running")
            return
        try:
            self.logger.log_info(self.data.agent_id, "Stopping agent")
            self._is_running = False
            self._is_paused = False
            # Stop main loop
            if self._main_loop_task:
                self._main_loop_task.cancel()
                self._main_loop_task = None
            # Update status
            self.data.update_status(AgentStatus.OFFLINE)
            # Cleanup plugins
            for plugin in self._plugins:
                plugin.cleanup(self.data)
            # Cleanup components
            for component in self._components.values():
                if hasattr(component, "cleanup"):
                    component.cleanup()
            # Notify event handlers
            for handler in self._event_handlers:
                handler.on_agent_destroyed(self.data)
            # Cleanup executor
            self._executor.shutdown(wait=True)
            self.logger.log_info(self.data.agent_id, "Agent stopped successfully")
        except Exception as e:
            self.logger.log_error(self.data.agent_id, f"Error during agent shutdown: {e}")

    def pause(self) -> None:
        """Pause agent execution"""
        if not self._is_running:
            self.logger.log_warning(self.data.agent_id, "Cannot pause - agent is not running")
            return
        self._is_paused = True
        self.data.update_status(AgentStatus.IDLE)
        self.logger.log_info(self.data.agent_id, "Agent paused")

    def resume(self) -> None:
        """Resume agent execution"""
        if not self._is_running:
            self.logger.log_warning(self.data.agent_id, "Cannot resume - agent is not running")
            return
        self._is_paused = False
        self.logger.log_info(self.data.agent_id, "Agent resumed")

    def restart(self) -> None:
        """Restart the agent (stop and start)"""
        self.logger.log_info(self.data.agent_id, "Restarting agent")
        self.stop()
        time.sleep(0.1)  # Brief pause
        self.start()

    def _start_main_loop(self) -> None:
        """Start the main agent loop"""

        async def main_loop():
            while self._is_running:
                try:
                    if not self._is_paused:
                        await self._update_cycle()
                    # Calculate sleep time to maintain update rate
                    elapsed = datetime.now() - self._last_update_time
                    sleep_time = max(0, (self._update_interval - elapsed).total_seconds())
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.log_error(self.data.agent_id, f"Error in main loop: {e}")
                    self.data.update_status(AgentStatus.ERROR)
                    await asyncio.sleep(1)  # Error recovery delay

        # Start the main loop as a task - handle no event loop case
        try:
            self._main_loop_task = asyncio.create_task(main_loop())
        except RuntimeError:
            # No event loop running (e.g., in tests) - skip async main loop
            self.logger.log_warning(
                self.data.agent_id, "No event loop available, skipping async main loop"
            )
            self._main_loop_task = None

    async def _update_cycle(self) -> None:
        """Execute one update cycle using Template Method pattern"""
        self._prepare_cycle()
        try:
            await self._execute_cycle_phases()
        except Exception as e:
            self._handle_cycle_error(e)

    def _prepare_cycle(self) -> None:
        """Prepare the update cycle with timing and plugin updates"""
        current_time = datetime.now()
        delta_time = (current_time - self._last_update_time).total_seconds()
        self._last_update_time = current_time

        for plugin in self._plugins:
            plugin.update(self.data, delta_time)

    async def _execute_cycle_phases(self) -> None:
        """Execute the main phases of the agent cycle"""
        observations = await self._execute_perception_phase()
        await self._execute_decision_phase(observations)
        self._execute_memory_phase()

    async def _execute_perception_phase(self) -> Optional[Any]:
        """Execute perception phase and return observations"""
        if "perception" not in self._components:
            return None

        observations = self._components["perception"].perceive(self.data, self.world_interface)

        if observations is not None:
            await self._process_observations(observations)
        return observations

    async def _process_observations(self, observations: Any) -> None:
        """Process observations through Markov blanket and Active Inference"""
        await self._update_markov_blanket(observations)
        self._update_active_inference(observations)

    async def _update_markov_blanket(self, observations: Any) -> None:
        """Update Markov blanket state and check boundary integrity"""
        if "markov_blanket" in self._components:
            agent_state = self._create_agent_state()
            self._components["markov_blanket"].update_states(agent_state, observations)
            await self._check_boundary_integrity()

    def _update_active_inference(self, observations: Any) -> None:
        """Update Active Inference beliefs"""
        if self.active_inference_interface:
            self.active_inference_interface.update_beliefs(self.data, observations)

    async def _execute_decision_phase(self, observations: Optional[Any]) -> None:
        """Execute decision-making phase"""
        if not self._can_make_decisions():
            return

        context = self._build_decision_context()
        behavior = self._components["behavior_tree"].evaluate(self.data, context)

        if behavior:
            await self._execute_behavior(behavior, context)

    def _can_make_decisions(self) -> bool:
        """Check if agent can make decisions"""
        return "behavior_tree" in self._components and "decision" in self._components

    async def _execute_behavior(self, behavior: Any, context: Dict[str, Any]) -> None:
        """Execute a behavior and process its result"""
        self.data.update_status(AgentStatus.PLANNING)
        result = behavior.execute(self.data, context)

        if result and "action" in result:
            await self._execute_action(result["action"])

    def _execute_memory_phase(self) -> None:
        """Execute memory consolidation phase"""
        if "memory" in self._components:
            self._components["memory"].consolidate_memory(self.data)

    def _handle_cycle_error(self, error: Exception) -> None:
        """Handle errors during the update cycle"""
        self.logger.log_error(self.data.agent_id, f"Error in update cycle: {error}")
        self.data.update_status(AgentStatus.ERROR)

    def _build_decision_context(self) -> Dict[str, Any]:
        """Build context for decision making"""
        context = {
            "timestamp": datetime.now(),
            "delta_time": (datetime.now() - self._last_update_time).total_seconds(),
            "agent_data": self.data,
            "world_interface": self.world_interface,
        }
        # Add component states
        for name, component in self._components.items():
            if hasattr(component, "get_state"):
                context[f"{name}_state"] = component.get_state()
        return context

    async def _execute_action(self, action: Action) -> None:
        """Execute an action"""
        try:
            self.data.update_status(
                AgentStatus.MOVING
                if action.action_type.value == "move"
                else AgentStatus.INTERACTING
            )
            # Execute action through world interface
            if self.world_interface:
                result = self.world_interface.perform_action(self.data, action)
                # Process action result
                if result.get("success", False):
                    # Update agent state based on action
                    if action.action_type.value == "move" and "new_position" in result:
                        old_position = self.data.position
                        new_position = Position(**result["new_position"])
                        self.data.update_position(new_position)
                        # Notify event handlers
                        for handler in self._event_handlers:
                            handler.on_agent_moved(self.data, old_position, new_position)
                # Add to memory
                self.data.add_to_memory(
                    {
                        "action": action.to_dict(),
                        "result": result,
                        "timestamp": datetime.now(),
                    }
                )
            self.data.update_status(AgentStatus.IDLE)
        except Exception as e:
            self.logger.log_error(self.data.agent_id, f"Error executing action: {e}")
            self.data.update_status(AgentStatus.ERROR)

    def add_behavior(self, behavior: IAgentBehavior) -> None:
        """Add a behavior to the agent"""
        if "behavior_tree" in self._components:
            self._components["behavior_tree"].add_behavior(behavior)
            self.logger.log_info(
                self.data.agent_id,
                f"Added behavior: {
                    type(behavior).__name__}",
            )

    def remove_behavior(self, behavior: IAgentBehavior) -> None:
        """Remove a behavior from the agent"""
        if "behavior_tree" in self._components:
            self._components["behavior_tree"].remove_behavior(behavior)
            self.logger.log_info(
                self.data.agent_id,
                f"Removed behavior: {
                    type(behavior).__name__}",
            )

    def add_plugin(self, plugin: IAgentPlugin) -> None:
        """Add a plugin to the agent."""
        self._plugins.append(plugin)
        if self._is_running:
            plugin.initialize(self.data)
        self.logger.log_info(
            self.data.agent_id,
            f"Added plugin: {
                plugin.get_name()}",
        )

    def remove_plugin(self, plugin: IAgentPlugin) -> None:
        """Remove a plugin from the agent"""
        if plugin in self._plugins:
            if self._is_running:
                plugin.cleanup(self.data)
            self._plugins.remove(plugin)
            self.logger.log_info(
                self.data.agent_id,
                f"Removed plugin: {
                    plugin.get_name()}",
            )

    def add_event_handler(self, handler: IAgentEventHandler) -> None:
        """Add an event handler"""
        self._event_handlers.append(handler)

    def remove_event_handler(self, handler: IAgentEventHandler) -> None:
        """Remove an event handler"""
        if handler in self._event_handlers:
            self._event_handlers.remove(handler)

    def get_component(self, name: str) -> Any | None:
        """Get a component by name"""
        return self._components.get(name)

    def get_state_summary(self) -> Dict[str, Any]:
        """Get comprehensive agent state summary"""
        summary = {
            "agent_id": self.data.agent_id,
            "name": self.data.name,
            "type": self.data.agent_type,
            "status": (
                self.data.status.value
                if hasattr(self.data.status, "value")
                else str(self.data.status)
            ),
            "position": {
                "x": self.data.position.x,
                "y": self.data.position.y,
                "z": self.data.position.z,
            },
            "is_running": self._is_running,
            "is_paused": self._is_paused,
            "last_update": self._last_update_time.isoformat(),
            "components": list(self._components.keys()),
            "plugins": [plugin.get_name() for plugin in self._plugins],
        }
        # Add component states
        for name, component in self._components.items():
            if hasattr(component, "get_state"):
                summary[f"{name}_state"] = component.get_state()
        # Add boundary metrics if available
        boundary_metrics = self.get_boundary_metrics()
        if "error" not in boundary_metrics:
            summary["boundary_metrics"] = boundary_metrics
        # Add Markov blanket state if available
        markov_state = self.get_markov_blanket_state()
        if "error" not in markov_state:
            summary["markov_blanket_state"] = markov_state
        return summary

    def __repr__(self) -> str:
        return (
            f"BaseAgent("
            f"id={self.data.agent_id}, "
            f"name={self.data.name}, "
            f"type={self.data.agent_type}, "
            f"status={self.data.status.value}"
            f")"
        )

    def _create_agent_state(self) -> AgentState:
        """Create an AgentState object from current agent data"""
        return AgentState(
            agent_id=self.data.agent_id,
            position=self.data.position,
            status=self.data.status.value if self.data.status else None,
            energy=self.data.resources.energy if self.data.resources else 1.0,
            health=self.data.resources.health if self.data.resources else 1.0,
            belief_state=getattr(self.data, "belief_state", None),
        )

    async def _check_boundary_integrity(self) -> None:
        """Check Markov blanket boundary integrity"""
        try:
            if "markov_blanket" not in self._components:
                return
            markov_blanket = self._components["markov_blanket"]
            # Verify statistical independence
            independence_score, details = markov_blanket.verify_independence()
            if independence_score < 0.7:  # Threshold for boundary integrity
                self.logger.log_warning(
                    self.data.agent_id,
                    f"Boundary integrity compromised: {
                        independence_score:.3f}",
                )
            # Detect violations
            violations = markov_blanket.detect_violations()
            if violations:
                self._boundary_violations.extend(violations)
                for violation in violations:
                    self.logger.log_warning(
                        self.data.agent_id,
                        f"Boundary violation detected: {
                            violation.violation_type} "
                        f"(severity: {
                            violation.severity})",
                    )
            # Update boundary check timestamp
            self._last_boundary_check = datetime.now()
        except Exception as e:
            self.logger.log_error(self.data.agent_id, f"Error checking boundary integrity: {e}")

    def _handle_boundary_violation(self, violation: BoundaryViolationEvent) -> None:
        """Handle boundary violation events"""
        self._boundary_violations.append(violation)
        self.logger.log_warning(
            self.data.agent_id,
            f"Boundary violation handler triggered: {
                violation.violation_type} "
            f"at {
                violation.timestamp} (severity: {
                violation.severity})",
        )
        # Update agent status if violation is severe
        if violation.severity > 0.8:
            self.data.update_status(AgentStatus.ERROR)
            self.logger.log_error(
                self.data.agent_id,
                "Severe boundary violation detected - agent status set to ERROR",
            )

    def get_boundary_metrics(self) -> Dict[str, Any]:
        """Get current boundary integrity metrics"""
        if "markov_blanket" not in self._components:
            return {"error": "MarkovBlanket not available"}
        try:
            markov_blanket = self._components["markov_blanket"]
            metrics = markov_blanket.get_metrics()
            return {
                "boundary_metrics": metrics,
                "violation_count": len(self._boundary_violations),
                "last_check": self._last_boundary_check.isoformat(),
                "recent_violations": [
                    {
                        "type": v.violation_type,
                        "severity": v.severity,
                        "timestamp": v.timestamp.isoformat(),
                    }
                    # Last 5 violations
                    for v in self._boundary_violations[-5:]
                ],
            }
        except Exception as e:
            return {"error": f"Failed to get boundary metrics: {e}"}

    def get_markov_blanket_state(self) -> Dict[str, Any]:
        """Get current Markov blanket state"""
        if "markov_blanket" not in self._components:
            return {"error": "MarkovBlanket not available"}
        try:
            markov_blanket = self._components["markov_blanket"]
            boundary_state = markov_blanket.get_boundary_state()
            dimensions = markov_blanket.get_dimensions()
            return {
                "dimensions": {
                    "internal": dimensions.internal_states,
                    "sensory": dimensions.sensory_states,
                    "active": dimensions.active_states,
                    "external": dimensions.external_states,
                },
                "boundary_state": boundary_state,
                "integrity_score": markov_blanket.verify_independence()[0],
            }
        except Exception as e:
            return {"error": f"Failed to get Markov blanket state: {e}"}


# Convenience function for creating agents
def create_agent(
    agent_type: str = "basic",
    name: str = "Agent",
    position: Position | None = None,
    **kwargs,
) -> BaseAgent:
    """
    Convenience function to create a new agent.
    Args:
        agent_type: Type of agent to create
        name: Agent name
        position: Initial position
        **kwargs: Additional agent data parameters
    Returns:
        BaseAgent instance
    """
    agent_data = AgentData(
        name=name,
        agent_type=agent_type,
        position=position or Position(0.0, 0.0, 0.0),
        **kwargs,
    )
    return BaseAgent(agent_data)
