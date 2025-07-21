"""Agent manager for creating and coordinating agents."""

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional

from agents.agent_adapter import ActiveInferenceGridAdapter
from agents.base_agent import ActiveInferenceAgent, BasicExplorerAgent
from world.grid_world import CellType, GridWorld, GridWorldConfig, Position

logger = logging.getLogger(__name__)


class AgentManager:
    """Manages the lifecycle and coordination of Active Inference agents."""

    def __init__(self):
        """Initialize the agent manager."""
        self.agents: Dict[str, ActiveInferenceAgent] = {}
        self.world: Optional[GridWorld] = None
        self.adapter = ActiveInferenceGridAdapter()
        self.running = False
        self._agent_counter = 0
        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=2)
        # Event queue for async broadcasts
        self._event_queue: List[Dict[str, Any]] = []
        self._event_lock = threading.Lock()

        logger.info("Agent manager initialized")

    def _find_free_position(self) -> Position:
        """Find a free position in the world for a new agent."""
        if not self.world:
            return Position(0, 0)

        # Try to find an unoccupied position
        for y in range(self.world.height):
            for x in range(self.world.width):
                pos = Position(x, y)
                if pos not in [agent.position for agent in self.world.agents.values()]:
                    # Check if the cell is empty/walkable
                    cell = self.world.get_cell(pos)
                    if cell and cell.type in [
                        CellType.EMPTY,
                        CellType.RESOURCE,
                    ]:
                        return pos

        # Fallback to (0,0) if no free position found
        return Position(0, 0)

    def create_world(self, size_or_config=10) -> GridWorld:
        """Create a world for agents to interact in.

        Args:
            size_or_config: Size of the grid world (int) or GridWorldConfig object

        Returns:
            Created world
        """
        if isinstance(size_or_config, GridWorldConfig):
            config = size_or_config
            size = config.width
        else:
            size = size_or_config
            config = GridWorldConfig(width=size, height=size)

        self.world = GridWorld(config)
        logger.info(f"Created {size}x{config.height} world")
        return self.world

    def create_agent(self, agent_type: str, name: str, **kwargs) -> str:
        """Create a new agent.

        Args:
            agent_type: Type of agent to create
            name: Human-readable name for the agent
            **kwargs: Additional configuration parameters

        Returns:
            Agent ID
        """
        self._agent_counter += 1
        # Generate ID based on agent type
        if agent_type == "active_inference":
            agent_id = f"ai_agent_{self._agent_counter}"
        else:
            agent_id = f"test_agent_{self._agent_counter}"

        # Validate that world is available
        if not self.world:
            raise ValueError(
                "Cannot create agent without a world. World must be initialized first."
            )

        # Create agent based on type
        try:
            if agent_type in [
                "explorer",
                "basic",
                "active_inference",
            ]:  # Support all names
                grid_size = self.world.width
                agent = BasicExplorerAgent(agent_id=agent_id, name=name, grid_size=grid_size)
                # Pass any PyMDP config parameters to the agent
                for key, value in kwargs.items():
                    if key in ["num_states", "num_obs", "num_controls", "num_actions"]:
                        setattr(agent, key, value)
            else:
                logger.warning(f"Unknown agent type: {agent_type}")
                return None
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            return None

        self.agents[agent_id] = agent

        # Add to world using adapter
        if self.world:
            # Choose starting position
            if "position" in kwargs:
                pos = kwargs["position"]
                if isinstance(pos, Position):
                    start_pos = pos
                else:
                    start_pos = Position(pos[0], pos[1])

                # Validate position is within world bounds
                if (
                    start_pos.x < 0
                    or start_pos.x >= self.world.width
                    or start_pos.y < 0
                    or start_pos.y >= self.world.height
                ):
                    raise ValueError(
                        f"Position {start_pos} is outside world bounds ({self.world.width}x{self.world.height})"
                    )
            else:
                # Find an unoccupied position
                start_pos = self._find_free_position()

            # Register with adapter and add to world
            grid_agent = self.adapter.register_agent(agent, start_pos)
            self.world.add_agent(grid_agent)

        logger.info(f"Created {agent_type} agent: {agent_id} ({name})")

        # Queue event for async broadcast
        self._queue_event(
            agent_id,
            "created",
            {
                "agent_type": agent_type,
                "name": name,
                "timestamp": datetime.now().isoformat(),
            },
        )

        return agent_id

    def _queue_event(self, agent_id: str, event_type: str, data: dict):
        """Queue an event for async broadcast."""
        with self._event_lock:
            self._event_queue.append({"agent_id": agent_id, "event_type": event_type, "data": data})

        # Submit async broadcast task to thread pool
        self._executor.submit(self._process_event_queue)

    def _process_event_queue(self):
        """Process queued events in a separate thread."""
        events_to_process = []

        with self._event_lock:
            events_to_process = self._event_queue.copy()
            self._event_queue.clear()

        if events_to_process:
            # Create new event loop for this thread if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Process all events
            for event in events_to_process:
                loop.run_until_complete(
                    self._broadcast_agent_event(
                        event["agent_id"], event["event_type"], event["data"]
                    )
                )

    async def _broadcast_agent_event(self, agent_id: str, event_type: str, data: dict):
        """Broadcast agent event via WebSocket."""
        try:
            from api.v1.websocket import broadcast_agent_event

            await broadcast_agent_event(agent_id, event_type, data)
        except Exception as e:
            logger.error(f"Failed to broadcast agent event: {e}")

    def start_agent(self, agent_id: str) -> bool:
        """Start an agent.

        Args:
            agent_id: Agent to start

        Returns:
            True if started successfully
        """
        if agent_id not in self.agents:
            logger.error(f"Agent {agent_id} not found")
            return False

        agent = self.agents[agent_id]
        agent.start()

        # Queue agent start event
        self._queue_event(agent_id, "started", {"timestamp": datetime.now().isoformat()})

        return True

    def stop_agent(self, agent_id: str) -> bool:
        """Stop an agent.

        Args:
            agent_id: Agent to stop

        Returns:
            True if stopped successfully
        """
        if agent_id not in self.agents:
            logger.error(f"Agent {agent_id} not found")
            return False

        agent = self.agents[agent_id]
        agent.stop()

        # Queue agent stop event
        self._queue_event(agent_id, "stopped", {"timestamp": datetime.now().isoformat()})

        return True

    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent (alias for delete_agent for compatibility)."""
        return self.delete_agent(agent_id)

    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent.

        Args:
            agent_id: Agent to delete

        Returns:
            True if deleted successfully
        """
        if agent_id not in self.agents:
            logger.error(f"Agent {agent_id} not found")
            return False

        # Stop agent if running
        self.stop_agent(agent_id)

        # Remove from world
        if self.world:
            self.world.remove_agent(agent_id)

        # Unregister from adapter
        self.adapter.unregister_agent(agent_id)

        # Delete agent
        del self.agents[agent_id]
        logger.info(f"Deleted agent {agent_id}")
        return True

    def step_agent(self, agent_id: str) -> Dict[str, Any]:
        """Execute one step for an agent.

        Args:
            agent_id: Agent to step

        Returns:
            Step result
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")

        if not self.world:
            raise RuntimeError("No world available")

        agent = self.agents[agent_id]
        grid_agent = self.adapter.get_grid_agent(agent_id)

        if not grid_agent:
            raise RuntimeError(f"No grid agent found for {agent_id}")

        # Get observation from world
        grid_observation = self.world.get_observation(agent_id)

        # Convert observation to AI agent format
        ai_observation = self.adapter.convert_observation(grid_observation)

        # AI agent processes observation and selects action
        action = agent.step(ai_observation)

        # Convert action to new position
        new_position = self.adapter.convert_action(action, grid_agent.position)

        # Move agent in world
        success = self.world.move_agent(agent_id, new_position)

        # Sync agent states
        self.adapter.sync_agent_state(agent_id)

        return {
            "agent_id": agent_id,
            "action": action,
            "new_position": {"x": new_position.x, "y": new_position.y},
            "success": success,
        }

    def step_all(self) -> Dict[str, Dict[str, Any]]:
        """Execute one step for all active agents.

        Returns:
            Results for each agent
        """
        results = {}

        # Step world
        if self.world:
            self.world.step()

        # Step each active agent
        for agent_id, agent in self.agents.items():
            if agent.is_active:
                try:
                    results[agent_id] = self.step_agent(agent_id)
                except Exception as e:
                    logger.error(f"Error stepping agent {agent_id}: {e}")
                    results[agent_id] = {"error": str(e)}

        return results

    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get status of an agent.

        Args:
            agent_id: Agent to query

        Returns:
            Agent status
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")

        return self.agents[agent_id].get_status()

    def get_all_agents_status(self) -> List[Dict[str, Any]]:
        """Get status of all agents.

        Returns:
            List of agent statuses
        """
        return [agent.get_status() for agent in self.agents.values()]

    def get_world_state(self) -> Optional[Dict[str, Any]]:
        """Get current world state.

        Returns:
            World state or None if no world
        """
        if self.world:
            return self.world.get_state()
        return None

    def render_world(self) -> Optional[str]:
        """Render world as ASCII art.

        Returns:
            ASCII representation or None if no world
        """
        if self.world:
            return self.world.render()
        return None

    def get_agent_belief_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get belief monitoring statistics for a specific agent.

        Args:
            agent_id: Agent to query

        Returns:
            Belief monitoring statistics
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")

        return self.agents[agent_id].get_belief_monitoring_stats()

    def get_all_belief_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get belief monitoring statistics for all agents.

        Returns:
            Dictionary mapping agent IDs to their belief statistics
        """
        results = {}
        for agent_id, agent in self.agents.items():
            try:
                results[agent_id] = agent.get_belief_monitoring_stats()
            except Exception as e:
                logger.error(f"Failed to get belief stats for agent {agent_id}: {e}")
                results[agent_id] = {"error": str(e)}

        return results

    def reset_belief_monitoring(self, agent_id: Optional[str] = None) -> bool:
        """Reset belief monitoring for one or all agents.

        Args:
            agent_id: Agent to reset (None for all agents)

        Returns:
            True if successful
        """
        try:
            from observability.belief_monitoring import belief_monitoring_hooks

            if agent_id:
                if agent_id not in self.agents:
                    raise ValueError(f"Agent {agent_id} not found")
                belief_monitoring_hooks.reset_agent_monitor(agent_id)
                logger.info(f"Reset belief monitoring for agent {agent_id}")
            else:
                belief_monitoring_hooks.reset_all()
                logger.info("Reset belief monitoring for all agents")

            return True
        except Exception as e:
            logger.error(f"Failed to reset belief monitoring: {e}")
            return False

    def get_coordination_stats(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get coordination statistics for one or all agents.

        Args:
            agent_id: Agent to query (None for all agents)

        Returns:
            Coordination statistics
        """
        try:
            from observability.coordination_metrics import (
                get_agent_coordination_stats,
                get_system_coordination_report,
            )

            if agent_id:
                if agent_id not in self.agents:
                    raise ValueError(f"Agent {agent_id} not found")
                return get_agent_coordination_stats(agent_id)
            else:
                # Get system-wide coordination report
                system_report = get_system_coordination_report()

                # Add individual agent stats
                agent_stats = {}
                for aid in self.agents.keys():
                    agent_stats[aid] = get_agent_coordination_stats(aid)

                return {
                    "system_report": system_report,
                    "agent_stats": agent_stats,
                }
        except Exception as e:
            logger.error(f"Failed to get coordination stats: {e}")
            return {"error": str(e)}

    def run_simulation(self, steps: int = 100) -> List[Dict[str, Dict[str, Any]]]:
        """Run simulation for multiple steps.

        Args:
            steps: Number of steps to run

        Returns:
            History of results
        """
        history = []

        logger.info(f"Starting simulation for {steps} steps")

        for i in range(steps):
            results = self.step_all()
            history.append(results)

            if i % 10 == 0:
                logger.info(f"Simulation step {i}/{steps}")

        logger.info("Simulation complete")
        return history

    def get_agent(self, agent_id: str) -> Optional[ActiveInferenceAgent]:
        """Get an agent by ID.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent instance or None if not found
        """
        return self.agents.get(agent_id)

    def list_agents(self) -> List[Dict[str, str]]:
        """List all agents with their information.

        Returns:
            List of agent information dictionaries
        """
        return [{"id": agent_id, "name": agent.name} for agent_id, agent in self.agents.items()]

    def start(self) -> None:
        """Start the agent manager.

        Currently no specific startup needed.
        """
        self.running = True
        logger.info("AgentManager started")

    def stop(self) -> None:
        """Stop the agent manager and all agents."""
        logger.info("Stopping AgentManager and all agents")
        self.running = False
        for agent_id in list(self.agents.keys()):
            self.stop_agent(agent_id)
        logger.info("AgentManager stopped")

    def update(self) -> None:
        """Update the agent manager state.

        Updates all agents if they have an update method.
        """
        for agent_id, agent in self.agents.items():
            if hasattr(agent, "update") and callable(getattr(agent, "update")):
                try:
                    agent.update()
                except Exception as e:
                    logger.error(f"Error updating agent {agent_id}: {e}")

    def set_world(self, world: GridWorld) -> None:
        """Set the world for the agent manager.

        Args:
            world: GridWorld instance
        """
        self.world = world
        logger.info(f"World set to {world.width}x{world.height} grid")
