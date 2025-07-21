"""GraphQL resolvers that integrate with actual FreeAgentics components."""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Import FreeAgentics components
try:
    from agents.agent_manager import AgentManager
    from coalitions.coalition_manager import CoalitionManager

    COMPONENTS_AVAILABLE = True
except ImportError:
    logger.warning("FreeAgentics components not available for GraphQL resolvers")
    COMPONENTS_AVAILABLE = False

# Global instances (in production, these would be dependency-injected)
if COMPONENTS_AVAILABLE:
    agent_manager = AgentManager()
    coalition_manager = CoalitionManager()

    # Create a default world
    try:
        agent_manager.create_world(size=20)
    except Exception as e:
        logger.warning(f"Could not create default world: {e}")


class GraphQLResolvers:
    """Real resolvers for GraphQL operations."""

    def __init__(self):
        """Initialize resolvers."""
        self.agent_manager = agent_manager if COMPONENTS_AVAILABLE else None
        self.coalition_manager = coalition_manager if COMPONENTS_AVAILABLE else None

    # Agent resolvers
    def get_agents(self, status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get list of agents."""
        if not self.agent_manager:
            return []

        try:
            agent_statuses = self.agent_manager.get_all_agents_status()

            if status:
                agent_statuses = [a for a in agent_statuses if a.get("status") == status]

            # Convert to GraphQL format
            agents = []
            for agent_status in agent_statuses[:limit]:
                agents.append(
                    {
                        "id": agent_status.get("agent_id", ""),
                        "name": agent_status.get("name", "Unknown"),
                        "status": ("active" if agent_status.get("is_active") else "inactive"),
                        "created_at": datetime.fromisoformat(
                            agent_status.get("created_at", datetime.now().isoformat())
                        ),
                        "last_active": (
                            datetime.fromisoformat(agent_status.get("last_action_at"))
                            if agent_status.get("last_action_at")
                            else None
                        ),
                        "total_steps": agent_status.get("total_steps", 0),
                        "capabilities": [
                            "exploration",
                            "navigation",
                        ],  # Default capabilities
                        "performance_score": agent_status.get("metrics", {}).get(
                            "avg_free_energy", 0.0
                        ),
                    }
                )

            return agents

        except Exception as e:
            logger.error(f"Error getting agents: {e}")
            return []

    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific agent by ID."""
        if not self.agent_manager:
            return None

        try:
            agent_status = self.agent_manager.get_agent_status(agent_id)

            return {
                "id": agent_status.get("agent_id", ""),
                "name": agent_status.get("name", "Unknown"),
                "status": "active" if agent_status.get("is_active") else "inactive",
                "created_at": datetime.fromisoformat(
                    agent_status.get("created_at", datetime.now().isoformat())
                ),
                "last_active": (
                    datetime.fromisoformat(agent_status.get("last_action_at"))
                    if agent_status.get("last_action_at")
                    else None
                ),
                "total_steps": agent_status.get("total_steps", 0),
                "capabilities": ["exploration", "navigation"],
                "performance_score": agent_status.get("metrics", {}).get("avg_free_energy", 0.0),
            }

        except Exception as e:
            logger.error(f"Error getting agent {agent_id}: {e}")
            return None

    def create_agent(
        self, name: str, template: str, parameters: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new agent."""
        if not self.agent_manager:
            raise Exception("Agent manager not available")

        try:
            # Parse parameters if provided
            params = {}
            if parameters:
                try:
                    params = json.loads(parameters)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON parameters: {parameters}")

            # Map template to agent type
            agent_type = "explorer"  # Default for now
            if template == "basic-explorer":
                agent_type = "explorer"

            agent_id = self.agent_manager.create_agent(agent_type, name, **params)

            return {
                "id": agent_id,
                "name": name,
                "status": "pending",
                "created_at": datetime.now(),
                "last_active": None,
                "total_steps": 0,
                "capabilities": ["exploration", "navigation"],
                "performance_score": 0.0,
            }

        except Exception as e:
            logger.error(f"Error creating agent: {e}")
            raise Exception(f"Failed to create agent: {str(e)}")

    def update_agent_status(self, agent_id: str, status: str) -> Optional[Dict[str, Any]]:
        """Update agent status."""
        if not self.agent_manager:
            return None

        try:
            success = False
            if status == "active":
                success = self.agent_manager.start_agent(agent_id)
            elif status == "inactive":
                success = self.agent_manager.stop_agent(agent_id)

            if success:
                return self.get_agent(agent_id)
            else:
                raise Exception(f"Failed to update agent {agent_id} to status {status}")

        except Exception as e:
            logger.error(f"Error updating agent status: {e}")
            return None

    # Coalition resolvers
    def get_coalitions(
        self, status: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get list of coalitions."""
        if not self.coalition_manager:
            return []

        try:
            coalitions = []
            for coalition in self.coalition_manager.coalitions.values():
                coalition_data = coalition.get_status()

                if status and coalition_data.get("status") != status:
                    continue

                coalitions.append(
                    {
                        "id": coalition_data.get("coalition_id", ""),
                        "name": coalition_data.get("name", "Unknown"),
                        "status": coalition_data.get("status", "unknown"),
                        "member_count": coalition_data.get("member_count", 0),
                        "leader_id": coalition_data.get("leader_id"),
                        "objectives_count": coalition_data.get("objectives_count", 0),
                        "completed_objectives": coalition_data.get("completed_objectives", 0),
                        "performance_score": coalition_data.get("performance_score", 0.0),
                        "coordination_efficiency": coalition_data.get(
                            "coordination_efficiency", 0.0
                        ),
                        "created_at": datetime.fromisoformat(
                            coalition_data.get("created_at", datetime.now().isoformat())
                        ),
                    }
                )

            return coalitions[:limit]

        except Exception as e:
            logger.error(f"Error getting coalitions: {e}")
            return []

    def get_coalition(self, coalition_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific coalition by ID."""
        if not self.coalition_manager:
            return None

        try:
            coalition = self.coalition_manager.get_coalition(coalition_id)
            if not coalition:
                return None

            coalition_data = coalition.get_status()

            return {
                "id": coalition_data.get("coalition_id", ""),
                "name": coalition_data.get("name", "Unknown"),
                "status": coalition_data.get("status", "unknown"),
                "member_count": coalition_data.get("member_count", 0),
                "leader_id": coalition_data.get("leader_id"),
                "objectives_count": coalition_data.get("objectives_count", 0),
                "completed_objectives": coalition_data.get("completed_objectives", 0),
                "performance_score": coalition_data.get("performance_score", 0.0),
                "coordination_efficiency": coalition_data.get("coordination_efficiency", 0.0),
                "created_at": datetime.fromisoformat(
                    coalition_data.get("created_at", datetime.now().isoformat())
                ),
            }

        except Exception as e:
            logger.error(f"Error getting coalition {coalition_id}: {e}")
            return None

    # World state resolvers
    def get_world_state(self) -> Optional[Dict[str, Any]]:
        """Get current world state."""
        if not self.agent_manager:
            return None

        try:
            world_state = self.agent_manager.get_world_state()
            if not world_state:
                return None

            return {
                "size": world_state.get("size", 0),
                "step_count": world_state.get("step_count", 0),
                "agent_count": world_state.get("num_agents", 0),
                "active_agents": len(
                    [a for a in self.agent_manager.get_all_agents_status() if a.get("is_active")]
                ),
            }

        except Exception as e:
            logger.error(f"Error getting world state: {e}")
            return None

    # System metrics resolvers
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        try:
            # Get agent metrics
            total_agents = 0
            active_agents = 0

            if self.agent_manager:
                agent_statuses = self.agent_manager.get_all_agents_status()
                total_agents = len(agent_statuses)
                active_agents = len([a for a in agent_statuses if a.get("is_active")])

            # Get coalition metrics
            total_coalitions = 0
            active_coalitions = 0
            pending_objectives = 0

            if self.coalition_manager:
                system_status = self.coalition_manager.get_system_status()
                total_coalitions = system_status.get("total_coalitions", 0)
                active_coalitions = system_status.get("active_coalitions", 0)
                pending_objectives = system_status.get("pending_objectives", 0)

            # Mock some metrics for now
            inference_rate = 12.5
            avg_response_time = 245.6

            return {
                "total_agents": total_agents,
                "active_agents": active_agents,
                "total_coalitions": total_coalitions,
                "active_coalitions": active_coalitions,
                "pending_objectives": pending_objectives,
                "inference_rate": inference_rate,
                "avg_response_time": avg_response_time,
            }

        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {
                "total_agents": 0,
                "active_agents": 0,
                "total_coalitions": 0,
                "active_coalitions": 0,
                "pending_objectives": 0,
                "inference_rate": 0.0,
                "avg_response_time": 0.0,
            }

    # Inference resolvers
    def perform_inference(
        self,
        agent_id: Optional[str],
        model: str,
        input_text: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Dict[str, Any]:
        """Perform inference using an agent or model."""
        try:
            # Mock implementation for now
            # In practice, this would integrate with the inference engine

            processing_start = datetime.now()

            # Simulate processing
            output_text = f"Processed input: {input_text[:50]}..."
            confidence = 0.85

            processing_time = (datetime.now() - processing_start).total_seconds()

            return {
                "agent_id": agent_id or "default_agent",
                "input_text": input_text,
                "output_text": output_text,
                "confidence": confidence,
                "processing_time": processing_time,
                "timestamp": datetime.now(),
            }

        except Exception as e:
            logger.error(f"Error performing inference: {e}")
            raise Exception(f"Inference failed: {str(e)}")


# Global resolver instance
resolvers = GraphQLResolvers()
